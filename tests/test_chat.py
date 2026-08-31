from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest

from kilix_ollama.chat import ChatEngine, conversation_title
from kilix_ollama.conversations import ConversationStore
from kilix_ollama.errors import RuntimeUnavailable, ToolError
from kilix_ollama.paths import RuntimePaths
from kilix_ollama.streaming import StreamDelta, StreamResult
from kilix_ollama.tools import ToolRegistry


def private_paths(root: Path) -> RuntimePaths:
    return RuntimePaths.from_environment(
        {
            "HOME": str(root / "home"),
            "XDG_RUNTIME_DIR": str(root / "run"),
            "XDG_STATE_HOME": str(root / "state"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
        },
        uid=os.getuid(),
    )


def result(
    content: str,
    *,
    calls: tuple[dict[str, object], ...] = (),
    thinking: str = "",
) -> StreamResult:
    return StreamResult(
        request_model="local:latest",
        response_model="local:latest",
        content=content,
        thinking=thinking,
        tool_calls=calls,
        done_reason="stop",
        metrics={"eval_count": 1},
        event_count=1,
    )


class _FakeClient:
    def __init__(self, results: list[StreamResult], *, installed: bool = True) -> None:
        self.results = list(results)
        self.installed = installed
        self.install_checks = []
        self.chat_calls = []

    def require_installed(self, model: str) -> dict[str, object]:
        self.install_checks.append(model)
        if not self.installed:
            raise RuntimeUnavailable("missing")
        return {"name": model}

    def chat(self, model: str, messages, **kwargs) -> StreamResult:
        self.chat_calls.append((model, [dict(message) for message in messages], kwargs))
        if not self.results:
            raise AssertionError("unexpected extra chat call")
        value = self.results.pop(0)
        callback = kwargs.get("on_delta")
        if callback is not None and value.content:
            callback(StreamDelta(content=value.content, done=True))
        return value


class ChatEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.identifiers = iter(("1" * 32, "2" * 32, "3" * 32))
        self.store = ConversationStore(
            private_paths(self.root),
            clock=lambda: datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
            identifier_factory=lambda: next(self.identifiers),
        )

    def test_plain_turn_streams_and_persists_typed_messages(self) -> None:
        client = _FakeClient([result("answer", thinking="private")])
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Test")
        deltas = []
        turn = engine.turn(conversation.identifier, "question", on_delta=deltas.append)
        self.assertEqual(turn.result.content, "answer")
        self.assertEqual(turn.tool_rounds, 0)
        self.assertEqual(client.install_checks, ["local:latest"])
        self.assertEqual([delta.content for delta in deltas], ["answer"])
        stored = self.store.load(conversation.identifier)
        self.assertEqual(
            [message["role"] for message in stored.messages],
            ["user", "assistant"],
        )
        self.assertEqual(stored.messages[1]["thinking"], "private")
        self.assertFalse(client.chat_calls[0][2]["check_installed"])

    def test_approved_tool_call_is_bounded_persisted_and_returned_to_model(self) -> None:
        tool_call = {
            "type": "function",
            "function": {"name": "kilix.current_time", "arguments": {}},
        }
        client = _FakeClient([result("", calls=(tool_call,)), result("It is noon.")])
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Tools")
        registry = ToolRegistry(
            clock=lambda: datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
        )
        previews = []
        turn = engine.turn(
            conversation.identifier,
            "What time is it?",
            registry=registry,
            approve=lambda preview: previews.append(preview) or True,
        )
        self.assertEqual(turn.tool_rounds, 1)
        self.assertEqual(turn.tool_calls, 1)
        self.assertEqual(turn.tool_denials, 0)
        self.assertEqual(len(previews), 1)
        self.assertEqual(len(client.chat_calls), 2)
        second_messages = client.chat_calls[1][1]
        self.assertEqual(second_messages[-1]["role"], "tool")
        tool_payload = json.loads(second_messages[-1]["content"])
        self.assertEqual(tool_payload["status"], "ok")
        self.assertEqual(
            [message["role"] for message in turn.conversation.messages],
            ["user", "assistant", "tool", "assistant"],
        )

    def test_denied_tool_is_reported_as_denied_and_does_not_execute(self) -> None:
        invoked = False

        def clock() -> datetime:
            nonlocal invoked
            invoked = True
            return datetime.now(timezone.utc)

        tool_call = {
            "type": "function",
            "function": {"name": "kilix.current_time", "arguments": {}},
        }
        client = _FakeClient([result("", calls=(tool_call,)), result("Denied safely.")])
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Deny")
        turn = engine.turn(
            conversation.identifier,
            "Try a tool",
            registry=ToolRegistry(clock=clock),
            approve=lambda _preview: False,
        )
        self.assertEqual(turn.tool_denials, 1)
        self.assertFalse(invoked)
        self.assertEqual(
            json.loads(client.chat_calls[1][1][-1]["content"])["status"],
            "denied",
        )

    def test_tool_calls_are_refused_when_registry_is_disabled_or_unknown(self) -> None:
        known = {
            "type": "function",
            "function": {"name": "kilix.current_time", "arguments": {}},
        }
        client = _FakeClient([result("", calls=(known,))])
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Disabled")
        with self.assertRaises(ToolError):
            engine.turn(conversation.identifier, "call", registry=None)
        self.assertEqual(len(self.store.load(conversation.identifier).messages), 2)

        unknown = {
            "type": "function",
            "function": {"name": "kilix.shell", "arguments": {}},
        }
        client = _FakeClient([result("", calls=(unknown,))])
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Unknown")
        with self.assertRaises(ToolError):
            engine.turn(
                conversation.identifier,
                "call",
                registry=ToolRegistry(),
                approve=lambda _preview: True,
            )
        self.assertEqual(len(client.chat_calls), 1)

    def test_tool_loop_stops_after_exact_round_budget(self) -> None:
        tool_call = {
            "type": "function",
            "function": {"name": "kilix.current_time", "arguments": {}},
        }
        client = _FakeClient([result("", calls=(tool_call,)) for _ in range(5)])
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Loop")
        with self.assertRaises(ToolError):
            engine.turn(
                conversation.identifier,
                "loop",
                registry=ToolRegistry(),
                approve=lambda _preview: False,
            )
        self.assertEqual(len(client.chat_calls), 5)

    def test_missing_model_fails_before_conversation_mutation(self) -> None:
        client = _FakeClient([], installed=False)
        engine = ChatEngine(client, self.store)  # type: ignore[arg-type]
        conversation = engine.create(model="local:latest", title="Missing")
        with self.assertRaises(RuntimeUnavailable):
            engine.turn(conversation.identifier, "question")
        self.assertEqual(self.store.load(conversation.identifier).messages, ())
        self.assertEqual(client.chat_calls, [])

    def test_conversation_title_is_safe_bounded_and_first_line_only(self) -> None:
        self.assertEqual(conversation_title("hello\nsecond"), "hello")
        self.assertEqual(conversation_title("\x1b]0;bad\x07safe"), "safe")
        self.assertEqual(conversation_title(""), "New chat")
        self.assertLessEqual(len(conversation_title("x" * 200)), 72)


if __name__ == "__main__":
    unittest.main()
