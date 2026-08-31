from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kilix_ollama.cli import main
from kilix_ollama.streaming import StreamDelta, StreamResult


def result(content: str, *, thinking: str = "") -> StreamResult:
    return StreamResult(
        request_model="local:latest",
        response_model="local:latest",
        content=content,
        thinking=thinking,
        tool_calls=(),
        done_reason="stop",
        metrics={"eval_count": 2},
        event_count=2,
    )


class _FakeClient:
    def __init__(self, content: str = "answer", *, thinking: str = "") -> None:
        self.content = content
        self.thinking = thinking
        self.generate_calls = []
        self.chat_calls = []

    def installed_models(self):
        return ({"name": "local:latest", "digest": "a" * 64, "size": 10},)

    def running_models(self):
        return ()

    def show_model(self, model: str):
        return {"model": model, "family": "fixture"}

    def require_installed(self, model: str):
        return {"name": model}

    def generate(self, model: str, prompt: str, **kwargs):
        self.generate_calls.append((model, prompt, kwargs))
        callback = kwargs.get("on_delta")
        if callback is not None:
            callback(StreamDelta(content=self.content[:3], done=False))
            callback(StreamDelta(content=self.content[3:], done=True))
        return result(self.content, thinking=self.thinking)

    def chat(self, model: str, messages, **kwargs):
        self.chat_calls.append((model, [dict(message) for message in messages], kwargs))
        callback = kwargs.get("on_delta")
        if callback is not None:
            callback(StreamDelta(content=self.content, done=True))
        return result(self.content, thinking=self.thinking)


class _BrokenStdout:
    def write(self, _value: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError

    def fileno(self) -> int:
        raise OSError("fixture has no descriptor")


class ProductCliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_offline_library_has_three_exact_blocked_models_without_client(self) -> None:
        with mock.patch(
            "kilix_ollama.cli.OllamaClient", side_effect=AssertionError("runtime access")
        ):
            status, stdout, stderr = self.invoke("library", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(record["curated_count"], 3)
        self.assertEqual(record["installed_count"], 0)
        self.assertFalse(record["automatic_install"])

    def test_models_ps_show_and_installed_library_use_structured_client(self) -> None:
        client = _FakeClient()
        with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
            status, stdout, _ = self.invoke("models", "--json")
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["count"], 1)
            status, stdout, _ = self.invoke("ps", "--json")
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["count"], 0)
            status, stdout, _ = self.invoke(
                "show", "local:latest", "--development-runtime", "--json"
            )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout)["family"], "fixture")
            status, stdout, _ = self.invoke("library", "--installed", "--json")
            self.assertEqual(status, 0)
            library = json.loads(stdout)
            self.assertEqual(library["installed_count"], 1)
            self.assertTrue(library["runtime_contacted"])

    def test_installed_library_records_a_successful_empty_runtime_contact(self) -> None:
        client = _FakeClient()
        client.installed_models = lambda: ()
        with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
            status, stdout, stderr = self.invoke("library", "--installed", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(record["installed_count"], 0)
        self.assertTrue(record["runtime_contacted"])

    def test_one_off_plain_output_is_streamed_and_terminal_sanitized(self) -> None:
        client = _FakeClient("ok\x1b[31mred\x1b[0m")
        with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
            status, stdout, stderr = self.invoke(
                "ask",
                "local:latest",
                "private",
                "prompt",
                "--development-runtime",
            )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, "okred\n")
        self.assertEqual(client.generate_calls[0][1], "private prompt")

    def test_one_off_json_and_json_lines_escape_raw_controls(self) -> None:
        client = _FakeClient("a\x1b[31mb")
        with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
            status, stdout, stderr = self.invoke(
                "ask", "local:latest", "prompt", "--json", "--development-runtime"
            )
            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(json.loads(stdout)["content"], "a\x1b[31mb")
            self.assertNotIn("\x1b", stdout)
            status, stdout, stderr = self.invoke(
                "ask",
                "local:latest",
                "prompt",
                "--stream-json",
                "--development-runtime",
            )
            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            lines = stdout.splitlines()
            self.assertEqual(json.loads(lines[-1])["type"], "terminal")
            self.assertNotIn("\x1b", stdout)

    def test_one_off_atomic_output_is_no_clobber_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.txt"
            client = _FakeClient("raw\x1b[31mdata")
            with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
                status, stdout, stderr = self.invoke(
                    "ask",
                    "local:latest",
                    "prompt",
                    "--output",
                    str(output),
                    "--development-runtime",
                )
                self.assertEqual(status, 0)
                self.assertEqual(stdout, "")
                self.assertIn("wrote", stderr)
                self.assertEqual(output.read_text(encoding="utf-8"), "raw\x1b[31mdata")
                status, _stdout, stderr = self.invoke(
                    "ask",
                    "local:latest",
                    "prompt",
                    "--output",
                    str(output),
                    "--development-runtime",
                )
                self.assertEqual(status, 2)
                self.assertIn("--overwrite", stderr)

    def test_broken_pipe_and_active_cancellation_have_fixed_exit_contracts(self) -> None:
        client = _FakeClient("answer")
        with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client), mock.patch(
            "sys.stdout", _BrokenStdout()
        ):
            self.assertEqual(
                main(("ask", "local:latest", "prompt", "--development-runtime")),
                0,
            )

        client.generate = mock.Mock(side_effect=KeyboardInterrupt)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client), redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            status = main(
                ("ask", "local:latest", "prompt", "--development-runtime")
            )
        self.assertEqual(status, 130)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "kilix-ollama: cancelled\n")

    def test_one_turn_chat_persists_and_json_output_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                "HOME": str(Path(directory) / "home"),
                "XDG_RUNTIME_DIR": str(Path(directory) / "run"),
                "XDG_STATE_HOME": str(Path(directory) / "state"),
                "XDG_DATA_HOME": str(Path(directory) / "data"),
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
            },
        ):
            client = _FakeClient("chat answer")
            with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
                status, stdout, stderr = self.invoke(
                    "chat",
                    "local:latest",
                    "--message",
                    "hello",
                    "--json",
                    "--development-runtime",
                )
            self.assertEqual(status, 0)
            record = json.loads(stdout)
            self.assertEqual(record["message_count"], 2)
            self.assertEqual(record["result"]["content"], "chat answer")
            self.assertIn("conversation:", stderr)
            status, stdout, stderr = self.invoke("conversations", "list", "--json")
            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(json.loads(stdout)["count"], 1)

    def test_thinking_is_private_in_json_until_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                "HOME": str(Path(directory) / "home"),
                "XDG_RUNTIME_DIR": str(Path(directory) / "run"),
                "XDG_STATE_HOME": str(Path(directory) / "state"),
                "XDG_DATA_HOME": str(Path(directory) / "data"),
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
            },
        ):
            client = _FakeClient("answer", thinking="PRIVATE_CHAIN")
            with mock.patch("kilix_ollama.cli.OllamaClient", return_value=client):
                status, stdout, _stderr = self.invoke(
                    "chat",
                    "local:latest",
                    "--message",
                    "hello",
                    "--json",
                    "--development-runtime",
                )
            self.assertEqual(status, 0)
            turn = json.loads(stdout)
            self.assertNotIn("thinking", turn["result"])
            self.assertNotIn("PRIVATE_CHAIN", stdout)
            identifier = turn["conversation_id"]

            status, stdout, stderr = self.invoke(
                "conversations", "show", identifier, "--json"
            )
            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            self.assertNotIn("thinking", json.loads(stdout)["messages"][1])
            self.assertNotIn("PRIVATE_CHAIN", stdout)

            status, stdout, stderr = self.invoke(
                "conversations", "show", identifier, "--json", "--thinking"
            )
            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(
                json.loads(stdout)["messages"][1]["thinking"], "PRIVATE_CHAIN"
            )

    def test_pull_refuses_before_runtime_or_network_and_states_denominator(self) -> None:
        with mock.patch(
            "kilix_ollama.cli.OllamaClient", side_effect=AssertionError("runtime access")
        ):
            status, stdout, stderr = self.invoke("pull", "granite4.1:3b")
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("0 of 1", stderr)
        self.assertIn("carrier/receipt", stderr)

    def test_tool_registry_command_is_exact_and_disabled(self) -> None:
        status, stdout, stderr = self.invoke("tools", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(record["tool_count"], 4)
        self.assertFalse(record["enabled_by_default"])
        self.assertFalse(record["shell_tool"])


if __name__ == "__main__":
    unittest.main()
