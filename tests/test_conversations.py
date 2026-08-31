from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from kilix_ollama.conversations import (
    ConversationStore,
    MAX_MESSAGE_BYTES,
    validate_message,
)
from kilix_ollama.errors import ConversationError
from kilix_ollama.paths import RuntimePaths


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


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = private_paths(self.root)
        self.now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
        identifiers = iter(("1" * 32, "2" * 32, "3" * 32))
        self.store = ConversationStore(
            self.paths,
            clock=lambda: self.now,
            identifier_factory=lambda: next(identifiers),
        )

    def test_create_load_append_and_list_use_owner_private_atomic_files(self) -> None:
        conversation = self.store.create(model="local:latest", title="First")
        self.assertEqual(conversation.identifier, "1" * 32)
        path = self.store.root / f"{conversation.identifier}.json"
        self.assertEqual(stat.S_IMODE(self.store.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE((self.store.root / ".lock").stat().st_mode), 0o600)

        self.now += timedelta(seconds=1)
        updated = self.store.append(
            conversation.identifier,
            (
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world", "thinking": "private"},
            ),
        )
        self.assertEqual(len(updated.messages), 2)
        self.assertEqual(self.store.load(conversation.identifier), updated)
        self.assertNotIn("thinking", updated.record()["messages"][1])
        self.assertEqual(
            updated.record(include_thinking=True)["messages"][1]["thinking"],
            "private",
        )
        self.assertEqual(
            updated.storage_record()["messages"][1]["thinking"], "private"
        )
        listed = self.store.list()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], conversation.identifier)
        self.assertEqual(listed[0]["message_count"], 2)
        self.assertEqual(list(self.store.root.glob("*.tmp")), [])

    def test_list_nonexistent_store_is_empty_and_side_effect_free(self) -> None:
        self.assertEqual(self.store.list(), ())
        self.assertFalse(self.store.root.exists())

    def test_multiple_conversations_are_sorted_by_update_time(self) -> None:
        first = self.store.create(model="one", title="First")
        self.now += timedelta(seconds=1)
        second = self.store.create(model="two", title="Second")
        self.assertEqual(
            [record["id"] for record in self.store.list()],
            [second.identifier, first.identifier],
        )
        self.now += timedelta(seconds=1)
        self.store.append(first.identifier, ({"role": "user", "content": "new"},))
        self.assertEqual(
            [record["id"] for record in self.store.list()],
            [first.identifier, second.identifier],
        )

    def test_markdown_export_strips_terminal_controls_and_hides_thinking_by_default(self) -> None:
        conversation = self.store.create(model="local", title="Unsafe\x1b]0;title\x07Safe")
        self.store.append(
            conversation.identifier,
            (
                {
                    "role": "assistant",
                    "content": "before\x1b[31mred\x1b[0mafter",
                    "thinking": "secret thought",
                    "tool_calls": [
                        {"function": {"name": "tool", "arguments": {}}}
                    ],
                },
            ),
        )
        ordinary = self.store.export_markdown(conversation.identifier)
        self.assertNotIn("\x1b", ordinary)
        self.assertNotIn("secret thought", ordinary)
        self.assertIn("beforeredafter", ordinary)
        self.assertIn("Tool-call metadata omitted", ordinary)
        explicit = self.store.export_markdown(
            conversation.identifier, include_thinking=True
        )
        self.assertIn("secret thought", explicit)

    def test_delete_removes_only_the_exact_verified_conversation(self) -> None:
        conversation = self.store.create(model="local", title="Delete")
        unrelated = self.store.root / "keep.txt"
        unrelated.write_text("keep", encoding="utf-8")
        self.store.delete(conversation.identifier)
        self.assertFalse((self.store.root / f"{conversation.identifier}.json").exists())
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")
        with self.assertRaises(ConversationError):
            self.store.delete(conversation.identifier)

    def test_symlink_wrong_mode_and_duplicate_json_are_refused(self) -> None:
        conversation = self.store.create(model="local", title="Safe")
        path = self.store.root / f"{conversation.identifier}.json"
        original = path.read_bytes()
        path.chmod(0o644)
        with self.assertRaises(ConversationError):
            self.store.load(conversation.identifier)
        path.chmod(0o600)

        path.write_bytes(
            original.replace(b'"title":"Safe"', b'"title":"Safe","title":"Other"')
        )
        path.chmod(0o600)
        with self.assertRaises(ConversationError):
            self.store.load(conversation.identifier)

        path.unlink()
        target = self.root / "target"
        target.write_bytes(original)
        target.chmod(0o600)
        path.symlink_to(target)
        with self.assertRaises(ConversationError):
            self.store.load(conversation.identifier)

    def test_embedded_id_and_schema_tampering_are_refused(self) -> None:
        conversation = self.store.create(model="local", title="Safe")
        path = self.store.root / f"{conversation.identifier}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        for key, value in (("id", "f" * 32), ("schema", "other"), ("extra", True)):
            with self.subTest(key=key):
                tampered = dict(record)
                tampered[key] = value
                path.write_text(json.dumps(tampered), encoding="utf-8")
                path.chmod(0o600)
                with self.assertRaises(ConversationError):
                    self.store.load(conversation.identifier)

    def test_unexpected_directory_entry_and_unsafe_root_are_refused(self) -> None:
        self.store.create(model="local", title="Safe")
        (self.store.root / "unexpected").write_text("x", encoding="utf-8")
        with self.assertRaises(ConversationError):
            self.store.list()
        (self.store.root / "unexpected").unlink()
        self.store.root.chmod(0o755)
        with self.assertRaises(ConversationError):
            self.store.list()

    def test_message_schema_role_tool_and_size_limits_are_strict(self) -> None:
        invalid = (
            {"role": "unknown", "content": "x"},
            {"role": "user", "content": "x", "extra": True},
            {"role": "user", "content": "x", "tool_calls": []},
            {"role": "assistant", "content": "x", "tool_name": "tool"},
            {"role": "tool", "content": "x"},
            {"role": "tool", "content": "x", "tool_name": "bad name"},
            {"role": "user", "content": "x" * (MAX_MESSAGE_BYTES + 1)},
        )
        for message in invalid:
            with self.subTest(message=list(message)), self.assertRaises(ConversationError):
                validate_message(message)

    def test_missing_invalid_identifier_and_naive_clock_fail_closed(self) -> None:
        for identifier in ("", "../escape", "g" * 32, "1" * 31):
            with self.subTest(identifier=identifier), self.assertRaises(ConversationError):
                self.store.load(identifier)
        bad = ConversationStore(
            self.paths,
            clock=lambda: datetime(2026, 8, 31),
            identifier_factory=lambda: "4" * 32,
        )
        with self.assertRaises(ConversationError):
            bad.create(model="local", title="Bad clock")


if __name__ == "__main__":
    unittest.main()
