from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from kilix_ollama.errors import ToolError
from kilix_ollama.tools import (
    MAX_SEARCH_RESULTS,
    MAX_TOOL_FILE_BYTES,
    ToolRegistry,
)


def call(name: str, arguments: object, **extra: object) -> dict[str, object]:
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        **extra,
    }


class ToolRegistryTests(unittest.TestCase):
    def test_registry_has_exact_four_read_only_disabled_by_default_tools(self) -> None:
        record = ToolRegistry().record()
        self.assertEqual(record["tool_count"], 4)
        self.assertEqual(len(record["tools"]), 4)
        self.assertFalse(record["enabled_by_default"])
        self.assertFalse(record["persistent_approval"])
        self.assertFalse(record["write_tools"])
        self.assertFalse(record["network_tools"])
        self.assertFalse(record["process_tools"])
        self.assertFalse(record["shell_tool"])
        names = [tool["function"]["name"] for tool in record["tools"]]
        self.assertEqual(
            names,
            [
                "kilix.current_time",
                "kilix.calculator",
                "kilix.read_selected_file",
                "kilix.search_selected_text",
            ],
        )

    def test_default_or_explicit_denial_runs_no_handler(self) -> None:
        invoked = False

        def clock() -> datetime:
            nonlocal invoked
            invoked = True
            return datetime.now(timezone.utc)

        registry = ToolRegistry(clock=clock)
        outcome = registry.execute(call("kilix.current_time", {}), approve=None)
        self.assertFalse(outcome.approved)
        self.assertFalse(invoked)
        self.assertEqual(json.loads(outcome.content)["status"], "denied")
        outcome = registry.execute(
            call("kilix.current_time", {}), approve=lambda _preview: False
        )
        self.assertFalse(outcome.approved)
        self.assertFalse(invoked)

    def test_current_time_is_utc_fixed_and_requires_per_call_approval(self) -> None:
        now = datetime(2026, 8, 31, 12, 34, 56, tzinfo=timezone.utc)
        previews = []
        outcome = ToolRegistry(clock=lambda: now).execute(
            call("kilix.current_time", {"timezone": "UTC"}),
            approve=lambda preview: previews.append(preview) or True,
        )
        self.assertTrue(outcome.approved)
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].risk, "low")
        self.assertEqual(
            json.loads(outcome.content)["result"]["value"],
            "2026-08-31T12:34:56Z",
        )
        with self.assertRaises(ToolError):
            ToolRegistry().preview(call("kilix.current_time", {"timezone": "Europe/London"}))

    def test_calculator_allows_only_bounded_arithmetic(self) -> None:
        registry = ToolRegistry()
        valid = {
            "1 + 2 * 3": 7,
            "(10 - 4) / 3": 2.0,
            "2 ** 10": 1024,
            "17 % 5": 2,
        }
        for expression, expected in valid.items():
            with self.subTest(expression=expression):
                outcome = registry.execute(
                    call("kilix.calculator", {"expression": expression}),
                    approve=lambda _preview: True,
                )
                self.assertEqual(json.loads(outcome.content)["result"]["result"], expected)
        invalid = (
            "__import__('os').system('id')",
            "open('/etc/passwd')",
            "[1, 2, 3]",
            "2 ** 1000",
            "1 / 0",
        )
        for expression in invalid:
            with self.subTest(expression=expression[:20]):
                outcome = registry.execute(
                    call("kilix.calculator", {"expression": expression}),
                    approve=lambda _preview: True,
                )
                self.assertEqual(json.loads(outcome.content)["status"], "error")
        with self.assertRaises(ToolError):
            registry.execute(
                call("kilix.calculator", {"expression": "9" * 300}),
                approve=lambda _preview: True,
            )

    def test_read_file_is_bounded_utf8_and_beneath_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            path = nested / "note.txt"
            path.write_text("hello 世界", encoding="utf-8")
            registry = ToolRegistry((root,))
            preview = registry.preview(
                call(
                    "kilix.read_selected_file",
                    {"selection": 0, "path": "nested/note.txt"},
                )
            )
            self.assertEqual(preview.affected_resources, ("selection[0]/nested/note.txt",))
            outcome = registry.execute(
                call(
                    "kilix.read_selected_file",
                    {"selection": 0, "path": "nested/note.txt"},
                ),
                approve=lambda _preview: True,
            )
            result = json.loads(outcome.content)["result"]
            self.assertEqual(result["content"], "hello 世界")
            self.assertEqual(result["path"], "nested/note.txt")

    def test_file_tool_refuses_escape_symlink_binary_large_and_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "selected"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.write_text("secret", encoding="utf-8")
            (root / "link").symlink_to(outside)
            (root / "binary").write_bytes(b"a\x00b")
            (root / "large").write_bytes(b"x" * (MAX_TOOL_FILE_BYTES + 1))
            registry = ToolRegistry((root,))
            invalid = ("../outside", "link", "binary", "large", "missing")
            for relative in invalid:
                with self.subTest(relative=relative):
                    if relative in ("binary", "large"):
                        outcome = registry.execute(
                            call(
                                "kilix.read_selected_file",
                                {"selection": 0, "path": relative},
                            ),
                            approve=lambda _preview: True,
                        )
                        self.assertEqual(json.loads(outcome.content)["status"], "error")
                    else:
                        with self.assertRaises(ToolError):
                            registry.preview(
                                call(
                                    "kilix.read_selected_file",
                                    {"selection": 0, "path": relative},
                                )
                            )
            with self.assertRaises(ToolError):
                ToolRegistry().preview(
                    call(
                        "kilix.read_selected_file",
                        {"selection": 0, "path": "anything"},
                    )
                )

    def test_search_is_case_insensitive_bounded_and_strips_terminal_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(5):
                (root / f"file-{index}.txt").write_text(
                    f"line\nNeedle {index}\x1b[31m\n",
                    encoding="utf-8",
                )
            registry = ToolRegistry((root,))
            outcome = registry.execute(
                call(
                    "kilix.search_selected_text",
                    {
                        "selection": 0,
                        "path": ".",
                        "query": "needle",
                        "max_results": 3,
                    },
                ),
                approve=lambda _preview: True,
            )
            result = json.loads(outcome.content)["result"]
            self.assertEqual(len(result["matches"]), 3)
            self.assertTrue(result["truncated"])
            self.assertNotIn("\x1b", json.dumps(result))

    def test_unknown_duplicate_extra_and_wrong_typed_arguments_fail_before_approval(self) -> None:
        registry = ToolRegistry()
        invalid = (
            call("unknown", {}),
            call("kilix.calculator", {}),
            call("kilix.calculator", {"expression": "1", "extra": True}),
            call("kilix.calculator", {"expression": 1}),
            call("kilix.current_time", [], unexpected=True),
            {"type": "other", "function": {"name": "kilix.current_time", "arguments": {}}},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ToolError):
                registry.execute(value, approve=lambda _preview: True)

    def test_approval_preview_refuses_terminal_spoofing_controls(self) -> None:
        registry = ToolRegistry()
        approved = False

        def approve(_preview) -> bool:
            nonlocal approved
            approved = True
            return True

        for expression in ("1\u202e+2", "1\x9b+2", "1\x1b[31m+2"):
            with self.subTest(expression=repr(expression)), self.assertRaises(ToolError):
                registry.execute(
                    call("kilix.calculator", {"expression": expression}),
                    approve=approve,
                )
        self.assertFalse(approved)

    def test_intermediate_symlink_swap_after_approval_cannot_escape_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "selected"
            nested = root / "nested"
            outside = base / "outside"
            nested.mkdir(parents=True)
            outside.mkdir()
            (nested / "note.txt").write_text("INSIDE", encoding="utf-8")
            (outside / "note.txt").write_text("OUTSIDE_SELECTION", encoding="utf-8")
            registry = ToolRegistry((root,))

            def swap(_preview) -> bool:
                nested.rename(root / "approved-original")
                nested.symlink_to(outside, target_is_directory=True)
                return True

            outcome = registry.execute(
                call(
                    "kilix.read_selected_file",
                    {"selection": 0, "path": "nested/note.txt"},
                ),
                approve=swap,
            )
            self.assertEqual(json.loads(outcome.content)["status"], "error")
            self.assertNotIn("OUTSIDE_SELECTION", outcome.content)

    def test_selected_root_count_and_search_result_bounds_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ToolError):
                ToolRegistry(tuple(root for _ in range(17)))
            registry = ToolRegistry((root,))
            for maximum in (0, MAX_SEARCH_RESULTS + 1, True, "2"):
                with self.subTest(maximum=maximum), self.assertRaises(ToolError):
                    registry.preview(
                        call(
                            "kilix.search_selected_text",
                            {
                                "selection": 0,
                                "path": ".",
                                "query": "x",
                                "max_results": maximum,
                            },
                        )
                    )


if __name__ == "__main__":
    unittest.main()
