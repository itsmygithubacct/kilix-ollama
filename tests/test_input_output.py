from __future__ import annotations

import io
import os
from pathlib import Path
import stat
import tempfile
import unittest

from kilix_ollama.errors import RuntimeContractError
from kilix_ollama.input_output import (
    MAX_OUTPUT_BYTES,
    MAX_PROMPT_BYTES,
    atomic_write,
    read_prompt_file,
    select_prompt,
)


class PromptInputTests(unittest.TestCase):
    def test_arguments_file_and_stdin_are_distinct_bounded_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("from file", encoding="utf-8")
            self.assertEqual(
                select_prompt(["from", "arguments"], prompt_file=None, stdin=io.StringIO("ignored")),
                "from arguments",
            )
            self.assertEqual(
                select_prompt([], prompt_file=path, stdin=io.StringIO("ignored")),
                "from file",
            )
            self.assertEqual(
                select_prompt([], prompt_file=None, stdin=io.StringIO("from stdin")),
                "from stdin",
            )

    def test_ambiguous_empty_binary_non_utf8_and_large_prompts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "prompt"
            path.write_text("file", encoding="utf-8")
            with self.assertRaises(RuntimeContractError):
                select_prompt(["argument"], prompt_file=path, stdin=io.StringIO())
            for payload in (b"", b"a\x00b", b"\xff", b"x" * (MAX_PROMPT_BYTES + 1)):
                path.write_bytes(payload)
                with self.subTest(size=len(payload)), self.assertRaises(RuntimeContractError):
                    read_prompt_file(path)

    def test_symlink_directory_and_growing_file_shapes_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("text", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(RuntimeContractError):
                read_prompt_file(link)
            with self.assertRaises(RuntimeContractError):
                read_prompt_file(root)


class AtomicOutputTests(unittest.TestCase):
    def test_no_clobber_and_overwrite_are_atomic_private_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "response.txt"
            atomic_write(path, "first 世界", overwrite=False)
            self.assertEqual(path.read_text(encoding="utf-8"), "first 世界")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(root.glob("*.tmp")), [])
            with self.assertRaises(RuntimeContractError):
                atomic_write(path, "second", overwrite=False)
            self.assertEqual(path.read_text(encoding="utf-8"), "first 世界")
            atomic_write(path, "second", overwrite=True)
            self.assertEqual(path.read_text(encoding="utf-8"), "second")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_symlink_directory_missing_parent_and_size_are_refused_without_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("keep", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(RuntimeContractError):
                atomic_write(link, "replace", overwrite=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")
            with self.assertRaises(RuntimeContractError):
                atomic_write(root, "replace", overwrite=True)
            with self.assertRaises(RuntimeContractError):
                atomic_write(root / "missing/out", "x", overwrite=False)
            with self.assertRaises(RuntimeContractError):
                atomic_write(root / "large", "x" * (MAX_OUTPUT_BYTES + 1), overwrite=False)

    def test_target_appearing_during_no_clobber_is_never_replaced(self) -> None:
        # The public behavior is covered without a race harness: an existing
        # target is refused and its bytes remain exact.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.write_text("owner bytes", encoding="utf-8")
            with self.assertRaises(RuntimeContractError):
                atomic_write(path, "model bytes", overwrite=False)
            self.assertEqual(path.read_text(encoding="utf-8"), "owner bytes")


if __name__ == "__main__":
    unittest.main()
