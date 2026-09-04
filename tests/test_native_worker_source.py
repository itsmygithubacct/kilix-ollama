"""`native/` carries sources this repository never builds; keep them bound anyway.

Nothing compiles `native/`, so without this test an edit there is invisible: the
suite stays green and the source silently diverges from the binary plebian-os
ships. Each check below fails on absence as well as on change -- a manifest that
only validated the files it happened to find would not notice one being deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
MANIFEST = NATIVE / "SOURCE-MANIFEST.json"
SCHEMA = "kilix.ollama.native-source-manifest/v1"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class NativeWorkerSourceTests(unittest.TestCase):
    def test_manifest_is_present_and_declares_its_schema(self) -> None:
        self.assertTrue(MANIFEST.is_file(), f"{MANIFEST} is absent")
        self.assertEqual(_manifest().get("schema"), SCHEMA)

    def test_every_declared_source_is_present_and_matches_its_digest(self) -> None:
        sources = _manifest()["sources"]
        self.assertGreater(len(sources), 0, "the manifest declares no sources")
        for source in sources:
            path = NATIVE / source["path"]
            with self.subTest(source=source["path"]):
                self.assertTrue(path.is_file(), f"declared source is absent: {path}")
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(
                    observed,
                    source["sha256"],
                    f"{source['path']} changed without SOURCE-MANIFEST.json changing with it",
                )

    def test_no_undeclared_source_sits_in_native(self) -> None:
        """Closure the other way: an added file must be declared, not just tolerated."""
        declared = {source["path"] for source in _manifest()["sources"]}
        documentation = {"README.md", "SOURCE-MANIFEST.json"}
        observed = {
            str(path.relative_to(NATIVE))
            for path in NATIVE.rglob("*")
            if path.is_file()
        } - documentation
        self.assertEqual(
            observed,
            declared,
            "native/ and SOURCE-MANIFEST.json disagree about which sources exist",
        )

    def test_declared_runtime_closure_matches_the_runtime_constant(self) -> None:
        """The source and the runtime must name the same closure, or one has moved."""
        runtime = (ROOT / "src" / "kilix_ollama" / "tts_runtime.py").read_text(encoding="utf-8")
        for source in _manifest()["sources"]:
            constant = source["runtime_constant"]
            with self.subTest(constant=constant):
                match = re.search(
                    rf'^{re.escape(constant)}\s*=\s*\(?\s*"([^"]+)"', runtime, re.MULTILINE
                )
                self.assertIsNotNone(match, f"{constant} is absent from tts_runtime.py")
                self.assertEqual(match.group(1), source["runtime_closure"])

    def test_declared_install_path_is_the_one_the_runtime_resolves(self) -> None:
        runtime = (ROOT / "src" / "kilix_ollama" / "tts_runtime.py").read_text(encoding="utf-8")
        for source in _manifest()["sources"]:
            with self.subTest(source=source["path"]):
                self.assertIn(source["installed_as"], runtime)


if __name__ == "__main__":
    unittest.main()
