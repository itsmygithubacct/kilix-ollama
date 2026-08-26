from __future__ import annotations

from pathlib import Path
import unittest

import kilix_ollama


ROOT = Path(__file__).resolve().parents[1]
MODEL_SUFFIXES = {".gguf", ".safetensors", ".ckpt", ".onnx", ".pt", ".pth"}


class PackagingTests(unittest.TestCase):
    def test_version_sources_match(self) -> None:
        self.assertEqual((ROOT / "VERSION").read_text(encoding="utf-8").strip(), kilix_ollama.__version__)

    def test_launcher_is_executable(self) -> None:
        self.assertTrue((ROOT / "kilix-ollama").stat().st_mode & 0o111)

    def test_no_model_or_weight_artifact_is_present(self) -> None:
        artifacts = [
            path
            for path in ROOT.rglob("*")
            if ".git" not in path.parts and path.is_file() and path.suffix.casefold() in MODEL_SUFFIXES
        ]
        self.assertEqual(artifacts, [])


if __name__ == "__main__":
    unittest.main()
