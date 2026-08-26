from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import urllib.error

from kilix_ollama.errors import InstallError
from kilix_ollama.paths import RuntimePaths
from kilix_ollama import tts_artifacts


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


class CatalogTests(unittest.TestCase):
    def test_catalog_is_exact_metadata_and_contains_no_product_voice(self) -> None:
        record = tts_artifacts.catalog_record()
        self.assertEqual(record["schema"], "kilix.ollama.optional-tts/v0-unfrozen")
        self.assertFalse(record["automatic_install"])
        self.assertFalse(record["kilix_mirror"])
        self.assertFalse(record["product_voice_included"])
        model = record["models"][0]
        self.assertEqual(model["source_revision"], tts_artifacts.SOURCE_REVISION)
        self.assertEqual(model["source_bytes"], 219_088_535)
        self.assertEqual(model["runtime_bytes"], 123_629_056)
        self.assertEqual(model["license"], "CC-BY-4.0")
        self.assertEqual(
            tts_artifacts.CONVERTER_PATH,
            Path("/usr/lib/plebian-os/ollama/pocket-tts-convert-hf-to-gguf"),
        )
        self.assertEqual(
            model["converter_closure"],
            "candidate:plebian-os-llama.cpp-b10434-pocket-q8_0-r1",
        )
        self.assertEqual(
            [artifact["sha256"] for artifact in model["derived_artifacts"]],
            [
                "87c310bfeae4ba8a9e4016c01114013823d08451d10292c2bd64a8e7b9b529c6",
                "174653c6675289fc1a12f6fbd60a6352741cf95ce71c94f3dc1d42dd90a62b00",
            ],
        )

    def test_catalog_read_does_not_touch_filesystem_or_network(self) -> None:
        with mock.patch("pathlib.Path.open", side_effect=AssertionError("I/O")), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            record = tts_artifacts.catalog_record()
        self.assertEqual(record["models"][0]["catalog_id"], tts_artifacts.MODEL_ID)


class InstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = private_paths(self.root)
        self.payloads = {
            "model.safetensors": b"source model fixture",
            "tokenizer.model": b"tokenizer fixture",
            "README.md": b"pinned model card fixture\n",
            "tiny-q8.gguf": b"converted model fixture",
            "tiny-mmproj-q8.gguf": b"converted projector fixture",
        }

        def artifact(role: str, filename: str, *, url_name: str | None = None):
            payload_name = filename if url_name is None else url_name
            payload = self.payloads[payload_name]
            return tts_artifacts.ArtifactSpec(
                role=role,
                filename=filename,
                url=(
                    f"https://example.invalid/{payload_name}"
                    if url_name is not None or role.endswith("source")
                    else None
                ),
                size=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )

        self.model = tts_artifacts.TtsModelSpec(
            catalog_id="tiny-pocket-q8",
            source_revision="0" * 40,
            source_artifacts=(
                artifact("model-source", "model.safetensors"),
                artifact("tokenizer-source", "tokenizer.model"),
            ),
            notice_artifact=artifact("model-card", "MODEL_CARD.md", url_name="README.md"),
            derived_artifacts=(
                artifact("runtime-model-q8_0", "tiny-q8.gguf"),
                artifact("runtime-projector-q8_0", "tiny-mmproj-q8.gguf"),
            ),
            converter_closure="fixture-converter-r1",
        )
        model_map = mock.patch.dict(
            tts_artifacts.MODEL_BY_ID,
            {self.model.catalog_id: self.model},
            clear=False,
        )
        model_map.start()
        self.addCleanup(model_map.stop)

    def fake_download(
        self,
        url: str,
        destination: str,
        report,
        expected_sha256: str,
    ) -> str:
        name = url.rsplit("/", 1)[-1]
        payload = self.payloads[name]
        self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)
        Path(destination).write_bytes(payload)
        report(f"fixture download {name}")
        return destination

    def fake_convert(self, source_dir, runtime_dir, model, report) -> None:
        self.assertEqual((source_dir / "model.safetensors").read_bytes(), self.payloads["model.safetensors"])
        self.assertEqual((source_dir / "tokenizer.model").read_bytes(), self.payloads["tokenizer.model"])
        for output in model.derived_artifacts:
            (runtime_dir / output.filename).write_bytes(self.payloads[output.filename])
        report("fixture conversion")

    def install(self, **overrides):
        arguments = {
            "paths": self.paths,
            "accept_terms": True,
            "downloader": self.fake_download,
            "converter": self.fake_convert,
            "minimum_free_bytes": 0,
        }
        arguments.update(overrides)
        return tts_artifacts.install_model(self.model.catalog_id, **arguments)

    def test_terms_are_required_before_io(self) -> None:
        downloader = mock.Mock(side_effect=AssertionError("download attempted"))
        converter = mock.Mock(side_effect=AssertionError("conversion attempted"))
        with self.assertRaises(InstallError) as caught:
            tts_artifacts.install_model(
                self.model.catalog_id,
                paths=self.paths,
                downloader=downloader,
                converter=converter,
                minimum_free_bytes=0,
            )
        self.assertIn("terms were not accepted", str(caught.exception))
        downloader.assert_not_called()
        converter.assert_not_called()
        self.assertFalse(self.paths.data_dir.exists())

    def test_install_stages_converts_verifies_and_publishes_atomically(self) -> None:
        reports: list[str] = []
        target, created = self.install(report=reports.append)
        self.assertTrue(created)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
        self.assertEqual((target / "MODEL_CARD.md").read_bytes(), self.payloads["README.md"])
        self.assertEqual(
            (target / "runtime/tiny-q8.gguf").read_bytes(),
            self.payloads["tiny-q8.gguf"],
        )
        self.assertEqual(
            (target / "source/model.safetensors").read_bytes(),
            self.payloads["model.safetensors"],
        )
        for path in target.rglob("*"):
            expected_mode = 0o700 if path.is_dir() else 0o600
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected_mode)
        manifest = json.loads((target / tts_artifacts.MANIFEST_NAME).read_text())
        self.assertTrue(manifest["sources_retained"])
        self.assertTrue(
            tts_artifacts.inspect_install(
                self.model.catalog_id,
                paths=self.paths,
                verify_hashes=True,
            )[0]
        )
        self.assertEqual(list(target.parent.glob(".*.stage-*")), [])
        self.assertIn("fixture conversion", reports)

    def test_verified_install_is_idempotent_without_download_or_conversion(self) -> None:
        first, created = self.install()
        self.assertTrue(created)
        second, created = self.install(
            downloader=mock.Mock(side_effect=AssertionError("download attempted")),
            converter=mock.Mock(side_effect=AssertionError("conversion attempted")),
        )
        self.assertFalse(created)
        self.assertEqual(first, second)

    def test_invalid_existing_target_is_never_replaced(self) -> None:
        target = self.paths.data_dir / "tts-models" / self.model.catalog_id
        target.mkdir(parents=True, mode=0o700)
        target.chmod(0o700)
        marker = target / "keep"
        marker.write_text("user data")
        with self.assertRaises(InstallError) as caught:
            self.install()
        self.assertIn("refusing to replace", str(caught.exception))
        self.assertEqual(marker.read_text(), "user data")

    def test_failed_conversion_leaves_no_partial_install(self) -> None:
        def fail(*_arguments):
            raise RuntimeError("conversion fixture failed")

        with self.assertRaises(RuntimeError):
            self.install(converter=fail)
        target = self.paths.data_dir / "tts-models" / self.model.catalog_id
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.glob(".*.stage-*")), [])

    def test_missing_default_converter_fails_before_downloading(self) -> None:
        downloader = mock.Mock(side_effect=AssertionError("download attempted"))
        with mock.patch.object(
            tts_artifacts,
            "_validate_converter",
            side_effect=InstallError("converter fixture missing"),
        ), self.assertRaises(InstallError) as caught:
            tts_artifacts.install_model(
                self.model.catalog_id,
                paths=self.paths,
                accept_terms=True,
                downloader=downloader,
                minimum_free_bytes=0,
            )
        self.assertIn("converter fixture missing", str(caught.exception))
        downloader.assert_not_called()

    def test_discard_source_removes_inputs_and_content_cache(self) -> None:
        target, _created = self.install(discard_verified_source=True)
        self.assertFalse((target / "source").exists())
        self.assertTrue((target / "MODEL_CARD.md").is_file())
        manifest = json.loads((target / tts_artifacts.MANIFEST_NAME).read_text())
        self.assertFalse(manifest["sources_retained"])
        cache_root = self.paths.cache_dir / "artifacts/sha256"
        self.assertEqual(list(cache_root.iterdir()), [])
        self.assertTrue(
            tts_artifacts.inspect_install(
                self.model.catalog_id,
                paths=self.paths,
                verify_hashes=True,
            )[0]
        )

    def test_tampering_and_symlink_substitution_fail_closed(self) -> None:
        target, _created = self.install()
        runtime_model = target / "runtime/tiny-q8.gguf"
        runtime_model.unlink()
        runtime_model.symlink_to(target / "MODEL_CARD.md")
        installed, detail = tts_artifacts.inspect_install(
            self.model.catalog_id,
            paths=self.paths,
            verify_hashes=True,
        )
        self.assertFalse(installed)
        self.assertIn("not a regular file", detail)

    def test_access_denial_points_to_upstream_without_mirror_fallback(self) -> None:
        def denied(*_arguments, **_kwargs):
            raise urllib.error.HTTPError("https://example.invalid", 403, "denied", {}, None)

        with self.assertRaises(InstallError) as caught:
            self.install(downloader=denied)
        message = str(caught.exception)
        self.assertIn("Review or accept", message)
        self.assertIn("neither stores account credentials", message)
        self.assertIn("unapproved mirror", message)

    def test_remove_requires_confirmation_and_refuses_unexpected_content(self) -> None:
        target, _created = self.install()
        with self.assertRaises(InstallError):
            tts_artifacts.remove_model(
                self.model.catalog_id,
                paths=self.paths,
            )
        extra = target / "do-not-delete"
        extra.write_text("user data")
        with self.assertRaises(InstallError) as caught:
            tts_artifacts.remove_model(
                self.model.catalog_id,
                paths=self.paths,
                confirm=True,
            )
        self.assertIn("refusing to remove", str(caught.exception))
        self.assertTrue(extra.exists())
        self.assertEqual(
            list(target.parent.glob(f".{self.model.catalog_id}.remove-*")),
            [],
        )

    def test_confirmed_remove_deletes_only_verified_install(self) -> None:
        target, _created = self.install()
        removed_target, removed = tts_artifacts.remove_model(
            self.model.catalog_id,
            paths=self.paths,
            confirm=True,
        )
        self.assertTrue(removed)
        self.assertEqual(removed_target, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
