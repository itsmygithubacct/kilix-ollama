from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

from kilix_ollama.catalog import (
    CURATED_MODELS,
    catalog_record,
    find_entry,
    project_library,
)
from kilix_ollama.errors import RuntimeContractError


class CatalogTests(unittest.TestCase):
    def test_exact_three_model_population_is_metadata_only_and_fail_closed(self) -> None:
        record = catalog_record()
        self.assertEqual(record["model_count"], 3)
        self.assertEqual(len(record["models"]), 3)
        self.assertEqual(record["payload_bytes"], 4_811_631_328)
        self.assertEqual(record["carrier_admitted_count"], 0)
        self.assertEqual(record["install_enabled_count"], 0)
        self.assertFalse(record["automatic_install"])
        self.assertFalse(record["weights_in_source_tree"])
        self.assertFalse(record["release_admission"])
        for model in record["models"]:
            self.assertFalse(model["carrier_admitted"])
            self.assertFalse(model["receipt_authority_available"])
            self.assertFalse(model["install_enabled"])
            self.assertFalse(model["release_admitted"])
            self.assertFalse(model["weights_in_source_tree"])

    def test_exact_roles_tags_and_manifest_digests_are_unique(self) -> None:
        self.assertEqual(
            [entry.role for entry in CURATED_MODELS],
            ["chat-tool", "embedding", "vision-tool"],
        )
        self.assertEqual(
            [entry.display_tag for entry in CURATED_MODELS],
            ["granite4.1:3b", "nomic-embed-text:v1.5", "granite3.2-vision:2b"],
        )
        self.assertEqual(len({entry.manifest_digest for entry in CURATED_MODELS}), 3)
        self.assertEqual(len({entry.catalog_id for entry in CURATED_MODELS}), 3)

    def test_catalog_import_and_record_generation_perform_no_io(self) -> None:
        with mock.patch("builtins.open", side_effect=AssertionError("filesystem access")), mock.patch(
            "socket.socket", side_effect=AssertionError("network access")
        ):
            record = catalog_record()
        self.assertEqual(record["model_count"], 3)

    def test_library_projection_separates_exact_mismatch_unknown_and_uncurated(self) -> None:
        installed = (
            {
                "name": "granite4.1:3b",
                "digest": CURATED_MODELS[0].manifest_digest.removeprefix("sha256:"),
            },
            {"name": "nomic-embed-text:v1.5", "digest": "f" * 64},
            {"name": "granite3.2-vision:2b"},
            {"name": "custom:latest", "digest": "e" * 64},
        )
        projection = project_library(installed)
        self.assertEqual(projection["curated_count"], 3)
        self.assertEqual(projection["installed_count"], 4)
        self.assertEqual(projection["installed_exact_count"], 1)
        self.assertEqual(
            [entry["local_state"] for entry in projection["curated"]],
            ["installed-exact", "installed-digest-mismatch", "installed-unverified"],
        )
        self.assertEqual(
            projection["uncurated_installed"][0]["local_state"],
            "installed-uncurated",
        )

    def test_duplicate_or_malformed_installed_records_fail_closed(self) -> None:
        with self.assertRaises(RuntimeContractError):
            project_library(({"name": "same"}, {"name": "same"}))
        for record in (1, {}, {"name": ""}, {"name": "x", "digest": "bad"}):
            with self.subTest(record=record):
                if record == {"name": "x", "digest": "bad"}:
                    projection = project_library((record,))
                    self.assertIsNone(projection["uncurated_installed"][0]["digest"])
                else:
                    with self.assertRaises(RuntimeContractError):
                        project_library((record,))  # type: ignore[arg-type]

    def test_entry_validation_rejects_identity_licence_and_payload_drift(self) -> None:
        original = CURATED_MODELS[0]
        variants = (
            replace(original, manifest_digest="sha256:bad"),
            replace(original, license_id="unknown"),
            replace(original, use_restrictions="unknown"),
            replace(original, payloads=()),
            replace(original, tasks=("made-up",)),
        )
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(RuntimeContractError):
                variant.record()

    def test_find_entry_accepts_only_exact_id_or_display_tag(self) -> None:
        entry = CURATED_MODELS[0]
        self.assertIs(find_entry(entry.catalog_id), entry)
        self.assertIs(find_entry(entry.display_tag), entry)
        for identity in ("granite4.1", "GRANITE4.1:3B", "", "unknown"):
            with self.subTest(identity=identity), self.assertRaises(RuntimeContractError):
                find_entry(identity)


if __name__ == "__main__":
    unittest.main()
