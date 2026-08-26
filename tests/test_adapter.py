from __future__ import annotations

import json
import unittest

from kilix_ollama.adapter import (
    CANDIDATE_VERSION,
    compatibility_for_version,
    decode_json_object,
    health_record,
    parse_model_list,
    parse_version,
)
from kilix_ollama.errors import UpstreamProtocolError


class AdapterTests(unittest.TestCase):
    def test_exact_candidate_remains_explicitly_unqualified(self) -> None:
        compatibility = compatibility_for_version(CANDIDATE_VERSION)
        self.assertEqual(compatibility.mode, "candidate-unqualified")
        self.assertFalse(compatibility.qualified)
        self.assertEqual(
            compatibility.allowed_routes,
            ("/api/ps", "/api/tags", "/api/version"),
        )

    def test_unknown_version_is_read_only(self) -> None:
        for version in ("0.32.14", "99.0.0"):
            with self.subTest(version=version):
                compatibility = compatibility_for_version(version)
                self.assertEqual(compatibility.mode, "read-only-diagnostics")
                self.assertNotIn("/api/chat", compatibility.allowed_routes)
                self.assertNotIn("/api/pull", compatibility.allowed_routes)

    def test_version_parser_is_bounded_and_strict(self) -> None:
        self.assertEqual(parse_version(b'{"version":"0.32.14"}'), "0.32.14")
        invalid = (
            b"[]",
            b"{}",
            b'{"version":1}',
            b'{"version":"x y"}',
            b'{"version":"x"} trailing',
            b'{"version":"a","version":"b"}',
            b'{"version":"x","metric":NaN}',
            b"\xff",
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(UpstreamProtocolError):
                parse_version(payload)

    def test_json_size_limit_is_enforced_before_parsing(self) -> None:
        with self.assertRaises(UpstreamProtocolError):
            decode_json_object(b'{"value":1}', limit=2)

    def test_model_lists_are_objects_and_counted(self) -> None:
        payload = json.dumps({"models": [{"name": "one"}, {"name": "two"}]}).encode()
        self.assertEqual(len(parse_model_list(payload, route="/api/tags")), 2)
        with self.assertRaises(UpstreamProtocolError):
            parse_model_list(b'{"models":[1]}', route="/api/ps")
        with self.assertRaises(UpstreamProtocolError):
            parse_model_list(payload, route="/api/show")

    def test_health_is_metadata_only(self) -> None:
        health = health_record(
            b'{"version":"0.32.14+plebian.security-license-r3"}',
            b'{"models":[{"name":"installed"}]}',
            b'{"models":[]}',
        )
        self.assertEqual(health["installed_model_count"], 1)
        self.assertEqual(health["running_model_count"], 0)
        self.assertFalse(health["inference_performed"])
        self.assertEqual(health["schema_status"], "development-unfrozen")
        self.assertEqual(health["compatibility"]["mode"], "candidate-unqualified")


if __name__ == "__main__":
    unittest.main()
