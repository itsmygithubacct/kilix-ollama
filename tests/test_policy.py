from __future__ import annotations

import unittest

from kilix_ollama.errors import PolicyDenied
from kilix_ollama.policy import (
    MAX_HEADER_VALUE_BYTES,
    MAX_REQUEST_BODY,
    ROUTE_METHODS,
    authorize_route,
    bounded_body,
    forward_request_headers,
    forward_response_headers,
    policy_record,
    verify_peer,
)


class RoutePolicyTests(unittest.TestCase):
    def test_every_declared_method_is_admitted(self) -> None:
        for route, methods in ROUTE_METHODS.items():
            for method in methods:
                with self.subTest(route=route, method=method):
                    self.assertEqual(authorize_route(method, route), route)

    def test_cloud_account_push_browser_and_compatibility_routes_are_denied(self) -> None:
        denied = (
            "/api/push",
            "/api/me",
            "/api/experimental/web_search",
            "/api/experimental/web_fetch",
            "/api/experimental/model-recommendations",
            "/v1/messages",
            "/v1/chat/completions",
            "/",
        )
        for route in denied:
            with self.subTest(route=route), self.assertRaises(PolicyDenied):
                authorize_route("POST", route)

    def test_ambiguous_or_remote_targets_are_denied(self) -> None:
        targets = (
            "http://127.0.0.1:11434/api/version",
            "//127.0.0.1/api/version",
            "/api/version?x=1",
            "/api/version#fragment",
            "/api/%76ersion",
            "/api\\version",
            "/api//version",
            "/api/../version",
            "/api/version\nInjected: yes",
            "//[invalid/api/version",
        )
        for target in targets:
            with self.subTest(target=target), self.assertRaises(PolicyDenied):
                authorize_route("GET", target)
        with self.assertRaises(PolicyDenied):
            authorize_route("GET", "/api/version\ud800")

    def test_method_is_exact_and_typed(self) -> None:
        for method in ("get", "GET ", "G3T", "GÉT", 1, None):
            with self.subTest(method=method), self.assertRaises(PolicyDenied):
                authorize_route(method, "/api/version")  # type: ignore[arg-type]
        with self.assertRaises(PolicyDenied):
            authorize_route("POST", "/api/version")

    def test_policy_record_is_explicitly_unfrozen(self) -> None:
        record = policy_record()
        self.assertEqual(record["schema_status"], "development-unfrozen")
        self.assertFalse(record["cloud_fallback"])
        self.assertEqual(record["unknown_routes"], "deny")


class MessagePolicyTests(unittest.TestCase):
    def test_body_limit_is_exact(self) -> None:
        body = b"x" * MAX_REQUEST_BODY
        self.assertIs(bounded_body(body), body)
        with self.assertRaises(PolicyDenied):
            bounded_body(body + b"x")
        with self.assertRaises(PolicyDenied):
            bounded_body("text")  # type: ignore[arg-type]

    def test_only_content_headers_cross_request_boundary(self) -> None:
        forwarded = forward_request_headers(
            {
                "Accept": " application/json ",
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
                "Cookie": "secret",
                "Origin": "https://host.invalid",
                "X-Forwarded-For": "203.0.113.1",
            }
        )
        self.assertEqual(
            forwarded,
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "kilix-ollama/0.2.1.dev0",
            },
        )

    def test_header_smuggling_and_normalized_duplicates_are_denied(self) -> None:
        with self.assertRaises(PolicyDenied):
            forward_request_headers({"Content-Type": "x\r\nInjected: yes"})
        with self.assertRaises(PolicyDenied):
            forward_request_headers({"Accept": "a", "accept": "b"})
        with self.assertRaises(PolicyDenied):
            forward_response_headers({"Content-Type": "a", "content-type": "b"})
        with self.assertRaises(PolicyDenied):
            forward_request_headers({"Accept": "x" * (MAX_HEADER_VALUE_BYTES + 1)})
        with self.assertRaises(PolicyDenied):
            forward_response_headers({"Content-Type": "application/json\x00"})
        with self.assertRaises(PolicyDenied):
            forward_request_headers({"Accept": "text/é"})

    def test_response_headers_are_minimal(self) -> None:
        self.assertEqual(
            forward_response_headers(
                {
                    "Content-Type": "application/x-ndjson",
                    "Set-Cookie": "secret=yes",
                    "Access-Control-Allow-Origin": "*",
                }
            ),
            {"Content-Type": "application/x-ndjson"},
        )

    def test_peer_uid_must_be_exact_integer(self) -> None:
        verify_peer(1000, owner_uid=1000)
        for peer in (1001, "1000", True):
            with self.subTest(peer=peer), self.assertRaises(PolicyDenied):
                verify_peer(peer, owner_uid=1000)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
