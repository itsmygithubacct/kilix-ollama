from __future__ import annotations

import base64
import http.server
import json
import os
from pathlib import Path
import socketserver
import tempfile
import threading
import unittest
from unittest import mock

from kilix_ollama.adapter import CANDIDATE_VERSION
from kilix_ollama.arbiter import AcceleratorArbiter
from kilix_ollama.client import OllamaClient, validate_model_name
from kilix_ollama.errors import (
    RuntimeContractError,
    RuntimeUnavailable,
    UpstreamProtocolError,
)
from kilix_ollama.mediation import MediationProxy
from kilix_ollama.paths import RuntimePaths
from tests.image_fixtures import png_image


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


def encoded(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def stream(*values: object) -> bytes:
    return b"".join(encoded(value) + b"\n" for value in values)


class _ClientFixtureHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._serve()

    def do_POST(self) -> None:
        self._serve()

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with self.server.request_lock:  # type: ignore[attr-defined]
            self.server.requests.append((self.command, self.path, body))  # type: ignore[attr-defined]
        status, content_type, payload = self.server.responses.get(  # type: ignore[attr-defined]
            self.path,
            (404, "application/json", b'{"error":"missing"}'),
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_arguments) -> None:
        pass


class OllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = private_paths(self.root)
        self.paths.runtime_dir.mkdir(parents=True, mode=0o700)
        self.paths.runtime_dir.chmod(0o700)
        self.upstream = socketserver.UnixStreamServer(
            str(self.paths.upstream_socket), _ClientFixtureHandler
        )
        self.paths.upstream_socket.chmod(0o600)
        self.upstream.requests = []
        self.upstream.request_lock = threading.Lock()
        self.upstream.responses = {
            "/api/version": (
                200,
                "application/json",
                encoded({"version": CANDIDATE_VERSION}),
            ),
            "/api/tags": (
                200,
                "application/json",
                encoded(
                    {
                        "models": [
                            {
                                "name": "local:latest",
                                "digest": "a" * 64,
                                "size": 123,
                            }
                        ]
                    }
                ),
            ),
            "/api/ps": (200, "application/json", encoded({"models": []})),
            "/api/show": (
                200,
                "application/json",
                encoded({"model_info": {"family": "fixture"}}),
            ),
            "/api/generate": (
                200,
                "application/x-ndjson",
                stream(
                    {
                        "model": "local:latest",
                        "response": "hel",
                        "done": False,
                    },
                    {
                        "model": "local:latest",
                        "response": "lo",
                        "done": True,
                        "eval_count": 2,
                    },
                ),
            ),
            "/api/chat": (
                200,
                "application/x-ndjson",
                stream(
                    {
                        "model": "local:latest",
                        "message": {"role": "assistant", "content": "answer"},
                        "done": True,
                    }
                ),
            ),
        }
        self.upstream_thread = threading.Thread(
            target=self.upstream.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self.upstream_thread.start()
        self.arbiter = AcceleratorArbiter()
        self.proxy = MediationProxy(
            self.paths,
            self.arbiter,
            queue_timeout=1,
            upstream_timeout=2,
            client_timeout=2,
        )
        self.proxy.start()
        self.addCleanup(self._close)

    def _close(self) -> None:
        self.proxy.close()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(timeout=1)

    def test_default_client_allows_discovery_but_not_candidate_inference(self) -> None:
        client = OllamaClient(self.paths, timeout=2)
        self.assertEqual(client.version(), CANDIDATE_VERSION)
        self.assertEqual(client.version(), CANDIDATE_VERSION)
        self.assertEqual(client.installed_models()[0]["name"], "local:latest")
        self.assertEqual(client.running_models(), ())
        with self.assertRaises(RuntimeUnavailable):
            client.generate("local:latest", "prompt")
        requested_paths = [path for _method, path, _body in self.upstream.requests]
        self.assertEqual(requested_paths.count("/api/version"), 5)
        self.assertNotIn("/api/generate", requested_paths)

    def test_explicit_development_candidate_streams_one_off_and_chat(self) -> None:
        client = OllamaClient(
            self.paths,
            allow_unqualified_candidate=True,
            timeout=2,
        )
        deltas = []
        result = client.generate(
            "local:latest",
            "private prompt",
            temperature=0.2,
            seed=7,
            context=4096,
            on_delta=deltas.append,
        )
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.metrics["eval_count"], 2)
        self.assertEqual([delta.content for delta in deltas], ["hel", "lo"])
        chat = client.chat(
            "local:latest",
            ({"role": "user", "content": "hello"},),
        )
        self.assertEqual(chat.content, "answer")
        bodies = {
            path: json.loads(body)
            for method, path, body in self.upstream.requests
            if method == "POST" and path in ("/api/generate", "/api/chat")
        }
        self.assertEqual(bodies["/api/generate"]["options"]["seed"], 7)
        self.assertTrue(bodies["/api/generate"]["stream"])
        self.assertEqual(bodies["/api/chat"]["messages"][0]["role"], "user")

    def test_missing_model_refuses_before_inference_and_never_pulls(self) -> None:
        client = OllamaClient(
            self.paths,
            allow_unqualified_candidate=True,
            timeout=2,
        )
        with self.assertRaises(RuntimeUnavailable):
            client.generate("missing:latest", "prompt")
        paths = [path for _method, path, _body in self.upstream.requests]
        self.assertNotIn("/api/generate", paths)
        self.assertNotIn("/api/pull", paths)

    def test_show_is_version_gated_and_development_mode_is_exact(self) -> None:
        with self.assertRaises(RuntimeUnavailable):
            OllamaClient(self.paths, timeout=2).show_model("local:latest")
        record = OllamaClient(
            self.paths, allow_unqualified_candidate=True, timeout=2
        ).show_model("local:latest")
        self.assertEqual(record["model_info"]["family"], "fixture")
        self.upstream.responses["/api/version"] = (
            200,
            "application/json",
            encoded({"version": "99.0.0"}),
        )
        with self.assertRaises(RuntimeUnavailable):
            OllamaClient(
                self.paths, allow_unqualified_candidate=True, timeout=2
            ).show_model("local:latest")

    def test_runtime_version_is_rechecked_for_every_protected_request(self) -> None:
        client = OllamaClient(
            self.paths, allow_unqualified_candidate=True, timeout=2
        )
        self.assertEqual(client.show_model("local:latest")["model_info"]["family"], "fixture")
        self.upstream.responses["/api/version"] = (
            200,
            "application/json",
            encoded({"version": "99.0.0"}),
        )
        with self.assertRaises(RuntimeUnavailable):
            client.show_model("local:latest")
        requested_paths = [path for _method, path, _body in self.upstream.requests]
        self.assertEqual(requested_paths.count("/api/version"), 2)
        self.assertEqual(requested_paths.count("/api/show"), 1)

    def test_cloud_control_and_invalid_model_names_fail_before_socket_request(self) -> None:
        initial = len(self.upstream.requests)
        invalid = ("", "has space", "m:cloud", "m:tag-cloud", "é", "x" * 1025)
        for model in invalid:
            with self.subTest(model=model[:20]), self.assertRaises(RuntimeContractError):
                validate_model_name(model)
        self.assertEqual(len(self.upstream.requests), initial)

    def test_socket_absence_mode_and_runtime_directory_mode_fail_closed(self) -> None:
        self.proxy.close()
        with self.assertRaises(RuntimeUnavailable):
            OllamaClient(self.paths, timeout=2).version()
        self.proxy = MediationProxy(
            self.paths,
            self.arbiter,
            queue_timeout=1,
            upstream_timeout=2,
            client_timeout=2,
        )
        self.proxy.start()
        self.paths.mediation_socket.chmod(0o660)
        with self.assertRaises(RuntimeUnavailable):
            OllamaClient(self.paths, timeout=2).version()
        self.paths.mediation_socket.chmod(0o600)
        self.paths.runtime_dir.chmod(0o750)
        with self.assertRaises(RuntimeUnavailable):
            OllamaClient(self.paths, timeout=2).version()
        self.paths.runtime_dir.chmod(0o700)

    def test_malformed_incomplete_wrong_content_and_status_streams_fail(self) -> None:
        client = OllamaClient(
            self.paths,
            allow_unqualified_candidate=True,
            timeout=2,
        )
        cases = (
            (
                "incomplete",
                200,
                "application/x-ndjson",
                stream({"model": "local:latest", "response": "x", "done": False}),
            ),
            (
                "duplicate",
                200,
                "application/x-ndjson",
                b'{"model":"local:latest","response":"x","done":true,"done":false}\n',
            ),
            (
                "content-type",
                200,
                "text/plain",
                stream({"model": "local:latest", "response": "x", "done": True}),
            ),
            ("status", 500, "application/json", b'{"error":"SECRET PROMPT"}'),
        )
        for name, status, content_type, payload in cases:
            with self.subTest(name=name):
                self.upstream.responses["/api/generate"] = (
                    status,
                    content_type,
                    payload,
                )
                with self.assertRaises(UpstreamProtocolError) as captured:
                    client.generate(
                        "local:latest", "prompt", check_installed=False
                    )
                self.assertNotIn("SECRET", str(captured.exception))

    def test_message_sampling_and_tool_declaration_bounds_fail_locally(self) -> None:
        client = OllamaClient(
            self.paths,
            allow_unqualified_candidate=True,
            timeout=2,
        )
        invalid_messages = (
            (),
            ({"role": "bad", "content": "x"},),
            ({"role": "user", "content": "x", "extra": True},),
        )
        for messages in invalid_messages:
            with self.subTest(messages=messages), self.assertRaises(RuntimeContractError):
                client.chat("local:latest", messages, check_installed=False)
        for kwargs in (
            {"temperature": -1},
            {"temperature": 3},
            {"seed": True},
            {"context": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeContractError):
                client.generate(
                    "local:latest", "prompt", check_installed=False, **kwargs
                )
        with self.assertRaises(RuntimeContractError):
            client.chat(
                "local:latest",
                ({"role": "user", "content": "x"},),
                tools=tuple({"type": "function"} for _ in range(17)),
                check_installed=False,
            )

    def test_chat_images_are_canonical_bounded_and_never_accept_storage_records(self) -> None:
        client = OllamaClient(
            self.paths,
            allow_unqualified_candidate=True,
            timeout=2,
        )
        image = base64.b64encode(png_image(3, 2)).decode("ascii")
        result = client.chat(
            "local:latest",
            ({"role": "user", "content": "describe", "images": [image]},),
            check_installed=False,
        )
        self.assertEqual(result.content, "answer")
        request = next(
            json.loads(body)
            for method, path, body in self.upstream.requests
            if method == "POST" and path == "/api/chat"
        )
        self.assertEqual(request["messages"][0]["images"], [image])

        before = len(self.upstream.requests)
        invalid = (
            {"role": "assistant", "content": "x", "images": [image]},
            {"role": "user", "content": "x", "images": [image, image]},
            {"role": "user", "content": "x", "images": ["not base64!"]},
            {"role": "user", "content": "x", "images": None},
            {
                "role": "user",
                "content": "x",
                "images": [base64.b64encode(b"not an image").decode("ascii")],
            },
            {"role": "user", "content": "x", "attachments": []},
        )
        for message in invalid:
            with self.subTest(message=list(message)), self.assertRaises(RuntimeContractError):
                client.chat("local:latest", (message,))

        oversized_encoded = "A" * (4 * ((4 * 1024 * 1024 + 2) // 3) + 4)
        with mock.patch(
            "kilix_ollama.client.base64.b64decode",
            side_effect=AssertionError("decoder reached for over-limit input"),
        ), self.assertRaises(RuntimeContractError):
            client.chat(
                "local:latest",
                ({"role": "user", "content": "x", "images": [oversized_encoded]},),
            )
        self.assertEqual(len(self.upstream.requests), before)


if __name__ == "__main__":
    unittest.main()
