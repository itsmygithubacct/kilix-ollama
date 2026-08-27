from __future__ import annotations

import http.server
import os
from pathlib import Path
import socket
import socketserver
import tempfile
import threading
import time
import unittest

from kilix_ollama.arbiter import LLM_TURN, TTS_UTTERANCE, AcceleratorArbiter
from kilix_ollama.errors import RuntimeContractError
from kilix_ollama.mediation import MediationProxy
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


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._serve()

    def do_POST(self) -> None:
        self._serve()

    def do_DELETE(self) -> None:
        self._serve()

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with self.server.request_lock:  # type: ignore[attr-defined]
            self.server.requests.append(  # type: ignore[attr-defined]
                (self.command, self.path, dict(self.headers.items()), body)
            )
        if self.path in ("/api/generate", "/api/chat", "/api/embed"):
            self.server.inference_started.set()  # type: ignore[attr-defined]
            self.server.release_inference.wait(timeout=2)  # type: ignore[attr-defined]
        payload = self.server.response_body  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Set-Cookie", "must-not-cross=yes")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_arguments) -> None:
        pass


class MediationProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = private_paths(self.root)
        self.paths.runtime_dir.mkdir(parents=True, mode=0o700)
        self.paths.runtime_dir.chmod(0o700)
        self.upstream = socketserver.UnixStreamServer(
            str(self.paths.upstream_socket),
            _FixtureHandler,
        )
        self.paths.upstream_socket.chmod(0o600)
        self.upstream.requests = []
        self.upstream.request_lock = threading.Lock()
        self.upstream.inference_started = threading.Event()
        self.upstream.release_inference = threading.Event()
        self.upstream.release_inference.set()
        self.upstream.response_body = b'{"version":"fixture"}\n'
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

    def exchange(self, payload: bytes) -> bytes:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(3)
        try:
            connection.connect(str(self.paths.mediation_socket))
            connection.sendall(payload)
            blocks: list[bytes] = []
            while True:
                block = connection.recv(64 * 1024)
                if not block:
                    return b"".join(blocks)
                blocks.append(block)
        finally:
            connection.close()

    def test_exact_route_is_forwarded_with_minimal_headers_and_chunked_response(self) -> None:
        response = self.exchange(
            b"GET /api/version HTTP/1.1\r\n"
            b"Host: untrusted.invalid\r\n"
            b"Accept: application/json\r\n"
            b"Authorization: Bearer secret\r\n"
            b"Cookie: secret=yes\r\n\r\n"
        )
        self.assertIn(b"HTTP/1.1 200 OK\r\n", response)
        self.assertIn(b"Transfer-Encoding: chunked\r\n", response)
        self.assertIn(b'{"version":"fixture"}', response)
        self.assertNotIn(b"Set-Cookie", response)
        method, path, headers, body = self.upstream.requests[0]
        self.assertEqual((method, path, body), ("GET", "/api/version", b""))
        self.assertEqual(headers["Host"], "localhost")
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Cookie", headers)
        self.assertEqual(headers["User-Agent"], "kilix-ollama/0.2.1.dev0")

    def test_cloud_and_unknown_routes_never_reach_upstream(self) -> None:
        response = self.exchange(
            b"POST /api/push HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 2\r\nContent-Type: application/json\r\n\r\n{}"
        )
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)
        self.assertEqual(self.upstream.requests, [])

    def test_duplicate_length_transfer_encoding_and_pipelining_fail_closed(self) -> None:
        cases = (
            b"POST /api/show HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
            b"Content-Length: 2\r\n\r\n{}",
            b"POST /api/show HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            b"GET /api/version HTTP/1.1\r\nHost: x\r\n\r\nextra",
        )
        for payload in cases:
            with self.subTest(payload=payload[:30]):
                response = self.exchange(payload)
                self.assertIn(b"HTTP/1.1 400 Bad Request", response)
        self.assertEqual(self.upstream.requests, [])

    def test_llm_stream_holds_device_until_response_then_tts_runs(self) -> None:
        self.upstream.inference_started.clear()
        self.upstream.release_inference.clear()
        response: list[bytes] = []

        def generate() -> None:
            body = b'{"model":"qwen","prompt":"hello"}'
            response.append(
                self.exchange(
                    b"POST /api/generate HTTP/1.1\r\n"
                    b"Host: localhost\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                    + body
                )
            )

        thread = threading.Thread(target=generate)
        thread.start()
        self.assertTrue(self.upstream.inference_started.wait(timeout=1))
        self.assertEqual(self.arbiter.record()["active"]["workload"], LLM_TURN)
        speech = self.arbiter.submit(TTS_UTTERANCE)
        self.assertEqual(self.arbiter.record()["queued"][TTS_UTTERANCE], 1)
        self.upstream.release_inference.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertIn(b"HTTP/1.1 200 OK", response[0])
        speech_lease = speech.wait(timeout=1)
        self.assertEqual(speech_lease.workload, TTS_UTTERANCE)
        speech_lease.release()

    def test_cloud_tag_in_allowed_inference_route_never_reaches_upstream(self) -> None:
        body = b'{"model":"qwen3.5:4b:cloud","prompt":"hello"}'
        response = self.exchange(
            b"POST /api/generate HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        self.assertIn(b"HTTP/1.1 403 Forbidden", response)
        self.assertEqual(self.upstream.requests, [])

    def test_response_limit_aborts_stream_without_a_second_status_line(self) -> None:
        self.proxy.close()
        self.upstream.response_body = b"x" * 32
        self.proxy = MediationProxy(
            self.paths,
            self.arbiter,
            max_response_bytes=16,
            queue_timeout=1,
            upstream_timeout=2,
            client_timeout=2,
        )
        self.proxy.start()
        response = self.exchange(
            b"GET /api/version HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        self.assertEqual(response.count(b"HTTP/1.1"), 1)
        self.assertIn(b"HTTP/1.1 200 OK", response)
        self.assertEqual(self.proxy.record()["counters"]["response_limit_failures"], 1)

    def test_proxy_removes_only_its_own_socket(self) -> None:
        self.proxy.close()
        self.assertFalse(self.paths.mediation_socket.exists())
        self.paths.mediation_socket.write_text("keep", encoding="utf-8")
        with self.assertRaises(RuntimeContractError):
            MediationProxy(self.paths, self.arbiter)
        self.assertEqual(self.paths.mediation_socket.read_text(encoding="utf-8"), "keep")

    def test_status_records_no_prompt_or_weight_access(self) -> None:
        record = self.proxy.record()
        self.assertTrue(record["same_uid_required"])
        self.assertFalse(record["request_content_logged"])
        self.assertFalse(record["cloud_fallback"])
        self.assertFalse(record["weight_access"])


if __name__ == "__main__":
    unittest.main()
