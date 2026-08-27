from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest

from kilix_ollama.backend import BackendSelection
from kilix_ollama.errors import SupervisorError
from kilix_ollama.paths import RuntimePaths
from kilix_ollama.service import build_launch_spec
from kilix_ollama.supervisor import (
    RuntimeSupervisor,
    TransportAdmission,
    _decode_http_response,
)


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


class RuntimeSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trusted = self.root / "trusted"
        self.launcher = self.trusted / "ollama-current"
        self.launcher.parent.mkdir(mode=0o700)
        self.launcher.write_text(
            "#!/bin/sh\n"
            "env | sort > \"$TMPDIR/child-environment\"\n"
            "trap 'exit 0' TERM INT\n"
            "while :; do sleep 1; done\n",
            encoding="utf-8",
        )
        self.launcher.chmod(0o700)
        self.paths = private_paths(self.root)
        selection = BackendSelection("cpu", "cpu", "fixture", ("cpu",))
        base = build_launch_spec(self.paths, selection, transport="unix")
        self.spec = replace(
            base,
            executable=self.launcher,
            argv=(str(self.launcher), "serve"),
        )
        digest = hashlib.sha256(self.launcher.read_bytes()).hexdigest()
        self.admission = TransportAdmission(
            transport="unix",
            runtime_sha256=digest,
            unix_listener_delta_verified=True,
        )
        self.supervisor = RuntimeSupervisor(
            self.paths,
            launcher=self.launcher,
            launcher_uid=os.getuid(),
            trusted_root=self.trusted,
        )
        self.addCleanup(self.supervisor.stop)

    def ready(self, _spec, process) -> None:
        environment = self.paths.temp_dir / "child-environment"
        deadline = time.monotonic() + 1
        while not environment.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(0.01)
        self.assertTrue(environment.exists())

    def test_fixed_process_starts_with_cleared_environment_and_stops_as_group(self) -> None:
        os.environ["HTTPS_PROXY"] = "https://secret.invalid"
        os.environ["LD_PRELOAD"] = "/tmp/hostile.so"
        self.addCleanup(os.environ.pop, "HTTPS_PROXY", None)
        self.addCleanup(os.environ.pop, "LD_PRELOAD", None)
        pid = self.supervisor.start(self.spec, self.admission, readiness=self.ready)
        self.assertGreater(pid, 0)
        self.assertEqual(self.supervisor.record()["state"], "running")
        child_environment = (self.paths.temp_dir / "child-environment").read_text()
        self.assertNotIn("HTTPS_PROXY", child_environment)
        self.assertNotIn("LD_PRELOAD", child_environment)
        self.assertIn("OLLAMA_NO_CLOUD=1", child_environment)
        self.assertTrue(self.supervisor.stop(timeout=2))
        self.assertEqual(self.supervisor.record()["state"], "stopped")
        self.assertFalse(self.supervisor.stop())

    def test_wrong_runtime_hash_fails_before_start(self) -> None:
        wrong = replace(self.admission, runtime_sha256="0" * 64)
        with self.assertRaises(SupervisorError):
            self.supervisor.start(self.spec, wrong, readiness=self.ready)
        self.assertEqual(self.supervisor.record()["state"], "stopped")

    def test_missing_listener_evidence_fails_before_start(self) -> None:
        unverified = replace(self.admission, unix_listener_delta_verified=False)
        with self.assertRaises(SupervisorError) as caught:
            self.supervisor.start(self.spec, unverified, readiness=self.ready)
        self.assertIn("listener", str(caught.exception))

    def test_extra_argv_and_environment_are_rejected(self) -> None:
        with self.assertRaises(SupervisorError):
            self.supervisor.start(
                replace(self.spec, argv=self.spec.argv + ("--arbitrary",)),
                self.admission,
                readiness=self.ready,
            )
        with self.assertRaises(SupervisorError):
            self.supervisor.start(
                replace(
                    self.spec,
                    environment=self.spec.environment + (("LD_PRELOAD", "/tmp/x"),),
                ),
                self.admission,
                readiness=self.ready,
            )

    def test_occupied_upstream_path_is_never_removed(self) -> None:
        self.paths.runtime_dir.mkdir(parents=True, mode=0o700)
        marker = self.paths.upstream_socket
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(SupervisorError):
            self.supervisor.start(self.spec, self.admission, readiness=self.ready)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_status_is_unfrozen_and_contains_no_weight_or_request_data(self) -> None:
        record = self.supervisor.record()
        self.assertEqual(record["schema_status"], "development-unfrozen")
        self.assertFalse(record["weight_access"])
        self.assertFalse(record["request_content_logged"])
        self.assertFalse(record["arbitrary_argv"])

    def test_readiness_parser_accepts_case_insensitive_content_length(self) -> None:
        body = b'{"version":"0.32.14+plebian.security-license-r3"}'
        payload = (
            b"HTTP/1.1 200 OK\r\n"
            b"content-type: application/json\r\n"
            + f"content-length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        self.assertEqual(_decode_http_response(payload), body)

    def test_failed_default_readiness_removes_only_observed_socket(self) -> None:
        threads: list[threading.Thread] = []

        def popen(*arguments, **keywords):
            process = subprocess.Popen(*arguments, **keywords)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.paths.upstream_socket))
            self.paths.upstream_socket.chmod(0o600)
            listener.listen(1)

            def serve_invalid_version() -> None:
                try:
                    connection, _address = listener.accept()
                    with connection:
                        connection.recv(4096)
                        connection.sendall(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
                        )
                finally:
                    listener.close()

            thread = threading.Thread(target=serve_invalid_version)
            threads.append(thread)
            thread.start()
            return process

        supervisor = RuntimeSupervisor(
            self.paths,
            launcher=self.launcher,
            launcher_uid=os.getuid(),
            trusted_root=self.trusted,
            popen_factory=popen,
        )
        with self.assertRaises(SupervisorError):
            supervisor.start(self.spec, self.admission, timeout=1.0)
        for thread in threads:
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
        self.assertFalse(os.path.lexists(self.paths.upstream_socket))


if __name__ == "__main__":
    unittest.main()
