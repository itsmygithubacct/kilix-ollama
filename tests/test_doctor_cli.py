from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from kilix_ollama.backend import VulkanReport
from kilix_ollama.cli import main
from kilix_ollama.doctor import run_doctor
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


class DoctorTests(unittest.TestCase):
    def test_complete_fixture_is_ok_without_touching_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = private_paths(root)
            paths.runtime_dir.mkdir(parents=True, mode=0o700)
            trusted_root = root / "trusted"
            binary = trusted_root / "usr/lib/plebian-os/ollama/ollama-current"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"#!/bin/sh\n")
            binary.chmod(0o755)
            listener = socket.socket(socket.AF_UNIX)
            self.addCleanup(listener.close)
            listener.bind(str(paths.mediation_socket))
            paths.mediation_socket.chmod(0o600)

            report = run_doctor(
                paths=paths,
                runtime_launcher=binary,
                runtime_owner_uid=os.getuid(),
                trusted_root=trusted_root,
                vulkan=VulkanReport(None, (), (), ()),
            )
            self.assertEqual(report.status, "warning")
            record = report.record()
            self.assertFalse(record["weights_accessed"])
            self.assertEqual(record["backend"]["selected"], "cpu")

    def test_missing_runtime_is_an_error_but_missing_idle_socket_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_doctor(
                paths=private_paths(root),
                runtime_launcher=root / "missing/ollama-current",
                runtime_owner_uid=os.getuid(),
                trusted_root=root,
                vulkan=VulkanReport(None, (), (), ()),
            )
            self.assertEqual(report.status, "error")
            checks = {check.name: check.status for check in report.checks}
            self.assertEqual(checks["runtime-launcher"], "error")
            self.assertEqual(checks["mediation-socket"], "warning")

    def test_symlinked_launcher_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"#!/bin/sh\n")
            target.chmod(0o755)
            launcher = root / "launcher"
            launcher.symlink_to(target)
            report = run_doctor(
                paths=private_paths(root),
                runtime_launcher=launcher,
                runtime_owner_uid=os.getuid(),
                trusted_root=root,
                vulkan=VulkanReport(None, (), (), ()),
            )
            check = next(check for check in report.checks if check.name == "runtime-launcher")
            self.assertEqual(check.status, "error")
            self.assertIn("symlink", check.detail)


class CliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_policy_json_is_machine_readable(self) -> None:
        status, stdout, stderr = self.invoke("policy", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["unknown_routes"], "deny")

    def test_service_contract_reports_gated_supervisor(self) -> None:
        status, stdout, _ = self.invoke("service-contract", "--json")
        self.assertEqual(status, 0)
        record = json.loads(stdout)
        self.assertTrue(record["process_launch_implemented"])
        self.assertFalse(record["automatic_launch_enabled"])

    def test_runtime_profiles_are_machine_readable_and_model_free(self) -> None:
        status, stdout, stderr = self.invoke("runtime-profiles", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertFalse(record["automatic_admission"])
        self.assertFalse(record["weight_access"])
        self.assertEqual(record["profiles"][0]["name"], "nvk-tu106-qwen35-4b-r1")

    def test_forced_vulkan_service_spec_fails_while_unqualified(self) -> None:
        status, _, stderr = self.invoke(
            "service-spec",
            "--backend",
            "vulkan",
            "--json",
        )
        self.assertEqual(status, 2)
        self.assertIn("not qualified", stderr)

    def test_co_residency_policy_is_machine_readable_and_model_free(self) -> None:
        status, stdout, stderr = self.invoke("co-residency", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertEqual(
            record["policies"][0]["arbitration"],
            "finish-llm-turn-then-prioritize-tts",
        )
        self.assertFalse(record["automatic_admission"])
        self.assertFalse(record["weight_access"])

    def test_tts_catalog_is_machine_readable_and_requires_explicit_install(self) -> None:
        status, stdout, stderr = self.invoke("tts-models", "--json")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertFalse(record["automatic_install"])
        self.assertFalse(record["kilix_mirror"])
        self.assertTrue(record["explicit_terms_acceptance_required"])
        self.assertFalse(record["product_voice_included"])

    def test_tts_install_without_acceptance_fails_before_downloading(self) -> None:
        status, stdout, stderr = self.invoke("tts-install")
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("terms were not accepted", stderr)

    def test_tts_status_is_network_free_and_missing_is_informational(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                "HOME": str(Path(directory) / "home"),
                "XDG_RUNTIME_DIR": str(Path(directory) / "run"),
                "XDG_STATE_HOME": str(Path(directory) / "state"),
                "XDG_DATA_HOME": str(Path(directory) / "data"),
                "XDG_CACHE_HOME": str(Path(directory) / "cache"),
            },
        ):
            status, stdout, stderr = self.invoke(
                "tts-status", "--verify", "--json"
            )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        record = json.loads(stdout)
        self.assertFalse(record["installed"])
        self.assertFalse(record["network_access"])


if __name__ == "__main__":
    unittest.main()
