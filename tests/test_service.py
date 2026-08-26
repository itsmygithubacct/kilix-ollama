from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from kilix_ollama.backend import BackendSelection
from kilix_ollama.errors import RuntimeContractError
from kilix_ollama.paths import RuntimePaths
from kilix_ollama.profiles import (
    NVK_TU106_QWEN35_4B_AVATAR_R1,
    NVK_TU106_QWEN35_4B_R1,
)
from kilix_ollama.service import build_launch_spec, service_contract_record


def paths(root: Path) -> RuntimePaths:
    return RuntimePaths.from_environment(
        {
            "HOME": str(root / "home"),
            "XDG_RUNTIME_DIR": str(root / "run"),
            "XDG_STATE_HOME": str(root / "state"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
        },
        uid=1000,
    )


class ServiceSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selection = BackendSelection("auto", "cpu", "CPU qualified", ("cpu",))

    def test_unix_spec_has_exact_argv_and_cleared_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ["OLLAMA_ORIGINS"] = "*"
            os.environ["HTTPS_PROXY"] = "https://secret.invalid"
            os.environ["LD_PRELOAD"] = "/tmp/hostile.so"
            self.addCleanup(os.environ.pop, "OLLAMA_ORIGINS", None)
            self.addCleanup(os.environ.pop, "HTTPS_PROXY", None)
            self.addCleanup(os.environ.pop, "LD_PRELOAD", None)
            spec = build_launch_spec(
                paths(root),
                self.selection,
                transport="unix",
            )
            self.assertEqual(
                spec.argv,
                ("/usr/lib/plebian-os/ollama/ollama-current", "serve"),
            )
            self.assertEqual(spec.endpoint, f"unix://{root}/run/kilix-ollama/upstream.sock")
            self.assertFalse(spec.requires_isolation_proof)
            self.assertTrue(spec.requires_listener_delta)
            self.assertEqual(spec.environ["OLLAMA_NO_CLOUD"], "1")
            self.assertEqual(spec.environ["OLLAMA_LLM_LIBRARY"], "cpu")
            self.assertEqual(spec.environ["OLLAMA_CONTEXT_LENGTH"], "4096")
            self.assertEqual(spec.environ["OLLAMA_MAX_LOADED_MODELS"], "1")
            self.assertNotIn("OLLAMA_VULKAN", spec.environ)
            self.assertNotIn("OLLAMA_ORIGINS", spec.environ)
            self.assertNotIn("HTTPS_PROXY", spec.environ)
            self.assertNotIn("LD_PRELOAD", spec.environ)
            self.assertFalse(spec.record()["launch_enabled"])
            self.assertIsNone(spec.record()["runtime_profile"])

    def test_qualified_nvk_profile_adds_only_fixed_tunings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection = BackendSelection(
                "auto",
                "vulkan-nouveau",
                "external evidence matched",
                ("cpu", "vulkan-nouveau"),
            )
            spec = build_launch_spec(
                paths(Path(directory)),
                selection,
                transport="unix",
                runtime_profile=NVK_TU106_QWEN35_4B_R1,
            )
            self.assertEqual(spec.environ["OLLAMA_LLM_LIBRARY"], "vulkan")
            self.assertEqual(spec.environ["OLLAMA_VULKAN"], "1")
            self.assertEqual(spec.environ["GGML_VK_FORCE_MMVQ"], "1")
            self.assertEqual(spec.environ["LLAMA_ARG_CACHE_RAM"], "0")
            self.assertEqual(spec.environ["LLAMA_ARG_CTX_CHECKPOINTS"], "0")
            self.assertNotIn("GGML_VK_VISIBLE_DEVICES", spec.environ)
            self.assertEqual(
                spec.record()["runtime_profile"]["name"],
                "nvk-tu106-qwen35-4b-r1",
            )

    def test_nvk_profile_cannot_be_applied_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeContractError):
                build_launch_spec(
                    paths(Path(directory)),
                    self.selection,
                    transport="unix",
                    runtime_profile=NVK_TU106_QWEN35_4B_R1,
                )

    def test_avatar_profile_reserves_gpu_memory_for_tts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection = BackendSelection(
                "auto",
                "vulkan-nouveau",
                "external evidence matched",
                ("cpu", "vulkan-nouveau"),
            )
            spec = build_launch_spec(
                paths(Path(directory)),
                selection,
                transport="unix",
                runtime_profile=NVK_TU106_QWEN35_4B_AVATAR_R1,
            )
            self.assertEqual(spec.environ["OLLAMA_GPU_OVERHEAD"], "268435456")
            self.assertEqual(spec.environ["OLLAMA_MAX_LOADED_MODELS"], "1")
            self.assertNotIn("GGML_VK_VISIBLE_DEVICES", spec.environ)

    def test_netns_loopback_is_marked_as_requiring_isolation_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = build_launch_spec(
                paths(Path(directory)),
                self.selection,
                transport="netns-tcp",
            )
            self.assertEqual(spec.endpoint, "127.0.0.1:11434")
            self.assertTrue(spec.requires_isolation_proof)
            self.assertFalse(spec.requires_listener_delta)

    def test_invalid_transport_and_forged_qualification_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = paths(Path(directory))
            with self.assertRaises(RuntimeContractError):
                build_launch_spec(layout, self.selection, transport="host-tcp")
            forged = BackendSelection("vulkan", "vulkan", "forged", ("cpu",))
            with self.assertRaises(RuntimeContractError):
                build_launch_spec(layout, forged, transport="unix")

    def test_contract_denies_unisolated_host_loopback(self) -> None:
        contract = service_contract_record()
        self.assertEqual(contract["host_loopback"], "denied-without-owner-isolation")
        self.assertIsNone(contract["admitted_transport"])
        self.assertEqual(contract["transport_status"], "development-unselected")
        self.assertEqual(
            contract["runtime_launcher"],
            "/usr/lib/plebian-os/ollama/ollama-current",
        )
        self.assertFalse(contract["process_launch_implemented"])
        self.assertFalse(contract["weight_access"])

    def test_overlong_unix_socket_path_fails(self) -> None:
        long_root = Path("/") / ("x" * 110)
        with self.assertRaises(RuntimeContractError):
            build_launch_spec(paths(long_root), self.selection, transport="unix")


if __name__ == "__main__":
    unittest.main()
