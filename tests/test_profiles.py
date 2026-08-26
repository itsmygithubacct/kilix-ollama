from __future__ import annotations

from dataclasses import replace
import unittest

from kilix_ollama.errors import RuntimeContractError
from kilix_ollama.profiles import (
    NVK_TU106_QWEN35_4B_AVATAR_R1,
    NVK_TU106_QWEN35_4B_R1,
    runtime_profile,
    runtime_profiles_record,
    validate_runtime_profile,
)


class RuntimeProfileTests(unittest.TestCase):
    def test_named_profile_is_bounded_and_does_not_embed_qualification(self) -> None:
        profile = runtime_profile("nvk-tu106-qwen35-4b-r1")
        self.assertIs(profile, NVK_TU106_QWEN35_4B_R1)
        self.assertEqual(profile.backend, "vulkan-nouveau")
        self.assertEqual(
            profile.environ,
            {
                "GGML_VK_FORCE_MMVQ": "1",
                "LLAMA_ARG_CACHE_RAM": "0",
                "LLAMA_ARG_CTX_CHECKPOINTS": "0",
            },
        )
        self.assertFalse(profile.record()["qualification_embedded"])

    def test_profile_rejects_backend_mismatch_and_environment_injection(self) -> None:
        with self.assertRaises(RuntimeContractError):
            validate_runtime_profile(NVK_TU106_QWEN35_4B_R1, backend="vulkan")
        forged = replace(
            NVK_TU106_QWEN35_4B_R1,
            environment=(("LD_PRELOAD", "/tmp/hostile.so"),),
        )
        with self.assertRaises(RuntimeContractError):
            validate_runtime_profile(forged, backend="vulkan-nouveau")
        modified = replace(
            NVK_TU106_QWEN35_4B_R1,
            environment=(("GGML_VK_FORCE_MMVQ", "0"),),
        )
        with self.assertRaises(RuntimeContractError):
            validate_runtime_profile(modified, backend="vulkan-nouveau")

    def test_avatar_profile_reserves_exact_tts_headroom(self) -> None:
        profile = runtime_profile("nvk-tu106-qwen35-4b-avatar-r1")
        self.assertIs(profile, NVK_TU106_QWEN35_4B_AVATAR_R1)
        self.assertEqual(profile.environ["OLLAMA_GPU_OVERHEAD"], "268435456")
        self.assertEqual(profile.environ["GGML_VK_FORCE_MMVQ"], "1")
        self.assertFalse(profile.record()["qualification_embedded"])

    def test_unknown_profile_fails_and_catalogue_accesses_no_weights(self) -> None:
        with self.assertRaises(RuntimeContractError):
            runtime_profile("unknown")
        record = runtime_profiles_record()
        self.assertFalse(record["automatic_admission"])
        self.assertFalse(record["weight_access"])


if __name__ == "__main__":
    unittest.main()
