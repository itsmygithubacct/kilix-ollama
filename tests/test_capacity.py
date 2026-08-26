from __future__ import annotations

from dataclasses import replace
import unittest

from kilix_ollama.capacity import (
    NVK_TU106_AVATAR_R1,
    co_residency_policy,
    co_residency_record,
    validate_co_residency_policy,
)
from kilix_ollama.errors import RuntimeContractError


class CoResidencyPolicyTests(unittest.TestCase):
    def test_exact_policy_serializes_gpu_work_and_accesses_no_weights(self) -> None:
        policy = co_residency_policy()
        self.assertIs(policy, NVK_TU106_AVATAR_R1)
        self.assertEqual(policy.ollama_gpu_overhead_bytes, 256 * 1024 * 1024)
        self.assertEqual(policy.arbitration, "finish-llm-turn-then-prioritize-tts")
        self.assertEqual(policy.tts_context_tokens, 512)
        record = co_residency_record()
        self.assertFalse(record["automatic_admission"])
        self.assertFalse(record["weight_access"])

    def test_forged_or_internally_inconsistent_policy_fails(self) -> None:
        forged = replace(NVK_TU106_AVATAR_R1, tts_batch_tokens=1024)
        with self.assertRaises(RuntimeContractError):
            validate_co_residency_policy(forged)
        with self.assertRaises(RuntimeContractError):
            co_residency_policy("unknown")


if __name__ == "__main__":
    unittest.main()
