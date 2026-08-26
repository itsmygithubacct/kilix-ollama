"""Measured Vulkan co-residency policy without model or launch access."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .errors import RuntimeContractError


@dataclass(frozen=True)
class CoResidencyPolicy:
    """One exact scheduling/resource candidate backed by external evidence."""

    name: str
    backend: str
    runtime_profile: str
    workloads: tuple[str, ...]
    ollama_gpu_overhead_bytes: int
    llm_context_tokens: int
    llm_max_loaded_models: int
    tts_context_tokens: int
    tts_batch_tokens: int
    tts_ubatch_tokens: int
    arbitration: str
    active_overlap: str
    first_use_warmup: str
    status: str = "measured-candidate-unqualified"

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "backend": self.backend,
            "runtime_profile": self.runtime_profile,
            "workloads": list(self.workloads),
            "ollama_gpu_overhead_bytes": self.ollama_gpu_overhead_bytes,
            "llm_context_tokens": self.llm_context_tokens,
            "llm_max_loaded_models": self.llm_max_loaded_models,
            "tts_context_tokens": self.tts_context_tokens,
            "tts_batch_tokens": self.tts_batch_tokens,
            "tts_ubatch_tokens": self.tts_ubatch_tokens,
            "arbitration": self.arbitration,
            "active_overlap": self.active_overlap,
            "first_use_warmup": self.first_use_warmup,
            "status": self.status,
            "qualification_embedded": False,
            "automatic_admission": False,
            "weight_access": False,
        }


NVK_TU106_AVATAR_R1 = CoResidencyPolicy(
    name="nvk-tu106-avatar-r1",
    backend="vulkan-nouveau",
    runtime_profile="nvk-tu106-qwen35-4b-avatar-r1",
    workloads=("qwen35-4b-conversation", "pockettts-q8-speech"),
    ollama_gpu_overhead_bytes=256 * 1024 * 1024,
    llm_context_tokens=4096,
    llm_max_loaded_models=1,
    tts_context_tokens=512,
    tts_batch_tokens=128,
    tts_ubatch_tokens=128,
    arbitration="finish-llm-turn-then-prioritize-tts",
    active_overlap="supported-with-latency-degradation",
    first_use_warmup="required-during-explicit-install-or-qualification",
)

CO_RESIDENCY_POLICIES = MappingProxyType(
    {NVK_TU106_AVATAR_R1.name: NVK_TU106_AVATAR_R1}
)


def validate_co_residency_policy(policy: CoResidencyPolicy) -> None:
    if not isinstance(policy, CoResidencyPolicy):
        raise RuntimeContractError("co-residency policy has an invalid type")
    if CO_RESIDENCY_POLICIES.get(policy.name) != policy:
        raise RuntimeContractError("co-residency policy is not an exact catalog entry")
    if policy.status != "measured-candidate-unqualified":
        raise RuntimeContractError("co-residency policy has an unknown status")
    integer_fields = (
        policy.ollama_gpu_overhead_bytes,
        policy.llm_context_tokens,
        policy.llm_max_loaded_models,
        policy.tts_context_tokens,
        policy.tts_batch_tokens,
        policy.tts_ubatch_tokens,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in integer_fields):
        raise RuntimeContractError("co-residency policy has an invalid resource bound")
    if policy.tts_batch_tokens > policy.tts_context_tokens:
        raise RuntimeContractError("TTS batch exceeds its context bound")
    if policy.tts_ubatch_tokens > policy.tts_batch_tokens:
        raise RuntimeContractError("TTS ubatch exceeds its batch bound")


def co_residency_policy(name: str = NVK_TU106_AVATAR_R1.name) -> CoResidencyPolicy:
    if not isinstance(name, str):
        raise RuntimeContractError("co-residency policy name must be a string")
    try:
        policy = CO_RESIDENCY_POLICIES[name]
    except KeyError as error:
        raise RuntimeContractError(f"unknown co-residency policy: {name}") from error
    validate_co_residency_policy(policy)
    return policy


def co_residency_record() -> dict[str, object]:
    return {
        "schema": "kilix.ollama.vulkan-co-residency/v0-unfrozen",
        "schema_status": "development-unfrozen",
        "policies": [policy.record() for policy in CO_RESIDENCY_POLICIES.values()],
        "automatic_admission": False,
        "weight_access": False,
    }


__all__ = [
    "CO_RESIDENCY_POLICIES",
    "NVK_TU106_AVATAR_R1",
    "CoResidencyPolicy",
    "co_residency_policy",
    "co_residency_record",
    "validate_co_residency_policy",
]
