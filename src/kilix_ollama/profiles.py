"""Fixed runtime tunings that still require external qualification evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .errors import RuntimeContractError


PROFILE_ENVIRONMENT_KEYS = frozenset(
    (
        "GGML_VK_FORCE_MMVQ",
        "LLAMA_ARG_CACHE_RAM",
        "LLAMA_ARG_CTX_CHECKPOINTS",
        "OLLAMA_GPU_OVERHEAD",
    )
)


@dataclass(frozen=True)
class RuntimeProfile:
    """A bounded tuning candidate; this object is not qualification evidence."""

    name: str
    backend: str
    workload: str
    status: str
    requirements: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]

    @property
    def environ(self) -> dict[str, str]:
        return dict(self.environment)

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "backend": self.backend,
            "workload": self.workload,
            "status": self.status,
            "requirements": list(self.requirements),
            "environment": self.environ,
            "qualification_embedded": False,
        }


# This profile is intentionally narrower than "Nouveau" or "Vulkan". Its
# settings were measured with the exact F105 runtime candidate and Qwen 3.5 4B
# on TU106. A release integrator must match all requirements with evidence
# outside this source tree before passing the profile to a launch spec.
NVK_TU106_QWEN35_4B_R1 = RuntimeProfile(
    name="nvk-tu106-qwen35-4b-r1",
    backend="vulkan-nouveau",
    workload="qwen35-4b-conversation",
    status="measured-candidate-unqualified",
    requirements=(
        "pci-vendor:10de",
        "pci-device:1f36",
        "drm-driver:nouveau",
        "vulkan-driver:nvk",
        "ollama:0.32.14+plebian.security-license-r3",
        "llama.cpp:b10434-7e4c0a96880d",
    ),
    environment=tuple(
        sorted(
            {
                # NVK exposes integer dot products but no fp16/cooperative
                # matrix path on the measured TU106 stack. MMVQ was the
                # consistently faster quantized matrix-vector path.
                "GGML_VK_FORCE_MMVQ": "1",
                # The hybrid recurrent Qwen path can enter a persistent slow
                # state when a new, unrelated prompt replaces an idle slot.
                # Disabling host prompt snapshots and recurrent checkpoints
                # made unrelated conversation turns stable.
                "LLAMA_ARG_CACHE_RAM": "0",
                "LLAMA_ARG_CTX_CHECKPOINTS": "0",
            }.items()
        )
    ),
)


# The avatar profile keeps the LLM tuning above and additionally prevents
# Ollama from treating the whole GPU as its own allocation pool. The 256 MiB
# reservation covers the measured Pocket TTS Q8_0/context-512 working set plus
# margin. It is not a general promise that every TTS model fits that budget.
NVK_TU106_QWEN35_4B_AVATAR_R1 = RuntimeProfile(
    name="nvk-tu106-qwen35-4b-avatar-r1",
    backend="vulkan-nouveau",
    workload="qwen35-4b-conversation-with-priority-tts",
    status="measured-candidate-unqualified",
    requirements=(
        "pci-vendor:10de",
        "pci-device:1f36",
        "drm-driver:nouveau",
        "vulkan-driver:nvk",
        "ollama:0.32.14+plebian.security-license-r3",
        "llama.cpp:b10434-7e4c0a96880d",
        "tts-working-set:maximum-256MiB",
        "tts-arbitration:utterance-boundary-priority",
    ),
    environment=tuple(
        sorted(
            {
                "GGML_VK_FORCE_MMVQ": "1",
                "LLAMA_ARG_CACHE_RAM": "0",
                "LLAMA_ARG_CTX_CHECKPOINTS": "0",
                "OLLAMA_GPU_OVERHEAD": "268435456",
            }.items()
        )
    ),
)

RUNTIME_PROFILES = MappingProxyType(
    {
        NVK_TU106_QWEN35_4B_R1.name: NVK_TU106_QWEN35_4B_R1,
        NVK_TU106_QWEN35_4B_AVATAR_R1.name: NVK_TU106_QWEN35_4B_AVATAR_R1,
    }
)


def validate_runtime_profile(profile: RuntimeProfile, *, backend: str) -> None:
    if not isinstance(profile, RuntimeProfile):
        raise RuntimeContractError("runtime profile has an invalid type")
    if RUNTIME_PROFILES.get(profile.name) != profile:
        raise RuntimeContractError("runtime profile is not an exact catalog entry")
    if profile.backend != backend:
        raise RuntimeContractError(
            f"runtime profile {profile.name} requires backend {profile.backend}"
        )
    if profile.status != "measured-candidate-unqualified":
        raise RuntimeContractError(f"runtime profile {profile.name} has an unknown status")
    if not profile.name or not profile.workload or not profile.requirements:
        raise RuntimeContractError("runtime profile metadata is incomplete")
    environment = profile.environ
    if len(environment) != len(profile.environment):
        raise RuntimeContractError("runtime profile contains a duplicate environment key")
    unexpected = set(environment) - PROFILE_ENVIRONMENT_KEYS
    if unexpected:
        raise RuntimeContractError(
            f"runtime profile contains a forbidden environment key: {sorted(unexpected)[0]}"
        )
    for name, value in environment.items():
        if not value or len(value) > 128:
            raise RuntimeContractError(f"runtime profile value is invalid: {name}")
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
            raise RuntimeContractError(f"runtime profile value is not printable ASCII: {name}")


def runtime_profile(name: str) -> RuntimeProfile:
    if not isinstance(name, str):
        raise RuntimeContractError("runtime profile name must be a string")
    try:
        return RUNTIME_PROFILES[name]
    except KeyError as error:
        raise RuntimeContractError(f"unknown runtime profile: {name}") from error


def runtime_profiles_record() -> dict[str, object]:
    return {
        "schema": "kilix.ollama.runtime-profiles/v0-unfrozen",
        "schema_status": "development-unfrozen",
        "profiles": [profile.record() for profile in RUNTIME_PROFILES.values()],
        "automatic_admission": False,
        "weight_access": False,
    }


__all__ = [
    "NVK_TU106_QWEN35_4B_AVATAR_R1",
    "NVK_TU106_QWEN35_4B_R1",
    "PROFILE_ENVIRONMENT_KEYS",
    "RUNTIME_PROFILES",
    "RuntimeProfile",
    "runtime_profile",
    "runtime_profiles_record",
    "validate_runtime_profile",
]
