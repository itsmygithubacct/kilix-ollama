"""Immutable service specifications; this module never starts a process."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from .backend import BackendSelection
from .errors import RuntimeContractError
from .paths import RuntimePaths
from .profiles import RuntimeProfile, validate_runtime_profile


DEFAULT_RUNTIME_LAUNCHER = Path("/usr/lib/plebian-os/ollama/ollama-current")
TRANSPORTS = frozenset(("unix", "netns-tcp"))
BACKEND_LIBRARY = {
    "cpu": "cpu",
    "vulkan": "vulkan",
    "vulkan-nouveau": "vulkan",
}


@dataclass(frozen=True)
class RuntimeLaunchSpec:
    executable: Path
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    backend: str
    transport: str
    endpoint: str
    requires_isolation_proof: bool
    requires_listener_delta: bool
    runtime_profile: RuntimeProfile | None = None

    @property
    def environ(self) -> dict[str, str]:
        return dict(self.environment)

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.service-launch/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "executable": str(self.executable),
            "argv": list(self.argv),
            "environment": self.environ,
            "backend": self.backend,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "requires_isolation_proof": self.requires_isolation_proof,
            "requires_listener_delta": self.requires_listener_delta,
            "runtime_profile": (
                self.runtime_profile.record()
                if self.runtime_profile is not None
                else None
            ),
            "launch_enabled": False,
        }


def _safe_absolute(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RuntimeContractError(f"{label} must be an absolute path")
    if ".." in candidate.parts:
        raise RuntimeContractError(f"{label} contains a parent traversal")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in str(candidate)):
        raise RuntimeContractError(f"{label} contains a control character")
    return candidate


def _fixed_environment(
    paths: RuntimePaths,
    selection: BackendSelection,
    endpoint: str,
    runtime_profile: RuntimeProfile | None,
) -> Mapping[str, str]:
    unix_listener = endpoint.startswith("unix://")
    environment = {
        "HOME": str(paths.upstream_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OLLAMA_CONTEXT_LENGTH": "4096",
        # A candidate without the independently verified listener delta binds
        # only an ephemeral loopback port, then fails readiness and is killed;
        # it never falls back to the public/default Ollama port.
        "OLLAMA_HOST": "127.0.0.1:0" if unix_listener else endpoint,
        "OLLAMA_KEEP_ALIVE": "5m",
        "OLLAMA_LLM_LIBRARY": BACKEND_LIBRARY[selection.selected],
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_MAX_QUEUE": "8",
        "OLLAMA_MODELS": str(paths.model_store),
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(paths.temp_dir),
        "XDG_CACHE_HOME": str(paths.cache_dir),
        "XDG_DATA_HOME": str(paths.data_dir),
        "XDG_STATE_HOME": str(paths.state_dir),
    }
    if unix_listener:
        environment["OLLAMA_KILIX_SOCKET"] = str(paths.upstream_socket)
    if selection.selected in ("vulkan", "vulkan-nouveau"):
        environment["OLLAMA_VULKAN"] = "1"
    if runtime_profile is not None:
        validate_runtime_profile(runtime_profile, backend=selection.selected)
        environment.update(runtime_profile.environ)
    return environment


def build_launch_spec(
    paths: RuntimePaths,
    selection: BackendSelection,
    *,
    transport: str,
    runtime_profile: RuntimeProfile | None = None,
) -> RuntimeLaunchSpec:
    """Build a closed launch description without creating paths or processes."""
    if selection.selected not in BACKEND_LIBRARY:
        raise RuntimeContractError(f"unsupported backend: {selection.selected}")
    if not set(selection.qualified_backends).issubset(BACKEND_LIBRARY):
        raise RuntimeContractError("backend qualification record contains an unknown backend")
    if selection.selected not in selection.qualified_backends:
        raise RuntimeContractError("selected backend is not present in its qualification record")
    if transport not in TRANSPORTS:
        raise RuntimeContractError(f"unsupported transport: {transport}")
    for label, value in paths.record().items():
        _safe_absolute(Path(value), label)

    executable = DEFAULT_RUNTIME_LAUNCHER
    if transport == "unix":
        if len(os.fsencode(paths.upstream_socket)) > 107:
            raise RuntimeContractError("upstream Unix socket path exceeds the Linux limit")
        endpoint = f"unix://{paths.upstream_socket}"
        isolation_required = False
        listener_delta_required = True
    else:
        endpoint = "127.0.0.1:11434"
        isolation_required = True
        listener_delta_required = False
    environment = tuple(
        sorted(_fixed_environment(paths, selection, endpoint, runtime_profile).items())
    )
    return RuntimeLaunchSpec(
        executable=executable,
        argv=(str(executable), "serve"),
        environment=environment,
        backend=selection.selected,
        transport=transport,
        endpoint=endpoint,
        requires_isolation_proof=isolation_required,
        requires_listener_delta=listener_delta_required,
        runtime_profile=runtime_profile,
    )


def service_contract_record() -> dict[str, object]:
    return {
        "schema": "kilix.ollama.service-contract/v0-unfrozen",
        "schema_status": "development-unfrozen",
        "candidate_transports": ["netns-tcp", "unix"],
        "admitted_transport": None,
        "transport_status": "development-unselected",
        "host_loopback": "denied-without-owner-isolation",
        "wildcard_bind": "denied",
        "environment_source": "cleared-fixed-allowlist",
        "runtime_launcher": str(DEFAULT_RUNTIME_LAUNCHER),
        "runtime_profiles": "external-qualification-required",
        "process_launch_implemented": True,
        "automatic_launch_enabled": False,
        "launch_admission": "exact-external-transport-and-runtime-evidence-required",
        "shared_accelerator_arbiter": "implemented-unfrozen",
        "weight_access": False,
    }


__all__ = [
    "DEFAULT_RUNTIME_LAUNCHER",
    "RuntimeLaunchSpec",
    "TRANSPORTS",
    "build_launch_spec",
    "service_contract_record",
]
