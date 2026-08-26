"""Side-effect-free diagnostics for the unpublished runtime spine."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Sequence

from .backend import (
    DEFAULT_QUALIFIED_BACKENDS,
    BackendSelection,
    VulkanReport,
    inspect_vulkan,
    select_backend,
)
from .errors import BackendUnavailable, RuntimeContractError
from .paths import RuntimePaths, private_directory_status
from .service import DEFAULT_RUNTIME_LAUNCHER, service_contract_record


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str

    def record(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Check, ...]
    paths: RuntimePaths
    vulkan: VulkanReport
    backend: BackendSelection | None

    @property
    def status(self) -> str:
        statuses = {check.status for check in self.checks}
        if "error" in statuses:
            return "error"
        if "warning" in statuses:
            return "warning"
        return "ok"

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.doctor/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "status": self.status,
            "checks": [check.record() for check in self.checks],
            "paths": self.paths.record(),
            "vulkan": self.vulkan.record(),
            "backend": self.backend.record() if self.backend is not None else None,
            "service_contract": service_contract_record(),
            "weights_accessed": False,
        }


def _launcher_check(
    executable: Path,
    *,
    expected_uid: int,
    trusted_root: Path,
) -> Check:
    """Validate a fixed launcher and each retained ancestor without following links."""
    if not executable.is_absolute() or not trusted_root.is_absolute():
        return Check("runtime-launcher", "error", "launcher validation paths are not absolute")
    try:
        relative = executable.relative_to(trusted_root)
    except ValueError:
        return Check("runtime-launcher", "error", "launcher is outside its trusted root")
    candidates = [trusted_root]
    cursor = trusted_root
    for part in relative.parts:
        cursor = cursor / part
        candidates.append(cursor)
    for index, candidate in enumerate(candidates):
        final = index == len(candidates) - 1
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            return Check("runtime-launcher", "error", f"missing: {candidate}")
        except OSError as error:
            return Check("runtime-launcher", "error", f"unreadable: {candidate}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            return Check("runtime-launcher", "error", f"symlink in trusted path: {candidate}")
        if final and not stat.S_ISREG(metadata.st_mode):
            return Check("runtime-launcher", "error", "runtime launcher is not a regular file")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            return Check("runtime-launcher", "error", f"ancestor is not a directory: {candidate}")
        if metadata.st_uid != expected_uid:
            return Check(
                "runtime-launcher",
                "error",
                f"{candidate} is owned by uid {metadata.st_uid}, expected {expected_uid}",
            )
        if metadata.st_mode & 0o022:
            return Check(
                "runtime-launcher",
                "error",
                f"group/world writable trusted path: {candidate}",
            )
    if not os.access(executable, os.X_OK):
        return Check("runtime-launcher", "error", "runtime launcher is not executable")
    return Check("runtime-launcher", "ok", str(executable))


def _socket_check(path: Path, *, uid: int) -> Check:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return Check("mediation-socket", "warning", "service is not active")
    except OSError as error:
        return Check("mediation-socket", "error", f"unreadable: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISSOCK(metadata.st_mode):
        return Check("mediation-socket", "error", "path is not a Unix socket")
    if metadata.st_uid != uid:
        return Check(
            "mediation-socket",
            "error",
            f"owned by uid {metadata.st_uid}, expected {uid}",
        )
    if metadata.st_mode & 0o077:
        return Check("mediation-socket", "error", "socket grants group or other access")
    return Check("mediation-socket", "ok", "private owner socket")


def run_doctor(
    *,
    paths: RuntimePaths | None = None,
    runtime_launcher: Path = DEFAULT_RUNTIME_LAUNCHER,
    runtime_owner_uid: int = 0,
    trusted_root: Path = Path("/"),
    requested_backend: str = "auto",
    vulkan: VulkanReport | None = None,
    qualified_backends: Sequence[str] = tuple(DEFAULT_QUALIFIED_BACKENDS),
    uid: int | None = None,
) -> DoctorReport:
    if uid is not None and (not isinstance(uid, int) or isinstance(uid, bool) or uid < 0):
        raise RuntimeContractError("uid must be a non-negative integer")
    if (
        not isinstance(runtime_owner_uid, int)
        or isinstance(runtime_owner_uid, bool)
        or runtime_owner_uid < 0
    ):
        raise RuntimeContractError("runtime owner uid must be a non-negative integer")
    owner = os.getuid() if uid is None else uid
    layout = RuntimePaths.from_environment(uid=owner) if paths is None else paths
    report = inspect_vulkan() if vulkan is None else vulkan
    checks: list[Check] = []

    private, detail = private_directory_status(layout.runtime_dir, uid=owner)
    checks.append(
        Check(
            "private-runtime-directory",
            "ok" if private else ("warning" if detail == "missing" else "error"),
            detail,
        )
    )
    checks.append(_socket_check(layout.mediation_socket, uid=owner))
    checks.append(
        _launcher_check(
            Path(runtime_launcher),
            expected_uid=runtime_owner_uid,
            trusted_root=Path(trusted_root),
        )
    )
    checks.append(
        Check(
            "transport-boundary",
            "warning",
            "release transport is unselected; candidates are private Unix or owner-isolated netns TCP",
        )
    )
    try:
        selection = select_backend(
            requested_backend,
            report,
            qualified_backends=qualified_backends,
        )
    except (BackendUnavailable, RuntimeContractError) as error:
        selection = None
        checks.append(Check("backend-selection", "error", str(error)))
    else:
        checks.append(
            Check(
                "backend-selection",
                "ok",
                f"{selection.selected}: {selection.reason}",
            )
        )
    if report.hardware_vulkan_available and not {
        "vulkan",
        "vulkan-nouveau",
    }.intersection(qualified_backends):
        checks.append(
            Check(
                "vulkan-qualification",
                "warning",
                "Vulkan is discoverable but remains unqualified; auto selection stays on CPU",
            )
        )
    checks.append(Check("weight-boundary", "ok", "no model or weight path was accessed"))
    return DoctorReport(tuple(checks), layout, report, selection)


__all__ = ["Check", "DoctorReport", "run_doctor"]
