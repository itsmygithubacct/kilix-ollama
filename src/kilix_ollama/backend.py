"""Read-only backend discovery and fail-closed backend selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Callable, Iterable, Sequence

from .errors import BackendUnavailable, RuntimeContractError


RUNTIME_PACKAGES = (
    "libvulkan1",
    "mesa-vulkan-drivers",
    "firmware-nvidia-graphics",
    "vulkan-tools",
)
BACKENDS = frozenset(("cpu", "vulkan", "vulkan-nouveau"))
DEFAULT_QUALIFIED_BACKENDS = frozenset(("cpu",))
DEFAULT_ICD_DIRECTORIES = (
    Path("/usr/share/vulkan/icd.d"),
    Path("/etc/vulkan/icd.d"),
)
DEFAULT_DRM_ROOT = Path("/sys/class/drm")
MAX_ICD_BYTES = 1024 * 1024
MAX_ICD_FILES = 128
MAX_DRM_UEVENT_BYTES = 64 * 1024
_SOFTWARE_ICD_MARKERS = ("lavapipe", "lvp", "swiftshader", "gfxstream")
_CARD_NAME = re.compile(r"card[0-9]+\Z")


@dataclass(frozen=True)
class PackageState:
    name: str
    installed: bool
    version: str | None = None
    detail: str = ""

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "installed": self.installed,
            "version": self.version,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class VulkanICD:
    manifest: Path
    library_path: str
    api_version: str | None

    @property
    def identity(self) -> str:
        return f"{self.manifest.stem} {self.library_path}".casefold()

    @property
    def is_nouveau(self) -> bool:
        return "nouveau" in self.identity or "nvk" in self.identity

    @property
    def is_software(self) -> bool:
        words = set(re.split(r"[^a-z0-9]+", self.identity))
        return any(marker in words for marker in _SOFTWARE_ICD_MARKERS)

    def record(self) -> dict[str, object]:
        return {
            "manifest": str(self.manifest),
            "library_path": self.library_path,
            "api_version": self.api_version,
            "nouveau": self.is_nouveau,
            "software": self.is_software,
        }


@dataclass(frozen=True)
class VulkanReport:
    loader: Path | None
    icds: tuple[VulkanICD, ...]
    drm_drivers: tuple[str, ...]
    packages: tuple[PackageState, ...]
    diagnostics: tuple[str, ...] = ()

    def package_installed(self, name: str) -> bool:
        return any(package.name == name and package.installed for package in self.packages)

    @property
    def hardware_vulkan_available(self) -> bool:
        return (
            self.hardware_vulkan_discoverable
            and self.package_installed("libvulkan1")
            and self.package_installed("mesa-vulkan-drivers")
        )

    @property
    def hardware_vulkan_discoverable(self) -> bool:
        return (
            self.loader is not None
            and bool(self.drm_drivers)
            and any(not icd.is_software for icd in self.icds)
        )

    @property
    def nouveau_icd_available(self) -> bool:
        return any(icd.is_nouveau and not icd.is_software for icd in self.icds)

    @property
    def nouveau_ready(self) -> bool:
        return (
            self.loader is not None
            and self.nouveau_icd_available
            and "nouveau" in self.drm_drivers
            and self.package_installed("libvulkan1")
            and self.package_installed("mesa-vulkan-drivers")
            and self.package_installed("firmware-nvidia-graphics")
        )

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.vulkan-discovery/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "loader": str(self.loader) if self.loader is not None else None,
            "icds": [icd.record() for icd in self.icds],
            "drm_drivers": list(self.drm_drivers),
            "packages": [package.record() for package in self.packages],
            "hardware_vulkan_discoverable": self.hardware_vulkan_discoverable,
            "hardware_vulkan_available": self.hardware_vulkan_available,
            "nouveau_ready": self.nouveau_ready,
            "diagnostics": list(self.diagnostics),
            "qualification_performed": False,
        }


@dataclass(frozen=True)
class BackendSelection:
    requested: str
    selected: str
    reason: str
    qualified_backends: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "reason": self.reason,
            "qualified_backends": list(self.qualified_backends),
        }


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _loader_candidates() -> tuple[Path, ...]:
    triplets = {
        "x86_64": "x86_64-linux-gnu",
        "amd64": "x86_64-linux-gnu",
        "aarch64": "aarch64-linux-gnu",
        "arm64": "aarch64-linux-gnu",
        "riscv64": "riscv64-linux-gnu",
    }
    triplet = triplets.get(platform.machine().casefold())
    candidates = [
        Path("/usr/lib/libvulkan.so.1"),
        Path("/lib/libvulkan.so.1"),
        Path("/usr/lib64/libvulkan.so.1"),
        Path("/lib64/libvulkan.so.1"),
    ]
    if triplet is not None:
        candidates[0:0] = [
            Path("/usr/lib") / triplet / "libvulkan.so.1",
            Path("/lib") / triplet / "libvulkan.so.1",
        ]
    return tuple(candidates)


def discover_loader(candidates: Iterable[Path] | None = None) -> Path | None:
    for candidate in _loader_candidates() if candidates is None else candidates:
        path = Path(candidate)
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _parse_icd(path: Path) -> VulkanICD:
    if path.is_symlink():
        raise ValueError("manifest is a symlink")
    stat = path.stat()
    if not path.is_file():
        raise ValueError("manifest is not a regular file")
    if stat.st_size > MAX_ICD_BYTES:
        raise ValueError("manifest exceeds size limit")
    with path.open("rb") as handle:
        payload = handle.read(MAX_ICD_BYTES + 1)
    if len(payload) > MAX_ICD_BYTES:
        raise ValueError("manifest exceeds size limit")
    document = json.loads(payload)
    if not isinstance(document, dict) or not isinstance(document.get("ICD"), dict):
        raise ValueError("manifest has no ICD object")
    icd = document["ICD"]
    library_path = icd.get("library_path")
    api_version = icd.get("api_version")
    if not isinstance(library_path, str) or not library_path or "\x00" in library_path:
        raise ValueError("manifest has no safe library_path")
    if api_version is not None and not isinstance(api_version, str):
        raise ValueError("manifest api_version is not a string")
    return VulkanICD(path, library_path, api_version)


def discover_icds(
    directories: Iterable[Path] = DEFAULT_ICD_DIRECTORIES,
) -> tuple[tuple[VulkanICD, ...], tuple[str, ...]]:
    manifests: list[Path] = []
    for directory in directories:
        try:
            manifests.extend(Path(directory).glob("*.json"))
        except OSError:
            continue
    manifests = sorted(set(manifests), key=lambda path: str(path))
    diagnostics: list[str] = []
    if len(manifests) > MAX_ICD_FILES:
        diagnostics.append(f"ICD manifest count exceeds {MAX_ICD_FILES}; extras ignored")
        manifests = manifests[:MAX_ICD_FILES]
    icds: list[VulkanICD] = []
    for manifest in manifests:
        try:
            icds.append(_parse_icd(manifest))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            diagnostics.append(f"{manifest}: {error}")
    return tuple(icds), tuple(diagnostics)


def discover_drm_drivers(root: Path = DEFAULT_DRM_ROOT) -> tuple[str, ...]:
    drivers: set[str] = set()
    try:
        cards = sorted(Path(root).glob("card*"), key=lambda path: path.name)
    except OSError:
        return ()
    for card in cards:
        if _CARD_NAME.fullmatch(card.name) is None:
            continue
        driver_link = card / "device/driver"
        try:
            if driver_link.exists():
                drivers.add(driver_link.resolve(strict=True).name.casefold())
                continue
        except OSError:
            pass
        uevent = card / "device/uevent"
        try:
            with uevent.open("rb") as handle:
                raw = handle.read(MAX_DRM_UEVENT_BYTES + 1)
            if len(raw) > MAX_DRM_UEVENT_BYTES:
                continue
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if line.startswith("DRIVER=") and line[7:]:
                    drivers.add(line[7:].casefold())
        except OSError:
            continue
    return tuple(sorted(drivers))


def query_runtime_packages(runner: RunCommand = subprocess.run) -> tuple[PackageState, ...]:
    states: list[PackageState] = []
    for package in RUNTIME_PACKAGES:
        try:
            result = runner(
                [
                    "/usr/bin/dpkg-query",
                    "-W",
                    "-f=${db:Status-Abbrev}\\t${Version}\\n",
                    package,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            states.append(PackageState(package, False, detail=f"query failed: {error}"))
            continue
        fields = result.stdout.rstrip("\n").split("\t", maxsplit=1)
        installed = result.returncode == 0 and bool(fields) and fields[0].startswith("ii")
        version = fields[1] if installed and len(fields) == 2 and fields[1] else None
        detail = "installed" if installed else "not installed"
        states.append(PackageState(package, installed, version, detail))
    return tuple(states)


def inspect_vulkan(
    *,
    icd_directories: Iterable[Path] = DEFAULT_ICD_DIRECTORIES,
    drm_root: Path = DEFAULT_DRM_ROOT,
    loader_candidates: Iterable[Path] | None = None,
    runner: RunCommand = subprocess.run,
) -> VulkanReport:
    icds, diagnostics = discover_icds(icd_directories)
    return VulkanReport(
        loader=discover_loader(loader_candidates),
        icds=icds,
        drm_drivers=discover_drm_drivers(drm_root),
        packages=query_runtime_packages(runner),
        diagnostics=diagnostics,
    )


def _available(backend: str, report: VulkanReport) -> bool:
    if backend == "cpu":
        return True
    if backend == "vulkan":
        return report.hardware_vulkan_available
    if backend == "vulkan-nouveau":
        return report.nouveau_ready
    return False


def select_backend(
    requested: str,
    report: VulkanReport,
    *,
    qualified_backends: Sequence[str] = tuple(DEFAULT_QUALIFIED_BACKENDS),
) -> BackendSelection:
    if requested not in BACKENDS | {"auto"}:
        raise RuntimeContractError(f"unknown backend selection: {requested}")
    qualified = tuple(sorted(set(qualified_backends)))
    unknown = set(qualified) - BACKENDS
    if unknown:
        raise RuntimeContractError(f"unknown qualified backend: {sorted(unknown)[0]}")
    if requested == "auto":
        for candidate in ("vulkan-nouveau", "vulkan", "cpu"):
            if candidate in qualified and _available(candidate, report):
                reason = (
                    "highest-preference available qualified backend"
                    if candidate != "cpu"
                    else "CPU is the only available qualified backend"
                )
                return BackendSelection(requested, candidate, reason, qualified)
        raise BackendUnavailable("no qualified backend is available")
    if requested not in qualified:
        raise BackendUnavailable(f"requested backend is not qualified: {requested}")
    if not _available(requested, report):
        raise BackendUnavailable(f"requested backend is not available: {requested}")
    return BackendSelection(requested, requested, "explicit available qualified backend", qualified)


__all__ = [
    "BACKENDS",
    "DEFAULT_QUALIFIED_BACKENDS",
    "BackendSelection",
    "PackageState",
    "VulkanICD",
    "VulkanReport",
    "discover_drm_drivers",
    "discover_icds",
    "discover_loader",
    "inspect_vulkan",
    "query_runtime_packages",
    "select_backend",
]
