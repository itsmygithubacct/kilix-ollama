from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from kilix_ollama.backend import (
    PackageState,
    VulkanICD,
    VulkanReport,
    discover_icds,
    inspect_vulkan,
    select_backend,
)
from kilix_ollama.errors import BackendUnavailable, RuntimeContractError


def ready_report(*, firmware: bool = True) -> VulkanReport:
    return VulkanReport(
        loader=Path("/usr/lib/libvulkan.so.1"),
        icds=(
            VulkanICD(
                Path("/usr/share/vulkan/icd.d/nouveau_icd.json"),
                "libvulkan_nouveau.so",
                "1.4.305",
            ),
        ),
        drm_drivers=("nouveau",),
        packages=(
            PackageState("libvulkan1", True, "1"),
            PackageState("mesa-vulkan-drivers", True, "1"),
            PackageState("firmware-nvidia-graphics", firmware, "1" if firmware else None),
            PackageState("vulkan-tools", False),
        ),
    )


class BackendSelectionTests(unittest.TestCase):
    def test_auto_remains_cpu_when_vulkan_is_discovered_but_unqualified(self) -> None:
        selection = select_backend("auto", ready_report())
        self.assertEqual(selection.selected, "cpu")
        self.assertEqual(selection.qualified_backends, ("cpu",))

    def test_admitted_nouveau_is_preferred_after_qualification(self) -> None:
        selection = select_backend(
            "auto",
            ready_report(),
            qualified_backends=("cpu", "vulkan", "vulkan-nouveau"),
        )
        self.assertEqual(selection.selected, "vulkan-nouveau")

    def test_explicit_unqualified_backend_fails_without_fallback(self) -> None:
        with self.assertRaises(BackendUnavailable):
            select_backend("vulkan-nouveau", ready_report())

    def test_missing_nouveau_firmware_fails_without_fallback(self) -> None:
        with self.assertRaises(BackendUnavailable):
            select_backend(
                "vulkan-nouveau",
                ready_report(firmware=False),
                qualified_backends=("cpu", "vulkan-nouveau"),
            )

    def test_unknown_qualified_backend_is_a_contract_error(self) -> None:
        with self.assertRaises(RuntimeContractError):
            select_backend("auto", ready_report(), qualified_backends=("cpu", "cuda"))


class VulkanDiscoveryTests(unittest.TestCase):
    def test_reads_bounded_manifests_and_nouveau_drm_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            icd_dir = root / "icd.d"
            icd_dir.mkdir()
            (icd_dir / "nouveau_icd.json").write_text(
                json.dumps(
                    {
                        "file_format_version": "1.0.0",
                        "ICD": {
                            "library_path": "libvulkan_nouveau.so",
                            "api_version": "1.4.305",
                        },
                    }
                ),
                encoding="utf-8",
            )
            drm = root / "drm"
            device = drm / "card0/device"
            device.mkdir(parents=True)
            (device / "uevent").write_text("DRIVER=nouveau\n", encoding="utf-8")
            loader = root / "libvulkan.so.1"
            loader.touch()

            def installed(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                package = args[-1]
                return subprocess.CompletedProcess(args, 0, f"ii \t1.0-{package}\n", "")

            report = inspect_vulkan(
                icd_directories=(icd_dir,),
                drm_root=drm,
                loader_candidates=(loader,),
                runner=installed,
            )
            self.assertTrue(report.nouveau_ready)
            self.assertEqual(report.drm_drivers, ("nouveau",))
            self.assertEqual(len(report.icds), 1)
            self.assertFalse(report.record()["qualification_performed"])

    def test_invalid_and_symlinked_manifests_are_not_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            (root / "bad.json").write_text("not-json", encoding="utf-8")
            (root / "linked.json").symlink_to(target)
            icds, diagnostics = discover_icds((root,))
            self.assertEqual(icds, ())
            self.assertGreaterEqual(len(diagnostics), 2)

    def test_software_icd_is_not_hardware(self) -> None:
        report = VulkanReport(
            loader=Path("/lib/libvulkan.so.1"),
            icds=(VulkanICD(Path("lvp_icd.json"), "libvulkan_lvp.so", "1.4"),),
            drm_drivers=("test",),
            packages=(),
        )
        self.assertFalse(report.hardware_vulkan_discoverable)
        self.assertFalse(report.hardware_vulkan_available)

    def test_discovery_without_runtime_packages_is_not_available(self) -> None:
        report = VulkanReport(
            loader=Path("/lib/libvulkan.so.1"),
            icds=(VulkanICD(Path("intel_icd.json"), "libvulkan_intel.so", "1.4"),),
            drm_drivers=("i915",),
            packages=(),
        )
        self.assertTrue(report.hardware_vulkan_discoverable)
        self.assertFalse(report.hardware_vulkan_available)


if __name__ == "__main__":
    unittest.main()
