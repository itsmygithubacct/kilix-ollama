"""Command-line diagnostics for the local development spine."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .backend import inspect_vulkan, select_backend
from .capacity import co_residency_record
from .doctor import DoctorReport, run_doctor
from .errors import KilixOllamaError
from .paths import RuntimePaths
from .policy import policy_record
from .profiles import runtime_profiles_record
from .service import build_launch_spec, service_contract_record
from . import tts_artifacts


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kilix-ollama",
        description="Inspect the unpublished, artifact-free Kilix Ollama runtime contract.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    doctor = commands.add_parser("doctor", help="run side-effect-free local diagnostics")
    doctor.add_argument("--json", action="store_true", help="emit provisional JSON")
    doctor.add_argument(
        "--backend",
        choices=("auto", "cpu", "vulkan", "vulkan-nouveau"),
        default="auto",
    )

    backend = commands.add_parser("backend", help="inspect backend availability")
    backend.add_argument("--json", action="store_true", help="emit provisional JSON")

    policy = commands.add_parser("policy", help="print the mediation allowlist")
    policy.add_argument("--json", action="store_true", help="retained for CLI symmetry")

    contract = commands.add_parser("service-contract", help="print the isolation contract")
    contract.add_argument("--json", action="store_true", help="retained for CLI symmetry")

    profiles = commands.add_parser(
        "runtime-profiles", help="print measured runtime tuning candidates"
    )
    profiles.add_argument("--json", action="store_true", help="retained for CLI symmetry")

    co_residency = commands.add_parser(
        "co-residency", help="print measured Vulkan LLM/TTS scheduling candidates"
    )
    co_residency.add_argument(
        "--json", action="store_true", help="retained for CLI symmetry"
    )

    tts_models = commands.add_parser(
        "tts-models", help="show the metadata-only optional TTS catalog"
    )
    tts_models.add_argument("--json", action="store_true", help="emit provisional JSON")

    tts_status = commands.add_parser(
        "tts-status", help="inspect an optional local TTS conversion without network access"
    )
    tts_status.add_argument(
        "model", nargs="?", default=tts_artifacts.MODEL_ID, choices=tts_artifacts.MODEL_IDS
    )
    tts_status.add_argument("--verify", action="store_true", help="hash every artifact")
    tts_status.add_argument("--json", action="store_true", help="emit provisional JSON")

    tts_install = commands.add_parser(
        "tts-install", help="explicitly download exact upstream inputs and convert locally"
    )
    tts_install.add_argument(
        "model", nargs="?", default=tts_artifacts.MODEL_ID, choices=tts_artifacts.MODEL_IDS
    )
    tts_install.add_argument(
        "--accept-model-terms",
        action="store_true",
        help="confirm review and acceptance of the pinned upstream terms",
    )
    tts_install.add_argument(
        "--discard-verified-source",
        action="store_true",
        help="remove source inputs after exact local conversion verifies",
    )

    tts_remove = commands.add_parser(
        "tts-remove", help="remove only an exact, fully verified optional TTS install"
    )
    tts_remove.add_argument(
        "model", nargs="?", default=tts_artifacts.MODEL_ID, choices=tts_artifacts.MODEL_IDS
    )
    tts_remove.add_argument(
        "--confirm", action="store_true", help="confirm removal of the verified install"
    )

    spec = commands.add_parser("service-spec", help="describe, but do not run, a service")
    spec.add_argument("--json", action="store_true", help="emit provisional JSON")
    spec.add_argument("--transport", choices=("unix", "netns-tcp"), default="unix")
    spec.add_argument(
        "--backend",
        choices=("auto", "cpu", "vulkan", "vulkan-nouveau"),
        default="auto",
    )
    return parser


def _print_checks(report: DoctorReport) -> None:
    print(f"status: {report.status}")
    for check in report.checks:
        print(f"{check.status:7} {check.name}: {check.detail}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "doctor":
            report = run_doctor(requested_backend=args.backend)
            record = report.record()
            print(_json(record)) if args.json else _print_checks(report)
            return 1 if report.status == "error" else 0
        if args.command == "backend":
            vulkan = inspect_vulkan()
            selection = select_backend("auto", vulkan)
            record = {
                "schema": "kilix.ollama.backend/v0-unfrozen",
                "schema_status": "development-unfrozen",
                "selection": selection.record(),
                "vulkan": vulkan.record(),
            }
            if args.json:
                print(_json(record))
            else:
                print(f"selected: {selection.selected} ({selection.reason})")
                print(f"Vulkan available: {vulkan.hardware_vulkan_available}")
                print(f"Nouveau ready: {vulkan.nouveau_ready}")
            return 0
        if args.command == "policy":
            print(_json(policy_record()))
            return 0
        if args.command == "service-contract":
            print(_json(service_contract_record()))
            return 0
        if args.command == "runtime-profiles":
            print(_json(runtime_profiles_record()))
            return 0
        if args.command == "co-residency":
            print(_json(co_residency_record()))
            return 0
        if args.command == "tts-models":
            record = tts_artifacts.catalog_record()
            if args.json:
                print(_json(record))
            else:
                for model in record["models"]:
                    print(f"model: {model['catalog_id']}")
                    print(f"upstream revision: {model['source_revision']}")
                    print(f"download bytes: {model['download_bytes']}")
                    print(f"converted runtime bytes: {model['runtime_bytes']}")
                    print(f"license: {model['license']} ({model['license_url']})")
                    print(f"attribution: {model['attribution']}")
                    print(f"terms: {model['terms_url']}")
                    print(f"notice: {model['terms_summary']}")
                    print("product voice included: no")
            return 0
        if args.command == "tts-status":
            paths = RuntimePaths.from_environment()
            installed, detail = tts_artifacts.inspect_install(
                args.model,
                paths=paths,
                verify_hashes=args.verify,
            )
            record = {
                "schema": "kilix.ollama.optional-tts-status/v0-unfrozen",
                "schema_status": "development-unfrozen",
                "model": args.model,
                "installed": installed,
                "hashes_verified": bool(installed and args.verify),
                "detail": detail,
                "network_access": False,
            }
            if args.json:
                print(_json(record))
            else:
                print(f"installed: {'yes' if installed else 'no'}")
                print(f"detail: {detail}")
            return 0
        if args.command == "tts-install":
            if args.accept_model_terms:
                print(f"Attribution: Kyutai; license: CC BY 4.0 ({tts_artifacts.LICENSE_URL})")
                print(f"Pinned terms: {tts_artifacts.UPSTREAM_TERMS_URL}")
                print(f"Notice: {tts_artifacts.TERMS_SUMMARY}")
            target, created = tts_artifacts.install_model(
                args.model,
                accept_terms=args.accept_model_terms,
                discard_verified_source=args.discard_verified_source,
                report=print,
            )
            action = "Installed" if created else "Already installed and verified"
            print(f"{action}: {target}")
            return 0
        if args.command == "tts-remove":
            target, removed = tts_artifacts.remove_model(
                args.model,
                confirm=args.confirm,
            )
            action = "Removed" if removed else "Not installed"
            print(f"{action}: {target}")
            return 0
        if args.command == "service-spec":
            vulkan = inspect_vulkan()
            selection = select_backend(args.backend, vulkan)
            spec = build_launch_spec(
                RuntimePaths.from_environment(),
                selection,
                transport=args.transport,
            )
            if args.json:
                print(_json(spec.record()))
            else:
                print("launch enabled: no")
                print(f"transport: {spec.transport}")
                print(f"backend: {spec.backend}")
                print(f"argv: {' '.join(spec.argv)}")
            return 0
    except KilixOllamaError as error:
        print(f"kilix-ollama: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["main"]
