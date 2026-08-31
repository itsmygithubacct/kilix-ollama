"""Artifact-free Kilix Ollama product and diagnostic command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from . import __version__, tts_artifacts
from .backend import inspect_vulkan, select_backend
from .capacity import co_residency_record
from .catalog import find_entry, project_library
from .chat import ChatEngine, conversation_title
from .client import OllamaClient
from .conversations import ConversationStore
from .doctor import DoctorReport, run_doctor
from .errors import KilixOllamaError, RuntimeContractError, RuntimeUnavailable
from .input_output import atomic_write, select_prompt
from .paths import RuntimePaths
from .policy import policy_record
from .profiles import runtime_profiles_record
from .sanitizer import TerminalSanitizer, sanitize_text
from .service import build_launch_spec, service_contract_record
from .streaming import StreamDelta
from .tools import ToolPreview, ToolRegistry


def _json(value: object, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def _runtime_switch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--development-runtime",
        action="store_true",
        help="allow the exact unqualified local R3 candidate; never allows an unknown runtime",
    )


def _sampling(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--context", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kilix-ollama",
        description="Use the private, local-only Kilix Ollama client and inspect its development contracts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    library = commands.add_parser("library", help="browse the exact artifact-free model catalog")
    library.add_argument("--installed", action="store_true", help="merge local installed state")
    library.add_argument("--json", action="store_true")

    models = commands.add_parser("models", help="list installed local models")
    models.add_argument("--json", action="store_true")

    running = commands.add_parser("ps", help="list loaded and running local models")
    running.add_argument("--json", action="store_true")

    show = commands.add_parser("show", help="inspect one installed local model")
    show.add_argument("model")
    show.add_argument("--json", action="store_true")
    _runtime_switch(show)

    ask = commands.add_parser("ask", help="run one prompt without retaining a conversation")
    ask.add_argument("model")
    ask.add_argument("prompt", nargs="*")
    ask.add_argument("--file", type=Path, dest="prompt_file")
    machine = ask.add_mutually_exclusive_group()
    machine.add_argument("--json", action="store_true")
    machine.add_argument("--stream-json", action="store_true")
    ask.add_argument("--output", type=Path)
    ask.add_argument("--overwrite", action="store_true")
    _sampling(ask)
    _runtime_switch(ask)

    chat = commands.add_parser("chat", help="run a persistent local chat")
    chat.add_argument("model", nargs="?")
    chat.add_argument("--conversation")
    chat.add_argument("--title")
    chat.add_argument("--message", help="run one turn and exit")
    chat.add_argument("--json", action="store_true", help="requires --message")
    chat.add_argument("--tools", action="store_true", help="offer the bounded read-only registry")
    chat.add_argument(
        "--tool-root",
        action="append",
        default=[],
        type=Path,
        help="explicit directory a file tool may read; repeatable",
    )
    _sampling(chat)
    _runtime_switch(chat)

    conversations = commands.add_parser("conversations", help="inspect private local chat state")
    conversation_commands = conversations.add_subparsers(dest="conversation_command")
    conversation_list = conversation_commands.add_parser("list", help="list conversations")
    conversation_list.add_argument("--json", action="store_true")
    conversation_show = conversation_commands.add_parser("show", help="show one conversation")
    conversation_show.add_argument("identifier")
    conversation_show.add_argument("--json", action="store_true")
    conversation_show.add_argument("--thinking", action="store_true")
    conversation_delete = conversation_commands.add_parser("delete", help="delete one conversation")
    conversation_delete.add_argument("identifier")
    conversation_delete.add_argument("--confirm", action="store_true")
    conversation_export = conversation_commands.add_parser("export", help="atomically export Markdown")
    conversation_export.add_argument("identifier")
    conversation_export.add_argument("--output", type=Path, required=True)
    conversation_export.add_argument("--overwrite", action="store_true")
    conversation_export.add_argument("--thinking", action="store_true")

    pull = commands.add_parser("pull", help="request an exact model install through shared authority")
    pull.add_argument("model")

    tools = commands.add_parser("tools", help="show the disabled-by-default bounded tool registry")
    tools.add_argument("--json", action="store_true")

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
    co_residency.add_argument("--json", action="store_true", help="retained for CLI symmetry")

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


def _client(args: argparse.Namespace) -> OllamaClient:
    return OllamaClient(
        RuntimePaths.from_environment(),
        allow_unqualified_candidate=bool(getattr(args, "development_runtime", False)),
    )


def _print_model_list(records: Sequence[dict[str, object]]) -> None:
    for record in records:
        name = sanitize_text(str(record.get("name", record.get("model", "unknown"))))
        digest = sanitize_text(str(record.get("digest", "digest unavailable")))
        size = record.get("size")
        suffix = f" — {size} bytes" if isinstance(size, int) else ""
        print(f"{name}  {digest}{suffix}")


def _library(args: argparse.Namespace) -> int:
    installed: Sequence[dict[str, object]] = ()
    if args.installed:
        installed = _client(args).installed_models()
    record = project_library(installed)
    if args.json:
        print(_json(record))
        return 0
    for model in record["curated"]:
        print(f"{sanitize_text(str(model['display_name']))}")
        print(f"  tag: {sanitize_text(str(model['display_tag']))}")
        print(f"  role: {sanitize_text(str(model['role']))}")
        print(f"  payload: {model['payload_bytes']} bytes")
        print(f"  license: {model['license']['id']} (exact engineering input)")
        print(f"  local: {model['local_state']}")
        print("  install: blocked until the shared carrier/receipt authority is admitted")
    return 0


def _ask(args: argparse.Namespace) -> int:
    if args.output is not None and (args.json or args.stream_json):
        raise RuntimeContractError("--output cannot be combined with JSON stdout modes")
    if args.overwrite and args.output is None:
        raise RuntimeContractError("--overwrite requires --output")
    prompt = select_prompt(args.prompt, prompt_file=args.prompt_file, stdin=sys.stdin)
    client = _client(args)
    sanitizer = TerminalSanitizer()

    def emit(delta: StreamDelta) -> None:
        if args.stream_json:
            if delta.content:
                print(
                    _json(
                        {
                            "schema": "kilix.ollama.inference-event/v0-unfrozen",
                            "type": "content",
                            "content": delta.content,
                            "done": delta.done,
                        },
                        compact=True,
                    ),
                    flush=True,
                )
        elif args.output is None and not args.json and delta.content:
            safe = sanitizer.feed(delta.content)
            if safe:
                sys.stdout.write(safe)
                sys.stdout.flush()

    result = client.generate(
        args.model,
        prompt,
        temperature=args.temperature,
        seed=args.seed,
        context=args.context,
        on_delta=emit,
    )
    sanitizer.finish()
    if args.output is not None:
        atomic_write(args.output, result.content, overwrite=args.overwrite)
        print(f"wrote {len(result.content.encode('utf-8'))} bytes", file=sys.stderr)
    elif args.json:
        print(_json(result.record()))
    elif args.stream_json:
        print(
            _json(
                {
                    "schema": "kilix.ollama.inference-event/v0-unfrozen",
                    "type": "terminal",
                    "result": result.record(include_content=False),
                    "done": True,
                },
                compact=True,
            )
        )
    elif result.content and not result.content.endswith("\n"):
        print()
    return 0


def _approval(preview: ToolPreview) -> bool:
    print("\nTool approval required:", file=sys.stderr)
    print(_json(preview.record()), file=sys.stderr)
    print("Approve this call only? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline()
    return answer.strip().casefold() in ("y", "yes")


def _chat(args: argparse.Namespace) -> int:
    if args.json and args.message is None:
        raise RuntimeContractError("--json requires --message")
    if args.tool_root and not args.tools:
        raise RuntimeContractError("--tool-root requires --tools")
    paths = RuntimePaths.from_environment()
    client = OllamaClient(paths, allow_unqualified_candidate=args.development_runtime)
    store = ConversationStore(paths)
    engine = ChatEngine(client, store)
    if args.conversation:
        conversation = store.load(args.conversation)
        if args.model is not None and args.model != conversation.model:
            raise RuntimeContractError("model argument disagrees with the stored conversation")
    else:
        if args.model is None:
            raise RuntimeContractError("a model is required for a new conversation")
        client.require_installed(args.model)
        title = args.title or conversation_title(args.message or "New chat")
        conversation = engine.create(model=args.model, title=title)
    registry = ToolRegistry(args.tool_root) if args.tools else None
    print(f"conversation: {conversation.identifier}", file=sys.stderr)

    def run(prompt: str, *, machine: bool) -> None:
        sanitizer = TerminalSanitizer()

        def emit(delta: StreamDelta) -> None:
            if not machine and delta.content:
                safe = sanitizer.feed(delta.content)
                if safe:
                    sys.stdout.write(safe)
                    sys.stdout.flush()

        turn = engine.turn(
            conversation.identifier,
            prompt,
            registry=registry,
            approve=_approval if registry is not None else None,
            on_delta=None if machine else emit,
            temperature=args.temperature,
            seed=args.seed,
            context=args.context,
        )
        sanitizer.finish()
        if machine:
            print(_json(turn.record()))
        elif turn.result.content and not turn.result.content.endswith("\n"):
            print()

    if args.message is not None:
        run(args.message, machine=args.json)
        return 0
    if not sys.stdin.isatty():
        raise RuntimeContractError("interactive chat requires a terminal; use --message")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not prompt.strip():
            continue
        if prompt.strip() in ("/exit", "/quit"):
            return 0
        run(prompt, machine=False)


def _conversations(args: argparse.Namespace) -> int:
    if args.conversation_command is None:
        raise RuntimeContractError("a conversations subcommand is required")
    store = ConversationStore(RuntimePaths.from_environment())
    if args.conversation_command == "list":
        records = store.list()
        if args.json:
            print(
                _json(
                    {
                        "schema": "kilix.ollama.conversation-list/v0-unfrozen",
                        "conversations": list(records),
                        "count": len(records),
                    }
                )
            )
        else:
            for record in records:
                print(
                    f"{record['id']}  {sanitize_text(str(record['title']))}  "
                    f"[{sanitize_text(str(record['model']))}]  {record['message_count']} messages"
                )
        return 0
    if args.conversation_command == "show":
        conversation = store.load(args.identifier)
        if args.json:
            print(_json(conversation.record()))
        else:
            print(store.export_markdown(args.identifier, include_thinking=args.thinking), end="")
        return 0
    if args.conversation_command == "delete":
        if not args.confirm:
            raise RuntimeContractError("conversation deletion requires --confirm")
        store.delete(args.identifier)
        print(f"deleted conversation {args.identifier}")
        return 0
    if args.conversation_command == "export":
        content = store.export_markdown(args.identifier, include_thinking=args.thinking)
        atomic_write(args.output, content, overwrite=args.overwrite)
        print(f"exported conversation {args.identifier}", file=sys.stderr)
        return 0
    raise RuntimeContractError("unknown conversations subcommand")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "library":
            return _library(args)
        if args.command == "models":
            records = _client(args).installed_models()
            if args.json:
                print(_json({"models": list(records), "count": len(records)}))
            else:
                _print_model_list(records)
            return 0
        if args.command == "ps":
            records = _client(args).running_models()
            if args.json:
                print(_json({"models": list(records), "count": len(records)}))
            else:
                _print_model_list(records)
            return 0
        if args.command == "show":
            print(_json(_client(args).show_model(args.model)))
            return 0
        if args.command == "ask":
            return _ask(args)
        if args.command == "chat":
            return _chat(args)
        if args.command == "conversations":
            return _conversations(args)
        if args.command == "pull":
            entry = find_entry(args.model)
            raise RuntimeUnavailable(
                f"model transfer is blocked for {entry.display_tag}: 0 of 1 shared F100 carrier/receipt authorities are admitted"
            )
        if args.command == "tools":
            print(_json(ToolRegistry().record()))
            return 0
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
            target, removed = tts_artifacts.remove_model(args.model, confirm=args.confirm)
            action = "Removed" if removed else "Not installed"
            print(f"{action}: {target}")
            return 0
        if args.command == "service-spec":
            vulkan = inspect_vulkan()
            selection = select_backend(args.backend, vulkan)
            spec = build_launch_spec(
                RuntimePaths.from_environment(), selection, transport=args.transport
            )
            if args.json:
                print(_json(spec.record()))
            else:
                print("launch enabled: no")
                print(f"transport: {spec.transport}")
                print(f"backend: {spec.backend}")
                print(f"argv: {' '.join(spec.argv)}")
            return 0
    except BrokenPipeError:
        return 0
    except KilixOllamaError as error:
        print(f"kilix-ollama: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["main"]
