"""Small, read-only, per-call-approved tool registry for local chat."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import operator
import os
from pathlib import Path
import stat
from typing import Callable, Mapping, Sequence

from .errors import ToolError
from .sanitizer import sanitize_text


MAX_TOOL_OUTPUT_BYTES = 64 * 1024
MAX_TOOL_FILE_BYTES = 256 * 1024
MAX_SEARCH_READ_BYTES = 4 * 1024 * 1024
MAX_SEARCH_FILES = 4096
MAX_SEARCH_ENTRIES = 8192
MAX_SEARCH_DEPTH = 64
MAX_SEARCH_RESULTS = 20
MAX_EXPRESSION_BYTES = 256
MAX_EXPRESSION_NODES = 64
MAX_TOOL_CALLS_PER_TURN = 8
MAX_TOOL_ROUNDS = 4


@dataclass(frozen=True)
class ToolPreview:
    tool_id: str
    risk: str
    normalized_arguments: dict[str, object]
    affected_resources: tuple[str, ...]
    data_returned_to_model: str
    _resource_identity: tuple[int, int] | None = None

    def record(self) -> dict[str, object]:
        return {
            "tool": self.tool_id,
            "risk": self.risk,
            "normalized_arguments": dict(self.normalized_arguments),
            "affected_resources": list(self.affected_resources),
            "data_returned_to_model": self.data_returned_to_model,
            "approval_scope": "this-call-only",
        }


@dataclass(frozen=True)
class ToolResult:
    tool_id: str
    approved: bool
    content: str

    def message(self) -> dict[str, object]:
        return {"role": "tool", "tool_name": self.tool_id, "content": self.content}


@dataclass(frozen=True)
class ToolDefinition:
    identifier: str
    description: str
    risk: str
    parameters: dict[str, object]

    def declaration(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.identifier,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


DEFINITIONS = (
    ToolDefinition(
        "kilix.current_time",
        "Return the current UTC time. No location or network lookup is performed.",
        "low",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"timezone": {"type": "string", "enum": ["UTC"]}},
        },
    ),
    ToolDefinition(
        "kilix.calculator",
        "Evaluate one bounded arithmetic expression without executing code.",
        "low",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"expression": {"type": "string", "maxLength": 256}},
            "required": ["expression"],
        },
    ),
    ToolDefinition(
        "kilix.read_selected_file",
        "Read one UTF-8 file beneath a directory explicitly selected by the user.",
        "local-data",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selection": {"type": "integer", "minimum": 0},
                "path": {"type": "string", "maxLength": 4096},
            },
            "required": ["selection", "path"],
        },
    ),
    ToolDefinition(
        "kilix.search_selected_text",
        "Search bounded UTF-8 files beneath a directory explicitly selected by the user.",
        "local-data",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selection": {"type": "integer", "minimum": 0},
                "path": {"type": "string", "maxLength": 4096},
                "query": {"type": "string", "maxLength": 128},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_SEARCH_RESULTS,
                },
            },
            "required": ["selection", "path", "query"],
        },
    ),
)
_DEFINITION_BY_ID = {definition.identifier: definition for definition in DEFINITIONS}


def _strict_json(value: object, *, maximum: int = MAX_TOOL_OUTPUT_BYTES) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        encoded = payload.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise ToolError("tool result is not strict JSON") from error
    if len(encoded) > maximum:
        raise ToolError("tool result exceeds its output limit")
    return payload


def _arguments(call: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    if not isinstance(call, Mapping) or set(call) - {"function", "id", "type"}:
        raise ToolError("tool call shape is invalid")
    if call.get("type", "function") != "function":
        raise ToolError("tool call type is invalid")
    function = call.get("function")
    if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
        raise ToolError("tool call function shape is invalid")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or name not in _DEFINITION_BY_ID:
        raise ToolError("unknown tool is refused")
    if not isinstance(arguments, dict):
        raise ToolError("tool arguments must be an object")
    _strict_json(arguments, maximum=256 * 1024)
    return name, arguments


def _exact_keys(arguments: Mapping[str, object], *, required: set[str], optional: set[str] = set()) -> None:
    if not required.issubset(arguments) or set(arguments) - required - optional:
        raise ToolError("tool arguments do not match the declared schema")


def _integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ToolError(f"tool argument {label} is outside its bounds")
    return value


def _text(value: object, label: str, *, maximum: int, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ToolError(f"tool argument {label} is not text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise ToolError(f"tool argument {label} contains invalid Unicode") from error
    if size > maximum or (not empty and not value):
        raise ToolError(f"tool argument {label} is outside its bounds")
    if (
        "\x00" in value
        or any(ord(character) < 0x20 for character in value)
        or sanitize_text(value) != value
    ):
        raise ToolError(f"tool argument {label} contains a control character")
    return value


class _Calculator:
    _BINARY = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def __init__(self) -> None:
        self._nodes = 0

    def evaluate(self, expression: str) -> int | float:
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError, MemoryError) as error:
            raise ToolError("calculator expression is invalid") from error
        result = self._visit(tree)
        if isinstance(result, float) and not math.isfinite(result):
            raise ToolError("calculator result is not finite")
        if abs(result) > 1e100:
            raise ToolError("calculator result exceeds its magnitude limit")
        return result

    def _visit(self, node: ast.AST) -> int | float:
        self._nodes += 1
        if self._nodes > MAX_EXPRESSION_NODES:
            raise ToolError("calculator expression exceeds its complexity limit")
        if isinstance(node, ast.Expression):
            return self._visit(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolError("calculator accepts numeric literals only")
            if isinstance(value, int) and value.bit_length() > 512:
                raise ToolError("calculator literal exceeds its magnitude limit")
            if isinstance(value, float) and not math.isfinite(value):
                raise ToolError("calculator literal is not finite")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY:
            return self._UNARY[type(node.op)](self._visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 12 or abs(left) > 1e12:
                    raise ToolError("calculator exponentiation exceeds its limit")
            try:
                result = self._BINARY[type(node.op)](left, right)
            except (ArithmeticError, OverflowError, ValueError) as error:
                raise ToolError("calculator operation failed safely") from error
            if isinstance(result, int) and result.bit_length() > 512:
                raise ToolError("calculator result exceeds its magnitude limit")
            if isinstance(result, float) and not math.isfinite(result):
                raise ToolError("calculator result is not finite")
            return result
        raise ToolError("calculator expression contains a forbidden operation")


class ToolRegistry:
    """Execute only declared read-only tools after one explicit approval."""

    def __init__(
        self,
        selected_roots: Sequence[Path] = (),
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        roots: list[Path] = []
        identities: list[tuple[int, int]] = []
        for index, raw in enumerate(selected_roots):
            try:
                root = Path(raw).resolve(strict=True)
                descriptor = os.open(
                    root,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
            except OSError as error:
                raise ToolError(f"selected tool root {index} is unavailable") from error
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ToolError(f"selected tool root {index} is not a real directory")
                identities.append((metadata.st_dev, metadata.st_ino))
            finally:
                os.close(descriptor)
            roots.append(root)
        if len(roots) > 16:
            raise ToolError("selected tool-root count exceeds its limit")
        self.roots = tuple(roots)
        self._root_identities = tuple(identities)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def declarations(self) -> list[dict[str, object]]:
        return [definition.declaration() for definition in DEFINITIONS]

    def _selection(self, value: object) -> tuple[int, Path]:
        index = _integer(value, "selection", minimum=0, maximum=max(0, len(self.roots) - 1))
        if not self.roots:
            raise ToolError("no local directory was selected for tools")
        return index, self.roots[index]

    def _relative(self, value: object) -> tuple[str, tuple[str, ...]]:
        relative_text = _text(value, "path", maximum=4096)
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ToolError("tool path must stay beneath the selected directory")
        components = tuple(
            component for component in relative.parts if component not in ("", ".")
        )
        normalized = "." if not components else Path(*components).as_posix()
        return normalized, components

    def _open_root(self, index: int) -> int:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self.roots[index], flags)
        except OSError as error:
            raise ToolError("selected tool root cannot be reopened safely") from error
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._root_identities[index]
        ):
            os.close(descriptor)
            raise ToolError("selected tool root identity changed")
        return descriptor

    def _open_beneath(
        self,
        index: int,
        value: object,
        *,
        directory: bool,
    ) -> tuple[str, int, tuple[int, int]]:
        normalized, components = self._relative(value)
        descriptor = self._open_root(index)
        try:
            if not components and not directory:
                raise ToolError("selected tool path is not a regular file")
            for offset, component in enumerate(components):
                final = offset == len(components) - 1
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                if not final or directory:
                    flags |= os.O_DIRECTORY
                else:
                    flags |= getattr(os, "O_NONBLOCK", 0)
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as error:
                    raise ToolError("selected tool path cannot be opened safely") from error
                os.close(descriptor)
                descriptor = child
            metadata = os.fstat(descriptor)
            expected_type = stat.S_ISDIR if directory else stat.S_ISREG
            if not expected_type(metadata.st_mode):
                label = "directory" if directory else "regular file"
                raise ToolError(f"selected tool path is not a {label}")
            return normalized, descriptor, (metadata.st_dev, metadata.st_ino)
        except BaseException:
            os.close(descriptor)
            raise

    def _normalize(self, name: str, arguments: dict[str, object]) -> ToolPreview:
        if name == "kilix.current_time":
            _exact_keys(arguments, required=set(), optional={"timezone"})
            zone = arguments.get("timezone", "UTC")
            if zone != "UTC":
                raise ToolError("current-time tool supports UTC only")
            return ToolPreview(name, "low", {"timezone": "UTC"}, (), "one UTC timestamp")
        if name == "kilix.calculator":
            _exact_keys(arguments, required={"expression"})
            expression = _text(arguments["expression"], "expression", maximum=MAX_EXPRESSION_BYTES)
            return ToolPreview(name, "low", {"expression": expression}, (), "one arithmetic result")
        if name == "kilix.read_selected_file":
            _exact_keys(arguments, required={"selection", "path"})
            index, _root = self._selection(arguments["selection"])
            relative, descriptor, identity = self._open_beneath(
                index, arguments["path"], directory=False
            )
            os.close(descriptor)
            return ToolPreview(
                name,
                "local-data",
                {"selection": index, "path": relative},
                (f"selection[{index}]/{relative}",),
                f"at most {MAX_TOOL_FILE_BYTES} UTF-8 bytes from the selected file",
                identity,
            )
        if name == "kilix.search_selected_text":
            _exact_keys(
                arguments,
                required={"selection", "path", "query"},
                optional={"max_results"},
            )
            index, _root = self._selection(arguments["selection"])
            relative, descriptor, identity = self._open_beneath(
                index, arguments["path"], directory=True
            )
            os.close(descriptor)
            query = _text(arguments["query"], "query", maximum=128)
            maximum = _integer(
                arguments.get("max_results", MAX_SEARCH_RESULTS),
                "max_results",
                minimum=1,
                maximum=MAX_SEARCH_RESULTS,
            )
            return ToolPreview(
                name,
                "local-data",
                {
                    "selection": index,
                    "path": relative,
                    "query": query,
                    "max_results": maximum,
                },
                (f"selection[{index}]/{relative}",),
                f"at most {maximum} matching UTF-8 lines",
                identity,
            )
        raise ToolError("unknown tool is refused")

    def preview(self, call: Mapping[str, object]) -> ToolPreview:
        name, arguments = _arguments(call)
        return self._normalize(name, arguments)

    def _read_file(self, preview: ToolPreview) -> dict[str, object]:
        arguments = preview.normalized_arguments
        index = int(arguments["selection"])
        relative, descriptor, identity = self._open_beneath(
            index, arguments["path"], directory=False
        )
        try:
            if identity != preview._resource_identity:
                raise ToolError("selected file identity changed after approval")
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_TOOL_FILE_BYTES:
                raise ToolError("selected file is not a bounded regular file")
            payload = bytearray()
            while len(payload) <= MAX_TOOL_FILE_BYTES:
                block = os.read(descriptor, min(64 * 1024, MAX_TOOL_FILE_BYTES + 1 - len(payload)))
                if not block:
                    break
                payload.extend(block)
            if len(payload) > MAX_TOOL_FILE_BYTES:
                raise ToolError("selected file exceeds its byte limit")
        finally:
            os.close(descriptor)
        try:
            content = bytes(payload).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ToolError("selected file is not UTF-8 text") from error
        if "\x00" in content:
            raise ToolError("selected file contains binary NUL bytes")
        return {"path": relative, "bytes": len(payload), "content": content}

    def _search(self, preview: ToolPreview) -> dict[str, object]:
        arguments = preview.normalized_arguments
        index = int(arguments["selection"])
        relative, start_descriptor, identity = self._open_beneath(
            index, arguments["path"], directory=True
        )
        if identity != preview._resource_identity:
            os.close(start_descriptor)
            raise ToolError("selected search root identity changed after approval")
        query = str(arguments["query"])
        folded_query = query.casefold()
        maximum = int(arguments["max_results"])
        results: list[dict[str, object]] = []
        files_scanned = 0
        bytes_read = 0
        entries_seen = 0
        start_parts = () if relative == "." else tuple(Path(relative).parts)

        def visit(directory_descriptor: int, display_parts: tuple[str, ...], depth: int) -> bool:
            nonlocal bytes_read, entries_seen, files_scanned
            if depth > MAX_SEARCH_DEPTH:
                raise ToolError("selected search exceeds its directory-depth limit")
            try:
                names = sorted(os.listdir(directory_descriptor))
            except OSError as error:
                raise ToolError("selected search directory cannot be listed safely") from error
            for name in names:
                entries_seen += 1
                if entries_seen > MAX_SEARCH_ENTRIES:
                    raise ToolError("selected search exceeds its entry-count limit")
                if not isinstance(name, str) or sanitize_text(name) != name:
                    continue
                try:
                    metadata = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    flags = (
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    try:
                        child = os.open(name, flags, dir_fd=directory_descriptor)
                    except OSError:
                        continue
                    try:
                        observed = os.fstat(child)
                        if (
                            not stat.S_ISDIR(observed.st_mode)
                            or (observed.st_dev, observed.st_ino)
                            != (metadata.st_dev, metadata.st_ino)
                        ):
                            continue
                        if visit(child, display_parts + (name,), depth + 1):
                            return True
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_TOOL_FILE_BYTES:
                    continue
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                )
                try:
                    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
                except OSError:
                    continue
                try:
                    observed = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(observed.st_mode)
                        or (observed.st_dev, observed.st_ino)
                        != (metadata.st_dev, metadata.st_ino)
                        or observed.st_size > MAX_TOOL_FILE_BYTES
                    ):
                        continue
                    payload = bytearray()
                    while len(payload) <= MAX_TOOL_FILE_BYTES:
                        block = os.read(
                            descriptor,
                            min(64 * 1024, MAX_TOOL_FILE_BYTES + 1 - len(payload)),
                        )
                        if not block:
                            break
                        payload.extend(block)
                finally:
                    os.close(descriptor)
                files_scanned += 1
                if files_scanned > MAX_SEARCH_FILES:
                    raise ToolError("selected search exceeds its file-count limit")
                bytes_read += len(payload)
                if bytes_read > MAX_SEARCH_READ_BYTES:
                    raise ToolError("selected search exceeds its read-byte limit")
                if len(payload) > MAX_TOOL_FILE_BYTES:
                    continue
                try:
                    content = bytes(payload).decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    continue
                if "\x00" in content:
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if folded_query in line.casefold():
                        safe_line = sanitize_text(line)
                        if len(safe_line) > 512:
                            safe_line = safe_line[:512] + "…"
                        results.append(
                            {
                                "path": "/".join(display_parts + (name,)),
                                "line": line_number,
                                "text": safe_line,
                            }
                        )
                        if len(results) >= maximum:
                            return True
            return False

        try:
            truncated = visit(start_descriptor, start_parts, 0)
        finally:
            os.close(start_descriptor)
        return {
            "root_path": relative,
            "query": query,
            "matches": results,
            "files_scanned": files_scanned,
            "truncated": truncated,
        }

    def _run(self, preview: ToolPreview) -> object:
        name = preview.tool_id
        if name == "kilix.current_time":
            value = self._clock()
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ToolError("current-time clock is invalid")
            return {
                "timezone": "UTC",
                "value": value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        if name == "kilix.calculator":
            expression = str(preview.normalized_arguments["expression"])
            return {"expression": expression, "result": _Calculator().evaluate(expression)}
        if name == "kilix.read_selected_file":
            return self._read_file(preview)
        if name == "kilix.search_selected_text":
            return self._search(preview)
        raise ToolError("unknown tool is refused")

    def execute(
        self,
        call: Mapping[str, object],
        *,
        approve: Callable[[ToolPreview], bool] | None,
    ) -> ToolResult:
        preview = self.preview(call)
        approved = bool(approve(preview)) if approve is not None else False
        if not approved:
            return ToolResult(
                preview.tool_id,
                False,
                _strict_json({"status": "denied", "reason": "user-did-not-approve"}),
            )
        try:
            result = self._run(preview)
            content = _strict_json({"status": "ok", "result": result})
        except ToolError as error:
            content = _strict_json({"status": "error", "reason": str(error)})
        return ToolResult(preview.tool_id, True, content)

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.tool-registry/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "tools": [definition.declaration() for definition in DEFINITIONS],
            "tool_count": len(DEFINITIONS),
            "selected_roots": [f"selection[{index}]" for index in range(len(self.roots))],
            "approval": "per-call",
            "persistent_approval": False,
            "write_tools": False,
            "network_tools": False,
            "process_tools": False,
            "shell_tool": False,
            "limits": {
                "calls_per_turn": MAX_TOOL_CALLS_PER_TURN,
                "rounds": MAX_TOOL_ROUNDS,
                "output_bytes": MAX_TOOL_OUTPUT_BYTES,
            },
            "enabled_by_default": False,
            "release_admission": False,
        }


__all__ = [
    "DEFINITIONS",
    "MAX_SEARCH_RESULTS",
    "MAX_TOOL_CALLS_PER_TURN",
    "MAX_TOOL_OUTPUT_BYTES",
    "MAX_TOOL_ROUNDS",
    "ToolDefinition",
    "ToolPreview",
    "ToolRegistry",
    "ToolResult",
]
