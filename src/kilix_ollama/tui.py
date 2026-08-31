"""Small, deterministic full-screen terminal application for Kilix Ollama.

The release-level nested-Kilix proof remains external.  This module owns the
local half of that boundary: bounded input parsing, control-free rendering,
alternate-screen setup, and best-effort restoration on normal exits and
catchable signals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import select
import signal
import stat
import sys
import termios
import tty
from types import FrameType
from typing import Mapping, Sequence
import unicodedata

from .catalog import project_library
from .errors import TerminalError
from .sanitizer import sanitize_text


MAX_INPUT_CHUNK = 4096
MAX_INPUT_BUFFER = 8192
MAX_PASTE_BYTES = 64 * 1024
MAX_TERMINAL_COLUMNS = 320
MAX_TERMINAL_ROWS = 120
MIN_TERMINAL_COLUMNS = 20
MIN_TERMINAL_ROWS = 6

PASTE_BEGIN = b"\x1b[200~"
PASTE_END = b"\x1b[201~"

ENTER_TERMINAL = (
    b"\x1b[?1049h"  # alternate screen
    b"\x1b[?25l"  # hide cursor
    b"\x1b[?2004h"  # bracketed paste
    b"\x1b[?1004h"  # focus events
    b"\x1b[?1000h"  # basic mouse events
    b"\x1b[?1006h"  # SGR mouse coordinates
    b"\x1b[>1u"  # Kitty: disambiguate escape codes
)
LEAVE_TERMINAL = (
    b"\x1b[<u"  # pop Kitty keyboard mode
    b"\x1b[?1006l"
    b"\x1b[?1000l"
    b"\x1b[?1004l"
    b"\x1b[?2004l"
    b"\x1b[0m"
    b"\x1b[?25h"
    b"\x1b[?1049l"
)
REPAINT = b"\x1b[H\x1b[2J"

VIEWS = (
    "home",
    "library",
    "detail",
    "chat",
    "one-off",
    "models",
    "activity",
    "settings",
)

_FIXED_SEQUENCES = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1b[Z": "shift-tab",
    b"\x1b[I": "focus-in",
    b"\x1b[O": "focus-out",
}
_KITTY_SPECIAL = {
    9: "tab",
    13: "enter",
    27: "escape",
    127: "backspace",
    57352: "up",
    57353: "down",
    57354: "right",
    57355: "left",
}


@dataclass(frozen=True)
class TerminalEvent:
    kind: str
    value: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in (
            "focus",
            "interrupt",
            "key",
            "mouse",
            "paste",
            "unknown",
        ):
            raise TerminalError("terminal event kind is invalid")
        if not isinstance(self.value, str):
            raise TerminalError("terminal event value is invalid")
        try:
            size = len(self.value.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise TerminalError("terminal event contains invalid Unicode") from error
        if size > MAX_PASTE_BYTES * 3:
            raise TerminalError("terminal event exceeds its byte limit")


def _utf8_width(first: int) -> int:
    if first < 0x80:
        return 1
    if 0xC2 <= first <= 0xDF:
        return 2
    if 0xE0 <= first <= 0xEF:
        return 3
    if 0xF0 <= first <= 0xF4:
        return 4
    return 0


def _marker_suffix(buffer: bytearray, marker: bytes) -> int:
    maximum = min(len(buffer), len(marker) - 1)
    for size in range(maximum, 0, -1):
        if buffer[-size:] == marker[:size]:
            return size
    return 0


def _bounded_decimal(value: str, *, maximum: int) -> int | None:
    """Parse a short ASCII decimal without invoking unbounded integer conversion."""

    if not 1 <= len(value) <= 10 or any(
        character < "0" or character > "9" for character in value
    ):
        return None
    parsed = 0
    for character in value:
        parsed = (parsed * 10) + (ord(character) - ord("0"))
        if parsed > maximum:
            return None
    return parsed


class TerminalInputParser:
    """Incrementally parse only the terminal controls the application uses."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._paste: bytearray | None = None

    def _paste_event(self) -> TerminalEvent:
        assert self._paste is not None
        text = bytes(self._paste).decode("utf-8", errors="replace")
        return TerminalEvent("paste", sanitize_text(text))

    def _consume_paste(self, events: list[TerminalEvent]) -> bool:
        assert self._paste is not None
        end = self._buffer.find(PASTE_END)
        if end >= 0:
            self._paste.extend(self._buffer[:end])
            del self._buffer[: end + len(PASTE_END)]
            if len(self._paste) > MAX_PASTE_BYTES:
                raise TerminalError("terminal paste exceeds its byte limit")
            events.append(self._paste_event())
            self._paste = None
            return True
        keep = _marker_suffix(self._buffer, PASTE_END)
        movable = len(self._buffer) - keep
        if movable:
            self._paste.extend(self._buffer[:movable])
            del self._buffer[:movable]
        if len(self._paste) + len(self._buffer) > MAX_PASTE_BYTES:
            raise TerminalError("terminal paste exceeds its byte limit")
        return False

    def _kitty_event(self, sequence: bytes) -> TerminalEvent:
        body = sequence[2:-1]
        try:
            fields = body.decode("ascii", errors="strict").split(";")
        except UnicodeDecodeError:
            return TerminalEvent("unknown")
        if not fields or len(fields) > 3:
            return TerminalEvent("unknown")

        key_fields = fields[0].split(":")
        if not 1 <= len(key_fields) <= 3:
            return TerminalEvent("unknown")
        key_codes: list[int | None] = []
        for index, value in enumerate(key_fields):
            if not value:
                if index == 1 and len(key_fields) == 3:
                    key_codes.append(None)
                    continue
                return TerminalEvent("unknown")
            code = _bounded_decimal(value, maximum=0x10FFFF)
            if code is None or 0xD800 <= code <= 0xDFFF:
                return TerminalEvent("unknown")
            key_codes.append(code)

        modifier = 1
        event_type = 1
        if len(fields) >= 2:
            modifier_fields = fields[1].split(":")
            if len(modifier_fields) > 2:
                return TerminalEvent("unknown")
            if modifier_fields[0]:
                parsed_modifier = _bounded_decimal(
                    modifier_fields[0],
                    maximum=256,
                )
                if parsed_modifier is None or parsed_modifier < 1:
                    return TerminalEvent("unknown")
                modifier = parsed_modifier
            elif len(modifier_fields) > 1:
                return TerminalEvent("unknown")
            if len(modifier_fields) == 2:
                parsed_event_type = _bounded_decimal(
                    modifier_fields[1],
                    maximum=3,
                )
                if parsed_event_type not in (1, 2, 3):
                    return TerminalEvent("unknown")
                event_type = parsed_event_type

        if len(fields) == 3:
            text_fields = fields[2].split(":")
            if not 1 <= len(text_fields) <= 32:
                return TerminalEvent("unknown")
            for value in text_fields:
                text_code = _bounded_decimal(value, maximum=0x10FFFF)
                if (
                    text_code is None
                    or text_code < 0x20
                    or 0x7F <= text_code <= 0x9F
                    or 0xD800 <= text_code <= 0xDFFF
                ):
                    return TerminalEvent("unknown")
            # This application does not request associated-text reporting.
            return TerminalEvent("unknown")

        codepoint = key_codes[0]
        assert codepoint is not None
        actual_modifiers = modifier - 1
        base_layout = key_codes[2] if len(key_codes) == 3 else None
        if (
            actual_modifiers & 0b100
            and event_type in (1, 2)
            and (codepoint == ord("c") or base_layout == ord("c"))
        ):
            return TerminalEvent("interrupt")
        if actual_modifiers or event_type != 1:
            return TerminalEvent("unknown")
        if codepoint in _KITTY_SPECIAL:
            return TerminalEvent("key", _KITTY_SPECIAL[codepoint])
        character = chr(codepoint)
        if sanitize_text(character) != character:
            return TerminalEvent("unknown")
        return TerminalEvent("key", character)

    def _csi_event(self, sequence: bytes) -> TerminalEvent:
        fixed = _FIXED_SEQUENCES.get(sequence)
        if fixed is not None:
            if fixed.startswith("focus-"):
                return TerminalEvent("focus", fixed.removeprefix("focus-"))
            return TerminalEvent("key", fixed)
        if sequence.endswith(b"u"):
            return self._kitty_event(sequence)
        if sequence.startswith(b"\x1b[<") and sequence[-1:] in (b"M", b"m"):
            body = sequence[3:-1]
            if len(body) <= 48 and body and all(
                byte in b"0123456789;" for byte in body
            ):
                return TerminalEvent("mouse", "press" if sequence[-1:] == b"M" else "release")
        return TerminalEvent("unknown")

    def _consume_escape(self) -> TerminalEvent | None:
        if self._buffer.startswith(PASTE_BEGIN):
            del self._buffer[: len(PASTE_BEGIN)]
            self._paste = bytearray()
            return None
        if PASTE_BEGIN.startswith(self._buffer):
            return None
        for sequence, name in _FIXED_SEQUENCES.items():
            if self._buffer.startswith(sequence):
                del self._buffer[: len(sequence)]
                if name.startswith("focus-"):
                    return TerminalEvent("focus", name.removeprefix("focus-"))
                return TerminalEvent("key", name)
        if any(sequence.startswith(self._buffer) for sequence in _FIXED_SEQUENCES):
            return None
        if len(self._buffer) == 1:
            return None
        if not self._buffer.startswith(b"\x1b["):
            del self._buffer[0]
            return TerminalEvent("key", "escape")
        final = None
        for index in range(2, len(self._buffer)):
            if 0x40 <= self._buffer[index] <= 0x7E:
                final = index
                break
        if final is None:
            if len(self._buffer) > MAX_INPUT_BUFFER:
                self._buffer.clear()
                return TerminalEvent("unknown")
            return None
        sequence = bytes(self._buffer[: final + 1])
        del self._buffer[: final + 1]
        return self._csi_event(sequence)

    def _consume_plain(self) -> TerminalEvent | None:
        first = self._buffer[0]
        controls = {
            0x03: "interrupt",
            0x09: "tab",
            0x0A: "enter",
            0x0D: "enter",
            0x0C: "refresh",
            0x7F: "backspace",
        }
        if first in controls:
            del self._buffer[0]
            name = controls[first]
            if name == "interrupt":
                return TerminalEvent("interrupt")
            return TerminalEvent("key", name)
        width = _utf8_width(first)
        if width == 0:
            del self._buffer[0]
            return TerminalEvent("unknown")
        if len(self._buffer) < width:
            return None
        payload = bytes(self._buffer[:width])
        del self._buffer[:width]
        try:
            character = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return TerminalEvent("unknown")
        if sanitize_text(character) != character:
            return TerminalEvent("unknown")
        return TerminalEvent("key", character)

    def feed(self, data: bytes) -> tuple[TerminalEvent, ...]:
        if not isinstance(data, bytes) or len(data) > MAX_INPUT_CHUNK:
            raise TerminalError("terminal input chunk is outside its byte limit")
        self._buffer.extend(data)
        if self._paste is None and len(self._buffer) > MAX_INPUT_BUFFER:
            raise TerminalError("terminal input buffer exceeds its byte limit")
        events: list[TerminalEvent] = []
        while self._buffer:
            if self._paste is not None:
                if not self._consume_paste(events):
                    break
                continue
            if self._buffer[0] == 0x1B:
                before = len(self._buffer)
                event = self._consume_escape()
                if event is not None:
                    events.append(event)
                if len(self._buffer) == before:
                    break
                continue
            event = self._consume_plain()
            if event is None:
                break
            events.append(event)
        return tuple(events)

    def flush_pending(self) -> tuple[TerminalEvent, ...]:
        if self._paste is not None:
            return ()
        if self._buffer == b"\x1b":
            self._buffer.clear()
            return (TerminalEvent("key", "escape"),)
        return ()


@dataclass(frozen=True)
class TuiState:
    view: int = 0
    model: int = 0
    help_visible: bool = False
    status: str = "local-only; no runtime contacted"

    def __post_init__(self) -> None:
        if not isinstance(self.view, int) or isinstance(self.view, bool) or not 0 <= self.view < len(VIEWS):
            raise TerminalError("terminal view selection is invalid")
        if not isinstance(self.model, int) or isinstance(self.model, bool) or self.model < 0:
            raise TerminalError("terminal model selection is invalid")
        if not isinstance(self.help_visible, bool):
            raise TerminalError("terminal help state is invalid")
        if not isinstance(self.status, str):
            raise TerminalError("terminal status is invalid")
        try:
            status_size = len(self.status.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise TerminalError("terminal status contains invalid Unicode") from error
        if status_size > 1024:
            raise TerminalError("terminal status is invalid")


def reduce_event(
    state: TuiState,
    event: TerminalEvent,
    *,
    model_count: int,
) -> tuple[TuiState, int | None]:
    if not isinstance(event, TerminalEvent):
        raise TerminalError("terminal event is invalid")
    if not isinstance(model_count, int) or isinstance(model_count, bool) or model_count < 0:
        raise TerminalError("terminal model count is invalid")
    if event.kind == "interrupt":
        return state, 130
    if event.kind == "paste":
        size = len(event.value.encode("utf-8"))
        return replace(
            state,
            status=f"paste ignored safely ({size}/{MAX_PASTE_BYTES * 3} sanitized bytes)",
        ), None
    if event.kind != "key":
        return state, None
    key = event.value.casefold() if len(event.value) == 1 else event.value
    if key == "q":
        return state, 0
    if key == "?":
        return replace(state, help_visible=not state.help_visible), None
    if key in ("tab", "right"):
        return replace(state, view=(state.view + 1) % len(VIEWS), help_visible=False), None
    if key in ("shift-tab", "left"):
        return replace(state, view=(state.view - 1) % len(VIEWS), help_visible=False), None
    if key in ("up", "k") and model_count:
        return replace(state, model=(state.model - 1) % model_count), None
    if key in ("down", "j") and model_count:
        return replace(state, model=(state.model + 1) % model_count), None
    if key == "enter" and VIEWS[state.view] == "library":
        return replace(state, view=VIEWS.index("detail")), None
    if key == "escape":
        if state.help_visible:
            return replace(state, help_visible=False), None
        return replace(state, view=0), None
    if key in "123456789":
        selected = ord(key) - ord("1")
        if selected < len(VIEWS):
            return replace(state, view=selected, help_visible=False), None
    return state, None


def _cell_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1


def _safe_line(value: object, width: int) -> str:
    text = sanitize_text(str(value)).replace("\t", "    ").replace("\r", "")
    text = " ".join(text.splitlines())
    output: list[str] = []
    occupied = 0
    for character in text:
        size = _cell_width(character)
        if occupied + size > width:
            break
        output.append(character)
        occupied += size
    return "".join(output)


def _body_lines(
    state: TuiState,
    catalog: Sequence[Mapping[str, object]],
    width: int,
) -> list[str]:
    view = VIEWS[state.view]
    if state.help_visible:
        return [
            "Keyboard help",
            "  Tab/Right and Shift-Tab/Left change views",
            "  Up/Down or j/k select a curated model",
            "  Enter opens model detail; 1-8 jump to a view",
            "  Bracketed paste is ignored; q exits; Ctrl-C cancels",
        ]
    if view == "home":
        return [
            "Local model workspace",
            f"  Curated metadata: {len(catalog)}/{len(catalog)} entries; transfers: disabled",
            "  Runtime contact: none",
            "  Start with Library, Chat, or One-off from the view bar.",
        ]
    if view == "library":
        lines = ["Curated library — metadata only"]
        for index, model in enumerate(catalog):
            marker = ">" if index == state.model % max(1, len(catalog)) else " "
            lines.append(
                f"{marker} {model.get('display_name', 'unnamed')}  "
                f"[{model.get('local_state', 'unknown')}]"
            )
        lines.append("Enter shows exact local metadata; no selection downloads bytes.")
        return lines
    if view == "detail":
        if not catalog:
            return ["Model detail", "  Curated metadata is unavailable."]
        model = catalog[state.model % len(catalog)]
        license_record = model.get("license", {})
        license_id = license_record.get("id", "unknown") if isinstance(license_record, Mapping) else "unknown"
        return [
            str(model.get("display_name", "unnamed")),
            f"  Tag: {model.get('display_tag', 'unknown')}",
            f"  Role: {model.get('role', 'unknown')}",
            f"  Payload descriptor: {model.get('payload_bytes', 'unknown')} bytes",
            f"  License input: {license_id}",
            f"  Local state: {model.get('local_state', 'unknown')}",
            "  Install: blocked pending shared carrier/receipt authority.",
        ]
    if view == "chat":
        return [
            "Persistent chat",
            "  Conversation storage is owner-private and thinking is collapsed.",
            "  Use `kilix-ollama chat MODEL` for the current streaming surface.",
            "  TUI prompt editing is not release-admitted.",
        ]
    if view == "one-off":
        return [
            "One-off prompt",
            "  Args, stdin, and explicit UTF-8 files are supported.",
            "  JSON, JSON Lines, cancellation, and atomic output are available.",
            "  Use `kilix-ollama ask MODEL PROMPT`.",
        ]
    if view == "models":
        return [
            "Installed and running models",
            "  Runtime contact is opt-in and has not occurred in this TUI.",
            "  Use `models`, `ps`, and `show` through the mediated socket.",
        ]
    if view == "activity":
        return [
            "Activity",
            "  Active transfers: 0/0 observed",
            "  Model transfer remains disabled until shared authority exists.",
        ]
    return [
        "Settings and doctor",
        "  Local-only policy: enforced",
        "  Cloud fallback: disabled",
        "  Tools: disabled unless explicitly selected for one session",
        "  Run `kilix-ollama doctor` for side-effect-free diagnostics.",
    ]


def render_frame(
    state: TuiState,
    catalog: Sequence[Mapping[str, object]],
    *,
    columns: int,
    rows: int,
) -> str:
    if not isinstance(state, TuiState):
        raise TerminalError("terminal state is invalid")
    if (
        not isinstance(columns, int)
        or isinstance(columns, bool)
        or not MIN_TERMINAL_COLUMNS <= columns <= MAX_TERMINAL_COLUMNS
    ):
        raise TerminalError("terminal column count is outside its bounds")
    if (
        not isinstance(rows, int)
        or isinstance(rows, bool)
        or not MIN_TERMINAL_ROWS <= rows <= MAX_TERMINAL_ROWS
    ):
        raise TerminalError("terminal row count is outside its bounds")
    if (
        not isinstance(catalog, Sequence)
        or isinstance(catalog, (str, bytes))
        or len(catalog) > 64
        or not all(isinstance(item, Mapping) for item in catalog)
    ):
        raise TerminalError("terminal catalog is outside its bounds")
    navigation = "  ".join(
        f"{index + 1}:{name.upper() if index == state.view else name}"
        for index, name in enumerate(VIEWS)
    )
    lines = [
        "Kilix Ollama — local-only development",
        navigation,
        "─" * min(columns, 80),
        *_body_lines(state, catalog, columns),
    ]
    footer = f"Status: {state.status} | ?:help q:quit"
    available = rows - 1
    clipped = [_safe_line(line, columns) for line in lines[:available]]
    while len(clipped) < available:
        clipped.append("")
    clipped.append(_safe_line(footer, columns))
    frame = "\n".join(clipped)
    if "\x1b" in frame or "\x9b" in frame:
        raise TerminalError("terminal frame contains an unsafe control")
    return frame


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        except OSError as error:
            raise TerminalError("terminal output failed") from error
        if written <= 0:
            raise TerminalError("terminal output made no progress")
        view = view[written:]


class _TerminalSignal(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class TerminalSession:
    """Own and restore one real terminal session."""

    _SIGNALS = tuple(
        value
        for value in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGQUIT", None),
            getattr(signal, "SIGTERM", None),
        )
        if value is not None
    )

    def __init__(self, input_descriptor: int, output_descriptor: int) -> None:
        if not isinstance(input_descriptor, int) or isinstance(input_descriptor, bool):
            raise TerminalError("terminal input descriptor is invalid")
        if not isinstance(output_descriptor, int) or isinstance(output_descriptor, bool):
            raise TerminalError("terminal output descriptor is invalid")
        self.input_descriptor = input_descriptor
        self.output_descriptor = output_descriptor
        self._attributes: list[object] | None = None
        self._handlers: dict[int, object] = {}
        self._entered = False

    @staticmethod
    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        raise _TerminalSignal(signum)

    def __enter__(self) -> "TerminalSession":
        if self._entered:
            raise TerminalError("terminal session is already active")
        if not os.isatty(self.input_descriptor) or not os.isatty(self.output_descriptor):
            raise TerminalError("full-screen terminal UI requires a real TTY")
        try:
            input_metadata = os.fstat(self.input_descriptor)
            output_metadata = os.fstat(self.output_descriptor)
            if not stat.S_ISCHR(input_metadata.st_mode) or not stat.S_ISCHR(output_metadata.st_mode):
                raise TerminalError("terminal descriptors are not character devices")
            input_identity = (
                input_metadata.st_dev,
                input_metadata.st_ino,
                input_metadata.st_rdev,
            )
            output_identity = (
                output_metadata.st_dev,
                output_metadata.st_ino,
                output_metadata.st_rdev,
            )
            if input_identity != output_identity:
                raise TerminalError("terminal input and output are different TTYs")
            self._attributes = termios.tcgetattr(self.input_descriptor)
            for signum in self._SIGNALS:
                self._handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            tty.setraw(self.input_descriptor, termios.TCSANOW)
            self._entered = True
            _write_all(self.output_descriptor, ENTER_TERMINAL)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> TerminalError | None:
        incomplete = False
        prior_mask: set[signal.Signals] | None = None
        if hasattr(signal, "pthread_sigmask"):
            try:
                prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, self._SIGNALS)
            except (OSError, ValueError):
                pass
        if self._entered:
            try:
                _write_all(self.output_descriptor, LEAVE_TERMINAL)
            except TerminalError:
                incomplete = True
        if self._attributes is not None:
            try:
                termios.tcsetattr(self.input_descriptor, termios.TCSANOW, self._attributes)
            except (OSError, termios.error):
                incomplete = True
        for signum, handler in self._handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                incomplete = True
        if prior_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            except (OSError, ValueError):
                incomplete = True
        self._handlers.clear()
        self._attributes = None
        self._entered = False
        if incomplete:
            return TerminalError("terminal restoration was incomplete")
        return None

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        error = self._restore()
        if error is not None and _kind is None:
            raise error

    def size(self) -> tuple[int, int]:
        try:
            size = os.get_terminal_size(self.output_descriptor)
        except OSError as error:
            raise TerminalError("terminal size is unavailable") from error
        columns = min(MAX_TERMINAL_COLUMNS, max(MIN_TERMINAL_COLUMNS, size.columns))
        rows = min(MAX_TERMINAL_ROWS, max(MIN_TERMINAL_ROWS, size.lines))
        return columns, rows

    def render(self, frame: str) -> None:
        if not self._entered:
            raise TerminalError("terminal session is not active")
        if not isinstance(frame, str) or "\x1b" in frame or "\x9b" in frame:
            raise TerminalError("terminal frame is invalid")
        try:
            payload = frame.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise TerminalError("terminal frame contains invalid Unicode") from error
        if len(payload) > MAX_TERMINAL_COLUMNS * MAX_TERMINAL_ROWS * 4:
            raise TerminalError("terminal frame exceeds its byte limit")
        _write_all(self.output_descriptor, REPAINT + payload)


def _catalog() -> tuple[Mapping[str, object], ...]:
    projection = project_library(())
    curated = projection.get("curated")
    if not isinstance(curated, list) or not all(isinstance(item, Mapping) for item in curated):
        raise TerminalError("curated terminal catalog is invalid")
    return tuple(curated)


def run_tui(
    *,
    input_descriptor: int | None = None,
    output_descriptor: int | None = None,
) -> int:
    try:
        source = sys.stdin.fileno() if input_descriptor is None else input_descriptor
        target = sys.stdout.fileno() if output_descriptor is None else output_descriptor
    except (AttributeError, OSError, ValueError) as error:
        raise TerminalError("terminal descriptors are unavailable") from error
    parser = TerminalInputParser()
    state = TuiState()
    catalog = _catalog()
    try:
        with TerminalSession(source, target) as session:
            columns, rows = session.size()
            session.render(render_frame(state, catalog, columns=columns, rows=rows))
            last_size = (columns, rows)
            while True:
                try:
                    ready, _writable, _exceptional = select.select((source,), (), (), 0.25)
                except InterruptedError:
                    ready = []
                current_size = session.size()
                changed = current_size != last_size
                if ready:
                    try:
                        payload = os.read(source, MAX_INPUT_CHUNK)
                    except OSError as error:
                        raise TerminalError("terminal input failed") from error
                    if not payload:
                        return 0
                    events = parser.feed(payload)
                else:
                    events = parser.flush_pending()
                for event in events:
                    updated, exit_status = reduce_event(
                        state,
                        event,
                        model_count=len(catalog),
                    )
                    changed = changed or updated != state
                    state = updated
                    if exit_status is not None:
                        return exit_status
                if changed:
                    columns, rows = current_size
                    session.render(
                        render_frame(state, catalog, columns=columns, rows=rows)
                    )
                    last_size = current_size
    except _TerminalSignal as caught:
        return 128 + caught.signum


__all__ = [
    "ENTER_TERMINAL",
    "LEAVE_TERMINAL",
    "MAX_INPUT_BUFFER",
    "MAX_PASTE_BYTES",
    "TerminalEvent",
    "TerminalInputParser",
    "TerminalSession",
    "TuiState",
    "VIEWS",
    "reduce_event",
    "render_frame",
    "run_tui",
]
