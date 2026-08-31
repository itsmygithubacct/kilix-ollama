"""Streaming terminal-output sanitization with cross-chunk state.

Model output is untrusted terminal input.  Sanitizing individual HTTP deltas is
not sufficient because an escape sequence can be split at any byte boundary.
This module therefore keeps the minimum parser state needed to consume ANSI,
OSC, DCS, APC, PM, and SOS controls across arbitrary text chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from .errors import UpstreamProtocolError


ESC = "\x1b"
BEL = "\x07"
MAX_CONTROL_SEQUENCE = 16 * 1024
_STRING_INTRODUCERS = frozenset(("]", "P", "_", "^", "X"))
_BIDI_CONTROLS = frozenset(
    chr(value)
    for value in (
        0x061C,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
    )
)


def _plain_character(character: str) -> str:
    if character in ("\n", "\t"):
        return character
    codepoint = ord(character)
    if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
        return ""
    if character in _BIDI_CONTROLS or character == "\ufeff":
        return ""
    if unicodedata.category(character) == "Cs":
        return "\ufffd"
    return character


@dataclass
class TerminalSanitizer:
    """Remove terminal controls while preserving printable streamed text."""

    max_sequence: int = MAX_CONTROL_SEQUENCE

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_sequence, int)
            or isinstance(self.max_sequence, bool)
            or self.max_sequence <= 0
        ):
            raise UpstreamProtocolError("terminal control-sequence limit is invalid")
        self._state = "plain"
        self._sequence_size = 0
        self._string_escape = False
        self._finished = False

    def _begin(self, state: str) -> None:
        self._state = state
        self._sequence_size = 1
        self._string_escape = False

    def _consume_sequence_character(self) -> None:
        self._sequence_size += 1
        if self._sequence_size > self.max_sequence:
            # Continue discarding until a real terminator.  Emitting the tail
            # would turn a bounded-parser failure back into terminal input.
            self._state = "discard-string"

    def feed(self, text: str) -> str:
        if self._finished:
            raise UpstreamProtocolError("terminal sanitizer is already finished")
        if not isinstance(text, str):
            raise UpstreamProtocolError("terminal sanitizer input must be text")
        output: list[str] = []
        for character in text:
            if self._state == "plain":
                if character == ESC:
                    self._begin("escape")
                elif character == "\x9b":
                    self._begin("csi")
                elif character in ("\x90", "\x9d", "\x98", "\x9e", "\x9f"):
                    self._begin("string")
                else:
                    safe = _plain_character(character)
                    if safe:
                        output.append(safe)
                continue

            if self._state == "escape":
                self._consume_sequence_character()
                if character == "[":
                    self._state = "csi"
                elif character in _STRING_INTRODUCERS:
                    self._state = "string"
                elif character == ESC:
                    self._begin("escape")
                elif 0x20 <= ord(character) <= 0x2F:
                    self._state = "escape-intermediate"
                else:
                    self._state = "plain"
                continue

            if self._state == "escape-intermediate":
                self._consume_sequence_character()
                if 0x30 <= ord(character) <= 0x7E:
                    self._state = "plain"
                elif character == ESC:
                    self._begin("escape")
                continue

            if self._state == "csi":
                self._consume_sequence_character()
                codepoint = ord(character)
                if 0x40 <= codepoint <= 0x7E:
                    self._state = "plain"
                elif character == ESC:
                    self._begin("escape")
                continue

            if self._state in ("string", "discard-string"):
                self._sequence_size += 1
                if character == BEL and not self._string_escape:
                    self._state = "plain"
                    self._string_escape = False
                    continue
                if self._string_escape:
                    if character == "\\":
                        self._state = "plain"
                        self._string_escape = False
                        continue
                    self._string_escape = character == ESC
                else:
                    self._string_escape = character == ESC
                if self._sequence_size > self.max_sequence:
                    self._state = "discard-string"
                continue

            raise UpstreamProtocolError("terminal sanitizer entered an invalid state")
        return "".join(output)

    def finish(self) -> str:
        """Drop any incomplete trailing control and make the parser immutable."""
        if self._finished:
            return ""
        self._finished = True
        self._state = "finished"
        self._string_escape = False
        return ""

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.terminal-sanitizer/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "stream_reassembly": True,
            "control_sequence_limit": self.max_sequence,
            "ansi_csi": "strip",
            "osc_dcs_apc_pm_sos": "strip",
            "bidi_controls": "strip",
            "incomplete_trailing_control": "drop",
        }


def sanitize_text(text: str) -> str:
    sanitizer = TerminalSanitizer()
    return sanitizer.feed(text) + sanitizer.finish()


__all__ = [
    "MAX_CONTROL_SEQUENCE",
    "TerminalSanitizer",
    "sanitize_text",
]
