from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time
import unittest

from kilix_ollama.catalog import project_library
from kilix_ollama.errors import TerminalError
from kilix_ollama.tui import (
    ENTER_TERMINAL,
    LEAVE_TERMINAL,
    MAX_INPUT_BUFFER,
    MAX_PASTE_BYTES,
    REPAINT,
    TerminalEvent,
    TerminalInputParser,
    TerminalSession,
    TuiState,
    VIEWS,
    reduce_event,
    render_frame,
)


def fragmented(payload: bytes) -> tuple[TerminalEvent, ...]:
    parser = TerminalInputParser()
    events: list[TerminalEvent] = []
    for byte in payload:
        events.extend(parser.feed(bytes((byte,))))
    return tuple(events)


def set_size(descriptor: int, *, columns: int = 96, rows: int = 28) -> None:
    fcntl.ioctl(
        descriptor,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", rows, columns, 0, 0),
    )


def drain(descriptor: int) -> bytes:
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    fcntl.fcntl(descriptor, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    output = bytearray()
    try:
        while True:
            try:
                block = os.read(descriptor, 64 * 1024)
            except BlockingIOError:
                break
            if not block:
                break
            output.extend(block)
    finally:
        fcntl.fcntl(descriptor, fcntl.F_SETFL, flags)
    return bytes(output)


class TerminalInputParserTests(unittest.TestCase):
    def test_navigation_focus_mouse_and_kitty_sequences_survive_every_boundary(self) -> None:
        cases = (
            (b"\x1b[A", TerminalEvent("key", "up")),
            (b"\x1b[B", TerminalEvent("key", "down")),
            (b"\x1b[C", TerminalEvent("key", "right")),
            (b"\x1b[D", TerminalEvent("key", "left")),
            (b"\x1b[Z", TerminalEvent("key", "shift-tab")),
            (b"\x1b[I", TerminalEvent("focus", "in")),
            (b"\x1b[O", TerminalEvent("focus", "out")),
            (b"\x1b[113;1u", TerminalEvent("key", "q")),
            (b"\x1b[57352u", TerminalEvent("key", "up")),
            (b"\x1b[<0;12;4M", TerminalEvent("mouse", "press")),
            (b"\x1b[<0;12;4m", TerminalEvent("mouse", "release")),
        )
        for sequence, expected in cases:
            with self.subTest(sequence=sequence):
                self.assertEqual(fragmented(sequence), (expected,))

    def test_bracketed_paste_is_one_inert_sanitized_event(self) -> None:
        sequence = b"\x1b[200~q\x1b[31mprivate\x1b[201~"
        events = fragmented(sequence)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "paste")
        self.assertEqual(events[0].value, "qprivate")
        state, status = reduce_event(TuiState(), events[0], model_count=3)
        self.assertIsNone(status)
        self.assertIn("paste ignored safely", state.status)

    def test_plain_utf8_controls_and_lone_escape_are_deterministic(self) -> None:
        parser = TerminalInputParser()
        self.assertEqual(parser.feed("世".encode("utf-8")), (TerminalEvent("key", "世"),))
        self.assertEqual(parser.feed(b"\x03"), (TerminalEvent("interrupt"),))
        self.assertEqual(parser.feed(b"\x1b"), ())
        self.assertEqual(parser.flush_pending(), (TerminalEvent("key", "escape"),))
        self.assertEqual(parser.feed(b"\x80"), (TerminalEvent("unknown"),))

    def test_input_and_paste_limits_fail_closed(self) -> None:
        with self.assertRaises(TerminalError):
            TerminalInputParser().feed(b"x" * 4097)

        parser = TerminalInputParser()
        parser.feed(b"\x1b[")
        with self.assertRaises(TerminalError):
            for _index in range((MAX_INPUT_BUFFER // 4096) + 2):
                parser.feed(b"1" * 4096)

        parser = TerminalInputParser()
        parser.feed(b"\x1b[200~")
        with self.assertRaises(TerminalError):
            for _index in range((MAX_PASTE_BYTES // 4096) + 2):
                parser.feed(b"p" * 4096)


class TuiStateAndRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = tuple(project_library(())["curated"])

    def test_reducer_navigates_every_view_and_keeps_paste_inert(self) -> None:
        state = TuiState()
        observed = []
        for _index in range(len(VIEWS)):
            observed.append(VIEWS[state.view])
            state, status = reduce_event(
                state,
                TerminalEvent("key", "tab"),
                model_count=3,
            )
            self.assertIsNone(status)
        self.assertEqual(tuple(observed), VIEWS)
        self.assertEqual(state.view, 0)

        state, _status = reduce_event(
            replace_state := TuiState(view=VIEWS.index("library")),
            TerminalEvent("key", "down"),
            model_count=3,
        )
        self.assertEqual(replace_state.model, 0)
        self.assertEqual(state.model, 1)
        state, _status = reduce_event(state, TerminalEvent("key", "enter"), model_count=3)
        self.assertEqual(VIEWS[state.view], "detail")
        state, exit_status = reduce_event(state, TerminalEvent("key", "q"), model_count=3)
        self.assertEqual(exit_status, 0)
        _state, exit_status = reduce_event(
            state,
            TerminalEvent("interrupt"),
            model_count=3,
        )
        self.assertEqual(exit_status, 130)

    def test_every_view_renders_at_narrow_and_wide_bounds(self) -> None:
        for view in range(len(VIEWS)):
            for columns, rows in ((20, 6), (80, 24), (320, 120)):
                with self.subTest(view=view, columns=columns, rows=rows):
                    frame = render_frame(
                        TuiState(view=view),
                        self.catalog,
                        columns=columns,
                        rows=rows,
                    )
                    self.assertEqual(len(frame.splitlines()), rows)
                    self.assertNotIn("\x1b", frame)
                    for line in frame.splitlines():
                        self.assertLessEqual(len(line), columns)

    def test_untrusted_catalog_and_status_controls_never_enter_frame(self) -> None:
        malicious = (
            {
                "display_name": "safe\x1b]52;c;secret\x07\u202eforged",
                "display_tag": "tag\x9b31m",
                "role": "chat\nsecond",
                "payload_bytes": 1,
                "local_state": "missing\x1b[2J",
                "license": {"id": "MIT\x1b[31m"},
            },
        )
        frame = render_frame(
            TuiState(view=VIEWS.index("detail"), status="ok\x1b[31m\u202ebad"),
            malicious,
            columns=80,
            rows=24,
        )
        self.assertNotIn("\x1b", frame)
        self.assertNotIn("\u202e", frame)
        self.assertNotIn("secret", frame)

    def test_invalid_state_dimensions_catalog_and_events_fail_closed(self) -> None:
        for state in (
            {"view": -1},
            {"view": len(VIEWS)},
            {"model": -1},
            {"help_visible": 1},
            {"status": "x" * 1025},
            {"status": "\ud800"},
        ):
            with self.subTest(state=state), self.assertRaises(TerminalError):
                TuiState(**state)  # type: ignore[arg-type]
        for columns, rows in ((19, 24), (321, 24), (80, 5), (80, 121)):
            with self.subTest(columns=columns, rows=rows), self.assertRaises(TerminalError):
                render_frame(TuiState(), self.catalog, columns=columns, rows=rows)
        with self.assertRaises(TerminalError):
            reduce_event(TuiState(), TerminalEvent("key", "q"), model_count=-1)
        with self.assertRaises(TerminalError):
            TerminalEvent("invented")
        with self.assertRaises(TerminalError):
            TerminalEvent("key", "\ud800")
        with self.assertRaises(TerminalError):
            TerminalEvent("key", "x" * (MAX_PASTE_BYTES * 3 + 1))
        with self.assertRaises(TerminalError):
            render_frame(TuiState(), ("not-a-record",), columns=80, rows=24)  # type: ignore[arg-type]
        with self.assertRaises(TerminalError):
            render_frame(object(), self.catalog, columns=80, rows=24)  # type: ignore[arg-type]


class TerminalSessionTests(unittest.TestCase):
    def test_session_enters_renders_and_restores_exact_attributes(self) -> None:
        master, slave = pty.openpty()
        try:
            set_size(slave)
            before = termios.tcgetattr(slave)
            with TerminalSession(slave, slave) as session:
                during = termios.tcgetattr(slave)
                self.assertFalse(during[3] & termios.ECHO)
                self.assertFalse(during[3] & termios.ICANON)
                self.assertEqual(session.size(), (96, 28))
                session.render("safe\nframe")
            after = termios.tcgetattr(slave)
            self.assertEqual(after, before)
            output = drain(master)
            self.assertIn(ENTER_TERMINAL, output)
            self.assertIn(REPAINT + b"safe\nframe", output)
            self.assertIn(LEAVE_TERMINAL, output)
        finally:
            os.close(master)
            os.close(slave)

    def test_exception_and_rejected_frame_still_restore(self) -> None:
        master, slave = pty.openpty()
        try:
            set_size(slave)
            before = termios.tcgetattr(slave)
            with self.assertRaises(TerminalError):
                with TerminalSession(slave, slave) as session:
                    session.render("unsafe\x1b[2J")
            self.assertEqual(termios.tcgetattr(slave), before)
            self.assertIn(LEAVE_TERMINAL, drain(master))
        finally:
            os.close(master)
            os.close(slave)

    def test_non_tty_descriptors_are_rejected_without_output(self) -> None:
        reader, writer = os.pipe()
        output_reader, output_writer = os.pipe()
        try:
            with self.assertRaises(TerminalError):
                with TerminalSession(reader, output_writer):
                    self.fail("non-TTY session entered")
            os.set_blocking(output_reader, False)
            with self.assertRaises(BlockingIOError):
                os.read(output_reader, 1)
        finally:
            for descriptor in (reader, writer, output_reader, output_writer):
                os.close(descriptor)

    def test_different_input_and_output_ttys_are_rejected(self) -> None:
        first_master, first_slave = pty.openpty()
        second_master, second_slave = pty.openpty()
        try:
            with self.assertRaises(TerminalError):
                with TerminalSession(first_slave, second_slave):
                    self.fail("mismatched TTY session entered")
            self.assertNotIn(ENTER_TERMINAL, drain(second_master))
        finally:
            for descriptor in (
                first_master,
                first_slave,
                second_master,
                second_slave,
            ):
                os.close(descriptor)


class TuiSubprocessTests(unittest.TestCase):
    def run_session(self, action: str) -> tuple[int, bytes, list[object], list[object]]:
        launcher = Path(__file__).resolve().parents[1] / "kilix-ollama"
        master, slave = pty.openpty()
        set_size(slave)
        before = termios.tcgetattr(slave)
        process = subprocess.Popen(
            (sys.executable, str(launcher), "tui"),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        output = bytearray()
        try:
            deadline = time.monotonic() + 5
            while ENTER_TERMINAL not in output and time.monotonic() < deadline:
                ready, _write, _error = select.select((master,), (), (), 0.1)
                if ready:
                    output.extend(os.read(master, 64 * 1024))
            self.assertIn(ENTER_TERMINAL, output)
            if action == "quit":
                os.write(master, b"q")
            elif action == "interrupt-byte":
                os.write(master, b"\x03")
            elif action == "terminate-signal":
                process.send_signal(signal.SIGTERM)
            else:
                self.fail("unknown fixture action")
            status = process.wait(timeout=5)
            output.extend(drain(master))
            after = termios.tcgetattr(slave)
            return status, bytes(output), before, after
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
            os.close(slave)

    def test_quit_interrupt_and_termination_restore_real_pty(self) -> None:
        cases = (
            ("quit", 0),
            ("interrupt-byte", 130),
            ("terminate-signal", 128 + signal.SIGTERM),
        )
        for action, expected_status in cases:
            with self.subTest(action=action):
                status, output, before, after = self.run_session(action)
                self.assertEqual(status, expected_status)
                self.assertEqual(after, before)
                self.assertIn(ENTER_TERMINAL, output)
                self.assertIn(LEAVE_TERMINAL, output)
                self.assertNotIn(b"Traceback", output)


if __name__ == "__main__":
    unittest.main()
