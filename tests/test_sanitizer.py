from __future__ import annotations

import unittest

from kilix_ollama.errors import UpstreamProtocolError
from kilix_ollama.sanitizer import TerminalSanitizer, sanitize_text


class TerminalSanitizerTests(unittest.TestCase):
    SEQUENCES = (
        "\x1b[31m",
        "\x1b[2J",
        "\x1b]0;forged title\x07",
        "\x1b]8;;https://example.invalid\x1b\\",
        "\x1b]52;c;c2VjcmV0\x1b\\",
        "\x1bP1;2;3+qpayload\x1b\\",
        "\x1b_Gf=100;AAAA\x1b\\",
        "\x1b^privacy-message\x1b\\",
        "\x1bXstart-of-string\x1b\\",
        "\x9b31m",
        "\x9dtitle\x07",
    )

    def test_controls_are_removed_at_every_two_chunk_boundary(self) -> None:
        for sequence in self.SEQUENCES:
            for boundary in range(len(sequence) + 1):
                with self.subTest(sequence=repr(sequence), boundary=boundary):
                    sanitizer = TerminalSanitizer()
                    output = sanitizer.feed("before" + sequence[:boundary])
                    output += sanitizer.feed(sequence[boundary:] + "after")
                    output += sanitizer.finish()
                    self.assertEqual(output, "beforeafter")
                    self.assertNotIn("\x1b", output)

    def test_controls_are_removed_when_every_character_is_a_delta(self) -> None:
        for sequence in self.SEQUENCES:
            with self.subTest(sequence=repr(sequence)):
                sanitizer = TerminalSanitizer()
                output = sanitizer.feed("left")
                for character in sequence:
                    output += sanitizer.feed(character)
                output += sanitizer.feed("right")
                output += sanitizer.finish()
                self.assertEqual(output, "leftright")

    def test_incomplete_trailing_sequences_are_dropped(self) -> None:
        for suffix in ("\x1b", "\x1b[31", "\x1b]52;c;secret", "\x1bPpayload\x1b"):
            with self.subTest(suffix=repr(suffix)):
                sanitizer = TerminalSanitizer()
                self.assertEqual(sanitizer.feed("safe" + suffix), "safe")
                self.assertEqual(sanitizer.finish(), "")

    def test_printable_unicode_newlines_and_tabs_survive(self) -> None:
        text = "Hello, 世界 👩‍💻\nnext\tcolumn"
        self.assertEqual(sanitize_text(text), text)

    def test_c0_c1_bidi_bom_and_surrogates_do_not_reach_terminal(self) -> None:
        unsafe = "a\r\b\x00\x7f\x80\u202eb\ufeff\ud800c"
        self.assertEqual(sanitize_text(unsafe), "ab\ufffdc")

    def test_overlong_control_is_discarded_without_unbounded_buffering(self) -> None:
        sanitizer = TerminalSanitizer(max_sequence=8)
        output = sanitizer.feed("a\x1b]" + ("x" * 1000))
        output += sanitizer.feed("\x1b\\b")
        self.assertEqual(output, "ab")

    def test_finished_parser_refuses_more_input(self) -> None:
        sanitizer = TerminalSanitizer()
        sanitizer.finish()
        self.assertEqual(sanitizer.finish(), "")
        with self.assertRaises(UpstreamProtocolError):
            sanitizer.feed("late")

    def test_invalid_limit_and_non_text_input_fail_closed(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(UpstreamProtocolError):
                TerminalSanitizer(max_sequence=value)  # type: ignore[arg-type]
        with self.assertRaises(UpstreamProtocolError):
            TerminalSanitizer().feed(b"not text")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
