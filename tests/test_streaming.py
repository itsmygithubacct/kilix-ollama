from __future__ import annotations

import json
import unittest

from kilix_ollama.errors import UpstreamProtocolError
from kilix_ollama.streaming import (
    JSONLinesDecoder,
    StreamAccumulator,
    consume_stream,
)


def event(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8") + b"\n"


class JSONLinesDecoderTests(unittest.TestCase):
    def test_every_byte_boundary_reassembles_two_strict_objects(self) -> None:
        payload = event({"value": "世界"}) + event({"done": True})
        for boundary in range(len(payload) + 1):
            with self.subTest(boundary=boundary):
                decoder = JSONLinesDecoder()
                values = list(decoder.feed(payload[:boundary]))
                values.extend(decoder.feed(payload[boundary:]))
                values.extend(decoder.finish())
                self.assertEqual(values, [{"value": "世界"}, {"done": True}])
                self.assertEqual(decoder.event_count, 2)
                self.assertEqual(decoder.byte_count, len(payload))

    def test_crlf_and_terminal_line_without_newline_are_supported(self) -> None:
        decoder = JSONLinesDecoder()
        values = list(decoder.feed(b'{"one":1}\r\n{"two":2}'))
        values.extend(decoder.finish())
        self.assertEqual(values, [{"one": 1}, {"two": 2}])

    def test_empty_duplicate_non_object_and_nonfinite_events_fail(self) -> None:
        invalid = (
            b"\n",
            b'{"a":1,"a":2}\n',
            b"[]\n",
            b'{"value":NaN}\n',
            b"\xff\n",
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(UpstreamProtocolError):
                JSONLinesDecoder().feed(payload)

    def test_byte_line_and_event_limits_are_independent(self) -> None:
        with self.assertRaises(UpstreamProtocolError):
            JSONLinesDecoder(max_total=4, max_line=4).feed(b"12345")
        with self.assertRaises(UpstreamProtocolError):
            JSONLinesDecoder(max_total=10, max_line=3).feed(b"1234")
        decoder = JSONLinesDecoder(max_total=20, max_line=10, max_events=1)
        with self.assertRaises(UpstreamProtocolError):
            decoder.feed(b"{}\n{}\n")

    def test_invalid_limits_and_feed_after_finish_fail(self) -> None:
        for kwargs in (
            {"max_total": 0},
            {"max_line": True},
            {"max_events": -1},
            {"max_total": 2, "max_line": 3},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(UpstreamProtocolError):
                JSONLinesDecoder(**kwargs)  # type: ignore[arg-type]
        decoder = JSONLinesDecoder()
        decoder.finish()
        with self.assertRaises(UpstreamProtocolError):
            decoder.feed(b"{}\n")


class StreamAccumulatorTests(unittest.TestCase):
    def test_generate_stream_collects_content_thinking_and_metrics(self) -> None:
        blocks = (
            event(
                {
                    "model": "local:latest",
                    "response": "hel",
                    "thinking": "plan",
                    "done": False,
                }
            ),
            event(
                {
                    "model": "local:latest",
                    "response": "lo",
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 2,
                    "eval_duration": 30,
                }
            ),
        )
        deltas = []
        result = consume_stream(
            blocks,
            route="/api/generate",
            request_model="local:latest",
            on_delta=deltas.append,
        )
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.thinking, "plan")
        self.assertEqual(result.response_model, "local:latest")
        self.assertEqual(result.done_reason, "stop")
        self.assertEqual(result.metrics, {"eval_count": 2, "eval_duration": 30})
        self.assertEqual(result.event_count, 2)
        self.assertEqual([delta.content for delta in deltas], ["hel", "lo"])
        self.assertNotIn("thinking", result.record())
        self.assertEqual(result.record(include_thinking=True)["thinking"], "plan")

    def test_chat_stream_validates_and_collects_structured_tool_calls(self) -> None:
        call = {
            "type": "function",
            "function": {"name": "kilix.current_time", "arguments": {}},
        }
        result = consume_stream(
            (
                event(
                    {
                        "model": "tool-model",
                        "message": {"role": "assistant", "content": "Checking."},
                        "done": False,
                    }
                ),
                event(
                    {
                        "model": "tool-model",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [call],
                        },
                        "done": True,
                    }
                ),
            ),
            route="/api/chat",
            request_model="tool-model",
        )
        self.assertEqual(result.content, "Checking.")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(
            result.tool_calls[0]["function"]["name"],
            "kilix.current_time",
        )

    def test_model_identity_change_and_post_terminal_data_are_refused(self) -> None:
        accumulator = StreamAccumulator(route="/api/generate", request_model="one")
        accumulator.accept({"model": "one", "response": "", "done": False})
        with self.assertRaises(UpstreamProtocolError):
            accumulator.accept({"model": "two", "response": "", "done": True})

        accumulator = StreamAccumulator(route="/api/generate", request_model="one")
        accumulator.accept({"model": "one", "response": "", "done": True})
        with self.assertRaises(UpstreamProtocolError):
            accumulator.accept({"model": "one", "response": "late", "done": True})

    def test_first_response_model_must_exactly_match_printable_ascii_request(self) -> None:
        for request_model, response_model in (
            ("requested:exact", "different:model"),
            ("requested:exact", "requested:exact\u202e"),
            ("requested:exact", "requested exact"),
        ):
            with self.subTest(response_model=response_model), self.assertRaises(
                UpstreamProtocolError
            ):
                StreamAccumulator(
                    route="/api/generate", request_model=request_model
                ).accept(
                    {"model": response_model, "response": "", "done": True}
                )
        for request_model in ("bad model", "bad\u202e", ""):
            with self.subTest(request_model=request_model), self.assertRaises(
                UpstreamProtocolError
            ):
                StreamAccumulator(route="/api/generate", request_model=request_model)

    def test_missing_terminal_model_message_role_and_metrics_fail_closed(self) -> None:
        cases = (
            ("/api/generate", [{"model": "m", "response": "x", "done": False}]),
            ("/api/generate", [{"response": "x", "done": True}]),
            ("/api/chat", [{"model": "m", "done": True}]),
            (
                "/api/chat",
                [{"model": "m", "message": {"role": "user", "content": "x"}, "done": True}],
            ),
            (
                "/api/generate",
                [{"model": "m", "response": "x", "done": True, "eval_count": -1}],
            ),
        )
        for route, events in cases:
            with self.subTest(route=route, events=events):
                accumulator = StreamAccumulator(route=route, request_model="m")
                with self.assertRaises(UpstreamProtocolError):
                    for value in events:
                        accumulator.accept(value)
                    accumulator.finish()

    def test_upstream_error_does_not_echo_untrusted_error_text(self) -> None:
        accumulator = StreamAccumulator(route="/api/generate", request_model="m")
        with self.assertRaises(UpstreamProtocolError) as captured:
            accumulator.accept({"error": "SECRET PROMPT", "done": True})
        self.assertNotIn("SECRET", str(captured.exception))

    def test_tool_call_shape_unknown_fields_and_argument_limits_are_refused(self) -> None:
        invalid_calls = (
            {"function": {"name": "tool", "arguments": []}},
            {"function": {"name": "tool", "arguments": {}}, "extra": True},
            {"type": "other", "function": {"name": "tool", "arguments": {}}},
            {"function": {"name": "bad name", "arguments": {}}},
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(UpstreamProtocolError):
                StreamAccumulator(route="/api/chat", request_model="m").accept(
                    {
                        "model": "m",
                        "message": {"role": "assistant", "content": "", "tool_calls": [call]},
                        "done": True,
                    }
                )


if __name__ == "__main__":
    unittest.main()
