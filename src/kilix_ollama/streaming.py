"""Strict, bounded Ollama JSON-lines streaming contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .adapter import decode_json_object
from .errors import UpstreamProtocolError


MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_STREAM_LINE = 4 * 1024 * 1024
MAX_STREAM_EVENTS = 131_072
MAX_RESPONSE_TEXT = 16 * 1024 * 1024
MAX_TOOL_CALLS = 16
MAX_TOOL_ARGUMENT_BYTES = 256 * 1024


class JSONLinesDecoder:
    """Decode strict JSON objects from arbitrarily fragmented UTF-8 bytes."""

    def __init__(
        self,
        *,
        max_total: int = MAX_STREAM_BYTES,
        max_line: int = MAX_STREAM_LINE,
        max_events: int = MAX_STREAM_EVENTS,
    ) -> None:
        for value, label in (
            (max_total, "stream byte limit"),
            (max_line, "stream line limit"),
            (max_events, "stream event limit"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise UpstreamProtocolError(f"{label} is invalid")
        if max_line > max_total:
            raise UpstreamProtocolError("stream line limit exceeds total byte limit")
        self.max_total = max_total
        self.max_line = max_line
        self.max_events = max_events
        self._buffer = bytearray()
        self._total = 0
        self._events = 0
        self._finished = False

    def _decode_line(self, line: bytes) -> dict[str, Any]:
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line:
            raise UpstreamProtocolError("upstream stream contains an empty event")
        if len(line) > self.max_line:
            raise UpstreamProtocolError("upstream stream event exceeds its byte limit")
        self._events += 1
        if self._events > self.max_events:
            raise UpstreamProtocolError("upstream stream exceeds its event-count limit")
        return decode_json_object(line, limit=self.max_line)

    def feed(self, block: bytes) -> tuple[dict[str, Any], ...]:
        if self._finished:
            raise UpstreamProtocolError("JSON-lines decoder is already finished")
        if not isinstance(block, bytes):
            raise UpstreamProtocolError("upstream stream block must be bytes")
        self._total += len(block)
        if self._total > self.max_total:
            raise UpstreamProtocolError("upstream stream exceeds its total byte limit")
        self._buffer.extend(block)
        events: list[dict[str, Any]] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) > self.max_line:
                    raise UpstreamProtocolError(
                        "upstream stream event exceeds its byte limit"
                    )
                break
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            events.append(self._decode_line(line))
        return tuple(events)

    def finish(self) -> tuple[dict[str, Any], ...]:
        if self._finished:
            return ()
        self._finished = True
        if not self._buffer:
            return ()
        line = bytes(self._buffer)
        self._buffer.clear()
        return (self._decode_line(line),)

    @property
    def event_count(self) -> int:
        return self._events

    @property
    def byte_count(self) -> int:
        return self._total


def _bounded_text(value: object, label: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise UpstreamProtocolError(f"upstream {label} is not text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise UpstreamProtocolError(f"upstream {label} contains invalid Unicode") from error
    if size > limit:
        raise UpstreamProtocolError(f"upstream {label} exceeds its byte limit")
    return value


def _tool_call(value: object, index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UpstreamProtocolError(f"upstream tool call {index} is not an object")
    if set(value) - {"function", "id", "type"}:
        raise UpstreamProtocolError(f"upstream tool call {index} has unknown members")
    function = value.get("function")
    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
        raise UpstreamProtocolError(
            f"upstream tool call {index} has an invalid function object"
        )
    name = _bounded_text(function.get("name"), "tool name", limit=256)
    if not name or any(ord(character) < 0x21 or ord(character) > 0x7E for character in name):
        raise UpstreamProtocolError(f"upstream tool call {index} has an invalid name")
    arguments = function.get("arguments")
    if not isinstance(arguments, dict):
        raise UpstreamProtocolError(
            f"upstream tool call {index} arguments are not an object"
        )
    import json

    try:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise UpstreamProtocolError(
            f"upstream tool call {index} arguments are not strict JSON"
        ) from error
    if len(encoded) > MAX_TOOL_ARGUMENT_BYTES:
        raise UpstreamProtocolError(
            f"upstream tool call {index} arguments exceed their byte limit"
        )
    result: dict[str, object] = {
        "function": {"name": name, "arguments": arguments},
        "type": "function",
    }
    call_id = value.get("id")
    if call_id is not None:
        result["id"] = _bounded_text(call_id, "tool call id", limit=256)
    kind = value.get("type", "function")
    if kind != "function":
        raise UpstreamProtocolError(f"upstream tool call {index} has an invalid type")
    return result


@dataclass(frozen=True)
class StreamDelta:
    content: str = ""
    thinking: str = ""
    tool_calls: tuple[dict[str, object], ...] = ()
    done: bool = False


@dataclass(frozen=True)
class StreamResult:
    request_model: str
    response_model: str
    content: str
    thinking: str
    tool_calls: tuple[dict[str, object], ...]
    done_reason: str | None
    metrics: dict[str, int]
    event_count: int

    def record(
        self,
        *,
        include_content: bool = True,
        include_thinking: bool = False,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": "kilix.ollama.inference-result/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "request_model": self.request_model,
            "response_model": self.response_model,
            "done_reason": self.done_reason,
            "metrics": dict(self.metrics),
            "event_count": self.event_count,
            "tool_calls": list(self.tool_calls),
        }
        if include_content:
            record["content"] = self.content
        if include_thinking:
            record["thinking"] = self.thinking
        return record


class StreamAccumulator:
    """Validate a complete `/api/chat` or `/api/generate` event sequence."""

    _METRICS = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )

    def __init__(
        self,
        *,
        route: str,
        request_model: str,
        max_text_bytes: int = MAX_RESPONSE_TEXT,
    ) -> None:
        if route not in ("/api/chat", "/api/generate"):
            raise UpstreamProtocolError("stream accumulator route is invalid")
        if not isinstance(request_model, str) or not request_model:
            raise UpstreamProtocolError("stream request model is invalid")
        try:
            request_model_bytes = request_model.encode("ascii", errors="strict")
        except UnicodeEncodeError as error:
            raise UpstreamProtocolError("stream request model is not printable ASCII") from error
        if len(request_model_bytes) > 1024 or any(
            byte < 0x21 or byte > 0x7E for byte in request_model_bytes
        ):
            raise UpstreamProtocolError("stream request model is outside its ASCII bounds")
        if (
            not isinstance(max_text_bytes, int)
            or isinstance(max_text_bytes, bool)
            or max_text_bytes <= 0
        ):
            raise UpstreamProtocolError("stream text limit is invalid")
        self.route = route
        self.request_model = request_model
        self.max_text_bytes = max_text_bytes
        self._response_model: str | None = None
        self._content: list[str] = []
        self._thinking: list[str] = []
        self._content_bytes = 0
        self._thinking_bytes = 0
        self._tool_calls: list[dict[str, object]] = []
        self._done = False
        self._done_reason: str | None = None
        self._metrics: dict[str, int] = {}
        self._events = 0

    def _append(self, value: object, *, thinking: bool) -> str:
        if value in (None, ""):
            return ""
        label = "thinking" if thinking else "content"
        text = _bounded_text(value, label, limit=self.max_text_bytes)
        size = len(text.encode("utf-8"))
        if thinking:
            self._thinking_bytes += size
            if self._thinking_bytes > self.max_text_bytes:
                raise UpstreamProtocolError("upstream thinking exceeds its total limit")
            self._thinking.append(text)
        else:
            self._content_bytes += size
            if self._content_bytes > self.max_text_bytes:
                raise UpstreamProtocolError("upstream response exceeds its total limit")
            self._content.append(text)
        return text

    def accept(self, event: dict[str, Any]) -> StreamDelta:
        if self._done:
            raise UpstreamProtocolError("upstream emitted data after its terminal event")
        self._events += 1
        error = event.get("error")
        if error is not None:
            # Do not reflect an upstream string: it can contain prompt or model
            # material.  The category is enough for a stable user-facing error.
            raise UpstreamProtocolError("local runtime reported an inference error")
        model_value = event.get("model")
        if model_value is not None:
            model = _bounded_text(model_value, "response model", limit=1024)
            try:
                model_bytes = model.encode("ascii", errors="strict")
            except UnicodeEncodeError as error:
                raise UpstreamProtocolError("upstream response model is not printable ASCII") from error
            if not model_bytes or any(byte < 0x21 or byte > 0x7E for byte in model_bytes):
                raise UpstreamProtocolError("upstream response model is outside its ASCII bounds")
            if model != self.request_model:
                raise UpstreamProtocolError("upstream response model differs from the request")
            if self._response_model is not None and model != self._response_model:
                raise UpstreamProtocolError("upstream changed model identity mid-stream")
            self._response_model = model

        done = event.get("done")
        if not isinstance(done, bool):
            raise UpstreamProtocolError("upstream stream event has no Boolean done member")

        content = ""
        thinking = ""
        calls: tuple[dict[str, object], ...] = ()
        if self.route == "/api/chat":
            message = event.get("message")
            if not isinstance(message, dict):
                raise UpstreamProtocolError("upstream chat event has no message object")
            role = message.get("role")
            if role not in (None, "assistant"):
                raise UpstreamProtocolError("upstream chat event has an invalid role")
            content = self._append(message.get("content", ""), thinking=False)
            thinking = self._append(message.get("thinking", ""), thinking=True)
            raw_calls = message.get("tool_calls", ())
            if raw_calls is None:
                raw_calls = ()
            if not isinstance(raw_calls, (list, tuple)):
                raise UpstreamProtocolError("upstream tool_calls is not an array")
            if len(raw_calls) > MAX_TOOL_CALLS:
                raise UpstreamProtocolError("upstream tool call count exceeds its limit")
            parsed_calls = tuple(_tool_call(value, index) for index, value in enumerate(raw_calls))
            self._tool_calls.extend(parsed_calls)
            if len(self._tool_calls) > MAX_TOOL_CALLS:
                raise UpstreamProtocolError("upstream cumulative tool calls exceed their limit")
            calls = parsed_calls
        else:
            content = self._append(event.get("response", ""), thinking=False)
            thinking = self._append(event.get("thinking", ""), thinking=True)

        if done:
            self._done = True
            raw_reason = event.get("done_reason")
            if raw_reason is not None:
                self._done_reason = _bounded_text(raw_reason, "done reason", limit=128)
            for name in self._METRICS:
                value = event.get(name)
                if value is None:
                    continue
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise UpstreamProtocolError(f"upstream metric {name} is invalid")
                self._metrics[name] = value
        return StreamDelta(content, thinking, calls, done)

    def finish(self) -> StreamResult:
        if not self._done:
            raise UpstreamProtocolError("upstream stream ended without a terminal event")
        if self._response_model is None:
            raise UpstreamProtocolError("upstream stream did not identify its model")
        return StreamResult(
            request_model=self.request_model,
            response_model=self._response_model,
            content="".join(self._content),
            thinking="".join(self._thinking),
            tool_calls=tuple(self._tool_calls),
            done_reason=self._done_reason,
            metrics=dict(self._metrics),
            event_count=self._events,
        )


def consume_stream(
    blocks: Iterable[bytes],
    *,
    route: str,
    request_model: str,
    on_delta: Callable[[StreamDelta], None] | None = None,
) -> StreamResult:
    decoder = JSONLinesDecoder()
    accumulator = StreamAccumulator(route=route, request_model=request_model)
    for block in blocks:
        for event in decoder.feed(block):
            delta = accumulator.accept(event)
            if on_delta is not None:
                on_delta(delta)
    for event in decoder.finish():
        delta = accumulator.accept(event)
        if on_delta is not None:
            on_delta(delta)
    return accumulator.finish()


__all__ = [
    "JSONLinesDecoder",
    "MAX_RESPONSE_TEXT",
    "MAX_STREAM_BYTES",
    "MAX_STREAM_EVENTS",
    "MAX_STREAM_LINE",
    "MAX_TOOL_ARGUMENT_BYTES",
    "MAX_TOOL_CALLS",
    "StreamAccumulator",
    "StreamDelta",
    "StreamResult",
    "consume_stream",
]
