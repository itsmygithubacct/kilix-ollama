"""Version-gated client for the private Kilix Ollama mediation socket."""

from __future__ import annotations

import base64
import binascii
import http.client
import json
import os
from pathlib import Path
import socket
import stat
from typing import Callable, Iterable, Mapping, Sequence

from .adapter import (
    CANDIDATE_VERSION,
    MAX_METADATA_RESPONSE,
    compatibility_for_version,
    decode_json_object,
    parse_model_list,
    parse_version,
)
from .attachments import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_TOTAL_BYTES,
    MAX_ATTACHMENTS_PER_MESSAGE,
    inspect_image,
)
from .conversations import validate_message
from .errors import (
    ConversationError,
    RuntimeContractError,
    RuntimeUnavailable,
    UpstreamProtocolError,
)
from .mediation import UnixHTTPConnection
from .paths import RuntimePaths, private_directory_status
from .policy import MAX_REQUEST_BODY, ROUTE_METHODS, has_explicit_cloud_source
from .streaming import (
    JSONLinesDecoder,
    StreamAccumulator,
    StreamDelta,
    StreamResult,
)


DEFAULT_TIMEOUT = 300.0
MAX_CLIENT_MESSAGES = 512
MAX_CLIENT_MESSAGE_BYTES = 8 * 1024 * 1024
_STREAM_CONTENT_TYPES = frozenset(
    ("application/json", "application/jsonl", "application/x-jsonlines", "application/x-ndjson")
)


def validate_model_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError("model name must be non-empty text")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise RuntimeContractError("model name must be printable ASCII") from error
    if len(encoded) > 1024 or any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise RuntimeContractError("model name is outside its printable ASCII bounds")
    if has_explicit_cloud_source(value):
        raise RuntimeContractError("cloud-tagged models are not available")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise RuntimeContractError("request is not strict JSON") from error
    if len(payload) > MAX_REQUEST_BODY:
        raise RuntimeContractError("request exceeds the mediation byte limit")
    return payload


def _content_type(response: http.client.HTTPResponse) -> str | None:
    values = response.getheaders()
    found: str | None = None
    for name, value in values:
        if name.casefold() != "content-type":
            continue
        if found is not None:
            raise UpstreamProtocolError("local runtime repeated Content-Type")
        found = value.split(";", 1)[0].strip().casefold()
    return found


def _validate_images(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= MAX_ATTACHMENTS_PER_MESSAGE
    ):
        raise RuntimeContractError("chat images are not a bounded array")
    checked: list[str] = []
    total = 0
    for image in value:
        if not isinstance(image, str) or not image:
            raise RuntimeContractError("chat image is not base64 text")
        try:
            encoded = image.encode("ascii", errors="strict")
            payload = base64.b64decode(encoded, validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as error:
            raise RuntimeContractError("chat image is not canonical base64") from error
        if base64.b64encode(payload) != encoded:
            raise RuntimeContractError("chat image is not canonical base64")
        if not 1 <= len(payload) <= MAX_ATTACHMENT_BYTES:
            raise RuntimeContractError("chat image exceeds its byte limit")
        try:
            inspect_image(payload)
        except ConversationError as error:
            raise RuntimeContractError(str(error)) from error
        total += len(payload)
        if total > MAX_ATTACHMENT_TOTAL_BYTES:
            raise RuntimeContractError("chat images exceed their total byte limit")
        if image in checked:
            raise RuntimeContractError("chat images repeat the same payload")
        checked.append(image)
    return checked


class OllamaClient:
    """Call one same-UID mediation socket; never a caller-supplied URL."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        allow_unqualified_candidate: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        owner_uid: int | None = None,
    ) -> None:
        if not isinstance(allow_unqualified_candidate, bool):
            raise RuntimeContractError("development-runtime flag must be Boolean")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise RuntimeContractError("client timeout must be positive")
        self.paths = paths
        self.allow_unqualified_candidate = allow_unqualified_candidate
        self.timeout = float(timeout)
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        if (
            not isinstance(self.owner_uid, int)
            or isinstance(self.owner_uid, bool)
            or self.owner_uid < 0
        ):
            raise RuntimeContractError("client owner uid must be non-negative")
        self._version: str | None = None

    def _validate_socket(self) -> None:
        private, detail = private_directory_status(
            self.paths.runtime_dir, uid=self.owner_uid
        )
        if not private:
            raise RuntimeUnavailable(f"private runtime directory is unavailable: {detail}")
        try:
            metadata = self.paths.mediation_socket.lstat()
        except FileNotFoundError as error:
            raise RuntimeUnavailable("private Ollama mediation socket is not running") from error
        except OSError as error:
            raise RuntimeUnavailable("private Ollama mediation socket cannot be inspected") from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != self.owner_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeUnavailable("private Ollama mediation socket is not owner-only")

    def _begin(
        self,
        method: str,
        path: str,
        payload: bytes = b"",
    ) -> tuple[UnixHTTPConnection, http.client.HTTPResponse]:
        self._validate_socket()
        connection = UnixHTTPConnection(self.paths.mediation_socket, timeout=self.timeout)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            connection.putheader("Host", "localhost")
            connection.putheader("Accept", "application/json")
            connection.putheader("Connection", "close")
            if payload:
                connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(payload)))
            connection.endheaders(payload)
            response = connection.getresponse()
            if not 100 <= response.status <= 599:
                raise UpstreamProtocolError("local runtime returned an invalid HTTP status")
            if not 200 <= response.status <= 299:
                # Drain only a small bounded prefix, without reflecting it.  It
                # may contain prompt content or registry-controlled text.
                response.read(min(64 * 1024, MAX_METADATA_RESPONSE))
                raise UpstreamProtocolError(
                    f"local runtime request failed with HTTP {response.status}"
                )
            return connection, response
        except (RuntimeUnavailable, UpstreamProtocolError):
            connection.close()
            raise
        except (OSError, socket.error, http.client.HTTPException) as error:
            connection.close()
            raise RuntimeUnavailable("private Ollama mediation request failed") from error

    def _request_json(
        self,
        method: str,
        path: str,
        value: object | None = None,
        *,
        limit: int = MAX_METADATA_RESPONSE,
    ) -> dict[str, object]:
        payload = b"" if value is None else _json_bytes(value)
        connection, response = self._begin(method, path, payload)
        try:
            content_type = _content_type(response)
            if content_type not in (None, "application/json"):
                raise UpstreamProtocolError("local runtime returned an unexpected content type")
            body = response.read(limit + 1)
            if len(body) > limit:
                raise UpstreamProtocolError("local runtime response exceeds its byte limit")
            return decode_json_object(body, limit=limit)
        except (OSError, http.client.HTTPException) as error:
            raise RuntimeUnavailable("private Ollama response ended unexpectedly") from error
        finally:
            connection.close()

    def version(self, *, refresh: bool = False) -> str:
        if self._version is None or refresh:
            response = self._request_json("GET", "/api/version")
            self._version = parse_version(_json_bytes(response))
        return self._version

    def _authorize(self, route: str) -> None:
        if route not in ROUTE_METHODS:
            raise RuntimeContractError(f"client route is not mediated: {route}")
        # Admission is per protected request. A cached health result must not
        # survive a runtime restart or socket replacement.
        version = self.version(refresh=True)
        compatibility = compatibility_for_version(version)
        if route in compatibility.allowed_routes:
            return
        if self.allow_unqualified_candidate and version == CANDIDATE_VERSION:
            return
        if version == CANDIDATE_VERSION:
            raise RuntimeUnavailable(
                "runtime candidate is not release-qualified; pass --development-runtime only for local development"
            )
        raise RuntimeUnavailable("runtime version is not eligible for inference or management")

    def installed_models(self) -> tuple[dict[str, object], ...]:
        self._authorize("/api/tags")
        response = self._request_json("GET", "/api/tags")
        return parse_model_list(_json_bytes(response), route="/api/tags")

    def running_models(self) -> tuple[dict[str, object], ...]:
        self._authorize("/api/ps")
        response = self._request_json("GET", "/api/ps")
        return parse_model_list(_json_bytes(response), route="/api/ps")

    def show_model(self, model: str) -> dict[str, object]:
        checked = validate_model_name(model)
        self._authorize("/api/show")
        return self._request_json("POST", "/api/show", {"model": checked})

    def require_installed(self, model: str) -> dict[str, object]:
        checked = validate_model_name(model)
        matches: list[dict[str, object]] = []
        for record in self.installed_models():
            name = record.get("name", record.get("model"))
            if name == checked:
                matches.append(record)
        if len(matches) != 1:
            if not matches:
                raise RuntimeUnavailable(
                    "model is not installed; noninteractive inference never pulls implicitly"
                )
            raise UpstreamProtocolError("local runtime reported a duplicate model identity")
        return matches[0]

    def _messages(self, messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise RuntimeContractError("chat messages must be a sequence")
        if not messages or len(messages) > MAX_CLIENT_MESSAGES:
            raise RuntimeContractError("chat message count is outside its bounds")
        checked: list[dict[str, object]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise RuntimeContractError("chat message is not an object")
            raw = dict(message)
            if "attachments" in raw:
                raise RuntimeContractError(
                    "stored attachment records must be resolved by the private attachment store"
                )
            has_images = "images" in raw
            images = raw.pop("images", None)
            try:
                stored = validate_message(raw)
            except ConversationError as error:
                raise RuntimeContractError(str(error)) from error
            wire = {key: value for key, value in stored.items() if key != "thinking"}
            if has_images:
                if stored["role"] != "user":
                    raise RuntimeContractError("only a user chat message may contain images")
                wire["images"] = _validate_images(images)
            checked.append(wire)
        if len(_json_bytes(checked)) > MAX_CLIENT_MESSAGE_BYTES:
            raise RuntimeContractError("chat context exceeds its byte limit")
        return checked

    def _options(
        self,
        *,
        temperature: float | None,
        seed: int | None,
        context: int | None,
    ) -> dict[str, object]:
        options: dict[str, object] = {}
        if temperature is not None:
            if (
                not isinstance(temperature, (int, float))
                or isinstance(temperature, bool)
                or not 0 <= float(temperature) <= 2
            ):
                raise RuntimeContractError("temperature must be between 0 and 2")
            options["temperature"] = float(temperature)
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool) or not -(2**63) <= seed < 2**63:
                raise RuntimeContractError("seed is outside its signed 64-bit bounds")
            options["seed"] = seed
        if context is not None:
            if not isinstance(context, int) or isinstance(context, bool) or not 256 <= context <= 65_536:
                raise RuntimeContractError("context must be between 256 and 65536")
            options["num_ctx"] = context
        return options

    def _stream(
        self,
        *,
        route: str,
        model: str,
        payload: Mapping[str, object],
        on_delta: Callable[[StreamDelta], None] | None,
    ) -> StreamResult:
        self._authorize(route)
        body = _json_bytes(payload)
        connection, response = self._begin("POST", route, body)
        decoder = JSONLinesDecoder()
        accumulator = StreamAccumulator(route=route, request_model=model)
        try:
            content_type = _content_type(response)
            if content_type not in _STREAM_CONTENT_TYPES:
                raise UpstreamProtocolError("local runtime returned an unexpected stream content type")
            while True:
                block = response.read1(64 * 1024)
                if not block:
                    break
                for event in decoder.feed(block):
                    delta = accumulator.accept(event)
                    if on_delta is not None:
                        on_delta(delta)
            for event in decoder.finish():
                delta = accumulator.accept(event)
                if on_delta is not None:
                    on_delta(delta)
            return accumulator.finish()
        except (OSError, http.client.HTTPException) as error:
            raise RuntimeUnavailable("private Ollama stream ended unexpectedly") from error
        finally:
            connection.close()

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        context: int | None = None,
        on_delta: Callable[[StreamDelta], None] | None = None,
        check_installed: bool = True,
    ) -> StreamResult:
        checked_model = validate_model_name(model)
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeContractError("prompt must be non-empty text")
        try:
            prompt_bytes = len(prompt.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise RuntimeContractError("prompt contains invalid Unicode") from error
        if prompt_bytes > 1024 * 1024:
            raise RuntimeContractError("prompt exceeds its byte limit")
        if check_installed:
            self.require_installed(checked_model)
        payload: dict[str, object] = {
            "model": checked_model,
            "prompt": prompt,
            "stream": True,
        }
        options = self._options(temperature=temperature, seed=seed, context=context)
        if options:
            payload["options"] = options
        return self._stream(
            route="/api/generate",
            model=checked_model,
            payload=payload,
            on_delta=on_delta,
        )

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] = (),
        temperature: float | None = None,
        seed: int | None = None,
        context: int | None = None,
        on_delta: Callable[[StreamDelta], None] | None = None,
        check_installed: bool = True,
    ) -> StreamResult:
        checked_model = validate_model_name(model)
        if check_installed:
            self.require_installed(checked_model)
        checked_messages = self._messages(messages)
        payload: dict[str, object] = {
            "model": checked_model,
            "messages": checked_messages,
            "stream": True,
        }
        if tools:
            if len(tools) > 16:
                raise RuntimeContractError("tool declaration count exceeds its limit")
            checked_tools = [dict(tool) for tool in tools]
            if len(_json_bytes(checked_tools)) > 256 * 1024:
                raise RuntimeContractError("tool declarations exceed their byte limit")
            payload["tools"] = checked_tools
        options = self._options(temperature=temperature, seed=seed, context=context)
        if options:
            payload["options"] = options
        return self._stream(
            route="/api/chat",
            model=checked_model,
            payload=payload,
            on_delta=on_delta,
        )

    def record(self) -> dict[str, object]:
        return {
            "schema": "kilix.ollama.client/v0-unfrozen",
            "schema_status": "development-unfrozen",
            "transport": "same-uid-private-unix-mediation",
            "endpoint": str(self.paths.mediation_socket),
            "caller_supplied_url": False,
            "cloud_fallback": False,
            "implicit_pull": False,
            "allow_unqualified_candidate": self.allow_unqualified_candidate,
        }


__all__ = [
    "DEFAULT_TIMEOUT",
    "MAX_CLIENT_MESSAGES",
    "OllamaClient",
    "validate_model_name",
]
