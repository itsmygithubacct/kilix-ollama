"""Deny-by-default policy for the private Ollama mediation surface."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping
from urllib.parse import urlsplit

from .errors import PolicyDenied


# This is a development allowlist derived from the accepted F105 packaging
# plan. It is intentionally not named as the future frozen adapter schema.
ROUTE_METHODS: dict[str, frozenset[str]] = {
    "/api/version": frozenset(("GET",)),
    "/api/tags": frozenset(("GET",)),
    "/api/show": frozenset(("POST",)),
    "/api/ps": frozenset(("GET",)),
    "/api/chat": frozenset(("POST",)),
    "/api/generate": frozenset(("POST",)),
    "/api/embed": frozenset(("POST",)),
    "/api/pull": frozenset(("POST",)),
    "/api/delete": frozenset(("DELETE",)),
    "/api/copy": frozenset(("POST",)),
    "/api/status": frozenset(("GET",)),
}

MAX_TARGET_BYTES = 2048
MAX_REQUEST_BODY = 16 * 1024 * 1024
MAX_HEADER_VALUE_BYTES = 8192
FORWARDED_REQUEST_HEADERS = frozenset(("accept", "content-type"))
FORWARDED_RESPONSE_HEADERS = frozenset(("content-type",))
MODEL_BODY_FIELDS: dict[str, tuple[str, ...]] = {
    "/api/chat": ("model",),
    "/api/generate": ("model",),
    "/api/embed": ("model",),
    "/api/show": ("model", "name"),
    "/api/pull": ("model", "name"),
    "/api/delete": ("model", "name"),
    "/api/copy": ("source", "destination"),
}
MAX_MODEL_NAME_BYTES = 1024


def _header_value(value: str, *, source: str) -> str:
    if len(value) > MAX_HEADER_VALUE_BYTES:
        raise PolicyDenied(f"{source} header exceeds the mediation limit")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise PolicyDenied(f"{source} header contains a non-printable character")
    return value.strip()


def authorize_route(method: str, target: str) -> str:
    """Return the exact allowed path or raise without normalizing ambiguity."""
    if not isinstance(method, str):
        raise PolicyDenied("HTTP method must be a string")
    verb = method.upper()
    if not verb.isascii() or not verb.isalpha() or verb != method:
        raise PolicyDenied("HTTP method must be canonical uppercase ASCII")
    if not isinstance(target, str) or not target:
        raise PolicyDenied("request target is empty")
    try:
        target_size = len(target.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise PolicyDenied("request target is not valid UTF-8") from error
    if target_size > MAX_TARGET_BYTES:
        raise PolicyDenied("request target exceeds the mediation limit")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in target):
        raise PolicyDenied("request target contains a control character")
    if "\\" in target or "%" in target:
        raise PolicyDenied("encoded or backslash request targets are not accepted")
    try:
        parsed = urlsplit(target)
    except ValueError as error:
        raise PolicyDenied("request target cannot be parsed canonically") from error
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise PolicyDenied("only an exact local API path is accepted")
    if parsed.path != target or "//" in target or "/./" in target or "/../" in target:
        raise PolicyDenied("request target is not canonical")
    methods = ROUTE_METHODS.get(target)
    if methods is None:
        raise PolicyDenied(f"route is not allowed: {target}")
    if verb not in methods:
        raise PolicyDenied(f"method {verb} is not allowed for {target}")
    return target


def bounded_body(body: bytes, *, limit: int = MAX_REQUEST_BODY) -> bytes:
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise PolicyDenied("request body limit is invalid")
    if not isinstance(body, bytes):
        raise PolicyDenied("request body must be bytes")
    if len(body) > limit:
        raise PolicyDenied(f"request body exceeds {limit} bytes")
    return body


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyDenied(f"request JSON repeats member {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PolicyDenied(f"request JSON contains non-finite number {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PolicyDenied("request JSON contains an out-of-range number")
    return parsed


def _json_body(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except PolicyDenied:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise PolicyDenied(f"request body is not a strict JSON object: {error}") from error
    if not isinstance(value, dict):
        raise PolicyDenied("request body must be a JSON object")
    return value


def _model_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyDenied(f"request field {field!r} must be a non-empty model name")
    try:
        size = len(value.encode("ascii", errors="strict"))
    except UnicodeEncodeError as error:
        raise PolicyDenied(f"request field {field!r} must be printable ASCII") from error
    if size > MAX_MODEL_NAME_BYTES:
        raise PolicyDenied(f"request field {field!r} exceeds the model-name limit")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise PolicyDenied(f"request field {field!r} must be printable ASCII without spaces")
    return value


def has_explicit_cloud_source(model: str) -> bool:
    """Match the candidate Ollama source suffix grammar, including legacy tags."""
    if not isinstance(model, str):
        return False
    cursor = model.strip()
    while True:
        index = cursor.rfind(":")
        if index < 0:
            return False
        raw_suffix = cursor[index + 1 :].strip()
        suffix = raw_suffix.casefold()
        if suffix == "cloud":
            return True
        if "/" not in raw_suffix and suffix.endswith("-cloud"):
            return True
        if suffix == "local":
            cursor = cursor[:index]
            continue
        return False


def authorize_request_body(method: str, path: str, body: bytes) -> bytes:
    """Deny explicit cloud selectors and ambiguous model-management bodies."""
    bounded = bounded_body(body)
    fields = MODEL_BODY_FIELDS.get(path)
    if fields is None:
        if bounded:
            raise PolicyDenied(f"{method} {path} does not accept a request body")
        return bounded
    value = _json_body(bounded)
    supplied: dict[str, str] = {}
    for field in fields:
        if field in value and value[field] not in (None, ""):
            supplied[field] = _model_name(value[field], field)
    if path == "/api/copy":
        if set(supplied) != {"source", "destination"}:
            raise PolicyDenied("copy requires exact non-empty source and destination model names")
    elif not supplied:
        raise PolicyDenied(f"{path} requires a non-empty model name")
    elif "model" in supplied and "name" in supplied and supplied["model"] != supplied["name"]:
        raise PolicyDenied("model and deprecated name fields disagree")
    for field, model in supplied.items():
        if has_explicit_cloud_source(model):
            raise PolicyDenied(f"cloud-tagged model in field {field!r} is denied")
    if path == "/api/pull":
        if value.get("insecure") is True:
            raise PolicyDenied("insecure model pulls are denied")
        if value.get("username") not in (None, "") or value.get("password") not in (
            None,
            "",
        ):
            raise PolicyDenied("model pull credentials are not accepted")
    return bounded


def forward_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy only content negotiation headers; never credentials or origins."""
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise PolicyDenied("request header names and values must be strings")
        lowered = name.casefold()
        if lowered not in FORWARDED_REQUEST_HEADERS:
            continue
        canonical = lowered.title()
        if canonical in forwarded:
            raise PolicyDenied(f"duplicate forwarded request header: {lowered}")
        forwarded[canonical] = _header_value(value, source="request")
    forwarded["User-Agent"] = "kilix-ollama/0.2.1.dev0"
    return forwarded


def forward_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise PolicyDenied("response header names and values must be strings")
        lowered = name.casefold()
        if lowered not in FORWARDED_RESPONSE_HEADERS:
            continue
        canonical = lowered.title()
        if canonical in forwarded:
            raise PolicyDenied(f"duplicate forwarded response header: {lowered}")
        forwarded[canonical] = _header_value(value, source="upstream response")
    return forwarded


def verify_peer(peer_uid: int, *, owner_uid: int) -> None:
    if (
        not isinstance(peer_uid, int)
        or isinstance(peer_uid, bool)
        or not isinstance(owner_uid, int)
        or isinstance(owner_uid, bool)
        or peer_uid != owner_uid
    ):
        raise PolicyDenied("mediation peer is not the owning uid")


@dataclass(frozen=True)
class MediationRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes = b""

    def prepared(self) -> "MediationRequest":
        path = authorize_route(self.method, self.path)
        return MediationRequest(
            method=self.method,
            path=path,
            headers=forward_request_headers(self.headers),
            body=authorize_request_body(self.method, path, self.body),
        )


def policy_record() -> dict[str, object]:
    return {
        "schema": "kilix.ollama.mediation-policy/v0-unfrozen",
        "schema_status": "development-unfrozen",
        "routes": {
            route: sorted(methods) for route, methods in sorted(ROUTE_METHODS.items())
        },
        "request_body_limit": MAX_REQUEST_BODY,
        "header_value_limit": MAX_HEADER_VALUE_BYTES,
        "forwarded_request_headers": sorted(FORWARDED_REQUEST_HEADERS),
        "forwarded_response_headers": sorted(FORWARDED_RESPONSE_HEADERS),
        "model_body_fields": {
            route: list(fields) for route, fields in sorted(MODEL_BODY_FIELDS.items())
        },
        "explicit_cloud_model_source": "deny",
        "pull_credentials": "deny",
        "insecure_pull": "deny",
        "cloud_fallback": False,
        "unknown_routes": "deny",
    }


__all__ = [
    "MAX_REQUEST_BODY",
    "MAX_HEADER_VALUE_BYTES",
    "MAX_MODEL_NAME_BYTES",
    "MODEL_BODY_FIELDS",
    "MediationRequest",
    "ROUTE_METHODS",
    "authorize_request_body",
    "authorize_route",
    "bounded_body",
    "forward_request_headers",
    "forward_response_headers",
    "has_explicit_cloud_source",
    "policy_record",
    "verify_peer",
]
