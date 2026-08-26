"""Deny-by-default policy for the private Ollama mediation surface."""

from __future__ import annotations

from dataclasses import dataclass
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
        return MediationRequest(
            method=self.method,
            path=authorize_route(self.method, self.path),
            headers=forward_request_headers(self.headers),
            body=bounded_body(self.body),
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
        "cloud_fallback": False,
        "unknown_routes": "deny",
    }


__all__ = [
    "MAX_REQUEST_BODY",
    "MAX_HEADER_VALUE_BYTES",
    "MediationRequest",
    "ROUTE_METHODS",
    "authorize_route",
    "bounded_body",
    "forward_request_headers",
    "forward_response_headers",
    "policy_record",
    "verify_peer",
]
