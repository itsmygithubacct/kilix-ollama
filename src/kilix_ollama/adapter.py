"""Bounded parsers for the provisional Ollama compatibility boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from .errors import UpstreamProtocolError


CANDIDATE_VERSION = "0.32.14+plebian.security-license-r3"
MAX_METADATA_RESPONSE = 4 * 1024 * 1024
MAX_MODELS = 4096
DISCOVERY_ONLY_ROUTES = ("/api/ps", "/api/tags", "/api/version")


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpstreamProtocolError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise UpstreamProtocolError(f"non-finite JSON number: {value}")


def decode_json_object(payload: bytes, *, limit: int = MAX_METADATA_RESPONSE) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise UpstreamProtocolError("upstream payload must be bytes")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise UpstreamProtocolError("upstream payload limit is invalid")
    if len(payload) > limit:
        raise UpstreamProtocolError(f"upstream payload exceeds {limit} bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise UpstreamProtocolError(f"invalid upstream JSON: {error}") from error
    if not isinstance(value, dict):
        raise UpstreamProtocolError("upstream JSON is not an object")
    return value


def parse_version(payload: bytes) -> str:
    version = decode_json_object(payload).get("version")
    if not isinstance(version, str) or not version or len(version) > 128:
        raise UpstreamProtocolError("upstream version is missing or invalid")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in version):
        raise UpstreamProtocolError("upstream version is not printable ASCII")
    return version


def parse_model_list(payload: bytes, *, route: str) -> tuple[dict[str, Any], ...]:
    if route not in ("/api/tags", "/api/ps"):
        raise UpstreamProtocolError(f"unsupported model-list route: {route}")
    models = decode_json_object(payload).get("models")
    if not isinstance(models, list):
        raise UpstreamProtocolError(f"{route} response has no models list")
    if len(models) > MAX_MODELS:
        raise UpstreamProtocolError(f"{route} response exceeds the model-count limit")
    checked: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise UpstreamProtocolError(f"{route} model {index} is not an object")
        name = model.get("name")
        if name is not None and (not isinstance(name, str) or len(name) > 1024):
            raise UpstreamProtocolError(f"{route} model {index} has an invalid name")
        checked.append(model)
    return tuple(checked)


@dataclass(frozen=True)
class Compatibility:
    version: str
    mode: str
    allowed_routes: tuple[str, ...]
    qualified: bool
    reason: str

    def record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mode": self.mode,
            "allowed_routes": list(self.allowed_routes),
            "qualified": self.qualified,
            "reason": self.reason,
        }


def compatibility_for_version(version: str) -> Compatibility:
    if version == CANDIDATE_VERSION:
        return Compatibility(
            version,
            "candidate-unqualified",
            DISCOVERY_ONLY_ROUTES,
            False,
            "exact R3 candidate; release compatibility evidence is incomplete",
        )
    return Compatibility(
        version,
        "read-only-diagnostics",
        DISCOVERY_ONLY_ROUTES,
        False,
        "unknown version; management, inference, and tools remain disabled",
    )


def health_record(version_payload: bytes, tags_payload: bytes, ps_payload: bytes) -> dict[str, object]:
    version = parse_version(version_payload)
    tags = parse_model_list(tags_payload, route="/api/tags")
    running = parse_model_list(ps_payload, route="/api/ps")
    return {
        "schema": "kilix.ollama.health/v0-unfrozen",
        "schema_status": "development-unfrozen",
        "compatibility": compatibility_for_version(version).record(),
        "installed_model_count": len(tags),
        "running_model_count": len(running),
        "inference_performed": False,
    }


__all__ = [
    "CANDIDATE_VERSION",
    "Compatibility",
    "decode_json_object",
    "compatibility_for_version",
    "health_record",
    "parse_model_list",
    "parse_version",
]
