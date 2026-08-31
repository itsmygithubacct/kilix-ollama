"""Artifact-free, exact-identity development catalog for F105.

The three records mirror the cleared engineering inputs, but deliberately do
not claim an accepted F100 carrier, receipt, install transaction, or release
admission.  Importing this module performs no filesystem or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Mapping

from .errors import RuntimeContractError


CATALOG_SCHEMA = "kilix.ai.models/v0-unfrozen"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASKS = frozenset(
    ("chat", "code", "completion", "embedding", "reasoning", "vision")
)
_CAPABILITIES = frozenset(
    ("completion", "embedding", "thinking", "tools", "vision")
)


@dataclass(frozen=True)
class ModelCatalogEntry:
    catalog_id: str
    display_name: str
    display_tag: str
    role: str
    manifest_digest: str
    payloads: tuple[tuple[str, str, int], ...]
    tasks: tuple[str, ...]
    capabilities: tuple[str, ...]
    publisher: str
    publisher_repository: str
    publisher_revision: str
    model_card_sha256: str
    license_id: str
    license_digest: str
    license_bytes: int
    use_restrictions: str

    def validate(self) -> None:
        ascii_fields = (
            (self.catalog_id, "catalog id", 128),
            (self.display_tag, "display tag", 256),
            (self.role, "role", 64),
            (self.publisher_revision, "publisher revision", 64),
            (self.license_id, "license id", 64),
        )
        for value, label, limit in ascii_fields:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > limit
                or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
            ):
                raise RuntimeContractError(f"model catalog {label} is invalid")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise RuntimeContractError("model catalog display name is invalid")
        if not _SHA256.fullmatch(self.manifest_digest):
            raise RuntimeContractError("model catalog manifest digest is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.model_card_sha256):
            raise RuntimeContractError("model catalog model-card digest is invalid")
        if not _SHA256.fullmatch(self.license_digest) or self.license_bytes <= 0:
            raise RuntimeContractError("model catalog licence identity is invalid")
        if not self.tasks or not set(self.tasks).issubset(_TASKS):
            raise RuntimeContractError("model catalog task set is invalid")
        if not self.capabilities or not set(self.capabilities).issubset(_CAPABILITIES):
            raise RuntimeContractError("model catalog capability set is invalid")
        roles: set[str] = set()
        for role, digest, byte_count in self.payloads:
            if role in roles or role not in ("model", "projector"):
                raise RuntimeContractError("model catalog payload role is invalid")
            roles.add(role)
            if not _SHA256.fullmatch(digest):
                raise RuntimeContractError("model catalog payload digest is invalid")
            if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
                raise RuntimeContractError("model catalog payload byte count is invalid")
        if "model" not in roles:
            raise RuntimeContractError("model catalog entry has no model payload")
        if self.license_id != "Apache-2.0":
            raise RuntimeContractError("development catalog contains an unselected licence")
        if self.use_restrictions != "none-recorded":
            raise RuntimeContractError("development catalog use restrictions are not exact")

    @property
    def payload_bytes(self) -> int:
        return sum(byte_count for _role, _digest, byte_count in self.payloads)

    def record(self) -> dict[str, object]:
        self.validate()
        return {
            "catalog_id": self.catalog_id,
            "display_name": self.display_name,
            "display_tag": self.display_tag,
            "role": self.role,
            "manifest_digest": self.manifest_digest,
            "payloads": [
                {"role": role, "digest": digest, "bytes": byte_count}
                for role, digest, byte_count in self.payloads
            ],
            "payload_bytes": self.payload_bytes,
            "tasks": list(self.tasks),
            "capabilities": list(self.capabilities),
            "publisher": self.publisher,
            "publisher_repository": self.publisher_repository,
            "publisher_revision": self.publisher_revision,
            "model_card_sha256": self.model_card_sha256,
            "license": {
                "id": self.license_id,
                "digest": self.license_digest,
                "bytes": self.license_bytes,
                "weight_determination": "engineering-cleared",
                "redistribution": "permitted-with-apache-section-4-conditions",
                "use_restrictions": self.use_restrictions,
            },
            "manifest_mutable_alias": True,
            "carrier_admitted": False,
            "receipt_authority_available": False,
            "install_enabled": False,
            "release_admitted": False,
            "weights_in_source_tree": False,
        }


CURATED_MODELS = (
    ModelCatalogEntry(
        catalog_id="ollama.granite4.1-3b.chat-tool",
        display_name="Granite 4.1 3B — chat and tools",
        display_tag="granite4.1:3b",
        role="chat-tool",
        manifest_digest="sha256:6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb",
        payloads=(
            (
                "model",
                "sha256:662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29",
                2_099_501_664,
            ),
        ),
        tasks=("chat", "code", "completion"),
        capabilities=("completion", "tools"),
        publisher="IBM",
        publisher_repository="ibm-granite/granite-4.1-3b",
        publisher_revision="c0650403e44e78ec0262dab1c90914c65b196c4e",
        model_card_sha256="7868b8ead19d157ad23ec827f252f6709a380d1b2cedc35c0e7a533b91550f1f",
        license_id="Apache-2.0",
        license_digest="sha256:58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
        license_bytes=11_357,
        use_restrictions="none-recorded",
    ),
    ModelCatalogEntry(
        catalog_id="ollama.nomic-embed-text-v1.5.embedding",
        display_name="Nomic Embed Text v1.5",
        display_tag="nomic-embed-text:v1.5",
        role="embedding",
        manifest_digest="sha256:0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
        payloads=(
            (
                "model",
                "sha256:970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6",
                274_290_656,
            ),
        ),
        tasks=("embedding",),
        capabilities=("embedding",),
        publisher="Nomic",
        publisher_repository="nomic-ai/nomic-embed-text-v1.5",
        publisher_revision="e9b6763023c676ca8431644204f50c2b100d9aab",
        model_card_sha256="4430178908fbd574fdb6457e302d23e643d733ed036b5777180b0806946767b2",
        license_id="Apache-2.0",
        license_digest="sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        license_bytes=11_357,
        use_restrictions="none-recorded",
    ),
    ModelCatalogEntry(
        catalog_id="ollama.granite3.2-vision-2b.vision-tool",
        display_name="Granite 3.2 Vision 2B",
        display_tag="granite3.2-vision:2b",
        role="vision-tool",
        manifest_digest="sha256:3be41a661804ad72cd08269816c5a145f1df6479ad07e2b3a7e29dba575d2669",
        payloads=(
            (
                "model",
                "sha256:1aefcd9a8a15091b670951963b5f8a7e6653bb1350345e9621e179685ac9bc5f",
                1_545_294_400,
            ),
            (
                "projector",
                "sha256:4d464be24899cf8dc1862945432e0cef4366c4181fa38b14754cc9279b727608",
                892_544_608,
            ),
        ),
        tasks=("chat", "vision"),
        capabilities=("completion", "tools", "vision"),
        publisher="IBM",
        publisher_repository="ibm-granite/granite-vision-3.2-2b",
        publisher_revision="ec894191f2e9e9be3d90eab9ac6949583c46f03b",
        model_card_sha256="ff91fe6064bd00d2c58c84b404ecc15102f956a79a8579fd355007d0fea9d72d",
        license_id="Apache-2.0",
        license_digest="sha256:2e68075caee4c571d43cb9d8d636415d8fe0fe59f695370cf56bc9f872f1ff3f",
        license_bytes=11_331,
        use_restrictions="none-recorded",
    ),
)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise RuntimeContractError("model catalog is not canonical JSON") from error


def catalog_record() -> dict[str, object]:
    entries = [entry.record() for entry in CURATED_MODELS]
    source_digest = hashlib.sha256(_canonical(entries)).hexdigest()
    return {
        "schema": CATALOG_SCHEMA,
        "schema_status": "development-unfrozen",
        "source": "f105-exact-engineering-inputs",
        "source_digest": f"sha256:{source_digest}",
        "models": entries,
        "model_count": len(entries),
        "payload_bytes": sum(entry.payload_bytes for entry in CURATED_MODELS),
        "carrier_admitted_count": 0,
        "install_enabled_count": 0,
        "automatic_install": False,
        "weights_in_source_tree": False,
        "release_admission": False,
    }


def _installed_digest(record: Mapping[str, object]) -> str | None:
    value = record.get("digest")
    if not isinstance(value, str):
        return None
    if value.startswith("sha256:"):
        return value if _SHA256.fullmatch(value) else None
    candidate = f"sha256:{value}"
    return candidate if _SHA256.fullmatch(candidate) else None


def project_library(
    installed: Iterable[Mapping[str, object]] = (),
    *,
    runtime_contacted: bool = False,
) -> dict[str, object]:
    if not isinstance(runtime_contacted, bool):
        raise RuntimeContractError("runtime-contacted provenance must be Boolean")
    installed_by_name: dict[str, Mapping[str, object]] = {}
    for index, record in enumerate(installed):
        if not isinstance(record, Mapping):
            raise RuntimeContractError(f"installed model record {index} is invalid")
        name = record.get("name", record.get("model"))
        if not isinstance(name, str) or not name or len(name) > 1024:
            raise RuntimeContractError(f"installed model record {index} has no safe name")
        if name in installed_by_name:
            raise RuntimeContractError(f"installed model name is duplicated: {name}")
        installed_by_name[name] = record

    curated: list[dict[str, object]] = []
    matched: set[str] = set()
    exact_count = 0
    for entry in CURATED_MODELS:
        record = entry.record()
        local = installed_by_name.get(entry.display_tag)
        state = "not-installed"
        local_digest = None
        if local is not None:
            matched.add(entry.display_tag)
            local_digest = _installed_digest(local)
            if local_digest == entry.manifest_digest:
                state = "installed-exact"
                exact_count += 1
            elif local_digest is None:
                state = "installed-unverified"
            else:
                state = "installed-digest-mismatch"
        record["local_state"] = state
        record["local_manifest_digest"] = local_digest
        curated.append(record)

    uncurated = [
        {
            "name": name,
            "digest": _installed_digest(record),
            "local_state": "installed-uncurated",
            "trusted_catalog_metadata": False,
        }
        for name, record in sorted(installed_by_name.items())
        if name not in matched
    ]
    return {
        "schema": "kilix.ollama.library/v0-unfrozen",
        "schema_status": "development-unfrozen",
        "curated": curated,
        "uncurated_installed": uncurated,
        "curated_count": len(curated),
        "installed_count": len(installed_by_name),
        "installed_exact_count": exact_count,
        "runtime_contacted": runtime_contacted,
        "automatic_install": False,
        "release_admission": False,
    }


def find_entry(identity: str) -> ModelCatalogEntry:
    matches = [
        entry
        for entry in CURATED_MODELS
        if identity in (entry.catalog_id, entry.display_tag)
    ]
    if len(matches) != 1:
        raise RuntimeContractError(f"unknown curated model identity: {identity}")
    return matches[0]


__all__ = [
    "CATALOG_SCHEMA",
    "CURATED_MODELS",
    "ModelCatalogEntry",
    "catalog_record",
    "find_entry",
    "project_library",
]
