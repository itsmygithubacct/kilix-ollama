"""Opt-in, hash-pinned Pocket TTS source download and local conversion.

Importing this module performs no filesystem or network I/O.  The catalog
contains metadata only; model bytes always come directly from the immutable
upstream revision after explicit terms acceptance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

from .errors import InstallError
from .paths import RuntimePaths


CATALOG_SCHEMA = "kilix.ollama.optional-tts/v0-unfrozen"
INSTALL_SCHEMA = "kilix.ollama.optional-tts-install/v0-unfrozen"
MODEL_ID = "pocket-tts-english-q8_0"
SOURCE_REPOSITORY = "kyutai/pocket-tts-without-voice-cloning"
SOURCE_REVISION = "d29db7978e464fb90cb3359ee0c69a273b9142cc"
UPSTREAM_TERMS_URL = (
    "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning"
)
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CONVERTER_PATH = Path(
    "/usr/lib/plebian-os/ollama/pocket-tts-convert-hf-to-gguf"
)
CONVERTER_CLOSURE = (
    "candidate:plebian-os-llama.cpp-b10434-pocket-q8_0-r1"
)
MINIMUM_FREE_BYTES = 512 * 1024 * 1024
MANIFEST_NAME = "kilix-tts-model.json"
MODEL_CARD_NAME = "MODEL_CARD.md"

TERMS_SUMMARY = (
    "CC BY 4.0; credit Kyutai, preserve supplied notices, identify changes, "
    "and do not imply endorsement. Model use must be lawful and must not "
    "facilitate unauthorized voice impersonation, deception, or harmful or "
    "privacy-invasive content. Review the pinned upstream model card."
)


@dataclass(frozen=True)
class ArtifactSpec:
    """One immutable upstream or locally derived artifact."""

    role: str
    filename: str
    size: int
    sha256: str
    url: str | None = None

    def record(self, *, include_url: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "role": self.role,
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
        }
        if include_url and self.url is not None:
            record["url"] = self.url
        return record


@dataclass(frozen=True)
class TtsModelSpec:
    """A source-to-runtime conversion candidate with exact byte identities."""

    catalog_id: str
    source_revision: str
    source_artifacts: tuple[ArtifactSpec, ...]
    notice_artifact: ArtifactSpec
    derived_artifacts: tuple[ArtifactSpec, ...]
    converter_closure: str

    @property
    def source_bytes(self) -> int:
        return sum(artifact.size for artifact in self.source_artifacts)

    @property
    def download_bytes(self) -> int:
        return self.source_bytes + self.notice_artifact.size

    @property
    def runtime_bytes(self) -> int:
        return sum(artifact.size for artifact in self.derived_artifacts)


def _source_url(relative_path: str) -> str:
    return (
        f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/"
        f"{SOURCE_REVISION}/{relative_path}"
    )


POCKET_Q8 = TtsModelSpec(
    catalog_id=MODEL_ID,
    source_revision=SOURCE_REVISION,
    source_artifacts=(
        ArtifactSpec(
            role="model-source",
            filename="model.safetensors",
            url=_source_url("languages/english/model.safetensors"),
            size=219_029_196,
            sha256=(
                "be9c6b4876d3f30740a8225dfcaa2e43"
                "dc4aeb753c15272735bee16bbb4abb0a"
            ),
        ),
        ArtifactSpec(
            role="tokenizer-source",
            filename="tokenizer.model",
            url=_source_url("languages/english/tokenizer.model"),
            size=59_339,
            sha256=(
                "d461765ae179566678c93091c5fa6f298"
                "4c31bbe990bf1aa62d92c64d91bc3f6"
            ),
        ),
    ),
    notice_artifact=ArtifactSpec(
        role="model-card",
        filename=MODEL_CARD_NAME,
        url=_source_url("README.md"),
        size=11_187,
        sha256=(
            "ae2ebac6f8039d761ca90e2b742136dc"
            "e9d7872ec8dd2105e3b2de1e3021e3aa"
        ),
    ),
    derived_artifacts=(
        ArtifactSpec(
            role="runtime-model-q8_0",
            filename="pocket-tts-english-q8_0.gguf",
            size=84_769_056,
            sha256=(
                "87c310bfeae4ba8a9e4016c0111401382"
                "3d08451d10292c2bd64a8e7b9b529c6"
            ),
        ),
        ArtifactSpec(
            role="runtime-projector-q8_0",
            filename="mmproj-pocket-tts-english-q8_0.gguf",
            size=38_860_000,
            sha256=(
                "174653c6675289fc1a12f6fbd60a63527"
                "41cf95ce71c94f3dc1d42dd90a62b00"
            ),
        ),
    ),
    converter_closure=CONVERTER_CLOSURE,
)

MODELS = (POCKET_Q8,)
MODEL_BY_ID = {model.catalog_id: model for model in MODELS}
MODEL_IDS = tuple(MODEL_BY_ID)


def get_model(catalog_id: str = MODEL_ID) -> TtsModelSpec:
    try:
        return MODEL_BY_ID[catalog_id]
    except KeyError as error:
        raise InstallError(
            f"unknown optional TTS model {catalog_id!r}; choose one of: "
            f"{', '.join(MODEL_IDS)}"
        ) from error


def catalog_record() -> dict[str, object]:
    """Return the metadata-only optional TTS catalog without touching disk."""
    return {
        "schema": CATALOG_SCHEMA,
        "schema_status": "development-unfrozen",
        "automatic_install": False,
        "explicit_terms_acceptance_required": True,
        "download_origin": "immutable-upstream-only",
        "kilix_mirror": False,
        "credentials_stored": False,
        "product_voice_included": False,
        "publication_authorized": False,
        "models": [
            {
                "catalog_id": model.catalog_id,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": model.source_revision,
                "source_bytes": model.source_bytes,
                "download_bytes": model.download_bytes,
                "runtime_bytes": model.runtime_bytes,
                "peak_free_space_required": MINIMUM_FREE_BYTES,
                "source_artifacts": [
                    artifact.record() for artifact in model.source_artifacts
                ],
                "notice_artifact": model.notice_artifact.record(),
                "derived_artifacts": [
                    artifact.record() for artifact in model.derived_artifacts
                ],
                "converter_closure": model.converter_closure,
                "converter_status": "candidate-not-release-admitted",
                "license": "CC-BY-4.0",
                "license_url": LICENSE_URL,
                "attribution": "Kyutai",
                "terms_url": UPSTREAM_TERMS_URL,
                "terms_summary": TERMS_SUMMARY,
            }
            for model in MODELS
        ],
    }


def _models_root(paths: RuntimePaths) -> Path:
    return paths.data_dir / "tts-models"


def _model_dir(paths: RuntimePaths, model: TtsModelSpec) -> Path:
    return _models_root(paths) / model.catalog_id


def _cache_root(paths: RuntimePaths) -> Path:
    return paths.cache_dir / "artifacts" / "sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_private_directory(path: Path) -> Path:
    """Create one owner-only directory and reject pre-existing substitutions."""
    existed = os.path.lexists(path)
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as error:
        raise InstallError(f"cannot create private directory {path}: {error}") from error
    try:
        info = path.lstat()
    except OSError as error:
        raise InstallError(f"cannot inspect private directory {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise InstallError(f"private path {path} is not a real directory")
    if info.st_uid != os.getuid():
        raise InstallError(
            f"private directory {path} is owned by uid {info.st_uid}, "
            f"expected {os.getuid()}"
        )
    if not existed:
        try:
            path.chmod(0o700)
            info = path.lstat()
        except OSError as error:
            raise InstallError(f"cannot secure private directory {path}: {error}") from error
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise InstallError(f"private directory {path} must have mode 0700")
    return path


def _regular_file_status(
    path: Path,
    artifact: ArtifactSpec,
    *,
    verify_hash: bool,
) -> tuple[bool, str]:
    try:
        info = path.lstat()
    except OSError as error:
        return False, f"required artifact {artifact.filename} is unavailable: {error}"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return False, f"required artifact {artifact.filename} is not a regular file"
    if info.st_uid != os.getuid():
        return False, f"required artifact {artifact.filename} has an unexpected owner"
    if stat.S_IMODE(info.st_mode) != 0o600:
        return False, f"required artifact {artifact.filename} must have mode 0600"
    if info.st_size != artifact.size:
        return False, (
            f"artifact {artifact.filename} has {info.st_size} bytes; "
            f"expected {artifact.size}"
        )
    if verify_hash:
        try:
            actual_sha256 = _sha256(path)
        except OSError as error:
            return False, f"artifact {artifact.filename} cannot be hashed: {error}"
        if actual_sha256 != artifact.sha256:
            return False, f"artifact {artifact.filename} failed SHA-256 verification"
    return True, "verified"


def _manifest(model: TtsModelSpec, *, sources_retained: bool) -> dict[str, object]:
    return {
        "schema": INSTALL_SCHEMA,
        "catalog_schema": CATALOG_SCHEMA,
        "model": model.catalog_id,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": model.source_revision,
        "sources_retained": sources_retained,
        "converter_closure": model.converter_closure,
        "source_artifacts": [
            artifact.record(include_url=False) for artifact in model.source_artifacts
        ],
        "notice_artifact": model.notice_artifact.record(include_url=False),
        "derived_artifacts": [
            artifact.record(include_url=False) for artifact in model.derived_artifacts
        ],
        "license": "CC-BY-4.0",
        "license_url": LICENSE_URL,
        "attribution": "Kyutai",
        "terms_url": UPSTREAM_TERMS_URL,
        "terms_summary": TERMS_SUMMARY,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("manifest is not a regular file")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("manifest owner or mode is unsafe")
        if info.st_size > 64 * 1024:
            raise ValueError("manifest is unexpectedly large")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise InstallError(f"model manifest is missing or invalid: {error}") from error
    if not isinstance(value, dict):
        raise InstallError("model manifest must be a JSON object")
    return value


def _directory_status(path: Path, label: str) -> tuple[bool, str]:
    try:
        info = path.lstat()
    except OSError as error:
        return False, f"{label} is unavailable: {error}"
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False, f"{label} is not a real directory"
    if info.st_uid != os.getuid():
        return False, f"{label} has an unexpected owner"
    if stat.S_IMODE(info.st_mode) != 0o700:
        return False, f"{label} must have mode 0700"
    return True, "owner-only directory"


def _exact_entries(directory: Path, expected: set[str]) -> tuple[bool, str]:
    try:
        actual = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        return False, f"cannot enumerate {directory}: {error}"
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        return False, f"unexpected entries={unexpected}; missing entries={missing}"
    return True, "exact entries"


def inspect_install(
    catalog_id: str = MODEL_ID,
    *,
    paths: RuntimePaths | None = None,
    verify_hashes: bool = False,
) -> tuple[bool, str]:
    """Inspect one local conversion without network access or mutation."""
    model = get_model(catalog_id)
    runtime_paths = RuntimePaths.from_environment() if paths is None else paths
    target = _model_dir(runtime_paths, model)
    return _inspect_model_directory(target, model, verify_hashes=verify_hashes)


def verified_runtime_artifacts(
    catalog_id: str = MODEL_ID,
    *,
    paths: RuntimePaths | None = None,
) -> tuple[Path, Path]:
    """Return the exact local model/projector pair after full hash verification."""
    model = get_model(catalog_id)
    runtime_paths = RuntimePaths.from_environment() if paths is None else paths
    installed, detail = inspect_install(
        model.catalog_id,
        paths=runtime_paths,
        verify_hashes=True,
    )
    if not installed:
        raise InstallError(f"optional TTS runtime is not verified: {detail}")
    by_role = {
        artifact.role: artifact
        for artifact in model.derived_artifacts
    }
    try:
        model_artifact = by_role["runtime-model-q8_0"]
        projector_artifact = by_role["runtime-projector-q8_0"]
    except KeyError as error:
        raise InstallError("optional TTS catalog has no exact runtime pair") from error
    root = _model_dir(runtime_paths, model) / "runtime"
    return root / model_artifact.filename, root / projector_artifact.filename


def _inspect_model_directory(
    target: Path,
    model: TtsModelSpec,
    *,
    verify_hashes: bool,
) -> tuple[bool, str]:
    """Inspect one exact path, including a detached removal quarantine."""
    try:
        target.lstat()
    except FileNotFoundError:
        return False, f"not installed; run: kilix-ollama tts-install {model.catalog_id}"
    except OSError as error:
        return False, f"cannot inspect {target}: {error}"

    ok, detail = _directory_status(target, "model directory")
    if not ok:
        return False, detail
    try:
        installed_manifest = _read_manifest(target / MANIFEST_NAME)
    except InstallError as error:
        return False, str(error)
    sources_retained = installed_manifest.get("sources_retained")
    if not isinstance(sources_retained, bool):
        return False, "model manifest has an invalid sources_retained value"
    if installed_manifest != _manifest(model, sources_retained=sources_retained):
        return False, "model manifest does not match this catalog"

    expected_top = {MANIFEST_NAME, MODEL_CARD_NAME, "runtime"}
    if sources_retained:
        expected_top.add("source")
    ok, detail = _exact_entries(target, expected_top)
    if not ok:
        return False, f"model directory layout mismatch: {detail}"

    runtime_dir = target / "runtime"
    ok, detail = _directory_status(runtime_dir, "runtime artifact directory")
    if not ok:
        return False, detail
    ok, detail = _exact_entries(
        runtime_dir, {artifact.filename for artifact in model.derived_artifacts}
    )
    if not ok:
        return False, f"runtime artifact layout mismatch: {detail}"
    for artifact in model.derived_artifacts:
        ok, detail = _regular_file_status(
            runtime_dir / artifact.filename,
            artifact,
            verify_hash=verify_hashes,
        )
        if not ok:
            return False, detail

    ok, detail = _regular_file_status(
        target / MODEL_CARD_NAME,
        model.notice_artifact,
        verify_hash=verify_hashes,
    )
    if not ok:
        return False, detail

    if sources_retained:
        source_dir = target / "source"
        ok, detail = _directory_status(source_dir, "source artifact directory")
        if not ok:
            return False, detail
        ok, detail = _exact_entries(
            source_dir, {artifact.filename for artifact in model.source_artifacts}
        )
        if not ok:
            return False, f"source artifact layout mismatch: {detail}"
        for artifact in model.source_artifacts:
            ok, detail = _regular_file_status(
                source_dir / artifact.filename,
                artifact,
                verify_hash=verify_hashes,
            )
            if not ok:
                return False, detail
    verification = "fully verified" if verify_hashes else "structurally verified"
    retention = "source retained" if sources_retained else "source discarded"
    return True, f"installed and {verification}; {retention}"


@contextlib.contextmanager
def _install_lock(paths: RuntimePaths, model: TtsModelSpec) -> Iterator[None]:
    root = _ensure_private_directory(_models_root(paths))
    lock_path = root / f".{model.catalog_id}.install.lock"
    descriptor = -1
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise InstallError("installation lock is not an owner-controlled file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except OSError as error:
        raise InstallError(f"cannot lock model installation: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _kilix_download():
    """Load Kilix Content only for an explicitly accepted install."""
    try:
        return importlib.import_module("kilix_content.install").download
    except ImportError:
        sibling = (
            Path(__file__).resolve().parents[4]
            / "kilix-modules"
            / "kilix-content"
            / "src"
        )
        if sibling.is_dir() and str(sibling) not in sys.path:
            sys.path.insert(0, str(sibling))
        try:
            return importlib.import_module("kilix_content.install").download
        except ImportError as error:
            raise InstallError(
                "kilix-content is required for optional model downloads"
            ) from error


def _looks_like_access_denial(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        if (
            code in (401, 403)
            or "HTTP Error 401" in str(current)
            or "HTTP Error 403" in str(current)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _verify_cached(path: Path, artifact: ArtifactSpec) -> tuple[bool, str]:
    return _regular_file_status(path, artifact, verify_hash=True)


def _cache_artifact(
    artifact: ArtifactSpec,
    cache_root: Path,
    *,
    download: Callable[..., str],
    report: Callable[[str], None],
) -> Path:
    if artifact.url is None:
        raise InstallError(f"source artifact {artifact.filename} has no upstream URL")
    cached = cache_root / artifact.sha256
    if os.path.lexists(cached):
        ok, detail = _verify_cached(cached, artifact)
        if not ok:
            raise InstallError(
                f"refusing invalid cached artifact {cached}: {detail}. "
                "Move it aside and retry."
            )
        report(f"using verified cache entry for {artifact.filename}")
        return cached
    try:
        download(
            artifact.url,
            str(cached),
            report=report,
            expected_sha256=artifact.sha256,
        )
    except Exception as error:
        try:
            cached.unlink(missing_ok=True)
        except OSError:
            pass
        if _looks_like_access_denial(error):
            raise InstallError(
                f"upstream denied {artifact.filename}. Review or accept the model "
                f"terms at {UPSTREAM_TERMS_URL}, then retry. Kilix neither stores "
                "account credentials nor switches to an unapproved mirror."
            ) from error
        raise InstallError(
            f"could not download {artifact.filename} from the pinned upstream: {error}"
        ) from error
    try:
        os.chmod(cached, 0o600, follow_symlinks=False)
    except OSError as error:
        try:
            cached.unlink(missing_ok=True)
        except OSError:
            pass
        raise InstallError(f"cannot secure cached artifact {cached}: {error}") from error
    ok, detail = _verify_cached(cached, artifact)
    if not ok:
        try:
            cached.unlink(missing_ok=True)
        except OSError:
            pass
        raise InstallError(f"downloaded artifact failed verification: {detail}")
    with cached.open("rb") as stream:
        os.fsync(stream.fileno())
    return cached


def _copy_verified(source: Path, destination: Path, artifact: ArtifactSpec) -> None:
    ok, detail = _verify_cached(source, artifact)
    if not ok:
        raise InstallError(f"cached artifact changed before staging: {detail}")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    source_descriptor = -1
    try:
        source_descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(source_descriptor, "rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output:
            source_descriptor = -1
            descriptor = -1
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
    ok, detail = _regular_file_status(destination, artifact, verify_hash=True)
    if not ok:
        raise InstallError(f"staged artifact failed verification: {detail}")


def _stage_verified(source: Path, destination: Path, artifact: ArtifactSpec) -> None:
    """Prefer a same-filesystem hard link, with an exact-copy fallback."""
    ok, detail = _verify_cached(source, artifact)
    if not ok:
        raise InstallError(f"cached artifact changed before staging: {detail}")
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EOPNOTSUPP,
            errno.ENOTSUP,
        ):
            raise InstallError(
                f"cannot stage cached artifact {artifact.filename}: {error}"
            ) from error
        _copy_verified(source, destination, artifact)
        return
    ok, detail = _regular_file_status(destination, artifact, verify_hash=True)
    if not ok:
        raise InstallError(f"staged artifact failed verification: {detail}")


def _validate_converter(path: Path = CONVERTER_PATH) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise InstallError("pinned converter path must be absolute and traversal-free")
    for ancestor in reversed(path.parents):
        try:
            ancestor_info = ancestor.lstat()
        except OSError as error:
            raise InstallError(
                f"cannot inspect pinned converter ancestor {ancestor}: {error}"
            ) from error
        if stat.S_ISLNK(ancestor_info.st_mode) or not stat.S_ISDIR(
            ancestor_info.st_mode
        ):
            raise InstallError(
                f"pinned converter ancestor {ancestor} is not a real directory"
            )
        if ancestor_info.st_uid != 0 or stat.S_IMODE(ancestor_info.st_mode) & 0o022:
            raise InstallError(
                f"pinned converter ancestor {ancestor} is not root-controlled"
            )
    try:
        info = path.lstat()
    except OSError as error:
        raise InstallError(
            f"the pinned local converter closure is unavailable at {path}; "
            "run 'sudo plebian-os-install-deps --vulkan-tts' and then "
            "'sudo plebian-os-install-ollama-converter' first"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise InstallError(f"pinned converter {path} is not a regular file")
    if info.st_uid != 0:
        raise InstallError(f"pinned converter {path} must be owned by root")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise InstallError(f"pinned converter {path} is group/world writable")
    if not os.access(path, os.X_OK):
        raise InstallError(f"pinned converter {path} is not executable")


def _default_convert(
    source_dir: Path,
    runtime_dir: Path,
    model: TtsModelSpec,
    report: Callable[[str], None],
) -> None:
    _validate_converter()
    private_home = _ensure_private_directory(runtime_dir.parent / ".converter-home")
    temporary = _ensure_private_directory(runtime_dir.parent / ".converter-tmp")
    environment = {
        "HOME": str(private_home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "TMPDIR": str(temporary),
    }
    for artifact in model.derived_artifacts:
        arguments = [
            str(CONVERTER_PATH),
            str(source_dir),
            "--outtype",
            "q8_0",
            "--outfile",
            str(runtime_dir / artifact.filename),
        ]
        if artifact.role == "runtime-projector-q8_0":
            arguments.append("--mmproj")
        report(f"converting {artifact.filename} with the pinned offline closure")
        try:
            result = subprocess.run(
                arguments,
                cwd=runtime_dir.parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise InstallError(f"local conversion failed: {error}") from error
        if result.returncode != 0:
            raise InstallError(
                f"local conversion of {artifact.filename} exited "
                f"with status {result.returncode}"
            )
    shutil.rmtree(private_home)
    shutil.rmtree(temporary)


def _write_manifest(
    directory: Path,
    model: TtsModelSpec,
    *,
    sources_retained: bool,
) -> None:
    target = directory / MANIFEST_NAME
    data = (
        json.dumps(
            _manifest(model, sources_retained=sources_retained),
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prune_cache(
    cache_root: Path,
    artifacts: tuple[ArtifactSpec, ...],
    report: Callable[[str], None],
) -> None:
    for artifact in artifacts:
        cached = cache_root / artifact.sha256
        try:
            ok, _detail = _verify_cached(cached, artifact)
            if ok:
                cached.unlink()
        except OSError as error:
            report(f"warning: could not remove verified source cache {cached}: {error}")


def install_model(
    catalog_id: str = MODEL_ID,
    *,
    paths: RuntimePaths | None = None,
    accept_terms: bool = False,
    discard_verified_source: bool = False,
    report: Callable[[str], None] = lambda _message: None,
    downloader: Callable[..., str] | None = None,
    converter: Callable[
        [Path, Path, TtsModelSpec, Callable[[str], None]], None
    ]
    | None = None,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
) -> tuple[Path, bool]:
    """Download exact sources, convert offline, and atomically publish locally."""
    if not accept_terms:
        raise InstallError(
            f"model terms were not accepted. Review {UPSTREAM_TERMS_URL} and "
            "rerun with --accept-model-terms"
        )
    if minimum_free_bytes < 0:
        raise InstallError("minimum_free_bytes must be non-negative")
    model = get_model(catalog_id)
    runtime_paths = RuntimePaths.from_environment() if paths is None else paths
    target = _model_dir(runtime_paths, model)
    convert = _default_convert if converter is None else converter

    with _install_lock(runtime_paths, model):
        if os.path.lexists(target):
            installed, detail = inspect_install(
                model.catalog_id,
                paths=runtime_paths,
                verify_hashes=True,
            )
            if installed:
                return target, False
            raise InstallError(
                f"refusing to replace the existing invalid model at {target}: "
                f"{detail}. Move it aside and retry."
            )

        # Refuse before allocating cache space or reaching the network when the
        # independently packaged converter closure is not ready.
        if converter is None:
            _validate_converter()

        models_root = _models_root(runtime_paths)
        free_bytes = shutil.disk_usage(models_root).free
        if free_bytes < minimum_free_bytes:
            raise InstallError(
                f"optional TTS conversion needs at least {minimum_free_bytes} "
                f"free bytes; {free_bytes} are available"
            )
        cache_root = _ensure_private_directory(_cache_root(runtime_paths))
        if cache_root.stat().st_dev != models_root.stat().st_dev:
            cache_free_bytes = shutil.disk_usage(cache_root).free
            if cache_free_bytes < model.download_bytes:
                raise InstallError(
                    f"optional TTS download needs at least {model.download_bytes} "
                    f"free cache bytes; {cache_free_bytes} are available"
                )
        download = _kilix_download() if downloader is None else downloader
        downloads = model.source_artifacts + (model.notice_artifact,)
        cached = {
            artifact.role: _cache_artifact(
                artifact,
                cache_root,
                download=download,
                report=report,
            )
            for artifact in downloads
        }

        stage: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{model.catalog_id}.stage-", dir=models_root)
        )
        stage.chmod(0o700)
        try:
            source_dir = stage / "source"
            runtime_dir = stage / "runtime"
            source_dir.mkdir(mode=0o700)
            runtime_dir.mkdir(mode=0o700)
            for artifact in model.source_artifacts:
                _stage_verified(
                    cached[artifact.role],
                    source_dir / artifact.filename,
                    artifact,
                )
            _stage_verified(
                cached[model.notice_artifact.role],
                stage / MODEL_CARD_NAME,
                model.notice_artifact,
            )

            convert(source_dir, runtime_dir, model, report)
            ok, detail = _exact_entries(
                source_dir,
                {artifact.filename for artifact in model.source_artifacts},
            )
            if not ok:
                raise InstallError(f"converter changed the source layout: {detail}")
            for artifact in model.source_artifacts:
                ok, detail = _regular_file_status(
                    source_dir / artifact.filename,
                    artifact,
                    verify_hash=True,
                )
                if not ok:
                    raise InstallError(f"converter changed a source artifact: {detail}")
            ok, detail = _exact_entries(
                runtime_dir,
                {artifact.filename for artifact in model.derived_artifacts},
            )
            if not ok:
                raise InstallError(f"converter produced an unexpected layout: {detail}")
            for artifact in model.derived_artifacts:
                output = runtime_dir / artifact.filename
                try:
                    output.chmod(0o600, follow_symlinks=False)
                except OSError as error:
                    raise InstallError(
                        f"cannot secure converted artifact {artifact.filename}: {error}"
                    ) from error
                ok, detail = _regular_file_status(
                    output, artifact, verify_hash=True
                )
                if not ok:
                    raise InstallError(f"converted artifact failed verification: {detail}")
                with output.open("rb") as stream:
                    os.fsync(stream.fileno())

            sources_retained = not discard_verified_source
            if discard_verified_source:
                shutil.rmtree(source_dir)
            _write_manifest(
                stage,
                model,
                sources_retained=sources_retained,
            )
            expected_top = {MANIFEST_NAME, MODEL_CARD_NAME, "runtime"}
            if sources_retained:
                expected_top.add("source")
            ok, detail = _exact_entries(stage, expected_top)
            if not ok:
                raise InstallError(f"staged model layout is not exact: {detail}")
            _fsync_directory(runtime_dir)
            if sources_retained:
                _fsync_directory(source_dir)
            _fsync_directory(stage)
            os.rename(stage, target)
            stage = None
            _fsync_directory(models_root)
        except BaseException:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
            raise

        if discard_verified_source:
            _prune_cache(cache_root, downloads, report)

    verified, detail = inspect_install(
        model.catalog_id,
        paths=runtime_paths,
        verify_hashes=True,
    )
    if not verified:
        raise InstallError(f"published model did not verify: {detail}")
    return target, True


def remove_model(
    catalog_id: str = MODEL_ID,
    *,
    paths: RuntimePaths | None = None,
    confirm: bool = False,
) -> tuple[Path, bool]:
    """Remove only an exact, fully verified local install after confirmation."""
    if not confirm:
        raise InstallError("removal requires --confirm")
    model = get_model(catalog_id)
    runtime_paths = RuntimePaths.from_environment() if paths is None else paths
    target = _model_dir(runtime_paths, model)
    with _install_lock(runtime_paths, model):
        if not os.path.lexists(target):
            return target, False
        quarantine = target.parent / f".{model.catalog_id}.remove-{os.getpid()}"
        if os.path.lexists(quarantine):
            raise InstallError(f"refusing existing removal quarantine {quarantine}")
        # Bind verification to the directory that can actually be deleted.
        # Checking target first and renaming later would leave a replacement
        # window in a destructive operation. Invalid content is restored.
        os.rename(target, quarantine)
        _fsync_directory(target.parent)
        verified, detail = _inspect_model_directory(
            quarantine,
            model,
            verify_hashes=True,
        )
        if not verified:
            if os.path.lexists(target):
                raise InstallError(
                    f"refusing to remove an unverified directory; it remains "
                    f"detached at {quarantine} because {target} was recreated: "
                    f"{detail}"
                )
            os.rename(quarantine, target)
            _fsync_directory(target.parent)
            raise InstallError(
                f"refusing to remove an unverified directory at {target}: {detail}"
            )
        try:
            shutil.rmtree(quarantine)
        except OSError as error:
            raise InstallError(
                f"model was detached but cleanup failed at {quarantine}: {error}"
            ) from error
        _fsync_directory(target.parent)
    return target, True


__all__ = [
    "ArtifactSpec",
    "CATALOG_SCHEMA",
    "CONVERTER_CLOSURE",
    "CONVERTER_PATH",
    "INSTALL_SCHEMA",
    "LICENSE_URL",
    "MANIFEST_NAME",
    "MINIMUM_FREE_BYTES",
    "MODEL_CARD_NAME",
    "MODEL_ID",
    "MODEL_IDS",
    "MODELS",
    "POCKET_Q8",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "TERMS_SUMMARY",
    "TtsModelSpec",
    "UPSTREAM_TERMS_URL",
    "catalog_record",
    "get_model",
    "inspect_install",
    "install_model",
    "remove_model",
    "verified_runtime_artifacts",
]
