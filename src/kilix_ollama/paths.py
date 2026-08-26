"""Deterministic XDG layout for the private client and service boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Mapping

from .errors import RuntimeContractError


def _absolute(value: str | os.PathLike[str], label: str) -> Path:
    try:
        path = Path(value)
    except (TypeError, ValueError) as error:
        raise RuntimeContractError(f"{label} is not a filesystem path") from error
    if not path.is_absolute():
        raise RuntimeContractError(f"{label} must be an absolute path")
    if path == Path("/"):
        raise RuntimeContractError(f"{label} must not be the filesystem root")
    if ".." in path.parts:
        raise RuntimeContractError(f"{label} contains a parent traversal")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in str(path)):
        raise RuntimeContractError(f"{label} contains a control character")
    return path


@dataclass(frozen=True)
class RuntimePaths:
    """Private paths owned by one unprivileged Kilix Ollama instance."""

    runtime_dir: Path
    mediation_socket: Path
    upstream_socket: Path
    state_dir: Path
    upstream_home: Path
    data_dir: Path
    model_store: Path
    cache_dir: Path
    temp_dir: Path

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        uid: int | None = None,
        home: Path | None = None,
    ) -> "RuntimePaths":
        values = os.environ if environ is None else environ
        if uid is not None and (not isinstance(uid, int) or isinstance(uid, bool) or uid < 0):
            raise RuntimeContractError("uid must be a non-negative integer")
        owner = os.getuid() if uid is None else uid
        user_home = _absolute(
            home if home is not None else values.get("HOME", str(Path.home())),
            "home",
        )
        runtime_base = values.get("XDG_RUNTIME_DIR", f"/run/user/{owner}")
        state_base = values.get("XDG_STATE_HOME", str(user_home / ".local/state"))
        data_base = values.get("XDG_DATA_HOME", str(user_home / ".local/share"))
        cache_base = values.get("XDG_CACHE_HOME", str(user_home / ".cache"))

        runtime_dir = _absolute(runtime_base, "XDG_RUNTIME_DIR") / "kilix-ollama"
        state_dir = _absolute(state_base, "XDG_STATE_HOME") / "kilix-ollama"
        data_dir = _absolute(data_base, "XDG_DATA_HOME") / "kilix-ollama"
        cache_dir = _absolute(cache_base, "XDG_CACHE_HOME") / "kilix-ollama"
        return cls(
            runtime_dir=runtime_dir,
            mediation_socket=runtime_dir / "api.sock",
            upstream_socket=runtime_dir / "upstream.sock",
            state_dir=state_dir,
            upstream_home=state_dir / "upstream-home",
            data_dir=data_dir,
            model_store=data_dir / "models",
            cache_dir=cache_dir,
            temp_dir=runtime_dir / "tmp",
        )

    def record(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def private_directory_status(path: Path, *, uid: int | None = None) -> tuple[bool, str]:
    """Inspect a private directory without creating or repairing it."""
    if uid is not None and (not isinstance(uid, int) or isinstance(uid, bool) or uid < 0):
        return False, "invalid expected uid"
    owner = os.getuid() if uid is None else uid
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return False, "missing"
    except OSError as error:
        return False, f"unreadable: {error}"
    if path.is_symlink():
        return False, "is a symlink"
    if not path.is_dir():
        return False, "is not a directory"
    if stat.st_uid != owner:
        return False, f"owned by uid {stat.st_uid}, expected {owner}"
    mode = stat.st_mode & 0o777
    if mode != 0o700:
        return False, f"mode is {mode:04o}, expected 0700"
    return True, "owner-only directory"


__all__ = ["RuntimePaths", "private_directory_status"]
