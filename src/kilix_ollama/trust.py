"""Checks for fixed executables below a non-writable trusted root."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from .errors import RuntimeContractError


def validate_trusted_executable(
    executable: Path,
    *,
    expected_uid: int = 0,
    trusted_root: Path = Path("/"),
) -> Path:
    """Reject links, ownership changes, writable components, and non-executables."""
    path = Path(executable)
    root = Path(trusted_root)
    if not path.is_absolute() or not root.is_absolute():
        raise RuntimeContractError("launcher validation paths must be absolute")
    if (
        not isinstance(expected_uid, int)
        or isinstance(expected_uid, bool)
        or expected_uid < 0
    ):
        raise RuntimeContractError("trusted executable uid must be non-negative")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RuntimeContractError("launcher is outside its trusted root") from error

    candidates = [root]
    cursor = root
    for part in relative.parts:
        cursor /= part
        candidates.append(cursor)
    for index, candidate in enumerate(candidates):
        final = index == len(candidates) - 1
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise RuntimeContractError(
                f"cannot inspect trusted launcher path {candidate}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeContractError(f"symlink in trusted launcher path: {candidate}")
        if final and not stat.S_ISREG(metadata.st_mode):
            raise RuntimeContractError("runtime launcher is not a regular file")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeContractError(
                f"trusted launcher ancestor is not a directory: {candidate}"
            )
        if metadata.st_uid != expected_uid:
            raise RuntimeContractError(
                f"trusted launcher path {candidate} is owned by uid "
                f"{metadata.st_uid}, expected {expected_uid}"
            )
        if metadata.st_mode & 0o022:
            raise RuntimeContractError(
                f"group/world writable trusted launcher path: {candidate}"
            )
    if not os.access(path, os.X_OK):
        raise RuntimeContractError("runtime launcher is not executable")
    return path


def trusted_executable_sha256(
    executable: Path,
    *,
    expected_uid: int = 0,
    trusted_root: Path = Path("/"),
) -> str:
    """Hash the validated regular file through a no-follow descriptor."""
    path = validate_trusted_executable(
        executable,
        expected_uid=expected_uid,
        trusted_root=trusted_root,
    )
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise RuntimeContractError(f"cannot open trusted runtime launcher: {error}") from error
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != expected_uid:
            raise RuntimeContractError("trusted runtime launcher changed during validation")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


__all__ = ["trusted_executable_sha256", "validate_trusted_executable"]
