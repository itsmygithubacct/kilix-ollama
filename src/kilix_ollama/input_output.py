"""Bounded prompt input and atomic output-file helpers."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import BinaryIO, TextIO
import uuid

from .errors import RuntimeContractError


MAX_PROMPT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def _decode_prompt(payload: bytes, source: str) -> str:
    if len(payload) > MAX_PROMPT_BYTES:
        raise RuntimeContractError(f"{source} prompt exceeds its byte limit")
    if b"\x00" in payload:
        raise RuntimeContractError(f"{source} prompt contains binary NUL bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeContractError(f"{source} prompt is not UTF-8 text") from error
    if not text:
        raise RuntimeContractError("prompt is empty")
    return text


def read_prompt_file(path: Path) -> str:
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise RuntimeContractError("selected prompt file cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PROMPT_BYTES:
            raise RuntimeContractError("selected prompt file is not a bounded regular file")
        payload = bytearray()
        while len(payload) <= MAX_PROMPT_BYTES:
            block = os.read(descriptor, min(64 * 1024, MAX_PROMPT_BYTES + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
        return _decode_prompt(bytes(payload), "file")
    finally:
        os.close(descriptor)


def read_prompt_stdin(stream: TextIO) -> str:
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        payload = buffer.read(MAX_PROMPT_BYTES + 1)
        if not isinstance(payload, bytes):
            raise RuntimeContractError("stdin did not return bytes")
        return _decode_prompt(payload, "stdin")
    text = stream.read(MAX_PROMPT_BYTES + 1)
    if not isinstance(text, str):
        raise RuntimeContractError("stdin did not return text")
    try:
        payload = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RuntimeContractError("stdin prompt contains invalid Unicode") from error
    return _decode_prompt(payload, "stdin")


def select_prompt(
    arguments: list[str],
    *,
    prompt_file: Path | None,
    stdin: TextIO,
) -> str:
    if arguments and prompt_file is not None:
        raise RuntimeContractError("prompt arguments and --file are mutually exclusive")
    if arguments:
        return _decode_prompt(" ".join(arguments).encode("utf-8"), "argument")
    if prompt_file is not None:
        return read_prompt_file(prompt_file)
    if hasattr(stdin, "isatty") and stdin.isatty():
        raise RuntimeContractError("prompt is required when stdin is a terminal")
    return read_prompt_stdin(stdin)


def atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    if not isinstance(text, str):
        raise RuntimeContractError("output must be text")
    try:
        payload = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RuntimeContractError("output contains invalid Unicode") from error
    if len(payload) > MAX_OUTPUT_BYTES:
        raise RuntimeContractError("output exceeds its byte limit")
    target = Path(path)
    parent = target.parent if target.parent != Path("") else Path(".")
    try:
        parent_metadata = parent.lstat()
    except OSError as error:
        raise RuntimeContractError("output directory cannot be inspected") from error
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeContractError("output parent is not a real directory")
    if os.path.lexists(target):
        try:
            metadata = target.lstat()
        except OSError as error:
            raise RuntimeContractError("output target cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeContractError("output target is not a regular file")
        if not overwrite:
            raise RuntimeContractError("output exists; pass --overwrite to replace it")
    temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeContractError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError as error:
                raise RuntimeContractError("output target appeared during no-clobber write") from error
            temporary.unlink()
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except RuntimeContractError:
        raise
    except OSError as error:
        raise RuntimeContractError("output cannot be committed atomically") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


__all__ = [
    "MAX_OUTPUT_BYTES",
    "MAX_PROMPT_BYTES",
    "atomic_write",
    "read_prompt_file",
    "read_prompt_stdin",
    "select_prompt",
]
