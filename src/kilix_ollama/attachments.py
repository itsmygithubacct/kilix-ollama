"""Owner-private, bounded image snapshots for persistent local chat."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
from typing import Mapping, Sequence
import uuid
import zlib

from .errors import ConversationError
from .paths import RuntimePaths


ATTACHMENT_SCHEMA = "kilix.ollama.attachment/v0-unfrozen"
MAX_ATTACHMENTS_PER_MESSAGE = 4
MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024
MAX_ATTACHMENT_TOTAL_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 16_777_216
MAX_IMAGE_DECODE_BYTES = 64 * 1024 * 1024
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})\Z")
_MEDIA_TYPES = frozenset(("image/jpeg", "image/png"))
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF = frozenset((0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))


def _bounded_integer(value: object, label: str, *, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ConversationError(f"attachment {label} is outside its limit")
    return value


def _dimensions(width: int, height: int) -> tuple[int, int]:
    checked_width = _bounded_integer(
        width,
        "width",
        maximum=MAX_IMAGE_DIMENSION,
    )
    checked_height = _bounded_integer(
        height,
        "height",
        maximum=MAX_IMAGE_DIMENSION,
    )
    if checked_width * checked_height > MAX_IMAGE_PIXELS:
        raise ConversationError("attachment pixel count exceeds its limit")
    return checked_width, checked_height


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(_PNG_SIGNATURE):
        raise ConversationError("attachment is not a supported PNG image")
    offset = len(_PNG_SIGNATURE)
    width = height = 0
    saw_header = False
    saw_data = False
    saw_end = False
    saw_palette = False
    data_closed = False
    bit_depth = color_type = interlace = -1
    compressed: list[bytes] = []
    chunk_index = 0
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ConversationError("attachment PNG chunk is truncated")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if length > MAX_ATTACHMENT_BYTES or end > len(payload):
            raise ConversationError("attachment PNG chunk exceeds its bounds")
        if len(kind) != 4 or not all(
            0x41 <= byte <= 0x5A or 0x61 <= byte <= 0x7A for byte in kind
        ):
            raise ConversationError("attachment PNG chunk type is invalid")
        if kind[2] & 0x20:
            raise ConversationError("attachment PNG chunk reserved bit is invalid")
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
            raise ConversationError("attachment PNG chunk checksum is invalid")
        if chunk_index == 0 and kind != b"IHDR":
            raise ConversationError("attachment PNG header is not first")
        if kind == b"IHDR":
            if saw_header or length != 13:
                raise ConversationError("attachment PNG header is invalid")
            width, height = struct.unpack(">II", data[:8])
            bit_depth, color_type, compression, filtering, interlace = data[8:]
            valid_depths = {
                0: frozenset((1, 2, 4, 8, 16)),
                2: frozenset((8, 16)),
                3: frozenset((1, 2, 4, 8)),
                4: frozenset((8, 16)),
                6: frozenset((8, 16)),
            }
            if (
                color_type not in valid_depths
                or bit_depth not in valid_depths[color_type]
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                raise ConversationError("attachment PNG encoding is unsupported")
            _dimensions(width, height)
            saw_header = True
        elif kind == b"IDAT":
            if not saw_header or saw_end or data_closed:
                raise ConversationError("attachment PNG data ordering is invalid")
            if color_type == 3 and not saw_palette:
                raise ConversationError("attachment indexed PNG has no palette")
            saw_data = True
            compressed.append(data)
        elif kind == b"IEND":
            if length != 0 or not saw_header or not saw_data or saw_end:
                raise ConversationError("attachment PNG end marker is invalid")
            saw_end = True
            if end != len(payload):
                raise ConversationError("attachment PNG has trailing data")
        elif kind == b"PLTE":
            if (
                saw_palette
                or saw_data
                or color_type in (0, 4)
                or not 3 <= length <= 768
                or length % 3
                or (color_type == 3 and length // 3 > 2**bit_depth)
            ):
                raise ConversationError("attachment PNG palette is invalid")
            saw_palette = True
        elif (kind[0] & 0x20) == 0:
            raise ConversationError("attachment PNG has an unknown critical chunk")
        elif saw_end:
            raise ConversationError("attachment PNG has data after its end marker")
        if saw_data and kind not in (b"IDAT", b"IEND"):
            data_closed = True
        offset = end
        chunk_index += 1
    if not (saw_header and saw_data and saw_end):
        raise ConversationError("attachment PNG is incomplete")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]

    def row_bytes(pixels: int) -> int:
        return (pixels * channels * bit_depth + 7) // 8

    rows: list[int] = []
    if interlace == 0:
        rows = [row_bytes(width)] * height
    else:
        passes = (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        )
        for x_start, y_start, x_step, y_step in passes:
            pass_width = max(0, (width - x_start + x_step - 1) // x_step)
            pass_height = max(0, (height - y_start + y_step - 1) // y_step)
            if pass_width and pass_height:
                rows.extend([row_bytes(pass_width)] * pass_height)
    expected = sum(length + 1 for length in rows)
    if expected > MAX_IMAGE_DECODE_BYTES:
        raise ConversationError("attachment PNG decoded bytes exceed their limit")
    decoder = zlib.decompressobj()
    decoded = bytearray()
    try:
        for part in compressed:
            remaining = expected + 1 - len(decoded)
            decoded.extend(decoder.decompress(part, max(1, remaining)))
            if decoder.unconsumed_tail or len(decoded) > expected:
                raise ConversationError("attachment PNG decoded bytes exceed their dimensions")
        decoded.extend(decoder.flush(max(1, expected + 1 - len(decoded))))
    except zlib.error as error:
        raise ConversationError("attachment PNG image data is invalid") from error
    if (
        len(decoded) != expected
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ConversationError("attachment PNG decoded bytes do not match its dimensions")
    decoded_offset = 0
    for length in rows:
        if decoded[decoded_offset] > 4:
            raise ConversationError("attachment PNG row filter is invalid")
        decoded_offset += length + 1
    return _dimensions(width, height)


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 8 or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ConversationError("attachment is not a complete JPEG image")
    offset = 2
    width = height = 0
    saw_scan = False
    while offset < len(payload) - 2:
        if payload[offset] != 0xFF:
            raise ConversationError("attachment JPEG marker is invalid")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            raise ConversationError("attachment JPEG marker is truncated")
        marker = payload[offset]
        offset += 1
        if marker == 0x00:
            raise ConversationError("attachment JPEG contains escaped data before a scan")
        if marker == 0xDA:
            if width == 0 or height == 0:
                raise ConversationError("attachment JPEG has no bounded dimensions")
            if offset + 2 > len(payload):
                raise ConversationError("attachment JPEG scan header is truncated")
            length = struct.unpack(">H", payload[offset : offset + 2])[0]
            if length < 2 or offset + length > len(payload) - 2:
                raise ConversationError("attachment JPEG scan header is invalid")
            saw_scan = True
            break
        if marker == 0xD9:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            raise ConversationError("attachment JPEG segment is truncated")
        length = struct.unpack(">H", payload[offset : offset + 2])[0]
        if length < 2 or offset + length > len(payload) - 2:
            raise ConversationError("attachment JPEG segment is invalid")
        if marker in _JPEG_SOF:
            if length < 8:
                raise ConversationError("attachment JPEG frame is invalid")
            frame = payload[offset + 2 : offset + length]
            frame_height, frame_width = struct.unpack(">HH", frame[1:5])
            components = frame[5]
            if components not in (1, 3, 4):
                raise ConversationError("attachment JPEG component count is unsupported")
            if width and (width != frame_width or height != frame_height):
                raise ConversationError("attachment JPEG dimensions are ambiguous")
            width, height = _dimensions(frame_width, frame_height)
        offset += length
    if not saw_scan:
        raise ConversationError("attachment JPEG has no image scan")
    return _dimensions(width, height)


def inspect_image(payload: bytes) -> tuple[str, int, int]:
    if not isinstance(payload, bytes) or not payload:
        raise ConversationError("attachment image is empty")
    if len(payload) > MAX_ATTACHMENT_BYTES:
        raise ConversationError("attachment image exceeds its byte limit")
    if payload.startswith(_PNG_SIGNATURE):
        width, height = _png_dimensions(payload)
        return "image/png", width, height
    if payload.startswith(b"\xff\xd8"):
        width, height = _jpeg_dimensions(payload)
        return "image/jpeg", width, height
    raise ConversationError("attachment is not a supported PNG or JPEG image")


def validate_attachment_record(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConversationError("attachment record is not an object")
    expected = {
        "schema",
        "schema_status",
        "kind",
        "digest",
        "media_type",
        "bytes",
        "width",
        "height",
    }
    if set(value) != expected:
        raise ConversationError("attachment record members do not match the schema")
    if value.get("schema") != ATTACHMENT_SCHEMA:
        raise ConversationError("attachment schema is unsupported")
    if value.get("schema_status") != "development-unfrozen":
        raise ConversationError("attachment schema status is invalid")
    if value.get("kind") != "image":
        raise ConversationError("attachment kind is unsupported")
    digest = value.get("digest")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ConversationError("attachment digest is invalid")
    media_type = value.get("media_type")
    if media_type not in _MEDIA_TYPES:
        raise ConversationError("attachment media type is unsupported")
    size = _bounded_integer(value.get("bytes"), "byte count", maximum=MAX_ATTACHMENT_BYTES)
    width, height = _dimensions(value.get("width"), value.get("height"))  # type: ignore[arg-type]
    return {
        "schema": ATTACHMENT_SCHEMA,
        "schema_status": "development-unfrozen",
        "kind": "image",
        "digest": digest,
        "media_type": media_type,
        "bytes": size,
        "width": width,
        "height": height,
    }


def validate_attachment_records(value: object) -> list[dict[str, object]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not 1 <= len(value) <= MAX_ATTACHMENTS_PER_MESSAGE
    ):
        raise ConversationError("message attachments are not a bounded array")
    records = [validate_attachment_record(record) for record in value]
    if sum(int(record["bytes"]) for record in records) > MAX_ATTACHMENT_TOTAL_BYTES:
        raise ConversationError("message attachments exceed their total byte limit")
    digests = [str(record["digest"]) for record in records]
    if len(set(digests)) != len(digests):
        raise ConversationError("message attachments repeat a digest")
    return records


def _ensure_private_directory(path: Path, *, uid: int) -> None:
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise ConversationError("no safe anchor exists for attachment storage")
        cursor = parent
    try:
        anchor = cursor.lstat()
    except OSError as error:
        raise ConversationError("attachment storage anchor cannot be inspected") from error
    if stat.S_ISLNK(anchor.st_mode) or not stat.S_ISDIR(anchor.st_mode):
        raise ConversationError("attachment storage anchor is not a real directory")
    if anchor.st_uid not in (0, uid) or anchor.st_mode & 0o022:
        raise ConversationError("attachment storage anchor is not owner-controlled")
    for candidate in reversed(missing):
        descriptor = -1
        try:
            candidate.mkdir(mode=0o700)
            os.chmod(candidate, 0o700, follow_symlinks=False)
            descriptor = os.open(
                candidate,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise ConversationError("attachment directory cannot be created") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ConversationError("new attachment directory is unsafe")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ConversationError("attachment directory cannot be inspected") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ConversationError("attachment directory must be owner-only mode 0700")


def _read_descriptor(descriptor: int, *, maximum: int) -> bytes:
    payload = bytearray()
    while True:
        block = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
        if not block:
            break
        payload.extend(block)
        if len(payload) > maximum:
            raise ConversationError("attachment image exceeds its byte limit")
    return bytes(payload)


class AttachmentStore:
    """Copy explicitly selected images into immutable private snapshots."""

    def __init__(self, paths: RuntimePaths, *, uid: int | None = None) -> None:
        self.paths = paths
        self.uid = os.getuid() if uid is None else uid
        if not isinstance(self.uid, int) or isinstance(self.uid, bool) or self.uid < 0:
            raise ConversationError("attachment owner uid is invalid")
        self.root = paths.data_dir / "attachments"

    def _prepare(self, path: Path) -> tuple[dict[str, object], bytes]:
        try:
            selected = Path(path)
        except (TypeError, ValueError) as error:
            raise ConversationError("attachment selection is not a filesystem path") from error
        descriptor = -1
        try:
            descriptor = os.open(
                selected,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ConversationError("attachment selection is not a regular file")
            if not 1 <= before.st_size <= MAX_ATTACHMENT_BYTES:
                raise ConversationError("attachment image is outside its byte limit")
            payload = _read_descriptor(descriptor, maximum=MAX_ATTACHMENT_BYTES)
            after = os.fstat(descriptor)
        except ConversationError:
            raise
        except OSError as error:
            raise ConversationError("attachment selection cannot be read safely") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(payload) != before.st_size:
            raise ConversationError("attachment selection changed while it was read")
        media_type, width, height = inspect_image(payload)
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        record = validate_attachment_record(
            {
                "schema": ATTACHMENT_SCHEMA,
                "schema_status": "development-unfrozen",
                "kind": "image",
                "digest": digest,
                "media_type": media_type,
                "bytes": len(payload),
                "width": width,
                "height": height,
            }
        )
        return record, payload

    def _target(self, record: Mapping[str, object]) -> Path:
        match = _DIGEST.fullmatch(str(record["digest"]))
        if match is None:
            raise ConversationError("attachment digest is invalid")
        return self.root / match.group(1)

    def _publish(self, record: dict[str, object], payload: bytes) -> None:
        _ensure_private_directory(self.root, uid=self.uid)
        target = self._target(record)
        if os.path.lexists(target):
            if self.read(record) != payload:
                raise ConversationError("attachment digest target has different content")
            return
        temporary = self.root / f".{target.name}.{uuid.uuid4().hex}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ConversationError("attachment write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                if self.read(record) != payload:
                    raise ConversationError("attachment target appeared with different content")
            temporary.unlink()
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except ConversationError:
            raise
        except OSError as error:
            raise ConversationError("attachment cannot be committed atomically") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def snapshot_many(self, paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
        if (
            not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes, bytearray))
            or not 1 <= len(paths) <= MAX_ATTACHMENTS_PER_MESSAGE
        ):
            raise ConversationError("attachment selections are not a bounded array")
        prepared = tuple(self._prepare(path) for path in paths)
        records = validate_attachment_records([record for record, _payload in prepared])
        for record, payload in prepared:
            self._publish(record, payload)
        return tuple(records)

    def snapshot(self, path: Path) -> dict[str, object]:
        return self.snapshot_many((path,))[0]

    def read(self, value: Mapping[str, object]) -> bytes:
        record = validate_attachment_record(value)
        _ensure_private_directory(self.root, uid=self.uid)
        target = self._target(record)
        descriptor = -1
        try:
            descriptor = os.open(
                target,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != self.uid
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != record["bytes"]
            ):
                raise ConversationError("attachment snapshot metadata is unsafe")
            payload = _read_descriptor(descriptor, maximum=MAX_ATTACHMENT_BYTES)
        except ConversationError:
            raise
        except OSError as error:
            raise ConversationError("attachment snapshot cannot be read safely") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if digest != record["digest"]:
            raise ConversationError("attachment snapshot digest does not match")
        media_type, width, height = inspect_image(payload)
        if (
            media_type != record["media_type"]
            or width != record["width"]
            or height != record["height"]
        ):
            raise ConversationError("attachment snapshot metadata does not match")
        return payload

    def wire_images(self, value: object) -> list[str]:
        records = validate_attachment_records(value)
        return [
            base64.b64encode(self.read(record)).decode("ascii")
            for record in records
        ]


__all__ = [
    "ATTACHMENT_SCHEMA",
    "AttachmentStore",
    "MAX_ATTACHMENT_BYTES",
    "MAX_ATTACHMENT_TOTAL_BYTES",
    "MAX_ATTACHMENTS_PER_MESSAGE",
    "MAX_IMAGE_DIMENSION",
    "MAX_IMAGE_DECODE_BYTES",
    "MAX_IMAGE_PIXELS",
    "inspect_image",
    "validate_attachment_record",
    "validate_attachment_records",
]
