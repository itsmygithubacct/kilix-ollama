"""Owner-private, bounded image snapshots for persistent local chat."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
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
_JPEG_SOF = frozenset((0xC0, 0xC1, 0xC2))
_JPEG_OTHER_SOF = frozenset((0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))
_MAX_JPEG_MARKERS = 4096


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


def _jpeg_quantization_tables(segment: bytes, tables: set[int]) -> None:
    offset = 0
    while offset < len(segment):
        descriptor = segment[offset]
        offset += 1
        precision = descriptor >> 4
        identifier = descriptor & 0x0F
        if precision not in (0, 1) or identifier > 3:
            raise ConversationError("attachment JPEG quantization table is invalid")
        size = 64 * (precision + 1)
        end = offset + size
        if end > len(segment):
            raise ConversationError("attachment JPEG quantization table is truncated")
        values = segment[offset:end]
        step = precision + 1
        if any(not any(values[index : index + step]) for index in range(0, size, step)):
            raise ConversationError("attachment JPEG quantization table contains zero")
        tables.add(identifier)
        offset = end
    if offset == 0:
        raise ConversationError("attachment JPEG quantization segment is empty")


def _jpeg_huffman_tables(segment: bytes, tables: set[tuple[int, int]]) -> None:
    offset = 0
    while offset < len(segment):
        if offset + 17 > len(segment):
            raise ConversationError("attachment JPEG Huffman table is truncated")
        descriptor = segment[offset]
        table_class = descriptor >> 4
        identifier = descriptor & 0x0F
        if table_class not in (0, 1) or identifier > 3:
            raise ConversationError("attachment JPEG Huffman table is invalid")
        counts = segment[offset + 1 : offset + 17]
        symbol_count = sum(counts)
        if not 1 <= symbol_count <= 256 or offset + 17 + symbol_count > len(segment):
            raise ConversationError("attachment JPEG Huffman symbols are invalid")
        available = 1
        for count in counts:
            available = available * 2 - count
            if available < 0:
                raise ConversationError("attachment JPEG Huffman tree is oversubscribed")
        tables.add((table_class, identifier))
        offset += 17 + symbol_count
    if offset == 0:
        raise ConversationError("attachment JPEG Huffman segment is empty")


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    if (
        len(payload) < 8
        or not payload.startswith(b"\xff\xd8")
        or not payload.endswith(b"\xff\xd9")
    ):
        raise ConversationError("attachment is not a complete JPEG image")
    offset = 2
    width = height = 0
    frame_marker: int | None = None
    frame_components: dict[int, tuple[int, int, int]] = {}
    quantization_tables: set[int] = set()
    huffman_tables: set[tuple[int, int]] = set()
    dc_scanned: set[int] = set()
    scan_count = 0
    marker_count = 1
    saw_end = False
    while offset < len(payload):
        if payload[offset] != 0xFF:
            raise ConversationError("attachment JPEG marker is invalid")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            raise ConversationError("attachment JPEG marker is truncated")
        marker = payload[offset]
        offset += 1
        marker_count += 1
        if marker_count > _MAX_JPEG_MARKERS:
            raise ConversationError("attachment JPEG marker count exceeds its limit")
        if marker == 0xD9:
            saw_end = True
            if offset != len(payload):
                raise ConversationError("attachment JPEG has trailing data")
            break
        if marker == 0x00:
            raise ConversationError("attachment JPEG has escaped data outside a scan")
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            raise ConversationError("attachment JPEG standalone marker is unsupported")
        if offset + 2 > len(payload):
            raise ConversationError("attachment JPEG segment is truncated")
        length = struct.unpack(">H", payload[offset : offset + 2])[0]
        if length < 2 or offset + length > len(payload):
            raise ConversationError("attachment JPEG segment is invalid")
        segment = payload[offset + 2 : offset + length]
        offset += length

        if marker == 0xDB:
            _jpeg_quantization_tables(segment, quantization_tables)
            continue
        if marker == 0xC4:
            _jpeg_huffman_tables(segment, huffman_tables)
            continue
        if marker == 0xDD:
            if len(segment) != 2:
                raise ConversationError("attachment JPEG restart interval is invalid")
            continue
        if marker in _JPEG_OTHER_SOF:
            raise ConversationError("attachment JPEG frame encoding is unsupported")
        if marker in _JPEG_SOF:
            if frame_marker is not None or len(segment) < 6:
                raise ConversationError("attachment JPEG frame is repeated or truncated")
            precision = segment[0]
            frame_height, frame_width = struct.unpack(">HH", segment[1:5])
            component_count = segment[5]
            if (
                precision != 8
                or component_count not in (1, 3, 4)
                or len(segment) != 6 + 3 * component_count
            ):
                raise ConversationError("attachment JPEG frame shape is unsupported")
            width, height = _dimensions(frame_width, frame_height)
            for index in range(component_count):
                start = 6 + 3 * index
                identifier, sampling, table = segment[start : start + 3]
                horizontal = sampling >> 4
                vertical = sampling & 0x0F
                if (
                    identifier == 0
                    or identifier in frame_components
                    or not 1 <= horizontal <= 4
                    or not 1 <= vertical <= 4
                    or table > 3
                ):
                    raise ConversationError("attachment JPEG frame component is invalid")
                frame_components[identifier] = (horizontal, vertical, table)
            frame_marker = marker
            continue
        if marker == 0xDA:
            if frame_marker is None or not segment:
                raise ConversationError("attachment JPEG scan has no frame or table")
            component_count = segment[0]
            if (
                not 1 <= component_count <= len(frame_components)
                or len(segment) != 4 + 2 * component_count
            ):
                raise ConversationError("attachment JPEG scan component table is invalid")
            selected: list[tuple[int, int, int]] = []
            seen_components: set[int] = set()
            for index in range(component_count):
                identifier = segment[1 + 2 * index]
                selector = segment[2 + 2 * index]
                dc_table = selector >> 4
                ac_table = selector & 0x0F
                if (
                    identifier not in frame_components
                    or identifier in seen_components
                    or dc_table > 3
                    or ac_table > 3
                ):
                    raise ConversationError("attachment JPEG scan component is invalid")
                seen_components.add(identifier)
                selected.append((identifier, dc_table, ac_table))
            spectral_start, spectral_end, approximation = segment[-3:]
            high = approximation >> 4
            low = approximation & 0x0F
            if frame_marker in (0xC0, 0xC1):
                if (spectral_start, spectral_end, high, low) != (0, 63, 0, 0):
                    raise ConversationError("attachment sequential JPEG scan is invalid")
                required_tables = {
                    table
                    for _identifier, dc_table, ac_table in selected
                    for table in ((0, dc_table), (1, ac_table))
                }
            else:
                if (
                    not 0 <= spectral_start <= spectral_end <= 63
                    or (spectral_start == 0 and spectral_end != 0)
                    or (spectral_start > 0 and component_count != 1)
                    or high > 13
                    or low > 13
                    or (high > 0 and high != low + 1)
                ):
                    raise ConversationError("attachment progressive JPEG scan is invalid")
                if spectral_start == 0 and high == 0:
                    required_tables = {(0, dc_table) for _, dc_table, _ in selected}
                elif spectral_start > 0:
                    required_tables = {(1, ac_table) for _, _, ac_table in selected}
                else:
                    required_tables = set()
            if not required_tables <= huffman_tables:
                raise ConversationError("attachment JPEG scan references a missing Huffman table")
            if any(
                frame_components[identifier][2] not in quantization_tables
                for identifier, _dc, _ac in selected
            ):
                raise ConversationError("attachment JPEG scan references a missing quantization table")
            if spectral_start == 0:
                dc_scanned.update(seen_components)
            scan_count += 1

            entropy_bytes = 0
            while offset < len(payload):
                if payload[offset] != 0xFF:
                    entropy_bytes += 1
                    offset += 1
                    continue
                marker_offset = offset
                offset += 1
                while offset < len(payload) and payload[offset] == 0xFF:
                    offset += 1
                if offset >= len(payload):
                    raise ConversationError("attachment JPEG entropy marker is truncated")
                entropy_marker = payload[offset]
                offset += 1
                if entropy_marker == 0x00:
                    entropy_bytes += 1
                    continue
                if 0xD0 <= entropy_marker <= 0xD7:
                    marker_count += 1
                    if marker_count > _MAX_JPEG_MARKERS:
                        raise ConversationError("attachment JPEG marker count exceeds its limit")
                    continue
                offset = marker_offset
                break
            if entropy_bytes == 0:
                raise ConversationError("attachment JPEG scan contains no entropy data")
            continue
        if not (0xE0 <= marker <= 0xEF or marker == 0xFE):
            raise ConversationError("attachment JPEG segment type is unsupported")
    if (
        not saw_end
        or frame_marker is None
        or scan_count == 0
        or set(frame_components) != dc_scanned
    ):
        raise ConversationError("attachment JPEG image structure is incomplete")
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


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _open_private_directory(path: Path, *, uid: int, create: bool) -> int:
    if not path.is_absolute() or not path.parts or path.parts[0] != path.anchor:
        raise ConversationError("attachment storage path must be absolute")
    components = path.parts[1:]
    if not components or any(component in ("", ".", "..") for component in components):
        raise ConversationError("attachment storage path is invalid")
    descriptor = -1
    try:
        descriptor = os.open(path.anchor, _DIRECTORY_FLAGS)
        for index, component in enumerate(components):
            final = index == len(components) - 1
            created = False
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise ConversationError("attachment directory does not exist")
                parent = os.fstat(descriptor)
                if not stat.S_ISDIR(parent.st_mode) or parent.st_uid not in (0, uid):
                    raise ConversationError("attachment storage parent is not controlled")
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                os.chmod(component, 0o700, dir_fd=descriptor)
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                created = True
            metadata = os.fstat(child)
            mode = stat.S_IMODE(metadata.st_mode)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in (0, uid)
                or (metadata.st_uid == uid and mode & 0o022)
                or (created and (metadata.st_uid != uid or mode != 0o700))
                or (final and (metadata.st_uid != uid or mode != 0o700))
            ):
                os.close(child)
                raise ConversationError("attachment directory component is unsafe")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except ConversationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ConversationError("attachment directory cannot be opened safely") from error


def _verify_private_directory(path: Path, descriptor: int, *, uid: int) -> None:
    verification = _open_private_directory(path, uid=uid, create=False)
    try:
        expected = os.fstat(descriptor)
        observed = os.fstat(verification)
        if (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino):
            raise ConversationError("attachment directory changed during use")
    finally:
        os.close(verification)


def _open_selected_file(path: Path) -> int:
    anchor = path.anchor if path.is_absolute() else "."
    components = path.parts[1:] if path.is_absolute() else path.parts
    if not components or any(component in ("", ".") for component in components):
        raise ConversationError("attachment selection is not a file path")
    parent = -1
    try:
        parent = os.open(anchor, _DIRECTORY_FLAGS)
        for component in components[:-1]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent)
            os.close(parent)
            parent = child
        return os.open(
            components[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
    except OSError as error:
        raise ConversationError("attachment selection cannot be opened without symlinks") from error
    finally:
        if parent >= 0:
            os.close(parent)


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


_PREPARED_TOKEN = object()


@dataclass(frozen=True)
class PreparedAttachmentBatch:
    """Opaque, validated image bytes retained in memory until commit."""

    _token: object
    _records: tuple[Mapping[str, object], ...]
    _payloads: tuple[bytes, ...]


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
            descriptor = _open_selected_file(selected)
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

    def _target_name(self, record: Mapping[str, object]) -> str:
        match = _DIGEST.fullmatch(str(record["digest"]))
        if match is None:
            raise ConversationError("attachment digest is invalid")
        return match.group(1)

    def _read_at(self, record: Mapping[str, object], directory: int) -> bytes:
        target = self._target_name(record)
        descriptor = -1
        try:
            descriptor = os.open(
                target,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory,
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

    def _publish(self, record: dict[str, object], payload: bytes) -> None:
        directory = -1
        descriptor = -1
        temporary: str | None = None
        try:
            directory = _open_private_directory(self.root, uid=self.uid, create=True)
            target = self._target_name(record)
            try:
                os.stat(target, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if self._read_at(record, directory) != payload:
                    raise ConversationError("attachment digest target has different content")
                _verify_private_directory(self.root, directory, uid=self.uid)
                return
            temporary = f".{target}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory,
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
                os.link(
                    temporary,
                    target,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if self._read_at(record, directory) != payload:
                    raise ConversationError("attachment target appeared with different content")
            os.unlink(temporary, dir_fd=directory)
            temporary = None
            os.fsync(directory)
            _verify_private_directory(self.root, directory, uid=self.uid)
        except ConversationError:
            raise
        except OSError as error:
            raise ConversationError("attachment cannot be committed atomically") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None and directory >= 0:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except OSError:
                    pass
            if directory >= 0:
                os.close(directory)

    def _validated_prepared(
        self,
        value: PreparedAttachmentBatch,
    ) -> tuple[tuple[dict[str, object], bytes], ...]:
        if (
            type(value) is not PreparedAttachmentBatch
            or value._token is not _PREPARED_TOKEN
            or len(value._records) != len(value._payloads)
        ):
            raise ConversationError("prepared attachment batch is invalid")
        records = validate_attachment_records(value._records)
        entries: list[tuple[dict[str, object], bytes]] = []
        for record, payload in zip(records, value._payloads, strict=True):
            if (
                not isinstance(payload, bytes)
                or len(payload) != record["bytes"]
                or f"sha256:{hashlib.sha256(payload).hexdigest()}" != record["digest"]
            ):
                raise ConversationError("prepared attachment payload does not match")
            entries.append((record, payload))
        return tuple(entries)

    def prepare_many(self, paths: Sequence[Path]) -> PreparedAttachmentBatch:
        if (
            not isinstance(paths, Sequence)
            or isinstance(paths, (str, bytes, bytearray))
            or not 1 <= len(paths) <= MAX_ATTACHMENTS_PER_MESSAGE
        ):
            raise ConversationError("attachment selections are not a bounded array")
        prepared = tuple(self._prepare(path) for path in paths)
        records = validate_attachment_records([record for record, _payload in prepared])
        return PreparedAttachmentBatch(
            _PREPARED_TOKEN,
            tuple(MappingProxyType(dict(record)) for record in records),
            tuple(payload for _record, payload in prepared),
        )

    def prepared_records(
        self,
        value: PreparedAttachmentBatch,
    ) -> tuple[dict[str, object], ...]:
        return tuple(record for record, _payload in self._validated_prepared(value))

    def wire_prepared(self, value: PreparedAttachmentBatch) -> list[str]:
        return [
            base64.b64encode(payload).decode("ascii")
            for _record, payload in self._validated_prepared(value)
        ]

    def publish_prepared(
        self,
        value: PreparedAttachmentBatch,
    ) -> tuple[dict[str, object], ...]:
        entries = self._validated_prepared(value)
        for record, payload in entries:
            self._publish(record, payload)
        return tuple(record for record, _payload in entries)

    def snapshot_many(self, paths: Sequence[Path]) -> tuple[dict[str, object], ...]:
        return self.publish_prepared(self.prepare_many(paths))

    def snapshot(self, path: Path) -> dict[str, object]:
        return self.snapshot_many((path,))[0]

    def read(self, value: Mapping[str, object]) -> bytes:
        record = validate_attachment_record(value)
        directory = -1
        try:
            directory = _open_private_directory(self.root, uid=self.uid, create=False)
            payload = self._read_at(record, directory)
            _verify_private_directory(self.root, directory, uid=self.uid)
            return payload
        finally:
            if directory >= 0:
                os.close(directory)

    def wire_images(self, value: object) -> list[str]:
        records = validate_attachment_records(value)
        directory = -1
        try:
            directory = _open_private_directory(self.root, uid=self.uid, create=False)
            images = [
                base64.b64encode(self._read_at(record, directory)).decode("ascii")
                for record in records
            ]
            _verify_private_directory(self.root, directory, uid=self.uid)
            return images
        finally:
            if directory >= 0:
                os.close(directory)


__all__ = [
    "ATTACHMENT_SCHEMA",
    "AttachmentStore",
    "PreparedAttachmentBatch",
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
