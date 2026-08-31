from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest import mock
import zlib

from kilix_ollama.attachments import (
    AttachmentStore,
    MAX_ATTACHMENT_BYTES,
    inspect_image,
    validate_attachment_record,
    validate_attachment_records,
)
from kilix_ollama.errors import ConversationError
from kilix_ollama.paths import RuntimePaths
from tests.image_fixtures import jpeg_image, png_image


def private_paths(root: Path) -> RuntimePaths:
    return RuntimePaths.from_environment(
        {
            "HOME": str(root / "home"),
            "XDG_RUNTIME_DIR": str(root / "run"),
            "XDG_STATE_HOME": str(root / "state"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
        },
        uid=os.getuid(),
    )


def padded_png(size: int) -> bytes:
    image = png_image()
    kind = b"raNd"
    padding = b"x" * size
    chunk = (
        struct.pack(">I", len(padding))
        + kind
        + padding
        + struct.pack(">I", zlib.crc32(kind + padding) & 0xFFFFFFFF)
    )
    return image[:33] + chunk + image[33:]


def png_with_raw_rows(rows: bytes) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class AttachmentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = private_paths(self.root)
        self.store = AttachmentStore(self.paths)

    def selected(self, name: str, payload: bytes) -> Path:
        path = self.root / name
        path.write_bytes(payload)
        return path

    def test_png_and_jpeg_dimensions_are_bounded_and_structurally_checked(self) -> None:
        self.assertEqual(inspect_image(png_image(3, 2)), ("image/png", 3, 2))
        self.assertEqual(inspect_image(jpeg_image()), ("image/jpeg", 3, 2))
        self.assertEqual(
            inspect_image(jpeg_image(progressive=True)),
            ("image/jpeg", 3, 2),
        )

        corrupt = bytearray(png_image())
        corrupt[-1] ^= 1
        oversized = bytearray(png_image())
        oversized[16:24] = struct.pack(">II", 16_384, 16_384)
        oversized[29:33] = struct.pack(
            ">I", zlib.crc32(bytes(oversized[12:29])) & 0xFFFFFFFF
        )
        for payload in (
            bytes(corrupt),
            bytes(oversized),
            b"GIF89a",
            jpeg_image()[:-2],
            b"\xff\xd8\xff\xc0\x00\x08\x08\x00\x02\x00\x02\x01\xff\xda\x00\x02\xff\xd9",
            b"\xff\xd8" + b"\xff\x01" * 4097 + b"\xff\xd9",
            png_with_raw_rows(b"\x00" * 100),
            png_with_raw_rows(b"\x05" + b"\x00" * 16 + b"\x00" + b"\x00" * 16),
        ):
            with self.subTest(size=len(payload)), self.assertRaises(ConversationError):
                inspect_image(payload)

    def test_snapshot_is_private_content_addressed_path_free_and_deduplicated(self) -> None:
        payload = png_image(3, 2)
        source = self.selected("PRIVATE-NAME.png", payload)
        previous_umask = os.umask(0o777)
        try:
            record = self.store.snapshot(source)
        finally:
            os.umask(previous_umask)
        digest = hashlib.sha256(payload).hexdigest()
        target = self.store.root / digest

        self.assertEqual(record["digest"], f"sha256:{digest}")
        self.assertEqual(record["media_type"], "image/png")
        self.assertEqual((record["width"], record["height"]), (3, 2))
        self.assertNotIn("PRIVATE-NAME", str(record))
        self.assertNotIn(str(source), str(record))
        self.assertEqual(self.store.read(record), payload)
        self.assertEqual(stat.S_IMODE(self.store.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

        repeated = self.store.snapshot(source)
        self.assertEqual(repeated, record)
        self.assertEqual(tuple(path.name for path in self.store.root.iterdir()), (digest,))

    def test_invalid_sources_and_over_total_batch_publish_nothing(self) -> None:
        source = self.selected("source.png", png_image())
        symlink = self.root / "link.png"
        symlink.symlink_to(source)
        real_directory = self.root / "real-directory"
        real_directory.mkdir()
        (real_directory / "image.png").write_bytes(png_image())
        intermediate_symlink = self.root / "linked-directory"
        intermediate_symlink.symlink_to(real_directory, target_is_directory=True)
        large = self.root / "large.png"
        with large.open("wb") as stream:
            stream.truncate(MAX_ATTACHMENT_BYTES + 1)
        fifo = self.root / "fifo.png"
        os.mkfifo(fifo)
        for path in (
            symlink,
            intermediate_symlink / "image.png",
            self.root,
            fifo,
            large,
            self.selected("text", b"not image"),
        ):
            with self.subTest(path=path.name), self.assertRaises(ConversationError):
                self.store.snapshot(path)

        first = self.selected("first.png", padded_png(3 * 1024 * 1024))
        second = self.selected("second.png", padded_png(3 * 1024 * 1024 + 1))
        with self.assertRaises(ConversationError):
            self.store.snapshot_many((first, second))
        self.assertFalse(self.store.root.exists())

    def test_source_mutation_and_no_clobber_race_fail_without_substitution(self) -> None:
        payload = png_image()
        source = self.selected("mutable.png", payload)
        real_read = os.read
        reads = 0

        def mutate_after_read(descriptor: int, count: int) -> bytes:
            nonlocal reads
            block = real_read(descriptor, count)
            reads += 1
            if reads == 1:
                with source.open("ab") as stream:
                    stream.write(b"x")
            return block

        with mock.patch(
            "kilix_ollama.attachments.os.read",
            side_effect=mutate_after_read,
        ), self.assertRaises(ConversationError):
            self.store.snapshot(source)
        self.assertFalse(self.store.root.exists())
        source.write_bytes(payload)

        digest = hashlib.sha256(payload).hexdigest()
        target = self.store.root / digest
        real_link = os.link

        def inject_target(
            source_path,
            destination,
            *,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=True,
        ):
            target.write_bytes(b"concurrent owner data")
            target.chmod(0o600)
            return real_link(
                source_path,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        with mock.patch(
            "kilix_ollama.attachments.os.link",
            side_effect=inject_target,
        ), self.assertRaises(ConversationError):
            self.store.snapshot(source)
        self.assertEqual(target.read_bytes(), b"concurrent owner data")
        self.assertEqual(tuple(self.store.root.glob("*.tmp")), ())

    def test_snapshot_root_component_swap_never_redirects_publication(self) -> None:
        payload = png_image()
        source = self.selected("source.png", payload)
        digest = hashlib.sha256(payload).hexdigest()
        data_dir = self.paths.data_dir
        moved_data = self.root / "moved-data"
        outside = self.root / "outside"
        outside.mkdir(mode=0o700)
        real_link = os.link
        swapped = False

        def swap_component(
            source_name,
            destination_name,
            *,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=True,
        ):
            nonlocal swapped
            if not swapped:
                data_dir.rename(moved_data)
                data_dir.symlink_to(outside, target_is_directory=True)
                swapped = True
            return real_link(
                source_name,
                destination_name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        with mock.patch(
            "kilix_ollama.attachments.os.link",
            side_effect=swap_component,
        ), self.assertRaises(ConversationError):
            self.store.snapshot(source)
        self.assertTrue(swapped)
        self.assertEqual(tuple(outside.iterdir()), ())
        self.assertEqual(
            (moved_data / "attachments" / digest).read_bytes(),
            payload,
        )

    def test_tampered_snapshot_and_record_shapes_fail_closed(self) -> None:
        payload = jpeg_image()
        record = self.store.snapshot(self.selected("image.jpg", payload))
        target = self.store.root / str(record["digest"]).removeprefix("sha256:")
        target.chmod(0o644)
        with self.assertRaises(ConversationError):
            self.store.read(record)
        target.chmod(0o600)
        target.write_bytes(b"x" * len(payload))
        with self.assertRaises(ConversationError):
            self.store.read(record)

        invalid = (
            {**record, "extra": True},
            {**record, "digest": "sha256:" + "A" * 64},
            {**record, "media_type": "image/gif"},
            {**record, "width": True},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ConversationError):
                validate_attachment_record(value)
        with self.assertRaises(ConversationError):
            validate_attachment_records([record, record])


if __name__ == "__main__":
    unittest.main()
