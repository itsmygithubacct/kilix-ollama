from __future__ import annotations

import base64
import struct
import zlib


_BASELINE_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8L"
    "CwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUF"
    "BQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh4eHh4eHh7/wAARCAACAAMDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEA"
    "AAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIh"
    "MUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
    "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZ"
    "mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx"
    "8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
    "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAV"
    "YnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hp"
    "anN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPE"
    "xcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDj"
    "/E+mabJ4y8TCTT7RxDr+pQRhoVOyOO7lSNBxwqoqqB0AUAcCiiiv3jh3/kUYX/r3"
    "D/0lFPc//9k=",
    validate=True,
)
_PROGRESSIVE_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8L"
    "CwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUF"
    "BQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh4eHh4eHh7/wgARCAACAAMDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAA"
    "AAAAAAAAAAX/xAAVAQEBAAAAAAAAAAAAAAAAAAABBv/aAAwDAQACEAMQAAABji8f"
    "/8QAFhABAQEAAAAAAAAAAAAAAAAAAwQB/9oACAEBAAEFAqTPbP/EABQRAQAAAAAA"
    "AAAAAAAAAAAAAAD/2gAIAQMBAT8Bf//EABoRAAAHAAAAAAAAAAAAAAAAAAABAwU0"
    "crH/2gAIAQIBAT8BboiVSwf/xAAbEAABBAMAAAAAAAAAAAAAAAABAgMRIQAEQf/a"
    "AAgBAQAGPwLZltNPuAVwKMZ//8QAGRABAAIDAAAAAAAAAAAAAAAAAQAhEXGB/9oA"
    "CAEBAAE/Ic9m/GghoAOT/9oADAMBAAIAAwAAABAD/8QAFhEAAwAAAAAAAAAAAAAA"
    "AAAAAAEx/9oACAEDAQE/EHT/xAAWEQADAAAAAAAAAAAAAAAAAAAAUfD/2gAIAQIB"
    "AT8Qkof/xAAYEAEBAAMAAAAAAAAAAAAAAAABIQARMf/aAAgBAQABPxAwDFI1OyBB"
    "wAEM/9k=",
    validate=True,
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def png_image(width: int = 2, height: int = 2) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x00\x00\x00\xff" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


def jpeg_image(
    width: int = 3,
    height: int = 2,
    *,
    progressive: bool = False,
) -> bytes:
    if (width, height) != (3, 2):
        raise ValueError("the real JPEG fixture has fixed dimensions 3x2")
    return _PROGRESSIVE_JPEG if progressive else _BASELINE_JPEG


__all__ = ["jpeg_image", "png_image"]
