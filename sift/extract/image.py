"""Extractor for images — format/dimensions from headers (stdlib), EXIF via PIL.

The pixels themselves are handed to a vision-capable LLM (llava, llama3.2-vision,
LM Studio VLMs) by the summarizer; here we recover everything we can without one.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional

from ..detect import Detection
from .base import Extracted, human_size

VISION_BYTE_CAP = 8_000_000  # don't base64 enormous images into a prompt
#: The formats a vision server takes. Measured 2026-10-16 on LM Studio with a GGUF model (granite-vision-3.3-2b) and
#: an MLX one (qwen/qwen3-vl-30b), each picture sent with its true type: PNG, JPEG and GIF answered; WEBP, BMP, ICO,
#: TIFF and AVIF got HTTP 400 "'url' field must be a base64 encoded image." on both -- the whole request refused, the
#: analysis failed (every vision model on an AVIF of the host application benchmark). Any other picture is described from
#: its metadata, and the prompt says the model cannot see it.
VISION_FORMATS = frozenset({"PNG", "JPEG", "GIF"})


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    with path.open("rb") as fh:
        data = fh.read(max_bytes) if max_bytes > 0 else fh.read()
    complete = len(data) >= det.size
    fmt, width, height = _dimensions(data)
    mime = _mime(fmt)
    meta = {
        "format": fmt or "unknown",
        "width": width,
        "height": height,
        # The file's size, not the read's: reporting the truncated length as
        # size_bytes would contradict what the directory listing shows.
        "size_bytes": det.size,
        "size_human": human_size(det.size),
    }
    if width is None and height is None and fmt:
        meta["dimensions_note"] = "header truncated or unsupported variant"
    if fmt == "PNG":
        meta.update(_png_text(data))
    meta.update(_pillow_details(path, (width, height)))

    dims = f"{width}×{height}px" if width and height else "unknown dimensions"
    text = (f"# Image ({fmt or 'unknown'})\n"
            f"Dimensions: {dims}\nFile size: {human_size(det.size)}\n"
            f"Filename: {path.name}\n\n"
            "Visual content requires a vision-capable model to describe.")

    ex = Extracted("image", mime, text=text, metadata=meta, rendered=True,
                   truncated=not complete)
    if complete and fmt not in VISION_FORMATS:
        ex.note(f"the picture ({fmt or 'unknown format'}) is not shown to a vision model: the servers decode "
                f"{', '.join(sorted(VISION_FORMATS))} only")
    elif complete and len(data) <= VISION_BYTE_CAP:
        ex.image_data = data
        ex.image_mime = mime
    elif not complete:
        # Half an image decodes to garbage or nothing; a vision model would
        # describe whichever it got. Better to say we could not show it.
        ex.note(f"read only {human_size(len(data))} of {human_size(det.size)}; "
                f"the image was not shown to a vision model")
    else:
        ex.note(f"image too large for vision prompt ({human_size(len(data))})")
    return ex


#: Enough for a Description or a Copyright, short enough that a novel pasted
#: into a tEXt chunk cannot crowd out the rest of the prompt.
_PNG_TEXT_CAP = 2000
#: A PNG carrying more than this many text chunks is not documenting itself.
#: Twenty thousand distinct keywords in a 1 MB file produced 2 GB of resident
#: memory and 40 MB of JSON — the cap on each *value* protected the prompt and
#: nothing else, and `sift serve` accepts 60 MB uploads.
_PNG_MAX_CHUNKS = 64
#: zlib.decompress with no bound turned a 261 KB PNG into 257 MB. The reader
#: only ever keeps _PNG_TEXT_CAP characters, so there is no reason to
#: materialize more than a small multiple of that.
_PNG_MAX_INFLATE = _PNG_TEXT_CAP * 8


def _png_text(data: bytes) -> dict:
    """Title, Author, Description and friends out of a PNG's text chunks.

    Real PNGs carry them — every screenshot tool and camera writes at least
    Software — and they were being dropped, so a summary could not mention
    what the file said about itself. They are also somewhere a payload can sit
    unread: whatever lands here goes through the same injection check as the
    body, because it goes into the same prompt.
    """
    found: dict[str, str] = {}
    seen_chunks = 0
    offset = 8                              # past the signature
    while offset + 8 <= len(data):
        try:
            length = struct.unpack(">I", data[offset:offset + 4])[0]
        except struct.error:
            break
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        if kind == b"IEND" or len(body) < length:
            break
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            seen_chunks += 1
            if seen_chunks > _PNG_MAX_CHUNKS:
                # Counted before the decompression, not after. The cap was on
                # distinct *keywords*, and setdefault discards a repeat only
                # once it has been inflated: 250,000 zTXt chunks sharing one
                # keyword cost five seconds and produced a single entry.
                found["png_text_note"] = (
                    f"more than {_PNG_MAX_CHUNKS} text chunks; the rest were "
                    f"not read")
                return found
            pair = _png_pair(kind, body)
            if pair:
                key, value = pair
                # First one wins: a second chunk with the same keyword must not
                # be able to overwrite what the file already said.
                found.setdefault(f"png_{key}", value[:_PNG_TEXT_CAP])
        offset += 12 + length               # length + type + data + CRC
    return found


def _inflate(blob: bytes) -> bytes:
    """Decompress, but never more than the reader can keep."""
    obj = zlib.decompressobj()
    out = obj.decompress(blob, _PNG_MAX_INFLATE)
    return out


def _png_pair(kind: bytes, body: bytes) -> Optional[tuple]:
    keyword, _, rest = body.partition(b"\x00")
    if not keyword:
        return None
    try:
        if kind == b"zTXt":
            rest = _inflate(rest[1:])                   # skip compression byte
        elif kind == b"iTXt":
            compressed = rest[:1] == b"\x01"
            rest = rest[2:]                             # flag + method
            for _ in range(2):                          # language, translated
                _, _, rest = rest.partition(b"\x00")
            if compressed:
                rest = _inflate(rest)
        key = keyword.decode("latin-1").strip().lower().replace(" ", "_")
        value = rest.decode("utf-8", "replace").strip()
    except (zlib.error, ValueError, UnicodeDecodeError):
        return None
    return (key, value) if key and value else None


def _mime(fmt: Optional[str]) -> str:
    return {
        "PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif",
        "BMP": "image/bmp", "WEBP": "image/webp", "TIFF": "image/tiff",
        "ICO": "image/x-icon", "CUR": "image/x-icon", "HEIF": "image/heif",
        "AVIF": "image/avif",
    }.get(fmt or "", "image/*")


#: The types the data: URI of a picture sent carries (sift.llm.base.image_mime reads them off the bytes).
VISION_MIMES = frozenset(_mime(fmt) for fmt in VISION_FORMATS)


def _dimensions(data: bytes):
    """Format and pixel size. A truncated header yields the format and None.

    Two failure modes were possible before, both worse than "unknown": a short
    WEBP produced a fabricated 1x1 because slicing past the end gives b"" and
    int.from_bytes(b"") + 1 == 1; a short PNG raised out of struct.unpack, and
    the caller then reported no facts at all — not even the file size, and
    without attaching the bytes a vision model could still have looked at.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) < 24:
            return "PNG", None, None
        w, h = struct.unpack(">II", data[16:24])
        return "PNG", w, h
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG", *_jpeg_dims(data)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        if len(data) < 10:
            return "GIF", None, None
        w, h = struct.unpack("<HH", data[6:10])
        return "GIF", w, h
    if data[:2] == b"BM":
        if len(data) < 26:
            return "BMP", None, None
        w, h = struct.unpack("<ii", data[18:26])
        return "BMP", w, abs(h)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP", *_webp_dims(data)
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF", None, None
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return "AVIF", None, None
    if data[4:12] in (b"ftypheic", b"ftypmif1", b"ftypheix"):
        return "HEIF", None, None
    if data[:4] == b"\x00\x00\x01\x00":
        return "ICO", None, None
    if data[:4] == b"\x00\x00\x02\x00":
        return "CUR", None, None
    return None, None, None


def _jpeg_dims(data: bytes):
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None, None


def _webp_dims(data: bytes):
    fourcc = data[12:16]
    try:
        if fourcc == b"VP8 " and len(data) >= 30:
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if fourcc == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if fourcc == b"VP8X" and len(data) >= 30:
            # Slicing past the end returns b"", and int.from_bytes(b"") is 0,
            # so the +1 alone reported a truncated file as a 1x1 image.
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
    except struct.error:
        pass
    return None, None


#: EXIF fields worth reporting. The first five say how the picture was taken;
#: the last four are the only ones that carry prose — a caption, a credit, a
#: licence — and dropping them meant a photo's own description never reached a
#: summary, and a sentence hidden there was read by nobody. They are longer
#: than the rest because a caption is a sentence.
_EXIF_FACTS = ("Make", "Model", "DateTime", "Software", "Orientation")
_EXIF_PROSE = ("ImageDescription", "UserComment", "Artist", "Copyright")


def _pillow_details(path: Path, known: tuple = (None, None)) -> dict:
    try:
        from PIL import Image, ExifTags  # type: ignore
    except ModuleNotFoundError:
        return {}
    out: dict = {}
    try:
        with Image.open(path) as img:
            out["mode"] = img.mode
            # Only where the header parse had no answer. Letting Pillow
            # overwrite it meant the same file reported different dimensions
            # depending on whether an optional package happened to be
            # installed, with nothing saying so.
            if known[0] is None or known[1] is None:
                out["width"], out["height"] = img.size
            exif = getattr(img, "getexif", lambda: {})()
            if exif:
                named = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                for key in _EXIF_FACTS:
                    if key in named:
                        out.setdefault("exif", {})[key] = str(named[key])[:80]
                for key in _EXIF_PROSE:
                    text = _exif_text(named.get(key))
                    if text:
                        out.setdefault("exif", {})[key] = text[:500]
    except Exception:  # noqa: BLE001 - PIL raises many errors on odd files
        pass
    return out


def _exif_text(value) -> str:
    """A caption as text, whatever the camera stored it as.

    UserComment is bytes with an 8-byte character-code prefix, and cameras
    fill it with ASCII, UTF-16 or nothing at all.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        if value[:8] in (b"ASCII\x00\x00\x00", b"UNICODE\x00"):
            value = value[8:]
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                value = value.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            return ""
    return " ".join(str(value).split())
