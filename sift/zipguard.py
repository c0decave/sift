"""Cheap ZIP central-directory limits applied before ``zipfile.ZipFile``.

``ZipFile`` materializes every central-directory entry while opening an
archive. A tiny archive holding hundreds of thousands of empty members can
therefore consume substantial CPU and memory before an extractor sees a
single member. The end record already declares both the entry count and the
directory byte size, and lives in the last 64 KiB, so inspect that first.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_DIRECTORY_BYTES = 16 * 1024 * 1024

_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_EOCD = struct.Struct("<4s4H2LH")
_ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_MAX_COMMENT = 65_535


@dataclass(frozen=True)
class ZipDirectoryInfo:
    entries: int
    size: int


class ZipLimitError(ValueError):
    def __init__(self, info: ZipDirectoryInfo, reason: str):
        super().__init__(reason)
        self.info = info


def directory_info(path: Path) -> Optional[ZipDirectoryInfo]:
    """Read the ZIP/ZIP64 end record without parsing member entries."""
    try:
        with Path(path).open("rb") as fh:
            return directory_info_of(fh)
    except OSError:
        return None


def directory_info_of(fh) -> Optional[ZipDirectoryInfo]:
    """directory_info of an open, seekable binary file -- a zip held in
    memory, such as one inside another archive or attached to a mail."""
    try:
        fh.seek(0, 2)
        length = fh.tell()
        amount = min(length, _EOCD.size + _MAX_COMMENT)
        fh.seek(length - amount)
        tail = fh.read(amount)
        start = length - amount
        position = len(tail)
        record = None
        absolute = None
        while True:
            position = tail.rfind(_EOCD_SIGNATURE, 0, position)
            if position < 0:
                return None
            raw = tail[position:position + _EOCD.size]
            if len(raw) == _EOCD.size:
                candidate = _EOCD.unpack(raw)
                comment_size = candidate[-1]
                if position + _EOCD.size + comment_size == len(tail):
                    record = candidate
                    absolute = start + position
                    break
            if position == 0:
                return None

        entries, size = int(record[4]), int(record[5])
        if entries != 0xFFFF and size != 0xFFFFFFFF:
            return ZipDirectoryInfo(entries, size)

        locator_at = absolute - _ZIP64_LOCATOR.size
        if locator_at < 0:
            return None
        fh.seek(locator_at)
        locator = fh.read(_ZIP64_LOCATOR.size)
        if len(locator) != _ZIP64_LOCATOR.size:
            return None
        signature, _disk, zip64_at, _disks = _ZIP64_LOCATOR.unpack(locator)
        if signature != _ZIP64_LOCATOR_SIGNATURE:
            return None
        fh.seek(zip64_at)
        raw64 = fh.read(_ZIP64_EOCD.size)
        if len(raw64) != _ZIP64_EOCD.size:
            return None
        record64 = _ZIP64_EOCD.unpack(raw64)
        if record64[0] != _ZIP64_EOCD_SIGNATURE:
            return None
        return ZipDirectoryInfo(int(record64[7]), int(record64[8]))
    except (OSError, struct.error, ValueError):
        return None


def check(path: Path) -> Optional[ZipDirectoryInfo]:
    """Return directory facts, or raise before ``ZipFile`` on excess."""
    return _judged(directory_info(path))


def check_file(fh) -> Optional[ZipDirectoryInfo]:
    """check() of an open, seekable binary file."""
    return _judged(directory_info_of(fh))


def _judged(info: Optional[ZipDirectoryInfo]) -> Optional[ZipDirectoryInfo]:
    if info is None:
        return None
    if info.entries > MAX_ZIP_ENTRIES:
        raise ZipLimitError(
            info, f"ZIP entry limit exceeded: {info.entries} entries "
                  f"(maximum {MAX_ZIP_ENTRIES})")
    if info.size > MAX_ZIP_DIRECTORY_BYTES:
        raise ZipLimitError(
            info, f"ZIP directory-size limit exceeded: {info.size} bytes "
                  f"(maximum {MAX_ZIP_DIRECTORY_BYTES})")
    return info
