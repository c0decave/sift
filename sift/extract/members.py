"""What reading one member of a disc image or a 7z archive raises, the
file-like shape its bytes come in, and the type a member is of (iso9660,
sevenzip).

A host application extracts an archive's members through these readers: it opens a
member, reads it a piece at a time up to its own limit, and hashes what it
read. A reader is a forward-only binary file: read(size) hands out at most
``size`` bytes, b"" at the member's end, and raises MemberError where the
bytes cannot be had -- encrypted, compressed with a coder sift does not
decode, past the end of the file, damaged, or over a bound.
"""
from __future__ import annotations

import stat

#: A member's type, as a zip's (the type bits of its Unix mode) and a tar's
#: (TarInfo.isreg, issym, isfifo, ...) members say theirs: a regular file, a
#: directory, a symbolic link, a named pipe, a character or block device, a
#: socket -- or "unknown", where the archive gives a type no file has or the
#: record that says the type cannot be read.
KINDS = ("file", "dir", "link", "fifo", "device", "socket", "unknown")
#: The kinds whose bytes a reader does not hand out as a file's: a tar's
#: special members are not extracted either (TarFile.extractfile gives
#: None), and a host application skips them as "not a regular file".
NOT_READ = frozenset({"fifo", "device", "socket", "unknown"})
_BY_TYPE = {stat.S_IFREG: "file", stat.S_IFDIR: "dir", stat.S_IFLNK: "link",
            stat.S_IFIFO: "fifo", stat.S_IFCHR: "device",
            stat.S_IFBLK: "device", stat.S_IFSOCK: "socket"}
_WORDS = {"fifo": "a named pipe", "device": "a device",
          "socket": "a socket", "unknown": "of a type sift cannot tell"}


def kind_of(mode: int) -> str:
    """The kind the type bits of the Unix ``mode`` name; "unknown" for bits
    that name no type -- 0 among them: a reader whose format stores the
    permission bits alone says what that means."""
    return _BY_TYPE.get(stat.S_IFMT(mode), "unknown")


def not_read(kind: str) -> str:
    """Why a member of ``kind`` (one of NOT_READ) is not read: a MemberError's
    message."""
    return (f"not a regular file ({_WORDS[kind]}): its bytes are not read "
            f"as a file's")


class MemberError(ValueError):
    """A member's bytes cannot be read; the message says why, in sift's
    words and without the file's."""


class MemberReader:
    """A member's bytes read forward: read(), readable(), a context
    manager. Subclasses give _read(size), size never negative."""

    #: How much read() with no size reads at a time.
    PIECE = 1 << 20

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            parts = []
            while True:
                piece = self._read(self.PIECE)
                if not piece:
                    return b"".join(parts)
                parts.append(piece)
        return self._read(size) if size else b""

    def _read(self, size: int) -> bytes:        # pragma: no cover - abstract
        raise NotImplementedError

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
