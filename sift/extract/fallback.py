"""Extractor of last resort — binaries, archives, unknown formats.

Recovers what a human would look at first: the magic bytes, printable strings,
a hex preview, entropy (a compression/encryption hint) and, for archives, the
member list.
"""
from __future__ import annotations

import bz2
import inspect
import lzma
import math
import re
import struct
import tarfile
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .. import deadline
from ..detect import HEAD_BYTES, Detection, _archive_format, stream_format
from ..zipguard import ZipLimitError, check as check_zip
from . import iso9660
from .base import Extracted, human_size

try:                                    # Python 3.14 and later
    from compression import zstd as _zstd
except ImportError:                     # pragma: no cover - older Python
    _zstd = None

MAX_STRINGS = 120
STRING_RE = re.compile(rb"[\x20-\x7e]{4,}")
MAX_TAR_ENTRIES = 10_000
#: TAR has no central directory. Stop before walking/decompressing an
#: arbitrarily large logical stream merely to count names.
MAX_TAR_SCAN_BYTES = 64 * 1024 * 1024
#: How many members a listing names; the count goes on past it.
MAX_LISTED = 60
#: What tarfile may read of headers for one member: the member's own, a GNU
#: long name or link and pax records before it, an old GNU sparse file's
#: extension blocks, a GNU sparse 1.0 map. tarfile reads each whole, as the
#: header declares it, before sift's bounds are asked: a 400 KB .tar.gz
#: declaring a 400 MiB long name peaked at 837 MiB. A path is
#: at most 4 KiB (PATH_MAX); a pax header -- a path, a link, times, a
#: file's extended attributes -- a few KiB more. A megabyte is far past both.
MAX_TAR_HEADER_BYTES = 1 << 20
#: Global pax records (type "g") hold for every member after them: tarfile
#: keeps them to the end of the archive and copies them into each member.
#: git archive writes one, the commit id.
MAX_TAR_GLOBAL_RECORDS = 64

#: What a single compressed stream (app.log.gz) is decompressed to at most,
#: to learn its size: the budget a tar's members are walked under.
MAX_STREAM_BYTES = MAX_TAR_SCAN_BYTES
#: And how far it may expand, once past STREAM_RATIO_FLOOR: bytes out per
#: byte read. Measured on 101 real streams (a host application's viewer census):
#: documents, logs and source tarballs expand up to 176:1; every one past
#: 1,000:1 was a zero-filled disk image, which is also what a bomb is. The
#: byte budget is what bounds the work -- 64 MiB is under half a second for
#: gzip, xz or zstd and about three for bzip2 -- and this stops a stream
#: that is plainly a bomb before it is spent.
MAX_STREAM_RATIO = 1_000
#: Below this a stream is measured whole, whatever its ratio: a small file
#: of repeated bytes is not a bomb.
STREAM_RATIO_FLOOR = 1 << 20
#: How much of a compress (.Z) file is decoded. There is no stdlib decoder:
#: sift counts the lengths of its LZW codes in Python. Measured: 0.46 s of
#: CPU for 1 MiB of the worst case (noise, whose codes are all short), and
#: the 21 real .Z files of the census -- 73 to 173 KB, 64 and 100 MiB
#: decompressed -- 0.04 to 0.13 s each, every size equal to gzip -dc's.
MAX_LZW_INPUT = 1 << 20

_READ = 1 << 16                         # compressed bytes read at a time
_CHUNK = 1 << 16                        # decompressed bytes asked for at a time

#: What decompressing may raise on damaged input, besides a clean end.
_STREAM_ERRORS = (OSError, EOFError, zlib.error, lzma.LZMAError, ValueError) \
    + ((_zstd.ZstdError,) if _zstd is not None else ())
#: And what tarfile raises on a damaged tar besides its own errors: it
#: indexes a cut sparse extension block (IndexError), and it reads a chain
#: of long-name and pax headers by recursion (RecursionError).
_TAR_ERRORS = (tarfile.TarError, IndexError, RecursionError) + _STREAM_ERRORS
#: tarfile keeps every member it reads, its pax records with it, unless it
#: is told the archive is a stream (stream=True, where TarFile takes it; the
#: listing lets go of them itself where it does not): each is read once.
_TAR_STREAM = ({"stream": True}
               if "stream" in inspect.signature(tarfile.TarFile).parameters
               else {})

#: A compression suffix, and what the name is without it: gunzip's rule
#: (x.tgz becomes x.tar).
_STREAM_SUFFIXES = {".gz": "", ".bz2": "", ".xz": "", ".zst": "", ".zstd": "",
                    ".z": "", ".tgz": ".tar", ".taz": ".tar", ".tbz": ".tar",
                    ".tbz2": ".tar", ".txz": ".tar", ".tzst": ".tar"}


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    if det.kind == "empty":
        return Extracted("empty", det.mime, text="(empty file)",
                         metadata={"size_bytes": 0})

    with path.open("rb") as fh:
        head = fh.read(min(max_bytes, 1_000_000))
    size = path.stat().st_size

    found = [s.decode("latin-1") for s in STRING_RE.findall(head)]
    strings = found[:MAX_STRINGS]
    scanned = min(size, len(head))
    meta = {
        "size_bytes": size,
        "size_human": human_size(size),
        "magic_hex": head[:16].hex(" "),
        # Scoped names: the entropy is of the first 64 KB and the string count
        # is of the bytes we read, not of the file. "120 printable strings" for
        # a file with 500 was the display cap being reported as a measurement.
        "entropy_first_64k": round(_entropy(head[:65536]), 3),
        "printable_strings_found": len(found),
        "printable_strings_shown": len(strings),
        "bytes_scanned": scanned,
        "detected_kind": det.kind,
    }
    if scanned < size:
        meta["scan_note"] = (f"figures cover the first {human_size(scanned)} "
                             f"of {human_size(size)}")

    listing = None
    if det.kind in ("archive", "epub"):
        listing = _list_archive(path, head[:HEAD_BYTES])
        meta["archive_members"] = listing.names
        meta["archive_member_count"] = listing.total
        if listing.format:
            meta["archive_format"] = listing.format
        if listing.sizes:
            meta["archive_member_sizes"] = listing.sizes
        if listing.refused:
            meta["archive_limit"] = listing.refused
        if listing.note:
            meta["archive_note"] = listing.note

    parts = [f"# Binary / {det.kind}",
             f"size: {human_size(size)}  magic: {meta['magic_hex']}  "
             f"entropy (first 64 KB): {meta['entropy_first_64k']}"]
    if meta.get("archive_members"):
        listed = meta["archive_members"]
        sizes = meta.get("archive_member_sizes") or []
        line = "members: " + ", ".join(
            _member(name, sizes[i] if i < len(sizes) else "")
            for i, name in enumerate(listed[:30]))
        total = meta.get("archive_member_count", len(listed))
        if total > len(listed):
            line += f", … and {total - len(listed)} more"
        parts.append(line)
    if meta.get("archive_note"):
        parts.append(f"archive: {meta['archive_note']}")
    if strings:
        parts.append("\n## Printable strings")
        parts.append("\n".join(strings))
    parts.append("\n## Hex preview")
    parts.append(_hexdump(head[:256]))

    ex = Extracted(det.kind, det.mime, text="\n".join(parts), metadata=meta,
                     rendered=True)
    if det.kind in ("binary", "unknown"):
        ex.note("unrecognized format — summary is based on strings and structure only")
    if listing is not None and listing.note:
        ex.note(listing.note)
    return ex


def _member(name: str, size) -> str:
    """A member as the listing line shows it: its size where it is known."""
    if isinstance(size, int) and not isinstance(size, bool):
        return f"{name} ({human_size(size)})"
    if size is None:
        return f"{name} (size not measured)"
    return name


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


@dataclass
class _Listing:
    """What reading an archive's members found: the first MAX_LISTED names
    and their sizes (None where not measured), how many there are, the limit
    that stopped the reading (``refused``), the format, and a sentence on
    what could not be read (``note``)."""
    names: list = field(default_factory=list)
    total: int = 0
    refused: str = ""
    sizes: list = field(default_factory=list)
    format: str = ""
    note: str = ""

    def add(self, name: str, size) -> None:
        self.total += 1
        if len(self.names) < MAX_LISTED:
            self.names.append(name)
            self.sizes.append(size)


def _archive_members(path: Path) -> tuple[list[str], int, str]:
    """Member names, the total count and the limit that stopped the reading,
    for every format _list_archive reads."""
    with Path(path).open("rb") as fh:
        head = fh.read(HEAD_BYTES)
    listing = _list_archive(Path(path), head)
    return listing.names, listing.total, listing.refused


def _list_archive(path: Path, head: bytes) -> _Listing:
    """The members of an archive, whatever the archive is.

    Zip-only listing left every .tar.gz / .tar.bz2 with an empty member list
    and no note — indistinguishable from an empty archive. Zip and tar alone
    did the same to a .gz of a log (tarfile cannot read it as a tar), a .deb,
    a .cab and a .cpio: a single compressed stream is one member now, named
    as gunzip names what it writes, and the rest are read from their headers.
    A 7z and an ISO 9660 image are read by sift's own readers (sevenzip,
    iso9660), which a host application extracts their members through.
    """
    try:
        check_zip(path)
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            listing = _Listing(format="zip")
            for info in infos:
                listing.add(info.filename, info.file_size)
            return listing
    except ZipLimitError as exc:
        return _Listing(total=exc.info.entries, refused=str(exc), format="zip")
    except (zipfile.BadZipFile, OSError):
        pass
    form = _archive_format(head)
    if form == "ar":
        return _ar_listing(path)
    if form == "cpio":
        return _cpio_listing(path)
    if form == "cab":
        return _cab_listing(path)
    if form == "7z":
        return _seven_zip_listing(path)
    if form == "iso9660":
        # Before tarfile, which reads the 32 KiB of zeros an image opens
        # with as an empty tar.
        with path.open("rb") as fh:
            return _read_listing(iso9660.IsoImage(fh))
    stream = stream_format(head)
    # tarfile.open tried every decompressor; zstd's is the one that also
    # reads a file opening with a skippable frame (pzstd writes them).
    tar = _tar_listing(path, stream or (
        "zstd" if head[:4] in _ZSTD_SKIPPABLE else None))
    if tar is not None and (tar.total or tar.refused or not stream):
        # Zeros are an empty tar to tarfile; called one only where the
        # header says so or a member was read.
        if tar.total or tar.refused or form == "tar":
            tar.format = f"tar ({stream})" if stream else "tar"
        return tar
    if stream:
        return _stream_listing(path, stream)
    return _Listing()


def _seven_zip_listing(path: Path) -> _Listing:
    # sevenzip derives its bounds from this module's: imported here, where
    # this module is whole.
    from . import sevenzip
    with path.open("rb") as fh:
        return _read_listing(sevenzip.SevenZip(fh))


def _read_listing(archive) -> _Listing:
    """What one of sift's archive readers (iso9660.IsoImage,
    sevenzip.SevenZip) read, as the other formats report it: the paths, a
    directory's ending in "/" as zipfile names one, a directory's size 0
    and a size the header could not give None; the count, the bound that
    stopped the reading and the note."""
    listing = _Listing(format=archive.format)
    for member in archive.members:
        if member.is_dir:
            listing.add(member.name + "/", 0)
        else:
            listing.add(member.name, member.size)
    listing.total = archive.total
    listing.refused = archive.refused
    listing.note = archive.note
    return listing


def _damaged(count: int) -> str:
    return f"the archive ends early or is damaged after {count} member(s)"


def _tar_listing(path: Path, form):
    """A tar's members -- in a stream compressed as ``form`` (a name
    stream_format gives) or as the file is -- or None where tarfile cannot
    read the file as a tar at all.

    tarfile reads the bytes through _TarBytes, which bounds the headers it
    reads for each member, and a compressed tar through _Inflated rather
    than gzip.GzipFile and its kin, whose loops do not look at the deadline.
    """
    if form == "compress" or (form == "zstd" and _zstd is None):
        return None                     # no reader for either in the stdlib
    listing = _Listing()
    with path.open("rb") as fh:
        view = _TarBytes(_Inflated(fh, form) if form else fh)
        try:
            tf = tarfile.TarFile(fileobj=view, **_TAR_STREAM)
        except _TarLimit as exc:
            listing.refused = str(exc)
            return listing
        except _TAR_ERRORS:
            return None
        try:
            while True:
                deadline.check("an archive member")
                view.budget = MAX_TAR_HEADER_BYTES
                member = tf.next()
                if not _TAR_STREAM:
                    tf.members.clear()
                if member is None:
                    break
                if member.size < 0:
                    listing.note = _damaged(listing.total)
                    break
                if _too_many_global_records(tf.pax_headers):
                    listing.refused = (
                        f"TAR header limit exceeded: more than "
                        f"{MAX_TAR_GLOBAL_RECORDS} global pax records, or more "
                        f"than {MAX_TAR_HEADER_BYTES:,} bytes of them")
                    break
                if listing.total + 1 > MAX_TAR_ENTRIES:
                    listing.total += 1
                    listing.refused = (f"TAR entry limit exceeded: more than "
                                       f"{MAX_TAR_ENTRIES} entries")
                    break
                logical_end = member.offset_data + member.size
                if logical_end > MAX_TAR_SCAN_BYTES:
                    listing.total += 1
                    listing.refused = (
                        f"TAR scan-size limit exceeded: member data ends at "
                        f"{logical_end} bytes (maximum {MAX_TAR_SCAN_BYTES})")
                    break
                listing.add(member.name, member.size)
        except _TarLimit as exc:
            listing.refused = str(exc)
        except _TAR_ERRORS:
            if not listing.total:
                return None
            # The members before the damage were read; a truncated download
            # of a .tar.gz listed none of them.
            listing.note = _damaged(listing.total)
    return listing


def _too_many_global_records(records: dict) -> bool:
    return bool(records) and (
        len(records) > MAX_TAR_GLOBAL_RECORDS
        or sum(len(key) + len(value) for key, value in records.items())
        > MAX_TAR_HEADER_BYTES)


class _TarLimit(Exception):
    """A tar's headers are more than sift reads. Not an OSError, ValueError
    or tarfile error: tarfile makes those "not a tar" or "the archive ends
    here", and this is neither."""


#: _TarBytes' scan limit where its caller names none: MAX_TAR_SCAN_BYTES, as
#: it stands when the reader seeks.
_LISTING_BOUND = object()


class _TarBytes:
    """A tar's bytes as tarfile reads them: forward only, and no more than
    ``budget`` bytes read (it is set again for each member).

    tarfile reads a member's header -- and a long name, pax records or a
    sparse map before it -- as long as the header says; the member's data
    it passes by seeking, which is bounded by ``scan_limit``
    (MAX_TAR_SCAN_BYTES; None for no bound -- a host application walks a tar to its
    own limits and sets ``budget`` to a member's size while it reads the
    member). A read past the budget is refused before it happens.
    """

    def __init__(self, source, scan_limit=_LISTING_BOUND):
        self._source = source
        self._scan_limit = scan_limit
        self.budget = MAX_TAR_HEADER_BYTES

    def read(self, size=-1):
        if size is None or size < 0 or size > self.budget:
            raise _TarLimit(
                f"TAR header limit exceeded: the headers of one member (its "
                f"long name, pax records or sparse map) are more than "
                f"{MAX_TAR_HEADER_BYTES:,} bytes")
        data = self._source.read(size)
        self.budget -= len(data)
        return data

    def seek(self, offset, whence=0):
        if whence != 0 or offset < self._source.tell():
            raise tarfile.ReadError("a tar header points backwards")
        limit = MAX_TAR_SCAN_BYTES if self._scan_limit is _LISTING_BOUND \
            else self._scan_limit
        if limit is not None and offset > limit + tarfile.BLOCKSIZE:
            raise _TarLimit(
                f"TAR scan-size limit exceeded: the next header is at "
                f"{offset} bytes (maximum {limit})")
        self._source.seek(offset)
        return offset

    def tell(self):
        return self._source.tell()


def _stream_member_name(name: str) -> str:
    stem, dot, suffix = name.rpartition(".")
    replacement = _STREAM_SUFFIXES.get("." + suffix.lower())
    if dot and stem and replacement is not None:
        return stem + replacement
    return name


def _stream_listing(path: Path, form: str) -> _Listing:
    """A single compressed stream as the one member it holds: named after
    the file without its compression suffix, its size decompressed under
    MAX_STREAM_BYTES and MAX_STREAM_RATIO, and the file's deadline."""
    name = _stream_member_name(path.name)
    listing = _Listing(format=form)
    if form == "zstd" and _zstd is None:
        listing.add(name, None)
        listing.note = (f"a zstd stream: this Python has no zstd decoder "
                        f"(3.14 and later have one), so the size of {name} "
                        f"is not known")
        return listing
    if form == "compress":
        out, ended, all_zero, trailing = _lzw_size(path)
    else:
        out, ended, all_zero, trailing = _stream_size(path, form)
    if ended == "complete" and all_zero and name.lower().endswith(".tar"):
        # A compressed tar that holds nothing: tarfile read it and found no
        # member, and the stream is all zeros, as an empty tar is.
        return _Listing(format=f"tar ({form})")
    at_least = f"at least {human_size(out)} ({out:,} bytes)"
    if ended == "complete":
        listing.add(name, out)
        if trailing:
            listing.note = (f"{trailing:,} bytes after the stream are not "
                            f"{form} data and were not counted")
        return listing
    listing.add(name, None)
    if ended == "bytes":
        listing.refused = (f"stream size limit reached: more than "
                           f"{MAX_STREAM_BYTES:,} bytes decompressed")
        listing.note = f"{name} is {at_least}; it was not decompressed further"
    elif ended == "ratio":
        listing.refused = (f"stream expansion limit reached: more than "
                           f"{MAX_STREAM_RATIO:,} bytes out for each byte in")
        listing.note = f"{name} is {at_least}; it was not decompressed further"
    elif ended == "input":
        listing.note = (f"{name} is {at_least}: only the first "
                        f"{human_size(MAX_LZW_INPUT)} of the compressed file "
                        f"were decoded")
    else:
        listing.note = (f"the stream is cut short or damaged: {name} is "
                        f"{at_least}, read before it ended")
    return listing


class _GzipMember:
    """zlib's decompressor for one gzip member, with the interface of bz2's,
    lzma's and zstd's: decompress(data, max_length) keeps what it did not
    consume, and says so in needs_input."""

    def __init__(self):
        self._zlib = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def decompress(self, data: bytes, max_length: int) -> bytes:
        tail = self._zlib.unconsumed_tail
        return self._zlib.decompress(tail + data if tail else data,
                                     max_length)

    @property
    def needs_input(self) -> bool:
        return not self._zlib.unconsumed_tail

    @property
    def eof(self) -> bool:
        return self._zlib.eof

    @property
    def unused_data(self) -> bytes:
        return self._zlib.unused_data


def _decompressor(form: str):
    if form == "gzip":
        return _GzipMember()
    if form == "bzip2":
        return bz2.BZ2Decompressor()
    if form == "xz":
        return lzma.LZMADecompressor(lzma.FORMAT_XZ)
    return _zstd.ZstdDecompressor()


#: zstd's skippable frames (a seek table, pzstd's frame sizes) are frames.
_ZSTD_SKIPPABLE = tuple(bytes([0x50 | low, 0x2A, 0x4D, 0x18])
                        for low in range(16))
#: How each stream of a form begins.
_MAGIC = {"gzip": (b"\x1f\x8b",), "bzip2": (b"BZh",), "xz": (b"\xfd7zXZ\x00",),
          "zstd": (b"\x28\xb5\x2f\xfd",) + _ZSTD_SKIPPABLE}


def _stream_size(path: Path, form: str):
    """(bytes out, how it ended, whether they were all zeros, bytes of
    trailing data). It ends "complete", "bytes" or "ratio" (a budget),
    or "damaged" -- which a stream cut short is too."""
    all_zero = True
    with path.open("rb") as fh:
        source = _Inflated(fh, form)
        while True:
            try:
                chunk = source.next_chunk()
            except _STREAM_ERRORS:
                return source.produced, "damaged", all_zero, 0
            if not chunk:
                return source.produced, "complete", all_zero, source.trailing
            if all_zero and chunk.strip(b"\x00"):
                all_zero = False
            out = source.produced
            if out > MAX_STREAM_BYTES:
                return out, "bytes", all_zero, 0
            if out > STREAM_RATIO_FLOOR \
                    and out > MAX_STREAM_RATIO * max(source.consumed, 1):
                return out, "ratio", all_zero, 0


class _Inflated:
    """A compressed file as the bytes it holds, read forward: one stream
    after another, as gunzip reads a file of several (pigz, pbzip2 and xz
    write them), NUL padding between them skipped; anything else after the
    last is not decompressed, and counted (``trailing``). Damage raises what
    the decompressor raises, or EOFError where the file ends inside a
    stream.

    Every step looks at the file's deadline. gzip.GzipFile and its kin,
    which tarfile read a compressed tar through, pass a member's name, NUL
    padding and empty members in loops that do not: 600 MiB of padding
    after a small .gz, or 64 MiB of empty members, ran past a budget of 20
    seconds.
    """

    def __init__(self, fh, form: str):
        self._fh = fh
        self._form = form
        self._magic = _MAGIC[form]
        self._need = max(len(one) for one in self._magic)
        self._stream = _decompressor(form)
        self._pending = b""             # read, not yet decompressed
        self._chunk = b""               # decompressed, not yet handed out
        self._at = 0
        self._ended = False
        self.consumed = 0               # bytes of the file read
        self.produced = 0               # bytes decompressed
        self.position = 0               # bytes handed out or passed over
        self.trailing = 0               # bytes after the last stream

    def next_chunk(self) -> bytes:
        """The next piece of what the file holds; b"" at its end."""
        while not self._ended:
            deadline.check("a compressed stream")
            if self._stream.eof and not self._next_stream():
                self._ended = True
                break
            if self._stream.needs_input and not self._pending:
                self._pending = self._fh.read(_READ)
                self.consumed += len(self._pending)
                if not self._pending:
                    raise EOFError("the file ends inside a compressed stream")
            chunk = self._stream.decompress(self._pending, _CHUNK)
            self._pending = b""
            if chunk:
                self.produced += len(chunk)
                return chunk
        return b""

    def _next_stream(self) -> bool:
        """Past the end of a stream: the next one set up, after any NUL
        padding -- or False where the file ends, or what follows is not a
        stream of this form. Its magic is asked of as many bytes as it has:
        a chunk of the file can end inside it."""
        rest = self._stream.unused_data
        while True:
            rest = rest.lstrip(b"\x00")
            if len(rest) >= self._need:
                break
            deadline.check("a compressed stream")
            more = self._fh.read(_READ)
            self.consumed += len(more)
            if not more:
                break
            rest += more
        if not rest:
            return False
        if not rest.startswith(self._magic):
            self.trailing = len(rest) + _remaining(self._fh)
            return False
        self._stream, self._pending = _decompressor(self._form), rest
        return True

    def read(self, size: int) -> bytes:
        parts = []
        while size > 0 and self._fill():
            piece = self._chunk[self._at:self._at + size]
            self._at += len(piece)
            size -= len(piece)
            parts.append(piece)
        data = b"".join(parts)
        self.position += len(data)
        return data

    def seek(self, offset: int) -> int:
        """Forward to ``offset``, what lies before it decompressed and
        passed over; to the end, where the end comes first."""
        if offset < self.position:
            raise ValueError("a compressed stream is read forward only")
        while self.position < offset and self._fill():
            step = min(offset - self.position, len(self._chunk) - self._at)
            self._at += step
            self.position += step
        return self.position

    def tell(self) -> int:
        return self.position

    def _fill(self) -> bool:
        """Whether there is a byte to hand out, decompressing one if not."""
        if self._at == len(self._chunk):
            self._chunk, self._at = self.next_chunk(), 0
        return self._at < len(self._chunk)


def _remaining(fh) -> int:
    here = fh.tell()
    fh.seek(0, 2)
    return fh.tell() - here


def _lzw_size(path: Path):
    """What _stream_size says, for compress (.Z): how many bytes its LZW
    codes stand for, counted without writing them -- each code's length is
    one more than its prefix's -- in the reader gzip's unlzw is: codes of 9
    bits up to maxbits, read least significant bit first, the codes of one
    width a whole group of that many bytes before the width grows or a
    CLEAR resets it. compress has no end marker: a file cut short reads as
    a shorter one, and only a code no reader could have written says the
    stream is damaged."""
    with path.open("rb") as fh:
        data = fh.read(3 + MAX_LZW_INPUT + 1)
    capped = len(data) > 3 + MAX_LZW_INPUT
    data = data[:3 + MAX_LZW_INPUT]
    flags = data[2] if len(data) > 2 else 0
    maxbits, block = flags & 0x1F, flags & 0x80
    if not 9 <= maxbits <= 16:
        return 0, "damaged", False, 0
    maxmax = 1 << maxbits
    lengths = [1] * 256 + [0] * (maxmax - 255)
    n_bits, maxcode, free = 9, 511, 257 if block else 256
    pos = origin = 24
    end = len(data) * 8
    out = codes = 0
    old = -1
    while end - pos >= n_bits:
        if free > maxcode:
            group = n_bits * 8
            pos = origin + -(-(pos - origin) // group) * group
            origin = pos
            n_bits += 1
            maxcode = maxmax if n_bits == maxbits else (1 << n_bits) - 1
            continue
        at = pos >> 3
        code = (int.from_bytes(data[at:at + 3], "little") >> (pos & 7)) \
            & ((1 << n_bits) - 1)
        pos += n_bits
        codes += 1
        if codes & 0xFFF == 0:
            deadline.check("a compressed stream")
        if old < 0:
            if code >= 256:
                return out, "damaged", False, 0
            out, old = 1, code
            continue
        if code == 256 and block:
            group = n_bits * 8
            pos = origin + -(-(pos - origin) // group) * group
            origin = pos
            n_bits, maxcode, free = 9, 511, 256
            continue
        if code < free:
            out += lengths[code]
        elif code == free:
            out += lengths[old] + 1
        else:
            return out, "damaged", False, 0
        if free < maxmax:
            lengths[free] = lengths[old] + 1
            free += 1
        old = code
        if out > MAX_STREAM_BYTES:
            return out, "bytes", False, 0
        if out > STREAM_RATIO_FLOOR and out > MAX_STREAM_RATIO * (pos >> 3):
            return out, "ratio", False, 0
    return out, "input" if capped else "complete", False, 0


def _ar_listing(path: Path) -> _Listing:
    """An ar archive's members -- a .deb's debian-binary, control.tar.* and
    data.tar.*, a static library's objects -- from their 60-byte headers:
    GNU's long-name table ("//", "/123") and BSD's "#1/<length>" read, the
    symbol tables left out."""
    listing = _Listing(format="ar")
    long_names = b""
    headers = 0
    with path.open("rb") as fh:
        length = _remaining(fh)
        fh.seek(8)
        while True:
            deadline.check("an archive member")
            header = fh.read(60)
            if len(header) < 60 or header[58:60] != b"`\n":
                if header.strip(b"\n") and not listing.note:
                    listing.note = _damaged(listing.total)
                return listing
            # The symbol and name tables are not listed, and count toward
            # the limit all the same: a file of nothing else was read to its
            # end, one header at a time.
            headers += 1
            if headers > MAX_TAR_ENTRIES:
                listing.total += 1
                listing.refused = (f"ar entry limit exceeded: more than "
                                   f"{MAX_TAR_ENTRIES} entries")
                return listing
            size = _number(header[48:58].rstrip(b" "), 10)
            if size is None:
                listing.note = _damaged(listing.total)
                return listing
            start = fh.tell()
            if start + size > length and not listing.note:
                listing.note = ("the archive is cut short: a member runs "
                                "past the end of the file")
            raw = header[:16].rstrip(b" ")
            skip = 0
            if raw == b"//":
                long_names = fh.read(min(size, 1 << 20))
            elif raw.startswith(b"#1/") and raw[3:].isdigit():
                skip = int(raw[3:])     # the name's length, in the data
                if skip > size:
                    listing.note = _damaged(listing.total)
                    return listing
                raw = fh.read(min(skip, 4096))
            elif raw.startswith(b"/") and raw[1:].isdigit():
                offset = int(raw[1:])
                raw = long_names[offset:offset + 4096].split(b"\n", 1)[0]
            # A name ends at its first NUL, as zipfile and tarfile end one.
            raw = raw.split(b"\x00", 1)[0]
            if raw not in (b"/", b"//", b"/SYM64/") \
                    and not raw.startswith(b"__.SYMDEF"):
                name = raw[:-1] if raw.endswith(b"/") else raw
                listing.add(name.decode("utf-8", "replace"), size - skip)
            fh.seek(start + size + (size & 1))


_DIGITS = {8: b"01234567", 10: b"0123456789", 16: b"0123456789abcdefABCDEF"}


def _number(field: bytes, base: int):
    """A header's number as the format writes it: digits of ``base`` and
    nothing else -- or None. int() also takes a sign, blanks, underscores
    and a 0x prefix: a negative size seeked an ar or cpio listing before the
    start of the file, or back onto a header it then read again, or read
    the rest of the file at once."""
    if not field or field.strip(_DIGITS[base]):
        return None
    return int(field, base)


def _cpio_listing(path: Path) -> _Listing:
    """A cpio archive's members, portable ASCII headers ("newc", "crc",
    "odc"), up to the TRAILER!!! entry that ends it. detect matches the
    first header only; each one after it is read as strictly."""
    listing = _Listing(format="cpio")
    with path.open("rb") as fh:
        length = _remaining(fh)
        fh.seek(0)
        while True:
            deadline.check("an archive member")
            start = fh.tell()
            magic = fh.read(6)
            size = namesize = None
            if magic in (b"070701", b"070702"):
                fields, pad = fh.read(104), 4
                if len(fields) == 104:
                    size = _number(fields[48:56], 16)
                    namesize = _number(fields[88:96], 16)
            elif magic == b"070707":
                fields, pad = fh.read(70), 1
                if len(fields) == 70:
                    namesize = _number(fields[53:59], 8)
                    size = _number(fields[59:70], 8)
            # The name holds its NUL at least. No header at all is damage
            # too: an archive ends with its TRAILER!!! entry.
            if size is None or namesize is None or namesize < 1:
                if not listing.note:
                    listing.note = _damaged(listing.total)
                return listing
            raw = fh.read(min(namesize, 4096)).split(b"\x00", 1)[0]
            if raw == b"TRAILER!!!":
                return listing
            if listing.total >= MAX_TAR_ENTRIES:
                listing.total += 1
                listing.refused = (f"cpio entry limit exceeded: more than "
                                   f"{MAX_TAR_ENTRIES} entries")
                return listing
            data = start + len(magic) + len(fields) + namesize
            data += -data % pad
            end = data + size
            if end > length and not listing.note:
                listing.note = ("the archive is cut short: a member runs "
                                "past the end of the file")
            listing.add(raw.decode("utf-8", "replace"), size)
            fh.seek(end + (-end % pad))


def _cab_listing(path: Path) -> _Listing:
    """A Microsoft cabinet's files, from its CFFILE entries: the header says
    where they start and how many there are; each gives a size and a name."""
    listing = _Listing(format="cab")
    with path.open("rb") as fh:
        header = fh.read(36)
        if len(header) < 36:
            listing.note = "the cabinet header is cut short"
            return listing
        first, = struct.unpack_from("<I", header, 16)
        declared, = struct.unpack_from("<H", header, 28)
        if first < len(header):
            # Read on, the header itself was listed as file entries.
            listing.note = ("the cabinet is damaged: its file table would "
                            "begin inside its header")
            return listing
        fh.seek(first)
        for _ in range(declared):
            deadline.check("an archive member")
            entry = fh.read(16)
            if len(entry) < 16:
                break
            size, = struct.unpack_from("<I", entry, 0)
            attributes, = struct.unpack_from("<H", entry, 14)
            at = fh.tell()
            raw = fh.read(257)
            end = raw.find(b"\x00")
            if end < 0:
                break
            fh.seek(at + end + 1)
            if listing.total >= MAX_TAR_ENTRIES:
                listing.total += 1
                listing.refused = (f"cab entry limit exceeded: more than "
                                   f"{MAX_TAR_ENTRIES} entries")
                return listing
            listing.add(raw[:end].decode("utf-8" if attributes & 0x80
                                         else "cp1252", "replace"), size)
    if listing.total < declared:
        listing.note = (f"the cabinet declares {declared} files; "
                        f"{listing.total} could be read")
    return listing


def _hexdump(data: bytes) -> str:
    lines = []
    for off in range(0, len(data), 16):
        chunk = data[off:off + 16]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        text = "".join(chr(b) if 0x20 <= b < 0x7f else "." for b in chunk)
        lines.append(f"{off:08x}  {hexs:<47}  {text}")
    return "\n".join(lines)
