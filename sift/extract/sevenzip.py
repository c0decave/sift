"""7z archives: their members listed, and one member's bytes decoded.

A 7z opens with a signature header of 32 bytes: the magic, the format's
version, and where the next header lies (its offset after those 32 bytes,
its size, its CRC). The next header is the archive's directory -- plain
(kHeader) or, as 7-Zip writes it, encoded (kEncodedHeader): a description
of a folder whose decoded bytes are the header. The header describes the
packed streams, the folders that decode them (each a chain of coders), the
substreams each folder's output is cut into, and the files: names, which
of them have no stream (a directory or an empty file), times, attributes.
Members of one folder are "solid": one member's bytes are those after the
members before it in the folder's output.

sift decodes a folder whose coders are copy, LZMA, LZMA2, or a branch
converter (BCJ for x86, and ARM, ARM Thumb, PowerPC, SPARC, IA-64) or delta
before one of them, with the standard library's lzma and raw filters. An
encrypted folder (7-Zip's AES) and one of a coder sift does not decode
(PPMd, BCJ2, Deflate, BZip2, ...) have their members listed with their
names, and their bytes said to be unreadable; an encrypted header lists
nothing, and says so.

Every number is the archive's to choose, so each is bounded before it is
used, with zipguard's bounds as the other archives: at most MAX_ENTRIES
files (more is refused before a name is read, as a zip's end record is
judged), a header of at most MAX_HEADER_BYTES read or decoded, every count
the header declares held to what a list may hold before the list is made,
and the deadline looked at wherever a loop can run long. Decoding a member
is held to ``budget`` bytes of output (default MAX_DECODE_BYTES: the
members before it in its folder count) and to fallback's expansion rule --
past RATIO_FLOOR, no more than MAX_RATIO bytes out for each byte in --, and
the LZMA dictionary to what that output can need. A member's bytes are
checked against its CRC when it has been read to its end.
"""
from __future__ import annotations

import lzma
import stat
import struct
import zlib
from dataclasses import dataclass, field

from .. import deadline
from ..zipguard import MAX_ZIP_DIRECTORY_BYTES, MAX_ZIP_ENTRIES
from .fallback import MAX_STREAM_BYTES, MAX_STREAM_RATIO, STREAM_RATIO_FLOOR
from .members import NOT_READ, MemberError, MemberReader, kind_of, not_read

SIGNATURE = b"7z\xbc\xaf\x27\x1c"
#: Files listed: zipguard's bound on a zip's entries.
MAX_ENTRIES = MAX_ZIP_ENTRIES
#: Bytes of header read, and decoded where it is encoded: zipguard's bound
#: on a zip's central directory.
MAX_HEADER_BYTES = MAX_ZIP_DIRECTORY_BYTES
#: Bytes one open() of a member may decode, the members before it in its
#: folder counted: the bound a tar is walked and a stream decoded under.
MAX_DECODE_BYTES = MAX_STREAM_BYTES
#: fallback's rule for a compressed stream: past RATIO_FLOOR bytes out, no
#: more than MAX_RATIO bytes out for each byte in.
MAX_RATIO = MAX_STREAM_RATIO
RATIO_FLOOR = STREAM_RATIO_FLOOR
#: Coders in one folder and streams of one coder: 7-Zip's own bound.
MAX_CODERS = 64
#: A header encoded in a header encoded in ...: 7-Zip writes one level.
MAX_HEADER_LEVELS = 4
#: What liblzma decodes in one raw chain (LZMA_FILTERS_MAX).
MAX_FILTERS = 4
#: The smallest dictionary liblzma takes (LZMA_DICT_SIZE_MIN).
_MIN_DICTIONARY = 4096

_START = struct.Struct("<QQI")          # next header: offset, size, CRC
_READ = 1 << 16                         # packed bytes read at a time

# Property ids (7zFormat.txt).
K_END, K_HEADER, K_ARCHIVE_PROPERTIES = 0x00, 0x01, 0x02
K_ADDITIONAL_STREAMS, K_MAIN_STREAMS, K_FILES_INFO = 0x03, 0x04, 0x05
K_PACK_INFO, K_UNPACK_INFO, K_SUBSTREAMS_INFO = 0x06, 0x07, 0x08
K_SIZE, K_CRC, K_FOLDER, K_UNPACK_SIZE = 0x09, 0x0A, 0x0B, 0x0C
K_NUM_UNPACK_STREAM, K_EMPTY_STREAM, K_EMPTY_FILE = 0x0D, 0x0E, 0x0F
K_ANTI, K_NAME, K_MTIME, K_ATTRIBUTES = 0x10, 0x11, 0x14, 0x15
K_ENCODED_HEADER, K_DUMMY = 0x17, 0x19

_COPY = b"\x00"
_DELTA = b"\x03"
_LZMA = b"\x03\x01\x01"
_LZMA2 = b"\x21"
_AES = b"\x06\xf1\x07\x01"
_BRANCH = {b"\x03\x03\x01\x03": lzma.FILTER_X86,
           b"\x03\x03\x02\x05": lzma.FILTER_POWERPC,
           b"\x03\x03\x04\x01": lzma.FILTER_IA64,
           b"\x03\x03\x05\x01": lzma.FILTER_ARM,
           b"\x03\x03\x07\x01": lzma.FILTER_ARMTHUMB,
           b"\x03\x03\x08\x05": lzma.FILTER_SPARC}
#: Coders sift names and does not decode.
_NAMED = {b"\x03\x03\x01\x1b": "BCJ2", b"\x03\x04\x01": "PPMd",
          b"\x04\x01\x08": "Deflate", b"\x04\x01\x09": "Deflate64",
          b"\x04\x02\x02": "BZip2", b"\x0a": "ARM64", b"\x0b": "RISC-V",
          b"\x04\xf7\x11\x01": "Zstandard", b"\x04\xf7\x11\x02": "Brotli",
          b"\x04\xf7\x11\x04": "LZ4", b"\x04\xf7\x11\x05": "LZ5",
          b"\x04\xf7\x11\x06": "Lizard"}

#: FILETIME's epoch (1601) before POSIX's, in seconds.
_FILETIME_EPOCH = 11_644_473_600
_ATTRIBUTE_DIRECTORY = 0x10
_ATTRIBUTE_UNIX = 0x8000


class _Damaged(Exception):
    """The header cannot be read as 7z writes one."""


class _Refused(Exception):
    """A bound: the message is the listing's ``refused``; ``total`` the
    entries it declares, where it is their count."""

    def __init__(self, message: str, total: int = 0):
        super().__init__(message)
        self.total = total


class _Encrypted(Exception):
    """The header is encrypted."""


class _Stopped(Exception):
    """What stopped the reading is said already (a note); nothing is
    listed."""


@dataclass
class SevenZipMember:
    """A file or a directory of a 7z. ``name`` is its path as the archive
    holds it; ``size`` its bytes (0 for a directory); ``mtime`` POSIX
    seconds or None; ``crc`` the CRC-32 its bytes must have, or None;
    ``problem`` why its bytes cannot be read, "" where they can.
    ``folder`` and ``offset``: where its bytes lie in the output of which
    folder (None for a member without a stream). ``mode``: the Unix mode
    p7zip stores in the attributes' high half, None where there is none
    (7-Zip on Windows); ``kind`` the member's type (members.KINDS) -- by
    that mode, a file where there is none; "dir" and "link" where
    ``is_dir`` and ``is_link`` say so."""
    name: str
    size: int
    mtime: "int | None" = None
    is_dir: bool = False
    is_link: bool = False
    crc: "int | None" = None
    problem: str = ""
    folder: "int | None" = None
    offset: int = 0
    mode: "int | None" = None
    kind: str = ""

    def __post_init__(self):
        if not self.kind:
            self.kind = ("dir" if self.is_dir else
                         "link" if self.is_link else "file")


@dataclass
class _Coder:
    id: bytes
    ins: int
    outs: int
    props: bytes


@dataclass
class _Folder:
    coders: list
    pairs: list                         # (in index, out index)
    packed: list                        # the in indices packed streams feed
    sizes: list = field(default_factory=list)   # per out stream
    crc: "int | None" = None
    first_pack: int = 0                 # its first packed stream's index
    start: int = 0                      # where its packed stream begins
    pack_size: int = 0

    @property
    def size(self) -> int:
        bound = {out for _, out in self.pairs}
        for index, size in enumerate(self.sizes):
            if index not in bound:
                return size
        return 0


@dataclass
class _Streams:
    pack_pos: int = 0
    pack_sizes: list = field(default_factory=list)
    folders: list = field(default_factory=list)
    # Per folder, its substreams: (size, crc or None).
    substreams: list = field(default_factory=list)


class _Bytes:
    """A header's bytes read forward; _Damaged past their end."""

    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def left(self) -> int:
        return len(self.data) - self.at

    def byte(self) -> int:
        if self.at >= len(self.data):
            raise _Damaged("the header ends early")
        self.at += 1
        return self.data[self.at - 1]

    def take(self, n: int) -> bytes:
        if n < 0 or n > self.left():
            raise _Damaged("the header ends early")
        self.at += n
        return self.data[self.at - n:self.at]

    def number(self) -> int:
        """7z's NUMBER: the first byte's leading one bits count the bytes
        after it; its other bits are the value's high bits."""
        first = self.byte()
        value, mask = 0, 0x80
        for i in range(8):
            if not first & mask:
                return value | ((first & (mask - 1)) << (8 * i))
            value |= self.byte() << (8 * i)
            mask >>= 1
        return value

    def count(self, bound: int, what: str, each: int = 0) -> int:
        """A NUMBER that counts things a list is made of, held before the
        list is made: refused past ``bound``, damage where the header has
        not the ``each`` bytes every one of them takes."""
        n = self.number()
        if n > bound:
            raise _Refused(f"7z {what} limit exceeded: {n} (maximum "
                           f"{bound})", n if what == "entry" else 0)
        if n * each > self.left():
            raise _Damaged(f"the header ends before the {n} {what}s it "
                           f"declares")
        return n

    def uint32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def uint64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def bits(self, n: int) -> list:
        raw = self.take((n + 7) // 8)
        return [bool(raw[i // 8] & (0x80 >> (i % 8))) for i in range(n)]

    def defined(self, n: int) -> list:
        """A vector of which of ``n`` items are defined: "all" in one byte,
        or a bit field."""
        return [True] * n if self.byte() else self.bits(n)

    def expect(self, prop: int) -> None:
        if self.byte() != prop:
            raise _Damaged("a header property is not where 7z puts it")


class SevenZip:
    """The members of the 7z the open, seekable binary file ``fh`` holds,
    read when it is made: ``members`` (in the archive's order, directories
    among them), ``total`` (how many the archive declares), ``refused``
    (the bound that stopped the reading, "" if none), ``note`` (what could
    not be listed or read, in sentences). ``open(member)`` decodes one
    member's bytes, through ``fh``, which must stay open while it does."""

    format = "7z"

    def __init__(self, fh):
        self._fh = fh
        self.members: list = []
        self.total = 0
        self.refused = ""
        self._notes: list = []
        self._folders: list = []
        self._cursor = None             # (folder index, _Unpacked)
        self._generation = 0
        fh.seek(0, 2)
        self._length = fh.tell()
        try:
            self._read()
        except _Refused as exc:
            self.members, self.total, self.refused = [], exc.total, str(exc)
        except _Stopped:
            self.members = []
        except _Encrypted:
            self.members = []
            self._notes = ["the archive's header is encrypted: its members "
                           "are not listed"]
        except _Damaged as exc:
            self.members = []
            self._notes = [f"the archive is damaged: {exc}; nothing is "
                           f"listed"]
        else:
            self.total = len(self.members)
            self._judge()

    @property
    def note(self) -> str:
        return "; ".join(self._notes)

    # ---- headers ----------------------------------------------------------
    def _read(self) -> None:
        self._fh.seek(0)
        head = self._fh.read(32)
        if len(head) < 32:
            self._cut("its signature header is cut short")
            return
        if head[:6] != SIGNATURE:
            raise _Damaged("no 7z signature")
        if head[6] != 0:
            raise _Damaged(f"a format version {head[6]}.{head[7]} sift does "
                           f"not read")
        if zlib.crc32(head[12:32]) != struct.unpack_from("<I", head, 8)[0]:
            raise _Damaged("its start header's checksum does not match")
        offset, size, crc = _START.unpack_from(head, 12)
        if not size:
            return                      # an empty archive: 32 bytes
        if size > MAX_HEADER_BYTES:
            raise _Refused(f"7z header-size limit exceeded: {size} bytes "
                           f"(maximum {MAX_HEADER_BYTES})")
        if 32 + offset + size > self._length:
            self._cut("its header lies past the end of the file")
            return
        self._fh.seek(32 + offset)
        data = self._fh.read(size)
        if zlib.crc32(data) != crc:
            raise _Damaged("its header's checksum does not match")
        header = _Bytes(data)
        prop = header.byte()
        for _ in range(MAX_HEADER_LEVELS):
            if prop != K_ENCODED_HEADER:
                break
            deadline.check("a 7z header")
            header = _Bytes(self._decoded_header(_streams(header)))
            prop = header.byte()
        if prop != K_HEADER:
            raise _Damaged("its header is neither plain nor encoded")
        self._header(header)

    def _cut(self, what: str) -> None:
        self._notes.append(f"the archive is cut short: {what}; nothing is "
                           f"listed")

    def _decoded_header(self, streams: _Streams) -> bytes:
        if not streams.folders:
            raise _Damaged("its encoded header names no folder")
        self._place(streams)
        folder = streams.folders[0]
        filters, problem = _plan(folder)
        if problem == _ENCRYPTED:
            raise _Encrypted
        if problem:
            raise _Damaged(f"its header is {problem}")
        size = folder.size
        if size > MAX_HEADER_BYTES:
            raise _Refused(f"7z header-size limit exceeded: its encoded "
                           f"header declares {size} bytes (maximum "
                           f"{MAX_HEADER_BYTES})")
        if folder.start + folder.pack_size > self._length:
            self._cut("its header lies past the end of the file")
            raise _Stopped
        try:
            data = _Unpacked(self._fh, folder, filters, size).read(size)
        except MemberError as exc:
            raise _Damaged(f"its header cannot be decoded ({exc})") from None
        if len(data) != size:
            raise _Damaged("its encoded header ends early")
        if folder.crc is not None and zlib.crc32(data) != folder.crc:
            raise _Damaged("its header's checksum does not match")
        return data

    def _place(self, streams: _Streams) -> None:
        """Where each folder's packed stream lies, from PackPos and the
        sizes of the packed streams before it."""
        offsets, at = [], 32 + streams.pack_pos
        for size in streams.pack_sizes:
            offsets.append(at)
            at += size
        index = 0
        for folder in streams.folders:
            folder.first_pack = index
            index += len(folder.packed)
            if index > len(streams.pack_sizes):
                raise _Damaged("its folders use more packed streams than "
                               "it has")
            folder.start = offsets[folder.first_pack]
            folder.pack_size = streams.pack_sizes[folder.first_pack]

    def _header(self, header: _Bytes) -> None:
        prop = header.byte()
        if prop == K_ARCHIVE_PROPERTIES:
            while header.byte() != K_END:
                header.take(header.number())
            prop = header.byte()
        if prop == K_ADDITIONAL_STREAMS:
            _streams(header)
            prop = header.byte()
        streams = _Streams()
        if prop == K_MAIN_STREAMS:
            streams = _streams(header)
            self._place(streams)
            prop = header.byte()
        files = []
        if prop == K_FILES_INFO:
            files = _files(header)
            prop = header.byte()
        if prop != K_END:
            raise _Damaged("its header does not end where 7z ends one")
        self._folders = streams.folders
        self._members(files, streams)

    def _members(self, files: list, streams: _Streams) -> None:
        slots = []                      # (folder, offset, size, crc)
        for index, (folder, subs) in enumerate(zip(streams.folders,
                                                   streams.substreams)):
            at = 0
            for size, crc in subs:
                slots.append((index, at, size, crc))
                at += size
        with_streams = sum(1 for f in files if not f["empty"])
        if with_streams != len(slots):
            raise _Damaged(f"it names {with_streams} files with data and "
                           f"holds {len(slots)} data streams")
        plans = [_plan(folder) for folder in streams.folders]
        slot = iter(slots)
        for entry in files:
            deadline.check("a 7z header")
            if entry["anti"]:
                continue                # a deletion marker, no file
            kind = "dir" if entry["dir"] else entry["kind"]
            member = SevenZipMember(entry["name"], 0, entry["mtime"],
                                    is_dir=kind == "dir",
                                    is_link=kind == "link",
                                    mode=entry["mode"], kind=kind)
            if not entry["empty"]:
                folder, offset, size, crc = next(slot)
                member.folder, member.offset = folder, offset
                member.size, member.crc = size, crc
                member.problem = self._problem(streams.folders[folder],
                                               plans[folder][1])
            self.members.append(member)

    def _problem(self, folder: _Folder, problem: str) -> str:
        if problem == _ENCRYPTED:
            return "encrypted (7-Zip's AES): its bytes are not read"
        if problem:
            return f"{problem}: its bytes are not read"
        if folder.start + folder.pack_size > self._length:
            return ("the archive is cut short: the member's packed data "
                    "runs past the end of the file")
        return ""

    def _judge(self) -> None:
        said = {}
        for member in self.members:
            if member.problem:
                key = member.problem.split(":")[0]
                said[key] = said.get(key, 0) + 1
        for problem, count in said.items():
            if problem.startswith("the archive is cut short"):
                self._notes.append(f"the archive is cut short: the packed "
                                   f"data of {count} member(s) runs past the "
                                   f"end of the file")
            else:
                self._notes.append(f"{count} member(s) {problem}: their "
                                   f"bytes are not read")
        unknown = sum(1 for m in self.members if m.kind == "unknown")
        if unknown:
            self._notes.append(f"{unknown} member(s) of a type sift cannot "
                               f"tell (their Unix mode names no type a file "
                               f"has): not read")

    # ---- reading ----------------------------------------------------------
    def open(self, member: SevenZipMember,
             budget: int = MAX_DECODE_BYTES) -> MemberReader:
        """A reader of ``member``'s bytes, decoded as they are read, no more
        than ``budget`` bytes of its folder's output decoded for it -- the
        members before it in the folder count. Members read in the
        archive's order continue one decoder; opening a member ends the
        reading of the one opened before it. MemberError for a directory
        and for a member whose bytes cannot be read (``problem``) or are
        not a file's (a special file: members.NOT_READ)."""
        if member.is_dir:
            raise MemberError("a directory has no bytes to read")
        if member.kind in NOT_READ:
            raise MemberError(not_read(member.kind))
        if member.problem:
            raise MemberError(member.problem)
        self._generation += 1
        if member.folder is None or not member.size:
            return _Member(self, None, member, budget)
        cursor = self._cursor
        if cursor is None or cursor[0] != member.folder \
                or not cursor[1].reaches(member.offset, budget):
            folder = self._folders[member.folder]
            filters, _ = _plan(folder)
            cursor = (member.folder,
                      _Unpacked(self._fh, folder, filters, budget))
            self._cursor = cursor
        return _Member(self, cursor[1], member, budget)


def _streams(header: _Bytes) -> _Streams:
    """A StreamsInfo: PackInfo, the folders (CodersInfo), SubStreamsInfo."""
    streams = _Streams()
    prop = header.byte()
    if prop == K_PACK_INFO:
        streams.pack_pos = header.number()
        n = header.count(4 * MAX_ENTRIES, "packed stream", each=1)
        prop = header.byte()
        if prop == K_SIZE:
            streams.pack_sizes = [header.number() for _ in range(n)]
            prop = header.byte()
        if prop == K_CRC:
            defined = header.defined(n)
            for flag in defined:
                if flag:
                    header.uint32()
            prop = header.byte()
        if prop != K_END:
            raise _Damaged("its pack information does not end")
        if len(streams.pack_sizes) != n:
            raise _Damaged("its packed streams have no sizes")
        prop = header.byte()
    if prop == K_UNPACK_INFO:
        header.expect(K_FOLDER)
        n = header.count(MAX_ENTRIES, "folder", each=3)
        if header.byte():
            raise _Damaged("its folders lie outside the header")
        streams.folders = [_folder(header) for _ in range(n)]
        header.expect(K_UNPACK_SIZE)
        for folder in streams.folders:
            outs = sum(c.outs for c in folder.coders)
            folder.sizes = [header.number() for _ in range(outs)]
        prop = header.byte()
        if prop == K_CRC:
            defined = header.defined(n)
            for folder, flag in zip(streams.folders, defined):
                if flag:
                    folder.crc = header.uint32()
            prop = header.byte()
        if prop != K_END:
            raise _Damaged("its folder information does not end")
        prop = header.byte()
    streams.substreams = [[(f.size, f.crc)] for f in streams.folders]
    if prop == K_SUBSTREAMS_INFO:
        _substreams(header, streams)
        prop = header.byte()
    if prop != K_END:
        raise _Damaged("its stream information does not end")
    return streams


def _folder(header: _Bytes) -> _Folder:
    deadline.check("a 7z header")
    n = header.number()
    if not 1 <= n <= MAX_CODERS:
        raise _Damaged(f"a folder of {n} coders")
    coders = []
    for _ in range(n):
        flags = header.byte()
        if flags & 0xC0:
            raise _Damaged("a coder of alternative methods")
        cid = header.take(flags & 0x0F)
        ins = outs = 1
        if flags & 0x10:
            ins, outs = header.number(), header.number()
            if not (1 <= ins <= MAX_CODERS and 1 <= outs <= MAX_CODERS):
                raise _Damaged("a coder of more streams than 7z writes")
        props = header.take(header.number()) if flags & 0x20 else b""
        coders.append(_Coder(cid, ins, outs, props))
    total_in = sum(c.ins for c in coders)
    total_out = sum(c.outs for c in coders)
    if total_in > MAX_CODERS or total_out > MAX_CODERS:
        raise _Damaged("a folder of more streams than 7z writes")
    pairs = []
    for _ in range(total_out - 1):
        in_index, out_index = header.number(), header.number()
        if in_index >= total_in or out_index >= total_out:
            raise _Damaged("a folder binds a stream it does not have")
        pairs.append((in_index, out_index))
    packed_count = total_in - len(pairs)
    if packed_count < 1:
        raise _Damaged("a folder with no packed stream")
    if packed_count == 1:
        bound = {i for i, _ in pairs}
        packed = [i for i in range(total_in) if i not in bound][:1]
        if not packed:
            raise _Damaged("a folder with no packed stream")
    else:
        packed = [header.number() for _ in range(packed_count)]
    return _Folder(coders, pairs, packed)


def _substreams(header: _Bytes, streams: _Streams) -> None:
    folders = streams.folders
    counts = [1] * len(folders)
    prop = header.byte()
    if prop == K_NUM_UNPACK_STREAM:
        counts = []
        for _ in folders:
            counts.append(header.count(MAX_ENTRIES, "entry"))
            if sum(counts) > MAX_ENTRIES:
                raise _Refused(f"7z entry limit exceeded: more than "
                               f"{MAX_ENTRIES} data streams", sum(counts))
        prop = header.byte()
    sizes = []
    for folder, count in zip(folders, counts):
        if not count:
            sizes.append([])
            continue
        mine = []
        if prop == K_SIZE:
            for _ in range(count - 1):
                mine.append(header.number())
        elif count > 1:
            raise _Damaged("a folder of several files without their sizes")
        rest = folder.size - sum(mine)
        if rest < 0:
            raise _Damaged("a folder's files are larger than the folder")
        sizes.append(mine + [rest])
    if prop == K_SIZE:
        prop = header.byte()
    # A folder of one stream whose CRC the folder gives has it; the digests
    # here are the rest's, in order.
    known = [count == 1 and folder.crc is not None
             for folder, count in zip(folders, counts)]
    unknown = sum(count for count, k in zip(counts, known) if not k)
    crcs = [None] * unknown
    while prop != K_END:
        if prop == K_CRC:
            defined = header.defined(unknown)
            crcs = [header.uint32() if flag else None for flag in defined]
        else:
            header.take(header.number())
        prop = header.byte()
    digests = iter(crcs)
    streams.substreams = []
    for folder, count, mine, k in zip(folders, counts, sizes, known):
        if k:
            streams.substreams.append([(mine[0], folder.crc)])
        else:
            streams.substreams.append([(size, next(digests))
                                       for size in mine])


def _files(header: _Bytes) -> list:
    n = header.number()
    if n > MAX_ENTRIES:
        raise _Refused(f"7z entry limit exceeded: {n} entries (maximum "
                       f"{MAX_ENTRIES})", n)
    empty, empty_file, anti = [False] * n, [], []
    names, mtimes, attributes = None, [None] * n, [None] * n
    while True:
        deadline.check("a 7z header")
        prop = header.number()
        if prop == K_END:
            break
        body = _Bytes(header.take(header.number()))
        if prop == K_EMPTY_STREAM:
            empty = body.bits(n)
        elif prop == K_EMPTY_FILE:
            empty_file = body.bits(sum(empty))
        elif prop == K_ANTI:
            anti = body.bits(sum(empty))
        elif prop == K_NAME:
            if body.byte():
                raise _Damaged("its names lie outside the header")
            text = body.take(body.left()).decode("utf-16-le", "replace")
            names = text.split("\x00")
            if len(names) < n + 1:
                raise _Damaged("it holds fewer names than files")
        elif prop == K_MTIME:
            defined = body.defined(n)
            if body.byte():
                raise _Damaged("its times lie outside the header")
            mtimes = [_posix(body.uint64()) if flag else None
                      for flag in defined]
        elif prop == K_ATTRIBUTES:
            defined = body.defined(n)
            if body.byte():
                raise _Damaged("its attributes lie outside the header")
            attributes = [body.uint32() if flag else None for flag in defined]
    files, at = [], 0
    for index in range(n):
        attribute = attributes[index]
        is_dir = False
        mode = None
        if attribute is not None:
            is_dir = bool(attribute & _ATTRIBUTE_DIRECTORY)
            if attribute & _ATTRIBUTE_UNIX:
                mode = attribute >> 16
        kind = _kind(mode)
        entry = {"name": names[index] if names is not None else f"{index}",
                 "empty": empty[index], "anti": False, "mtime": mtimes[index],
                 "mode": mode, "kind": kind}
        if empty[index]:
            file_flag = empty_file[at] if at < len(empty_file) else False
            entry["anti"] = anti[at] if at < len(anti) else False
            at += 1
            # No stream and no empty-file bit is 7-Zip's directory -- unless
            # the Unix mode names a type that is not one (a FIFO, say).
            entry["dir"] = is_dir or kind == "dir" or (
                not file_flag and not (mode and stat.S_IFMT(mode)))
        else:
            entry["dir"] = False
        if kind == "dir" and not entry["dir"]:
            entry["kind"] = "unknown"   # a directory's mode on bytes
        files.append(entry)
    return files


def _kind(mode: "int | None") -> str:
    """The kind of a member whose Unix mode is ``mode``: a file where there
    is none or it holds the permission bits alone (as a host application reads a
    zip's), else what its type bits name (members.kind_of)."""
    if mode is None or not stat.S_IFMT(mode):
        return "file"
    return kind_of(mode)


def _posix(filetime: int) -> "int | None":
    if not filetime:
        return None
    return filetime // 10_000_000 - _FILETIME_EPOCH


_ENCRYPTED = "encrypted"


def _plan(folder: _Folder):
    """(the raw filter chain that decodes ``folder`` -- [] for copy --,
    "") or (None, why it is not decoded)."""
    for coder in folder.coders:
        if coder.id == _AES:
            return None, _ENCRYPTED
    for coder in folder.coders:
        if coder.id in _NAMED:
            return None, (f"compressed with {_NAMED[coder.id]}, which sift "
                          f"does not decode")
        if coder.id not in (_COPY, _DELTA, _LZMA, _LZMA2) \
                and coder.id not in _BRANCH:
            return None, (f"compressed with a coder sift does not know "
                          f"(id {coder.id.hex() or 'empty'})")
        if coder.ins != 1 or coder.outs != 1:
            return None, "compressed by a folder sift does not decode"
    # Coder k reads in stream k and writes out stream k. From the out
    # stream no pair binds (the folder's output) to the packed stream.
    bound_outs = {out for _, out in folder.pairs}
    feeds = {in_index: out for in_index, out in folder.pairs}
    main = [k for k in range(len(folder.coders)) if k not in bound_outs]
    if len(main) != 1:
        return None, "compressed by a folder sift does not decode"
    chain, k = [], main[0]
    while True:
        if k in chain:
            return None, "compressed by a folder sift does not decode"
        chain.append(k)
        if k in feeds:
            k = feeds[k]
        elif folder.packed == [k]:
            break
        else:
            return None, "compressed by a folder sift does not decode"
    if len(chain) != len(folder.coders):
        return None, "compressed by a folder sift does not decode"
    filters = []
    for k in chain:
        coder = folder.coders[k]
        try:
            one = _filter(coder)
        except (IndexError, ValueError):
            return None, "compressed with coder properties sift cannot read"
        if one is not None:
            filters.append(one)
    compressors = [f for f in filters
                   if f["id"] in (lzma.FILTER_LZMA1, lzma.FILTER_LZMA2)]
    if filters and (len(compressors) != 1 or filters[-1] is not compressors[0]
                    or len(filters) > MAX_FILTERS):
        return None, "compressed by a folder sift does not decode"
    return filters, ""


def _filter(coder: _Coder):
    """The raw filter of one coder, None for copy. Its LZMA dictionary is
    the declared one; _Unpacked holds it to what the decoder may write --
    decoding n bytes never looks further back than n."""
    props = coder.props
    if coder.id == _COPY:
        return None
    if coder.id == _DELTA:
        return {"id": lzma.FILTER_DELTA, "dist": props[0] + 1}
    if coder.id in _BRANCH:
        one = {"id": _BRANCH[coder.id]}
        if len(props) == 4:
            one["start_offset"] = struct.unpack("<I", props)[0]
        elif props:
            raise ValueError("branch converter properties")
        return one
    if coder.id == _LZMA:
        if len(props) != 5 or props[0] >= 9 * 5 * 5:
            raise ValueError("LZMA properties")
        d = props[0]
        return {"id": lzma.FILTER_LZMA1, "lc": d % 9, "lp": d // 9 % 5,
                "pb": d // 45,
                "dict_size": struct.unpack_from("<I", props, 1)[0]}
    if len(props) != 1 or props[0] > 40:
        raise ValueError("LZMA2 properties")
    d = props[0]
    size = 0xFFFFFFFF if d == 40 else (2 | (d & 1)) << (d // 2 + 11)
    return {"id": lzma.FILTER_LZMA2, "dict_size": size}


class _Unpacked:
    """A folder's output, decoded forward from its packed stream: no more
    than ``limit`` bytes of it (with the dictionary held to that), the
    packed bytes read a piece at a time through ``fh`` (sought before each
    read), and fallback's expansion rule looked at as it goes."""

    def __init__(self, fh, folder: _Folder, filters, limit: int):
        self._fh = fh
        self._at = folder.start
        self._left = folder.pack_size
        self.produced = self.consumed = 0
        # A decoder that may write the whole folder can take any member of
        # it; one whose dictionary was cut to ``limit`` only what lies in
        # the first ``limit`` bytes.
        self._whole = True
        self._limit = limit
        chain = []
        for one in filters:
            one = dict(one)
            if "dict_size" in one:
                wanted = min(one["dict_size"], max(folder.size, 1))
                one["dict_size"] = max(_MIN_DICTIONARY, min(wanted, limit))
                if one["dict_size"] < wanted:
                    self._whole = False
            chain.append(one)
        try:
            self._decoder = lzma.LZMADecompressor(
                lzma.FORMAT_RAW, filters=chain) if chain else None
        except (lzma.LZMAError, ValueError) as exc:
            raise MemberError("the folder's coders cannot be set up "
                              "(damaged properties)") from exc
        self._pending = b""

    def reaches(self, offset: int, budget: int) -> bool:
        """Whether reading on from here to ``offset`` and then ``budget``
        bytes stays within what this decoder can write."""
        if self.produced > offset:
            return False
        return self._whole or offset + budget <= self._limit

    def read(self, size: int) -> bytes:
        """Up to ``size`` bytes more of the folder's output; fewer only
        where its packed stream ends."""
        parts, want = [], size
        while want:
            deadline.check("a 7z member")
            if self._decoder is None:
                piece = self._packed(min(want, _READ))
                if not piece:
                    break
            else:
                if self._decoder.eof:
                    break
                if self._decoder.needs_input and not self._pending:
                    self._pending = self._packed(_READ)
                    if not self._pending:
                        break
                try:
                    piece = self._decoder.decompress(self._pending,
                                                     min(want, _READ))
                except lzma.LZMAError as exc:
                    raise MemberError("the member's packed data is "
                                      "damaged") from exc
                self._pending = b""
            self.produced += len(piece)
            want -= len(piece)
            parts.append(piece)
            if self.produced > RATIO_FLOOR \
                    and self.produced > MAX_RATIO * max(self.consumed, 1):
                raise MemberError(
                    f"expansion limit reached: more than {MAX_RATIO:,} "
                    f"bytes out for each byte in")
        return b"".join(parts)

    def _packed(self, size: int) -> bytes:
        size = min(size, self._left)
        if not size:
            return b""
        self._fh.seek(self._at)
        data = self._fh.read(size)
        self._at += len(data)
        self._left -= len(data)
        self.consumed += len(data)
        if len(data) < size:
            self._left = 0
        return data


class _Member(MemberReader):
    """One member's bytes from its folder's decoder: the members before it
    passed over first, no more than ``budget`` bytes decoded for it, its
    CRC checked at its end."""

    def __init__(self, archive: SevenZip, stream, member: SevenZipMember,
                 budget: int):
        self._archive = archive
        self._generation = archive._generation
        self._stream = stream
        self._member = member
        self._left = member.size if stream is not None else 0
        self._budget = self._allowed = budget
        self._crc = 0
        self._skipped = stream is None

    def _read(self, size: int) -> bytes:
        if self._archive._generation != self._generation:
            raise MemberError("another member of the archive was opened "
                              "since: this one is no longer read")
        try:
            return self._decoded(size)
        except MemberError:
            # A decoder that failed is not continued by the next member.
            if self._archive._cursor is not None \
                    and self._archive._cursor[1] is self._stream:
                self._archive._cursor = None
            raise

    def _decoded(self, size: int) -> bytes:
        if not self._skipped:
            self._skip()
        want = min(size, self._left)
        if not want:
            return b""
        if not self._budget:
            raise MemberError(
                f"decode budget reached: reading the member needs more than "
                f"{self._allowed:,} bytes of its folder decoded")
        want = min(want, self._budget)
        data = self._stream.read(want)
        self._budget -= len(data)
        self._left -= len(data)
        if len(data) < want:
            raise MemberError("the member's packed data ends early: the "
                              "archive is cut short or damaged")
        self._crc = zlib.crc32(data, self._crc)
        if not self._left and self._member.crc is not None \
                and self._crc != self._member.crc:
            raise MemberError("the member's checksum does not match: its "
                              "bytes are damaged")
        return data

    def _skip(self) -> None:
        """The folder's output before the member, decoded and passed over."""
        gap = self._member.offset - self._stream.produced
        if gap > self._budget:
            raise MemberError(
                f"decode budget reached: {gap:,} bytes of the members before "
                f"it in its folder would be decoded, more than the budget of "
                f"{self._budget:,}")
        while gap:
            piece = self._stream.read(min(gap, _READ))
            if not piece:
                raise MemberError("the member's packed data ends early: the "
                                  "archive is cut short or damaged")
            gap -= len(piece)
            self._budget -= len(piece)
        self._skipped = True
