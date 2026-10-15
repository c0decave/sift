"""ISO 9660 disc images: the files they hold listed, and one file's bytes
read.

An image is a file system: volume descriptors from sector 16 on, a root
directory record in the primary one -- and in a Joliet supplementary one,
whose names are UCS-2 --, directories of records, each naming a file's or a
directory's extent (where its bytes lie) and size. sift walks the
directories from the root, depth first in the image's order, and names each
file by its path: Joliet's names where the image has them, else Rock
Ridge's (the NM entries of the primary tree's system use areas), else ISO
9660's own "NAME.EXT;1" without its version. A file of several extents
(a record flagged "not final", then the rest under the same name) is one
file. A UDF image without an ISO 9660 tree is named, not listed.

A file's type is Rock Ridge's to say (PX's mode, SL for a symbolic link):
a regular file, a link, a named pipe, a device or a socket (members.KINDS).
Joliet's records carry no Rock Ridge, so where an image has both -- as
genisoimage -J -R and bsdtar write one -- the primary tree is read for the
types too, under bounds of its own, and each Joliet file takes the type of
the primary tree's file of the same path and extent, else of its extent
alone. A type that cannot be read -- a PX too short for its mode or naming
no type a file has, an entry damaged or a continuation area not read before
PX, a part of the primary tree not read -- is "unknown": not a regular
file, and not read as one, as a special file is not.

Every number is the image's to choose, so each is bounded before it is used
(zipguard's bounds, as the other archives): at most MAX_ENTRIES records
(the walk stops there and the listing is refused past it), at most
MAX_DIRECTORY_BYTES of directory records read in all (a directory declaring
more is refused before it is read), directories at most MAX_DEPTH deep, and
the file's deadline looked at per record. A directory whose extent was
walked already -- a record pointing back at an ancestor, or two directories
pointing at each other -- is listed and not walked again; an extent past the
end of the file is listed with its declared size and said to be past it;
files whose extents overlap a directory's, the descriptors' or another
file's only in part are listed and said to (a file sharing another's extent
whole is a hard link, as genisoimage writes one). Nothing is read by its
declared size: a directory is read up to the bound, a file a piece at a time.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from .. import deadline
from ..zipguard import MAX_ZIP_DIRECTORY_BYTES, MAX_ZIP_ENTRIES
from .members import NOT_READ, MemberError, MemberReader, kind_of, not_read

#: The size of a volume descriptor, and where the first begins (sector 16).
SECTOR = 2048
FIRST_DESCRIPTOR = 16
#: Volume descriptors read before the terminator, and UDF's recognition
#: sequence after it: an image has a handful (primary, Joliet, El Torito's
#: boot record; BEA01, NSR02 or NSR03, TEA01).
MAX_DESCRIPTORS = 64
#: Records listed, directories and files: zipguard's bound on a zip.
MAX_ENTRIES = MAX_ZIP_ENTRIES
#: Bytes of directory records (and Rock Ridge continuation areas) read in
#: all: zipguard's bound on a zip's central directory.
MAX_DIRECTORY_BYTES = MAX_ZIP_DIRECTORY_BYTES
#: How deep the walk goes: a member at depth MAX_DEPTH is listed, a
#: directory there is not walked. ISO 9660 allows 8 levels; Rock Ridge and
#: Joliet images go deeper, and 64 levels of short names are already past
#: Linux's PATH_MAX of 4,096 bytes.
MAX_DEPTH = 64
#: Rock Ridge continuation areas (CE) followed for one record.
MAX_CONTINUATIONS = 8
#: A Joliet supplementary descriptor's escape sequences: UCS-2 levels 1-3.
_JOLIET = (b"%/@", b"%/C", b"%/E")
#: UDF's volume recognition sequence; NSR02 and NSR03 name a UDF volume.
_UDF = (b"BEA01", b"NSR02", b"NSR03", b"TEA01", b"BOOT2", b"CDW02")
_UDF_VOLUME = (b"NSR02", b"NSR03")
#: A directory record's flags.
_DIRECTORY, _ASSOCIATED, _NOT_FINAL = 0x02, 0x04, 0x80
#: The fixed part of a directory record, before its name.
_RECORD = 33
_BOTH32 = struct.Struct("<I")


@dataclass
class IsoMember:
    """A file or a directory of an image. ``name`` is its path, "/" between
    the directories, none at either end; ``size`` its bytes (0 for a
    directory); ``mtime`` the record's date as POSIX seconds, or None;
    ``extents`` (offset in the image, length) in order; ``problem`` why its
    bytes cannot be read, "" where they can. ``mode``: Rock Ridge's PX mode
    of a file, None where there is none (and for a directory); ``kind`` its
    type (members.KINDS) -- a file where the image has no Rock Ridge; "dir"
    and "link" where ``is_dir`` and ``is_link`` say so."""
    name: str
    size: int
    mtime: "int | None" = None
    is_dir: bool = False
    is_link: bool = False
    extents: tuple = ()
    problem: str = ""
    mode: "int | None" = None
    kind: str = ""

    def __post_init__(self):
        if not self.kind:
            self.kind = ("dir" if self.is_dir else
                         "link" if self.is_link else "file")


@dataclass
class _Record:
    lba: int
    offset: int                         # where its data begins, in bytes
    size: int
    flags: int
    name: bytes
    system_use: bytes
    mtime: "int | None"


class _Stop(Exception):
    """A bound of the whole walk is reached; ``IsoImage.refused`` says
    which."""


class _Cut(Exception):
    """A bound of the reading of the primary tree's Rock Ridge (for a
    Joliet image's types) is reached: the types not read are unknown."""


class _Types:
    """What the primary tree's Rock Ridge says of each file's type, for the
    walk of a Joliet tree: (kind, mode) by the file's path and its extent
    (the first record's), and the set of them by the extent alone; ``cut``
    where part of the tree was not read."""

    def __init__(self):
        self.by_path: dict = {}
        self.by_extent: dict = {}
        self.cut = False

    def add(self, path: str, lba: int, kind: str, mode) -> None:
        self.by_path[(path, lba)] = (kind, mode)
        self.by_extent.setdefault(lba, set()).add((kind, mode))

    def of(self, path: str, lba: int):
        """(kind, mode) of the Joliet file at ``path`` whose extent is at
        ``lba``: the primary tree's file of that path and extent; else the
        one type every file of that extent has (Joliet cuts a name at 64
        characters, a writer may name a file otherwise in each tree); else
        "unknown" where files of several types share the extent or the tree
        was not read whole -- a file only where Rock Ridge has no file
        there."""
        found = self.by_path.get((path, lba))
        if found is not None:
            return found
        found = self.by_extent.get(lba, ())
        kinds = {kind for kind, _ in found}
        if len(kinds) == 1:
            modes = {mode for _, mode in found}
            return kinds.pop(), (modes.pop() if len(modes) == 1 else None)
        if kinds or self.cut:
            return "unknown", None
        return "file", None


@dataclass
class _Frame:
    records: object                     # an iterator of _Record
    prefix: str
    depth: int
    pending: list = field(default_factory=list)   # a multi-extent file's


class IsoImage:
    """The files of the ISO 9660 image the open, seekable binary file
    ``fh`` holds, read when it is made: ``members`` (at most MAX_ENTRIES of
    them, directories among them, in the walk's order), ``total`` (how many
    records were seen; one more than MAX_ENTRIES where the walk was
    refused), ``refused`` (the bound that stopped the walk, "" if none),
    ``note`` (what could not be listed or read, in sentences; "" if
    nothing), ``names`` ("joliet", "rock ridge" or "iso9660": whose names
    ``members`` carry). ``open(member)`` reads one file's bytes, through
    ``fh``, which must stay open while it does."""

    format = "iso9660"

    def __init__(self, fh):
        self._fh = fh
        self.members: list = []
        self.total = 0
        self.refused = ""
        self.names = ""
        self._notes: list = []
        self._spent = 0
        self._block = SECTOR
        self._rock_ridge_skip = None
        self._joliet = False
        self._types = None              # a Joliet image's, from Rock Ridge
        self._visited: set = set()
        self._directories: list = []    # (start, end) of each walked
        self._records = 0
        self._descriptor_end = FIRST_DESCRIPTOR * SECTOR
        self._counts = {"loop": 0, "past": 0, "deep": 0, "damaged": 0,
                        "unfinished": 0}
        fh.seek(0, 2)
        self._length = fh.tell()
        try:
            self._read()
        except _Stop:
            pass
        self.total = len(self.members) + (1 if self.refused.startswith(
            "ISO entry limit") else 0)
        self._judge()

    @property
    def note(self) -> str:
        return "; ".join(self._notes)

    # ---- descriptors ------------------------------------------------------
    def _read(self) -> None:
        primary, joliet, udf, count = self._descriptors()
        self._descriptor_end = (FIRST_DESCRIPTOR + count) * SECTOR
        if primary is None:
            if udf:
                self._notes.append("a UDF disc image without an ISO 9660 "
                                   "file system: sift does not list the "
                                   "files inside it")
            else:
                self._notes.append("an ISO 9660 image without a primary "
                                   "volume descriptor: nothing is listed")
            return
        if udf:
            self._notes.append("the image also holds a UDF file system, "
                               "which sift does not read: the files listed "
                               "are its ISO 9660 ones")
        tree = joliet if joliet is not None else primary
        block = struct.unpack_from("<H", tree, 128)[0]
        if block not in (512, 1024, 2048):
            self._notes.append(f"an ISO 9660 image with a logical block of "
                               f"{block} bytes, which no image has: nothing "
                               f"is listed")
            return
        self._block = block
        root = self._record(tree, 156, len(tree))
        if root is None or not root.flags & _DIRECTORY:
            self._notes.append("an ISO 9660 image whose root directory "
                               "record is damaged: nothing is listed")
            return
        self._joliet = joliet is not None
        if self._joliet:
            self._types = self._rock_ridge_types(primary)
        records = self._directory(root, "")
        if records is None:
            return
        if not self._joliet:
            self._rock_ridge_skip = _rock_ridge(records)
        self.names = "joliet" if self._joliet else (
            "rock ridge" if self._rock_ridge_skip is not None else "iso9660")
        self._walk(records)

    def _descriptors(self):
        """(the primary descriptor, the Joliet one, whether a UDF volume
        was recognized, how many sectors of descriptors were read)."""
        primary = joliet = None
        udf = ended = False
        count = 0
        for n in range(MAX_DESCRIPTORS):
            self._fh.seek((FIRST_DESCRIPTOR + n) * SECTOR)
            d = self._fh.read(SECTOR)
            if len(d) < SECTOR:
                break
            ident = d[1:6]
            if ident == b"CD001" and not ended:
                if d[0] == 255:
                    ended = True
                elif d[0] == 1 and primary is None:
                    primary = d
                elif d[0] == 2 and joliet is None and d[88:91] in _JOLIET:
                    joliet = d
            elif ident in _UDF:
                udf = udf or ident in _UDF_VOLUME
            else:
                break
            count = n + 1
        return primary, joliet, udf, count

    # ---- records and directories -----------------------------------------
    def _record(self, buf: bytes, at: int, end: int):
        """The directory record at ``at`` of ``buf`` (which ends at
        ``end``), or None where it is damaged."""
        length = buf[at]
        if length < _RECORD + 1 or at + length > end:
            return None
        name_length = buf[at + 32]
        if not name_length or _RECORD + name_length > length:
            return None
        lba = _BOTH32.unpack_from(buf, at + 2)[0]
        attributes = buf[at + 1]
        use = _RECORD + name_length + (0 if name_length % 2 else 1)
        return _Record(
            lba=lba, offset=(lba + attributes) * self._block,
            size=_BOTH32.unpack_from(buf, at + 10)[0], flags=buf[at + 25],
            name=buf[at + 33:at + 33 + name_length],
            system_use=buf[at + use:at + length] if use < length else b"",
            mtime=_date(buf[at + 18:at + 25]))

    def _directory(self, rec: _Record, path: str):
        """The records of directory ``rec`` (at ``path``), "." and ".."
        among them; None where it is not read, said."""
        self._visited.add(rec.lba)
        if rec.offset + rec.size > self._length:
            self._counts["past"] += 1
            return None
        if self._spent + rec.size > MAX_DIRECTORY_BYTES:
            self.refused = (f"ISO directory-size limit exceeded: more than "
                            f"{MAX_DIRECTORY_BYTES} bytes of directory "
                            f"records")
            raise _Stop
        self._spent += rec.size
        self._fh.seek(rec.offset)
        buf = self._fh.read(rec.size)
        self._directories.append((rec.offset, rec.offset + rec.size))
        records, damaged = self._records_of(buf)
        if damaged:
            self._counts["damaged"] += 1
        return records

    def _records_of(self, buf: bytes):
        """(the records of the directory ``buf``, whether one is damaged:
        the records end there)."""
        records, at = [], 0
        while at < len(buf):
            if not buf[at]:
                at = (at // self._block + 1) * self._block
                continue
            one = self._record(buf, at, len(buf))
            if one is None:
                return records, True
            records.append(one)
            at += buf[at]
        return records, False

    def _walk(self, root_records) -> None:
        stack = [_Frame(iter(root_records), "", 1)]
        while stack:
            frame = stack[-1]
            for rec in frame.records:
                deadline.check("a disc image's directories")
                child = self._entry(frame, rec)
                if child is not None:
                    stack.append(child)
                    break
            else:
                self._finish(frame)
                stack.pop()

    def _entry(self, frame: _Frame, rec: _Record):
        """List ``rec`` of ``frame``'s directory; the frame of a directory
        to walk next, or None."""
        if rec.name in (b"\x00", b"\x01") or rec.flags & _ASSOCIATED:
            return None
        su = self._system_use(rec)
        if su.get("relocated"):
            return None                 # listed where its CL entry is
        name = self._name(rec, su)
        if frame.pending and frame.pending[0][0] != name:
            self._finish(frame)
        self._records += 1
        if self._records > MAX_ENTRIES:
            self._finish(frame)
            self.refused = (f"ISO entry limit exceeded: more than "
                            f"{MAX_ENTRIES} entries")
            raise _Stop
        if su.get("child") is not None:
            # Rock Ridge's relocated directory (CL): a file record standing
            # for the directory rr_moved holds, whose "." record says where
            # it is and how large.
            moved = self._relocated(su["child"])
            if moved is None:
                self._counts["damaged"] += 1
                self._finish(frame)
                self.members.append(IsoMember(frame.prefix + name, 0,
                                              rec.mtime, is_dir=True))
                return None
            rec = replace(moved, name=rec.name, mtime=rec.mtime)
        if rec.flags & _DIRECTORY:
            self._finish(frame)
            path = frame.prefix + name
            self.members.append(IsoMember(path, 0, rec.mtime, is_dir=True))
            if rec.lba in self._visited:
                self._counts["loop"] += 1
                return None
            if frame.depth >= MAX_DEPTH:
                self._counts["deep"] += 1
                return None
            records = self._directory(rec, path)
            if records is None:
                return None
            return _Frame(iter(records), path + "/", frame.depth + 1)
        frame.pending.append((name, rec, su))
        if not rec.flags & _NOT_FINAL:
            self._finish(frame)
        return None

    def _finish(self, frame: _Frame) -> None:
        """The file of ``frame``'s pending records, listed."""
        if not frame.pending:
            return
        name, first, su = frame.pending[0]
        if frame.pending[-1][1].flags & _NOT_FINAL:
            self._counts["unfinished"] += 1
        extents = tuple((rec.offset, rec.size) for _, rec, _ in frame.pending)
        size = sum(length for _, length in extents)
        frame.pending = []
        problem = ""
        # An extent of no bytes is not read, wherever it points (bsdtar
        # points an empty file's, a link's and a device's past the image).
        if any(length and offset + length > self._length
               for offset, length in extents):
            problem = ("the image is cut short: the file runs past its end "
                       "(its size or place is more than the image holds)")
            self._counts["past"] += 1
        path = frame.prefix + name
        if self._types is not None:
            kind, mode = self._types.of(path, first.lba)
        else:
            kind, mode = _kind(su), su.get("mode")
        self.members.append(IsoMember(
            path, size, first.mtime, is_link=kind == "link", extents=extents,
            problem=problem, mode=mode, kind=kind))

    # ---- names ------------------------------------------------------------
    def _name(self, rec: _Record, su: dict) -> str:
        if not self._joliet:
            return _primary_name(rec, su)
        text = _without_version(rec.name.decode("utf-16-be", "replace"))
        return text.replace("/", "_") or "_"

    def _system_use(self, rec: _Record, skip=None) -> dict:
        """What Rock Ridge says of ``rec``, its entries ``skip`` bytes into
        the system use area (None: the image's own skip; {} where the tree
        has no Rock Ridge): its name (NM), its mode (PX, "mode"), whether it
        names a link's target (SL, "link"), is a relocated directory (RE) or
        stands for one (CL) -- and "unread" where an entry that may say its
        type is not read: a PX too short for its mode, an entry damaged
        before the area ends, a continuation area not followed."""
        skip = self._rock_ridge_skip if skip is None else skip
        if skip is None:
            return {}
        found: dict = {}
        name = []
        area = rec.system_use[skip:]
        continuations = 0
        while area is not None:
            follow = None
            at = 0
            while at + 4 <= len(area):
                sig, length = area[at:at + 2], area[at + 2]
                if length < 4 or at + length > len(area):
                    if any(area[at:]):  # zeros are padding, not an entry
                        found["unread"] = True
                    break
                data = area[at + 4:at + length]
                if sig == b"NM" and data:
                    if not data[0] & 0x06:      # not "." or ".."
                        name.append(data[1:])
                elif sig == b"PX":
                    # The mode both byte orders (8 bytes); the first read.
                    if len(data) >= 8:
                        found["mode"] = _BOTH32.unpack_from(data, 0)[0]
                    else:
                        found["unread"] = True
                elif sig == b"SL":
                    found["link"] = True
                elif sig == b"RE":
                    found["relocated"] = True
                elif sig == b"CL" and len(data) >= 4:
                    found["child"] = _BOTH32.unpack_from(data, 0)[0]
                elif sig == b"CE" and len(data) >= 24:
                    follow = (_BOTH32.unpack_from(data, 0)[0],
                              _BOTH32.unpack_from(data, 8)[0],
                              _BOTH32.unpack_from(data, 16)[0])
                elif sig == b"ST":
                    break
                at += length
            area = None
            if follow is not None:
                if continuations < MAX_CONTINUATIONS:
                    continuations += 1
                    area = self._continuation(*follow)
                if area is None:
                    found["unread"] = True
        if name:
            found["name"] = b"".join(name)
        return found

    def _relocated(self, lba: int):
        """The "." record of the directory at ``lba`` -- where a Rock Ridge
        CL entry says a relocated directory is --, or None."""
        at = lba * self._block
        if at + self._block > self._length \
                or self._spent + self._block > MAX_DIRECTORY_BYTES:
            return None
        self._spent += self._block
        self._fh.seek(at)
        first = self._record(self._fh.read(self._block), 0, self._block)
        if first is None or first.name != b"\x00" \
                or not first.flags & _DIRECTORY or first.lba != lba:
            return None
        return first

    def _continuation(self, lba: int, offset: int, length: int):
        """A Rock Ridge continuation area, at most a block, charged to the
        directory bytes; None where it is past the end or the bound."""
        length = min(length, self._block)
        at = lba * self._block + offset
        if at + length > self._length \
                or self._spent + length > MAX_DIRECTORY_BYTES:
            return None
        self._spent += length
        self._fh.seek(at)
        return self._fh.read(length)

    # ---- the types of a Joliet image's files -----------------------------
    def _rock_ridge_types(self, primary: bytes):
        """The types the primary tree's Rock Ridge gives its files (_Types),
        for a Joliet image, whose own records carry none; None where the
        primary tree has no Rock Ridge. Read under bounds of its own and
        counted apart from the listing's: at most MAX_DIRECTORY_BYTES of
        directories and continuation areas, MAX_ENTRIES records, MAX_DEPTH
        levels, a directory once, the deadline per record; where one stops
        the reading, or a directory is damaged or past the end, the rest's
        types are unknown."""
        root = self._record(primary, 156, len(primary))
        if root is None or not root.flags & _DIRECTORY \
                or struct.unpack_from("<H", primary, 128)[0] != self._block:
            return None
        types = _Types()
        try:
            records = self._types_directory(root, types)
            skip = _rock_ridge(records)
            if skip is None:
                return None
            self._types_walk(root.lba, records, skip, types)
        except _Cut:
            types.cut = True
        finally:
            self._spent = 0
        return types

    def _types_walk(self, root_lba: int, root_records, skip: int,
                    types: _Types) -> None:
        visited = {root_lba}
        stack = [(iter(root_records), "", 1)]
        seen = 0
        while stack:
            records, prefix, depth = stack[-1]
            for rec in records:
                deadline.check("a disc image's Rock Ridge")
                if rec.name in (b"\x00", b"\x01") or rec.flags & _ASSOCIATED:
                    continue
                seen += 1
                if seen > MAX_ENTRIES:
                    raise _Cut
                su = self._system_use(rec, skip)
                if su.get("relocated"):
                    continue            # walked where its CL entry is
                name = _primary_name(rec, su)
                if su.get("child") is not None:
                    moved = self._relocated(su["child"])
                    if moved is None:
                        types.cut = True
                        continue
                    rec = replace(moved, name=rec.name)
                if not rec.flags & _DIRECTORY:
                    types.add(prefix + name, rec.lba, _kind(su), su.get("mode"))
                    continue
                if rec.lba in visited:
                    continue
                if depth >= MAX_DEPTH:
                    types.cut = True
                    continue
                visited.add(rec.lba)
                children = self._types_directory(rec, types)
                stack.append((iter(children), prefix + name + "/", depth + 1))
                break
            else:
                stack.pop()

    def _types_directory(self, rec: _Record, types: _Types) -> list:
        """The records of the primary tree's directory ``rec``, charged to
        the directory bytes the types are read under."""
        if rec.offset + rec.size > self._length:
            types.cut = True
            return []
        if self._spent + rec.size > MAX_DIRECTORY_BYTES:
            raise _Cut
        self._spent += rec.size
        self._fh.seek(rec.offset)
        records, damaged = self._records_of(self._fh.read(rec.size))
        types.cut = types.cut or damaged
        return records

    # ---- the whole ------------------------------------------------------
    def _judge(self) -> None:
        counts = self._counts
        if counts["loop"]:
            self._notes.append(
                f"{counts['loop']} director{'y' if counts['loop'] == 1 else 'ies'}"
                f" of the image refer back to a directory already listed (a "
                f"loop): not walked again")
        if counts["deep"]:
            self._notes.append(f"directories deeper than {MAX_DEPTH} levels "
                               f"were not walked")
        if counts["past"]:
            self._notes.append(
                f"the image is cut short: {counts['past']} file(s) or "
                f"directories run past its end")
        if counts["damaged"]:
            self._notes.append(
                f"the image is damaged: {counts['damaged']} director"
                f"{'y' if counts['damaged'] == 1 else 'ies'} ended in a "
                f"record that cannot be read")
        if counts["unfinished"]:
            self._notes.append("the image is damaged: a file of several "
                               "extents ends without its last one")
        overlapping = self._overlaps()
        if overlapping:
            self._notes.append(
                f"{overlapping} file(s) overlap a directory, the volume "
                f"descriptors or another file in part (overlapping extents)")
        unknown = sum(1 for m in self.members if m.kind == "unknown")
        if unknown:
            self._notes.append(
                f"{unknown} file(s) of a type sift cannot tell (Rock Ridge's "
                f"entry for it is damaged, not read, or names no type a file "
                f"has): not read")

    def _overlaps(self) -> int:
        """How many files have an extent that overlaps a directory's, the
        descriptors', or another file's in part; a file whose extent is
        another's whole shares it (a hard link)."""
        spans = [(start, end, None) for start, end in self._directories]
        spans.append((FIRST_DESCRIPTOR * SECTOR, self._descriptor_end, None))
        for index, member in enumerate(self.members):
            if member.is_dir or member.problem:
                continue
            for offset, length in member.extents:
                if length:
                    spans.append((offset, offset + length, index))
        spans.sort(key=lambda s: (s[0], s[1]))
        flagged = set()
        reach = None                    # the span reaching furthest so far
        for span in spans:
            start, end, owner = span
            if start >= end:
                continue
            if reach is not None and start < reach[1]:
                shared = (reach[0], reach[1]) == (start, end) \
                    and owner is not None and reach[2] is not None
                if not shared:
                    for one in (owner, reach[2]):
                        if one is not None:
                            flagged.add(one)
            if reach is None or end > reach[1]:
                reach = span
        return len(flagged)

    # ---- reading ----------------------------------------------------------
    def open(self, member: IsoMember) -> MemberReader:
        """A reader of ``member``'s bytes; MemberError for a directory, for
        a file whose bytes are not a file's (a special file:
        members.NOT_READ) and for a file whose bytes cannot be read
        (``problem``)."""
        if member.is_dir:
            raise MemberError("a directory has no bytes to read")
        if member.kind in NOT_READ:
            raise MemberError(not_read(member.kind))
        if member.problem:
            raise MemberError(member.problem)
        return _Extents(self._fh, member.extents)


class _Extents(MemberReader):
    """A file's extents read in order, a piece at a time, through ``fh``
    (sought before every read: the listing and other readers share it)."""

    def __init__(self, fh, extents):
        self._fh = fh
        self._extents = list(extents)
        self._at = 0                    # into the current extent

    def _read(self, size: int) -> bytes:
        parts, want = [], size
        while want and self._extents:
            deadline.check("a disc image's file")
            offset, length = self._extents[0]
            take = min(want, length - self._at, self.PIECE)
            self._fh.seek(offset + self._at)
            data = self._fh.read(take)
            if len(data) < take:
                raise MemberError("the image is cut short: the file runs "
                                  "past its end")
            parts.append(data)
            want -= take
            self._at += take
            if self._at == length:
                self._extents.pop(0)
                self._at = 0
        return b"".join(parts)


def _rock_ridge(root_records) -> "int | None":
    """How many bytes of each system use area to pass over before Rock
    Ridge's entries (SUSP's SP entry, in the root's "." record), or None
    where the image has no Rock Ridge."""
    if not root_records or root_records[0].name != b"\x00":
        return None
    su = root_records[0].system_use
    if len(su) >= 7 and su[:2] == b"SP" and su[4:6] == b"\xbe\xef":
        return su[6]
    return None


def _primary_name(rec: _Record, su: dict) -> str:
    """The name of a record of the primary tree: Rock Ridge's (NM), else
    ISO 9660's without its version -- one component either way: a slash in
    it would place the file in a directory it is not in."""
    if su.get("name") is not None:
        text = su["name"].decode("utf-8", "replace")
    else:
        text = _without_version(rec.name.decode("utf-8", "replace"))
        if not rec.flags & _DIRECTORY and text.endswith(".") \
                and text.count(".") == 1:
            text = text[:-1]            # "README." is README
    return text.replace("/", "_") or "_"


def _kind(su: dict) -> str:
    """The type Rock Ridge's entries ``su`` (_system_use) give a record that
    is no directory: PX's -- unknown where it names a directory, no type, or
    another type than a link beside SL --, else unknown where an entry that
    may say it was not read, else a link where SL names a target, else a
    file (no Rock Ridge, or none of its entries on this record)."""
    mode = su.get("mode")
    if mode is None:
        if su.get("unread"):
            return "unknown"
        return "link" if su.get("link") else "file"
    kind = kind_of(mode)
    if kind == "dir" or (su.get("link") and kind != "link"):
        return "unknown"
    return kind


def _without_version(name: str) -> str:
    """ISO 9660's "NAME.EXT;1" as NAME.EXT."""
    head, sep, tail = name.rpartition(";")
    return head if sep and tail.isdigit() else name


def _date(raw: bytes) -> "int | None":
    """A directory record's date (years since 1900, month, day, hour,
    minute, second, offset from GMT in quarter hours) as POSIX seconds."""
    if len(raw) < 7 or not raw[1] or not raw[2]:
        return None
    offset = raw[6] - 256 if raw[6] > 127 else raw[6]
    try:
        when = datetime(1900 + raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
                        tzinfo=timezone(timedelta(minutes=15 * offset)))
    except ValueError:
        return None
    return int(when.timestamp())
