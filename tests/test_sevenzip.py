"""7z archives: their members listed and read.

A census of real files held a real .7z that listed nothing: sift
named the format and read none of it. A host application builds its member
extraction on sift's readers, so sift parses a 7z's headers -- the
signature header, the next header, plain or encoded (LZMA-compressed, as
7-Zip writes it), the files' names -- and decodes a member of a folder
compressed with LZMA, LZMA2, copy or a branch filter (BCJ) with the
standard library's lzma and raw filters. An encrypted (AES) member or one
compressed with a coder sift does not decode (PPMd, BCJ2) is listed with its
name and said to be unreadable; an encrypted header lists nothing, and says
why.

The archives are assembled here byte by byte (seven_zip), in the layout of
7-Zip's 7zFormat.txt, and the one real archive of the census is read as a
check on the writer: its every member decoded, its CRC matched.
"""
import io
import lzma
import os
import shutil
import stat
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from sift import deadline, zipguard
from sift.detect import HEAD_BYTES, detect
from sift.extract import extract, fallback, sevenzip
from sift.extract.members import MemberError

SIG = b"7z\xbc\xaf\x27\x1c"
#: Real archives to read besides the ones written here: a directory named by
#: SIFT_ARCHIVE_SAMPLES (a host application's viewer census keeps them), searched
#: for *.7z; the tests that read them skip where it is not set.
SAMPLES = os.environ.get("SIFT_ARCHIVE_SAMPLES", "")


def samples(pattern: str) -> list:
    return sorted(Path(SAMPLES).rglob(pattern)) if SAMPLES else []

LZMA1 = b"\x03\x01\x01"
LZMA2 = b"\x21"
COPY = b"\x00"
BCJ = b"\x03\x03\x01\x03"
ARM = b"\x03\x03\x05\x01"
DELTA = b"\x03"
AES = b"\x06\xf1\x07\x01"
PPMD = b"\x03\x04\x01"
DICT = 1 << 16


def num(n: int) -> bytes:
    """7z's NUMBER: a first byte whose leading one bits count the bytes
    after it, the value's high bits in the rest of it."""
    for k in range(8):
        if n < 1 << (7 * (k + 1)):
            first = ((0xFF << (8 - k)) & 0xFF) | (n >> (8 * k))
            return bytes([first]) + (n & ((1 << (8 * k)) - 1)).to_bytes(
                k, "little")
    return b"\xff" + n.to_bytes(8, "little")


def bits(flags) -> bytes:
    out = bytearray((len(flags) + 7) // 8)
    for i, flag in enumerate(flags):
        if flag:
            out[i // 8] |= 0x80 >> (i % 8)
    return bytes(out)


def _lzma1(data: bytes) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=[
        {"id": lzma.FILTER_LZMA1, "dict_size": DICT, "lc": 3, "lp": 0,
         "pb": 2}])


def _raw(data: bytes, filters) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=filters)


LZMA1_PROPS = bytes([0x5D]) + struct.pack("<I", DICT)
LZMA2_PROPS = bytes([8])                # (2 | 0) << (8 // 2 + 11) = 64 KiB


def method(name: str, data: bytes):
    """(coders as 7z lists them -- the one whose output is the folder's
    first --, bind pairs, the packed bytes) for ``name``."""
    lz2 = {"id": lzma.FILTER_LZMA2, "dict_size": DICT}
    lz1 = {"id": lzma.FILTER_LZMA1, "dict_size": DICT, "lc": 3, "lp": 0,
           "pb": 2}
    if name == "copy":
        return [(COPY, b"")], [], data
    if name == "lzma":
        return [(LZMA1, LZMA1_PROPS)], [], _raw(data, [lz1])
    if name == "lzma2":
        return [(LZMA2, LZMA2_PROPS)], [], _raw(data, [lz2])
    if name == "bcj+lzma2":
        return ([(BCJ, b""), (LZMA2, LZMA2_PROPS)], [(0, 1)],
                _raw(data, [{"id": lzma.FILTER_X86}, lz2]))
    if name == "7-zip order":
        # BCJ and LZMA2 as 7-Zip lists them: the coder that reads the
        # packed stream first, the one whose output is the folder's last
        # (the census archive: AES, LZMA, BCJ).
        return ([(LZMA2, LZMA2_PROPS), (BCJ, b"")], [(1, 0)],
                _raw(data, [{"id": lzma.FILTER_X86}, lz2]))
    if name == "arm+lzma2":
        return ([(ARM, b""), (LZMA2, LZMA2_PROPS)], [(0, 1)],
                _raw(data, [{"id": lzma.FILTER_ARM}, lz2]))
    if name == "delta+lzma":
        return ([(DELTA, b"\x03"), (LZMA1, LZMA1_PROPS)], [(0, 1)],
                _raw(data, [{"id": lzma.FILTER_DELTA, "dist": 4}, lz1]))
    if name == "copy+lzma2":
        return ([(COPY, b""), (LZMA2, LZMA2_PROPS)], [(0, 1)],
                _raw(data, [lz2]))
    if name == "aes":
        return [(AES, b"\x13\x1f" + bytes(16))], [], os.urandom(
            max(16, -(-len(data) // 16) * 16))
    if name == "lzma+aes":
        return ([(LZMA1, LZMA1_PROPS), (AES, b"\x13\x1f" + bytes(16))],
                [(0, 1)], os.urandom(max(16, -(-len(data) // 16) * 16)))
    if name == "ppmd":
        return [(PPMD, b"\x06\x00\x00\x00\x01")], [], os.urandom(
            max(1, len(data) // 2))
    raise ValueError(name)


def folder(coders, pairs) -> bytes:
    out = num(len(coders))
    for cid, props in coders:
        out += bytes([len(cid) | (0x20 if props else 0)]) + cid
        if props:
            out += num(len(props)) + props
    for in_index, out_index in pairs:
        out += num(in_index) + num(out_index)
    return out


def streams_info(pack_pos, packs, folders, substreams=None,
                 folder_crcs=None) -> bytes:
    """``folders``: (folder bytes, unpack sizes); ``substreams``: per
    folder a list of (size, crc)."""
    out = (b"\x06" + num(pack_pos) + num(len(packs)) + b"\x09"
           + b"".join(num(len(p)) for p in packs) + b"\x00")
    out += b"\x07\x0b" + num(len(folders)) + b"\x00"
    out += b"".join(f for f, _ in folders) + b"\x0c"
    out += b"".join(num(s) for _, sizes in folders for s in sizes)
    if folder_crcs is not None:
        out += b"\x0a\x01" + b"".join(struct.pack("<I", c)
                                      for c in folder_crcs)
    out += b"\x00"
    if substreams is not None:
        out += b"\x08"
        if any(len(s) != 1 for s in substreams):
            out += b"\x0d" + b"".join(num(len(s)) for s in substreams)
        if any(len(s) > 1 for s in substreams):
            out += b"\x09" + b"".join(num(size) for s in substreams
                                      for size, _ in s[:-1])
        crcs = [c for s in substreams for _, c in s]
        if crcs:
            out += b"\x0a\x01" + b"".join(struct.pack("<I", c) for c in crcs)
        out += b"\x00"
    return out + b"\x00"


def files_info(entries, mtime=None, attributes=None) -> bytes:
    """``entries``: (name, data) -- data None for a directory."""
    n = len(entries)
    empty = [data is None or data == b"" for _, data in entries]
    out = b"\x05" + num(n)
    if any(empty):
        field = bits(empty)
        out += b"\x0e" + num(len(field)) + field
        empty_file = [data == b"" for (_, data), e in zip(entries, empty) if e]
        if any(empty_file):
            field = bits(empty_file)
            out += b"\x0f" + num(len(field)) + field
    names = b"".join(name.encode("utf-16-le") + b"\x00\x00"
                     for name, _ in entries)
    out += b"\x11" + num(len(names) + 1) + b"\x00" + names
    if mtime is not None:
        out += b"\x14" + num(2 + 8 * n) + b"\x01\x00" + struct.pack(
            "<Q", mtime) * n
    if attributes is not None:
        out += b"\x15" + num(2 + 4 * n) + b"\x01\x00" + b"".join(
            struct.pack("<I", attributes(name, data)) for name, data in entries)
    return out + b"\x00"


#: 2026-09-29 12:00:00 UTC as a FILETIME.
FILETIME = (1_790_683_200 + 11_644_473_600) * 10_000_000


def _attributes(name, data):
    return 0x10 if data is None else 0x20


def seven_zip(entries, how="lzma2", solid=True, encode_header=True,
              header_how="lzma", mtime=FILETIME,
              attributes=_attributes) -> bytes:
    """A 7z archive of ``entries`` ((name, bytes), bytes None for a
    directory), its data compressed ``how``, in one folder (``solid``) or
    one per file, its header plain or encoded with ``header_how``."""
    streams = [(name, data) for name, data in entries if data]
    body, packs, folders, substreams = b"", [], [], []
    groups = [streams] if solid and streams else [[s] for s in streams]
    for group in groups:
        joined = b"".join(data for _, data in group)
        coders, pairs, packed = method(how, joined)
        folders.append((folder(coders, pairs), [len(joined)] * len(coders)))
        packs.append(packed)
        body += packed
        substreams.append([(len(d), zlib.crc32(d)) for _, d in group])
    header = b"\x01"
    if folders:
        header += b"\x04" + streams_info(0, packs, folders, substreams)
    header += files_info(entries, mtime, attributes) + b"\x00"
    if encode_header:
        coders, pairs, packed = method(header_how, header)
        encoded = b"\x17" + streams_info(
            len(body), [packed],
            [(folder(coders, pairs), [len(header)] * len(coders))],
            folder_crcs=[zlib.crc32(header)])
        body += packed
        header = encoded
    return wrap(body, header)


def wrap(body: bytes, header: bytes, offset=None) -> bytes:
    start = struct.pack("<QQI", len(body) if offset is None else offset,
                        len(header), zlib.crc32(header))
    return (SIG + b"\x00\x04" + struct.pack("<I", zlib.crc32(start)) + start
            + body + header)


ENTRIES = [("docs", None), ("docs/readme.txt", b"Read me first.\n"),
           ("docs/guide.txt", b"A guide to the archive.\n" * 200),
           ("empty.txt", b""), ("bin/tool.exe", b"MZ\x90\x00" + bytes(300)
                                + b"\xe8\x10\x00\x00\x00" * 50)]


class _SevenZip(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def archive(self, data: bytes, seconds: float = 10):
        with deadline.limit(seconds):
            return sevenzip.SevenZip(io.BytesIO(data))

    def listing(self, data: bytes, name: str = "a.7z", seconds: float = 10):
        path = self.dir / name
        path.write_bytes(data)
        with deadline.limit(seconds):
            return fallback._list_archive(path, data[:HEAD_BYTES])

    def member(self, archive, name):
        return next(m for m in archive.members if m.name == name)

    def read(self, archive, name, **kwargs) -> bytes:
        with archive.open(self.member(archive, name), **kwargs) as fh:
            return fh.read()


class ListingTest(_SevenZip):
    def test_names_sizes_directories_and_empty_files(self):
        for encode in (True, False):
            with self.subTest(encode_header=encode):
                archive = self.archive(seven_zip(ENTRIES,
                                                 encode_header=encode))
                listed = [(m.name, m.size, m.is_dir) for m in archive.members]
                self.assertEqual(
                    [("docs", 0, True), ("docs/readme.txt", 15, False),
                     ("docs/guide.txt", 4_800, False),
                     ("empty.txt", 0, False),
                     ("bin/tool.exe", 554, False)], listed)
                self.assertEqual(5, archive.total)
                self.assertEqual("", archive.note)
                self.assertEqual("", archive.refused)

    def test_the_mtime_and_crc(self):
        archive = self.archive(seven_zip(ENTRIES))
        readme = self.member(archive, "docs/readme.txt")
        self.assertEqual(1_790_683_200, readme.mtime)
        self.assertEqual(zlib.crc32(b"Read me first.\n"), readme.crc)

    def test_a_link_is_known_by_its_unix_mode(self):
        """p7zip keeps a symbolic link's target as its data and its mode in
        the high half of the attributes (0x8000 set)."""
        def attributes(name, data):
            if name == "link":
                return 0x8000 | 0x20 | (0o120777 << 16)
            return 0x8000 | 0x20 | (0o100644 << 16)
        archive = self.archive(seven_zip(
            [("file", b"data"), ("link", b"file")], attributes=attributes))
        self.assertEqual({"file": False, "link": True},
                         {m.name: m.is_link for m in archive.members})

    def test_an_empty_archive(self):
        """7-Zip writes 32 bytes for an archive of nothing."""
        start = struct.pack("<QQI", 0, 0, 0)
        empty = SIG + b"\x00\x04" + struct.pack("<I", zlib.crc32(start)) \
            + start
        archive = self.archive(empty)
        self.assertEqual([], archive.members)
        self.assertEqual(0, archive.total)
        self.assertEqual("", archive.note)
        listing = self.listing(empty)
        self.assertEqual(("7z", 0, []), (listing.format, listing.total,
                                         listing.names))

    def test_an_archive_of_directories_only(self):
        archive = self.archive(seven_zip([("a", None), ("a/b", None)]))
        self.assertEqual([("a", True), ("a/b", True)],
                         [(m.name, m.is_dir) for m in archive.members])

    def test_the_listing_of_sift(self):
        listing = self.listing(seven_zip(ENTRIES))
        self.assertEqual("7z", listing.format)
        self.assertEqual(["docs/", "docs/readme.txt", "docs/guide.txt",
                          "empty.txt", "bin/tool.exe"], listing.names)
        self.assertEqual([0, 15, 4_800, 0, 554], listing.sizes)
        self.assertEqual(5, listing.total)

    def test_extract_names_the_members(self):
        path = self.dir / "bundle.7z"
        path.write_bytes(seven_zip(ENTRIES))
        det = detect(path)
        self.assertEqual("archive", det.kind)
        ex = extract(path, det, 25_000_000)
        self.assertEqual(5, ex.metadata["archive_member_count"])
        self.assertEqual("7z", ex.metadata["archive_format"])
        self.assertIn("docs/readme.txt (15 B)", ex.text)


class ReadingTest(_SevenZip):
    METHODS = ("copy", "lzma", "lzma2", "bcj+lzma2", "7-zip order",
               "arm+lzma2", "delta+lzma", "copy+lzma2")

    def test_every_method_reads_back(self):
        for how in self.METHODS:
            for solid in (True, False):
                with self.subTest(how=how, solid=solid):
                    archive = self.archive(seven_zip(ENTRIES, how, solid))
                    for name, data in ENTRIES:
                        if data is None:
                            continue
                        self.assertEqual(data, self.read(archive, name))

    def test_reading_in_pieces_and_out_of_order(self):
        archive = self.archive(seven_zip(ENTRIES))
        guide = self.member(archive, "docs/guide.txt")
        with archive.open(guide) as fh:
            pieces = [fh.read(1000) for _ in range(6)]
        self.assertEqual([1000] * 4 + [800, 0], [len(p) for p in pieces])
        self.assertEqual(ENTRIES[2][1], self.read(archive, "docs/guide.txt"))
        self.assertEqual(ENTRIES[1][1], self.read(archive, "docs/readme.txt"))
        self.assertEqual(ENTRIES[4][1], self.read(archive, "bin/tool.exe"))

    def test_a_solid_folder_read_in_order_is_decoded_once(self):
        """Members read one after another continue the folder's decoder:
        its packed bytes are read once, not once per member."""
        entries = [(f"f{n}.txt", os.urandom(20_000)) for n in range(20)]
        data = seven_zip(entries)
        fh = _Counting(data)
        archive = sevenzip.SevenZip(fh)
        fh.count = 0
        for name, wanted in entries:
            self.assertEqual(wanted, self.read(archive, name))
        self.assertLess(fh.count, len(data) * 1.5)

    def test_a_directory_is_not_read_and_an_empty_file_is_empty(self):
        archive = self.archive(seven_zip(ENTRIES))
        with self.assertRaises(MemberError):
            archive.open(self.member(archive, "docs"))
        self.assertEqual(b"", self.read(archive, "empty.txt"))

    def test_a_damaged_member_fails_its_checksum(self):
        data = bytearray(seven_zip(ENTRIES, "copy", encode_header=False))
        data[32 + 3] ^= 0xFF                 # the third byte of readme.txt
        archive = self.archive(bytes(data))
        with self.assertRaises(MemberError) as caught:
            self.read(archive, "docs/readme.txt")
        self.assertIn("checksum", str(caught.exception))
        self.assertEqual(ENTRIES[2][1], self.read(archive, "docs/guide.txt"))

    def test_real_archives(self):
        """Real 7z archives, 7-Zip's own writing (a census of a host application's
        viewer: a header encoded with LZMA and no end marker, folders of
        AES, LZMA and BCJ): each listed with names and sizes, every member
        decoded to its CRC or its problem said -- and an encrypted one
        refused, not read."""
        paths = samples("*.7z")
        if not paths:
            self.skipTest("SIFT_ARCHIVE_SAMPLES not set")
        for path in paths:
            with self.subTest(path.name), path.open("rb") as fh:
                with deadline.limit(60):
                    archive = sevenzip.SevenZip(fh)
                self.assertTrue(archive.members, archive.note)
                self.assertEqual("", archive.refused)
                for member in archive.members:
                    if member.is_dir:
                        continue
                    if member.problem:
                        with self.assertRaises(MemberError):
                            archive.open(member)
                        continue
                    with archive.open(member, budget=1 << 30) as reader:
                        got = reader.read()
                    self.assertEqual(member.size, len(got))
                    if member.crc is not None:
                        self.assertEqual(member.crc, zlib.crc32(got))


def unix(mode: int, directory: bool = False) -> int:
    """7z's attributes as p7zip writes them: the Unix mode in the high half,
    0x8000 saying it is there, Windows' archive or directory bit below."""
    return 0x8000 | (0x10 if directory else 0x20) | (mode << 16)


#: A member of each type: its bytes, its Unix mode, the kind sift says.
TYPED = [("file", b"data", 0o100644, "file"),
         ("pipe", b"", stat.S_IFIFO | 0o644, "fifo"),
         ("tty", b"", stat.S_IFCHR | 0o620, "device"),
         ("disk", b"", stat.S_IFBLK | 0o660, "device"),
         ("sock", b"", stat.S_IFSOCK | 0o755, "socket"),
         ("pipe-with-bytes", b"payload", stat.S_IFIFO | 0o644, "fifo"),
         ("link", b"file", stat.S_IFLNK | 0o777, "link")]


class TypeTest(_SevenZip):
    """A member's type as its Unix mode says it, the way a zip's and a
    tar's members say theirs: ``kind`` one of file,
    dir, link, fifo, device, socket or unknown, ``mode`` the Unix mode
    (None where the archive stores none). A host application skips what is not a
    regular file by it; sift's own reader does not open a special file."""

    def typed(self, entries=TYPED, **kwargs):
        modes = {name: mode for name, _, mode, _ in entries}

        def attributes(name, data):
            return unix(modes[name]) if modes[name] is not None else 0x20
        return self.archive(seven_zip([(n, d) for n, d, _, _ in entries],
                                      attributes=attributes, **kwargs))

    def test_the_unix_mode_says_each_type(self):
        for encode in (True, False):
            with self.subTest(encode_header=encode):
                archive = self.typed(encode_header=encode)
                self.assertEqual({n: k for n, _, _, k in TYPED},
                                 {m.name: m.kind for m in archive.members})
                self.assertEqual({n: mode for n, _, mode, _ in TYPED},
                                 {m.name: m.mode for m in archive.members})
                self.assertEqual({"link"}, {m.name for m in archive.members
                                            if m.is_link})
                self.assertFalse(any(m.is_dir for m in archive.members))
                self.assertEqual("", archive.note)

    def test_a_special_file_is_not_opened_and_a_regular_one_reads(self):
        archive = self.typed()
        for name, _, _, kind in TYPED:
            if kind in ("file", "link"):
                continue
            with self.subTest(name):
                with self.assertRaises(MemberError) as caught:
                    archive.open(self.member(archive, name))
                self.assertIn("not a regular file", str(caught.exception))
        self.assertEqual(b"data", self.read(archive, "file"))
        # A link reads as before: its target, the bytes p7zip stores.
        self.assertEqual(b"file", self.read(archive, "link"))

    def test_windows_attributes_and_none_are_a_file(self):
        """7-Zip on Windows stores no Unix mode (0x8000 clear), and an
        archive may store no attributes: a file, as before."""
        for attributes in (_attributes, None):
            with self.subTest(attributes=attributes):
                archive = self.archive(seven_zip(ENTRIES, attributes=attributes))
                kinds = {m.name: m.kind for m in archive.members}
                self.assertEqual({"docs": "dir", "docs/readme.txt": "file",
                                  "docs/guide.txt": "file", "empty.txt": "file",
                                  "bin/tool.exe": "file"}, kinds)
                self.assertEqual({None}, {m.mode for m in archive.members})

    def test_permission_bits_alone_are_a_file(self):
        """A Unix mode without type bits: a regular file, as a host application reads
        a zip's (many writers store only the permission bits)."""
        archive = self.typed([("f", b"x", 0o644, "file")])
        self.assertEqual(("file", 0o644), (archive.members[0].kind,
                                           archive.members[0].mode))
        self.assertEqual(b"x", self.read(archive, "f"))

    def test_type_bits_no_file_has_are_unknown(self):
        """0o170000 and BSD's whiteout (0o160000): no type sift names, so no
        regular file, and not opened."""
        archive = self.typed([("a", b"x", 0o170644, "unknown"),
                              ("b", b"y", 0o160644, "unknown"),
                              ("c", b"z", 0o100644, "file")])
        self.assertEqual({"a": "unknown", "b": "unknown", "c": "file"},
                         {m.name: m.kind for m in archive.members})
        self.assertIn("2 member(s) of a type sift cannot tell", archive.note)
        for name in ("a", "b"):
            with self.assertRaises(MemberError):
                archive.open(self.member(archive, name))
        self.assertEqual(b"z", self.read(archive, "c"))

    def test_a_directory_by_its_type_and_its_bits(self):
        """An entry without a stream and without the empty-file bit was a
        directory; a Unix mode that names another type makes it that type.
        A directory's bit, or a directory's mode, stays a directory; a mode
        that calls an entry with bytes a directory names no type it can
        have."""
        modes = {"pipe": unix(stat.S_IFIFO | 0o644),
                 "plain": 0x20,
                 "flagged": unix(stat.S_IFIFO | 0o644, directory=True),
                 "by-mode": unix(stat.S_IFDIR | 0o755),
                 "odd": unix(stat.S_IFDIR | 0o755)}
        archive = self.archive(seven_zip(
            [("pipe", None), ("plain", None), ("flagged", None),
             ("by-mode", None), ("odd", b"bytes")],
            attributes=lambda name, data: modes[name]))
        self.assertEqual({"pipe": ("fifo", False), "plain": ("dir", True),
                          "flagged": ("dir", True), "by-mode": ("dir", True),
                          "odd": ("unknown", False)},
                         {m.name: (m.kind, m.is_dir) for m in archive.members})

    def test_the_listing_of_sift_is_unchanged(self):
        def attributes(name, data):
            return unix(next(mode for n, _, mode, _ in TYPED if n == name))
        listing = self.listing(seven_zip([(n, d) for n, d, _, _ in TYPED],
                                         attributes=attributes))
        self.assertEqual([n for n, _, _, _ in TYPED], listing.names)
        self.assertEqual("", listing.note)


class _Counting(io.BytesIO):
    """Counts the bytes read of it."""

    count = 0

    def read(self, size=-1):
        data = super().read(size)
        self.count += len(data)
        return data


class UnreadableTest(_SevenZip):
    def test_encrypted_members_are_listed_not_read(self):
        archive = self.archive(seven_zip(ENTRIES, "lzma+aes"))
        self.assertEqual([n for n, _ in ENTRIES],
                         [m.name for m in archive.members])
        tool = self.member(archive, "bin/tool.exe")
        self.assertIn("encrypted", tool.problem)
        with self.assertRaises(MemberError):
            archive.open(tool)
        self.assertIn("encrypted", archive.note)
        self.assertEqual(b"", self.read(archive, "empty.txt"))

    def test_an_encrypted_header_lists_nothing(self):
        archive = self.archive(seven_zip(ENTRIES, "lzma+aes",
                                         header_how="lzma+aes"))
        self.assertEqual([], archive.members)
        self.assertIn("encrypted", archive.note)
        listing = self.listing(seven_zip(ENTRIES, header_how="aes"))
        self.assertEqual([], listing.names)
        self.assertIn("encrypted", listing.note)

    def test_an_unsupported_coder_is_named(self):
        archive = self.archive(seven_zip(ENTRIES, "ppmd"))
        self.assertEqual(5, len(archive.members))
        tool = self.member(archive, "bin/tool.exe")
        self.assertIn("PPMd", tool.problem)
        with self.assertRaises(MemberError):
            archive.open(tool)
        self.assertIn("PPMd", archive.note)

    def test_an_unknown_coder_is_named_by_its_id(self):
        """LZMA2's id (21) made 7f, which names no coder."""
        data = _edited(seven_zip(ENTRIES, "lzma2", encode_header=False),
                       lambda h: h.replace(b"\x21\x21\x01\x08",
                                           b"\x21\x7f\x01\x08", 1))
        archive = self.archive(data)
        self.assertIn("7f", self.member(archive, "bin/tool.exe").problem)


def _edited(data: bytes, edit) -> bytes:
    """``data`` with ``edit`` applied to its next header, the header's CRC
    and the start header's made right again."""
    offset, size = struct.unpack_from("<QQ", data, 12)
    header = edit(data[32 + offset:32 + offset + size])
    return wrap(data[32:32 + offset], header)


class HostileTest(_SevenZip):
    def test_the_entry_count_is_bounded_by_zipguards(self):
        count = zipguard.MAX_ZIP_ENTRIES + 1
        data = seven_zip([(f"f{n}", b"") for n in range(count)],
                         encode_header=False, mtime=None, attributes=None)
        archive = self.archive(data)
        self.assertEqual([], archive.members)
        self.assertEqual(count, archive.total)
        self.assertIn("entry limit", archive.refused)
        listing = self.listing(data)
        self.assertEqual((count, []), (listing.total, listing.names))
        self.assertIn("entry limit", listing.refused)

    def test_a_huge_next_header_is_refused_unread(self):
        start = struct.pack("<QQI", 0, sevenzip.MAX_HEADER_BYTES + 1, 0)
        data = SIG + b"\x00\x04" + struct.pack("<I", zlib.crc32(start)) \
            + start + bytes(64)
        archive = self.archive(data)
        self.assertEqual([], archive.members)
        self.assertIn("header", archive.refused)

    def test_an_encoded_header_declaring_too_much_is_refused(self):
        """The encoded header's unpack size (kCodersUnPackSize, after the
        LZMA coder's properties) made 1 GiB."""
        def edit(header):
            at = header.index(LZMA1_PROPS + b"\x0c") + len(LZMA1_PROPS) + 1
            old = len(num(_unpacked_size(header, at)))
            return header[:at] + num(1 << 30) + header[at + old:]
        archive = self.archive(_edited(seven_zip(ENTRIES), edit))
        self.assertEqual([], archive.members)
        self.assertIn("header", archive.refused)

    def test_a_truncated_archive_says_so(self):
        data = seven_zip(ENTRIES)
        for cut in (31, 40, len(data) // 2, len(data) - 1):
            with self.subTest(cut=cut):
                archive = self.archive(data[:cut])
                self.assertEqual([], archive.members)
                self.assertIn("cut short", archive.note)

    def test_data_past_the_end_is_noted_per_member(self):
        """A header that says the packed data lies further than the file
        reaches: the members listed, their bytes said to be missing."""
        entries = [("a.txt", b"alpha" * 100)]
        # kPackInfo's PackPos (the header begins 01 04 06) made 1 MiB.
        archive = self.archive(_edited(
            seven_zip(entries, "copy", encode_header=False),
            lambda h: h[:3] + num(1 << 20) + h[4:]))
        member = self.member(archive, "a.txt")
        self.assertIn("past", member.problem)
        with self.assertRaises(MemberError):
            archive.open(member)
        self.assertIn("cut short", archive.note)

    def test_a_bad_checksum_is_damage(self):
        data = bytearray(seven_zip(ENTRIES))
        data[8] ^= 1                         # the start header's CRC
        self.assertIn("damaged", self.archive(bytes(data)).note)
        data = bytearray(seven_zip(ENTRIES))
        data[-1] ^= 1                        # the next header itself
        self.assertIn("damaged", self.archive(bytes(data)).note)

    def test_garbage_after_the_magic_is_no_exception(self):
        for tail in (b"", b"\x00" * 40, b"\x00\x04" + os.urandom(300)):
            with self.subTest(len(tail)):
                listing = self.listing(SIG + tail)
                self.assertEqual([], listing.names)
                self.assertTrue(listing.note)

    def test_a_huge_declared_size_is_listed_and_not_decoded(self):
        """A member declaring a terabyte from a few bytes: its size listed as
        declared; reading stops at the budget, nothing of that size held."""
        def edit(header):
            # The copy folder (one coder, id 00) and its unpack size.
            at = header.index(b"\x01\x01\x00\x0c") + 4
            old = len(num(_unpacked_size(header, at)))
            return header[:at] + num(1 << 40) + header[at + old:]
        archive = self.archive(_edited(seven_zip(
            [("zeros.bin", bytes(4096))], "copy", encode_header=False), edit))
        member = self.member(archive, "zeros.bin")
        self.assertEqual(1 << 40, member.size)
        with self.assertRaises(MemberError):
            with archive.open(member, budget=1 << 20) as fh:
                while fh.read(1 << 16):
                    pass

    def test_a_bomb_stops_at_the_expansion_ratio(self):
        """Eight MiB of zeros from about a kilobyte: past the ratio floor,
        more than MAX_RATIO bytes out for each byte in -- refused as
        fallback refuses a single stream."""
        self.assertEqual(fallback.MAX_STREAM_RATIO, sevenzip.MAX_RATIO)
        self.assertEqual(fallback.STREAM_RATIO_FLOOR, sevenzip.RATIO_FLOOR)
        archive = self.archive(seven_zip([("zeros.img", bytes(8 << 20))]))
        member = self.member(archive, "zeros.img")
        with self.assertRaises(MemberError) as caught:
            with archive.open(member) as fh:
                while fh.read(1 << 16):
                    pass
        self.assertIn("expansion", str(caught.exception))

    def test_a_member_past_the_budget_is_refused(self):
        entries = [("a.bin", os.urandom(300_000)), ("b.bin", b"late")]
        archive = self.archive(seven_zip(entries))
        with self.assertRaises(MemberError) as caught:
            self.read(archive, "b.bin", budget=100_000)
        self.assertIn("budget", str(caught.exception))
        self.assertEqual(b"late", self.read(archive, "b.bin"))

    def test_counts_that_would_allocate_are_refused(self):
        """A folder count, a pack-stream count or a substream count of a
        billion in a header of a few bytes: no list of a billion made."""
        for body in (b"\x06\x00" + num(10 ** 9) + b"\x00",
                     b"\x07\x0b" + num(10 ** 9) + b"\x00",
                     b"\x08\x0d" + num(10 ** 9)):
            with self.subTest(body=body[:3]):
                header = b"\x01\x04" + body + b"\x00" * 8
                archive = self.archive(wrap(b"", header), seconds=5)
                self.assertEqual([], archive.members)
                self.assertTrue(archive.note or archive.refused)

    def test_the_bounds_are_derived(self):
        self.assertEqual(zipguard.MAX_ZIP_ENTRIES, sevenzip.MAX_ENTRIES)
        self.assertEqual(zipguard.MAX_ZIP_DIRECTORY_BYTES,
                         sevenzip.MAX_HEADER_BYTES)
        self.assertEqual(fallback.MAX_STREAM_BYTES, sevenzip.MAX_DECODE_BYTES)


def _unpacked_size(header: bytes, at: int) -> int:
    """The NUMBER at ``at``."""
    first, value = header[at], 0
    mask = 0x80
    for i in range(8):
        if not first & mask:
            return value | ((first & (mask - 1)) << (8 * i))
        value |= header[at + 1 + i] << (8 * i)
        mask >>= 1
    return value


if __name__ == "__main__":
    unittest.main()
