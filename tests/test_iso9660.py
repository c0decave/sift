"""ISO 9660 disc images: their files listed and read.

A census of real files held a real .iso that listed nothing: sift
said "an ISO 9660 or UDF disc image: sift does not list the files inside
it". A host application builds its member extraction on sift's readers, so sift lists
an image's files -- Joliet's names where the image has them, Rock Ridge's
where it has no Joliet, else ISO 9660's own -- and reads one file's bytes
where its extents say, standard library only.

Every number in an image is the image's to choose: a directory that points
back at one above it, an extent past the end of the file, a size of 4 GiB in
a file of 100 KB, ten thousand and one records. Each is refused or noted,
never an endless walk and never the declared size in memory.

The images are written here, record by record (_Image), as genisoimage
writes them: volume descriptors from sector 16, directories of records that
do not cross a sector, "." and ".." first, a file's version ";1".
"""
import io
import os
import shutil
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sift import deadline, zipguard
from sift.detect import HEAD_BYTES, detect
from sift.extract import extract, fallback, iso9660
from sift.extract.members import MemberError

BLOCK = 2048
#: Real archives to read besides the ones written here: a directory named by
#: SIFT_ARCHIVE_SAMPLES (a host application's viewer census keeps them), searched
#: for *.iso; the tests that read them skip where it is not set.
SAMPLES = os.environ.get("SIFT_ARCHIVE_SAMPLES", "")


def samples(pattern: str) -> list:
    return sorted(Path(SAMPLES).rglob(pattern)) if SAMPLES else []


def _both16(n: int) -> bytes:
    return struct.pack("<H", n) + struct.pack(">H", n)


def _both32(n: int) -> bytes:
    return struct.pack("<I", n) + struct.pack(">I", n)


#: 2026-09-29 12:00:00 at UTC+2 (eight quarter hours).
DATE = bytes([126, 9, 29, 12, 0, 0, 8])
DATE_POSIX = 1_790_676_000


def record(name: bytes, extent: int, size: int, flags: int = 0,
           su: bytes = b"", date: bytes = DATE) -> bytes:
    """One directory record: its length, the extended attribute length, the
    extent and size (both byte orders), the date, the flags, the name and
    its padding, the system use area."""
    body = (bytes([0]) + _both32(extent) + _both32(size) + date
            + bytes([flags, 0, 0]) + _both16(1) + bytes([len(name)]) + name)
    if len(name) % 2 == 0:
        body += b"\x00"
    body += su
    if (1 + len(body)) % 2:
        body += b"\x00"
    return bytes([1 + len(body)]) + body


def directory(records) -> bytes:
    """Records packed into sectors, none crossing one, the rest zeros."""
    out, sector = bytearray(), bytearray()
    for one in records:
        if len(sector) + len(one) > BLOCK:
            out += sector + bytes(BLOCK - len(sector))
            sector = bytearray()
        sector += one
    if sector or not out:
        out += sector + bytes(BLOCK - len(sector))
    return bytes(out)


def descriptor(kind: int, root: bytes = b"", blocks: int = 0,
               escape: bytes = b"", ident: bytes = b"CD001") -> bytes:
    d = bytearray(BLOCK)
    d[0], d[1:6], d[6] = kind, ident, 1
    if ident == b"CD001" and kind in (1, 2):
        d[8:40] = b"LINUX".ljust(32)
        d[40:72] = b"CDROM".ljust(32)
        d[80:88] = _both32(blocks)
        d[88:88 + len(escape)] = escape
        d[120:124] = _both16(1)
        d[124:128] = _both16(1)
        d[128:132] = _both16(BLOCK)
        d[156:156 + len(root)] = root
        d[881] = 1
    return bytes(d)


class _Image:
    """An image built sector by sector: put(lba, bytes), then bytes()."""

    def __init__(self):
        self.sectors = {}

    def put(self, lba: int, data: bytes) -> None:
        for n in range(0, max(len(data), 1), BLOCK):
            self.sectors[lba + n // BLOCK] = data[n:n + BLOCK]

    def bytes(self) -> bytes:
        last = max(self.sectors) + 1
        return b"".join(self.sectors.get(n, b"").ljust(BLOCK, b"\x00")
                        for n in range(last))


def _rr_name(name: str) -> bytes:
    raw = name.encode()
    return b"NM" + bytes([5 + len(raw), 1, 0]) + raw


def _rr_mode(mode: int) -> bytes:
    return b"PX" + bytes([36, 1]) + _both32(mode) + _both32(1) \
        + _both32(0) + _both32(0)


RR_SP = b"SP\x07\x01\xbe\xef\x00"


def iso(files: dict, joliet: bool = False, rock_ridge: bool = False,
        udf: bool = False, links=(), modes=None, joliet_names=None,
        entries=None) -> bytes:
    """An image of ``files`` (path -> bytes; a path ending "/" an empty
    directory): a primary tree of ISO 9660 names (upper case, ";1"), a
    Joliet tree of the real names where ``joliet`` (``joliet_names``
    {path: the name Joliet gives it instead}), Rock Ridge's NM and PX
    entries in the primary tree where ``rock_ridge`` (``links`` the paths
    PX calls symbolic links, ``modes`` {path: the mode PX gives it},
    ``entries`` {path: the system use entries written after NM in place of
    PX}), UDF's recognition sequence after the terminator where ``udf``."""
    tree = {}
    for path, data in files.items():
        parts = path.strip("/").split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if path.endswith("/"):
            node.setdefault(parts[-1], {})
        else:
            node[parts[-1]] = data

    def primary_name(name, is_dir):
        text = name.upper()
        return text.encode() if is_dir else text.encode() + b";1"

    def joliet_name(name, is_dir, path=""):
        name = (joliet_names or {}).get(path, name)
        return name.encode("utf-16-be") + (b"" if is_dir else
                                           ";1".encode("utf-16-be"))

    def su_of(name, path, is_dir):
        if not rock_ridge:
            return b""
        if path in (entries or {}):
            return _rr_name(name) + entries[path]
        mode = 0o40755 if is_dir else (0o120777 if path in links
                                       else (modes or {}).get(path, 0o100644))
        return _rr_name(name) + _rr_mode(mode)

    def name_of(namer, name, is_dir, path):
        return namer(name, is_dir, path) if namer is joliet_name \
            else namer(name, is_dir)

    descriptors = 1 + (1 if joliet else 0)
    terminator = 16 + descriptors
    next_lba = terminator + 1 + (3 if udf else 0)
    trees = [("primary", primary_name)] + (
        [("joliet", joliet_name)] if joliet else [])
    # Directory sizes first: records' lengths depend on names, not extents.
    layout = {}                         # (tree, path) -> (lba, size)

    def plan(tree_name, namer, node, path):
        records = 2 + len(node)
        dummy = [record(b"\x00", 0, 0, 2, RR_SP if (rock_ridge and not path
                                                     and tree_name == "primary")
                        else b"")] * 2
        dummy += [record(name_of(namer, k, isinstance(v, dict),
                                 f"{path}{k}"), 0, 0,
                         su=su_of(k, f"{path}{k}", isinstance(v, dict))
                         if tree_name == "primary" else b"")
                  for k, v in sorted(node.items())]
        assert len(dummy) == records
        return len(directory(dummy))

    def assign(tree_name, namer, node, path):
        nonlocal next_lba
        size = plan(tree_name, namer, node, path)
        layout[(tree_name, path)] = (next_lba, size)
        next_lba += size // BLOCK
        for k, v in sorted(node.items()):
            if isinstance(v, dict):
                assign(tree_name, namer, v, f"{path}{k}/")

    for tree_name, namer in trees:
        assign(tree_name, namer, tree, "")
    data_at = {}

    def place(node, path):
        nonlocal next_lba
        for k, v in sorted(node.items()):
            if isinstance(v, dict):
                place(v, f"{path}{k}/")
            else:
                data_at[f"{path}{k}"] = next_lba
                next_lba += max(1, -(-len(v) // BLOCK))
    place(tree, "")
    image = _Image()

    def write(tree_name, namer, node, path, parent):
        lba, size = layout[(tree_name, path)]
        me = (lba, size)
        root_su = RR_SP if (rock_ridge and not path
                            and tree_name == "primary") else b""
        records = [record(b"\x00", lba, size, 2, root_su),
                   record(b"\x01", parent[0], parent[1], 2)]
        for k, v in sorted(node.items()):
            is_dir = isinstance(v, dict)
            su = su_of(k, f"{path}{k}", is_dir) \
                if tree_name == "primary" else b""
            if is_dir:
                child = layout[(tree_name, f"{path}{k}/")]
                records.append(record(name_of(namer, k, True, f"{path}{k}"),
                                      child[0], child[1], 2, su))
            else:
                records.append(record(name_of(namer, k, False, f"{path}{k}"),
                                      data_at[f"{path}{k}"], len(v), 0, su))
        image.put(lba, directory(records))
        for k, v in sorted(node.items()):
            if isinstance(v, dict):
                write(tree_name, namer, v, f"{path}{k}/", me)

    for tree_name, namer in trees:
        root = layout[(tree_name, "")]
        write(tree_name, namer, tree, "", root)

    def data(node, path):
        for k, v in sorted(node.items()):
            if isinstance(v, dict):
                data(v, f"{path}{k}/")
            else:
                image.put(data_at[f"{path}{k}"], v)
    data(tree, "")
    blocks = max(max(image.sectors) + 1, next_lba)
    primary_root = layout[("primary", "")]
    image.put(16, descriptor(1, record(b"\x00", *primary_root, 2), blocks))
    if joliet:
        joliet_root = layout[("joliet", "")]
        image.put(17, descriptor(2, record(b"\x00", *joliet_root, 2), blocks,
                                 escape=b"%/E"))
    image.put(terminator, descriptor(255))
    if udf:
        for n, ident in enumerate((b"BEA01", b"NSR02", b"TEA01")):
            image.put(terminator + 1 + n, descriptor(0, ident=ident))
    image.put(blocks - 1, image.sectors.get(blocks - 1, b""))
    return image.bytes()


def hand_made(dirs: dict, root_lba: int = 20, blocks: int = None,
              extra: dict = None) -> bytes:
    """An image whose directories are given record by record: ``dirs`` maps
    a directory's lba to its records (after "." and ".."), each (name,
    extent, size, flags); the root is at ``root_lba``; ``extra`` puts more
    sectors (lba -> bytes)."""
    image = _Image()
    sizes = {lba: len(directory([record(b"\x00", 0, 0, 2)] * 2 + [
        record(n, e, s, f) for n, e, s, f in recs]))
        for lba, recs in dirs.items()}
    for lba, recs in dirs.items():
        image.put(lba, directory(
            [record(b"\x00", lba, sizes[lba], 2),
             record(b"\x01", root_lba, sizes[root_lba], 2)]
            + [record(n, e, s, f) for n, e, s, f in recs]))
    for lba, data in (extra or {}).items():
        image.put(lba, data)
    total = blocks or max(image.sectors) + 1
    image.put(16, descriptor(1, record(b"\x00", root_lba, sizes[root_lba], 2),
                             total))
    image.put(17, descriptor(255))
    return image.bytes()


class _Iso(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def image(self, data: bytes, seconds: float = 10):
        with deadline.limit(seconds):
            return iso9660.IsoImage(io.BytesIO(data))

    def listing(self, data: bytes, name: str = "disc.iso", seconds=10):
        path = self.dir / name
        path.write_bytes(data)
        with deadline.limit(seconds):
            return fallback._list_archive(path, data[:HEAD_BYTES])

    def read(self, image, name: str) -> bytes:
        member = next(m for m in image.members if m.name == name)
        with image.open(member) as fh:
            return fh.read()


FILES = {"readme.txt": b"Read me first.\n",
         "docs/guide.txt": b"A guide\n" * 600,           # 4,800 bytes
         "docs/deep/deeper/note.md": b"# Note\n",
         "empty.bin": b"",
         "Mixed Case Name.txt": b"case kept"}


class ListingTest(_Iso):
    def test_joliet_names_and_sizes(self):
        image = self.image(iso(FILES, joliet=True))
        self.assertEqual("joliet", image.names)
        listed = {m.name: (m.size, m.is_dir) for m in image.members}
        self.assertEqual({
            "Mixed Case Name.txt": (9, False), "docs": (0, True),
            "docs/deep": (0, True), "docs/deep/deeper": (0, True),
            "docs/deep/deeper/note.md": (7, False),
            "docs/guide.txt": (4_800, False), "empty.bin": (0, False),
            "readme.txt": (15, False)}, listed)
        self.assertEqual(8, image.total)
        self.assertEqual("", image.note)
        self.assertEqual("", image.refused)

    def test_a_directory_comes_before_what_it_holds(self):
        """Walked depth first, in the image's order: a directory, then its
        files and directories, then the next one."""
        image = self.image(iso(FILES, joliet=True))
        names = [m.name for m in image.members]
        self.assertEqual(["Mixed Case Name.txt", "docs", "docs/deep",
                          "docs/deep/deeper", "docs/deep/deeper/note.md",
                          "docs/guide.txt", "empty.bin", "readme.txt"], names)

    def test_iso9660_names_without_joliet(self):
        """The primary tree's names, their version ";1" dropped."""
        image = self.image(iso({"readme.txt": b"x", "sub/a.txt": b"y"}))
        self.assertEqual("iso9660", image.names)
        self.assertEqual(["README.TXT", "SUB", "SUB/A.TXT"],
                         [m.name for m in image.members])

    def test_rock_ridge_names_where_there_is_no_joliet(self):
        image = self.image(iso({"readme.txt": b"x", "Sub Dir/a.txt": b"y",
                                "run": b"target"}, rock_ridge=True,
                               links=("run",)))
        self.assertEqual("rock ridge", image.names)
        self.assertEqual(["Sub Dir", "Sub Dir/a.txt", "readme.txt", "run"],
                         [m.name for m in image.members])
        link = {m.name: m.is_link for m in image.members}
        self.assertEqual({"Sub Dir": False, "Sub Dir/a.txt": False,
                          "readme.txt": False, "run": True}, link)

    def test_the_mtime_is_the_records(self):
        image = self.image(iso({"a.txt": b"x"}, joliet=True))
        self.assertEqual(DATE_POSIX, image.members[0].mtime)

    def test_an_empty_image_lists_nothing(self):
        image = self.image(iso({}, joliet=True))
        self.assertEqual([], image.members)
        self.assertEqual(0, image.total)
        self.assertEqual("", image.note)

    def test_the_listing_of_sift(self):
        """fallback's listing: the paths, a directory's with "/", the sizes,
        the count and the format, as for the other archives."""
        listing = self.listing(iso({"a.txt": b"alpha", "d/b.txt": b"beta"},
                                   joliet=True))
        self.assertEqual("iso9660", listing.format)
        self.assertEqual(["a.txt", "d/", "d/b.txt"], listing.names)
        self.assertEqual([5, 0, 4], listing.sizes)
        self.assertEqual(3, listing.total)
        self.assertEqual("", listing.note)

    def test_extract_names_the_members(self):
        path = self.dir / "disc.iso"
        path.write_bytes(iso({"report.pdf": b"%PDF-1.4 x", "x/y.txt": b"y"},
                             joliet=True))
        det = detect(path)
        self.assertEqual("archive", det.kind)
        ex = extract(path, det, 25_000_000)
        self.assertEqual(["report.pdf", "x/", "x/y.txt"],
                         ex.metadata["archive_members"])
        self.assertEqual(3, ex.metadata["archive_member_count"])
        self.assertEqual("iso9660", ex.metadata["archive_format"])
        self.assertIn("members: report.pdf (10 B)", ex.text)
        self.assertNotIn("archive_note", ex.metadata)

    def test_a_udf_only_image_is_named_not_listed(self):
        image = _Image()
        for n, ident in enumerate((b"BEA01", b"NSR02", b"TEA01")):
            image.put(16 + n, descriptor(0, ident=ident))
        image.put(40, b"x")
        listing = self.listing(image.bytes(), "win.iso")
        self.assertEqual([], listing.names)
        self.assertEqual(0, listing.total)
        self.assertIn("UDF", listing.note)
        self.assertIn("does not list", listing.note)

    def test_a_udf_bridge_image_lists_its_iso9660_files_and_says_so(self):
        listing = self.listing(iso({"README.TXT": b"see UDF"}, udf=True))
        self.assertEqual(["README.TXT"], listing.names)
        self.assertIn("UDF", listing.note)

    def test_an_image_without_a_root_says_so(self):
        """A primary descriptor and nothing else, as sift's older test wrote
        one: no member, a note that names the format."""
        data = bytearray(40_000)
        data[0x8000:0x8007] = b"\x01CD001\x01"
        listing = self.listing(bytes(data))
        self.assertEqual([], listing.names)
        self.assertIn("ISO 9660", listing.note)


class RockRidgeTest(_Iso):
    """Rock Ridge beyond a name in the record: a directory relocated past
    ISO 9660's eight levels (CL where it belongs, RE where it lies), and a
    name continued in a continuation area (CE)."""

    def test_a_relocated_directory_is_listed_where_it_belongs(self):
        cl = b"CL" + bytes([12, 1]) + _both32(23)
        re_ = b"RE" + bytes([4, 1])
        image = _Image()
        image.put(20, directory([
            record(b"\x00", 20, BLOCK, 2, RR_SP), record(b"\x01", 20, BLOCK, 2),
            record(b"A", 21, BLOCK, 2, _rr_name("a")),
            record(b"RR_MOVED", 22, BLOCK, 2, _rr_name("rr_moved"))]))
        image.put(21, directory([
            record(b"\x00", 21, BLOCK, 2), record(b"\x01", 20, BLOCK, 2),
            record(b"DEEP", 0, 0, 0, _rr_name("deep") + cl)]))
        image.put(22, directory([
            record(b"\x00", 22, BLOCK, 2), record(b"\x01", 20, BLOCK, 2),
            record(b"DEEP", 23, BLOCK, 2, _rr_name("deep") + re_)]))
        image.put(23, directory([
            record(b"\x00", 23, BLOCK, 2), record(b"\x01", 22, BLOCK, 2),
            record(b"F.TXT;1", 30, 4, 0, _rr_name("f.txt"))]))
        image.put(30, b"deep")
        image.put(16, descriptor(1, record(b"\x00", 20, BLOCK, 2), 31))
        image.put(17, descriptor(255))
        listed = self.image(image.bytes())
        self.assertEqual("rock ridge", listed.names)
        self.assertEqual([("a", True), ("a/deep", True),
                          ("a/deep/f.txt", False), ("rr_moved", True)],
                         [(m.name, m.is_dir) for m in listed.members])
        self.assertEqual(b"deep", self.read(listed, "a/deep/f.txt"))
        self.assertEqual("", listed.note)

    def test_a_name_continued_in_a_continuation_area(self):
        first = b"NM" + bytes([5 + 5, 1, 1]) + b"part1"      # continues
        ce = b"CE" + bytes([28, 1]) + _both32(25) + _both32(0) + _both32(12)
        area = b"NM" + bytes([5 + 5, 1, 0]) + b"part2" + b"ST\x04\x01"
        image = _Image()
        image.put(20, directory([
            record(b"\x00", 20, BLOCK, 2, RR_SP), record(b"\x01", 20, BLOCK, 2),
            record(b"P.TXT;1", 30, 2, 0, first + ce)]))
        image.put(25, area)
        image.put(30, b"ok")
        image.put(16, descriptor(1, record(b"\x00", 20, BLOCK, 2), 31))
        image.put(17, descriptor(255))
        listed = self.image(image.bytes())
        self.assertEqual(["part1part2"], [m.name for m in listed.members])

    def test_a_continuation_chain_is_bounded(self):
        """Every continuation area pointing at itself: followed
        MAX_CONTINUATIONS times, the name what those held."""
        ce = b"CE" + bytes([28, 1]) + _both32(25) + _both32(0) + _both32(38)
        area = b"NM" + bytes([5 + 1, 1, 1]) + b"x" + ce
        image = _Image()
        image.put(20, directory([
            record(b"\x00", 20, BLOCK, 2, RR_SP), record(b"\x01", 20, BLOCK, 2),
            record(b"P.TXT;1", 30, 2, 0, ce)]))
        image.put(25, area)
        image.put(30, b"ok")
        image.put(16, descriptor(1, record(b"\x00", 20, BLOCK, 2), 31))
        image.put(17, descriptor(255))
        listed = self.image(image.bytes(), seconds=5)
        self.assertEqual(["x" * iso9660.MAX_CONTINUATIONS],
                         [m.name for m in listed.members])


class ReadingTest(_Iso):
    def test_every_file_reads_back(self):
        for joliet in (True, False):
            with self.subTest(joliet=joliet):
                image = self.image(iso(FILES, joliet=joliet))
                for member in image.members:
                    if member.is_dir:
                        continue
                    wanted = next(v for k, v in FILES.items()
                                  if k.upper() == member.name.upper())
                    with image.open(member) as fh:
                        self.assertEqual(wanted, fh.read())

    def test_reading_in_pieces(self):
        image = self.image(iso(FILES, joliet=True))
        member = next(m for m in image.members if m.name == "docs/guide.txt")
        with image.open(member) as fh:
            pieces = [fh.read(1000) for _ in range(6)]
        self.assertEqual([1000] * 4 + [800, 0], [len(p) for p in pieces])
        self.assertEqual(FILES["docs/guide.txt"], b"".join(pieces))

    def test_a_directory_is_not_read(self):
        image = self.image(iso(FILES, joliet=True))
        member = next(m for m in image.members if m.name == "docs")
        with self.assertRaises(MemberError):
            image.open(member)

    def test_a_multi_extent_file_reads_its_extents_in_order(self):
        """A file of two records of the same name, the first flagged "not
        final" (0x80): one member, its size the sum."""
        data = hand_made({20: [(b"BIG.BIN;1", 30, 3000, 0x80),
                               (b"BIG.BIN;1", 25, 100, 0)]},
                         extra={30: b"A" * 3000, 25: b"B" * 100})
        image = self.image(data)
        self.assertEqual([("BIG.BIN", 3100)],
                         [(m.name, m.size) for m in image.members])
        self.assertEqual(b"A" * 3000 + b"B" * 100,
                         self.read(image, "BIG.BIN"))

    def test_the_same_file_twice_is_no_overlap(self):
        """Hard links share an extent, as genisoimage writes them."""
        data = hand_made({20: [(b"A.TXT;1", 30, 5, 0), (b"B.TXT;1", 30, 5, 0)]},
                         extra={30: b"hello"})
        image = self.image(data)
        self.assertEqual("", image.note)
        self.assertEqual(b"hello", self.read(image, "B.TXT"))


class HostileTest(_Iso):
    def test_a_directory_loop_is_walked_once(self):
        """A directory whose record points back at the root: listed, not
        walked again, and said."""
        data = hand_made({20: [(b"A.TXT;1", 30, 1, 0), (b"LOOP", 20, 2048, 2)]},
                         extra={30: b"a"})
        image = self.image(data, seconds=5)
        self.assertEqual(["A.TXT", "LOOP"], [m.name for m in image.members])
        self.assertIn("loop", image.note)

    def test_two_directories_pointing_at_each_other(self):
        data = hand_made({20: [(b"X", 21, 2048, 2)],
                          21: [(b"Y", 22, 2048, 2)],
                          22: [(b"BACK", 21, 2048, 2)]})
        image = self.image(data, seconds=5)
        self.assertEqual(["X", "X/Y", "X/Y/BACK"],
                         [m.name for m in image.members])
        self.assertIn("loop", image.note)

    def test_a_size_past_the_end_is_noted_and_not_read(self):
        """A file declaring 4 GiB in an image of a few sectors: listed with
        its declared size, not readable, the image said to be cut short."""
        data = hand_made({20: [(b"HUGE.BIN;1", 30, 0xFFFFFFFF, 0),
                               (b"OK.TXT;1", 31, 2, 0)]},
                         extra={30: b"x" * 10, 31: b"ok"})
        image = self.image(data)
        huge = image.members[0]
        self.assertEqual(("HUGE.BIN", 0xFFFFFFFF), (huge.name, huge.size))
        self.assertIn("past", huge.problem)
        with self.assertRaises(MemberError):
            image.open(huge)
        self.assertIn("cut short", image.note)
        self.assertEqual(b"ok", self.read(image, "OK.TXT"))

    def test_a_truncated_image_lists_what_it_holds(self):
        full = iso({"a.txt": b"a" * 5000, "b.txt": b"b" * 5000}, joliet=True)
        image = self.image(full[:-4000])
        names = {m.name: m.problem for m in image.members}
        self.assertEqual("", names["a.txt"])
        self.assertIn("past", names["b.txt"])
        self.assertEqual(b"a" * 5000, self.read(image, "a.txt"))

    def test_a_directory_past_the_end_is_noted(self):
        data = hand_made({20: [(b"GONE", 900, 2048, 2), (b"A.TXT;1", 30, 1, 0)]},
                         extra={30: b"a"})
        image = self.image(data)
        self.assertEqual(["GONE", "A.TXT"], [m.name for m in image.members])
        self.assertIn("past", image.note)

    def test_overlapping_extents_are_noted(self):
        """A file whose extent lies over a directory's, and two files that
        overlap only in part: listed, readable, and said."""
        data = hand_made({20: [(b"A.TXT;1", 30, 4096, 0),
                               (b"B.TXT;1", 31, 100, 0),
                               (b"DIRBYTES;1", 20, 50, 0)]},
                         extra={30: b"a" * 4096})
        image = self.image(data)
        self.assertIn("overlap", image.note)
        self.assertEqual(3, len(image.members))

    def test_the_entry_count_is_bounded_by_zipguards(self):
        """MAX_ZIP_ENTRIES records and one more: the walk stops at the
        bound, the listing says how many it saw and that it was refused."""
        count = zipguard.MAX_ZIP_ENTRIES + 1
        recs = [(f"F{n:05d}.TXT;1".encode(), 30, 0, 0) for n in range(count)]
        image = self.image(hand_made({20: recs}), seconds=30)
        self.assertEqual(zipguard.MAX_ZIP_ENTRIES, len(image.members))
        self.assertEqual(count, image.total)
        self.assertIn("entry limit", image.refused)
        listing = self.listing(hand_made({20: recs}), seconds=30)
        self.assertEqual(fallback.MAX_LISTED, len(listing.names))
        self.assertEqual(count, listing.total)
        self.assertIn("entry limit", listing.refused)

    def test_a_huge_directory_is_refused_before_it_is_read(self):
        """A directory declaring 4 GiB: more than the directory bytes sift
        reads, refused without reading them."""
        data = hand_made({20: [(b"BIG", 21, 0xFFFFF800, 2)],
                          21: [(b"A.TXT;1", 30, 0, 0)]}, blocks=64)
        image = self.image(data)
        self.assertEqual(["BIG"], [m.name for m in image.members])
        self.assertTrue(image.note or image.refused)

    def test_the_directory_bytes_are_bounded(self):
        self.assertEqual(zipguard.MAX_ZIP_DIRECTORY_BYTES,
                         iso9660.MAX_DIRECTORY_BYTES)
        self.assertEqual(zipguard.MAX_ZIP_ENTRIES, iso9660.MAX_ENTRIES)

    def test_a_deep_chain_stops_at_the_depth_bound(self):
        depth = iso9660.MAX_DEPTH + 5
        dirs = {20 + n: [(f"D{n}".encode(), 21 + n, 2048, 2)]
                for n in range(depth)}
        dirs[20 + depth] = []
        image = self.image(hand_made(dirs), seconds=10)
        deepest = max(m.name.count("/") for m in image.members)
        self.assertEqual(iso9660.MAX_DEPTH - 1, deepest)
        self.assertIn("deeper", image.note)

    def test_a_damaged_record_ends_its_directory_only(self):
        good = hand_made({20: [(b"A.TXT;1", 30, 1, 0), (b"SUB", 21, 2048, 2)],
                          21: [(b"B.TXT;1", 31, 1, 0)]},
                         extra={30: b"a", 31: b"b"})
        data = bytearray(good)
        # The second record of SUB after "." and "..": a length of 3.
        at = 21 * BLOCK + 34 + 34
        data[at] = 3
        image = self.image(bytes(data))
        self.assertEqual(["A.TXT", "SUB"], [m.name for m in image.members])
        self.assertIn("damaged", image.note)

    def test_garbage_after_the_magic_is_no_exception(self):
        for length in (0x8006, 0x8800, 40_000):
            data = bytearray(length)
            data[0x8000:0x8007] = b"\x01CD001\x01"
            if length > 0x8000 + 156:
                data[0x8000 + 156] = 34     # a root record of zeros
            with self.subTest(length=length):
                listing = self.listing(bytes(data))
                self.assertEqual([], listing.names)


#: A member of each type, and the mode Rock Ridge's PX gives it.
TYPED = {"a.txt": (b"alpha", 0o100644, "file"),
         "pipe": (b"", stat.S_IFIFO | 0o644, "fifo"),
         "tty": (b"", stat.S_IFCHR | 0o620, "device"),
         "disk": (b"", stat.S_IFBLK | 0o660, "device"),
         "sock": (b"", stat.S_IFSOCK | 0o755, "socket"),
         "run": (b"a.txt", stat.S_IFLNK | 0o777, "link"),
         "d/inner.txt": (b"inner", 0o100600, "file"),
         "d/fifo": (b"", stat.S_IFIFO | 0o600, "fifo")}
SPECIAL = ("pipe", "tty", "disk", "sock", "d/fifo")


def typed(joliet: bool, **kwargs) -> bytes:
    return iso({path: data for path, (data, _, _) in TYPED.items()},
               joliet=joliet, rock_ridge=True, links=("run",),
               modes={path: mode for path, (_, mode, _) in TYPED.items()},
               **kwargs)


def flat(primary, joliet=None, blocks=None) -> bytes:
    """An image of one root directory given record by record, Rock Ridge
    in the primary tree (its root at 20) and a Joliet tree (at 21) where
    ``joliet``: each record (name, extent, size, system use or b"")."""
    image = _Image()
    root = [record(b"\x00", 20, BLOCK, 2, RR_SP), record(b"\x01", 20, BLOCK, 2)]
    image.put(20, directory(root + [record(n, e, s, 0, su)
                                    for n, e, s, su in primary]))
    if joliet is not None:
        image.put(21, directory(
            [record(b"\x00", 21, BLOCK, 2), record(b"\x01", 21, BLOCK, 2)]
            + [record(n.encode("utf-16-be"), e, s, 0) for n, e, s in joliet]))
    image.put(30, b"data")
    total = blocks or max(image.sectors) + 1
    image.put(16, descriptor(1, record(b"\x00", 20, BLOCK, 2), total))
    at = 17
    if joliet is not None:
        image.put(17, descriptor(2, record(b"\x00", 21, BLOCK, 2), total,
                                 escape=b"%/E"))
        at = 18
    image.put(at, descriptor(255))
    return image.bytes()


def _px(mode: int) -> bytes:
    return _rr_mode(mode)


SL = b"SL" + bytes([4 + 1 + 2 + 5, 1, 0]) + bytes([0, 5]) + b"a.txt"


class TypeTest(_Iso):
    """A member's type as Rock Ridge's PX (and SL) says it, the way a zip's
    and a tar's members say theirs: ``kind`` one of
    file, dir, link, fifo, device, socket or unknown, ``mode`` PX's mode.
    A host application skips what is not a regular file by it; sift's own reader
    does not open a special file as one."""

    def kinds(self, image) -> dict:
        return {m.name: m.kind for m in image.members}

    def test_rock_ridge_says_each_type(self):
        image = self.image(typed(joliet=False))
        self.assertEqual("rock ridge", image.names)
        wanted = {path: kind for path, (_, _, kind) in TYPED.items()}
        wanted["d"] = "dir"
        self.assertEqual(wanted, self.kinds(image))
        modes = {m.name: m.mode for m in image.members if not m.is_dir}
        self.assertEqual({p: mode for p, (_, mode, _) in TYPED.items()}, modes)
        self.assertEqual({"run"}, {m.name for m in image.members if m.is_link})
        self.assertEqual({"d"}, {m.name for m in image.members if m.is_dir})
        self.assertEqual("", image.note)

    def test_joliet_names_and_rock_ridges_types(self):
        """genisoimage -J -R, the most common image: sift walks Joliet's
        tree for the names and reads the primary tree's Rock Ridge for the
        types -- before, every special file and link of such an image was a
        regular file."""
        image = self.image(typed(joliet=True))
        self.assertEqual("joliet", image.names)
        wanted = {path: kind for path, (_, _, kind) in TYPED.items()}
        wanted["d"] = "dir"
        self.assertEqual(wanted, self.kinds(image))
        self.assertEqual(stat.S_IFBLK | 0o660,
                         next(m.mode for m in image.members if m.name == "disk"))
        self.assertEqual("", image.note)

    def test_a_symlink_in_joliet_and_rock_ridge(self):
        image = self.image(iso({"target.txt": b"t", "link": b"target.txt"},
                               joliet=True, rock_ridge=True, links=("link",)))
        link = next(m for m in image.members if m.name == "link")
        self.assertTrue(link.is_link)
        self.assertEqual("link", link.kind)
        self.assertEqual(stat.S_IFLNK | 0o777, link.mode)
        self.assertEqual("file", next(m.kind for m in image.members
                                      if m.name == "target.txt"))

    def test_a_special_file_is_not_opened_and_a_regular_one_reads(self):
        for joliet in (False, True):
            image = self.image(typed(joliet=joliet))
            for name in SPECIAL:
                with self.subTest(joliet=joliet, name=name):
                    member = next(m for m in image.members if m.name == name)
                    with self.assertRaises(MemberError) as caught:
                        image.open(member)
                    self.assertIn("not a regular file", str(caught.exception))
            with self.subTest(joliet=joliet, name="a.txt"):
                self.assertEqual(b"alpha", self.read(image, "a.txt"))
                self.assertEqual(b"inner", self.read(image, "d/inner.txt"))

    def test_without_rock_ridge_every_file_is_a_file(self):
        for joliet in (False, True):
            with self.subTest(joliet=joliet):
                image = self.image(iso({"a.txt": b"a", "d/b.txt": b"b"},
                                       joliet=joliet))
                self.assertEqual({"a.txt": "file", "d": "dir", "d/b.txt": "file"},
                                 {m.name.lower(): m.kind for m in image.members})
                self.assertEqual({None}, {m.mode for m in image.members})

    def test_a_joliet_name_other_than_rock_ridges(self):
        """Joliet cuts a name at 64 characters, and a writer may name a file
        otherwise in each tree: the type found by the file's extent, in a
        directory named otherwise too."""
        image = self.image(typed(joliet=True, joliet_names={
            "pipe": "Pipe (Joliet)", "d": "D-joliet"}))
        kinds = self.kinds(image)
        self.assertEqual("fifo", kinds["Pipe (Joliet)"])
        self.assertEqual("fifo", kinds["D-joliet/fifo"])
        self.assertEqual("file", kinds["D-joliet/inner.txt"])
        self.assertNotIn("pipe", kinds)

    def test_zero_sized_files_sharing_an_extent(self):
        """Empty files and special files may all point at one extent: the
        path tells them apart; a Joliet name Rock Ridge does not know at
        that extent is of no one type, so not a regular file; a Joliet
        file at an extent the primary tree does not hold is ISO 9660's."""
        primary = [(b"EMPTY.;1", 0, 0, _rr_name("empty") + _px(0o100644)),
                   (b"PIPE.;1", 0, 0, _rr_name("pipe") + _px(stat.S_IFIFO | 0o644)),
                   (b"TTY.;1", 0, 0, _rr_name("tty") + _px(stat.S_IFCHR | 0o600))]
        joliet = [("empty;1", 0, 0), ("other;1", 0, 0), ("pipe;1", 0, 0),
                  ("tty;1", 0, 0), ("lone;1", 30, 4)]
        image = self.image(flat(primary, joliet))
        self.assertEqual({"empty": "file", "other": "unknown", "pipe": "fifo",
                          "tty": "device", "lone": "file"}, self.kinds(image))

    def test_a_zero_sized_file_needs_no_extent_inside_the_image(self):
        """libarchive's writer (bsdtar --format iso9660) gives an empty file,
        a link or a device an extent past the image: nothing of it is read,
        so the image is not cut short."""
        far = 0xFFFFFFF0
        image = self.image(flat(
            [(b"EMPTY.;1", far, 0, _rr_name("empty") + _px(0o100644)),
             (b"PIPE.;1", far + 1, 0,
              _rr_name("pipe") + _px(stat.S_IFIFO | 0o644)),
             (b"DATA.;1", 30, 4, _rr_name("data") + _px(0o100644))]))
        self.assertEqual({"empty": "", "pipe": "", "data": ""},
                         {m.name: m.problem for m in image.members})
        self.assertEqual("", image.note)
        self.assertEqual(b"", self.read(image, "empty"))
        self.assertEqual(b"data", self.read(image, "data"))

    #: System use entries after NM where PX, or the entry that would say
    #: the type, cannot be read: (entries, the kind they make).
    DAMAGED = {
        "PX too short for its mode": (b"PX" + bytes([8, 1]) + b"\xa4\x81\x00\x00",
                                      "unknown"),
        "PX longer than the area": (b"PX" + bytes([200, 1]) + _both32(0o100644),
                                    "unknown"),
        "PX without a type": (_px(0o644), "unknown"),
        "PX calls a file a directory": (_px(0o40755), "unknown"),
        "PX of a type no file has": (_px(0o170644), "unknown"),
        "SL where PX says a file": (_px(0o100644) + SL, "unknown"),
        "a damaged entry before PX": (b"XX\x02\x01" + _px(0o100644), "unknown"),
        "SL alone": (SL, "link"),
        "PX then a damaged entry": (_px(stat.S_IFIFO | 0o644) + b"XX\x02\x01",
                                    "fifo"),
        "NM alone, zeros after it": (b"\x00" * 6, "file"),
    }

    def test_damaged_px_and_sl_are_no_regular_file(self):
        """A PX that cannot be read, or names no type a file has, and an
        area that breaks before its PX: the type is unknown -- not a regular
        file, not opened. Rock Ridge writes a PX for every record; a record
        without one, and SL without PX, keep what they said before."""
        for joliet in (False, True):
            for case, (entries, kind) in self.DAMAGED.items():
                with self.subTest(case, joliet=joliet):
                    image = self.image(iso(
                        {"x.bin": b"xx", "ok.txt": b"ok", "d/in.txt": b"in"},
                        joliet=joliet, rock_ridge=True,
                        entries={"x.bin": entries, "d": entries}))
                    kinds = self.kinds(image)
                    self.assertEqual(kind, kinds["x.bin"])
                    self.assertEqual("file", kinds["ok.txt"])
                    # A directory is a directory by its record's flag.
                    self.assertEqual("dir", kinds["d"])
                    self.assertEqual("file", kinds["d/in.txt"])
                    self.assertEqual(kind == "link",
                                     next(m.is_link for m in image.members
                                          if m.name == "x.bin"))
                    member = next(m for m in image.members if m.name == "x.bin")
                    self.assertEqual(kind == "unknown",
                                     "of a type sift cannot tell" in image.note)
                    if kind in ("file", "link"):
                        self.assertEqual(b"xx", self.read(image, "x.bin"))
                    else:
                        with self.assertRaises(MemberError):
                            image.open(member)

    def test_a_continuation_not_read_leaves_the_type_unknown(self):
        """PX may lie in a continuation area; one past the end of the image
        is not read, and the type with it is unknown."""
        ce = b"CE" + bytes([28, 1]) + _both32(900) + _both32(0) + _both32(44)
        for joliet in (None, [("x;1", 30, 4)]):
            with self.subTest(joliet=bool(joliet)):
                image = self.image(flat(
                    [(b"X.;1", 30, 4, _rr_name("x") + ce)], joliet))
                self.assertEqual({"x": "unknown"}, self.kinds(image))

    def test_the_rock_ridge_of_a_joliet_image_is_read_within_bounds(self):
        """The primary tree read for its types under bounds of its own and
        counted apart from the listing's: a loop is walked once, a directory
        past the end is passed over, and one past the directory bytes stops
        that reading -- the Joliet listing whole and not refused, the files
        whose type was not read unknown, not opened, and said."""
        def build(third):
            image = _Image()
            image.put(20, directory([
                record(b"\x00", 20, BLOCK, 2, RR_SP),
                record(b"\x01", 20, BLOCK, 2),
                record(b"A.;1", 30, 4, 0, _rr_name("a") + _px(0o100644)),
                record(b"SUB", 23, BLOCK, 2, _rr_name("sub") + _px(0o40755)),
                third,
                record(b"P.;1", 31, 0, 0,
                       _rr_name("p") + _px(stat.S_IFIFO | 0o644))]))
            image.put(23, directory([record(b"\x00", 23, BLOCK, 2),
                                     record(b"\x01", 20, BLOCK, 2)]))
            image.put(21, directory([
                record(b"\x00", 21, BLOCK, 2), record(b"\x01", 21, BLOCK, 2),
                record("a;1".encode("utf-16-be"), 30, 4, 0),
                record("p;1".encode("utf-16-be"), 31, 0, 0)]))
            image.put(30, b"data")
            image.put(16, descriptor(1, record(b"\x00", 20, BLOCK, 2), 32))
            image.put(17, descriptor(2, record(b"\x00", 21, BLOCK, 2), 32,
                                     escape=b"%/E"))
            image.put(18, descriptor(255))
            return image.bytes()

        loop = record(b"LOOP", 20, BLOCK, 2, _rr_name("loop") + _px(0o40755))
        past = record(b"PAST", 900, BLOCK, 2, _rr_name("past") + _px(0o40755))
        big = record(b"BIG", 24, 2 * BLOCK, 2, _rr_name("big") + _px(0o40755))
        for third in (loop, past):
            listed = self.image(build(third), seconds=5)
            self.assertEqual({"a": "file", "p": "fifo"}, self.kinds(listed))
            self.assertEqual(("", ""), (listed.note, listed.refused))
        # Two blocks of directory bytes: the root's and SUB's spent, BIG's
        # two more stop the reading of the types before P. The Joliet root
        # is read under the listing's two blocks all the same.
        with mock.patch.object(iso9660, "MAX_DIRECTORY_BYTES", 2 * BLOCK):
            listed = self.image(build(big), seconds=5)
        self.assertEqual({"a": "file", "p": "unknown"}, self.kinds(listed))
        self.assertEqual("", listed.refused)
        self.assertIn("Rock Ridge", listed.note)
        with self.assertRaises(MemberError):
            listed.open(listed.members[1])

    def test_the_listing_of_sift_is_unchanged(self):
        listing = self.listing(typed(joliet=True))
        self.assertEqual(sorted([*TYPED, "d/"]), sorted(listing.names))
        self.assertEqual("", listing.note)


class RealSampleTest(_Iso):
    """Real images (a census of real files: genisoimage's, Joliet
    and Rock Ridge): each listed without an exception, every file read back
    to its size or its problem said."""

    @unittest.skipUnless(samples("*.iso"), "SIFT_ARCHIVE_SAMPLES not set")
    def test_real_images(self):
        listed = 0
        for path in samples("*.iso"):
            with self.subTest(path.name), path.open("rb") as fh:
                with deadline.limit(60):
                    image = iso9660.IsoImage(fh)
                for member in image.members:
                    if member.is_dir or member.problem:
                        continue
                    if member.kind not in ("file", "link"):
                        with self.assertRaises(MemberError):
                            image.open(member)
                        continue
                    with image.open(member) as reader:
                        self.assertEqual(member.size, len(reader.read()))
                listed += bool(image.members)
        self.assertTrue(listed, "no sample image listed a file")


if __name__ == "__main__":
    unittest.main()
