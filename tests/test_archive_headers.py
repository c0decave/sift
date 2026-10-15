"""Archive headers from a damaged or hostile file.

sift lists the members of every archive in a home directory, in a worker
with 1 GiB of memory. The numbers a header declares -- a size, a name's
length, where the file table starts -- are the file's to choose, so each is
read strictly (its digits and nothing else), every seek goes forward, every
read whose length the file declares is bounded before it happens, and damage
is a note on the listing with the members before it kept: never an
exception out of _list_archive, never a loop, never the declared size in
memory.
"""
import bz2
import gzip
import io
import lzma
import shutil
import struct
import tarfile
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

from sift import deadline
from sift.detect import HEAD_BYTES
from sift.extract import fallback

try:                                    # Python 3.14 and later
    from compression import zstd
except ImportError:                     # pragma: no cover - older Python
    zstd = None

MIB = 1 << 20


class _Archives(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def listing(self, name, data, seconds=10):
        """_list_archive on ``data``, under a deadline: a loop fails the
        test instead of hanging it."""
        path = self.write(name, data)
        with deadline.limit(seconds):
            return fallback._list_archive(path, data[:HEAD_BYTES])

    def assertDamagedAfter(self, listing, count):
        self.assertIn(f"damaged after {count} member(s)", listing.note)


class _Largest:
    """Records the longest read a reader asks of the file it opens."""

    def __init__(self, fh, seen):
        self._fh, self._seen = fh, seen

    def read(self, size=-1):
        data = self._fh.read(size)
        self._seen.append(len(data))
        return data

    def __getattr__(self, name):
        return getattr(self._fh, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()


def largest_read(reader, path):
    """What ``reader(path)`` returns, and the most bytes one read of the
    file handed it."""
    seen = []
    real_open = Path.open

    def spied(self, *args, **kwargs):
        return _Largest(real_open(self, *args, **kwargs), seen)

    with mock.patch.object(Path, "open", spied):
        with deadline.limit(10):
            result = reader(path)
    return result, max(seen, default=0)


# -------------------------------------------------------------------- ar

def ar(members):
    """An ar archive of (name field, size field, data), the fields written
    as given -- a damaged size is written as it is damaged."""
    out = b"!<arch>\n"
    for name, size, data in members:
        out += (name.ljust(16) + b"0".ljust(12) + b"0".ljust(6) * 2
                + b"100644".ljust(8) + size.ljust(10) + b"`\n") + data
        if len(data) % 2:
            out += b"\n"
    return out


def member(name, data):
    return (name, b"%d" % len(data), data)


class ArHeaderTest(_Archives):
    """_ar_listing read a size with int(), which takes a sign, blanks and
    underscores: a negative size seeked before the start of the file
    (OSError), read the rest of the file into memory ("//" of size -1:
    read(-1)) or seeked back onto a header -- a symbol table, never counted,
    read again until the deadline."""

    def test_a_negative_size_is_damage_and_the_members_before_stay(self):
        listing = self.listing("neg.a", ar([member(b"a.txt/", b"abcd"),
                                            (b"b.txt/", b"-200", b"")]))
        self.assertEqual(listing.names, ["a.txt"])
        self.assertDamagedAfter(listing, 1)

    def test_a_negative_name_table_reads_nothing_of_the_rest(self):
        path = self.write("negnames.a", ar([(b"//", b"-1", b"")])
                          + b"x" * (4 * MIB))
        listing, largest = largest_read(fallback._ar_listing, path)
        self.assertEqual(listing.total, 0)
        self.assertDamagedAfter(listing, 0)
        self.assertLessEqual(largest, 64 * 1024)

    def test_a_size_that_points_back_does_not_loop(self):
        for name in (b"/", b"a.txt/"):
            with self.subTest(name):
                listing = self.listing("loop.a", ar([(name, b"-60", b"")]),
                                       seconds=3)
                self.assertEqual(listing.total, 0)
                self.assertDamagedAfter(listing, 0)

    def test_only_decimal_digits_are_a_size(self):
        for field in (b"+4", b"0_4", b" 4", b"4x", b"0x4", b"4.0", b"",
                      b"\xd9\xa4"):
            with self.subTest(field):
                listing = self.listing("f.a", ar([(b"a.txt/", field, b"abcd")]))
                self.assertEqual(listing.names, [])
                self.assertDamagedAfter(listing, 0)

    def test_a_member_ending_at_the_end_of_the_file(self):
        listing = self.listing("exact.a", ar([member(b"a.txt/", b"abcd")]))
        self.assertEqual((listing.names, listing.sizes), (["a.txt"], [4]))
        self.assertEqual(listing.note, "")

    def test_a_member_one_byte_past_the_end_is_cut_short(self):
        listing = self.listing("cut.a", ar([(b"a.txt/", b"5", b"abcd")]))
        self.assertEqual(listing.names, ["a.txt"])
        self.assertIn("cut short", listing.note)

    def test_a_member_of_size_zero(self):
        listing = self.listing("zero.a", ar([member(b"e/", b""),
                                             member(b"f/", b"xy")]))
        self.assertEqual((listing.names, listing.sizes), (["e", "f"], [0, 2]))
        self.assertEqual(listing.note, "")

    def test_an_empty_archive(self):
        listing = self.listing("empty.a", b"!<arch>\n")
        self.assertEqual((listing.total, listing.note), (0, ""))

    def test_a_static_library_with_gnu_and_bsd_long_names(self):
        table = b"a_rather_long_object_name.o/\nsecond_long_object.o/\n"
        gnu = ar([(b"/", b"4", b"\x00\x00\x00\x00"),
                  (b"//", b"%d" % len(table), table),
                  member(b"/0", b"\x7fELF"), member(b"/29", b"\x7fELF12"),
                  member(b"short.o/", b"x")])
        listing = self.listing("libgnu.a", gnu)
        self.assertEqual(listing.names, ["a_rather_long_object_name.o",
                                         "second_long_object.o", "short.o"])
        self.assertEqual(listing.sizes, [4, 6, 1])
        bsd = ar([(b"#1/20", b"28", b"__.SYMDEF SORTED\x00\x00\x00\x00abcdefgh"),
                  (b"#1/24", b"28", b"a_long_bsd_object_name.o" + b"ELF!"),
                  (b"#1/8", b"8", b"c.o\x00\x00\x00\x00\x00")])
        listing = self.listing("libbsd.a", bsd)
        self.assertEqual(listing.names, ["a_long_bsd_object_name.o", "c.o"])
        self.assertEqual(listing.sizes, [4, 0])

    def test_a_bsd_name_longer_than_its_member_is_damage(self):
        listing = self.listing("bsd.a", ar([member(b"a.o/", b"ab"),
                                            (b"#1/20", b"12", b"x" * 12)]))
        self.assertEqual(listing.names, ["a.o"])
        self.assertDamagedAfter(listing, 1)

    def test_a_nul_inside_a_name_ends_it(self):
        table = b"lo\x00ng_object_name.o/\n"
        listing = self.listing("nul.a", ar([
            (b"#1/8", b"12", b"ab\x00cd\x00\x00\x00data"),
            (b"//", b"%d" % len(table), table), member(b"/0", b"x"),
            member(b"s\x00rt.o/", b"y")]))
        self.assertEqual(listing.names, ["ab", "lo", "s"])

    def test_tables_count_toward_the_entry_limit(self):
        tables = [(b"/", b"0", b"")] * 50
        with mock.patch.object(fallback, "MAX_TAR_ENTRIES", 10):
            listing = self.listing("tables.a",
                                   ar(tables + [member(b"a.o/", b"x")]))
        self.assertIn("entry limit", listing.refused)


# -------------------------------------------------------------------- cpio

def newc(name, data, size=None, namesize=None):
    raw = name + b"\x00"
    size = b"%08X" % len(data) if size is None else size
    namesize = b"%08X" % len(raw) if namesize is None else namesize
    out = (b"070701" + b"%08X" % 1 + b"%08X" % 0o100644 + b"0" * 32 + size
           + b"0" * 32 + namesize + b"0" * 8) + raw
    out += b"\x00" * (-len(out) % 4)
    out += data
    return out + b"\x00" * (-len(out) % 4)


def odc(name, data, size=None):
    raw = name + b"\x00"
    size = b"%011o" % len(data) if size is None else size
    return (b"070707" + b"0" * 42 + b"0" * 11 + b"%06o" % len(raw) + size
            + raw + data)


TRAILER = newc(b"TRAILER!!!", b"")


class CpioHeaderTest(_Archives):
    """_cpio_listing read its hex and octal fields with int(), which takes a
    sign, blanks and underscores (only the first header is matched by
    detect's pattern): a negative size seeked before the start of the file
    or back onto a header, and reached the listing; a member running past
    the end, or an archive with no TRAILER!!!, looked complete."""

    def test_newc_and_odc_list_as_before(self):
        listing = self.listing("a.cpio", newc(b"etc/hosts", b"127.0.0.1\n")
                               + newc(b"init", b"#!/bin/sh\n") + TRAILER)
        self.assertEqual((listing.names, listing.sizes),
                         (["etc/hosts", "init"], [10, 10]))
        self.assertEqual(listing.note, "")
        listing = self.listing("o.cpio", odc(b"a.txt", b"alpha")
                               + odc(b"TRAILER!!!", b""))
        self.assertEqual((listing.names, listing.sizes), (["a.txt"], [5]))

    def test_a_negative_size_is_damage_and_the_members_before_stay(self):
        for size in (b"-FFFFFFF", b"-0000078"):
            with self.subTest(size):
                listing = self.listing("neg.cpio", newc(b"a.txt", b"alpha")
                                       + newc(b"b.txt", b"", size=size))
                self.assertEqual((listing.names, listing.sizes),
                                 (["a.txt"], [5]))
                self.assertDamagedAfter(listing, 1)

    def test_only_hex_digits_are_a_newc_number(self):
        for field in (b"+0000005", b" 0000005", b"0000_005", b"0x000005",
                      b"0000005 ", b"00000G05"):
            with self.subTest(field):
                listing = self.listing("f.cpio", newc(b"a.txt", b"alpha")
                                       + newc(b"b.txt", b"12345", size=field)
                                       + TRAILER)
                self.assertEqual(listing.names, ["a.txt"])
                self.assertDamagedAfter(listing, 1)
        listing = self.listing("n.cpio", newc(b"a.txt", b"alpha")
                               + newc(b"b.txt", b"", namesize=b"-0000006"))
        self.assertDamagedAfter(listing, 1)

    def test_only_octal_digits_are_an_odc_number(self):
        for field in (b"-0000000200", b"+0000000005", b" 0000000005",
                      b"00000000009"):
            with self.subTest(field):
                listing = self.listing("o.cpio", odc(b"a.txt", b"alpha")
                                       + odc(b"b.txt", b"", size=field)
                                       + odc(b"TRAILER!!!", b""))
                self.assertEqual(listing.names, ["a.txt"])
                self.assertDamagedAfter(listing, 1)

    def test_a_member_past_the_end_is_cut_short(self):
        listing = self.listing("past.cpio", newc(b"a.txt", b"alpha")
                               + newc(b"b.txt", b"", size=b"00100000"))
        self.assertEqual(listing.names, ["a.txt", "b.txt"])
        self.assertIn("cut short", listing.note)

    def test_an_archive_without_its_trailer_says_so(self):
        listing = self.listing("cut.cpio", newc(b"a.txt", b"alpha"))
        self.assertEqual(listing.names, ["a.txt"])
        self.assertDamagedAfter(listing, 1)

    def test_a_nul_inside_a_name_ends_it(self):
        listing = self.listing("nul.cpio", newc(b"a\x00evil", b"alpha")
                               + TRAILER)
        self.assertEqual(listing.names, ["a"])

    def test_edges(self):
        listing = self.listing("empty.cpio", TRAILER)
        self.assertEqual((listing.total, listing.note), (0, ""))
        listing = self.listing("zero.cpio", newc(b"e", b"") + TRAILER)
        self.assertEqual((listing.names, listing.sizes, listing.note),
                         (["e"], [0], ""))
        # The last member's data ends at the end of the file (four bytes,
        # no padding): complete as a member, but the trailer is missing.
        listing = self.listing("end.cpio", newc(b"abc", b"abcd"))
        self.assertEqual(listing.names, ["abc"])
        self.assertNotIn("cut short", listing.note)
        self.assertDamagedAfter(listing, 1)

    def test_the_entry_limit_holds(self):
        with mock.patch.object(fallback, "MAX_TAR_ENTRIES", 3):
            listing = self.listing("many.cpio", b"".join(
                newc(b"f%d" % i, b"x") for i in range(10)) + TRAILER)
        self.assertEqual(listing.total, 4)
        self.assertIn("entry limit", listing.refused)


# -------------------------------------------------------------------- cab

def cab(names_sizes, first=44, declared=None):
    files = b"".join(struct.pack("<IIHHHH", size, 0, 0, 0, 0, 0x20)
                     + name + b"\x00" for name, size in names_sizes)
    header = struct.pack("<4sIIIIIBBHHHHH", b"MSCF", 0, 0, 0, first, 0, 3, 1,
                         1, len(names_sizes) if declared is None else declared,
                         0, 0, 0)
    return header + struct.pack("<IHH", 0, 0, 0) + files


class CabHeaderTest(_Archives):
    """The cabinet reader's numbers are unsigned and its count is 16 bits:
    what could go wrong was a file table said to start inside the header,
    which read the header as file entries."""

    def test_a_file_table_inside_the_header_is_damage(self):
        for first in (0, 20, 35):
            with self.subTest(first):
                listing = self.listing("back.cab", cab([(b"a.inf", 1)],
                                                       first=first))
                self.assertEqual(listing.names, [])
                self.assertIn("damaged", listing.note)

    def test_a_file_table_past_the_end(self):
        listing = self.listing("past.cab", cab([(b"a.inf", 1)],
                                               first=0xFFFFFFF0))
        self.assertEqual(listing.names, [])
        self.assertIn("0 could be read", listing.note)

    def test_edges(self):
        listing = self.listing("none.cab", cab([]))
        self.assertEqual((listing.total, listing.note), (0, ""))
        name = b"n" * 256                  # CB_MAX_FILENAME
        listing = self.listing("max.cab", cab([(name, 7)]))
        self.assertEqual((listing.names, listing.sizes), ([name.decode()], [7]))
        listing = self.listing("over.cab", cab([(name + b"n", 7)]))
        self.assertEqual(listing.names, [])
        self.assertIn("0 could be read", listing.note)
        listing = self.listing("count.cab", cab([(b"a", 1)], declared=65535))
        self.assertEqual(listing.names, ["a"])
        self.assertIn("declares 65535 files", listing.note)


# -------------------------------------------------------------------- tar

BLOCK = 512


def header(name, kind=tarfile.REGTYPE, size=0, fmt=tarfile.GNU_FORMAT):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = size
    return info.tobuf(format=fmt)[:BLOCK]


def padded(data):
    return data + b"\x00" * (-len(data) % BLOCK)


def file_member(name, data=b""):
    return header(name, size=len(data)) + padded(data)


def pax(records):
    return b"".join(b"%d %s=%s\n" % (_record_length(k, v), k, v)
                    for k, v in records)


def _record_length(key, value):
    body = len(key) + len(value) + 3            # " ", "=", "\n"
    length = body + 1
    while len(str(length)) + body != length:
        length += 1
    return length


def with_checksum(block):
    block = bytearray(block)
    block[148:156] = b" " * 8
    block[148:156] = b"%06o\x00 " % sum(block)
    return bytes(block)


def sparse_header(name, isextended, realsize=0):
    """An old GNU sparse member ("S"), no sparse entries in the header."""
    block = bytearray(header(name, tarfile.GNUTYPE_SPARSE))
    block[482] = 1 if isextended else 0
    block[483:495] = b"%011o\x00" % realsize
    return with_checksum(block)


def sparse_extension(more):
    block = bytearray(BLOCK)
    for i in range(21):
        block[i * 24:i * 24 + 12] = b"%011o\x00" % (i + 1)
        block[i * 24 + 12:i * 24 + 24] = b"%011o\x00" % 1
    block[504] = 1 if more else 0
    return bytes(block)


def end():
    return b"\x00" * (2 * BLOCK)


class TarHeaderBoundTest(_Archives):
    """tarfile reads a GNU long name or link, pax records and a sparse
    file's map whole, as the header declares them, before sift's bounds
    were asked: a 400 KB .tar.gz declaring a 400 MiB long name peaked at
    837 MiB. A chain of such headers recursed until RecursionError, a cut
    sparse extension raised IndexError, and a pax global header's records
    were copied into every member tarfile kept."""

    def test_a_declared_long_name_is_not_read_into_memory(self):
        for kind in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                     tarfile.XHDTYPE, tarfile.XGLTYPE,
                     tarfile.SOLARIS_XHDTYPE):
            with self.subTest(kind):
                declared = 8 * MIB
                body = pax([(b"k", b"v" * (declared - 11))]) \
                    if kind in (tarfile.XHDTYPE, tarfile.XGLTYPE,
                                tarfile.SOLARIS_XHDTYPE) else b"a" * declared
                self.assertEqual(len(body), declared)
                data = gzip.compress(header("././@LongLink", kind, declared)
                                     + body + file_member("a.txt", b"x")
                                     + end(), compresslevel=1)
                path = self.write("big.tar.gz", data)
                tracemalloc.start()
                try:
                    with deadline.limit(10):
                        listing = fallback._list_archive(path, data[:HEAD_BYTES])
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertIn("header limit", listing.refused)
                self.assertEqual(listing.format, "tar (gzip)")
                self.assertLess(peak, 3 * fallback.MAX_TAR_HEADER_BYTES)

    def test_the_reader_seeks_as_far_as_its_scan_limit(self):
        """A host application reads its members through this reader (header budget,
        forward only) and walks a tar past sift's listing bound, to its own
        limits: the scan limit is the caller's, MAX_TAR_SCAN_BYTES by
        default, None for none."""
        edge = fallback.MAX_TAR_SCAN_BYTES + tarfile.BLOCKSIZE
        view = fallback._TarBytes(io.BytesIO())
        self.assertEqual(view.seek(edge), edge)
        with self.assertRaisesRegex(fallback._TarLimit, "scan-size limit"):
            fallback._TarBytes(io.BytesIO()).seek(edge + 1)
        free = fallback._TarBytes(io.BytesIO(), scan_limit=None)
        self.assertEqual(free.seek(10 * edge), 10 * edge)
        with self.assertRaisesRegex(tarfile.ReadError, "backwards"):
            free.seek(edge)
        own = fallback._TarBytes(io.BytesIO(), scan_limit=1000)
        self.assertEqual(own.seek(1000 + tarfile.BLOCKSIZE), 1512)
        with self.assertRaisesRegex(fallback._TarLimit, r"maximum 1000\)"):
            own.seek(1513)
        with mock.patch.object(fallback, "MAX_TAR_SCAN_BYTES", 1000), \
                self.assertRaisesRegex(fallback._TarLimit, r"maximum 1000\)"):
            fallback._TarBytes(io.BytesIO()).seek(1513)  # as it stands then
        self.assertEqual(free.budget, fallback.MAX_TAR_HEADER_BYTES)
        with self.assertRaisesRegex(fallback._TarLimit, "header limit"):
            free.read(fallback.MAX_TAR_HEADER_BYTES + 1)

    def test_a_long_name_at_the_limit_is_read_and_one_block_more_is_not(self):
        with mock.patch.object(fallback, "MAX_TAR_HEADER_BYTES", 4096):
            name = "n" * 3000                  # 3072 bytes in blocks
            data = header("././@LongLink", tarfile.GNUTYPE_LONGNAME,
                          len(name) + 1) + padded(name.encode() + b"\x00") \
                + file_member("x", b"abc") + end()
            listing = self.listing("limit.tar", data)
            self.assertEqual((listing.names, listing.sizes), ([name], [3]))
            name = "n" * 3100                  # 3584 bytes in blocks
            data = header("././@LongLink", tarfile.GNUTYPE_LONGNAME,
                          len(name) + 1) + padded(name.encode() + b"\x00") \
                + file_member("x", b"abc") + end()
            listing = self.listing("over.tar", data)
            self.assertEqual(listing.names, [])
            self.assertIn("header limit", listing.refused)

    def test_normal_long_names_links_and_pax_records_still_list(self):
        buf = io.BytesIO()
        long_name = "d/" + "n" * 3000 + ".txt"
        with tarfile.open(fileobj=buf, mode="w:gz",
                          format=tarfile.GNU_FORMAT) as tf:
            for name in (long_name, "short.txt"):
                info = tarfile.TarInfo(name)
                info.size = 3
                tf.addfile(info, io.BytesIO(b"abc"))
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "t/" + "l" * 2000
            tf.addfile(info)
        listing = self.listing("gnu.tgz", buf.getvalue())
        self.assertEqual(listing.names, [long_name, "short.txt", "link"])
        self.assertEqual(listing.format, "tar (gzip)")
        buf = io.BytesIO()
        pax_name = "p/" + "é" * 2000 + ".txt"
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": "0" * 40}) as tf:
            info = tarfile.TarInfo(pax_name)
            info.size = 2
            info.pax_headers = {"comment": "x" * 20000}
            tf.addfile(info, io.BytesIO(b"hi"))
            info = tarfile.TarInfo("plain.txt")
            info.size = 1
            tf.addfile(info, io.BytesIO(b"!"))
        listing = self.listing("pax.tar", buf.getvalue())
        self.assertEqual((listing.names, listing.sizes),
                         ([pax_name, "plain.txt"], [2, 1]))
        self.assertEqual((listing.refused, listing.note), ("", ""))

    def test_an_old_gnu_sparse_member_lists_with_its_real_size(self):
        data = sparse_header("disk.img", True, realsize=4096) \
            + sparse_extension(more=False) + file_member("after", b"z") + end()
        listing = self.listing("sparse.tar", data)
        self.assertEqual((listing.names, listing.sizes),
                         (["disk.img", "after"], [4096, 1]))

    def test_a_chain_of_long_name_headers_is_not_an_exception(self):
        data = gzip.compress(file_member("first", b"1")
                             + header("././@LongLink",
                                      tarfile.GNUTYPE_LONGNAME) * 2000
                             + file_member("a.txt", b"x") + end())
        listing = self.listing("chain.tar.gz", data)
        self.assertEqual(listing.names, ["first"])
        self.assertTrue(listing.note or listing.refused)

    def test_a_cut_sparse_extension_is_damage(self):
        data = file_member("a.txt", b"x") + sparse_header("s", True)
        listing = self.listing("cut.tar", data)
        self.assertEqual(listing.names, ["a.txt"])
        self.assertDamagedAfter(listing, 1)

    def test_sparse_extensions_without_end_are_bounded(self):
        data = gzip.compress(file_member("a.txt", b"x")
                             + sparse_header("s", True)
                             + sparse_extension(more=True) * 8192)
        listing = self.listing("ext.tar.gz", data)
        self.assertEqual(listing.names, ["a.txt"])
        self.assertIn("header limit", listing.refused)

    def test_a_gnu_sparse_map_without_end_is_bounded(self):
        records = pax([(b"GNU.sparse.major", b"1"),
                       (b"GNU.sparse.minor", b"0")])
        data = gzip.compress(file_member("a.txt", b"x")
                             + header("pax", tarfile.XHDTYPE, len(records))
                             + padded(records) + header("s", size=0)
                             + b"99999999999\n" + b"1\n" * (2 * MIB))
        listing = self.listing("map.tar.gz", data)
        self.assertEqual(listing.names, ["a.txt"])
        self.assertIn("header limit", listing.refused)

    def test_global_pax_records_are_bounded(self):
        records = pax([(b"k%04d" % i, b"") for i in range(200)])
        data = file_member("first", b"1") \
            + header("g", tarfile.XGLTYPE, len(records)) + padded(records) \
            + b"".join(file_member("m%d" % i) for i in range(3)) + end()
        listing = self.listing("global.tar", data)
        self.assertEqual(listing.names, ["first"])
        self.assertIn("global", listing.refused)

    def test_a_git_archive_global_comment_is_fine(self):
        records = pax([(b"comment", b"0123456789abcdef0123456789abcdef01234567")])
        data = header("pax_global_header", tarfile.XGLTYPE, len(records)) \
            + padded(records) + file_member("README", b"hi") + end()
        listing = self.listing("git.tar", data)
        self.assertEqual((listing.names, listing.refused), (["README"], ""))

    def test_pax_records_are_not_kept_for_every_member(self):
        """tarfile kept every member it read, and each member its pax
        records: sixty members of a megabyte of records each held 452 MiB.
        Where TarFile takes no stream=True, the listing lets go of them
        itself."""
        blocks = []
        for i in range(20):
            records = pax([(b"k%06d" % (i * 4000 + j), b"")
                           for j in range(4000)])
            blocks += [header("x", tarfile.XHDTYPE, len(records)),
                       padded(records), file_member("m%d" % i)]
        data = b"".join(blocks) + end()
        for stream in (fallback._TAR_STREAM, {}):
            with self.subTest(stream=stream), \
                    mock.patch.object(fallback, "_TAR_STREAM", stream):
                tracemalloc.start()
                try:
                    listing = self.listing("records.tar", data)
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertEqual(listing.total, 20)
                self.assertLess(peak, 4 * MIB)

    def test_a_negative_size_is_damage(self):
        block = bytearray(header("d", tarfile.DIRTYPE))
        block[124:136] = b"\xff" * 12            # base-256: -1
        data = file_member("a.txt", b"x") + with_checksum(block) + end()
        listing = self.listing("neg.tar", data)
        self.assertEqual(listing.names, ["a.txt"])
        self.assertDamagedAfter(listing, 1)

    def test_an_empty_tar(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w"):
            pass
        listing = self.listing("empty.tar", buf.getvalue())
        self.assertEqual((listing.total, listing.refused, listing.note),
                         (0, "", ""))

    def test_the_tar_is_read_under_the_file_deadline(self):
        """gzip.GzipFile, which tarfile read a .tar.gz through, passes empty
        members, NUL padding and a member's name a byte at a time in loops
        that never look at the deadline."""
        path = self.write("many.tar.gz", gzip.compress(b"") * 2000
                          + gzip.compress(file_member("a", b"x") + end()))
        with mock.patch.object(fallback.deadline, "check",
                               wraps=deadline.check) as check:
            listing = fallback._tar_listing(path, "gzip")
        self.assertEqual(listing.names, ["a"])
        self.assertGreaterEqual(check.call_count, 2000)

    def test_a_tar_zst_that_opens_with_a_skippable_frame_still_lists(self):
        if zstd is None:
            self.skipTest("no zstd in this Python")
        skippable = b"\x50\x2a\x4d\x18" + struct.pack("<I", 4) + b"abcd"
        data = skippable + zstd.compress(file_member("a.txt", b"x") + end())
        listing = self.listing("p.tar.zst", data)
        self.assertEqual(listing.names, ["a.txt"])


# -------------------------------------------------------------------- streams

class StreamBoundTest(_Archives):
    """A single compressed stream is measured under 64 MiB out and 1,000:1,
    one stream after another. What was not bounded: the NUL padding between
    streams was skipped without a look at the deadline, and a stream's magic
    was asked of fewer bytes than it has where a chunk ended inside it --
    the rest of the file was then called trailing data."""

    def test_nul_padding_is_skipped_under_the_deadline(self):
        path = self.write("pad.gz", gzip.compress(b"hello\n")
                          + b"\x00" * (16 * 65536))
        with mock.patch.object(fallback.deadline, "check",
                               wraps=deadline.check) as check:
            size = fallback._stream_size(path, "gzip")
        self.assertEqual(size, (6, "complete", False, 0))
        self.assertGreaterEqual(check.call_count, 16)

    def test_a_stream_boundary_at_the_end_of_a_chunk(self):
        empty = bz2.compress(b"")                 # 14 bytes
        data = empty * 10_000 + bz2.compress(b"payload")
        listing = self.listing("many.bz2", data)
        self.assertEqual(listing.sizes, [7])
        self.assertEqual(listing.note, "")
        # One gzip member of 65,535 bytes: its end leaves one byte of the
        # next member's two-byte magic in the first 64 KiB read.
        import random
        noise = random.Random(1).randbytes(65_535)
        for size in range(65_535, 65_000, -1):
            first = gzip.compress(noise[:size], compresslevel=0, mtime=0)
            if len(first) == 65_535:
                noise = noise[:size]
                break
        self.assertEqual(len(first), 65_535)
        listing = self.listing("two.gz", first + gzip.compress(b"tail"))
        self.assertEqual(listing.sizes, [len(noise) + 4])
        self.assertEqual(listing.note, "")

    def test_zstd_skippable_frames_between_frames_are_not_trailing_data(self):
        if zstd is None:
            self.skipTest("no zstd in this Python")
        skippable = b"\x5a\x2a\x4d\x18" + struct.pack("<I", 3) + b"xyz"
        data = zstd.compress(b"one") + skippable + zstd.compress(b"two!")
        listing = self.listing("frames.zst", data)
        self.assertEqual((listing.sizes, listing.note), ([7], ""))

    def test_every_compressor_stops_a_multi_member_bomb(self):
        """Forty streams of 2 MiB of zeros each, one after another."""
        zeros = b"\x00" * (2 * MIB)
        forms = {"gzip": lambda: gzip.compress(zeros),
                 "bzip2": lambda: bz2.compress(zeros),
                 "xz": lambda: lzma.compress(zeros)}
        if zstd is not None:
            forms["zstd"] = lambda: zstd.compress(zeros)
        for form, compress in forms.items():
            with self.subTest(form):
                listing = self.listing("bomb.img", compress() * 40)
                self.assertEqual(listing.format, form)
                self.assertEqual(listing.sizes, [None])
                self.assertTrue(listing.refused)
                self.assertIn("at least", listing.note)


# -------------------------------------------------------------------- all

class NeverAnExceptionTest(_Archives):
    """Every archive cut short, and with its header bytes changed one at a
    time to a sign, a negative number, a blank, 0xff, a digit or a NUL, is
    listed -- a note, maybe, and never an exception or a loop."""

    CHANGES = (b"-", b"-9", b"+1", b" ", b"\xff", b"9", b"\x00")

    def archives(self):
        """(name, data, the offsets changed): every byte of the small ones;
        for the tar, the number fields of each header (100 to 160) and the
        type; for the .tar.gz, every byte of the compressed file."""
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w", format=tarfile.GNU_FORMAT) as tf:
            for name in ("a/" + "n" * 120, "b.txt"):
                info = tarfile.TarInfo(name)
                info.size = 3
                tf.addfile(info, io.BytesIO(b"abc"))
        tar = tar.getvalue()[:6 * BLOCK]
        fields = [block + at for block in (0, 2 * BLOCK, 4 * BLOCK)
                  for at in range(100, 160)]
        small = {
            "x.a": ar([member(b"debian-binary", b"2.0\n"),
                       member(b"#1/12", b"long_name.oELF"),
                       member(b"data.tar.xz", b"y" * 11)]),
            "x.cpio": newc(b"a.txt", b"alpha") + newc(b"b", b"12") + TRAILER,
            "o.cpio": odc(b"a.txt", b"alpha") + odc(b"b", b"12")
            + odc(b"TRAILER!!!", b""),
            "x.cab": cab([(b"a.inf", 120), (b"b.sys", 4096)]),
        }
        for name, data in small.items():
            yield name, data, range(8, len(data))
        yield "x.tar", tar, fields
        packed = gzip.compress(tar + end(), mtime=0)
        yield "x.tar.gz", packed, range(10, len(packed))

    def test_cut_and_changed_headers(self):
        for name, data, offsets in self.archives():
            for at in offsets:
                samples = [data[:at]] + [data[:at] + change
                                         + data[at + len(change):]
                                         for change in self.CHANGES]
                for sample in samples:
                    with self.subTest(name=name, at=at):
                        listing = self.listing(name, sample, seconds=5)
                        self.assertIsInstance(listing, fallback._Listing)


if __name__ == "__main__":
    unittest.main()
