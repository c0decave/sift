"""Legacy Office: a .xls (BIFF5/BIFF8), a .doc (Word 6/95 and 97+) and a .ppt
read for their text, tables and sheets (sift/extract/legacy.py).

Three kinds of file are used. The real ones in tests/data/legacy were written
by LibreOffice, an independent writer, and are compared with what
LibreOffice itself reads from them. The synthetic ones are built here, byte
by byte from [MS-XLS], [MS-DOC] and [MS-PPT], for the paths no real file in
the corpus takes: a shared-string table split over CONTINUE records, a
string result after a formula, BIFF5, a Word 6/95 text, a piece table of
8-bit and UTF-16 pieces. The hostile ones are damaged on purpose.
"""
import struct
import tempfile
import unittest
from pathlib import Path

from _fixtures import make_xlsx
from test_active_content import cfb

from sift.detect import detect
from sift.extract import extract, fallback, iter_table, legacy, read_table
from sift.extract import office

DATA = Path(__file__).parent / "data" / "legacy"


# ---- builders: BIFF --------------------------------------------------------

def rec(rid, payload=b""):
    return struct.pack("<HH", rid, len(payload)) + payload


def bof(dt, version=0x0600):
    return rec(0x0809, struct.pack("<HHHHII", version, dt, 0x0DBB, 0x07CC, 0,
                                   6))


EOF = rec(0x000A)


def xl_string(text, wide_length=True):
    wide = any(ord(c) > 0xFF for c in text)
    head = struct.pack("<H" if wide_length else "<B", len(text))
    head += bytes([1 if wide else 0])
    return head + (text.encode("utf-16-le") if wide else text.encode("latin-1"))


def labelsst(row, col, index, ixfe=0):
    return rec(0x00FD, struct.pack("<HHHI", row, col, ixfe, index))


def label8(row, col, text, ixfe=0):
    return rec(0x0204, struct.pack("<HHH", row, col, ixfe) + xl_string(text))


def label5(row, col, text, codec="cp1252", ixfe=0):
    raw = text.encode(codec)
    return rec(0x0204, struct.pack("<HHHH", row, col, ixfe, len(raw)) + raw)


def number(row, col, value, ixfe=0):
    return rec(0x0203, struct.pack("<HHHd", row, col, ixfe, value))


def rk_int(row, col, n, x100=False, ixfe=0):
    raw = ((n << 2) & 0xFFFFFFFF) | 2 | (1 if x100 else 0)
    return rec(0x027E, struct.pack("<HHHI", row, col, ixfe, raw))


def rk_float(row, col, value, ixfe=0):
    raw = (struct.unpack("<Q", struct.pack("<d", value))[0] >> 32) \
        & 0xFFFFFFFC
    return rec(0x027E, struct.pack("<HHHI", row, col, ixfe, raw))


def mulrk(row, first, raws, ixfe=0):
    body = b"".join(struct.pack("<HI", ixfe, raw) for raw in raws)
    return rec(0x00BD, struct.pack("<HH", row, first) + body
               + struct.pack("<H", first + len(raws) - 1))


def boolerr(row, col, value, is_error=False):
    return rec(0x0205, struct.pack("<HHHBB", row, col, 0, value,
                                   1 if is_error else 0))


def formula(row, col, result8, ixfe=0):
    return rec(0x0006, struct.pack("<HHH", row, col, ixfe) + result8
               + struct.pack("<HIH", 0, 0, 0))


def formula_number(row, col, value, ixfe=0):
    return formula(row, col, struct.pack("<d", value), ixfe)


def formula_string(row, col):
    return formula(row, col, bytes([0, 0, 0, 0, 0, 0, 0xFF, 0xFF]))


def string_record(text):
    return rec(0x0207, xl_string(text))


def sst(strings, limit=8224):
    """The SST record and the CONTINUE records that follow it, cut at
    ``limit`` bytes the way Excel cuts them: a cut inside a string's
    characters begins the continuation with the flag byte of the rest."""
    segments = [bytearray(struct.pack("<II", len(strings), len(strings)))]
    for text in strings:
        wide = any(ord(c) > 0xFF for c in text)
        head = struct.pack("<HB", len(text), 1 if wide else 0)
        if len(segments[-1]) + len(head) + 1 > limit:
            segments.append(bytearray())
        segments[-1] += head
        data = text.encode("utf-16-le") if wide else text.encode("latin-1")
        width = 2 if wide else 1
        at = 0
        while at < len(data):
            room = limit - len(segments[-1])
            if room < width:
                segments.append(bytearray([1 if wide else 0]))
                room = limit - 1
            take = min(room, len(data) - at) // width * width
            segments[-1] += data[at:at + take]
            at += take
    out = rec(0x00FC, bytes(segments[0]))
    for segment in segments[1:]:
        out += rec(0x003C, bytes(segment))
    return out


def sheet(*records, dt=0x10):
    return bof(dt) + b"".join(records) + EOF


def workbook_stream(sheets, strings=(), biff=8, date1904=False,
                    formats=None, xfs=(), codepage=1252, states=None,
                    kinds=None, sst_limit=8224, globals_extra=b""):
    """The Workbook (BIFF8) or Book (BIFF5) stream: ``sheets`` is a list of
    (name, substream bytes), ``xfs`` the number format of each XF record
    (the cell's ``ixfe`` is the index)."""
    version = 0x0600 if biff == 8 else 0x0500

    def head(offsets):
        out = bof(0x05, version)
        out += rec(0x0042, struct.pack("<H", codepage))
        out += rec(0x0022, struct.pack("<H", 1 if date1904 else 0))
        for index, code in (formats or {}).items():
            if biff == 8:
                out += rec(0x041E, struct.pack("<H", index)
                           + xl_string(code))
            else:
                raw = code.encode("cp1252")
                out += rec(0x041E, struct.pack("<HB", index, len(raw)) + raw)
        for index in xfs:
            out += rec(0x00E0, struct.pack("<HH", 0, index) + bytes(16))
        for n, ((name, _), offset) in enumerate(zip(sheets, offsets)):
            state = (states or {}).get(n, 0)
            kind = (kinds or {}).get(n, 0)
            if biff == 8:
                raw = struct.pack("<IBB", offset, state, kind) \
                    + bytes([len(name), 0]) + name.encode("latin-1")
            else:
                raw = struct.pack("<IBB", offset, state, kind) \
                    + bytes([len(name)]) + name.encode("latin-1")
            out += rec(0x0085, raw)
        if biff == 8 and strings:
            out += sst(strings, sst_limit)
        out += globals_extra + EOF
        return out

    size = len(head([0] * len(sheets)))
    offsets, at = [], size
    for _, body in sheets:
        offsets.append(at)
        at += len(body)
    return head(offsets) + b"".join(body for _, body in sheets)


def xls_file(*args, biff=8, **kw):
    name = "Workbook" if biff == 8 else "Book"
    return cfb([((name,), workbook_stream(*args, biff=biff, **kw))])


# ---- builders: Word and PowerPoint -----------------------------------------

_FIB_PAIRS = 93
_FIB_LEN = 154 + _FIB_PAIRS * 8


def word_file(parts, lengths=None, nfib=0x00C1, flags=0x0200, table=True):
    """A Word 97+ document: ``parts`` is a list of (text, compressed) pieces;
    ``lengths`` the characters of each part of the document in file order
    (main text, footnotes, headers, comments, endnotes, text boxes)."""
    body, pcds, cps = b"", b"", [0]
    for text, compressed in parts:
        raw = text.encode("cp1252") if compressed \
            else text.encode("utf-16-le")
        at = _FIB_LEN + len(body)
        fc = ((at * 2) | 0x40000000) if compressed else at
        body += raw
        cps.append(cps[-1] + len(text))
        pcds += struct.pack("<HIH", 0, fc, 0)
    plc = struct.pack(f"<{len(cps)}I", *cps) + pcds
    clx = b"\x02" + struct.pack("<I", len(plc)) + plc
    rglw = [0] * 22
    given = lengths or {"main": cps[-1]}
    for name, index in (("main", 3), ("footnotes", 4), ("headers", 5),
                        ("comments", 7), ("endnotes", 8), ("textboxes", 9),
                        ("header_textboxes", 10)):
        rglw[index] = given.get(name, 0)
    fib = struct.pack("<HHHHHH", 0xA5EC, nfib, 0, 0x0409, 0, flags) + bytes(20)
    fib += struct.pack("<H", 14) + bytes(28)
    fib += struct.pack("<H", 22) + struct.pack("<22I", *rglw)
    fib += struct.pack("<H", _FIB_PAIRS)
    pairs = [(0, 0)] * _FIB_PAIRS
    pairs[33] = (0, len(clx))
    fib += b"".join(struct.pack("<II", *pair) for pair in pairs)
    assert len(fib) == _FIB_LEN
    entries = [(("WordDocument",), fib + body)]
    if table:
        entries.append(((("1Table" if flags & 0x0200 else "0Table"),), clx))
    return cfb(entries)


def word6_file(text):
    raw = text.encode("cp1252")
    head = bytearray(0x200)
    struct.pack_into("<HH", head, 0, 0xA5EC, 0x0065)
    struct.pack_into("<II", head, 0x18, 0x200, 0x200 + len(raw))
    return cfb([(("WordDocument",), bytes(head) + raw)])


def ppt_rec(rtype, data=b"", ver=0, inst=0):
    return struct.pack("<HHI", (inst << 4) | ver, rtype, len(data)) + data


def container(rtype, children, inst=0):
    return ppt_rec(rtype, b"".join(children), ver=0xF, inst=inst)


def chars(text):
    return ppt_rec(0x0FA0, text.encode("utf-16-le"))


def latin(text):
    return ppt_rec(0x0FA8, text.encode("cp1252"))


PERSIST = ppt_rec(0x03F3, bytes(20))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def path(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def rows(self, path, sheet=None, limit=1000):
        det = detect(path)
        page = read_table(path, det, sheet=sheet, limit=limit)
        self.assertIsNone(page.error, page.error)
        return page.header, page.rows, page

    def all_rows(self, path, sheet=None):
        """(header, every row with the header first, the stream): sift
        takes a first row of text for the table's header."""
        with iter_table(path, detect(path), sheet=sheet) as stream:
            body = list(stream)
            self.assertIsNone(stream.error, stream.error)
            header = list(stream.header)
            return header, ([header] if header else []) + body, stream


# ---- .xls: the real files --------------------------------------------------

class RealWorkbook(Base):
    """book.xls: LibreOffice's BIFF8, against LibreOffice's own reading and
    against the values the file was made from."""

    def setUp(self):
        super().setUp()
        self.file = DATA / "book.xls"

    def test_kind_stays_the_one_sift_always_gave(self):
        self.assertEqual(detect(self.file).kind, "office-legacy")
        self.assertEqual(legacy.stream_kind(self.file), "xls")

    def test_sheets_in_file_order_hidden_ones_included(self):
        _h, _r, page = self.rows(self.file)
        self.assertEqual(page.sheets, ["People", "Zahlen", "Many", "Hidden"])

    def test_people_rows_header_dates_booleans_and_formulas(self):
        header, rows, page = self.rows(self.file, "People")
        self.assertEqual(header, ["Name", "Amount", "Joined", "Active", "Note"])
        self.assertEqual(rows[0], ["Müller", "1234.5", "2024-03-05", "1",
                                   "Grüße € ☃"])
        self.assertEqual(rows[1][:4], ["O'Brien", "-7", "1999-12-31", "0"])
        self.assertTrue(rows[1][4].startswith("Lorem ipsum dolor sit amet, "))
        self.assertEqual(rows[2], ["Σίσυφος", "0.1", "2024-02-29", "1"])
        # the cached results of the formulas, not recalculated
        self.assertEqual(rows[3], ["Total", "1227.6", "2469"])
        self.assertEqual(rows[4], ["hidden row secret", "1"])
        self.assertEqual(rows[5], ["", "", "", "", "far right"])
        self.assertEqual(page.total, 6)
        self.assertFalse(page.has_more)

    def test_numbers_as_general_shows_them(self):
        _header, flat, _page = self.all_rows(self.file, "Zahlen")
        self.assertEqual(flat[0], ["0.1", "1E+21", "123456789012", "-0.000001",
                                   "100", "3.14159265358979"])

    def test_hidden_sheet_is_read(self):
        _h, body, _s = self.all_rows(self.file, "Hidden")
        self.assertEqual(body, [["secret-in-hidden-sheet", "42"]])

    def test_shared_strings_across_continue_records(self):
        _h, rows, page = self.rows(self.file, "Many")
        self.assertEqual(page.total, 300)
        want = [f"unique string number {i:05d} with some padding to make it "
                f"long enough" for i in range(300)]
        got = [r[0] for r in rows]
        # the first row may be judged a header by sift's rule; either way
        # every string is intact and in order
        self.assertEqual(([page.header[0]] if page.header else []) + got, want)

    def test_page_at_the_end_has_no_more(self):
        det = detect(self.file)
        page = read_table(self.file, det, sheet="Many", offset=295, limit=50)
        self.assertEqual(len(page.rows), 5)
        self.assertFalse(page.has_more)
        self.assertEqual(page.rows[-1][0].split()[3], "00299")

    def test_the_stream_equals_the_pages(self):
        det = detect(self.file)
        header, whole, _s = self.all_rows(self.file, "People")
        page = read_table(self.file, det, sheet="People", limit=1000)
        self.assertEqual(whole, [page.header] + page.rows)
        self.assertEqual(header, page.header)

    def test_extract_renders_the_workbook_for_a_summary(self):
        ex = extract(self.file, detect(self.file), 10_000_000)
        self.assertEqual(ex.kind, "office-legacy")
        self.assertTrue(ex.text.startswith("# Spreadsheet\n## Sheet: People"))
        self.assertIn("Müller\t1234.5\t2024-03-05", ex.text)
        self.assertIn("## Sheet: Hidden [hidden]", ex.text)
        self.assertIn("secret-in-hidden-sheet", ex.text)
        self.assertEqual(ex.metadata["sheet_count"], 4)
        self.assertEqual(ex.metadata["hidden_sheets"], ["Hidden"])
        self.assertEqual(ex.metadata["format"], "Excel BIFF8")
        self.assertIn("recalculated", ex.metadata["computed_cells"])
        self.assertTrue(ex.truncated)            # Many: 300 rows, 200 read
        self.assertEqual(ex.metadata["active_content"]["classes"], [])

    def test_matches_libreoffice_own_reading(self):
        want = [["Name", "Amount", "Joined", "Active", "Note"],
                ["Müller", "1234.5", "2024-03-05", "1", "Grüße € ☃"]]
        _h, rows, page = self.rows(self.file, "People", limit=1)
        self.assertEqual([page.header] + rows[:1], want)


class RealDocuments(Base):
    LETTER = ("Überschrift Eins\n"
              "First paragraph with umlauts: äöüß and a euro sign €.\n"
              "Greek: Σίσυφος and a snowman ☃ force Unicode pieces.\n"
              "cell-a1\tcell-b1\n"
              "cell-a2\tcell-b2\n")

    def test_unicode_document_equals_libreoffice_text(self):
        path = DATA / "letter.doc"
        self.assertEqual(legacy.stream_kind(path), "doc")
        ex = extract(path, detect(path), 10_000_000)
        self.assertTrue(ex.text.startswith("# Document\n"))
        self.assertEqual(ex.text[len("# Document\n"):][:len(self.LETTER)],
                         self.LETTER)

    def test_latin1_only_document(self):
        path = DATA / "plain.doc"
        ex = extract(path, detect(path), 10_000_000)
        self.assertEqual(ex.text, "# Document\nPlain ASCII and latin1 only: "
                                  "Grüße aus Köln.\nSecond line.")

    def test_a_document_is_no_table(self):
        page = read_table(DATA / "letter.doc", detect(DATA / "letter.doc"))
        self.assertEqual(page.rows, [])
        self.assertIn("no table", page.error)


# ---- .xls: synthetic -------------------------------------------------------

class WorkbookCells(Base):
    def test_every_cell_record_type(self):
        body = sheet(
            labelsst(0, 0, 0), label8(0, 1, "plain label"),
            number(0, 2, 2.5), rk_int(0, 3, 7), rk_int(0, 4, 1234, True),
            rk_float(0, 5, 0.5),
            mulrk(1, 0, [(5 << 2) | 2, (6 << 2) | 2, (7 << 2) | 2]),
            boolerr(2, 0, 1), boolerr(2, 1, 0), boolerr(2, 2, 7, True),
            boolerr(2, 3, 0x2A, True),
            formula_number(3, 0, 42.0),
            formula_string(3, 1), string_record("formula text"),
            formula(3, 2, bytes([1, 0, 1, 0, 0, 0, 0xFF, 0xFF])),
            formula(3, 3, bytes([2, 0, 0x17, 0, 0, 0, 0xFF, 0xFF])),
            formula(3, 4, bytes([3, 0, 0, 0, 0, 0, 0xFF, 0xFF])))
        path = self.path("c.xls", xls_file([("S", body)], ["shared"]))
        _h, flat, _p = self.all_rows(path, "S")
        self.assertEqual(flat[0], ["shared", "plain label", "2.5", "7",
                                   "12.34", "0.5"])
        self.assertEqual(flat[1], ["5", "6", "7"])
        self.assertEqual(flat[2], ["TRUE", "FALSE", "#DIV/0!", "#N/A"])
        self.assertEqual(flat[3], ["42", "formula text", "TRUE", "#REF!"])

    def test_string_result_after_array_record_still_found(self):
        body = sheet(formula_string(0, 0), rec(0x0221, bytes(12)),
                     string_record("after the array record"))
        path = self.path("a.xls", xls_file([("S", body)]))
        _h, rows, _p = self.all_rows(path)
        self.assertEqual(rows, [["after the array record"]])

    def test_shared_strings_cut_inside_characters_and_both_encodings(self):
        strings = ["a" * 40, "Ωmega" * 12, "b" * 55, "ß" * 30, "日本語" * 20,
                   "tail"]
        cells = [labelsst(i, 0, i) for i in range(len(strings))]
        for limit in (64, 100, 8224):
            data = xls_file([("S", sheet(*cells))], strings, sst_limit=limit)
            path = self.path(f"s{limit}.xls", data)
            _h, rows, _p = self.all_rows(path)
            self.assertEqual([r[0] for r in rows], strings, limit)

    def test_dates_by_cell_format(self):
        # xf 0 General, xf 1 built-in 14 (date), xf 2 custom date, xf 3 custom
        # that only looks like a unit, xf 4 built-in time 21
        body = sheet(number(0, 0, 45356.0, 1), number(0, 1, 45356.75, 1),
                     number(0, 2, 36526.0, 2), number(0, 3, 12.0, 3),
                     number(0, 4, 0.5, 4), number(0, 5, 45356.0, 0),
                     number(0, 6, 61.0, 1), number(0, 7, 1.0, 1))
        data = xls_file([("S", body)], formats={170: "dd.mm.yyyy",
                                                171: '0.0 "days"'},
                        xfs=[0, 14, 170, 171, 21])
        _h, rows, _p = self.all_rows(self.path("d.xls", data))
        self.assertEqual(rows[0], ["2024-03-05", "2024-03-05 18:00:00",
                                   "2000-01-01", "12", "12:00:00", "45356",
                                   "1900-03-01", "1900-01-01"])

    def test_date_1904_and_the_1900_leap_day(self):
        body = sheet(number(0, 0, 0.0, 1), number(0, 1, 60.0, 1))
        path = self.path("d1904.xls", xls_file([("S", body)], date1904=True,
                                               xfs=[0, 14]))
        self.assertEqual(self.all_rows(path)[1], [["1904-01-01",
                                                   "1904-03-01"]])
        path = self.path("d1900.xls", xls_file(
            [("S", sheet(number(0, 0, 60.0, 1)))], xfs=[0, 14]))
        self.assertEqual(self.all_rows(path)[1], [["1900-02-29"]])

    def test_biff5_book_stream_and_codepage(self):
        body = bof(0x10, 0x0500) + label5(0, 0, "Grüße") \
            + label5(0, 1, "Да", "cp1251") + number(1, 0, 3.0) \
            + EOF
        path = self.path("b5.xls", xls_file([("Blatt", body)], biff=5,
                                            codepage=1251))
        _h, rows, page = self.all_rows(path)
        self.assertEqual(legacy.stream_kind(path), "xls")
        # codepage 1251 decodes both byte strings: Grüße's ü and ß are
        # Cyrillic there; the point is that the bytes are not read as UTF-8
        self.assertEqual(rows[1], ["3"])
        self.assertEqual(rows[0][1], "Да")
        self.assertEqual(page.sheets, ["Blatt"])

    def test_chart_sheet_is_no_table_and_is_said(self):
        sheets = [("Data", sheet(label8(0, 0, "x"))),
                  ("Chart1", sheet(dt=0x20))]
        path = self.path("ch.xls", xls_file(sheets, kinds={1: 2}))
        _h, _r, page = self.rows(path)
        self.assertEqual(page.sheets, ["Data"])
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.metadata["sheet_count"], 2)
        self.assertIn("not a worksheet", ex.metadata["sheets"][1]["error"])

    def test_notes_and_text_boxes_are_in_the_text(self):
        txo = rec(0x01B6, bytes(10) + struct.pack("<HHI", 13, 0, 0))
        text = rec(0x003C, b"\x00" + b"hidden note!!")
        body = sheet(label8(0, 0, "x"), txo + text)
        path = self.path("n.xls", xls_file([("S", body)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text.count("hidden note!!"), 1)
        self.assertIn("## Notes and text boxes\nhidden note!!", ex.text)


class WorkbookShapes(Base):
    def test_empty_sheet_has_no_rows(self):
        path = self.path("e.xls", xls_file([("S", sheet())]))
        header, rows, page = self.rows(path)
        self.assertEqual((header, rows, page.total), ([], [], 0))

    def test_gap_rows_are_blank_rows(self):
        body = sheet(label8(0, 0, "top"), label8(4, 1, "far"))
        path = self.path("g.xls", xls_file([("S", body)]))
        self.assertEqual(self.all_rows(path)[1],
                         [["top"], [], [], [], ["", "far"]])

    def test_rows_out_of_order_come_back_in_order(self):
        body = sheet(label8(3, 0, "d"), label8(0, 0, "a"), label8(1, 1, "b"),
                     label8(0, 1, "c"))
        path = self.path("o.xls", xls_file([("S", body)]))
        self.assertEqual(self.all_rows(path)[1],
                         [["a", "c"], ["", "b"], [], ["d"]])

    def test_last_cell_of_a_biff8_sheet(self):
        body = sheet(label8(65535, 255, "corner"))
        path = self.path("m.xls", xls_file([("S", body)]))
        _h, rows, _p = self.all_rows(path)
        # sift drops the blank rows that lead a table
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][255], "corner")
        self.assertEqual(len(rows[0]), 256)
        ex = extract(path, detect(path), 10**7)
        self.assertIn("1 rows × 256 cols", ex.text)

    def test_sheet_asked_for_by_index_and_unknown_name(self):
        sheets = [("A", sheet(label8(0, 0, "one"))),
                  ("B", sheet(label8(0, 0, "two")))]
        path = self.path("two.xls", xls_file(sheets))
        det = detect(path)
        self.assertEqual(read_table(path, det, sheet=1).rows, [["two"]])
        self.assertEqual(read_table(path, det, sheet="A").rows, [["one"]])
        with self.assertRaises(office.NoSuchSheet):
            read_table(path, det, sheet="C")

    def test_many_columns_are_cut_at_the_render_width(self):
        cells = [label8(0, c, str(c)) for c in range(250)]
        path = self.path("w.xls", xls_file([("S", sheet(*cells))]))
        _h, rows, _p = self.all_rows(path)
        self.assertEqual(len(rows[0]), 250)
        ex = extract(path, detect(path), 10**7)
        self.assertIn("250 cols", ex.text)


class WorkbookSad(Base):
    def test_encrypted_workbook_is_said_not_read(self):
        data = xls_file([("S", sheet(label8(0, 0, "secret")))],
                        globals_extra=rec(0x002F, bytes(54)))
        path = self.path("enc.xls", data)
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "")
        self.assertTrue(ex.metadata["encrypted"])
        self.assertIn("encrypted", " ".join(ex.notes))
        page = read_table(path, detect(path))
        self.assertIn("encrypted", page.error)
        self.assertEqual(page.rows, [])

    def test_workbook_without_a_bof_is_unreadable_not_a_crash(self):
        path = self.path("nobof.xls", cfb([(("Workbook",), b"\x01" * 600)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "")
        self.assertIn("BOF", ex.metadata["error"])

    def test_biff4_workbook_is_refused_in_words(self):
        data = rec(0x0409, struct.pack("<HHHH", 0x0400, 0x10, 0, 0)) + EOF
        path = self.path("b4.xls", cfb([(("Workbook",), data)]))
        page = read_table(path, detect(path))
        self.assertIn("BIFF2-4", page.error)

    def test_sheet_offset_outside_the_stream(self):
        data = workbook_stream([("S", sheet(label8(0, 0, "x")))])
        broken = bytearray(data)
        # BOUNDSHEET: lbPlyPos is its first four bytes; point it past the end
        at = bytes(broken).index(struct.pack("<HH", 0x0085, 0)[:2])
        struct.pack_into("<I", broken, at + 4, 0x7FFFFFF0)
        path = self.path("off.xls", cfb([(("Workbook",), bytes(broken))]))
        page = read_table(path, detect(path))
        self.assertEqual(page.rows, [])
        self.assertIn("could not read the spreadsheet", page.error)

    def test_record_cut_off_in_the_middle_does_not_raise(self):
        data = xls_file([("S", sheet(labelsst(0, 0, 0), number(0, 1, 1.0)))],
                        ["x" * 50])
        for cut in range(0, len(data), 97):
            path = self.path("cut.xls", data[:cut])
            det = detect(path)
            extract(path, det, 10**7)             # never raises
            if det.kind == "office-legacy":
                read_table(path, det)

    def test_every_byte_flipped_in_a_small_workbook_never_raises(self):
        stream = workbook_stream([("S", sheet(labelsst(0, 0, 0),
                                              number(0, 1, 1.5),
                                              formula_string(1, 0),
                                              string_record("s")))],
                                 ["strong"])
        for i in range(0, len(stream), 3):
            mutated = bytearray(stream)
            mutated[i] ^= 0xFF
            path = self.path("fz.xls", cfb([(("Workbook",), bytes(mutated))]))
            det = detect(path)
            extract(path, det, 10**7)
            if det.kind == "office-legacy":
                read_table(path, det)
                with iter_table(path, det) as stream_:
                    list(stream_)

    def test_absurd_counts_cost_what_the_file_holds(self):
        # SST claims 4 billion strings, MULRK claims nothing, a record claims
        # to be 65535 bytes long and the file ends
        sst_bad = rec(0x00FC, struct.pack("<II", 0xFFFFFFFF, 0xFFFFFFFF)
                      + xl_string("only one"))
        data = workbook_stream([("S", sheet(labelsst(0, 0, 0)))],
                               globals_extra=sst_bad)
        path = self.path("abs.xls", cfb([(("Workbook",), data
                                          + b"\xff\xff\xff\xff")]))
        ex = extract(path, detect(path), 10**7)
        self.assertIsNotNone(ex)

    def test_a_compound_file_larger_than_the_bound_is_not_read(self):
        path = self.path("big.xls", xls_file([("S", sheet())]))
        real = legacy.MAX_FILE
        legacy.MAX_FILE = 100
        try:
            ex = extract(path, detect(path), 10**7)
        finally:
            legacy.MAX_FILE = real
        self.assertEqual(ex.text, "")
        self.assertIn("larger than", ex.metadata["error"])


# ---- .doc ------------------------------------------------------------------

class WordText(Base):
    def test_compressed_and_utf16_pieces_in_order(self):
        data = word_file([("Grüße ", True), ("Σίσυφος\r", False),
                          ("end\r", True)])
        path = self.path("a.doc", data)
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "# Document\nGrüße Σίσυφος\nend")
        self.assertEqual(ex.metadata["format"], "Word binary document")

    def test_fields_keep_their_shown_result_only(self):
        text = ("see \x13 HYPERLINK \"http://example.test/\" \x14shown "
                "text\x15 and \x13 PAGE \x15 here\r"
                "nested \x13 IF \x13 DATE \x14today\x15 \x14yes\x15!\r")
        path = self.path("f.doc", word_file([(text, False)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "# Document\nsee shown text and  here\n"
                                  "nested yes!")

    def test_table_cells_become_tabs_and_rows_lines(self):
        text = "one\x07two\x07\x07three\x07four\x07\x07after\r"
        path = self.path("t.doc", word_file([(text, True)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "# Document\none\ttwo\nthree\tfour\nafter")

    def test_footnotes_headers_comments_textboxes_are_sections(self):
        main, foot, head, note, box = ("Body\r", "Footnote text\r",
                                       "Header text\r", "A comment\r",
                                       "In a box\r")
        parts = [(main + foot + head + note + box, True)]
        lengths = {"main": len(main), "footnotes": len(foot),
                   "headers": len(head), "comments": len(note),
                   "textboxes": len(box)}
        path = self.path("s.doc", word_file(parts, lengths))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text.split("\n\n"), [
            "# Document\nBody", "## Footnotes\nFootnote text",
            "## Headers\nHeader text", "## Comments\nA comment",
            "## Text boxes\nIn a box"])

    def test_zero_table_stream_selected_by_the_flag(self):
        data = word_file([("zero table\r", True)], flags=0x0000)
        path = self.path("z.doc", data)
        ex = extract(path, detect(path), 10**7)
        self.assertIn("zero table", ex.text)

    def test_word6_text_is_one_run(self):
        path = self.path("w6.doc", word6_file("Old Word für DOS\r\rtwo"))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "# Document\nOld Word für DOS\n\ntwo")

    def test_special_characters_are_dropped_or_mapped(self):
        text = "a\x01b\x02c\x08d\x0be\x0cf\x1eg\x1fh\x00i\r"
        path = self.path("sp.doc", word_file([(text, True)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "# Document\nabcd\ne\nf-ghi")

    def test_hidden_text_is_in_the_stream_and_so_in_the_text(self):
        # Word's hidden text is a character property: the characters are
        # there. A reader that drops what Word hides would hide a secret.
        path = self.path("h.doc", word_file([("visible hidden-run\r", True)]))
        self.assertIn("hidden-run", extract(path, detect(path), 10**7).text)


class WordSad(Base):
    def test_encrypted_document(self):
        data = word_file([("secret\r", True)], flags=0x0300)
        path = self.path("e.doc", data)
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "")
        self.assertTrue(ex.metadata["encrypted"])

    def test_missing_table_stream(self):
        path = self.path("nt.doc", word_file([("x\r", True)], table=False))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "")
        self.assertIn("1Table", ex.metadata["error"])

    def test_piece_table_out_of_order(self):
        data = word_file([("abc", True), ("def", True)])
        # swap the two character positions in the piece table
        raw = bytearray(data)
        marker = struct.pack("<3I", 0, 3, 6)
        at = bytes(raw).find(marker)
        self.assertGreater(at, 0)
        raw[at:at + 12] = struct.pack("<3I", 0, 6, 3)
        path = self.path("oo.doc", bytes(raw))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "")
        self.assertIn("not in order", ex.metadata["error"])

    def test_piece_pointing_outside_the_stream_yields_no_crash(self):
        data = word_file([("abc", True)])
        raw = bytearray(data)
        fc = struct.pack("<I", ((_FIB_LEN * 2) | 0x40000000))
        at = bytes(raw).find(fc)
        self.assertGreater(at, 0)
        raw[at:at + 4] = struct.pack("<I", 0x7FFFFFFF)
        path = self.path("out.doc", bytes(raw))
        extract(path, detect(path), 10**7)

    def test_not_a_word_header(self):
        path = self.path("nh.doc", cfb([(("WordDocument",), b"\x00" * 600)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "")
        self.assertIn("Word header", ex.metadata["error"])

    def test_every_byte_flipped_never_raises(self):
        data = word_file([("Grüße\r", True), ("Σ\r", False)])
        for i in range(0, len(data), 7):
            mutated = bytearray(data)
            mutated[i] ^= 0xFF
            path = self.path("fz.doc", bytes(mutated))
            extract(path, detect(path), 10**7)


# ---- .ppt ------------------------------------------------------------------

class PowerPoint(Base):
    def stream(self):
        slides = container(0x0FF0, [PERSIST, chars("Title one"),
                                    latin("Bullet für one"), PERSIST,
                                    chars("Title two")], inst=0)
        masters = container(0x0FF0, [PERSIST, chars("Click to edit Master")],
                            inst=1)
        notes = container(0x0FF0, [PERSIST, chars("Speaker note")], inst=2)
        document = container(0x03E8, [slides, masters, notes])
        master = container(0x03F8, [chars("Master title style")])
        slide1 = container(0x03EE, [container(0x040C, [container(
            0xF00D, [chars("Free text box")])])])
        notes_page = container(0x03F0, [container(0x040C, [container(
            0xF00D, [chars("Note on the page")])])])
        return document + master + slide1 + notes_page

    def test_slide_text_notes_and_shapes_masters_left_out(self):
        path = self.path("p.ppt", cfb([(("PowerPoint Document",),
                                        self.stream())]))
        self.assertEqual(legacy.stream_kind(path), "ppt")
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "\n\n".join([
            "# Presentation\n## Slide 1\nTitle one\nBullet für one\n"
            "Free text box", "## Slide 2\nTitle two",
            "## Notes\nSpeaker note\nNote on the page"]))
        self.assertEqual(ex.metadata["slide_count"], 2)
        self.assertNotIn("Master", ex.text)

    def test_symbol_only_note_placeholders_are_dropped(self):
        notes = container(0x0FF0, [PERSIST, chars("*"), chars("real note")],
                          inst=2)
        stream = container(0x03E8, [notes])
        path = self.path("n.ppt", cfb([(("PowerPoint Document",), stream)]))
        ex = extract(path, detect(path), 10**7)
        self.assertEqual(ex.text, "# Presentation\n## Notes\nreal note")

    def test_truncated_record_stream(self):
        stream = self.stream()
        for cut in range(0, len(stream), 13):
            path = self.path("t.ppt", cfb([(("PowerPoint Document",),
                                            stream[:cut])]))
            extract(path, detect(path), 10**7)

    def test_container_claiming_more_than_the_stream(self):
        bogus = struct.pack("<HHI", 0x000F, 0x03E8, 0xFFFFFFF0) \
            + chars("inside")
        path = self.path("b.ppt", cfb([(("PowerPoint Document",), bogus)]))
        ex = extract(path, detect(path), 10**7)
        self.assertIn("inside", ex.text)


# ---- everything else stays as it was ---------------------------------------

class OtherCompoundFiles(Base):
    def test_a_compound_file_with_no_office_stream_is_the_fallbacks(self):
        data = cfb([(("__properties_version1.0",), b"x" * 64),
                    (("__substg1.0_0037001F",), "subject".encode("utf-16-le"))])
        path = self.path("mail.msg", data)
        det = detect(path)
        self.assertEqual(det.kind, "office-legacy")
        self.assertIsNone(legacy.stream_kind(path))
        before = fallback.extract(path, det, 10**7)
        after = extract(path, det, 10**7)
        self.assertEqual((after.text, after.notes, after.metadata.get("error")),
                         (before.text, before.notes, before.metadata.get(
                             "error")))

    def test_read_table_still_refuses_what_is_no_table(self):
        path = self.path("x.txt", b"hello\n")
        with self.assertRaises(ValueError):
            read_table(path, detect(path))

    def test_xlsx_is_untouched(self):
        path = self.dir / "a.xlsx"
        make_xlsx(path)
        det = detect(path)
        self.assertEqual(det.kind, "xlsx")
        self.assertIsNone(read_table(path, det).error)


if __name__ == "__main__":
    unittest.main()


# ---- bounds a hostile workbook meets ---------------------------------------

class WorkbookBounds(Base):
    """What a planted workbook costs is bounded by what a real one can hold (measured 2026-10-15: one cell in column
    65,000 on each of 3,000 rows, a 43 KiB file, took 18.5 s a page; 1.2 million two-character shared strings, a
    5.8 MiB file, 77 MiB)."""

    def test_a_cell_past_the_formats_last_column_is_not_read_and_said(self):
        import time
        body = sheet(*[label8(r, 65000, "far") for r in range(3000)], label8(0, 0, "near"), label8(1, 255, "edge"))
        path = self.path("wide.xls", xls_file([("S", body)]))
        started = time.monotonic()
        page = read_table(path, detect(path), limit=10)
        self.assertLess(time.monotonic() - started, 2.0)
        rows = ([page.header] if page.header else []) + page.rows
        self.assertEqual(rows[0][0], "near")
        self.assertEqual(rows[1][255], "edge")
        self.assertTrue(all(len(r) <= 256 for r in rows))
        self.assertIn("past column 256", " ".join(page.notes))
        ex = extract(path, detect(path), 10**7)
        self.assertNotIn("far", ex.text)
        self.assertIn("past column 256", " ".join(ex.notes))

    def test_the_shared_strings_past_the_bound_are_not_read_and_said(self):
        strings = [f"s{i}" for i in range(10)]
        body = sheet(*[labelsst(i, 0, i) for i in range(10)])
        path = self.path("sst.xls", xls_file([("S", body)], strings))
        real = legacy.MAX_STRINGS
        legacy.MAX_STRINGS = 4
        try:
            page = read_table(path, detect(path), limit=100)
            ex = extract(path, detect(path), 10**7)
        finally:
            legacy.MAX_STRINGS = real
        shown = [r[0] for r in ([page.header] if page.header else []) + page.rows if r]
        self.assertEqual(shown, ["s0", "s1", "s2", "s3"])
        self.assertIn("shared strings", " ".join(page.notes))
        self.assertIn("shared strings", " ".join(ex.notes))

    def test_the_shared_strings_characters_are_bounded_too(self):
        strings = ["a" * 100, "b" * 100, "c" * 100]
        path = self.path("chars.xls", xls_file([("S", sheet(*[labelsst(i, 0, i) for i in range(3)]))], strings))
        real = legacy.MAX_STRING_CHARS
        legacy.MAX_STRING_CHARS = 150
        try:
            page = read_table(path, detect(path), limit=100)
        finally:
            legacy.MAX_STRING_CHARS = real
        shown = [r[0] for r in ([page.header] if page.header else []) + page.rows if r]
        self.assertEqual(shown, ["a" * 100])

    def test_a_workbook_larger_than_a_tables_bound_is_no_table_but_still_text(self):
        path = self.path("big.xls", xls_file([("S", sheet(label8(0, 0, "cell")))]))
        real = legacy.MAX_TABLE_FILE
        legacy.MAX_TABLE_FILE = 100
        try:
            page = read_table(path, detect(path))
            with iter_table(path, detect(path)) as stream:
                rows = list(stream)
                error = stream.error
            ex = extract(path, detect(path), 10**7)
        finally:
            legacy.MAX_TABLE_FILE = real
        self.assertEqual(page.rows, [])
        self.assertIn("not read as a table", page.error)
        self.assertEqual(rows, [])
        self.assertIn("not read as a table", error)
        self.assertIn("cell", ex.text)                       # the summary's bound is the compound file's own

    def test_the_tables_bound_is_an_xlsx_members(self):
        self.assertEqual(legacy.MAX_TABLE_FILE, office._MAX_MEMBER)
        self.assertLessEqual(legacy.MAX_TABLE_FILE, legacy.MAX_FILE)
