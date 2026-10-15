"""Pages that cost what they show, and the rows of a table as they are read.

sift 0.1.55 read a whole csv (up to max_bytes), a whole sheet's XML tree or
a whole ods for every page, and then counted every row after the page for
the total: page 0 of a 50 MiB csv took 0.71 s, of a 200,000-row xlsx 2.36 s
at 614 MiB (measured 2026-09-28 in a container). A page now
reads the table up to the row after it and stops; iter_table hands out
every row as it is read, for a caller that needs them all.

What a page holds must not change: every page here is compared with
0.1.55's (tests/_table_reference.py). Where the new page stopped before the
end of its table its total is None -- 0.1.55 counted it -- and a fact only
the rest of the file holds (a later line the CSV reader cannot parse) is
reported by the page that reaches it.
"""
import codecs
import csv
import gc
import io
import os
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

from sift import deadline
from sift.detect import detect
from sift.extract import office, tabular
from sift.extract.base import TablePage

import _table_reference as reference

SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
ODS_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
          'xmlns:dc="http://purl.org/dc/elements/1.1/" '
          'xmlns:foo="urn:example:foo"')

#: The notes a page that stopped early cannot have: they are about rows
#: after it (0.1.55 counted those rows; the stream does not read them).
TAIL_NOTES = ("the rest of the file is not valid CSV",
              "ran out of time counting the rows after this page")


def read_table(path, det=None, **kw):
    from sift.extract import read_table as current
    return current(path, det or detect(path), **kw)


def compare(test, new: TablePage, old: TablePage) -> str:
    """How ``new`` stands to 0.1.55's ``old``: "same", or "early" -- it
    stopped at its page, so its total is None and the notes about rows past
    it are not on it. Anything else fails ``test``."""
    if new == old:
        return "same"
    test.assertIsNone(old.error, f"0.1.55 said {old.error!r}; now {new}")
    for f in fields(TablePage):
        if f.name not in ("total", "notes", "truncated"):
            test.assertEqual(getattr(old, f.name), getattr(new, f.name),
                             f.name)
    test.assertTrue(new.has_more, f"a page that reached the end must be "
                                  f"0.1.55's: {new} != {old}")
    test.assertIsNone(new.total)
    kept = [n for n in old.notes if n in new.notes]
    test.assertEqual(kept, new.notes, "the notes are 0.1.55's, in order")
    missing = [n for n in old.notes if n not in new.notes]
    test.assertTrue(all(n.startswith(TAIL_NOTES) for n in missing), missing)
    if new.truncated != old.truncated:
        test.assertTrue(old.truncated and any(
            n.startswith(TAIL_NOTES[0]) for n in missing))
    return "early"


def pages_of(total: int):
    """(offset, limit) pairs that cross every edge of a table of ``total``
    data rows: the first rows, the page cap, the last row and past it."""
    out = {(0, 1), (0, 5), (0, 100), (3, 7), (0, 1000), (0, 5000)}
    for at in (1, 99, 100, 101, 511, 512, 513, total - 1, total, total + 3):
        if at >= 0:
            out |= {(at, 1), (at, 5), (at, 100)}
    return sorted(out)


class _Files(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.counts = {}

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data if isinstance(data, bytes) else
                         data.encode("utf-8"))
        return path

    def assert_pages(self, path, det=None, *, sheets=(None,), **kw):
        """Every page of ``path`` against 0.1.55's."""
        det = det or detect(path)
        for sheet in sheets:
            try:
                whole = reference.read_table(path, det, sheet=sheet,
                                             limit=10 ** 9, **kw)
            except ValueError:
                with self.assertRaises(ValueError):
                    read_table(path, det, sheet=sheet, **kw)
                continue
            total = whole.total if whole.total is not None else \
                len(whole.rows) + 1000
            for offset, limit in pages_of(total):
                with self.subTest(path=path.name, sheet=sheet, offset=offset,
                                  limit=limit, **kw):
                    old = reference.read_table(path, det, sheet=sheet,
                                               offset=offset, limit=limit,
                                               **kw)
                    new = read_table(path, det, sheet=sheet, offset=offset,
                                     limit=limit, **kw)
                    kind = compare(self, new, old)
                    self.counts[kind] = self.counts.get(kind, 0) + 1


# -- csv and tsv ------------------------------------------------------------
def _csv_text(rows, delimiter=",", end="\n"):
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=delimiter, lineterminator=end).writerows(rows)
    return buffer.getvalue()


def csv_corpus():
    """(name, bytes): csv and tsv files of every shape the reader knows."""
    numbered = [["id", "name", "city"]] + [
        [str(i), f"name {i}", f"city {i % 7}"] for i in range(1, 251)]
    quoted = [["id", "text", "note"]] + [
        [str(i), f'says "hi", {i}', f"line one\nline two {i}\r\nthree"]
        for i in range(1, 201)]
    umlauts = [["Name", "Stadt", "Betrag"]] + [
        [f"Müller {i}", "Köln €", f"{i},50"] for i in range(1, 301)]
    yield "plain.csv", _csv_text(numbered).encode()
    yield "crlf.csv", _csv_text(numbered, end="\r\n").encode()
    yield "quoted.csv", _csv_text(quoted).encode()
    yield "bom.csv", b"\xef\xbb\xbf" + _csv_text(umlauts).encode()
    yield "utf8.csv", _csv_text(umlauts).encode()
    yield "cp1252.csv", _csv_text(umlauts).encode("cp1252")
    yield "latin1.csv", _csv_text(umlauts).encode("cp1252") + b"x\x81y\n"
    late = _csv_text(numbered).encode()
    yield "late-bad-byte.csv", (_csv_text(umlauts[:3]).encode() + late
                                + b"last,\xff,row\n")
    yield "ascii-then-bad-byte.csv", late + b"last,\xfc,row\n"
    yield "utf16.csv", codecs.BOM_UTF16_LE + _csv_text(numbered).encode(
        "utf-16-le")
    yield "sep.csv", ("\n  \nsep=;\n" + _csv_text(umlauts, ";")).encode()
    yield "mac.csv", _csv_text(numbered).replace("\n", "\r").encode()
    yield "blanks.csv", (",,\n\n" + _csv_text(numbered).replace(
        "\n", "\n\n   \n", 40) + ",,\n,,\n").encode()
    yield "preamble.csv", ("Monthly report\nGenerated 2026-09-28\n"
                           + _csv_text(numbered)).encode()
    repeated = numbered[:60] + [numbered[0]] + numbered[60:] + [numbered[0]]
    yield "repeated-header.csv", _csv_text(repeated).encode()
    ragged = [r + ["extra"] if int(r[0] or 0) % 9 == 0 else r
              for r in numbered[1:]]
    yield "ragged.csv", _csv_text(numbered[:1] + ragged).encode()
    wide = [[f"c{i}" for i in range(600)]] + [
        [str(r * i) for i in range(600)] for r in range(1, 30)]
    yield "wide.csv", _csv_text(wide).encode()
    yield "huge-cell.csv", _csv_text(
        [["id", "blob"], ["1", "x" * 300_000], ["2", "small"],
         ["3", "y" * 150_000], ["4", "z" * 60_000]]).encode()
    yield "unparseable.csv", (_csv_text(numbered[:150]) + "150,bad\rfield,x\n"
                              + _csv_text(numbered[150:])).encode()
    yield "headerless.csv", _csv_text(
        [[str(i), str(i * i), f"2026-01-{i % 28 + 1:02d}"]
         for i in range(1, 300)]).encode()
    yield "only-header.csv", b"id,name,city\n"
    yield "tabs.tsv", _csv_text(
        [["a", "b, c"]] + [[str(i), f"x, {i}"] for i in range(200)],
        "\t").encode()
    yield "nul.csv", _csv_text(numbered[:40]).replace(
        "name 7", "na\x00me").encode()
    yield "wide-chars.csv", _csv_text(
        [["k", "v"]] + [[f"🙂{i}", "€ü" * (i % 5)] for i in range(400)]
    ).encode()


class CsvParityTest(_Files):
    """Every page of every csv shape, read whole and read as a stream in
    pieces as small as a few bytes, is 0.1.55's page."""

    def _each(self, **kw):
        for name, data in csv_corpus():
            path = self.write(name, data)
            self.assert_pages(path, **kw)

    def test_the_default_reading(self):
        self._each()

    def test_read_in_pieces_of_a_few_bytes(self):
        for piece in (7, 64, 4099):
            with mock.patch.object(tabular, "_WHOLE_BYTES", 16), \
                    mock.patch.object(tabular, "_PIECE_BYTES", piece):
                self._each()
        # Every file's records fit the header scan: the rest is counted.
        self.assertEqual({"same"}, set(self.counts))

    def test_cut_at_max_bytes_anywhere(self):
        with mock.patch.object(tabular, "_WHOLE_BYTES", 16), \
                mock.patch.object(tabular, "_PIECE_BYTES", 13):
            for name, data in csv_corpus():
                path = self.write(name, data)
                for cut in (17, 18, 19, 20, 777, len(data) // 2,
                            len(data) - 3, len(data) - 2, len(data) - 1):
                    if 0 < cut < len(data):
                        self.assert_pages(path, max_bytes=cut)

    def test_classic_mac_line_ends_longer_than_the_head(self):
        """A text without any "\\n" is read with its carriage returns as line
        ends; the head the stream decides on is 65,536 characters, and a file
        longer than that is one line to it."""
        text = "id,name\r" + "".join(f"{i},name {i}\r" for i in range(9000))
        path = self.write("mac-long.csv", text)
        self.assertGreater(len(text), tabular._HEAD_CHARS)
        with mock.patch.object(tabular, "_WHOLE_BYTES", 16), \
                mock.patch.object(tabular, "_PIECE_BYTES", 4099):
            self.assert_pages(path)
            page = read_table(path, offset=8990, limit=100)
        self.assertEqual(["8990", "name 8990"], page.rows[0])
        self.assertEqual(9000, page.total)

    def test_a_multibyte_character_cut_by_max_bytes(self):
        """decode_text trims up to three bytes of a cut file to find a
        character boundary; a stream must decide the same trim."""
        body = ("id,v\n" + "".join(f"{i},{'é€🙂'[i % 3] * 3}\n"
                                   for i in range(300))).encode()
        path = self.write("cut.csv", body)
        with mock.patch.object(tabular, "_WHOLE_BYTES", 16), \
                mock.patch.object(tabular, "_PIECE_BYTES", 5):
            for cut in range(len(body) - 40, len(body)):
                self.assert_pages(path, max_bytes=cut)
            bad = self.write("bad-tail.csv", body[:-10] + b"\xff\n" + b"a,b\n")
            for cut in range(len(body) - 16, len(body) - 4):
                self.assert_pages(bad, max_bytes=cut)


class CsvReadsOnlyItsPageTest(_Files):
    """A page of a csv read as a stream, whose records go on past the
    header scan, reads the file up to the row after it."""

    def setUp(self):
        super().setUp()
        streamed = mock.patch.object(tabular, "_WHOLE_BYTES", 16)
        streamed.start()
        self.addCleanup(streamed.stop)

    def _big(self, rows=40_000, tail=""):
        text = "id,name\n" + "".join(f"{i},name{i}\n" for i in range(rows))
        return self.write("big.csv", text + tail)

    def test_page_zero_parses_only_the_rows_it_needs(self):
        path = self._big()
        seen = []
        real = tabular._records_of_lines

        def counted(lines, delimiter):
            for row in real(lines, delimiter):
                seen.append(1)
                yield row
        with mock.patch.object(tabular, "_records_of_lines", counted):
            page = read_table(path, offset=0, limit=10)
        self.assertEqual(10, len(page.rows))
        self.assertTrue(page.has_more)
        self.assertIsNone(page.total)
        self.assertLessEqual(len(seen), tabular.MAX_ROWS_SCAN + 1 + 11)

    def test_a_far_page_parses_up_to_it(self):
        path = self._big()
        page = read_table(path, offset=30_000, limit=10)
        self.assertEqual([str(30_000), "name30000"], page.rows[0])
        self.assertTrue(page.has_more)

    def test_a_line_past_the_page_that_csv_cannot_parse(self):
        """0.1.55 reported it on every page; it is now the page that reaches
        it that says so, and the rows before it are all reachable."""
        path = self._big(tail="x,bad\rfield\n1,2\n")
        first = read_table(path, offset=0, limit=10)
        self.assertFalse(first.truncated)
        self.assertEqual([], first.notes)
        last = read_table(path, offset=39_995, limit=100)
        self.assertEqual(5, len(last.rows))
        self.assertTrue(last.truncated)
        self.assertTrue(any("not valid CSV" in n for n in last.notes))
        self.assertIsNone(last.total)

    def test_the_last_page_knows_the_total(self):
        path = self._big(rows=3000)
        page = read_table(path, offset=2990, limit=100)
        self.assertEqual(3000, page.total)
        self.assertFalse(page.has_more)

    def test_the_deadline_is_checked_while_the_page_is_read(self):
        path = self._big()
        with self.assertRaises(deadline.DeadlineExceeded):
            read_table(path, offset=30_000, time_budget=1e-9)

    def test_the_read_for_the_encoding_is_under_the_deadline(self):
        """A byte past ASCII makes the rest of the file be read once for
        decode_text's choice: 25 MB at most, and every piece of it checked
        against the clock."""
        path = self._big(tail="x,Ü\n")
        real = deadline.check

        def expire(where=""):
            if where == "a table's encoding":
                raise deadline.DeadlineExceeded(1, where)
            return real(where)
        with mock.patch.object(tabular, "_WHOLE_BYTES", 16), \
                mock.patch.object(tabular, "_PIECE_BYTES", 4096), \
                mock.patch.object(deadline, "check", expire):
            with self.assertRaises(deadline.DeadlineExceeded):
                read_table(path, offset=39_990, limit=100)
        page = read_table(path, offset=39_999, limit=100)
        self.assertEqual([["39999", "name39999"], ["x", "Ü"]], page.rows)


# -- xlsx -------------------------------------------------------------------
def _ref(col):
    return office._column_name(col)


def xlsx_bytes(sheets, *, shared=True, sst_extra="", sheet_xml=None,
               stored=False, names=None):
    """An xlsx of ``sheets`` [(name, rows)]: cells as shared strings (with a
    rich-text run and a phonetic run among them) or inline strings; a cell
    "=n" is a number with a formula, "#n" a plain number."""
    strings, index = [], {}

    def si(value):
        if value not in index:
            index[value] = len(strings)
            strings.append(value)
        return index[value]

    def cell(ref, value):
        if value.startswith("#"):
            return f'<c r="{ref}"><v>{value[1:]}</v></c>'
        if value.startswith("="):
            return f'<c r="{ref}"><f>A1*2</f><v>{value[1:]}</v></c>'
        if shared:
            return f'<c r="{ref}" t="s"><v>{si(value)}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'

    parts = []
    for _name, rows in sheets:
        body = "".join(
            f'<row r="{r + 1}">' + "".join(
                cell(f"{_ref(c)}{r + 1}", v) for c, v in enumerate(row) if v)
            + "</row>" for r, row in enumerate(rows))
        parts.append(f'<worksheet xmlns="{SS}"><sheetData>{body}'
                     "</sheetData></worksheet>")
    buffer = io.BytesIO()
    method = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w", method) as z:
        z.writestr("xl/workbook.xml",
                   f'<workbook xmlns="{SS}" xmlns:r="{REL}"><sheets>' + "".join(
                       f'<sheet name="{n}" r:id="rId{i}"/>'
                       for i, (n, _) in enumerate(sheets, 1))
                   + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<Relationships xmlns="{PKG_REL}">' + "".join(
                       f'<Relationship Id="rId{i}" '
                       f'Target="worksheets/sheet{i}.xml"/>'
                       for i in range(1, len(sheets) + 1))
                   + "</Relationships>")
        if shared and strings:
            items = "".join(
                f"<si><r><t>{s[:1]}</t></r><r><t>{s[1:]}</t></r>"
                f"<rPh><t>ph</t></rPh></si>" if i % 5 == 1
                else f"<si><t>{s}</t></si>" for i, s in enumerate(strings))
            z.writestr("xl/sharedStrings.xml",
                       f'<sst xmlns="{SS}">{items}{sst_extra}</sst>')
        for i, part in enumerate(parts, 1):
            data = part if sheet_xml is None else sheet_xml(i, part)
            z.writestr(f"xl/worksheets/sheet{i}.xml", data)
    return buffer.getvalue()


def _rows(count, start=1, width=3):
    return [[f"id{i}", f"name {i}", f"#{i * 1.5}"][:width]
            for i in range(start, start + count)]


def xlsx_corpus():
    head = [["id", "name", "amount"]]
    yield "shared.xlsx", xlsx_bytes([("Data", head + _rows(300))])
    yield "inline.xlsx", xlsx_bytes([("Data", head + _rows(300))],
                                    shared=False)
    yield "sheets.xlsx", xlsx_bytes([("First", head + _rows(20)),
                                     ("Empty", []),
                                     ("Third", _rows(40)),
                                     ("Blank rows", [[], [], ["x", "y"]]
                                      + _rows(30))])
    yield "dates.xlsx", xlsx_bytes([("D", [["when", "n"]] + [
        [f"#{44197 + i}", f"={i}"] for i in range(250)])])
    wide = [[f"h{c}" for c in range(520)]] + [
        [f"v{r}.{c}" for c in range(520)] for r in range(12)]
    yield "wide.xlsx", xlsx_bytes([("W", wide)])
    sparse = [["a", "", "", "d"]] + [
        ["", f"b{i}", "", ""] if i % 3 else [f"a{i}", "", "", f"d{i}"]
        for i in range(200)]
    yield "sparse.xlsx", xlsx_bytes([("S", sparse)])
    yield "headerless.xlsx", xlsx_bytes([("N", [[f"#{i}", f"#{i * i}"]
                                                for i in range(300)])])

    def odd_refs(_i, part):
        # A cell without a reference, a reference outside A-Z, a shared
        # string index that names nothing, a row nested in a cell (read in
        # document order, the outer row first).
        return part.replace('<c r="A3"', '<c').replace(
            'r="B4"', 'r="ß4"').replace(
            "<v>2</v></c></row>", '<v>999999</v></c></row>', 1).replace(
            '<row r="9">', '<row r="9"><c r="A9"><row r="99"><c r="A1" '
                           't="inlineStr"><is><t>inner</t></is></c></row>'
                           '<v>1</v></c>', 1)
    yield "odd.xlsx", xlsx_bytes([("O", head + _rows(250))],
                                 sheet_xml=odd_refs)


def xlsx_damaged():
    """(name, bytes, original bytes): workbooks damaged in one place."""
    good = xlsx_bytes([("Data", [["id", "name", "amount"]] + _rows(400))])
    yield "late-xml.xlsx", xlsx_bytes(
        [("Data", [["id", "name", "amount"]] + _rows(400))],
        sheet_xml=lambda i, p: p.replace("</sheetData>", "<<</sheetData>")), \
        good
    yield "early-xml.xlsx", xlsx_bytes(
        [("Data", [["id", "name", "amount"]] + _rows(400))],
        sheet_xml=lambda i, p: p.replace('<row r="2">', '<row r="2"<')), None
    yield "late-sst.xlsx", xlsx_bytes(
        [("Data", [["id", "name", "amount"]] + _rows(400))],
        sst_extra="<si><t>x</si>"), good
    stored = xlsx_bytes([("Data", [["id", "name", "amount"]] + _rows(400))],
                        stored=True)
    # A value changed in a stored member: the XML stays well-formed, the
    # CRC does not match, and zipfile says so at the member's end.
    yield "crc.xlsx", stored.replace(b"<v>598.5</v>", b"<v>598.6</v>"), \
        stored


class XlsxParityTest(_Files):
    """Every page of every xlsx shape is 0.1.55's page, read whole (a member
    of at most _WHOLE_MEMBER bytes) and read as a stream."""

    def _each(self, corpus):
        for name, data in corpus():
            path = self.write(name, data)
            sheets = [None, 0, 1, 2, 3, 4, "Third", "Nope", -1]
            self.assert_pages(path, sheets=sheets)

    def test_read_whole(self):
        self._each(xlsx_corpus)
        self.assertEqual({"same"}, set(self.counts))

    def test_read_as_a_stream(self):
        for piece in (97, 4096):
            with mock.patch.object(office, "_WHOLE_MEMBER", 0), \
                    mock.patch.object(office, "_PIECE", piece):
                self._each(xlsx_corpus)
        self.assertGreater(self.counts.get("early", 0), 0)

    def test_a_sheet_cut_at_the_member_cap(self):
        """0.1.55 parsed the first _MAX_MEMBER bytes back to their last row
        and said so on every page; the stream stops at the same row. Its
        shared strings cut at the cap were an error on every page."""
        table = [["id", "name", "amount"]] + _rows(900)
        inline = self.write("capped.xlsx", xlsx_bytes([("Data", table)],
                                                      shared=False))
        shared = self.write("capped-sst.xlsx", xlsx_bytes([("Data", table)]))
        # Rows inside an element _closed does not close: the cut tree does
        # not parse, and 0.1.55's page was an error without the cut note.
        unclosed = self.write("capped-unclosed.xlsx", xlsx_bytes(
            [("Data", table)], shared=False, sheet_xml=lambda i, p: p.replace(
                "<sheetData>", "<sheetData><extra>").replace(
                "</sheetData>", "</extra></sheetData>")))
        for cap in (2000, 5000, 5001, 20_000):
            for whole in (0, 1 << 20):
                with mock.patch.object(office, "_MAX_MEMBER", cap), \
                        mock.patch.object(office, "_WHOLE_MEMBER", whole), \
                        mock.patch.object(office, "_PIECE", 333):
                    self.assert_pages(inline)
                    self.assert_pages(shared)
                    # Streamed, the cut's failure is the page's that
                    # reaches it; that page is 0.1.55's, the note not on it.
                    end = read_table(unclosed, offset=10 ** 6)
                    self.assertEqual(reference.read_table(
                        unclosed, detect(unclosed), offset=10 ** 6), end)
                    self.assertEqual([], end.notes)
                    if whole:
                        self.assert_pages(unclosed)
                    page = read_table(inline)
                    self.assertTrue(page.truncated)
                    self.assertTrue(any("sheet larger than" in n
                                        for n in page.notes))
                    self.assertTrue(read_table(shared).error)

    def test_damage_read_whole_is_0155s_error(self):
        for name, data, _good in xlsx_damaged():
            path = self.write(name, data)
            self.assert_pages(path)
        self.assertEqual({"same"}, set(self.counts) - {"early"})

    def test_damage_past_the_page_is_reported_by_the_page_reaching_it(self):
        """Streamed, a page before the damage holds the rows 0.1.55 showed
        of the undamaged file; the page that reaches it is 0.1.55's error
        page."""
        with mock.patch.object(office, "_WHOLE_MEMBER", 0), \
                mock.patch.object(office, "_PIECE", 512):
            for name, data, good in xlsx_damaged():
                path = self.write(name, data)
                old_last = reference.read_table(path, detect(path),
                                                offset=395, limit=100)
                new_last = read_table(path, offset=395, limit=100)
                with self.subTest(name):
                    self.assertEqual(old_last, new_last)
                    self.assertTrue(new_last.error)
                    first = read_table(path, offset=0, limit=10)
                    if good is None:
                        self.assertEqual(reference.read_table(
                            path, detect(path), offset=0, limit=10), first)
                        continue
                    self.assertIsNone(first.error)
                    good_path = self.write("good-" + name, good)
                    expect = reference.read_table(good_path,
                                                  detect(good_path),
                                                  offset=0, limit=10)
                    self.assertEqual(expect.rows, first.rows)
                    self.assertEqual(expect.header, first.header)


class XlsxReadsOnlyItsPageTest(_Files):
    def test_rows_before_the_offset_are_counted_not_read(self):
        data = xlsx_bytes([("Data", [["id", "name", "amount"]]
                            + _rows(5000))])
        path = self.write("big.xlsx", data)
        calls = []
        real = office._sheet_row

        def counted(*a, **kw):
            calls.append(1)
            return real(*a, **kw)
        with mock.patch.object(office, "_WHOLE_MEMBER", 0), \
                mock.patch.object(office, "_sheet_row", counted):
            page = read_table(path, offset=4000, limit=10)
        self.assertEqual(["id4001", "name 4001", "6001.5"], page.rows[0])
        self.assertTrue(page.has_more)
        self.assertLessEqual(len(calls), 9 + 11)

    def test_shared_strings_are_read_as_far_as_the_page_needs(self):
        data = xlsx_bytes([("Data", [["id", "name", "amount"]]
                            + _rows(3000))], sst_extra="<si><t>x</si>")
        path = self.write("sst.xlsx", data)
        with mock.patch.object(office, "_WHOLE_MEMBER", 0):
            page = read_table(path, offset=0, limit=10)
        self.assertIsNone(page.error)
        self.assertEqual(["id1", "name 1", "1.5"], page.rows[0])


# -- ods --------------------------------------------------------------------
def ods_bytes(body, *, stored=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED if stored
                         else zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument."
                               "spreadsheet", compress_type=zipfile.ZIP_STORED)
        z.writestr("content.xml",
                   f'<?xml version="1.0"?><office:document-content {ODS_NS}>'
                   f"<office:body><office:spreadsheet>{body}"
                   "</office:spreadsheet></office:body>"
                   "</office:document-content>")
    return buffer.getvalue()


def _cell(text, attrs=""):
    return (f"<table:table-cell{attrs}><text:p>{text}</text:p>"
            "</table:table-cell>")


def _row(cells, attrs=""):
    return (f"<table:table-row{attrs}>" + "".join(_cell(c) for c in cells)
            + "</table:table-row>")


def _table(name, rows_xml):
    named = f' table:name="{name}"' if name else ""
    return f"<table:table{named}>{rows_xml}</table:table>"


def ods_corpus():
    rows = _row(["id", "name"]) + "".join(_row([str(i), f"n{i}"])
                                          for i in range(300))
    yield "plain.ods", ods_bytes(_table("Kalkulation", rows))
    yield "tables.ods", ods_bytes(
        _table("Empty", _row(["", " "]))
        + _table("", _row(["a", "b"]) + _row(["1", "2"]))
        + _table("Second", rows) + _table("Third", _row(["x"])))
    yield "repeats.ods", ods_bytes(_table("R", (
        _row(["h1", "h2"]) + _row(["same", "x"], ' table:number-rows-'
                                                 'repeated="250"')
        + _row(["", ""], ' table:number-rows-repeated="1000"')
        + _row(["after", "y"]) + _row(["again"], ' table:number-rows-'
                                                 'repeated="300"')
        + "<table:table-row>" + _cell("c", ' table:number-columns-'
                                           'repeated="3"')
        + "<table:covered-table-cell><text:p>cv</text:p>"
          "</table:covered-table-cell></table:table-row>")))
    yield "groups.ods", ods_bytes(_table("G", (
        "<table:table-header-rows>" + _row(["Name", "Qty"])
        + "</table:table-header-rows><table:table-row-group>"
        + "".join(_row([f"g{i}", str(i)]) for i in range(120))
        + "<table:table-rows>" + _row(["inner", "1"]) + "</table:table-rows>"
        + "</table:table-row-group><foo:table-row-group>" + _row(["foo"])
        + "</foo:table-row-group>" + _row(["END", "0"]))))
    nested = _table("Inner", _row(["i1"]) + _row(["i2"]))
    yield "nested.ods", ods_bytes(_table("Outer", (
        "<table:table-row><table:table-cell>" + nested
        + "</table:table-cell></table:table-row>"
        + "".join(_row([f"o{i}"]) for i in range(30)))))
    yield "empty-first.ods", ods_bytes(_table("Outer", (
        "<table:table-row><table:table-cell>"
        + _table("Inner", _row(["deep"]))
        + "</table:table-cell></table:table-row>")))
    yield "many.ods", ods_bytes("".join(
        _table(f"T{i}", _row([f"v{i}"])) for i in range(40)))
    yield "no-table.ods", ods_bytes("<table:table/>")
    yield "wide.ods", ods_bytes(_table("W", _row(["h"] * 3) + (
        "<table:table-row>" + _cell("x", ' table:number-columns-repeated='
                                         '"5000"') + _cell("END")
        + "</table:table-row>") * 20))


class OdsParityTest(_Files):
    def test_every_page(self):
        for whole, piece in ((1 << 20, 1 << 16), (0, 1 << 16), (0, 101)):
            with mock.patch.object(office, "_PIECE", piece), \
                    mock.patch.object(office, "_WHOLE_MEMBER", whole):
                for name, data in ods_corpus():
                    path = self.write(name, data)
                    self.assert_pages(path, sheets=[None, 0, 1, 2, 31, 32,
                                                    "Second", "Inner",
                                                    "Table 1", "Nope"])
        # The content is read to its end: every page counts its table.
        self.assertEqual({"same"}, set(self.counts))

    def test_damage_is_0155s_error_on_every_page(self):
        """The whole content is read for the list of sheets, so damage
        anywhere is found by every page."""
        rows = "".join(_row([str(i), f"n{i}"]) for i in range(400))
        good = _table("T", _row(["id", "name"]) + rows)
        for name, data in (
                ("late.ods", ods_bytes(good.replace("n399", "n<<"))),
                ("early.ods", ods_bytes(good.replace("n1<", "n1<<"))),
                ("crc.ods", self._crc(ods_bytes(good, stored=True)))):
            path = self.write(name, data)
            for whole in (0, 1 << 20):
                with mock.patch.object(office, "_PIECE", 512), \
                        mock.patch.object(office, "_WHOLE_MEMBER", whole):
                    self.assert_pages(path)
                    self.assertTrue(read_table(path).error)
        self.assertEqual({"same"}, set(self.counts))

    @staticmethod
    def _crc(data):
        return data.replace(b"<text:p>n398</text:p>", b"<text:p>n39Q</text:p>")

    def test_content_cut_at_the_member_cap(self):
        rows = "".join(_row([str(i), f"n{i}"]) for i in range(400))
        path = self.write("cap.ods", ods_bytes(_table("T", rows)))
        for whole in (0, 1 << 20):
            with mock.patch.object(office, "_MAX_MEMBER", 3000), \
                    mock.patch.object(office, "_WHOLE_MEMBER", whole), \
                    mock.patch.object(office, "_PIECE", 700):
                self.assert_pages(path)
                self.assertTrue(read_table(path).error)


# -- iter_table -------------------------------------------------------------
class IterTableTest(_Files):
    """Every row of a table, handed out as it is read."""

    def _all_pages(self, path, sheet=None, **kw):
        """(the last page, every page's rows, whether a cell was clipped)."""
        page = read_table(path, sheet=sheet, limit=1000, **kw)
        rows, offset, clipped = list(page.rows), 0, page.cells_clipped
        while page.has_more:
            offset += len(page.rows)
            page = read_table(path, sheet=sheet, offset=offset, limit=1000,
                              **kw)
            rows += page.rows
            clipped |= page.cells_clipped
        return page, rows, clipped

    def test_is_exported(self):
        import sift.extract as extract
        self.assertIn("iter_table", extract.__all__)
        self.assertIn("TableStream", extract.__all__)

    def test_the_rows_are_the_pages_rows(self):
        from sift.extract import iter_table
        files = [self.write(n, d) for n, d in csv_corpus()]
        files += [self.write(n, d) for n, d in xlsx_corpus()]
        files += [self.write(n, d) for n, d in ods_corpus()]
        for path in files:
            det = detect(path)
            if det.kind not in ("csv", "tsv", "xlsx", "ods"):
                continue
            with self.subTest(path.name):
                last, rows, clipped = self._all_pages(path)
                table = iter_table(path, det)
                self.assertEqual(last.header, table.header)
                self.assertEqual(last.preamble, table.preamble)
                got = list(table)
                if clipped:
                    # A page clips a cell to its budget, and marks the cut
                    # with "…"; the stream hands out the whole cell.
                    self.assertEqual(len(rows), len(got))
                    for shown, whole in zip(rows, got):
                        for a, b in zip(shown, whole):
                            self.assertTrue(a == b or a.endswith("…")
                                            and b.startswith(a[:-1]))
                else:
                    self.assertEqual(rows, got)
                self.assertEqual(last.sheets, table.sheets)
                self.assertEqual(last.sheet, table.sheet)
                self.assertEqual(last.error, table.error)
                self.assertEqual(last.truncated, table.truncated)
                self.assertEqual([n for n in last.notes if "at most" not in n],
                                 table.notes)

    def test_every_sheet_by_index(self):
        from sift.extract import iter_table
        path = self.write("s.xlsx", dict(xlsx_corpus())["sheets.xlsx"])
        first = iter_table(path, detect(path))
        list(first)
        self.assertEqual(["First", "Empty", "Third", "Blank rows"],
                         first.sheets)
        for index in range(len(first.sheets)):
            table = iter_table(path, detect(path), sheet=index)
            self.assertEqual(first.sheets[index], table.sheet)
            _page, rows, _clipped = self._all_pages(path, sheet=index)
            self.assertEqual(rows, list(table))

    def test_an_unreadable_file_is_an_error_and_no_rows(self):
        from sift.extract import iter_table
        fake = self.write("fake.xlsx", b"PK\x03\x04 no zip")
        table = iter_table(fake, replace(detect(fake), kind="xlsx"))
        self.assertIn("could not read the spreadsheet", table.error)
        self.assertEqual([], list(table))
        gone = self.write("gone.csv", "a\n1\n")
        det = detect(gone)
        gone.unlink()
        table = iter_table(gone, det)
        self.assertIn("No such file", table.error)
        self.assertEqual([], list(table))

    def test_damage_ends_the_rows_with_an_error(self):
        from sift.extract import iter_table
        name, data, _good = next(iter(xlsx_damaged()))
        path = self.write(name, data)
        with mock.patch.object(office, "_WHOLE_MEMBER", 0), \
                mock.patch.object(office, "_PIECE", 512):
            table = iter_table(path, detect(path))
            self.assertIsNone(table.error)
            rows = list(table)
        self.assertEqual(400, len(rows))
        self.assertIn("could not read the spreadsheet", table.error)

    def test_arguments_that_name_nothing_are_refused(self):
        from sift.extract import iter_table
        csv_path = self.write("t.csv", "a,b\n1,2\n")
        text = self.write("t.txt", "prose\n")
        with self.assertRaises(ValueError):
            iter_table(text, detect(text))
        with self.assertRaises(ValueError):
            iter_table(csv_path, detect(csv_path), sheet=0)
        book = self.write("b.xlsx", dict(xlsx_corpus())["sheets.xlsx"])
        for sheet in ("Nope", 9, -1, True):
            with self.subTest(sheet=sheet), self.assertRaises(ValueError):
                iter_table(book, detect(book), sheet=sheet)
        ods = self.write("b.ods", dict(ods_corpus())["tables.ods"])
        with self.assertRaises(ValueError):
            iter_table(ods, detect(ods), sheet="Nope")

    def test_an_empty_table(self):
        from sift.extract import iter_table
        blank = self.write("blank.csv", "\n\n  \n")
        table = iter_table(blank, detect(blank))
        self.assertEqual(([], [], None), (table.header, list(table),
                                          table.error))
        none = self.write("none.ods", dict(ods_corpus())["no-table.ods"])
        table = iter_table(none, detect(none))
        self.assertEqual([], list(table))
        self.assertEqual(["the file holds no table"], table.notes)

    def test_max_bytes_bounds_a_csv(self):
        from sift.extract import iter_table
        path = self.write("t.csv", "id,name\n" + "".join(
            f"{i},n{i}\n" for i in range(1000)))
        table = iter_table(path, detect(path), max_bytes=2000)
        rows = list(table)
        self.assertTrue(table.truncated)
        self.assertLess(len(rows), 1000)
        self.assertEqual(rows, self._all_pages(path, max_bytes=2000)[1])

    def test_wide_rows_are_cut_to_the_column_cap(self):
        from sift.extract import iter_table
        path = self.write("wide.csv", dict(csv_corpus())["wide.csv"])
        table = iter_table(path, detect(path))
        rows = list(table)
        self.assertEqual(tabular._MAX_COLUMNS, len(table.header))
        self.assertEqual({tabular._MAX_COLUMNS}, {len(r) for r in rows})
        self.assertTrue(table.columns_truncated)

    def test_its_time_budget_runs_out_while_rows_are_read(self):
        from sift.extract import iter_table
        path = self.write("t.csv", "id,name\n" + "".join(
            f"{i},n{i}\n" for i in range(20_000)))
        table = iter_table(path, detect(path), time_budget=60)
        with mock.patch("time.monotonic", return_value=10 ** 9):
            with self.assertRaises(deadline.DeadlineExceeded):
                list(table)

    def test_its_budget_and_csv_limit_do_not_leak_into_the_caller(self):
        """Between two rows the caller runs under its own clock and csv's
        own field limit, not the stream's."""
        from sift.extract import iter_table
        path = self.write("t.csv", "id,name\n" + "".join(
            f"{i},n{i}\n" for i in range(5000)))
        before = csv.field_size_limit()
        table = iter_table(path, detect(path), time_budget=5)
        for _row in table:
            self.assertIsNone(deadline.seconds())
            self.assertEqual(before, csv.field_size_limit())
        with deadline.limit(1000):
            table = iter_table(path, detect(path))
            for _row in table:
                self.assertEqual(1000, deadline.seconds())

    def test_the_callers_deadline_applies_without_a_budget(self):
        from sift.extract import iter_table
        path = self.write("t.csv", "id,name\n" + "".join(
            f"{i},n{i}\n" for i in range(20_000)))
        with deadline.limit(60):
            table = iter_table(path, detect(path))
            with mock.patch("time.monotonic", return_value=10 ** 9):
                with self.assertRaises(deadline.DeadlineExceeded):
                    list(table)

    def test_closing_it_early_closes_the_file(self):
        from sift.extract import iter_table
        path = self.write("t.xlsx", dict(xlsx_corpus())["shared.xlsx"])
        with iter_table(path, detect(path)) as table:
            next(iter(table))
        self.assertEqual([], list(table))



# -- a review of 0.1.58 ---------------------------------------------------
def _fds():
    """The file descriptors this process holds, or None where /proc does
    not say."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


class _NoCollector:
    """The cyclic collector off: what is closed here is closed by the code,
    not by a collection that happens to run."""

    def __enter__(self):
        self.was = gc.isenabled()
        gc.collect()
        gc.disable()
        return self

    def __exit__(self, *exc):
        if self.was:
            gc.enable()


def _lazy_sst_xlsx(rows=60_000):
    """An xlsx whose shared strings declare more than _WHOLE_MEMBER, so
    that they are read as the cells need them."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml",
                   f'<workbook xmlns="{SS}" xmlns:r="{REL}"><sheets>'
                   '<sheet name="S" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<Relationships xmlns="{PKG_REL}"><Relationship '
                   'Id="rId1" Target="worksheets/sheet1.xml"/>'
                   '</Relationships>')
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{SS}">' + "".join(
            f"<si><t>string number {i}</t></si>" for i in range(rows))
            + "</sst>")
        z.writestr("xl/worksheets/sheet1.xml",
                   f'<worksheet xmlns="{SS}"><sheetData>' + "".join(
                       f'<row r="{r + 1}"><c r="A{r + 1}" t="s"><v>{r}</v>'
                       '</c></row>' for r in range(rows))
                   + "</sheetData></worksheet>")
    return buffer.getvalue()


def _deep_ods(depth, rows, inner=False):
    """An ods whose one table holds ``rows`` rows inside ``depth`` nested
    row groups; with ``inner``, a row whose cell holds a table of its own,
    itself in row groups, among them."""
    group = "<table:table-row-group>"
    body = "".join(_row([str(i), f"n{i}"]) for i in range(rows))
    if inner:
        nested = _table("Inner", group * 3 + _row(["i1"]) + _row(["i2"])
                        + "</table:table-row-group>" * 3)
        body = (_row(["id", "name"]) + "<table:table-row><table:table-cell>"
                + nested + "</table:table-cell></table:table-row>" + body)
    return ods_bytes(_table("Deep", group * depth + body
                            + "</table:table-row-group>" * depth)
                     + _table("After", _row(["a", "b"])))


class _CountedGroups(frozenset):
    """_ODS_ROW_GROUPS, counting how often a tag is looked up in it."""

    looked = 0

    def __contains__(self, item):
        type(self).looked += 1
        return frozenset.__contains__(self, item)


class OdsRowGroupDepthTest(_Files):
    """M1: a row's table was found by walking every row group around it at
    each row's end: deep-20k.ods took 0.42 s for page 0 in 0.1.55 and 45.92
    s in 0.1.58. Each element's table is taken when it opens."""

    def test_a_row_costs_the_same_however_deep_its_groups(self):
        path = self.write("deep.ods", _deep_ods(depth=2000, rows=3000))
        counted = _CountedGroups(office._ODS_ROW_GROUPS)
        _CountedGroups.looked = 0
        with mock.patch.object(office, "_ODS_ROW_GROUPS", counted), \
                mock.patch.object(office, "_WHOLE_MEMBER", 0):
            page = read_table(path, offset=2990, limit=100)
        self.assertEqual(["2990", "n2990"], page.rows[0])
        self.assertEqual(3000, page.total)
        # Once an element as it opens, never once a row per group around it
        # (3,000 rows under 2,000 groups were 6,000,000 lookups).
        elements = 2000 + 3000 * 5 + 20
        self.assertLessEqual(_CountedGroups.looked, elements)

    def test_deep_groups_read_as_0155_read_them(self):
        for inner in (False, True):
            path = self.write(f"deep-{inner}.ods",
                              _deep_ods(depth=40, rows=250, inner=inner))
            for whole in (0, 1 << 20):
                with mock.patch.object(office, "_WHOLE_MEMBER", whole), \
                        mock.patch.object(office, "_PIECE", 211):
                    self.assert_pages(path, sheets=[None, 0, 1, 2, "Inner",
                                                    "After", "Nope"])
        self.assertEqual({"same"}, set(self.counts))

    def test_a_row_outside_any_table_belongs_to_none(self):
        """A row in groups that sit in no table, and one in a cell but not
        in a table, is no table's row; the table after is read as 0.1.55
        read it."""
        stray = ("<table:table-row-group>" + _row(["stray"])
                 + "</table:table-row-group>")
        path = self.write("stray.ods", ods_bytes(
            stray + _table("T", _row(["id", "v"]) + "".join(
                _row([str(i), "x"]) for i in range(30)))))
        for whole in (0, 1 << 20):
            with mock.patch.object(office, "_WHOLE_MEMBER", whole):
                self.assert_pages(path)
        self.assertEqual({"same"}, set(self.counts))


class TotalsWhereTheyAreCheapTest(_Files):
    """M2: a page read from a source held whole (a csv or tsv of at most
    _WHOLE_BYTES, one whose records all fit the header scan, an xlsx sheet
    of at most _WHOLE_MEMBER) and every ods page (its content is read to
    the end for the list of sheets) count the rows after the page, as
    0.1.55 did: the total is 0.1.55's."""

    def test_a_small_csv_has_0155s_total_on_every_page(self):
        for name, data in csv_corpus():
            self.assert_pages(self.write(name, data))
        self.assertEqual({"same"}, set(self.counts))

    def test_a_large_csv_whose_rows_fit_the_scan_has_the_total(self):
        rows = tabular.MAX_ROWS_SCAN - 10
        path = self.write("scan.csv", "id,blob\n" + "".join(
            f"{i},{'x' * 300}\n" for i in range(rows)))
        self.assertGreater(path.stat().st_size, tabular._WHOLE_BYTES)
        self.assert_pages(path)
        self.assertEqual({"same"}, set(self.counts))
        self.assertEqual(rows, read_table(path, limit=10).total)

    def test_a_large_csv_past_the_scan_keeps_no_total(self):
        rows = tabular.MAX_ROWS_SCAN * 50
        path = self.write("long.csv", "id,name\n" + "".join(
            f"{i},name {i}\n" for i in range(rows)))
        self.assertGreater(path.stat().st_size, tabular._WHOLE_BYTES)
        page = read_table(path, limit=10)
        self.assertTrue(page.has_more)
        self.assertIsNone(page.total)
        self.assertEqual(rows, read_table(path, offset=rows - 5).total)

    def test_an_xlsx_sheet_read_whole_has_the_total(self):
        path = self.write("x.xlsx", dict(xlsx_corpus())["sheets.xlsx"])
        self.assert_pages(path, sheets=[None, 0, 1, 2, 3])
        self.assertEqual({"same"}, set(self.counts))

    def test_an_xlsx_sheet_read_as_a_stream_keeps_no_total(self):
        path = self.write("x.xlsx", xlsx_bytes([("D", _rows(400))]))
        with mock.patch.object(office, "_WHOLE_MEMBER", 0):
            page = read_table(path, limit=10)
        self.assertTrue(page.has_more)
        self.assertIsNone(page.total)

    def test_every_ods_page_has_the_total(self):
        for whole, piece in ((1 << 20, 1 << 16), (0, 1 << 16), (0, 101)):
            with mock.patch.object(office, "_PIECE", piece), \
                    mock.patch.object(office, "_WHOLE_MEMBER", whole):
                for name, data in ods_corpus():
                    self.assert_pages(self.write(name, data),
                                      sheets=[None, 0, 1, 2, 31, "Second"])
        self.assertEqual({"same"}, set(self.counts))

    def test_an_ods_row_is_counted_as_its_cells_would_be_read(self):
        """The rows after an ods page are counted without their cells
        (_ods_blank): a row is counted exactly where _ods_row makes a cell
        of it that is not white space, within the first 512 columns. A
        comment or a shape in a cell is no part of its value (0.1.78): a
        cell holding nothing else is empty."""
        rows = [
            "<table:table-row><table:table-cell><office:annotation><dc:creator>"
            "x</dc:creator><text:p>note</text:p></office:annotation>"
            "</table:table-cell></table:table-row>",
            "<table:table-row><table:table-cell><draw:custom-shape><text:p>"
            "shape</text:p></draw:custom-shape></table:table-cell>"
            "</table:table-row>",
            "<table:table-row><table:table-cell><office:annotation><text:p>"
            "note</text:p></office:annotation><text:p>v</text:p>"
            "</table:table-cell></table:table-row>",
            "<table:table-row>" + _cell("", ' table:number-columns-repeated'
                                           '="600"') + _cell("far") +
            "</table:table-row>",
            "<table:table-row>" + _cell("", ' table:number-columns-repeated'
                                           '="511"') + _cell("edge") +
            "</table:table-row>",
            "<table:table-row><table:covered-table-cell><text:p>cv</text:p>"
            "</table:covered-table-cell></table:table-row>",
            "<table:table-row><table:table-cell><text:p> <text:span>\u00a0"
            "</text:span>\t</text:p></table:table-cell></table:table-row>",
            "<table:table-row><table:table-cell><text:p> <text:span>x"
            "</text:span></text:p></table:table-cell></table:table-row>",
            "<table:table-row><foo:table-cell><text:p>other</text:p>"
            "</foo:table-cell></table:table-row>",
            "<table:table-row/>",
        ]
        from xml.etree import ElementTree as ET
        for xml in rows:
            row = ET.fromstring(f"<r {ODS_NS}>{xml}</r>")[0]
            with self.subTest(xml=xml[:60]):
                self.assertEqual(not any(c.strip() for c in
                                         office._ods_row(row)),
                                 office._ods_blank(row))
        beside = [ET.fromstring(f"<r {ODS_NS}>{xml}</r>")[0]
                  for xml in rows[:3]]
        self.assertEqual([[], [], ["v"]],
                         [office._ods_row(row) for row in beside])
        body = _row(["h1", "h2"]) + "".join(
            xml.replace("<table:table-row", '<table:table-row table:number-'
                                            'rows-repeated="3"', 1)
            for xml in rows) * 40
        path = self.write("blank.ods", ods_bytes(_table("B", body)))
        for whole in (0, 1 << 20):
            with mock.patch.object(office, "_WHOLE_MEMBER", whole):
                self.assert_pages(path)
        self.assertEqual({"same"}, set(self.counts))

    def test_counting_out_of_time_says_so_as_0155_did(self):
        path = self.write("t.csv", "id,name\n" + "".join(
            f"{i},name {i}\n" for i in range(3000)))
        real = deadline.check

        def expire(where=""):
            if where == "the rest of a table":
                raise deadline.DeadlineExceeded(1, where)
            return real(where)
        with mock.patch.object(deadline, "check", expire):
            page = read_table(path, limit=10)
            old = reference.read_table(path, detect(path), limit=10)
        self.assertEqual(old, page)
        self.assertEqual(10, len(page.rows))
        self.assertIsNone(page.total)
        self.assertTrue(page.notes[0].startswith(TAIL_NOTES[1]))


def _sheet_member_xlsx(xml: bytes) -> bytes:
    """An xlsx of one sheet whose member is ``xml``, as it is."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml",
                   f'<workbook xmlns="{SS}" xmlns:r="{REL}"><sheets>'
                   '<sheet name="S" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<Relationships xmlns="{PKG_REL}"><Relationship '
                   'Id="rId1" Target="worksheets/sheet1.xml"/>'
                   '</Relationships>')
        z.writestr("xl/worksheets/sheet1.xml", xml)
    return buffer.getvalue()


class SheetAtTheMemberCapTest(_Files):
    """L1, L2: 0.1.55 read a member's first _MAX_MEMBER bytes and called it
    cut (_is_truncated) when it read all of them and they did not parse;
    0.1.58 called a sheet the zip declares larger cut before it parsed a
    byte, and one of exactly _MAX_MEMBER bytes never."""

    BODY = "".join(f'<row r="{r + 1}"><c r="A{r + 1}"><v>{r}</v></c>'
                   f'<c r="B{r + 1}" t="inlineStr"><is><t>n{r}</t></is></c>'
                   "</row>" for r in range(300))

    def _pages(self, xml: bytes, cap: int):
        path = self.write("s.xlsx", _sheet_member_xlsx(xml))
        for whole in (0, 1 << 20):
            with mock.patch.object(office, "_MAX_MEMBER", cap), \
                    mock.patch.object(office, "_WHOLE_MEMBER", whole), \
                    mock.patch.object(office, "_PIECE", 257):
                self.assert_pages(path)
        return path

    def test_a_damaged_sheet_of_exactly_the_cap_is_cut_back(self):
        xml = (f'<worksheet xmlns="{SS}"><sheetData>{self.BODY}'
               '<row r="999"><c r="A999"><v>1').encode()
        path = self._pages(xml, len(xml))
        with mock.patch.object(office, "_MAX_MEMBER", len(xml)), \
                mock.patch.object(office, "_WHOLE_MEMBER", 0):
            page = read_table(path, offset=290, limit=100)
        self.assertEqual(["299", "n299"], page.rows[-1])
        self.assertTrue(page.truncated)
        self.assertTrue(any("sheet larger than" in n for n in page.notes))

    def test_a_damaged_sheet_one_byte_short_of_the_cap_is_an_error(self):
        """Not cut: its damage is an error, on every page read whole, on
        the page that reaches it read as a stream."""
        xml = (f'<worksheet xmlns="{SS}"><sheetData>{self.BODY}'
               '<row r="999"><c r="A999"><v>1').encode()
        path = self.write("s.xlsx", _sheet_member_xlsx(xml))
        with mock.patch.object(office, "_MAX_MEMBER", len(xml) + 1):
            self.assert_pages(path)
            with mock.patch.object(office, "_WHOLE_MEMBER", 0):
                last = read_table(path, offset=290)
                self.assertEqual(reference.read_table(
                    path, detect(path), offset=290), last)
                first = read_table(path, limit=5)
        self.assertTrue(last.error)
        self.assertEqual([], last.notes)
        self.assertEqual(([], False), (first.notes, first.truncated))

    def test_a_larger_sheet_whose_first_part_is_whole_is_not_cut(self):
        """White space, or a comment, after the sheet's end: its first
        _MAX_MEMBER bytes parse, and 0.1.55 read it as it is."""
        xml = f'<worksheet xmlns="{SS}"><sheetData>{self.BODY}' \
              '</sheetData></worksheet>'
        for tail in (" " * 5000, "<!-- c -->" + "\n" * 5000,
                     "<?pi x?>\r\n" * 700):
            with self.subTest(tail=tail[:8]):
                data = (xml + tail).encode()
                path = self._pages(data, len(xml) + 20)
                with mock.patch.object(office, "_MAX_MEMBER", len(xml) + 20), \
                        mock.patch.object(office, "_WHOLE_MEMBER", 0):
                    page = read_table(path, limit=5)
                self.assertFalse(page.truncated)
                self.assertEqual([], page.notes)

    def test_a_larger_sheet_cut_mid_row_is_cut_on_every_page(self):
        xml = (f'<worksheet xmlns="{SS}"><sheetData>{self.BODY}'
               '</sheetData></worksheet>').encode()
        path = self._pages(xml, len(xml) // 2)
        with mock.patch.object(office, "_MAX_MEMBER", len(xml) // 2), \
                mock.patch.object(office, "_WHOLE_MEMBER", 0):
            page = read_table(path, limit=5)
        self.assertTrue(page.truncated)
        self.assertTrue(any("sheet larger than" in n for n in page.notes))

    def test_a_utf16_sheet_at_the_cap(self):
        xml = (f'<?xml version="1.0" encoding="UTF-16"?><worksheet '
               f'xmlns="{SS}"><sheetData>{self.BODY}</sheetData>'
               '</worksheet>').encode("utf-16")
        self._pages(xml, len(xml) // 2)
        self._pages(xml + " ".encode("utf-16-le") * 30, len(xml) + 10)


def _declaring(data: bytes, member: str, size: int) -> bytes:
    """The zip ``data`` with ``member``'s uncompressed size declared as
    ``size`` in its local header and its central directory entry (the
    reviewer's liar.py)."""
    import struct
    out = bytearray(data)
    info = zipfile.ZipFile(io.BytesIO(data)).getinfo(member)
    struct.pack_into("<I", out, info.header_offset + 22, size)
    at = out.rfind(b"PK\x01\x02")
    while at >= 0:
        length = struct.unpack_from("<H", out, at + 28)[0]
        if bytes(out[at + 46:at + 46 + length]) == member.encode():
            struct.pack_into("<I", out, at + 24, size)
        at = out.rfind(b"PK\x01\x02", 0, at)
    return bytes(out)


class SheetDeclaredAtTheCapTest(_Files):
    """A review: a sheet whose zip declares _MAX_MEMBER bytes but
    holds far fewer was called cut before a byte of it was parsed -- "sheet
    larger than 16.0 MB" of a 140 KB sheet, truncated on its first page and
    an error on its last (liar.py). 0.1.55 decided on the bytes it read
    (_is_truncated), as _closed_pieces does: so is it decided now, and a
    sheet that holds no more than _WHOLE_MEMBER is parsed whole, as 0.1.55
    parsed every sheet."""

    BODY = SheetAtTheMemberCapTest.BODY
    CAP = 1 << 21

    def _liar(self, tail: str, declared: int = CAP):
        xml = (f'<worksheet xmlns="{SS}"><sheetData>{self.BODY}'
               f'{tail}').encode()
        return self.write("liar.xlsx", _declaring(
            _sheet_member_xlsx(xml), "xl/worksheets/sheet1.xml", declared)), xml

    def test_a_well_formed_sheet_declared_at_the_cap_is_read_whole(self):
        path, xml = self._liar("</sheetData></worksheet>")
        with mock.patch.object(office, "_MAX_MEMBER", self.CAP):
            self.assert_pages(path)
            page = read_table(path, limit=5)
        self.assertEqual((page.notes, page.truncated, page.total),
                         ([], False, 300))

    def test_a_damaged_sheet_declared_at_the_cap_is_an_error_not_a_cut(self):
        """Sad path: its damage is 0.1.55's error on every page, never
        "sheet larger than"."""
        path, xml = self._liar('<row r="999"><c r="A999"><v>1')
        with mock.patch.object(office, "_MAX_MEMBER", self.CAP):
            self.assert_pages(path)
            for offset in (0, 290):
                page = read_table(path, offset=offset, limit=5)
                self.assertFalse(any("sheet larger than" in n
                                     for n in page.notes), page.notes)
                self.assertFalse(page.truncated)
                self.assertTrue(page.error)

    def test_a_sheet_declared_at_the_cap_holding_more_than_whole_streams(self):
        """Edge: one that holds more than _WHOLE_MEMBER, short of the cap, is
        read as a stream, and its damage is an error on the page that
        reaches it, not a cut."""
        path, xml = self._liar('<row r="999"><c r="A999"><v>1')
        with mock.patch.object(office, "_MAX_MEMBER", self.CAP), \
                mock.patch.object(office, "_WHOLE_MEMBER", 1000):
            first = read_table(path, limit=5)
            last = read_table(path, offset=290, limit=100)
        self.assertEqual((first.notes, first.truncated), ([], False))
        self.assertEqual(["0", "n0"], first.rows[0])
        self.assertTrue(last.error)
        self.assertFalse(any("sheet larger than" in n for n in last.notes))

    def test_a_sheet_declared_at_the_cap_and_holding_it_is_cut(self):
        """Edge: one that declares the cap and holds as much is cut, as
        before -- the bytes read fill it."""
        xml = (f'<worksheet xmlns="{SS}"><sheetData>{self.BODY}'
               '<row r="999"><c r="A999"><v>1').encode()
        path = self.write("full.xlsx", _sheet_member_xlsx(xml))
        with mock.patch.object(office, "_MAX_MEMBER", len(xml)), \
                mock.patch.object(office, "_WHOLE_MEMBER", 0):
            page = read_table(path, offset=290, limit=100)
        self.assertTrue(page.truncated)
        self.assertTrue(any("sheet larger than" in n for n in page.notes))


class ClosedDeterministicallyTest(_Files):
    """L3, L4: a table's file is let go by the code that read it, not by
    the cyclic collector; closing a TableStream, or exhausting it, closes
    its file at once."""

    def setUp(self):
        super().setUp()
        if _fds() is None:
            self.skipTest("no /proc/self/fd here")

    def test_a_page_of_an_xlsx_with_lazy_shared_strings(self):
        path = self.write("s.xlsx", _lazy_sst_xlsx())
        with _NoCollector():
            before = _fds()
            for _ in range(3):
                page = read_table(path, limit=10)
            self.assertEqual(before, _fds())
        self.assertEqual(["string number 0"], page.rows[0])

    def test_a_stream_closed_early_closes_its_file(self):
        """close() closes the file itself, while the caller still holds the
        rows: nothing is left to be closed as it is let go (ResourceWarning
        says where a file is)."""
        import warnings
        from sift.extract import iter_table
        for name, data in (("s.xlsx", _lazy_sst_xlsx()),
                           ("t.csv", "id,v\n" + "".join(
                               f"{i},{i}\n" for i in range(100_000))),
                           ("d.ods", _deep_ods(depth=3, rows=5000))):
            path = self.write(name, data)
            with self.subTest(name), _NoCollector(), \
                    warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                with mock.patch.object(office, "_WHOLE_MEMBER", 0), \
                        mock.patch.object(tabular, "_WHOLE_BYTES", 16):
                    before = _fds()
                    table = iter_table(path, detect(path))
                    rows = iter(table)
                    next(rows)
                    self.assertGreater(_fds(), before)
                    table.close()
                    self.assertEqual(before, _fds())
                    self.assertEqual([], list(table))
                    del rows, table
                self.assertEqual([], [str(w.message) for w in caught
                                      if w.category is ResourceWarning])

    def test_closing_a_sheets_rows_lets_every_member_go(self):
        """open_sheet_rows' close lets the sheet's member go and the shared
        strings' it reads as the cells need them, while the zip, which the
        caller owns, stays open: zipfile counts what holds its file."""
        path = self.write("s.xlsx", _lazy_sst_xlsx())
        from sift.extract.base import TablePage
        with mock.patch.object(office, "_WHOLE_MEMBER", 0), \
                zipfile.ZipFile(path) as zf:
            sink = TablePage(kind="xlsx")
            _header, rows, close = office.open_sheet_rows(
                zf, detect(path), None, sink)
            next(rows)
            self.assertEqual(3, zf._fileRefCnt)
            close()
            self.assertEqual(1, zf._fileRefCnt)
            self.assertIsNotNone(rows)

    def test_a_stream_let_go_unread_closes_its_file(self):
        from sift.extract import iter_table
        path = self.write("s.xlsx", _lazy_sst_xlsx())
        with _NoCollector():
            before = _fds()
            table = iter_table(path, detect(path))
            next(iter(table))
            del table
            self.assertEqual(before, _fds())

    def test_a_stream_read_to_its_end_closes_its_file(self):
        from sift.extract import iter_table
        path = self.write("s.xlsx", _lazy_sst_xlsx(rows=5000))
        with _NoCollector():
            before = _fds()
            table = iter_table(path, detect(path))
            self.assertEqual(5000, len(list(table)))
            self.assertEqual(before, _fds())
            table.close()
            table.close()
            self.assertEqual(before, _fds())


class StreamClockTest(_Files):
    """L5: iter_table's time_budget is a clock that runs from the call on:
    the time the caller spends between two rows is spent of it too, though
    the caller's code never runs under it."""

    def test_the_callers_time_between_rows_is_spent_of_the_budget(self):
        from sift.extract import iter_table
        path = self.write("t.csv", "id,v\n" + "".join(
            f"{i},{i}\n" for i in range(20_000)))
        now = [1000.0]
        with mock.patch("time.monotonic", lambda: now[0]):
            table = iter_table(path, detect(path), time_budget=10)
            rows = iter(table)
            for _ in range(tabular.MAX_ROWS_SCAN + table.BATCH):
                next(rows)
            now[0] += 11                # the caller took 11 s
            with self.assertRaises(deadline.DeadlineExceeded):
                list(rows)

    def test_a_caller_within_the_budget_reads_every_row(self):
        from sift.extract import iter_table
        path = self.write("t.csv", "id,v\n" + "".join(
            f"{i},{i}\n" for i in range(2000)))
        now = [1000.0]
        with mock.patch("time.monotonic", lambda: now[0]):
            table = iter_table(path, detect(path), time_budget=10)
            count = 0
            for _row in table:
                count += 1
                now[0] += 0.004
        self.assertEqual(2000, count)


class CarriageReturnSepTest(_Files):
    """L6: a csv whose lines end in carriage returns alone and which opens
    with a sep= line lost every row -- the line cut as the directive's was
    the whole text (0.1.55 alike)."""

    TEXT = "sep=;\rid;name\r" + "".join(f"{i};n{i}\r" for i in range(300))

    def test_its_rows_are_read_whole_and_as_a_stream(self):
        path = self.write("mac-sep.csv", self.TEXT)
        for whole in (1 << 20, 16):
            with mock.patch.object(tabular, "_WHOLE_BYTES", whole), \
                    mock.patch.object(tabular, "_PIECE_BYTES", 97):
                page = read_table(path, limit=5)
                self.assertEqual(["id", "name"], page.header)
                self.assertEqual(["0", "n0"], page.rows[0])
                self.assertEqual(300, read_table(path, offset=299).total)
                self.assert_pages(path)

    def test_the_summary_reads_it_too(self):
        from sift.extract import extract
        path = self.write("mac-sep.csv", self.TEXT)
        ex = extract(path, detect(path), 25_000_000)
        self.assertEqual(300, ex.metadata.get("rows"))

    def test_blank_lines_before_the_directive_and_crlf_are_as_before(self):
        for text in ("\r\r  \rsep=;\rid;name\r1;2\r",
                     "sep=;\r\nid;name\r\n1;2\r\n",
                     "sep=;\nid;name\r1;2\r"):
            path = self.write("s.csv", text)
            with self.subTest(text=text):
                self.assert_pages(path)
        self.assertEqual(["id", "name"], read_table(self.write(
            "b.csv", "\r\r  \rsep=;\rid;name\r1;2\r")).header)


class HeadBoundsTest(unittest.TestCase):
    """L7: the characters the header decision and the Sniffer read are
    typed once, and every reading of a csv reads that many."""

    def test_the_header_decision_reads_the_head(self):
        seen = []
        real = csv.reader

        def reader(lines, *a, **kw):
            if isinstance(lines, io.StringIO):
                seen.append(len(lines.getvalue()))
            return real(lines, *a, **kw)
        text = "a,b\n" + "1,2\n" * 50_000
        with mock.patch.object(tabular.csv, "reader", reader):
            tabular._has_header(text, ",")
        self.assertEqual([tabular._HEAD_CHARS], seen)

    def test_the_sniffer_reads_the_same_characters_everywhere(self):
        seen = []
        real = csv.Sniffer.sniff

        def sniff(self, sample, *a, **kw):
            seen.append(len(sample))
            return real(self, sample, *a, **kw)
        text = "a b c\n" + "x y z\n" * 50_000
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.csv"
            path.write_text(text)
            det = replace(detect(path), extra={})
            with mock.patch.object(csv.Sniffer, "sniff", sniff):
                tabular._prepare(path, det, 25_000_000)
                with mock.patch.object(tabular, "_WHOLE_BYTES", 16):
                    opened = tabular._open_text(path, det, 25_000_000)
                    opened[4]()
        self.assertEqual([tabular._SNIFF_CHARS] * 2, seen)
        self.assertLessEqual(tabular._SNIFF_CHARS, tabular._HEAD_CHARS)


if __name__ == "__main__":
    unittest.main()
