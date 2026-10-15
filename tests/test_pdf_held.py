"""What pypdf's reader holds of one document, bounded as a whole.

0.1.59 bounded the entries of the cross-reference sections alone. pypdf parses and keeps every object of an object
stream the first time one of them is asked for, and walking the page tree asks for every stream that holds a page:
t2100p.pdf, a tagged report of 21 MB and 935,000 objects, took pypdf past 1 GiB and was OOM-killed at --memory=1g. sift's copy of the object-stream reader now parses the objects asked for, and what the reader holds --
the entries, the objects it parses and keeps, the pages it walks, the object streams sift keeps decoded, the index
of "N G obj" -- is charged against MAX_DOCUMENT_HELD as it is made.
"""
import gc
import io
import re
import time
import tracemalloc
import unittest
import zlib
from unittest import mock

import test_extract_formats as formats
from sift.extract import pdf as _pdf


def _tagged_pdf(pages: int, rows: int, cols: int, per: int = 50, packed: bool = True) -> bytes:
    """The reviewer's gen_tagged.py, scaled: ``pages`` pages of ``rows`` x ``cols`` cells, one TD StructElem a cell,
    one TR a row, one Table a page, the struct elements in object streams of ``per`` objects; where ``packed``, each
    page's dictionary too, before its struct elements. Cross-reference stream W [1 4 2]."""
    cat, tree, root, font = 1, 2, 3, 4
    nxt = 5
    plain, packed_objects, tables, page_numbers = {}, [], [], []
    for p in range(pages):
        page, content = nxt, nxt + 1
        nxt += 2
        page_numbers.append(page)
        table = nxt
        nxt += 1
        tables.append(table)
        drawn, trs, mcid = bytearray(), [], 0
        first = len(packed_objects)
        for r in range(rows):
            tr = nxt
            nxt += 1
            trs.append(tr)
            tds = []
            for c in range(cols):
                td = nxt
                nxt += 1
                tds.append(td)
                drawn += (b"/TD <</MCID %d>> BDC BT /F1 7 Tf %d %d Td (p%d r%d c%d) Tj ET EMC\n"
                          % (mcid, 20 + c * 28, 780 - r * 12, p, r, c))
                packed_objects.append((td, b"<< /Type /StructElem /S /TD /P %d 0 R /Pg %d 0 R /K %d >>"
                                       % (tr, page, mcid)))
                mcid += 1
            packed_objects.append((tr, b"<< /Type /StructElem /S /TR /P %d 0 R /K [%s] >>"
                                   % (table, b" ".join(b"%d 0 R" % t for t in tds))))
        packed_objects.append((table, b"<< /Type /StructElem /S /Table /P %d 0 R /K [%s] >>"
                               % (root, b" ".join(b"%d 0 R" % t for t in trs))))
        data = zlib.compress(bytes(drawn))
        plain[content] = b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream"
        body = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                b"/Resources << /Font << /F1 4 0 R >> >> /StructParents %d >>" % (content, p))
        if packed:
            packed_objects.insert(first, (page, body))
        else:
            plain[page] = body
    plain[cat] = b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 3 0 R /MarkInfo << /Marked true >> >>"
    plain[tree] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % n for n in page_numbers),
                                                                 pages)
    plain[root] = b"<< /Type /StructTreeRoot /K [%s] >>" % b" ".join(b"%d 0 R" % t for t in tables)
    plain[font] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    where = {}
    for i in range(0, len(packed_objects), per):
        chunk, stm = packed_objects[i:i + per], nxt
        nxt += 1
        head, body = bytearray(), bytearray()
        for j, (number, obj) in enumerate(chunk):
            head += b"%d %d " % (number, len(body))
            body += obj + b"\n"
            where[number] = (stm, j)
        data = zlib.compress(bytes(head) + bytes(body))
        plain[stm] = (b"<< /Type /ObjStm /N %d /First %d /Length %d /Filter /FlateDecode >>\nstream\n"
                      % (len(chunk), len(head), len(data)) + data + b"\nendstream")
    out, offsets = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"), {}
    for number in sorted(plain):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + plain[number] + b"\nendobj\n"
    xref, at = nxt, len(out)
    rows_of = {n: (1, o, 0) for n, o in offsets.items()}
    rows_of.update({n: (2, s, j) for n, (s, j) in where.items()})
    rows_of[xref] = (1, at, 0)
    packed_rows = zlib.compress(formats.PdfStructureTest._xref_rows(rows_of, xref + 1))
    out += (b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R /Filter /FlateDecode /Length %d >>\nstream\n"
            % (xref, xref + 1, len(packed_rows)) + packed_rows + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at)
    return bytes(out)


#: What the parse under way may pass the bound by before it is walked (_Structure.in_flight).
_OVERSHOOT = _pdf.READ_HELD * _pdf.IN_FLIGHT_READS


def _alone(data: bytes):
    """(pypdf's own text of ``data`` as sift joins it, its reader, what it logged)."""
    import pypdf
    with formats._PypdfLog() as log:
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n\n".join(f"[page {i + 1}]\n{page.extract_text()}".strip() for i, page in enumerate(reader.pages))
    return text, reader, log.records


def _same_page_pdf(kids: int, keys: int) -> bytes:
    """A page tree of ``kids`` kids, every one the same page, whose dictionary holds ``keys`` keys besides a page's
    own: pypdf copies the page's dictionary into a PageObject for each kid (the reviewer-sized samekids20k.pdf,
    137 KB, took 0.1.59 past 1 GiB)."""
    extra = b" ".join(b"/K%d 0" % k for k in range(keys))
    page = formats.PdfStructureTest.PAGE[:-2] + extra + b" >>"
    bodies = [b"<< /Type /Catalog /Pages 2 0 R >>",
              b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b"5 0 R " * kids, kids),
              None, formats.PdfStructureTest.FONT, page]
    out, offsets = bytearray(b"%PDF-1.7\n"), [0]
    content = formats.PdfStructureTest.CONTENT
    for number, body in enumerate(bodies, 1):
        offsets.append(len(out))
        if body is None:
            out += b"%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (number, len(content), content)
        else:
            out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    table = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets[1:])
    out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % table
    return bytes(out)


def _array_page_pdf(item: bytes, count: int, pages: int = 1) -> bytes:
    """``pages`` pages, the last of which has /Annots of ``count`` times ``item`` (b"0 ": numbers)."""
    base = formats.PdfReaderBuildTest._pages_pdf(pages)
    last = b"/Contents %d 0 R" % (5 + 2 * (pages - 1))
    at = base.index(last)
    grown = base[:at] + b"/Annots [" + item * count + b"] " + base[at:]
    # The offsets after the grown page move by what was inserted: rebuild the table.
    return _reoffset(grown)


def _reoffset(data: bytes) -> bytes:
    """``data`` with its classic table's offsets written anew from where each "N 0 obj" now stands."""
    table = data.rindex(b"\nxref\n") + 1
    size = int(re.search(rb"/Size (\d+)", data[table:]).group(1))
    offsets = {int(m.group(1)): m.start() for m in re.finditer(rb"(?m)^(\d+) 0 obj", data[:table])}
    lines = b"".join(b"%010d 00000 n \n" % offsets[n] if n in offsets else b"0000000000 65535 f \n"
                     for n in range(1, size))
    trailer = data.index(b"trailer", table)
    return (data[:table] + b"xref\n0 %d\n0000000000 65535 f \n" % size + lines + data[trailer:data.rindex(b"startxref")]
            + b"startxref\n%d\n%%%%EOF\n" % table)


def _interleaved_pdf(each: int) -> bytes:
    """2 x ``each`` pages whose dictionaries sit in two object streams, the page tree alternating between them."""
    out, xref = bytearray(b"%PDF-1.7\n"), {}

    def obj(number, text):
        xref[number] = (1, len(out), 0)
        out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, text))
    kids = [10 + k for k in range(2 * each)]
    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids)))
    content = formats.PdfStructureTest.CONTENT
    obj(3, b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
    obj(4, formats.PdfStructureTest.FONT)
    for holder, numbers in ((40, kids[0::2]), (41, kids[1::2])):
        head, body = bytearray(), bytearray()
        for k, number in enumerate(numbers):
            head += b"%d %d " % (number, len(body))
            body += formats.PdfStructureTest.PAGE + b"\n"
            xref[number] = (2, holder, k)
        packed = zlib.compress(bytes(head) + bytes(body))
        obj(holder, b"<< /Type /ObjStm /N %d /First %d /Filter /FlateDecode /Length %d >>\nstream\n"
            % (len(numbers), len(head), len(packed)) + packed + b"\nendstream")
    size = max(xref) + 2
    at = len(out)
    xref[size - 1] = (1, at, 0)
    rows = zlib.compress(formats.PdfStructureTest._xref_rows(xref, size))
    out.extend(b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R /Filter /FlateDecode /Length %d >>\n"
               b"stream\n" % (size - 1, size, len(rows)) + rows
               + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at)
    return bytes(out)


def _raw_objstm_pdf(numbers, header: bytes, body: bytes) -> bytes:
    """Objects ``numbers`` listed by the cross-reference stream in one object stream of ``header`` and ``body``,
    written as they are, unfiltered."""
    out, xref = bytearray(b"%PDF-1.7\n"), {}

    def obj(number, text):
        xref[number] = (1, len(out), 0)
        out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, text))
    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, b"<< /Type /Pages /Kids [] /Count 0 >>")
    raw = header + body
    obj(5, b"<< /Type /ObjStm /N %d /First %d /Length %d >>\nstream\n" % (len(numbers), len(header), len(raw))
        + raw + b"\nendstream")
    for k, number in enumerate(numbers):
        xref[number] = (2, 5, k)
    size = max(xref) + 2
    at = len(out)
    xref[size - 1] = (1, at, 0)
    rows = zlib.compress(formats.PdfStructureTest._xref_rows(xref, size))
    out.extend(b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R /Filter /FlateDecode /Length %d >>\n"
               b"stream\n" % (size - 1, size, len(rows)) + rows
               + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at)
    return bytes(out)


class PdfHeldTest(unittest.TestCase):
    """What pypdf's reader holds of one document: the objects of object streams parsed as they are asked for, and
    everything the reader holds charged against MAX_DOCUMENT_HELD."""

    def setUp(self):
        import importlib.util
        if importlib.util.find_spec("pypdf") is None:
            self.skipTest("no pypdf")

    def _opened(self, data: bytes):
        return formats.PdfReaderBuildTest._opened(self, data)

    @staticmethod
    def _charged(structure) -> int:
        return structure.most

    # ---- object streams: the objects asked for -----------------------------------------------------------------

    def test_an_object_stream_parses_the_objects_asked_for(self):
        """Happy path: a tagged report whose pages sit in object streams beside their struct elements reads as
        pypdf reads it, logs what pypdf logs, and the reader holds none of the struct elements it never asked
        for -- pypdf parsed and kept every object of a stream the first time one was asked for."""
        data = _tagged_pdf(6, 4, 5, per=30)
        text, alone, records = _alone(data)
        with formats._PypdfLog() as log:
            (got, meta, notes, _), readers = self._opened(data)
        self.assertEqual((got, meta.get("pages"), notes), (text, 6, []))
        self.assertEqual(log.records, records)
        self.assertGreater(sum(1 for obj in alone.resolved_objects.values()
                               if hasattr(obj, "get") and obj.get("/Type") == "/StructElem"), 100)
        # _close empties resolved_objects once the document is read: look while it is read instead.
        seen = []
        from sift.extract import pdf
        real = pdf._read_pages

        def reading(reader, *args):
            result = real(reader, *args)
            seen.extend(obj for obj in reader.resolved_objects.values()
                        if hasattr(obj, "get") and obj.get("/Type") == "/StructElem")
            return result
        with formats._PypdfLog(), mock.patch.object(pdf, "_read_pages", reading):
            pdf._with_pypdf(data)
        self.assertEqual(seen, [])

    def test_sifts_parse_of_an_object_stream_is_pypdfs_asked_in_any_order(self):
        """What sift's reader makes of each object of an object stream, and what pypdf logs, is what pypdf's own
        makes, asked for in any order: backwards, an object the stream's header lists twice (the first is taken),
        one its header lacks (null), one pypdf cannot read."""
        import pypdf
        from sift.extract import pdf
        PAGE = formats.PdfStructureTest.PAGE
        shapes = {
            "backwards": formats.PdfStructureTest._object_streams_pdf(4, bodies=[PAGE] * 4),
            "an unreadable object": formats.PdfStructureTest._object_streams_pdf(
                3, bodies=[PAGE, b"<< /A (unterminated", b"[1 2 (x) /N]"]),
            "numbers": formats.PdfStructureTest._object_streams_pdf(3, bodies=[b"42", b"43", b"(s)"]),
        }
        shapes["listed twice"] = _raw_objstm_pdf([10, 12], b"10 0 10 8 ", b"(first)\n(second)\n")
        shapes["lacking from the header"] = _raw_objstm_pdf([10, 12], b"10 0 99 8 ", b"(first)\n(second)\n")

        def read(opener, data, order):
            with formats._PypdfLog() as log:
                reader = opener(data)
                made = []
                for number in order(sorted(reader.xref_objStm)):
                    try:
                        made.append(re.sub(r"IndirectObject\((\d+), (\d+), \d+\)", r"IndirectObject(\1, \2)",
                                           repr(reader.get_object(number))))
                    except Exception as exc:  # noqa: BLE001 - both raise alike or not
                        made.append(type(exc).__name__)
            return made, log.records
        for name, data in shapes.items():
            for order in (list, lambda numbers: list(reversed(numbers))):
                with self.subTest(name, order=order):
                    structure = pdf._Structure(pdf._structure_budget(len(data)), pdf._entry_budget(len(data)))
                    mine = read(lambda d: pdf._open(d, structure), data, order)
                    theirs = read(lambda d: pypdf.PdfReader(io.BytesIO(d)), data, order)
                    self.assertEqual(mine[0], theirs[0])
                    self.assertEqual(sorted(mine[1]), sorted(theirs[1]))

    def test_an_object_stream_is_decoded_once_while_it_is_kept(self):
        """Happy path: the pages of one stream, each asked for on its own, decode it once: it is kept, and charged
        to MAX_DOCUMENT_HELD, while the document is read."""
        data = formats.PdfStructureTest._object_streams_pdf(5, bodies=[formats.PdfStructureTest.PAGE] * 5)
        with formats._PypdfLog():
            (text, _, notes, _), readers = self._opened(data)
        self.assertEqual((text.count("hello"), notes), (5, []))
        structure = readers[0].structure
        decoded = len(zlib.decompress(re.search(rb"/ObjStm.*?stream\n(.*?)\nendstream", data, re.S).group(1)))
        self.assertEqual(structure.spent - self._xref_spent(data), decoded)
        self.assertEqual(len(structure.kept), 0)            # let go of with the document

    @staticmethod
    def _xref_spent(data: bytes) -> int:
        """What the cross-reference stream of ``data`` decodes to."""
        body = re.search(rb"/XRef.*?stream\n(.*?)\nendstream", data, re.S).group(1)
        return len(zlib.decompress(body))

    def test_an_object_stream_let_go_of_is_decoded_again_within_the_budget(self):
        """Edge: past MAX_OBJECT_STREAMS_HELD the stream used longest ago is let go of, and decoded again -- and
        charged again to the structure budget -- when an object of it is asked for next; the text is the same.
        Pages alternate between two streams here, so that each is let go of before it is asked again."""
        from sift.extract import pdf
        data = _interleaved_pdf(3)
        text, _, _ = _alone(data)
        with formats._PypdfLog():
            (got, _, notes, _), kept_all = self._opened(data)
        self.assertEqual((got, notes), (text, []))
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_OBJECT_STREAMS_HELD", 1):
            (got, _, notes, _), readers = self._opened(data)
        self.assertEqual((got, notes), (text, []))
        once = kept_all[0].structure.spent - self._xref_spent(data)
        self.assertEqual(readers[0].structure.spent - self._xref_spent(data), 3 * once)

    # ---- what the reader holds, charged -------------------------------------------------------------------------

    def test_the_reviewers_tagged_report_reads_every_page_within_the_bound(self):
        """t2100p.pdf, scaled: where parsing every object of the streams the page tree touches would pass the
        bound, the objects asked for fit it, and every page is read."""
        from sift.extract import pdf
        data = _tagged_pdf(30, 8, 8, per=60)
        text, alone, _ = _alone(data)
        with formats._PypdfLog():
            (_, _, _, _), readers = self._opened(data)
        charged = self._charged(readers[0].structure)
        # The premise: pypdf keeps every struct element, which would pass the bound here.
        elements = [obj for obj in alone.resolved_objects.values()
                    if hasattr(obj, "get") and obj.get("/Type") == "/StructElem"]
        self.assertGreater(len(elements), 1_000)
        self.assertGreater(sum(map(pdf._held_by, elements)), charged)
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", charged + _OVERSHOOT):
            (got, meta, notes, cut), _ = self._opened(data)
        self.assertEqual((got, meta.get("pages"), notes, cut), (text, 30, [], False))

    def test_the_bound_passed_reading_a_page_ends_the_document_there(self):
        """Sad path: past MAX_DOCUMENT_HELD while a page is read, the pages before it are kept, the text is marked
        cut, and the notes say on which page it ended."""
        from sift.extract import pdf
        data = formats.PdfReaderBuildTest._pages_pdf(40)
        with formats._PypdfLog():
            (_, _, _, _), readers = self._opened(data)
        whole = self._charged(readers[0].structure)
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", whole - 5_000):
            (text, meta, notes, cut), readers = self._opened(data)
        self.assertTrue(cut)
        self.assertIn("[page 1]\npage 1", text)
        self.assertNotIn("[page 40]", text)
        page = int(re.search(r"reading page (\d+);", " ".join(notes)).group(1))
        self.assertIn(f"[page {page - 1}]", text)
        self.assertNotIn(f"[page {page}]\n", text)
        self.assertIn(f"pypdf would hold more than {whole - 5_000} bytes of this document's objects -- half of "
                      f"the {pdf.WORKER_MEMORY} bytes a worker reads a document in --, reading page {page}; "
                      f"the text of page {page} and after is not in the body", notes)
        self.assertEqual(meta.get("pages"), 40)

    def test_the_bound_passed_walking_the_page_tree_falls_back(self):
        """Sad path: a page tree of one page many times -- pypdf copies the page for each kid -- passes the bound
        before a page is read; pypdf is not read with, and the notes say why. The reviewer's shape at full size,
        137 KB, took 0.1.59 past 1 GiB."""
        from sift.extract import pdf
        data = _same_page_pdf(400, 300)
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", 2_000_000):
            (text, _, notes, _), peak = formats.PdfStructureTest._peak(lambda: pdf._with_pypdf(data))
        self.assertIsNone(text)
        self.assertIn("pypdf would hold more than 2000000 bytes of this document's objects -- half of the "
                      f"{pdf.WORKER_MEMORY} bytes a worker reads a document in --, walking its page tree; "
                      f"used fallback", notes)
        self.assertLess(peak, 2_000_000 + 1_000_000, f"{peak} bytes held")
        with formats._PypdfLog():
            text, _, notes, _ = pdf._with_pypdf(_same_page_pdf(3, 300))
        self.assertEqual((text.count("hello"), notes), (3, []))

    def test_what_a_page_tree_copies_is_charged_as_it_is_walked(self):
        """What pypdf copies of a page for each kid is charged as it walks: 400 more kids of a page of 300 keys
        charge 400 copies of the page more, each what it holds (_page_held; test_pdf_charged holds that within 15 %
        of what tracemalloc traces)."""
        import pypdf
        from sift.extract import pdf
        with formats._PypdfLog():
            _, few = self._opened(_same_page_pdf(4, 300))
            _, many = self._opened(_same_page_pdf(404, 300))
            copy = pypdf.PdfReader(io.BytesIO(_same_page_pdf(1, 300))).pages[0]
        grown = self._charged(many[0].structure) - self._charged(few[0].structure)
        self.assertGreaterEqual(grown, 400 * pdf._page_held(copy))
        self.assertLessEqual(grown, 400 * pdf._page_held(copy) + 400 * 128)     # and the kids' references

    def test_the_bound_passed_opening_the_document_falls_back(self):
        """Sad path: past MAX_DOCUMENT_HELD while the reader is made, the reader is not made, and the notes say
        why."""
        from sift.extract import pdf
        data = formats.PdfReaderBuildTest._pages_pdf(40)
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", 1_000):
            text, _, notes, _ = pdf._with_pypdf(data)
        self.assertIsNone(text)
        self.assertIn("pypdf would hold more than 1000 bytes of this document's objects -- half of the "
                      f"{pdf.WORKER_MEMORY} bytes a worker reads a document in --, opening it",
                      " ".join(notes))

    def test_one_object_past_the_bound_ends_as_it_is_parsed(self):
        """Sad path: one object that alone passes the bound -- a page whose /Annots holds 300,000 numbers, 20 MB
        parsed -- ends as pypdf parses it, not once it is parsed: what is held stays near the bound."""
        from sift.extract import pdf
        data = _array_page_pdf(b"0 ", 300_000, pages=2)
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", 3_000_000):
            (text, _, notes, cut), peak = formats.PdfStructureTest._peak(lambda: pdf._with_pypdf(data))
        self.assertLess(peak, 3_000_000 + 2_000_000, f"{peak} bytes held")
        self.assertIsNone(text)                  # the walk resolves the page: before any page is read
        self.assertIn("walking its page tree", " ".join(notes))
        with formats._PypdfLog():
            text, _, notes, _ = pdf._with_pypdf(_array_page_pdf(b"0 ", 3_000, pages=2))
        self.assertEqual((text, notes), ("[page 1]\npage 1\n\n[page 2]\npage 2", []))

    def test_one_object_of_an_object_stream_past_the_bound_ends_as_it_is_parsed(self):
        """Sad path: the same, the object in an object stream."""
        from sift.extract import pdf
        page = formats.PdfStructureTest.PAGE[:-2] + b"/Annots [" + b"0 " * 300_000 + b"] >>"
        data = formats.PdfStructureTest._object_streams_pdf(1, bodies=[page])
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", 3_000_000):
            (text, _, notes, _), peak = formats.PdfStructureTest._peak(lambda: pdf._with_pypdf(data))
        self.assertLess(peak, 3_000_000 + 2_000_000, f"{peak} bytes held")
        self.assertIsNone(text)
        self.assertIn("walking its page tree", " ".join(notes))

    def test_the_bound_is_inclusive(self):
        """Edge: what the reader is charged may reach MAX_DOCUMENT_HELD, and not pass it (_Structure.hold); a
        document is read whole under a bound of the most it was charged, and the parse under way may pass it by
        _OVERSHOOT before it is walked, and a byte under the most it is charged, it is not read whole. What a parse
        holds on its way may be more than what it keeps -- pypdf's dictionary as it grows, the bytes it reads --:
        a bound of exactly what it keeps is walked past, and ends it."""
        from sift.extract import pdf
        structure = pdf._Structure(10 ** 6, 10 ** 6)
        structure.bound = 1_000
        structure.hold(1_000)
        with self.assertRaises(pdf._HeldBound):
            structure.hold(1)
        data = formats.PdfReaderBuildTest._pages_pdf(12)
        with formats._PypdfLog():
            (text, _, notes, _), readers = self._opened(data)
        exact = self._charged(readers[0].structure)
        for bound, whole in ((exact + _OVERSHOOT, True), (exact - 1, False)):
            with self.subTest(bound=bound), formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", bound):
                got, _, said, _ = pdf._with_pypdf(data)
                self.assertEqual((got, said) == (text, notes), whole, said)

    def test_the_index_of_objects_is_charged_and_ends_at_the_bound(self):
        """The index of "N G obj" is made of at most what is left of the bound at its peak, INDEX_KEY_PEAK a key,
        and charged what it holds (_index_held); where what is left is less than a full index, it ends there, and
        pypdf's search answers past it (the reviewer: 112 MB at its cap, uncounted)."""
        from sift.extract import pdf
        marks = b"%PDF-1.7\n" + b"".join(b"\n%d 0 obj" % n for n in range(1, 1_001))
        for room, keys in ((10 ** 6, 1_000), (20 * pdf.INDEX_KEY_PEAK + 5, 20)):
            with self.subTest(room=room):
                structure = pdf._Structure(10 ** 6, 10 ** 6)
                structure.data = marks
                structure.held = structure.bound - room
                self.assertEqual(structure.find(1, 0), (9, 17))
                self.assertEqual(len(structure.found.index), keys)
                self.assertEqual(structure.held, structure.bound - room + pdf._index_held(structure.found))
                self.assertEqual(structure.found.upto < len(marks), keys < 1_000)
                self.assertEqual(structure.find(900, 0)[0], marks.index(b"\n900 0 obj"))
        structure = pdf._Structure(10 ** 6, 10 ** 6)       # nothing left: not even an empty index
        structure.data, structure.held = marks, structure.bound
        with self.assertRaises(pdf._HeldBound):
            structure.find(1, 0)

    # ---- the charges, against what pypdf holds ------------------------------------------------------------------

    def test_a_read_of_the_parse_under_way_is_counted_at_the_worst_case(self):
        """What a parsed object holds a read pypdf makes of it -- the costliest shapes measured: hex strings 28.4
        bytes a read, strings 27.2, one-key dictionaries 27.1 -- is within READ_HELD, the worst case the parse under
        way is counted at until it is walked (_Structure.in_flight)."""
        import pypdf
        from pypdf.generic import read_object
        from sift.extract import pdf
        with formats._PypdfLog():
            reader = pypdf.PdfReader(io.BytesIO(formats.PdfReaderBuildTest._pages_pdf(1)))

        class Counting(io.BytesIO):
            calls = 0

            def read(self, size=-1):
                data = super().read(size)
                Counting.calls += 1 if len(data) <= pdf.READ_HELD else len(data) / pdf.READ_HELD
                return data
        for name, item in {"numbers": b"0 ", "references": b"5 0 R ", "names": b"/a ", "strings": b"(a)",
                           "hex strings": b"<a>", "dictionaries": b"<</a 0>>", "arrays": b"[]",
                           "long strings": b"(" + b"a" * 200 + b")", "pages": formats.PdfStructureTest.PAGE}.items():
            with self.subTest(name):
                Counting.calls = 0
                gc.collect()
                tracemalloc.start()
                try:
                    read_object(Counting(b"[" + item * 20_000 + b"]"), reader)
                    held = tracemalloc.get_traced_memory()[0]
                finally:
                    tracemalloc.stop()
                self.assertLess(held / Counting.calls, pdf.READ_HELD)

    def test_the_documents_bound_is_derived_from_the_worker_memory(self):
        """MAX_DOCUMENT_HELD is half the 1 GiB a host application's workers read a document in; the entries may take half
        of it (MAX_XREF_HELD, the count bound's), the object streams kept a sixteenth."""
        from sift.extract import pdf
        self.assertEqual(pdf.MAX_DOCUMENT_HELD, pdf.WORKER_MEMORY // 2)
        self.assertEqual(pdf.MAX_XREF_HELD, pdf.MAX_DOCUMENT_HELD // 2)
        self.assertEqual(pdf.MAX_OBJECT_STREAMS_HELD, pdf.MAX_DOCUMENT_HELD // 16)

    def test_reads_of_the_file_are_counted_only_while_an_object_is_parsed(self):
        """A counted read costs pypdf half as much again as io.BytesIO's (a classic table of 100,004 objects: its
        page tree walked in 4.7 to 5.0 s, 7.4 s counted, a review): the file's reads are io.BytesIO's but while
        sift parses an object of it, and after one."""
        from sift.extract import pdf
        data = formats.PdfReaderBuildTest._pages_pdf(3)
        with formats._PypdfLog():
            structure = pdf._Structure(10 ** 6, 10 ** 6)
            reader = pdf._open(data, structure)
            try:
                seen = []
                names = pdf._get_object_names()

                def parsing(stream, *args):
                    seen.append(type(stream).read)
                    return names[6](stream, *args)
                with mock.patch.object(pdf, "_GET_OBJECT_NAMES", names[:6] + (parsing,) + names[7:]):
                    reader.pages[0].get_contents()
                self.assertTrue(seen)
                self.assertTrue(all(read is not io.BytesIO.read for read in seen))
                self.assertIs(type(reader.stream).read, io.BytesIO.read)
            finally:
                pdf._close(reader)

    # ---- a review M3: a rebuild's writes ----------------------------------------------------------------------

    def test_rebuilding_a_table_counts_every_entry_it_writes(self):
        """Object streams of "0 0 0 ..." met rebuilding a lost table write entry 0 over and over; 0.1.59 counted new
        entries only, and read an 11 MB file to the 60 s budget with no text, where 0.1.58 took 3.05 s.
        Every entry a rebuild writes is counted against the entry bound, as 0.1.58 counted it; an object held is
        still charged once."""
        from sift.extract import pdf
        data = formats.PdfReaderBuildTest._rebuilt_pdf(20, 200_000)
        started = time.monotonic()
        with formats._PypdfLog():
            (text, _, notes, _), readers = self._opened(data)
        took = time.monotonic() - started
        structure = readers[0].structure
        self.assertEqual(text, "[page 1]\nhello")
        self.assertEqual(structure.written, structure.entries)
        self.assertEqual(structure.listed, 5 + 20 + 1)
        self.assertIn(f"pypdf rebuilt this document's lost cross-reference table writing more than "
                      f"{structure.entries} entries -- one a {pdf.XREF_FILE_BYTES_AN_ENTRY} bytes of it --; what "
                      f"is past that was not read", notes)
        self.assertLess(took, 2.0)

    def test_a_rebuild_writing_up_to_the_bound_is_read_whole(self):
        """Edge: the write bound is inclusive: a rebuild of the entries it writes is read whole at a bound of
        exactly them, and said to be cut one below."""
        from sift.extract import pdf
        data = formats.PdfReaderBuildTest._rebuilt_pdf(2, 64, digits=b"7 0 ")
        with formats._PypdfLog():
            (text, _, notes, _), readers = self._opened(data)
        written = readers[0].structure.written
        self.assertEqual((text, notes), ("[page 1]\nhello", []))
        for bound, said in ((written, False), (written - 1, True)):
            with self.subTest(bound=bound), formats._PypdfLog(), \
                    mock.patch.object(pdf, "_entry_budget", lambda size: bound):
                (_, _, notes, _), readers = self._opened(data)
                self.assertEqual(any("writing more than" in note for note in notes), said, notes)
                self.assertEqual(readers[0].structure.written, bound if said else written)


class PypdfFloorTest(unittest.TestCase):
    """A review: _older_than_the_floor read the version's leading numbers and a pre-release written without a
    separator; "6.18.1-rc1", "6.18.1-dev0" and "v6.18.0" were read with. Versions are read as PEP 440 orders them:
    a pre- or development release of the floor is older than it."""

    def test_versions_are_ordered_as_pep_440_orders_them(self):
        from sift.extract import pdf
        older = ("6.18.0", "6.18.1rc1", "6.18.1-rc1", "6.18.1_rc1", "6.18.1.RC1", "6.18.1-dev0", "6.18.1.dev0",
                 "6.18.1dev", "v6.18.0", "V6.18.0", "6.18.1a1", "6.18.1alpha2", "6.18.1b2", "6.18.1beta",
                 "6.18.1c1", "6.18.1-pre1", "6.18.1preview1", "6.18.1rc1.post1", "6.18.1rc1.dev3", "6.18",
                 "6.17.9", "6.18.0.post5", " 6.18.0 ", "0!6.18.0")
        not_older = ("6.18.1", "v6.18.1", "6.18.1.0", "06.18.01", "6.18.1.post1", "6.18.1-1", "6.18.1post0",
                     "6.18.1.post1.dev0", "6.18.1+local.7", "6.18.2.dev0", "6.18.2a1", "6.19.0", "7", "1!0.1",
                     " 6.18.1\n")
        for version in older:
            with self.subTest(version=version):
                self.assertTrue(pdf._older_than_the_floor(version))
        for version in not_older:
            with self.subTest(version=version):
                self.assertFalse(pdf._older_than_the_floor(version))

    def test_a_version_that_names_none_is_not_older(self):
        """Sad path: what PEP 440 cannot read is read with, as before -- the methods sift needs are looked for
        besides (_bounded_reader)."""
        from sift.extract import pdf
        for version in ("(of no version)", "", "six", "6.18.1-foo", "6..18", "latest"):
            with self.subTest(version=version):
                self.assertFalse(pdf._older_than_the_floor(version))


if __name__ == "__main__":
    unittest.main()
