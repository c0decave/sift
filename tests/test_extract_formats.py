"""PDF, image and structured-data extraction, at unit level.

These three were the thinnest-covered modules, and they are the ones whose
failures are quietest: a PDF that yields glued-together words, an image whose
dimensions are invented, a JSON blamed for a cut we made ourselves. Nothing
here raises — it just reports something untrue — so each case pins a fact.
"""
import base64
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from sift.config import Config
from sift.detect import detect
from sift.extract import extract
from sift.report import render_json
from sift.summarize import summarize_path
from sift.extract.image import _dimensions, _jpeg_dims, _webp_dims
from sift.extract.pdf import _kerned, _page_count, _text_ops

from _timing import cpu_ratio, linear_bound


def png(width: int, height: int, colour=(70, 130, 180)) -> bytes:
    """A real, decodable PNG — not a header stub."""
    raw = b"".join(b"\x00" + bytes(colour) * width for _ in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


class ImageHeaderTest(unittest.TestCase):
    def test_png_dimensions(self):
        self.assertEqual(_dimensions(png(16, 12)), ("PNG", 16, 12))

    def test_gif_dimensions(self):
        blob = b"GIF89a" + struct.pack("<HH", 640, 480)
        self.assertEqual(_dimensions(blob), ("GIF", 640, 480))

    def test_bmp_dimensions_with_a_negative_height(self):
        """Top-down bitmaps store height as a negative number."""
        header = (b"BM" + (1000).to_bytes(4, "little") + b"\x00" * 4
                  + (54).to_bytes(4, "little") + (40).to_bytes(4, "little")
                  + struct.pack("<ii", 320, -240) + b"\x00" * 8)
        self.assertEqual(_dimensions(header), ("BMP", 320, 240))

    def test_jpeg_dimensions_from_the_frame_marker(self):
        # SOI, then an APP0 whose declared length must be walked over exactly,
        # then SOF0 carrying height before width.
        app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
        sof0 = (b"\xff\xc0" + struct.pack(">H", 17) + b"\x08"
                + struct.pack(">HH", 300, 400) + b"\x03" + b"\x00" * 9)
        self.assertEqual(_jpeg_dims(b"\xff\xd8" + app0 + sof0), (400, 300))

    def test_a_truncated_header_reports_unknown_never_a_guess(self):
        for blob, fmt in ((b"\x89PNG\r\n\x1a\n\x00\x00", "PNG"),
                          (b"GIF89a\x01", "GIF"),
                          (b"BM" + b"\x00" * 8, "BMP"),
                          (b"RIFF\x24\x00\x00\x00WEBPVP8X", "WEBP")):
            self.assertEqual(_dimensions(blob), (fmt, None, None), fmt)

    def test_webp_lossy_and_lossless(self):
        vp8 = (b"RIFF" + b"\x00" * 4 + b"WEBPVP8 " + b"\x00" * 10
               + struct.pack("<HH", 200, 100))
        self.assertEqual(_webp_dims(vp8), (200 & 0x3FFF, 100 & 0x3FFF))

    def test_the_reported_size_is_the_files_size_not_the_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            p.write_bytes(png(16, 12))
            ex = extract(p, detect(p), 20)          # deliberately short read
        self.assertEqual(ex.metadata["size_bytes"], p_size := ex.metadata["size_bytes"])
        self.assertGreater(p_size, 20)
        self.assertTrue(ex.truncated)
        self.assertIsNone(ex.image_data, "half an image must not go to a model")


class PdfTextTest(unittest.TestCase):
    def test_a_kerning_array_becomes_spaces_where_the_gap_is_wide(self):
        self.assertEqual(_kerned(b"(Page)-250(two)-250(here)"), "Page two here")

    def test_a_narrow_kern_is_not_a_space(self):
        self.assertEqual(_kerned(b"(Kern)-20(ing)"), "Kerning")

    def test_positive_kerns_never_insert_a_space(self):
        self.assertEqual(_kerned(b"(A)120(B)"), "AB")

    def test_escaped_parentheses_survive(self):
        self.assertEqual(_kerned(rb"(a\(b\))"), "a(b)")

    def test_a_plain_tj_operator(self):
        self.assertEqual(_text_ops(b"BT (Hello world) Tj ET"), "Hello world")

    def test_page_count_from_the_page_tree(self):
        self.assertEqual(_page_count(b"<< /Type /Pages /Kids [1 2] /Count 7 >>"), 7)
        self.assertEqual(_page_count(b"<< /Count 3 /Type /Pages >>"), 3)

    def test_page_count_does_not_count_the_word_Pages(self):
        """Counting the substring '/Page' also matched '/Pages'."""
        body = b"<< /Type /Page >> << /Type /Page >> << /Type /Pages >>"
        self.assertEqual(_page_count(body), 2)

    def test_a_whole_small_pdf_end_to_end(self):
        stream = zlib.compress(b"BT (Hello from the stream) Tj ET")
        pdf = b"".join([
            b"%PDF-1.4\n",
            b"1 0 obj << /Type /Pages /Count 1 >> endobj\n",
            b"2 0 obj << /Type /Page >> endobj\n",
            b"3 0 obj << /Length " + str(len(stream)).encode()
            + b" /Filter /FlateDecode >> stream\n",
            stream,
            b"\nendstream endobj\n%%EOF\n",
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.pdf"
            p.write_bytes(pdf)
            det = detect(p)
            ex = extract(p, det, 25_000_000)
        self.assertEqual(det.kind, "pdf")
        self.assertIn("Hello from the stream", ex.text)
        self.assertEqual(ex.metadata.get("pages"), 1)

    @classmethod
    def _text_pdf(cls, pages: int, line: bytes, lines: int) -> bytes:
        """A real PDF pypdf reads: ``pages`` pages, each ``lines`` times ``line`` in Helvetica."""
        return cls._content_pdf(pages, b"BT /F1 8 Tf 20 780 Td 8 TL\n" + (b"(" + line + b") '\n") * lines + b"ET")

    @staticmethod
    def _content_pdf(pages: int, content: bytes) -> bytes:
        """A real PDF pypdf reads: ``pages`` pages that share ``content`` as their content stream, font /F1
        Helvetica."""
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   b"<< /Type /Pages /Kids [%s] /Count %d >>"
                   % (b" ".join(b"%d 0 R" % (5 + i) for i in range(pages)), pages),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                   b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"]
        objects += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    b"/Contents 4 0 R >>"] * pages
        return PdfTextTest._assemble(objects)

    @staticmethod
    def _assemble(objects) -> bytes:
        """Number ``objects`` from 1, with a classic xref table and a trailer whose /Root is object 1."""
        out = bytearray(b"%PDF-1.4\n")
        offsets = []
        for number, body in enumerate(objects, 1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
        out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
        return bytes(out)

    def _extract_pdf(self, data: bytes):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.pdf"
            p.write_bytes(data)
            return extract(p, detect(p), 25_000_000)

    def test_the_time_budget_bounds_a_pdf_of_many_pages(self):
        """A per-file budget across pages: pypdf reads each page whole, and a host application's editor runs this in the web
        request, so a document of many pages was bounded only by the request's own timeout. deadline.check runs once
        per page, and extract() re-raises DeadlineExceeded rather than falling to the stdlib reader."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift import deadline
        data = self._text_pdf(50, b"A line of the budget test, the same on every line of every page.", 10)
        real, seen = deadline.check, []

        def expire_on_the_second_page(where=""):
            seen.append(where)
            if where == "a PDF" and seen.count("a PDF") >= 2:
                raise deadline.DeadlineExceeded(60, where)
            return real(where)
        with mock.patch.object(deadline, "check", expire_on_the_second_page), \
                self.assertRaises(deadline.DeadlineExceeded):
            with deadline.limit(60):
                self._extract_pdf(data)
        self.assertGreaterEqual(seen.count("a PDF"), 2)                       # checked once per page

    def test_a_pdf_of_many_pages_reads_whole_within_its_budget(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift import deadline
        data = self._text_pdf(20, b"A line that stays well under the character budget.", 3)
        with deadline.limit(60):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf")
        self.assertEqual(ex.metadata.get("pages"), 20)
        self.assertIn("[page 20]", ex.text)

    def test_pypdf_stops_reading_pages_once_the_text_is_past_what_is_kept(self):
        """A text PDF of 63 MB held all its pages' text, some 60 million characters, before the body was cut to
        MAX_CHARS: 540 MB and 100 s in a host application's editor. The pages past the cut are no longer read."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract.pdf import MAX_CHARS
        read = []
        real = pypdf.PageObject.extract_text

        def counted(page, *args, **kwargs):
            read.append(1)
            return real(page, *args, **kwargs)
        data = self._text_pdf(400, b"A line of the stop test, the same on every line of every page.", 20)
        with mock.patch.object(pypdf.PageObject, "extract_text", counted):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf")
        self.assertEqual(ex.metadata.get("pages"), 400)                       # every page counted still
        self.assertTrue(ex.truncated)
        self.assertLessEqual(len(read), MAX_CHARS // 1200 + 2)                 # a page holds over 1200 characters

    def test_the_body_is_the_same_as_reading_every_page_and_cutting_after(self):
        """The pages joined so far are a prefix of all of them joined, and clip() reads no further than MAX_CHARS:
        stopping once past it gives the same bytes -- under the limit, just over it, and far over it."""
        try:
            import io

            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift.extract.base import clip
        from sift.extract.pdf import MAX_CHARS
        line = b"A line of the sameness test, the same on every line of every page."
        per_page = len(pypdf.PdfReader(io.BytesIO(self._text_pdf(1, line, 20))).pages[0].extract_text())
        for pages in (MAX_CHARS // per_page - 1, MAX_CHARS // per_page, MAX_CHARS // per_page + 1,
                      MAX_CHARS // per_page + 2, 3 * MAX_CHARS // per_page, 1):
            with self.subTest(pages=pages):
                data = self._text_pdf(pages, line, 20)
                reader = pypdf.PdfReader(io.BytesIO(data))
                every = "\n\n".join(f"[page {i+1}]\n{page.extract_text() or ''}".strip()
                                     for i, page in enumerate(reader.pages))
                expected, truncated = clip(every, MAX_CHARS)
                ex = self._extract_pdf(data)
                self.assertEqual(ex.text, expected)
                self.assertEqual(ex.truncated, truncated)

    def test_a_page_that_fails_past_the_cut_is_never_read_and_one_before_it_still_falls_back(self):
        """The one difference to reading every page: a page past the cut that pypdf cannot read no longer throws the
        whole document onto the best-effort reader -- its text would have been cut off anyway. A page before the cut
        that fails still does."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        real = pypdf.PageObject.extract_text
        data = self._text_pdf(400, b"A line of the failing-page test, the same on every line of every page.", 20)
        for failing, engine in ((399, "pypdf"), (0, "stdlib-fallback")):
            seen = []

            def extract_text(page, *args, **kwargs):
                seen.append(1)
                if len(seen) - 1 == failing:
                    raise ValueError("a page pypdf cannot read")
                return real(page, *args, **kwargs)
            with self.subTest(failing=failing), mock.patch.object(pypdf.PageObject, "extract_text", extract_text):
                ex = self._extract_pdf(data)
                self.assertEqual(ex.metadata.get("engine"), engine)
                self.assertEqual(any("pypdf failed" in note for note in ex.notes), engine != "pypdf")
                if engine == "pypdf":                                               # the fallback cuts its own way
                    self.assertTrue(ex.truncated)

    def test_the_stop_is_at_the_limit_itself_not_one_page_early(self):
        """The sameness test above holds on reading every page by construction, and none of its page counts lands the
        joined text on MAX_CHARS itself: ``>=`` for ``>``, or counting from 0 instead of -2, passed it as well. Here
        the pages joined so far are exactly MAX_CHARS long, and one longer, each with pages after. At MAX_CHARS the
        next page still decides ``truncated`` and is read; one past it, no further page is."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract.base import clip
        from sift.extract.pdf import MAX_CHARS

        def text(length: int) -> str:                             # lines of words, nothing for strip() to take
            line = "A line of the limit test.\n"
            return (line * (length // len(line) + 1))[:length - 1] + "x"
        data = self._text_pdf(4, b"A line pypdf is never asked for.", 1)
        for landing in (MAX_CHARS, MAX_CHARS + 1):
            second = landing - len(f"[page 1]\n{text(MAX_CHARS // 2)}") - len("\n\n[page 2]\n")
            texts = [text(MAX_CHARS // 2), text(second), text(500), text(500)]
            parts = [f"[page {i+1}]\n{t}".strip() for i, t in enumerate(texts)]
            self.assertEqual(len("\n\n".join(parts[:2])), landing)                  # the test's own arithmetic
            expected, truncated = clip("\n\n".join(parts), MAX_CHARS)
            needed = next(k for k in range(1, len(parts) + 1) if len("\n\n".join(parts[:k])) > MAX_CHARS)
            read = []

            def extract_text(page, *args, **kwargs):
                read.append(1)
                return texts[len(read) - 1]
            with self.subTest(landing=landing), mock.patch.object(pypdf.PageObject, "extract_text", extract_text):
                ex = self._extract_pdf(data)
                self.assertEqual(len(read), needed)
                self.assertEqual(ex.text, expected)
                self.assertEqual(ex.truncated, truncated)

    def test_one_page_holding_all_the_text_is_parsed_only_up_to_the_bound(self):
        """Stopping between pages bounds nothing when one page holds it all: pypdf parses a page's content stream
        whole, every operator into an object. 810,000 lines on one page, Flate-compressed to 11 MB, had not finished
        after 600 s and held 407 MB; 160,000 lines, a 42 KB file, took 243 s. Now only the first MAX_INFLATED bytes
        of a page's content are parsed -- and a page of plain text holds far more than MAX_CHARS characters in them,
        so the body is still the one reading the whole page gives."""
        try:
            import io

            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract.base import clip
        from sift.extract.pdf import MAX_CHARS, MAX_INFLATED
        line = b"A line of the one-page test, the same on every line of the only page."
        data = self._text_pdf(1, line, MAX_INFLATED // (len(line) + 5) + 100)
        whole = pypdf.PdfReader(io.BytesIO(data)).pages[0]
        self.assertGreater(len(whole.get_contents().get_data()), MAX_INFLATED)     # the test's own premise
        expected, _ = clip(f"[page 1]\n{whole.extract_text()}".strip(), MAX_CHARS)
        parsed = []
        real = pypdf.PageObject.extract_text

        def extract_text(page, *args, **kwargs):
            parsed.append(len(page.get_contents().get_data()))
            return real(page, *args, **kwargs)
        with mock.patch.object(pypdf.PageObject, "extract_text", extract_text):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf")
        self.assertEqual(len(parsed), 1)
        self.assertLessEqual(parsed[0], MAX_INFLATED)
        self.assertEqual(ex.text, expected)
        self.assertTrue(ex.truncated)
        self.assertTrue(any("page 1's content" in note for note in ex.notes), ex.notes)

    def test_a_page_at_the_bound_is_read_whole_and_one_byte_over_it_is_cut(self):
        try:
            import io

            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        data = self._text_pdf(1, b"A line of the bound test, the same on every line.", 20)
        size = len(pypdf.PdfReader(io.BytesIO(data)).pages[0].get_contents().get_data())
        whole = self._extract_pdf(data)
        for bound, cut in ((size, False), (size - 1, True)):
            with self.subTest(bound=bound), mock.patch.object(pdf, "MAX_INFLATED", bound):
                ex = self._extract_pdf(data)
                self.assertEqual(ex.metadata.get("engine"), "pypdf")
                self.assertEqual(any("page 1's content" in note for note in ex.notes), cut, ex.notes)
                self.assertEqual(ex.truncated, cut)
                self.assertEqual(ex.text, whole.text)                   # the byte over is the "ET" that ends the text

    def test_a_cut_inside_a_token_goes_back_to_the_last_whole_one(self):
        """pypdf gives up on a content stream that ends inside a string, an array, a dictionary or an inline image,
        and a page it gives up on sends the whole document to the fallback. A comment is no string, even with a
        parenthesis in it."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        head = b"BT /F1 8 Tf 20 780 Td 8 TL\n(the kept line) '\n"
        for inside in (rb"(a (nested\) string) with (parentheses) in it) '" + b"\n",
                       b"<48656c6c6f2c20776f726c642c20616761696e> Tj\n",
                       b"[(Hel) -250 (lo, wor) 120 (ld, once more)] TJ\n",
                       b"/Span << /ActualText (Hello) /MCID 3 >> BDC (Hello) Tj EMC\n",
                       b"BI /W 8 /H 1 /CS /G /BPC 8 ID )((a)(b( EI\n",
                       b"% a comment with a ( in it, and more words\n"):
            content = head + inside + b"(a line past the cut) '\nET"
            with self.subTest(inside=inside), mock.patch.object(pdf, "MAX_INFLATED", len(head) + len(inside) // 2):
                ex = self._extract_pdf(self._content_pdf(1, content))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.text, "[page 1]\nthe kept line")
                self.assertTrue(ex.truncated)
                self.assertTrue(any("page 1's content" in note for note in ex.notes), ex.notes)

    def test_where_the_content_is_cut(self):
        from sift.extract import pdf
        cut = getattr(pdf, "_content_cut", None)
        self.assertIsNotNone(cut, "sift.extract.pdf has no _content_cut")
        self.assertEqual(cut(b"BT (short) Tj ET", 100), 16)                    # shorter than the bound: all of it
        self.assertEqual(cut(b"BT (short) Tj ET", 16), 16)                     # exactly the bound: all of it
        self.assertEqual(cut(b"BT (short) Tj ET", 14), 13)                     # before the space after "Tj"
        self.assertEqual(cut(b"BT (short) Tj ET", 12), 10)                     # "T" may be "Tj": not whole yet
        self.assertEqual(cut(b"BT (sh) (ort)Tj ET", 9), 7)                     # inside the second string
        self.assertEqual(cut(b"(" + b"x" * 100 + b") Tj", 50), 0)              # no whole token before the bound
        self.assertEqual(cut(b"BT\x00(a)\x00Tj\x00ET", 9), 6)                  # NUL is white space in PDF

    def test_several_pages_cut_are_counted_in_one_note(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        data = self._text_pdf(3, b"A line of the several-pages test.", 20)
        with mock.patch.object(pdf, "MAX_INFLATED", 100):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf")
        notes = [note for note in ex.notes if "content" in note and "bytes" in note]
        self.assertEqual(len(notes), 1, ex.notes)
        self.assertIn("3 pages", notes[0])
        self.assertTrue(ex.truncated)

    @staticmethod
    def _form_pdf(page: bytes, draws, forms: dict, pages: int = 1, compress: bool = False) -> bytes:
        """A real PDF pypdf reads: ``pages`` pages that share content ``page`` and resources that name the XObjects in
        ``draws``. ``forms`` maps a name to (content, the names the form's own resources hold), Flate-compressed, or to
        (content, names, filter) written as is under that /Filter; a name that starts with "Im" is a 256-byte grey
        image instead, and a name in no map points at an object that is not there. Font /F1 Helvetica on the pages and
        in every form. ``compress`` Flate-compresses the pages' content too."""
        number = {name: 5 + pages + i for i, name in enumerate(forms)}

        def resources(names):
            xobjects = b" ".join(b"/%s %d 0 R" % (name, number.get(name, 99)) for name in names)
            return b"<< /Font << /F1 3 0 R >> /XObject << %s >> >>" % xobjects
        body = zlib.compress(page) if compress else page
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   b"<< /Type /Pages /Kids [%s] /Count %d >>"
                   % (b" ".join(b"%d 0 R" % (5 + i) for i in range(pages)), pages),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                   b"<<%s /Length %d >>\nstream\n" % (b" /Filter /FlateDecode" if compress else b"", len(body))
                   + body + b"\nendstream"]
        objects += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources %s /Contents 4 0 R >>"
                    % resources(draws)] * pages
        for name, form in forms.items():
            if name.startswith(b"Im"):
                objects.append(b"<< /Type /XObject /Subtype /Image /Width 256 /Height 1 /ColorSpace /DeviceGray "
                               b"/BitsPerComponent 8 /Length 256 >>\nstream\n" + b"\x80" * 256 + b"\nendstream")
                continue
            content, inner, *written = form
            body = content if written else zlib.compress(content)
            objects.append(b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources %s /Filter %s "
                           b"/Length %d >>\nstream\n" % (resources(inner), written[0] if written else b"/FlateDecode",
                                                          len(body)) + body + b"\nendstream")
        return PdfTextTest._assemble(objects)

    @staticmethod
    def _form_lines(tag: str, lines: int) -> bytes:
        return b"".join(b"BT /F1 8 Tf 20 %d Td (%s line %d) Tj ET\n" % (780 - i, tag.encode(), i)
                        for i in range(lines))

    def test_a_form_xobject_is_parsed_only_up_to_the_bound(self):
        """_bound_content cut the page's own content only, and pypdf parses a form XObject's content whole each time
        the page draws it: one form of 160,000 lines drawn once, a 27 KB file, had not finished after 10 minutes,
        while the same lines as the page's content took 0.95 s under the cut. Now a form is cut where a page is."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        form = self._form_lines("form", 40)
        data = self._form_pdf(b"/Fm0 Do\n", (b"Fm0",), {b"Fm0": (form, ())})
        whole = self._extract_pdf(data)
        self.assertEqual(whole.metadata.get("engine"), "pypdf")
        self.assertIn("form line 39", whole.text)                               # the test's own premise
        self.assertFalse(whole.truncated)
        self.assertFalse(any("form" in note for note in whole.notes), whole.notes)
        with mock.patch.object(pdf, "MAX_INFLATED", len(form) // 2):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertIn("form line 0", ex.text)
        self.assertNotIn("form line 39", ex.text)
        self.assertTrue(ex.truncated)
        self.assertIn(f"read only the first {pdf._content_cut(form, len(form) // 2)} of {len(form)} bytes of a form "
                      f"XObject page 1 draws; its text after that is not in the body", ex.notes)

    def test_a_form_at_the_bound_is_read_whole_and_one_byte_over_it_is_cut(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        form = self._form_lines("form", 5)
        data = self._form_pdf(b"/Fm0 Do\n", (b"Fm0",), {b"Fm0": (form, ())})
        for bound, cut in ((len(form), False), (len(form) - 1, True)):
            with self.subTest(bound=bound), mock.patch.object(pdf, "MAX_INFLATED", bound):
                ex = self._extract_pdf(data)
                self.assertEqual(ex.metadata.get("engine"), "pypdf")
                self.assertEqual(any("form XObject" in note for note in ex.notes), cut, ex.notes)
                self.assertEqual(ex.truncated, cut)
                self.assertIn("form line 4", ex.text)                  # the byte over is the newline after the "ET"

    def test_every_form_the_page_draws_is_cut_however_it_is_reached(self):
        """A form drawn by another form, one drawn by the page and by a form, one drawn three times and one that
        draws itself: each is cut once, counted once, and none of them hangs."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        a, b = self._form_lines("alpha", 40), self._form_lines("beta", 40)
        forms = {b"Fm0": (b"/Fm1 Do\n/Fm2 Do\n", (b"Fm1", b"Fm2")),
                 b"Fm1": (a, ()),
                 b"Fm2": (b + b"/Fm2 Do\n", (b"Fm2",))}
        data = self._form_pdf(b"/Fm0 Do\n/Fm1 Do\n/Fm1 Do\n", (b"Fm0", b"Fm1"), forms)
        with mock.patch.object(pdf, "MAX_INFLATED", len(a) // 2):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        for tag in ("alpha", "beta"):
            self.assertIn(f"{tag} line 0", ex.text)
            self.assertNotIn(f"{tag} line 39", ex.text)
        self.assertEqual(ex.text.count("alpha line 0"), 3)                      # drawn three times, cut each time
        notes = [note for note in ex.notes if "form XObject" in note]
        self.assertEqual(notes, [f"read at most the first {len(a) // 2} bytes of the content of 2 form XObjects, "
                                 f"the first of them drawn on page 1; their text after that is not in the body"])
        self.assertTrue(ex.truncated)

    def test_an_image_xobject_and_a_missing_one_are_left_to_pypdf(self):
        """Only a form has content pypdf parses as operators. An image past the bound is not cut, and a name that
        points at nothing is pypdf's to judge, as before: neither ever reaches sift's form bound, which the form drawn
        beside them does -- without it, this passed before the bound existed."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        text = b"BT /F1 8 Tf 20 700 Td (the line of the page itself) Tj ET\n"
        page = b"q 256 0 0 1 0 0 cm /Im0 Do Q\n/Gone Do\n/Fm0 Do\n" + text
        data = self._form_pdf(page, (b"Im0", b"Gone", b"Fm0"),
                              {b"Im0": None, b"Fm0": (self._form_lines("form", 2), ())})
        self.assertLess(len(page), 200)                         # the test's own premise: only the image is past it
        handed, real = [], pdf._bound_form

        def bound_form(xform, *args, **kwargs):
            handed.append(xform.get("/Subtype"))
            return real(xform, *args, **kwargs)
        with mock.patch.object(pdf, "MAX_INFLATED", 200), mock.patch.object(pdf, "_bound_form", bound_form):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertIn("the line of the page itself", ex.text)
        self.assertIn("form line 1", ex.text)
        self.assertEqual(handed, ["/Form"])                     # the image and the missing name never reach it
        self.assertFalse(any("form XObject" in note for note in ex.notes), ex.notes)
        self.assertFalse(ex.truncated)

    def test_the_time_budget_is_checked_inside_a_page_and_its_forms(self):
        """deadline.check ran between pages only, and pypdf reads a page and every form it draws in one call: a
        972-byte page drawing one 200-line form 5,000 times took 74.7 s, past a 60 s budget nothing looked at. Now
        the budget is checked before every operator pypdf runs, the forms' own included. pypdf swallows any exception
        raised inside a form and goes on with the page, so the check after the page is what raises: a one-page file
        whose budget ran out inside its form came back with part of its text and not a word about it."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift import deadline
        form = self._form_lines("form", 20)
        data = self._form_pdf(b"/Fm0 Do\n", (b"Fm0",), {b"Fm0": (form, ())})
        real, seen = deadline.check, []

        def expire_inside_the_form(where=""):
            seen.append(where)
            if len(seen) >= 10:                             # one check per page before, one per operator now
                raise deadline.DeadlineExceeded(60, where)
            return real(where)
        with mock.patch.object(deadline, "check", expire_inside_the_form), \
                self.assertRaises(deadline.DeadlineExceeded) as raised:
            with deadline.limit(60):
                self._extract_pdf(data)
        self.assertEqual(raised.exception.where, "a page of a PDF")
        self.assertEqual(len(seen), 11)                     # the one inside the form, then the one after the page

    def _extract_keeping_the_reader(self, data: bytes):
        """(the extraction, the PdfReader sift read it with) -- to look at what pypdf still holds afterwards.

        It put a function in pypdf.PdfReader's place, and sift, finding no PdfReader it could bound, read these
        documents with that plain reader: sift's own reader is kept now."""
        from unittest import mock
        from sift.extract import pdf
        readers, real = [], pdf._open

        def reader(*args, **kwargs):
            readers.append(real(*args, **kwargs))
            return readers[-1]
        with mock.patch.object(pdf, "_open", reader):
            ex = self._extract_pdf(data)
        self.assertEqual(len(readers), 1)
        return ex, readers[0]

    def test_a_form_drawn_again_is_parsed_once(self):
        """sift handed pypdf a stream for each form, and pypdf wraps a stream in a new ContentStream and parses it on
        every draw: one form of 1.12 MB, cut to 800,000 bytes and drawn 20 times, took 16.0 s against 1.38 s drawn
        once. Now pypdf is handed a ContentStream parsed once per page, whether the form was cut or not."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from pypdf.generic import ContentStream
        parses, real = [], ContentStream._parse_content_stream

        def parse(stream, *args, **kwargs):
            parses.append(1)
            return real(stream, *args, **kwargs)
        data = self._form_pdf(b"/Fm0 Do\n" * 20, (b"Fm0",), {b"Fm0": (self._form_lines("form", 20), ())})
        with mock.patch.object(ContentStream, "_parse_content_stream", parse):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text.count("form line 19"), 20)                      # every draw still read
        self.assertEqual(len(parses), 2)                                          # the page's content, the form once
        self.assertFalse(ex.truncated)
        self.assertEqual(ex.notes, [])

    def test_a_form_shared_by_pages_is_cut_once_and_counted_once(self):
        """What was cut was kept per page: a letterhead past the bound on 5 pages was cut 5 times, and the note spoke
        of "5 form XObjects" where there was one. Now a form is cut once per reader and counted once."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        form = self._form_lines("letterhead", 40)
        data = self._form_pdf(b"/Fm0 Do\n/Fm0 Do\n", (b"Fm0",), {b"Fm0": (form, ())}, pages=3)
        from pypdf.generic import ContentStream
        cuts, real, parses, parse = [], pdf._bound_form, [], ContentStream._parse_content_stream

        def bound_form(xform, *args, **kwargs):
            cuts.append(1)
            return real(xform, *args, **kwargs)

        def counted(stream, *args, **kwargs):
            parses.append(1)
            return parse(stream, *args, **kwargs)
        with mock.patch.object(pdf, "MAX_INFLATED", len(form) // 2), \
                mock.patch.object(pdf, "_bound_form", bound_form), \
                mock.patch.object(ContentStream, "_parse_content_stream", counted):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text.count("letterhead line 0"), 6)                  # two draws on each of three pages
        self.assertNotIn("letterhead line 39", ex.text)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(len(parses), 3 + 1)                                      # each page's content, the form once
        self.assertEqual([note for note in ex.notes if "form XObject" in note],
                         [f"read only the first {pdf._content_cut(form, len(form) // 2)} of {len(form)} bytes of a "
                          f"form XObject page 1 draws; its text after that is not in the body"])
        self.assertTrue(ex.truncated)

    def test_what_is_kept_parsed_across_pages_is_bounded(self):
        """A form parsed on one page is kept parsed for the next only up to MAX_FORM_CONTENT bytes of forms in all --
        the operations of 800,000 bytes of path content hold 35 MB. Four pages drawing Fm0, Fm1, Fm1, Fm0, each form
        the whole bound: Fm0 is kept and parsed once, Fm1 no longer fits and is parsed on each page that draws it."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from pypdf.generic import ContentStream
        from sift.extract import pdf
        zero, one = self._form_lines("zero", 5), self._form_lines("one-", 5)
        self.assertEqual(len(zero), len(one))                                    # the test's own premise
        draws = (b"Fm0", b"Fm1", b"Fm1", b"Fm0")
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   b"<< /Type /Pages /Kids [%s] /Count 4 >>" % b" ".join(b"%d 0 R" % (6 + 2 * i) for i in range(4)),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        for name, content in ((b"Fm0", zero), (b"Fm1", one)):
            objects.append(b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> "
                           b">> /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        for i, name in enumerate(draws):
            objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Fm0 4 0 R "
                           b"/Fm1 5 0 R >> >> /Contents %d 0 R >>" % (7 + 2 * i))
            objects.append(b"<< /Length 8 >>\nstream\n/%s Do\n\nendstream" % name)
        parses, real = [], ContentStream._parse_content_stream

        def parse(stream, *args, **kwargs):
            parses.append(1)
            return real(stream, *args, **kwargs)
        with mock.patch.object(pdf, "MAX_FORM_CONTENT", len(zero)), \
                mock.patch.object(ContentStream, "_parse_content_stream", parse):
            ex = self._extract_pdf(self._assemble(objects))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual((ex.text.count("zero line 4"), ex.text.count("one- line 4")), (2, 2))
        self.assertEqual(len(parses), 4 + 1 + 2)                     # each page's content, Fm0 once, Fm1 on each page
        self.assertFalse(ex.truncated)

    def test_the_form_content_one_page_hands_pypdf_is_bounded(self):
        """A bound that does not depend on the clock: the content of every form a page draws, counted on every draw,
        is at most MAX_FORM_CONTENT bytes -- the bound on the page's own content. At it every draw is read, one byte
        under it the last draw is not, and the next page starts from nothing."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        form = self._form_lines("form", 5)
        data = self._form_pdf(b"/Fm0 Do\n" * 3, (b"Fm0",), {b"Fm0": (form, ())}, pages=2)
        for bound, read in ((3 * len(form), 3), (3 * len(form) - 1, 2)):
            with self.subTest(bound=bound), mock.patch.object(pdf, "MAX_FORM_CONTENT", bound):
                ex = self._extract_pdf(data)
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.text.count("form line 4"), 2 * read)
                skipped = [note for note in ex.notes if note.startswith("skipped")]
                self.assertEqual(ex.truncated, read < 3)
                if read == 3:
                    self.assertEqual(skipped, [])
                else:
                    self.assertEqual(skipped, [f"skipped the form XObjects 2 pages draw past the first {bound} bytes "
                                               f"of their content on each, the first of them page 1; their text is "
                                               f"not in the body"])
        with mock.patch.object(pdf, "MAX_FORM_CONTENT", 2 * len(form)):
            ex = self._extract_pdf(self._form_pdf(b"/Fm0 Do\n" * 3, (b"Fm0",), {b"Fm0": (form, ())}))
        self.assertIn(f"skipped the form XObjects page 1 draws after the first {2 * len(form)} bytes of their "
                      f"content; their text is not in the body", ex.notes)

    def test_a_form_pypdf_cannot_decode_is_tried_once_and_said(self):
        """A form pypdf cannot decode -- one that inflates past its 75,000,000-byte limit took 0.39 s drawn once and
        6.16 s drawn 100 times -- was decoded again on every draw, and its text left out without a word. Now it is
        tried once, pypdf still logs why, and the body says the text is missing."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf.filters
        text = b"BT /F1 8 Tf 20 700 Td (the line of the page itself) Tj ET\n"
        data = self._form_pdf(text + b"/Fm0 Do\n" * 5, (b"Fm0",),
                              {b"Fm0": (self._form_lines("form", 2), (), b"/NoSuchFilter")})
        decoded, real = [], pypdf.filters.decode_stream_data

        def decode(stream, *args, **kwargs):
            decoded.append(stream.get("/Filter"))
            return real(stream, *args, **kwargs)
        with mock.patch.object(pypdf.filters, "decode_stream_data", decode), \
                self.assertLogs("pypdf", "WARNING") as logged:
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertIn("the line of the page itself", ex.text)
        self.assertNotIn("form line", ex.text)
        self.assertEqual(decoded.count("/NoSuchFilter"), 1)
        self.assertEqual(sum("XFormObject" in line for line in logged.output), 1, logged.output)
        self.assertIn("could not read a form XObject page 1 draws; its text is not in the body", ex.notes)
        self.assertTrue(ex.truncated)
        broken = self._form_pdf(b"/Fm0 Do\n/Fm1 Do\n", (b"Fm0", b"Fm1"),
                                {b"Fm0": (b"x", (), b"/NoSuchFilter"), b"Fm1": (b"y", (), b"/NoSuchFilter")}, pages=2)
        with self.assertLogs("pypdf", "WARNING"):
            ex = self._extract_pdf(broken)
        self.assertIn("could not read 2 form XObjects, the first of them drawn on page 1; their text is not in the "
                      "body", ex.notes)

    def test_a_form_pypdf_cannot_parse_is_tried_once_and_said(self):
        """A form whose content pypdf gives up on half-way -- here an inline image with nothing after "BI" -- loses
        all its text, as it did. pypdf's ContentStream keeps the operations it read before it gave up, so a stream
        parsed once for every draw must not be handed over again: that would read text pypdf never reads."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from pypdf.generic import ContentStream
        text = b"BT /F1 8 Tf 20 700 Td (the line of the page itself) Tj ET\n"
        data = self._form_pdf(text + b"/Fm0 Do\n" * 3, (b"Fm0",), {b"Fm0": (self._form_lines("broken", 2) + b"BI", ())})
        parses, real = [], ContentStream._parse_content_stream

        def parse(stream, *args, **kwargs):
            parses.append(1)
            return real(stream, *args, **kwargs)
        with mock.patch.object(ContentStream, "_parse_content_stream", parse), \
                self.assertLogs("pypdf", "WARNING") as logged:
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertIn("the line of the page itself", ex.text)
        self.assertNotIn("broken line", ex.text)
        self.assertEqual(len(parses), 2)                                          # the page's content, the form once
        self.assertEqual(sum("XFormObject" in line for line in logged.output), 1, logged.output)
        self.assertEqual(ex.notes, ["could not read a form XObject page 1 draws; its text is not in the body"])
        self.assertTrue(ex.truncated)

    def test_what_pypdf_decoded_is_not_kept_past_the_cut(self):
        """pypdf keeps what it decoded on the reader's own object, so a cut kept the whole of it anyway: eight forms of
        70 MB grew the process by 67 MB each, to 593 MB. The original's decoded copy is dropped once it is cut, the
        page's own content as well; within the bound it stays for the page, since pypdf reads it from there again,
        and goes with the page."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from pypdf.generic import IndirectObject
        from sift.extract import pdf
        form, small = self._form_lines("form", 40), self._form_lines("small", 2)
        page = b"/Fm0 Do\n/Fm1 Do\n" + self._form_lines("page", 40)                   # the draws before the cut
        data = self._form_pdf(page, (b"Fm0", b"Fm1"), {b"Fm0": (form, ()), b"Fm1": (small, ())}, compress=True)
        with mock.patch.object(pdf, "MAX_INFLATED", len(form) // 2):
            ex, reader = self._extract_keeping_the_reader(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertIn("small line 1", ex.text)
        self.assertTrue(any("page 1's content" in note for note in ex.notes), ex.notes)   # the premise: both are cut
        self.assertTrue(any("form XObject" in note for note in ex.notes), ex.notes)
        content, cut, kept = (IndirectObject(number, 0, reader).get_object() for number in (4, 6, 7))
        self.assertIsNone(content.decoded_self)
        self.assertIsNone(cut.decoded_self)
        self.assertIsNone(kept.decoded_self)

    _CMAP = (b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n1 begincodespacerange\n<00> <FF>\n"
             b"endcodespacerange\n2 beginbfchar\n<41> <005A>\n<42> <0059>\nendbfchar\nendcmap\nend\nend\n")

    @classmethod
    def _cmap(cls, size: int) -> bytes:
        """The ToUnicode map that reads "ABBA" as "ZYYZ", padded with a comment to ``size`` bytes."""
        return cls._CMAP + b"%" + b"x" * (size - len(cls._CMAP) - 2) + b"\n"

    @staticmethod
    def _cmap_pdf(cmap: bytes, encoding: bytes = b"/FlateDecode", pages: int = 1, draws: int = 0) -> bytes:
        """A real PDF pypdf reads: ``pages`` pages that show "ABBA" in /F1, Helvetica with ``cmap`` as its /ToUnicode,
        written through ``encoding`` (b"" for none; ASCIIHex in lines of 64 digits). With ``draws``, the text is in a
        form XObject each page draws that often instead, and the font is in the form's resources only. The map is
        object 4."""
        raw = {b"/FlateDecode": zlib.compress,
               b"/ASCIIHexDecode": lambda b: re.sub(rb"(.{64})", rb"\1\n", b.hex().encode()) + b">",
               b"[/FlateDecode /FlateDecode]": lambda b: zlib.compress(zlib.compress(b)),
               b"/ASCII85Decode": lambda b: base64.a85encode(b, adobe=True), b"": lambda b: b}[encoding](cmap)
        text = b"BT /F1 12 Tf 20 700 Td (ABBA) Tj ET\n"
        page, resources = ((b"/Fm0 Do\n" * draws, b"<< /XObject << /Fm0 6 0 R >> >>") if draws
                           else (text, b"<< /Font << /F1 3 0 R >> >>"))
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   b"<< /Type /Pages /Kids [%s] /Count %d >>"
                   % (b" ".join(b"%d 0 R" % (7 + i) for i in range(pages)), pages),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 4 0 R >>",
                   b"<<%s /Length %d >>\nstream\n" % (b" /Filter " + encoding if encoding else b"", len(raw))
                   + raw + b"\nendstream",
                   b"<< /Length %d >>\nstream\n" % len(page) + page + b"\nendstream",
                   b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                   b"/Length %d >>\nstream\n" % len(text) + text + b"\nendstream"]
        objects += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources %s /Contents 5 0 R >>"
                    % resources] * pages
        return PdfTextTest._assemble(objects)

    def test_a_tounicode_map_within_the_bound_is_read(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        for encoding in (b"", b"/FlateDecode", b"/ASCIIHexDecode"):
            with self.subTest(encoding=encoding):
                ex, reader = self._extract_keeping_the_reader(self._cmap_pdf(self._CMAP, encoding))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.text, "[page 1]\nZYYZ")                    # the map applied
                self.assertFalse(ex.truncated)
                self.assertEqual(ex.notes, [])

    def test_a_tounicode_map_past_the_bound_is_ignored_and_said(self):
        """pypdf reads every /ToUnicode map of every font in the resources -- whole, split and joined several times
        over, and again for the page and for every form draw -- before the first operator: a 30 KB file whose map
        inflated to 30 MB took 2.99 s and 1400 MB, and drawn five times 13.65 s. A map past MAX_TO_UNICODE is now
        left out of a copy of the font pypdf is given. Flate is measured by inflating to one byte past the bound,
        never further; another filter is decoded by pypdf and the decoded copy dropped."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        ex, reader = self._extract_keeping_the_reader(self._cmap_pdf(self._cmap(pdf.MAX_TO_UNICODE + 1)))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text, "[page 1]\nABBA")                              # the map not applied
        self.assertEqual(ex.notes, [f"ignored the ToUnicode map of a font past {pdf.MAX_TO_UNICODE} bytes' worth of "
                                    f"reading; its text may be wrong"])
        self.assertTrue(ex.truncated)
        from pypdf.generic import IndirectObject
        self.assertIsNone(IndirectObject(4, 0, reader).get_object().decoded_self)   # never inflated whole
        with mock.patch.object(pdf, "MAX_TO_UNICODE", 1000):
            for encoding in (b"", b"/ASCIIHexDecode"):
                with self.subTest(encoding=encoding):
                    ex, reader = self._extract_keeping_the_reader(self._cmap_pdf(self._cmap(2000), encoding))
                    self.assertEqual(ex.text, "[page 1]\nABBA")
                    self.assertTrue(ex.truncated)
                    self.assertTrue(any("ToUnicode" in note for note in ex.notes), ex.notes)
                    self.assertIsNone(getattr(IndirectObject(4, 0, reader).get_object(), "decoded_self", None))

    def test_a_tounicode_map_at_the_bound_is_read_and_one_byte_over_it_is_not(self):
        """At MAX_TO_UNICODE the map is read, and one unit of reading past it, or one byte, it is not: its bytes are
        measured through every filter before it is read, and what reading it costs counted from them (_map_reading)."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        cmap = self._cmap(1000)
        units = pdf._map_reading(cmap, 10 ** 12).units
        self.assertGreater(units, len(cmap))                                     # its lines and entries
        for encoding in (b"", b"/FlateDecode", b"/ASCIIHexDecode", b"[/FlateDecode /FlateDecode]", b"/ASCII85Decode"):
            for bound, read in ((units, True), (units - 1, False), (len(cmap) - 1, False)):
                with self.subTest(encoding=encoding, bound=bound), mock.patch.object(pdf, "MAX_TO_UNICODE", bound):
                    ex = self._extract_pdf(self._cmap_pdf(cmap, encoding))
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(ex.text, "[page 1]\n" + ("ZYYZ" if read else "ABBA"))
                    self.assertEqual(ex.truncated, not read)
                    self.assertEqual(any("ToUnicode" in note for note in ex.notes), not read, ex.notes)

    def test_a_font_is_measured_once_however_often_pypdf_reads_it(self):
        """pypdf reads a font's map for the page and again for every form draw that names it. sift measures each font
        once per reader, on pages and in forms, and counts it once."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        for draws in (0, 3):
            measured, real = [], pdf._decoded_size

            def decoded_size(stream, *args, **kwargs):
                measured.append(1)
                return real(stream, *args, **kwargs)
            with self.subTest(draws=draws), mock.patch.object(pdf, "MAX_TO_UNICODE", 1000), \
                    mock.patch.object(pdf, "_decoded_size", decoded_size):
                ex = self._extract_pdf(self._cmap_pdf(self._cmap(2000), pages=3, draws=draws))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.text.count("ABBA"), 3 * max(draws, 1))
                self.assertEqual(len(measured), 1)
                self.assertEqual([note for note in ex.notes if "ToUnicode" in note],
                                 ["ignored the ToUnicode map of a font past 1000 bytes' worth of reading; its text "
                                  "may be wrong"])

    #: What a font past the page's font budget is handed without.
    _WITHOUT = "without their ToUnicode maps, widths, encoding differences, glyph procedures and font files"

    def test_the_maps_pypdf_reads_for_one_page_are_bounded(self):
        """pypdf reads every map of the fonts in play again for the page and for every form draw, so a map within
        MAX_TO_UNICODE was bounded per draw only: 3 MB drawn 50 times by a 4 KB file took 13.5 s. Every font a page
        loads is charged FONT_COST and its map against MAX_FONT_LOAD_PER_PAGE; draws past it get their fonts without
        their maps. At the budget every draw keeps its map, one byte under it the last does not, and each page starts
        from nothing -- but for what counting the map cost sift, charged to the page that first names its font. The
        page's own fonts are charged too."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        cmap = self._cmap(1000)
        reading = pdf._map_reading(cmap, 10 ** 12)
        load = pdf.FONT_COST + reading.units                            # one draw: the form names /F1, nothing else
        first = len(cmap) // pdf.MAP_HASH_BYTES_A_UNIT + reading.read   # knowing and counting the map, on page 1
        self.assertLess(first, load)                                    # the test's own premise
        for bound, read in ((3 * load + first, (3, 3)), (3 * load + first - 1, (2, 3)), (3 * load - 1, (2, 2))):
            with self.subTest(bound=bound), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
                ex = self._extract_pdf(self._cmap_pdf(cmap, pages=2, draws=3))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual((ex.text.count("ZYYZ"), ex.text.count("ABBA")), (sum(read), 6 - sum(read)))
                self.assertEqual(ex.truncated, read != (3, 3))
                self.assertEqual([note for note in ex.notes if "font budget" in note],
                                 [] if read == (3, 3) else
                                 [f"page 1 loads fonts past a page's font budget of {bound}: 1 of them "
                                  f"{self._WITHOUT}; their text may be wrong"] if read == (2, 3) else
                                 [f"2 pages load fonts past a page's font budget of {bound}, the first of them page "
                                  f"1: 2 of them {self._WITHOUT}; their text may be wrong"])
        with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", 3 * load - 1):
            ex = self._extract_pdf(self._cmap_pdf(cmap, draws=3))
        self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {3 * load - 1}: 1 of them "
                                    f"{self._WITHOUT}; their text may be wrong"])
        for bound, read in ((load + first, True), (load + first - 1, False)):
            with self.subTest(own=bound), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
                ex = self._extract_pdf(self._cmap_pdf(cmap))
                self.assertEqual(ex.text, "[page 1]\n" + ("ZYYZ" if read else "ABBA"))
                self.assertEqual(ex.notes, [] if read else
                                 [f"page 1 loads fonts past a page's font budget of {bound}: 1 of them "
                                  f"{self._WITHOUT}; their text may be wrong"])

    def test_a_map_within_the_bound_drawn_fifty_times_is_read_once(self):
        """The reviewer's case: a map of 3,000,000 bytes, under MAX_TO_UNICODE, in a form a page draws 50 times.
        pypdf read it 50 times, 13.5 s; now once, and the other 49 draws are said to be without it. sift prepares it
        itself to count what reading it costs (_map_reading, _prepared)."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf._cmap
        from pypdf.generic import StreamObject
        from sift.extract import pdf
        cmap = b"<a>" * 1_000_000
        self.assertLess(len(cmap), pdf.MAX_TO_UNICODE)                        # the test's own premise
        read, real = [], pypdf._cmap.prepare_cm

        def prepare_cm(font, *args, **kwargs):
            if isinstance(font["/ToUnicode"], StreamObject):
                read.append(1)
            return real(font, *args, **kwargs)
        with mock.patch.object(pypdf._cmap, "prepare_cm", prepare_cm):
            ex = self._extract_pdf(self._cmap_pdf(cmap, draws=50))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(len(read), 1)                                      # pypdf's one build
        self.assertEqual(ex.text.count("ABBA"), 50)                         # this map maps nothing: text unchanged
        self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {pdf.MAX_FONT_LOAD_PER_PAGE}: 49 "
                                    f"of them {self._WITHOUT}; their text may be wrong"])
        self.assertTrue(ex.truncated)

    @staticmethod
    def _cjk_pdf(draws: int) -> bytes:
        """A real PDF pypdf reads: one page drawing ``draws`` times a form that shows "中文" in a Type0 font,
        Identity-H, whose /ToUnicode maps every two-byte code to itself -- 65,536 entries, 924,291 bytes."""
        blocks = b"".join(b"256 beginbfchar\n" + b"".join(b"<%04X> <%04X>\n" % (code, code)
                                                            for code in range(start, start + 256)) + b"endbfchar\n"
                          for start in range(0, 65536, 256))
        cmap = zlib.compress(b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n1 begincodespacerange\n"
                             b"<0000> <FFFF>\nendcodespacerange\n" + blocks + b"endcmap\nend\nend\n")
        text, page = b"BT /F1 12 Tf 20 700 Td <4E2D6587> Tj ET\n", b"/Fm0 Do\n" * draws
        return PdfTextTest._assemble([
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [7 0 R] /Count 1 >>",
            b"<< /Type /Font /Subtype /Type0 /BaseFont /Song /Encoding /Identity-H /DescendantFonts [8 0 R] "
            b"/ToUnicode 4 0 R >>",
            b"<< /Filter /FlateDecode /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream",
            b"<< /Length %d >>\nstream\n" % len(page) + page + b"\nendstream",
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            b"/Length %d >>\nstream\n" % len(text) + text + b"\nendstream",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Fm0 6 0 R >> >> "
            b"/Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Song /CIDSystemInfo << /Registry (Adobe) "
            b"/Ordering (Identity) /Supplement 0 >> /FontDescriptor 9 0 R /DW 1000 >>",
            b"<< /Type /FontDescriptor /FontName /Song /Flags 4 /FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 880 "
            b"/Descent -120 /CapHeight 700 /StemV 80 >>"])

    def test_a_full_two_byte_map_is_read_once_on_a_page(self):
        """The largest map a real font has, every two-byte code: its 65,536 bfchar entries cost pypdf 258 to 303 ms a
        build -- 2.9 million units -- not the 924,291 bytes it was charged, and a page read it three times over a
        form's draws within a budget of 3,200,000. As written it fits a page's budget once: the first draw
        is read with it, the other two without, said; the costliest map of 111 real documents cost 11,967. Its codes
        map onto themselves, and handed to pypdf as bfrange lines (_as_ranges) it costs 1.42 million: two draws are
        read with it."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        for ranges_from, whole in ((pdf.MAP_RANGES_FROM, 2), (10 ** 12, 1)):
            for draws in (1, 3):
                with self.subTest(draws=draws, whole=whole), mock.patch.object(pdf, "MAP_RANGES_FROM", ranges_from):
                    ex, reader = self._extract_keeping_the_reader(self._cjk_pdf(draws))
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(ex.text, "[page 1]\n" + "\n".join(["中文"] * draws))   # Identity-H: the same
                    self.assertEqual(ex.notes, [] if draws <= whole else
                                     [f"page 1 loads fonts past a page's font budget of {pdf.MAX_FONT_LOAD_PER_PAGE}: "
                                      f"{draws - whole} of them {self._WITHOUT}; their text may be wrong"])
                    self.assertEqual(ex.truncated, draws > whole)
        from pypdf.generic import IndirectObject
        full = pdf._map_reading(IndirectObject(4, 0, reader).get_object().get_data(), 10 ** 12)
        self.assertEqual(full.entries, 65536)
        self.assertLess(pdf.FONT_COST + full.units, pdf.MAX_FONT_LOAD_PER_PAGE)
        self.assertGreater(2 * (pdf.FONT_COST + full.units), pdf.MAX_FONT_LOAD_PER_PAGE)

    @staticmethod
    def _stream(data: bytes, flate: bool = True) -> bytes:
        body = zlib.compress(data) if flate else data
        return (b"<<%s /Length %d >>\nstream\n" % (b" /Filter /FlateDecode" if flate else b"", len(body)) + body
                + b"\nendstream")

    @staticmethod
    def _fonts_pdf(font: bytes, names: int, extra=(), draws: int = 0, form: bytes = None) -> bytes:
        """A real PDF pypdf reads: "hello" shown in /T, Helvetica, with ``font`` named ``names`` times after it, /F0
        on, in the page's resources -- or with ``draws`` in those of a form XObject the page draws that often, or
        in the form's dictionary ``form`` in their place. ``font`` is object 4, the form 6; ``extra`` are objects 8
        and on."""
        fonts = b"/T 3 0 R " + b" ".join(b"/F%d 4 0 R" % i for i in range(names))
        text = b"BT /T 12 Tf 20 700 Td (hello) Tj ET\n"
        page, resources = ((b"/Fm0 Do\n" * draws, b"<< /XObject << /Fm0 6 0 R >> >>") if draws
                           else (text, b"<< /Font << %s >> >>" % fonts))
        form = b"/Resources << /Font << %s >> >>" % fonts if form is None else form
        return PdfTextTest._assemble([
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [7 0 R] /Count 1 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            font,
            b"<< /Length %d >>\nstream\n" % len(page) + page + b"\nendstream",
            b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] %s /Length %d >>\n"
            b"stream\n" % (form, len(text)) + text + b"\nendstream",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources %s /Contents 5 0 R >>" % resources,
            *extra])

    @staticmethod
    def _parts(font) -> str:
        """What pypdf reads of ``font`` beyond the dictionary itself, "+"-joined, or "bare"."""
        from pypdf.generic import DictionaryObject, StreamObject

        def at(owner, key):
            return owner[key] if isinstance(owner, DictionaryObject) and key in owner else None
        parts = []
        if isinstance(at(font, "/ToUnicode"), StreamObject):
            parts.append("map")
        if any(at(descendant.get_object(), "/W") is not None for descendant in at(font, "/DescendantFonts") or ()):
            parts.append("W")
        if at(at(font, "/Encoding"), "/Differences") is not None:
            parts.append("Differences")
        if at(font, "/CharProcs"):
            parts.append("CharProcs")
        if at(at(font, "/FontDescriptor"), "/FontFile") is not None:
            parts.append("FontFile")
        return "+".join(parts) or "bare"

    def _loads(self, data: bytes, built=None):
        """(the extraction, what pypdf built each font of /BaseFont /X from, in order: its parts, see _parts), recorded
        into ``built`` where given."""
        from unittest import mock
        import pypdf._font
        built, real = [] if built is None else built, pypdf._font.Font.from_font_resource.__func__

        def record(cls, font):
            if font.get("/BaseFont") == "/X":
                built.append(self._parts(font))
            return real(cls, font)
        with mock.patch.object(pypdf._font.Font, "from_font_resource", classmethod(record)):
            return self._extract_pdf(data), built

    #: A Type1 font program with one encoding entry, 2,000 bytes; pypdf reads it for a Type1 font without a map.
    _PROGRAM = b"%!PS-AdobeFont-1.0: X\n/Encoding 256 array\ndup 65 /A put\nreadonly def\n%"
    _PROGRAM += b"x" * (2000 - len(_PROGRAM) - 1) + b"\n"

    def _font_kinds(self):
        """kind -> (the font, its extra objects, what pypdf reads of it beyond the dictionary, in FONT_COST's
        units, its parts as _parts names them): one of every part a font is charged for, and a /W whose elements
        are more than its widths -- pypdf walks them all."""
        from sift.extract import pdf
        self.assertEqual(len(self._PROGRAM), 2000)                                  # the test's own premise
        descendant = (b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) "
                      b"/Ordering (Identity) /Supplement 0 >> /W [0 999 500] >>")
        composite = b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [%s] >>"
        return {
            "W": (composite % descendant, (), pdf.WIDTH_COST * 1000, "W"),
            "W elements": (composite % descendant.replace(b"[0 999 500]", b"[" + b"0 [] " * 500 + b"]"), (),
                           pdf.WIDTH_COST * 1000, "W"),
            "Differences": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding << /Differences [0"
                            + b" /a" * 1000 + b"] >> >>", (), 1001, "Differences"),
            "CharProcs": (b"<< /Type /Font /Subtype /Type3 /BaseFont /X /FontBBox [0 0 1 1] /FontMatrix [0.001 0 0 "
                          b"0.001 0 0] /CharProcs << " + b" ".join(b"/g%d 8 0 R" % i for i in range(1000))
                          + b" >> /FirstChar 0 /LastChar 0 /Widths [1] >>", (self._stream(b""),), 1000, "CharProcs"),
            "FontFile": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 8 0 R >>",
                         (b"<< /Type /FontDescriptor /FontName /X /Flags 32 /FontFile 9 0 R >>",
                          self._stream(self._PROGRAM)), 2000, "FontFile"),
            "map": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /ToUnicode 8 0 R >>",
                    (self._stream(self._cmap(2000)),), pdf._map_reading(self._cmap(2000), 10 ** 12).units, "map"),
        }

    def test_the_fonts_pypdf_loads_for_one_page_are_bounded_by_what_they_cost(self):
        """pypdf builds a Font for every name in /Font before the first operator -- the page's, and every form draw's
        -- and reads for each what the font holds: up to 100,000 /W widths, a /Differences of any length, the
        /CharProcs of a Type3 font, the font file of a Type1 font without a map. Only the /ToUnicode map was bounded:
        one /W font named 170 times, a 4,807-byte file, took 3.3 s and 1,034 MB. Every name is now charged
        FONT_COST and what pypdf reads of it against MAX_FONT_LOAD_PER_PAGE: a name that does not fit whole is handed
        without those parts, and one that does not fit even so is left out. At the budget two names are read whole,
        one unit under it the last name is left out, one name's worth over it a third is read whole."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        unit = pdf.FONT_COST
        for kind, (font, extra, cost, parts) in self._font_kinds().items():
            self.assertGreater(cost, unit)                      # the test's own premise: no name fits whole in two bare
            first = (2000 // pdf.MAP_HASH_BYTES_A_UNIT + pdf._map_reading(self._cmap(2000), 10 ** 12).read
                     if kind == "map" else 0)                   # knowing and counting its map, once
            budget = first + unit + 2 * (unit + cost) + 2 * unit        # /T, two names whole, two bare
            for bound, loads, left_out in ((budget, [parts] * 2 + ["bare"] * 2, 0),
                                           (budget - 1, [parts] * 2 + ["bare"], 1),
                                           (budget + cost, [parts] * 3 + ["bare"], 0)):
                with self.subTest(kind=kind, bound=bound), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
                    ex, built = self._loads(self._fonts_pdf(font, 4, extra))
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(built, loads)
                    self.assertIn("hello", ex.text)
                    self.assertTrue(ex.truncated)
                    stripped = loads.count("bare")
                    self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {bound}: {stripped} "
                                                f"of them {self._WITHOUT}"
                                                + (f", {left_out} left out altogether" if left_out else "")
                                                + "; their text may be wrong"])
        font, extra, cost, _ = self._font_kinds()["W"]
        bound = 2 * (2 * unit + cost) + 2 * unit                    # two draws whole, /T and /F0 bare on the third
        with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
            ex, built = self._loads(self._fonts_pdf(font, 1, extra, draws=3))
        self.assertEqual(built, ["W", "W", "bare"])                  # charged on every draw
        self.assertEqual(ex.text.count("hello"), 3)
        self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {bound}: 1 of them "
                                    f"{self._WITHOUT}; their text may be wrong"])

    def test_a_font_box_pypdf_would_drop_is_not_handed_to_it(self):
        """pypdf floats every element of a /FontBBox -- a font's descriptor's, a Type3 font's own, each descendant's
        descriptor's -- once per name, before it keeps only one of exactly four (Font._parse_bbox): 2,000 names of a
        font with a box of 1,000,000 elements ran 77 s past a deadline of 60 before the first operator (review of
        sift 0.1.48). A box that is not four elements is left out of what pypdf is handed: it would drop it, so
        nothing is lost, no note is due and the page is not cut; a box of four is handed as it is."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf._font
        from pypdf.generic import ArrayObject, DictionaryObject
        big = b"[" + b"0 " * 1000 + b"]"
        descriptor = b"<< /Type /FontDescriptor /FontName /X /Flags 32 /FontBBox %s >>"
        kinds = {
            "descriptor": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 8 0 R >>",
                           (descriptor % big,)),
            "Type3": (b"<< /Type /Font /Subtype /Type3 /BaseFont /X /FontBBox " + big + b" /FontMatrix [0.001 0 0 "
                      b"0.001 0 0] /CharProcs << >> /FirstChar 0 /LastChar 0 /Widths [1] >>", ()),
            "descendant": (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< "
                           b"/Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) "
                           b"/Ordering (Identity) /Supplement 0 >> /FontDescriptor 8 0 R >>] >>", (descriptor % big,)),
        }

        def boxes(font):
            """The /FontBBox arrays of ``font`` pypdf reads, by where they stand."""
            def at(owner, key):
                return owner[key] if isinstance(owner, DictionaryObject) and key in owner else None
            found = {"Type3": at(font, "/FontBBox"), "descriptor": at(at(font, "/FontDescriptor"), "/FontBBox")}
            for descendant in at(font, "/DescendantFonts") or ():
                found["descendant"] = at(at(descendant.get_object(), "/FontDescriptor"), "/FontBBox")
            return {k: len(v) for k, v in found.items() if isinstance(v, ArrayObject)}

        real = pypdf._font.Font.from_font_resource.__func__
        for kind, (font, extra) in kinds.items():
            for box, handed in ((big, {}), (b"[0 0 1 1]", {kind: 4}), (b"[0 0 1]", {})):
                with self.subTest(kind=kind, box=box[:12]):
                    seen = []

                    def record(cls, font_dict):
                        if font_dict.get("/BaseFont") == "/X":
                            seen.append(boxes(font_dict))
                        return real(cls, font_dict)
                    data = self._fonts_pdf(font.replace(big, box), 3, tuple(e.replace(big, box) for e in extra))
                    with mock.patch.object(pypdf._font.Font, "from_font_resource", classmethod(record)):
                        ex = self._extract_pdf(data)
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(seen, [handed] * 3)
                    self.assertIn("hello", ex.text)
                    self.assertEqual((ex.notes, ex.truncated), ([], False))

    def test_a_font_whose_map_is_left_out_is_charged_for_its_font_file(self):
        """Without its map, a Type1 font's font file is what pypdf reads instead: a font handed without a map past
        MAX_TO_UNICODE is charged for it, and past the budget handed without it too."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        unit = pdf.FONT_COST
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /ToUnicode 8 0 R /FontDescriptor 9 0 R >>"
        extra = (self._stream(self._cmap(2000)),
                 b"<< /Type /FontDescriptor /FontName /X /Flags 32 /FontFile 10 0 R >>", self._stream(self._PROGRAM))
        bound = unit + 2 * (unit + len(self._PROGRAM)) + 2 * unit
        with mock.patch.object(pdf, "MAX_TO_UNICODE", 1000), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
            ex, built = self._loads(self._fonts_pdf(font, 4, extra))
        self.assertEqual(built, ["FontFile"] * 2 + ["bare"] * 2)
        self.assertEqual(ex.notes, ["ignored the ToUnicode map of a font past 1000 bytes' worth of reading; its text "
                                    "may be wrong",
                                    f"page 1 loads fonts past a page's font budget of {bound}: 2 of them "
                                    f"{self._WITHOUT}; their text may be wrong"])

    def test_one_font_of_a_hundred_thousand_widths_named_170_times(self):
        """The reviewer's case, at the real bounds: 100,000 /W widths, pypdf's own limit, in one Type0 font that a
        page names 170 times -- 3.3 s and 1,034 MB. Each name costs 640 + 3 * 100,000; 13 fit the budget of
        4,000,000 after /T, 142 more are read without their widths, 640 each, and the last 15 are left out."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift.extract import pdf
        font = (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< /Type /Font "
                b"/Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) "
                b"/Supplement 0 >> /W [0 65535 500 0 34463 500] >>] >>")
        ex, built = self._loads(self._fonts_pdf(font, 170))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        whole = (pdf.MAX_FONT_LOAD_PER_PAGE - pdf.FONT_COST) // (pdf.FONT_COST + pdf.WIDTH_COST * 100_000)
        bare = (pdf.MAX_FONT_LOAD_PER_PAGE - pdf.FONT_COST - whole * (pdf.FONT_COST + pdf.WIDTH_COST * 100_000)) \
            // pdf.FONT_COST
        self.assertEqual((whole, bare), (13, 142))
        self.assertEqual((built.count("W"), built.count("bare"), len(built)), (13, 142, 155))
        self.assertEqual(built[:13], ["W"] * 13)
        self.assertIn("hello", ex.text)
        self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {pdf.MAX_FONT_LOAD_PER_PAGE}: "
                                    f"142 of them {self._WITHOUT}, 15 left out altogether; their text may be wrong"])

    def test_a_font_box_that_is_no_array_is_handed_to_pypdf_only_where_pypdf_keeps_it(self):
        """pypdf floats whatever a /FontBBox holds, element by element, before it keeps only four (Font._parse_bbox):
        a string of digits character by character. 2,000 names of a font whose descriptor's box was a string of
        1,000,000 digits took 170 s past a 60 s deadline; sift left out only arrays
        of other than four elements. Now any box of other than four is left out -- pypdf drops it -- and one of four,
        a string pypdf reads as four numbers included, is handed as it is, as is one that is no sequence at all."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from collections.abc import Sized
        from unittest import mock
        import pypdf._font
        from pypdf.generic import DictionaryObject
        box = b"(" + b"1" * 1000 + b")"
        descriptor = b"<< /Type /FontDescriptor /FontName /X /Flags 32 /FontBBox %s >>"
        kinds = {
            "descriptor": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 8 0 R >>",
                           (descriptor % box,)),
            "Type3": (b"<< /Type /Font /Subtype /Type3 /BaseFont /X /FontBBox " + box + b" /FontMatrix [0.001 0 0 "
                      b"0.001 0 0] /CharProcs << >> /FirstChar 0 /LastChar 0 /Widths [1] >>", ()),
            "descendant": (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< "
                           b"/Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) "
                           b"/Ordering (Identity) /Supplement 0 >> /FontDescriptor 8 0 R >>] >>", (descriptor % box,)),
        }

        def boxes(font):
            """The /FontBBox entries of ``font`` pypdf reads, by where they stand: their length, or their type."""
            def at(owner, key):
                return owner[key] if isinstance(owner, DictionaryObject) and key in owner else None
            found = {"Type3": at(font, "/FontBBox"), "descriptor": at(at(font, "/FontDescriptor"), "/FontBBox")}
            for descendant in at(font, "/DescendantFonts") or ():
                found["descendant"] = at(at(descendant.get_object(), "/FontDescriptor"), "/FontBBox")
            return {k: len(v) if isinstance(v, Sized) else type(v).__name__ for k, v in found.items() if v is not None}

        real = pypdf._font.Font.from_font_resource.__func__
        for kind, (font, extra) in kinds.items():
            for value, handed in ((box, {}), (b"<7F7F7F>", {}), (b"(1234)", {kind: 4}), (b"<7F7F7F7F>", {kind: 4}),
                                  (b"7", {kind: "NumberObject"})):
                with self.subTest(kind=kind, box=value[:12]):
                    seen = []

                    def record(cls, font_dict):
                        if font_dict.get("/BaseFont") == "/X":
                            seen.append(boxes(font_dict))
                        return real(cls, font_dict)
                    data = self._fonts_pdf(font.replace(box, value), 3, tuple(e.replace(box, value) for e in extra))
                    with mock.patch.object(pypdf._font.Font, "from_font_resource", classmethod(record)):
                        ex = self._extract_pdf(data)
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(seen, [handed] * 3)
                    self.assertIn("hello", ex.text)
                    self.assertEqual((ex.notes, ex.truncated), ([], False))

    def test_glyph_procedures_in_an_array_and_widths_in_a_dictionary_are_charged(self):
        """pypdf walks a Type3 font's /CharProcs whatever it is, for whether every entry is an Adobe glyph name
        (Font.from_font_resource) -- an array as well as a dictionary -- and a descendant's /W whatever it is, a
        dictionary key by key, logging each (Font._collect_cid_character_widths). sift charged a dictionary of glyph
        procedures and an array of widths only: 2,000 names of a font with an array of 100,000 glyph names took 6.9 s,
        of one with a /W dictionary of 100,000 keys 134 s, past a 60 s deadline. Each is now charged as
        the other shape is -- a key of /W, which pypdf logs, as the record it is -- and past the budget
        handed without it."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import contextlib
        from unittest import mock
        from sift.extract import pdf
        unit = pdf.FONT_COST
        kinds = {
            "CharProcs array": (b"<< /Type /Font /Subtype /Type3 /BaseFont /X /FontBBox [0 0 1 1] /FontMatrix [0.001 0 "
                                b"0 0.001 0 0] /CharProcs [" + b"/a " * 1000 + b"] /FirstChar 0 /LastChar 0 /Widths [1] "
                                b">>", 1000, "CharProcs"),
            "W dictionary": (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< "
                             b"/Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) "
                             b"/Ordering (Identity) /Supplement 0 >> /W << "
                             + b" ".join(b"/k%d 1" % i for i in range(1000)) + b" >> >>] >>",
                             sum(pdf.WIDTH_COST + pdf.LOG_RECORD_COST + pdf._NOT_A_WIDTH + len("/k%d" % i)
                                 for i in range(1000)), "W"),
        }
        for kind, (font, cost, parts) in kinds.items():
            budget = unit + 2 * (unit + cost) + 2 * unit                # /T, two names whole, two bare
            for bound, loads in ((budget, [parts] * 2 + ["bare"] * 2), (budget + cost, [parts] * 3 + ["bare"])):
                logs = (self.assertLogs("pypdf", "WARNING") if kind == "W dictionary"   # a warning a key
                        else contextlib.nullcontext())
                with self.subTest(kind=kind, bound=bound), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
                    with logs:
                        ex, built = self._loads(self._fonts_pdf(font, 4))
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(built, loads)
                    self.assertIn("hello", ex.text)
                    stripped = loads.count("bare")
                    self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {bound}: {stripped} "
                                                f"of them {self._WITHOUT}; their text may be wrong"])

    def test_a_font_resource_that_is_no_dictionary_is_charged_as_pypdf_reads_it(self):
        """pypdf walks /Font whatever it is and builds a Font of ``fonts[entry]`` for each entry that indexes it
        (PageObject._extract_text): an array [1 4 0 R 1 1 ...] builds the font it holds at index 1 once for every 1 in
        it, uncharged -- 201 builds of a font of 99,999 widths from a 1,132-byte file took 2.8 s -- and a string is
        walked character by character, 4.6 s for 10,000,000 of them. pypdf is now handed a dictionary of
        what it builds, keyed as it keys it, charged as any /Font is; a string builds nothing, and a byte string whose
        bytes index past its end still sends the document to the fallback, as pypdf's IndexError did."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf._page
        from sift.extract import pdf
        unit = pdf.FONT_COST
        font, extra, cost, _ = self._font_kinds()["W"]
        self.assertEqual(extra, ())                                                     # the test's own premise

        def document(fonts):
            return self._assemble([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
                                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", font,
                                   b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font %s >> "
                                   b"/Contents 6 0 R >>" % fonts,
                                   self._stream(b"BT 1 12 Tf 20 700 Td (hello) Tj ET\n", flate=False)])
        handed, real = [], pypdf._page._get_page_resources

        def resources(obj):
            found = real(obj)
            handed.append(type(found.get("/Font")).__name__)
            return found
        indexed = b"[1 4 0 R" + b" 1" * 50 + b" 0 3 0 R]"          # font 4 at index 1; index 0 holds a number
        for fonts, bound, loads, text, notes in (
                (indexed, pdf.MAX_FONT_LOAD_PER_PAGE, ["W"], "hello", []),
                (indexed, 2 * unit, ["bare"], "hello", [f"page 1 loads fonts past a page's font budget of {2 * unit}: "
                                                        f"1 of them {self._WITHOUT}; their text may be wrong"]),
                (b"(" + b"x" * 1000 + b")", pdf.MAX_FONT_LOAD_PER_PAGE, [], "\ufffd" * 5, [])):  # no font, as in 0.1.50
            with self.subTest(fonts=fonts[:12], bound=bound), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound), \
                    mock.patch.object(pypdf._page, "_get_page_resources", resources):
                handed.clear()
                ex, built = self._loads(document(fonts))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(handed, ["DictionaryObject"])
                self.assertEqual(built, loads)
                self.assertEqual(ex.text, "[page 1]\n" + text)
                self.assertEqual(ex.notes, notes)
        ex = self._extract_pdf(document(b"<FFFE00>"))                   # 255 indexes past three bytes
        self.assertEqual(ex.metadata.get("engine"), "stdlib-fallback")
        self.assertIn("pypdf failed (index out of range); used fallback", ex.notes)

    def test_the_strings_pypdf_copies_or_logs_of_a_font_are_charged(self):
        """pypdf copies a font's /BaseFont and /Subtype into every Font it builds of it, and logs its /Encoding with
        each where it knows no such encoding (Font.from_font_resource, _cmap._parse_encoding): as names, it reads them
        to 4,096 characters at most (NameObject._LENGTH_LIMIT), but a string has no bound. 2,000 names of a font whose
        /BaseFont was a string of 1,000,000 characters held 1,968 MB, one with such a /Subtype 1,967 MB, one with such
        an /Encoding wrote 2,000 MB of log. Each character of such a string is now charged -- an
        /Encoding, which pypdf logs, as the record it is --, the font
        stripped of its parts or not; a name that does not fit even so is left out, and the names after it are
        still read."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import contextlib
        from unittest import mock
        import pypdf._font
        from sift.extract import pdf
        unit, long = pdf.FONT_COST, b"(" + b"X" * 2000 + b")"
        kinds = {
            "/BaseFont": b"<< /Type /Font /Subtype /Type1 /BaseFont %s >>" % long,
            "/Subtype": b"<< /Type /Font /Subtype %s /BaseFont /X /DescendantFonts [] >>" % long,
            "/Encoding": b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding %s >>" % long,
        }
        real = pypdf._font.Font.from_font_resource.__func__
        for key, font in kinds.items():
            cost = 2000 + (pdf.LOG_RECORD_COST + pdf._ADVANCED if key == "/Encoding" else 0)    # and the record
            bound = unit + 2 * (unit + cost) + unit                     # /T, two long names, and /Z
            built = []

            def record(cls, font_dict):
                built.append("long" if len(str(font_dict.get(key))) == 2000 else str(font_dict.get("/BaseFont")))
                return real(cls, font_dict)
            data = self._assemble([
                b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", font,
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /T 3 0 R /F0 4 0 R /F1 4 "
                b"0 R /F2 4 0 R /F3 4 0 R /Z 3 0 R >> >> /Contents 6 0 R >>",
                self._stream(b"BT /Z 12 Tf 20 700 Td (hello) Tj ET\n", flate=False)])
            logs = self.assertLogs("pypdf", "ERROR") if key == "/Encoding" else contextlib.nullcontext()
            with self.subTest(key=key), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
                with logs, mock.patch.object(pypdf._font.Font, "from_font_resource", classmethod(record)):
                    ex = self._extract_pdf(data)
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(built, ["/Helvetica", "long", "long", "/Helvetica"])
                self.assertIn("hello", ex.text)
                self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {bound}: 2 left out "
                                            f"altogether; their text may be wrong"])

    def test_a_map_or_font_file_past_its_bound_is_never_decoded_whole(self):
        """sift measured a Flate map by inflating it to one byte past the bound, but any other -- two Flate filters,
        ASCIIHex, ASCII85 -- by having pypdf decode it whole, and dropped pypdf's copy after: the reader looked the
        same afterwards either way. 2,000 fonts with maps of 70 MB inflated twice took 117 s past a 60 s budget. Now
        pypdf decodes none of them: a chain of Flate filters is inflated stage by stage to one byte past the bound,
        ASCIIHex and ASCII85 are measured from their encoded bytes. A font file past the page's budget likewise."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf.filters
        from sift.extract import pdf
        decoded, real = [], pypdf.filters.decode_stream_data

        def decode(stream, *args, **kwargs):
            decoded.append(stream.indirect_reference.idnum if stream.indirect_reference else None)
            return real(stream, *args, **kwargs)
        for encoding in (b"/FlateDecode", b"[/FlateDecode /FlateDecode]", b"/ASCIIHexDecode", b"/ASCII85Decode"):
            with self.subTest(encoding=encoding), mock.patch.object(pdf, "MAX_TO_UNICODE", 1000), \
                    mock.patch.object(pypdf.filters, "decode_stream_data", decode):
                decoded.clear()
                ex = self._extract_pdf(self._cmap_pdf(self._cmap(2000), encoding))
                self.assertEqual(ex.text, "[page 1]\nABBA")
                self.assertNotIn(4, decoded)                                              # the map
                self.assertEqual(ex.notes, ["ignored the ToUnicode map of a font past 1000 bytes' worth of reading; "
                                            "its text may be wrong"])
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor 8 0 R >>"
        extra = (b"<< /Type /FontDescriptor /FontName /X /Flags 32 /FontFile 9 0 R >>",
                 self._stream(self._PROGRAM + b"%" * 20_000 + b"\n"))
        decoded.clear()
        with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", 10_000), \
                mock.patch.object(pypdf.filters, "decode_stream_data", decode):
            ex, built = self._loads(self._fonts_pdf(font, 1, extra))
        self.assertEqual(built, ["bare"])
        self.assertNotIn(9, decoded)                                                      # the font file
        self.assertIn("hello", ex.text)

    def test_what_sift_measures_is_what_pypdf_decodes(self):
        """sift's own count of a stream's decoded length must be pypdf's: ASCIIHex with white space and an odd digit,
        ASCII85 with "z" groups, a short last group and no "<~", Flate chains of one to three stages -- at the bound,
        one byte over it, and far past it, where the count stops one byte past the bound."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from pypdf.filters import ASCII85Decode, ASCIIHexDecode, FlateDecode
        from sift.extract import pdf
        for data in (b"", b"a", b"4142 43\n44\t4>", b"41 42>junk 4344", b"41\r\n42\x0b43\x0c44"):
            with self.subTest(hex=data):
                self.assertEqual(pdf._hex_size(data), len(ASCIIHexDecode.decode(data)))
        for raw in (b"", b"abc", b"\0" * 8 + b"abcde", b"x" * 1001, bytes(range(256)) * 3):
            for encoded in (base64.a85encode(raw, adobe=True), base64.a85encode(raw, adobe=True, wrapcol=10),
                            base64.a85encode(raw, adobe=True)[2:]):
                with self.subTest(a85=encoded[:20]):
                    self.assertEqual(pdf._a85_size(encoded), len(ASCII85Decode.decode(encoded)))
        raw = bytes(range(256)) * 400                                               # 102,400 bytes
        for stages in (1, 2, 3):
            data = raw
            for _ in range(stages):
                data = zlib.compress(data)
            for cap, size in ((len(raw), len(raw)), (len(raw) - 1, len(raw)), (1000, 1001)):
                with self.subTest(stages=stages, cap=cap):
                    self.assertEqual(pdf._inflated_size(data, stages, cap), size)
        for data in (zlib.compress(raw)[:-4], zlib.compress(b"a" * 65537)[:-4]):    # no checksum; the last byte past
            with self.subTest(truncated=len(data)):                                 # a step waits in the inflater
                self.assertEqual(pdf._inflated_size(data, 1, 10**9), len(FlateDecode.decode(data)))
        with self.assertRaises(zlib.error):
            pdf._inflated_size(b"not zlib", 1, 10**9)

    def test_the_time_budget_is_checked_while_sift_reads_a_pages_fonts(self):
        """sift walks every font a page names before pypdf reads the page, measuring maps as it goes, and never looked
        at the clock doing it: 2,000 fonts with double-Flate maps of 70 MB took 117 s past a 60 s budget. Now the
        budget is checked for every name, so running out on the fourth name stops before pypdf builds any font."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift import deadline
        font, extra, *_ = self._font_kinds()["map"]
        real, seen, built = deadline.check, [], []

        def expire_on_the_fifth(where=""):
            seen.append(where)
            if len(seen) >= 5:                          # the page, /T, /F0, /F1, and /F2
                raise deadline.DeadlineExceeded(60, where)
            return real(where)
        with mock.patch.object(deadline, "check", expire_on_the_fifth), \
                self.assertRaises(deadline.DeadlineExceeded) as raised:
            with deadline.limit(60):
                self._loads(self._fonts_pdf(font, 50, extra), built)
        self.assertEqual(raised.exception.where, "a page of a PDF")
        self.assertEqual(len(seen), 5)
        self.assertEqual(built, [])                                      # before any operator, pypdf's or a font

    def test_forms_past_the_page_budget_are_neither_decoded_nor_said_to_be_read(self):
        """Once the forms a page draws are past MAX_FORM_CONTENT the rest are skipped -- each after being decoded,
        to learn its length: one page drawing 1,000 forms of 790,000 bytes took 797 MB where 44 MB do. And the form
        that went past was cut first, and the cut said: "read only the first 997 of 86,852 bytes of a form XObject
        page 1 draws", of a form not a byte of which was read. Now nothing is decoded after the first form past the
        budget, a cut is said only of a form pypdf reads, and a small form after a skipped one is skipped too, as the
        note says."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf.filters
        from sift.extract import pdf
        zero, one, two = self._form_lines("zero", 5), self._form_lines("one", 200), self._form_lines("two", 5)
        data = self._form_pdf(b"/Fm0 Do\n/Fm1 Do\n/Fm2 Do\n/Fm3 Do\n", (b"Fm0", b"Fm1", b"Fm2", b"Fm3"),
                              {b"Fm0": (zero, ()), b"Fm1": (one, ()), b"Fm2": (two, ()), b"Fm3": (two, ())})
        decoded, real = [], pypdf.filters.decode_stream_data

        def decode(stream, *args, **kwargs):
            decoded.append(stream.indirect_reference.idnum)
            return real(stream, *args, **kwargs)
        with mock.patch.object(pdf, "MAX_FORM_CONTENT", len(zero) + len(two)), \
                mock.patch.object(pdf, "MAX_INFLATED", len(one) // 2), \
                mock.patch.object(pypdf.filters, "decode_stream_data", decode):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertGreater(pdf._content_cut(one, len(one) // 2), len(two))             # the premise: Fm1 goes past
        self.assertEqual(sorted(set(decoded)), [6, 7])                                   # Fm0 and Fm1; never Fm2, Fm3
        self.assertIn("zero line 4", ex.text)
        self.assertNotIn("one line", ex.text)
        self.assertNotIn("two line", ex.text)                                            # it would fit: skipped too
        self.assertEqual(ex.notes, [f"skipped the form XObjects page 1 draws after the first {len(zero)} bytes of "
                                    f"their content; their text is not in the body"])
        self.assertTrue(ex.truncated)

    @staticmethod
    def _pages_pdf(pages, forms: dict) -> bytes:
        """A real PDF pypdf reads: a page for each content in ``pages``, every page naming every form XObject in
        ``forms`` (name -> content, written as is) and /F1, Helvetica, as does every form."""
        names, first = list(forms), 4 + 2 * len(pages)
        xobjects = b" ".join(b"/%s %d 0 R" % (name, first + i) for i, name in enumerate(names))
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   b"<< /Type /Pages /Kids [%s] /Count %d >>"
                   % (b" ".join(b"%d 0 R" % (4 + 2 * i) for i in range(len(pages))), len(pages)),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        for i, content in enumerate(pages):
            objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> "
                           b"/XObject << %s >> >> /Contents %d 0 R >>" % (xobjects, 5 + 2 * i))
            objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        for name in names:
            objects.append(b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> "
                           b">> /Length %d >>\nstream\n" % len(forms[name]) + forms[name] + b"\nendstream")
        return PdfTextTest._assemble(objects)

    def test_a_form_past_what_is_kept_parsed_is_parsed_once_on_its_page(self):
        """What does not fit the forms kept parsed across pages is still kept parsed for its own page: Fm1, drawn
        twice on page 2 after page 1's Fm0 and Fm2 filled what is kept, is parsed once there, not once a draw."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from pypdf.generic import ContentStream
        from sift.extract import pdf
        zero, one, two = (self._form_lines(tag, 5) for tag in ("zero", "one-", "two-"))
        self.assertEqual(len({len(zero), len(one), len(two)}), 1)                      # the test's own premise
        data = self._pages_pdf([b"/Fm0 Do\n/Fm2 Do\n", b"/Fm1 Do\n/Fm1 Do\n"], {b"Fm0": zero, b"Fm1": one,
                                                                                  b"Fm2": two})
        parses, real = [], ContentStream._parse_content_stream

        def parse(stream, *args, **kwargs):
            parses.append(1)
            return real(stream, *args, **kwargs)
        with mock.patch.object(pdf, "MAX_FORM_CONTENT", 2 * len(zero)), \
                mock.patch.object(ContentStream, "_parse_content_stream", parse):
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual([ex.text.count(f"{tag} line 4") for tag in ("zero", "one-", "two-")], [1, 2, 1])
        self.assertEqual(len(parses), 2 + 3)                             # each page's content, each form once
        self.assertEqual(ex.notes, [])

    def test_a_cut_is_said_of_the_page_that_first_reads_the_form(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        form = self._form_lines("form", 40)
        text = b"BT /F1 8 Tf 20 700 Td (the first page) Tj ET\n"
        with mock.patch.object(pdf, "MAX_INFLATED", len(form) // 2):
            ex = self._extract_pdf(self._pages_pdf([text, b"/Fm0 Do\n", b"/Fm0 Do\n"], {b"Fm0": form}))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text.count("form line 0"), 2)
        self.assertEqual(ex.notes, [f"read only the first {pdf._content_cut(form, len(form) // 2)} of {len(form)} "
                                    f"bytes of a form XObject page 2 draws; its text after that is not in the body"])

    def test_nothing_pypdf_decoded_for_a_page_is_kept_past_it(self):
        """pypdf keeps what it decodes on the reader's own objects, and the reader lives as long as the document is
        read: 1,000 pages of 790 KB of content each held 800 MB; 500 pages with a form of their own 421 MB; 300 pages
        with a font of their own, each with a 3.1 MB map, 936 MB; 30 pages with a 60 MB font file each 1,816 MB. And
        sift's cut copy of a page's content stayed on the page: 500 of them, 433 MB. Now what was decoded for a page
        -- its content, the forms it drew, the maps and font files of its fonts -- is dropped when the page is read,
        and the page has its own /Contents and /Resources back. What is kept of a form parsed across pages is its
        parsed operations, bounded by MAX_FORM_CONTENT."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf.filters
        from sift.extract import pdf
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [8 0 R 14 0 R 20 0 R] /Count 3 >>",
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 4 0 R >>",
                   self._stream(self._CMAP),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
                   self._stream(self._cmap(3000)),                                        # past MAX_TO_UNICODE below
                   b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                   b"/Filter /FlateDecode /Length %d >>\nstream\n" % len(zlib.compress(self._form_lines("letter", 2)))
                   + zlib.compress(self._form_lines("letter", 2)) + b"\nendstream"]
        for number in (1, 2, 3):
            first = 8 + 6 * (number - 1)             # page, content, form, font, descriptor, font file
            content = (b"BT /F1 8 Tf 20 700 Td (ABBA) Tj ET\n/Fm0 Do\n/Fm9 Do\nBT /F2 8 Tf 20 600 Td (page %d) Tj ET\n"
                       % number + (self._form_lines("long", 100) if number == 2 else b""))
            form = self._form_lines("wide" if number == 3 else "form%d" % number, 100 if number == 3 else 2)
            objects += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 "
                        b"%d 0 R /F3 5 0 R >> /XObject << /Fm0 %d 0 R /Fm9 7 0 R >> >> /Contents %d 0 R >>"
                        % (first + 3, first + 2, first + 1),
                        self._stream(content),
                        b"<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                        b"/Filter /FlateDecode /Length %d >>\nstream\n" % len(zlib.compress(form)) + zlib.compress(form)
                        + b"\nendstream",
                        b"<< /Type /Font /Subtype /Type1 /BaseFont /Y /FontDescriptor %d 0 R >>" % (first + 4),
                        b"<< /Type /FontDescriptor /FontName /Y /Flags 32 /FontFile %d 0 R >>" % (first + 5),
                        self._stream(self._PROGRAM)]
        decoded, real = set(), pypdf.filters.decode_stream_data

        def decode(stream, *args, **kwargs):
            decoded.add(stream.indirect_reference.idnum)
            return real(stream, *args, **kwargs)
        with mock.patch.object(pdf, "MAX_INFLATED", 2000), mock.patch.object(pdf, "MAX_TO_UNICODE", 2000), \
                mock.patch.object(pypdf.filters, "decode_stream_data", decode):
            ex, reader = self._extract_keeping_the_reader(self._assemble(objects))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        for line in ("page 1", "page 2", "page 3", "form1 line 1", "form2 line 1", "wide line 0", "long line 0"):
            self.assertIn(line, ex.text)
        self.assertEqual(ex.text.count("letter line 1"), 3)
        self.assertEqual(ex.text.count("ZYYZ"), 3)                                     # page 1's map, read on each
        self.assertTrue(any("page 2's content" in note for note in ex.notes), ex.notes)  # the premise: cut, and a
        self.assertTrue(any("form XObject page 3" in note for note in ex.notes), ex.notes)  # form cut; each read,
        self.assertTrue({4, 7, 9, 10, 13, 15, 16, 19, 21, 22, 25} <= decoded, decoded)    # and decoded by pypdf
        held = sorted(number for (_, number), obj in reader.resolved_objects.items()
                      if getattr(obj, "decoded_self", None) is not None)
        self.assertEqual(held, [])
        for page, first in zip(reader.pages, (8, 14, 20)):
            self.assertEqual(page.get("/Contents").idnum, first + 1)                  # its own, not the cut copy
            self.assertIn("/ToUnicode", page["/Resources"]["/Font"]["/F3"])            # its own, not sift's copy

    def test_a_forms_fonts_inherited_through_its_parent_are_charged_as_its_own(self):
        """pypdf takes a form's /Resources through /Parent where it has none of its own (PageObject._extract_text reads
        them with _get_page_resources, DictionaryObject.get_inherited); sift charged a form's own only, so pypdf built
        every font a parent names, unbudgeted: 2.2 KB with 100 names of a /W font took 1,021 MB, drawn 20 times 7,641
        MB. They are now found as pypdf finds them and charged as a form's own are, through up to
        MAX_INHERITED parents; a form whose parents go on past that, or round a cycle, is not read, and said."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        font, extra, cost, _ = self._font_kinds()["W"]
        self.assertEqual(extra, ())                                                     # the test's own premise
        unit = pdf.FONT_COST
        bound = 2 * (2 * unit + cost) + 2 * unit                    # two draws whole, /T and /F0 bare on the third
        last = b"<< /Resources << /Font << /T 3 0 R /F0 4 0 R >> >> >>"

        def parents(depth, first=b"/Parent 8 0 R"):
            """Objects 8 on: ``depth`` parents, the last with the form's fonts."""
            return self._fonts_pdf(font, 1, tuple(b"<< /Parent %d 0 R >>" % (9 + i) for i in range(depth - 1))
                                   + (last,), draws=3, form=first)
        with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
            own, own_built = self._loads(self._fonts_pdf(font, 1, draws=3))
            self.assertEqual(own_built, ["W", "W", "bare"])                            # as its own are charged
            for depth in (1, pdf.MAX_INHERITED):
                with self.subTest(depth=depth):
                    ex, built = self._loads(parents(depth))
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(built, own_built)
                    self.assertEqual(ex.text.count("hello"), 3)
                    self.assertEqual(ex.notes, own.notes)
            for name, data in (("past MAX_INHERITED", parents(pdf.MAX_INHERITED + 1)),
                               ("a cycle", self._fonts_pdf(font, 1, (b"<< /Parent 6 0 R >>",), draws=3,
                                                           form=b"/Parent 8 0 R"))):
                with self.subTest(parents=name):
                    with self.assertLogs("pypdf", "WARNING") as logged:
                        ex, built = self._loads(data)
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual(built, [])
                    self.assertNotIn("hello", ex.text)
                    self.assertEqual(sum("XFormObject" in line for line in logged.output), 1, logged.output)
                    self.assertEqual(ex.notes, ["could not read a form XObject page 1 draws; its text is not in the "
                                                "body"])

    @staticmethod
    def _font_pages_pdf(fonts: bytes, pages: int, font: bytes = b"<< /Type /Font /Subtype /Type1 /BaseFont /X >>",
                        extra=()):
        """A real PDF pypdf reads: ``pages`` pages, each showing "hello" in /T, Helvetica, and naming ``fonts`` in
        its /Font, where ``font`` is object 4, the pages 6 on, and ``extra`` the objects after them."""
        return PdfTextTest._assemble(
            [b"<< /Type /Catalog /Pages 2 0 R >>",
             b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % (6 + i) for i in range(pages)), pages),
             b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>", font,
             PdfTextTest._stream(b"BT /T 12 Tf 20 700 Td (hello) Tj ET\n")]
            + [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << %s >> >> /Contents 5 0 "
               b"R >>" % fonts] * pages + list(extra))

    def test_what_pypdf_built_for_a_page_is_collected_past_it(self):
        """What pypdf builds to read a page is a reference cycle: its TextExtraction's operation_handlers hold methods
        bound to it (pypdf/_text_extraction/_text_extractor.py:97-111 in 6.19.0), and it holds the page's Fonts, their
        widths and maps. Only the collector frees a cycle, and it runs as the objects it tracks are allocated -- a
        dict of numbers and strings is not tracked, so what makes a Font large never counts. 50 pages of a 12.8 KB
        file, each within its font budget, held 3,171 MB, 141 MB collected after every page: the
        per-page budget bounds a page, not the process. sift now collects once what the fonts built since it last
        did hold reaches MAX_FONT_HELD_UNCOLLECTED; here one page's worth, so that eight pages leave less behind than
        one page's fonts do uncollected. Python's own collector is off throughout, so what sift collects is young."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import gc
        import tracemalloc
        from unittest import mock
        from sift.extract import pdf
        widths, names = 20_000, 5
        font = (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< /Type /Font "
                b"/Subtype /CIDFontType2 /BaseFont /X /W [0 %d 500] >>] >>" % (widths - 1))
        fonts = b"/T 3 0 R " + b" ".join(b"/F%d 4 0 R" % i for i in range(names))
        page = pdf.FONT_HELD + names * (pdf.FONT_HELD + pdf.WIDTH_HELD * widths)      # what one page's fonts hold
        held = {}
        enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            for pages, bound in ((1, 10 ** 12), (8, page)):                  # never collected; after every page
                data = self._font_pages_pdf(fonts, pages, font)
                with mock.patch.object(pdf, "MAX_FONT_HELD_UNCOLLECTED", bound), \
                        mock.patch.object(pdf, "_young", 0), mock.patch.object(pdf, "_old", 0):
                    tracemalloc.start()
                    try:
                        ex = self._extract_pdf(data)
                        text, engine = ex.text, ex.metadata.get("engine")
                        del ex
                        held[pages] = tracemalloc.get_traced_memory()[0]
                    finally:
                        tracemalloc.stop()
                self.assertEqual(engine, "pypdf")
                self.assertEqual(text.count("hello"), pages)
        finally:
            if enabled:
                gc.enable()
        self.assertGreater(held[1], 5_000_000, held)          # the premise: one page's fonts, 20,000 widths each
        self.assertLess(held[8], held[1] // 4, held)

    def test_pypdfs_fonts_are_collected_once_what_they_hold_reaches_the_bound(self):
        """sift collects when what the fonts built for the pages read since it last collected hold reaches
        MAX_FONT_HELD_UNCOLLECTED, counted across documents, since the process is what holds them: at the bound,
        and not one byte before it. Pages that load no font never make it collect."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import gc
        from unittest import mock
        from sift.extract import pdf
        font, _, _, _ = self._font_kinds()["W"]
        page = pdf.FONT_HELD + 2 * (pdf.FONT_HELD + pdf.WIDTH_HELD * 1000)              # /T, and /F0 and /F1
        six = self._font_pages_pdf(b"/T 3 0 R /F0 4 0 R /F1 4 0 R", 6, font)
        one = self._font_pages_pdf(b"/T 3 0 R /F0 4 0 R /F1 4 0 R", 1, font)
        for documents, bound, collections in (((six,), page, 6), ((six,), 2 * page, 3), ((six,), 2 * page + 1, 2),
                                              ((one,), 2 * page, 0), ((one, one), 2 * page, 1),
                                              ((self._font_pages_pdf(b"", 6),), 1, 0)):
            with self.subTest(documents=len(documents), bound=bound), \
                    mock.patch.object(pdf, "MAX_FONT_HELD_UNCOLLECTED", bound), \
                    mock.patch.object(pdf, "_young", 0), mock.patch.object(pdf, "_old", 0), \
                    mock.patch.object(pdf.gc, "collect", wraps=gc.collect) as collect:
                for data in documents:
                    ex = self._extract_pdf(data)
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(collect.call_count, collections)

    def test_the_bounds_are_derived_from_max_inflated(self):
        """What reading a map may cost, MAX_TO_UNICODE, holds the largest map pypdf builds -- 100,000 entries, one
        bfrange line -- and a full map of two-byte codes, 65,536 bfchar lines, each once: counted in entries as pypdf
        builds them, not in bytes, three of the latter no longer fit, as they never fitted what pypdf
        spends. A page's forms get what the page's own content gets. The fonts one page loads may cost what one map
        at the bound does and a fourth of it, what counting a well-formed map costs sift at most: 23
        times the most any page of 111 real documents loaded (170,212, soa.pdf's first page), 260 times their most
        font names on a page (24, pm.pdf), 267 times their costliest map (11,967), and each of pypdf's largest maps
        with what counting it costs sift, once; none of their maps is rewritten as ranges (16 times their costliest). pypdf may log 126 times what any of them logged building fonts (6,341
        characters, speexdsp-manual.pdf), and 446 times what any of them logged at all (7,173 characters with a
        prefix a record) -- and what it may log building fonts, each record 40 characters at least, as well; what
        its Fonts hold is collected past seven times what any of their pages held (4,750,240 bytes, soa.pdf); and a
        form's resources are inherited through eight times as many parents as their deepest page tree has (4)."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift.extract import pdf
        self.assertEqual(pdf.MAX_FORM_CONTENT, pdf.MAX_INFLATED)
        self.assertEqual(pdf.MAX_TO_UNICODE, 4 * pdf.MAX_INFLATED)
        self.assertEqual(pdf.MAX_FONT_LOAD_PER_PAGE, pdf.MAX_TO_UNICODE + pdf.MAX_TO_UNICODE // 4)
        self.assertGreaterEqual(pdf.MAX_FONT_LOAD_PER_PAGE, 23 * 170_212)
        self.assertGreaterEqual(pdf.MAX_FONT_LOAD_PER_PAGE // pdf.FONT_COST, 260 * 24)
        self.assertGreaterEqual(pdf.MAX_TO_UNICODE, 267 * 11_967)
        self.assertGreaterEqual(pdf.MAP_RANGES_FROM, 16 * 11_967)
        self.assertGreaterEqual(pdf.MAX_FONT_LOG_PER_DOCUMENT, 126 * 6_341)
        self.assertGreaterEqual(pdf.MAX_OP_LOG_PER_DOCUMENT, 446 * 7_173)
        self.assertGreaterEqual(pdf.MAX_OP_LOG_PER_DOCUMENT,
                                pdf.MAX_FONT_LOG_PER_DOCUMENT * (40 + pdf.LOG_RECORD_PREFIX) // 40)
        self.assertGreaterEqual(pdf.MAX_FONT_HELD_UNCOLLECTED, 7 * 4_750_240)
        self.assertEqual(pdf.MAX_INHERITED, 8 * 4)
        largest = pdf._map_reading(b"beginbfrange\n<00000000> <0001869F> <00000041>\nendbfrange\n", 10 ** 12)
        self.assertEqual(largest.entries, 100_000)
        full = (b"begincmap\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n65536 beginbfchar\n"
                + b"".join(b"<%04X> <%04X>\n" % (code, code) for code in range(65536)) + b"endbfchar\nendcmap\n")
        full = pdf._map_reading(full, 10 ** 12)
        self.assertEqual(full.entries, 65536)
        for reading in (largest, full):
            self.assertLess(pdf.FONT_COST + reading.units, pdf.MAX_TO_UNICODE)
            self.assertLessEqual(pdf.FONT_COST + reading.units + reading.read, pdf.MAX_FONT_LOAD_PER_PAGE)
            self.assertGreater(2 * (pdf.FONT_COST + reading.units), pdf.MAX_FONT_LOAD_PER_PAGE)

    #: /ToUnicode maps of every shape pypdf reads -- and of the shapes it logs about or gives up on -- by name.
    _MAPS = {
        "one range": b"beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\n",
        "the reviewer's two ranges": b"2 beginbfrange\n<0000> <FFFF> <0000>\n<0000> <7FFF> <0000>\nendbfrange\n",
        "ranges of one entry": b"beginbfrange\n" + b"".join(b"<%04X> <%04X> <0041>\n" % (i, i) for i in range(300))
                               + b"endbfrange\n",
        "a range backwards": b"beginbfrange\n<0010> <0001> <0041>\nendbfrange\n",
        "one-byte codes": b"beginbfrange\n<00> <FF> <0041>\nendbfrange\n",
        "four-byte codes": b"beginbfrange\n<00000000> <000003E7> <00410042>\nendbfrange\n",
        "a destination past its digits": b"beginbfrange\n<0000> <00FF> <FFF0>\nendbfrange\n",
        "a destination of six digits": b"beginbfrange\n<0000> <0010> <004100>\nendbfrange\n",
        "a list": b"beginbfrange\n<0000> <0004> [<0041> <0042> <0043> <0044> <0045>]\nendbfrange\n",
        "a list over lines": b"beginbfrange\n<0000> <0009> [<0041> <0042>\n<0043> <0044>\n<0045>]\n<0020> <0021> "
                             b"<0061>\nendbfrange\n",
        "a list with a broken token": b"beginbfrange\n<0000> <0004> [<0041> <004> <0043>]\nendbfrange\n",
        "a list past its codes": b"beginbfrange\n<00> <01> [" + b"<0041> " * 300 + b"]\nendbfrange\n",
        "bfchar": b"3 beginbfchar\n<41> <005A>\n<42> <0059>\n<43> <>\nendbfchar\n",
        "bfchar pairs on a line": b"beginbfchar\n" + b" ".join(b"<%04X> <%04X>" % (i, 0x41 + i % 26)
                                                                for i in range(500)) + b"\nendbfchar\n",
        "bfchar odd tokens": b"beginbfchar\n<41> <005A> <42>\nendbfchar\n",
        "bfchar bad destinations": b"beginbfchar\n<41> <5A> <42> <00G9> <43> <059> <44> <0059>\nendbfchar\n",
        "bfchar a three-byte destination": b"beginbfchar\n<41> <005A> <42> <00590A> <43> <0059>\nendbfchar\n",
        "bfchar a bad source": b"beginbfchar\n<0041> <005A> <004> <0059>\nendbfchar\n",
        "broken lines": b"beginbfrange\n" + b"x\n" * 50 + b"]]]}}\n<0000>\n<GG> <00> <00>\n<00> <01> <0G>\n"
                        + b"<000> <001> <0041>\nendbfrange\nbeginbfchar\n" + b"x y\n" * 20 + b"endbfchar\n",
        "lines outside a section": b"x\n" * 100 + b"%comment\n\n\n" + b"<a>" * 100,
        "past pypdf's entries": b"beginbfrange\n<00000000> <0001869F> <00000041>\n<0000> <0000> <0041>\nendbfrange\n",
        "a list past pypdf's entries": b"beginbfrange\n<00000000> <0001869F> <00000041>\n<0000> <0004> [<0041> <0042>]\n"
                                       b"endbfrange\n",
        "a width of three bytes": b"beginbfrange\n<00000> <1869F> <0041>\n<0000> <0000> <0041>\nendbfrange\n",
        "a long code": b"beginbfrange\n<" + b"0" * 17 + b"> <0001> <0041>\nendbfrange\n",
        "a negative range": b"beginbfrange\n-1 1 <0041>\nendbfrange\n",
    }

    def _read_as_pypdf_does(self, data: bytes):
        """(entries, log messages, what pypdf raised or None) of pypdf reading ``data`` as a font's /ToUnicode map
        (_cmap._parse_to_unicode in 6.19.0)."""
        import logging
        from pypdf._cmap import _parse_to_unicode
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        stream = DecodedStreamObject()
        stream.set_data(data)
        messages = []

        class Said(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())
        logger, handler = logging.getLogger("pypdf"), Said()
        logger.addHandler(handler)
        entries, raised = None, None
        try:
            try:
                entries = len(_parse_to_unicode(DictionaryObject({NameObject("/ToUnicode"): stream}))[1])
            except Exception as exc:  # noqa: BLE001 - what pypdf raises out of a Font build
                raised = type(exc).__name__
        finally:
            logger.removeHandler(handler)
        return entries, messages, raised

    def test_a_maps_reading_is_counted_as_pypdf_reads_it(self):
        """pypdf expands every bfrange line of a /ToUnicode map into an entry per code (_cmap.parse_bfrange in
        6.19.0), up to 100,000 a map, on every Font it builds -- and sift charged a map by its decoded bytes: 191 bytes
        built 98,304 entries, 60 names of it 10.7 s and 674 MB from a 1,747-byte file. sift now counts
        what pypdf does with a map without building it: the lines, the entries of each shape, the element copies a
        bfchar line of many pairs costs, the records pypdf logs of broken lines and their characters, and where
        pypdf gives up on the map. For each shape, sift's count of entries and records is pypdf's, and its count of
        characters is at least what pypdf logged."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift.extract import pdf
        for name, data in self._MAPS.items():
            with self.subTest(map=name):
                entries, messages, raised = self._read_as_pypdf_does(data)
                reading = pdf._map_reading(data, 10 ** 12)
                self.assertEqual(reading.raised, raised is not None, raised)
                if raised is None:
                    self.assertEqual(reading.entries, entries)
                self.assertEqual(reading.records, len(messages), messages[:3])
                self.assertGreaterEqual(reading.logged, sum(len(message) for message in messages))
                self.assertLessEqual(reading.logged, 3 * sum(len(message) for message in messages) + 200)
        from pypdf._cmap import prepare_cm
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        def lines(data):
            """The lines of ``data`` pypdf reads past the first check (_cmap.process_cm_line)."""
            stream = DecodedStreamObject()
            stream.set_data(data)
            split = prepare_cm(DictionaryObject({NameObject("/ToUnicode"): stream})).split(b"\n")
            return sum(1 for line in split if line.strip(b" \t") and not line.strip(b" \t").startswith(b"%"))
        two = self._MAPS["the reviewer's two ranges"]
        reading = pdf._map_reading(two, 10 ** 12)
        self.assertEqual(reading.entries, 65536 + 32768)
        self.assertEqual(reading.units, len(two) + lines(two) * pdf.MAP_LINE_COST + 98304 * pdf.MAP_RANGE_ENTRY_COST)
        many = self._MAPS["bfchar pairs on a line"]
        pairs = pdf._map_reading(many, 10 ** 12)
        self.assertEqual(pairs.entries, 500)
        self.assertEqual(pairs.units, len(many) + lines(many) * pdf.MAP_LINE_COST + 500 * pdf.MAP_CHAR_ENTRY_COST
                         + sum(1000 - 2 * k for k in range(1, 501)) // pdf.MAP_COPIES_A_UNIT)
        broken = pdf._map_reading(self._MAPS["broken lines"], 10 ** 12)
        self.assertGreaterEqual(broken.units, broken.records * pdf.LOG_RECORD_COST + broken.logged)
        capped = pdf._map_reading(self._MAPS["ranges of one entry"], 1000)            # counted no further than
        self.assertEqual(capped.units, 1001)                                            # one past the cap

    @staticmethod
    def _cjk_map(entries: int, style: str = "bfchar", digits: int = 4) -> bytes:
        """A /ToUnicode map as LibreOffice and Word write one for a CJK font: codes from 1 onto consecutive letters
        from U+4E01, written in ``digits`` digits, in sections of 100 lines -- bfchar lines of one pair each, or
        bfrange lines of up to 256 codes that do not cross a low byte."""
        head = (b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo << /Registry (Adobe) "
                b"/Ordering (UCS) /Supplement 0 >> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
                b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n")
        if style == "bfchar":
            lines = [b"<%04X> <%0*X>\n" % (code, digits, 0x4E00 + code) for code in range(1, entries + 1)]
        else:
            lines, code = [], 1
            while code <= entries:
                last = min(code | 0xFF, entries)
                lines.append(b"<%04X> <%04X> <%0*X>\n" % (code, last, digits, 0x4E00 + code))
                code = last + 1
        body = b"".join(b"%d begin%s\n" % (len(lines[at:at + 100]), style.encode()) + b"".join(lines[at:at + 100])
                        + b"end%s\n" % style.encode() for at in range(0, len(lines), 100))
        return head + body + b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"

    #: Maps whose well-formed lines come in runs, and lines next to them that only walking them one by one decides.
    _RUN_MAPS = {
        "cjk bfchar": _cjk_map.__func__(20902),
        "cjk bfrange": _cjk_map.__func__(20902, "bfrange"),
        "cjk bfchar onto eight digits": _cjk_map.__func__(3000, digits=8),
        "cjk bfrange onto eight digits": _cjk_map.__func__(3000, "bfrange", 8),
        "one-byte codes": b"beginbfchar\n" + b"".join(b"<%02X> <%04X>\n" % (code, code) for code in range(256))
                          + b"endbfchar\nbeginbfrange\n<00> <7F> <0041>\n<80> <FF> <00C0>\nendbfrange\n",
        "codes of one byte and of two": b"beginbfchar\n<41> <0041>\n<0042> <0042>\n<43> <0043>\nendbfchar\n"
                                        b"beginbfrange\n<00> <0100> <0041>\n<0101> <01> <0041>\nendbfrange\n",
        "empty and short destinations": b"beginbfchar\n<0041> <>\n<0042> <42>\n<0043> < >\nendbfchar\n",
        "blank lines and tabs": b"beginbfchar\n\t<0041>\t<0042>\t\n\n  \n<0043>  <0044>\nendbfchar\nbeginbfrange\n\n"
                                b"\t<0000>\t<0010>\t<0041>\n\nendbfrange\n",
        "a range past its four digits": b"beginbfrange\n<0000> <00FF> <0041>\n<0100> <01FF> <FFF0>\n<0200> <0200> "
                                        b"<0041>\nendbfrange\n",
        "a range backwards in a run": b"beginbfrange\n<0000> <00FF> <0041>\n<0010> <0001> <0041>\n<0200> <0201> "
                                      b"<0041>\nendbfrange\n",
        "a list after a run": b"beginbfrange\n<0000> <0001> <0041>\n<0002> <0004> [<0041> <0042>\n<0043>]\n<0005> "
                              b"<0006> <0041>\nendbfrange\n",
        "both sections at once": b"beginbfrange\nbeginbfchar\n<0041> <0042>\n<0000> <0001> <0041>\nendbfrange\n"
                                 b"<0043> <0044>\nendbfchar\n",
        "a section without its end": b"beginbfchar\n" + b"<0041> <0042>\n" * 50,
        "past 300 entries in a run": b"beginbfchar\n" + b"".join(b"<%04X> <%04X>\n" % (code, code)
                                                                  for code in range(301)) + b"endbfchar\n",
        "past 300 entries in a range run": b"beginbfrange\n" + b"<0000> <00FF> <0000>\n" * 2 + b"endbfrange\n",
        "lines of other white space": b"beginbfchar\n<0041>\x0b<0042>\n<0043> <0044>\x0c\nendbfchar\n",
    }

    def test_a_well_formed_maps_lines_are_counted_in_runs(self):
        """sift counted a map's lines one by one as pypdf reads them, at 0.82 to 0.90 of the cost of pypdf's own
        reading, and charged none of it where the font was handed whole: a page with fresh fonts took 1.85 times its
        budget, and 150 pages each with a font of its own of 20,902 bfchar entries 27.1 s against 0.1.51's 14.2 s. sift now prepares a map in steps that run in C (_prepared), and counts a run of well-formed
        lines -- bfchar lines of one pair, bfrange lines of one range within four digits -- in one match, walking
        one by one only the lines a run does not decide. What it counts is what walking every line counts, for every
        shape and at every cap; what it spends is a fourth of what it counts, or less, for such maps."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        import pypdf._cmap
        from sift.extract import pdf
        for limit in (pypdf._cmap.MAPPING_DICTIONARY_SIZE_LIMIT, 300):         # pypdf's entries at most, and fewer
            for name, data in {**self._MAPS, **self._RUN_MAPS}.items():
                with mock.patch.object(pypdf._cmap, "MAPPING_DICTIONARY_SIZE_LIMIT", limit):
                    walked = pdf._map_reading(data, 10 ** 12, runs=False)
                    for cap in (10 ** 12, walked.units // 2, len(data) + pdf.MAP_LINE_COST, len(data) - 1):
                        with self.subTest(map=name, cap=cap, limit=limit):
                            run, walk = pdf._map_reading(data, cap), pdf._map_reading(data, cap, runs=False)
                            self.assertEqual((run.units, run.entries, run.records, run.logged, run.raised),
                                             (walk.units, walk.entries, walk.records, walk.logged, walk.raised))
                            self.assertLessEqual(run.read, walk.read)
        for name in ("cjk bfchar", "cjk bfrange", "cjk bfchar onto eight digits", "one-byte codes"):
            with self.subTest(spent=name):
                reading = pdf._map_reading(self._RUN_MAPS[name], 10 ** 12)
                self.assertEqual(reading.records, 0)
                self.assertLessEqual(reading.read, reading.units // 4)
        entries, _, _ = self._read_as_pypdf_does(self._RUN_MAPS["cjk bfchar"])
        self.assertEqual(pdf._map_reading(self._RUN_MAPS["cjk bfchar"], 10 ** 12).entries, entries)

    @staticmethod
    def _drawn_map(draw) -> bytes:
        """A /ToUnicode map drawn at random from ``draw``, a random.Random: bfchar sections of runs of consecutive codes
        onto consecutive strings, of every length, between lines of other shapes -- one-byte codes, eight-digit
        strings, empty ones, broken lines -- and bfrange sections of ranges and of lists, closed or not, blank lines,
        comments, sections without their end."""
        def pair():
            return b"<%04X> <%04X>" % (draw.randrange(65536), draw.randrange(65536))
        odd = [lambda: b"<%02X> <%04X>" % (draw.randrange(256), draw.randrange(65536)),
               lambda: b"<%04X> <%08X>" % (draw.randrange(65536), draw.randrange(1 << 32)),
               lambda: b"<%04X> <>" % draw.randrange(65536), lambda: b"<%04X> <%04X> <0041>" % (1, 2),
               lambda: b"<0041>", lambda: b"", lambda: b"% a comment", lambda: pair() + b" " + pair()]
        parts = [b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n1 begincodespacerange\n<0000> "
                 b"<FFFF>\nendcodespacerange\n"]
        for _ in range(draw.randrange(1, 6)):
            if draw.random() < 0.7:
                lines = []
                for _ in range(draw.randrange(1, 12)):
                    if draw.random() < 0.6:
                        length = draw.choice([1, 2, 7, 8, 9, 30, 255, 300])
                        source, string = draw.randrange(65536 - length), draw.randrange(65536 - length)
                        step = 1 if draw.random() < 0.8 else draw.choice([0, 2])
                        lines += [b"<%04X> <%04X>" % (source + i, string + i * step) for i in range(length)]
                    else:
                        lines.append(draw.choice(odd)())
                parts.append(b"%d beginbfchar\n" % len(lines) + b"\n".join(lines) + b"\n"
                             + (b"endbfchar\n" if draw.random() < 0.9 else b""))
            else:
                lines = [b"<%04X> <%04X> <%04X>" % (first, first + draw.randrange(300), draw.randrange(65536))
                         for first in (draw.randrange(60000) for _ in range(draw.randrange(1, 5)))]
                if draw.random() < 0.4:
                    lines.append(b"<0000> <0004> [<0041> <0042>" + (b"]" if draw.random() < 0.5 else b""))
                parts.append(b"beginbfrange\n" + b"\n".join(lines) + b"\n"
                             + (b"endbfrange\n" if draw.random() < 0.9 else b""))
        return b"".join(parts) + b"endcmap\nend\nend\n"

    @staticmethod
    def _built_by_pypdf(data: bytes):
        """What pypdf builds of the /ToUnicode map ``data`` (_cmap._parse_to_unicode in 6.19.0) -- its entries in the
        order pypdf builds them, as a list, and its list of codes -- or what it raised."""
        from pypdf._cmap import _parse_to_unicode
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        stream = DecodedStreamObject()
        stream.set_data(data)
        try:
            built, codes = _parse_to_unicode(DictionaryObject({NameObject("/ToUnicode"): stream}))
        except Exception as exc:  # noqa: BLE001 - what pypdf raises out of a Font build
            return type(exc).__name__
        return list(built.items()), codes

    def test_a_map_written_as_ranges_builds_what_it_builds_as_written(self):
        """A map of 20,902 bfchar lines of consecutive codes onto consecutive strings costs pypdf 936,445 units, as
        it builds an entry a line; four of them no longer fitted a page, and 0.1.52 read 30 of 40 lines of such text
        that 0.1.51 read. The same entries written as bfrange lines cost 475,396, a line of 21 units an
        entry (_cmap.parse_bfrange in 6.19.0). sift hands pypdf the map so where it costs MAP_RANGES_FROM or more, and
        what pypdf builds of it -- every code, every string, in the same order -- is what it builds of the map as
        written: for every map the tests hold, and for 600 drawn at random, 88 of which it rewrites."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import random
        from sift.extract import pdf
        draw = random.Random(11)
        maps = [*self._MAPS.values(), *self._RUN_MAPS.values(), *(self._drawn_map(draw) for _ in range(600))]
        rewritten = 0
        for number, data in enumerate(maps):
            ranges = pdf._as_ranges(data, 10 ** 12)
            if ranges is None:
                continue
            rewritten += 1
            with self.subTest(map=number):
                self.assertEqual(self._built_by_pypdf(ranges), self._built_by_pypdf(data))
                self.assertNotIn(b"<", ranges)
                shorter, written = pdf._map_reading(ranges, 10 ** 12), pdf._map_reading(data, 10 ** 12)
                self.assertEqual((shorter.entries, shorter.records, shorter.raised), (written.entries, 0, False))
                self.assertLess(shorter.units, written.units)
        self.assertGreaterEqual(rewritten, 88)                              # the test's own premise
        cjk = self._RUN_MAPS["cjk bfchar"]
        self.assertEqual((pdf._map_reading(cjk, 10 ** 12).units, pdf._map_reading(pdf._as_ranges(cjk, 10 ** 12),
                                                                                  10 ** 12).units), (936_445, 475_396))
        for name in ("a list after a run", "both sections at once", "lines of other white space"):
            self.assertIsNone(pdf._as_ranges(self._RUN_MAPS[name], 10 ** 12), name)   # nothing to write, or in doubt

    @staticmethod
    def _type0_pdf(maps, pages: int = 1) -> bytes:
        """A real PDF pypdf reads: ``pages`` pages each showing, in a Type0 font of its own per map of ``maps`` --
        Identity-H, its /ToUnicode that map -- codes 1 to 4 and 10 on from 11 times its number: "丁丂七丄" and so on
        for _cjk_map's maps. The fonts are objects 4 on, three each, shared by every page."""
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None, b"<< >>"]
        fonts = []
        for number, cmap in enumerate(maps):
            first = len(objects) + 1
            fonts.append(b"/F%d %d 0 R" % (number, first))
            objects += [b"<< /Type /Font /Subtype /Type0 /BaseFont /Song /Encoding /Identity-H /DescendantFonts "
                        b"[%d 0 R] /ToUnicode %d 0 R >>" % (first + 1, first + 2),
                        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Song /CIDSystemInfo << /Registry (Adobe) "
                        b"/Ordering (Identity) /Supplement 0 >> /DW 1000 >>", PdfTextTest._stream(cmap)]
        content = b"".join(b"BT /F%d 12 Tf 20 %d Td <000100020003%04X> Tj ET\n" % (k, 700 - 20 * k, 11 * k + 10)
                           for k in range(len(maps)))
        objects.append(PdfTextTest._stream(content))
        kids, shown = [], len(objects)
        for _ in range(pages):
            objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << %s >> >> "
                           b"/Contents %d 0 R >>" % (b" ".join(fonts), shown))
            kids.append(len(objects))
        objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), pages)
        return PdfTextTest._assemble(objects)

    def test_four_fonts_of_a_large_map_each_fit_a_page(self):
        """The reviewer's table: four fonts to a page, each with a map of 20,902 bfchar entries -- 0.1.51 read 40 of
        40 lines, 0.1.52 30. Handed the map as bfrange lines, pypdf reads all four whole, and the text is
        the same as that of the map as written."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        maps = [self._cjk_map(20902) + b"%% font %d\n" % k for k in range(4)]      # bytes of their own each
        expected = "\n".join("丁丂七" + chr(0x4E00 + 11 * k + 10) for k in range(4))
        ex = self._extract_pdf(self._type0_pdf(maps, pages=3))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.notes, [])
        self.assertEqual(ex.text, "\n\n".join(f"[page {n}]\n{expected}" for n in (1, 2, 3)))
        with mock.patch.object(pdf, "MAP_RANGES_FROM", 10 ** 12):                   # as written: not all fit
            ex = self._extract_pdf(self._type0_pdf(maps, pages=3))
        self.assertTrue(ex.truncated)
        self.assertIn("font budget", ex.notes[0])

    @staticmethod
    def _rewritten_map(tag: int, runs: int = 200, repeated: int = 1000) -> bytes:
        """A map sift hands pypdf written as ranges (_as_ranges), of bytes of its own per ``tag``: ``runs`` runs of
        eight codes onto consecutive strings, a bfrange line each so, and ``repeated`` lines of one code that stay as
        they are -- the reviewer's shape, a twentieth of its size. Codes 1 to 4 are "丁丂七丄"."""
        lines = [b"<%04X> <%04X>\n" % (c, 0x4E00 + c) for c in range(8)] * runs + [b"<0100> <0041>\n"] * repeated
        return (b"%% font %d\nbegincmap\n%d beginbfchar\n" % (tag, len(lines)) + b"".join(lines)
                + b"endbfchar\nendcmap\n")

    @staticmethod
    def _own_fonts_pdf(maps, shared: int = 0) -> bytes:
        """A real PDF pypdf reads: a page per map of ``maps``, each showing codes 1 to 4 in a Type0 font of its own
        -- Identity-H, its /ToUnicode that map -- and ``shared`` pages more showing them in the first page's font."""
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
                   PdfTextTest._stream(b"BT /F0 12 Tf 20 700 Td <0001000200030004> Tj ET\n")]
        kids, fonts = [], []
        for cmap in maps:
            first = len(objects) + 1
            fonts.append(first)
            objects += [b"<< /Type /Font /Subtype /Type0 /BaseFont /Song /Encoding /Identity-H /DescendantFonts "
                        b"[%d 0 R] /ToUnicode %d 0 R >>" % (first + 1, first + 2),
                        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Song /CIDSystemInfo << /Registry (Adobe) "
                        b"/Ordering (Identity) /Supplement 0 >> /DW 1000 >>", PdfTextTest._stream(cmap)]
        for font in fonts + fonts[:1] * shared:
            objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F0 %d 0 R "
                           b">> >> /Contents 3 0 R >>" % font)
            kids.append(len(objects))
        objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
        return PdfTextTest._assemble(objects)

    def test_maps_written_as_ranges_are_not_held_for_the_whole_document(self):
        """sift kept every map it hands pypdf written as ranges for the reader's life, in the map's reading and in
        the font's copy: 300 pages each with a font of its own grew the process to 196 MB from a 1 MB file, 0.40 MB a
        page, where 0.1.52 stayed at 64 MB. It now keeps them up to MAX_MAPS_REWRITTEN bytes a document,
        and past that only while the page that hands them is read: what 30 pages leave held past page 6 grows no
        more than it does with the maps handed as written."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import gc
        import tracemalloc
        from unittest import mock
        from sift.extract import pdf
        pages = 30
        maps = [self._rewritten_map(k) for k in range(pages)]
        rewrite = len(pdf._as_ranges(maps[0], 10 ** 12))
        data = self._own_fonts_pdf(maps)
        grown = {}
        enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        real = pdf._Bounds.forget
        try:
            for name, ranges_from in (("as ranges", 1), ("as written", 10 ** 12)):
                held, rewritten = [], []

                def forget(bounds):
                    real(bounds)
                    held.append(tracemalloc.get_traced_memory()[0])

                def ranged_map(reading, fours, ranged=pdf._ranged_map):
                    # A spy that keeps nothing of the map: a mock keeps its arguments.
                    spent = ranged(reading, fours)
                    rewritten.append(reading.ranges is not None)
                    return spent
                with mock.patch.object(pdf, "MAP_RANGES_FROM", ranges_from), \
                        mock.patch.object(pdf, "MAX_MAPS_REWRITTEN", 2 * rewrite), \
                        mock.patch.object(pdf, "MAX_FONT_HELD_UNCOLLECTED", 1), \
                        mock.patch.object(pdf._Bounds, "forget", forget), \
                        mock.patch.object(pdf, "_ranged_map", ranged_map):
                    tracemalloc.start()
                    try:
                        ex = self._extract_pdf(data)
                    finally:
                        tracemalloc.stop()
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.notes, [])
                self.assertEqual(ex.text, "\n\n".join(f"[page {n}]\n丁丂七丄" for n in range(1, pages + 1)))
                self.assertEqual(len(held), pages)
                grown[name] = held[-1] - held[5]
                if name == "as ranges":         # the premise: every map was handed as ranges
                    self.assertEqual(rewritten, [True] * pages)
        finally:
            if enabled:
                gc.enable()
        self.assertGreater(rewrite, 15_000)
        self.assertLess(grown["as ranges"] - grown["as written"], rewrite, (grown, rewrite))

    def test_maps_written_as_ranges_are_kept_up_to_their_bound_and_made_again_past_it(self):
        """A map written as ranges that fits MAX_MAPS_REWRITTEN with those kept before it is kept for the document:
        a font on every page is counted and rewritten once. One that does not is dropped with the page that handed
        it and counted and rewritten again on the next page that names its font -- charged to that page as the
        first time, and read the same. At the bound it is kept; a byte below, it is not."""
        try:
            import pypdf  # noqa: F401
            from pypdf import PdfReader
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import io
        from unittest import mock
        from sift.extract import pdf
        maps = [self._rewritten_map(0), self._rewritten_map(1)]
        sizes = [len(pdf._as_ranges(cmap, 10 ** 12)) for cmap in maps]
        data = self._own_fonts_pdf(maps, shared=3)      # fonts 0, 1, then 0 on three pages more
        for bound, rewrites in ((10 ** 12, 2), (sizes[0], 2), (sizes[0] + sizes[1] - 1, 2),
                                (sizes[0] - 1, 5), (0, 5)):
            with self.subTest(bound=bound), mock.patch.object(pdf, "MAP_RANGES_FROM", 1), \
                    mock.patch.object(pdf, "MAX_MAPS_REWRITTEN", bound), \
                    mock.patch.object(pdf, "_ranged_map", wraps=pdf._ranged_map) as ranged:
                reader = PdfReader(io.BytesIO(data))
                bounds, loads, texts = pdf._Bounds(reader, len(data)), [], []
                for number, page in enumerate(reader.pages, 1):
                    texts.append(bounds.read(page, number))
                    loads.append(bounds.load)
                    bounds.forget()
                self.assertEqual(ranged.call_count, rewrites)
                self.assertEqual(texts, ["丁丂七丄"] * 5)
                self.assertEqual(bounds.notes(), [None] * 6)
                if rewrites == 2:                               # counted once: the pages after charged less
                    self.assertEqual(loads[3:], [loads[2]] * 2)
                    self.assertLess(loads[2], loads[0])
                else:                                           # counted again: charged as the first time
                    self.assertEqual(loads[2:], [loads[0]] * 3)
        self.assertLessEqual(pdf.MAX_TO_UNICODE - pdf.MAX_TO_UNICODE // 4, pdf.MAX_MAPS_REWRITTEN)   # one fits

    def test_a_maps_bytes_are_counted_once_whichever_streams_hold_them(self):
        """Producers write the same map into a font of every page, each its own stream: sift counted each, and
        counting them was what it spent most on such a page. A map whose bytes sift counted before is
        known by their digest and not counted again; one byte apart, it is."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        expected = "\n".join("丁丂七" + chr(0x4E00 + 11 * k + 10) for k in range(3))
        for maps, counted in (([self._cjk_map(300)] * 3, 1),
                              ([self._cjk_map(300) + b"%%%d\n" % k for k in range(3)], 3)):
            with self.subTest(counted=counted):
                calls, real = [], pdf._map_reading

                def map_reading(data, *args, **kwargs):
                    calls.append(len(data))
                    return real(data, *args, **kwargs)
                with mock.patch.object(pdf, "_map_reading", map_reading):
                    ex = self._extract_pdf(self._type0_pdf(maps, pages=2))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.text, f"[page 1]\n{expected}\n\n[page 2]\n{expected}")
                self.assertEqual(len(calls), counted)

    def test_sifts_preparation_of_a_map_is_pypdfs(self):
        """_prepared is pypdf's prepare_cm (_cmap.py:222-260 in 6.19.0) byte for byte, without its loop in Python
        over every "<" of the map, which cost 11.7 ms of the 77.6 ms pypdf spends reading a map of 20,902 entries:
        on every map the tests hold, and on 3,000 strings of what prepare_cm looks for, drawn at random."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import random
        from pypdf._cmap import prepare_cm
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        from sift.extract import pdf
        draw = random.Random(7)
        pieces = [b"<", b">", b" ", b"  ", b"<>", b"< >", b"<<", b">>", b"[", b"]", b"\r", b"\n", b"\r\n", b"\t",
                  b"\x0b", b"%", b"0041", b"4E 00", b".", b"beginbfchar", b"endbfchar", b"beginbfrange", b"endbfrange",
                  b"x", b"<0041>", b"< 00 41 >"]
        drawn = [b"".join(draw.choice(pieces) for _ in range(draw.randrange(0, 40))) for _ in range(3000)]
        for data in [*self._MAPS.values(), *self._RUN_MAPS.values(), *drawn, b"", b" \n ", b">", b"a>b<c>d>e"]:
            stream = DecodedStreamObject()
            stream.set_data(data)
            self.assertEqual(pdf._prepared(data), prepare_cm(DictionaryObject({NameObject("/ToUnicode"): stream})),
                             data[:200])

    def test_what_sift_spends_counting_a_map_is_charged_to_the_page_that_first_names_it(self):
        """What counting a map cost sift was charged only where its font was not handed whole: now on
        the page that first names the font, whatever it is handed of it, and on no page after -- its digest, which
        finds a map sift counted in another stream, and its reading (_Bounds._reading)."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        cmap = self._cmap(3000)
        reading = pdf._map_reading(cmap, 10 ** 12)
        spent = len(cmap) // pdf.MAP_HASH_BYTES_A_UNIT + reading.read      # its digest, and its reading
        self.assertGreater(reading.read, 0)
        whole = pdf.FONT_COST + reading.units
        for bound, first in ((whole + spent, "ZYYZ"), (whole + spent - 1, "ABBA")):
            with self.subTest(bound=bound), mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound):
                ex = self._extract_pdf(self._cmap_pdf(cmap, pages=2))
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.text, f"[page 1]\n{first}\n\n[page 2]\nZYYZ")
                self.assertEqual(ex.notes, [] if first == "ZYYZ" else
                                 [f"page 1 loads fonts past a page's font budget of {bound}: 1 of them "
                                  f"{self._WITHOUT}; their text may be wrong"])

    def test_a_maps_entries_are_charged_not_its_bytes(self):
        """The reviewer's case: 191 bytes of map that pypdf expands to 98,304 entries, named 60 times on a page --
        10.7 s and 674 MB, and not a word said. Each name now costs what pypdf does with the map, two
        million units: one name is read with it, and the other 59 without, said."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift.extract import pdf
        cmap = self._MAPS["the reviewer's two ranges"]
        font = (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 8 0 R /DescendantFonts "
                b"[<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X /CIDSystemInfo << /Registry (Adobe) /Ordering "
                b"(Identity) /Supplement 0 >> >>] >>")
        cost = pdf._map_reading(cmap, 10 ** 12).units
        self.assertGreater(cost, 2_000_000)
        self.assertLessEqual(pdf.FONT_COST + (pdf.FONT_COST + cost), pdf.MAX_FONT_LOAD_PER_PAGE)     # one fits whole,
        self.assertGreater(pdf.FONT_COST + 2 * (pdf.FONT_COST + cost), pdf.MAX_FONT_LOAD_PER_PAGE)   # two do not
        ex, built = self._loads(self._fonts_pdf(font, 60, (self._stream(cmap),)))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(built, ["map"] + ["bare"] * 59)
        self.assertIn("hello", ex.text)
        self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {pdf.MAX_FONT_LOAD_PER_PAGE}: 59 "
                                    f"of them {self._WITHOUT}; their text may be wrong"])

    def test_what_pypdf_holds_of_a_map_is_counted_for_the_collector(self):
        """What makes a map's Font large is its entries -- 10.5 MB for the reviewer's 98,304 -- and sift counted the
        map's 191 bytes towards collecting: 20 pages held 242 MB and sift never collected. What pypdf
        holds of a page's fonts is now counted in bytes, entries and widths included, and collected past
        MAX_FONT_HELD_UNCOLLECTED."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import gc
        from unittest import mock
        from sift.extract import pdf
        cmap = self._MAPS["the reviewer's two ranges"]
        font = (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 12 0 R /DescendantFonts "
                b"[<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X >>] >>")
        data = self._font_pages_pdf(b"/T 3 0 R /F0 4 0 R", 6, font, (self._stream(cmap),))       # the map is 12
        held = 2 * pdf.FONT_HELD + 98304 * pdf.MAP_ENTRY_HELD                    # /T and /F0
        self.assertGreater(held, 10_000_000)
        with mock.patch.object(pdf, "_young", 0), mock.patch.object(pdf, "_old", 0), \
                mock.patch.object(pdf.gc, "collect", wraps=gc.collect) as collect:
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text.count("hello"), 6)
        self.assertEqual(collect.call_count, 6 * held // pdf.MAX_FONT_HELD_UNCOLLECTED)

    def test_pypdfs_fonts_are_collected_young_unless_a_collection_promoted_them(self):
        """Collecting everything after every page past the bound cost a full collection each time -- 240 ms with a
        server's heap, 38.9 s against 9.5 s for 100 pages. What pypdf built for a page is young unless
        Python's own collector ran over the oldest generations while the page was read: sift collects the young
        generations then (gc.collect(1)), and everything only once what may have grown old reaches the bound."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import gc
        from unittest import mock
        from sift.extract import pdf
        font, _, cost, _ = self._font_kinds()["W"]
        page = pdf.FONT_HELD + 2 * (pdf.FONT_HELD + pdf.WIDTH_HELD * 1000)             # /T, and /F0 and /F1
        six = self._font_pages_pdf(b"/T 3 0 R /F0 4 0 R /F1 4 0 R", 6, font)
        real_stats = gc.get_stats

        def aging(calls):
            """gc.get_stats, with one more collection of generation 1 counted on each of its first ``calls`` calls --
            sift asks before and after each page, so the first calls // 2 pages see one run while they are read."""
            count = [0, 0]

            def stats():
                found = real_stats()
                count[0] += 1
                count[1] += count[0] <= calls
                found[1] = dict(found[1], collections=found[1]["collections"] + count[1])
                return found
            return stats
        enabled = gc.isenabled()
        gc.disable()
        try:
            for name, stats, bound, calls in (
                    ("young", aging(0), 2 * page, [(1,)] * 3),
                    ("young, below the bound", aging(0), 7 * page, []),
                    ("every page promoted", aging(12), 2 * page, [(2,)] * 3),
                    ("the first two pages promoted", aging(4), 2 * page, [(2,), (1,), (1,)]),
                    ("the first page promoted", aging(2), 2 * page, [(1,)] * 3)):
                with self.subTest(name), mock.patch.object(pdf, "MAX_FONT_HELD_UNCOLLECTED", bound), \
                        mock.patch.object(pdf, "_young", 0), mock.patch.object(pdf, "_old", 0), \
                        mock.patch.object(pdf.gc, "get_stats", stats), \
                        mock.patch.object(pdf.gc, "collect", wraps=gc.collect) as collect:
                    ex = self._extract_pdf(six)
                    self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                    self.assertEqual([call.args for call in collect.call_args_list], calls)
        finally:
            if enabled:
                gc.enable()

    def test_what_pypdf_logs_building_a_font_is_charged(self):
        """pypdf formats into a log record, on every Font it builds, a /Differences that is no array (whole), a Type1
        font's dictionary where its font file is CFF and fontTools is missing, each /W element that is no number, and
        an /Encoding name it does not know (_cmap._parse_encoding, _cmap._parse_to_unicode,
        Font._collect_cid_character_widths in 6.19.0) -- and sift charged none of it: 3,000 names of a font whose
        /Differences was a dictionary ran 79 s past a 20 s deadline and wrote 4.17 GB of log. Each record
        is now charged LOG_RECORD_COST and a unit a character of what it says, where pypdf would log it. A name that
        does not fit whole is handed without the part that logs, where it can be; an encoding it cannot."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import pypdf._font
        from unittest import mock
        from pypdf.generic import IndirectObject
        from sift.extract import pdf
        long = b"/" + b"N" * 4000
        kinds = {
            "Differences": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding << /Differences << "
                            + b" ".join(b"/k%d 1" % i for i in range(300)) + b" >> >> >>", (), "bare"),
            "W": (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< /Type /Font "
                  b"/Subtype /CIDFontType2 /BaseFont /X /W [" + b"8 0 R " * 30 + b"] >>] >>", (long,), "bare"),
            "Encoding": (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding " + long + b" >>", (), None),
        }
        if not pypdf._font.HAS_FONTTOOLS:
            kinds["Type1C"] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /X /FontDescriptor << /Type /FontDescriptor "
                               b"/FontName /X /Flags 32 /FontFile3 8 0 R >> /Junk [" + b"0 " * 2000 + b"] >>",
                               (b"<< /Subtype /Type1C /Length 4 >>\nstream\n\x01\x00\x04\x01\nendstream",), "bare")
        unit = pdf.FONT_COST
        for kind, (font, extra, stripped) in kinds.items():
            with self.subTest(kind=kind):
                data = self._fonts_pdf(font, 4, extra)
                with self.assertLogs("pypdf", "WARNING") as one:                        # what one build logs
                    reader = pypdf.PdfReader(__import__("io").BytesIO(data))
                    pypdf._font.Font.from_font_resource(IndirectObject(4, 0, reader).get_object())
                messages = [record.getMessage() for record in one.records]
                charge = len(messages) * pdf.LOG_RECORD_COST + sum(map(len, messages))
                bounds = pdf._Bounds(reader, len(data))
                cost, _ = bounds._font(IndirectObject(4, 0, reader).get_object())
                self.assertGreaterEqual(cost.parts + cost.kept, charge)
                self.assertLessEqual(cost.parts + cost.kept, charge + 2 * len(messages) + 5 * unit, messages[0][:80])
                budget = unit + 2 * (unit + cost.kept + cost.parts) + 2 * (unit + cost.kept)   # two whole, two not
                with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", budget), \
                        self.assertLogs("pypdf", "WARNING") as logged:
                    ex, built = self._loads(data)
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertIn("hello", ex.text)
                if stripped:
                    self.assertEqual(len(logged.records), 2 * len(messages))            # the other two log nothing
                    self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {budget}: 2 of them "
                                                f"{self._WITHOUT}; their text may be wrong"])
                else:
                    self.assertEqual(len(logged.records), 4 * len(messages))
                    self.assertEqual(ex.notes, [])

    def test_a_widths_reference_to_an_object_the_file_lacks_is_charged_and_not_looked_for(self):
        """A reference to an object the file does not hold is looked for in the whole file every time pypdf resolves
        it, and logged twice (PdfReader.get_object in 6.19.0 caches it as None, and looks again): pypdf resolves
        every /W element on every Font it builds, and so did sift measuring it -- a 71 KB file with 10,000 such
        elements in one font named once took 22 s -- and a /Font entry naming such an object is resolved on every
        build too. sift no longer resolves what the file lacks, and charges each resolve pypdf makes of it as a
        search of the file."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        font = (b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< /Type /Font "
                b"/Subtype /CIDFontType2 /BaseFont /X /W [" + b"99 0 R " * 2000 + b"] >>] >>")
        data = self._fonts_pdf(font, 3)
        # Resolved by sift's reader, which searches the file once for what it lacks, and so counted.
        kind = pdf._reader_class(pypdf)
        looked, real = [], kind.get_object

        def get_object(reader, reference):
            if getattr(reference, "idnum", reference) == 99:
                looked.append(1)
            return real(reader, reference)
        with mock.patch.object(kind, "get_object", get_object):
            ex, built = self._loads(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(built, ["bare"] * 3)
        self.assertEqual(looked, [])                                    # neither sift nor pypdf looked for it
        self.assertIn("hello", ex.text)
        search = pdf._Bounds(pypdf.PdfReader(__import__("io").BytesIO(data)), len(data)).missing
        self.assertEqual(search, len(data) // pdf.MISSING_REF_BYTES_A_UNIT + 2 * pdf.LOG_RECORD_COST
                         + pdf.MISSING_REF_LOGGED)
        with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", 10 ** 12), self.assertLogs("pypdf", "WARNING"), \
                mock.patch.object(kind, "get_object", get_object):
            looked.clear()
            ex, built = self._loads(self._fonts_pdf(font.replace(b"99 0 R " * 2000, b"99 0 R " * 20), 1))
        self.assertEqual(built, ["W"])
        self.assertEqual(len(looked), 20)                                # pypdf's own, once an element
        data = self._assemble([b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
                               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /T 3 0 R "
                               b"/F0 99 0 R /F1 99 0 R /F2 99 0 R >> >> /Contents 5 0 R >>",
                               self._stream(b"BT /T 12 Tf 20 700 Td (hello) Tj ET\n")])
        search = pdf._Bounds(pypdf.PdfReader(__import__("io").BytesIO(data)), len(data)).missing
        bound = pdf.FONT_COST + (pdf.FONT_COST + search)                     # /T, and one name of the three
        with mock.patch.object(pdf, "MAX_FONT_LOAD_PER_PAGE", bound), \
                mock.patch.object(kind, "get_object", get_object):
            looked.clear()
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text, "[page 1]\nhello")
        self.assertEqual(len(looked), 1)                                 # pypdf, for the one name it was handed
        self.assertEqual(ex.notes, [f"page 1 loads fonts past a page's font budget of {bound}: 2 left out altogether; "
                                    f"their text may be wrong"])

    def test_what_pypdf_logs_of_a_documents_fonts_is_bounded(self):
        """Log volume was bounded only by the deadline: /W elements naming one name of 4,000 characters, 606 KB,
        wrote 16.2 GB to stderr; an unknown /Encoding name, 227 KB, 6.8 GB. What pypdf logs building a
        document's fonts is now counted across its pages, and past MAX_FONT_LOG_PER_DOCUMENT a font is handed
        without the parts that log, or where it would log even so, left out -- said once for the document."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        long = b"/" + b"N" * 4000
        for kind, font, extra, handed in (
                ("W", b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< /Type "
                      b"/Font /Subtype /CIDFontType2 /BaseFont /X /W [9 0 R 9 0 R] >>] >>", long, "bare"),
                ("Encoding", b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding " + long + b" >>", b"null",
                 None)):
            data = self._font_pages_pdf(b"/T 3 0 R /F0 4 0 R", 10, font.replace(b"9 0 R", b"16 0 R"), (extra,))
            with self.assertLogs("pypdf", "WARNING") as one:
                ex, built = self._loads(self._font_pages_pdf(b"/T 3 0 R /F0 4 0 R", 1, font) if kind == "Encoding"
                                        else data)
            per = sum(len(record.getMessage()) for record in one.records) // (1 if kind == "Encoding" else 10)
            bound = 3 * per + per // 2                                                  # three pages' worth
            with self.subTest(kind=kind), mock.patch.object(pdf, "MAX_FONT_LOG_PER_DOCUMENT", bound), \
                    self.assertLogs("pypdf", "WARNING") as logged:
                ex, built = self._loads(data)
            self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
            self.assertEqual(ex.text.count("hello"), 10)
            self.assertLessEqual(sum(len(record.getMessage()) for record in logged.records), bound)
            if handed:
                self.assertEqual(built, [kind] * 3 + ["bare"] * 7)
                what = f"7 of them {self._WITHOUT}"
            else:
                self.assertEqual(built, ["bare"] * 3)
                what = "7 left out altogether"
            self.assertEqual(ex.notes, [f"pypdf would log more than {bound} characters building the fonts of this "
                                        f"document, the first past it on page 4: {what}; their text may be wrong"])

    def test_the_size_of_what_pypdf_logs_is_counted_without_formatting_it(self):
        """What a log record says of an object is its "%s", a container's repr, each reference in it as a reference:
        sift counts its length, to one past a cap, walking the object rather than formatting it."""
        try:
            import pypdf
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import io
        from pypdf.generic import (ArrayObject, BooleanObject, ByteStringObject, DictionaryObject, FloatObject,
                                   IndirectObject, NameObject, NullObject, NumberObject, TextStringObject)
        from sift.extract import pdf
        reader = pypdf.PdfReader(io.BytesIO(self._fonts_pdf(b"<< >>", 1)))
        inner = ArrayObject([NumberObject(1), FloatObject("2.5"), NameObject("/N"), TextStringObject("t'\"\n"),
                             ByteStringObject(b"\x00\xff"), BooleanObject(True), NullObject(),
                             IndirectObject(4, 0, reader), DictionaryObject()])
        outer = DictionaryObject({NameObject("/A"): inner, NameObject("/B"): DictionaryObject({NameObject("/C"): inner}),
                                  NameObject("/D"): ArrayObject()})
        circular = ArrayObject([NumberObject(1)])
        circular.append(circular)
        for value in (outer, inner, NameObject("/" + "E" * 4000), TextStringObject("x" * 10), NumberObject(7),
                      NullObject(), circular, ByteStringObject(b"ab")):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(pdf._logged_size(value, 10 ** 9), len("%s" % (value,)))
        self.assertEqual(pdf._logged_size(outer, 20), 21)

    @staticmethod
    def _log_size(records) -> int:
        """What ``records`` come to as MAX_OP_LOG_PER_DOCUMENT counts them: each message and a line's prefix."""
        from sift.extract import pdf
        return sum(len(record.getMessage()) + pdf.LOG_RECORD_PREFIX for record in records)

    #: Content pypdf logs a record about for every few bytes of it -- parsing it, the page's or a form's, and running
    #: it -- and the forms the pages name.
    _LOGGING_CONTENT = {
        "a number pypdf cannot read, parsed": (b"1-1 " * 4000, {}),     # NumberObject, generic/_base.py:567 in 6.19.0
        "a number of two points, parsed": (b"1.. " * 4000, {}),         # FloatObject, generic/_base.py:507
        "a form's numbers, parsed": (b"/Fm0 Do\n", {b"Fm0": b"1-1 " * 4000}),
        "a form the page lacks, drawn": (b"/X Do\n" * 3000, {}),        # _page.py:1983
    }

    def test_what_pypdf_logs_reading_a_documents_content_is_bounded(self):
        """What pypdf logs reading a page was bounded only by the deadline: 12 pages sharing one compressed content
        stream of numbers it cannot read, 2,772 bytes, wrote 230 MB of log in 2.4 million records in 38 s -- one
        record a token while it parses the content (pypdf/generic/_base.py:507, :567 in 6.19.0), and one a failed
        Do while it runs it (_page.py:1983) -- and a host application keeps every record. sift now counts what
        pypdf logs while it reads a document, as it logs it, and past MAX_OP_LOG_PER_DOCUMENT the page being read
        ends -- mid-parse, or at the next operator -- and the document with it, said."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        hello = b"BT /F1 12 Tf 20 700 Td (hello) Tj ET\n"
        for shape, (logging_content, forms) in self._LOGGING_CONTENT.items():
            with self.subTest(shape=shape):
                with self.assertLogs("pypdf", "WARNING") as whole:
                    ex = self._extract_pdf(self._pages_pdf([hello + logging_content], forms))
                self.assertEqual(ex.notes, [])                                  # within the bound: read, not said
                self.assertEqual(ex.text, "[page 1]\nhello")
                bound = self._log_size(whole.records) // 3                      # a third of one page's worth
                largest = max(len(record.getMessage()) + pdf.LOG_RECORD_PREFIX for record in whole.records)
                data = self._pages_pdf([hello, hello, hello + logging_content, hello], forms)
                with mock.patch.object(pdf, "MAX_OP_LOG_PER_DOCUMENT", bound), \
                        self.assertLogs("pypdf", "WARNING") as logged:
                    ex = self._extract_pdf(data)
                self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
                self.assertEqual(ex.metadata.get("pages"), 4)
                self.assertEqual(ex.text, "[page 1]\nhello\n\n[page 2]\nhello")
                self.assertTrue(ex.truncated)
                self.assertLessEqual(self._log_size(logged.records), bound + largest)   # one record past it at most
                self.assertEqual(ex.notes, [f"pypdf logged more than {bound} characters reading this document, "
                                            f"past them on page 3; the text of page 3 and after is not in the body"])

    def test_the_reviewers_document_of_twelve_pages_logs_within_the_bound(self):
        """The reviewer's shape at the bound itself: 12 pages sharing one content stream of 800,000 bytes of numbers
        pypdf cannot read wrote 230 MB. Now the first page ends at MAX_OP_LOG_PER_DOCUMENT, and nothing after it is
        read."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from sift.extract import pdf
        with self.assertLogs("pypdf", "WARNING") as logged:
            ex = self._extract_pdf(self._content_pdf(12, b"1-1 " * 200_000))
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text, "")
        largest = max(len(record.getMessage()) + pdf.LOG_RECORD_PREFIX for record in logged.records)
        self.assertLessEqual(self._log_size(logged.records), pdf.MAX_OP_LOG_PER_DOCUMENT + largest)
        self.assertGreater(self._log_size(logged.records), pdf.MAX_OP_LOG_PER_DOCUMENT)
        self.assertEqual(ex.notes[:1], [f"pypdf logged more than {pdf.MAX_OP_LOG_PER_DOCUMENT} characters reading "
                                        f"this document, past them on page 1; the text of page 1 and after is not in "
                                        f"the body"])

    def test_what_pypdf_logs_is_counted_per_document_across_threads(self):
        """What pypdf logs is counted for the document the thread that logs it reads: a server reads uploads in
        threads of its own, and one document's records must not end another's pages. The threads provably overlap:
        each noisy document holds on its first record until the quiet one is being read, and the quiet one holds on
        its own first record until both noisy ones are done -- so every record they log past the bound is logged
        while all three documents are read, and a count shared by the threads ends the quiet document (a mutant
        with one count for every thread passed 9 of 13 runs of this test without that, a review). Whatever sift
        installs to count them is gone once no document is read, and pypdf's records reach where they reached
        before -- Python's last resort handler where logging is not configured at all."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import contextlib
        import io
        import logging
        import threading
        from unittest import mock
        from sift.extract import pdf
        hello = b"BT /F1 12 Tf 20 700 Td (hello) Tj ET\n"
        noisy = self._pages_pdf([hello + b"1-1 " * 4000] * 3, {})
        quiet = self._pages_pdf([hello + b"1-1 "] + [hello] * 59, {})          # one record of its own, page 1
        logger = logging.getLogger("pypdf")
        before = list(logger.handlers)
        results, errors, held = {}, [], []
        noisy_logging = threading.Semaphore(0)
        quiet_read, noisy_done = threading.Event(), threading.Barrier(3)
        wait = 30

        class Overlap(logging.Handler):
            """Holds each thread on its first record, in the logging path, until the others are where they must
            be. Without the handler lock: the threads wait here for one another."""

            def __init__(self):
                super().__init__()
                self.seen = set()

            def handle(self, record):
                name = threading.current_thread().name
                if name in self.seen:
                    return True
                self.seen.add(name)
                if name.startswith("noisy"):
                    noisy_logging.release()
                    held.append((name, quiet_read.wait(wait)))
                elif name == "quiet":
                    quiet_read.set()
                    try:
                        noisy_done.wait(wait)
                        held.append((name, True))
                    except threading.BrokenBarrierError:
                        held.append((name, False))
                return True

        def read(name, data):
            try:
                results[name] = self._extract_pdf(data)
            except Exception as exc:  # noqa: BLE001 - said below
                errors.append(exc)
            finally:
                if name.startswith("noisy"):
                    noisy_done.wait(wait)

        def read_quiet():
            # Begun once both noisy documents are read and logging, so that its count is the one set last.
            if all(noisy_logging.acquire(timeout=wait) for _ in range(2)):
                read("quiet", quiet)
            else:
                errors.append("the noisy documents never logged")
        with mock.patch.object(pdf, "MAX_OP_LOG_PER_DOCUMENT", 20_000), self.assertLogs("pypdf", "WARNING"):
            overlap = Overlap()
            logger.addHandler(overlap)
            try:
                threads = [threading.Thread(target=read, args=(name, noisy), name=name)
                           for name in ("noisy", "noisy again")]
                threads.append(threading.Thread(target=read_quiet, name="quiet"))
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(3 * wait)
            finally:
                logger.removeHandler(overlap)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(sorted(held), [("noisy", True), ("noisy again", True), ("quiet", True)])    # they overlapped
        self.assertEqual(errors, [])
        self.assertEqual(results["quiet"].text.count("hello"), 60)
        self.assertEqual(results["quiet"].notes, [])
        for name in ("noisy", "noisy again"):
            self.assertEqual(results[name].text, "")
            self.assertEqual(results[name].notes[:1], ["pypdf logged more than 20000 characters reading this document, "
                                                   "past them on page 1; the text of page 1 and after is not in the "
                                                   "body"])
        self.assertEqual(logger.handlers, before)                               # nothing of sift's left behind
        self.assertEqual(self._extract_pdf(b"%PDF-1.4\nbroken").metadata.get("engine"), "stdlib-fallback")
        self.assertEqual(logger.handlers, before)                               # nor where pypdf failed
        stderr = io.StringIO()
        with mock.patch.object(logging.root, "handlers", []), mock.patch.object(logger, "handlers", []), \
                mock.patch.object(logger, "propagate", True), contextlib.redirect_stderr(stderr):
            ex = self._extract_pdf(self._pages_pdf([hello + b"1-1 "], {}))
        self.assertEqual(ex.text, "[page 1]\nhello")
        self.assertEqual(stderr.getvalue(), "NumberObject(b'1-1') invalid; use 0 instead\n")   # the last resort's

    _CONFIGURE_WHILE_LOGGING = """
import io, logging, logging.config, os, threading, time
from sift.extract import pdf
entered, go = threading.Event(), threading.Event()


class Slow:
    def __str__(self):
        entered.set()
        go.wait(20)
        return "slow"


logging.root.addHandler(logging.StreamHandler(io.StringIO()))      # the host's handler


def log():
    with pdf._counting_pypdf_log():
        logging.getLogger("pypdf.generic").warning(Slow())


reader = threading.Thread(target=log, daemon=True)
reader.start()
entered.wait(20)                                    # the reader is inside sift's count of the record
configure = threading.Thread(target=logging.config.dictConfig, daemon=True, args=({
    "version": 1, "disable_existing_loggers": False,
    "handlers": {"h": {"class": "logging.StreamHandler", "stream": "ext://sys.stdout", "level": "CRITICAL"}},
    "root": {"handlers": ["h"], "level": "WARNING"}},))
configure.start()
while configure.is_alive():                         # until it holds logging's module lock, or is done
    if not logging._lock.acquire(blocking=False):
        break
    logging._lock.release()
    time.sleep(0.001)
go.set()
reader.join(10)
configure.join(10)
print(f"reader alive={reader.is_alive()} configure alive={configure.is_alive()}", flush=True)
os._exit(0)                                         # a hung thread would hang logging's shutdown at exit
"""

    def test_counting_what_pypdf_logs_does_not_deadlock_a_logging_configuration(self):
        """sift's _LogWatch ran under its handler lock (Handler.handle) and looked the record's logger up with
        logging.getLogger, which takes logging's module lock: the reverse of logging's own order -- dictConfig holds
        the module lock and then takes every handler's lock to shut it down. A thread reading a noisy PDF while the
        host configured logging hung the process, 2 of 30 fresh processes and 3 of 3 with repeated configurations. Here deterministically: the reader held inside the count while dictConfig runs, in a process of
        its own -- a hang there cannot take this one with it."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        import subprocess
        import sys
        import sift
        root = str(Path(sift.__file__).resolve().parent.parent)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
        try:
            done = subprocess.run([sys.executable, "-c", self._CONFIGURE_WHILE_LOGGING], env=env, timeout=60,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"the process hung: {exc}")
        self.assertEqual((done.stdout, done.returncode), ("reader alive=False configure alive=False\n", 0),
                         done.stderr[-2000:])

    def test_the_log_watch_takes_no_lock_of_logging(self):
        """What _LogWatch does with a record takes neither logging's module lock nor its own handler lock -- the
        order of those two is what deadlocked -- and still decides as logging does whether it is the one handler
        the record reaches: a logger the record names that logging never made is not made, and its nearest
        ancestor stands in for it, as it would be its parent."""
        import contextlib
        import io
        import logging
        from unittest import mock
        from sift.extract import pdf

        class Forbidden:
            def __enter__(self):
                raise AssertionError("a lock of logging taken")
            acquire = __enter__

            def __exit__(self, *_):
                return False
            release = __exit__
        logger = logging.getLogger("pypdf")
        absent = "pypdf.sift-test.no-such-logger"
        self.assertNotIn(absent, logging.Logger.manager.loggerDict)                    # the test's own premise

        def record(name):
            return logging.makeLogRecord({"name": name, "levelno": logging.WARNING, "levelname": "WARNING",
                                          "msg": "said"})
        other = logging.StreamHandler(io.StringIO())
        for name, handlers, root_handlers, propagate, reaches_last_resort in (
                ("pypdf", [], [], True, True),                      # nothing configured: the last resort
                ("pypdf", [], [other], True, False),                # the host's handler on the root
                ("pypdf", [other], [], True, False),                # one on pypdf's logger
                ("pypdf", [], [other], False, True),                # the root's not reached
                (absent, [], [], True, True),                       # a logger logging never made
                (absent, [other], [], True, False),                 # ... whose ancestor has a handler
                ("root", [], [], True, True)):                      # the root's own records
            with self.subTest(name=name, handlers=len(handlers), root=len(root_handlers), propagate=propagate):
                stderr = io.StringIO()
                logged = pdf._LogCount()
                token = pdf._log_count.set(logged)
                try:
                    with mock.patch.object(logging, "_lock", Forbidden()), \
                            mock.patch.object(pdf._LOG_WATCH, "lock", Forbidden()), \
                            mock.patch.object(logger, "handlers", [pdf._LOG_WATCH, *handlers]), \
                            mock.patch.object(logger, "propagate", propagate), \
                            mock.patch.object(logging.root, "handlers", list(root_handlers)), \
                            contextlib.redirect_stderr(stderr):
                        pdf._LOG_WATCH.handle(record(name))
                finally:
                    pdf._log_count.reset(token)
                self.assertEqual((logged.records, logged.characters), (1, len("said") + pdf.LOG_RECORD_PREFIX))
                self.assertEqual(stderr.getvalue(), "said\n" if reaches_last_resort else "")
        self.assertNotIn(absent, logging.Logger.manager.loggerDict)
        stderr = io.StringIO()
        with mock.patch.object(logger, "handlers", [pdf._LOG_WATCH]), mock.patch.object(logging.root, "handlers", []), \
                mock.patch.object(logger, "propagate", True), contextlib.redirect_stderr(stderr):
            pdf._LOG_WATCH.handle(record("pypdf"))                  # no document read: counted nowhere, passed on
        self.assertEqual(stderr.getvalue(), "said\n")

    def test_what_pypdf_logs_of_a_font_name_of_an_object_the_file_lacks_is_bounded(self):
        """A /Font entry naming an object the file lacks is resolved on every Font pypdf builds, and logged
        (PdfReader.get_object in 6.19.0): sift charged the search of the file, and none of what pypdf logs of it
        against MAX_FONT_LOG_PER_DOCUMENT. Each is now charged MISSING_REF_LOGGED characters."""
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")
        from unittest import mock
        from sift.extract import pdf
        data = self._font_pages_pdf(b"/T 3 0 R /F0 99 0 R /F1 99 0 R", 10)
        bound = 5 * pdf.MISSING_REF_LOGGED + pdf.MISSING_REF_LOGGED // 2               # five names' worth
        with mock.patch.object(pdf, "MAX_FONT_LOG_PER_DOCUMENT", bound), \
                self.assertLogs("pypdf", "WARNING") as logged:
            ex = self._extract_pdf(data)
        self.assertEqual(ex.metadata.get("engine"), "pypdf", ex.notes)
        self.assertEqual(ex.text.count("hello"), 10)
        self.assertLessEqual(sum(len(record.getMessage()) for record in logged.records), bound)
        self.assertEqual(sum("Object 99 0 not defined" in line for line in logged.output), 5)
        self.assertEqual(ex.notes, [f"pypdf would log more than {bound} characters building the fonts of this "
                                    f"document, the first past it on page 3: 15 left out altogether; their text may "
                                    f"be wrong"])

    def test_max_bytes_is_honoured_and_said_out_loud(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.pdf"
            p.write_bytes(b"%PDF-1.4\n" + b"x" * 5000)
            ex = extract(p, detect(p), 100)
        self.assertTrue(ex.truncated)
        self.assertTrue(any("read only the first" in n for n in ex.notes))


class PdfStructureTest(unittest.TestCase):
    """What pypdf decodes and parses of a document's structure -- its cross-reference streams and object streams --
    bounded per document and within the time budget."""

    PAGE = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 3 0 R "
            b"/Resources << /Font << /F1 4 0 R >> >> >>")
    CONTENT = b"BT /F1 12 Tf 20 100 Td (hello) Tj ET"
    FONT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"

    def setUp(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")

    @staticmethod
    def _xref_rows(xref: dict, size: int) -> bytes:
        """The rows of a cross-reference stream of /W [1 4 2] for ``xref``, number -> (type, field 2, field 3)."""
        rows = b"".join(bytes([t]) + a.to_bytes(4, "big") + b.to_bytes(2, "big")
                        for n in range(size) for (t, a, b) in [xref.get(n, (0, 0, 0))])
        return b"\x00" * 5 + b"\xff\xff" + rows[7:]

    @classmethod
    def _xref_streams_pdf(cls, count: int, padding: int) -> bytes:
        """One page in a classic table, then ``count`` incremental updates, each a cross-reference stream of one
        entry padded with zeros to ``padding`` bytes decoded (the reviewer's xrefs.py, scaled)."""
        out, offsets = bytearray(b"%PDF-1.7\n"), [0]
        bodies = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>", None, cls.FONT,
                  cls.PAGE, b"<< /Title (Old title) >>"]
        for number, body in enumerate(bodies, 1):
            offsets.append(len(out))
            if body is None:
                out += b"%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (
                    number, len(cls.CONTENT), cls.CONTENT)
            else:
                out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
        table = len(out)
        out += b"xref\n0 7\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets[1:])
        out += b"trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\nstartxref\n%d\n%%%%EOF\n" % table
        previous = table
        for k in range(count):
            number, at = 7 + k, len(out)
            entry = bytes([1]) + at.to_bytes(4, "big") + bytes([0])
            packer = zlib.compressobj(9)
            data = packer.compress(entry) + packer.compress(bytes(padding - len(entry))) + packer.flush()
            out += (b"%d 0 obj\n<< /Type /XRef /Size %d /Index [%d 1] /W [1 4 1] /Root 1 0 R /Info 6 0 R /Prev %d "
                    b"/Filter /FlateDecode /Length %d >>\nstream\n" % (number, number + 1, number, previous, len(data)))
            out += data + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at
            previous = at
        return bytes(out)

    @classmethod
    def _object_streams_pdf(cls, pages: int, padding: int = 0, shared: bool = False, bodies=None) -> bytes:
        """``pages`` pages whose dictionaries are in object streams, each padded with zeros to ``padding`` bytes
        decoded: one stream a page (the reviewer's objstm.py), or where ``shared``, one stream whose header lists
        every page at offset 0 (shared_objstm.py). ``bodies`` replaces the page dictionaries, one a page, in one
        stream at their own offsets."""
        out, xref = bytearray(b"%PDF-1.7\n"), {}

        def obj(number, body):
            xref[number] = (1, len(out), 0)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))

        def stream(number, entries, data):
            xref[number] = (1, len(out), 0)
            out.extend(b"%d 0 obj\n<< %s /Length %d >>\nstream\n" % (number, entries, len(data)) + data
                       + b"\nendstream\nendobj\n")

        def packed(raw):
            packer = zlib.compressobj(9)
            return packer.compress(raw) + packer.compress(bytes(max(padding - len(raw), 0))) + packer.flush()
        numbers = [10 + 2 * k for k in range(pages)]
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % p for p in numbers), pages))
        stream(3, b"", cls.CONTENT)
        obj(4, cls.FONT)
        if shared or bodies is not None:
            kept = bodies if bodies is not None else [cls.PAGE] * pages
            offsets, at = [], 0
            for body in kept:
                offsets.append(0 if shared else at)
                at += len(body) + 1
            header = b" ".join(b"%d %d" % (p, o) for p, o in zip(numbers, offsets)) + b" "
            raw = header + (kept[0] if shared else b"\n".join(kept) + b"\n")
            holder = 10 + 2 * pages
            stream(holder, b"/Type /ObjStm /N %d /First %d /Filter /FlateDecode" % (pages, len(header)), packed(raw))
            for k, p in enumerate(numbers):
                xref[p] = (2, holder, k)
        else:
            for p in numbers:
                header = b"%d 0 " % p
                stream(p + 1, b"/Type /ObjStm /N 1 /First %d /Filter /FlateDecode" % len(header),
                       packed(header + cls.PAGE + b"\n"))
                xref[p] = (2, p + 1, 0)
        size = max(xref) + 2
        at = len(out)
        xref[size - 1] = (1, at, 0)
        rows = zlib.compress(cls._xref_rows(xref, size))
        out.extend(b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R /Filter /FlateDecode /Length %d >>\n"
                   b"stream\n" % (size - 1, size, len(rows)) + rows
                   + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at)
        return bytes(out)

    @staticmethod
    def _peak(call):
        """(what ``call`` returned, the most Python held meanwhile over what it held before, in bytes)."""
        import gc
        import tracemalloc
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            result = call()
            return result, tracemalloc.get_traced_memory()[1] - before
        finally:
            tracemalloc.stop()

    def _opened(self, data: bytes):
        """(_with_pypdf of ``data``, the readers it opened)."""
        from unittest import mock
        from sift.extract import pdf
        readers, real = [], pdf._open

        def opening(*args, **kwargs):
            readers.append(real(*args, **kwargs))
            return readers[-1]
        with mock.patch.object(pdf, "_open", opening):
            return pdf._with_pypdf(data), readers

    def test_a_cross_reference_stream_within_the_budget_is_read_and_not_held(self):
        """Happy path: a cross-reference stream a document needs is read, and its decoded copy goes once pypdf read
        it -- pypdf kept it in resolved_objects for the reader's life (_reader.py:1234 in 6.19.0)."""
        from sift.extract import pdf
        (text, meta, notes, _), readers = self._opened(self._xref_streams_pdf(1, 64))
        self.assertEqual((text, meta.get("engine"), notes), ("[page 1]\nhello", "pypdf", []))
        self.assertEqual(len(readers), 1)
        self.assertEqual(readers[0].resolved_objects, {})     # nothing kept past the document
        self.assertLessEqual(readers[0].structure.spent, pdf._structure_budget(len(self._xref_streams_pdf(1, 64))))

    def test_cross_reference_streams_decode_within_the_documents_budget(self):
        """Twelve cross-reference streams of 73 MB decoded in an 839 KB file held 949 MB: pypdf decoded
        each to its own 75 MB limit and kept every copy. Each now decodes within what is left of the document's
        budget, and none is kept once pypdf read it."""
        from sift.extract import pdf
        data = self._xref_streams_pdf(4, 4_000_000)
        budget = pdf._structure_budget(len(data))
        self.assertLess(budget, 4_000_000)
        (text, meta, notes, _), peak = self._peak(lambda: pdf._with_pypdf(data))
        self.assertLess(peak, 3 * budget + (2 << 20), f"{peak} bytes held")
        said = (f"pypdf's cross-reference and object streams of this document decode to more than {budget} bytes, "
                f"{pdf.STRUCTURE_BYTES_A_FILE_BYTE} times its size; what is past that was not read")
        # pypdf gives up on a document whose newest cross-reference stream it cannot read, and the fallback reads it.
        self.assertIsNone(text)
        self.assertIn(said, notes)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.pdf"
            path.write_bytes(data)
            ex = extract(path, detect(path), 25_000_000)
        self.assertIn("hello", ex.text)
        self.assertIn(said, ex.notes)

    def test_object_streams_are_charged_to_the_documents_budget(self):
        """24 pages each alone in an object stream of 73 MB held 1.8 GB, and 1.7 GB after sift returned.
        What object streams decode to is charged to the document's budget, and their decoded copies dropped."""
        from sift.extract import pdf
        data = self._object_streams_pdf(6, padding=300_000)
        budget = pdf._structure_budget(len(data))
        self.assertLess(budget, 6 * 300_000)
        (text, meta, notes, _), peak = self._peak(lambda: pdf._with_pypdf(data))
        self.assertLess(peak, 3 * budget + (2 << 20), f"{peak} bytes held")
        self.assertIn(f"pypdf's cross-reference and object streams of this document decode to more than {budget} "
                      f"bytes", " ".join(notes))

    def test_object_streams_within_the_budget_read_as_before_and_are_not_held(self):
        """Happy path: pages in object streams within the budget read as pypdf reads them, and nothing the reader
        decoded or parsed is held once the document is read, collector or not: the reader is a reference cycle
        pypdf's text extraction keeps (TextExtraction), and every stream pypdf decoded stayed with it until a full
        collection -- 892 MB after sift returned, 52 MB once collected."""
        import gc
        import tracemalloc
        from sift.extract import pdf
        data = self._object_streams_pdf(3, padding=200_000)
        pdf._with_pypdf(data)                       # what pypdf and sift build once for any document
        gc.collect()
        gc.disable()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            (text, meta, notes, _), readers = self._opened(data)
            held = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()
            gc.enable()
        self.assertEqual(text, "[page 1]\nhello\n\n[page 2]\nhello\n\n[page 3]\nhello")
        self.assertEqual((meta.get("engine"), notes), ("pypdf", []))
        self.assertEqual(readers[0].resolved_objects, {})
        self.assertLess(held, 200_000, f"{held} bytes held past the document of 600,000 decoded")

    def test_one_object_stream_at_the_budget_is_read_and_one_past_it_is_not(self):
        """Edge: the budget is inclusive."""
        from unittest import mock
        from sift.extract import pdf
        data = self._object_streams_pdf(1, padding=5_000)
        with mock.patch.object(pdf, "_structure_budget", lambda size: 5_000 + 1_000):
            text, _, notes, _ = pdf._with_pypdf(data)
        self.assertEqual((text, notes), ("[page 1]\nhello", []))
        with mock.patch.object(pdf, "_structure_budget", lambda size: 4_000):
            text, _, notes, _ = pdf._with_pypdf(data)
        self.assertIn("decode to more than 4000 bytes", " ".join(notes))

    def test_the_budget_is_derived_from_the_file_and_the_corpus(self):
        """A multiple of the file's size, at least MAX_INFLATED: 12 times the most any of 111 real documents
        decoded of their cross-reference and object streams against their size (0.66, testPopupAnnotation.pdf)."""
        from sift.extract import pdf
        self.assertEqual(pdf._structure_budget(10), pdf.MAX_INFLATED)
        self.assertEqual(pdf._structure_budget(10 ** 7), pdf.STRUCTURE_BYTES_A_FILE_BYTE * 10 ** 7)
        self.assertGreaterEqual(pdf.STRUCTURE_BYTES_A_FILE_BYTE, 12 * 0.66)

    def _within(self, data: bytes, seconds: float):
        """(what reading ``data`` under a budget of ``seconds`` came to, how long it took)."""
        import time
        from sift import deadline
        from sift.extract import pdf
        started = time.monotonic()
        try:
            with deadline.limit(seconds):
                what = pdf._with_pypdf(data)
        except deadline.DeadlineExceeded as exc:
            what = exc
        return what, time.monotonic() - started

    def test_objects_sharing_the_bytes_of_one_object_stream_stop_within_the_time_budget(self):
        """20 pages at offset 0 of one object stream, 72 KB of file, ran 47 s past a 5 s budget: pypdf
        parses every object of the stream in one call, and skips the zeros after each byte by byte. The budget is
        now checked while it parses."""
        data = self._object_streams_pdf(300, padding=700_000, shared=True)
        what, took = self._within(data, 0.5)
        self.assertLess(took, 0.5 + 0.5, what)

    def test_objects_sharing_the_bytes_of_one_object_stream_are_bounded_without_a_budget(self):
        """Sad path: with no time budget set, parsing one object stream reads at most OBJECT_STREAM_READS_A_BYTE
        times its bytes; past that its objects are not read, and the notes say so. Unbounded, each object at offset
        0 reads the zeros after it, objects times bytes: eight times both is 64 times the reads, bounded 8 -- so
        eight times the objects and the bytes cost at most linear_bound(8) times the CPU, the two read in turn.
        Two seconds of wall clock for 300 objects and 700,000 bytes failed at 2.14 s under a host load of 27 to 49
        (0.1.73)."""
        from sift.extract import pdf

        def reading(pages, padding):
            data = self._object_streams_pdf(pages, padding=padding, shared=True)
            return lambda: pdf._with_pypdf(data)
        ratio = cpu_ratio(reading(160, 400_000), reading(20, 50_000))
        self.assertLess(ratio, linear_bound(8), f"{ratio:.1f}x the CPU for 8x the objects and bytes")
        text, meta, notes, _ = reading(160, 400_000)()
        self.assertIn("an object stream pypdf parses past", " ".join(notes))

    def test_references_in_a_fonts_widths_to_one_object_stream_stop_within_the_time_budget(self):
        """A /W of 20 references to objects at offset 0 of one object stream ran 46.8 s past a 5 s budget: sift's own walk of /W resolves each."""
        refs = [100 + i for i in range(300)]
        out, xref = bytearray(b"%PDF-1.7\n"), {}

        def obj(number, body):
            xref[number] = (1, len(out), 0)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
               b"/Resources << /Font << /F0 5 0 R >> >> >>")
        xref[4] = (1, len(out), 0)
        content = b"BT /F0 12 Tf 20 100 Td <0001> Tj ET"
        out.extend(b"4 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (len(content), content))
        obj(5, b"<< /Type /Font /Subtype /Type0 /BaseFont /Song /Encoding /Identity-H /DescendantFonts [6 0 R] >>")
        obj(6, b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Song /CIDSystemInfo << /Registry (Adobe) "
               b"/Ordering (Identity) /Supplement 0 >> /W [%s] >>" % b" ".join(b"%d 0 R" % r for r in refs))
        header = b" ".join(b"%d 0" % r for r in refs) + b" "
        packer = zlib.compressobj(9)
        raw = header + b"<< /A 1 >>"
        data = packer.compress(raw) + packer.compress(bytes(700_000 - len(raw))) + packer.flush()
        holder = 100 + len(refs)
        xref[holder] = (1, len(out), 0)
        out.extend(b"%d 0 obj\n<< /Type /ObjStm /N %d /First %d /Filter /FlateDecode /Length %d >>\nstream\n"
                   % (holder, len(refs), len(header), len(data)) + data + b"\nendstream\nendobj\n")
        for k, r in enumerate(refs):
            xref[r] = (2, holder, k)
        size = max(xref) + 2
        at = len(out)
        xref[size - 1] = (1, at, 0)
        rows = zlib.compress(self._xref_rows(xref, size))
        out.extend(b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R /Filter /FlateDecode /Length %d >>\n"
                   b"stream\n" % (size - 1, size, len(rows)) + rows
                   + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at)
        what, took = self._within(bytes(out), 0.5)
        self.assertLess(took, 0.5 + 0.5, what)

    def test_sifts_parse_of_an_object_stream_is_pypdfs(self):
        """What sift's reader makes of every object of an object stream -- well-formed, /N past what the stream can
        hold, an object pypdf cannot read, two objects at one offset -- is what pypdf's own makes of it, and pypdf
        logs the same."""
        import io
        import logging
        import pypdf
        from sift.extract import pdf
        shapes = {
            "well-formed": self._object_streams_pdf(3, bodies=[self.PAGE] * 3),
            "an unreadable object": self._object_streams_pdf(2, bodies=[self.PAGE, b"<< /A (unterminated"]),
            "two at one offset": self._object_streams_pdf(3, padding=64, shared=True),
            "a number and an array": self._object_streams_pdf(2, bodies=[b"42", b"[1 2 (x) /N]"]),
        }
        stated = self._object_streams_pdf(2, bodies=[self.PAGE, self.PAGE])
        shapes["/N past its stream"] = stated.replace(b"/N 2 /First", b"/N 9999 /First")

        def read(opener, data):
            sink = io.StringIO()
            handler = logging.StreamHandler(sink)
            logger = logging.getLogger("pypdf")
            level = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
            try:
                reader = opener(data)
                made = [re.sub(r"IndirectObject\((\d+), (\d+), \d+\)", r"IndirectObject(\1, \2)",
                               repr(reader.get_object(number))) for number in sorted(reader.xref_objStm)]
            except Exception as exc:  # noqa: BLE001 - both raise alike or not
                made = [type(exc).__name__]
            finally:
                logger.removeHandler(handler)
                logger.setLevel(level)          # as it was: the level is the process's
            return made, sink.getvalue()
        before = logging.getLogger("pypdf").level
        for name, data in shapes.items():
            with self.subTest(name):
                structure = pdf._Structure(pdf._structure_budget(len(data)), pdf._entry_budget(len(data)))
                self.assertEqual(read(lambda d: pdf._open(d, structure), data),
                                 read(lambda d: pypdf.PdfReader(io.BytesIO(d)), data))
        self.assertEqual(logging.getLogger("pypdf").level, before)

    def test_the_font_names_a_page_loads_at_most_are_said_as_the_constants_make_them(self):
        """FONT_COST's comment said it bounds the names one page loads at 5,000; MAX_FONT_LOAD_PER_PAGE made it
        6,250."""
        import inspect
        from sift.extract import pdf
        source = inspect.getsource(pdf)
        comment = source[source.index("#: What every name pypdf builds a Font for costs"):source.index("FONT_COST =")]
        self.assertIn(f"{pdf.MAX_FONT_LOAD_PER_PAGE // pdf.FONT_COST:,}", comment)

    def test_bounds_derived_from_max_inflated_say_so_and_hold_their_documented_ratios(self):
        """MAX_FONT_LOG_PER_DOCUMENT, MAX_OP_LOG_PER_DOCUMENT and MAX_MAPS_REWRITTEN are derived from MAX_INFLATED
        (MAX_TO_UNICODE is 4 times it) while their comments justified them only as ratios to measured maxima: each comment now names the constant it is derived from, and the ratios it states hold."""
        import inspect
        from sift.extract import pdf
        source = inspect.getsource(pdf)
        for name, derived, measured in (("MAX_FONT_LOG_PER_DOCUMENT", "MAX_INFLATED", 6_341),
                                        ("MAX_OP_LOG_PER_DOCUMENT", "MAX_INFLATED", 7_173),
                                        ("MAX_MAPS_REWRITTEN", "MAX_TO_UNICODE", None)):
            with self.subTest(name):
                end = source.index(f"\n{name} = ")
                comment = source[source.rindex("\n\n", 0, end):end].replace("\n#: ", " ")
                self.assertIn(derived, comment)
                value = getattr(pdf, name)
                if measured is not None:
                    stated = int(re.search(r"(\d+) times the most", comment).group(1))
                    self.assertEqual(value // measured, stated)
        self.assertEqual(pdf.MAX_FONT_LOG_PER_DOCUMENT, pdf.MAX_INFLATED)
        self.assertEqual(pdf.MAX_OP_LOG_PER_DOCUMENT, 4 * pdf.MAX_INFLATED)
        self.assertEqual(pdf.MAX_MAPS_REWRITTEN, pdf.MAX_TO_UNICODE)
        self.assertLessEqual(pdf.MAX_TO_UNICODE - pdf.MAX_TO_UNICODE // 4, pdf.MAX_MAPS_REWRITTEN)


class _PypdfLog:
    """What pypdf logs while it is in force, as (logger name, message); its logger's level is put back after."""

    def __enter__(self):
        import logging
        self.records, self.logger = [], logging.getLogger("pypdf")
        self.level = self.logger.level
        outer = self

        class Keep(logging.Handler):
            def emit(self, record):
                outer.records.append((record.name, record.getMessage()))
        self.handler = Keep()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.level)

    def characters(self):
        from sift.extract import pdf
        return sum(len(message) + pdf.LOG_RECORD_PREFIX for _, message in self.records)


class PdfReaderBuildTest(unittest.TestCase):
    """What pypdf does while its reader is made -- the cross-reference sections it lists, the table it rebuilds, the
    file it searches for an object it lacks -- bounded per document, within the time budget and the log bound."""

    PAGE, CONTENT, FONT = PdfStructureTest.PAGE, PdfStructureTest.CONTENT, PdfStructureTest.FONT

    def setUp(self):
        try:
            import pypdf  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("no pypdf")

    @classmethod
    def _entries_pdf(cls, kind: int, count: int, padding: int = 0) -> bytes:
        """One page in a classic table, then a cross-reference stream of ``count`` entries of type ``kind`` (1: at
        offset 5, where no object is; 2: in object stream 5) of two bytes each (/W [1 1 0]), numbered from 1000, and
        ``padding`` bytes of a stream nothing refers to (the reviewer's entries.py)."""
        out, offsets = bytearray(b"%PDF-1.7\n"), [0]
        bodies = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                  cls.PAGE.replace(b"/Contents 3 0 R", b"/Contents 4 0 R").replace(b"4 0 R >> >>", b"5 0 R >> >>"),
                  None, cls.FONT, b"<< /Title (Old title) >>", b"pad"]
        for number, body in enumerate(bodies, 1):
            offsets.append(len(out))
            if body is None or body == b"pad":
                data = cls.CONTENT if body is None else bytes(range(256)) * (padding // 256) + b"x"
                out += b"%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (number, len(data), data)
            else:
                out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
        table = len(out)
        out += b"xref\n0 8\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets[1:])
        out += b"trailer\n<< /Size 8 /Root 1 0 R /Info 6 0 R >>\nstartxref\n%d\n%%%%EOF\n" % table
        at, rows = len(out), zlib.compress(bytes([kind, 5]) * count, 9)
        out += (b"8 0 obj\n<< /Type /XRef /Size %d /Index [1000 %d] /W [1 1 0] /Root 1 0 R /Info 6 0 R /Prev %d "
                b"/Filter /FlateDecode /Length %d >>\nstream\n" % (1000 + count, count, table, len(rows)))
        out += rows + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at
        return bytes(out)

    @classmethod
    def _rebuilt_pdf(cls, streams: int, size: int, digits: bytes = b"0 ") -> bytes:
        """One page whose table is lost (startxref 0), and ``streams`` object streams that decode to ``size``
        bytes of ``digits`` repeated, found only by rebuilding the table (the reviewer's rebuild.py)."""
        out = bytearray(b"%PDF-1.7\n")
        bodies = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                  cls.PAGE.replace(b"/Contents 3 0 R", b"/Contents 4 0 R").replace(b"4 0 R >> >>", b"5 0 R >> >>"),
                  None, cls.FONT]
        for number, body in enumerate(bodies, 1):
            if body is None:
                out += b"%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (
                    number, len(cls.CONTENT), cls.CONTENT)
            else:
                out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
        packed = zlib.compress(digits * (size // len(digits)), 9)
        for k in range(streams):
            out += (b"%d 0 obj\n<< /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode /Length %d >>\nstream\n"
                    % (100 + k, len(packed)) + packed + b"\nendstream\nendobj\n")
        out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
        return bytes(out)

    @classmethod
    def _lacking_pdf(cls, references, padding: int = 0, contents: bytes = None) -> bytes:
        """One page whose /Contents is an array of ``references`` (object numbers, generation 0) -- or
        ``contents`` -- and ``padding`` bytes of a stream nothing refers to that holds no "obj" (the reviewer's
        missing.py)."""
        array = b"[" + b"".join(b"%d 0 R " % number for number in references) + b"]"
        bodies = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                  cls.PAGE.replace(b"/Contents 3 0 R", b"/Contents " + (contents or array)), cls.FONT, None]
        out, offsets = bytearray(b"%PDF-1.7\n"), [0]
        for number, body in enumerate(bodies, 1):
            offsets.append(len(out))
            if body is None:
                data = (bytes(range(256)) * (padding // 256 + 1))[:padding]
                out += b"%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (number, len(data), data)
            else:
                out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
        table = len(out)
        out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets[1:])
        out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % table
        return bytes(out)

    @classmethod
    def _pages_pdf(cls, pages: int, shift: bytes = b"", end: bytes = b" \n", free: int = 0) -> bytes:
        """A classic-table PDF of ``pages`` pages, each "page N" in its own content, whose table lines end in
        ``end`` (b" \\r\\n": 21 bytes) and list ``free`` free entries past the objects; ``shift`` inserted after
        the header moves every object off the offset the table gives, as an ASCII-mode transfer does (the
        reviewer's gen_shift.py, insfix7.pdf)."""
        out, offsets, kids = bytearray(b"%PDF-1.4\n"), {}, []

        def obj(number, body):
            offsets[number] = len(out)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))
        for page in range(pages):
            content = b"BT /F1 12 Tf 20 100 Td (page %d) Tj ET" % (page + 1)
            obj(5 + 2 * page, b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
            obj(4 + 2 * page, cls.PAGE.replace(b"/Contents 3 0 R", b"/Contents %d 0 R" % (5 + 2 * page))
                .replace(b"/F1 4 0 R", b"/F1 3 0 R"))
            kids.append(4 + 2 * page)
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), pages))
        obj(3, cls.FONT)
        size = 4 + 2 * pages + free
        table = len(out)
        out += b"xref\n0 %d\n" % size + b"0000000000 65535 f" + end + b"".join(
            (b"%010d 00000 n" % offsets[n] if n in offsets else b"0000000000 65535 f") + end
            for n in range(1, size))
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, table)
        data = bytes(out)
        if shift:
            data = data[:9] + shift + data[9:]
            data = data.replace(b"startxref\n%d" % table, b"startxref\n%d" % (table + len(shift)))
        return data

    @classmethod
    def _hybrid_pdf(cls, pages: int) -> bytes:
        """A hybrid-reference PDF, as Word writes one: the pages in an object stream, which its /XRefStm lists
        (type 2) and its classic table lists as free."""
        out, xref = bytearray(b"%PDF-1.5\n"), {}

        def obj(number, body):
            xref[number] = len(out)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = [10 + 2 * p for p in range(pages)]
        obj(2, b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), pages))
        obj(3, cls.FONT)
        for p, kid in enumerate(kids):
            content = b"BT /F1 12 Tf 20 100 Td (page %d) Tj ET" % (p + 1)
            obj(kid + 1, b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
        body, head = bytearray(), bytearray()
        for kid in kids:
            head += b"%d %d " % (kid, len(body))
            body += cls.PAGE.replace(b"/Contents 3 0 R", b"/Contents %d 0 R" % (kid + 1)).replace(
                b"/F1 4 0 R", b"/F1 3 0 R") + b"\n"
        packed = zlib.compress(bytes(head) + bytes(body))
        stm = 10 + 2 * pages
        obj(stm, b"<< /Type /ObjStm /N %d /First %d /Filter /FlateDecode /Length %d >>\nstream\n" % (
            len(kids), len(head), len(packed)) + packed + b"\nendstream")
        size = stm + 2
        rows = zlib.compress(b"".join(bytes([2]) + stm.to_bytes(4, "big") + kids.index(n).to_bytes(2, "big")
                                      if n in kids else bytes(7) for n in range(size)))
        xs = len(out)
        xref[size - 1] = xs
        out += b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Filter /FlateDecode /Length %d >>\nstream\n" % (
            size - 1, size, len(rows)) + rows + b"\nendstream\nendobj\n"
        table = len(out)
        out += b"xref\n0 %d\n" % size + b"".join(
            b"%010d 00000 n \n" % xref[n] if n in xref else b"0000000000 65535 f \n" for n in range(size))
        out += b"trailer\n<< /Size %d /Root 1 0 R /XRefStm %d >>\nstartxref\n%d\n%%%%EOF\n" % (size, xs, table)
        return bytes(out)

    @classmethod
    def _sparse_pdf(cls, base: int) -> bytes:
        """One page whose objects are numbered from ``base`` on, its cross-reference stream listing every number
        from 0, all but those free (the reviewer's gen_sparse.py)."""
        out, offsets = bytearray(b"%PDF-1.7\n"), {}

        def obj(number, body):
            offsets[number] = len(out)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))
        obj(base, b"<< /Type /Catalog /Pages %d 0 R >>" % (base + 1))
        obj(base + 1, b"<< /Type /Pages /Kids [%d 0 R] /Count 1 >>" % (base + 2))
        obj(base + 2, cls.PAGE.replace(b"/Contents 3 0 R", b"/Contents %d 0 R" % (base + 3)).replace(
            b"/F1 4 0 R", b"/F1 %d 0 R" % (base + 4)))
        obj(base + 3, b"<< /Length %d >>\nstream\n%s\nendstream" % (len(cls.CONTENT), cls.CONTENT))
        obj(base + 4, cls.FONT)
        at = len(out)
        offsets[base + 5] = at
        rows = zlib.compress(b"".join(bytes([1]) + offsets[n].to_bytes(4, "big") + bytes(2) if n in offsets
                                      else bytes([0]) + bytes(4) + b"\xff\xff" for n in range(base + 6)), 9)
        out += b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root %d 0 R /Filter /FlateDecode /Length %d >>" \
               b"\nstream\n" % (base + 5, base + 6, base, len(rows)) + rows
        out += b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at
        return bytes(out)

    @classmethod
    def _annots_pdf(cls, count: int) -> bytes:
        """One page whose /Annots is an array of ``count`` references to one annotation (the reviewer's
        bigarr.py annots): pypdf reads it from the file a byte at a time once the reader is made."""
        page = cls.PAGE + b""
        page = page[:-2] + b"/Annots [" + b"5 0 R " * count + b"] >>"
        out, offsets = bytearray(b"%PDF-1.7\n"), [0]
        for number, body in enumerate([b"<< /Type /Catalog /Pages 2 0 R >>",
                                       b"<< /Type /Pages /Kids [6 0 R] /Count 1 >>", None, cls.FONT,
                                       b"<< /Type /Annot /Subtype /Link /Rect [0 0 1 1] >>", page], 1):
            offsets.append(len(out))
            if body is None:
                out += b"%d 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (
                    number, len(cls.CONTENT), cls.CONTENT)
            else:
                out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
        table = len(out)
        out += b"xref\n0 7\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets[1:])
        out += b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % table
        return bytes(out)

    def _opened(self, data: bytes):
        """(_with_pypdf of ``data``, the readers it opened)."""
        from unittest import mock
        from sift.extract import pdf
        readers, real = [], pdf._open

        def opening(*args, **kwargs):
            try:
                readers.append(real(*args, **kwargs))
            except BaseException as exc:
                readers.append(exc)
                raise
            return readers[-1]
        with mock.patch.object(pdf, "_open", opening):
            return pdf._with_pypdf(data), readers

    @staticmethod
    def _within(data: bytes, seconds: float):
        """(what reading ``data`` under a budget of ``seconds`` came to, how long it took)."""
        import time
        from sift import deadline
        from sift.extract import pdf
        started = time.monotonic()
        try:
            with deadline.limit(seconds):
                what = pdf._with_pypdf(data)
        except deadline.DeadlineExceeded as exc:
            what = exc
        return what, time.monotonic() - started

    # ---- F1: what the cross-reference sections list --------------------------------------------------------------

    def test_entries_of_a_cross_reference_stream_within_the_bound_are_all_read(self):
        """Happy path: a cross-reference stream's entries are read as pypdf reads them where they are within the
        document's entry bound, and nothing is said of them. Counted are the entries of objects in use pypdf
        makes, each once."""
        import io
        import pypdf
        data = self._entries_pdf(2, 50)
        with _PypdfLog():
            (text, meta, notes, _), readers = self._opened(data)
            alone = pypdf.PdfReader(io.BytesIO(data))
            _, table = self._opened(self._entries_pdf(2, 0))
        self.assertEqual((text, notes), ("[page 1]\nhello", []))
        self.assertEqual(len(alone.xref_objStm), 50)
        # The stream's 50 entries, and the table's 7 objects in use (its free line 0 is not counted).
        self.assertEqual(readers[0].structure.listed - table[0].structure.listed, 50)
        self.assertEqual(table[0].structure.listed, 7)

    def test_entries_a_cross_reference_stream_lists_past_the_bound_are_not_read(self):
        """12,000,000 entries of two bytes in a 2.9 MB file took 2,090 MB and 13.3 s: pypdf makes a
        dictionary entry of each. What the sections make pypdf hold is bounded by the file's size; the entries past
        it are not read, and the notes say so."""
        from sift.extract import pdf
        data = self._entries_pdf(2, 400_000)
        bound = pdf._entry_budget(len(data))
        self.assertLess(bound, 400_000)
        with _PypdfLog():
            (text, meta, notes, _), readers = self._opened(data)
        self.assertEqual(text, "[page 1]\nhello")
        self.assertEqual(readers[0].structure.listed, bound)
        self.assertIn(f"pypdf's cross-reference sections of this document list more than {bound} objects in use -- "
                      f"one a {pdf.XREF_FILE_BYTES_AN_ENTRY} bytes of it, {pdf.MAX_XREF_ENTRIES} at most --; those "
                      f"past that were not read", notes)

    def test_entries_at_the_bound_are_read_and_one_past_it_is_not(self):
        """Edge: the entry bound is inclusive. The newest section, the stream, is read first: at the bound its last
        entry is cut, and the classic table after it, which then starts at the bound, is not read -- pypdf finds
        its objects by searching the file. A classic table that crosses the bound ends the reader: it cannot be
        cut."""
        from unittest import mock
        from sift.extract import pdf
        data = self._entries_pdf(2, 50)
        with _PypdfLog():
            _, readers = self._opened(data)
        whole = readers[0].structure.listed
        for bound, text, listed in ((whole, "[page 1]\nhello", whole), (whole - 1, None, None),
                                    (50, "[page 1]\nhello", 50), (49, "[page 1]\nhello", 49)):
            with self.subTest(bound=bound), _PypdfLog(), mock.patch.object(pdf, "_entry_budget", lambda size: bound):
                (got, _, notes, _), readers = self._opened(data)
                self.assertEqual(got, text)
                said = f"list more than {bound} objects" in " ".join(notes)
                self.assertEqual(said, bound < whole, notes)
                if listed is not None:
                    self.assertEqual(readers[0].structure.listed, listed)

    def test_entries_at_offsets_that_hold_no_object_are_looked_at_within_the_bound(self):
        """399,000 type-1 entries in a 1 KB file ran 6.73 s past a 5 s budget and logged 399,000 records, 19.8 M
        characters: pypdf looks at every entry's offset while the reader is made, and logs each that
        holds no object. They are listed within the entry bound now."""
        import time
        from sift.extract import pdf
        data = self._entries_pdf(1, 399_000)
        started = time.monotonic()
        with _PypdfLog() as log:
            text, _, notes, _ = pdf._with_pypdf(data)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(text, "[page 1]\nhello")
        looked = [m for _, m in log.records if m.startswith("Ignoring wrong pointing object")]
        self.assertEqual(len(looked), pdf._entry_budget(len(data)))
        self.assertIn("list more than", " ".join(notes))

    def test_the_reader_is_made_within_the_time_budget(self):
        """Sad path: the entry bound aside, making the reader looks at the time budget -- as it looks at every
        entry's offset (type 1) and as it lists entries (type 2)."""
        from unittest import mock
        from sift import deadline
        from sift.extract import pdf
        for kind, count in ((1, 400_000), (2, 3_000_000)):
            with self.subTest(kind=kind), _PypdfLog(), mock.patch.object(pdf, "_entry_budget", lambda size: 10 ** 9), \
                    mock.patch.object(pdf, "_structure_budget", lambda size: 10 ** 10):
                what, took = self._within(self._entries_pdf(kind, count), 0.3)
                self.assertIsInstance(what, deadline.DeadlineExceeded)
                self.assertLess(took, 0.3 + 0.5, what)

    def test_what_pypdf_logs_while_the_reader_is_made_is_bounded(self):
        """Sad path: MAX_OP_LOG_PER_DOCUMENT was looked at only between pages, and 399,000 records were logged
        before the first. Past it the reader is not made, pypdf is not read with, and the notes say
        why."""
        from unittest import mock
        from sift.extract import pdf
        with _PypdfLog() as log, mock.patch.object(pdf, "_entry_budget", lambda size: 10 ** 9), \
                mock.patch.object(pdf, "MAX_OP_LOG_PER_DOCUMENT", 20_000):
            text, _, notes, _ = pdf._with_pypdf(self._entries_pdf(1, 20_000))
        self.assertIsNone(text)
        self.assertIn("pypdf logged more than 20000 characters opening this document", " ".join(notes))
        self.assertLess(log.characters(), 20_000 + 1024 * 120)

    def test_classic_tables_listing_past_the_bound_end_the_reader(self):
        """Sad path: a classic table is read a line of 20 bytes an entry, each counted; past the bound pypdf is
        not read with, and the notes say so."""
        from unittest import mock
        from sift.extract import pdf
        with _PypdfLog(), mock.patch.object(pdf, "_entry_budget", lambda size: 3):
            text, _, notes, _ = pdf._with_pypdf(self._entries_pdf(2, 0))
        self.assertIsNone(text)
        self.assertIn("list more than 3 objects", " ".join(notes))

    def test_the_entry_bound_is_derived_from_memory(self):
        """One entry of an object in use a XREF_FILE_BYTES_AN_ENTRY bytes of the file, at most MAX_XREF_ENTRIES:
        MAX_XREF_HELD, a quarter of the 1 GiB a host application's workers run in, of XREF_ENTRY_HELD each, the most an
        entry was measured to hold (244 bytes). The reviewer's largest legitimate file, 890,426 objects in 17.6 MB,
        19.7 bytes an object, the fewest of any, fits both."""
        from sift.extract import pdf
        self.assertEqual(pdf.MAX_XREF_ENTRIES, pdf.MAX_XREF_HELD // pdf.XREF_ENTRY_HELD)
        self.assertEqual(pdf.MAX_XREF_HELD, pdf.WORKER_MEMORY // 4)
        self.assertEqual(pdf.WORKER_MEMORY, 1 << 30)
        self.assertGreaterEqual(pdf.XREF_ENTRY_HELD, 244)
        self.assertEqual(pdf._entry_budget(10 ** 6), 10 ** 6 // pdf.XREF_FILE_BYTES_AN_ENTRY)
        self.assertEqual(pdf._entry_budget(10 ** 10), pdf.MAX_XREF_ENTRIES)
        self.assertGreaterEqual(pdf._entry_budget(17_583_030), 890_426)
        self.assertLessEqual(3 * pdf.XREF_FILE_BYTES_AN_ENTRY, 19.7)

    def test_what_an_entry_holds_is_within_what_is_charged(self):
        """What pypdf's reader holds of each entry it makes of an object in use -- a classic table's line, a
        stream's entry of type 1 and of type 2 -- is within XREF_ENTRY_HELD (traced: 99 to 184 bytes)."""
        import gc
        import io
        import tracemalloc
        import pypdf
        from sift.extract import pdf
        shapes = {"classic": lambda n: self._pages_pdf(n // 2),
                  "type 1": lambda n: self._streamed(n, 1),
                  "type 2": lambda n: self._streamed(n, 2)}
        for name, make in shapes.items():
            small, large = make(10), make(10_010)
            held = []
            for data in (small, large):
                gc.collect()
                tracemalloc.start()
                with _PypdfLog():
                    reader = pypdf.PdfReader(io.BytesIO(data))
                gc.collect()
                held.append((tracemalloc.get_traced_memory()[0] - len(data),
                             sum(len(v) for v in reader.xref.values()) + len(reader.xref_objStm)))
                tracemalloc.stop()
                del reader
            (few, n_few), (many, n_many) = held
            with self.subTest(name):
                self.assertGreater(n_many - n_few, 9_000)
                self.assertLess((many - few) / (n_many - n_few), pdf.XREF_ENTRY_HELD)

    @classmethod
    def _streamed(cls, count: int, kind: int) -> bytes:
        """``count`` objects in use listed by a cross-reference stream as entries of ``kind``: 1, "null" objects in
        the file; 2, in object streams of 100."""
        out, rows = bytearray(b"%PDF-1.7\n"), {}

        def obj(number, body):
            rows[number] = (1, len(out), 0)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [] /Count 0 >>")
        numbers = list(range(3, 3 + count))
        if kind == 1:
            for number in numbers:
                obj(number, b"null")
        else:
            for at in range(0, count, 100):
                chunk, stm = numbers[at:at + 100], 3 + count + at // 100
                head = b"".join(b"%d %d " % (n, 2 * j) for j, n in enumerate(chunk))
                packed = zlib.compress(head + b"0\n" * len(chunk))
                obj(stm, b"<< /Type /ObjStm /N %d /First %d /Filter /FlateDecode /Length %d >>\nstream\n" % (
                    len(chunk), len(head), len(packed)) + packed + b"\nendstream")
                for j, number in enumerate(chunk):
                    rows[number] = (2, stm, j)
        size = max(rows) + 2
        at = len(out)
        rows[size - 1] = (1, at, 0)
        packed = zlib.compress(PdfStructureTest._xref_rows(rows, size))
        out += (b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R /Filter /FlateDecode /Length %d >>\n"
                b"stream\n" % (size - 1, size, len(packed)) + packed + b"\nendstream\nendobj\nstartxref\n%d\n"
                b"%%%%EOF\n" % at)
        return bytes(out)

    # ---- a review ----------------------------------------------------------------------------------------------

    def test_a_classic_line_read_twice_is_counted_once(self):
        """A table whose lines end in " \\r\\n" (21 bytes) is read by pypdf a line and a re-read at a time, and
        0.1.58 counted both: cl21.pdf, 341,000 objects in 22 MB, went to the fallback and lost 34 % of its text. Each object in use is counted once, as pypdf enters it."""
        import io
        import pypdf
        counted = {}
        for end in (b" \n", b" \r\n", b"\r\n"):
            data = self._pages_pdf(20, end=end)
            with _PypdfLog():
                (text, _, notes, _), readers = self._opened(data)
                alone = pypdf.PdfReader(io.BytesIO(data))
                expected = "\n\n".join(f"[page {i}]\npage {i}" for i in range(1, 21))
            with self.subTest(end=end):
                self.assertEqual((text, notes), (expected, []))
                self.assertEqual(readers[0].structure.listed, sum(len(v) for v in alone.xref.values()))
                counted[end] = readers[0].structure.listed
        self.assertEqual({3 + 2 * 20}, set(counted.values()))

    def test_free_entries_are_not_counted(self):
        """sparse.pdf, one page numbered from 5,000 in a file of 844 bytes, its stream listing every number free
        but those, got a note in 0.1.58: 0.1.58 counted the free entries, which pypdf makes nothing of.
        A classic table's free lines are not counted either: each takes 20 bytes of the file."""
        from unittest import mock
        from sift.extract import pdf
        data = self._sparse_pdf(5000)
        self.assertLess(pdf._entry_budget(len(data)), 5000)
        with _PypdfLog():
            (text, _, notes, _), readers = self._opened(data)
        self.assertEqual((text, notes), ("[page 1]\nhello", []))
        self.assertEqual(readers[0].structure.listed, 6)
        data = self._pages_pdf(2, free=5000)
        with _PypdfLog(), mock.patch.object(pdf, "_entry_budget", lambda size: 7):
            (text, _, notes, _), readers = self._opened(data)
        self.assertEqual((text, notes), ("[page 1]\npage 1\n\n[page 2]\npage 2", []))
        self.assertEqual(readers[0].structure.listed, 7)

    def test_an_object_a_rebuild_finds_twice_is_counted_once(self):
        """Edge: a file updated in place holds an object twice; a table rebuilt of it enters the object's newer
        offset over the older, which holds nothing more, and is counted once: at a bound of the distinct objects,
        nothing is cut."""
        from unittest import mock
        from sift.extract import pdf
        data = self._rebuilt_pdf(0, 0)
        at = data.index(b"trailer")
        data = data[:at] + b"4 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (
            len(self.CONTENT), self.CONTENT) + b"5 0 obj\n%s\nendobj\n" % self.FONT + data[at:]
        for bound, cut in ((5, False), (4, True)):
            with self.subTest(bound=bound), _PypdfLog(), mock.patch.object(pdf, "_entry_budget", lambda size: bound):
                (text, _, notes, _), readers = self._opened(data)
                if not cut:
                    self.assertEqual((text, notes), ("[page 1]\nhello", []))
                self.assertEqual(readers[0].structure.listed, bound)
                self.assertEqual(readers[0].structure.listed_over, cut)

    def test_a_hybrid_files_objects_are_counted_once(self):
        """A hybrid-reference file lists its compressed objects as free in its table and as type 2 in its
        /XRefStm; 0.1.58 counted both, and a 780-page one read 735 pages."""
        import io
        import pypdf
        data = self._hybrid_pdf(30)
        with _PypdfLog() as log:
            (text, _, notes, _), readers = self._opened(data)
        with _PypdfLog() as alone_log:
            alone = pypdf.PdfReader(io.BytesIO(data))
            pages = "\n\n".join(f"[page {i + 1}]\n{page.extract_text()}" for i, page in enumerate(alone.pages))
        objects = {n for v in alone.xref.values() for n in v} | set(alone.xref_objStm)
        self.assertEqual((text, notes), (pages, []))
        self.assertEqual(readers[0].structure.listed, len(objects))
        self.assertEqual(log.records, alone_log.records)

    def test_objects_off_their_offsets_are_found_from_one_index(self):
        """insfix7.pdf, 2,000 pages whose objects all sit past where the table says: pypdf copies and searches the
        whole file for each, 0.1.55 read it in 14.4 s, 0.1.58 read 407 pages within MAX_SEARCHED.
        The document's index answers every search: every page is read, what pypdf logs is its own, and the file is
        not copied once -- and eight times the pages cost at most linear_bound(8) times the CPU, the two read in turn,
        where searching the file for each page is pages times the file. Five seconds of wall clock for 300 pages was
        the measure: 0.8 s of CPU on a quiet core (0.1.74), within the 6.4 times a CPU second varies on this host.
        A reading that made the index again for each search came out 29 to 41 times for 8x the pages, one that
        does not 9 to 11."""
        import io
        import pypdf
        shift = b"%" + b"x" * 37 + b"\n"

        def reading(pages):
            data = self._pages_pdf(pages, shift=shift)
            return lambda: self._opened(data)
        with _PypdfLog():
            ratio = cpu_ratio(reading(320), reading(40))
        self.assertLess(ratio, linear_bound(8), f"{ratio:.1f}x the CPU for 8x the pages")
        data = self._pages_pdf(320, shift=shift)
        with _PypdfLog() as alone_log:
            alone = pypdf.PdfReader(io.BytesIO(data))
            pages = "\n\n".join(f"[page {i + 1}]\n{page.extract_text()}" for i, page in enumerate(alone.pages))
        with _PypdfLog() as log:
            (text, _, notes, _), readers = self._opened(data)
        self.assertEqual((text, notes), (pages, []))
        self.assertIn("[page 320]\npage 320", text)
        self.assertEqual(log.records, alone_log.records)
        self.assertTrue(any("ref repaired" in m for _, m in log.records))
        self.assertEqual(readers[0].structure.searched, 0)

    def test_sifts_lookup_of_an_object_is_pypdfs(self):
        """What sift's get_object makes, caches and logs of an object -- where the table says, off it, missing,
        free, only in the file, numbered with a leading zero -- is what pypdf's own makes of it (_get_object)."""
        import io
        import pypdf
        from sift.extract import pdf
        content = b"BT /F1 12 Tf 20 100 Td (found) Tj ET"
        stored = b"9 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (len(content), content)
        base = self._lacking_pdf([], contents=b"[9 0 R 7 0 R 8 0 R 5 0 R 4 0 R 3 0 R]")
        at = base.index(b"\nxref\n") + 1
        # An object twice: pypdf's search finds the first. A number with a leading zero: its search finds no
        # "007 0 obj" for object 7.
        older = b"8 0 obj\n(older)\nendobj\n007 0 obj\n(zero)\nendobj\n7 0 obj\n(first)\nendobj\n"
        added = stored + b"8 0 obj\n(newer)\nendobj\n"
        head = base.index(b"1 0 obj")
        in_file = base[:head] + older + base[head:at] + added + base[at:]
        shapes = {
            "in the file only": in_file.replace(b"startxref\n%d" % at,
                                                b"startxref\n%d" % (at + len(older) + len(added))),
            "off its offset": self._pages_pdf(3, shift=b"%" + b"y" * 11 + b"\n"),
            "missing": self._lacking_pdf([777, 778, 777]),
            "free": self._pages_pdf(2, free=3).replace(b"/Kids [4 0 R 6 0 R]", b"/Kids [4 0 R 6 0 R 9 0 R]"),
        }

        def read(opener, data):
            with _PypdfLog() as log:
                reader = opener(data)
                texts = []
                for page in reader.pages:
                    texts.append(page.extract_text())
                made = (dict(reader.xref), dict(reader.xref_objStm), sorted(reader.trailer), texts,
                        sorted((k, re.sub(r"(IndirectObject\(\d+, \d+, )\d+", r"\1X", repr(v)))
                               for k, v in reader.resolved_objects.items()))
            return made, log.records
        for name, data in shapes.items():
            with self.subTest(name):
                structure = pdf._Structure(pdf._structure_budget(len(data)), pdf._entry_budget(len(data)))
                mine, theirs = read(lambda d: pdf._open(d, structure), data), \
                    read(lambda d: pypdf.PdfReader(io.BytesIO(d)), data)
                self.assertEqual(mine, theirs)
                self.assertEqual(structure.searched, 0)

    def test_a_page_read_from_the_file_stops_at_the_time_budget(self):
        """The /Annots of 4,000,000 references took 21 s past a 3 s budget, all resolving the first page: pypdf
        reads it from the file a byte at a time, and nothing watched a read once the reader was made.
        Every read of the file raises once the budget is spent (_Alarm)."""
        from sift import deadline
        from sift.extract import pdf
        data = self._annots_pdf(600_000)
        what, took = self._within(data, 0.3)
        self.assertIsInstance(what, deadline.DeadlineExceeded)
        self.assertLess(took, 0.3 + 0.5, what)
        with _PypdfLog():
            text, _, notes, _ = pdf._with_pypdf(self._annots_pdf(2_000))
        self.assertEqual((text, notes), ("[page 1]\nhello", []))

    def test_reads_of_the_file_cost_what_io_bytesios_do(self):
        """A read that looked at the clock made making cl20.pdf's reader take 3.09 to 3.79 s, 1.65 to 2.14 s
        unwatched: the file's reads are io.BytesIO's own until the budget is spent, and then raise."""
        import io
        from sift import deadline
        from sift.extract import pdf
        with deadline.limit(60), _PypdfLog():
            structure = pdf._Structure(10 ** 6, 10 ** 6)
            reader = pdf._open(self._lacking_pdf([]), structure)
            try:
                for stream in structure.streams:
                    self.assertIs(type(stream).read, io.BytesIO.read)
                self.assertIsNotNone(structure.alarm)
                structure.trip("deadline")
                for stream in structure.streams:
                    with self.assertRaises(pdf._ReaderBound):
                        stream.read(1)
            finally:
                structure.disarm()
                pdf._close(reader)
        with _PypdfLog():
            structure = pdf._Structure(10 ** 6, 10 ** 6)
            pdf._close(pdf._open(self._lacking_pdf([]), structure))
            self.assertIsNone(structure.alarm)

    def test_a_search_while_the_reader_is_made_looks_at_the_log_bound(self):
        """A copy of the whole file pypdf takes while the reader is made -- to recover a line of a table it cannot
        read -- is not taken once the log bound was passed: the reader is not made, and the file is not copied."""
        from sift.extract import pdf
        data = self._pages_pdf(2)
        data = data.replace(b"0000000000 65535 f \n", b"000000000x 65535 f \n", 1)
        structure = pdf._Structure(10 ** 6, 10 ** 6)
        structure.logged = pdf._LogCount()
        structure.logged.over = True
        with _PypdfLog(), self.assertRaises(pdf._Unread) as caught:
            pdf._open(data, structure)
        self.assertIn("pypdf logged more than", str(caught.exception))
        self.assertEqual(structure.searched, 0)
        structure = pdf._Structure(10 ** 6, 10 ** 6)
        with _PypdfLog():
            pdf._close(pdf._open(data, structure))
        self.assertEqual(structure.searched, 1)

    def test_the_search_bound_holds_its_derivation(self):
        """MAX_SEARCHED is about 1.6 s of searching at the 6.1 to 6.3 ms a MB measured."""
        from sift.extract import pdf
        self.assertLessEqual(pdf.MAX_SEARCHED / (1 << 20) * 6.3e-3, 1.7)
        self.assertGreaterEqual(pdf.MAX_SEARCHED / (1 << 20) * 6.1e-3, 1.5)

    def test_a_pypdf_older_than_the_floor_is_not_read_with(self):
        """pypdf 6.18.0 has all sift bounds pypdf with, and sift's tests fail on it: it was read with, silently. A pypdf older than PYPDF_FLOOR is not read with, and the notes say why."""
        import types
        from unittest import mock
        import pypdf
        from sift.extract import pdf
        for version, read in (("6.18.0", False), ("6.18.1rc1", False), ("6.17.9", False), (pdf.PYPDF_FLOOR, True),
                              ("6.19.0", True), ("7.0", True)):
            old = types.ModuleType("pypdf")
            old.__dict__.update({k: v for k, v in vars(pypdf).items() if not k.startswith("__")})
            old.__version__ = version
            with self.subTest(version=version), mock.patch.dict("sys.modules", {"pypdf": old}), \
                    mock.patch.dict(pdf._READERS, clear=True), _PypdfLog():
                text, _, notes, _ = pdf._with_pypdf(self._lacking_pdf([]))
                if read:
                    self.assertEqual((text, notes), ("[page 1]", []))
                else:
                    self.assertIsNone(text)
                    self.assertEqual(notes, [f"pypdf {version} is older than {pdf.PYPDF_FLOOR}, the oldest sift "
                                             f"reads with, which is tested with it; used fallback"])

    # ---- F2: object streams met rebuilding a broken table --------------------------------------------------------

    def test_object_streams_met_rebuilding_a_table_are_charged(self):
        """200 object streams of 790,000 bytes, a 174 KB file whose table is lost, took 44 s past a 5 s budget: the rebuild decoded each within what was left of the budget and charged none. Each is
        charged now; past the budget the rest are not decoded, and the page is still read."""
        from sift.extract import pdf
        data = self._rebuilt_pdf(20, 200_000)
        budget = pdf._structure_budget(len(data))
        self.assertLess(budget, 20 * 200_000)
        with _PypdfLog():
            ((text, _, notes, _), readers), peak = PdfStructureTest._peak(lambda: self._opened(data))
        self.assertEqual(text, "[page 1]\nhello")
        self.assertIn(f"decode to more than {budget} bytes", " ".join(notes))
        self.assertLess(peak, 3 * budget + (2 << 20), f"{peak} bytes held")

    def test_rebuilding_a_table_stops_within_the_time_budget(self):
        """Sad path: the budget and the entry bound aside, the rebuild looks at the time budget object by object and
        as it parses an object stream."""
        from unittest import mock
        from sift import deadline
        from sift.extract import pdf
        with _PypdfLog(), mock.patch.object(pdf, "_structure_budget", lambda size: 10 ** 10), \
                mock.patch.object(pdf, "_entry_budget", lambda size: 10 ** 9):
            what, took = self._within(self._rebuilt_pdf(40, 400_000), 0.3)
        self.assertIsInstance(what, deadline.DeadlineExceeded)
        self.assertLess(took, 0.3 + 0.5, what)

    def test_sifts_rebuild_of_a_table_is_pypdfs(self):
        """What sift's reader makes of a file whose table is lost -- its objects, those in object streams, its
        trailer -- and what it logs, is what pypdf's own makes of it."""
        import io
        import pypdf
        from sift.extract import pdf
        numbered = self._rebuilt_pdf(3, 64, digits=b"7 0 ")
        shapes = {"object streams": numbered, "none": self._rebuilt_pdf(0, 0),
                  "an object stream it cannot parse": self._rebuilt_pdf(1, 64, digits=b"x ")}

        def read(opener, data):
            with _PypdfLog() as log:
                reader = opener(data)
                made = (reader.xref, reader.xref_objStm, sorted(reader.trailer), reader.pages[0].extract_text())
            return made, log.records
        for name, data in shapes.items():
            with self.subTest(name):
                structure = pdf._Structure(pdf._structure_budget(len(data)), pdf._entry_budget(len(data)))
                self.assertEqual(read(lambda d: pdf._open(d, structure), data),
                                 read(lambda d: pypdf.PdfReader(io.BytesIO(d)), data))

    # ---- F3: references to objects the file lacks ----------------------------------------------------------------

    def test_references_to_one_object_the_file_lacks_search_the_file_once(self):
        """100 references to one object a 4.9 MB file lacks ran 9.81 s past a 5 s budget: pypdf copies
        and searches the whole file each time it resolves one, and sift resolved each three times. The document's
        index of "N G obj" answers now, made once, and the file is never copied; pypdf logs what it logs resolving
        the page's content once."""
        import io
        import pypdf
        data = self._lacking_pdf([999_999] * 200, padding=1_000_000)
        with _PypdfLog() as log:
            (text, _, notes, _), readers = self._opened(data)
        with _PypdfLog() as alone:
            pypdf.PdfReader(io.BytesIO(data)).pages[0].get_contents()
        self.assertEqual(readers[0].structure.searched, 0)
        self.assertIsNotNone(readers[0].structure.found)
        self.assertEqual([m for _, m in log.records if "999999" in m], [m for _, m in alone.records if "999999" in m])
        self.assertEqual(notes, [])

    def test_references_to_many_objects_the_file_lacks_search_it_within_a_bound(self):
        """Sad path: an object past what the index holds (_index_budget) is searched for in the file, within
        MAX_SEARCHED bytes in all; one looked for after that is taken as absent, and the notes say so."""
        from unittest import mock
        from sift.extract import pdf
        data = self._lacking_pdf(range(900_000, 900_200), padding=100_000)
        with _PypdfLog(), mock.patch.object(pdf, "_index_budget", lambda size: 2), \
                mock.patch.object(pdf, "MAX_SEARCHED", 5 * len(data)):
            (text, _, notes, _), readers = self._opened(data)
        structure = readers[0].structure
        upto = structure.found.upto
        self.assertLess(upto, len(data))
        searched = 5 * len(data) // (len(data) - upto)
        self.assertEqual(structure.searched, searched)
        self.assertEqual(text, "[page 1]")
        self.assertIn(f"pypdf searched this document's whole file {searched} times, {5 * len(data)} bytes at "
                      f"most, for objects its cross-reference tables lack; objects it looked for after that were "
                      f"taken as absent", notes)

    def test_searches_at_the_bound_are_made_and_one_past_it_is_not(self):
        """Edge: the search bound is inclusive; an object searched for and not found is not searched for again."""
        from unittest import mock
        from sift.extract import pdf
        for lacking, said in ((3, False), (4, True)):
            data = self._lacking_pdf(list(range(900_001, 900_001 + lacking)) * 2,
                                     padding=10_000 - 22 * lacking)
            with self.subTest(lacking=lacking), _PypdfLog(), mock.patch.object(pdf, "_index_budget", lambda size: 1):
                upto = pdf._Found(data, 1, lambda: None).upto
                with mock.patch.object(pdf, "MAX_SEARCHED", 3 * (len(data) - upto)):
                    (_, _, notes, _), readers = self._opened(data)
                self.assertEqual(readers[0].structure.searched, 3)
                self.assertEqual("whole file 3 times" in " ".join(notes), said, notes)

    def test_searching_for_objects_the_file_lacks_stops_within_the_time_budget(self):
        """Sad path: the index is made under the time budget, and each search past it looks at it."""
        from unittest import mock
        from sift import deadline
        from sift.extract import pdf
        filler = b"".join(b"\n%d 0 obj" % n for n in range(10, 700_000))
        data = self._lacking_pdf([5_000_000], contents=b"[5000000 0 R] " + b"% " + filler.replace(b"\n", b" "))
        with _PypdfLog():
            what, took = self._within(data, 0.3)
        self.assertIsInstance(what, deadline.DeadlineExceeded)
        self.assertLess(took, 0.3 + 0.5, what)
        with _PypdfLog(), mock.patch.object(pdf, "_index_budget", lambda size: 1), \
                mock.patch.object(pdf, "MAX_SEARCHED", 10 ** 12):
            what, took = self._within(self._lacking_pdf(range(900_000, 902_000), padding=2_000_000), 0.3)
        self.assertIsInstance(what, deadline.DeadlineExceeded)
        self.assertLess(took, 0.3 + 0.5, what)

    def test_an_object_the_tables_lack_that_the_file_holds_is_found(self):
        """Happy path: an object the table does not list but the file holds is found by searching, as pypdf finds
        it, and read."""
        content = b"BT /F1 12 Tf 20 100 Td (found) Tj ET"
        stored = b"9 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n" % (len(content), content)
        data = self._lacking_pdf([], contents=b"9 0 R")
        table = data.index(b"\nxref\n") + 1
        data = (data[:table] + stored + data[table:]).replace(b"startxref\n%d" % table,
                                                              b"startxref\n%d" % (table + len(stored)))
        from sift.extract import pdf
        with _PypdfLog():
            text, _, notes, _ = pdf._with_pypdf(data)
        self.assertEqual((text, notes), ("[page 1]\nfound", []))

    # ---- F4: the pypdf sift reads with ---------------------------------------------------------------------------

    def test_a_pypdf_without_what_sift_bounds_it_with_is_not_read_with(self):
        """A pypdf lacking the configuration API (6.17.0 and older) got the plain, unbounded PdfReader, silently. It is not read with now, and the notes say why."""
        import types
        from unittest import mock
        import pypdf
        from sift.extract import pdf
        old = types.ModuleType("pypdf")
        old.__version__, old.PdfReader = "6.14.2", pypdf.PdfReader
        with mock.patch.dict("sys.modules", {"pypdf": old}), mock.patch.dict(pdf._READERS, clear=True), \
                mock.patch.object(pypdf.PdfReader, "__init__", side_effect=AssertionError("read with")):
            text, meta, notes, _ = pdf._with_pypdf(self._lacking_pdf([]))
        self.assertIsNone(text)
        self.assertEqual(notes, [f"pypdf 6.14.2 lacks apply_configuration and get_configuration, with which sift "
                                 f"bounds what it reads; pypdf {pdf.PYPDF_FLOOR} and later have them; used fallback"])

    def test_the_pypdf_sift_needs_is_stated_alike_where_it_is_installed_from(self):
        """pyproject.toml said pypdf>=6.8 and that the tests passed on 6.8.0 to 6.19.0, while sift imports
        apply_configuration, which pypdf has since 6.18.0; its tests pass on 6.18.1 and 6.19.0."""
        from sift.extract import pdf
        root = Path(__file__).resolve().parent.parent
        pyproject = (root / "pyproject.toml").read_text()
        wanted = f'"pypdf>={pdf.PYPDF_FLOOR}"'
        self.assertEqual(re.findall(r'"pypdf[^"]*"', pyproject), [wanted, wanted])
        self.assertIn(f"pypdf>={pdf.PYPDF_FLOOR} ", (root / "requirements-optional.txt").read_text())
        self.assertIn(f"sift's tests pass against {pdf.PYPDF_FLOOR} and 6.19.0", " ".join(pyproject.split()))
        for readme in ("README.md", "README.de.md"):
            self.assertIn(f"pypdf {pdf.PYPDF_FLOOR} ", " ".join((root / readme).read_text().split()), readme)

    # ---- F5: a limit pypdf sets itself -----------------------------------------------------------------------------

    def test_a_limit_pypdf_sets_itself_is_not_blamed_on_the_budget(self):
        """An older cross-reference stream naming more /Columns than pypdf allows raised LimitReachedError inside
        sift's decoding, and sift spent the budget: the page's object stream was not read and pypdf failed, where
        pypdf alone logs "Previous trailer cannot be read" and reads the page."""
        import io
        import pypdf
        out, xref = bytearray(b"%PDF-1.7\n"), {}

        def obj(number, body):
            xref[number] = (1, len(out), 0)
            out.extend(b"%d 0 obj\n%s\nendobj\n" % (number, body))

        def stream(number, entries, data):
            xref[number] = (1, len(out), 0)
            out.extend(b"%d 0 obj\n<< %s /Length %d >>\nstream\n" % (number, entries, len(data)) + data
                       + b"\nendstream\nendobj\n")
        obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj(2, b"<< /Type /Pages /Kids [10 0 R] /Count 1 >>")
        stream(3, b"", self.CONTENT)
        obj(4, self.FONT)
        stream(11, b"/Type /ObjStm /N 1 /First 5 /Filter /FlateDecode", zlib.compress(b"10 0 " + self.PAGE + b"\n"))
        xref[10] = (2, 11, 0)
        old = len(out)
        bad = zlib.compress(b"\x00" * 64)
        out.extend(b"20 0 obj\n<< /Type /XRef /Size 21 /Index [20 1] /W [1 4 2] /Filter /FlateDecode "
                   b"/DecodeParms << /Predictor 12 /Columns 300000 >> /Length %d >>\nstream\n" % len(bad) + bad
                   + b"\nendstream\nendobj\n")
        at = len(out)
        xref[21] = (1, at, 0)
        rows = zlib.compress(PdfStructureTest._xref_rows(xref, 22))
        out.extend(b"21 0 obj\n<< /Type /XRef /Size 22 /W [1 4 2] /Root 1 0 R /Prev %d /Filter /FlateDecode "
                   b"/Length %d >>\nstream\n" % (old, len(rows)) + rows
                   + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % at)
        data = bytes(out)
        with _PypdfLog() as alone:
            self.assertEqual(pypdf.PdfReader(io.BytesIO(data)).pages[0].extract_text(), "hello")
        with _PypdfLog() as log:
            (text, _, notes, _), readers = self._opened(data)
        self.assertEqual((text, notes), ("[page 1]\nhello", []))
        self.assertFalse(readers[0].structure.over)
        self.assertEqual(log.records, alone.records)

    # ---- F6: what the mutants of 0.1.55 showed untested ------------------------------------------------------------

    def test_the_reader_is_let_go_of_without_the_collector(self):
        """What _close clears -- the validated root, the page map, the named destinations -- held the reader in a
        reference cycle: without it the reader outlived the document until a full collection. (pypdf's
        TextExtraction is a cycle of its own that holds the fonts of the page it read, and through them the reader,
        until a collection: these pages have none.)"""
        import gc
        import weakref
        from unittest import mock
        from sift.extract import pdf
        page = self.PAGE.replace(b"/Resources << /Font << /F1 4 0 R >> >>", b"/Resources << >>")
        data = PdfStructureTest._object_streams_pdf(2, bodies=[page, page])
        refs, real = [], pdf._open

        def opening(*args, **kwargs):
            reader = real(*args, **kwargs)
            refs.append(weakref.ref(reader))
            return reader
        pdf._with_pypdf(data)
        gc.collect()
        gc.disable()
        try:
            with mock.patch.object(pdf, "_open", opening):
                text, _, _, _ = pdf._with_pypdf(data)
            self.assertEqual(text, "[page 1]\n\n[page 2]")
            self.assertIsNone(refs[0]())
        finally:
            gc.enable()

    def test_what_a_cross_reference_stream_decodes_to_is_charged(self):
        """Every cross-reference stream is charged what it decoded to, whether or not it is near the budget."""
        (_, _, notes, _), readers = self._opened(PdfStructureTest._xref_streams_pdf(3, 5_000))
        self.assertEqual(notes, [])
        self.assertEqual(readers[0].structure.spent, 3 * 5_000)

    def test_an_object_streams_parse_looks_at_the_time_budget_as_it_reads(self):
        """Sad path: the read bound aside, an object stream's parse looks at the time budget every 1,024 reads."""
        from unittest import mock
        from sift import deadline
        from sift.extract import pdf
        data = PdfStructureTest._object_streams_pdf(300, padding=700_000, shared=True)
        with mock.patch.object(pdf, "OBJECT_STREAM_READS_A_BYTE", 10 ** 6):
            what, took = self._within(data, 0.3)
        self.assertIsInstance(what, deadline.DeadlineExceeded)
        self.assertLess(took, 0.3 + 0.5, what)

    # ---- F7 ------------------------------------------------------------------------------------------------------

    def test_the_object_stream_read_bound_is_derived_from_the_corpus(self):
        """OBJECT_STREAM_READS_A_BYTE is twice the most reads a byte any object stream of the 111 real documents
        took (1.92, soa.pdf), rounded up; its comment says so, and a test holds it."""
        import inspect
        from sift.extract import pdf
        source = inspect.getsource(pdf)
        end = source.index("\nOBJECT_STREAM_READS_A_BYTE = ")
        comment = source[source.rindex("\n\n", 0, end):end].replace("\n#: ", " ")
        self.assertIn("1.92 reads a byte at most", comment)
        self.assertGreaterEqual(pdf.OBJECT_STREAM_READS_A_BYTE, 2 * 1.92)

    def test_a_reader_is_opened_with_the_documents_budget(self):
        """_open made a budget of its own where none was handed, which only a test did."""
        import inspect
        from sift.extract import pdf
        self.assertTrue(all(parameter.default is inspect.Parameter.empty
                            for parameter in inspect.signature(pdf._open).parameters.values()))


class StructuredTest(unittest.TestCase):
    def _meta(self, name: str, body: str, max_bytes: int = 25_000_000):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(body, encoding="utf-8")
            return extract(p, detect(p), max_bytes).metadata

    def test_json_object_structure(self):
        meta = self._meta("t.json", '{"service": "api", "port": 8080, '
                                    '"tags": ["a", "b"]}')
        self.assertEqual(meta["top_level_type"], "dict")
        self.assertNotIn("parse_error", meta)

    def test_json_top_level_list(self):
        meta = self._meta("t.json", '[{"a": 1}, {"a": 2}]')
        self.assertEqual(meta["top_level_type"], "list")

    def test_malformed_json_is_reported_as_malformed(self):
        meta = self._meta("t.json", '{"a": ')
        self.assertIn("parse_error", meta)
        self.assertNotIn("parse_error_cause", meta,
                         "the file really is broken; do not excuse it")

    def test_our_own_truncation_is_not_blamed_on_the_document(self):
        body = '{"a": 1, "b": "' + "x" * 500 + '"}'
        meta = self._meta("t.json", body, max_bytes=100)
        self.assertIn("parse_error_cause", meta)
        self.assertIn("max_bytes", meta["parse_error_cause"])

    def test_jsonl_counts_records(self):
        meta = self._meta("t.jsonl", '{"a":1}\n{"a":2}\n{"a":3}\n')
        self.assertEqual(meta.get("records"), 3)

    def test_xml_with_an_encoding_declaration(self):
        meta = self._meta("t.xml", '<?xml version="1.0" encoding="utf-8"?>'
                                   "<root><item>x</item></root>")
        self.assertNotIn("parse_error", meta)

    def test_ini_sections(self):
        meta = self._meta("t.ini", "[server]\nhost = a\n[client]\nretries = 3\n")
        self.assertIn("sections", meta)


if __name__ == "__main__":
    unittest.main()


class PromptEconomyTest(unittest.TestCase):
    """The same fact must not reach the model more than once.

    A 371-byte JSON produced a 1,243-character prompt: its structure appeared
    as metadata, again as an outline, and a third time in the document itself.
    """

    def _extract(self, name: str, body: str):
        with tempfile.TemporaryDirectory() as d:
            p = Path(name := str(Path(d) / name))
            p.write_text(body, encoding="utf-8")
            return detect(p), extract(p, detect(p), 25_000_000)

    def test_a_small_json_is_not_prefixed_with_its_own_outline(self):
        det, ex = self._extract("t.json", '{"a": 1, "b": {"c": true}}')
        self.assertNotIn("# JSON structure", ex.text)
        self.assertIn('"c": true', ex.text)
        self.assertIn("structure", ex.metadata)      # still in the report
        self.assertIn("structure", ex.text_covers)   # but not in the prompt

    def test_a_truncated_json_keeps_its_outline(self):
        """Then the outline is the only view of the whole shape."""
        from sift.extract import structured
        big = '{"items": [' + ", ".join(
            f'{{"id": {i}, "name": "n{i}"}}' for i in range(9000)) + "]}"
        _, rendered = structured._json(big)
        self.assertIn("# JSON structure", rendered)

    def test_the_prompt_states_each_fact_once(self):
        from sift.llm import prompts
        det, ex = self._extract("t.json", '{"service": "api", "port": 8080}')
        built = prompts.build_user_prompt(
            filename="t.json", kind=det.kind, mime=det.mime,
            metadata=ex.metadata, content=ex.text, covered=ex.text_covers)
        self.assertEqual(built.count('"service"'), 1)
        self.assertEqual(built.count("8080"), 1)


class SqliteHonestyTest(unittest.TestCase):
    """A database that is partly unreadable has to say so."""

    def _db(self, name):
        path = Path(self.dir) / name
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE good (id INTEGER, name TEXT)")
        con.executemany("INSERT INTO good VALUES (?,?)", [(1, "anna"), (2, "bo")])
        con.execute("CREATE TABLE broken (id INTEGER, value TEXT)")
        con.executemany("INSERT INTO broken VALUES (?,?)",
                        [(i, f"v{i}") for i in range(500)])
        con.commit()
        con.close()
        return path

    def test_a_corrupt_table_is_not_reported_as_an_empty_one(self):
        path = self._db("half.db")
        raw = bytearray(path.read_bytes())
        raw[4096 * 3:4096 * 4] = b"\xff" * 4096       # one page destroyed
        path.write_bytes(bytes(raw))
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)
        said = " ".join(summ.notes + summ.warnings)
        self.assertIn("could not be read", said)
        self.assertIn("broken", said)

    def test_an_intact_database_says_nothing_of_the_kind(self):
        summ = summarize_path(self._db("whole.db"),
                              Config(no_llm=True, quiet=True, cache=False),
                              None, None)
        self.assertNotIn("could not be read",
                         " ".join(summ.notes + summ.warnings))

    def test_a_large_database_is_not_called_truncated(self):
        """sqlite3 reads the whole file; --max-bytes never applied to it."""
        path = Path(self.dir) / "big.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE log (id INTEGER, text TEXT)")
        con.executemany("INSERT INTO log VALUES (?,?)",
                        [(i, "x" * 200) for i in range(20_000)])
        con.commit()
        con.close()
        summ = summarize_path(path, Config(no_llm=True, quiet=True, cache=False,
                                           max_bytes=100_000), None, None)
        self.assertFalse(summ.truncated)
        self.assertEqual([], [n for n in summ.notes
                              if "read only the first" in n])
        self.assertEqual(20_000, summ.metadata["tables"][0]["rows"])

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


@unittest.skipUnless(hasattr(os, "geteuid") and os.geteuid() != 0,
                     "needs a user that a directory's permissions apply to")
class SqliteWalReadOnlyTest(unittest.TestCase):
    """A database in WAL mode, in a directory sift may not write to.

    A reader of a WAL database needs the -shm index beside it, and creates
    it when it is missing. On a read-only mount it cannot, so mode=ro fails
    — "attempt to write a readonly database" for a cleanly closed database,
    "unable to open database file" when a -wal is left without its -shm —
    and the whole file came back as unreadable. immutable=1 reads the main
    file and ignores the log, which is the honest fallback as long as it
    says what it left out.
    """

    NOTE = ("read without the write-ahead log; changes not yet checkpointed "
            "are not included")

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _lock(self, directory: Path) -> None:
        for entry in directory.iterdir():
            entry.chmod(0o444)
        directory.chmod(0o555)
        self.addCleanup(directory.chmod, 0o755)

    def _wal_db(self, name="app.db") -> Path:
        path = self.dir / name
        con = sqlite3.connect(path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE orders (id INTEGER, item TEXT)")
        con.executemany("INSERT INTO orders VALUES (?,?)",
                        [(1, "desk"), (2, "lamp"), (3, "chair")])
        con.commit()
        con.close()                  # the last close checkpoints the log
        return path

    def _extract(self, path: Path):
        return extract(path, detect(path), 25_000_000)

    def test_a_wal_database_on_a_read_only_directory_is_read(self):
        """Cleanly closed: the last close checkpointed everything and
        removed the log, so nothing was left out and the note that says
        something was would be false."""
        path = self._wal_db()
        self.assertEqual(["app.db"], sorted(p.name for p in self.dir.iterdir()))
        self._lock(self.dir)
        ex = self._extract(path)
        self.assertNotIn("parse_error", ex.metadata, ex.notes)
        self.assertEqual(3, ex.metadata["tables"][0]["rows"])
        self.assertNotIn(self.NOTE, ex.notes)
        self.assertEqual(["app.db"], sorted(p.name for p in self.dir.iterdir()))

    def test_an_empty_log_left_behind_is_no_reason_for_the_note(self):
        path = self._wal_db()
        Path(str(path) + "-wal").write_bytes(b"")
        self._lock(self.dir)
        ex = self._extract(path)
        self.assertEqual(3, ex.metadata["tables"][0]["rows"])
        self.assertNotIn(self.NOTE, ex.notes)

    def test_a_log_without_its_index_reads_only_what_was_checkpointed(self):
        """The -wal is there and the -shm is not: mode=ro cannot open it."""
        source = self.dir / "src"
        source.mkdir()
        writer = sqlite3.connect(source / "live.db")
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE settled (a)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("CREATE TABLE pending (a)")
        writer.commit()
        copy = self.dir / "copy"
        copy.mkdir()
        shutil.copy(source / "live.db", copy / "live.db")
        shutil.copy(source / "live.db-wal", copy / "live.db-wal")
        writer.close()
        self._lock(copy)
        ex = self._extract(copy / "live.db")
        self.assertEqual(["settled"],
                         [t["name"] for t in ex.metadata["tables"]])
        self.assertIn(self.NOTE, ex.notes)

    def test_a_writable_directory_needs_no_fallback(self):
        ex = self._extract(self._wal_db())
        self.assertEqual(3, ex.metadata["tables"][0]["rows"])
        self.assertNotIn(self.NOTE, ex.notes)

    def test_a_rollback_journal_database_is_not_retried(self):
        path = self.dir / "plain.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE t (a)")
        con.commit()
        con.close()
        self._lock(self.dir)
        ex = self._extract(path)
        self.assertEqual("t", ex.metadata["tables"][0]["name"])
        self.assertNotIn(self.NOTE, ex.notes)

    def test_a_hot_rollback_journal_is_never_read_past(self):
        """A rollback-journal database copied mid-transaction: mode=ro fails
        with the same "attempt to write a readonly database" a WAL one does,
        because the hot journal has to be rolled back first. immutable=1
        would read the half-written transaction — measured, 1,746 of 1,789
        rows already changed — so only the header's WAL bytes may license
        that retry."""
        from unittest import mock
        from sift.extract import sqlite_db
        source = self.dir / "src"
        source.mkdir()
        writer = sqlite3.connect(source / "j.db", isolation_level=None)
        writer.execute("PRAGMA journal_mode=DELETE")
        writer.execute("CREATE TABLE t (a, b)")
        writer.executemany("INSERT INTO t VALUES (?, ?)",
                           [(i, "x" * 200) for i in range(2000)])
        writer.execute("PRAGMA cache_size=1")     # spill changed pages
        writer.execute("BEGIN")
        writer.execute("UPDATE t SET b = 'changed' || b")
        copy = self.dir / "copy"
        copy.mkdir()
        shutil.copy(source / "j.db", copy / "j.db")
        shutil.copy(source / "j.db-journal", copy / "j.db-journal")
        writer.execute("ROLLBACK")
        writer.close()
        self.assertEqual(b"\x01\x01",
                         (copy / "j.db").read_bytes()[18:20])  # rollback mode
        self._lock(copy)
        with mock.patch.object(sqlite_db, "_attempt",
                               wraps=sqlite_db._attempt) as attempt:
            ex = self._extract(copy / "j.db")
        tried = [call.args[0] for call in attempt.call_args_list]
        self.assertEqual(1, len(tried), tried)
        self.assertNotIn("immutable", tried[0])
        self.assertIn("readonly", ex.metadata["parse_error"])
        self.assertNotIn(self.NOTE, ex.notes)

    def test_a_corrupt_file_with_a_wal_header_still_says_it_is_unreadable(self):
        """Both attempts fail; the second one's reason is the one that
        describes the file, and the note about the log would be false."""
        header = bytearray(b"SQLite format 3\x00" + b"\x00" * 200)
        header[16:18] = (4096).to_bytes(2, "big")
        header[18] = header[19] = 2              # write/read version: WAL
        path = self.dir / "broken.db"
        path.write_bytes(bytes(header))
        self._lock(self.dir)
        ex = self._extract(path)
        self.assertIn("file is not a database", ex.metadata["parse_error"])
        self.assertNotIn(self.NOTE, ex.notes)
        self.assertEqual("sqlite", ex.kind)


class RtfEncodingTest(unittest.TestCase):
    """A run of \\'hh escapes is one sequence, not a series of bytes."""

    CASES = {
        "utf8.rtf": (rb"{\rtf1\ansi\ansicpg65001\deff0 Gr\'c3\'bc\'c3\'9fe aus "
                     rb"M\'c3\'bcnchen. Betrag: 1.234,56 \'e2\'82\'ac\par}",
                     "utf-8", "Grüße aus München. Betrag: 1.234,56 €"),
        "cp1252.rtf": (rb"{\rtf1\ansi\ansicpg1252\deff0 Gr\'fc\'dfe aus "
                       rb"M\'fcnchen. 5 \'80\par}",
                       "cp1252", "Grüße aus München. 5 €"),
    }

    def test_multibyte_codepages_decode_to_the_words_in_the_file(self):
        for name, (raw, encoding, expected) in self.CASES.items():
            path = Path(self.dir) / name
            path.write_bytes(raw)
            ex = extract(path, detect(path), 10_000_000)
            self.assertEqual(encoding, ex.metadata.get("encoding"), name)
            self.assertEqual(expected, ex.text.strip(), name)

    def test_a_dbcs_document_is_not_mojibake_either(self):
        path = Path(self.dir) / "cp932.rtf"
        body = "".join(f"\\'{b:02x}" for b in "日本語の文書".encode("cp932"))
        path.write_bytes(f"{{\\rtf1\\ansi\\ansicpg932\\deff0 {body}\\par}}"
                         .encode("ascii"))
        ex = extract(path, detect(path), 10_000_000)
        self.assertEqual("日本語の文書", ex.text.strip())

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class RenderingIsDeclaredTest(unittest.TestCase):
    """A reader that strips the file's own form has to say it stripped it.

    The flag drives a note with two branches. The second one — "the markup
    was taken out here, and its absence says nothing about the file" — is
    what stops a model from reporting the absence as a finding, and three
    readers were not setting it: a real model warned that an .rtf file
    "contains only plain text without visible RTF markup", and earlier that
    a press release "lacks standard structural tags (<html>, <head>,
    <body>), resembling plain text" — of a file that has all three.

    Held as a list because the failure is per-reader and silent: nothing
    breaks when a new extractor forgets, and the cost only shows up in a
    model's output.
    """

    CASES = {
        "a.rtf": rb"{\rtf1\ansi\deff0 Gr\'fc\'dfe aus M\'fcnchen.\par}",
        "a.html": b"<html><head><title>Bericht</title></head><body>"
                  b"<p>Umsatz plus 12 Prozent.</p></body></html>",
        "a.csv": b"id,name\n1,anna\n2,bruno\n",
        "a.json": b'{"a": 1, "b": [2, 3]}',
        "a.eml": b"From: a@b\nTo: c@d\nSubject: x\n\nHallo.\n",
    }

    def test_readers_that_strip_the_original_form_declare_it(self):
        for name, raw in self.CASES.items():
            path = Path(self.dir) / name
            path.write_bytes(raw)
            ex = extract(path, detect(path), 10_000_000)
            self.assertTrue(ex.rendered, f"{name} does not declare a rendering")

    def test_and_plain_text_does_not_claim_to_be_one(self):
        """A .txt is its own bytes; saying otherwise would be the same lie."""
        path = Path(self.dir) / "a.txt"
        path.write_text("Rechnung 2026-014\nBetrag 1.240,00 EUR\n",
                        encoding="utf-8")
        ex = extract(path, detect(path), 10_000_000)
        self.assertFalse(ex.rendered)

    def test_the_note_says_the_absence_of_markup_means_nothing(self):
        from sift.llm.prompts import _rendering_note
        note = _rendering_note("Grüße aus München.")
        self.assertIn("removed by sift", note)
        self.assertIn("says nothing about", note)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class SingularTest(unittest.TestCase):
    """"1 headings", "1 slides", "1 printable strings".

    Each was written separately and each read as carelessness in the one
    place a reader looks first. Checked as a class rather than one at a
    time: a file of every kind, holding exactly one of whatever it counts.
    """

    WRONG = re.compile(r"\b1 (?!of\b)[a-z]+(?:s|es)\b", re.IGNORECASE)
    #: Words that are plural at one and correct that way.
    ALLOWED = ("1 less", "1 this", "1 its", "1 https")

    def _one_of_everything(self):
        d = Path(self.dir)
        (d / "one.csv").write_text("column\n1\n", encoding="utf-8")
        (d / "one.txt").write_text("one line\n", encoding="utf-8")
        (d / "one.log").write_text("2026-08-01 00:00:00 ERROR one line\n",
                                   encoding="utf-8")
        (d / "one.md").write_text("# One\n\nOne paragraph.\n", encoding="utf-8")
        (d / "one.yaml").write_text("a: 1\n", encoding="utf-8")
        (d / "one.ini").write_text("[a]\nb = 1\n", encoding="utf-8")
        (d / "one.rtf").write_bytes(rb"{\rtf1\ansi\deff0 One word.\par}")
        (d / "one.eml").write_text("From: a@b\nTo: c@d\nSubject: x\n\nOne.\n",
                                   encoding="utf-8")
        con = sqlite3.connect(d / "one.db")
        con.execute("CREATE TABLE t (a INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
        con.commit()
        con.close()
        with zipfile.ZipFile(d / "one.zip", "w") as archive:
            archive.writestr("a.txt", "x")
        return sorted(d.iterdir())

    def test_nothing_counts_one_of_something_in_the_plural(self):
        wrong = []
        for path in self._one_of_everything():
            summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                               cache=False), None, None)
            said = " ".join([summ.summary or ""] + (summ.key_points or [])
                            + (summ.notes or []) + (summ.warnings or []))
            wrong += [f"{path.name}: {m.group(0)}"
                      for m in self.WRONG.finditer(said)
                      if not m.group(0).lower().startswith(self.ALLOWED)]
        self.assertEqual([], wrong)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class SectionlessConfigTest(unittest.TestCase):
    """.env and .properties have no [sections]. That is the format.

    "INI/config file that could not be parsed: File contains no section
    headers" was sift describing the format it expected rather than the file
    it read — of a file whose keys were right there to list.
    """

    PARSED = {
        "app.env": ("DB_HOST=db.intern\nDB_PORT=5432\n"
                    "DB_PASSWORD=Sup3rGeheim!2026\nLOG_LEVEL=debug\n",
                    ["DB_HOST", "DB_PORT", "DB_PASSWORD", "LOG_LEVEL"]),
        "app.properties": ("jdbc.url=jdbc:pg://db/app\njdbc.user=svc\n"
                           "jdbc.password=Herbst2024!\npool.size=20\n",
                           ["jdbc.url", "jdbc.user", "jdbc.password",
                            "pool.size"]),
    }
    #: Still an honest failure. configparser reads a .pgpass line as one key
    #: with a four-field value and reports success, which is worse than the
    #: failure it replaces; redis.conf separates with spaces, not "=".
    NOT_PARSED = {
        "pgpass.conf": "db.intern:5432:buchhaltung:svc:Winterreise2026!\n",
        "redis.conf": "bind 127.0.0.1\nport 6379\nrequirepass Kaese!23\n",
    }

    def test_the_keys_are_listed_with_their_own_case(self):
        for name, (body, keys) in self.PARSED.items():
            path = Path(self.dir) / name
            path.write_text(body, encoding="utf-8")
            summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                               cache=False), None, None)
            self.assertNotIn("could not be parsed", summ.summary, name)
            for key in keys:
                self.assertIn(key, summ.summary + " ".join(summ.key_points),
                              f"{name}: {key}")

    def test_and_the_credentials_in_them_are_still_hidden(self):
        for name, (body, _) in self.PARSED.items():
            path = Path(self.dir) / name
            path.write_text(body, encoding="utf-8")
            summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                               cache=False), None, None)
            rendered = render_json([summ])
            for secret in ("Sup3rGeheim", "Herbst2024!"):
                self.assertNotIn(secret, rendered, name)

    def test_a_file_that_only_looks_section_less_still_says_so(self):
        for name, body in self.NOT_PARSED.items():
            path = Path(self.dir) / name
            path.write_text(body, encoding="utf-8")
            summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                               cache=False), None, None)
            self.assertIn("could not be parsed", summ.summary, name)

    def test_a_real_ini_is_unchanged(self):
        path = Path(self.dir) / "real.ini"
        path.write_text("[db]\nhost = db.intern\n[log]\nlevel = debug\n",
                        encoding="utf-8")
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)
        self.assertIn("2 sections: db, log", summ.summary)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class BodyPointsTest(unittest.TestCase):
    """A list in a mail body keeps its own order and loses its own bullet.

    The shipped example read "• - Tuesday: Meet your team", "• - Monday: IT
    setup" — a bullet on top of a bullet, and a week starting on Tuesday.
    Both from the same function: the renderer writes its own marker, and the
    ranking preferred lines containing digits, which "1:1 with your manager"
    has and "IT setup and account provisioning" does not.
    """

    MAIL = ("From: a@b.de\nTo: c@d.de\nSubject: Erste Woche\n\n"
            "Willkommen im Team. So sieht die erste Woche aus:\n\n"
            "- Montag: Einrichtung der Zugaenge und der Hardware\n"
            "- Dienstag: Vorstellung im Team und 1:1 mit der Leitung\n"
            "- Mittwoch: Produktschulung im Detail\n"
            "- Donnerstag: Sicherheit und Datenschutz\n"
            "- Freitag: Rueckblick und offene Fragen\n")

    def test_no_key_point_carries_its_own_bullet(self):
        summ = self._summarize(self.MAIL)
        for point in summ.key_points:
            self.assertFalse(point.lstrip().startswith(("-", "*", "•", "+")),
                             point)

    def test_the_list_stays_in_the_order_the_file_wrote_it(self):
        summ = self._summarize(self.MAIL)
        days = [p.split(":")[0] for p in summ.key_points
                if p.split(":")[0] in ("Montag", "Dienstag", "Mittwoch",
                                       "Donnerstag", "Freitag")]
        self.assertEqual(["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                          "Freitag"][:len(days)], days)

    def test_a_prose_line_with_a_figure_still_outranks_pleasantries(self):
        """The preference is not gone, it is only kept out of lists."""
        summ = self._summarize(
            "From: a@b.de\nTo: c@d.de\nSubject: Rechnung\n\n"
            "Ich hoffe, es geht Ihnen gut und der Sommer war erholsam.\n"
            "Der offene Betrag ist 1.240,00 EUR, faellig am 2026-09-01.\n"
            "Mit freundlichen Gruessen und bis bald in Muenchen.\n")
        self.assertIn("1.240,00", summ.key_points[0])

    def _summarize(self, body):
        path = Path(self.dir) / "m.eml"
        path.write_text(body, encoding="utf-8")
        return summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class OpenDocumentTest(unittest.TestCase):
    """odt as paragraphs, ods as tables, odp as slides.

    All three were read as one flat list of paragraphs, which is right for a
    text document and loses the shape of the other two: a spreadsheet came
    back as "OpenDocument spreadsheet, ~6 words" where the .xlsx beside it
    said "1 sheet: Quartal (3×2)".
    """

    def _odf(self, name, mime, content):
        path = Path(self.dir) / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", mime,
                             compress_type=zipfile.ZIP_STORED)
            archive.writestr(
                "META-INF/manifest.xml",
                '<?xml version="1.0"?><manifest:manifest xmlns:manifest='
                '"urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
                f'<manifest:file-entry manifest:full-path="/" '
                f'manifest:media-type="{mime}"/></manifest:manifest>')
            archive.writestr("content.xml", content)
        return summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)

    NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"')

    def test_a_spreadsheet_reports_its_tables(self):
        rows = "".join(
            "<table:table-row>"
            + "".join(f"<table:table-cell><text:p>{cell}</text:p>"
                      f"</table:table-cell>" for cell in row)
            + "</table:table-row>"
            for row in (("Posten", "Betrag"), ("Material", "312000"),
                        ("Montage", "168000")))
        summ = self._odf(
            "k.ods", "application/vnd.oasis.opendocument.spreadsheet",
            f'<?xml version="1.0"?><office:document-content {self.NS}>'
            f'<office:body><office:spreadsheet>'
            f'<table:table table:name="Kalkulation">{rows}</table:table>'
            f'</office:spreadsheet></office:body></office:document-content>')
        self.assertIn("Kalkulation", summ.summary)
        self.assertIn("1 sheet", summ.summary)
        self.assertEqual(1, summ.metadata["sheet_count"])
        self.assertEqual(2, summ.metadata["sheets"][0]["rows"])   # header off
        self.assertEqual(2, summ.metadata["sheets"][0]["columns"])

    def test_a_repeated_cell_does_not_inflate_the_row(self):
        """ODF collapses runs, and a saved sheet ends rows with 16384."""
        summ = self._odf(
            "r.ods", "application/vnd.oasis.opendocument.spreadsheet",
            f'<?xml version="1.0"?><office:document-content {self.NS}>'
            f'<office:body><office:spreadsheet><table:table table:name="T">'
            f'<table:table-row><table:table-cell><text:p>a</text:p>'
            f'</table:table-cell><table:table-cell '
            f'table:number-columns-repeated="16384"/></table:table-row>'
            f'</table:table></office:spreadsheet></office:body>'
            f'</office:document-content>')
        self.assertLessEqual(summ.metadata["sheets"][0]["columns"], 512)

    def test_a_presentation_counts_its_slides(self):
        pages = "".join(
            f'<draw:page draw:name="Folie {i}"><draw:frame><draw:text-box>'
            f'<text:p>Inhalt {i}</text:p></draw:text-box></draw:frame>'
            f'</draw:page>' for i in (1, 2, 3))
        summ = self._odf(
            "p.odp", "application/vnd.oasis.opendocument.presentation",
            f'<?xml version="1.0"?><office:document-content {self.NS}>'
            f'<office:body><office:presentation>{pages}'
            f'</office:presentation></office:body></office:document-content>')
        self.assertIn("3 slides", summ.summary)
        self.assertIn("Inhalt 1", " ".join(summ.key_points))

    def test_a_text_document_is_still_paragraphs(self):
        summ = self._odf(
            "t.odt", "application/vnd.oasis.opendocument.text",
            f'<?xml version="1.0"?><office:document-content {self.NS}>'
            f'<office:body><office:text>'
            f'<text:p>Rahmenvertrag zwischen Bauer GmbH und Keller AG</text:p>'
            f'<text:p>Laufzeit 24 Monate.</text:p>'
            f'</office:text></office:body></office:document-content>')
        self.assertIn("OpenDocument text", summ.summary)
        self.assertIn("Rahmenvertrag zwischen Bauer GmbH und Keller AG",
                      summ.key_points)

    def test_a_spreadsheet_with_no_tables_keeps_its_words(self):
        """Losing the file's text to gain its shape is not a trade."""
        summ = self._odf(
            "odd.ods", "application/vnd.oasis.opendocument.spreadsheet",
            f'<?xml version="1.0"?><office:document-content {self.NS}>'
            f'<office:body><office:spreadsheet>'
            f'<text:p>Umsatz 100</text:p>'
            f'</office:spreadsheet></office:body></office:document-content>')
        self.assertIn("Umsatz 100", " ".join(summ.key_points))

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class ReadTableTest(unittest.TestCase):
    """read_table pages through every row of csv, tsv, xlsx and ods.

    The rendering the model reads shows twelve rows a sheet and profiles at
    most 5000 records; a viewer that pages through a table needs all of
    them, under the same byte, member, column and time limits.
    """

    SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    REL = ("http://schemas.openxmlformats.org/officeDocument/2006/"
           "relationships")
    ODS_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
              'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
              'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"')

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    # -- builders ----------------------------------------------------------
    def _csv(self, name, rows, delimiter=","):
        path = self.dir / name
        path.write_text("\n".join(delimiter.join(r) for r in rows) + "\n",
                        encoding="utf-8")
        return path

    def _xlsx(self, sheets, name="book.xlsx"):
        """sheets: [(name, rows)]; every cell an inline string."""
        path = self.dir / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("xl/workbook.xml",
                       f'<workbook xmlns="{self.SS}" xmlns:r="{self.REL}">'
                       "<sheets>" + "".join(
                           f'<sheet name="{n}" r:id="rId{i}"/>'
                           for i, (n, _) in enumerate(sheets, 1))
                       + "</sheets></workbook>")
            z.writestr("xl/_rels/workbook.xml.rels",
                       '<Relationships xmlns="http://schemas.openxmlformats.org'
                       '/package/2006/relationships">' + "".join(
                           f'<Relationship Id="rId{i}" '
                           f'Target="worksheets/sheet{i}.xml"/>'
                           for i in range(1, len(sheets) + 1))
                       + "</Relationships>")
            for i, (_, rows) in enumerate(sheets, 1):
                body = "".join(
                    f'<row r="{r + 1}">' + "".join(
                        f'<c r="{chr(65 + c)}{r + 1}" t="inlineStr"><is><t>'
                        f"{value}</t></is></c>" for c, value in enumerate(row))
                    + "</row>" for r, row in enumerate(rows))
                z.writestr(f"xl/worksheets/sheet{i}.xml",
                           f'<worksheet xmlns="{self.SS}"><sheetData>{body}'
                           "</sheetData></worksheet>")
        return path

    def _ods(self, tables, name="book.ods"):
        mime = "application/vnd.oasis.opendocument.spreadsheet"
        body = "".join(
            f'<table:table table:name="{n}">' + "".join(
                "<table:table-row>" + "".join(
                    f"<table:table-cell><text:p>{c}</text:p></table:table-cell>"
                    for c in row) + "</table:table-row>" for row in rows)
            + "</table:table>" for n, rows in tables)
        path = self.dir / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("mimetype", mime, compress_type=zipfile.ZIP_STORED)
            z.writestr("content.xml",
                       f'<?xml version="1.0"?><office:document-content '
                       f'{self.ODS_NS}><office:body><office:spreadsheet>'
                       f"{body}</office:spreadsheet></office:body>"
                       f"</office:document-content>")
        return path

    def _page(self, path, **kw):
        from sift.extract import read_table
        return read_table(path, detect(path), **kw)

    @staticmethod
    def _numbered(count, start=1):
        return [[str(i), f"name{i}"] for i in range(start, start + count)]

    # -- happy -------------------------------------------------------------
    def test_a_csv_pages_past_the_rows_the_extractor_profiles(self):
        path = self._csv("big.csv", [["id", "name"]] + self._numbered(6000))
        first = self._page(path, offset=0, limit=100)
        self.assertEqual(["id", "name"], first.header)
        self.assertEqual(["1", "name1"], first.rows[0])
        self.assertEqual(100, len(first.rows))
        # Read whole (no more than 1 MiB), the rows after the page are
        # counted for the total, as 0.1.55 counted them.
        self.assertEqual(6000, first.total)
        self.assertTrue(first.has_more)
        self.assertFalse(first.truncated)
        self.assertEqual([], first.sheets)
        self.assertIsNone(first.sheet)
        last = self._page(path, offset=5990, limit=100)
        self.assertEqual([[str(i), f"name{i}"] for i in range(5991, 6001)],
                         last.rows)
        self.assertEqual(5990, last.offset)
        self.assertFalse(last.has_more)
        self.assertEqual(6000, last.total)

    def test_a_tsv_splits_on_tabs(self):
        path = self._csv("t.tsv", [["a", "b, c"], ["1", "x, y"], ["2", "z"]],
                         delimiter="\t")
        page = self._page(path)
        self.assertEqual("tsv", page.kind)
        self.assertEqual(["a", "b, c"], page.header)
        self.assertEqual([["1", "x, y"], ["2", "z"]], page.rows)

    def test_an_xlsx_sheet_pages_past_the_sheet_row_cap(self):
        path = self._xlsx([("First", [["k", "v"], ["a", "1"]]),
                           ("Second", [["id", "name"]] + self._numbered(450))])
        page = self._page(path, sheet="Second", offset=400, limit=100)
        self.assertEqual(["First", "Second"], page.sheets)
        self.assertEqual("Second", page.sheet)
        self.assertEqual(["id", "name"], page.header)
        self.assertEqual(self._numbered(50, start=401), page.rows)
        self.assertEqual(450, page.total)
        self.assertFalse(page.has_more)
        by_index = self._page(path, sheet=1, offset=0, limit=2)
        self.assertEqual("Second", by_index.sheet)
        self.assertEqual(self._numbered(2), by_index.rows)
        default = self._page(path)
        self.assertEqual("First", default.sheet)
        self.assertEqual([["a", "1"]], default.rows)

    def test_an_ods_table_pages_past_its_row_cap(self):
        path = self._ods([("Kalkulation",
                           [["Posten", "Betrag"]]
                           + [[f"p{i}", str(i)] for i in range(1, 301)])])
        page = self._page(path, offset=250, limit=100)
        self.assertEqual(["Kalkulation"], page.sheets)
        self.assertEqual(["Posten", "Betrag"], page.header)
        self.assertEqual(300, page.total)
        self.assertEqual(50, len(page.rows))
        self.assertEqual(["p251", "251"], page.rows[0])

    def test_the_total_is_the_extractors_row_count(self):
        """One word, one meaning: rows are records here too."""
        table = [["name", "dept"], ["Anna", "Eng"], ["Bob", "Ops"],
                 ["Cara", "Fin"]]
        for path in (self._csv("t.csv", table), self._xlsx([("S", table)]),
                     self._ods([("S", table)])):
            with self.subTest(path.suffix):
                meta = extract(path, detect(path), 25_000_000).metadata
                rows = meta.get("rows", (meta.get("sheets") or [{}])[0].get(
                    "rows"))
                self.assertEqual(rows, self._page(path).total)

    def test_a_csv_repeats_the_extractors_reading_of_the_file(self):
        """Preamble lines, a repeated header and trailing ",," lines are
        not records to the extractor, and are not rows here either."""
        path = self.dir / "report.csv"
        path.write_text("Monthly report\nid,name,city\n1,Ann,Rome\n"
                        "id,name,city\n2,Bob,Oslo\n,,\n,,\n", encoding="utf-8")
        page = self._page(path)
        self.assertEqual(["Monthly report"], page.preamble)
        self.assertEqual(["id", "name", "city"], page.header)
        self.assertEqual([["1", "Ann", "Rome"], ["2", "Bob", "Oslo"]],
                         page.rows)
        self.assertEqual(extract(path, detect(path), 25_000_000)
                         .metadata["rows"], page.total)

    # -- sad ---------------------------------------------------------------
    def test_arguments_that_cannot_name_a_page_are_refused(self):
        path = self._csv("t.csv", [["a"], ["1"]])
        for bad in ({"offset": -1}, {"limit": 0}, {"offset": "3"},
                    {"limit": 2.5}):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    self._page(path, **bad)
        text = self.dir / "notes.txt"
        text.write_text("just prose\n")
        with self.assertRaises(ValueError):
            self._page(text)

    def test_an_unknown_sheet_names_the_ones_there_are(self):
        path = self._xlsx([("Alpha", [["a"]]), ("Beta", [["b"]])])
        for sheet in ("Gamma", 2, -1):
            with self.subTest(sheet):
                with self.assertRaises(ValueError) as caught:
                    self._page(path, sheet=sheet)
                self.assertIn("Alpha", str(caught.exception))
        with self.assertRaises(ValueError):
            self._page(self._csv("t.csv", [["a"], ["1"]]), sheet="Alpha")

    def test_a_broken_spreadsheet_is_an_error_not_an_exception(self):
        from dataclasses import replace
        from sift.extract import read_table
        fake = self.dir / "fake.xlsx"
        fake.write_bytes(b"PK\x03\x04 this is not a zip at all")
        # Detection calls this an archive; a caller that trusts the
        # extension must still get an answer, not a traceback.
        page = read_table(fake, replace(detect(fake), kind="xlsx"))
        self.assertIn("could not read the spreadsheet", page.error)
        self.assertEqual([], page.rows)
        malformed = self._xlsx([("S", [["a"], ["1"]])], name="bad.xlsx")
        with zipfile.ZipFile(malformed, "a") as z:
            z.writestr("xl/worksheets/sheet1.xml",
                       f'<worksheet xmlns="{self.SS}"><sheetData><row>')
        self.assertEqual("xlsx", detect(malformed).kind)
        page = self._page(malformed)
        self.assertTrue(page.error)
        self.assertEqual([], page.rows)
        broken = self.dir / "broken.ods"
        with zipfile.ZipFile(broken, "w") as z:
            z.writestr("mimetype",
                       "application/vnd.oasis.opendocument.spreadsheet")
            z.writestr("content.xml", "<office:document-content><unclosed>")
        page = self._page(broken)
        self.assertTrue(page.error)
        self.assertEqual([], page.rows)

    def test_a_corrupt_deflate_stream_is_an_error_not_an_exception(self):
        """zipfile hands zlib's own error through for a damaged member, and
        it is none of the types the reader caught."""
        path = self._xlsx([("S", [["a"], ["1"]])], name="deflate.xlsx")
        data = bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("xl/worksheets/sheet1.xml")
        self.assertEqual(zipfile.ZIP_STORED, info.compress_type)
        # Rewritten deflated, then the compressed bytes damaged.
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, body in members.items():
                z.writestr(name, body)
        data = bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("xl/worksheets/sheet1.xml")
        start = (info.header_offset + 30 + len(info.filename.encode())
                 + len(info.extra))
        for index in range(start + 2, start + 12):
            data[index] ^= 0xFF
        path.write_bytes(bytes(data))
        page = self._page(path)
        self.assertTrue(page.error)
        self.assertEqual([], page.rows)

    def test_a_cell_reference_with_a_letter_outside_a_to_z(self):
        """"ß".upper() is "SS", two characters, and ord() refused it: the
        page and the whole extraction of the workbook failed on one cell."""
        path = self._xlsx([("S", [["a", "b"], ["1", "2"]])], name="ref.xlsx")
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        members["xl/worksheets/sheet1.xml"] = members[
            "xl/worksheets/sheet1.xml"].replace(b'r="A1"',
                                                'r="ß1"'.encode())
        with zipfile.ZipFile(path, "w") as z:
            for name, body in members.items():
                z.writestr(name, body)
        page = self._page(path)
        self.assertIsNone(page.error)
        self.assertEqual([["1", "2"]], page.rows)
        from sift.extract import extract
        ex = extract(path, detect(path), 10_000_000)
        self.assertFalse([n for n in ex.notes if "failed" in n], ex.notes)
        from sift.extract.office import _column_index
        self.assertEqual(0, _column_index("ß1"))
        self.assertEqual(3, _column_index("D4"))
        self.assertEqual(27, _column_index("AB9"))

    def test_a_missing_file_is_an_error_for_every_kind(self):
        """The csv reader let FileNotFoundError out while the xlsx reader
        set page.error: one contract, two behaviours."""
        from dataclasses import replace
        from sift.extract import read_table
        for name, make in (
                ("gone.csv", lambda p: self._csv(p, [["a"], ["1"]])),
                ("gone.tsv", lambda p: self._csv(p, [["a", "b"], ["1", "2"]],
                                                 delimiter="\t")),
                ("gone.xlsx", lambda p: self._xlsx([("S", [["a"]])], name=p)),
                ("gone.ods", lambda p: self._ods([("S", [["a"]])], name=p))):
            with self.subTest(name):
                path = make(name)
                det = detect(path)
                path.unlink()
                page = read_table(path, det)
                self.assertTrue(page.error)
                self.assertIn("No such file", page.error)
                self.assertEqual([], page.rows)

    def test_an_unexpected_failure_inside_a_reader_is_an_error(self):
        """The boundary, not a list of types: the next unexpected exception
        from a parser has not been written yet."""
        from unittest import mock
        from sift.extract import office, read_table, tabular
        xlsx = self._xlsx([("S", [["a"], ["1"]])], name="boom.xlsx")
        csv_path = self._csv("boom.csv", [["a"], ["1"]])
        for module, name, path in ((office, "_page_of_sheet", xlsx),
                                   (tabular, "fill_page", csv_path)):
            with self.subTest(name):
                with mock.patch.object(module, name,
                                       side_effect=TypeError("boom")):
                    page = read_table(path, detect(path))
                self.assertIn("boom", page.error)
                self.assertEqual([], page.rows)

    def test_an_ods_row_repeated_past_the_cap_is_not_a_complete_total(self):
        """ODF writes a million identical rows as one row repeated a million
        times. Expanding it is capped, and the page said "200 rows, all
        read" of a table the file says holds a million and one."""
        path = self.dir / "repeat.ods"
        body = ('<table:table table:name="S"><table:table-row>'
                "<table:table-cell><text:p>h</text:p></table:table-cell>"
                "</table:table-row>"
                '<table:table-row table:number-rows-repeated="1000000">'
                "<table:table-cell><text:p>same</text:p></table:table-cell>"
                "</table:table-row></table:table>")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("mimetype",
                       "application/vnd.oasis.opendocument.spreadsheet",
                       compress_type=zipfile.ZIP_STORED)
            z.writestr("content.xml",
                       f'<?xml version="1.0"?><office:document-content '
                       f'{self.ODS_NS}><office:body><office:spreadsheet>'
                       f"{body}</office:spreadsheet></office:body>"
                       f"</office:document-content>")
        page = self._page(path, limit=10)
        self.assertEqual(10, len(page.rows))
        self.assertTrue(page.truncated)
        self.assertIsNone(page.total)
        self.assertTrue(any("1000000" in n for n in page.notes), page.notes)
        small = self._ods([("S", [["h"], ["a"], ["b"]])], name="plain.ods")
        page = self._page(small)
        self.assertFalse(page.truncated)
        self.assertEqual([], page.notes)

    def test_columns_truncated_speaks_for_the_rows_on_the_page(self):
        """A row wider than the column cap far down the sheet set the flag
        on every page, including the pages it is not on."""
        from sift.extract.office import MAX_RENDER_COLUMNS
        rows = [["id", "name"]] + self._numbered(30)
        path = self._xlsx([("S", rows)], name="wide.xlsx")
        with zipfile.ZipFile(path) as z:
            members = {n: z.read(n) for n in z.namelist()}
        wide = "".join(
            f'<c r="{self._ref(col)}25" t="inlineStr"><is><t>v{col}</t></is>'
            f"</c>" for col in range(MAX_RENDER_COLUMNS + 5))
        sheet = members["xl/worksheets/sheet1.xml"].decode()
        start = sheet.index('<row r="25">')
        end = sheet.index("</row>", start) + len("</row>")
        members["xl/worksheets/sheet1.xml"] = (
            sheet[:start] + f'<row r="25">{wide}</row>' + sheet[end:]).encode()
        with zipfile.ZipFile(path, "w") as z:
            for name, body in members.items():
                z.writestr(name, body)
        first = self._page(path, offset=0, limit=5)
        self.assertEqual(5, len(first.rows))
        self.assertFalse(first.columns_truncated)
        with_it = self._page(path, offset=20, limit=10)
        self.assertTrue(with_it.columns_truncated)

    @staticmethod
    def _ref(col: int) -> str:
        from sift.extract.office import _column_name
        return _column_name(col)

    def test_a_csv_that_stops_parsing_keeps_the_rows_before(self):
        path = self.dir / "cr.csv"
        path.write_text("a,b\n1,x\ry,2\n3,4\n", encoding="utf-8")
        page = self._page(path)
        self.assertTrue(page.truncated)
        self.assertIsNone(page.total)
        self.assertTrue(any("not valid CSV" in n for n in page.notes),
                        page.notes)

    # -- edge --------------------------------------------------------------
    def test_an_empty_file_is_an_empty_table(self):
        from dataclasses import replace
        from sift.extract import read_table
        blank = self.dir / "blank.csv"
        blank.write_text("\n\n  \n")
        empty = self.dir / "empty.csv"
        empty.write_text("")
        # Zero bytes is kind "empty" to detection, and not a table to page.
        with self.assertRaises(ValueError):
            self._page(empty)
        for page in (self._page(blank),
                     read_table(empty, replace(detect(empty), kind="csv"))):
            self.assertEqual(([], [], 0, False, None),
                             (page.header, page.rows, page.total,
                              page.has_more, page.error))
        page = self._page(self._xlsx([("Empty", [])]))
        self.assertEqual(0, page.total)
        self.assertEqual([], page.rows)

    def test_an_offset_past_the_end_is_an_empty_page(self):
        path = self._csv("t.csv", [["id", "name"]] + self._numbered(5))
        page = self._page(path, offset=50, limit=10)
        self.assertEqual([], page.rows)
        self.assertEqual(5, page.total)
        self.assertFalse(page.has_more)

    def test_a_row_wider_than_the_column_cap_is_cut_and_says_so(self):
        from sift.extract.tabular import _MAX_COLUMNS
        path = self._csv("wide.csv", [[f"c{i}" for i in range(2000)],
                                      [str(i) for i in range(2000)]])
        page = self._page(path)
        self.assertTrue(page.columns_truncated)
        self.assertEqual(_MAX_COLUMNS, max(len(r) for r in page.rows))
        self.assertEqual(_MAX_COLUMNS, len(page.header))

    def test_a_huge_cell_is_clipped_to_the_page_budget(self):
        from sift.extract import MAX_PAGE_CHARS
        path = self._csv("fat.csv", [["id", "blob"], ["1", "x" * 1_000_000],
                                     ["2", "small"]])
        page = self._page(path, limit=10)
        self.assertTrue(page.cells_clipped)
        self.assertEqual(1, len(page.rows), "the page ends at its budget")
        self.assertLessEqual(sum(len(c) for c in page.rows[0]),
                             MAX_PAGE_CHARS + 1)
        self.assertTrue(page.has_more)
        after = self._page(path, offset=1, limit=10)
        self.assertEqual([["2", "small"]], after.rows)
        named = self._csv("named.csv", [["id", "n" * 1_000_000], ["1", "a"]])
        page = self._page(named)
        self.assertTrue(page.cells_clipped)
        self.assertLessEqual(sum(len(c) for c in page.header),
                             MAX_PAGE_CHARS + 1)
        # A header the size of the page left no room, and every row under
        # it came back as "…" cells.
        self.assertEqual([["1", "a"]], page.rows)
        self.assertLessEqual(sum(len(c) for c in page.header),
                             MAX_PAGE_CHARS // 2 + 1)

    def test_a_page_is_never_longer_than_the_page_cap(self):
        from sift.extract import MAX_PAGE_ROWS
        path = self._csv("t.csv", [["id", "name"]]
                         + self._numbered(MAX_PAGE_ROWS + 50))
        page = self._page(path, limit=MAX_PAGE_ROWS * 10)
        self.assertEqual(MAX_PAGE_ROWS, len(page.rows))
        self.assertTrue(page.has_more)

    def test_a_file_past_max_bytes_has_no_known_total(self):
        path = self._csv("t.csv", [["id", "name"]] + self._numbered(1000))
        page = self._page(path, offset=0, limit=5000, max_bytes=2000)
        self.assertTrue(page.truncated)
        self.assertIsNone(page.total)
        self.assertFalse(page.has_more)
        self.assertTrue(page.rows)
        self.assertLess(len(page.rows), 1000)

    def test_the_time_budget_applies(self):
        from sift.deadline import DeadlineExceeded
        path = self._csv("t.csv", [["id", "name"]] + self._numbered(50))
        with self.assertRaises(DeadlineExceeded):
            self._page(path, time_budget=1e-9)

    def test_the_rows_after_the_page_are_not_read_for_a_total(self):
        """0.1.55 converted and counted every row after the page for the
        total -- 2.4 s a page of a 200,000-row sheet. A sheet held whole
        (no more than _WHOLE_MEMBER) has its rows after the page counted,
        never converted; one read as a stream stops at the row
        that says there is another, and its total is unknown."""
        from unittest import mock
        from sift.extract import office
        path = self._xlsx([("S", [["id", "name"]] + self._numbered(3000))])
        for whole, total in ((office._WHOLE_MEMBER, 3000), (0, None)):
            calls = []
            real = office._sheet_row

            def counted(*args, **kw):
                calls.append(1)
                return real(*args, **kw)
            with self.subTest(whole=whole), \
                    mock.patch.object(office, "_sheet_row", counted), \
                    mock.patch.object(office, "_WHOLE_MEMBER", whole):
                page = self._page(path, offset=0, limit=10)
                self.assertEqual(self._numbered(10), page.rows)
                self.assertEqual(total, page.total)
                self.assertTrue(page.has_more)
                self.assertEqual([], page.notes)
                self.assertLessEqual(len(calls), 9 + 11)


class SheetCountsAndNamespacesTest(unittest.TestCase):
    """What a sheet's cells and counts say, read as the file's program
    reads them.

    An xlsx shared-string index went through str.isdigit(), which takes
    "²", and then int(), which refuses it: read_sheet_page raised a
    ValueError past the 200 rows extract() reads. An ods row or cell of any
    namespace was read as the table's, and a repeat count went through
    int(), which reads "1_0" as 10. LibreOffice 25.2 (the host application oracle,
    2026-09-27) ignores a table:table-row, table:table-cell or table:table
    of another namespace and a number-rows-repeated of another namespace,
    and reads a count as white space, an optional "+", and the ASCII
    digits up to the first other character: "1_0" is 1, "3x" 3, " 3 " 3,
    "+3" 3; "²", "٣", "0", "-2" and eleven nines are 1.
    """

    SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    REL = ("http://schemas.openxmlformats.org/officeDocument/2006/"
           "relationships")
    ODS_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
              'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
              'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
              'xmlns:foo="urn:example:foo"')

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _xlsx(self, rows, name="s.xlsx"):
        path = self.dir / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("xl/workbook.xml",
                       f'<workbook xmlns="{self.SS}" xmlns:r="{self.REL}">'
                       '<sheets><sheet name="S" r:id="rId1"/></sheets>'
                       "</workbook>")
            z.writestr("xl/_rels/workbook.xml.rels",
                       '<Relationships xmlns="http://schemas.openxmlformats.'
                       'org/package/2006/relationships"><Relationship '
                       'Id="rId1" Target="worksheets/sheet1.xml"/>'
                       "</Relationships>")
            z.writestr("xl/sharedStrings.xml",
                       f'<sst xmlns="{self.SS}"><si><t>zero</t></si>'
                       "<si><t>one</t></si></sst>")
            z.writestr("xl/worksheets/sheet1.xml",
                       f'<worksheet xmlns="{self.SS}"><sheetData>{rows}'
                       "</sheetData></worksheet>")
        return path

    def _ods(self, body, name="s.ods"):
        path = self.dir / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("mimetype",
                       "application/vnd.oasis.opendocument.spreadsheet",
                       compress_type=zipfile.ZIP_STORED)
            z.writestr("content.xml",
                       f'<?xml version="1.0"?><office:document-content '
                       f'{self.ODS_NS}><office:body><office:spreadsheet>'
                       f"{body}</office:spreadsheet></office:body>"
                       f"</office:document-content>")
        return path

    @staticmethod
    def _cell(text, attrs=""):
        return (f"<table:table-cell{attrs}><text:p>{text}</text:p>"
                "</table:table-cell>")

    def _row(self, text, attrs="", tag="table:table-row"):
        return f"<{tag}{attrs}>{self._cell(text)}</{tag}>"

    def _lines(self, path, sheet=None):
        from sift.extract import office
        page = office.read_sheet_page(path, detect(path), sheet, 0, 1000,
                                      1_000_000)
        self.assertIsNone(page.error)
        return [list(page.header)] + page.rows if page.header else page.rows

    # -- xlsx ----------------------------------------------------------------
    def test_a_shared_string_index_that_is_no_ascii_number_names_none(self):
        """Past the 200 rows extract() reads and inside them."""
        for index in ("²", "٣", "1_0", " 1", "+1", "-1",
                      "9" * 5000, "1" * 11):
            with self.subTest(index=index[:12]):
                filler = "".join(f'<row r="{i}"><c r="A{i}" t="s"><v>1</v>'
                                 "</c></row>" for i in range(1, 205))
                odd = (f'<row r="205"><c r="A205" t="s"><v>{index}</v></c>'
                       '<c r="B205" t="s"><v>0</v></c></row>')
                path = self._xlsx(filler + odd)
                lines = self._lines(path)
                self.assertEqual(["", "zero"], lines[-1])
                self.assertEqual(["one"], lines[0])
                inside = self._xlsx('<row r="1"><c r="A1" t="s"><v>1</v></c>'
                                    "</row>" + odd.replace("205", "2"),
                                    name="in.xlsx")
                ex = extract(inside, detect(inside), 25_000_000)
                self.assertNotIn("extractor_error", ex.metadata)
                self.assertIn("one", ex.text)

    def test_a_shared_string_index_of_ascii_digits_names_its_string(self):
        path = self._xlsx('<row r="1"><c r="A1" t="s"><v>1</v></c>'
                          '<c r="B1" t="s"><v>0000000000</v></c>'
                          '<c r="C1" t="s"><v>2</v></c></row>')
        self.assertEqual([["one", "zero"]], self._lines(path))

    # -- ods -----------------------------------------------------------------
    def test_an_ods_row_count_is_read_as_libreoffice_reads_it(self):
        for count, times in (("1_0", 1), ("3x", 3), (" 3 ", 3), ("+3", 3),
                             ("03", 3), ("²", 1), ("٣", 1),
                             ("0", 1), ("-2", 1), ("9" * 11, 1),
                             ("9" * 5000, 1), ("", 1), ("2", 2)):
            with self.subTest(count=count[:12]):
                path = self._ods(
                    '<table:table table:name="S">' + self._row("TOP")
                    + self._row("x", f' table:number-rows-repeated="{count}"')
                    + self._row("END") + "</table:table>")
                self.assertEqual([["TOP"]] + [["x"]] * times + [["END"]],
                                 self._lines(path))

    def test_an_ods_cell_count_is_read_as_libreoffice_reads_it(self):
        for count, times in (("1_0", 1), ("2", 2), ("²", 1)):
            with self.subTest(count=count):
                path = self._ods(
                    '<table:table table:name="S"><table:table-row>'
                    + self._cell("x", f' table:number-columns-repeated='
                                      f'"{count}"')
                    + self._cell("END") + "</table:table-row>"
                    + self._row("next") + "</table:table>")
                self.assertEqual([["x"] * times + ["END"], ["next"]],
                                 self._lines(path))

    def test_an_ods_is_read_in_the_table_namespace_only(self):
        """A row, a cell, a count or a table of another namespace is no
        part of the sheet, as LibreOffice 25.2 reads it."""
        body = ('<foo:table foo:name="F">' + self._row("FT") + "</foo:table>"
                '<table:table table:name="S">' + self._row("TOP")
                + self._row("F", tag="foo:table-row")
                + self._row("x", ' foo:number-rows-repeated="3"')
                + "<table:table-row>" + self._cell("a")
                + "<foo:table-cell><text:p>FC</text:p></foo:table-cell>"
                + self._cell("b", ' foo:number-columns-repeated="3"')
                + "</table:table-row>" + self._row("END") + "</table:table>")
        path = self._ods(body)
        from sift.extract import office
        page = office.read_sheet_page(path, detect(path), None, 0, 100,
                                      1_000_000)
        self.assertEqual(["S"], page.sheets)
        self.assertEqual([["TOP"], ["x"], ["a", "b"], ["END"]],
                         self._lines(path))
        ex = extract(path, detect(path), 25_000_000)
        self.assertEqual(["S"], [s["name"] for s in ex.metadata["sheets"]])

    def test_an_ods_row_group_of_the_table_namespace_is_read(self):
        """The rows of table:table-header-rows, table:table-rows and
        table:table-row-group, nested too, are the sheet's rows in document
        order, as LibreOffice shows them and the host application ods codec numbers
        them (a review, finding 7: a header row "Name,
        Qty" was not read, and the viewer showed "a, 1" as the header)."""
        for tag in ("table:table-row-group", "table:table-header-rows",
                    "table:table-rows"):
            with self.subTest(tag=tag):
                path = self._ods(
                    '<table:table table:name="S">' + self._row("TOP")
                    + f"<{tag}>" + self._row(
                        "G", ' table:number-rows-repeated="2"')
                    + self._row("H") + f"</{tag}>" + self._row("END")
                    + "</table:table>")
                self.assertEqual([["TOP"], ["G"], ["G"], ["H"], ["END"]],
                                 self._lines(path))
        nested = self._ods(
            '<table:table table:name="S"><table:table-header-rows>'
            + self._row("Name") + "</table:table-header-rows>"
            "<table:table-row-group>" + self._row("a")
            + "<table:table-row-group><table:table-rows>" + self._row("b")
            + "</table:table-rows></table:table-row-group>" + self._row("c")
            + "</table:table-row-group>" + self._row("END")
            + "</table:table>")
        self.assertEqual([["Name"], ["a"], ["b"], ["c"], ["END"]],
                         self._lines(nested))

    def test_an_ods_row_group_of_another_namespace_is_not_read(self):
        """Its rows are no part of the sheet (the table namespace only), nor
        is a table namespace group inside it; the rows after it are."""
        path = self._ods(
            '<table:table table:name="S">' + self._row("TOP")
            + "<foo:table-row-group>" + self._row("G")
            + "<table:table-row-group>" + self._row("GG")
            + "</table:table-row-group></foo:table-row-group>"
            + self._row("END") + "</table:table>")
        self.assertEqual([["TOP"], ["END"]], self._lines(path))

    def test_a_table_whose_rows_are_all_in_groups_is_listed(self):
        path = self._ods(
            '<table:table table:name="Only groups"><table:table-header-rows>'
            + self._row("head") + "</table:table-header-rows>"
            "</table:table>")
        from sift.extract import office
        page = office.read_sheet_page(path, detect(path), None, 0, 100,
                                      1_000_000)
        self.assertEqual(["Only groups"], page.sheets)
        ex = extract(path, detect(path), 25_000_000)
        self.assertEqual(["Only groups"],
                         [s["name"] for s in ex.metadata["sheets"]])
        self.assertIn("head", ex.text)

    def test_row_groups_nested_past_the_recursion_limit_are_read(self):
        """Groups are walked with a stack, not by recursion: 3,000 levels
        (Python's limit is 1,000) and a row at the bottom."""
        depth = 3000
        path = self._ods(
            '<table:table table:name="S">' + "<table:table-row-group>" * depth
            + self._row("deep") + "</table:table-row-group>" * depth
            + self._row("END") + "</table:table>")
        self.assertEqual([["deep"], ["END"]], self._lines(path))
        ex = extract(path, detect(path), 25_000_000)
        self.assertNotIn("extractor_error", ex.metadata)
        self.assertIn("deep", ex.text)
