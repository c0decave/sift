"""Active content: what in a file would run, or load something, if the file
were shown -- found by reading its bytes (sift/extract/active.py).

A host application refuses to render a file that carries it.
Every fixture here is built in the test and harmless: alert(1), the name
calc.exe, a 64-byte MZ stub that is no program at all.

Each class has a file that carries it and a file that does not; each escape
the format allows has a file that uses it; and every way the scan can fall
short of the whole file -- encrypted, damaged, unlistable, over a budget --
says so with ``checked`` False.
"""
import base64
import io
import struct
import tarfile
import tempfile
import time
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

from sift import deadline
from sift.config import Config
from sift.detect import detect
from sift.extract import active, extract
from sift.extract.base import scrub
from sift.llm.prompts import build_user_prompt, render_metadata
from sift.summarize import _has_structure, summarize_path

from _fixtures import EML_TEXT, make_docx, make_xlsx

MZ = b"MZ" + b"\x00" * 58 + struct.pack("<I", 64) + b"PE\x00\x00" + b"\x00" * 60
ELF = b"\x7fELF\x02\x01\x01" + b"\x00" * 57
CLASS = b"\xca\xfe\xba\xbe\x00\x00\x00\x34" + b"\x00" * 24


# ---- builders ------------------------------------------------------------

def pdf(objects, trailer=b"", header=b"%PDF-1.7\n"):
    """A PDF with a classic cross-reference table; objects[i] is the body of
    object i + 1, and object 1 is the catalog."""
    out, offsets = bytearray(header), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += (b"trailer\n<< /Size %d /Root 1 0 R " % (len(objects) + 1)
            + trailer + b">>\nstartxref\n%d\n%%%%EOF\n" % at)
    return bytes(out)


PAGES = [b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
         b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
         b"/Contents 4 0 R >>",
         b"<< /Length 36 >>\nstream\nBT /F1 12 Tf 20 100 Td (Hi) Tj ET\n\n"
         b"endstream"]


def catalog(extra=b""):
    return b"<< /Type /Catalog /Pages 2 0 R " + extra + b" >>"


def plain_pdf(extra=b"", more=()):
    return pdf([catalog(extra)] + PAGES + list(more))


def pdf_with_object_stream(plain, packed, filters=b"/FlateDecode", typed=True,
                           encode=zlib.compress):
    """Objects 1..len(plain) written plainly, the next len(packed) packed
    into one object stream, and a cross-reference stream after it."""
    out, offsets = bytearray(b"%PDF-1.7\n"), {}
    for number, body in enumerate(plain, 1):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    first_packed = len(plain) + 1
    heads, bodies, at = [], b"", 0
    for i, body in enumerate(packed):
        heads.append(b"%d %d" % (first_packed + i, at))
        bodies += body + b"\n"
        at = len(bodies)
    head = b" ".join(heads) + b"\n"
    content = encode(head + bodies)
    stm = first_packed + len(packed)
    offsets[stm] = len(out)
    kind = b"/Type /ObjStm " if typed else b""
    out += (b"%d 0 obj\n<< %s/N %d /First %d /Filter %s /Length %d >>\nstream\n"
            % (stm, kind, len(packed), len(head), filters, len(content))
            + content + b"\nendstream\nendobj\n")
    xref = stm + 1
    offsets[xref] = len(out)
    rows = [b"\x00" + struct.pack(">IH", 0, 65535)]
    for number in range(1, xref + 1):
        if first_packed <= number < stm:
            rows.append(b"\x02" + struct.pack(">IH", stm, number - first_packed))
        else:
            rows.append(b"\x01" + struct.pack(">IH", offsets[number], 0))
    data = b"".join(rows)
    out += (b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 4 2] /Root 1 0 R "
            b"/Length %d >>\nstream\n" % (xref, xref + 1, len(data))
            + data + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n"
            % offsets[xref])
    return bytes(out)


CT = "http://schemas.openxmlformats.org/package/2006/content-types"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
OREL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def zipped(files, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def content_types(*overrides):
    return (f'<?xml version="1.0"?><Types xmlns="{CT}"><Default Extension="xml" '
            'ContentType="application/xml"/>'
            + "".join(f'<Override PartName="{p}" ContentType="{c}"/>'
                      for p, c in overrides) + "</Types>")


def rels(*relationships):
    return (f'<?xml version="1.0"?><Relationships xmlns="{REL}">'
            + "".join(f'<Relationship Id="rId{i}" Type="{OREL}/{t}" '
                      f'Target="{target}"{" TargetMode=%s" % chr(34) + mode + chr(34) if mode else ""}/>'
                      for i, (t, target, mode) in enumerate(relationships, 1))
            + "</Relationships>")


def document(body, prefix="w"):
    return (f'<?xml version="1.0"?><{prefix}:document xmlns:{prefix}="{W}">'
            f'<{prefix}:body>{body}</{prefix}:body></{prefix}:document>')


def docx(body="<w:p><w:r><w:t>Hello</w:t></w:r></w:p>", extra=None,
         ctype="application/vnd.openxmlformats-officedocument."
               "wordprocessingml.document.main+xml", prefix="w"):
    files = {"[Content_Types].xml": content_types(("/word/document.xml", ctype)),
             "_rels/.rels": rels(("officeDocument", "word/document.xml", "")),
             "word/document.xml": document(body, prefix)}
    files.update(extra or {})
    return zipped(files)


def field(code, prefix="w"):
    """A complex field whose instruction is split over two runs."""
    half = len(code) // 2
    p = prefix
    return (f'<{p}:p><{p}:r><{p}:fldChar {p}:fldCharType="begin"/></{p}:r>'
            f'<{p}:r><{p}:instrText>{code[:half]}</{p}:instrText></{p}:r>'
            f'<{p}:r><{p}:instrText>{code[half:]}</{p}:instrText></{p}:r>'
            f'<{p}:r><{p}:fldChar {p}:fldCharType="separate"/></{p}:r>'
            f'<{p}:r><{p}:t>shown</{p}:t></{p}:r>'
            f'<{p}:r><{p}:fldChar {p}:fldCharType="end"/></{p}:r></{p}:p>')


def xlsx(sheet="<sheetData/>", extra=None):
    files = {"[Content_Types].xml": content_types(
                 ("/xl/workbook.xml", "application/vnd.openxmlformats-"
                  "officedocument.spreadsheetml.sheet.main+xml")),
             "_rels/.rels": rels(("officeDocument", "xl/workbook.xml", "")),
             "xl/workbook.xml": f'<workbook xmlns="{SS}"><sheets><sheet '
                                f'name="S" sheetId="1"/></sheets></workbook>',
             "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{SS}">{sheet}'
                                         '</worksheet>'}
    files.update(extra or {})
    return zipped(files)


ODF_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:script="urn:oasis:names:tc:opendocument:xmlns:script:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
          'xmlns:xlink="http://www.w3.org/1999/xlink"')


def odt(body="<text:p>Hello</text:p>", head="", extra=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.text",
                   compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/manifest.xml", "<manifest/>")
        z.writestr("content.xml", f'<?xml version="1.0"?><office:document-content '
                   f'{ODF_NS}>{head}<office:body><office:text>{body}'
                   '</office:text></office:body></office:document-content>')
        z.writestr("styles.xml", f'<office:document-styles {ODF_NS}/>')
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def cfb(entries):
    """A compound file (the container of .doc, .xls, .ppt, .msg).

    ``entries`` is a list of (path, data): a path is a tuple of names, data
    is bytes for a stream and None for a storage. Streams under 4096 bytes
    go to the mini stream, as a writer puts them."""
    sector = 512
    tree = {(): []}
    for path, data in entries:
        tree.setdefault(path[:-1], []).append((path, data))
        if data is None:
            tree.setdefault(path, [])
    order = [((), None)]
    index = {(): 0}

    def visit(parent):
        for path, data in tree.get(parent, []):
            index[path] = len(order)
            order.append((path, data))
        for path, data in tree.get(parent, []):
            if data is None:
                visit(path)
    visit(())
    mini, big = bytearray(), bytearray()
    starts = {}
    for path, data in order[1:]:
        if data is None:
            continue
        if len(data) < 4096:
            starts[path] = ("mini", len(mini) // 64)
            mini += data + b"\x00" * (-len(data) % 64)
        else:
            starts[path] = ("big", len(big) // sector)
            big += data + b"\x00" * (-len(data) % sector)
    ministream = bytes(mini) + b"\x00" * (-len(mini) % sector)
    minifat = []
    for path, data in order[1:]:
        if data is not None and starts[path][0] == "mini":
            n = -(-len(data) // 64) or 1
            first = starts[path][1]
            minifat += [first + k + 1 for k in range(n - 1)] + [0xFFFFFFFE]
    minifat_bytes = struct.pack("<%dI" % len(minifat), *minifat) if minifat else b""
    minifat_bytes += b"\xff" * (-len(minifat_bytes) % sector)
    dirs = bytearray()
    layout = []                   # (what, bytes) in sector order after the FAT
    layout.append(("big", bytes(big)))
    layout.append(("ministream", ministream))
    layout.append(("minifat", minifat_bytes))
    n_dir = -(-len(order) * 128 // sector)
    layout.append(("dir", b"\x00" * n_dir * sector))
    body_sectors = sum(len(b) // sector for _, b in layout)
    n_fat = 1
    while n_fat * (sector // 4) < body_sectors + n_fat:
        n_fat += 1
    fat = [0xFFFFFFFD] * n_fat
    first_of = {}
    at = n_fat
    for what, blob in layout:
        count = len(blob) // sector
        first_of[what] = at if count else 0xFFFFFFFE
        at += count
    # chains
    for what, blob in layout:
        count = len(blob) // sector
        if what == "big":
            for path, data in order[1:]:
                if data is not None and starts[path][0] == "big":
                    n = -(-len(data) // sector)
                    first = first_of["big"] + starts[path][1]
                    fat += [first + k + 1 for k in range(n - 1)] + [0xFFFFFFFE]
        elif count:
            fat += [first_of[what] + k + 1 for k in range(count - 1)] + [0xFFFFFFFE]
    fat += [0xFFFFFFFF] * (n_fat * sector // 4 - len(fat))
    for n, (path, data) in enumerate(order):
        name = (path[-1] if path else "Root Entry").encode("utf-16-le")
        kind = 5 if not path else (1 if data is None else 2)
        siblings = tree.get(path[:-1], []) if path else []
        right = 0xFFFFFFFF
        if path:
            place = [p for p, _ in siblings].index(path)
            if place + 1 < len(siblings):
                right = index[siblings[place + 1][0]]
        child = 0xFFFFFFFF
        if kind in (1, 5) and tree.get(path):
            child = index[tree[path][0][0]]
        if kind == 5:
            start = first_of["ministream"] if mini else 0xFFFFFFFE
            size = len(mini)
        elif kind == 2:
            where, first = starts[path]
            start = first if where == "mini" else first_of["big"] + first
            size = len(data)
        else:
            start, size = 0, 0
        dirs += (name + b"\x00" * (64 - len(name))
                 + struct.pack("<HBBIII", len(name) + 2, kind, 1, 0xFFFFFFFF,
                               right, child)
                 + b"\x00" * 16 + b"\x00" * 4 + b"\x00" * 16
                 + struct.pack("<IQ", start, size))
    dirs += b"\x00" * (n_dir * sector - len(dirs))
    header = bytearray(sector)
    header[:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<HHHHH", header, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<IIIIIIII", header, 40, 0, n_fat, first_of["dir"], 0,
                     4096, first_of["minifat"] if minifat else 0xFFFFFFFE,
                     len(minifat_bytes) // sector, 0xFFFFFFFE)
    struct.pack_into("<I", header, 72, 0)
    difat = list(range(n_fat)) + [0xFFFFFFFF] * (109 - n_fat)
    struct.pack_into("<109I", header, 76, *difat)
    fat_bytes = struct.pack("<%dI" % len(fat), *fat)
    out = bytes(header) + fat_bytes
    for what, blob in layout:
        out += dirs if what == "dir" else blob
    return out


def boundsheet(name, kind):
    raw = name.encode("latin-1")
    body = struct.pack("<IBBBB", 0, 0, kind, len(raw), 0) + raw
    return struct.pack("<HH", 0x0085, len(body)) + body


def workbook(*sheets):
    bof = struct.pack("<HH", 0x0809, 16) + struct.pack("<HHHHII", 0x0600, 5,
                                                         0, 0, 0, 0)
    eof = struct.pack("<HH", 0x000A, 0)
    stream = bof + b"".join(sheets) + eof
    return stream + b"\x00" * (4096 - len(stream))


def mail(*parts, boundary="b1"):
    """A multipart/mixed message; each part is (headers, body)."""
    out = ("From: a@example.invalid\r\nTo: b@example.invalid\r\nSubject: s\r\n"
           "Date: Mon, 1 Sep 2026 10:00:00 +0200\r\nMIME-Version: 1.0\r\n"
           f"Content-Type: multipart/mixed; boundary=\"{boundary}\"\r\n\r\n")
    for headers, body in parts:
        out += f"--{boundary}\r\n{headers}\r\n\r\n{body}\r\n"
    return (out + f"--{boundary}--\r\n").encode()


def attachment(name, data, ctype="application/octet-stream"):
    return (f"Content-Type: {ctype}; name=\"{name}\"\r\nContent-Disposition: "
            f"attachment; filename=\"{name}\"\r\nContent-Transfer-Encoding: "
            f"base64", base64.encodebytes(data).decode())


HTML_PART = "Content-Type: text/html; charset=utf-8"


def tgz(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _Files(unittest.TestCase):
    """Writes a fixture to a temporary file and scans it as sift would."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def path(self, name, data):
        p = Path(self._dir.name) / name
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return p

    def found(self, name, data, max_bytes=25_000_000):
        p = self.path(name, data)
        return active.scan(p, detect(p), max_bytes)

    def classes(self, name, data):
        got = self.found(name, data).metadata()
        return set(got["classes"])

    def assertClasses(self, name, data, expected, checked=True):
        got = self.found(name, data).metadata()
        self.assertEqual(set(got["classes"]), set(expected), name)
        self.assertEqual(got["checked"], checked, name)

    def assertNone(self, name, data):
        self.assertClasses(name, data, set())


# ---- PDF -----------------------------------------------------------------

class PdfTest(_Files):

    def test_an_ordinary_pdf_carries_nothing(self):
        self.assertNone("plain.pdf", plain_pdf())

    def test_javascript_in_the_open_action(self):
        self.assertClasses("a.pdf", plain_pdf(
            b"/OpenAction << /S /JavaScript /JS (app.alert\\(1\\)) >>"),
            {"pdf-javascript", "pdf-open-action"})

    def test_javascript_in_the_name_tree(self):
        self.assertClasses("a.pdf", plain_pdf(
            b"/Names << /JavaScript << /Names [(x) 5 0 R] >> >>",
            more=[b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"]),
            {"pdf-javascript"})

    def test_javascript_in_additional_actions_of_a_page(self):
        data = pdf([catalog(), PAGES[0],
                    b"<< /Type /Page /Parent 2 0 R /AA << /O << /S /JavaScript "
                    b"/JS 5 0 R >> >> >>", PAGES[2],
                    b"<< /Length 12 >>\nstream\napp.alert(1)\nendstream"])
        self.assertClasses("a.pdf", data, {"pdf-javascript"})

    def test_each_action_type_has_its_class(self):
        for action, cls in ((b"/Launch /F (calc.exe)", "pdf-launch"),
                            (b"/SubmitForm /F (http://x.invalid/)",
                             "pdf-submit-form"),
                            (b"/ImportData /F (data.fdf)", "pdf-import-data")):
            with self.subTest(cls):
                data = plain_pdf(more=[b"<< /Type /Annot /Subtype /Link "
                                       b"/Rect [0 0 9 9] /A << /S " + action
                                       + b" >> >>"])
                self.assertClasses("a.pdf", data, {cls})

    def test_embedded_files_richmedia_and_xfa(self):
        self.assertClasses("a.pdf", plain_pdf(
            b"/Names << /EmbeddedFiles << /Names [(tool.exe) 5 0 R] >> >>",
            more=[b"<< /Type /Filespec /F (tool.exe) /EF << /F 6 0 R >> >>",
                  b"<< /Type /EmbeddedFile /Length 4 >>\nstream\nMZ..\n"
                  b"endstream"]), {"pdf-embedded-file"})
        self.assertClasses("a.pdf", plain_pdf(more=[
            b"<< /Type /Annot /Subtype /RichMedia /RichMediaContent 6 0 R >>",
            b"<< >>"]), {"pdf-richmedia"})
        self.assertClasses("a.pdf", plain_pdf(
            b"/AcroForm << /Fields [] /XFA 5 0 R >>",
            more=[b"<< /Length 5 >>\nstream\n<xdp>\nendstream"]), {"pdf-xfa"})

    def test_remote_actions_that_run_by_themselves(self):
        self.assertClasses("a.pdf", plain_pdf(
            b"/OpenAction << /S /URI /URI (http://x.invalid/) >>"),
            {"pdf-open-action", "pdf-goto-remote"})
        self.assertClasses("a.pdf", plain_pdf(
            b"/OpenAction << /S /GoToR /F (other.pdf) /D [0 /Fit] >>"),
            {"pdf-open-action", "pdf-goto-remote"})
        data = pdf([catalog(), PAGES[0],
                    b"<< /Type /Page /Parent 2 0 R /AA << /O 5 0 R >> >>",
                    PAGES[2], b"<< /S /GoToE /T << /R /C /N (x) >> >>"])
        self.assertClasses("a.pdf", data, {"pdf-goto-remote"})

    def test_a_link_the_reader_clicks_is_not_active(self):
        self.assertNone("a.pdf", plain_pdf(more=[
            b"<< /Type /Annot /Subtype /Link /Rect [0 0 9 9] "
            b"/A << /S /URI /URI (https://example.invalid/) >> >>"]))

    def test_an_open_action_that_only_turns_to_a_page(self):
        self.assertNone("a.pdf", plain_pdf(b"/OpenAction [3 0 R /Fit]"))
        self.assertNone("a.pdf", plain_pdf(b"/OpenAction 5 0 R",
                                           more=[b"[3 0 R /XYZ 0 792 0]"]))
        self.assertNone("a.pdf", plain_pdf(
            b"/OpenAction << /S /GoTo /D [3 0 R /Fit] >>"))

    def test_an_open_action_that_cannot_be_resolved_counts(self):
        self.assertClasses("a.pdf", plain_pdf(b"/OpenAction 99 0 R"),
                           {"pdf-open-action"})

    def test_names_written_in_hex_escapes(self):
        self.assertClasses("a.pdf", plain_pdf(
            b"/Open#41ction << /S /Java#53cript /#4A#53 (app.alert\\(1\\)) >>"),
            {"pdf-javascript", "pdf-open-action"})
        self.assertClasses("a.pdf", plain_pdf(
            b"/OpenAction << /S /Java#73cript /J#53 <6170702e616c65727428312"
            b"9> >>"), {"pdf-javascript", "pdf-open-action"})

    def test_javascript_in_a_compressed_object_stream(self):
        data = pdf_with_object_stream(
            [catalog(b"/OpenAction 5 0 R")] + PAGES,
            [b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"])
        self.assertClasses("a.pdf", data, {"pdf-javascript", "pdf-open-action"})
        # The raw bytes do not say it: only the stream does.
        self.assertNotIn(b"/JavaScript", data)

    def test_object_stream_with_a_filter_chain_and_no_type(self):
        data = pdf_with_object_stream(
            [catalog()] + PAGES,
            [b"<< /Type /Annot /Subtype /Link /A << /S /Launch /F (calc) >> >>"],
            filters=b"[/ASCIIHexDecode /FlateDecode]", typed=False,
            encode=lambda raw: zlib.compress(raw).hex().encode() + b">")
        self.assertClasses("a.pdf", data, {"pdf-launch"})

    def test_an_object_stream_whose_dictionary_says_obj(self):
        """The dictionary is found from the object's header: a string or a
        name holding "obj" inside it does not cut it short."""
        data = pdf_with_object_stream(
            [catalog(b"/OpenAction 5 0 R")] + PAGES,
            [b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"])
        data = data.replace(b"<< /Type /ObjStm ",
                            b"<< /Note (an obj) /Xobj 1 /Type /ObjStm ")
        self.assertClasses("a.pdf", data, {"pdf-javascript", "pdf-open-action"})

    def test_an_object_stream_without_n_and_first_is_read_whole(self):
        """No reader can find an object in it, but every byte of it can be
        read: it is checked, and what it says is said."""
        for inside, expected in ((b"<< /Type /Page >>", set()),
                                 (b"<< /S /JavaScript /JS (x) >>",
                                  {"pdf-javascript"})):
            with self.subTest(expected):
                packed = zlib.compress(inside)
                data = pdf([catalog()] + PAGES + [
                    b"<< /Type /ObjStm /Filter /FlateDecode /Length %d >>\n"
                    b"stream\n" % len(packed) + packed + b"\nendstream"])
                self.assertClasses("a.pdf", data, expected)

    def test_an_object_stream_whose_dictionary_is_padded(self):
        data = pdf_with_object_stream(
            [catalog()] + PAGES,
            [b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"])
        data = data.replace(b"<< /Type /ObjStm ",
                            b"<< /Type /ObjStm /Pad [%s] " % (b"0 " * 50_000))
        self.assertIn("pdf-javascript", self.classes("a.pdf", data))

    def test_a_comment_inside_an_object_header(self):
        hidden = b"9 %x\n0 obj << /S /JavaScript /JS (x) >> endobj"
        data = pdf([catalog(), PAGES[0], PAGES[1],
                    b"<< /Length %d >>\nstream\n" % len(hidden) + hidden
                    + b"\nendstream"])
        self.assertIn("pdf-javascript", self.classes("a.pdf", data))

    def test_words_inside_strings_and_streams_are_not_names(self):
        self.assertNone("a.pdf", plain_pdf(
            b"/Title (see /JS and /OpenAction \\) /AA) /Subject <2f4a53>"))
        stream = b"BT /JS 12 Tf (x) Tj ET /Launch Do"
        self.assertNone("a.pdf", pdf([catalog(), PAGES[0], PAGES[1],
                                      b"<< /Length %d >>\nstream\n" % len(stream)
                                      + stream + b"\nendstream"]))

    def test_an_object_hidden_inside_a_stream_body_is_found(self):
        hidden = b"9 0 obj << /S /JavaScript /JS (x) >> endobj"
        data = pdf([catalog(), PAGES[0], PAGES[1],
                    b"<< /Length %d >>\nstream\n" % len(hidden) + hidden
                    + b"\nendstream"])
        self.assertIn("pdf-javascript", self.classes("a.pdf", data))

    def test_a_string_that_never_closes_does_not_hide_the_objects_after_it(self):
        data = pdf([catalog(b"/Title (never closed"), PAGES[0], PAGES[1],
                    PAGES[2], b"<< /S /JavaScript /JS 1 >>"])
        self.assertIn("pdf-javascript", self.classes("a.pdf", data))

    def test_an_encrypted_pdf_is_not_checked(self):
        self.assertClasses("a.pdf", pdf(
            [catalog()] + PAGES + [b"<< /Filter /Standard /V 2 /R 3 >>"],
            trailer=b"/Encrypt 5 0 R /ID [<00> <00>] "), set(), checked=False)

    def test_an_object_stream_that_cannot_be_decoded_is_not_checked(self):
        data = pdf_with_object_stream([catalog()] + PAGES, [b"<< >>"],
                                      filters=b"/DCTDecode", encode=bytes)
        self.assertEqual(self.found("a.pdf", data).metadata()["checked"], False)
        broken = pdf_with_object_stream([catalog()] + PAGES, [b"<< >>"],
                                        encode=lambda raw: b"not zlib at all")
        self.assertEqual(self.found("a.pdf", broken).metadata()["checked"], False)

    def test_a_pdf_read_only_in_part_is_not_checked(self):
        data = plain_pdf(more=[b"<< /Filler (%s) >>" % (b"x" * 5000)])
        got = self.found("a.pdf", data, max_bytes=1000).metadata()
        self.assertFalse(got["checked"])

    def test_the_decoded_budget_cuts_the_scan(self):
        data = pdf_with_object_stream([catalog()] + PAGES,
                                      [b"<< /Pad (%s) >>" % (b"y" * 50_000)])
        with mock.patch.object(active, "DECODE_BUDGET", 10_000):
            got = self.found("a.pdf", data).metadata()
        self.assertFalse(got["checked"])


# ---- OOXML, ODF, RTF -------------------------------------------------------

class OfficeTest(_Files):

    def test_ordinary_documents_carry_nothing(self):
        d = Path(self._dir.name)
        for p in (make_docx(d / "a.docx"), make_xlsx(d / "b.xlsx")):
            got = active.scan(p, detect(p), 25_000_000).metadata()
            self.assertEqual(got, {"classes": [], "checked": True}, p.name)
        self.assertNone("c.docx", docx())
        self.assertNone("d.xlsx", xlsx())

    def test_macros_by_part_and_by_content_type(self):
        self.assertClasses("a.docm", docx(
            ctype="application/vnd.ms-word.document.macroEnabled.main+xml",
            extra={"word/vbaProject.bin": b"\xd0\xcf\x11\xe0VBA"}),
            {"office-macros"})
        self.assertClasses("a.xlsm", xlsx(extra={
            "xl/vbaProject.bin": b"\xd0\xcf\x11\xe0VBA"}), {"office-macros"})
        self.assertClasses("a.xlsx", xlsx(extra={
            "xl/macrosheets/sheet1.xml": f'<xm:macrosheet xmlns:xm="{SS}"/>'}),
            {"office-macros"})
        self.assertClasses("a.docx", docx(
            ctype="application/vnd.ms-word.document.macroEnabled.main+xml"),
            {"office-macros"})

    def test_activex_and_ole_objects(self):
        self.assertClasses("a.docx", docx(extra={
            "word/activeX/activeX1.xml": "<ax:ocx xmlns:ax='urn:x'/>"}),
            {"office-activex"})
        self.assertClasses("a.docx", docx(extra={
            "word/embeddings/oleObject1.bin": b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"}),
            {"office-ole-object"})

    def test_an_embedded_workbook_for_a_chart_is_not_active(self):
        self.assertNone("a.docx", docx(extra={
            "word/embeddings/Microsoft_Excel_Worksheet.xlsx": xlsx()}))

    def test_dde_fields_split_over_runs_and_under_any_prefix(self):
        code = ' DDEAUTO c:\\\\windows\\\\system32\\\\cmd.exe "/k calc.exe" '
        self.assertClasses("a.docx", docx(field(code)), {"office-dde"})
        self.assertClasses("a.docx", docx(field(code, "x"), prefix="x"),
                           {"office-dde"})
        self.assertClasses("a.docx", docx(
            '<w:p><w:fldSimple w:instr=" dde calc x"/></w:p>'), {"office-dde"})
        # A character reference is the same letter to Word.
        self.assertClasses("a.docx", docx(field("&#68;DEAUTO calc x")),
                           {"office-dde"})

    def test_ordinary_fields_are_not_dde(self):
        self.assertNone("a.docx", docx(
            field(' HYPERLINK "http://example.invalid/dde" ')
            + field(" PAGE ") + '<w:p><w:r><w:t>DDEAUTO in text</w:t></w:r></w:p>'))

    def test_dde_in_a_workbook(self):
        self.assertClasses("a.xlsx", xlsx(extra={
            "xl/externalLinks/externalLink1.xml":
                f'<externalLink xmlns="{SS}"><ddeLink ddeService="cmd" '
                'ddeTopic="/c calc"/></externalLink>'}), {"office-dde"})
        self.assertClasses("a.xlsx", xlsx(
            "<sheetData><row r='1'><c r='A1'><f>cmd|' /C calc'!A0</f></c>"
            "</row></sheetData>"), {"office-dde"})
        self.assertNone("a.xlsx", xlsx(
            "<sheetData><row r='1'><c r='A1'><f>SUM(B1:B9)</f></c><c r='B1' "
            "t='inlineStr'><is><t>a | b ! c</t></is></c></row></sheetData>"))

    def test_external_relationships_that_load_something(self):
        settings = rels(("attachedTemplate", "https://x.invalid/t.dotm",
                         "External"))
        self.assertClasses("a.docx", docx(extra={
            "word/_rels/settings.xml.rels": settings}),
            {"office-external-template"})
        self.assertClasses("a.docx", docx(extra={
            "word/_rels/document.xml.rels": rels(
                ("image", "http://x.invalid/pixel.png", "External"))}),
            {"office-external-template"})

    def test_an_external_hyperlink_is_not_a_template(self):
        self.assertNone("a.docx", docx(extra={
            "word/_rels/document.xml.rels": rels(
                ("hyperlink", "https://example.invalid/", "External"))}))

    def test_program_and_macro_actions_in_a_presentation(self):
        slide = ('<p:sld xmlns:p="http://schemas.openxmlformats.org/'
                 'presentationml/2006/main" xmlns:a="http://schemas.'
                 'openxmlformats.org/drawingml/2006/main"><a:hlinkClick '
                 'action="ppaction://program" r:id="rId1" xmlns:r="%s"/>'
                 '</p:sld>' % OREL)
        files = {"[Content_Types].xml": content_types(),
                 "ppt/presentation.xml": "<p/>",
                 "ppt/slides/slide1.xml": slide}
        self.assertClasses("a.pptx", zipped(files), {"office-macros"})

    def test_opendocument_scripts(self):
        self.assertNone("a.odt", odt())
        self.assertClasses("a.odt", odt(extra={
            "Basic/Standard/Module1.xml": "<module>Sub Main</module>"}),
            {"odf-scripts"})
        self.assertClasses("a.odt", odt(head=(
            '<office:scripts><office:event-listeners><script:event-listener '
            'script:language="ooo:script" script:event-name="dom:load" '
            'xlink:href="vnd.sun.star.script:Standard.Module1.Main?language='
            'Basic&amp;location=document"/></office:event-listeners>'
            '</office:scripts>')), {"odf-scripts"})
        self.assertClasses("a.odt", odt(
            '<text:p><text:a xlink:href="vnd.sun.star.script:S.M.Main">x'
            '</text:a></text:p>'), {"odf-scripts"})
        self.assertClasses("a.odt", odt(
            '<draw:frame><draw:object-ole xlink:href="./Obj"/></draw:frame>'),
            {"office-ole-object"})

    def test_rtf_objects(self):
        self.assertNone("a.rtf", "{\\rtf1\\ansi An object lesson, \\b bold\\b0.}")
        self.assertClasses("a.rtf", "{\\rtf1 x{\\object\\objemb{\\*\\objclass "
                           "Package}{\\*\\objdata 0105}}}", {"rtf-object"})
        self.assertClasses("a.rtf", "{\\rtf1 x{\\objupdate1}}", {"rtf-object"})
        self.assertClasses("a.rtf", "{\\rtf1 {\\field{\\*\\fldinst {DD}EAUTO "
                           "c:\\\\cmd.exe}{\\fldrslt x}}}", {"office-dde"})
        self.assertNone("a.rtf", "{\\rtf1 {\\field{\\*\\fldinst HYPERLINK "
                        "\"http://x.invalid/dde\"}{\\fldrslt x}}}")


# ---- HTML and XML --------------------------------------------------------

class HtmlTest(_Files):

    def test_an_ordinary_page_carries_nothing(self):
        self.assertNone("a.html", "<!doctype html><html><head><title>t</title>"
                        "<link rel=stylesheet href='site.css'></head><body>"
                        "<p>Hi <a href='#x'>x</a> <a href='mailto:a@b.c'>m</a>"
                        "<img src='img/a.png' alt='javascript: the good parts'>"
                        "</p></body></html>")

    def test_each_class(self):
        cases = {
            "<script>alert(1)</script>": "html-script",
            "<body onload='alert(1)'>": "html-event-handler",
            "<a href='javascript:alert(1)'>x</a>": "html-javascript-url",
            "<iframe src='x.html'></iframe>": "html-frame",
            "<object data='x.swf'></object>": "html-frame",
            "<embed src='x.swf'>": "html-frame",
            "<img src='https://x.invalid/p.gif'>": "html-remote-resource",
            "<img src='//x.invalid/p.gif'>": "html-remote-resource",
            "<img srcset='a.png 1x, http://x.invalid/b.png 2x'>":
                "html-remote-resource",
            "<p style='background:url(\"ftp://x.invalid/a\")'>":
                "html-remote-resource",
            "<style>@import 'https://x.invalid/a.css';</style>":
                "html-remote-resource",
            "<link rel=icon href=file:///etc/passwd>": "html-remote-resource",
        }
        for markup, cls in cases.items():
            with self.subTest(markup):
                self.assertIn(cls, self.classes("a.html", markup))

    def test_case_entities_and_whitespace_do_not_hide_anything(self):
        for markup, cls in (
                ("<SCRIPT>alert(1)</SCRIPT>", "html-script"),
                ("<ScRiPt src=x.js></sCrIpT>", "html-script"),
                ("<img src=x OnErRoR=alert(1)>", "html-event-handler"),
                ("<img src=x onerror=&#97;&#108;ert(1)>", "html-event-handler"),
                ("<svg/onload=alert(1)>", "html-event-handler"),
                ("<a href='java&#x09;script:alert(1)'>x</a>",
                 "html-javascript-url"),
                ("<a href='jav&#x0A;ascript:alert(1)'>x</a>",
                 "html-javascript-url"),
                ("<a href='&#106;avascript:alert(1)'>x</a>",
                 "html-javascript-url"),
                ("<a href='javascript&colon;alert(1)'>x</a>",
                 "html-javascript-url"),
                ("<a href=' \x01JaVaScRiPt:alert(1)'>x</a>",
                 "html-javascript-url"),
                ("<a href='vbscript:msgbox(1)'>x</a>", "html-javascript-url"),
                ("<iframe src='data:text/html,<b>x</b>'></iframe>",
                 "html-javascript-url"),
                ("<object data='data:image/svg+xml;base64,PHN2Zz4='></object>",
                 "html-javascript-url"),
                ("<meta http-equiv=refresh content='0;url=javascript:alert(1)'>",
                 "html-javascript-url"),
                ("<p style='width:expression(alert(1))'>", "html-javascript-url"),
                ("<p style='background:url(\\68ttp://x.invalid/a)'>",
                 "html-remote-resource"),
                ("<img src='\\\\x.invalid\\share\\a.png'>",
                 "html-remote-resource"),
                ("<iframe srcdoc='&lt;script&gt;alert(1)&lt;/script&gt;'>",
                 "html-script"),
                ("<svg><script>alert(1)</script></svg>", "html-script"),
                ("<svg><set attributeName=onmouseover to=alert(1)></svg>",
                 "html-event-handler"),
                ("<!--><script>alert(1)</script>-->", "html-script"),
                ("<!---><script>alert(1)</script>-->", "html-script"),
                ("<style><!--</style><img src=x onerror=alert(1)>-->",
                 "html-event-handler"),
                ("<title><!--</title><script>alert(1)</script>-->",
                 "html-script"),
                ("<noscript><p title='</noscript><img src=x onerror=alert(1)>'>",
                 "html-event-handler"),
                ("<p \"><script>alert(1)</script>\">", "html-script")):
            with self.subTest(markup):
                self.assertIn(cls, self.classes("a.html", markup))

    def test_what_is_not_active(self):
        for markup in ("<!-- <script>alert(1)</script> -->",
                       "<img src='data:image/png;base64,iVBORw0KGgo='>",
                       "<img src='data:image/svg+xml;base64,PHN2Zz4='>",
                       "<img src='cid:part1@example.invalid'>",
                       "<p>on the javascript: scheme and onload=</p>",
                       "<a title='javascript: tips' href='/x'>x</a>",
                       "<p data-onload='x' online='yes'>x</p>"):
            with self.subTest(markup):
                self.assertNone("a.html", markup)

    def test_a_utf16_page(self):
        data = "\ufeff<p>x</p><script>alert(1)</script>".encode("utf-16-le")
        self.assertIn("html-script", self.classes("a.html", data))

    def test_a_page_read_only_in_part_is_not_checked(self):
        data = "<p>" + "x" * 5000 + "</p><script>alert(1)</script>"
        got = self.found("a.html", data, max_bytes=1000).metadata()
        self.assertFalse(got["checked"])


class XmlTest(_Files):
    SVG = "http://www.w3.org/2000/svg"

    def test_an_ordinary_svg_and_an_ordinary_xml_carry_nothing(self):
        self.assertNone("a.svg", f'<svg xmlns="{self.SVG}"><rect width="1"/>'
                        '<a href="#x"><text>t</text></a></svg>')
        self.assertNone("a.xml", '<?xml version="1.0"?><catalog><object id="1" '
                        'onhand="3" href="http://x.invalid/"/><embed/></catalog>')

    def test_svg_scripts_under_any_prefix(self):
        self.assertIn("html-script", self.classes(
            "a.svg", f'<svg xmlns="{self.SVG}"><script>alert(1)</script></svg>'))
        self.assertIn("html-script", self.classes(
            "a.svg", f'<s:svg xmlns:s="{self.SVG}"><s:script>alert(1)'
                     '</s:script></s:svg>'))
        self.assertClasses("a.svg", f'<svg xmlns="{self.SVG}" onload="alert(1)"'
                           '/>', {"html-event-handler"})
        self.assertClasses(
            "a.svg", f'<svg xmlns="{self.SVG}" xmlns:l="http://www.w3.org/1999/'
            'xlink"><a l:href="javascript:alert(1)"><text>x</text></a></svg>',
            {"html-javascript-url"})
        self.assertClasses(
            "a.svg", f'<svg xmlns="{self.SVG}"><image href="https://x.invalid/'
            'a.png"/></svg>', {"html-remote-resource"})

    def test_an_entity_does_not_hide_a_javascript_url(self):
        self.assertIn("html-javascript-url", self.classes(
            "a.svg", '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY j '
            '"javascript:alert(1)">]>' f'<svg xmlns="{self.SVG}"><a href="&j;">'
            '<text>x</text></a></svg>'))

    def test_a_remote_stylesheet_instruction(self):
        self.assertClasses("a.xml", '<?xml version="1.0"?><?xml-stylesheet '
                           'type="text/xsl" href="https://x.invalid/s.xsl"?>'
                           '<doc/>', {"html-remote-resource"})

    def test_a_malformed_svg_is_still_read(self):
        self.assertIn("html-script", self.classes(
            "a.svg", f'<svg xmlns="{self.SVG}"><script>alert(1)</script><p></svg>'))


# ---- mail ----------------------------------------------------------------

class MailTest(_Files):

    def test_an_ordinary_mail_carries_nothing(self):
        self.assertNone("a.eml", EML_TEXT)
        self.assertNone("b.eml", mail((HTML_PART, "<p>Hi <img src='cid:logo'>"
                                       "</p>"), attachment("r.pdf", plain_pdf())))

    def test_the_html_part(self):
        self.assertClasses("a.eml", mail((HTML_PART,
                                          "<p>Hi</p><script>alert(1)</script>"
                                          "<img src='https://t.invalid/p.gif'>")),
                           {"html-script", "html-remote-resource"})

    def test_executables_by_content_and_by_name(self):
        self.assertIn("mail-executable-attachment", self.classes(
            "a.eml", mail(attachment("Rechnung.pdf", MZ, "application/pdf"))))
        for name in ("run.js", "x.vbs", "x.ps1", "x.bat", "x.cmd", "x.hta",
                     "x.jar", "x.lnk", "x.scr"):
            with self.subTest(name):
                self.assertIn("mail-executable-attachment", self.classes(
                    "a.eml", mail(attachment(name, b"harmless"))))

    def test_an_encoded_name_does_not_hide_its_extension(self):
        for disposition in ('attachment; filename="=?utf-8?b?cnVuLmpz?="',
                            "attachment; filename*=utf-8''run%2Ejs"):
            with self.subTest(disposition):
                part = ("Content-Type: application/octet-stream\r\n"
                        f"Content-Disposition: {disposition}\r\n"
                        "Content-Transfer-Encoding: base64",
                        base64.encodebytes(b"harmless").decode())
                self.assertIn("mail-executable-attachment",
                              self.classes("a.eml", mail(part)))

    def test_an_office_attachment_with_macros(self):
        got = self.classes("a.eml", mail(attachment("a.docm", docx(
            ctype="application/vnd.ms-word.document.macroEnabled.main+xml",
            extra={"word/vbaProject.bin": b"VBA"}))))
        self.assertTrue({"mail-executable-attachment", "office-macros"} <= got)

    def test_attachments_are_read_as_what_they_are(self):
        self.assertIn("pdf-javascript", self.classes("a.eml", mail(attachment(
            "a.pdf", plain_pdf(b"/OpenAction << /S /JavaScript /JS (x) >>")))))
        self.assertIn("archive-executable", self.classes("a.eml", mail(
            attachment("a.zip", zipped({"tool.dat": MZ})))))

    def test_an_encrypted_part_is_not_checked(self):
        self.assertClasses("a.eml", mail(
            ("Content-Type: application/pkcs7-mime; smime-type=enveloped-data;"
             " name=smime.p7m\r\nContent-Transfer-Encoding: base64",
             base64.encodebytes(b"\x30\x80opaque").decode())), set(),
            checked=False)

    def test_every_message_of_a_mailbox(self):
        box = (b"From a@example.invalid Mon Sep  1 10:00:00 2026\n"
               + EML_TEXT.encode() + b"\n"
               + b"From b@example.invalid Mon Sep  1 11:00:00 2026\n"
               + mail((HTML_PART, "<script>alert(1)</script>")))
        self.assertIn("html-script", self.classes("a.mbox", box))


# ---- archives and legacy office ------------------------------------------

class ArchiveTest(_Files):

    def test_an_ordinary_archive_carries_nothing(self):
        self.assertNone("a.zip", zipped({"a.txt": "x", "b/c.csv": "1,2"}))
        self.assertNone("a.tar.gz", tgz({"a.txt": b"x"}))

    def test_executables_by_magic_and_scripts_by_name_say_which(self):
        found = self.found("a.zip", zipped({"tool.dat": MZ, "lib": ELF,
                                            "run.bat": "@echo off"}))
        self.assertEqual(found.metadata(), {"classes": ["archive-executable"],
                                            "checked": True})
        note = found.note()
        for said in ("pe", "elf", "by magic", ".bat", "by extension"):
            self.assertIn(said, note)
        self.assertClasses("a.tar.gz", tgz({"bin/x": ELF}), {"archive-executable"})

    def test_java(self):
        self.assertClasses("a.zip", zipped({"a/B.class": CLASS}), {"archive-java"})
        self.assertClasses("a.jar", zipped({"META-INF/MANIFEST.MF": "x",
                                            "a/B.class": CLASS}), {"archive-java"})

    def test_an_archive_inside_an_archive(self):
        self.assertClasses("a.zip", zipped({"inner.zip": zipped({"x": MZ})}),
                           {"archive-executable"})

    def test_what_cannot_be_listed_is_not_checked(self):
        self.assertClasses("a.zip", b"PK\x03\x04" + b"\x00" * 40, set(),
                           checked=False)
        self.assertClasses("a.7z", b"7z\xbc\xaf\x27\x1c" + b"\x00" * 40, set(),
                           checked=False)
        encrypted = bytearray(zipped({"secret.dat": b"x" * 100},
                                     zipfile.ZIP_STORED))
        encrypted[6] |= 1                     # the local header's flag
        at = encrypted.rfind(b"PK\x01\x02")
        encrypted[at + 8] |= 1                # and the directory's
        self.assertEqual(self.found("a.zip", bytes(encrypted)).metadata()
                         ["checked"], False)

    def test_the_budget_cuts_the_scan(self):
        data = zipped({"inner.zip": zipped({f"f{i}": b"x" * 3000
                                            for i in range(20)})})
        with mock.patch.object(active, "DECODE_BUDGET", 10_000):
            self.assertFalse(self.found("a.zip", data).metadata()["checked"])


class LegacyOfficeTest(_Files):

    def test_an_ordinary_compound_file_carries_nothing(self):
        self.assertNone("a.doc", cfb([(("WordDocument",), b"\x00" * 5000),
                                      (("\x01CompObj",), b"\x00" * 100)]))

    def test_vba_storages(self):
        self.assertClasses("a.doc", cfb([
            (("WordDocument",), b"\x00" * 5000), (("Macros",), None),
            (("Macros", "VBA"), None), (("Macros", "VBA", "dir"), b"\x01" * 80)]),
            {"office-macros"})

    def test_an_excel_4_macro_sheet(self):
        self.assertClasses("a.xls", cfb([(("Workbook",), workbook(
            boundsheet("Sheet1", 0), boundsheet("Macro1", 1)))]),
            {"office-macros"})
        self.assertNone("a.xls", cfb([(("Workbook",), workbook(
            boundsheet("Sheet1", 0), boundsheet("Chart1", 2)))]))

    def test_embedded_objects_and_dde(self):
        self.assertClasses("a.doc", cfb([
            (("WordDocument",), b"\x00" * 5000), (("ObjectPool",), None),
            (("ObjectPool", "_1234"), None),
            (("ObjectPool", "_1234", "\x01Ole10Native"), b"\x00" * 64)]),
            {"office-ole-object"})
        self.assertClasses("a.doc", cfb([(("WordDocument",), b"\x00" * 600
                                          + b"\x13 DDEAUTO c:\\cmd.exe x\x14"
                                          + b"\x00" * 5000)]), {"office-dde"})

    def test_a_presentation_with_vba_or_an_embedded_object(self):
        vba = (b"\x0f\x00\xff\x03\x14\x00\x00\x00\x02\x00\x00\x04\x0c\x00\x00\x00"
               + b"\x00" * 12)
        ole = b"\x10\x00\x11\x10" + b"\x20\x00\x00\x00" * 2 + b"\x78\x9c"
        noise = bytes(range(256)) * 16 + b"\x00\x00\x11\x10" * 64
        for extra, expected in ((vba, {"office-macros"}),
                                (ole, {"office-ole-object"}), (b"", set())):
            with self.subTest(expected):
                stream = noise + extra
                stream += b"\x00" * (8192 - len(stream))
                self.assertClasses("a.ppt", cfb([(("PowerPoint Document",),
                                                  stream)]), expected)

    def test_an_outlook_message_with_an_executable_attachment(self):
        att = "__attach_version1.0_#00000000"
        self.assertIn("mail-executable-attachment", self.classes("a.msg", cfb([
            (("__substg1.0_0037001F",), "Subject".encode("utf-16-le")),
            ((att,), None),
            ((att, "__substg1.0_3707001F"), "invoice.pdf".encode("utf-16-le")),
            ((att, "__substg1.0_37010102"), MZ)])))

    def test_a_damaged_compound_file_is_not_checked(self):
        self.assertClasses("a.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
                           + b"\xff" * 600, set(), checked=False)


# ---- where it is said ----------------------------------------------------

class SaidTest(_Files):

    def test_every_listed_kind_carries_the_field_and_no_other(self):
        d = Path(self._dir.name)
        for name, data in (("a.pdf", plain_pdf()), ("a.html", "<p>x</p>"),
                           ("a.svg", "<svg xmlns='http://www.w3.org/2000/svg'/>"),
                           ("a.eml", EML_TEXT), ("a.zip", zipped({"a": "b"})),
                           ("a.rtf", "{\\rtf1 x}"), ("a.docx", docx()),
                           ("a.odt", odt()), ("a.doc", cfb([(("x",), b"1")]))):
            p = self.path(name, data)
            meta = extract(p, detect(p), 25_000_000).metadata
            self.assertEqual(meta.get("active_content"),
                             {"classes": [], "checked": True}, name)
        for name, data in (("a.txt", "hello"), ("a.csv", "a,b\n1,2\n"),
                           ("a.json", '{"a": 1}')):
            p = self.path(name, data)
            self.assertNotIn("active_content",
                             extract(p, detect(p), 25_000_000).metadata, name)
        del d

    def test_the_note_and_the_json(self):
        p = self.path("a.html", "<p>Hi</p><script>alert(1)</script>")
        summ = summarize_path(p, Config(no_llm=True, cache=False))
        self.assertEqual(summ.metadata["active_content"],
                         {"classes": ["html-script"], "checked": True})
        self.assertIn("this file carries active content (html-script); "
                      "a host application does not render it", summ.notes)
        self.assertEqual(summ.to_dict()["metadata"]["active_content"],
                         {"classes": ["html-script"], "checked": True})

    def test_a_file_not_checked_says_so(self):
        p = self.path("a.7z", b"7z\xbc\xaf\x27\x1c" + b"\x00" * 40)
        ex = extract(p, detect(p), 25_000_000)
        self.assertEqual(ex.metadata["active_content"],
                         {"classes": [], "checked": False})
        self.assertTrue(any("could not be checked for active content" in n
                            for n in ex.notes), ex.notes)

    def test_the_prompt_is_not_changed(self):
        meta = {"title": "t", "links": 2}
        with_it = dict(meta, active_content={"classes": ["html-script"],
                                             "checked": True})
        self.assertEqual(render_metadata(meta), render_metadata(with_it))
        kwargs = dict(filename="a.html", kind="html", mime="text/html",
                      content="x")
        strip = lambda s: s.split("<<<FILE-CONTENT")[0]
        self.assertEqual(strip(build_user_prompt(metadata=meta, **kwargs)),
                         strip(build_user_prompt(metadata=with_it, **kwargs)))

    def test_the_metadata_survives_the_redaction_boundary(self):
        value = {"classes": sorted(active.CLASSES), "checked": False}
        self.assertEqual(scrub({"active_content": value})["active_content"], value)

    def test_the_field_is_not_a_fact_about_the_contents(self):
        self.assertFalse(_has_structure({"active_content": {
            "classes": ["html-script"], "checked": True}}))

    def test_a_scanner_that_fails_does_not_fail_the_file(self):
        p = self.path("a.html", "<p>x</p>")
        with mock.patch.object(active, "_html", side_effect=TypeError("boom")):
            ex = extract(p, detect(p), 25_000_000)
        self.assertEqual(ex.metadata["active_content"],
                         {"classes": [], "checked": False})
        self.assertIn("x", ex.text)

    def test_the_scan_gets_a_share_of_the_time_and_no_more(self):
        p = self.path("a.html", "<p>x</p>")

        def slow(*args):
            while True:
                deadline.check("a test")

        with mock.patch.object(active, "_html", side_effect=slow), \
                deadline.limit(0.4):
            began = time.monotonic()
            found = active.scan(p, detect(p), 25_000_000)
            spent = time.monotonic() - began
            self.assertFalse(deadline.spent())
        self.assertFalse(found.metadata()["checked"])
        self.assertLess(spent, 0.35)

    def test_the_class_names_are_the_interface(self):
        self.assertEqual(active.CLASSES, frozenset({
            "pdf-javascript", "pdf-open-action", "pdf-launch", "pdf-submit-form",
            "pdf-import-data", "pdf-embedded-file", "pdf-richmedia", "pdf-xfa",
            "pdf-goto-remote", "office-macros", "office-activex",
            "office-ole-object", "office-dde", "office-external-template",
            "odf-scripts", "rtf-object", "html-script", "html-event-handler",
            "html-javascript-url", "html-frame", "html-remote-resource",
            "mail-executable-attachment", "archive-executable",
            "archive-java"}))


class BytesPatternTest(unittest.TestCase):
    """test_backtracking tries every pattern of the package on hostile
    text; the scan's patterns over bytes need hostile bytes."""

    def test_no_bytes_pattern_is_super_linear(self):
        import re
        n = 30_000
        baits = [b" " * n, b"1" * n, b"1 %" * (n // 3), b"1 " * (n // 2),
                 b"/" + b"a" * n, b"/a#" * (n // 3), b"(" * n, b"\\" * n,
                 b"<" * n, b"%" * n, b"obj" * (n // 3), b"1 0 obj " * (n // 8),
                 b"stream " * (n // 7), b"\x13" + b" " * n, b"#" * n,
                 b"\\object" * (n // 7), b"\\'" * n, b"{" * n,
                 b"\x00\x00\x11\x10" * (n // 4), b"/Length 1 " * (n // 10)]
        patterns = [(name, value) for name, value in vars(active).items()
                    if isinstance(value, re.Pattern)
                    and isinstance(value.pattern, bytes)]
        self.assertGreater(len(patterns), 10)
        slow = []
        for name, pattern in patterns:
            for bait in baits:
                began = time.process_time()
                pattern.search(bait)
                spent = time.process_time() - began
                if spent > 1.0:
                    slow.append(f"{name}: {spent:.1f}s on {bait[:12]!r}")
                    break
        self.assertEqual([], slow)


class LinearTest(_Files):
    """CPU time grows with the size, not faster: GROWTH times the file costs
    about GROWTH times the time, and less than GROWTH ** 1.5 -- halfway, on a
    log scale, between linear and quadratic. The short and the long run take
    turns, so a burst of load on the host slows both alike, and the best of
    five of each is compared."""

    GROWTH = 8

    def _cpu(self, p):
        det = detect(p)
        began = time.process_time()
        active.scan(p, det, 100_000_000)
        return time.process_time() - began

    def _ratio(self, name, make, unit):
        short, long_ = self.path("s" + name, make(unit)), \
            self.path("l" + name, make(unit * self.GROWTH))
        times = {short: [], long_: []}
        for _ in range(5):
            for p in (short, long_):
                times[p].append(self._cpu(p))
        return min(times[long_]) / max(min(times[short]), 1e-4)

    def test_html_pdf_and_zip_are_linear(self):
        page = ("<div class=x onclick='f()'><a href='https://x.invalid/a'>a</a>"
                "<!-- c --><script>var s='<p>';</script><style>p{}</style>"
                "<img src=a.png srcset='b.png 2x'></div>\n")
        cases = {
            ".html": (lambda n: page * n, 2000),
            ".pdf": (lambda n: pdf([catalog()] + PAGES + [
                b"<< /Type /Annot /A << /S /URI /URI (http://x/%d) >> /T (s) >>"
                % i for i in range(n)]), 2000),
            ".zip": (lambda n: zipped({f"d/f{i}.txt": "x" for i in range(n)}),
                     500),
        }
        for ext, (make, unit) in cases.items():
            with self.subTest(ext):
                ratio = self._ratio(ext, make, unit)
                self.assertLess(ratio, self.GROWTH ** 1.5,
                                f"{ext}: {self.GROWTH}x the size cost "
                                f"{ratio:.1f}x")


# ---- how far the scan reads ------------------------------------------------

class ScanReachTest(_Files):
    """A viewer may hand a renderer more of a file than sift reads: a host application
    gives pdf.js a PDF of up to 64 MiB, sift reads 25 MB of it, and the scan
    of a clean 30 MB PDF said "not checked" -- its pages were refused.
    ``active_max_bytes`` asks the scan alone to read further; the extractors
    still read ``max_bytes``."""

    #: A PDF whose open action is JavaScript, in an object after 5,000
    #: bytes of filler: past a max_bytes of 1,000.
    LATE_JS = plain_pdf(b"/OpenAction 6 0 R", more=[
        b"<< /Filler (%s) >>" % (b"x" * 5000),
        b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"])
    CLEAN = plain_pdf(more=[b"<< /Filler (%s) >>" % (b"x" * 5000)])

    def extracted(self, name, data, max_bytes, active_bytes=0):
        p = self.path(name, data)
        return extract(p, detect(p), max_bytes, active_bytes=active_bytes)

    def test_the_scan_reads_as_far_as_it_is_asked(self):
        cut = self.extracted("a.pdf", self.CLEAN, 1000)
        self.assertEqual(cut.metadata["active_content"],
                         {"classes": [], "checked": False})
        whole = self.extracted("a.pdf", self.CLEAN, 1000, active_bytes=10_000)
        self.assertEqual(whole.metadata["active_content"],
                         {"classes": [], "checked": True})
        self.assertFalse(any("active content" in n for n in whole.notes),
                         whole.notes)
        self.assertEqual(whole.text, cut.text)   # the extractor reads 1,000

    def test_javascript_past_max_bytes_is_found_where_the_scan_reaches(self):
        cut = self.extracted("a.pdf", self.LATE_JS, 1000)
        self.assertNotIn("pdf-javascript", cut.metadata["active_content"]["classes"])
        self.assertFalse(cut.metadata["active_content"]["checked"])
        whole = self.extracted("a.pdf", self.LATE_JS, 1000, active_bytes=len(self.LATE_JS))
        self.assertEqual(whole.metadata["active_content"],
                         {"classes": ["pdf-javascript", "pdf-open-action"], "checked": True})

    def test_a_reach_short_of_the_file_is_still_not_checked(self):
        got = self.extracted("a.pdf", self.CLEAN, 1000, active_bytes=len(self.CLEAN) - 1)
        self.assertFalse(got.metadata["active_content"]["checked"])
        exact = self.extracted("a.pdf", self.CLEAN, 1000, active_bytes=len(self.CLEAN))
        self.assertTrue(exact.metadata["active_content"]["checked"])

    def test_the_reach_is_never_less_than_max_bytes(self):
        got = self.extracted("a.pdf", self.CLEAN, 25_000_000, active_bytes=100)
        self.assertTrue(got.metadata["active_content"]["checked"])
        for max_bytes, asked, reach in ((1000, 0, 1000), (1000, 5000, 5000), (5000, 1000, 5000),
                                        (1000, -3, 1000), (0, 5000, 0), (0, 0, 0)):
            with self.subTest(max_bytes=max_bytes, asked=asked):
                self.assertEqual(active.reach(max_bytes, asked), reach)

    def test_summarize_passes_it_on_and_the_text_stays_cut(self):
        p = self.path("a.pdf", self.CLEAN)
        summ = summarize_path(p, Config(no_llm=True, cache=False, max_bytes=1000,
                                        active_max_bytes=10_000))
        self.assertEqual(summ.metadata["active_content"], {"classes": [], "checked": True})
        self.assertFalse(any("active content" in n for n in summ.notes), summ.notes)
        self.assertTrue(summ.truncated)            # the extractors read 1,000
        default = summarize_path(p, Config(no_llm=True, cache=False, max_bytes=1000))
        self.assertFalse(default.metadata["active_content"]["checked"])

    def test_a_negative_reach_is_refused_like_a_negative_max_bytes(self):
        from sift.config import ConfigError, validate
        validate(Config(active_max_bytes=0))
        validate(Config(active_max_bytes=64 << 20))
        with self.assertRaises(ConfigError):
            validate(Config(active_max_bytes=-5))


if __name__ == "__main__":
    unittest.main()
