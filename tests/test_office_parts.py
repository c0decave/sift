"""The parts of an office document besides its body: a secret there is a
secret in the file.

A census of real files (2026-09-29) measured 5 of 18 real .docx
files with words in their headers, footers or footnotes that sift's text
did not carry (header words 3 of 3 and 7 of 7 missing, footer 4 of 5,
footnotes 8 of 15): _docx read word/document.xml and nothing else. Each of
these is a file whose secret find_secrets never saw, and a host application showed
sift's text as the document's text. The same class in xlsx: its cell
comments and its sheets' headers and footers were not read either; in
the siblings: a pptx's speaker notes, an odt's or ods's master-page
headers and footers, an odp's header and footer declarations. And the
text read beside where it belongs: an ods comment glued to its cell's
value, a table of a nest of three docx tables not rendered at all.
"""
import dataclasses
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from sift import zipguard
from sift.config import Config
from sift.detect import detect
from sift.extract import extract, office
from sift.extract.base import find_secrets
from sift.llm.base import LLMResult
from sift.report import render_json, render_markdown, render_text
from sift.summarize import summarize_path

SECRET = "Kx9vQ2mZ7pL4Wq8rTz"
LINE = f"password: {SECRET}"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
SS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
ODF_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
          'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:'
          'presentation:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/"')
BODY = "Body text of the quarterly report."


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write(path: Path, files: dict) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _spans_hold(test, text: str, name: str, secrets) -> None:
    """Every span find_secrets reports lies inside ``text``, none overlaps
    the next, and each of ``secrets`` is the text of one of them at its
    offsets: a host application masks and highlights by these offsets."""
    spans = find_secrets(text, name)
    shown = [s for s in spans if s.reason != "bidi"]
    for span in shown:
        test.assertTrue(0 <= span.start < span.end <= len(text), span)
    for one, two in zip(shown, shown[1:]):
        test.assertLessEqual(one.end, two.start)
    found = {text[s.start:s.end] for s in shown}
    for secret in secrets:
        test.assertIn(secret, found, f"{secret} not at a span of {name}")


# ---- docx ---------------------------------------------------------------
def _para(text: str) -> str:
    return (f'<w:p><w:r><w:t xml:space="preserve">{_escape(text)}</w:t>'
            f"</w:r></w:p>")


def _hdr(*lines: str, root: str = "hdr") -> str:
    return (f'<w:{root} xmlns:w="{W}">' + "".join(_para(t) for t in lines)
            + f"</w:{root}>")


def _notes(kind: str, *items) -> str:
    """A footnotes, endnotes or comments part: Word's two separator notes
    first where it writes them, then one item per entry of ``items``, each
    a tuple of its paragraphs."""
    one = {"footnotes": "footnote", "endnotes": "endnote",
           "comments": "comment"}[kind]
    head = ""
    if kind != "comments":
        head = (f'<w:{one} w:type="separator" w:id="-1"><w:p><w:r>'
                f'<w:separator/></w:r></w:p></w:{one}><w:{one} '
                f'w:type="continuationSeparator" w:id="0"><w:p><w:r>'
                f"<w:continuationSeparator/></w:r></w:p></w:{one}>")
    body = "".join(
        f'<w:{one} w:id="{n + 1}" w:author="Reviewer">'
        + "".join(_para(t) for t in paragraphs) + f"</w:{one}>"
        for n, paragraphs in enumerate(items))
    return f'<w:{kind} xmlns:w="{W}">{head}{body}</w:{kind}>'


def _docx(parts=(), references=None, rels=True, body=(BODY,), extra=None):
    """The members of a docx: ``parts`` are (type, target, xml) with a
    relationship each (rId1, rId2, ...), ``references`` the ids the body's
    section properties name, in order (default: every header and footer in
    ``parts`` order), ``rels`` False for no relationships part at all, and
    ``extra`` more members as they are."""
    files, entries, refs = {}, [], []
    for index, (kind, target, xml) in enumerate(parts, 1):
        rid = f"rId{index}"
        entries.append(f'<Relationship Id="{rid}" Type="{REL}{kind}" '
                       f'Target="{target}"/>')
        if xml is not None:
            files[f"word/{target}"] = xml
        if kind in ("header", "footer"):
            refs.append((kind, rid))
    if references is not None:
        refs = references
    sect = "".join(f'<w:{kind}Reference w:type="default" r:id="{rid}"/>'
                   for kind, rid in refs)
    files["word/document.xml"] = (
        f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>'
        + "".join(_para(t) for t in body)
        + (f"<w:sectPr>{sect}</w:sectPr>" if sect else "")
        + "</w:body></w:document>")
    if rels:
        files["word/_rels/document.xml.rels"] = (
            f'<Relationships xmlns="{PR}">{"".join(entries)}</Relationships>')
    files.update(extra or {})
    return files


class _Office(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def path(self, name: str, files: dict) -> Path:
        return _write(self.dir / name, files)

    def extracted(self, name: str, files: dict):
        path = self.path(name, files)
        return extract(path, detect(path), 25_000_000)

    def assert_found(self, name: str, files: dict) -> None:
        """The secret is in the text, at a span of find_secrets, counted in
        the note on credentials and repeated in nothing sift says."""
        path = self.path(name, files)
        ex = extract(path, detect(path), 25_000_000)
        self.assertIn(SECRET, ex.text)
        _spans_hold(self, ex.text, name, [SECRET])
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None,
                              keep_text=True)
        self.assertEqual(ex.text, summ.text)
        notes = [f["text"] for f in summ.findings
                 if f.get("kind") == "credential"]
        self.assertEqual(1, len(notes), summ.findings)
        said = dataclasses.asdict(summ)
        said.pop("text")
        self.assertNotIn(SECRET, json.dumps(said, ensure_ascii=False))

    def assert_spans_are_the_secret(self, text: str, name: str) -> None:
        """Every span find_secrets reports in ``text`` is the secret, no
        more and no less: what is glued to it -- a cell's value, a
        comment's author, the word after a tab -- is no part of it."""
        spans = [s for s in find_secrets(text, name) if s.reason != "bidi"]
        self.assertTrue(spans, text)
        self.assertEqual({SECRET}, {text[s.start:s.end] for s in spans},
                         text)

    def assert_hidden_from_a_model(self, name: str, files: dict) -> None:
        """A model that repeats the bare secret has it hidden in all sift
        says: sift learned the secret itself from its text, not the secret
        with the text glued to it."""
        path = self.path(name, files)
        summ = summarize_path(path, Config(quiet=True, cache=False),
                              _Echo(), None)
        said = {"text": render_text(summ, color=False),
                "json": render_json([summ]),
                "markdown": render_markdown([summ])}
        self.assertEqual({}, {k: v for k, v in said.items() if SECRET in v})


class _Echo:
    """A model that repeats the bare secret back."""

    name = provider = model = "stub"
    endpoint = "memory://stub"

    def available(self):
        return True

    def list_models(self):
        return ["stub"]

    def generate(self, system, user, *, images=None, temperature=0.2,
                 max_tokens=800):
        return LLMResult(
            text=(f"TYPE: Document\nSUMMARY: The comment reads {SECRET}.\n"
                  f"KEY POINTS:\n- {SECRET}\nENTITIES: {SECRET}\n"
                  f"NOTES: none"), model="stub", provider="stub")


class DocxPartsTest(_Office):
    PARTS = {
        "header": ("header", "header1.xml", _hdr(LINE)),
        "footer": ("footer", "footer1.xml", _hdr(LINE, root="ftr")),
        "footnotes": ("footnotes", "footnotes.xml",
                      _notes("footnotes", (LINE,))),
        "endnotes": ("endnotes", "endnotes.xml", _notes("endnotes", (LINE,))),
        "comments": ("comments", "comments.xml", _notes("comments", (LINE,))),
    }

    def test_a_secret_in_each_part_is_found(self):
        for kind, part in self.PARTS.items():
            with self.subTest(part=kind):
                self.assert_found(f"{kind}.docx", _docx([part]))

    def test_the_body_comes_first_and_is_unchanged(self):
        """The parts follow the body: the text a docx without them gave is
        where it was, so every offset into it is too."""
        plain = self.extracted("plain.docx", _docx()).text
        ex = self.extracted("parts.docx", _docx(list(self.PARTS.values())))
        self.assertTrue(ex.text.startswith(plain + "\n\n## Header\n"),
                        ex.text)

    def test_the_parts_follow_in_the_documents_order(self):
        """Headers and footers in the order the section properties name
        them, then the footnotes, the endnotes and the comments."""
        ex = self.extracted("order.docx", _docx(
            [("header", "header1.xml", _hdr("Header one")),
             ("header", "header2.xml", _hdr("Header two")),
             ("footer", "footer1.xml", _hdr("Footer one", root="ftr")),
             ("comments", "comments.xml",
              _notes("comments", ("First comment",), ("Second comment",))),
             ("endnotes", "endnotes.xml", _notes("endnotes", ("An endnote",))),
             ("footnotes", "footnotes.xml",
              _notes("footnotes", ("A footnote",), ("Another footnote",)))],
            references=[("header", "rId2"), ("footer", "rId3"),
                        ("header", "rId1")]))
        expected = [BODY, "## Header", "Header two", "Header one",
                    "## Footer", "Footer one", "## Footnotes", "A footnote",
                    "Another footnote", "## Endnotes", "An endnote",
                    "## Comments", "First comment", "Second comment"]
        self.assertEqual(expected, [ln for ln in ex.text.splitlines() if ln])
        self.assertEqual((2, 1, 2), (ex.metadata["footnotes"],
                                     ex.metadata["endnotes"],
                                     ex.metadata["comments"]))
        self.assertEqual(1, ex.metadata["paragraphs"])

    def test_a_part_no_relationship_names_is_read(self):
        """A member Word would name header1.xml, with no relationship to
        it: its text is in the file, so it is read, after the named ones."""
        ex = self.extracted("orphan.docx", _docx(
            [("header", "header2.xml", _hdr("Named header"))],
            extra={"word/header1.xml": _hdr(LINE),
                   "word/comments.xml": _notes("comments", ("A comment",))}))
        lines = [ln for ln in ex.text.splitlines() if ln]
        self.assertEqual([BODY, "## Header", "Named header", LINE,
                          "## Comments", "A comment"], lines)

    def test_without_relationships_the_names_give_the_order(self):
        """No word/_rels/document.xml.rels at all: Word's member names, in
        the order of their numbers (header2 before header10)."""
        ex = self.extracted("norels.docx", _docx(rels=False, extra={
            "word/header10.xml": _hdr("Tenth"),
            "word/header2.xml": _hdr("Second"),
            "word/footer1.xml": _hdr("Foot", root="ftr"),
            "word/footnotes.xml": _notes("footnotes", ("Note",))}))
        self.assertEqual([BODY, "## Header", "Second", "Tenth", "## Footer",
                          "Foot", "## Footnotes", "Note"],
                         [ln for ln in ex.text.splitlines() if ln])

    def test_a_relationship_to_a_missing_member_is_passed_over(self):
        ex = self.extracted("missing.docx", _docx(
            [("header", "header9.xml", None),
             ("footer", "footer1.xml", _hdr(LINE, root="ftr"))]))
        self.assertEqual([BODY, "## Footer", LINE],
                         [ln for ln in ex.text.splitlines() if ln])
        self.assertEqual([], ex.notes)

    def test_a_relationship_out_of_the_package_is_passed_over(self):
        """A target that climbs out of the package or is external names no
        member of it."""
        files = _docx([("header", "../../header1.xml", None),
                       ("footer", "footer1.xml", _hdr("Foot", root="ftr"))])
        files["word/_rels/document.xml.rels"] = files[
            "word/_rels/document.xml.rels"].replace(
                "</Relationships>",
                f'<Relationship Id="rId7" Type="{REL}header" '
                f'Target="http://example.org/h.xml" TargetMode="External"/>'
                "</Relationships>")
        ex = self.extracted("outside.docx", files)
        self.assertEqual([BODY, "## Footer", "Foot"],
                         [ln for ln in ex.text.splitlines() if ln])

    def test_a_part_that_is_not_well_formed_leaves_the_rest(self):
        """The body and every other part are read, and a note says one part
        could not be."""
        ex = self.extracted("broken.docx", _docx(
            [("header", "header1.xml", f'<w:hdr xmlns:w="{W}"><w:p>'),
             ("footer", "footer1.xml", _hdr(LINE, root="ftr")),
             ("comments", "comments.xml", b"\x00\xff not xml")]))
        self.assertEqual([BODY, "## Footer", LINE],
                         [ln for ln in ex.text.splitlines() if ln])
        self.assertEqual(1, len(ex.notes), ex.notes)
        self.assertIn("2 of the document's header, footer, note and comment "
                      "parts could not be read", ex.notes[0])

    def test_broken_relationships_fall_back_to_the_names(self):
        files = _docx(extra={"word/header1.xml": _hdr(LINE)})
        files["word/_rels/document.xml.rels"] = "<Relationships"
        ex = self.extracted("badrels.docx", files)
        self.assertEqual([BODY, "## Header", LINE],
                         [ln for ln in ex.text.splitlines() if ln])

    def test_an_empty_header_adds_nothing(self):
        plain = self.extracted("plain.docx", _docx())
        for name, part in (
                ("empty", ("header", "header1.xml", _hdr(""))),
                ("blank", ("header", "header1.xml", _hdr("   ", ""))),
                ("bare", ("header", "header1.xml", f'<w:hdr xmlns:w="{W}"/>')),
                ("separators", ("footnotes", "footnotes.xml",
                                _notes("footnotes")))):
            with self.subTest(part=name):
                ex = self.extracted(f"{name}.docx", _docx([part]))
                self.assertEqual(plain.text, ex.text)
                self.assertEqual(plain.metadata, ex.metadata)
                self.assertEqual([], ex.notes)

    def test_a_part_named_twice_is_read_once(self):
        """Two relationships to one member, both named by the sections:
        its text once. Headers that repeat each other's text word for word
        (a first-page header as the default one) are written once too."""
        ex = self.extracted("twice.docx", _docx(
            [("header", "header1.xml", _hdr("Company confidential")),
             ("header", "header1.xml", None),
             ("header", "header2.xml", _hdr("Company confidential")),
             ("header", "header3.xml", _hdr("Draft"))]))
        self.assertEqual([BODY, "## Header", "Company confidential", "Draft"],
                         [ln for ln in ex.text.splitlines() if ln])

    def test_breaks_tabs_and_text_boxes(self):
        """A break and a tab as the body reader keeps them; a text box
        inside a paragraph is a line of its own, its text not also part of
        the paragraph around it."""
        box = (f"<w:p><w:r><w:t>Before</w:t></w:r><w:r><w:pict>"
               f"<w:txbxContent>{_para('Boxed')}</w:txbxContent></w:pict>"
               f"</w:r><w:r><w:t>after</w:t><w:tab/><w:t>tabbed</w:t><w:br/>"
               f"<w:t>broken</w:t></w:r></w:p>")
        ex = self.extracted("box.docx", _docx(
            [("header", "header1.xml", f'<w:hdr xmlns:w="{W}">{box}</w:hdr>')]))
        tail = ex.text.split("## Header\n", 1)[1]
        self.assertEqual("Beforeafter\ttabbed\nbroken\nBoxed", tail)

    def test_a_deep_nest_of_text_boxes_is_read_in_linear_time(self):
        """Every paragraph's text is read once: an iter() under each of
        5,000 nested paragraphs read 12.5 million elements."""
        depth = 5_000
        nest = ("<w:p><w:r><w:t>x</w:t><w:pict><w:txbxContent>" * depth
                + "</w:txbxContent></w:pict></w:r></w:p>" * depth)
        start = time.monotonic()
        ex = self.extracted("nest.docx", _docx(
            [("header", "header1.xml", f'<w:hdr xmlns:w="{W}">{nest}</w:hdr>')]))
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertEqual(depth, ex.text.split("## Header\n", 1)[1].count("x"))

    def test_ten_thousand_header_parts_stay_bounded(self):
        """As many parts as zipguard lets a zip hold, each named by a
        relationship and a section: reading stops once the text is past
        MAX_CHARS, which is all clip keeps."""
        count = zipguard.MAX_ZIP_ENTRIES - 3
        line = "A header line of forty characters, long."
        files = _docx([("header", f"header{n}.xml", _hdr(f"{line} {n}"))
                       for n in range(1, count + 1)])
        self.assertLessEqual(len(files), zipguard.MAX_ZIP_ENTRIES)
        reads = []
        real = office._read_member

        def counted(zf, name):
            reads.append(name)
            return real(zf, name)
        start = time.monotonic()
        with mock.patch.object(office, "_read_member", counted):
            ex = self.extracted("many.docx", files)
        self.assertLess(time.monotonic() - start, 20.0)
        self.assertTrue(ex.truncated)
        self.assertLessEqual(len(ex.text),
                             office.MAX_CHARS + len("\n…[truncated]"))
        headers = [n for n in reads if n.startswith("word/header")]
        self.assertLessEqual(len(headers),
                             office.MAX_CHARS // len(line) + 2)
        self.assertEqual(len(headers), len(set(headers)))

    def test_ten_thousand_empty_header_parts_stay_bounded(self):
        """Parts without text never fill MAX_CHARS: each is read once, as
        many as the zip may hold, and the clock is looked at per part."""
        count = zipguard.MAX_ZIP_ENTRIES - 3
        files = _docx([("header", f"header{n}.xml", _hdr(""))
                       for n in range(1, count + 1)])
        checks = []
        real = office.deadline.check

        def counted(where=""):
            checks.append(where)
            return real(where)
        start = time.monotonic()
        with mock.patch.object(office.deadline, "check", counted):
            ex = self.extracted("empty.docx", files)
        self.assertLess(time.monotonic() - start, 20.0)
        self.assertEqual(BODY, ex.text)
        self.assertGreaterEqual(len(checks), count)

    def test_many_relationships_to_one_member_read_it_once(self):
        big = _hdr(*[f"Line {n} of a long header" for n in range(20_000)])
        rels = "".join(f'<Relationship Id="rId{n}" Type="{REL}header" '
                       f'Target="header1.xml"/>' for n in range(20_000))
        files = _docx(extra={"word/header1.xml": big})
        files["word/_rels/document.xml.rels"] = (
            f'<Relationships xmlns="{PR}">{rels}</Relationships>')
        reads = []
        real = office._read_member

        def counted(zf, name):
            reads.append(name)
            return real(zf, name)
        with mock.patch.object(office, "_read_member", counted):
            self.extracted("once.docx", files)
        self.assertEqual(1, reads.count("word/header1.xml"))

    def test_the_parts_share_the_archive_budget(self):
        """What the zip declares for the body and the parts together is
        held to _MAX_TOTAL_INFLATED, as an epub's chapters and a workbook's
        sheets are; the parts past it are not read, and the metadata says
        so."""
        files = _docx([("header", "header1.xml", _hdr("First header")),
                       ("footer", "footer1.xml", _hdr(LINE, root="ftr"))])
        path = self.path("budget.docx", files)
        with zipfile.ZipFile(path) as zf:
            body = zf.getinfo("word/document.xml").file_size
            rels = zf.getinfo("word/_rels/document.xml.rels").file_size
            first = zf.getinfo("word/header1.xml").file_size
        with mock.patch.object(office, "_MAX_TOTAL_INFLATED",
                               body + rels + first):
            ex = extract(path, detect(path), 25_000_000)
        self.assertIn("First header", ex.text)
        self.assertNotIn(SECRET, ex.text)
        self.assertIn("were not read", ex.metadata["read_limit"])

    def test_every_span_lies_on_the_extracted_text(self):
        """A secret in the body and one in each part: each is found at its
        offsets into sift's text, the text a host application shows and masks."""
        secrets = [f"Zq{n}vX8mT2rL5wK9pB{n}" for n in range(6)]
        files = _docx([
            ("header", "header1.xml", _hdr(f"token = {secrets[1]}")),
            ("footer", "footer1.xml",
             _hdr("Page 1", f"api_key: {secrets[2]}", root="ftr")),
            ("footnotes", "footnotes.xml",
             _notes("footnotes", (f"The password is {secrets[3]}.",))),
            ("endnotes", "endnotes.xml",
             _notes("endnotes", (f"secret={secrets[4]}",))),
            ("comments", "comments.xml",
             _notes("comments", ("Looks fine.", f"password: {secrets[5]}")))],
            body=(BODY, f"Vault password: {secrets[0]}"))
        path = self.path("all.docx", files)
        ex = extract(path, detect(path), 25_000_000)
        _spans_hold(self, ex.text, path.name, secrets)
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None,
                              keep_text=True)
        _spans_hold(self, summ.text, path.name, secrets)

    def test_innocent_parts_add_no_span(self):
        """Precision: the headings and the page furniture of an ordinary
        document are not secrets."""
        ex = self.extracted("innocent.docx", _docx([
            ("header", "header1.xml", _hdr("ACME Corp — Internal")),
            ("footer", "footer1.xml", _hdr("Page 1 of 3", root="ftr")),
            ("footnotes", "footnotes.xml",
             _notes("footnotes", ("See the annual report, p. 12.",))),
            ("comments", "comments.xml",
             _notes("comments", ("Please check the figures.",)))]))
        self.assertEqual([], find_secrets(ex.text, "innocent.docx"))


# ---- xlsx ---------------------------------------------------------------
def _xlsx(sheets, comments=None, orphans=None):
    """``sheets``: (name, rows xml, header/footer xml or ""). ``comments``
    maps a sheet's index to its comments part's xml, related from the
    sheet; ``orphans`` are comments parts nothing relates, by member."""
    comments = comments or {}
    book = "".join(f'<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>'
                   for n, (name, _, _) in enumerate(sheets, 1))
    rels = "".join(f'<Relationship Id="rId{n}" Type="{REL}worksheet" '
                   f'Target="worksheets/sheet{n}.xml"/>'
                   for n in range(1, len(sheets) + 1))
    files = {"xl/workbook.xml": f'<workbook xmlns="{SS}" xmlns:r="{R}">'
                                f"<sheets>{book}</sheets></workbook>",
             "xl/_rels/workbook.xml.rels":
                 f'<Relationships xmlns="{PR}">{rels}</Relationships>'}
    for n, (_, rows, head_foot) in enumerate(sheets, 1):
        files[f"xl/worksheets/sheet{n}.xml"] = (
            f'<worksheet xmlns="{SS}"><sheetData>{rows}</sheetData>'
            f"{head_foot}</worksheet>")
        if n - 1 in comments:
            files[f"xl/comments{n}.xml"] = comments[n - 1]
            files[f"xl/worksheets/_rels/sheet{n}.xml.rels"] = (
                f'<Relationships xmlns="{PR}"><Relationship Id="rId1" '
                f'Type="{REL}comments" Target="../comments{n}.xml"/>'
                f"</Relationships>")
    files.update(orphans or {})
    return files


def _cell(ref: str, text: str) -> str:
    return (f'<c r="{ref}" t="inlineStr"><is><t>{_escape(text)}</t></is>'
            f"</c>")


def _comments(*entries) -> str:
    """(ref, text) pairs, as Excel writes them: the author in a bold run,
    the text in the next."""
    body = "".join(
        f'<comment ref="{ref}" authorId="0"><text><r><rPr><b/></rPr>'
        f'<t>Reviewer:</t></r><r><t xml:space="preserve">\n{_escape(text)}'
        f"</t></r></text></comment>" for ref, text in entries)
    return (f'<comments xmlns="{SS}"><authors><author>Reviewer</author>'
            f"</authors><commentList>{body}</commentList></comments>")


ROWS = f'<row r="1">{_cell("A1", "Item")}{_cell("B1", "Qty")}</row>'


class XlsxPartsTest(_Office):
    def test_a_secret_in_a_cell_comment_is_found(self):
        self.assert_found("comment.xlsx", _xlsx(
            [("Data", ROWS, "")], comments={0: _comments(("B1", LINE))}))

    def test_a_secret_in_a_sheet_header_or_footer_is_found(self):
        for kind in ("oddHeader", "oddFooter", "evenHeader", "firstFooter"):
            with self.subTest(kind=kind):
                value = _escape(f'&L&"Arial,Bold"&12{LINE}&RPage &P of &N')
                self.assert_found(f"{kind}.xlsx", _xlsx([(
                    "Data", ROWS,
                    f"<headerFooter><{kind}>{value}</{kind}></headerFooter>")]))

    def test_comments_headers_and_footers_follow_every_sheet(self):
        """The sheets as they were, then per sheet in the workbook's order
        its header, its footer and its comments, each under a heading
        _own_text leaves out as sift's."""
        plain = self.extracted("plain.xlsx", _xlsx(
            [("Data", ROWS, ""), ("More", ROWS, "")]))
        hf = ("<headerFooter><oddHeader>&amp;CQuarterly figures</oddHeader>"
              "<oddFooter>&amp;LPage &amp;P</oddFooter></headerFooter>")
        ex = self.extracted("parts.xlsx", _xlsx(
            [("Data", ROWS, hf), ("More", ROWS, "")],
            comments={0: _comments(("A1", "Check this"), ("B1", "And this")),
                      1: _comments(("A1", "Second sheet note"))}))
        self.assertEqual(
            plain.text + "\n## Sheet header: Data\nQuarterly figures\n\n"
            "## Sheet footer: Data\nPage\n\n## Sheet comments: Data\n"
            "A1: Reviewer:\nCheck this\nB1: Reviewer:\nAnd this\n\n"
            "## Sheet comments: More\nA1: Reviewer:\nSecond sheet note\n",
            ex.text)
        self.assertEqual(3, ex.metadata["comments"])

    def test_header_and_footer_codes_are_not_text(self):
        """Excel's codes are formatting and fields: a font, a size, a
        colour, the page, the sheet's name. LibreOffice writes a header of
        nothing else, which is no line at all; "&&" is an ampersand."""
        cases = {
            '&C&"DejaVu Serif,Book"&12&A': [],
            '&C&"DejaVu Serif,Book"&12Page &P': ["Page"],
            "&L&F&C&D&R&P/&N": [],
            "&KFF0000Red&K01+000 theme&B bold&I": ["Red theme bold"],
            "Smith && Sons&RTotal": ["Smith & Sons", "Total"],
            "&14": [],
            "Two\nlines": ["Two", "lines"],
            "trailing &": ["trailing"],
            '&"unterminated font': [],
        }
        for value, lines in cases.items():
            with self.subTest(value=value):
                self.assertEqual(lines, office._header_footer_lines(value))

    def test_a_comments_part_no_sheet_names_is_read(self):
        ex = self.extracted("orphan.xlsx", _xlsx(
            [("Data", ROWS, "")],
            orphans={"xl/comments7.xml": _comments(("C3", LINE))}))
        self.assertIn(f"## Sheet comments\nC3: Reviewer:\n{LINE}", ex.text)

    def test_a_broken_comments_part_leaves_the_sheet(self):
        plain = self.extracted("plain.xlsx", _xlsx([("Data", ROWS, "")]))
        ex = self.extracted("broken.xlsx", _xlsx(
            [("Data", ROWS, "")], comments={0: "<comments><commentList>"}))
        self.assertEqual(plain.text, ex.text)
        self.assertEqual(1, len(ex.notes), ex.notes)
        self.assertIn("1 comments part could not be read", ex.notes[0])

    def test_every_span_lies_on_the_extracted_text(self):
        secrets = ["Zq1vX8mT2rL5wK9pB1", "Zq2vX8mT2rL5wK9pB2"]
        path = self.path("spans.xlsx", _xlsx(
            [("Data", ROWS, "<headerFooter><oddHeader>&amp;Ctoken = "
                            f"{secrets[0]}</oddHeader></headerFooter>")],
            comments={0: _comments(("A1", f"password: {secrets[1]}"))}))
        ex = extract(path, detect(path), 25_000_000)
        _spans_hold(self, ex.text, path.name, secrets)

    def test_innocent_comments_add_no_span(self):
        ex = self.extracted("innocent.xlsx", _xlsx(
            [("Data", ROWS, "<headerFooter><oddFooter>&amp;LPage &amp;P of "
                            "&amp;N&amp;RACME Corp</oddFooter></headerFooter>")],
            comments={0: _comments(("A1", "Figures from the Q3 close."))}))
        self.assertEqual([], find_secrets(ex.text, "innocent.xlsx"))



# ---- docx tables --------------------------------------------------------
def _tc(text: str, inner: str = "") -> str:
    return f"<w:tc>{_para(text)}{inner}</w:tc>"


def _tbl(*rows: str) -> str:
    """A table of ``rows``, each the cells of one row."""
    return "<w:tbl>" + "".join(f"<w:tr>{r}</w:tr>" for r in rows) + "</w:tbl>"


def _body(xml: str) -> dict:
    return {"word/document.xml": f'<w:document xmlns:w="{W}"><w:body>{xml}'
                                 "</w:body></w:document>"}


def _chain(depth: int) -> str:
    """``depth`` tables, each the last child of the one before it, each a
    row of one cell reading "x" (test_secrets' nest)."""
    return ("<w:tbl><w:tr><w:tc><w:p><w:r><w:t>x</w:t></w:r></w:p></w:tc>"
            "</w:tr>" * depth + "</w:tbl>" * depth)


class DocxNestedTablesTest(_Office):
    def test_every_table_of_a_nest_of_three_is_rendered(self):
        """A table in a cell of a table in a cell of a table: each is a
        table of its own rows and cells. The middle one was not rendered at
        all -- its rows were taken for rows of a table nested in another
        (inner_rows_all), which it is -- so its text was in no line."""
        inner = _tbl(_tc("c1") + _tc("c2"))
        middle = _tbl(_tc("b1", inner), _tc("b2"))
        outer = _tbl(_tc("a1", middle) + _tc("a2"))
        ex = self.extracted("nest.docx", _body(_para("Intro") + outer))
        self.assertEqual("Intro\n\n## Table\na1 | a2\n\n## Table\nb1\nb2"
                         "\n\n## Table\nc1 | c2", ex.text)
        self.assertEqual(3, ex.metadata["tables"])
        self.assertEqual(["1×2", "2×1", "1×2"], ex.metadata["table_shapes"])
        self.assertEqual(6, ex.metadata["table_cells"])
        self.assertEqual(1, ex.metadata["paragraphs"])

    def test_a_secret_in_the_middle_table_is_found(self):
        inner = _tbl(_tc("c1") + _tc("c2"))
        middle = _tbl(_tc(LINE, inner))
        self.assert_found("middle.docx", _body(_tbl(_tc("a1", middle))))

    def test_a_table_in_a_cell_is_no_row_or_cell_of_the_one_around_it(self):
        """As before: a 2x2 inside a 2x2 is two 2x2 tables, each cell once."""
        inner = _tbl(_tc("i1") + _tc("i2"), _tc("i3") + _tc("i4"))
        outer = _tbl(_tc("o1", inner) + _tc("o2"), _tc("o3") + _tc("o4"))
        ex = self.extracted("two.docx", _body(outer))
        self.assertEqual("\n\n## Table\no1 | o2\no3 | o4\n\n## Table\n"
                         "i1 | i2\ni3 | i4", ex.text)
        self.assertEqual(["2×2", "2×2"], ex.metadata["table_shapes"])

    def test_every_table_of_a_deep_chain_is_rendered(self):
        """Fifty tables, each inside the one before: fifty tables of one
        cell. The first and the last were the only two rendered."""
        ex = self.extracted("chain.docx", _body(_chain(50)))
        self.assertEqual(50, ex.metadata["tables"])
        self.assertEqual(50, ex.text.count("## Table\nx"))

    def test_the_clock_is_looked_at_while_the_tables_are_walked(self):
        """Not once per table after every table's rows and cells were
        gathered, which is where the time went (6,400 nested tables ran
        50 s against a 5 s budget): one table of 3,000 rows looks at it as
        its rows are read, and a spent budget stops the walk."""
        files = _body(_tbl(*[_tc(f"row {n}") for n in range(3_000)]))
        calls = []
        real = office.deadline.check

        def counted(where=""):
            if where == "a document's tables":
                calls.append(where)
                if len(calls) == 2:
                    raise office.deadline.DeadlineExceeded(1.0, where)
            return real(where)
        with mock.patch.object(office.deadline, "check", counted):
            with self.assertRaises(office.deadline.DeadlineExceeded):
                self.extracted("rows.docx", files)
        self.assertEqual(2, len(calls))


# ---- pptx ---------------------------------------------------------------
def _shape(*paragraphs: str) -> str:
    return ("<p:sp><p:txBody>" + "".join(
        f"<a:p><a:r><a:t>{_escape(t)}</a:t></a:r></a:p>" for t in paragraphs)
        + "</p:txBody></p:sp>")


def _notes_slide(*paragraphs: str, number: str = "", raw: str = "") -> str:
    """A notes slide as PowerPoint writes one: the notes' paragraphs, and
    with ``number`` the slide-number field of its number placeholder."""
    field = (f'<p:sp><p:txBody><a:p><a:fld id="{{B6F15528}}" type="slidenum">'
             f"<a:t>{number}</a:t></a:fld></a:p></p:txBody></p:sp>"
             if number else "")
    return (f'<p:notes xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>'
            f"{_shape(*paragraphs)}{raw}{field}</p:spTree></p:cSld>"
            f"</p:notes>")


def _pptx(slides, notes=None, orphans=None):
    """``slides``: each slide's paragraphs; ``notes`` maps a slide's index
    to its notes slide's xml, related from the slide as PowerPoint relates
    it (from the package's root); ``orphans`` are members as they are."""
    notes = notes or {}
    files = {"ppt/presentation.xml": f'<p:presentation xmlns:p="{P}"/>'}
    for n, paragraphs in enumerate(slides, 1):
        files[f"ppt/slides/slide{n}.xml"] = (
            f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>'
            f"{_shape(*paragraphs)}</p:spTree></p:cSld></p:sld>")
        if n - 1 in notes:
            files[f"ppt/notesSlides/notesSlide{n}.xml"] = notes[n - 1]
            files[f"ppt/slides/_rels/slide{n}.xml.rels"] = (
                f'<Relationships xmlns="{PR}"><Relationship Id="rId1" '
                f'Type="{REL}slideLayout" Target="../slideLayouts/'
                f'slideLayout1.xml"/><Relationship Id="rId2" '
                f'Type="{REL}notesSlide" '
                f'Target="/ppt/notesSlides/notesSlide{n}.xml"/>'
                f"</Relationships>")
    files.update(orphans or {})
    return files


class PptxNotesTest(_Office):
    def test_a_secret_in_the_speaker_notes_is_found(self):
        self.assert_found("notes.pptx", _pptx(
            [["Quarterly review"]], notes={0: _notes_slide(LINE, number="1")}))

    def test_the_notes_follow_every_slide(self):
        """The slides as they were, every offset into them too; then each
        slide's notes in the slides' order, under a heading that begins
        "## Slide", sift's own. The slide number on a notes page is a
        field, not a note."""
        slides = [["First slide"], ["Second slide"], ["Third slide"]]
        plain = self.extracted("plain.pptx", _pptx(slides))
        ex = self.extracted("order.pptx", _pptx(slides, notes={
            2: _notes_slide("Wrap up", number="3"),
            0: _notes_slide("Say hello", "Then the agenda", number="1")}))
        self.assertEqual(
            plain.text + "\n\n## Slide 1 notes\nSay hello\nThen the agenda"
            "\n\n## Slide 3 notes\nWrap up", ex.text)
        self.assertEqual("# Presentation\n## Slide 1\nFirst slide\n\n"
                         "## Slide 2\nSecond slide\n\n## Slide 3\n"
                         "Third slide", plain.text)
        self.assertEqual(3, ex.metadata["slides"])
        self.assertEqual(2, ex.metadata["notes"])

    def test_a_note_is_read_a_paragraph_at_a_time(self):
        """Runs of one paragraph are one line, a break a line of its own:
        a password whose value is a run of its own is the password line."""
        runs = (f"<p:sp><p:txBody><a:p><a:r><a:t>password: </a:t></a:r>"
                f"<a:r><a:t>{SECRET}</a:t></a:r><a:br/><a:r><a:t>next"
                f"</a:t></a:r></a:p></p:txBody></p:sp>")
        ex = self.extracted("runs.pptx", _pptx(
            [["Only slide"]], notes={0: _notes_slide(raw=runs)}))
        self.assertTrue(ex.text.endswith(f"## Slide 1 notes\n{LINE}\nnext"),
                        ex.text)
        self.assert_spans_are_the_secret(ex.text, "runs.pptx")

    def test_notes_no_slide_names_are_read(self):
        ex = self.extracted("orphan.pptx", _pptx(
            [["Only slide"]],
            orphans={"ppt/notesSlides/notesSlide4.xml": _notes_slide(LINE)}))
        self.assertTrue(ex.text.endswith(f"## Slide notes\n{LINE}"), ex.text)

    def test_broken_notes_leave_the_slides(self):
        plain = self.extracted("plain.pptx", _pptx([["Only slide"]]))
        ex = self.extracted("broken.pptx", _pptx(
            [["Only slide"]], notes={0: "<p:notes"}))
        self.assertEqual(plain.text, ex.text)
        self.assertIn("1 notes part could not be read", " ".join(ex.notes))

    def test_empty_notes_add_nothing(self):
        """PowerPoint writes a notes slide for every slide: an empty body
        placeholder and the slide number."""
        plain = self.extracted("plain.pptx", _pptx([["Only slide"]]))
        ex = self.extracted("empty.pptx", _pptx(
            [["Only slide"]], notes={0: _notes_slide("", number="1")}))
        self.assertEqual(plain.text, ex.text)
        self.assertEqual(plain.metadata, ex.metadata)
        self.assertEqual([], ex.notes)

    def test_notes_and_slides_share_the_archive_budget(self):
        """The slides first, as before; the relationships and the notes on
        what the slides leave of _MAX_TOTAL_INFLATED."""
        files = _pptx([["First slide"], ["Second slide"]],
                      notes={0: _notes_slide(LINE)})
        path = self.path("budget.pptx", files)
        with zipfile.ZipFile(path) as zf:
            slides = sum(zf.getinfo(f"ppt/slides/slide{n}.xml").file_size
                         for n in (1, 2))
            rels = zf.getinfo("ppt/slides/_rels/slide1.xml.rels").file_size
        for budget, said in ((slides - 1, "later slides were not read"),
                             (slides + rels, "later notes were not read")):
            with self.subTest(budget=budget), mock.patch.object(
                    office, "_MAX_TOTAL_INFLATED", budget):
                ex = extract(path, detect(path), 25_000_000)
                self.assertIn("First slide", ex.text)
                self.assertNotIn(SECRET, ex.text)
                self.assertIn(said, ex.metadata["read_limit"])

    def test_reading_the_notes_stops_past_max_chars(self):
        """Four hundred notes parts nothing relates: read until the text is
        past MAX_CHARS, which is all clip keeps."""
        line = "A speaker note of forty characters, long"
        files = _pptx([["Only slide"]], orphans={
            f"ppt/notesSlides/notesSlide{n}.xml": _notes_slide(line)
            for n in range(1, 401)})
        reads = []
        real = office._read_member

        def counted(zf, name):
            reads.append(name)
            return real(zf, name)
        with mock.patch.object(office, "MAX_CHARS", 2_000), \
                mock.patch.object(office, "_read_member", counted):
            ex = self.extracted("many.pptx", files)
        self.assertTrue(ex.truncated)
        notes = [n for n in reads if n.startswith("ppt/notesSlides/")]
        self.assertLessEqual(len(notes), 2_000 // len(line) + 2)

    def test_every_span_lies_on_the_extracted_text(self):
        secrets = ["Zq1vX8mT2rL5wK9pB1", "Zq2vX8mT2rL5wK9pB2"]
        path = self.path("spans.pptx", _pptx(
            [[f"token = {secrets[0]}"], ["Plain"]],
            notes={1: _notes_slide(f"password: {secrets[1]}", number="2")}))
        ex = extract(path, detect(path), 25_000_000)
        _spans_hold(self, ex.text, path.name, secrets)
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None,
                              keep_text=True)
        _spans_hold(self, summ.text, path.name, secrets)

    def test_innocent_notes_add_no_span(self):
        ex = self.extracted("innocent.pptx", _pptx(
            [["Quarterly review"]], notes={0: _notes_slide(
                "Thank the team first.", "Revenue grew 12% on Q2.",
                number="1")}))
        self.assertIn("Thank the team first.", ex.text)
        self.assertEqual([], find_secrets(ex.text, "innocent.pptx"))


# ---- OpenDocument -------------------------------------------------------
def _odf(kind: str, body: str, styles: "str | None" = None):
    mime = {"odt": "text", "ods": "spreadsheet", "odp": "presentation"}[kind]
    files = {"mimetype": f"application/vnd.oasis.opendocument.{mime}",
             "content.xml": f"<office:document-content {ODF_NS}><office:body>"
                            f"{body}</office:body></office:document-content>"}
    if styles is not None:
        files["styles.xml"] = styles
    return files


def _master(*regions, pages: int = 1) -> str:
    """A styles.xml of ``pages`` master pages, each holding ``regions``:
    (element, paragraphs xml)."""
    inner = "".join(f"<style:{name}>{xml}</style:{name}>"
                    for name, xml in regions)
    return (f"<office:document-styles {ODF_NS}><office:master-styles>"
            + "".join(f'<style:master-page style:name="M{n}">{inner}'
                      "</style:master-page>" for n in range(pages))
            + "</office:master-styles></office:document-styles>")


def _annotation(*paragraphs: str) -> str:
    """A comment as LibreOffice writes one: its author, its date, then its
    paragraphs."""
    return ("<office:annotation><dc:creator>Reviewer</dc:creator><dc:date>"
            "2026-09-29T10:00:00</dc:date>"
            + "".join(f"<text:p>{_escape(t)}</text:p>" for t in paragraphs)
            + "</office:annotation>")


ODT_BODY = "<office:text><text:p>Body text of the letter.</text:p></office:text>"
ODS_BODY = ('<office:spreadsheet><table:table table:name="S"><table:table-row>'
            "<table:table-cell><text:p>Item</text:p></table:table-cell>"
            "</table:table-row></table:table></office:spreadsheet>")
ODP_BODY = ('<office:presentation><draw:page draw:name="p1"><draw:frame>'
            "<draw:text-box><text:p>Quarterly review</text:p></draw:text-box>"
            "</draw:frame></draw:page></office:presentation>")


class OdfPartsTest(_Office):
    def test_a_secret_in_a_master_page_header_or_footer_is_found(self):
        for kind, body in (("odt", ODT_BODY), ("ods", ODS_BODY)):
            for region in ("header", "footer", "header-left",
                           "footer-first"):
                with self.subTest(kind=kind, region=region):
                    self.assert_found(f"{region}.{kind}", _odf(
                        kind, body, _master(
                            (region, f"<text:p>{LINE}</text:p>"))))

    def test_a_secret_in_a_presentations_header_or_footer_is_found(self):
        for decl in ("footer-decl", "header-decl"):
            with self.subTest(decl=decl):
                body = ODP_BODY.replace(
                    "<office:presentation>",
                    f'<office:presentation><presentation:{decl} '
                    f'presentation:name="d1">{LINE}</presentation:{decl}>')
                self.assert_found(f"{decl}.odp", _odf("odp", body))

    def test_the_headings_and_the_order(self):
        """After the body, whose text and offsets are as they were: the
        headers, then the footers; an ods's under headings that begin
        "## Sheet", after its sheets as an xlsx's are."""
        styles = _master(
            ("header", "<text:p>Letter <text:span>head</text:span>ing</text:p>"
                       "<text:h>Second line</text:h>"),
            ("footer", "<text:p>Foot<text:tab/>note</text:p>"))
        plain = self.extracted("plain.odt", _odf("odt", ODT_BODY))
        odt = self.extracted("order.odt", _odf("odt", ODT_BODY, styles))
        self.assertEqual(plain.text + "\n\n## Header\nLetter heading\n"
                         "Second line\n\n## Footer\nFoot note", odt.text)
        plain = self.extracted("plain.ods", _odf("ods", ODS_BODY))
        ods = self.extracted("order.ods", _odf("ods", ODS_BODY, styles))
        self.assertEqual(plain.text + "\n## Sheet header\nLetter heading\n"
                         "Second line\n\n## Sheet footer\nFoot note\n",
                         ods.text)
        body = ODP_BODY.replace(
            "<office:presentation>",
            '<office:presentation><presentation:footer-decl presentation:'
            'name="f1">ACME</presentation:footer-decl><presentation:'
            'footer-decl presentation:name="f2">ACME</presentation:'
            'footer-decl><presentation:header-decl presentation:name="h1">'
            "Notes header</presentation:header-decl>")
        odp = self.extracted("order.odp", _odf("odp", body))
        self.assertEqual("Quarterly review\n\n## Header\nNotes header\n\n"
                         "## Footer\nACME", odp.text)

    def test_libreoffices_page_fields_are_not_text(self):
        """The fields of the header and footer LibreOffice writes into every
        spreadsheet hold what the last layout put there -- "???" for the
        sheet's name and the title, "00/00/0000" for the date: measured on
        6 of 6 ods in a census. What the author wrote around them stays."""
        styles = _master(
            ("header", "<text:p><text:sheet-name>???</text:sheet-name></text:p>"),
            ("footer", "<text:p>Page <text:page-number>1</text:page-number>"
                       "</text:p>"),
            ("header-left", '<style:region-left><text:p><text:sheet-name>???'
                            "</text:sheet-name><text:s/>(<text:title>???"
                            "</text:title>)</text:p></style:region-left>"
                            '<style:region-right><text:p><text:date style:'
                            'data-style-name="N2" text:date-value="2012-06-06">'
                            "00/00/0000</text:date>, <text:time>00:00:00"
                            "</text:time></text:p></style:region-right>"),
            ("footer-left", "<text:p>Page <text:page-number>1"
                            "</text:page-number><text:s/>/ <text:page-count>"
                            "99</text:page-count></text:p>"))
        ex = self.extracted("fields.ods", _odf("ods", ODS_BODY, styles))
        for placeholder in ("???", "00/00/0000", "00:00:00", "99"):
            self.assertNotIn(placeholder, ex.text)
        self.assertNotIn("## Sheet header", ex.text)
        self.assertTrue(ex.text.endswith("## Sheet footer\nPage\nPage /\n"),
                        ex.text)

    def test_a_header_every_master_page_repeats_is_written_once(self):
        ex = self.extracted("repeat.odt", _odf("odt", ODT_BODY, _master(
            ("header", "<text:p>Company confidential</text:p>"), pages=40)))
        self.assertEqual(1, ex.text.count("Company confidential"))

    def test_a_broken_styles_part_leaves_the_content(self):
        for kind, body in (("odt", ODT_BODY), ("ods", ODS_BODY)):
            with self.subTest(kind=kind):
                plain = self.extracted(f"plain.{kind}", _odf(kind, body))
                ex = self.extracted(f"broken.{kind}", _odf(
                    kind, body, "<office:document-styles"))
                self.assertEqual(plain.text, ex.text)
                self.assertIn("styles.xml could not be read",
                              " ".join(ex.notes))

    def test_a_deeply_nested_header_leaves_the_content(self):
        """Spans nested past Python's recursion limit: read with a stack,
        so the body is kept and the header read."""
        depth = 50_000
        header = ("<text:p>" + "<text:span>" * depth + LINE
                  + "</text:span>" * depth + "</text:p>")
        ex = self.extracted("deep.odt", _odf(
            "odt", ODT_BODY, _master(("header", header))))
        self.assertEqual(f"Body text of the letter.\n\n## Header\n{LINE}",
                         ex.text)

    def test_empty_regions_add_nothing(self):
        plain = self.extracted("plain.odt", _odf("odt", ODT_BODY))
        ex = self.extracted("empty.odt", _odf("odt", ODT_BODY, _master(
            ("header", "<text:p/>"), ("footer", "<text:p>   </text:p>"),
            ("header-left", ""))))
        self.assertEqual(plain.text, ex.text)
        self.assertEqual([], ex.notes)

    def test_the_styles_share_the_archive_budget(self):
        files = _odf("odt", ODT_BODY, _master(("header", f"<text:p>{LINE}"
                                                         "</text:p>")))
        path = self.path("budget.odt", files)
        with zipfile.ZipFile(path) as zf:
            content = zf.getinfo("content.xml").file_size
        with mock.patch.object(office, "_MAX_TOTAL_INFLATED", content):
            ex = extract(path, detect(path), 25_000_000)
        self.assertEqual("Body text of the letter.", ex.text)
        self.assertIn("were not read", ex.metadata["read_limit"])

    def test_reading_a_header_stops_past_max_chars(self):
        line = "A header line of forty characters, long."
        styles = _master(("header", "".join(
            f"<text:p>{line} {n}</text:p>" for n in range(1_000))))
        calls = []
        real = office._odf_line

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        with mock.patch.object(office, "MAX_CHARS", 2_000), \
                mock.patch.object(office, "_odf_line", counted):
            ex = self.extracted("many.odt", _odf("odt", ODT_BODY, styles))
        self.assertTrue(ex.truncated)
        self.assertLessEqual(len(calls), 2_000 // len(line) + 3)

    def test_every_span_lies_on_the_extracted_text(self):
        secrets = ["Zq1vX8mT2rL5wK9pB1", "Zq2vX8mT2rL5wK9pB2"]
        cases = {
            "spans.odt": _odf("odt", ODT_BODY.replace(
                "letter.", f"letter. token = {secrets[0]}"), _master(
                ("footer", f"<text:p>password: {secrets[1]}</text:p>"))),
            "spans.ods": _odf("ods", ODS_BODY.replace(
                "Item", f"token = {secrets[0]}"), _master(
                ("header", f"<text:p>password: {secrets[1]}</text:p>"))),
            "spans.odp": _odf("odp", ODP_BODY.replace(
                "Quarterly review", f"token = {secrets[0]}").replace(
                "<office:presentation>", "<office:presentation><presentation:"
                f"footer-decl>password: {secrets[1]}</presentation:"
                "footer-decl>")),
        }
        for name, files in cases.items():
            with self.subTest(file=name):
                path = self.path(name, files)
                ex = extract(path, detect(path), 25_000_000)
                _spans_hold(self, ex.text, name, secrets)
                summ = summarize_path(path, Config(
                    no_llm=True, quiet=True, cache=False), None, None,
                    keep_text=True)
                _spans_hold(self, summ.text, name, secrets)

    def test_innocent_headers_and_footers_add_no_span(self):
        styles = _master(
            ("header", "<text:p>ACME Corp — Internal</text:p>"),
            ("footer", "<text:p>Page <text:page-number>1</text:page-number>"
                       " of <text:page-count>3</text:page-count></text:p>"))
        for kind, body in (("odt", ODT_BODY), ("ods", ODS_BODY)):
            with self.subTest(kind=kind):
                ex = self.extracted(f"innocent.{kind}",
                                    _odf(kind, body, styles))
                self.assertIn("ACME Corp — Internal", ex.text)
                self.assertEqual([], find_secrets(ex.text, f"i.{kind}"))
        ex = self.extracted("innocent.odp", _odf("odp", ODP_BODY.replace(
            "<office:presentation>", "<office:presentation><presentation:"
            "footer-decl>Confidential, do not distribute</presentation:"
            "footer-decl>")))
        self.assertIn("## Footer\nConfidential", ex.text)
        self.assertEqual([], find_secrets(ex.text, "innocent.odp"))


class OdfBesideTheTextTest(_Office):
    """What a paragraph or a cell holds beside its own text -- a comment
    with its author and date, a footnote, a text box, a shape -- was read
    into it, glued to the words around it: sift learned the secret with
    the cell's value on its end ("Kx9vQ2mZ7pL4Wq8rTzItem", or nothing at
    all behind "A2026-09-29T10:00:00password:"), and a model's sentence
    that repeated the bare secret was published unhidden. So were a tab
    and a line break, read as nothing between two words."""

    def test_an_ods_comment_is_a_line_of_its_own_after_the_sheets(self):
        plain = self.extracted("plain.ods", _odf("ods", ODS_BODY))
        ex = self.extracted("comment.ods", _odf("ods", ODS_BODY.replace(
            "<text:p>Item", _annotation(LINE) + "<text:p>Item")))
        self.assertEqual(plain.text + f"\n## Sheet comments: S\n{LINE}\n",
                         ex.text)
        self.assertEqual(1, ex.metadata["comments"])
        for author_or_date in ("Reviewer", "2026-09-29"):
            self.assertNotIn(author_or_date, ex.text)
        self.assert_spans_are_the_secret(ex.text, "comment.ods")

    def test_a_model_repeating_an_ods_comments_secret_is_hidden(self):
        self.assert_hidden_from_a_model("comment.ods", _odf(
            "ods", ODS_BODY.replace("<text:p>Item",
                                    _annotation(LINE) + "<text:p>Item")))

    def test_a_shape_in_a_cell_is_a_line_of_its_own_after_the_sheets(self):
        """A shape anchored to a cell is drawn over the sheet, no part of
        the cell's value (a host application's ods codec: BESIDE_THE_VALUE and the
        drawing namespace)."""
        shape = f"<draw:custom-shape><text:p>{LINE}</text:p></draw:custom-shape>"
        files = _odf("ods", ODS_BODY.replace("<text:p>Item",
                                             shape + "<text:p>Item"))
        plain = self.extracted("plain.ods", _odf("ods", ODS_BODY))
        ex = self.extracted("shape.ods", files)
        self.assertEqual(plain.text + f"\n## Sheet shapes: S\n{LINE}\n",
                         ex.text)
        self.assert_spans_are_the_secret(ex.text, "shape.ods")
        self.assert_hidden_from_a_model("shape.ods", files)

    def test_a_page_of_the_sheet_shows_the_value_alone(self):
        """read_sheet_page's cells are what LibreOffice shows in them."""
        path = self.path("page.ods", _odf("ods", ODS_BODY.replace(
            "<text:p>Item", _annotation(LINE) + "<text:p>Item")))
        page = office.read_sheet_page(path, detect(path), None, 0, 10,
                                      10_000)
        self.assertIsNone(page.error)
        self.assertEqual([["Item"]], page.rows)

    def test_a_cell_holding_only_a_comment_is_empty(self):
        """A census's comment.ods: one cell, a comment and no value. Its
        sheet holds no row, so it is read as paragraphs, the comment's
        among them, and neither its author nor its date."""
        ex = self.extracted("only.ods", _odf("ods", ODS_BODY.replace(
            "<text:p>Item</text:p>", _annotation("Komentarz"))))
        self.assertEqual("Komentarz", ex.text)

    def test_the_comments_of_each_sheet_under_its_name(self):
        second = ODS_BODY.replace('table:name="S"', 'table:name="T"').replace(
            "<text:p>Item", _annotation("On T") + "<text:p>Item")
        body = ODS_BODY.replace(
            "<text:p>Item", _annotation("On S", "Second line") +
            "<text:p>Item").replace("</office:spreadsheet>", "") + \
            second.replace("<office:spreadsheet>", "")
        ex = self.extracted("two.ods", _odf("ods", body))
        self.assertTrue(ex.text.endswith(
            "\n## Sheet comments: S\nOn S\nSecond line\n\n"
            "## Sheet comments: T\nOn T\n"), ex.text)
        self.assertEqual(2, ex.metadata["comments"])

    def test_an_odt_comment_is_not_glued_to_its_paragraph(self):
        """A comment on a selection is written before the selected words:
        its author, its date and its text were glued to them."""
        files = _odf("odt", "<office:text><text:p>A commented "
                            f"{_annotation(LINE)}Item<office:annotation-end/>"
                            " here.</text:p></office:text>")
        ex = self.extracted("comment.odt", files)
        self.assertEqual(f"A commented Item here.\n{LINE}", ex.text)
        self.assert_spans_are_the_secret(ex.text, "comment.odt")
        self.assert_hidden_from_a_model("comment.odt", files)

    def test_a_footnote_and_a_text_box_are_lines_of_their_own(self):
        """Their paragraphs were read twice: as lines of their own, and
        glued into the paragraph that holds them -- a span over the secret
        and the word after it."""
        body = ("<office:text><text:p>See<text:note text:note-class="
                '"footnote"><text:note-citation>1</text:note-citation>'
                f"<text:note-body><text:p>{LINE}</text:p></text:note-body>"
                "</text:note>Item more.</text:p><text:p>Before<draw:frame>"
                f"<draw:text-box><text:p>{LINE}</text:p></draw:text-box>"
                "</draw:frame>After</text:p></office:text>")
        ex = self.extracted("note.odt", _odf("odt", body))
        self.assertEqual(f"See1Item more.\n{LINE}\nBeforeAfter\n{LINE}",
                         ex.text)
        self.assert_spans_are_the_secret(ex.text, "note.odt")

    def test_a_tab_a_space_and_a_line_break_are_white_space(self):
        """text:tab, text:s and text:line-break hold no text: read as
        nothing, "password:<tab>SECRET<line break>User" was one word, the
        bare secret not hidden. A line break is a line break, as a docx's
        is. So were the last word of a cell's paragraph and the first of
        its next one word."""
        files = _odf("odt", "<office:text><text:p>password:<text:tab/>"
                            f"{SECRET}<text:line-break/>User<text:s "
                            'text:c="3"/>bob</text:p></office:text>')
        ex = self.extracted("tab.odt", files)
        self.assertEqual(f"password: {SECRET}\nUser bob", ex.text)
        self.assert_spans_are_the_secret(ex.text, "tab.odt")
        self.assert_hidden_from_a_model("tab.odt", files)
        cells = _odf("ods", ODS_BODY.replace(
            "<text:p>Item</text:p>",
            f"<text:p>password:<text:tab/>{SECRET}</text:p><text:p>Item"
            "</text:p>"))
        ex = self.extracted("tab.ods", cells)
        # A cell's value is one line of its row: its paragraphs apart, by a
        # space, which keeps the bare secret among what is hidden.
        self.assertIn(f"\npassword: {SECRET} Item\n", ex.text)
        self.assert_hidden_from_a_model("tab.ods", cells)

    def test_a_deeply_nested_body_is_read(self):
        """Spans nested past Python's recursion limit: the body was lost to
        "nesting too deep"; read with a stack, it is read."""
        depth = 50_000
        body = ("<office:text><text:p>" + "<text:span>" * depth + LINE
                + "</text:span>" * depth + "</text:p></office:text>")
        ex = self.extracted("deep.odt", _odf("odt", body))
        self.assertEqual(LINE, ex.text)

    def test_reading_the_comments_stops_past_max_chars(self):
        line = "A comment line of forty characters, long"
        cells = "".join(
            f"<table:table-cell>{_annotation(f'{line} {n}')}<text:p>v{n}"
            "</text:p></table:table-cell>" for n in range(1_000))
        body = ODS_BODY.replace(
            "<table:table-cell><text:p>Item</text:p></table:table-cell>",
            cells)
        calls = []
        real = office._odf_line

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)
        with mock.patch.object(office, "MAX_CHARS", 2_000), \
                mock.patch.object(office, "_odf_line", counted):
            ex = self.extracted("many.ods", _odf("ods", body))
        self.assertTrue(ex.truncated)
        self.assertLessEqual(len(calls), 2_000 // len(line) + 3)

    def test_the_clock_is_looked_at_while_the_comments_are_gathered(self):
        rows = "".join("<table:table-row><table:table-cell><text:p>v"
                       "</text:p></table:table-cell></table:table-row>"
                       for _ in range(3_000))
        body = ODS_BODY.replace(
            "<text:p>Item", _annotation("A note") + "<text:p>Item").replace(
            "</table:table>", rows + "</table:table>")
        calls = []
        real = office.deadline.check

        def counted(where=""):
            if where == "a spreadsheet's content":
                calls.append(where)
            return real(where)
        with mock.patch.object(office.deadline, "check", counted):
            self.extracted("clock.ods", _odf("ods", body))
        self.assertGreaterEqual(len(calls), 2)

    def test_every_span_lies_on_the_extracted_text(self):
        secrets = ["Zq1vX8mT2rL5wK9pB1", "Zq2vX8mT2rL5wK9pB2"]
        cases = {
            "spans.ods": _odf("ods", ODS_BODY.replace(
                "<text:p>Item", _annotation(f"password: {secrets[1]}")
                + f"<text:p>token = {secrets[0]}")),
            "spans.odt": _odf("odt", "<office:text><text:p>Item "
                                     f"{_annotation(f'password: {secrets[1]}')}"
                                     f"token = {secrets[0]}</text:p>"
                                     "</office:text>"),
        }
        for name, files in cases.items():
            with self.subTest(file=name):
                path = self.path(name, files)
                ex = extract(path, detect(path), 25_000_000)
                _spans_hold(self, ex.text, name, secrets)
                summ = summarize_path(path, Config(
                    no_llm=True, quiet=True, cache=False), None, None,
                    keep_text=True)
                _spans_hold(self, summ.text, name, secrets)

    def test_innocent_comments_add_no_span(self):
        ex = self.extracted("innocent.ods", _odf("ods", ODS_BODY.replace(
            "<text:p>Item", _annotation("Figures from the Q3 close.")
            + "<text:p>Item")))
        self.assertIn("Figures from the Q3 close.", ex.text)
        self.assertEqual([], find_secrets(ex.text, "innocent.ods"))



# ---- docx text boxes --------------------------------
def _boxed(box: str, after: str = "Item") -> str:
    """A body paragraph holding a text box of ``box`` and then the run
    ``after``, as Word writes a floating text box: a drawing's txbxContent
    inside a run of the paragraph around it."""
    return (f"<w:p><w:r><w:drawing><w:txbxContent>{_para(box)}"
            f"</w:txbxContent></w:drawing></w:r><w:r><w:t>{_escape(after)}"
            f"</w:t></w:r></w:p>")


class DocxTextBoxTest(_Office):
    """A text box's paragraphs sit inside the body paragraph that anchors
    it. _docx read that paragraph with every descendant's text, the box's
    among them, so "password: SECRET" in a box followed by the run "Item"
    was learned as "password: SECRETItem" -- a span "SECRETItem" beside the
    box's own line. A text box is a line of its
    own now, and no part of the paragraph around it, as a header's is
    (_paragraph_lines)."""

    def test_a_text_box_is_a_line_of_its_own(self):
        ex = self.extracted("box.docx", _body(
            _para("Before") + _boxed("Boxed text", after="after")
            + _para("Last")))
        self.assertEqual(["Before", "after", "Boxed text", "Last"],
                         ex.text.splitlines())
        self.assertEqual(4, ex.metadata["paragraphs"])

    def test_a_secret_in_a_text_box_is_the_secret(self):
        files = _body(_boxed(LINE))
        self.assert_found("secret.docx", files)
        ex = self.extracted("secret.docx", files)
        self.assert_spans_are_the_secret(ex.text, "secret.docx")
        self.assertNotIn(SECRET + "Item", ex.text)

    def test_a_model_repeating_the_boxed_secret_has_it_hidden(self):
        self.assert_hidden_from_a_model("box.docx", _body(_boxed(LINE)))

    def test_a_text_box_in_a_table_cell_is_not_glued(self):
        cell = ("<w:tc>" + _boxed(LINE) + "</w:tc>")
        ex = self.extracted("cell.docx", _body(_tbl(cell)))
        self.assertNotIn(SECRET + "Item", ex.text)
        self.assert_spans_are_the_secret(ex.text, "cell.docx")

    def test_a_paragraph_without_a_box_reads_as_before(self):
        """Breaks and tabs as _docx_paragraph kept them."""
        ex = self.extracted("plain.docx", _body(
            "<w:p><w:r><w:t>a</w:t><w:tab/><w:t>b</w:t><w:br/><w:t>c</w:t>"
            "</w:r></w:p>"))
        self.assertEqual("a\tb\nc", ex.text)

    def test_a_deep_nest_of_text_boxes_is_read_in_linear_time(self):
        """Each paragraph's text is read once: an iter() under each of
        5,000 nested paragraphs read 12.5 million elements."""
        depth = 5_000
        nest = ("<w:p><w:r><w:t>x</w:t><w:pict><w:txbxContent>" * depth
                + "</w:txbxContent></w:pict></w:r></w:p>" * depth)
        start = time.monotonic()
        ex = self.extracted("nest.docx", _body(nest))
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertEqual(depth, ex.text.count("x"))


# ---- pptx comments ----------------------------------
P188 = "http://schemas.microsoft.com/office/powerpoint/2018/8/main"
MODERN_REL = "http://schemas.microsoft.com/office/2018/10/relationships/comments"


def _legacy_comments(*texts: str) -> str:
    """ppt/comments/commentN.xml as PowerPoint 2007-2019 writes it: each
    comment's author by id, its date, its position, its text."""
    return (f'<p:cmLst xmlns:p="{P}">' + "".join(
        f'<p:cm authorId="0" dt="2026-09-29T10:00:00.000" idx="{n + 1}">'
        f'<p:pos x="10" y="10"/><p:text>{_escape(t)}</p:text></p:cm>'
        for n, t in enumerate(texts)) + "</p:cmLst>")


def _modern_comments(*threads) -> str:
    """ppt/comments/modernComment_*.xml as Microsoft 365 writes it: each
    thread (text, reply, ...) a comment whose replies come before its own
    text body in the file."""
    out = [f'<p188:cmLst xmlns:a="{A}" xmlns:p188="{P188}">']
    for n, (text, *replies) in enumerate(threads):
        body = ("<p188:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{}"
                "</a:t></a:r></a:p></p188:txBody>")
        out.append(f'<p188:cm id="{{C{n}}}" authorId="{{A1}}" '
                   f'created="2026-09-29T10:00:00.000"><p188:pos x="1" y="1"/>')
        if replies:
            out.append("<p188:replyLst>" + "".join(
                f'<p188:reply id="{{R{n}{m}}}" authorId="{{A2}}" '
                f'created="2026-09-29T11:00:00.000">'
                + body.format(_escape(r)) + "</p188:reply>"
                for m, r in enumerate(replies)) + "</p188:replyLst>")
        out.append(body.format(_escape(text)) + "</p188:cm>")
    return "".join(out) + "</p188:cmLst>"


def _commented(files: dict, slide: int, member: str, xml: str,
               modern: bool = False) -> dict:
    """``files`` with ``member`` related from slide ``slide`` (1-based) as
    its comments part."""
    files = dict(files)
    rels = f"ppt/slides/_rels/slide{slide}.xml.rels"
    kind = MODERN_REL if modern else f"{REL}comments"
    entry = (f'<Relationship Id="rId9{slide}" Type="{kind}" '
             f'Target="../comments/{member.rsplit("/", 1)[-1]}"/>')
    if rels in files:
        files[rels] = files[rels].replace("</Relationships>",
                                          entry + "</Relationships>")
    else:
        files[rels] = f'<Relationships xmlns="{PR}">{entry}</Relationships>'
    files[member] = xml
    return files


class PptxCommentsTest(_Office):
    """A presentation's comments -- ppt/comments/commentN.xml, and the
    modern threads of Microsoft 365 -- were read by nothing: a password in a review comment was not in sift's text. They
    are read after the notes now, as the notes after the slides."""

    def test_a_secret_in_a_comment_is_found(self):
        for modern in (False, True):
            with self.subTest(modern=modern):
                xml = _modern_comments((LINE,)) if modern \
                    else _legacy_comments(LINE)
                member = ("ppt/comments/modernComment_100_2F3A5B1C.xml"
                          if modern else "ppt/comments/comment1.xml")
                self.assert_found(f"c{int(modern)}.pptx", _commented(
                    _pptx([["Quarterly review"]]), 1, member, xml, modern))

    def test_the_comments_follow_the_notes(self):
        """The slides and the notes as they were; then each slide's
        comments in the slides' order, a thread's text before its replies;
        no author, no date."""
        slides = [["First slide"], ["Second slide"]]
        base = _pptx(slides, notes={0: _notes_slide("Say hello")})
        plain = self.extracted("plain.pptx", base)
        files = _commented(base, 2, "ppt/comments/comment2.xml",
                           _legacy_comments("Check the figures", "Typo"))
        files = _commented(files, 1,
                           "ppt/comments/modernComment_100_1.xml",
                           _modern_comments(("Why this chart?", "Agreed",
                                             "Fixed")), modern=True)
        ex = self.extracted("order.pptx", files)
        self.assertEqual(
            plain.text + "\n\n## Slide 1 comments\nWhy this chart?\nAgreed"
            "\nFixed\n\n## Slide 2 comments\nCheck the figures\nTypo",
            ex.text)
        self.assertEqual(5, ex.metadata["comments"])
        self.assertEqual(1, ex.metadata["notes"])
        self.assertNotIn("2026", ex.text)

    def test_comments_no_slide_names_are_read(self):
        files = _pptx([["Only slide"]], orphans={
            "ppt/comments/comment3.xml": _legacy_comments(LINE),
            "ppt/comments/modernComment_7_AB.xml": _modern_comments(
                ("Second orphan",))})
        ex = self.extracted("orphan.pptx", files)
        self.assertTrue(ex.text.endswith(
            f"## Slide comments\n{LINE}\nSecond orphan"), ex.text)

    def test_broken_comments_leave_the_slides(self):
        plain = self.extracted("plain.pptx", _pptx([["Only slide"]]))
        ex = self.extracted("broken.pptx", _commented(
            _pptx([["Only slide"]]), 1, "ppt/comments/comment1.xml",
            "<p:cmLst"))
        self.assertEqual(plain.text, ex.text)
        self.assertIn("1 comments part could not be read", " ".join(ex.notes))

    def test_empty_comments_add_nothing(self):
        plain = self.extracted("plain.pptx", _pptx([["Only slide"]]))
        ex = self.extracted("empty.pptx", _commented(
            _pptx([["Only slide"]]), 1, "ppt/comments/comment1.xml",
            _legacy_comments("  ")))
        self.assertEqual(plain.text, ex.text)
        self.assertEqual(plain.metadata, ex.metadata)

    def test_a_model_repeating_a_commented_secret_has_it_hidden(self):
        self.assert_hidden_from_a_model("hidden.pptx", _commented(
            _pptx([["Only slide"]]), 1, "ppt/comments/comment1.xml",
            _legacy_comments(LINE)))

    def test_innocent_comments_add_no_span(self):
        ex = self.extracted("innocent.pptx", _commented(
            _pptx([["Quarterly review"]]), 1, "ppt/comments/comment1.xml",
            _legacy_comments("Move the logo left.")))
        self.assertIn("Move the logo left.", ex.text)
        self.assertEqual([], find_secrets(ex.text, "innocent.pptx"))

    def test_comments_share_the_archive_budget(self):
        """The slides, their relationships and the notes first; the
        comments on what they leave of _MAX_TOTAL_INFLATED."""
        files = _commented(_pptx([["First slide"]]), 1,
                           "ppt/comments/comment1.xml", _legacy_comments(LINE))
        path = self.path("budget.pptx", files)
        with zipfile.ZipFile(path) as zf:
            used = sum(zf.getinfo(n).file_size for n in (
                "ppt/slides/slide1.xml", "ppt/slides/_rels/slide1.xml.rels"))
        with mock.patch.object(office, "_MAX_TOTAL_INFLATED", used):
            ex = extract(path, detect(path), 25_000_000)
        self.assertIn("First slide", ex.text)
        self.assertNotIn(SECRET, ex.text)
        self.assertIn("later comments were not read",
                      ex.metadata["read_limit"])

    def test_a_nest_of_comments_is_read_in_linear_time(self):
        """Comments inside comments, 20,000 deep: each read once, not once
        per level (a walk under every comment for its replies was)."""
        depth = 20_000
        nest = ('<p188:cm id="{C}">' * depth + "</p188:cm>" * depth)
        xml = (f'<p188:cmLst xmlns:a="{A}" xmlns:p188="{P188}">{nest}'
               "</p188:cmLst>")
        start = time.monotonic()
        ex = self.extracted("nest.pptx", _commented(
            _pptx([["Only slide"]]), 1, "ppt/comments/comment1.xml", xml))
        self.assertLess(time.monotonic() - start, 5.0)
        self.assertIn("Only slide", ex.text)

    def test_reading_the_comments_stops_past_max_chars(self):
        line = "A review comment of forty characters, ok"
        files = _pptx([["Only slide"]], orphans={
            f"ppt/comments/comment{n}.xml": _legacy_comments(line)
            for n in range(1, 401)})
        reads = []
        real = office._read_member

        def counted(zf, name):
            reads.append(name)
            return real(zf, name)
        with mock.patch.object(office, "MAX_CHARS", 2_000), \
                mock.patch.object(office, "_read_member", counted):
            ex = self.extracted("many.pptx", files)
        self.assertTrue(ex.truncated)
        read = [n for n in reads if n.startswith("ppt/comments/")]
        self.assertLessEqual(len(read), 2_000 // len(line) + 2)


if __name__ == "__main__":
    unittest.main()
