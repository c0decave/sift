import io
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sift.detect import detect
from sift.extract import extract
from tests import _fixtures as fx

BIG = 25_000_000


class ExtractTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _ex(self, path):
        return extract(path, detect(path), BIG)

    def test_csv_columns_and_types(self):
        ex = self._ex(fx.write(self.dir / "d.csv", fx.CSV_TEXT))
        self.assertEqual(ex.metadata["columns"], 4)
        self.assertEqual(ex.metadata["rows"], 3)
        types = {c["name"]: c["type"] for c in ex.metadata["column_profile"]}
        self.assertEqual(types["id"], "integer")
        self.assertEqual(types["score"], "float")
        self.assertEqual(types["active"], "boolean")
        self.assertEqual(types["name"], "string")

    def test_email_headers_and_body(self):
        ex = self._ex(fx.write(self.dir / "m.eml", fx.EML_TEXT))
        h = ex.metadata["headers"]
        self.assertIn("a@example.com", h["from"])
        self.assertEqual(h["subject"], "Quarterly numbers")
        self.assertIn("revenue", ex.text)

    def test_batch_risk_signal(self):
        ex = self._ex(fx.write(self.dir / "b.bat", fx.BAT_TEXT))
        self.assertIn("recursive/forced delete", ex.metadata["risk_signals"])

    def test_docx_text_and_props(self):
        ex = self._ex(fx.make_docx(self.dir / "d.docx",
                                   ("Alpha heading", "Second paragraph body.")))
        self.assertIn("Second paragraph body.", ex.text)
        self.assertEqual(ex.metadata["creator"], "Tester")
        self.assertEqual(ex.metadata["paragraphs"], 2)

    def test_xlsx_shared_strings(self):
        ex = self._ex(fx.make_xlsx(self.dir / "s.xlsx"))
        self.assertEqual(ex.metadata["sheets"][0]["name"], "Data")
        self.assertIn("Apples", ex.text)
        self.assertIn("42", ex.text)

    def test_png_dimensions(self):
        ex = self._ex(fx.make_png(self.dir / "i.png", w=8, h=5))
        self.assertEqual(ex.metadata["format"], "PNG")
        self.assertEqual((ex.metadata["width"], ex.metadata["height"]), (8, 5))
        self.assertIsNotNone(ex.image_data)

    def test_json_structure(self):
        ex = self._ex(fx.write(self.dir / "a.json", fx.JSON_TEXT))
        self.assertIn("name", ex.metadata["structure"])
        self.assertIn("nested", ex.metadata["structure"])

    def test_html_title_and_text(self):
        ex = self._ex(fx.write(self.dir / "p.html", fx.HTML_TEXT))
        self.assertEqual(ex.metadata["title"], "Report")
        self.assertIn("Some body text", ex.text)
        self.assertNotIn("<h1>", ex.text)

    def test_log_levels(self):
        ex = self._ex(fx.write(self.dir / "s.log", fx.LOG_TEXT))
        self.assertEqual(ex.metadata["level_counts"].get("ERROR"), 1)

    def test_xml_with_encoding_declaration(self):
        xml = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<config><server name="db"/><server name="cache"/></config>')
        ex = self._ex(fx.write(self.dir / "c.xml", xml))
        self.assertNotIn("parse_error", ex.metadata)
        self.assertEqual(ex.metadata["root"], "config")
        self.assertEqual(ex.metadata["children"].get("server"), 2)


class TarArchiveLimitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "many.tar"

    def _write(self, sizes):
        with tarfile.open(self.path, "w") as tf:
            for index, size in enumerate(sizes):
                info = tarfile.TarInfo(f"entry-{index}")
                info.size = size
                tf.addfile(info, io.BytesIO(b"x" * size))

    def test_an_ordinary_tar_is_listed(self):
        from sift.extract.fallback import _archive_members
        self._write([0, 0])
        names, total, refused = _archive_members(self.path)
        self.assertEqual((names, total, refused),
                         (["entry-0", "entry-1"], 2, ""))

    def test_tar_entry_count_is_bounded(self):
        from sift.extract import fallback
        self._write([0, 0, 0])
        with mock.patch.object(fallback, "MAX_TAR_ENTRIES", 2):
            names, total, refused = fallback._archive_members(self.path)
        self.assertEqual(total, 3)
        self.assertEqual(names, ["entry-0", "entry-1"])
        self.assertIn("TAR entry limit exceeded", refused)

    def test_a_large_declared_member_stops_before_its_data_is_scanned(self):
        from sift.extract import fallback
        self._write([1024])
        with mock.patch.object(fallback, "MAX_TAR_SCAN_BYTES", 512):
            names, total, refused = fallback._archive_members(self.path)
        self.assertEqual((names, total), ([], 1))
        self.assertIn("TAR scan-size limit exceeded", refused)

    def test_tar_listing_honours_the_per_file_deadline(self):
        from sift import deadline
        from sift.extract.fallback import _archive_members
        self._write([0])
        with mock.patch("sift.deadline.time.monotonic",
                        side_effect=[0.0, 2.0]):
            with deadline.limit(1.0):
                with self.assertRaises(deadline.DeadlineExceeded):
                    _archive_members(self.path)


if __name__ == "__main__":
    unittest.main()


class TabularPrecisionTest(unittest.TestCase):
    """Numbers sift prints are checked against the file, not against a sample.

    Every case here was measured wrong before: a summary that is confidently
    off by 25x is worse than no summary at all.
    """

    def _meta(self, text: str, name="t.csv"):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(text, encoding="utf-8")
            return extract(p, detect(p), 25_000_000).metadata

    def test_statistics_cover_every_row_not_the_first_two_hundred(self):
        rows = "\n".join(f"{i},{i * 2}" for i in range(5000))
        meta = self._meta("id,amount\n" + rows + "\n")
        col = meta["column_profile"][0]
        self.assertEqual(meta["rows"], 5000)
        self.assertFalse(meta["rows_truncated"])
        self.assertEqual(col["max"], 4999.0)
        self.assertEqual(col["mean"], 2499.5)

    def test_a_column_that_turns_to_text_late_is_not_called_numeric(self):
        rows = "\n".join(f"{i},{i}" for i in range(250))
        meta = self._meta("id,amount\n" + rows + "\n250,N/A\n251,unknown\n")
        self.assertEqual(meta["column_profile"][1]["type"], "string")

    def test_thousands_separators_are_not_decimal_points(self):
        meta = self._meta('city,population\nA,"1,234"\nB,"2,500"\nC,"9,000"\n')
        col = meta["column_profile"][1]
        self.assertEqual(col["min"], 1234.0)
        self.assertEqual(col["max"], 9000.0)

    def test_european_decimal_comma_still_works(self):
        meta = self._meta('item,preis\nA,"1,5"\nB,"2,25"\n')
        self.assertEqual(meta["column_profile"][1]["max"], 2.25)

    def test_years_as_column_names_are_a_header(self):
        meta = self._meta("2020,2021,2022\n10,20,30\n40,50,60\n70,10,20\n")
        self.assertTrue(meta["has_header"])
        self.assertEqual(meta["rows"], 3)
        self.assertEqual(meta["column_profile"][0]["max"], 70.0)

    def test_a_headerless_numeric_file_is_not_given_one(self):
        meta = self._meta("1,10\n2,20\n3,30\n4,40\n")
        self.assertFalse(meta["has_header"])
        self.assertEqual(meta["rows"], 4)

    def test_a_field_larger_than_the_csv_default_limit(self):
        meta = self._meta("id,blob\n1," + "x" * 200_000 + "\n")
        self.assertEqual(meta["rows"], 1)
        self.assertNotIn("extractor_error", meta)

    def test_a_delimiter_inside_a_quoted_field_survives_re_rendering(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text('name,note\n"Doe, Jane",ok\n', encoding="utf-8")
            ex = extract(p, detect(p), 25_000_000)
        self.assertIn('"Doe, Jane",ok', ex.text)
        self.assertEqual(ex.metadata["columns"], 2)

    def test_a_date_shaped_prefix_is_not_a_date(self):
        meta = self._meta("id,when\n1,2024-99-99 not a date at all\n"
                          "2,2024-88-88 also not\n")
        self.assertEqual(meta["column_profile"][1]["type"], "string")


class DetectionSignatureTest(unittest.TestCase):
    def _kind(self, data: bytes, name: str):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_bytes(data)
            return detect(p).kind

    def test_text_starting_with_BM_is_not_a_bitmap(self):
        self.assertEqual(
            self._kind(b"BMW and Mercedes are German car makers.\n", "cars.txt"),
            "text")

    def test_text_starting_with_MZ_is_not_an_executable(self):
        self.assertEqual(
            self._kind(b"MZ is the start of this note about executables.\n",
                       "note.txt"), "text")

    def test_a_real_bmp_is_still_an_image(self):
        header = (b"BM" + (1000).to_bytes(4, "little") + b"\x00\x00\x00\x00"
                  + (54).to_bytes(4, "little") + (40).to_bytes(4, "little")
                  + b"\x00" * 24)
        self.assertEqual(self._kind(header, "pic.bmp"), "image")

    def test_a_real_pe_is_still_an_executable(self):
        head = bytearray(b"MZ" + b"\x00" * 62)
        head[0x3C:0x40] = (64).to_bytes(4, "little")
        head += b"PE\x00\x00" + b"\x00" * 20
        self.assertEqual(self._kind(bytes(head), "app.exe"), "pe-executable")

    def test_utf16_text_is_text_not_binary(self):
        blob = "Hallo Welt\nZweite Zeile: Größe\n".encode("utf-16")
        self.assertEqual(self._kind(blob, "notes.txt"), "text")

    def test_utf16le_without_bom_is_decoded_properly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_bytes("name,wert\nfoo,42\nbar,7\n".encode("utf-16-le"))
            ex = extract(p, detect(p), 25_000_000)
        self.assertIn("utf-16", ex.metadata["encoding"])
        self.assertEqual(ex.metadata["column_profile"][0]["name"], "name")

    def test_a_named_pipe_is_refused_instead_of_blocking(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fifo"
            os.mkfifo(p)
            with self.assertRaises(OSError) as ctx:
                detect(p)
        self.assertIn("not a regular file", str(ctx.exception))


class ContainerAndFormatTest(unittest.TestCase):
    """Formats whose structure was being flattened or mislabelled."""

    def _extract(self, name: str, write):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            write(p)
            return detect(p), extract(p, detect(p), 25_000_000)

    def test_html_table_cells_are_separated(self):
        """'RegionSales / North100' welded every label to its number."""
        html = ("<table><tr><th>Region</th><th>Sales</th></tr>"
                "<tr><td>North</td><td>100</td></tr></table>")
        _, ex = self._extract("t.html", lambda p: p.write_text(html))
        self.assertNotIn("RegionSales", ex.text)
        self.assertNotIn("North100", ex.text)
        self.assertIn("North", ex.text)

    def test_declared_charset_is_honoured(self):
        page = '<html><head><meta charset="windows-1251"></head><body>Привет</body></html>'
        _, ex = self._extract("t.html", lambda p: p.write_bytes(page.encode("cp1251")))
        self.assertIn("Привет", ex.text)
        self.assertEqual(ex.metadata["encoding"], "windows-1251")

    def test_pdf_kerning_arrays_become_spaces(self):
        from sift.extract.pdf import _kerned
        self.assertEqual(_kerned(b"(Page)-250(two)-250(here)"), "Page two here")
        self.assertEqual(_kerned(b"(Kern)-20(ing)"), "Kerning")

    def test_page_count_does_not_count_the_page_tree(self):
        from sift.extract.pdf import _page_count
        self.assertEqual(_page_count(b"<< /Type /Pages /Kids [...] /Count 7 >>"), 7)

    def test_a_truncated_webp_has_unknown_size_not_one_by_one(self):
        from sift.extract.image import _dimensions
        self.assertEqual(_dimensions(b"RIFF\x24\x00\x00\x00WEBPVP8X"),
                         ("WEBP", None, None))

    def test_a_truncated_png_still_reports_its_format(self):
        from sift.extract.image import _dimensions
        self.assertEqual(_dimensions(b"\x89PNG\r\n\x1a\n\x00\x00"),
                         ("PNG", None, None))

    def test_line_count_ignores_form_feeds(self):
        _, ex = self._extract("t.txt", lambda p: p.write_text("a\nb\x0cc\nd\n"))
        self.assertEqual(ex.metadata["lines"], 3)

    def test_two_lines_of_prose_are_not_a_table(self):
        det, _ = self._extract("t.txt",
                               lambda p: p.write_text("Hello, world\nGoodbye, world\n"))
        self.assertEqual(det.kind, "text")

    def test_a_real_csv_is_still_detected_without_its_extension(self):
        det, _ = self._extract("t.txt", lambda p: p.write_text(
            "id,name,dept\n1,Ada,Eng\n2,Bob,Sales\n3,Cy,Ops\n4,Di,Eng\n"))
        self.assertEqual(det.kind, "csv")

    def test_an_mbox_is_read_as_every_message(self):
        body = "".join(
            f"From s{i}@x.com Mon Jan  1 00:0{i}:00 2024\nFrom: s{i}@x.com\n"
            f"To: r@y.com\nSubject: Message number {i}\n\nBody {i}.\n\n"
            for i in (1, 2, 3))
        _, ex = self._extract("m.mbox", lambda p: p.write_text(body))
        self.assertEqual(ex.metadata["message_count"], 3)
        self.assertEqual(len(ex.metadata["senders"]), 3)

    def test_a_zip_holding_an_xl_path_is_not_a_workbook(self):
        import zipfile

        def write(p):
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("xl/notes.txt", "hi")
                z.writestr("readme.md", "hello")
        det, ex = self._extract("t.zip", write)
        self.assertEqual(det.kind, "archive")
        self.assertIn("xl/notes.txt", ex.metadata["archive_members"])

    def test_an_ods_is_not_reported_as_a_text_document(self):
        import zipfile

        def write(p):
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("mimetype",
                           "application/vnd.oasis.opendocument.spreadsheet")
                z.writestr("content.xml",
                           '<?xml version="1.0"?><o xmlns:t="urn:t">'
                           '<t:p>Umsatz 100</t:p></o>')
        det, ex = self._extract("book.ods", write)
        self.assertEqual(det.kind, "ods")
        self.assertIn("Umsatz", ex.text)

    def test_a_tar_archive_lists_its_members(self):
        import io
        import tarfile

        def write(p):
            with tarfile.open(p, "w:gz") as t:
                for n in ("a.txt", "b/c.txt"):
                    info = tarfile.TarInfo(n)
                    info.size = 5
                    t.addfile(info, io.BytesIO(b"hello"))
        _, ex = self._extract("bundle.tar.gz", write)
        self.assertEqual(sorted(ex.metadata["archive_members"]), ["a.txt", "b/c.txt"])

    def test_truncation_is_not_reported_as_a_malformed_document(self):
        valid = '{"a": 1, "b": {"c": [1, 2, 3]}, "d": "' + "x" * 400 + '"}'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(valid)
            meta = extract(p, detect(p), 200).metadata
        self.assertIn("parse_error_cause", meta)
        self.assertIn("max_bytes", meta["parse_error_cause"])


class LogSeverityTest(unittest.TestCase):
    """A log is read for its worst moment, not its first."""

    def _meta(self, body: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.log"
            p.write_text(body, encoding="utf-8")
            return extract(p, detect(p), 25_000_000).metadata

    def test_a_critical_in_the_middle_is_shown_before_early_warnings(self):
        lines = [f"2026-01-01 00:00:{i:02d} WARNING svc slow request {i}"
                 for i in range(20)]
        lines.insert(10, "2026-01-01 00:00:10 CRITICAL svc database unreachable")
        samples = self._meta("\n".join(lines) + "\n")["error_samples"]
        self.assertIn("CRITICAL", samples[0])

    def test_errors_come_before_warnings(self):
        body = ("2026-01-01 WARNING a slow\n" * 5
                + "2026-01-01 ERROR b failed\n")
        samples = self._meta(body)["error_samples"]
        self.assertIn("ERROR", samples[0])

    def test_the_sample_says_how_many_it_stands_for(self):
        body = "2026-01-01 ERROR svc failure\n" * 50
        meta = self._meta(body)
        self.assertIn("50", meta["error_samples_note"])

    def test_equal_severity_keeps_file_order(self):
        body = "".join(f"2026-01-01 ERROR svc failure {i}\n" for i in range(4))
        samples = self._meta(body)["error_samples"]
        self.assertIn("failure 0", samples[0])
        self.assertIn("failure 1", samples[1])


class BatchRobustnessTest(unittest.TestCase):
    """One unreadable file must not cost the other nine.

    Both layers promise this — extract() catches everything so a bad file
    becomes a record rather than an exception, and summarize_path's docstring
    says it never raises. Neither promise had a test that made them keep it.
    """

    def test_an_extractor_that_raises_becomes_a_record(self):
        from unittest import mock
        from sift.extract import extract as dispatch
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            det = detect(p)
            with mock.patch("sift.extract.tabular.extract",
                            side_effect=RuntimeError("boom")):
                ex = dispatch(p, det, 25_000_000)
        self.assertIn("boom", ex.metadata["extractor_error"])
        self.assertTrue(ex.notes)

    def test_a_batch_survives_a_file_that_explodes(self):
        from unittest import mock
        from sift.config import Config
        from sift.summarize import summarize_path
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.csv"
            bad = Path(d) / "bad.csv"
            good.write_text("id,name\n1,Ada\n", encoding="utf-8")
            bad.write_text("id,name\n2,Bob\n", encoding="utf-8")
            real = __import__("sift.extract.tabular", fromlist=["extract"]).extract

            def sometimes(path, det, max_bytes):
                if path.name == "bad.csv":
                    raise MemoryError("out of memory")
                return real(path, det, max_bytes)

            with mock.patch("sift.extract.tabular.extract", sometimes):
                results = [summarize_path(p, Config(no_llm=True))
                           for p in (good, bad)]
        self.assertIn("1 row and 2 columns", results[0].summary)
        self.assertIn("out of memory", str(results[1].metadata))
        self.assertIsNone(results[0].error)

    def test_summarize_path_never_raises_on_hostile_input(self):
        from sift.config import Config
        from sift.summarize import summarize_path
        hostile = {
            "empty.csv": b"",
            "nul.txt": b"\x00" * 100,
            "one-quote.csv": b'"',
            "bad.json": b"{",
            "bad.xml": b"<a>",
            "bad.zip": b"PK\x03\x04garbage",
            "bad.png": b"\x89PNG\r\n\x1a\n",
            "bad.pdf": b"%PDF-1.4 truncated",
            "bad.eml": b"From: \n\n",
            "highbytes.txt": bytes(range(128, 256)),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, body in hostile.items():
                p = Path(d) / name
                p.write_bytes(body)
                summ = summarize_path(p, Config(no_llm=True))
                self.assertTrue(summ.summary.strip() or summ.error, name)


class LogDigestTest(unittest.TestCase):
    """A log is mostly repetition; the model should see the signal.

    Sending 200,000 characters of log through eleven model calls cost five
    minutes and buried the four lines that matter. The extractor already knows
    which lines those are.
    """

    def _digest(self, body: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.log"
            p.write_text(body, encoding="utf-8")
            return extract(p, detect(p), 25_000_000).digest

    def _big_log(self) -> str:
        lines = [f"2026-01-01 00:00:{i % 60:02d} INFO svc routine event {i}"
                 for i in range(4000)]
        lines[1500] = "2026-01-01 00:25:00 CRITICAL svc database unreachable"
        lines[3000] = "2026-01-01 00:50:00 ERROR svc disk full"
        return "\n".join(lines) + "\n"

    def test_every_flagged_line_survives(self):
        digest = self._digest(self._big_log())
        self.assertIn("database unreachable", digest)
        self.assertIn("disk full", digest)

    def test_it_states_the_counts_for_the_whole_file(self):
        digest = self._digest(self._big_log())
        self.assertIn("log file of 4000 lines", digest)
        self.assertIn("CRITICAL: 1", digest)

    def test_it_says_it_is_an_extract(self):
        """A summary built from a sample must not read as one of everything."""
        self.assertIn("not every line", self._digest(self._big_log()))

    def test_it_is_far_smaller_than_the_file(self):
        body = self._big_log()
        self.assertLess(len(self._digest(body)), len(body) * 0.1)

    def test_ordinary_lines_are_sampled_across_the_file(self):
        digest = self._digest(self._big_log())
        self.assertIn("event 0", digest)
        self.assertTrue(any(f"event {n}" in digest for n in range(3500, 4000)),
                        "the sample should reach the end of the file")

    def test_only_logs_get_one(self):
        self.assertIsNone(self._digest("just some prose\n").__class__ is str
                          and None)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            p.write_text("just some prose\n", encoding="utf-8")
            self.assertIsNone(extract(p, detect(p), 25_000_000).digest)


class ZipDirectoryLimitTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _zip(self, names):
        import zipfile

        path = self.dir / "many.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, b"")
        return path

    def test_an_ordinary_zip_is_still_listed(self):
        path = self._zip(["one.txt", "two.txt"])
        ex = extract(path, detect(path), BIG)
        self.assertEqual(ex.metadata["archive_member_count"], 2)
        self.assertEqual(ex.metadata["archive_members"], ["one.txt", "two.txt"])
        self.assertNotIn("archive_limit", ex.metadata)

    def test_too_many_entries_are_rejected_before_zipfile_opens_them(self):
        from unittest import mock
        from sift import zipguard

        path = self._zip([f"empty-{i}" for i in range(4)])
        with mock.patch.object(zipguard, "MAX_ZIP_ENTRIES", 3), \
             mock.patch("zipfile.ZipFile",
                        side_effect=AssertionError("central directory parsed")) as opened:
            det = detect(path)
            ex = extract(path, det, BIG)

        opened.assert_not_called()
        self.assertEqual(det.kind, "archive")
        self.assertEqual(ex.metadata["archive_member_count"], 4)
        self.assertIn("entry limit", ex.metadata["archive_limit"])

    def test_an_oversize_central_directory_is_rejected_too(self):
        from unittest import mock
        from sift import zipguard

        path = self._zip(["a-name-long-enough-to-have-a-directory-record.txt"])
        with mock.patch.object(zipguard, "MAX_ZIP_DIRECTORY_BYTES", 1), \
             mock.patch("zipfile.ZipFile",
                        side_effect=AssertionError("central directory parsed")) as opened:
            ex = extract(path, detect(path), BIG)

        opened.assert_not_called()
        self.assertEqual(ex.metadata["archive_member_count"], 1)
        self.assertIn("directory-size limit", ex.metadata["archive_limit"])
