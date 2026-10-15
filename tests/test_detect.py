import shutil
import tempfile
import unittest
from pathlib import Path

from sift.detect import detect
from tests import _fixtures as fx


class DetectTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        # Removed when the test ends. Without this the suite left one
        # directory per test method behind — 43 a run, and a machine
        # that had run it often enough held 45,000 of them.
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _f(self, name, text):
        return fx.write(self.dir / name, text)

    def test_csv_by_content(self):
        # .txt extension but CSV content — content should win
        p = self._f("data.txt", fx.CSV_TEXT)
        self.assertEqual(detect(p).kind, "csv")

    def test_json(self):
        self.assertEqual(detect(self._f("a.json", fx.JSON_TEXT)).kind, "json")

    def test_email(self):
        self.assertEqual(detect(self._f("m.eml", fx.EML_TEXT)).kind, "email")

    def test_batch(self):
        self.assertEqual(detect(self._f("b.bat", fx.BAT_TEXT)).kind, "batch")

    def test_html(self):
        self.assertEqual(detect(self._f("p.html", fx.HTML_TEXT)).kind, "html")

    def test_markdown(self):
        self.assertEqual(detect(self._f("n.md", "# Hi\n\ntext")).kind, "markdown")

    def test_png_is_image(self):
        det = detect(fx.make_png(self.dir / "i.png"))
        self.assertEqual(det.kind, "image")
        self.assertTrue(det.is_binary)

    def test_docx(self):
        self.assertEqual(detect(fx.make_docx(self.dir / "d.docx")).kind, "docx")

    def test_xlsx(self):
        self.assertEqual(detect(fx.make_xlsx(self.dir / "s.xlsx")).kind, "xlsx")

    def test_empty(self):
        self.assertEqual(detect(self._f("e.txt", "")).kind, "empty")

    def test_binary(self):
        p = self.dir / "raw.bin"
        p.write_bytes(bytes([0, 1, 2, 3, 255, 254, 0, 0]) * 20)
        det = detect(p)
        self.assertTrue(det.is_binary)
        self.assertEqual(det.kind, "binary")

    def test_plain_text(self):
        p = self._f("readme.txt", "Just some prose without structure.\n")
        self.assertEqual(detect(p).kind, "text")

    def test_utf8_cjk_and_emoji_is_text_not_binary(self):
        # high-byte-heavy UTF-8 must not be misclassified as binary
        p = self._f("uni.txt", "日本語のテキスト 🎉🎊 中文内容 émojis\n" * 50)
        det = detect(p)
        self.assertFalse(det.is_binary)
        self.assertEqual(det.kind, "text")

    def test_german_umlauts_text(self):
        p = self._f("de.txt", "Schöne Grüße über die Straße — Mötörhead\n" * 30)
        self.assertFalse(detect(p).is_binary)

    def test_png_magic_beats_wrong_ext(self):
        # PNG bytes with a .txt name must still be detected as image
        det = detect(fx.make_png(self.dir / "fake.txt"))
        self.assertEqual(det.kind, "image")


if __name__ == "__main__":
    unittest.main()
