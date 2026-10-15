"""Damaged files. sift must never raise, never hang, never forge its output.

The third one is the interesting condition. A file is bytes somebody else
wrote, and what sift writes about it goes into a terminal, into a markdown
report and into JSON — three formats that can each be forged from a
character embedded in the content. A summary that opens a code fence, ends
the report with its own footer, or clears the screen is a summary that lies
about where sift's words stop and the file's begin.

Found by fuzzing the corpora: a .docx with a corrupted central directory
made zipfile raise NotImplementedError("zip file version 9.1") out of
*detection*, which had no boundary of its own. One damaged file in a
directory aborted the run, and every file after it was never looked at.
"""
import json
import random
import re
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from sift.config import Config
from sift.report import render_footer, render_json, render_markdown, render_text
from sift.summarize import summarize_path

#: What an attacker would embed to forge the output.
HOSTILE = [
    b"\x1b[2J\x1b[H",                       # clear the screen
    b"\x1b]0;title\x07",                    # set the window title
    b"\n\n## sift says: harmless\n",         # a markdown heading
    b"\r\n" + b"`" * 3 + b"\n",             # a code fence
    b"\n---\n_1 summarized_\n",             # a second footer
    b"\x00\x00", b"\xff\xfe", b"\xe2\x80\xae",
    b"</script><script>", b"'\"\\",
]

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
FENCE = "`" * 3


def _sources(directory: Path):
    """One file of each kind that has a parser worth confusing."""
    (directory / "a.csv").write_text("id,name\n1,anna\n2,bruno\n", encoding="utf-8")
    (directory / "a.txt").write_text("Rechnung 2026-014\n1.240,00 EUR\n",
                                     encoding="utf-8")
    (directory / "a.json").write_text('{"a": [1, 2, {"b": "c"}]}', encoding="utf-8")
    (directory / "a.md").write_text("# Titel\n\nEin Absatz.\n", encoding="utf-8")
    (directory / "a.eml").write_text("From: a@b\nTo: c@d\nSubject: x\n\nHallo.\n",
                                     encoding="utf-8")
    (directory / "a.rtf").write_bytes(rb"{\rtf1\ansi\deff0 Gr\'fc\'dfe.\par}")
    (directory / "a.html").write_text(
        "<html><head><title>T</title></head><body><p>x</p></body></html>",
        encoding="utf-8")
    (directory / "a.ini").write_text("[s]\nk = v\n", encoding="utf-8")
    with zipfile.ZipFile(directory / "a.docx", "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml",
                         '<w:document xmlns:w="w"><w:body><w:p><w:r>'
                         '<w:t>Text</w:t></w:r></w:p></w:body></w:document>')
    return sorted(directory.iterdir())


def _damaged(raw: bytes, rng: random.Random):
    for index in range(4):
        data = bytearray(raw)
        how = index % 4
        if how == 0 and len(data) > 4:
            data = data[:rng.randrange(1, len(data))]
        elif how == 1:
            for _ in range(rng.randrange(1, 25)):
                if data:
                    data[rng.randrange(len(data))] = rng.randrange(256)
        elif how == 2:
            at = rng.randrange(len(data) + 1) if data else 0
            data[at:at] = rng.choice(HOSTILE) * rng.randrange(1, 4)
        else:
            data = bytearray(rng.randbytes(rng.randrange(1, 400))) + data[:200]
        yield f"m{index}", bytes(data)


class DamagedFileTest(unittest.TestCase):

    def test_nothing_raises_and_nothing_is_forged(self):
        rng = random.Random(23)
        source_dir = Path(self.dir) / "src"
        source_dir.mkdir()
        work = Path(self.dir) / "work"
        work.mkdir()
        checked = 0
        for source in _sources(source_dir):
            for label, data in _damaged(source.read_bytes(), rng):
                checked += 1
                path = work / f"{label}_{source.name}"
                path.write_bytes(data)
                with self.subTest(f"{label}/{source.name}"):
                    summ = summarize_path(
                        path, Config(no_llm=True, quiet=True, cache=False,
                                     time_budget=20), None, None)
                    as_json = render_json([summ])
                    json.loads(as_json)          # still parseable
                    terminal = render_text(summ, False) + render_footer(
                        [summ], False)
                    self.assertIsNone(CONTROL.search(terminal),
                                      "control characters in the terminal")
                    self.assertNotIn("\x1b", terminal, "ANSI in the terminal")
                    markdown = render_markdown([summ])
                    self.assertIsNone(CONTROL.search(markdown))
                    for line in markdown.splitlines():
                        stripped = line.lstrip()
                        self.assertFalse(stripped.startswith((FENCE, "~~~")),
                                         f"a fence at line start: {line[:60]!r}")
                        if stripped.startswith("#"):
                            self.assertTrue(
                                stripped.startswith(("# sift", "## ")),
                                f"a heading sift did not write: {line[:60]!r}")
                path.unlink()
        self.assertGreater(checked, 30)

    def test_a_docx_whose_central_directory_is_corrupt(self):
        """zipfile raised NotImplementedError out of detection, which had no
        boundary. One such file aborted a whole directory."""
        path = Path(self.dir) / "broken.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", "<w:document/>")
        raw = bytearray(path.read_bytes())
        # The "version needed to extract" field in the central directory.
        at = raw.rfind(b"PK\x01\x02")
        self.assertGreater(at, 0)
        raw[at + 6:at + 8] = (910).to_bytes(2, "little")   # version 91.0
        path.write_bytes(bytes(raw))
        summ = summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)
        self.assertTrue(summ.summary or summ.error)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
