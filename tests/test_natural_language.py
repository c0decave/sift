"""detect_natural_language: German, English, or neither.

Summary.language is the *programming* language — "python" for a .py file —
and full-text search needs the other one: which stemmer to index a file
with. A wrong answer picks the wrong stemmer; no answer picks the neutral
one. So the function says "de" or "en" only when the text says so, and None
for everything else — other languages, code, tables, a single word.

It reads what it is given. For a file whose text sift renders — a CSV shown
as "# Tabular data", "## Schema", "## First rows" — the rendering's English
headings would make every file English, so the summary asks it about the
file's own bytes instead.
"""
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.config import Config
from sift.natural_language import detect_natural_language
from sift.summarize import summarize_path
from tests import _fixtures as fx

import corpus_language as corpus

GERMAN = ("Sehr geehrte Damen und Herren, anbei erhalten Sie die Rechnung für "
          "den Monat März. Bitte überweisen Sie den Betrag bis zum 15. April.")
ENGLISH = ("Dear customer, please find attached the invoice for March. The "
           "amount is due by April 15 and can be paid by bank transfer.")


def measure(passages, expected):
    """(right, wrong, declined): answered as expected, answered otherwise,
    answered None."""
    right = wrong = declined = 0
    for text in passages:
        code, _confidence = detect_natural_language(text)
        if code is None:
            declined += 1
        elif code == expected:
            right += 1
        else:
            wrong += 1
    return right, wrong, declined


class DetectNaturalLanguageTest(unittest.TestCase):

    # -- happy -------------------------------------------------------------
    def test_german_prose_is_german(self):
        code, confidence = detect_natural_language(GERMAN)
        self.assertEqual("de", code)
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)

    def test_english_prose_is_english(self):
        code, confidence = detect_natural_language(ENGLISH)
        self.assertEqual("en", code)
        self.assertGreater(confidence, 0.5)

    def test_german_written_without_umlauts_is_still_german(self):
        code, _ = detect_natural_language(
            "Fuer die Abrechnung brauchen wir noch die Belege, die ueber das "
            "Portal eingereicht wurden, sonst koennen wir nichts erstatten.")
        self.assertEqual("de", code)

    # -- sad ---------------------------------------------------------------
    def test_nothing_to_read_is_no_language(self):
        for text in ("", "   \n\t", None):
            with self.subTest(text=text):
                self.assertEqual((None, 0.0), detect_natural_language(text))

    def test_code_and_tables_are_no_language(self):
        for text in (fx.CSV_TEXT, fx.JSON_TEXT, fx.LOG_TEXT,
                     "def f(x):\n    return x * 2\n"):
            with self.subTest(text=text[:30]):
                self.assertIsNone(detect_natural_language(text)[0])

    def test_neighbouring_languages_are_not_mistaken_for_either(self):
        """Dutch and Danish share function words with German; French and
        Spanish with English. None of them is either."""
        for text in corpus.NEITHER[:6]:
            with self.subTest(text=text[:40]):
                self.assertIsNone(detect_natural_language(text)[0])

    # -- edge --------------------------------------------------------------
    def test_one_word_is_not_enough_to_tell(self):
        for text in ("Hallo", "Invoice", "und", "the"):
            with self.subTest(text=text):
                self.assertIsNone(detect_natural_language(text)[0])

    def test_half_and_half_is_not_either(self):
        code, _ = detect_natural_language(GERMAN + "\n" + ENGLISH)
        self.assertIsNone(code)

    def test_a_little_english_in_a_german_text_is_still_german(self):
        code, _ = detect_natural_language(
            GERMAN + " " + GERMAN + " Siehe auch: the attached file.")
        self.assertEqual("de", code)

    def test_a_long_text_is_read_in_bounded_time(self):
        text = (GERMAN + "\n") * 40_000               # ~6 MB
        started = time.monotonic()
        code, _ = detect_natural_language(text)
        self.assertEqual("de", code)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_case_does_not_matter(self):
        self.assertEqual("de", detect_natural_language(GERMAN.upper())[0])
        self.assertEqual("en", detect_natural_language(ENGLISH.upper())[0])


class CorpusPrecisionTest(unittest.TestCase):
    """Precision in both directions, and no confusion between the two.

    Recall matters less than precision here: a declined passage is indexed
    with the neutral configuration, which still finds every word; a wrong one
    is stemmed by rules for another language. Measured when this was
    written: German 23/24 and 10/12 held out, English 23/24 and 11/12, none
    of them called the other language and none of the 26 others answered —
    the declined ones are passages of 13 to 17 words with three distinct
    function words, below MIN_DISTINCT. The floor below is under that, so
    it catches a regression, not noise.
    """

    def test_german_is_never_called_english_and_mostly_german(self):
        for passages in (corpus.GERMAN, corpus.GERMAN_HELD_OUT):
            right, wrong, declined = measure(passages, "de")
            self.assertEqual(0, wrong)
            self.assertGreaterEqual(right / len(passages), 0.8,
                                    f"{right} of {len(passages)}")

    def test_english_is_never_called_german_and_mostly_english(self):
        for passages in (corpus.ENGLISH, corpus.ENGLISH_HELD_OUT):
            right, wrong, declined = measure(passages, "en")
            self.assertEqual(0, wrong)
            self.assertGreaterEqual(right / len(passages), 0.8,
                                    f"{right} of {len(passages)}")

    def test_everything_else_is_declined(self):
        for passages in (corpus.NEITHER, corpus.NEITHER_HELD_OUT):
            answered = [(text[:40], detect_natural_language(text))
                        for text in passages
                        if detect_natural_language(text)[0] is not None]
            self.assertEqual([], answered)


class SummaryNaturalLanguageTest(unittest.TestCase):
    """Summary.natural_language: from the file's bytes for a text type, from
    sift's rendering without sift's headings for everything else."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def summarize(self, name, text, **config):
        path = fx.write(self.dir / name, text)
        return summarize_path(path, Config(no_llm=True, cache=False,
                                           **config))

    # -- happy -------------------------------------------------------------
    def test_a_german_text_file_is_german(self):
        self.assertEqual("de", self.summarize("brief.txt", GERMAN)
                         .natural_language)

    def test_a_csv_is_read_from_its_bytes_not_from_the_rendering(self):
        """The rendering is sift's: "# Tabular data", "## Schema", a few
        sample values and the first rows. A table whose words start after
        those rows reads as nothing there, and as German in the file."""
        from sift.detect import detect
        from sift.extract import extract
        notes = ("Die Rechnung wurde bezahlt und ist erledigt",
                 "Der Beleg fehlt noch, aber der Betrag ist gebucht",
                 "Kunde hat sich nicht gemeldet, bitte nachfassen",
                 "Wird mit der nächsten Lieferung verrechnet")
        rows = ([f"{i};-;{i * 10}" for i in range(1, 31)]
                + [f"{i};{notes[i % len(notes)]};{i * 10}"
                   for i in range(31, 61)])
        summ = self.summarize("buchungen.csv",
                              "Nr;Vermerk;Betrag\n" + "\n".join(rows))
        self.assertEqual("csv", summ.kind)
        self.assertEqual("de", summ.natural_language)
        path = self.dir / "buchungen.csv"
        rendering = extract(path, detect(path), 25_000_000).text
        self.assertIn("## Schema", rendering)
        self.assertIsNone(detect_natural_language(rendering)[0])

    def test_an_email_is_read_without_sifts_headings(self):
        mail = fx.EML_TEXT.replace(
            "Hello team, the revenue is up 12 percent this quarter.",
            GERMAN + " " + GERMAN)
        summ = self.summarize("m.eml", mail)
        self.assertEqual("email", summ.kind)
        self.assertEqual("de", summ.natural_language)

    def test_it_is_part_of_the_json(self):
        data = self.summarize("brief.txt", ENGLISH).to_dict()
        self.assertEqual("en", data["natural_language"])

    # -- sad ---------------------------------------------------------------
    def test_a_table_of_numbers_has_no_language(self):
        summ = self.summarize("d.csv", fx.CSV_TEXT)
        self.assertIsNone(summ.natural_language)
        self.assertIn("natural_language", summ.to_dict())
        self.assertIsNone(summ.to_dict()["natural_language"])

    def test_an_image_has_no_language(self):
        path = fx.make_png(self.dir / "i.png")
        summ = summarize_path(path, Config(no_llm=True, cache=False))
        self.assertIsNone(summ.natural_language)

    def test_an_unreadable_path_has_no_language(self):
        summ = summarize_path(self.dir / "missing.txt",
                              Config(no_llm=True, cache=False))
        self.assertEqual("error", summ.kind)
        self.assertIsNone(summ.natural_language)

    # -- edge --------------------------------------------------------------
    def test_a_cache_hit_keeps_the_language(self):
        from sift.cache import SummaryCache
        cache = SummaryCache(directory=self.dir / "cache")
        path = fx.write(self.dir / "brief.txt", GERMAN)
        config = Config(no_llm=True)
        first = summarize_path(path, config, cache=cache)
        again = summarize_path(path, config, cache=cache)
        self.assertTrue(again.from_cache)
        self.assertEqual("de", first.natural_language)
        self.assertEqual("de", again.natural_language)


if __name__ == "__main__":
    unittest.main()
