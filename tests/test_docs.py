"""The documentation says numbers. The numbers have to be true.

A README that quietly falls behind the code is worse than no README: it
states its figures with the same confidence whether or not they still hold,
and a reader has no way to tell. The three that go stale on their own — the
test count, the language count, and the two documents agreeing with each
other — are checked here.

The count comes from *loading* the suite, not from running it. Running it
from inside itself is a recursion, and the first version of this file did
exactly that: it spawned the whole suite in a subprocess, which loaded this
module, which spawned the whole suite.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGLISH = ROOT / "README.md"
GERMAN = ROOT / "README.de.md"

#: The number words the README is allowed to use for the language count.
_WORDS = {"twenty-one": 21, "twenty-two": 22, "twenty-three": 23,
          "twenty-four": 24, "twenty-five": 25, "twenty-six": 26,
          "twenty-seven": 27, "twenty-eight": 28, "twenty-nine": 29,
          "thirty": 30, "thirty-one": 31, "thirty-two": 32}
_WORDS.update({"twenty-nine": 29})


def _count_tests() -> int:
    """Loaded the way the README says to run it: `discover -s tests`.

    With a different top_level_dir the modules import as `tests.test_x` and
    the ones that do `from corpus_secrets import …` fail to import — each
    failure counting as one test instead of its real thirty, which is how
    the first version of this reported 932 for a suite of 964.
    """
    import sys
    here = str(ROOT / "tests")
    added = here not in sys.path
    if added:
        sys.path.insert(0, here)
    try:
        return unittest.TestLoader().discover(here).countTestCases()
    finally:
        if added:
            sys.path.remove(here)


class ReadmeNumbersTest(unittest.TestCase):

    def test_both_documents_exist_and_neither_is_a_stub(self):
        """The goal was documentation in German and English, not a summary."""
        for path in (ENGLISH, GERMAN):
            self.assertTrue(path.is_file(), path.name)
        english, german = (len(p.read_text(encoding="utf-8").splitlines())
                           for p in (ENGLISH, GERMAN))
        self.assertGreater(
            german, english * 0.7,
            f"README.de.md is {german} lines against {english} — that is a "
            f"summary, and the point of having both is that neither reader "
            f"gets the short answer")

    def test_the_stated_test_count_is_the_real_one(self):
        stated = {}
        for path in (ENGLISH, GERMAN):
            found = re.search(r"([\d,]{3,6}) [Tt]ests,",
                              path.read_text(encoding="utf-8"))
            self.assertIsNotNone(found, f"{path.name} states no test count")
            stated[path.name] = int(found.group(1).replace(",", ""))
        self.assertEqual(stated[ENGLISH.name], stated[GERMAN.name],
                         f"the two documents disagree: {stated}")
        actual = _count_tests()
        # Not exact: a commit that adds a test before the commit that updates
        # the README is a nuisance, not a defect. Off by more than a few is
        # the README having stopped tracking the suite at all.
        self.assertLessEqual(
            abs(actual - stated[ENGLISH.name]), 15,
            f"the README says {stated[ENGLISH.name]} tests; the suite holds "
            f"{actual}")

    def test_the_stated_language_count_is_the_real_one(self):
        """It said twenty-one for a while after there were twenty-eight."""
        import sys
        here = str(ROOT / "tests")
        if here not in sys.path:
            sys.path.insert(0, here)
        from corpus_injection import LANGUAGE_ATTACKS
        patterns = len(LANGUAGE_ATTACKS)
        found = re.search(r"(twenty-\w+|thirty(?:-\w+)?) languages",
                          ENGLISH.read_text(encoding="utf-8"))
        self.assertIsNotNone(found, "the README states no language count")
        stated = _WORDS.get(found.group(1))
        self.assertIsNotNone(stated, f"unrecognised number word: {found.group(1)}")
        self.assertEqual(
            patterns, stated,
            f"the README claims {stated} languages; the corpus holds "
            f"{patterns}, and every one of them is a tested sentence")

    def test_the_stated_corpus_sizes_are_the_real_ones(self):
        """Both numbers grew and the README kept the old ones for a while."""
        import sys
        here = str(ROOT / "tests")
        if here not in sys.path:
            sys.path.insert(0, here)
        from corpus_injection import (ATTACKS, ORDINARY, TRUE_STATEMENTS,
                                      embeddings)
        attacks = sum(len(embeddings(a if isinstance(a, str) else a[0]))
                      for a in ATTACKS)
        ordinary = sum(len(embeddings(x if isinstance(x, str) else x[0]))
                       for x in list(ORDINARY) + list(TRUE_STATEMENTS))
        english = ENGLISH.read_text(encoding="utf-8")
        claimed = re.search(r"\*\*([\d,]+) of ([\d,]+) payloads reported, "
                            r"0 of ([\d,]+) ordinary", english)
        self.assertIsNotNone(claimed, "the README states no corpus sizes")
        stated_attacks = int(claimed.group(2).replace(",", ""))
        stated_ordinary = int(claimed.group(3).replace(",", ""))
        self.assertEqual(claimed.group(1), claimed.group(2),
                         "the README claims some payloads are missed")
        self.assertEqual(attacks, stated_attacks,
                         f"README says {stated_attacks} payloads, "
                         f"the corpus holds {attacks}")
        self.assertEqual(ordinary, stated_ordinary,
                         f"README says {stated_ordinary} ordinary sentences, "
                         f"the corpus holds {ordinary}")

    def test_the_german_document_covers_the_same_sections(self):
        """Not the same words — the same ground."""
        # IT terms stay English in the German text too: its headings say
        # "Credentials", "Configuration", "Web UI" and "CLI".
        required = ("Credentials", "Warnungen", "Was sift nicht tut",
                    "Wie es arbeitet", "Tests", "Configuration",
                    "Web UI", "CLI", "Lizenz")
        german = GERMAN.read_text(encoding="utf-8")
        missing = [heading for heading in required
                   if f"## {heading}" not in german]
        self.assertEqual([], missing, f"README.de.md is missing {missing}")
        english = ENGLISH.read_text(encoding="utf-8")
        count = lambda text: sum(line.startswith("## ")
                                 for line in text.splitlines())
        self.assertEqual(count(english), count(german),
                         "README.md and README.de.md have as many sections")


if __name__ == "__main__":
    unittest.main()
