"""A secret value that holds whitespace other than one blank.

sift's rule for a value in prose -- "the new vault password: Kx9v…" --
ended it at the first whitespace, whatever that was. Written with a tab,
"password: Kx9v<TAB>Q2mZ7pL4Wq8rTz" had "Kx9v" hidden and "Q2mZ7pL4Wq8rTz"
published by the summary of the mail that held it, and by every viewer that
masks what sift finds. The same with two blanks, a no-break space, an em
space -- and the tab is what sift itself turns into four blanks where it
folds a line (one_line).

The rules that read a whole line -- redact_assignments, find_secrets'
keyed lines -- took the rest of the line all along; the ones that read a
value inside prose (_SECRET_IN_SENTENCE, requirepass and its family, a
.pgpass line) end it where prose ends a word: at one plain blank, a comma,
a semicolon, a bracket -- not at a tab or a run of blanks, which no
sentence puts between two words.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.config import Config
from sift.extract import find_secrets
from sift.extract.base import (hide_known, learn_secrets, one_line,
                               redact_value, remember_hidden, scrub)
from sift.llm.base import LLMResult
from sift.report import render_json, render_markdown, render_text
from sift.summarize import summarize_path

HEAD, TAIL = "Kx9v", "Q2mZ7pL4Wq8rTz"
#: Whitespace that no sentence puts between two words.
GAPS = {"tab": "\t", "two blanks": "  ", "four blanks": "    ",
        "no-break space": " ", "em space": " ",
        "narrow no-break space": " ", "ideographic space": "　",
        "blank and tab": " \t"}
#: The shapes a value takes inside text rather than as a line of its own.
LINES = {
    "assignment in prose": "the new vault password: {value}",
    "copula": "the vault password is {value}",
    "German": "Das Kennwort lautet {value}",
    "keyed line": "password: {value}",
    "requirepass": "requirepass {value}",
    "pgpass line": "db.intern:5432:app:svc:{value}",
}


def _value(gap: str) -> str:
    return f"{HEAD}{gap}{TAIL}"


class EveryRuleTest(unittest.TestCase):
    """Each rule that reads a value, on each gap, in each shape."""

    def test_scrub_hides_the_whole_value(self):
        failures = [(label, name) for label, line in LINES.items()
                    for name, gap in GAPS.items()
                    if TAIL in scrub(line.format(value=_value(gap)))]
        self.assertEqual([], failures)

    def test_find_secrets_spans_the_whole_value(self):
        failures = []
        for label, line in LINES.items():
            for name, gap in GAPS.items():
                text = f"Hi team,\n{line.format(value=_value(gap))}\nbye\n"
                spans = [text[s.start:s.end] for s in find_secrets(text)
                         if s.reason != "bidi"]
                if not any(TAIL in span for span in spans):
                    failures.append((label, name, spans))
        self.assertEqual([], failures)

    def test_what_is_learned_is_hidden_as_sift_and_a_model_write_it(self):
        """The value as the file writes it, with its tabs as four blanks
        (one_line folds a metadata string so) and with its whitespace as
        one blank (a preview joins words so, and so does a model)."""
        failures = []
        for name, gap in GAPS.items():
            value = _value(gap)
            with remember_hidden():
                learn_secrets(f"Hi team,\nthe new vault password: {value}\n")
                for spelled in (value, one_line(value),
                                " ".join(value.split())):
                    said = hide_known(f"the model says {spelled} here")
                    if TAIL in said:
                        failures.append((name, spelled, said))
        self.assertEqual([], failures)

    def test_a_value_whose_first_piece_is_a_word_is_judged_whole(self):
        """Judging the first piece alone let "Sonnen<TAB>schein1" through
        whole: "Sonnen" is a word. Each piece is judged; one that is a
        secret makes the value one."""
        for gap in GAPS.values():
            line = f"Das Kennwort lautet Sonnen{gap}schein1."
            with self.subTest(repr(gap)):
                said = scrub(line)
                self.assertNotIn("schein1", said)
                self.assertNotIn("Sonnen", said)

    def test_a_repetition_is_found_in_every_spelling(self):
        """find_secrets covers what hide_known would hide: the repeated
        value, with its whitespace collapsed, is a span too."""
        text = (f"password: {_value(chr(9))}\n"
                f"note: the old one was {HEAD} {TAIL} as well\n")
        spans = [text[s.start:s.end] for s in find_secrets(text)]
        self.assertIn(f"{HEAD} {TAIL}", spans)


class TheMailTest(unittest.TestCase):
    """The reviewer's case: a mail whose body sets a vault password with a
    tab in it, summarized with and without a model."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def mail(self, name: str, gap: str) -> Path:
        path = self.dir / name
        path.write_text(
            "From: a@example.com\nTo: b@example.com\nSubject: vault\n"
            "MIME-Version: 1.0\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            f"Hi team,\nthe new vault password: {_value(gap)}\nregards\n",
            encoding="utf-8")
        return path

    def outputs(self, summ) -> str:
        return (render_text(summ, color=False) + render_json([summ])
                + render_markdown([summ]))

    def test_no_output_shows_the_tail(self):
        for index, (name, gap) in enumerate(GAPS.items()):
            summ = summarize_path(self.mail(f"m{index}.eml", gap),
                                  Config(no_llm=True, quiet=True, cache=False),
                                  None, None)
            with self.subTest(name):
                self.assertTrue(summ.summary)
                self.assertNotIn(TAIL, self.outputs(summ))

    def test_a_model_quoting_the_value_shows_no_tail(self):

        class Model:
            provider = name = model = "stub"
            endpoint = "memory://stub"

            def __init__(self, said):
                self.said = said

            def available(self):
                return True

            def list_models(self):
                return ["stub"]

            def generate(self, system, user, **kw):
                return LLMResult(
                    text=(f"TYPE: Mail\nSUMMARY: Sets the vault password "
                          f"{self.said}.\nKEY POINTS:\n- password: "
                          f"{self.said}\nENTITIES: {self.said}\nNOTES: none"),
                    model="stub", provider="stub")

        for index, (name, gap) in enumerate(GAPS.items()):
            value = _value(gap)
            for said in (value, " ".join(value.split()), one_line(value)):
                summ = summarize_path(self.mail(f"l{index}.eml", gap),
                                      Config(quiet=True, cache=False),
                                      Model(said), None)
                with self.subTest(name=name, said=said):
                    self.assertNotIn(TAIL, self.outputs(summ))


class WhereAValueStillEndsTest(unittest.TestCase):
    """What must not change: prose separates words with one blank, and a
    sentence may end with two."""

    def test_one_blank_ends_a_value_in_prose(self):
        line = "the password is Kx9vQ2 and the user is anna"
        self.assertIn("and the user is anna", redact_value(line))
        self.assertNotIn("Kx9vQ2", redact_value(line))

    def test_two_blanks_after_a_full_stop_end_a_sentence(self):
        for mark in ".!?":
            line = f"Das Kennwort ist Winterreise2026{mark}  Bitte ändern."
            with self.subTest(mark):
                said = redact_value(line)
                self.assertIn("Bitte ändern.", said)
                self.assertNotIn("Winterreise2026", said)

    def test_a_word_after_the_key_is_judged_as_before(self):
        """The first word decides whether it is a secret; the gap only
        decides how far a secret goes."""
        for line in ("Password:\tsee\tvault", "Password: see  the vault",
                     "The password policy requires  12 characters.",
                     "The token budget is  4096 tokens."):
            with self.subTest(line):
                self.assertEqual(line, redact_value(line))

    def test_a_label_ending_in_a_colon_is_a_word(self):
        """A review: each piece is judged without the colon a
        label ends in -- "Notiz:" is a word, not a credential."""
        for line in ("Passwort:\tsiehe\tNotiz:\tunten",
                     "the password:  see  Notes:  below"):
            with self.subTest(line):
                self.assertEqual(line, redact_value(line))

    def test_a_column_after_a_header_named_password_is_a_column(self):
        header = "user\tpassword\tnote"
        self.assertEqual(header, redact_value(header))

    def test_a_line_break_ends_a_value(self):
        text = f"the vault password: {HEAD}\n{TAIL} is the build id"
        self.assertIn(f"{TAIL} is the build id", redact_value(text))

    def test_a_long_run_of_blanks_ends_a_value(self):
        """Nine blanks and more are a table's alignment, not a value."""
        line = f"password: {HEAD}Q2mZ" + " " * 12 + "Comment"
        self.assertIn("Comment", redact_value(line))

    def test_whitespace_after_the_value_is_not_the_value(self):
        line = f"the vault password: {HEAD}{TAIL}\t\n"
        said = redact_value(line)
        self.assertIn(f"<{len(HEAD + TAIL)} chars — not shown>", said)

    def test_a_shell_argument_still_ends_at_whitespace(self):
        """curl -u user:pass: a shell ends an argument at a tab too."""
        line = "curl -u svc:Kx9vQ2mZ\thttps://api.example.com/v1"
        self.assertIn("https://api.example.com/v1", redact_value(line))


class WhatFollowsAGapTest(unittest.TestCase):
    """A review, still at 0.1.69: a value ran over a gap into what
    followed it -- an arrow comment, a tag, an earlier marker -- and the
    guard for a value someone already hid ("<" anywhere in it) rejected the
    whole, so the password in front of the arrow was published, and a
    second keyword swallowed by the run was never read."""

    PASSWORD = "Kx9vQ2mZ7pL4"

    def assertHidden(self, line):
        said = redact_value(line)
        self.assertNotIn(self.PASSWORD, said, said)
        found = [line[s.start:s.end] for s in find_secrets(line)
                 if s.reason != "bidi"]
        self.assertTrue(any(self.PASSWORD in span for span in found),
                        found)
        return said

    def test_an_arrow_comment_after_a_gap(self):
        for line in (f"password: {self.PASSWORD}  <-- change this",
                     f"* the password is {self.PASSWORD}  <-- rotated",
                     f"Das Kennwort lautet {self.PASSWORD}\t<-- neu"):
            with self.subTest(line):
                said = self.assertHidden(line)
                self.assertIn("<--", said)

    def test_a_tag_after_a_gap(self):
        said = self.assertHidden(f"password: {self.PASSWORD}\t<b>")
        self.assertTrue(said.endswith("\t<b>"), said)

    def test_a_second_keyword_after_a_tag(self):
        said = self.assertHidden(
            f"Password:\tsee\t<vault>\tKennwort:\t{self.PASSWORD}")
        self.assertIn("Password:\tsee\t<vault>\tKennwort:", said)

    def test_an_earlier_marker_after_a_gap(self):
        """A marker another rule left behind is not the value, and the value
        in front of it is still judged."""
        line = f"password: {self.PASSWORD}\t<12 chars — not shown> later"
        said = redact_value(line)
        self.assertNotIn(self.PASSWORD, said)
        self.assertIn("\t<12 chars — not shown> later", said)

    # -- sad ---------------------------------------------------------------
    def test_a_value_that_is_a_marker_is_left_alone(self):
        """The guard stays where it belongs: the value is the marker."""
        for line in ('password "<16 chars — not shown>"',
                     "api_key=<20 chars — not shown>",
                     "password: <12 chars — not shown>  <-- old"):
            with self.subTest(line):
                self.assertEqual(line, redact_value(line))

    def test_a_placeholder_and_a_tag_are_no_secret(self):
        for line in ("Password: <your-password-here>",
                     "Password:\tsee\t<vault>", "the password is\t<b>"):
            with self.subTest(line):
                self.assertEqual(line, redact_value(line))

    # -- edge --------------------------------------------------------------
    def test_a_value_holding_a_less_than_sign(self):
        """A password may hold "<": only a marker is someone else's."""
        said = redact_value("password: Kx9v<Q2mZ7pL4")
        self.assertNotIn("Q2mZ7pL4", said)


class TableRowTest(unittest.TestCase):
    """A review, still at 0.1.69: a tab or a run of blanks let
    a value run across a ticket's columns, and one date or number among
    them made all of them a secret -- "INC0001<TAB>Password <28 chars --
    not shown>". A value joined to its key by nothing but whitespace is one
    piece; one joined by ":", "=" or a copula runs on over a gap, as the
    review cases need."""

    ROWS = ("INC0001\tPassword reset\t2024-05-01\tclosed\tanna",
            "user\tpassword\tchanged\t2024-05-01",
            "secret\tstring\tSHA-256\trequired",
            "Token count  4711  tokens",
            "LAP-0042\tThinkPad T14\tPIN pad\t2023-11-30\tin use")

    def test_a_row_of_a_table_is_kept(self):
        for row in self.ROWS:
            with self.subTest(row):
                self.assertEqual(row, redact_value(row))
                self.assertEqual([], [s for s in find_secrets(row, "t.tsv")
                                      if s.reason != "bidi"])

    def test_a_credential_in_a_column_is_still_hidden(self):
        """Joined by nothing but a tab, the value is one piece -- which is
        what a column holds."""
        for row in ("db1\tpassword\tKx9vQ2mZ7pL4",
                    "db1\tpassword Kx9vQ2mZ7pL4\t2024-05-01"):
            with self.subTest(row):
                said = redact_value(row)
                self.assertNotIn("Kx9vQ2mZ7pL4", said)
                self.assertTrue(said.startswith("db1\tpassword"), said)

    def test_a_joined_value_still_runs_over_a_gap(self):
        for line in ("Password:\tSonnen\tschein1", "the password is Kx9v  Q2mZ7"):
            with self.subTest(line):
                said = redact_value(line)
                self.assertNotIn("schein1", said)
                self.assertNotIn("Q2mZ7", said)

    def test_the_rows_are_in_the_innocent_corpus(self):
        import corpus_secrets
        self.assertTrue(set(self.ROWS) <= set(corpus_secrets.SENTENCES_KEPT))


class ACellAfterTheValueTest(unittest.TestCase):
    """A review (office), 0.1.78: a sheet row is rendered with a tab between its
    cells, and "password: Kx9vQ2mZ7pL4<TAB>Item" was found as the span
    "Kx9vQ2mZ7pL4<TAB>Item" -- by the sentence rule, whose value runs on over
    a gap, and by the key reader, which takes the rest of the line. The
    bare password was then no known value, and a model's sentence that named
    it with no key beside it was published. A value runs on over a tab into
    what is itself a piece of a secret -- a review's tails --, not into a word:
    that is the next cell."""

    SECRET = "Kx9vQ2mZ7pL4"

    def spans(self, text: str) -> "list[str]":
        return [text[s.start:s.end] for s in find_secrets(text)
                if s.reason != "bidi"]

    def test_the_bare_secret_is_the_span(self):
        for line in (f"password: {self.SECRET}\tItem\n",
                     f"password: {self.SECRET}\tItem\tQty\t4\n",
                     f"PASSWORD={self.SECRET}\tItem\n",
                     f"Das Kennwort lautet {self.SECRET}\tArtikel\n",
                     f"the vault password: {self.SECRET} \tItem\n"):
            with self.subTest(line):
                self.assertEqual([self.SECRET], self.spans(line))
                said = redact_value(line)
                self.assertNotIn(self.SECRET, said)
                self.assertIn("\tItem" if "Item" in line else "\tArtikel",
                              said)

    def test_a_model_naming_the_bare_secret_is_masked(self):
        """What the finding cost: the value is known as it is written."""
        from sift.extract.base import (find_secrets_and_pieces,
                                       remember_values)
        text = f"user\tanna\npassword: {self.SECRET}\tItem\n"
        with remember_hidden():
            spans, pieces = find_secrets_and_pieces(text)
            remember_values(text, spans, pieces)
            said = hide_known(f"It uses {self.SECRET} for the item.")
        self.assertNotIn(self.SECRET, said)

    def test_a_tail_that_is_a_secret_s_piece_still_runs_on(self):
        """A review's shapes: an opaque piece after the tab is the value's."""
        for line, value in ((f"password: {HEAD}\t{TAIL}\tItem\n",
                             f"{HEAD}\t{TAIL}"),
                            ("Password:\tSonnen\tschein1\tItem\n",
                             "Sonnen\tschein1"),
                            (f"the new vault password: {HEAD}\t{TAIL}\n",
                             f"{HEAD}\t{TAIL}")):
            with self.subTest(line):
                self.assertEqual([value], self.spans(line))

    def test_a_plain_password_is_the_first_cell(self):
        """Under a key that names a secret a plain word is a password: the
        first cell is the value whatever it is, the next one is not."""
        line = "password: Sonnenschein\tItem\n"
        self.assertEqual(["Sonnenschein"], self.spans(line))
        self.assertIn("\tItem", redact_value(line))

    def test_a_value_is_judged_whole_and_cut_after(self):
        """The cut decides how far a secret goes, not whether there is one:
        judged by its first cell, openssl.cnf's "$dir/certs" under the hint
        "certs" read as a credential (/etc/ssl/openssl.cnf, 6 lines); judged
        whole, with its comment, it is none, as it was."""
        for line in ("certs\t\t= $dir/certs\t\t# Where the issued certs are "
                     "kept\n",
                     "signer_cert\t= $dir/tsacert.pem \t# The TSA signing "
                     "certificate\n"):
            with self.subTest(line):
                self.assertEqual([], self.spans(line))

    def test_a_row_of_cells_after_a_key_that_names_nothing(self):
        """Sad path: nothing changes where nothing was a secret -- a key
        that names none, and a word the sentence rule reads as a word."""
        line = "user:\tanna\tItem\n"
        self.assertEqual([], self.spans(line))
        self.assertEqual(line, redact_value(line))
        self.assertEqual("Password:\tsee\tvault",
                         redact_value("Password:\tsee\tvault"))


if __name__ == "__main__":
    unittest.main()
