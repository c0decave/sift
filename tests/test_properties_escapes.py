"""A .properties file as Java reads it: escapes decoded, spans on the raw text.

java.util.Properties.load reads a key and a value through escapes: \\uXXXX
for any UTF-16 unit, \\t \\n \\r \\f, and a backslash before any other
character for that character -- \\= and \\: inside a key, "\\ " for a blank
in it, \\\\ for a backslash. A key ends at the first unescaped "=", ":" or
blank, and a line ending in an odd number of backslashes goes on on the next,
whose leading blanks are dropped.

sift read a .properties line through the assignment pattern of every other
file (_KEYED_LINE), whose key is [\\w.\\-\\[\\]]: a key holding any escape
matched no line, and its value was no setting at all -- not masked in
A host application's viewer, not remembered, repeated by a model --, and a value
written with escapes was remembered in its raw form only, so a model that
decoded it ("p\\u00e4ss" is "p\u00e4ss") published it.

The span stays on the raw text: a host application masks by position. The decoded
value is one more form of it (_value_forms), beside the raw one, as a DSN's
percent-decoded password is (test_value_pieces.DecodedFormTest).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_value_pieces as pieces  # noqa: E402 -- the helpers sit beside this file
from _timing import cpu_ratio, linear_bound  # noqa: E402

from sift.extract import find_secrets
from sift.extract.base import (SECRET_SCAN_LIMIT, _SCAN_OVERHANG,
                               _properties_unescaped, _setting_keys,
                               _value_forms, in_place_only)

NAME = "app.properties"
SECRET = "Kx9vQ2mZ7pL4"
#: A value with two umlauts, and what Java reads it as.
UMLAUTS = "p\\u00e4ssw\\u00f6rd2026Kx"
UMLAUTS_READ = "p\u00e4ssw\u00f6rd2026Kx"
#: SECRET with every character written as \\uXXXX.
ESCAPED_SECRET = "".join(f"\\u{ord(char):04x}" for char in SECRET)


def found(text: str, name: str = NAME) -> "list[str]":
    """The raw texts find_secrets masks in ``text`` under ``name``."""
    return [text[s.start:s.end] for s in find_secrets(text, name)
            if s.reason != "bidi"]


def key_of(text: str, value: str) -> str:
    """The key _setting_keys reads for the settings value ``value``."""
    start = text.index(value)
    return _setting_keys(text, {start})[start]


class _Files(pieces._Files):
    pass


class EscapedValueTest(_Files):
    """A value written with escapes is found where it stands and
    remembered as Java reads it too."""

    #: (label, body, the raw value, the value as Java reads it)
    VALUES = (
        ("umlauts", f"db.password={UMLAUTS}\n", UMLAUTS, UMLAUTS_READ),
        ("every character escaped", f"secret={ESCAPED_SECRET}\n",
         ESCAPED_SECRET, SECRET),
        ("a tab", "db.password=Kx9v\\tQ2mZ7pL4\n", "Kx9v\\tQ2mZ7pL4",
         "Kx9v\tQ2mZ7pL4"),
        ("a backslash", "db.password=Kx9v\\\\Q2mZ7pL4\n", "Kx9v\\\\Q2mZ7pL4",
         "Kx9v\\Q2mZ7pL4"),
        ("an escaped hash and equals", "db.password=Kx9v\\#Q2\\=mZ7pL4\n",
         "Kx9v\\#Q2\\=mZ7pL4", "Kx9v#Q2=mZ7pL4"),
        ("an escaped leading blank", "db.password=\\ Kx9vQ2mZ7pL4\n",
         "\\ Kx9vQ2mZ7pL4", " Kx9vQ2mZ7pL4"),
    )

    def test_the_span_is_the_raw_value(self):
        for label, body, raw, _read in self.VALUES:
            with self.subTest(label):
                self.assertEqual([raw], found(body))

    def test_the_value_is_remembered_as_java_reads_it(self):
        for label, body, raw, read in self.VALUES:
            with self.subTest(label):
                seen = pieces.remembered(body, NAME)
                self.assertIn(raw, seen)
                self.assertIn(read.strip(), seen)

    def test_a_model_that_decodes_the_value_is_not_repeated(self):
        failures = []
        for label, body, _raw, read in self.VALUES:
            shown = self.shown(NAME, body, read.strip())
            if shown:
                failures.append((label, shown))
        self.assertEqual([], failures)


class EscapedKeyTest(_Files):
    """A key written with escapes is a key: its value is a setting, and it
    is judged by the name Java reads."""

    #: (label, body, the raw value, the key as Java reads it)
    KEYS = (
        ("every character escaped",
         "\\u0070\\u0061\\u0073\\u0073\\u0077\\u006f\\u0072\\u0064=Sommer\n",
         "Sommer", "password"),
        ("one character escaped", "db.p\\u0061ssword=Sommer2026\nlog=info\n",
         "Sommer2026", "db.password"),
        ("an escaped equals sign", f"my\\=pass={SECRET}\n", SECRET,
         "my=pass"),
        ("an escaped colon", f"db\\:password:{SECRET}\n", SECRET,
         "db:password"),
        ("an escaped blank", f"db\\ password = {SECRET}\n", SECRET,
         "db password"),
        ("a blank as the separator", f"db.password {SECRET}\n", SECRET,
         "db.password"),
        ("blanks, then a colon", f"db.password   :   {SECRET}\n", SECRET,
         "db.password"),
    )

    def test_the_value_is_a_setting(self):
        for label, body, value, _key in self.KEYS:
            with self.subTest(label):
                self.assertIn(value, found(body))

    def test_the_key_is_read_as_java_reads_it(self):
        for label, body, value, key in self.KEYS:
            with self.subTest(label):
                self.assertEqual(key, key_of(body, value))

    def test_a_secret_name_keeps_a_plain_word_secret(self):
        """in_place_only judges the value by the key Java reads: "Sommer"
        under "password" is a password, not a plain word."""
        for label, body, value, _key in self.KEYS[:2]:
            with self.subTest(label):
                spans = find_secrets(body, NAME)
                start = body.index(value)
                where = (start, start + len(value))
                self.assertIn(where, [(s.start, s.end) for s in spans])
                self.assertNotIn(where, in_place_only(body, spans))

    def test_no_output_repeats_the_value(self):
        failures = []
        for label, body, value, _key in self.KEYS:
            shown = self.shown(NAME, body, value)
            if shown:
                failures.append((label, shown))
        self.assertEqual([], failures)


class ContinuationTest(unittest.TestCase):
    """A logical line over several natural lines, LF and CRLF."""

    def test_a_key_continued_on_the_next_line(self):
        body = f"db.pass\\\n    word={SECRET}\nport=5432\n"
        self.assertEqual([SECRET, "5432"], found(body))
        self.assertEqual("db.password", key_of(body, SECRET))

    def test_a_value_continued_is_read_joined(self):
        body = "db.password=Kx9v\\\n    Q2mZ\\\n\t7pL4\nport=5432\n"
        self.assertEqual(["Kx9v\\\n    Q2mZ\\\n\t7pL4", "5432"], found(body))
        self.assertIn(SECRET, pieces.remembered(body, NAME))

    def test_crlf(self):
        body = (f"db.p\\u0061ssword=Wq8r\\\r\n  Tz5Nb3\r\n"
                f"api\\u005fkey={ESCAPED_SECRET}\r\nport=5432\r\n")
        self.assertEqual(["Wq8r\\\r\n  Tz5Nb3", ESCAPED_SECRET, "5432"],
                         found(body))
        self.assertEqual("db.password", key_of(body, "Wq8r\\"))
        self.assertEqual("api_key", key_of(body, ESCAPED_SECRET))
        seen = pieces.remembered(body, NAME)
        self.assertIn(SECRET, seen)
        self.assertIn("Wq8rTz5Nb3", seen)

    def test_an_escaped_backslash_ends_the_line(self):
        body = "path=C:\\\\dir\\\\\nport=5432\n"
        self.assertEqual(["C:\\\\dir\\\\", "5432"], found(body))

    def test_the_last_line_continued_into_nothing(self):
        """Java drops a backslash that continues the last line."""
        self.assertEqual(f"a{SECRET}", _properties_unescaped(f"a{SECRET}\\"))
        self.assertEqual([f"{SECRET}\\"], found(f"k={SECRET}\\"))


class DecodingTest(unittest.TestCase):
    """_properties_unescaped, the one decoder, against Java's
    Properties.load: happy, sad and edge."""

    def test_every_escape_java_knows(self):
        self.assertEqual("a\tb\nc\rd\fe\\f=g:h#i!j k",
                         _properties_unescaped(
                             "a\\tb\\nc\\rd\\fe\\\\f\\=g\\:h\\#i\\!j\\ k"))

    def test_any_other_escaped_character_is_itself(self):
        """Java: "\\b" is "b", not a backspace; "\\q" is "q"."""
        self.assertEqual("bqz", _properties_unescaped("\\b\\q\\z"))

    def test_a_surrogate_pair_is_one_character(self):
        self.assertEqual("x\U0001F600y",
                         _properties_unescaped("x\\ud83d\\ude00y"))
        self.assertEqual("x\U0001F600y",
                         _properties_unescaped("x\\uD83D\\uDE00y"))

    def test_a_lone_surrogate_stays_as_written(self):
        """Java keeps a lone UTF-16 unit, which no UTF-8 text can hold: a
        form holding one would break every writer that encodes it. It
        stays the escape it was written as."""
        for raw in ("x\\ud83dy", "x\\ude00y", "x\\ud83d\\u0041y",
                    "x\\ude00\\ud83dy", "x\\ud83d"):
            with self.subTest(raw):
                read = _properties_unescaped(raw)
                read.encode("utf-8")
                self.assertIn("\\u", read)
        self.assertEqual("x\\ud83dAy", _properties_unescaped("x\\ud83d\\u0041y"))

    def test_a_malformed_unicode_escape_is_literal(self):
        """Java throws "Malformed \\uxxxx encoding."; sift reads on, and
        the escape stands as written."""
        for raw in ("\\u12", "a\\u12zz", "a\\uZZZZ", "a\\u", "a\\u00e"):
            with self.subTest(raw):
                self.assertEqual(raw, _properties_unescaped(raw))

    def test_nothing_to_decode(self):
        self.assertEqual("", _properties_unescaped(""))
        self.assertEqual(SECRET, _properties_unescaped(SECRET))

    def test_a_continuation_inside_the_text(self):
        self.assertEqual("abcd", _properties_unescaped("ab\\\n   cd"))
        self.assertEqual("abcd", _properties_unescaped("ab\\\r\n\t cd"))
        self.assertEqual("abcd", _properties_unescaped("ab\\\r cd"))


class FormsTest(unittest.TestCase):
    """The raw form and the decoded one, both; nothing new where nothing
    decodes."""

    def test_both_forms(self):
        forms = _value_forms(UMLAUTS)
        self.assertIn(UMLAUTS, forms)
        self.assertIn(UMLAUTS_READ, forms)

    def test_a_surrogate_pair_is_a_form_decoded(self):
        forms = _value_forms("Kx9v\\ud83d\\ude00Q2mZ")
        self.assertIn("Kx9v\U0001F600Q2mZ", forms)

    def test_no_form_holds_a_lone_surrogate(self):
        for raw in ("Kx9v\\ud83dQ2mZ7pL4", "Kx9v\\u12Q2mZ7pL4"):
            with self.subTest(raw):
                forms = _value_forms(raw)
                self.assertIn(raw, forms)
                for form in forms:
                    form.encode("utf-8")

    def test_a_value_with_no_backslash_gains_no_form(self):
        self.assertEqual(_value_forms(SECRET), {SECRET})


class MalformedFileTest(_Files):
    """Nothing in a .properties makes the reader throw; a malformed escape
    is read as Java would not, literally."""

    def test_malformed_and_lone_escapes(self):
        body = ("a.password=Kx9v\\u12Q2mZ7pL4\n"
                "b.password=Kx9v\\ud83dQ2mZ7pL4\n"
                "c\\u12=Wq8rTz5Nb3\n"
                "\\\n"
                "\\u\n"
                "d=\\\n")
        self.assertEqual(["Kx9v\\u12Q2mZ7pL4", "Kx9v\\ud83dQ2mZ7pL4",
                          "Wq8rTz5Nb3"], found(body))
        self.assertEqual("c\\u12", key_of(body, "Wq8rTz5Nb3"))
        self.assertEqual({}, self.shown(NAME, body, "Kx9v\\u12Q2mZ7pL4",
                                        "Wq8rTz5Nb3"))

    def test_a_key_with_no_value(self):
        """Java reads "key" and "key=" as the empty value: nothing to
        mask."""
        self.assertEqual([], found("lonely\nempty=\nblank =   \n"))


class CommentTest(unittest.TestCase):
    """A line whose first character after blanks is "#" or "!" is a
    comment, escapes and all; it goes on on no next line."""

    def test_no_setting_in_a_comment(self):
        for line in (f"# p\\u0061ssword={SECRET}", f"! p\\u0061ssword={SECRET}",
                     f"   #\\u0070assword {SECRET}", f"\t!key\\u003d{SECRET}"):
            with self.subTest(line):
                self.assertEqual([], found(line + "\n"))

    def test_a_comment_ending_in_a_backslash_does_not_continue(self):
        body = f"# a comment \\\ndb.p\\u0061ssword={SECRET}\n"
        self.assertEqual([SECRET], found(body))
        self.assertEqual("db.password", key_of(body, SECRET))


class OnlyPropertiesTest(unittest.TestCase):
    """The Java reading belongs to a .properties: a .env keeps its own."""

    def test_a_dotenv_reads_no_java_escape_in_a_key(self):
        self.assertEqual([], found(f"A\\u0041={SECRET}\n", ".env"))

    def test_a_properties_key_ends_at_its_first_blank(self):
        """Where a .env has no setting, Java reads key "A" and the value
        "B=x=y"."""
        self.assertEqual([], found("A B=x=y\n", ".env"))
        self.assertEqual(["B=x=y"], found("A B=x=y\n"))


class KeyListTest(unittest.TestCase):
    """The keys sift lists of a .properties are the keys Java reads. They
    came from configparser, for which a key ends at "=" or ":" only: the
    line "db.url Kx9vQ2mZ7pL4=x" was the key "db.url Kx9vQ2mZ7pL4", and the
    summary named it -- the value with it."""

    def summary(self, body: str):
        from sift.config import Config
        from sift.summarize import summarize_path
        import tempfile
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        path = folder / NAME
        path.write_bytes(body.encode("utf-8"))
        return summarize_path(path, Config(no_llm=True, quiet=True,
                                           cache=False), None, None)

    def test_a_blank_separated_value_is_no_part_of_a_key(self):
        summ = self.summary(f"db.url {SECRET}=x\nport=5432\n")
        said = summ.summary + " ".join(summ.key_points)
        self.assertNotIn(SECRET, said)
        self.assertIn("db.url", said)
        self.assertIn("port", said)

    def test_the_keys_are_listed_decoded(self):
        summ = self.summary("db.p\\u0061ssword=Sommer2026\n"
                            "pool\\ size : 20\nlog\\=level=info\n")
        self.assertIn("db.password, pool size, log=level", summ.summary)
        self.assertNotIn("Sommer2026", summ.summary)

    def test_a_key_holding_a_control_character_is_listed_escaped(self):
        """A key may decode to anything, a line break or a bidi control
        among it; the summary is one line, and a reordering control in it
        would turn what a reader sees."""
        summ = self.summary("a\\u000ab=1\nc\\u202ed=2\n")
        self.assertIn("a\\u000ab, c\\u202ed", summ.summary)
        self.assertNotIn("\u202e", summ.summary)

    def test_each_key_once_and_none_empty(self):
        summ = self.summary("a=1\na=2\n=3\nb 4\n")
        self.assertIn(": a, b.", summ.summary)


class ExportTest(_Files):
    """"export KEY=value" under a .properties name is a .env's line, as sift
    reads every settings file (test_settings_values' "export" shape): the
    value is KEY's. To Java it is the key "export" and the value
    "KEY=value", and the model's "Kx9vQ2mZ7pL4" was published."""

    def test_the_value_after_export_is_the_keys(self):
        body = f"export DEBUG=true\nexport DB_PASSWORD = {SECRET}\n"
        self.assertEqual(["true", SECRET], found(body))
        self.assertEqual("DB_PASSWORD", key_of(body, SECRET))
        self.assertEqual({}, self.shown(NAME, body, SECRET))

    def test_a_flag_after_export_stays_harmless(self):
        body = "export DEBUG=true\n"
        spans = find_secrets(body, NAME)
        self.assertEqual({(13, 17)}, in_place_only(body, spans))

    def test_export_with_no_assignment_after_it_is_javas(self):
        """Sad path: no "=" or ":" after the next word -- the key is
        "export", as Java reads it."""
        for body, value in (("export all of it\n", "all of it"),
                            ("export\tnothing\n", "nothing"),
                            ("export #x=1\n", "#x=1")):
            with self.subTest(body):
                self.assertEqual([value], found(body))
                self.assertEqual("export", key_of(body, value))

    def test_an_exported_key_with_escapes(self):
        body = f"export db.p\\u0061ssword={SECRET}\n"
        self.assertEqual([SECRET], found(body))
        self.assertEqual("db.password", key_of(body, SECRET))


class EqualsInAValueTest(_Files):
    """A value holding "=": "db.url Kx9vQ2mZ7pL4=x" is the key db.url and the
    value "Kx9vQ2mZ7pL4=x" to Java -- a key ends at its first "=", ":" or
    blank, and any later "=" is the value's. A model reads the line as the
    "key=value" every other settings line is, and hands back
    "Kx9vQ2mZ7pL4": remembered whole only, the value was published. Each
    side of an "=" in a value is a form of it where it holds a run no word
    holds (_opaque_run), under any key: a .properties value is a secret
    under every key, and a word on either side is not."""

    #: (label, body, what a model hands back)
    LEAKED = (
        ("a blank-separated line", f"db.url {SECRET}=x\n", SECRET),
        ("a secret key", f"db.password {SECRET}=x\n", SECRET),
        ("the tail", f"db.url x={SECRET}\n", SECRET),
        ("a value opening with =", f"db.url =={SECRET}\n", SECRET),
        ("an escaped equals", f"db.url {SECRET}\\=x\n", SECRET),
        ("base64 without its padding", "api.key=S2x5OXZRMm1aN3BMNA==\n",
         "S2x5OXZRMm1aN3BMNA"),
        ("a .env", f"SESSION={SECRET}=x\n", SECRET),
    )

    def test_the_side_a_model_names_is_not_repeated(self):
        failures = []
        for label, body, secret in self.LEAKED:
            name = ".env" if label == "a .env" else NAME
            shown = self.shown(name, body, secret)
            if shown:
                failures.append((label, shown))
        self.assertEqual([], failures)

    def test_the_side_is_remembered(self):
        for label, body, secret in self.LEAKED:
            name = ".env" if label == "a .env" else NAME
            with self.subTest(label):
                self.assertIn(secret, pieces.remembered(body, name))

    def test_the_span_stays_javas_value(self):
        self.assertEqual([f"{SECRET}=x"], found(f"db.url {SECRET}=x\n"))
        self.assertEqual("db.url", key_of(f"db.url {SECRET}=x\n",
                                          f"{SECRET}=x"))

    def test_a_word_on_either_side_is_no_form(self):
        """Sad path: words and numbers stay what they are -- hidden in a
        summary wherever they stand, "Hello", "true" and "2024" would take
        its words out."""
        for body, words in (("app.title Hello=World\n", {"Hello", "World"}),
                            ("feature.flags=enabled=true\n",
                             {"enabled", "true"}),
                            ("db.password=Sommer=2024\n", {"Sommer", "2024"}),
                            ("log.level x=debug\n", {"debug"})):
            with self.subTest(body):
                self.assertEqual(set(), words & pieces.remembered(body, NAME))

    def test_no_equals_gains_no_form(self):
        """Edge: a value without "=" is remembered as before; a value of
        "=" alone, or "==", gains no empty or one-character form."""
        self.assertEqual({SECRET}, pieces.remembered(f"db.url {SECRET}\n",
                                                     NAME))
        for body in ("db.url ==\n", "db.url = =\n", "db.url a==\n"):
            with self.subTest(body):
                self.assertTrue(all(len(form) >= 4 for form in
                                    pieces.remembered(body, NAME)))
        self.assertEqual({"Kx9v=Q2mZ"}, _value_forms("Kx9v=Q2mZ") - {""})


class BoundTest(unittest.TestCase):
    """The reader is bounded as find_secrets is: spans start in the first
    SECRET_SCAN_LIMIT characters, and the work is linear."""

    def test_spans_start_before_the_limit(self):
        line = f"k\\u0065y{{}}=\\u0041{SECRET}\n"
        body = "".join(line.format(n) for n in range(
            (SECRET_SCAN_LIMIT + 2 * _SCAN_OVERHANG) // len(line) + 1))
        spans = find_secrets(body, NAME)
        self.assertTrue(spans)
        self.assertLess(max(s.start for s in spans), SECRET_SCAN_LIMIT)

    def test_escaped_lines_cost_linear_work(self):
        line = f"a\\u0062\\ c\\=d\\:e = \\u0041{SECRET}\\\n  tail\n"
        short, long_ = line * 1_000, line * 8_000
        ratio = cpu_ratio(lambda: find_secrets(long_, NAME),
                          lambda: find_secrets(short, NAME))
        self.assertLess(ratio, linear_bound(8))

    def test_a_key_that_never_ends_costs_linear_work(self):
        """One line of escapes and no separator, and a key continued over
        every line: each read once."""
        for unit, count in (("\\u0041", 8_000), ("a\\\n", 16_000)):
            with self.subTest(unit):
                short, long_ = unit * count, unit * 8 * count
                ratio = cpu_ratio(lambda: find_secrets(long_, NAME),
                                  lambda: find_secrets(short, NAME))
                self.assertLess(ratio, linear_bound(8))


if __name__ == "__main__":
    unittest.main()
