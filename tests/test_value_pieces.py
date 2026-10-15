"""The pieces of a value, in what sift says about the file.

remember_values records every value find_secrets found, in the forms
_value_forms says a value is repeated in -- and a model that read the file
hands back the piece it means, not the value as the file wrote it:

- a secret inside a value: CONFIG='{"password": "Mv4Tq9Wz2Lk7Rx3Nb"}' over
  three lines of a .env. find_secrets found the member, _merged folded it
  into the setting around it, and only the setting and its lines were
  remembered -- "uses Mv4Tq9Wz2Lk7Rx3Nb as its credential" was published;
- a URL's password anywhere in a line of a value, not only in a value
  that begins with the URL: jdbc:postgresql://app:…@db, "url = redis://…";
- a shell default: ${POSTGRES_PASSWORD:-Gk4Wq8Zr2Mx7Tb} was remembered
  whole, and the default word, which is the password, was not;
- a decoded form: a DSN's password percent-encoded (Kx9v%40Q2mZ7pL4 is
  Kx9v@Q2mZ7pL4), an Authorization header's base64 and the password in it.

What must not change: an innocent word, a harmless ${VAR}, the default of a
setting that is no secret.
"""
import base64
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.config import Config
from sift.extract import find_secrets
from sift.extract.base import (find_secrets_and_pieces, hide_known,
                               in_place_only, remember_hidden,
                               remember_values)
from sift.llm.base import LLMResult
from sift.report import render_json, render_markdown, render_text
from sift.summarize import summarize_path

#: The password of a review's reproduction, and the other secrets here.
MEMBER = "Mv4Tq9Wz2Lk7Rx3Nb"
URL_PW = "Hn5Rt8Wq3Zm6Kd"
DEFAULT = "Gk4Wq8Zr2Mx7Tb"
BASIC_PW = "Pq7Zx2Lm9Wk4Tr"
BASIC = base64.b64encode(f"svc_user:{BASIC_PW}".encode()).decode()


def _remember(body: str, name: str) -> None:
    """What summarize remembers of ``body`` under ``name``, into the live
    remember_hidden: find_secrets' spans and their pieces, the harmless
    settings left out (summarize.py's own order)."""
    found, pieces = find_secrets_and_pieces(body, name)
    spans = [s for s in found if s.reason not in ("bidi", "repeated")]
    in_place = in_place_only(body, spans)
    remember_values(body, [s for s in spans
                           if (s.start, s.end) not in in_place], pieces)


def remembered(body: str, name: str = ".env") -> set:
    """Every form summarize remembers of ``body`` under ``name``."""
    with remember_hidden() as seen:
        _remember(body, name)
        return set(seen)


def said_about(body: str, name: str, text: str) -> str:
    """``text`` as hide_known says it once ``body`` taught it."""
    with remember_hidden():
        _remember(body, name)
        return hide_known(text)


class _Model:
    """A model that hands each of ``values`` back in every section."""

    provider = name = model = "stub"
    endpoint = "memory://stub"

    def __init__(self, *values: str):
        self.values = values

    def available(self):
        return True

    def list_models(self):
        return ["stub"]

    def generate(self, system, user, **kw):
        said = ", ".join(self.values)
        return LLMResult(
            text=(f"TYPE: Environment settings\nSUMMARY: The service uses "
                  f"{self.values[0]} as its credential ({said}).\n"
                  "KEY POINTS:\n" + "".join(f"- {v}\n" for v in self.values)
                  + f"ENTITIES: {said}\nTAGS: settings\nNOTES: holds {said}"),
            model="stub", provider="stub")


class _Files(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.count = 0

    def shown(self, name: str, body: str, *secrets: str) -> dict:
        """{output: [each of ``secrets`` it repeats]} of the summary of
        ``body`` under ``name`` by a model that says every secret back."""
        self.count += 1
        folder = self.dir / str(self.count)
        folder.mkdir()
        path = folder / name
        path.write_bytes(body.encode("utf-8"))
        summ = summarize_path(path, Config(quiet=True, cache=False, tags=5),
                              _Model(*secrets), None)
        outputs = {"terminal": render_text(summ, color=False),
                   "json": render_json([summ]),
                   "markdown": render_markdown([summ])}
        found = {out: [s for s in secrets if s in text]
                 for out, text in outputs.items()}
        return {out: hits for out, hits in found.items() if hits}


class InnerSecretTest(_Files):
    """Every secret find_secrets finds inside a value is remembered on its
    own, and every URL password in one of its lines."""

    #: (label, file name, body, secret)
    INSIDE = (
        ("the review's multi-line JSON", ".env",
         f"DEBUG=true\nCONFIG='{{\n  \"password\": \"{MEMBER}\"\n}}'\n", MEMBER),
        ("JSON on one line", ".env", f'CONFIG={{"password": "{MEMBER}"}}\n',
         MEMBER),
        ("JSON in a .properties", "app.properties",
         f'db.config={{"password": "{MEMBER}"}}\n', MEMBER),
        ("a keyed line inside a value", ".env",
         "CONFIG='[db]\npassword = sonnenschein\n'\n", "sonnenschein"),
        ("a Sentry DSN inside a value", ".env",
         "CONFIG='{\n  \"dsn\": \"https://abc123def456@o1.ingest.sentry.io/1\"\n"
         "}'\n", "abc123def456"),
        ("a password in the comment after a value", ".env",
         "DEBUG=true # password: hunter2x\n", "hunter2x"),
        ("a URL inside a keyed value of another file", "notes.yaml",
         f"password: hunter2 https://u:{URL_PW}@h/x\n", URL_PW),
    )
    #: A URL's password where no rule finds the URL on its own.
    URLS = (
        ("jdbc", f"JDBC_URL=jdbc:postgresql://app:{URL_PW}@db/x\n"),
        ("after an option", f"OPTS=--url postgres://app:{URL_PW}@db/x -v\n"),
        ("mid-line in a value over lines",
         f"CONFIG='[db]\nurl = redis://:{URL_PW}@cache\n'\n"),
        ("first line of a value over lines",
         f'DATABASE_URL="postgres://app:{URL_PW}@db:5432/app\n"\n'),
    )

    def test_each_secret_inside_a_value_is_remembered(self):
        failures = [(label, sorted(seen)) for label, name, body, secret
                    in self.INSIDE
                    for seen in [remembered(body, name)] if secret not in seen]
        self.assertEqual([], failures)

    def test_each_url_password_in_a_line_is_remembered(self):
        failures = [(label, sorted(seen)) for label, body in self.URLS
                    for seen in [remembered(body)] if URL_PW not in seen]
        self.assertEqual([], failures)

    def test_the_reported_file_repeats_no_password(self):
        _label, name, body, secret = self.INSIDE[0]
        self.assertEqual({}, self.shown(name, body, secret))

    def test_the_pieces_are_what_the_spans_fold(self):
        body = (f'CONFIG={{"password": "{MEMBER}"}}\n'
                f"URL=jdbc:postgresql://app:{URL_PW}@db/x\n")
        found, pieces = find_secrets_and_pieces(body, ".env")
        self.assertEqual(found, find_secrets(body, ".env"))
        self.assertEqual([MEMBER, f"postgresql://app:{URL_PW}@"],
                         [body[p.start:p.end] for p in pieces])
        self.assertTrue(all(any(s.start <= p.start and p.end <= s.end
                                for s in found) for p in pieces))

    def test_a_url_password_is_not_repeated(self):
        self.assertEqual({}, self.shown(".env", self.URLS[0][1], URL_PW))

    # -- sad ---------------------------------------------------------------
    def test_innocent_lines_inside_a_value_are_not_remembered(self):
        """A value over lines whose lines are no secret: its words stay
        words in what sift says (the value itself is remembered whole)."""
        body = "CONFIG='[db]\nhost = localhost\nlevel = info\nport = 5432\n'\n"
        seen = remembered(body)
        self.assertFalse({"localhost", "info", "5432", "host"} & seen, seen)
        said = said_about(body, ".env", "The db runs on localhost at info level.")
        self.assertEqual("The db runs on localhost at info level.", said)

    def test_a_url_with_no_user_information_gives_no_piece(self):
        seen = remembered("OPTS=--url https://docs.example.com/guide -v\n")
        self.assertEqual({"--url https://docs.example.com/guide -v"}, seen)

    # -- edge --------------------------------------------------------------
    def test_a_value_over_crlf_lines(self):
        body = f"CONFIG='{{\r\n  \"password\": \"{MEMBER}\"\r\n}}'\r\nDEBUG=true\r\n"
        self.assertIn(MEMBER, remembered(body))
        self.assertIn(URL_PW, remembered(
            f"CONFIG='[db]\r\nurl = redis://:{URL_PW}@cache\r\n'\r\n"))

    def test_a_value_over_two_hundred_lines(self):
        lines = [f"line_{i} = value number {i}" for i in range(250)]
        lines[150] = f'"password": "{MEMBER}",'
        lines[200] = f"url = postgres://app:{URL_PW}@db/x"
        body = "CONFIG='" + "\n".join(lines) + "'\nDEBUG=true\n"
        seen = remembered(body)
        self.assertIn(MEMBER, seen)
        self.assertIn(URL_PW, seen)
        self.assertNotIn("value number 7", seen)

    def test_json_with_escaped_quotes_in_a_double_quoted_value(self):
        """dotenv reads \\" inside double quotes as a quote: the JSON a
        reader of the file has is the unescaped one."""
        body = f'CONFIG="{{\\"password\\": \\"{MEMBER}\\"}}"\n'
        self.assertIn(MEMBER, remembered(body))


class ShellDefaultTest(_Files):
    """${VAR:-word} and its kin: the word is a piece of the value, judged
    and remembered as the value is; a bare ${VAR} or $VAR stays harmless."""

    OPERATORS = (":-", "-", ":=", "=", ":+", "+")

    def test_the_default_word_is_remembered_under_every_operator(self):
        failures = []
        for op in self.OPERATORS:
            body = f"POSTGRES_PASSWORD=${{POSTGRES_PASSWORD{op}{DEFAULT}}}\n"
            if DEFAULT not in remembered(body):
                failures.append(op)
        self.assertEqual([], failures)

    def test_the_reported_default_is_not_repeated(self):
        body = f"POSTGRES_PASSWORD=${{POSTGRES_PASSWORD:-{DEFAULT}}}\n"
        self.assertEqual({}, self.shown(".env", body, DEFAULT))

    def test_a_default_inside_a_longer_value(self):
        for body in (f"DATABASE_URL=postgres://u:${{DB_PASS:-{DEFAULT}}}@h/db\n",
                     f"OPTS=--password ${{DB_PASSWORD:-{DEFAULT}}} -v\n"):
            with self.subTest(body):
                self.assertIn(DEFAULT, remembered(body))

    def test_a_quoted_default_is_remembered_without_its_quotes(self):
        body = f"PGPASSWORD=${{PGPASSWORD:-\"{DEFAULT}\"}}\n"
        self.assertIn(DEFAULT, remembered(body))

    def test_a_default_in_another_file(self):
        """A compose file writes the same default under a key that names a
        secret: the value is found by its key, the word is a piece of it."""
        body = f"services:\n  db:\n    password: ${{DB_PASSWORD:-{DEFAULT}}}\n"
        self.assertIn(DEFAULT, remembered(body, "docker-compose.yml"))

    # -- sad ---------------------------------------------------------------
    def test_a_bare_variable_stays_harmless(self):
        for body in ("DB_PASSWORD=${DB_PASSWORD}\n", "DB_PASSWORD=$DB_PASSWORD\n",
                     "LOG_DIR=${HOME}\n"):
            with self.subTest(body):
                self.assertEqual(set(), remembered(body))

    def test_a_harmless_default_stays_harmless_and_ungarbled(self):
        body = "LOG_LEVEL=${LOG_LEVEL:-info}\nDEBUG=${DEBUG:-true}\n"
        self.assertEqual(set(), remembered(body))
        said = "Construe the information: level info, debug true."
        self.assertEqual(said, said_about(body, ".env", said))

    def test_a_harmless_default_inside_a_value_is_no_piece(self):
        """JAVA_OPTS is remembered whole; LOG_LEVEL's default "debug" is no
        secret under its variable's name or the setting's key, and hiding
        it would take the word out of every sentence about the file."""
        body = "JAVA_OPTS=-Xmx512m -Dlog=${LOG_LEVEL:-debug}\n"
        seen = remembered(body)
        self.assertNotIn("debug", seen)
        self.assertEqual("It runs in debug mode.",
                         said_about(body, ".env", "It runs in debug mode."))

    def test_a_default_is_no_secret_under_a_harmless_name_only(self):
        """Judged under the setting's key and the variable's name both:
        PASSWORD's value is "info" when LEVEL is unset."""
        self.assertIn("info", remembered("PASSWORD=${LEVEL:-info}\n"))
        self.assertIn("info", remembered("LEVEL=${PASSWORD:-info}\n"))

    # -- edge --------------------------------------------------------------
    def test_an_empty_default_is_harmless(self):
        for op in self.OPERATORS:
            with self.subTest(op):
                self.assertEqual(set(), remembered(
                    f"DB_PASSWORD=${{DB_PASSWORD{op}}}\n"))

    def test_a_nested_default(self):
        self.assertEqual(set(), remembered("DB_PASSWORD=${A:-${B}}\n"))
        self.assertIn(DEFAULT, remembered(
            f"DB_PASSWORD=${{A:-${{B:-{DEFAULT}}}}}\n"))

    def test_a_default_that_never_closes_is_the_value(self):
        body = f"DB_PASSWORD=${{A:-{DEFAULT}\n"
        seen = remembered(body)
        self.assertIn(f"${{A:-{DEFAULT}", seen)

    def test_deep_nesting_is_read_in_one_pass(self):
        import time
        depth = 20_000
        value = "".join(f"${{V{i}:-" for i in range(depth)) + DEFAULT + "}" * depth
        started = time.process_time()
        seen = remembered(f"DB_PASSWORD={value}\n")
        self.assertIn(DEFAULT, seen)
        self.assertLess(time.process_time() - started, 5.0)


class DecodedFormTest(_Files):
    """The forms a model decodes a value into: a percent-encoded password,
    the base64 of an Authorization header and the password in it -- each
    only where it differs from the raw form and is long enough to look for
    (four characters, as every remembered form)."""

    def test_a_percent_encoded_password_is_remembered_decoded(self):
        body = "DATABASE_URL=postgres://u:Kx9v%40Q2mZ7pL4@h/db\n"
        seen = remembered(body)
        self.assertIn("Kx9v@Q2mZ7pL4", seen)
        self.assertIn("Kx9v%40Q2mZ7pL4", seen)
        self.assertEqual({}, self.shown(".env", body, "Kx9v@Q2mZ7pL4"))

    def test_a_basic_header_is_remembered_as_its_base64_and_password(self):
        body = f"AUTH_HEADER=Basic {BASIC}\n"
        seen = remembered(body)
        self.assertIn(BASIC, seen)
        self.assertIn(BASIC_PW, seen)
        self.assertNotIn("svc_user", seen)
        self.assertEqual({}, self.shown(".env", body, BASIC, BASIC_PW))

    def test_an_authorization_line_in_another_file(self):
        body = f"curl -H 'Authorization: Basic {BASIC}' https://api.example\n"
        self.assertIn(BASIC_PW, remembered(body, "deploy.sh"))

    def test_a_bearer_header_is_remembered_as_its_token(self):
        token = "Tq7Lm2Xp9Rt4Vn8KsZw3"
        self.assertIn(token, remembered(f"AUTH_HEADER=Bearer {token}\n"))

    # -- sad ---------------------------------------------------------------
    def test_a_percent_with_no_valid_escape_adds_nothing(self):
        for password in ("Kx9v%zzQ2mZ7pL4", "Kx9vQ2mZ7pL4%", "Kx9v%4"):
            body = f"DATABASE_URL=postgres://u:{password}@h/db\n"
            with self.subTest(password):
                seen = remembered(body)
                self.assertIn(password, seen)
                self.assertFalse({form for form in seen if "%" not in form
                                  and form.startswith("Kx9v")}, seen)

    def test_an_escape_that_is_no_utf8_adds_nothing(self):
        seen = remembered("DATABASE_URL=postgres://u:Kx9v%ffQ2mZ7pL4@h/db\n")
        self.assertFalse({form for form in seen if "\ufffd" in form}, seen)

    def test_invalid_base64_adds_nothing_decoded(self):
        """The credential after the scheme is still a form of its own; no
        form is decoded out of it -- every form is a piece of the value."""
        for value in ("Basic not*base64!", "Basic Zm9vYmFy====",
                      "Basic Zm9vYmFyO"):
            with self.subTest(value):
                seen = remembered(f"AUTH_HEADER={value}\n")
                self.assertIn(value.split()[1], seen)
                self.assertEqual([], [form for form in seen
                                      if form not in value])

    def test_base64_with_no_colon_gives_no_password(self):
        encoded = base64.b64encode(b"just-a-token-value").decode()
        seen = remembered(f"AUTH_HEADER=Basic {encoded}\n")
        self.assertNotIn("just-a-token-value", seen)

    # -- edge --------------------------------------------------------------
    def test_a_short_decoded_password_is_not_looked_for(self):
        encoded = base64.b64encode(b"admin:abc").decode()
        seen = remembered(f"AUTH_HEADER=Basic {encoded}\n")
        self.assertNotIn("abc", seen)
        self.assertNotIn("admin", seen)
        self.assertIn(encoded, seen)

    def test_an_encoded_user_is_decoded_with_its_password(self):
        seen = remembered("DATABASE_URL=postgres://svc%2Bapp:Kx9v%2FQ2mZ7@h/db\n")
        self.assertIn("Kx9v/Q2mZ7", seen)
        self.assertIn("svc+app:Kx9v/Q2mZ7", seen)


class CompositeValueTest(_Files):
    """A review, still at 0.1.69: a value made of parts, whose
    part a model names on its own -- Spring's placeholder default, a
    user:password pair, a list of keys. Each part is a piece under a key
    that names or hints a secret; under any other key only a part no word
    holds by chance (_opaque_run), so a list of hosts stays words."""

    #: (file name, body, the part a model names)
    COMPOSITES = (
        ("app.properties",
         "spring.datasource.password=${DB_PASSWORD:Kx9vQ2mZ7pL4}\n",
         "Kx9vQ2mZ7pL4"),
        ("app.properties", "db.password=${db.password:Kx9vQ2mZ7pL4}\n",
         "Kx9vQ2mZ7pL4"),
        (".env", "BASIC_AUTH=admin:Sonnenschein2026\n", "Sonnenschein2026"),
        (".env", "API_KEYS=Kx9vQ2mZ7pL4Wq8r,Zt5Yn3Wq8rKx9vQ2\n",
         "Zt5Yn3Wq8rKx9vQ2"),
        (".env", "API_KEYS=Kx9vQ2mZ7pL4Wq8r;Zt5Yn3Wq8rKx9vQ2\n",
         "Kx9vQ2mZ7pL4Wq8r"),
        (".env", "BACKUP_USERS=anna:Herbstwind77,ben:Sommer24x\n", "Sommer24x"),
        (".env", "NODE_IDS=Kx9vQ2mZ7pL4Wq8r,Zt5Yn3Wq8rKx9vQ2\n",
         "Zt5Yn3Wq8rKx9vQ2"),
    )

    def test_each_part_is_remembered(self):
        failures = [(name, body) for name, body, part in self.COMPOSITES
                    if part not in remembered(body, name)]
        self.assertEqual([], failures)

    def test_the_reported_values_repeat_no_part(self):
        for name, body, part in self.COMPOSITES[:4]:
            with self.subTest(body):
                self.assertEqual({}, self.shown(name, body, part))

    # -- sad ---------------------------------------------------------------
    def test_parts_that_are_words_under_another_key_are_not(self):
        for body, words in (
                ("ALLOWED_HOSTS=localhost,example.com,intranet\n",
                 {"localhost", "example.com", "intranet"}),
                ("DB_HOST=db.internal:5432\n", {"5432", "db.internal"}),
                ("LOCALES=de_DE;en_US;fr_FR\n", {"de_DE", "en_US"}),
                ("MIRRORS=https://a.example/x,https://b.example/y\n",
                 {"//a.example/x", "https://a.example/x"})):
            with self.subTest(body):
                self.assertFalse(words & remembered(body), remembered(body))

    def test_a_placeholder_with_a_harmless_default_is_harmless(self):
        self.assertEqual(set(), remembered("log.level=${LOG_LEVEL:info}\n",
                                           "app.properties"))
        self.assertEqual(set(), remembered("db.password=${DB_PASSWORD:}\n",
                                           "app.properties"))

    def test_the_user_is_no_part(self):
        seen = remembered("BASIC_AUTH=admin:Sonnenschein2026\n")
        self.assertNotIn("admin", seen)

    # -- edge --------------------------------------------------------------
    def test_empty_parts_and_a_colon_in_a_url(self):
        seen = remembered("API_KEYS=,Kx9vQ2mZ7pL4Wq8r,,\n")
        self.assertIn("Kx9vQ2mZ7pL4Wq8r", seen)
        seen = remembered("DB_PASSWORD=postgres://app:Hn5Rt8Wq3Zm6Kd@db/x\n")
        self.assertNotIn("//app:Hn5Rt8Wq3Zm6Kd@db/x", seen)
        self.assertIn("Hn5Rt8Wq3Zm6Kd", seen)

    def test_a_shell_substring_is_read_as_a_default(self):
        """${NAME:2} is a substring to a shell; read as Spring's default it
        only ever adds a piece, and a short one is none."""
        self.assertEqual(set(), remembered("LOG_LEVEL=${LEVEL:2}\n"))


class MidLineRunTest(_Files):
    """A review: a value over several lines was remembered as the value and each
    of its lines; an opaque run (_opaque_run) in the middle of a line was
    no form, and a model naming the run alone was published. Each opaque
    piece of a line is a form now, as the sides of an "=" are (0.1.83), a
    bounded number of them per value (_OPAQUE_PIECES)."""

    RUN = "Kx9vQ2mZ7pL4" + "Wq8rTz"
    THREE = ('NOTE="line one of it\n'
             f'  the run {RUN} is in the middle\n'
             '  last line"\n')

    def test_the_run_in_the_middle_of_a_line_is_remembered(self):
        seen = remembered(self.THREE)
        self.assertIn(self.RUN, seen)
        self.assertIn(f"the run {self.RUN} is in the middle", seen)

    def test_the_run_is_remembered_with_its_punctuation_stripped(self):
        body = f'NOTE="first\n  ({self.RUN}),\n  last"\n'
        self.assertIn(self.RUN, remembered(body))

    def test_a_properties_continuation_too(self):
        body = f"note=first \\\n    x {self.RUN} y \\\n    last\n"
        self.assertIn(self.RUN, remembered(body, "app.properties"))

    def test_a_one_line_value_too(self):
        body = f"NOTE=the run {self.RUN} is here\n"
        self.assertIn(self.RUN, remembered(body))

    # -- sad ---------------------------------------------------------------
    def test_the_words_of_a_line_are_no_forms(self):
        seen = remembered(self.THREE)
        for word in ("line", "one", "the", "run", "middle", "last", "is in"):
            self.assertNotIn(word, seen)

    def test_a_run_that_is_a_word_or_a_number_is_none(self):
        body = 'NOTE="first\n  Dampfschiff 1234567890 x\n  last"\n'
        seen = remembered(body)
        self.assertNotIn("Dampfschiff", seen)
        self.assertNotIn("1234567890", seen)

    # -- edge --------------------------------------------------------------
    def test_the_pieces_of_a_value_are_bounded(self):
        from sift.extract.base import _OPAQUE_PIECES, _value_forms
        runs = [f"Kx9vQ2mZ7pL{n:04d}" for n in range(_OPAQUE_PIECES * 3)]
        forms = _value_forms("a " + " b ".join(runs), "NOTE")
        self.assertEqual(_OPAQUE_PIECES, len([f for f in forms if f in runs]))
        self.assertIn(runs[0], forms)
        self.assertNotIn(runs[-1], forms)


class HideSaidOrderTest(_Files):
    """A review, still at 0.1.69: hide_said hid the file's
    known values first, and a known value can be the very word scrub reads
    a sentence by -- a JSON member "secret": "password", a setting
    AUTH_METHOD=password. "The admin password is Kx9vQ2mZ7pL4" became "The
    admin <8 chars> is Kx9vQ2mZ7pL4", and scrub found no key left to read
    it by."""

    SAID = "The admin password is Kx9vQ2mZ7pL4."
    FILES = (
        ("vault.json",
         '{"type": "secret", "secret": "password", "hint": "Kx9vQ2mZ7pL4"}\n'),
        ("app.env", "AUTH_METHOD=password\nADMIN_HINT=first pet\n"),
        ("oauth.env", "GRANT_TYPE=password\nCLIENT_ID=portal\n# the old admin "
                      "password was Kx9vQ2mZ7pL4 until May\n"),
        ("creds.tsv", "system\tsecret\thint\ndb1\tpassword\tKx9vQ2mZ7pL4\n"),
    )

    def said(self, text, *known):
        from sift.extract.base import hide_said
        with remember_hidden() as seen:
            seen.update(known)
            return hide_said(text)

    def test_a_known_keyword_does_not_hide_the_value_after_it(self):
        said = self.said(self.SAID, "password")
        self.assertNotIn("Kx9vQ2mZ7pL4", said)
        self.assertNotIn("password", said)

    def test_the_reported_files_repeat_no_password(self):
        failures = []
        for name, body in self.FILES:

            class Speaker(_Model):
                def generate(self, system, user, **kw):
                    said = HideSaidOrderTest.SAID
                    return LLMResult(text=(
                        f"TYPE: Config\nSUMMARY: {said}\nKEY POINTS:\n- "
                        f"{said}\nENTITIES: none\nNOTES: none"),
                        model="stub", provider="stub")

            self.count += 1
            folder = self.dir / str(self.count)
            folder.mkdir()
            path = folder / name
            path.write_text(body, encoding="utf-8")
            summ = summarize_path(path, Config(quiet=True, cache=False),
                                  Speaker("x"), None)
            if "Kx9vQ2mZ7pL4" in render_json([summ]):
                failures.append((name, summ.summary))
        self.assertEqual([], failures)

    def test_a_value_known_whole_is_still_hidden_whole(self):
        """Why the known values come first: the
        sentence rule stops at the blank a model writes for the tab."""
        said = self.said("password: Kx9v Q2mZ7pL4Wq8rTz", "Kx9v\tQ2mZ7pL4Wq8rTz",
                         "Kx9v Q2mZ7pL4Wq8rTz")
        self.assertNotIn("Q2mZ7pL4Wq8rTz", said)

    # -- sad ---------------------------------------------------------------
    def test_a_sentence_with_no_secret_loses_only_the_known_word(self):
        said = self.said("The password policy requires 12 characters.",
                         "password")
        self.assertEqual("The <8 chars — not shown> policy requires 12 "
                         "characters.", said)

    # -- edge --------------------------------------------------------------
    def test_outside_a_file_it_is_scrub(self):
        from sift.extract.base import hide_said, scrub
        for text in (self.SAID, "The password policy requires 12 characters."):
            with self.subTest(text):
                self.assertEqual(scrub(text), hide_said(text))


if __name__ == "__main__":
    unittest.main()
