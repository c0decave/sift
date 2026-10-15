"""find_secrets: where the secrets are, in the raw text, by character.

Redaction replaces text and says nowhere what it replaced, and it starts by
rewriting bidi controls, which moves every offset after them. A viewer that
masks a file in place needs the positions instead — and needs them to cover
everything sift's own redaction would hide, or the masked view publishes
what the summary was careful to keep back.

The oracle says exactly that: after the spans are masked, scrub, hide_known
and redact_assignments(everything=True) find nothing left to change outside
the mask.
"""
import difflib
import json
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.extract import SECRET_SCAN_LIMIT, Span, find_secrets
from sift.extract.base import (hide_known, learn_secrets,
                               looks_like_a_credential, redact_assignments,
                               redact_value, remember_hidden, scrub)

import corpus_secrets as corpus
from _timing import cpu_ratio, least_cpus, linear_bound

#: Keys a `K=v` line can carry — _ASSIGNMENT's own key class.
_ASSIGNABLE = re.compile(r"^[\w.\-\[\]]{1,64}$")

_KEYED = [(k, v) for k, v in (corpus.SECRETS + corpus.SECRETS_HELD_OUT
                              + corpus.SECRETS_THIRD + corpus.SPAN_PAIRS)
          if k and _ASSIGNABLE.match(k)]
#: The pairs sift hides whole under their key. redact_assignments(everything=
#: True) rewrites a line's whole value, so it can only find nothing left
#: where the whole value is the secret.
ORACLE = [(k, v) for k, v in _KEYED if looks_like_a_credential(v, k)]
#: And the ones where the secret is part of the value — a connection string
#: whose AccountKey is the credential and whose protocol is not.
PARTIAL = [(k, v) for k, v in _KEYED if not looks_like_a_credential(v, k)]
BARE = [v for k, v in corpus.SECRETS + corpus.SECRETS_THIRD if not k]

#: What masks a span. Not a blank: a blank is a separator to every pattern
#: here, and blanking "DB_PASSWORD=hunter2\nAPI_TOKEN=…" put the next line's
#: key within reach of the first one's — scrub then hid "API_TOKEN=", which
#: is the mask talking, not the file. U+2588 is no word character, no
#: separator and part of no credential shape.
MASK = "\u2588"


def forms(key: str, value: str) -> dict:
    return {"K=v": f"{key}={value}", "k: v": f"{key}: {value}",
            "json": json.dumps({key: value})}


def mask(text: str, spans) -> str:
    """Every secret span overwritten, every offset kept."""
    chars = list(text)
    for span in spans:
        if span.reason != "bidi":
            chars[span.start:span.end] = MASK * (span.end - span.start)
    return "".join(chars)


def changed_outside(masked: str, after: str) -> "list[str]":
    """What ``after`` changed in ``masked`` beyond the masked characters.

    A redaction function may rewrite the mask itself — redact_assignments
    (everything=True) rewrites any value, its own marker included, so a
    literal fixpoint is out of reach whatever the spans are — and it may
    tidy the blanks around a value it rewrites. What must not happen is a
    change to a character the spans left standing: that is a secret they
    missed. Inserting text hides nothing, so only what is removed counts.
    """
    outside = []
    matcher = difflib.SequenceMatcher(None, masked, after, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in ("replace", "delete") and masked[i1:i2].replace(
                MASK, "").strip():
            outside.append(f"{masked[i1:i2]!r} -> {after[j1:j2]!r}")
    return outside


def left_to_hide(raw: str, masked: str, everything: bool) -> "list[str]":
    """What each redaction function would still hide outside the mask."""
    found = []
    # redact_assignments rejoins the lines it read, so it drops a final
    # line break even when it hides nothing; that is its unchanged output.
    rejoined = "\n".join(masked.splitlines())
    with remember_hidden():
        learn_secrets(raw)
        results = {"scrub": (masked, scrub(masked)),
                   "hide_known": (masked, hide_known(masked)),
                   "redact_assignments": (rejoined, redact_assignments(
                       masked, everything=everything))}
    for name, (before, after) in results.items():
        found += [f"{name}: {change}"
                  for change in changed_outside(before, after)]
    return found


class OracleTest(unittest.TestCase):
    """Nothing sift would hide survives the mask."""

    #: Files whose secrets sift recognises only by a CSV column's name,
    #: where the table is read: a raw text carries no key beside the cell.
    COLUMN_ONLY = {"keys.csv": ["hunter2", "s3cret!", "pw12345"],
                   "zugang.csv": ["Kastanie2026!", "Lindenblatt9"]}

    def test_the_oracle_pairs_are_secrets_to_sift(self):
        """Otherwise the oracle would be measuring a value sift keeps."""
        self.assertEqual([], [(k, v) for k, v in _KEYED
                              if redact_value(v, k) == v])
        self.assertGreater(len(ORACLE), 60, "the oracle lost its corpus")
        self.assertEqual([("storage", "DefaultEndpointsProtocol=https;"
                           "AccountKey=abcdEFGH1234567890==")], PARTIAL)

    def test_every_form_of_every_pair_is_a_fixpoint(self):
        failures = []
        for key, value in ORACLE:
            for name, raw in forms(key, value).items():
                masked = mask(raw, find_secrets(raw))
                problems = left_to_hide(raw, masked, everything=True)
                if name == "json":
                    parsed = json.loads(masked)
                    problems += changed_outside(parsed[key],
                                                scrub(parsed)[key])
                    if list(parsed) != [key]:
                        problems.append(f"the key was masked: {parsed!r}")
                elif not masked.startswith(key):
                    problems.append(f"the key was masked: {masked!r}")
                if problems:
                    failures.append((name, raw, problems))
        self.assertEqual([], failures,
                         f"{len(failures)} of {3 * len(ORACLE)} forms leave "
                         f"something to hide")

    def test_a_secret_inside_a_value_is_masked_and_the_rest_kept(self):
        for key, value in PARTIAL:
            for name, raw in forms(key, value).items():
                with self.subTest(form=name):
                    masked = mask(raw, find_secrets(raw))
                    self.assertEqual([], left_to_hide(raw, masked,
                                                      everything=False))
                    self.assertIn("DefaultEndpointsProtocol=https;", masked)
                    self.assertNotIn("abcdEFGH1234567890", masked)

    def test_a_value_standing_alone_is_a_fixpoint(self):
        failures = []
        for raw in BARE + [f"see {v} for details" for v in BARE]:
            problems = left_to_hide(raw, mask(raw, find_secrets(raw)),
                                    everything=False)
            if problems:
                failures.append((raw, problems))
        self.assertEqual([], failures)

    def test_a_repetition_is_masked_where_it_repeats(self):
        """What hide_known does for a summary: the value, found once with
        its key, is a secret wherever else it turns up."""
        raw = ("DB_PASSWORD=Winterreise2026!\n"
               "Notiz: bei Problemen Winterreise2026 eingeben.\n"
               "Backup-Kennwort ist dasselbe: Winterreise2026!\n")
        spans = find_secrets(raw)
        masked = mask(raw, spans)
        self.assertNotIn("Winterreise2026", masked)
        self.assertEqual([], left_to_hide(raw, masked, everything=False))
        self.assertIn("repeated", {s.reason for s in spans})
        self.assertIn("Notiz: bei Problemen", masked)

    def test_the_whole_corpus_files_are_fixpoints(self):
        failures, missed = [], {}
        for name, body, secrets in (corpus.FILES + corpus.CREDENTIAL_FILES
                                    + corpus.CREDENTIAL_FILES_FOURTH):
            masked = mask(body, find_secrets(body, filename=name))
            problems = left_to_hide(body, masked, everything=name.endswith(
                (".env", ".properties")))
            left = [s for s in secrets if s in masked]
            if left:
                missed[name] = left
            if problems:
                failures.append((name, problems))
        self.assertEqual([], failures)
        self.assertEqual(self.COLUMN_ONLY, missed)


def json_leaves(parsed, scrubbed, found=None):
    """Every string the redaction changed in a parsed JSON document, as the
    JSON text writes it — the values a structured file's summary hides."""
    found = [] if found is None else found
    if isinstance(parsed, dict):
        for key in parsed:
            json_leaves(parsed[key], scrubbed[key], found)
    elif isinstance(parsed, list):
        for before, after in zip(parsed, scrubbed):
            json_leaves(before, after, found)
    elif isinstance(parsed, str) and parsed != scrubbed:
        found.append(json.dumps(parsed)[1:-1])
    return found


class StructuredTextTest(unittest.TestCase):
    """A structured file's summary is built from the parsed document, and
    scrub judges each value there by its key — whatever the layout of the
    text around it. The spans have to reach the same values in the text:
    a pretty-printed or aligned member, an array under a secret key, a key
    or value longer than a line, a YAML block scalar."""

    PEM_BODY = "A1b2C3d4" * 540                 # 4,320 characters

    JSON_CASES = {
        "newline before the value": '{"password":\n "Sommer24!"}',
        "aligned colons": '{"password"      :           "Sommer24!"}',
        "pretty-printed": json.dumps(
            {"db": {"host": "db.intern", "password": "Sommer24!"},
             "api_token": corpus.GITHUB_TOKEN},
            indent=2),
        "an array under a secret key": '{"passwords": ["Sommer24!", "x"]}',
        "nested arrays under a secret key":
            '{"tokens": [["Winterreise2026!", "Kastanie2026!"]]}',
        "objects inside an array": json.dumps(
            {"users": [{"name": "anna", "password": "Sommer24!"},
                       {"name": "bob", "password": "Winter25?"}]}, indent=1),
        "a long value": json.dumps({"token": "A1b2C3d4" * 520}),
        "a long key": json.dumps({"k" * 59 + "password": "Sommer24!xyz"}),
        "a long private key": json.dumps({"private_key": PEM_BODY}),
    }

    def test_every_value_scrub_hides_in_a_json_document_is_masked(self):
        failures = {}
        for name, raw in self.JSON_CASES.items():
            parsed = json.loads(raw)
            hidden = json_leaves(parsed, scrub(parsed))
            self.assertTrue(hidden, f"{name}: sift hides nothing here")
            masked = mask(raw, find_secrets(raw, filename="config.json"))
            left = [value for value in hidden if value in masked]
            if left:
                failures[name] = left
        self.assertEqual({}, failures)

    def test_the_summary_of_each_file_hides_what_the_spans_mask(self):
        """The reviewer's comparison, run: summarize_path hides the value,
        so the masked view must not show it."""
        import shutil
        import tempfile
        from sift.config import Config
        from sift.summarize import summarize_path
        where = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, where, ignore_errors=True)
        for name, raw in self.JSON_CASES.items():
            with self.subTest(name):
                path = where / "c.json"
                path.write_text(raw, encoding="utf-8")
                shown = json.dumps(summarize_path(
                    path, Config(no_llm=True, cache=False)).to_dict())
                masked = mask(raw, find_secrets(raw, filename="c.json"))
                for value in json_leaves(json.loads(raw),
                                         scrub(json.loads(raw))):
                    if value not in shown:
                        self.assertNotIn(value, masked)

    def test_a_yaml_block_scalar_is_the_lines_under_the_key(self):
        cases = {
            "folded": ("password: >\n  Sommer24!\n", ["Sommer24!"]),
            "literal, nested": ("db:\n  host: db.intern\n  password: |\n"
                                "    Sommer24!\nport: 5432\n", ["Sommer24!"]),
            "chomping indicator": ("secret: |-\n  Winterreise2026!\n"
                                   "user: anna\n", ["Winterreise2026!"]),
        }
        for name, (raw, secrets) in cases.items():
            with self.subTest(name):
                spans = find_secrets(raw, filename="app.yaml")
                masked = mask(raw, spans)
                for secret in secrets:
                    self.assertNotIn(secret, masked)
                # The indicator is YAML's, not the secret; the lines after
                # the block are the file's.
                self.assertNotIn(">", "".join(raw[s.start:s.end]
                                              for s in spans))
                self.assertTrue(masked.split("\n")[0].rstrip().endswith(
                    (">", "|", "|-", ":")))

    def test_a_block_scalar_that_is_no_secret_is_kept(self):
        raw = "description: >\n  The nightly export of all invoices.\n"
        self.assertEqual([], find_secrets(raw, filename="app.yaml"))

    def test_a_key_line_longer_than_sixty_four_characters(self):
        key = "k" * 59 + "_password"
        raw = f"{key}: Sommer24!xyz\n"
        masked = mask(raw, find_secrets(raw, filename="app.yaml"))
        self.assertNotIn("Sommer24!xyz", masked)
        self.assertTrue(masked.startswith(key))


class FalsePositiveTest(unittest.TestCase):
    """The other rate. Spans that swallow ordinary values make a masked view
    as useless as a summary that hides everything."""

    LISTS = {"INNOCENT": corpus.INNOCENT,
             "INNOCENT_HELD_OUT": corpus.INNOCENT_HELD_OUT,
             "INNOCENT_THIRD": corpus.INNOCENT_THIRD,
             "INNOCENT_CONFIG": corpus.INNOCENT_CONFIG}

    @staticmethod
    def _rendered(key, value):
        """(text, where the value starts, the value as the text writes it)."""
        if not key:
            return {"bare": (value, 0, value)}
        out = {}
        for name, text in forms(key, value).items():
            shown = json.dumps(value)[1:-1] if name == "json" else value
            out[name] = (text, text.rfind(shown), shown)
        return out

    @staticmethod
    def _today_destroys(text, key, value, shown) -> bool:
        """What sift's redaction does to this value today: scrub of the text,
        the preview's redact_assignments, and scrub of the value under its
        key, which is what a parsed file's metadata gets."""
        return (shown not in scrub(text)
                or shown not in redact_assignments(text)
                or redact_value(value, key) != value)

    @staticmethod
    def _spans_destroy(text, where, shown) -> bool:
        end = where + len(shown)
        return any(s.reason != "bidi" and s.start < end and s.end > where
                   for s in find_secrets(text))

    def measure(self):
        rates = {}
        for list_name, pairs in self.LISTS.items():
            today = spans = total = 0
            for key, value in pairs:
                for text, where, shown in self._rendered(key, value).values():
                    total += 1
                    today += self._today_destroys(text, key, value, shown)
                    spans += self._spans_destroy(text, where, shown)
            rates[list_name] = (spans, today, total)
        return rates

    def test_no_list_loses_more_to_spans_than_to_redaction(self):
        rates = self.measure()
        worse = {name: rate for name, rate in rates.items()
                 if rate[0] > rate[1]}
        self.assertEqual({}, worse,
                         "(spans, redaction today, rendered values): "
                         f"{rates}")

    def test_sentences_about_secrets_keep_every_word(self):
        masked = {sentence: [s for s in find_secrets(sentence)
                             if s.reason != "bidi"]
                  for sentence in (corpus.SENTENCES_KEPT
                                   + corpus.SENTENCES_ABOUT_SECRETS)}
        self.assertEqual({}, {k: v for k, v in masked.items() if v})

    def test_the_first_list_loses_nothing(self):
        spans, today, total = self.measure()["INNOCENT"]
        self.assertEqual((0, 0), (spans, today), f"of {total}")


class SpanShapeTest(unittest.TestCase):

    # -- happy -------------------------------------------------------------
    def test_a_token_in_prose_is_exactly_the_token(self):
        text = f"der neue Token ist {corpus.GITHUB_TOKEN} bitte"
        spans = find_secrets(text)
        self.assertEqual(1, len(spans))
        self.assertEqual(corpus.GITHUB_TOKEN,
                         text[spans[0].start:spans[0].end])
        self.assertIsInstance(spans[0], Span)

    def test_a_private_key_is_the_whole_block(self):
        text = (f"id_rsa:\n{corpus.OPENSSH_BEGIN}\n"
                "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU\n"
                f"{corpus.OPENSSH_END}\ntrailer")
        (span,) = find_secrets(text)
        self.assertTrue(text[span.start:].startswith("-----BEGIN"))
        self.assertTrue(text[:span.end].endswith("PRIVATE KEY-----"))
        self.assertEqual("private_key", span.reason)

    def test_an_assignment_hides_the_value_and_keeps_the_key(self):
        text = "HOST=db.intern\r\nDB_PASSWORD=hunter2\r\nPORT=5432\r\n"
        (span,) = find_secrets(text)
        self.assertEqual("hunter2", text[span.start:span.end])

    def test_json_hides_the_value_under_a_secret_key(self):
        text = '{"host": "db", "password": "Sommer24!", "port": 5432}'
        (span,) = find_secrets(text)
        self.assertEqual("Sommer24!", text[span.start:span.end])

    def test_a_settings_file_has_every_value_hidden(self):
        """redact_assignments(everything=True): in a .env or .properties the
        values are the reason the file is not in version control."""
        text = "# comment\nHOST=db\nPORT=5432\n"
        for name in (".env", "prod.env", ".env.local", "app.properties"):
            with self.subTest(name):
                found = [text[s.start:s.end]
                         for s in find_secrets(text, filename=name)]
                self.assertEqual(["db", "5432"], found)
        self.assertEqual([], find_secrets(text))
        self.assertEqual([], find_secrets(text, filename="settings.ini"))

    def test_pgpass_and_netrc_by_name_take_every_password(self):
        pgpass = "# local\ndb:5432:app:app:geheim\n*:*:*:svc:pass\\:word\n"
        found = [pgpass[s.start:s.end]
                 for s in find_secrets(pgpass, filename=".pgpass")]
        self.assertEqual(["geheim", "pass\\:word"], found)
        netrc = ("machine ftp.example.de login backup password geheim\n"
                 "machine mail login anna\n  password sonne account kasse\n")
        found = [netrc[s.start:s.end]
                 for s in find_secrets(netrc, filename=".netrc")]
        self.assertEqual(["geheim", "sonne", "kasse"], found)

    def test_bidi_controls_are_their_own_spans_and_move_nothing(self):
        text = ("Betrag ‮00,001‬ EUR, "
                f"Token {corpus.GITHUB_TOKEN}")
        spans = find_secrets(text)
        bidi = [(s.start, s.end) for s in spans if s.reason == "bidi"]
        self.assertEqual([(7, 8), (14, 15)], bidi)
        (token,) = [s for s in spans if s.reason != "bidi"]
        self.assertEqual(corpus.GITHUB_TOKEN,
                         text[token.start:token.end])

    def test_the_offsets_are_characters_not_bytes(self):
        text = "Größe: 12 €, Übergabe — password=Sommer24!"
        (span,) = find_secrets(text)
        self.assertEqual("Sommer24!", text[span.start:span.end])

    # -- sad ---------------------------------------------------------------
    def test_nothing_secret_is_no_spans(self):
        for text in ("", "   ", "Die Sitzung beginnt um 14 Uhr.",
                     "The key is E-flat major and the salt is 10 g."):
            with self.subTest(text):
                self.assertEqual([], find_secrets(text))

    def test_only_text_is_scanned(self):
        for bad in (None, b"password=hunter2", 42):
            with self.subTest(bad):
                with self.assertRaises(TypeError):
                    find_secrets(bad)

    def test_it_leaves_the_callers_record_of_hidden_values_alone(self):
        with remember_hidden() as seen:
            find_secrets("DB_PASSWORD=hunter2 and ghp_" "16C7e42F292c6912E77")
            self.assertEqual(set(), seen)

    # -- edge --------------------------------------------------------------
    def test_spans_are_sorted_and_secret_spans_never_overlap(self):
        text = (f"token: {corpus.GITHUB_TOKEN}\n"
                f"api_key={corpus.GITHUB_TOKEN} "
                "‮ sk_live_" "51H8qLKGxYzAbCdEfGhIjKlMnOpQrSt\n")
        spans = find_secrets(text)
        self.assertEqual(sorted(spans), spans)
        secret = [s for s in spans if s.reason != "bidi"]
        for first, second in zip(secret, secret[1:]):
            self.assertLessEqual(first.end, second.start)

    def test_only_the_first_scan_limit_characters_are_scanned(self):
        """The same limit learn_secrets has, stated for the caller."""
        filler = "x" * (SECRET_SCAN_LIMIT - 30)
        text = (filler + " password=Sommer24!" + " " * 50
                + " password=Winter25!")
        spans = find_secrets(text)
        self.assertEqual(["Sommer24!"], [text[s.start:s.end] for s in spans])
        self.assertTrue(all(s.start < SECRET_SCAN_LIMIT for s in spans))
        self.assertEqual(400_000, SECRET_SCAN_LIMIT)

    def test_a_secret_that_starts_inside_the_limit_runs_to_its_end(self):
        """A private key beginning 500 characters before the limit was
        masked up to the limit and its last 811 characters were shown."""
        pem = (f"{corpus.OPENSSH_BEGIN}\n"
               + "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU" * 40
               + f"\n{corpus.OPENSSH_END}")
        start = SECRET_SCAN_LIMIT - 500
        text = "x" * (start - 1) + "\n" + pem + "\ntail"
        (span,) = find_secrets(text)
        self.assertEqual((start, start + len(pem)), (span.start, span.end))
        self.assertEqual("private_key", span.reason)
        # And a value on a line that crosses the limit.
        line = "x" * (SECRET_SCAN_LIMIT - 12) + "\napi_token="
        token = corpus.GITHUB_TOKEN
        (span,) = find_secrets(line + token + "\n")
        self.assertEqual(token, (line + token)[span.start:span.end])

    def test_the_scan_limit_does_not_cost_seconds(self):
        """Distinct values, as many as fit: each one found is looked for
        again everywhere, and looking for each separately made the work
        grow with the square of the text — 21 s for 36,000 of them. The
        text up to the limit costs at most linear_bound(8) times an eighth
        of it, the two scanned in turn; five seconds of wall clock read 2.4
        under a host load of 30 to 42 (0.1.73)."""
        text = "".join(f"pw=Q{i:05d}!\n"
                       for i in range(SECRET_SCAN_LIMIT // 10))
        found = []
        ratio = cpu_ratio(
            lambda: found.append(find_secrets(text[:SECRET_SCAN_LIMIT])),
            lambda: find_secrets(text[:SECRET_SCAN_LIMIT // 8]))
        self.assertLess(ratio, linear_bound(8), f"{ratio:.1f}x the CPU for "
                                                f"8x the text")
        self.assertGreater(len(found[0]), 30_000)

    def test_distinct_values_cost_time_in_proportion_to_the_text(self):
        """Eight times the text; quadratic work was 33 times the time, and
        the bound is between that and eight. The two sizes take turns and
        are timed in CPU: one after the other on the wall clock, a burst of
        host load on one of them read as growth or as its absence."""
        def scanning(size):
            text = "".join(f"password=Val{i:07d}xyz!\n"
                           for i in range(size // 20))[:size]
            return lambda: find_secrets(text)
        large, small = least_cpus(scanning(400_000), scanning(50_000))
        self.assertLess(large, max(small, 1e-6) * 16,
                        f"{small:.2f}s for 50k, {large:.2f}s for 400k")

    def test_the_time_budget_reaches_the_scan(self):
        from sift import deadline
        text = "".join(f"pw=Q{i:05d}!\n"
                       for i in range(SECRET_SCAN_LIMIT // 10))
        with deadline.limit(0.001):
            time.sleep(0.01)
            with self.assertRaises(deadline.DeadlineExceeded):
                find_secrets(text)

    def test_a_repetition_inside_a_longer_word_is_masked(self):
        """hide_known replaces the value wherever it stands, inside a word
        too; the spans do the same."""
        raw = "password=Kastanie2026!\nbackup: xKastanie2026y\n"
        masked = mask(raw, find_secrets(raw))
        self.assertNotIn("Kastanie2026", masked)
        self.assertIn("backup: x", masked)

    def test_the_keyless_pass_is_redact_value_exactly(self):
        """The spans are read off the same stages redact_value runs, in the
        same order, on the same intermediate text."""
        from sift.extract.base import _redaction_pass
        samples = [v for _, v in corpus.SECRETS + corpus.INNOCENT]
        samples += [s for s, _ in corpus.SECRETS_IN_SENTENCES]
        samples += list(corpus.SENTENCES_KEPT) + [b for _, b, _ in corpus.FILES]
        samples += ["Betrag ‮00,001‬ EUR sk_live_" "51H8qLKGxYzAbCdEfGhIj",
                    "db.example.de:5432:kunden:svc_kunden:Sonnenschein2024"]
        for text in samples:
            with self.subTest(text[:40]):
                self.assertEqual(redact_value(text), _redaction_pass(text).text)


#: The JSON scans as find_secrets ran them up to 0.1.63: the oracle for
#: what the linear ones must still find, member for member.
_OLD_MEMBER = re.compile(r'"((?:[^"\\\n]|\\.)+)"\s*:\s*"((?:[^"\\\n]|\\.)*)"')
_OLD_TOKEN = re.compile(r'"((?:[^"\\\n]|\\.)*)"|[\[\]{}]')


def _old_array_values(text):
    from sift.extract.base import _JSON_COLON
    stack = []
    for match in _OLD_TOKEN.finditer(text):
        token = match.group(0)
        if token in "{[":
            key = stack[-1][1] if stack else None
            stack.append([token, key if token == "[" else None])
        elif token in "]}":
            if stack and stack[-1][0] == ("[" if token == "]" else "{"):
                stack.pop()
            else:
                stack = []
        elif stack and stack[-1][0] == "{":
            if _JSON_COLON.match(text, match.end()):
                stack[-1][1] = match.group(1)
        elif stack and stack[-1][1] is not None:
            yield match.start(1), match.end(1), stack[-1][1]


def _cpu_ratio(short, long, least=0.05, most=4000):
    """CPU seconds of ``long()`` over ``short()``, the two run in turn, as
    many times as the short one needs to cost ``least`` seconds -- so a
    burst of load on the host slows both sides alike, and a run too short
    for the clock is not a ratio of two roundings."""
    spent_short = spent_long = 0.0
    runs = 0
    while runs < most and (spent_short < least or runs < 3):
        start = time.process_time()
        short()
        middle = time.process_time()
        long()
        spent_long += time.process_time() - middle
        spent_short += middle - start
        runs += 1
    return spent_long / max(spent_short, 1e-9), spent_short, spent_long


class JsonScanTest(unittest.TestCase):
    """find_secrets' JSON scans -- each string member, and each string in an
    array under a key -- read a line of escaped quotes once.

    A string that does not close on its line was scanned to the end of the
    line from its opening quote, then again from every escaped quote in it:
    quadratic, 1.1 s for 8,000 escaped quotes at 0.1.60, four times as long
    for each doubling, and with no clock looked at in between. Every quote
    inside such a string is the second character of an escape the first
    scan stepped over, so a scan from it takes the same steps and fails the
    same way; the scans now go on from where the first one stopped, which
    finds the same members.
    """

    SHAPES = {
        "escaped quotes": lambda n: '\\"' * n,
        "a closed key full of them": lambda n: '"' + '\\"' * n + '"x',
        "a value that never closes": lambda n: '"k": "' + '\\"' * n,
        "an array that never closes": lambda n: '{"k": ["' + '\\"' * n,
        # The scan of keys in single quotes (_QUOTED_MEMBER), and the
        # apostrophes of prose, each of which opens a string to it.
        "escaped single quotes": lambda n: "\\'" * n,
        "a closed single-quoted key full of them":
            lambda n: "'" + "\\'" * n + "'x",
        "a single-quoted value that never closes":
            lambda n: "'k' => '" + "\\'" * n,
        "apostrophes": lambda n: "it's " * n,
    }

    def test_a_line_of_escaped_quotes_costs_in_proportion_to_its_length(self):
        for name, shape in self.SHAPES.items():
            with self.subTest(name):
                short, long = shape(1_000), shape(4_000)
                ratio, a, b = _cpu_ratio(lambda: find_secrets(short),
                                         lambda: find_secrets(long))
                # Four times the text: linear work is about four times the
                # time, quadratic sixteen.
                self.assertLess(ratio, 8.0, f"{ratio:.1f}x ({a:.3f} s against "
                                            f"{b:.3f} s)")

    def test_the_whole_scan_limit_of_them_takes_seconds_not_an_hour(self):
        """200,000 escaped quotes: 0.7 s of CPU measured, most of it the
        keyless pass over the text. The quadratic scans, at the growth
        measured for them (5.3 s for 8,000), would have taken an hour."""
        text = '\\"' * (SECRET_SCAN_LIMIT // 2)
        start = time.process_time()
        self.assertEqual([], find_secrets(text))
        self.assertLess(time.process_time() - start, 5.0)

    def _members(self, text):
        from sift.extract.base import _json_members
        return [(key, start, end) for key, start, end in _json_members(text)]

    def test_the_same_members_and_array_values_as_before(self):
        """Against the old patterns, on text that is JSON, text that is not,
        and a generated jumble of quotes, backslashes, colons and brackets
        -- where the old scans' restarts decided what they found."""
        import random
        from sift.extract.base import _json_array_values
        texts = [json.dumps({k: v, "list": [v, k], "n": {"k": [[v]]}},
                            indent=indent)
                 for k, v in ORACLE for indent in (None, 2)]
        texts += ['{"a": "b": "c"}', '"": "x"', '"k" : \n "v"',
                  '\\"password": "Kastanie2026"', 'x = "a\\\\", "b": "c"',
                  '{"k": ["a\\"b", "c"]}', '"k": "\\', '["a",\n"b"]']
        rnd = random.Random(20260929)
        alphabet = ['"', "\\", ":", "[", "]", "{", "}", " ", ",", "\n", "a",
                    "K", '\\"', '": "', '"k": ["', "\t"]
        texts += ["".join(rnd.choice(alphabet)
                          for _ in range(rnd.randint(0, 60)))
                  for _ in range(8_000)]
        for text in texts:
            old = [(m.group(1), m.start(2), m.end(2))
                   for m in _OLD_MEMBER.finditer(text)]
            self.assertEqual(old, self._members(text), repr(text))
            self.assertEqual(list(_old_array_values(text)),
                             list(_json_array_values(text)), repr(text))

    def test_keys_in_single_quotes_are_read_as_keys_in_double_quotes(self):
        """_QUOTED_MEMBER is _JSON_MEMBER with the quotes swapped: on a
        text with no single quote, the members of the text with its quotes
        swapped are the same members, at the same places -- whatever "=>",
        colons, escapes and brackets stand around them."""
        import random
        swap = str.maketrans("\"'", "'\"")
        rnd = random.Random(20260929)
        alphabet = ['"', "\\", ":", "[", "]", "{", "}", " ", ",", "\n", "a",
                    "K", '\\"', '": "', '"k": ["', "\t", " => ", "=>", "="]
        texts = ['{"password": "Kastanie2026"}', '"k" => "v"', '"a\\"b": "c"']
        texts += ["".join(rnd.choice(alphabet)
                          for _ in range(rnd.randint(0, 60)))
                  for _ in range(8_000)]
        for text in texts:
            swapped = text.translate(swap)
            self.assertEqual(self._members(text),
                             [(key.translate(swap), start, end)
                              for key, start, end in self._members(swapped)],
                             repr(text))

    def test_the_tokens_from_any_start_are_the_old_patterns(self):
        """_json_tokens(text, start) yields what finditer of the old pattern
        yields from ``start`` -- a caller that reads a text in pieces
        (a host application's windows over a large file) starts inside it, also
        inside a string. A host application read _JSON_TOKEN itself, and 0.1.64's
        pattern for it matches a string that never closes: the brackets
        inside it went unread and values of an array after a deep
        structure were judged by no key."""
        import random
        from sift.extract.base import _json_tokens
        rnd = random.Random(20260929)
        alphabet = ['"', "\\", ":", "[", "]", "{", "}", " ", ",", "\n", "a",
                    '\\"', '": "', '"k": ["', "\t"]
        texts = ['{"k": ["a\\"b", "c"]}', '"k": "\\', 'x "[{" ]', '"a\\"[b']
        texts += ["".join(rnd.choice(alphabet)
                          for _ in range(rnd.randint(0, 40)))
                  for _ in range(3_000)]
        for text in texts:
            for start in range(len(text) + 1):
                old = [(m.start(), m.end(), m.group(0))
                       for m in _OLD_TOKEN.finditer(text, start)]
                new = [(m.start(), m.end(), m.group(0))
                       for m, _token in _json_tokens(text, start)]
                self.assertEqual(old, new, (text, start))

    def test_tokens_start_at_the_beginning_by_default(self):
        from sift.extract.base import _json_tokens
        text = '{"k": ["v"]}'
        self.assertEqual([t for _m, t in _json_tokens(text)],
                         [t for _m, t in _json_tokens(text, 0)])

    def test_a_member_is_still_found_after_a_string_that_never_closes(self):
        text = 'note: 5" wide\n{"password": "Kastanie2026"}\n'
        found = [text[s.start:s.end] for s in find_secrets(text)]
        self.assertEqual(["Kastanie2026"], found)


if __name__ == "__main__":
    unittest.main()
