"""Shared types and helpers for extractors."""
from __future__ import annotations

import base64
import binascii
import bisect
import collections
import contextlib
import contextvars
import datetime
import ipaddress
import re
import unicodedata
import urllib.parse

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from .. import deadline


@dataclass
class Extracted:
    """Normalized result of pulling content out of a file.

    ``text`` is what we feed the LLM; ``metadata`` holds structured facts the
    heuristic summarizer and the UI can use directly. ``image_data`` carries raw
    bytes for vision models when the file *is* an image.
    """

    kind: str
    mime: str
    text: str = ""
    metadata: dict = field(default_factory=dict)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    image_data: Optional[bytes] = None
    image_mime: Optional[str] = None
    #: Metadata keys the rendered ``text`` already states in full. They stay in
    #: the report, where a machine reads them, but are left out of the prompt:
    #: sending the column profile as JSON *and* as a schema block spent a
    #: quarter of a small file's prompt saying the same thing twice.
    text_covers: tuple = ()
    #: Text that is in the file but deliberately not shown to the model — an
    #: HTML comment, a hidden cell. It is scanned for content addressed to an
    #: assistant, because that is a fact about the file worth reporting, and it
    #: is never put in a prompt: sending it to prove it exists would be the
    #: exposure the stripping prevents.
    hidden_text: str = ""
    #: True when the reader consulted the whole file regardless of
    #: ``--max-bytes``. A SQLite reader queries the file through sqlite3 and
    #: never honours a byte ceiling, so the generic note "read only the first
    #: 10 MB of 400 MB; figures describe that part" was false on every large
    #: database — the row counts are exact and describe all of it.
    reads_whole_file: bool = False
    #: The end of the file, when ``--max-bytes`` stopped the read before it.
    #: Never put in a prompt — the ceiling exists for a reason — but scanned,
    #: because what a reader is owed about a file does not stop at the
    #: ceiling.
    unread_tail: str = ""
    #: What the reader is sure of. See warn() — notes are for the rest.
    warnings: list = field(default_factory=list)
    #: True when ``text`` is sift's rendering of the file rather than its own
    #: bytes. A real model warned a user that settings.json "contains a leading
    #: '# JSON' comment which is invalid JSON" — that heading is sift's, and the
    #: file has no comment. The prompt has to say whose the layout is.
    rendered: bool = False
    #: A representative extract for content too large to send whole, built by
    #: the extractor that knows which parts matter. Where this exists it is
    #: better than map-reduce: a model reading 3,900 routine log lines to
    #: rediscover a severity count we already computed costs five minutes and
    #: produces a worse summary than one shown the errors directly.
    digest: Optional[str] = None
    #: The kind of finding (see summarize.FINDING_KINDS) each warning, and
    #: each note that is a finding, stands for — by its message. The summary
    #: turns them into records of Summary.findings.
    finding_kinds: dict = field(default_factory=dict)

    def note(self, message: str, kind: Optional[str] = None) -> None:
        """A note; with ``kind`` when it is a finding about the file rather
        than a remark about the reading."""
        if message and message not in self.notes:
            self.notes.append(message)
            if kind:
                self.finding_kinds[message] = kind

    def warn(self, message: str, kind: str) -> None:
        """For what the reader *measured*, as against what it suspects.

        Both volumes existed at the summary level and only one of them here,
        so an extractor that had compared two things and found them
        different could say so no louder than one that had a hunch. An
        attachment named .zip whose bytes are a PDF is not a hunch: the same
        contradiction one level up, about the file itself, has been a
        warning all along. Every warning is a finding, so it says its kind.
        """
        if message and message not in self.warnings:
            self.warnings.append(message)
            self.finding_kinds[message] = kind


@dataclass
class TablePage:
    """One page of a table's rows, for a reader paging through all of them.

    The cells are the file's own text: not redacted and not rendered. A
    caller that shows them to someone masks them itself.
    """

    kind: str                       # csv | tsv | xlsx | ods | office-legacy
    #: The sheets a reader can ask for, in file order; empty for csv/tsv.
    sheets: list = field(default_factory=list)
    sheet: Optional[str] = None
    #: Column names when the table's first row is a header, else empty.
    header: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    #: Where ``rows[0]`` sits among the data rows. The next page starts at
    #: ``offset + len(rows)`` — which may be short of the limit asked for,
    #: because a page also ends at its character budget.
    offset: int = 0
    #: Data rows in the table, or None when the page does not know it: the
    #: page stopped before the end of a table it does not hold whole
    #: (``has_more``: such a page reads the table up to the row after it,
    #: and no further; see read_table), counting ran out of time, or the
    #: file goes on past what sift reads (``truncated``).
    total: Optional[int] = None
    #: A row exists after this page.
    has_more: bool = False
    #: The table goes on past what sift reads (max_bytes, the member cap,
    #: a line the CSV reader cannot parse). Those rows are not reachable.
    truncated: bool = False
    #: Some row on this page was wider than the column cap.
    columns_truncated: bool = False
    #: Some cell on this page was cut to fit the page's character budget.
    cells_clipped: bool = False
    #: Lines above a CSV's table — a report's title — as the extractor
    #: sets them apart in its metadata.
    preamble: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    #: Why the file could not be read as a table at all.
    error: Optional[str] = None

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)


class CutRow(list):
    """A row that lost cells before fill_page saw it — wider than the reader
    keeps — so its length cannot say so. fill_page flags the page only when
    such a row is on it."""


def fill_page(page: TablePage, rows, *, header=(), offset: int, limit: int,
              max_columns: int, max_chars: int, count_rest=False) -> None:
    """Put data rows ``offset`` to ``offset + limit`` of ``rows`` on ``page``,
    under ``header``.

    A page ends at ``limit`` rows or at ``max_chars`` characters, header
    included, whichever comes first; a single row longer than the budget is
    cut to it, so one megabyte cell costs a page, not the process. The time
    budget in force applies throughout. A row before ``offset`` is counted,
    never looked at: a reader may hand a placeholder for it, and so for a
    row after the one that says there is another.

    ``rows`` is read up to the row after the page, which says there is one
    (``has_more``), and no further unless ``count_rest``: counting every row
    after the page for the total cost a 50 MiB csv 0.7 s and a 200,000-row
    sheet 2.4 s a page in 0.1.55, for a total that the viewer shows as a
    number. Without it a page that stops early has no total; the page that
    reaches the end has it. A reader that holds its rows whole, or reads
    them to the end anyway, counts the rest as 0.1.55 did (``count_rest``):
    running out of time then costs only the total -- the rows are already
    in hand, and discarding them for a count would be the wrong trade.
    """
    used = 0
    if header:
        page.columns_truncated = (len(header) > max_columns
                                  or isinstance(header, CutRow))
        page.header = list(header[:max_columns])
        # Half the page at most. A header clipped to the whole budget left
        # no room under it, and every row came back as a line of "…".
        if sum(len(cell) for cell in page.header) > max_chars // 2:
            page.header = _clip_cells(page.header, max_chars // 2)
            page.cells_clipped = True
        used = sum(len(cell) for cell in page.header)
    seen = 0
    rows = iter(rows)
    for row in rows:
        if not seen % 512:
            deadline.check("a table")
        seen += 1
        if seen <= offset:
            continue
        if len(page.rows) >= limit:
            page.has_more = True
            break
        wide = len(row) > max_columns or isinstance(row, CutRow)
        row = row[:max_columns]
        size = sum(len(cell) for cell in row)
        if page.rows and used + size > max_chars:
            page.has_more = True          # this row opens the next page
            break
        if size > max_chars - used:
            # The first row of the page, alone longer than the budget.
            row = _clip_cells(row, max_chars - used)
            size = sum(len(cell) for cell in row)
            page.cells_clipped = True
        page.columns_truncated |= wide
        page.rows.append(list(row))
        used += size
    if page.has_more and count_rest:
        try:
            rest = getattr(rows, "rest", None)
            if rest is not None:
                seen += rest()        # the reader counts them itself
            else:
                for _ in rows:
                    if not seen % 512:
                        deadline.check("the rest of a table")
                    seen += 1
        except deadline.DeadlineExceeded as exc:
            page.note(f"ran out of time counting the rows after this page "
                      f"({exc}); the page itself is complete")
            page.total = None
            return
        page.total = None if page.truncated else seen
        return
    page.total = None if page.truncated or page.has_more else seen


def _clip_cells(row: list, room: int) -> list:
    """The row with its cells cut so the whole fits ``room`` characters; a
    cut cell ends in "…" so it cannot pass for the whole value."""
    out = []
    for cell in row:
        if len(cell) > room:
            cell = cell[:max(room, 0)] + "…"
        room -= len(cell)
        out.append(cell)
    return out


def clip(text: str, max_chars: int) -> tuple[str, bool]:
    """Return (text, truncated). Cuts on a whitespace boundary when possible."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    cut = text.rfind("\n", 0, max_chars)
    if cut < max_chars * 0.6:
        cut = text.rfind(" ", 0, max_chars)
    if cut < max_chars * 0.6:
        cut = max_chars
    return text[:cut].rstrip() + "\n…[truncated]", True


def human_size(num_bytes: float) -> str:
    """A size as someone would write it.

    Rounding before the comparison, not after: 1048575 bytes is 1023.999 KB,
    which "%.1f" prints as "1024.0 KB" — a size nobody writes, and one that
    reads as larger than the megabyte it is just short of.
    """
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if unit == "B":
            if abs(num_bytes) < step:
                return f"{int(num_bytes)} {unit}"
        elif round(abs(num_bytes), 1) < step:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} PB"


def decode_text(data: bytes, cut_short: bool = False) -> tuple[str, str]:
    """Best-effort decode. Returns (text, encoding-name).

    Two traps here, both of which produce plausible-looking mojibake rather
    than an error.

    UTF-16 without a BOM. PowerShell's Out-File writes it; latin-1 accepts it
    and yields "n a m e" for "name".

    A multi-byte character cut in half by a byte cap. UTF-8 then fails for the
    whole buffer and cp1252 — which accepts almost anything — takes over, so a
    30 MB German document truncated at 25 MB has every umlaut mangled and
    reports its encoding as cp1252 with full confidence.

    ``cut_short`` says whether the caller actually truncated the data, and it
    is not optional politeness. b"caf\\xe9" is either latin-1 "café" or UTF-8
    "café…" cut mid-character, and nothing in the bytes distinguishes them.
    Trimming unconditionally silently drops the é from every complete latin-1
    file; only the caller knows which case this is.
    """
    from ..detect import utf16_encoding

    trims = (0, 1, 2, 3) if cut_short else (0,)
    wide = utf16_encoding(data[:4096])
    if wide:
        for trim in trims:
            try:
                return data[:len(data) - trim].decode(wide), wide
            except UnicodeDecodeError:
                continue
    for enc, name in (("utf-8-sig", "utf-8"), ("utf-8", "utf-8")):
        for trim in trims:
            try:
                return data[:len(data) - trim].decode(enc), name
            except UnicodeDecodeError:
                continue
    for enc in ("cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8/replace"

#: Where one word of a key ends and the next begins, beside no separator: a
#: camelCase hump -- userPin, pinCode, mTAN. Case-sensitive inside a
#: pattern that is not.
_WORD_HUMP = r"(?-i:(?<=[a-z])(?=[A-Z]))"
#: And where a word begins after an acronym -- APIPin, SIMPin: its first
#: capital before a small letter. Not where a word ends: base64 is letters
#: of both cases, and every RSA certificate's "...nTANBgkqhkiG9w0BAQ..."
#: ended "TAN" there (/etc/ca-certificates, 8 lines).
_ACRONYM_END = r"(?-i:(?<=[A-Z])(?=[A-Z][a-z]))"
#: PIN and TAN as a key or a sentence writes them: a word of its own, at the
#: start or the end of the key, beside a separator, or at a hump
#: (_WORD_HUMP, _ACRONYM_END) -- PIN, user_pin, userPin, pinCode,
#: PIN_NUMBER, pins, TAN_LIST, mTAN, pushTAN, TANs --, each part in one
#: case, as a word is written: "MTAn" in a certificate's "...iMTAnMAoG..."
#: is base64. The TAN procedures of German banks are one word written
#: together too: itan, etan, mtan, smstan, pushtan, chiptan, phototan.
#: Three letters with no boundary in front are a part of every other word:
#: Pakistan, Afghanistan, Titan and sultan end in "tan", Philippines,
#: alpine and spine hold "pin" and the "e" and "es" the longer names take
#: -- /usr/lib/python3.14's tables of countries, 'Pakistan': 'PK' and
#: 'Philippines': 'PH', were passwords to the strong tier once quoted keys
#: were read, and "pin" in "spinner", "mapping" and "typing"
#: named a secret to the sentence rule. A plural is "pins" and "tans", no
#: "pines". Written together with another word in one case -- USERPIN,
#: adminpin -- a PIN is lost: nothing tells it from SPIN, ALPINE or opine.
_PIN_OR_TAN = (
    rf"(?:(?<![a-z0-9])|{_WORD_HUMP}|{_ACRONYM_END})"
    r"(?-i:(?:pin|Pin|PIN)(?:code|Code|CODE)?"
    r"|(?:[iemIEM]|sms|Sms|SMS|push|Push|PUSH|chip|Chip|CHIP"
    r"|photo|Photo|PHOTO)?(?:tan|Tan|TAN))s?"
    rf"(?:(?![a-z0-9])|{_WORD_HUMP})")


def _key_word(*words: str, then: str = "") -> str:
    """A short name of a secret as _PIN_OR_TAN reads PIN and TAN: a word of
    its own -- at the start or the end of the key, beside a separator, at a
    hump --, written in one case (otp, Otp, OTP), ``then`` (a pattern)
    after it, a plural "s". Each of ``words`` in lower case."""
    cased = "|".join(f"{word}|{word.capitalize()}|{word.upper()}"
                     for word in words)
    return (rf"(?:(?<![a-z0-9])|{_WORD_HUMP}|{_ACRONYM_END})"
            rf"(?-i:(?:{cased}){then})s?(?:(?![a-z0-9])|{_WORD_HUMP})")


#: The short names of the strong tier, as words of their own (_key_word):
#: a one-time password, MFA, a pre-shared key, a seed, a DSN. With no
#: boundary in front, linseed, aniseed and rapeseed were a seed, cropsk a
#: pre-shared key and odsn a DSN, and every value under them a password.
#: Written together with another word in one case -- WIFIPSK, RANDOMSEED
#: -- the name is lost, as USERPIN is.
_SHORT_SECRET_NAMES = _key_word("otp", "totp", "mfa", "psk", "seed", "dsn")
#: A one-time password in a sentence (_SECRET_IN_SENTENCE), a word of its
#: own as well: "otp" with no boundary at all read "footprint: 4096" and
#: "spotprice = 42.10" as a secret's name and value.
_OTP = _key_word("otp", "totp", then="(?:code|Code|CODE)?")

#: Key names whose value must not be repeated in a summary. The summary
#: travels further than the file does: into --json, the on-disk cache, a
#: markdown report someone pastes into a ticket, and a browser card. The
#: file's contents going to a local model is one thing; a live Stripe key
#: landing in a bug report is another.
#: Key names that settle it on their own. Under one of these, the value is a
#: credential whatever it looks like: a passphrase has spaces, a German
#: password has an umlaut, a PIN is four digits, and every one of those was
#: surviving a test that demanded one opaque ASCII token.
_NAMES_A_SECRET = re.compile(
    # No boundary in front. Requiring one lost every run-together name people
    # actually use — PGPASSWORD, MYSQLPASSWORD, dbpassword, userPassword,
    # BenutzerPasswort — and under IGNORECASE "[^a-z0-9]" excludes capitals
    # too, so camelCase failed as well. It is safe to drop precisely because
    # the value still has to survive _clearly_not_a_secret: "monkey" reaches
    # the weak tier and "CHARS_PER_TOKEN = 4" is a count.
    # A name of two English words is written with one blank as well as a
    # separator or nothing -- "API Key" as a column, 'api key' as a quoted
    # key, "api key" in a sentence: api_key named a secret and
    # "api key" nothing, in every list below (SECRETISH, _ABOUT_A_SECRET,
    # _SECRET_IN_SENTENCE).
    r"(?:(?:"
    r"pass(?:word|wd|phrase|code)?|pwd|passw|secret|token|api[_.\- ]?keys?"
    r"|credentials?|private[_.\- ]?keys?|access[_.\- ]?keys?"
    r"|bindpw|webhook"
    r"|connection[_.\- ]?string|database[_.\- ]?url"
    r"|client[_.\- ]?secret|auth[_.\- ]?token|bearer[_.\- ]?token"
    # German, because this tool is pointed at German files.
    r"|pass?wort|kennwort|geheim(?:nis|schl(?:ue|ü)ssel)"
    r"|zugangsdaten|anmeldedaten"
    r")e?s?(?:$|[^a-z0-9])"
    # …except the short words, which need a boundary at each end
    # (_PIN_OR_TAN, _SHORT_SECRET_NAMES).
    r"|" + _PIN_OR_TAN + r"|" + _SHORT_SECRET_NAMES + r")", re.IGNORECASE)

#: Key names that only *hint*. A key can be musical, a salt can be a recipe,
#: a signature can be a name at the foot of a letter, a session can be a
#: meeting, a cookie can be a biscuit. Under one of these the value still has
#: to look like a credential.
SECRETISH = re.compile(
    # Two groups, because the words differ in how much they mean on their
    # own. A long unambiguous one may sit anywhere in the name — "dbpassword",
    # "PGPASSWORD", "BenutzerPasswort" are how people write these, and
    # requiring a boundary in front lost every one of them.
    r"(?:pass(?:word|wd|phrase|code)|pwd|passw|api[_.\- ]?keys?|credentials?"
    # "secret" and "token" belong here and not below: they are long enough to
    # mean one thing, and "secretmin" and "passwordUpper" are key names
    # people write. A trailing boundary lost both.
    r"|secret|token"
    r"|keystore|truststore|session[_.\- ]?id"
    r"|pass?wort|kennwort|geheim(?:nis)?|schl(?:ue|\u00fc)ssel"
    r"|zugangsdaten|anmeldedaten"
    # …and the short ones, which mean too many other things without a
    # boundary at each end: "key" matched inside "monkey" and "keyboard",
    # and destroyed "keyboard_layout: de" and "monkey_patch: TrueValue".
    r"|(?<![a-z])(?:key|auth|pin|[ie]?tan|pw|otp|sid|salt|cert|login|bearer"
    r"|private|signature|session|cookie|pass)s?(?![a-z])"
    r")", re.IGNORECASE)
# A word of _KEY_HEADS glued to "key" (MASTERKEY, masterKey) hints as
# "key" does: _glued_key, beside SECRETISH wherever a hint is asked for.

#: The name/value pair, where the key is not on the value's line at all.
#:
#:     env:
#:       - name: DB_PASSWORD
#:         value: Herbstlaub2026!
#:
#: Kubernetes writes environment variables this way, and so do Docker
#: Compose, GitHub Actions, CloudFormation and OpenAPI. The key beside the
#: value is the literal word "value", which names nothing, so both tiers
#: looked at it and found nothing — and a real model handed the password
#: back as an *entity*, stripped of the two lines that made it recognisable.
#:
#: Three groups: everything to keep, the name to judge by, and the value.
#: The name is not tested here — it goes to looks_like_a_credential, which
#: is where that judgement lives and already knows both tiers.
_NAME_VALUE_PAIR = re.compile(
    r"([\"']?names?[\"']?\s{0,4}[:=]\s{0,4}[\"']?([\w.\-]{1,64})[\"']?"
    r"\s{0,4},?\s{0,4}(?:\r?\n|\u23ce)?\s{0,12}(?:-\s{1,4})?"
    r"[\"']?values?[\"']?"
    r"\s{0,4}[:=]\s{0,4}[\"']?)([^\"'\n]{1,200})",
    re.IGNORECASE)

#: A name in quotes, as a key writes one: 'k' or "k", a backslash escaping
#: the character after it. No ":" or "=" in it, so the first of those on
#: the line is the separator and a lookahead can bound the key's length
#: (_ASSIGNMENT_SHAPE).
_QUOTED_NAME = (r"'(?:[^\n'\\:=]|\\[^\n:=])+'"
                r'|"(?:[^\n"\\:=]|\\[^\n:=])+"')
#: An assignment, in the shapes a config or environment file writes one.
#: The key's length is a parameter: the preview's redact_assignments takes
#: keys up to 64 characters, and find_secrets takes longer ones, because a
#: YAML file is parsed before scrub sees it and the parsed key has no limit.
#:
#: Three groups: the key as the line writes it, blanks and "export" before
#: it included; the separator; the rest of the line. The key is a bare
#: name -- [\w.\-\[\]] --, or a name in quotes, or one subscripted with a
#: name in quotes: 'PASSWORD', "password", ['password'],
#: os.environ['DB_PASSWORD'], $db['password'], a['b']["c"]; the separator
#: ":", "=" or "=>". The bare key alone, a quote holding no name, missed
#: the password of every Django settings.py, a PHP array, a Ruby hash and
#: os.environ['DB_PASSWORD'] = '...'. A quoted key
#: is at most ``longest`` characters too, with the blanks before its
#: separator: a host application's viewer reaches back that far for a key. Its first
#: character is looked at before that length is: a line of 30,000 blanks
#: took the lookahead once for every blank the indent gave back -- 1.1 s
#: (test_backtracking).
_ASSIGNMENT_SHAPE = (
    r"^([ \t]*(?:export[ \t]+)?(?:[\w.\-\[\]]{{1,{longest}}}"
    r"|(?=[$\['\"\w])(?=[^\n:=]{{1,{longest}}}(?:=>|[:=]))(?:" + _QUOTED_NAME
    + r"|\$?[\w.\-\[\]]{{0,{longest}}}(?:\[(?:" + _QUOTED_NAME + r")\])+)))"
    r"[ \t]*(=>|[:=](?!>))[ \t]*(\S.*)$")
_ASSIGNMENT = re.compile(_ASSIGNMENT_SHAPE.format(longest=64))
#: A string in quotes, single or double, a backslash escaping the character
#: after it: its text is group 1 or group 2.
_QUOTED_STRING = re.compile(r"'((?:[^\n'\\]|\\.)*)'|\"((?:[^\n\"\\]|\\.)*)\"")


def _key_name(key: str) -> str:
    """The name a key read off a line (_ASSIGNMENT_SHAPE) is judged by: the
    name in its quotes, the last where it has several -- 'PASSWORD' is
    PASSWORD, os.environ['DB_PASSWORD'] is DB_PASSWORD and a['b']['c'] is
    c, as scrub judges a parsed file's value by its own key --, else the
    key as the line writes it. Read with its quotes, "'password_hash'"
    ends in no suffix _names_the_secret_not_the_value knows."""
    last = None
    for last in _QUOTED_STRING.finditer(key):
        pass
    if last is None:
        return key
    return last.group(1) if last.group(1) is not None else last.group(2)


def _assigned_value(match) -> "Optional[tuple[int, int]]":
    """Where the value of an assignment line (a match of _ASSIGNMENT or
    _KEYED_LINE) is, as the key reader judges it: the rest of the line after
    a bare key, as it always was. After a quoted key -- a dict's, an
    array's, a JSON member's own line -- the text between the quotes of a
    value in quotes, as the member rule masks a JSON member's; else the
    rest without the "," or ";" that ends an item, where that is a literal
    (_a_literal), and None -- no value to judge -- where it is an
    expression of the code the key is in. Read as the rest of the line,
    '"password": "",' was the secret '",' and '"token": null,' the secret
    "null,"; and the quotes and the comma of a pretty-printed JSON file
    were masked, where the member rule masks the string's text."""
    start, end = match.span(3)
    if _QUOTED_STRING.search(match.group(1)) is None:
        return start, end
    string = _QUOTED_STRING.match(match.string, start, end)
    if string is not None:
        inside = 1 if string.group(1) is not None else 2
        return string.span(inside)
    rest = match.string[start:end].rstrip().rstrip(",;").rstrip()
    return (start, start + len(rest)) if _a_literal(rest) else None


#: What makes a value in code an expression rather than a literal: a blank,
#: a call's or a subscript's bracket, a collection's, a quote; a
#: variable's sigil in front.
_EXPRESSION = re.compile(r"[\s()\[\]{}\"'`]|^[$@&*]")
#: A name as code writes one, dotted or not: password, self.password,
#: settings.DB_PASSWORD, None.
_A_NAME = re.compile(r"[A-Za-z_][\w.]*")
#: A number: 4096, -1, 1.5.
_A_NUMBER = re.compile(r"[+-]?[\d.]+")


def _a_literal(value: str) -> bool:
    """Whether a value written without quotes after a key in quotes is a
    literal -- YAML's plain Kx9vQ2mZ7pL4 or Sommer24! -- rather than an
    expression of the code the key is in: 'password': password,
    'PASSWORD': os.environ['DB_PASSWORD'], 'password' => env('DB_PASSWORD',
    ''), "access_token": {"required": False}, 'password': KeyInfo(),
    'pass_stmt': tree.KeywordStatement. One token, no bracket, quote or
    sigil in it, no number -- the member rule reads strings only, and a
    pretty-printed JSON file's "key_tag": 18948 stays what it was --, and
    either a character no name is written with or, in a name, a run mixing
    letters and digits (_opaque_run's second kind: a name of sixteen
    letters is a name). Judged as values, the dict literals of the ansible
    collections in /usr/lib/python3 had 3,483 values masked, and
    "KeyInfo()" wherever else it stood: 13,803 repetitions."""
    if not value or _EXPRESSION.search(value) or _A_NUMBER.fullmatch(value):
        return False
    if not _A_NAME.fullmatch(value):
        return True
    return any(len(run) >= 8 and not run.isdigit() and not run.isalpha()
               for run in _TOKEN_RUN.findall(value))


#: The longest value the sentence rule reads, in characters
#: (_SECRET_IN_SENTENCE).
_SENTENCE_VALUE = 120
#: Whitespace that spaces two characters without ending a line.
_BLANKS = "[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
#: A line break as sift writes one it folds out of a value it shows on one
#: line -- " \u23ce " (one_line; the mail extractor's header values) -- is
#: the line break it stands for to the rules below, wherever they read one
#: between a keyword and its value. An encoded-word's line break in a
#: Subject reached every rule as "password: \u23ce Kx9vQ2mZ7pL4", which none
#: read as the "password:\nKx9vQ2mZ7pL4" it is, and the summary published
#: the password the same Subject on one line had hidden. In
#: the rules' whitespace between a keyword and its value, never inside a
#: value: a value ends at the blank before the mark, as at the break.
_SEPARATING = "[\\s\u23ce]"
#: Whitespace inside a value that does not end it: a tab, a no-break or an
#: em space, a run of blanks -- anything but the one plain blank prose puts
#: between two words, and nothing after a sentence's closing mark
#: ("Winterreise2026!  Bitte ändern." is two sentences). Up to eight, which
#: is two tabs as one_line writes them; a longer run is a table's alignment.
#: The rules that read a whole line (redact_assignments, find_secrets'
#: keyed lines) take the rest of it; the ones that read a value inside
#: text ended it at the first whitespace of any kind, and
#: "password: Kx9v<TAB>Q2mZ7pL4Wq8rTz" had its first four characters hidden
#: and the other fourteen published.
#: Never into a piece holding "<": an arrow comment ("Kx9vQ2mZ7pL4  <--
#: change this"), a tag, a marker another rule left behind. Run on into
#: one, the sentence rule's value held a "<", its guard for a value someone
#: already hid rejected the whole, and the password in front was published
#:. A piece is looked into no further than a sentence's
#: value reaches (_SENTENCE_VALUE).
_INNER_GAP = (rf"(?<![.!?])(?! (?!{_BLANKS})){_BLANKS}{{1,8}}"
              rf"(?![^\s<]{{0,{_SENTENCE_VALUE - 2}}}<)")


#: The other ways a file sets a credential, none of which is "key = value".
#: wp-config.php writes define('DB_PASSWORD', '…'), MySQL grants say
#: IDENTIFIED BY '…', redis.conf says requirepass …, and a shell script
#: passes curl -u user:pass. Each was published in full because the line did
#: not look like an assignment — a whole wp-config.php, verbatim, in the
#: preview.
#:
#: Two groups each: what to keep, and what to hide.
_OTHER_ASSIGNMENTS = [
    # define('DB_PASSWORD', 'value')  /  putenv("PASS=value")
    re.compile(r"((?:define|putenv|setenv|set_config)\s*\(\s*[\'\"]?"
               r"[\w.\-]*(?:pass(?:word|wd)?|secret|token|key|pwd)[\w.\-]*"
               r"[\'\"]?\s*,\s*[\'\"])([^\'\"]{1,200})", re.IGNORECASE),
    # IDENTIFIED BY 'value'  /  PASSWORD 'value'
    re.compile(r"((?:identified\s+by|encrypted\s+password|password)\s+[\'\"])"
               r"([^\'\"]{1,200})", re.IGNORECASE),
    # requirepass value  /  masterauth value  /  auth_pass value
    re.compile(r"^([ \t]*(?:requirepass|masterauth|auth_pass|user_pass"
               r"|proxy_password)[ \t]+)"
               rf"(\S(?:\S|{_INNER_GAP}(?=\S)){{0,199}})",
               re.IGNORECASE | re.MULTILINE),
    # curl -u user:password  /  --user user:password
    re.compile(r"((?:-u|--user)[ \t]+[^\s:]{1,64}:)(\S{1,200})"),
    # Authorization: Basic <base64>  — the credential is the base64
    re.compile(r"((?:authorization|proxy-authorization)\s*:\s*basic"
               rf"{_SEPARATING}+)"
               r"([A-Za-z0-9+/=]{8,})", re.IGNORECASE),
]


#: A password handed to a command on its command line: mysql's -p<password>, sshpass -p, redis-cli -a, openssl's
#: pass:<password>, keytool's -storepass and -keypass, htpasswd -b's last
#: argument, and in code the second of auth=(user, password) and of
#: .login(user, password). Only where the flag or the position names a
#: password: -p is a port to ssh, "parents" to mkdir and "preserve" to tar,
#: and -a means "all" to most commands, so each flag is read after its own
#: command. A value the rule finds is judged by _command_password.
#:
#: Each with the words one of its matches holds, at least one of them: a
#: text with none is not scanned by these (_stages).
#:
#: What stands between two arguments: blanks, and a line a backslash
#: continues, as a long command in a script is written.
_ARG_GAP = r"(?:[ \t]|\\\r?\n){1,16}"
#: An argument the rule does not hide: in quotes, or bare.
_ARG_SHAPE = r"(?:'[^'\n]{1,200}'|\"[^\"\n]{1,200}\"|[^\s'\"|;&<>]\S{0,199})"
#: The argument that is the password: the text between its quotes, or all
#: of it up to the next blank (single, double, bare).
_PASSWORD_ARG = (r"(?:'(?P<single>[^'\n]{1,200})'"
                 r"|\"(?P<double>[^\"\n]{1,200})\""
                 r"|(?P<bare>[^\s'\"|;&<>]\S{0,199}))")
#: The same, in quotes only: in code a bare name is a variable.
_QUOTED_PASSWORD = (r"(?:'(?P<single>[^'\n]{1,200})'"
                    r"|\"(?P<double>[^\"\n]{1,200})\")")
#: The arguments of a command before the flag looked for: twenty at most,
#: and none past a pipe, a ";", a "&" or a redirection.
_ARGS_BEFORE = rf"(?:{_ARG_GAP}[^\s|;&<>]\S*){{0,20}}?"
#: A user as code passes one: a string that is no path or URL, or a name.
_A_USER = r"(?:'[^'\n/]{1,64}'|\"[^\"\n/]{1,64}\"|[A-Za-z_][\w.]{0,64})"
#: htpasswd's options: letters, and -C's cost.
_HTPASSWD_FLAGS = (rf"(?P<flags>(?:{_ARG_GAP}-[A-Za-z]{{1,8}}"
                   rf"(?:{_ARG_GAP}\d{{1,4}})?){{1,6}})")
_COMMAND_LINE_PASSWORDS = [
    # mysqldump -u root -pKx9v... app: glued to -p, as MySQL's clients read
    # it; "-p app" asks for the password and names the database.
    (re.compile(r"(?<![\w-])(?:mysql|mysqldump|mysqladmin|mysqlimport"
                r"|mysqlcheck|mysqlshow|mysqlpump|mysqlslap"
                r"|mariadb(?:-dump|-admin|-import|-check|-show|-slap)?)"
                rf"{_ARGS_BEFORE}{_ARG_GAP}-p{_PASSWORD_ARG}"), "any",
     ("mysql", "mariadb")),
    # sshpass -p Kx9v... ssh host, glued or not, after sshpass's other
    # options (-P takes an argument).
    (re.compile(rf"(?<![\w-])sshpass(?:{_ARG_GAP}-[^p\s]\S*"
                rf"(?:{_ARG_GAP}[^-\s]\S*)?){{0,4}}{_ARG_GAP}-p[ \t]*"
                rf"{_PASSWORD_ARG}"), "any", ("sshpass",)),
    # redis-cli -h cache -a Kx9v... ping  /  --pass Kx9v...
    (re.compile(rf"(?<![\w-])redis-cli{_ARGS_BEFORE}{_ARG_GAP}(?:-a|--pass)"
                rf"{_ARG_GAP}{_PASSWORD_ARG}"), "any", ("redis-cli",)),
    # openssl ... -passout pass:Kx9v...  /  -passin 'pass:Kx9v...': only
    # "pass:" holds the password itself -- env:, file:, fd: and stdin say
    # where it is.
    (re.compile(r"(?<![\w-])-(?:passin|passout|pass|password)[ \t]+"
                r"(?:'pass:(?P<single>[^'\n]{1,200})'"
                r"|\"pass:(?P<double>[^\"\n]{1,200})\""
                r"|pass:(?P<bare>[^\s'\"]\S{0,199}))"), "any", ("pass:",)),
    # keytool / jarsigner -storepass Kx9v...  /  -keypass, -srcstorepass,
    # -deststorepass, -srckeypass, -destkeypass. -storepass:env and
    # -storepass:file say where it is.
    (re.compile(rf"(?<![\w-])-(?:src|dest)?(?:store|key)pass{_ARG_GAP}"
                rf"{_PASSWORD_ARG}"), "any", ("storepass", "keypass")),
    # htpasswd -nb user Kx9v...: batch mode, no file, the second argument.
    (re.compile(rf"(?<![\w-])htpasswd{_HTPASSWD_FLAGS}{_ARG_GAP}{_ARG_SHAPE}"
                rf"{_ARG_GAP}{_PASSWORD_ARG}"), "htpasswd -n", ("htpasswd",)),
    # htpasswd -b /etc/nginx/.htpasswd admin Kx9v...: the third.
    (re.compile(rf"(?<![\w-])htpasswd{_HTPASSWD_FLAGS}{_ARG_GAP}{_ARG_SHAPE}"
                rf"{_ARG_GAP}{_ARG_SHAPE}{_ARG_GAP}{_PASSWORD_ARG}"),
     "htpasswd", ("htpasswd",)),
    # requests.get(url, auth=('admin', 'Kx9v...')), session.auth = (...)
    (re.compile(rf"\bauth\s*=\s*\(\s*{_A_USER}\s*,\s*{_QUOTED_PASSWORD}"
                r"\s*\)"), "any", ("auth",)),
    # server.login('ops@example.org', 'Kx9v...') -- smtplib, imaplib,
    # ftplib and their kin.
    (re.compile(rf"\.login\s*\(\s*{_A_USER}\s*,\s*{_QUOTED_PASSWORD}\s*\)"),
     "any", (".login",)),
]


#: Values that are a credential whatever sits beside them. Only shapes that
#: nothing else has: a service's own key prefix, a PEM header, a JWT, a URL
#: carrying a password. Deliberately *not* bare hex or bare base64 — a git
#: SHA, an MD5 checksum, a product code and a DNA sequence all match those,
#: and a checksum column is exactly what a data CSV has.
_CREDENTIAL_SHAPE = re.compile(
    r"^(?:sk|pk|rk)[-_](?:live|test|prod)[-_][A-Za-z0-9]{8,}"
    r"|^(?:sk|pk|rk|tok)[-_](?=[-_A-Za-z0-9]*\d)(?=[-_A-Za-z0-9]*[A-Z])[-_A-Za-z0-9]{16,}"
    r"|^(?:ghp|gho|ghs|ghu|ghr|glpat|xox[bpsaor])[-_][-_A-Za-z0-9]{8,}"
    r"|^github_pat_[A-Za-z0-9_]{20,}"
    r"|^SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"
    r"|^\$(?:2[aby]|apr1|1|5|6)\$[./A-Za-z0-9$]{12,}"     # htpasswd / crypt
    r"|^(?:npm_|dop_v1_|dockerhub_|shpss_|shpat_)[A-Za-z0-9]{16,}"
    r"|^(?:AKIA|ASIA)[A-Z0-9]{12,}$"
    r"|^(?:AIza|ya29\.)[-_A-Za-z0-9]{10,}"
    r"|^ey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|^[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"
    r"|AccountKey=[A-Za-z0-9+/=]{16,}"
    r"|^Bearer\s+[A-Za-z0-9._-]{16,}$", re.IGNORECASE)

#: What a value has to look like under a key that only *hints*. "private:
#: true" is a flag, "key: E-flat major" is a musical key, "salt: 10 g" is a
#: recipe, and `session = requests.Session()` is code.
_OPAQUE = re.compile(r"^[!-~\u00a0-\uffff]{5,}$")

#: Values that are not a credential under any key name. A flag, an algorithm,
#: a scheme, a placeholder, a reference to where the secret actually lives.
_NOT_A_SECRET = frozenset({
    "true", "false", "null", "none", "nil", "yes", "no", "on", "off", "ja",
    "nein", "required", "optional", "auto", "default", "disabled", "enabled",
    "rsa", "dsa", "hmac", "ed25519", "ecdsa", "sha256", "sha512", "sha1",
    "md5", "bcrypt", "scrypt", "argon2", "pbkdf2", "aes", "aes256", "gcm",
    "bearer", "basic", "digest", "plain", "login", "oauth", "oauth2", "jwt",
    "ntlm", "kerberos", "saml", "oidc", "tls", "ssl", "none.",
    "changeme", "todo", "tbd", "xxx", "redacted", "hidden", "unset",
    "anonymous", "guest",           # the FTP and SMB conventions
    "n/a", "na", "-", "<none>", "***", "********"})

#: A value that names where the secret is, rather than being it: an
#: environment variable, a placeholder, a vault path, a file path.
_A_REFERENCE = re.compile(
    r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$"           # $FOO, ${FOO}
    r"|^%[A-Za-z_][A-Za-z0-9_]*%$"                 # %FOO%
    # An underscore required: "^[A-Z][A-Z0-9_]{2,}$" read every all-capital
    # value as an environment variable, so "GEHEIM2024" and "SOMMER2024GEHEIM"
    # were published as references to themselves.
    r"|^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$"            # SIFT_API_KEY
    r"|^<[^>]{1,40}>$"                             # <your-key-here>
    r"|^\{\{[^}]{1,60}\}\}$"                      # {{ secrets.KEY }}
    r"|^(?:[/~]|\.{1,2}/)[^\s]{1,120}$"            # /etc/ssl/private/x.pem
    r"|^[a-z][a-z0-9+.-]*://[^/\s:@]+(?::\d+)?(?:/|$)")  # a URL with no user


#: Suffixes that turn a secret's name into a name *for* the secret.
#: "secret_manager" is a service, "api_key_env" is the variable holding one,
#: "password_hash" is a hash, "token_expiry" is a date, "pin_required" is a
#: flag. Under those the value is configuration, not a credential.
_ABOUT_A_SECRET = re.compile(
    # A separator before the suffix. Anchored to "$" alone, three characters
    # bought an exemption from both tiers — "secretmin" was a "min" — and
    # "keystore" and "truststore" exempted themselves by their own last five
    # letters while being named in SECRETISH. A blank is a separator too
    #: "Token Expiry" and "API Key Name" name a column as
    # token_expiry and api_key_name name a key.
    r"[_.\- ]"
    # Deliberately not "id": a session_id and a token_id are live
    # credentials, whatever the grammar of the name suggests.
    # "url" and "uri" are gone: a webhook URL *is* the credential, and the
    # strong tier already declines to treat a URL as a reference. A
    # "private_notes_url" survives on the weak tier, where it does.
    # So are the ones that name a *place* a secret is used rather than
    # metadata about it — field, label, prompt, header, param, type, store —
    # which between them exempted twelve of fourteen real credential keys.
    r"(?:manager|name|names|arn|path|file|filename|env|var"
    r"|ref|reference|algorithm|algo"
    r"|required|enabled|disabled|length|len|min|max|expiry|expires|ttl"
    r"|budget|limit|size|quota|window|total|sum|threshold|interval|retries"
    r"|rotation|policy|hash|hashed|digest|hint|column"
    r"|count|age|days|format|location"
    # Time units, because "token_ttl_seconds" is how a name says how long a
    # token lives. "ttl" alone reached the ones that end there and no others.
    r"|seconds|secs|millis|minutes|mins|hours|weeks|months"
    r"|strength|complexity|attempts|last_changed|changed_at|created_at"
    # Nothing shorter than three letters: "_at", ".on" and "-by" bought an
    # exemption from both tiers for three characters, which is the same
    # price the previous audit put on this same list.
    r"|date|datum|time|timestamp|owner|user|subject|issuer|serial"
    r"|version|status|state|mode|note|notes|comment|description|title)$",
    re.IGNORECASE)


#: Words that, in front of a secret's name, say the value is a count or a
#: setting about it: max_tokens, num_tokens, context_tokens, stop_token,
#: pad_token. A suffix list alone could not reach these — the metadata word
#: is on the other side.
_ABOUT_A_SECRET_PREFIX = re.compile(
    r"^(?:max|min|num|no|n|total|count|context|stop|pad|unk|bos|eos|sep|cls"
    r"|mask|default|initial|target|expected|chars?|words?|bytes?"
    r"|anzahl|maximal|minimal)[_.\- ]", re.IGNORECASE)


def _names_the_secret_not_the_value(key: str) -> bool:
    name = str(key).strip().rstrip("_.-")
    return bool(_ABOUT_A_SECRET.search(name)
                or _ABOUT_A_SECRET_PREFIX.match(name))


#: The words of a key, a camelCase one split too: "masterKey" is "master"
#: and "Key", "APIKey" is "API" and "Key", "MASTERKEY" is one word.
_KEY_WORDS = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")
#: The words that name a key with "key" after them, whether glued, or a
#: separator or one blank between: MASTERKEY, masterKey, SigningKey, "api
#: key", "master key". SECRETISH's "key" wants a separator on each side,
#: so a glued one named a secret to the settings rule alone (_secret_key)
#: and an assignment line and a sentence read MASTERKEY as no key.
#: A list of the heads and not every word ending in "key": monkey, hockey,
#: hotkey and turnkey name none, and "hotkey: Ctrl+Shift+K" is no secret.
_KEY_HEADS = frozenset({
    "api", "secret", "private", "access", "master", "auth", "encryption",
    "signing", "session", "license", "app", "ssh", "hmac", "jwt", "service",
    "account"})
#: The same as a pattern: the head, then "key" glued or one blank or a
#: separator away, a plural "s".
_KEY_WORD_SHAPE = (r"(?:" + "|".join(sorted(_KEY_HEADS)) + r")[_.\- ]?keys?")


def _glued_key(key: str) -> bool:
    """Whether a word of ``key`` (_KEY_WORDS) is a head of _KEY_HEADS glued
    to "key" -- MASTERKEY, masterkey --, or "key" after such a head where
    the case splits the two -- masterKey, APIKey, SigningKey. A key that
    only hints: the value has to look like a credential under it, as
    under SECRETISH's "key"."""
    words = [word.lower() for word in _KEY_WORDS.findall(key)]
    for before, word in zip(("", *words), words):
        if word.endswith("keys"):
            word = word[:-1]
        if word == "key":
            word = before + word
        if word.endswith("key") and word[:-3] in _KEY_HEADS:
            return True
    return False


def _is_a_plain_word(text: str) -> bool:
    """One run of letters in one case — "primary", "kosher", "Sonnenschein".

    Under a key that only hints, this is what tells a musical key from a
    cryptographic one. A password that is a single lowercase dictionary word
    is a weak password; a password field is why the *strong* tier exists,
    and it does not consult this.
    """
    # Words joined by underscores or hyphens count as one word each:
    # "rechnung_id" under a key called "foreign_key" is a column name, and
    # "aws-secretsmanager" is a service. A credential does not read as a
    # sequence of words.
    # Bracketed tokens are labels: "[PAD]", "&lt;EOS&gt;", "{{ default }}".
    stripped = text.strip("[]<>{}()")
    if stripped and stripped != text:
        text = stripped
    parts = re.split(r"[_\-]", text)
    if not all(re.fullmatch(r"[A-Za-z\u00c0-\u024f]{1,}", part) for part in parts):
        return False
    # CamelCase counts as words: "LdapBackend", "BertTokenizer",
    # "TimesNewRoman", "DigiCert" are identifiers a configuration file is
    # full of, and requiring one case throughout called every one of them a
    # credential. A password with no digit and no symbol in it is a weak
    # password, and this tier is the one where the key only hints.
    def _words(part: str) -> bool:
        if part.islower() or part.isupper() or part.istitle():
            return True
        return bool(re.fullmatch(r"(?:[A-Z][a-z]+){2,}", part))
    return all(_words(part) for part in parts)


#: An all-capital name. Under a hint it is an environment variable being
#: pointed at; under a key that says "password" it is the password —
#: "SOMMER_2024" was published because it has an underscore in it, and
#: "GEHEIM2024" was hidden because it does not, which is not a distinction
#: anybody's password policy makes.
_ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: A URL naming a resource. Under a hint this is documentation; under a key
#: that names a secret it is the credential.
_IS_A_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)

#: A call or a subscript on an identifier — `requests.Session()`,
#: `parsed["key_points"]`, `os.environ.get('X')`. Not merely a bracket
#: somewhere in the string.
_LOOKS_LIKE_CODE = re.compile(
    r"^[A-Za-z_][\w.]*\s*[(\[][^)\]]*[)\]]$"
    r"|^[A-Za-z_][\w.]*\s*=\s*[A-Za-z_][\w.]*\s*\(")


def _clearly_not_a_secret(text: str, strong: bool = False) -> bool:
    """Whether a value under a secret-sounding key is plainly something else.

    Under a key that names a secret outright — password, passwd, token,
    Kennwort — the default is to hide the value, because the shapes a real
    password takes are unbounded: "correct horse battery staple" has spaces,
    "Sömmerung99" is not ASCII, a PIN is four digits. Demanding one opaque
    six-character ASCII token let all three through, and those are what
    people actually use.

    So the question is turned round: what is *demonstrably* not a secret. A
    flag, an algorithm name, a placeholder, a reference to where the secret
    really lives, a number small enough to be a length or a count, and code.
    """
    lowered = text.lower()
    if lowered in _NOT_A_SECRET:
        return True
    if _A_REFERENCE.match(text) and not (strong and (
            _IS_A_URL.match(text) or _ALL_CAPS.match(text)
            or _reads_as_itself(text))):
        # A URL is a reference to a resource — unless the key says it is a
        # credential, and then it is one: a Slack webhook URL *is* the
        # secret, and so is a signed S3 link. "webhook_url" was exempted by
        # the branch meant for "private_notes_url".
        # So is "$ADMIN2024" under a key that says password: python-dotenv
        # reads a bare $NAME as itself (_reads_as_itself).
        return True
    names, core = _defaulted(text)
    if names and (not core or (_A_VARIABLE.fullmatch(core) and not (
            strong and _reads_as_itself(core)))):
        # "${DB_PASSWORD:-}" and "${A:-${B}}" say where the value lives and
        # hold none, as "${DB_PASSWORD}" does. A default that is a word is
        # judged where the key/value rule judges it (_harmless_setting).
        return True
    # Code means code, not "contains a bracket". "contains a bracket" cost an
    # attacker one character: password: Sonne(2024)! and P@ssw0rt[2024] both
    # went out in full, and a password policy that demands a symbol makes
    # that the *likely* shape rather than a contrived one. What code looks
    # like is a call or a subscript on an identifier, whole.
    if not strong and _LOOKS_LIKE_CODE.match(text):
        # Only under a hint. Under a key that says "password", code is not
        # what the value is — and the test cannot tell `Session()` from
        # `Sonne(2024)`, which a password policy demanding a symbol makes
        # the likelier of the two.
        return True
    if text.isdigit():
        # A PIN is a secret; a timeout, a port and a count are not. Under a
        # key that says "password" or "PIN", any run of four digits or more
        # is the secret — the eight-digit ceiling published "123456789" for
        # one character. Under a hint, the window still applies, because
        # "session: 1700000000" is a timestamp.
        return (len(text) < _SHORTEST_PIN if strong
                else not _SHORTEST_PIN <= len(text) <= 8)
    return False


#: The fewest digits a PIN has: under a key that names one -- a password,
#: a PIN, a code (_names_a_code) -- a number this long is the secret.
_SHORTEST_PIN = 4


def looks_like_a_credential(value: str, key: str = "") -> bool:
    """Whether repeating this value in a summary would publish a secret.

    Three ways to qualify, in falling order of confidence.

    The value can be one on its own — a Stripe key, a PEM block, a URL
    carrying a password are credentials wherever they appear.

    The key can name one outright — password, passwd, token, Kennwort — in
    which case the value is a secret unless it is demonstrably something
    else. This tier exists because a real password's *shape* is unbounded:
    judging shape first let "hunter2", "Sonnenschein1!" and "correct horse
    battery staple" through under keys that said what they were.

    Or the key can merely hint — key, salt, session, signature — and then
    the value has to look like a credential too. Judging on a hint alone
    rewrote seven of nine lines of an ordinary Python module and turned
    npm's `"private": true` into a redaction.
    """
    text = str(value).strip().strip("\"'")
    if not text:
        return False
    found = _CREDENTIAL_SHAPE.search(text)
    if found and len(found.group(0)) >= len(text) * 0.6:
        # Only when the credential is essentially the whole value. Two of the
        # alternatives are unanchored — a PEM header, an Azure AccountKey —
        # and redact_value replaces the *whole string* when this returns True,
        # so thirty-one characters of "-----BEGIN RSA PRIVATE KEY-----"
        # anywhere in a preview deleted the entire summary and left
        # "<173 chars — not shown>" in its place. The same trick blanked an
        # injection warning. A match inside a longer string is handled by the
        # in-text substitution instead, which keeps the sentence.
        return True
    name = str(key)
    if _names_the_secret_not_the_value(name):
        # "secret_manager" is a service, "api_key_env" is the variable that
        # holds one, "token_expiry" is a date. Exempt from *both* tiers:
        # falling through to the weaker one redacted all three anyway.
        return False
    if _NAMES_A_SECRET.search(name):
        return not _clearly_not_a_secret(text, strong=True)
    if not name or not (SECRETISH.search(name) or _glued_key(name)):
        return False
    if _clearly_not_a_secret(text) or not _OPAQUE.match(text):
        return False
    # "key: primary", "salt: kosher", "session: morning" — one plain word
    # under a hint is a word. This is the tier where that has to be decided
    # on the value, because the key was never going to settle it.
    return not _is_a_plain_word(text)


#: The same shapes, findable inside a sentence. A model or a heuristic
#: quoting a file writes "der neue Token ist ghp_16C7e42F…" — one token in a
#: line of prose, which no whole-value test can see.
#:
#: Each \b below becomes an ASCII word boundary (_OPENS). In a str
#: pattern \b is Unicode-aware, and Chinese and Japanese write no blank
#: between a word and the token after it: "新的令牌是ghp_…" had no \b before
#: the token, and it went out whole. Glued to a Latin letter or a digit a
#: token is part of a longer identifier, as before; glued to anything else
#: it is a token.
_OPENS = r"(?<![A-Za-z0-9_])"
_CREDENTIAL_IN_TEXT = re.compile((
    r"\b(?:sk|pk|rk)[-_](?:live|test|prod)[-_][A-Za-z0-9]{8,}"
    r"|\b(?:sk|pk|rk|tok)[-_](?=[-_A-Za-z0-9]*\d)(?=[-_A-Za-z0-9]*[A-Z])[-_A-Za-z0-9]{16,}"
    r"|\b(?:ghp|gho|ghs|ghu|ghr|glpat|xox[bpsaor])[-_][-_A-Za-z0-9]{8,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"
    r"|\$(?:2[aby]|apr1|1|5|6)\$[./A-Za-z0-9$]{12,}"
    r"|\b(?:npm_|dop_v1_|dockerhub_|shpss_|shpat_)[A-Za-z0-9]{16,}"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{12,}"
    r"|\b(?:AIza|ya29\.)[-_A-Za-z0-9]{10,}"
    r"|\bey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"
    r"|AccountKey=[A-Za-z0-9+/=]{16,}"
    r"|\bBearer_SEPARATING+[A-Za-z0-9._-]{16,}").replace(r"\b", _OPENS)
    .replace("_SEPARATING", _SEPARATING))

#: An assignment carried inside prose. The scrub of a *summary* has no key
#: to pass — the key is inside the sentence: "The file sets
#: DB_PASSWORD=hunter2", "Kennwort: Sonnenschein1", "the console passcode is
#: Sonnenschein1!". Every one of those reached the markdown report, which is
#: the surface a reader pastes into a ticket.
_SECRET_IN_SENTENCE = re.compile(
    r"((?:^|[^A-Za-z0-9_.])(?:[A-Za-z_][\w.\-]{0,40}?)?"
    # The word itself, named: what is glued in front of it is judged as the
    # key reader judges a key (_sentence_value_is_secret). A key's name of
    # two words first (_KEY_WORD_SHAPE: "api key", "secret key",
    # MASTERKEY, masterKey): tried after "secret", "secret key: x" was the
    # word "secret" with the value "key", judged no secret, and the scan
    # resumed past the key.
    r"(?P<word>" + _KEY_WORD_SHAPE
    + r"|pass(?:word|wd|phrase|code)|pwd|secret|token"
    # \b before the alternation is not enough for these: the group sits
    # inside a longer key pattern, and "tan" without its own boundary
    # matched inside "Stand" — so a date after the word Stand was hidden;
    # "pin" had none, and named a secret inside "spinner", "mapping" and
    # "Philippines". Each is a word of the key (_PIN_OR_TAN): user_pin,
    # pinCode, mTAN. And "otp" (_OTP): footprint, spotprice.
    r"|" + _PIN_OR_TAN + r"|" + _OTP
    + r"|passphrase|zugangscode|sicherheitscode"
    r"|pass?wort|kennwort|geheimnis|credential)s?"
    # The rest of the key the word is written in: password_v2, secret.key.
    # Not over a "." that ends a sentence -- one with a blank or the text's
    # end after it: "reveal the secrets. password: Kx9v..." was the
    # key "secrets." with the value "password", judged no secret, and the
    # scan resumed past the password. "secrets;" and "secrets!" never were
    # a key's characters, and read on as they did.
    r"(?:[\w\-]|\.(?!_SEPARATING|\Z)){0,20}"
    # A key in quotes, or a name in quotes subscripting one, joined to its
    # value by ":", "=" or "=>" and nothing else: 'PASSWORD': ...,
    # os.environ["DB_PASSWORD"] = ..., $db['password'] = ..., 'password' =>
    # ... Nothing but [\w.\-] was read here, so the quote ended the rule and
    # nothing in a Django settings.py, a PHP array or a Ruby hash was a
    # sentence's. Not a blank or a copula after
    # the quote: an HTML attribute's value is a key in quotes to this rule
    # -- href="#developer-mapping" title="Link" is "pin" and a blank --,
    # and /usr/share/doc lost 479 of them to it. The key is kept (quoted),
    # and _sentence_value_is_secret judges its name and value as the member
    # rule judges a member's; a value after it may open with a quote too
    # (opened).
    r"(?:(?P<quoted>[\"'])\]?_SEPARATING{0,8}(?:=>|[:=])"
    # A label between the word and its value: "TAN 01: 449120", "Passwort 1:
    # …", "Kennwort neu: …", "password (old): …". Without this the label
    # became the *value*, was declared clearly-not-a-secret, and re.sub
    # resumed past the credential — the same shape as the "lautet:" bug, and
    # a numbered TAN list is exactly how the German files this tool is
    # pointed at are written.
    r"|(?:_SEPARATING{1,3}(?:\d{1,4}|neu|alt|new|old|\(\w{1,12}\)"
    r"|Nr\.?\s?\d{1,4})){0,2}"
    # A bare space counts as the separator too. .netrc and .pgpass and half
    # the handover notes ever written put the value straight after the word:
    # "machine ftp.example.de login backupsvc password Sonnenschein2024".
    # Up to four words may stand between the key and the copula: "Die TAN
    # fuer die Freigabe ist 884213" is how a handover note writes it, and
    # requiring the two to touch missed every such sentence. Only for the
    # copula forms — a bare space at that distance would match anything.
    # Every whitespace run bounded. Written with bare "\s+" and "\s*" the
    # three runs in a row could split one run of spaces between them in
    # combinatorially many ways: "password" followed by 1,200 spaces took
    # five seconds, 3,000 took eighty-five, and learn_secrets runs this over
    # 400 kB of raw text. The third catastrophic-backtracking bug in this
    # file, and the same cure as the other two.
    #
    # Which way the two are joined is kept (copula, joined, sign): a value
    # joined by nothing but whitespace is one piece (_JOINED_VALUE). "=>"
    # joins them as ":" does: PHP's, Ruby's and Perl's arrays and hashes.
    r"(?:(?:_SEPARATING{1,8}\w{1,40}){0,4}_SEPARATING{1,8}"
    r"(?P<copula>\bis\b|\bist\b|\blautet\b)"
    r"|_SEPARATING{0,8}(?:(?P<joined>=>|[:=]|\bis\b|\bist\b|\blautet\b)"
    r"|_SEPARATING)))"
    # After a key in quotes, a value in quotes is the text between them
    # (_QUOTED_VALUE), the opening quote kept with the key: as the member
    # rule masks a JSON member, and '"password": "Sommer24!"' is masked
    # as it was before this rule read it -- its quotes left standing.
    r"_SEPARATING{0,8}(?P<sign>[:=])?_SEPARATING{0,8}"
    r"(?(quoted)(?P<opened>[\"'])?))"
    r"(?P<value>(?(opened)_QUOTED_VALUE|(?(quoted)_JOINED_VALUE"
    r"|(?(copula)_JOINED_VALUE|(?(joined)_JOINED_VALUE"
    r"|(?(sign)_JOINED_VALUE|_ONE_PIECE))))))"
    .replace("_SEPARATING", _SEPARATING)
    # Up to the quote that opened it, over blanks and punctuation, a
    # backslash escaping the character after it; no further than a value
    # reaches (_SENTENCE_VALUE), nor over a line break, sift's " \u23ce "
    # included. A quote that does not close within that is no value in
    # quotes, and the value is read as after a bare key.
    .replace("_QUOTED_VALUE",
             r"(?:(?!(?P=opened))[^\r\n\u23ce\\]|\\[^\r\n\u23ce])"
             rf"{{1,{_SENTENCE_VALUE}}}(?=(?P=opened))")
    .replace("_JOINED_VALUE",
             rf"(?:[^\s,;)\]:](?:[^\s,;)\]]|{_INNER_GAP}(?=[^\s,;)\]]))"
             rf"{{1,{_SENTENCE_VALUE - 2}}}[^\s,;)\]:.]|[^\s,;)\]:]{{2}})")
    .replace("_ONE_PIECE",
             rf"(?:[^\s,;)\]:][^\s,;)\]]{{1,{_SENTENCE_VALUE - 2}}}"
             r"[^\s,;)\]:.]|[^\s,;)\]:]{2})"),
    re.IGNORECASE)


#: A .pgpass line: host:port:database:user:password, positionally. No key
#: name anywhere in it, so neither tier could see it — and the whole file is
#: nothing but credentials. A closed shape rather than a wordlist: five
#: colon-separated fields whose second is a port.
_PGPASS_LINE = re.compile(
    # Not anchored to a line: a preview folds the newlines out, and that is
    # the surface the credential actually reached.
    # The password runs over a gap that is not one plain blank, as the
    # sentence rule's value does (_INNER_GAP).
    r"(?:(?<=\s)|^)([^\s:]{1,253}:\d{1,5}:[^\s:]*:[^\s:]{1,64}:)"
    rf"(\S+(?:{_INNER_GAP}\S+)*)")


#: A PEM block, header to footer. Matching the header alone hid the words
#: "-----BEGIN OPENSSH PRIVATE KEY-----" and published the key underneath
#: them, which is the whole of the file that matters.
_PEM_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", re.DOTALL)


#: The same words as _ABOUT_A_SECRET, unanchored, for prose: in a key they
#: are a suffix ("token_budget"), in a sentence they are a word standing on
#: its own ("the token budget for this model is 4096").
_ABOUT_A_SECRET_WORD = re.compile(
    # "(?<![a-z0-9])", not "\b": an underscore is a word character, so \b
    # never matches between "token" and "budget" in "token_budget" — the
    # exemption a *key* gets could not be given to the same words in prose,
    # and "token_budget: 4096" was destroyed by the rule whose comment says
    # it was fixed.
    r"(?<![a-z0-9])(?:manager|name|names|arn|path|file|filename|env|var"
    r"|ref|reference"
    r"|algorithm|algo|required|enabled|disabled|length|len|min|max|expiry"
    r"|expires|ttl|rotation|policy|hash|hashed|digest|budget|limit|size"
    r"|quota|window|total|sum|threshold|interval|retries|count|age|days"
    r"|format|location|date|datum|time|timestamp|owner|user|subject|issuer"
    r"|serial|version|status|state|mode|note|notes|comment|description"
    r"|title|strength|complexity|attempts|num|count|total|amount"
    r"|seconds|secs|ms|millis|minutes|hours|days|weeks|interval|period"
    r"|Budget|Groesse|Gr\u00f6\u00dfe|L\u00e4nge|Laenge|Dauer|Anzahl"
    r"|G\u00fcltigkeit|Gueltigkeit|Frist)(?![a-z0-9])", re.IGNORECASE)


#: What is glued in front of the sentence rule's word, up to it: "user_"
#: of user_password (_sentence_value_is_secret).
_GLUED_KEY_HEAD = re.compile(r"[A-Za-z_][\w.\-]*\Z")
#: A key's run of characters after the sentence rule's word, to the end
#: of its key (the rule's own run, which stops before a "." that ends a
#: sentence).
_KEY_RUN = re.compile(r"[\w.\-]*")


def _sentence_value_is_secret(match) -> bool:
    """Whether the value of an assignment carried in prose is a secret.

    In prose the key and the value are separated by whatever the writer felt
    like — a colon, an "is", or nothing at all: .netrc writes "login backupsvc
    password Sonnenschein2024" and half the handover notes ever written do
    the same. Accepting a bare space is what reaches those, and it is also
    what made "The password policy requires 12 characters", "Set the password
    via the DB_PASSWORD environment variable" and "CHARS_PER_TOKEN is 4 in
    the tokenizer configuration" come back mangled.

    So the value has to look like one here too: not a word anybody would
    recognise. The cost is that a passphrase of ordinary words quoted in
    prose survives — "the password is correct horse battery staple" — which
    the key-based path catches at the extractor, where the file wrote it.
    """
    prefix, value = match.group(1), match.group("value")
    start = match.start("value")
    if any(_ALREADY_HIDDEN.match(match.string, start + at)
           for at, char in enumerate(value) if char == "<"):
        # Someone got here first. The earlier rules leave a marker behind,
        # and the marker keeps whatever quote surrounded the value — so
        # `password "…"` became `password "<16 chars — not shown>"`, whose
        # first four characters this rule then hid as `<4 chars — not
        # shown>`. A marker, read in the text it stands in: "<" alone in a
        # value was taken for one, and "Kx9vQ2mZ7pL4  <-- change this" was
        # published whole.
        return False
    # When the prose *is* an assignment, ask the key path. Two exemption
    # lists that have to agree drift apart: this one had "budget" and
    # "seconds" while the key list had "context", so "token_budget = 4096"
    # survived and "context_tokens=262144" did not — in the same file, by
    # the same rule, for no reason a reader could see.
    # A key in quotes or subscripted too, joined by "=>" too, and a value's
    # opening quote after it (_SECRET_IN_SENTENCE): "'token_budget': '4096'"
    # is the key token_budget, as "token_budget: 4096" is.
    named = re.search(rf"([A-Za-z_][\w.\-]*)[\"']?\]?{_SEPARATING}*"
                      rf"(?:=>|[:=])?{_SEPARATING}*[\"']?$", prefix)
    if match.group("quoted") is not None:
        # A key in quotes is a member's, and its value is judged as the
        # member rule judges a member's (looks_like_a_credential, by the
        # name in the quotes) -- in code, not prose: the rule's own words
        # are no key here, "mapping" and "grouping" hold "pin" and are no
        # name of a secret. A value without quotes after it only where it
        # is a literal (_a_literal), as the key reader has it.
        if named is None or (match.group("opened") is None
                             and not _a_literal(value)):
            return False
        return looks_like_a_credential(value, named.group(1))
    if named and _names_the_secret_not_the_value(named.group(1)):
        return False
    # The key the rule's word is written in, with what is glued in front of
    # it -- user_password, max_tokens -- is judged by the key path's lists:
    # a word in front says the value is metadata only where
    # _ABOUT_A_SECRET_PREFIX names it. The prose words below are read from
    # the rule's word on. Read over the whole prefix, "user" (a suffix in
    # _ABOUT_A_SECRET: password_user) exempted "user_password: hunter2Xy",
    # which the key path hides, and the summary published it.
    # From the word's end on, since the word may hold a blank ("api key"),
    # which no run of a key's characters crosses.
    word = match.start("word")
    glued = _GLUED_KEY_HEAD.search(match.string, match.start(1), word)
    key = match.string[glued.start() if glued else word:
                       _KEY_RUN.match(match.string, match.end("word"),
                                      match.end(1)).end()]
    if _names_the_secret_not_the_value(key):
        return False
    if _ABOUT_A_SECRET_WORD.search(match.string, word, match.end(1)):
        # The same exemption a *key* gets, in prose: "token_budget = 4096" is
        # a number about a secret and so is "the token budget for this model
        # is 4096". Without it, widening the distance between the word and
        # the copula — which is what reaches "Die TAN fuer die Freigabe ist
        # 884213" — brought every such sentence with it.
        return False
    # Each piece between the gaps a value runs on over (_INNER_GAP) is
    # judged as a value without them was: "Password:<TAB>see<TAB>vault" is
    # words, and "Sonnen<TAB>schein1" is a secret by its second piece, which
    # judging the first alone let through whole.
    return any(_piece_is_secret(piece.rstrip(":"))
               for piece in value.split())


#: The marks that may close a sentence, a quote or a JSON string after a
#: value in prose, and say nothing about the value (_piece_is_secret).
_CLOSING_MARKS = "!?\"'“”„)]}>,.;:"


def _piece_is_secret(value: str) -> bool:
    """_sentence_value_is_secret's test of the value itself."""
    if _clearly_not_a_secret(value) or _is_a_plain_word(value):
        return False
    # Judged without the sentence's own punctuation, hidden with it. Letting
    # "!" into the value is what finally hid all of "Winterreise2026!"
    # instead of publishing its last character — and it also swept up "Ist
    # das Kennwort korrekt?" and "Das Passwort ist geheim!", where the word
    # is ordinary and only the mark made it look otherwise. Trimming for the
    # *decision* keeps both: a word with a mark after it is still a word, and
    # a credential with a mark after it is still a credential. The quote
    # and the bracket that close a JSON string after the word too:
    # '{"error": "api key invalid"}' -- an LLM server's answer -- was
    # 'api key <9 chars — not shown>', as '"password invalid"}' was.
    trimmed = value.rstrip(_CLOSING_MARKS)
    if trimmed != value and (_clearly_not_a_secret(trimmed)
                             or _is_a_plain_word(trimmed)):
        return False
    bare = re.sub(r"^\W+|\W+$", "", value, flags=re.UNICODE)
    if not value[:1].isalnum() and (not bare or _is_a_plain_word(bare)):
        # Punctuation *and* a word behind it is prose, not a credential:
        # sift's own note — "context window 262144 tokens (reported by the
        # server for this model)" — came back as "…tokens <9 chars — not
        # shown> by the server…", because "tokens" is a secret-ish word and
        # "(reported" is what followed the space.
        #
        # Punctuation and something that is not a word is a credential with a
        # scheme on the front: "{SSHA}5en6G6MezRroT3XKqkdPOmY" is how LDAP
        # writes a password hash, and rejecting on the brace alone published
        # it.
        return False
    return True


def _to_the_last_secret_cell(text: str, start: int, end: int) -> int:
    """Where a value ``text[start:end]`` ends that runs on over a tab: after
    its first cell, whatever that holds, and after each later one that holds
    a piece of a secret (_piece_is_secret) -- a tab is where a sheet row's
    cells meet, and a word after it is the next cell, not the password's
    tail. "password: Kx9vQ2mZ7pL4<TAB>Item" was the value
    "Kx9vQ2mZ7pL4<TAB>Item", the bare password no known value, and a model's
    sentence naming it with no key beside it was published. A tail
    that is itself a piece of a secret still runs on, as a review has it:
    "Kx9v<TAB>Q2mZ7pL4Wq8rTz", "Sonnen<TAB>schein1". One split of the text,
    so a line of any length costs as much as it is long."""
    value = text[start:end]
    if "\t" not in value:
        return end
    cells = value.split("\t")
    last = len(cells[0].rstrip())
    at = len(cells[0]) + 1
    for cell in cells[1:]:
        if any(_piece_is_secret(piece.rstrip(":")) for piece in cell.split()):
            last = at + len(cell.rstrip())
        at += len(cell) + 1
    return start + last


#: What a redaction looks like once written. Recognised so that a second
#: pass does not redact the first one's marker: "api_key=<20 chars — not
#: shown>" came back as "api_key=<3 chars — not shown> chars — not shown>",
#: where the 3 is the length of "<20".
_ALREADY_HIDDEN = re.compile(r"<\d{1,9} chars — not shown>")


#: Values hidden while reading the current file, so they can be hidden in
#: whatever is said about it afterwards.
#:
#: The model reads the file and answers with what it found — and a handover
#: note whose password sift redacted in the preview came back with
#: "Winterreise2026" in the *entities*, where there is no key to judge it
#: by and no assignment to recognise. Judging the same value twice, once
#: with evidence and once without, can only end one way.
#:
#: A ContextVar because sift serve answers each upload on its own thread,
#: and one file's secrets must not leak into another file's summary.
_hidden_here: "contextvars.ContextVar[set]" = contextvars.ContextVar(
    "sift_hidden", default=None)


@contextlib.contextmanager
def remember_hidden():
    """Collect what gets hidden while reading one file."""
    token = _hidden_here.set(set())
    try:
        yield _hidden_here.get()
    finally:
        _hidden_here.reset(token)


#: Punctuation a value may be quoted or terminated with. A model hands back
#: what it read, not the characters around it.
_AROUND_A_VALUE = "\"'\u201c\u201d\u201e!?.,;:"


def _hide(text: str) -> str:
    seen = _hidden_here.get()
    if seen is not None:
        # Both forms. The value as it stands in the file may carry the
        # sentence's punctuation — "Das Kennwort lautet Winterreise2026!"
        # hides all sixteen characters, which is right — but the model reads
        # the file and hands back "Winterreise2026", and a set holding only
        # the punctuated form does not match it. Hiding it in the preview
        # and publishing it in the entities is not an improvement over
        # publishing it twice.
        # Short values are not remembered: hiding every later occurrence of
        # a four-character string would take words out of the summary.
        seen.update(form for form in _hidden_forms(text) if len(form) >= 4)
    return _marker(text)


def _hidden_forms(text: str) -> "set[str]":
    """The forms _hide remembers a value in, and _repetitions looks for:
    trimmed of quotes, and of the punctuation around it, each also spelled
    as it is written again (_spellings)."""
    return {spelled
            for form in (text.strip("\"'"), text.strip(_AROUND_A_VALUE))
            for spelled in _spellings(form)}


def _spellings(form: str) -> "tuple[str, ...]":
    """A value holding whitespace, also as it is written again: its runs of
    whitespace as one blank -- a preview joins words so, and so does a model
    -- and its tabs as four blanks, as one_line folds a line. A value with a
    tab in it was found whole in the file and published in the forms it was
    repeated in."""
    if not any(char.isspace() for char in form):
        return (form,)
    return (form, " ".join(form.split()), form.replace("\t", "    "))


def _marker(text: str) -> str:
    """What a hidden value is replaced with: its length and nothing else."""
    return f"<{len(text)} chars — not shown>"


def learn_secrets(text: str) -> None:
    """Record what a redaction would hide here, without hiding anything.

    Run over the extracted text, whose own content is deliberately left
    intact. What comes back out of the model is the same values with the
    context stripped off — a key name, an assignment, a surrounding
    sentence — so the only chance to recognise them is while that context
    is still there.
    """
    if _hidden_here.get() is None or not text:
        return
    redact_value(text[:_LEARN_LIMIT])


#: How much of a file is read for this. The values a model quotes back come
#: from what it was shown, and what it was shown is clipped well below this.
_LEARN_LIMIT = 400_000
#: The same limit, for callers of find_secrets: spans start in the first
#: SECRET_SCAN_LIMIT characters of a text, and only one that starts there
#: reaches past it (see _SCAN_OVERHANG). Text after that point has to be
#: scanned by the caller, piece by piece — and a value learned before the
#: limit is then not known there.
SECRET_SCAN_LIMIT = _LEARN_LIMIT
#: How far past SECRET_SCAN_LIMIT find_secrets reads, so a span that starts
#: before the limit is found to its end. A PEM private key of 4096 bits is
#: about 3,300 characters and a line holds what it holds; 64 KiB covers both
#: with room, and costs a sixth more reading at most.
_SCAN_OVERHANG = 65_536


def hide_known(value):
    """Hide anything already hidden for this file, wherever it turns up."""
    if not isinstance(value, str):
        return value
    seen = _hidden_here.get()
    if not seen:
        return value
    # Longest first. The set holds a value both as the file writes it and
    # trimmed of the punctuation around it, and iterating in set order let
    # the shorter one win: "Winterreise2026!" came back as "<15 chars — not
    # shown>!", which hides the password and then names its last character.
    for secret in sorted(seen, key=len, reverse=True):
        if secret not in value:
            continue
        marker = f"<{len(secret)} chars — not shown>"
        if _word_like(secret):
            value = re.sub(rf"(?<![^\W_]){re.escape(secret)}(?![^\W_])",
                           lambda _match: marker, value)
        else:
            value = value.replace(secret, marker)
    return value


#: A character that continues a token: a letter or a digit, of any script.
#: An underscore, a hyphen, a blank or a quote ends one.
_TOKEN_CHAR = re.compile(r"[^\W_]")
#: A run of token characters, and one that is opaque: sixteen or more, or
#: eight or more mixing letters and digits.
_TOKEN_RUN = re.compile(r"[^\W_]+")


def _opaque_run(text: str) -> bool:
    """Whether ``text`` holds a run of letters and digits that is no word:
    sixteen characters or more, or eight or more mixing letters and digits
    -- "Kx9vQ2mZ7pL4", a hex digest, a webhook's key."""
    for run in _TOKEN_RUN.findall(text):
        if len(run) >= 16 or (len(run) >= 8
                              and any(char.isdigit() for char in run)
                              and not run.isdigit()):
            return True
    return False


#: What a word-like value may hold between its first and last character:
#: letters, digits, and the separators a word or a phrase is written with.
_WORDISH = re.compile(r"[^\W_](?:[^\W_]|[ ._\-])*")


def _word_like(form: str) -> bool:
    """Whether a remembered value is hidden in what sift says only as a
    token of its own (hide_known): letters and digits, joined by blanks,
    dots, underscores or hyphens, beginning and ending with a letter or a
    digit, and no opaque run (_opaque_run). "true" is word-like, and was
    hidden inside "Construe" and "information";
    "sonnenschein", "hunter2", "2026-01-01" and "Kaffee und Kekse" are
    too. Any other value is opaque -- "Kx9vQ2mZ7pL4", "Welcome2024!", a
    URL -- and is hidden wherever it stands, glued into a longer string
    too: no word holds it by chance."""
    return bool(form and _WORDISH.fullmatch(form)
                and _TOKEN_CHAR.match(form[-1]) and not _opaque_run(form))


def hide_said(text):
    """Something sift says about the file, through the redaction: what the
    file taught hide_known first, then scrub, then what scrub itself learned.

    Known values first, because scrub hides the part of a value its rules
    can see: in "password: Kx9v Q2mZ7pL4Wq8rTz" the sentence rule stops at
    the blank, and once "Kx9v" is a marker the value the file held whole --
    "Kx9v<TAB>Q2mZ7pL4Wq8rTz", said again with one blank -- matches nothing,
    and "Q2mZ7pL4Wq8rTz" was published.

    And what scrub hides in the text as written is learned before that: a
    known value can be the word scrub reads a sentence by -- a JSON member
    "secret": "password", a setting AUTH_METHOD=password --, and "The admin
    password is Kx9vQ2mZ7pL4" went to scrub as "The admin <8 chars — not
    shown> is Kx9vQ2mZ7pL4", with no key left to read.
    Learned, the value is hidden by the first pass with the keyword. Of the
    two cures measured -- this, and keeping a known value that holds a
    keyword out of the first pass -- this one hid all nine probes; the
    other let "password Kx9v Q2mZ7pL4Wq8rTz" through, a known value that
    holds the keyword. Only where the first pass hid something: elsewhere
    scrub reads the text as written anyway (a second scrub of each field
    cost 2.2 ms per 2,000 characters)."""
    first = hide_known(text)
    if first != text:
        scrub(text)
        first = hide_known(text)
    return hide_known(scrub(first))


def hide_known_deep(value):
    """What hide_known does, over every string a value of any shape holds
    -- a dict's values, a list's items -- and nothing else: a key is sift's
    field name or the file's own name for a field, never a value of it.

    For the metadata, which travels into --json, the cache, -v's details and
    the browser card beside the fields hide_known already reads: a shell
    script's sampled commands are the file's lines, values and all.

    In one pass over each string rather than one per value hidden: a
    prod.env of 16,000 settings lists 16,000 keys in its metadata, and
    hide_known on each -- every one of 14,176 values looked for in every
    key -- took its summary from 4 seconds to 14 with no clock looking.
    Every character inside an occurrence is hidden (_occurrences), which is
    at least what hide_known hides; overlapping ones become one marker.
    """
    seen = _hidden_here.get()
    if not seen:
        return value
    return _hide_occurrences(value, _known_automaton(seen, "the metadata"))


def _hide_occurrences(value, automaton: "_KnownAutomaton"):
    """``value`` with every occurrence of a known form hidden -- of an
    opaque one wherever it stands, of a word-like one where it is a token
    (_known_matches)."""
    if isinstance(value, str):
        merged = []
        found = _known_matches(automaton, value, "the metadata")
        for start, end in sorted(found):
            if merged and start < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        out, at = [], 0
        for start, end in merged:
            out += [value[at:start], _marker(value[start:end])]
            at = end
        return "".join(out) + value[at:]
    if isinstance(value, dict):
        return {key: _hide_occurrences(item, automaton)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_hide_occurrences(item, automaton)
                           for item in value)
    return value


def remember_values(text: str, spans, pieces=()) -> None:
    """Record the values find_secrets found in ``text`` as hidden for this
    file (inside remember_hidden; nothing outside it), in every form
    _value_forms says a value is repeated in.

    learn_secrets records what a redaction of the text hides, and that is
    not all the file's secrets: find_secrets judges a line by its key, a
    JSON member by its name, and every value of a .env or a .properties by
    the file's name. None of the last were ever recorded, so a model that
    read "TARGET_DB_PASSWORD=Kx9vQ2mZ7pL4" and answered with the value had
    its answer shown, one line from the note saying no value is repeated.

    And every value find_secrets finds inside one of them, in its forms
    too: find_secrets merges a finding into the span around it, and a model
    hands back the piece it means. The password of a JSON member written
    over three lines of a .env was remembered as the setting and its lines
    only, and "uses Mv4Tq9Wz2Lk7Rx3Nb as its credential" was published. ``pieces`` are the findings folded into a span
    (find_secrets_and_pieces); each that lies inside one of ``spans`` is
    remembered. A value over several lines is read again as a text of its
    own (_pieces) -- under a settings file's name none of its lines was
    judged by its key.
    """
    seen = _hidden_here.get()
    if seen is None:
        return
    spans = sorted(span for span in spans if span.reason != "bidi")
    starts = [span.start for span in spans]
    inside = [[] for _span in spans]
    for piece in pieces:
        at = bisect.bisect_right(starts, piece.start) - 1
        if at >= 0 and piece.end <= spans[at].end:
            inside[at].append(text[piece.start:piece.end])
    keys = None
    for span, found in zip(spans, inside):
        raw = text[span.start:span.end]
        key = None
        if span.reason == "setting" and any(
                mark in raw for mark in ("${", ",", ";", ":")):
            # A shell default is judged under the setting's key as well as
            # its variable's name, and a part of a value by the key
            # (_value_forms); any other span is a secret by a rule, and its
            # defaults and parts are judged as one.
            if keys is None:
                keys = _setting_keys(text, {s.start for s in spans
                                            if s.reason == "setting"})
            key = keys.get(span.start)
        forms = _value_forms(raw, key)
        if _LINE_BREAK.search(raw) or '\\"' in raw:
            found = found + list(_pieces(raw))
        for piece in found:
            forms |= _value_forms(piece)
        seen.update(form for form in forms if len(form) >= 4)


def _pieces(raw: str):
    """The values find_secrets finds inside the value found at a span, read
    as a text of its own: a line "password = ..." of a value over several
    lines, which a settings file's name never judges by its key, and what
    the file's own scan finds there too. And inside the value as dotenv
    reads a double-quoted one, its escaped quotes unescaped:
    {\\"password\\": ...} is JSON only so."""
    texts = [raw]
    unescaped = _settled(raw).replace('\\"', '"')
    if unescaped not in raw:
        texts.append(unescaped)
    for text in texts:
        for start, end, reason in find_secrets(text):
            if reason not in ("bidi", "repeated"):
                yield text[start:end]


#: The values that are no secret under a key that names or hints one: a
#: flag -- the row of them _NOT_A_SECRET opens with, "enabled" and
#: "disabled" as well: PASSWORD_AUTH=enabled was a known value, and hidden
#: in every sentence about the file. _NOT_A_SECRET's other words
#: are no flag: "required", "optional", "auto" and "default" name a mode,
#: and "changeme", "guest", "anonymous" are passwords people set, and are
#: known values here.
_FLAGS = frozenset({"true", "false", "none", "null", "nil", "yes", "no",
                    "on", "off", "ja", "nein", "enabled", "disabled"})
#: A reference to a variable, whole: $HOME, ${DB_PASSWORD}.
_A_VARIABLE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")
#: A variable's name as an environment writes one: capitals, and digits
#: only as a part of their own -- POSTGRES_PASSWORD, DB_PASSWORD_2, HOME;
#: not ADMIN2024, Sommer24 or ecret.
_ENV_NAME = re.compile(r"[A-Z]+(?:_(?:[A-Z]+|[0-9]+))*")
#: The opening of a shell expansion with a word: ${NAME:-word} and
#: ${NAME-word} (the word where NAME is unset), ${NAME:=word} and
#: ${NAME=word} (the same, assigned), ${NAME:+word} and ${NAME+word} (the
#: word where NAME is set). ${NAME:?word} is a message, not a value. And
#: Spring's placeholder, ${name:word} with a dotted name -- the default
#: where the property is unset; spring.datasource.password=${DB_PASSWORD:
#: Kx9vQ2mZ7pL4} published its default. To a shell
#: ${NAME:2} is a substring: read as a default, its word only ever adds a
#: piece, and a short one is none. The name is group 1 or group 3.
_EXPANSION_OPEN = re.compile(r"\$\{(?:([A-Za-z_][A-Za-z0-9_]*)(:?[-=+])"
                             r"|([A-Za-z_][\w.\-]*):(?![-=+]))")
#: What _default_words reads: every "${", with the name of one that has a
#: word (group 1 or group 3), and every "}".
_EXPANSION_PART = re.compile(
    r"\$\{(?:([A-Za-z_][A-Za-z0-9_]*)(:?[-=+])"
    r"|([A-Za-z_][\w.\-]*):(?![-=+]))?|\}")
#: Words of letters joined by "-", "_" or ".": "info", "en-US", "my_app.prod".
_PLAIN_WORD = re.compile(r"[^\W\d_]+(?:[-_.][^\W\d_]+)*")
#: The longest plain word, in characters.
_PLAIN_WORD_LONGEST = 32
#: A number of up to twelve digits -- a port, a size, a timeout, a
#: timestamp in seconds; a card number has thirteen or more.
_PLAIN_NUMBER = re.compile(r"[+-]?\d{1,12}(?:\.\d{1,12})?")
#: A path, absolute or relative, of word characters, dots and hyphens.
_PLAIN_PATH = re.compile(r"(?:~|\.{1,2})?(?:/[\w.\-]*)+")
#: A URL with no user information, no "key=value" in its query. Anchored:
#: searched from every position, the scheme read a run of letters to its
#: end from each one (test_backtracking: 3.8 s on 30 kB).
_PLAIN_URL = re.compile(r"^[a-z][a-z0-9+.\-]*://[\w.\-]+(?::\d{1,5})?"
                        r"(?:/[\w.\-~%+]*)*(?:\?[\w.\-~%+&]*)?(?:#[\w.\-/]*)?",
                        re.IGNORECASE)

# The shapes of a plain setting that is neither a plain word nor a plain
# number: ENCODING=UTF-8, DB_HOST=192.0.2.10, START_DATE=2026-01-01,
# TZ=Europe/Berlin and LANG=en_US.UTF-8 were known values, hidden in every
# sentence about the file. Each is anchored, reads ASCII digits only, and
# reads each run up to a separator the run cannot hold, a bounded number of
# times: linear (test_backtracking).

#: A number of an IPv4 address, 0 to 255, with no leading zero.
_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
#: An address: IPv4 (group 1), or IPv6 in brackets (group 2), each with a
#: port (group 3); or IPv6 alone (group 4), whose last group no port could
#: be told from. What _an_address reads the groups as decides.
_PLAIN_ADDRESS = re.compile(
    rf"^(?:(?:((?:{_OCTET}\.){{3}}{_OCTET})|\[([0-9A-Fa-f:.]{{2,45}})\])"
    r"(?::([0-9]{1,5}))?|([0-9A-Fa-f:.]{2,45}))", re.ASCII)
#: A time of day as ISO 8601 writes one (extended): 03:30, 23:59:59.5,
#: with a zone -- Z, +01:00, -0500 -- or none.
_TIME_OF_DAY = (r"(?:[01][0-9]|2[0-3]):[0-5][0-9]"
                r"(?::[0-5][0-9](?:[.,][0-9]{1,9})?)?"
                r"(?:Z|[+-](?:[01][0-9]|2[0-3])(?::?[0-5][0-9])?)?")
#: A date (year, month and day are groups 1 to 3), a date and a time
#: joined by "T", or a time: 2026-01-01, 2026-01-01T12:00:00+01:00, 03:30.
#: A date is a day of its month where _a_day says so.
_PLAIN_TIME = re.compile(
    r"^(?:([0-9]{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    rf"(?:T{_TIME_OF_DAY})?|{_TIME_OF_DAY})", re.ASCII)
#: A part of a time zone's name after its area, as tzdata writes one: a
#: capital, then letters joined by "_" or "-" (Port-au-Prince,
#: Buenos_Aires), fourteen characters at most.
_ZONE_PART = r"/[A-Z](?:[A-Za-z]|[_-](?=[A-Za-z])){0,13}"
#: An IANA time zone: an area and a city, or an area, a region and a city
#: -- Europe/Berlin, America/Argentina/Buenos_Aires, US/Pacific --, and
#: Etc's: Etc/GMT-14 to Etc/GMT+12, Etc/UTC. Europe/Berlin/../x is none.
_PLAIN_ZONE = re.compile(
    r"^(?:(?:Africa|America|Antarctica|Arctic|Asia|Atlantic|Australia|Europe"
    rf"|Indian|Pacific|Brazil|Canada|Chile|Mexico|US)(?:{_ZONE_PART}){{1,2}}"
    r"|Etc/(?:GMT(?:[+-]?0|-(?:1[0-4]|[1-9])|\+(?:1[0-2]|[1-9]))?"
    r"|UTC|UCT|Universal|Zulu|Greenwich))", re.ASCII)
#: A character set's name that holds a digit (one of letters alone --
#: ASCII, Shift_JIS, EUC-JP -- is a plain word): UTF-8 and its kin,
#: ISO-8859-1 to -16, windows-1250 to -1258, a code page (cp1252, IBM437),
#: latin1, KOI8-R, GB18030, Big5, MySQL's utf8mb4. In any case.
_CHARSET = (r"(?:UTF-?(?:7|8|(?:16|32)(?:[BL]E)?)|UCS-?[24]|utf8mb[34]"
            r"|ISO[-_]?8859[-_]?(?:1[0-6]|[1-9])"
            r"|ISO-2022-(?:JP(?:-[123])?|KR|CN)|windows-?125[0-8]"
            r"|(?:cp|IBM)-?[0-9]{3,5}|latin-?[1-9]|KOI8-?[RU]|GB2312|GB18030"
            r"|Big5(?:-HKSCS)?)")
_PLAIN_CHARSET = re.compile(rf"^(?i:{_CHARSET})", re.ASCII)
#: A POSIX locale: a language, and a territory, a character set -- as
#: glibc writes one too: utf8, eucJP -- and a modifier, each if it has one:
#: en_US.UTF-8, C.UTF-8, de_DE@euro, de_DE.ISO-8859-15@euro.
_PLAIN_LOCALE = re.compile(
    r"^(?:C|POSIX|[a-z]{2,3}(?:_(?:[A-Z]{2}|[0-9]{3}))?)"
    rf"(?:\.(?i:{_CHARSET}|euc-?(?:jp|kr|cn|tw)|tis-?620|gbk|sjis))?"
    r"(?:@[a-z]+)?", re.ASCII)
#: A pre-release or a build of a semantic version: identifiers of letters
#: and digits joined by "." or "-", eight at most.
_VERSION_PART = r"[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+){0,7}"
#: A semantic version: 1.2.3, v2.0.0-rc1, 1.0.0-alpha.1+build.5.
_PLAIN_VERSION = re.compile(
    r"^v?(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){2}"
    rf"(?:-{_VERSION_PART})?(?:\+{_VERSION_PART})?", re.ASCII)
#: A duration: 30s, 5m, 1h30m, 500ms, 1.5h, 7d as Go and systemd write
#: one, and PT30S, P1D, PT1H30M as ISO 8601 does -- a part at least.
_PLAIN_DURATION = re.compile(
    r"^(?:(?:[0-9]{1,12}(?:\.[0-9]{1,12})?(?:ns|us|ms|s|m|h|d|w)){1,7}"
    r"|P(?=T?[0-9])(?:[0-9]{1,12}Y)?(?:[0-9]{1,12}M)?(?:[0-9]{1,12}W)?"
    r"(?:[0-9]{1,12}D)?(?:T(?=[0-9])(?:[0-9]{1,12}H)?(?:[0-9]{1,12}M)?"
    r"(?:[0-9]{1,12}(?:[.,][0-9]{1,9})?S)?)?)", re.ASCII)
#: A size: 512M, 512m, 64k, 100KB, 2GiB, 256Mi, 1.5G, 4096B.
_PLAIN_SIZE = re.compile(
    r"^[0-9]{1,12}(?:\.[0-9]{1,12})?"
    r"(?:[KMGTPE]i|[KkMmGgTtPpEe](?:i?[Bb])?|[Bb])", re.ASCII)
#: The shapes that are what they look like, with no reading beyond it.
_PLAIN_SHAPES = (_PLAIN_ZONE, _PLAIN_CHARSET, _PLAIN_LOCALE, _PLAIN_VERSION,
                 _PLAIN_DURATION, _PLAIN_SIZE)


def _plain_shape(value: str) -> bool:
    """Whether ``value`` is a plain setting's shape that is no plain word
    and no plain number: an address, a date or a time, a time zone, a
    character set, a locale, a version, a duration or a size -- whole."""
    address = _PLAIN_ADDRESS.fullmatch(value)
    if address and _an_address(*address.groups()):
        return True
    moment = _PLAIN_TIME.fullmatch(value)
    if moment and _a_day(*moment.groups()):
        return True
    return any(shape.fullmatch(value) for shape in _PLAIN_SHAPES)


def _an_address(ipv4, bracketed, port, bare) -> bool:
    """Whether _PLAIN_ADDRESS's groups are an address: an IPv4 one (its
    numbers are read by the pattern), or IPv6 as ipaddress reads it -- in
    brackets or alone, where "1.2.3" and "cafe" are none --, and a port of
    65535 at most."""
    if port is not None and int(port) > 65535:
        return False
    if ipv4 is not None:
        return True
    try:
        ipaddress.IPv6Address(bare if bracketed is None else bracketed)
    except ValueError:
        return False
    return True


def _a_day(year, month, day) -> bool:
    """Whether _PLAIN_TIME's date is a day of its month -- 2024-02-29 is,
    2026-02-30 and 2025-02-29 are not; a time alone has none to check."""
    if year is None:
        return True
    try:
        datetime.date(int(year), int(month), int(day))
    except ValueError:
        return False
    return True


def _harmless_setting(key: str, value: str) -> bool:
    """Whether a settings value is harmless by its key and itself (the
    key/value rule of in_place_only). Nothing, a flag or a variable under
    any key; under a key that names or hints a secret (_secret_key) --
    with none of the exemptions (default_, target_, _env, _expiry) that let
    the preview publish TARGET_DB_PASSWORD's -- nothing else. Under any
    other key also a plain word, a plain number, a path or a URL, none
    holding an opaque run (_opaque_run), and no number of a PIN's length
    under a key that names a code (_names_a_code); and, under a key that
    names no code either, an address, a date or a time, a time zone, a
    character set, a locale, a version, a duration or a size (_plain_shape)
    -- a code can be written "12:34:56" or "203.0.113.47".

    A value that is a shell default, ${NAME:-word} and its kin, is its
    word, judged under the key and under the name of every variable it
    defaults (_defaulted): ${POSTGRES_PASSWORD:-Gk4Wq8Zr2Mx7Tb} was
    remembered whole, and the word -- the password -- was published. ${DB_PASSWORD:-} and ${A:-${B}} hold none."""
    names, core = _defaulted(value)
    return all(_harmless_value(name, core) for name in (key, *names))


def _harmless_value(key: Optional[str], value: str) -> bool:
    """_harmless_setting's judgement of one value under one key; None for
    a key that names a secret."""
    secret = key is None or _secret_key(key)
    if not value or value.lower() in _FLAGS:
        return True
    if _A_VARIABLE.fullmatch(value):
        # ${NAME} is a reference to every reader that expands one; python-
        # dotenv reads a bare $NAME as itself, so "ADMIN_PASSWORD=$ADMIN2024"
        # is the password "$ADMIN2024" to it.
        if value[1] == "{":
            return True
        return not (_reads_as_itself(value) if secret
                    else _opaque_run(value))
    if secret:
        return False
    if _opaque_run(value):
        return False
    if value.isdecimal() and len(value) >= _SHORTEST_PIN \
            and _names_a_code(key):
        return False
    if _PLAIN_WORD.fullmatch(value):
        return len(value) <= _PLAIN_WORD_LONGEST
    if _PLAIN_PATH.fullmatch(value) or _PLAIN_URL.fullmatch(value):
        return not _opaque_segments(value)
    return bool(_PLAIN_NUMBER.fullmatch(value)
                or (not _names_a_code(key) and _plain_shape(value)))


#: What cuts the tail of a URL or a path into segments: the "/" of a path,
#: the "?" and "#" that open a query and a fragment, the "&" between a
#: query's items.
_SEGMENT_CUT = re.compile(r"[/?#&]")
#: Where a URL's tail begins: after its host and port.
_URL_TAIL = re.compile(r"^[a-z][a-z0-9+.\-]*://[^/?#]*", re.IGNORECASE)


def _opaque_segments(value: str) -> bool:
    """Whether the segments of a path, or of a URL's tail after its host,
    that are no plain word (_PLAIN_WORD) hold an opaque run (_opaque_run)
    once joined: a webhook's key cut into segments of four,
    https://hooks.example.com/Kx9v/Q2mZ/7pL4/Wq8r, was a harmless URL to
    _PLAIN_URL and the run was shown wherever a model wrote it.
    /docs/api/v2, /2024/05/01 and /img/640x480/photo.jpg join to no such
    run: a word is left out, a number of digits is no run, and a short
    version is short."""
    tail = _URL_TAIL.sub("", value, count=1) if "://" in value else value
    joined = "".join(segment for segment in _SEGMENT_CUT.split(tail)
                     if segment and not (_PLAIN_WORD.fullmatch(segment)
                                         and len(segment) <= _PLAIN_WORD_LONGEST))
    return _opaque_run(joined)


def _reads_as_itself(value: str) -> bool:
    """Whether ``value`` is a bare $NAME whose name is none an environment
    would write (_ENV_NAME): "$ADMIN2024", "$ecret". Under a key that
    names a secret that is the secret, not a reference to it."""
    return (value[:1] == "$" and value[1:2] != "{"
            and not _ENV_NAME.fullmatch(value[1:]))


#: English words that end in "key" and name no key: under MONKEY_PATCH or
#: TURKEY a plain word stays a word.
_NOT_A_KEY = frozenset({"monkey", "turkey", "donkey", "jockey", "hockey",
                        "whiskey", "lackey", "flunkey"})


def _secret_key(key: str) -> bool:
    """Whether a settings key names or hints a secret, for the key/value
    rule: by either tier (_NAMES_A_SECRET, SECRETISH), by a word glued to
    "key" -- MASTERKEY, masterKey, SigningKey, which SECRETISH's "key" does
    not reach without a separator; any word but the English ones that end
    so (_NOT_A_KEY), wider than _glued_key's heads, since a value under an
    unknown key is kept from a summary and not masked --, or by a
    character of another script
    (_outside_latin): DB_P\u0410SSWORD with a Cyrillic A reads as
    DB_PASSWORD and matches neither tier, so fail closed."""
    if _outside_latin(key) or _NAMES_A_SECRET.search(key) \
            or SECRETISH.search(key):
        return True
    for word in _KEY_WORDS.findall(key):
        word = word.lower()
        if word.endswith("keys"):
            word = word[:-1]
        if word.endswith("key") and word not in _NOT_A_KEY:
            return True
    return False


def _outside_latin(key: str) -> bool:
    """Whether ``key`` holds a character outside ASCII that is no Latin
    letter -- the homoglyph of one: a Cyrillic or Greek letter, a fullwidth
    one (named FULLWIDTH LATIN, not LATIN), a digit of another script, a
    symbol, an emoji, a character with no name. A Latin letter is not:
    GR\u00d6SSE, STRASSE_N\u00c4HE and MA\u00dfE are how the operator's keys are
    written, and calling each a secret withheld their plain values from
    everything said about the file. Read as the text composes it (NFC), so
    a decomposed \u00d6 is the Latin letter, and a combining mark left over is
    no letter."""
    if key.isascii():
        return False
    return any(not char.isascii() and not (
        char.isalpha() and unicodedata.name(char, "").startswith("LATIN "))
        for char in unicodedata.normalize("NFC", key))


#: What makes a code a public one, an outcome or a transform: ZIP_CODE,
#: POSTCODE, COUNTRY_CODE, EXIT_CODE, HTTP_STATUS_CODE, BARCODE, ENCODE.
_PUBLIC_CODES = frozenset({
    "zip", "postal", "post", "area", "dial", "country", "region",
    "language", "lang", "locale", "currency", "iso", "exit", "status",
    "error", "err", "http", "response", "return", "result", "reason", "bar",
    "qr", "op", "uni", "en", "de", "re", "trans", "geo", "char", "key",
    "short", "source"})


def _names_a_code(key: str) -> bool:
    """Whether ``key`` names a code no public one is: a word "code" or
    "codes", or one ending in it (ALARMCODE, alarmCode), whose qualifier --
    what is glued in front of it, else the word before it -- is none of
    _PUBLIC_CODES. BACKUP_CODE=48291937 and ALARM_CODE=4711 are a PIN by
    another name; CODE alone is one too."""
    words = [word.lower() for word in _KEY_WORDS.findall(key)]
    for index, word in enumerate(words):
        stem = word[:-1] if word.endswith("codes") else word
        if not stem.endswith("code"):
            continue
        qualifier = stem[:-4] or (words[index - 1] if index else "")
        if qualifier not in _PUBLIC_CODES:
            return True
    return False


def _defaulted(value: str) -> "tuple[list, str]":
    """The names of the shell defaults ``value`` is, outermost first, and
    the word at their core: "${A:-${B:-x}}" is (["A", "B"], "x"); a value
    that is none is ([], value). Each level reads its own opening only, so
    a chain of any depth is read once. A brace that closes early leaves a
    core no rule calls harmless: "${A:-x}y}" is "x}y"."""
    names, at, end = [], 0, len(value)
    while end - at > 3 and value[end - 1] == "}":
        match = _EXPANSION_OPEN.match(value, at, end - 1)
        if not match:
            break
        names.append(match.group(1) or match.group(3))
        at, end = match.end(), end - 1
    return names, value[at:end]


def _default_words(text: str):
    """(name, word) of every shell default in ``text`` whose word holds no
    other expansion, wherever it stands -- "postgres://u:${DB_PASS:-x}@h"
    too. One pass with a stack of what is open; a word holding one (the
    outer word of ${A:-${B:-x}}) is not yielded, so what is yielded is
    disjoint and the work is linear however deep the nesting."""
    stack = []          # [name or None, where its word starts, holds one]
    for match in _EXPANSION_PART.finditer(text):
        if match.group(0) != "}":
            if stack:
                stack[-1][2] = True
            stack.append([match.group(1) or match.group(3), match.end(),
                          False])
        elif stack:
            name, start, holds = stack.pop()
            if name is not None and not holds:
                yield name, text[start:match.start()]


def _setting_keys(text: str, starts) -> dict:
    """{start: key} of the settings values starting at ``starts``: the key
    of the line each starts on, as find_secrets read it (_KEYED_LINE)."""
    keys = {}
    if not starts:
        return keys
    last = max(starts)
    scope = text[:SECRET_SCAN_LIMIT + _SCAN_OVERHANG]
    lines = list(_lines(scope))
    for offset, line in lines:
        if offset > last:
            break
        match = _KEYED_LINE.match(line)
        if match and offset + match.start(3) in starts:
            keys[offset + match.start(3)] = match.group(1).strip()
    # A .properties value's key as Java reads it (_properties_values), which
    # _KEYED_LINE does not where it holds an escape: p\u0061ssword is the
    # key "password", and a plain word under it no harmless setting. Where
    # both read a key for one value, it is the same key but for its escapes.
    for key, start, _end in _properties_values(scope, lines):
        if start > last:
            break
        if start in starts:
            keys[start] = key
    return keys


def in_place_only(text: str, spans) -> "set[tuple[int, int]]":
    """The (start, end) of the settings values among ``spans`` (reason
    "setting", find_secrets' over ``text`` under a settings file's name)
    that are hidden where they stand only -- the preview, a warning's
    quote -- and not remembered for what else sift says about the file: "true", "info" and "none" of a .env were
    hidden in every sentence written about it, and inside "Construe" and
    "information" too.

    Fail closed: a value is harmless only where both say so -- the
    key/value rule (_harmless_setting) of the value as its format reads it
    (_settled), and sift's judgement of ``text`` under no file name, which
    every other file gets, finding nothing that overlaps it. The second
    sees what the first cannot: a password in the comment after a value, a
    token a setting's span begins with (_merged labels it a setting).
    A host application's viewer masks the same values in place only (_in_place)."""
    settings = {(start, end) for start, end, reason in spans
                if reason == "setting"}
    if not settings:
        return set()
    keys = _setting_keys(text, {start for start, _end in settings})
    candidates = [(start, end) for start, end in settings
                  if start in keys
                  and _harmless_setting(keys[start], _settled(text[start:end]))]
    if not candidates:
        return set()
    # Sorted and apart (find_secrets merges its spans; a bidi one may sit
    # inside another and is no secret): the first that ends after a value
    # starts is the only one that can overlap it.
    found = [(start, end) for start, end, reason in find_secrets(text)
             if reason != "bidi"]
    ends = [end for _start, end in found]
    alone = set()
    for start, end in candidates:
        after = bisect.bisect_right(ends, start)
        if after < len(found) and found[after][0] < end:
            continue
        alone.add((start, end))
    return alone


def too_short_to_find(raw: str) -> bool:
    """Whether a value found at a span is one hide_known cannot look for:
    there is something to repeat, and no form of it is four characters
    long (see _hide: hiding every "on", "1" or "dev" in a summary would
    take its words out). The forms of the value as its format reads it
    (_settled): the span "abc # rotated", or a line continued after a
    backslash, is four characters and more where "abc" is not, and the
    note said a value a model repeats was not repeated."""
    settled = _settled(raw)
    return bool(settled.strip()) and not any(
        len(form) >= 4 for form in _value_forms(settled))


def mask_spans(text: str, spans) -> str:
    """``text`` with the characters of every secret span (not a bidi one)
    replaced by its marker: the value's length and nothing else. Where it
    stands, so a value of one character is hidden as surely as one of
    forty -- which looking for it in free text cannot do."""
    out, at = [], 0
    for span in spans:
        if span.reason == "bidi" or span.start < at:
            continue
        out.append(text[at:span.start])
        out.append(_marker(text[span.start:span.end]))
        at = span.end
    out.append(text[at:])
    return "".join(out)


#: The most of someone else's words an error of sift's repeats -- a model
#: server's: an HTTP error's body, a stream's error, a line sift could not
#: read; an exception's --, in characters (repeatable): the 500 an HTTP
#: body was cut at before. A server that fails the chat call and quotes the
#: user message back put the file's DB_PASSWORD into the error, and an
#: Ollama stream's error was not cut at all.
SERVER_WORDS = 500
#: The most an error of sift's holds, in characters: the words it quotes,
#: at most SERVER_WORDS, and what sift says around them, at most as much
#: again (a test holds every frame of the model clients within
#: SERVER_WORDS). summarize.Summary.error, categories.CategoryResult.error,
#: llm.CallUsage.error and an extractor's failure are cut there.
ERROR_CHARS = 2 * SERVER_WORDS


def repeatable(text, limit: int = SERVER_WORDS) -> str:
    """``text`` -- a server's words, an exception's -- as sift may repeat it:
    one line, redacted as everything sift says about a file is (scrub, and
    hide_known inside remember_hidden: what the file taught it), and at
    most ``limit`` characters, marked "…" where it was cut.

    Redacted before it is cut, and cut at a space: a value cut first is a
    value the redaction no longer knows, and its first characters were
    published. Redacted before it is folded onto one line, too: folded
    first, the tab in "password: Kx9v<TAB>Q2mZ7pL4Wq8rTz" became the one
    blank that ends a value in prose, and "Q2mZ7pL4Wq8rTz" was said. What is read of ``text`` for it is at most four times
    ``limit``, ended at a space as well."""
    text = "" if text is None else str(text)
    reach, cut = 4 * limit, False
    if len(text) > reach:
        head = text[:reach + 1].rsplit(None, 1)
        text, cut = (head[0] if len(head) == 2 else ""), True
    said = " ".join(hide_said(text).split())
    if len(said) > limit:
        cut = True
    if cut:
        head = said[:max(limit - 1, 0)]
        space = head.rfind(" ")
        said = (head[:space] if space > 0 else "") + "…"
    return said


def redact_value(value, key: str = ""):
    """The value, or its shape where repeating it would publish a secret."""
    if not isinstance(value, str):
        return value
    # Bidi controls first, at the same boundary: they are the other way a
    # value can say one thing and show another.
    value = neutralize_bidi(value)
    if looks_like_a_credential(value, key):
        return _hide(value.strip())
    out = value
    for pattern, secret_group, _reason in _stages(out):
        out = pattern.sub(
            lambda m, pick=secret_group: _hide_group(m, pick(m)), out)
    return out


def _secret_span(match, group) -> "tuple[int, int]":
    """Where a stage's secret is in its match's string: ``group`` names a
    group of the match, or is the span its judge cut the value to
    (_sentence_value)."""
    return group if isinstance(group, tuple) else match.span(group)


def _hide_group(match, group) -> str:
    """The match, with one group -- or the span _secret_span says -- replaced
    by its shape; unchanged for None."""
    whole = match.group(0)
    if group is None:
        return whole
    start, end = (at - match.start() for at in _secret_span(match, group))
    return whole[:start] + _hide(whole[start:end]) + whole[end:]


def _other_assignment(match):
    return None if _clearly_not_a_secret(match.group(2)) else 2


#: What begins an argument that is no password itself: a shell's expansion
#: -- $VAR, ${VAR}, $(cat pw), `cat pw` --, a format's field (%s), troff's
#: markup (\fI), a placeholder ([password], {password}, <password>).
_NOT_THE_PASSWORD = "$%`\\[{<"


def _command_password(match, form: str = "any"):
    """Which group of a command-line match (_COMMAND_LINE_PASSWORDS) is the
    password -- single, double or bare --, or None where it is none: an
    expansion, a format's field, markup or a placeholder
    (_NOT_THE_PASSWORD), a name for the secret rather than the secret --
    "-ppassword" and "-storepass storepass" in a manual's synopsis, a plain
    word that names one (_NAMES_A_SECRET) --, or a value that is clearly
    something else (_clearly_not_a_secret, as for the other assignments).
    htpasswd's forms: "htpasswd -n" with -n and -b among its options,
    "htpasswd" with -b and without -n."""
    if form.startswith("htpasswd"):
        letters = re.sub(r"[^A-Za-z]", "", match.group("flags"))
        if "b" not in letters or ("n" in letters) != form.endswith("-n"):
            return None
    for group in ("single", "double", "bare"):
        value = match.group(group)
        if value is not None:
            break
    else:
        return None
    if (value[:1] in _NOT_THE_PASSWORD or _clearly_not_a_secret(value)
            or (_is_a_plain_word(value) and _NAMES_A_SECRET.search(value))):
        return None
    return group


def _name_value(match):
    return (3 if looks_like_a_credential(match.group(3), match.group(2))
            else None)


def _pgpass_value(match):
    # …and the last field still has to look like a credential, or
    # "localhost:8080:api:v1:health" and "2026-01-01:12:30:00:fertig" are
    # both .pgpass lines.
    value = match.group(2)
    return (None if _is_a_plain_word(value) or _clearly_not_a_secret(value)
            else 2)


def _sentence_value(match):
    """The sentence rule's secret: its value, to its last cell of a secret
    (_to_the_last_secret_cell) -- a span where that is shorter."""
    if not _sentence_value_is_secret(match):
        return None
    start, end = match.span("value")
    cut = _to_the_last_secret_cell(match.string, start, end)
    return "value" if cut == end else (start, cut)


#: What redact_value hides inside a text, in the order it hides it: the
#: pattern, which group of a match is the secret (None when this match is
#: not one), and why — the reason find_secrets reports. One table for both,
#: because two lists of rules that have to agree drift apart; every stage
#: sees the text as the stages before it left it.
_IN_TEXT_STAGES = (
    # The whole key block first: _CREDENTIAL_IN_TEXT matches its header, and
    # replacing that alone hid the words "-----BEGIN OPENSSH PRIVATE KEY-----"
    # and published the key underneath them.
    (_PEM_BLOCK, lambda match: 0, "private_key"),
    # A sentence that quotes one. Substituted in place, because the rest of
    # the sentence is what the summary is for.
    (_CREDENTIAL_IN_TEXT, lambda match: 0, "shape"),
    # And a sentence that *assigns* one. The key is inside the text here,
    # which is why scrub's key-less call on a summary saw nothing.
    *[(pattern, _other_assignment, "assignment")
      for pattern in _OTHER_ASSIGNMENTS],
    # A password on a command line, where its flag or position says so.
    *[(pattern, lambda match, form=form: _command_password(match, form),
       "assignment") for pattern, form, _words in _COMMAND_LINE_PASSWORDS],
    # name: X / value: Y, where X is the key Y has to be judged by.
    (_NAME_VALUE_PAIR, _name_value, "assignment"),
    (_PGPASS_LINE, _pgpass_value, "pgpass"),
    (_SECRET_IN_SENTENCE, _sentence_value, "sentence"),
)
#: A word one of _COMMAND_LINE_PASSWORDS' matches holds: a text with none
#: is scanned without them (_stages). No marker a stage leaves holds one.
_COMMAND_WORD = re.compile("|".join(dict.fromkeys(
    re.escape(word) for _pattern, _form, words in _COMMAND_LINE_PASSWORDS
    for word in words)))
#: _IN_TEXT_STAGES without the command lines'.
_STAGES_WITHOUT_COMMANDS = tuple(
    stage for stage in _IN_TEXT_STAGES
    if stage[0] not in {pattern for pattern, _form, _words
                        in _COMMAND_LINE_PASSWORDS})


def _stages(text: str) -> tuple:
    """The stages that can find something in ``text``: all of them where it
    holds a word of a command line (_COMMAND_WORD), else the others. One
    search of the text, where a scan per stage and value put a third on
    find_secrets' time for a module of dict literals."""
    return (_IN_TEXT_STAGES if _COMMAND_WORD.search(text)
            else _STAGES_WITHOUT_COMMANDS)


#: Characters that end a line, in every form a file can write one. A value
#: sift interpolates into a line-oriented rendering must not carry any of
#: them: NEL, LS and PS are line breaks to a text renderer and to most
#: models, and a lone CR is one to a terminal.
_LINE_BREAK = re.compile(r"[\r\n\x0b\x0c\x1c-\x1e\x85\u2028\u2029]+")
#: A fence opener at the start of a line ends a markdown report's structure.
_FENCE_LINE = re.compile(r"^(\s*)(```+|~~~+)", re.MULTILINE)


#: Characters that reorder what follows them without being visible. A file
#: writing "Der Betrag <U+202E>00,001<U+202C> EUR" renders as "100,00 EUR"
#: in a terminal, a browser and a JSON viewer alike, and sift copied them
#: straight into the summary, the cache and the report. The number a reader
#: sees would then be a number that is in no file anywhere.
#:
#: Replaced by a visible mark rather than dropped: that the file contained a
#: reordering control is a fact about it, and one worth seeing.
_BIDI_CONTROL = re.compile("[\u202a-\u202e\u2066-\u2069\u200e\u200f\u061c]")


_BIDI_MARK = "⟪bidi⟫"


def neutralize_bidi(value: str) -> str:
    """Make a reordering control visible instead of letting it reorder."""
    return _BIDI_CONTROL.sub(_BIDI_MARK, value)


def one_line(value: str) -> str:
    """A value that cannot add a line to whatever sift writes it into.

    sift renders facts one per line — ``Filename: x``, ``png_title: y``,
    ``- amount (integer)``, ``Attachments: z``. Every one of those values
    comes from the file. A newline in one lets the file write sift's next
    line: its own ``Detected type:``, its own ``## Note from sift``, its own
    metadata block. The forged line then sits *outside* the content fence,
    in the position where sift's own findings live, and inherits their
    authority — the fence is random per call and unbreakable, and this walks
    around it.

    Folded to a visible mark rather than dropped, because the fact that the
    value contained a line break is itself part of what the file is.
    """
    return neutralize_bidi(
        _LINE_BREAK.sub(" ⏎ ", str(value)).replace("\t", "    "))


#: sift's own metadata fields whose contents are *names* the file used, not
#: values it held: the list of a YAML document's top-level keys, a CSV's
#: column names, a workbook's sheet names, an INI's sections.
#:
#: This distinction cannot be made from the value alone, and getting it
#: wrong is not symmetrical. Judged as values under the field name
#: "top_level_keys" — which contains "key" — a config file's own structure
#: came back as "5 top-level keys: key, salt, <11 chars — not shown>,
#: <14 chars — not shown>, <17 chars — not shown>", which tells the reader
#: nothing and looks like the file was full of secrets.
#:
#: Only the *key*-based tiers are switched off inside these. A value that is
#: a credential on its own — an API key used as a column name — is still
#: caught, because that test never consulted the key.
_HOLDS_NAMES = frozenset({
    "top_level_keys", "keys", "columns", "column_names", "fields",
    "headers", "header", "names", "sections", "sheets", "sheet_names",
    "tables", "properties", "attributes", "tags", "labels", "entries",
    "slides", "chapters", "parts", "worksheets", "top_level_tables"})


def scrub(obj, key: str = ""):
    """Walk a value of any shape, redacting what must not be repeated.

    One place, because every round of this project has found one more route
    into a summary that nobody had guarded: the text preview, then the INI
    parse error, then the ragged-row note, then a shell script's key points,
    then a CSV column's examples, then an email's headers. There is no
    counting the routes; there is only the boundary they all cross.
    """
    if isinstance(obj, str):
        return redact_value(obj, "" if key in _HOLDS_NAMES else key)
    if isinstance(obj, dict):
        return {k: scrub(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        kind = type(obj)
        return kind(scrub(v, key) for v in obj)
    return obj


def redact_secrets(value: str) -> str:
    """Kept for the callers that quote a bare field with no key beside it."""
    return redact_value(value)


def redact_assignments(text: str, everything: bool = False) -> str:
    """Replace the values of secret-looking assignments with their shape.

    ``everything`` for formats where every line is a setting — .env and
    .properties — because there the key names are the useful part and the
    values are the reason the file is not in version control. A dotenv
    holding STRIPE_SECRET_KEY was previewed in full, and a *bare* .env file
    took a different route and leaked a second way.
    """
    out = []
    for line in text.splitlines():
        match = _ASSIGNMENT.match(line)
        if not match:
            out.append(line)
            continue
        name, sep, value = match.groups()
        # looks_like_a_credential, not SECRETISH: this line judged on the key
        # name alone — the rule the two tiers exist to replace — and consulted
        # neither _clearly_not_a_secret nor _ABOUT_A_SECRET, so every
        # exemption they exist for was bypassed on the preview path. It
        # produced "token_budget= <4 chars — not shown>",
        # "CHARS_PER_TOKEN= <1 chars — not shown>", "tokenizer= <22 chars —
        # not shown>" (bert-base-german-cased) and "signature= <14 chars —
        # not shown>" (a person's name): sixteen of twenty-two destroyed
        # values in an independent corpus came from this one comparison.
        #
        # After a quoted key the value is read as find_secrets reads it
        # (_assigned_value), by the name in the key's quotes (_key_name):
        # between the quotes of a quoted value, which stay, as the comma
        # after them does; an expression is no value.
        where = match.span(3) if everything else _assigned_value(match)
        if where is None or not (everything or looks_like_a_credential(
                line[where[0]:where[1]], _key_name(name))):
            out.append(line)
        elif where == match.span(3):
            out.append(f"{name}{sep} <{len(value)} chars — not shown>")
        else:
            start, end = where
            out.append(line[:start] + _marker(line[start:end]) + line[end:])
    return "\n".join(out)


# ---- where the secrets are ------------------------------------------------
class Span(NamedTuple):
    """Characters [start, end) of a raw text, and why they are masked.

    ``reason`` is one of: private_key, shape, assignment, pgpass, sentence,
    netrc, setting, repeated — or bidi, for a reordering control, which is
    reported so it can be shown rather than because it is a secret.
    """

    start: int
    end: int
    reason: str


class _Rope:
    """A text being rewritten stage by stage, remembering where every part
    of it came from in the text it started as.

    redact_value's stages each run on what the stage before left behind —
    bidi marks six characters long where one character stood, markers where
    a value stood — so an offset into the current text is not an offset into
    the file. The segments map one to the other.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        #: (current start, current end, raw start, raw end, verbatim)
        self.segments = [(0, len(text), 0, len(text), True)] if text else []
        self._starts = [0] if text else []

    def _raw(self, index: int, end: bool = False) -> int:
        probe = index - 1 if end else index
        segment = self.segments[bisect.bisect_right(self._starts, probe) - 1]
        start, _, raw_start, raw_end, verbatim = segment
        if verbatim:
            return raw_start + (index - start)
        return raw_end if end else raw_start

    def replace(self, regions: list) -> list:
        """Replace sorted, disjoint (start, end, text) regions; return the
        raw (start, end) each of them covered."""
        if not regions:
            return []
        covered = [(self._raw(a), self._raw(b, end=True) if b > a
                    else self._raw(a)) for a, b, _ in regions]
        pieces, segments = [], []
        position = shift = index = 0

        def keep(upto: int) -> None:
            nonlocal index
            while index < len(self.segments) and self.segments[index][0] < upto:
                start, end, raw_start, raw_end, verbatim = self.segments[index]
                low, high = max(start, position), min(end, upto)
                if low < high:
                    if verbatim:
                        segments.append((low + shift, high + shift,
                                         raw_start + (low - start),
                                         raw_start + (high - start), True))
                    else:
                        segments.append((low + shift, high + shift,
                                         raw_start, raw_end, False))
                if end > upto:
                    return
                index += 1

        for (a, b, text), (raw_a, raw_b) in zip(regions, covered):
            keep(a)
            pieces.append(self.text[position:a])
            pieces.append(text)
            if text:
                segments.append((a + shift, a + shift + len(text),
                                 raw_a, raw_b, False))
            shift += len(text) - (b - a)
            position = b
        keep(len(self.text))
        pieces.append(self.text[position:])
        self.text = "".join(pieces)
        self.segments = segments
        self._starts = [segment[0] for segment in segments]
        return covered


class _Pass(NamedTuple):
    text: str               # exactly what redact_value returns
    spans: list             # (raw start, raw end, reason)


def _redaction_pass(text: str, key: str = "",
                    whole: str = "shape", cells: bool = False) -> _Pass:
    """redact_value's stages, run on a rope: the same result, and where in
    ``text`` each hidden value was. Nothing is recorded as hidden for this
    file — this looks, it does not learn. With ``cells``, a value that is a
    credential as a whole -- judged whole, as ever -- is found to its last
    cell of a secret (_to_the_last_secret_cell): the rest of a keyed line
    runs on over a tab into the next cell of a sheet row."""
    rope = _Rope(text)
    rope.replace([(m.start(), m.end(), _BIDI_MARK)
                  for m in _BIDI_CONTROL.finditer(text)])
    current = rope.text
    if looks_like_a_credential(current, key):
        lead = len(current) - len(current.lstrip())
        stripped = current.strip()
        start = rope._raw(lead)
        end = rope._raw(lead + len(stripped), end=True)
        if cells:
            end = _to_the_last_secret_cell(text, start, end)
        return _Pass(_marker(stripped), [(start, end, whole)])
    spans = []
    for pattern, secret_group, reason in _stages(rope.text):
        regions = []
        for match in pattern.finditer(rope.text):
            group = secret_group(match)
            if group is not None:
                start, end = _secret_span(match, group)
                regions.append((start, end,
                                _marker(match.string[start:end])))
        spans.extend((a, b, reason) for a, b in rope.replace(regions))
    return _Pass(rope.text, spans)


#: A JSON member whose value is a string: the key is what the value is
#: judged by, which is what scrub does to a parsed file's metadata.
#: Any whitespace around the colon and no length on either side, because the
#: parsed document has neither: a pretty-printer puts a line break before
#: the value, an aligner forty blanks, and a key of 67 characters or a
#: private key of 4,320 is judged by scrub like any other. Capping them
#: here published exactly the values the summary of the same file hid.
#:
#: One match per string that could be a key (_json_members): its value is
#: group 3, None when no member follows. The match ends where the next try
#: has to start -- after the value, or at the key's closing quote, or where
#: a key that never closes stops -- so each string is read once. A pattern
#: for the member alone restarted at every quote after a failed try, and
#: every quote inside a string that did not close on its line is escaped:
#: each restart there read the rest of the line again only to fail the same
#: way -- 1.1 s for a line of 8,000 escaped quotes, four times as long per
#: doubling (test_secret_spans.JsonScanTest).
#:
#: A value in single quotes too, and "=>" as well as ":" between the two
#: (_QUOTED_MEMBER has the members of a key in single quotes).
_MEMBER_SHAPE = (r'"((?:[^"\\\n]|\\.)+)'
                 r"""(?:"\s*(?::|=>)\s*(["'])((?:(?!\2)[^\\\n]|\\.)*)\2)?""")
_JSON_MEMBER = re.compile(_MEMBER_SHAPE)
#: The same, for a key in single quotes: a Python dict, a JavaScript
#: object, YAML's flow mappings, a PHP array, a Ruby or a Perl hash. Only
#: double quotes and ":" were read, and {'password': '...'} and
#: ['password' => '...'] were no member. A scan of
#: its own, beside _JSON_MEMBER's and not one pattern for both: an
#: apostrophe opens a string to the scan that reads single quotes -- "Don't
#: commit {"auth": "..."}" --, and one scan for both let it take the JSON
#: after it for its text; the member was found no more. Each scan steps
#: over only its own kind of quote, so each finds what it found alone.
_QUOTED_MEMBER = re.compile(_MEMBER_SHAPE.replace('"', "\x00")
                            .replace("'", '"').replace("\x00", "'"))
#: The tokens _json_array_values reads: a string -- group 2 its closing
#: quote, empty where the line ends first -- or a bracket.
_JSON_TOKEN = re.compile(r'"((?:[^"\\\n]|\\.)*)("?)|[\[\]{}]')
_JSON_BRACKET = re.compile(r"[\[\]{}]")
#: What follows a string that is a member's key. Only ever matched right
#: after a closing quote, and the lookbehind says so: without it, search()
#: over a run of blanks tries every position in the run and each try
#: consumes the rest of it — 1.3 s on 30,000 spaces, the quadratic shape
#: test_backtracking exists to catch. With it a try can only start where
#: a run does, and the match after a quote is the same as before.
_JSON_COLON = re.compile(r"(?<!\s)\s*:")
#: A YAML block scalar's indicator: the value is the indented lines below.
_BLOCK_SCALAR = re.compile(r"[|>][1-9+-]{0,2}[ \t]*(?:#.*)?$")
#: An assignment line as find_secrets reads one. Longer keys than the
#: preview's _ASSIGNMENT: see _ASSIGNMENT_SHAPE.
_KEYED_LINE = re.compile(_ASSIGNMENT_SHAPE.format(longest=1024))


def _lines(text: str):
    """(offset, line) for every line, split the way str.splitlines splits —
    which is how redact_assignments reads a text."""
    offset = 0
    for piece in text.splitlines(keepends=True):
        yield offset, piece.rstrip("\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")
        offset += len(piece)


def _json_members(text: str):
    """(key, start, end) of each JSON member whose value is a string: the
    value's span, judged by the key. The members a pattern for the member
    alone finds with finditer -- the same ones, and in one pass (see
    _JSON_MEMBER) --, then those of a key in single quotes, in a pass of
    their own (_QUOTED_MEMBER)."""
    for pattern in (_JSON_MEMBER, _QUOTED_MEMBER):
        for match in pattern.finditer(text):
            if match.group(3) is not None:
                yield match.group(1), match.start(3), match.end(3)


def _json_tokens(text: str, start: int = 0):
    """(match, token) of each string and bracket from ``start``, token "\\""
    for a string: what finditer of ``"…"|[\\[\\]{}]`` yields from there, in
    one pass. A caller that reads a text in pieces starts inside it, also
    inside a string (a host application's windows over a large file).

    That pattern, where a string did not close on its line, tried again
    from the next character, and from every escaped quote in the string --
    each such try reads the rest of the line and fails as the first did,
    since every quote in it is escaped (the quadratic _JSON_MEMBER
    describes). What those tries found were the brackets inside the
    string; they are yielded here from the one pass over it.
    """
    for match in _JSON_TOKEN.finditer(text, start):
        closed = match.group(2)
        if closed is None:
            yield match, match.group(0)
        elif closed:
            yield match, '"'
        else:
            for bracket in _JSON_BRACKET.finditer(text, match.start() + 1,
                                                  match.end()):
                yield bracket, bracket.group(0)


def _json_array_values(text: str):
    """(start, end, key) of every string in an array that is the value of a
    member named ``key`` — in nested arrays too, as scrub hands a list's key
    down to everything in it, and not in an object inside the array, whose
    members carry keys of their own.

    One pass over the tokens with a stack of what is open, so the work is
    linear however deeply the arrays nest. Text that is not JSON leaves the
    stack in no particular state, which costs nothing: a string is judged
    only when a key is in force, and the judgement is looks_like_a_credential
    either way.
    """
    stack = []          # [bracket, key]: the key in force, or the last seen
    for match, token in _json_tokens(text):
        if token in "{[":
            key = stack[-1][1] if stack else None
            stack.append([token, key if token == "[" else None])
        elif token in "]}":
            if stack and stack[-1][0] == ("[" if token == "]" else "{"):
                stack.pop()
            else:
                stack = []              # not the structure it looked like
        elif stack and stack[-1][0] == "{":
            if _JSON_COLON.match(text, match.end()):
                stack[-1][1] = match.group(1)
        elif stack and stack[-1][1] is not None:
            yield match.start(1), match.end(1), stack[-1][1]


def _block_lines(lines: list, index: int):
    """(start, end) of the content of each line of the YAML block scalar
    whose indicator ends ``lines[index]``: the lines below it that are
    indented further, blank lines included, up to the first that is not."""
    offset, line = lines[index]
    indent = len(line) - len(line.lstrip(" \t"))
    for offset, line in lines[index + 1:]:
        content = line.strip()
        if not content:
            continue
        if len(line) - len(line.lstrip(" \t")) <= indent:
            return
        start = offset + len(line) - len(line.lstrip(" \t"))
        yield start, start + len(line.strip(" \t"))


def _block_spans(scope: str, lines: list, index: int, key: str):
    """The spans of a YAML block scalar's value, judged as one value by its
    key — which is how the parsed file reaches scrub — and line by line for
    what the in-text stages find when the whole is no credential."""
    parts = list(_block_lines(lines, index))
    if not parts:
        return
    folded = lines[index][1].rstrip().split("#", 1)[0].strip()[:1] == ">"
    value = (" " if folded else "\n").join(scope[a:b] for a, b in parts)
    if looks_like_a_credential(value.strip(), key):
        for start, end in parts:
            yield start, end, "assignment"
        return
    for start, end in parts:
        for a, b, why in _redaction_pass(scope[start:end], key,
                                         "assignment").spans:
            yield start + a, start + b, why


def _judged_key(key: str) -> str:
    """The key a value is judged by, the way scrub picks it: none, for a
    field that holds names rather than values."""
    return "" if key in _HOLDS_NAMES else key


def _basename(filename: str) -> str:
    return filename.replace("\\", "/").rsplit("/", 1)[-1].lower()


#: Dotenv files by another name: direnv's, Flask's, Docker's --env-file.
_DOTENV_NAMES = frozenset({".envrc", ".flaskenv", "env.list"})
#: The stages a dotenv is named for without its dot: env.production.
_DOTENV_STAGES = frozenset({"local", "dev", "development", "prod",
                            "production", "stage", "staging", "test",
                            "testing"})


def _settings_format(name: str) -> Optional[str]:
    """"properties" for a .properties, "dotenv" for a .env, a prod.env or a
    .env.local -- and .env-prod, .env_local, env.production, .envrc,
    .flaskenv and env.list --, None for any other
    (lowercased) base name: not env.sh or env.py, which are code, nor a
    bare "env", which is how an upload is written (is_settings_file)."""
    if name.endswith(".properties"):
        return "properties"
    if (name == ".env" or name.endswith(".env")
            or name.startswith((".env.", ".env-", ".env_"))
            or name in _DOTENV_NAMES
            or (name.startswith("env.") and name[4:] in _DOTENV_STAGES)):
        return "dotenv"
    return None


def _every_value_is_secret(name: str) -> bool:
    """A .env or a .properties: redact_assignments(everything=True) files."""
    return _settings_format(name) is not None


def is_settings_file(filename: str) -> bool:
    """Whether every value in the file of this name is a secret -- a .env,
    a .properties -- as find_secrets(text, filename) judges it. The name a
    reader knows the file by: an upload called .env is written to disk as
    "env", which is no settings name at all."""
    return _every_value_is_secret(_basename(filename or ""))


#: The quotes a dotenv value may be written in; each may span lines.
_QUOTES = "\"'`"
#: An inline comment in an unquoted dotenv value: a blank, then "#".
_INLINE_COMMENT = re.compile(r"[ \t]#")
#: A continuation: a backslash ending the line, the line break, and the
#: blanks the next line starts with -- all of which Java drops from a
#: .properties value. (A shell keeps the blanks; _value_forms looks for
#: each line of a value on its own as well, which covers both.)
_CONTINUED = re.compile(r"\\(?:\r\n|[\r\n\f])[ \t\f]*")
#: A URL's user information, between "://" and "@": a password after a
#: colon (redis://:pw@cache), or a key with no colon at all (a Sentry DSN).
#: Anywhere in a form, not only at its start: "jdbc:postgresql://app:pw@db"
#: and a line "url = redis://:pw@cache" of a value over several lines were
#: remembered whole only. A try starts at the first
#: letter of a scheme only (the lookbehind), and reads no further than the
#: next "/", which every later scheme's "://" holds: linear.
_USERINFO = re.compile(r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://([^/\s@]+)@",
                       re.IGNORECASE)
#: An HTTP credential after its scheme, as an Authorization header or a
#: setting holding one writes it: "Basic <base64>", "Bearer <token>".
_SCHEME_CREDENTIAL = re.compile(r"(?:basic|bearer)\s+(\S+)", re.IGNORECASE)
#: Base64 as a Basic credential writes it, padded to a multiple of four.
_BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}")


def _closing_quote(text: str, start: int) -> int:
    """Where the quote opened at ``text[start]`` closes, or -1: the next
    one of the same kind, skipping a double quote escaped with a backslash
    as dotenv does. Linear: each quote found is looked at once, and each
    backslash before it once."""
    quote, at = text[start], start + 1
    while True:
        at = text.find(quote, at)
        if at < 0 or quote != '"':
            return at
        before = at
        while before > start + 1 and text[before - 1] == "\\":
            before -= 1
        if (at - before) % 2 == 0:
            return at
        at += 1


def _continues(line: str) -> bool:
    """A line ending in an odd number of backslashes goes on on the next
    line; an even number is escaped backslashes."""
    return (len(line) - len(line.rstrip("\\"))) % 2 == 1


def _setting_end(scope: str, lines: list, index: int, start: int,
                 form: str) -> int:
    """Where the value of a settings file that starts at ``start``, on
    ``lines[index]``, ends: the end of its line -- or of a later one, where
    the file's format continues it there.

    A .env value in quotes runs to the closing quote, as dotenv reads it,
    wherever that is: a private key or a certificate is written over
    several lines, and only the first was a span, so the preview and a
    viewer's mask showed the rest of it. A quote that never closes is a
    value of one line, as dotenv reads that too. Any other value runs on
    over every line that ends in a backslash, as Java reads a .properties
    and a shell a .env it sources; quotes mean nothing in a .properties.
    The end of the line the value ends on, so a comment after the closing
    quote belongs to the span as it does on a line of its own.
    """
    offset, line = lines[index]
    end = offset + len(line)
    if form == "dotenv" and scope[start] in _QUOTES:
        close = _closing_quote(scope, start)
        while close >= end and index + 1 < len(lines):
            index += 1
            offset, line = lines[index]
            end = offset + len(line)
        return end
    while _continues(line) and index + 1 < len(lines):
        index += 1
        offset, line = lines[index]
        end = offset + len(line)
    return end


# ---- a .properties, as java.util.Properties.load reads it ----------------
#: The blanks of a .properties line: before a key, after it, and at the
#: start of a line another goes on on.
_PROPERTIES_BLANKS = " \t\f"
#: What a backslash before one of these letters stands for; before any
#: other character (=, :, #, !, a blank, a backslash) it is that character.
_PROPERTIES_ESCAPES = {"t": "\t", "n": "\n", "r": "\r", "f": "\f"}
#: The four hex digits of a \uXXXX escape.
_HEX_UNIT = re.compile(r"[0-9A-Fa-f]{4}")


def _properties_unescaped(text: str) -> str:
    """``text`` -- a key or a value of a .properties -- as Java reads it:
    \\uXXXX as its UTF-16 unit, a high and a low surrogate as the one
    character they make, \\t \\n \\r \\f as those, a backslash before any
    other character as that character, a backslash before a line break as
    a line going on (the break and the blanks after it dropped), and a
    backslash that ends ``text`` as nothing.

    Where Java throws -- a \\u without four hex digits -- or keeps what no
    UTF-8 text can hold -- a lone surrogate --, the escape stays as it is
    written: sift reads on, and a form holding a lone surrogate would break
    every writer that encodes it. One pass: each backslash is read once.
    """
    if "\\" not in text:
        return text
    out, at, length = [], 0, len(text)
    while True:
        slash = text.find("\\", at)
        if slash < 0:
            out.append(text[at:])
            return "".join(out)
        out.append(text[at:slash])
        if slash + 1 == length:
            return "".join(out)
        char, at = text[slash + 1], slash + 2
        if char == "u":
            read, at = _unicode_escape(text, slash)
            out.append(read)
        elif char in "\r\n":
            if char == "\r" and text.startswith("\n", at):
                at += 1
            while at < length and text[at] in _PROPERTIES_BLANKS:
                at += 1
        else:
            out.append(_PROPERTIES_ESCAPES.get(char, char))


def _unicode_escape(text: str, slash: int) -> "tuple[str, int]":
    """What the "\\u" at ``slash`` reads as, and where ``text`` goes on
    after it (_properties_unescaped)."""
    unit = _utf16_unit(text, slash)
    if unit is None:
        return "\\u", slash + 2
    if 0xD800 <= unit < 0xDC00:
        low = _utf16_unit(text, slash + 6)
        if low is not None and 0xDC00 <= low < 0xE000:
            return (chr(0x10000 + (unit - 0xD800) * 0x400 + low - 0xDC00),
                    slash + 12)
    if 0xD800 <= unit < 0xE000:
        return text[slash:slash + 6], slash + 6
    return chr(unit), slash + 6


def _utf16_unit(text: str, at: int) -> Optional[int]:
    """The UTF-16 unit of the \\uXXXX at ``at``, None where none is."""
    if not text.startswith("\\u", at) \
            or not _HEX_UNIT.fullmatch(text, at + 2, at + 6):
        return None
    return int(text[at + 2:at + 6], 16)


def _logical_line(lines: list, index: int, column: int = 0):
    """(index, offset, character) of each character of the .properties
    logical line that begins at ``lines[index]`` -- from ``column`` of it
    on --, its leading blanks left out: a line ending in an odd number of
    backslashes goes on on the next (_continues), that backslash, the line
    break and the blanks the next line begins with dropped -- also where no
    next line follows. Lazily: a reader stops where it has what it
    needs."""
    while True:
        offset, line = lines[index]
        goes_on = _continues(line)
        rest = line[column:]
        begin = column + len(rest) - len(rest.lstrip(_PROPERTIES_BLANKS))
        for at in range(begin, len(line) - goes_on):
            yield index, offset + at, line[at]
        if not goes_on or index + 1 >= len(lines):
            return
        index, column = index + 1, 0


#: "export " before a key, as a .env sourced by a shell writes one
#: (_ASSIGNMENT_SHAPE reads it so too). Anchored: read with .match it is the
#: same, and a search over a line of blanks is tried once, not once per
#: blank (test_backtracking: 64x the CPU for 8x the blanks).
_EXPORTED = re.compile(r"\A[ \t\f]*export[ \t\f]+")


def _properties_entry(lines: list, index: int):
    """The entry whose logical line begins at ``lines[index]``, as
    java.util.Properties.load reads it: (key, value line, value offset) --
    the key decoded (_properties_unescaped), the index of the line its
    value begins on and where, as an offset into the text; (key, None,
    None) for a key with no value. None for a blank line and a comment,
    whose first character after blanks is "#" or "!" (a comment goes on on
    no next line).

    The key runs to the first "=", ":" or blank no backslash escapes; the
    value begins after the blanks that follow it and, where a blank ended
    the key, one "=" or ":" and the blanks after that.

    One reading Java has not: "export KEY=value" is KEY's, where "export"
    is followed by a key and its "=" or ":" -- a .env under a .properties
    name, as every settings file is read (_ASSIGNMENT_SHAPE). To Java it is
    the key "export" and the value "KEY=value", and a model hands back the
    value after the "="; "export all of it" stays Java's."""
    exported = _EXPORTED.match(lines[index][1])
    if exported:
        entry = _read_entry(lines, index, exported.end())
        if entry is not None and entry[3]:
            return entry[:3]
    entry = _read_entry(lines, index)
    return None if entry is None else entry[:3]


def _read_entry(lines: list, index: int, column: int = 0):
    """_properties_entry's reading from ``column`` of ``lines[index]``:
    (key, value line, value offset, whether an "=" or ":" separates the
    two), or None for a blank line or a comment."""
    chars = _logical_line(lines, index, column)
    first = next(chars, None)
    if first is None or first[2] in "#!":
        return None
    key, escaped, separated = [first[2]], first[2] == "\\", None
    if first[2] in "=:" or first[2] in _PROPERTIES_BLANKS:
        key, separated = [], first[2] in "=:"
    else:
        for _where, _at, char in chars:
            if not escaped and (char in "=:" or char in _PROPERTIES_BLANKS):
                separated = char in "=:"
                break
            key.append(char)
            escaped = char == "\\" and not escaped
    name = _properties_unescaped("".join(key))
    if separated is None:
        return name, None, None, False
    for where, at, char in chars:
        if char in _PROPERTIES_BLANKS:
            continue
        if not separated and char in "=:":
            separated = True
            continue
        return name, where, at, separated
    return name, None, None, separated


def _properties_values(scope: str, lines: list,
                       where: str = "the secret scan"):
    """(key, start, end) of every value of a .properties, the key as Java
    reads it (_properties_entry) and the value's span on the raw text, over
    every line it goes on on (_setting_end). Each line is read once: the
    lines a key or a value goes on on belong to it, and are no entry of
    their own. ``where`` names the reading to the clock."""
    covered = 0
    for index, (offset, _line) in enumerate(lines):
        if index % 1024 == 0:
            deadline.check(where)
        if offset < covered:
            continue
        entry = _properties_entry(lines, index)
        if entry is None:
            continue
        key, where, start = entry
        if where is None:
            covered = _setting_end(scope, lines, index, offset, "properties")
            continue
        covered = _setting_end(scope, lines, where, start, "properties")
        yield key, start, covered


def properties_keys(text: str) -> "list[str]":
    """The keys of the settings of a .properties, as Java reads them
    (_properties_values): each once, in the order it first appears, none
    empty, and a character that is not printable -- a line break, a bidi
    control a key's \\uXXXX decodes to -- as that escape, so a list of them
    stays one line and reads as it is ordered."""
    keys = dict.fromkeys(
        key for key, _start, _end
        in _properties_values(text, list(_lines(text)), "the settings' keys"))
    return ["".join(char if char.isprintable() else f"\\u{ord(char):04x}"
                    for char in key)
            for key in keys if key]


def _settled(raw: str) -> str:
    """A value found at a span, as the reader of its format has it: inside
    its quotes, or before an inline comment, a line it continues on joined
    to it."""
    value = raw.strip()
    if value[:1] in _QUOTES:
        close = _closing_quote(value, 0)
        if close > 0:
            return value[1:close]
    return _CONTINUED.sub("", _INLINE_COMMENT.split(value, 1)[0].rstrip())


def _value_forms(raw: str, key: Optional[str] = None) -> "set[str]":
    """Every form in which a value found at a span may be repeated,
    whatever its length.

    What _hide remembers -- the value trimmed of quotes, and of the
    punctuation around it -- of the span as it stands and of the value as
    its format reads it (_settled): "Kx9vQ2mZ7pL4 # rotated" is the span
    and Kx9vQ2mZ7pL4 is what a model hands back. Each line of a value over
    several lines on its own, since a model quotes the line it means and a
    preview joins them with blanks. And the user information of every URL
    in it, and the password in that: redis://:pw4Kx9vQ2@cache is a
    credential because of the nine characters a model can name by
    themselves.

    And the forms a model writes that the file does not: the word of a shell default, ${POSTGRES_PASSWORD:-Gk4Wq8Zr2Mx7Tb},
    unless it is harmless under ``key`` -- the setting's; None for a value
    that is a secret by a rule, whose defaults are judged as a secret's --
    and under its variable's name (_harmless_value); a URL's user
    information and password percent-decoded, Kx9v%40Q2 as Kx9v@Q2; the
    credential of "Basic <base64>" or "Bearer <token>" without its scheme,
    and the password a Basic credential's base64 decodes to. A decoded form
    is the raw one where nothing decodes, and short ones are left out
    where every form is (remember_values).
    """
    value = raw.strip()
    settled = _settled(value)
    words = [word for name, word in _default_words(settled)
             if not (_harmless_value(key, word)
                     and _harmless_value(name, word))]
    secret = key is None or _secret_key(key)
    parts = [part for part in _parts(settled)
             if secret or _opaque_run(part)]
    forms = set()
    for form in (value, settled, settled.replace('\\"', '"'), *words,
                 *parts):
        forms.update((form.strip("\"'"), form.strip(_AROUND_A_VALUE)))
    for form in list(forms):
        if "\n" in form or "\r" in form:
            forms.update(line.strip().rstrip("\\").strip()
                         for line in form.splitlines())
    # And each piece of the value, blank to blank, that holds a run no word
    # holds (_opaque_run), without the punctuation and brackets around it
    # -- the first _OPAQUE_PIECES of them, in the value's order. A value
    # over three lines was remembered as the value and its lines, and a
    # model naming the run in the middle of a line was published;
    # 0.1.83 read the sides of an "=" so, and not a line's pieces. Of the
    # value as its format reads it and as the file writes it, both: a
    # continuation joined by _settled glues a run to the word before it.
    pieces = dict.fromkeys(
        piece.strip(_AROUND_A_VALUE + "()[]{}<>")
        for text in (settled, value) for piece in text.split()
        if _opaque_run(piece))
    forms.update(list(pieces)[:_OPAQUE_PIECES])
    # And as a .properties reads it (_properties_unescaped), which is how a
    # JSON string's escapes read too: p\u00e4ss is what a model hands back
    # as päss. After the lines, so a \n it decodes makes no line of
    # its own a form: GREETING="Hello\nWorld" does not make "Hello" one.
    decoded = _properties_unescaped(value)
    forms.update((decoded.strip().strip("\"'"),
                  decoded.strip().strip(_AROUND_A_VALUE)))
    # And each side of an "=" in it that holds a run no word holds
    # (_opaque_run). To Java "db.url Kx9vQ2mZ7pL4=x" is the value
    # "Kx9vQ2mZ7pL4=x" -- any "=" after the key's end is the value's --, and
    # a model reads the line as a "key=value" and hands back Kx9vQ2mZ7pL4;
    # base64 loses its padding the same way. A word on either side ("true"
    # of enabled=true) is no form: hide_known would take it out of every
    # sentence of the summary.
    for form in list(forms):
        if "=" in form:
            forms.update(side.strip().strip(_AROUND_A_VALUE)
                         for side in form.split("=") if _opaque_run(side))
    for form in list(forms):
        for userinfo in _USERINFO.finditer(form):
            for part in (userinfo.group(1),
                         userinfo.group(1).split(":", 1)[-1]):
                forms.update((part, _percent_decoded(part)))
        credential = _SCHEME_CREDENTIAL.fullmatch(form)
        if credential:
            forms.add(credential.group(1))
    for form in list(forms):
        password = _basic_password(form)
        if password:
            forms.add(password)
    forms = {spelled for form in forms for spelled in _spellings(form)}
    forms.discard("")
    return forms


#: How many opaque pieces of one value are forms of it (_value_forms): a
#: value of a thousand tokens is a blob, not a thousand secrets, and every
#: form is looked for in everything sift says about the file.
_OPAQUE_PIECES = 64
#: What separates the items of a list a setting holds.
_ITEM_SEPARATOR = re.compile(r"[,;]")


def _parts(value: str):
    """The parts of a value made of parts, each as a model names it on its
    own: every item of a list written with "," or
    ";" (API_KEYS=Kx9v...,Zt5Y...), and what follows the first ":" of an
    item -- the password of user:password (BASIC_AUTH=admin:Sonnenschein
    2026) -- unless the item is a URL, whose user information is a form of
    its own (_USERINFO), or holds a shell default (_default_words). Not of
    JSON, whose members find_secrets reads by their names. _value_forms
    keeps a part under a key that names or hints a secret, and under any
    other only a part no word holds by chance: ALLOWED_HOSTS=localhost,
    example.com stays words."""
    if value[:1] in "{[" or not any(mark in value for mark in ",;:"):
        return
    for item in _ITEM_SEPARATOR.split(value):
        item = item.strip()
        if not item:
            continue
        if item != value:
            yield item
        if ":" in item and "://" not in item and "${" not in item:
            yield item.split(":", 1)[1]


def _percent_decoded(text: str) -> str:
    """``text`` with its %XX escapes decoded, as a URL's reader has it; the
    text itself where it has none, or where they spell no UTF-8 (a "%"
    followed by no two hex digits stays as it is)."""
    if "%" not in text:
        return text
    try:
        return urllib.parse.unquote(text, errors="strict")
    except UnicodeDecodeError:
        return text


def _basic_password(form: str) -> str:
    """The password a Basic credential's base64 decodes to -- what follows
    the first ":" of "user:password" --, or "" where ``form`` is no
    base64 of printable UTF-8 holding a ":"."""
    if len(form) % 4 or not _BASE64.fullmatch(form):
        return ""
    try:
        decoded = base64.b64decode(form, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return ""
    if ":" not in decoded or not decoded.isprintable():
        return ""
    return decoded.split(":", 1)[1]


def _pgpass_passwords(text: str):
    """The fifth field of every line, after four colons that are not
    escaped with a backslash — the .pgpass format."""
    for offset, line in _lines(text):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        colons, index = 0, 0
        while index < len(line) and colons < 4:
            if line[index] == "\\":
                index += 2
                continue
            if line[index] == ":":
                colons += 1
            index += 1
        if colons == 4 and index < len(line):
            yield offset + index, offset + len(line)


def _netrc_passwords(text: str):
    """The token after every "password" and "account" — .netrc's grammar,
    where the value follows the word after nothing but whitespace."""
    tokens = list(re.finditer(r"\S+", text))
    for word, value in zip(tokens, tokens[1:]):
        if word.group(0) in ("password", "account"):
            yield value.start(), value.end()


def find_secrets(text: str, filename: str = "") -> "list[Span]":
    """Where the secrets in ``text`` are, as character offsets into it.

    Everything sift's redaction would hide, found in the raw string instead
    of replaced in a copy of it:

    - every stage of redact_value, vendor shapes included, over the text;
    - each ``key = value`` / ``key: value`` line, a YAML block scalar's
      lines under its key, each JSON string member however it is laid out,
      and every string in an array under a key, judged by that key the way
      scrub judges a parsed file's fields;
    - with a ``filename`` that says so: every value of a .env or a
      .properties (redact_assignments(everything=True)) -- the whole of it,
      over every line its format continues it on (_setting_end) --, every
      password of a .pgpass or a .netrc;
    - every further occurrence of a value found by the above, which is what
      hide_known does for a summary;
    - bidi controls, as spans of their own (reason "bidi") rather than
      replaced, so no offset moves.

    Spans start in the first SECRET_SCAN_LIMIT characters. One that starts
    there runs to its natural end, up to _SCAN_OVERHANG characters past the
    limit: a private key that begins just before it is masked whole, not up
    to the limit with its tail shown. Secret spans are sorted and never
    overlap one another; a bidi span may sit inside one.

    Not here: a CSV column's name as the key of its cells. A raw text puts
    no key beside a cell; where a table is read (read_table), judge each
    cell with redact_value(cell, column_name), as the extractor does.
    """
    return find_secrets_and_pieces(text, filename)[0]


def find_secrets_and_pieces(text: str, filename: str = ""):
    """find_secrets' spans, and its pieces: every finding _merged folded
    into a longer span (a repetition left out), once per place, sorted --
    the JSON member of a setting, the URL inside a keyed value, the
    password in the comment after a value. Masking needs the spans, which
    never overlap; what sift says about the file needs the pieces too,
    since a model hands back the piece it means (remember_values, a review). One scan for both."""
    if not isinstance(text, str):
        raise TypeError(f"find_secrets reads text, not "
                        f"{type(text).__name__}")
    scope = text[:SECRET_SCAN_LIMIT + _SCAN_OVERHANG]
    name = _basename(filename)
    found = list(_redaction_pass(scope).spans)
    settings = _settings_format(name)
    lines = list(_lines(scope))
    if settings == "properties":
        # Every value of a .properties as Java reads the file: a key holding
        # an escape (p\u0061ssword, my\=key, "pass\ word") or ending at
        # a blank was no key to _KEYED_LINE, and its value no setting.
        found.extend((start, end, "setting")
                     for _key, start, end in _properties_values(scope, lines))
    # Where the last settings value ends. A line before it is inside that
    # value, and its own "key=value" shape is the value's text: read again
    # from each such line, a value running on over every line of a file --
    # a backslash ending each -- was read to the end once per line.
    covered = 0
    for index, (offset, line) in enumerate(
            lines if settings != "properties" else ()):
        if index % 1024 == 0:
            deadline.check("the secret scan")
        if offset < covered:
            continue
        match = _KEYED_LINE.match(line)
        if not match:
            continue
        base = offset + match.start(3)
        if settings:
            covered = _setting_end(scope, lines, index, base, settings)
            found.append((base, covered, "setting"))
            continue
        # Judged by the name in a quoted key's quotes (_key_name), the value
        # as the key reader reads it (_assigned_value).
        key = _key_name(match.group(1).strip())
        if _BLOCK_SCALAR.match(match.group(3)):
            # "password: >" — the ">" is YAML's, and the password is on the
            # lines below, which the parsed file hands scrub as the value.
            found.extend(_block_spans(scope, lines, index, key))
            continue
        value = _assigned_value(match)
        if value is None:
            continue
        start, end = value
        keyed = _redaction_pass(line[start:end], key, "assignment",
                                cells=True)
        found.extend((offset + start + a, offset + start + b, why)
                     for a, b, why in keyed.spans)
    for count, (key, base, end) in enumerate(_json_members(scope)):
        if count % 1024 == 0:
            deadline.check("the secret scan")
        keyed = _redaction_pass(scope[base:end], _judged_key(key),
                                "assignment")
        found.extend((base + a, base + b, why) for a, b, why in keyed.spans)
    for count, (start, end, key) in enumerate(_json_array_values(scope)):
        if count % 1024 == 0:
            deadline.check("the secret scan")
        keyed = _redaction_pass(scope[start:end], _judged_key(key),
                                "assignment")
        found.extend((start + a, start + b, why) for a, b, why in keyed.spans)
    if name in (".pgpass", "pgpass.conf"):
        found.extend((a, b, "pgpass") for a, b in _pgpass_passwords(scope))
    if name in (".netrc", "_netrc"):
        found.extend((a, b, "netrc") for a, b in _netrc_passwords(scope))
    found = [span for span in found if span[0] < SECRET_SCAN_LIMIT]
    found.extend(_repetitions(scope, found))
    bidi = [Span(m.start(), m.end(), "bidi")
            for m in _BIDI_CONTROL.finditer(scope)]
    merged = _merged(found)
    whole = {(start, end) for start, end, _reason in merged}
    # One piece per place: the member rule and the sentence rule both read
    # '"password": "..."' now, and a piece is its
    # text, whichever rule found it.
    pieces = []
    for piece in sorted({Span(start, end, reason)
                         for start, end, reason in found
                         if end > start and reason != "repeated"
                         and (start, end) not in whole}):
        if not pieces or pieces[-1][:2] != piece[:2]:
            pieces.append(piece)
    return ([span for span in sorted(merged + bidi)
             if span.start < SECRET_SCAN_LIMIT], pieces)


def _repetitions(text: str, found: list):
    """Every further occurrence of a value already found — in both the forms
    _hide remembers, and not the values of a settings file: "true" in a .env
    is not a secret everywhere else in it, so a settings value is a span
    where it stands only. (What sift *says* about the file hides it wherever
    it occurs -- remember_values, for that file's summary alone.)

    All the values in one pass. Looking for each value on its own read the
    whole text once per value, and a file is allowed to hold one per line:
    36,000 distinct passwords in 400,000 characters took 21 seconds, with
    no clock looking at it.
    """
    forms = set()
    for start, end, reason in found:
        if reason == "setting":
            continue
        forms.update(form for form in _hidden_forms(text[start:end])
                     if len(form) >= 4)
    if not forms:
        return
    for start, end in _occurrences(text, forms):
        yield start, end, "repeated"


#: How many characters the one-pass scans read between looks at the clock.
_CLOCK_EVERY = 65_536


def _occurrences(text: str, forms) -> "list[tuple[int, int]]":
    """Where any of ``forms`` occurs in ``text``, as (start, end): at each
    position where one ends, the longest one ending there — every character
    inside an occurrence is inside one of these.

    Aho–Corasick: a trie of the forms with a failure link per node, read in
    one pass over the text, so the work is the text plus the forms however
    many there are. Overlapping occurrences are all covered, which hides at
    least what hide_known's replace-one-after-another hides.
    """
    return _matches(_automaton(forms), text)


class _Automaton(NamedTuple):
    """_occurrences' trie of forms: built once, read over any number of
    texts (hide_known_deep reads one per metadata value)."""
    goto: list
    fail: list
    longest: list


def _automaton(forms, where: str = "the secret scan") -> _Automaton:
    """The trie of ``forms``, with a failure link and the longest form
    ending at each node."""
    goto = [{}]
    fail = [0]
    longest = [0]
    for count, form in enumerate(forms):
        if count % 4096 == 0:
            deadline.check(where)
        node = 0
        for char in form:
            child = goto[node].get(char)
            if child is None:
                child = len(goto)
                goto[node][char] = child
                goto.append({})
                fail.append(0)
                longest.append(0)
            node = child
        longest[node] = max(longest[node], len(form))
    # Breadth first, so a node's failure target — which is shallower — is
    # finished before the node inherits the longest form ending there.
    queue = collections.deque(goto[0].values())
    while queue:
        node = queue.popleft()
        for char, child in goto[node].items():
            queue.append(child)
            back = fail[node]
            while back and char not in goto[back]:
                back = fail[back]
            fail[child] = goto[back].get(char, 0)
            longest[child] = max(longest[child], longest[fail[child]])
    return _Automaton(goto, fail, longest)


def _matches(automaton: _Automaton, text: str,
             where: str = "the secret scan") -> "list[tuple[int, int]]":
    """_occurrences, with the trie already built."""
    goto, fail, longest = automaton
    spans, node = [], 0
    for index, char in enumerate(text):
        if index % _CLOCK_EVERY == 0:
            deadline.check(where)
        while node and char not in goto[node]:
            node = fail[node]
        node = goto[node].get(char, 0)
        if longest[node]:
            spans.append((index + 1 - longest[node], index + 1))
    return spans


class _KnownAutomaton(NamedTuple):
    """_Automaton, and for hide_known_deep what a form ending at a node is:
    its length (``own``, 0 where none ends there), whether it is word-like
    (``word``), and the next node on the failure chain a form ends at
    (``out``, 0 where none does)."""
    goto: list
    fail: list
    longest: list
    own: list
    word: list
    out: list


def _known_automaton(forms, where: str) -> _KnownAutomaton:
    goto, fail, longest = _automaton(forms, where)
    own, word = [0] * len(goto), [False] * len(goto)
    for count, form in enumerate(forms):
        if count % 4096 == 0:
            deadline.check(where)
        node = 0
        for char in form:
            node = goto[node][char]
        own[node], word[node] = len(form), _word_like(form)
    # Breadth first, so a node's failure target is linked before it.
    out = [0] * len(goto)
    queue = collections.deque(goto[0].values())
    while queue:
        node = queue.popleft()
        queue.extend(goto[node].values())
        back = fail[node]
        out[node] = back if own[back] else out[back]
    return _KnownAutomaton(goto, fail, longest, own, word, out)


def _known_matches(automaton: _KnownAutomaton, text: str,
                   where: str) -> "list[tuple[int, int]]":
    """_matches, where a word-like form counts only as a token of its own:
    at each position, the longest form ending there that is opaque or
    stands between token edges -- "apps" of "xmy-apps" where "my-apps" is
    glued to the x. One pass; a form that is no token there costs one step
    to the next shorter one ending at the same place."""
    goto, fail, longest, own, word, out = automaton
    spans, node = [], 0
    for index, char in enumerate(text):
        if index % _CLOCK_EVERY == 0:
            deadline.check(where)
        while node and char not in goto[node]:
            node = fail[node]
        node = goto[node].get(char, 0)
        if not longest[node]:
            continue
        end = index + 1
        found = node if own[node] else out[node]
        while found:
            start = end - own[found]
            if not word[found] or _at_token_edges(text, start, end):
                spans.append((start, end))
                break
            found = out[found]
    return spans


def _at_token_edges(text: str, start: int, end: int) -> bool:
    """Whether text[start:end] is not glued to a letter or a digit."""
    return not (start and _TOKEN_CHAR.match(text, start - 1)) \
        and not _TOKEN_CHAR.match(text, end)


def _merged(found: list) -> "list[Span]":
    """Overlapping spans as one, keeping the reason of the one that starts
    first — a repetition yields to the finding it repeats."""
    spans = []
    for start, end, reason in sorted(
            (s for s in found if s[1] > s[0]),
            key=lambda s: (s[0], -s[1], s[2] == "repeated")):
        if spans and start < spans[-1].end:
            if end > spans[-1].end:
                spans[-1] = spans[-1]._replace(end=end)
            continue
        spans.append(Span(start, end, reason))
    return spans

