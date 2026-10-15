"""Extractor for .eml / RFC-822 email (headers, body, attachments)."""
from __future__ import annotations

import base64
import contextlib
import contextvars
import quopri
import re
import sys

from email.feedparser import BytesFeedParser
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import EmailPolicy, linesep_splitter
from pathlib import Path

from .. import deadline, zipguard
from ..detect import Detection, _magic, contradicts_its_name
from .active import looks_like_html
from .base import (_BIDI_CONTROL, one_line, Extracted, clip, decode_text,
                   human_size)
from .fallback import MAX_LISTED

MAX_BODY_CHARS = 100_000


#: An mbox separator is a "From " line at the start of a line that is followed
#: by an RFC-822 header. The lookahead is what distinguishes it from a body
#: sentence: "From the desk of Alice, hello." used to split a message in two —
#: a phantom entry appeared and the real message lost everything after that
#: line. Prose is not followed by "Subject:".
_MBOX_SEP = re.compile(
    rb"(?:\A|\r?\n)(From \S+[^\r\n]*\r?\n)(?=[A-Za-z][A-Za-z0-9-]*:)")


# ---- what one header may cost ----------------------------------------------

#: The most recipients an SMTP server has to take for one message (RFC 5321
#: 4.5.3.1.8), and the longest path it has to take for one of them
#: (4.5.3.1.3).
_RECIPIENTS = 100
_PATH_CHARS = 256
#: The characters of one header value Python's header parser is handed: an
#: address list of every recipient a server must take, each as long as a
#: path may be. A Subject is a line (RFC 5322 2.1.1: at most 998). The
#: longest header of 418 mails of every shape (Thunderbird's and a mail
#: parser's test messages) is an Autocrypt key of 3,062.
#:
#: The parser's cost grows with the square of a header: it keeps, for every
#: encoded-word, the rest of the header behind it
#: (_header_value_parser.get_encoded_word, "ew.cte = value"), and reads a
#: run of words, addresses or comments in the same way. A Subject of 64 KiB
#: of encoded-words held 133 MiB, one of 256 KiB was killed at 1 GiB; 1 MiB
#: of words took 83 s. At this bound, measured one header at a
#: time in a fresh process: the shortest encoded-words 28 MiB more, words,
#: an address list, comments, RFC 2231 continuations or parameters at most
#: 0.12 s. A longer value is read to it, and a note says so.
_MAX_HEADER_CHARS = _RECIPIENTS * _PATH_CHARS

#: The mail's bytes fed to the parser between two looks at the clock.
#: Python's parser has no clock of its own: 40,000 parts took 6 s to parse
#: before sift read a header.
_PARSE_CHUNK = 1 << 16

#: The parts of one mail sift walks: as many as it lists members of an
#: archive (zipguard).
_MAX_PARTS = zipguard.MAX_ZIP_ENTRIES
#: The attachments named in the metadata and the "Attachments:" line; the
#: count goes on past them, as an archive listing's does.
_MAX_LISTED = MAX_LISTED


class _Said:
    """What reading one mail could not do as the mail asked -- a header cut
    to the bound, one Python could not read, a part past the limit --, for
    the notes."""

    def __init__(self) -> None:
        self.cut: dict = {}             # header name -> its longest length
        self.unread: dict = {}          # what -> the exception's name
        self.more_parts = False
        self.boundary_twice = False
        self.structure = ""             # the exception the parse ended on
        self.parsing = False
        self.nested = False             # recursion ran out inside the parse
        self.made = 0                   # the parts the parser has made

    def unreadable(self, what: str, exc: BaseException) -> None:
        self.unread.setdefault(what, type(exc).__name__)

    def notes(self) -> list:
        notes = []
        if self.structure == "RecursionError":
            notes.append("this message nests deeper than Python's mail parser "
                         "follows (RecursionError); only its headers are read")
        elif self.structure:
            notes.append(f"Python's mail parser could not read this message's "
                         f"structure ({self.structure}); only its headers are "
                         f"read")
        elif self.nested:
            notes.append("this message nests deeper than Python's mail parser "
                         "follows (RecursionError); what lies deeper is read "
                         "as text")
        if self.cut:
            names = sorted(self.cut, key=lambda name: -self.cut[name])
            longest = self.cut[names[0]]
            said = ", ".join(_hval_name(name) for name in names[:3])
            if len(names) > 3:
                said += f" and {len(names) - 3} more"
            notes.append(
                f"the {said} header{'s' if len(names) > 1 else ''} "
                f"{'were' if len(names) > 1 else 'was'} cut to "
                f"{_MAX_HEADER_CHARS:,} characters (the longest held "
                f"{longest:,}); a Subject or an address list holds far "
                f"fewer, and sift read no further")
        for what, error in list(self.unread.items())[:6]:
            notes.append(f"sift could not read {what} of this message "
                         f"({error}); the rest is read as usual")
        if len(self.unread) > 6:
            notes.append(f"… and {len(self.unread) - 6} more elements sift "
                         f"could not read")
        if self.more_parts:
            notes.append(f"this message has more than {_MAX_PARTS:,} parts; "
                         f"sift read the first {_MAX_PARTS:,}")
        if self.boundary_twice:
            notes.append("this message names its MIME boundary twice; sift "
                         "read the parts after the first, and a mail reader "
                         "may show the ones after the other")
        return notes


_said_here: "contextvars.ContextVar[_Said | None]" = contextvars.ContextVar(
    "sift_mail_said", default=None)


@contextlib.contextmanager
def _saying():
    token = _said_here.set(_Said())
    try:
        yield _said_here.get()
    finally:
        _said_here.reset(token)


def _said() -> _Said:
    """The record of the mail being read -- a throwaway one outside it."""
    return _said_here.get() or _Said()


def _parsed(message, name: str, raw):
    """One header of ``message``, parsed once: its object, or None where
    Python's parser cannot read it (recorded for the notes).

    Cut to _MAX_HEADER_CHARS before the parser sees it. Remembered by the
    raw value it was made of, so a header asked for again -- the
    Content-Type get_content_type, get_param, get_boundary, get_body and
    iter_attachments each ask for -- is not parsed again: 10,000 parts cost
    130,000 parses, 13 a part. What is remembered is what is read of it
    (_Kept), not the parse: 40,000 parts' Content-Types and
    Content-Dispositions kept whole held 460 MiB."""
    memo = message.__dict__.setdefault("_sift_parsed", {})
    kept = memo.get(id(raw))
    if kept is not None and kept[0] is raw:
        return kept[1]
    said = _said()
    if isinstance(raw, str) and len(raw) > _MAX_HEADER_CHARS:
        cut = _said().cut
        cut[name] = max(cut.get(name, 0), len(raw))
    deadline.check("the headers of a mail")
    try:
        value = _Kept.of(message.policy.header_fetch_parse(name, raw))
    except deadline.DeadlineExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - the sender's header, not sift
        # RecursionError too: 1,000 nested comments in a From exceed the
        # parser's recursion. Or the parse of the structure has used it up,
        # 980 messages deep: the header is then read as missing, and what
        # it describes as text.
        if isinstance(exc, RecursionError) and said.parsing:
            said.nested = True
        else:
            said.unreadable(f"the {_hval_name(name)} header", exc)
        value = None
    memo[id(raw)] = (raw, value)
    return value


class _Kept(str):
    """A parsed header as sift and the email package read it once parsed:
    its text, its name and defects, and what its kind parsed out of it --
    a Content-Disposition's disposition (is_attachment), a Content-Type's
    parts and parameters, a transfer encoding. Not the parse tree the
    header object keeps for folding it, which nothing here does: that is
    what grows with the header."""

    _READ = ("name", "defects", "content_disposition", "content_type",
             "maintype", "subtype", "params", "cte")

    @classmethod
    def of(cls, header) -> "_Kept":
        kept = cls(header)
        for attribute in cls._READ:
            try:
                kept.__dict__[attribute] = getattr(header, attribute)
            except AttributeError:
                continue
        return kept


class _Message(EmailMessage):
    """An EmailMessage whose headers are parsed once each (_parsed), and
    of which a header Python's parser cannot read is a missing header, not
    an exception out of whatever asked for it -- get_body's
    is_attachment(), get_content_type(), get_filename(). Counted as the
    parser makes them (_parse)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _said().made += 1

    def get(self, name, failobj=None):
        name = name.lower()
        for key, raw in self._headers:
            if key.lower() == name:
                value = _parsed(self, key, raw)
                return failobj if value is None else value
        return failobj


class _Policy(EmailPolicy):
    """policy.default, with every header value cut to _MAX_HEADER_CHARS
    before it is parsed -- also where the stdlib asks for one without
    _Message.get (items(), get_all())."""

    def header_fetch_parse(self, name, value):
        if isinstance(value, str) and len(value) > _MAX_HEADER_CHARS:
            value = value[:_MAX_HEADER_CHARS]
        return super().header_fetch_parse(name, value)


_POLICY = _Policy(message_factory=_Message)
#: What multipart/signed writes its parts with (_written_size).
_SIGNED_POLICY = _POLICY.clone(max_line_length=0)


def _parse(data: bytes):
    """The message, handed to Python's parser _PARSE_CHUNK bytes at a time
    with a look at the clock between two, and no more once it has made
    _MAX_PARTS parts: what follows is not walked either, and parsing it cost
    the time -- 40,000 parts took 6 s to parse. A message the parser cannot
    follow -- nested deeper than its recursion goes -- is read for its
    headers alone, and a note says so."""
    parser, said = BytesFeedParser(policy=_POLICY), _said()
    said.parsing, made = True, said.made
    try:
        for start in range(0, len(data), _PARSE_CHUNK):
            deadline.check("a mail's structure")
            if said.made - made > _MAX_PARTS:
                said.more_parts = True
                break
            parser.feed(data[start:start + _PARSE_CHUNK])
        return parser.close()
    except deadline.DeadlineExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - RecursionError among them
        said.structure = type(exc).__name__
    finally:
        said.parsing = False
    return BytesParser(policy=_POLICY).parsebytes(data, headersonly=True)


@contextlib.contextmanager
def _element(what: str):
    """One element of a mail read on its own: what it cannot read is a note
    and the rest of the mail is read as usual. 462 of 91,849 fuzzed mails
    raised out of this extractor, and extract()'s boundary then reported
    the whole mail as an error -- the masquerade warning with it."""
    try:
        yield
    except deadline.DeadlineExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - the sender's, not sift's
        _said().unreadable(what, exc)


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    with path.open("rb") as fh:
        data = fh.read(max_bytes)
    byte_capped = det.size > max_bytes > 0
    with _saying() as said:
        if _is_mbox(data, path):
            ex = _mbox(data, det, byte_capped, len(data))
        else:
            ex = _single(data, det)
            if byte_capped:
                # Attachments live at the end of the message. Cut the file
                # short and they vanish silently — an email with two
                # attachments reported none.
                ex.truncated = True
                ex.note(f"read only the first {max_bytes} of {det.size} "
                        f"bytes; attachments and body may be incomplete")
                ex.metadata["attachments_note"] = ("message truncated; "
                                                   "count may be low")
        for note in said.notes():
            ex.note(note)
    return ex


def _is_mbox(data: bytes, path: Path) -> bool:
    if not data.startswith(b"From "):
        return False
    return path.suffix.lower() == ".mbox" or len(_MBOX_SEP.findall(data)) > 1


def _mbox(data: bytes, det: Detection, byte_capped: bool,
          bytes_read: int) -> Extracted:
    """Summarize a mailbox as a mailbox: every message, not just the first."""
    starts = [m.start(1) for m in _MBOX_SEP.finditer(data)]
    chunks = [data[a:b] for a, b in zip(starts, starts[1:] + [len(data)])]
    messages = []
    for raw in chunks:
        deadline.check("a mailbox")
        _, _, rest = raw.partition(b"\n")        # drop the "From " separator
        # mboxo quotes a body line that would look like a separator. The
        # replacement is a group reference: rb"\\1" is the two literal
        # characters, and wrote "\1my home office" into every such line.
        rest = re.sub(rb"^>(>*From )", rb"\1", rest, flags=re.MULTILINE)
        msg = _parse(rest)
        body, _kind, chosen = _read_body(msg)
        messages.append({
            "from": _hval(msg, "From"), "to": _hval(msg, "To"),
            "subject": _hval(msg, "Subject"), "date": _hval(msg, "Date"),
            "attachments": len(_list_attachments(msg, chosen)),
            "body_chars": len(body),
            "_body": body,
        })
    meta = {
        "mailbox": True,
        "message_count": len(messages),
        "messages": [{k: v for k, v in m.items() if k != "_body"}
                     for m in messages[:50]],
        "attachment_count": sum(m["attachments"] for m in messages),
        "senders": sorted({m["from"] for m in messages if m["from"]})[:20],
    }
    blocks = [f"# Mailbox with {len(messages)} message(s)"]
    for i, m in enumerate(messages[:20], 1):
        blocks.append(f"\n## Message {i}: {m['subject'] or '(no subject)'}\n"
                      f"From: {m['from']}   To: {m['to']}   Date: {m['date']}\n"
                      f"{m['_body'][:4000]}")
    text, truncated = clip("\n".join(blocks), MAX_BODY_CHARS)
    ex = Extracted(det.kind, det.mime, text=text, metadata=meta, rendered=True,
                   truncated=truncated or byte_capped)
    if len(messages) > 20:
        ex.note(f"showed the first 20 of {len(messages)} messages")
    if byte_capped:
        ex.note(f"read only the first {bytes_read} of {det.size} bytes "
                f"of the mailbox; later messages are missing")
    return ex


def _readable(data: bytes) -> bytes:
    """The message as UTF-8, whatever eight-bit encoding it arrived in.

    policy.default decodes headers as ASCII with replacement, so an ordinary
    German mail lost every umlaut in its Subject — "Gebühren" came out
    "Geb�hren" — and so did the sender's name, *even when the message
    declared its charset correctly*. sift already owns the fix and applies it
    to the same bytes under a different extension: decode_text reads cp1252
    and latin-1 and gets that Subject right.

    Only when the bytes are not already UTF-8, so a well-formed message is
    handed on untouched.
    """
    try:
        data.decode("utf-8")
        return data
    except UnicodeDecodeError:
        pass
    # The header block only. Re-encoding the whole message meant a body that
    # declares charset=iso-8859-1 was then handed UTF-8 bytes and decoded as
    # latin-1 a second time — "Gebühren" became "GebÃ¼hren". The body's own
    # charset is right about the body; it is the headers, which have no
    # charset and must be ASCII, that arrive eight-bit and unreadable.
    split = data.find(b"\r\n\r\n")
    length = 4
    if split < 0:
        split, length = data.find(b"\n\n"), 2
    if split < 0:
        head, body = data, b""
    else:
        head, body = data[:split + length], data[split + length:]
    text, _ = decode_text(head, False)
    return text.encode("utf-8", "surrogateescape") + body


def _single(data: bytes, det: Detection) -> Extracted:
    msg = _parse(_readable(data))

    headers = {
        "from": _hval(msg, "From"),
        "to": _hval(msg, "To"),
        "cc": _hval(msg, "Cc"),
        "subject": _hval(msg, "Subject"),
        "date": _hval(msg, "Date"),
        "message_id": _hval(msg, "Message-ID"),
        "reply_to": _hval(msg, "Reply-To"),
    }
    headers = {k: v for k, v in headers.items() if v}
    # Every other header the message carries, minus the routing and signing
    # noise. _hval reads seven names; message_from_bytes swallows every
    # header-shaped line into the header block, so anything else the writer
    # put there vanished — a meeting note beginning Date:/Subject:/Attendees:
    # lost the attendees entirely, from the metadata, the text and the
    # prompt. A line of the user's file must not disappear because sift did
    # not recognize its name.
    # One pass over the headers as the message holds them, not a get_all()
    # per distinct name. get_all() scans every header in the message, so a
    # mail with n distinct names cost n² — 15,000 of them took 7.5 s and
    # 40,000 took 63 s, from a 614 kB file. raw_items() gives the same
    # (name, value) pairs once, and parses none: items() parsed every one,
    # the routing noise skipped below and the seven _hval reads too.
    #
    # Every value, not the first: two "Attendees:" lines are two lists of
    # attendees, and keeping one silently lost the other — the same shape of
    # loss this block was added to fix. Four are shown of a name; the rest
    # are counted, not parsed.
    collected: "dict[str, list[str]]" = {}
    beyond: "dict[str, int]" = {}
    for name, raw in msg.raw_items():
        low = name.lower()
        if low in headers or low in _KNOWN or _NOISE.match(low):
            continue
        if len(collected) >= _MAX_OTHER_HEADERS and low not in collected:
            continue                     # bounded: the rest are counted below
        if len(collected.get(low, ())) >= 4:
            if str(raw).strip():
                beyond[low] = beyond.get(low, 0) + 1
            continue
        cleaned = _HEADER_BREAK.sub(" ⏎ ",
                                    _header_text(msg, name, raw)).strip()
        if cleaned:
            collected.setdefault(low, []).append(cleaned)

    other = {}
    for low, values in collected.items():
        if values and low not in other:
            rest = len(values) - 4 + beyond.get(low, 0)
            other[low] = " / ".join(values[:4]) + (
                f" / … and {rest} more" if rest > 0 else "")
    if other:
        headers["other"] = other

    body, body_kind, bodies = _read_body(msg)
    disagreement = _ALTERNATIVE_MISMATCH.search(body)
    attachments = _list_attachments(msg, bodies)
    # A forward carries the original as an attached message, and that
    # original is usually the whole point of the mail. Nothing of it reached
    # the summary or the prompt: the model was handed "Attachments:
    # (unnamed) (message/rfc822, 0 B)" and asked what the mail was about.
    forwarded = []
    with _element("a forwarded message"):
        forwarded = _forwarded_messages(msg, bodies=bodies)
    if forwarded:
        body = body.rstrip() + "\n\n" + "\n\n".join(forwarded)

    meta = {
        "headers": headers,
        "body_format": body_kind,
        "attachments": attachments[:_MAX_LISTED],
        "attachment_count": len(attachments),
    }
    # The parser records what it could not make sense of and nobody read it.
    # A message whose boundary does not match its declaration was reported as
    # an ordinary mail whose body happened to contain MIME headers — the
    # attachment lost, the raw source handed to the model as content, and
    # nothing to say the message was corrupt.
    defects = sorted({type(d).__name__ for part in _parts(msg)
                      for d in getattr(part, "defects", [])})
    if defects:
        meta["defects"] = defects

    clipped, truncated = clip(body, MAX_BODY_CHARS)
    text = _render(headers, clipped, attachments)
    ex = Extracted(det.kind, det.mime, text=text, metadata=meta, rendered=True,
                     truncated=truncated)
    # An attachment whose bytes are not what its name and Content-Type say.
    # Both of those come from whoever sent the message; the bytes do not.
    # "Rechnung_4711.pdf (application/pdf, 334 B)" was the whole report of a
    # Windows program.
    # warn, not note: this is a comparison, not a reading. The name and the
    # Content-Type are two claims by the sender and the bytes are the answer,
    # and the same contradiction one level up — about the file itself — has
    # been a warning all along. Measured on nine attachments, honest and
    # dishonest, including a .docx that really is a zip and an
    # application/octet-stream with an honest name: nine right, none wrong.
    disguised = [entry for entry in attachments
                 if entry.get("contents_are") or entry.get("name_reads_as")]
    for entry in disguised[:_MAX_LISTED]:
        if entry.get("name_reads_as"):
            # A reordering control in the name: a mail reader shows
            # "invoice<RLO>fdp.exe" as "invoiceexe.pdf", and the bytes
            # agree with the ".exe" nobody sees.
            ex.warn(f"the attachment “{one_line(entry['filename'])}” is "
                    f"shown by a mail reader as “"
                    f"{one_line(entry['name_reads_as'])}”: a reordering "
                    f"control hides that its name ends in “."
                    f"{one_line(_suffix(entry['filename']))}” — treat this "
                    f"message as untrusted", kind="masquerade")
        if entry.get("contents_are"):
            ex.warn(f"the attachment “{one_line(entry['filename'])}” is not "
                    f"what its name and type say: it is "
                    f"{entry['contents_are']} — treat this message as "
                    f"untrusted", kind="masquerade")
    if len(disguised) > _MAX_LISTED:
        ex.warn(f"… and {len(disguised) - _MAX_LISTED} more attachments are "
                f"not what their names and types say — treat this message "
                f"as untrusted", kind="masquerade")
    if disagreement:
        # A note as well as a line in the body. multipart/alternative means
        # "the same message, two renderings", and nothing enforces that: a
        # mail whose text part says 19,90 EUR to DE02 and whose HTML part
        # says 4.980,00 EUR to RO49 is valid, and every mail client shows
        # the HTML. sift reads the plain part, which is right — it is the
        # one that cannot carry a disguise — but reporting it as *the*
        # message with nothing said is how a reader pays the wrong amount
        # into the wrong account.
        ex.note(disagreement.group(0)[7:-1].strip()
                + " — treat the figures in this summary as the text version "
                  "only", kind="html_mismatch")
    if defects:
        ex.note("this message is malformed (" + ", ".join(defects[:3])
                + "); parts of it may be missing or shown as raw source")
    return ex


#: A header value occupies one line by definition — RFC 5322 folds long ones,
#: and unfolding rejoins them. A newline surviving into the value therefore
#: came from somewhere else: an RFC 2047 encoded-word decoding to one. A
#: Subject that decoded to "Invoice\nFrom: ceo@bigbank.example\nAttachments:
#: wire-authorisation.pdf (2.1 MB)" wrote both of those lines into sift's own
#: "# Email" block, where the model reads them as facts sift established —
#: while attachment_count sat at 0 two lines away. (The secret rules read
#: the fold as the break it stands for: base._SEPARATING.)
#: How many distinct unrecognised header names are kept. A mail has a
#: handful; this is a bound on work, and only the first twelve are rendered.
_MAX_OTHER_HEADERS = 256

_HEADER_BREAK = re.compile(r"[\r\n\x0b\x0c  ]+")


#: A header name is a token by RFC 5322 — printable ASCII without colon or
#: space. Anything else came from somewhere the parser was lenient about, and
#: it has no business being echoed.
_NAME_OK = re.compile(r"[^A-Za-z0-9!#$%&'*+._`|~^-]")


def _hval_name(name: str) -> str:
    return _NAME_OK.sub("?", str(name))[:60]


def _header_text(message, name: str, raw) -> str:
    """One header's value as text: as its kind parses it (_parsed), or, where
    Python cannot read it so, as a Subject would be read -- its
    encoded-words decoded -- or as the mail wrote it."""
    value = _parsed(message, name, raw)
    if value is not None:
        return str(value)
    text = "".join(linesep_splitter.split(str(raw)[:_MAX_HEADER_CHARS]))
    try:
        return str(_POLICY.header_factory["x-sift-text"](name, text))
    except Exception:  # noqa: BLE001 - as written, then
        return text.encode("utf-8", "surrogateescape").decode("utf-8",
                                                              "replace")


def _hval(msg, name: str) -> str:
    low = name.lower()
    for key, raw in msg.raw_items():
        if key.lower() == low:
            return _HEADER_BREAK.sub(" ⏎ ",
                                     _header_text(msg, key, raw)).strip()
    return ""


#: How much of an attached message to quote. Beyond this it is a document
#: in its own right and the outer mail is what was asked about.
_ATTACHED_LINES = 200


def _inner_message(part):
    """The message inside a message/rfc822 part, decoded if it was encoded.

    RFC 2045 forbids a content-transfer-encoding on message/rfc822 and
    gateways apply one anyway. Python then hands back an EmailMessage with no
    headers whose payload is still base64 — `hasattr(inner, "get")` passes,
    so the guard let it through and two lines of base64 went into the prompt
    while the sender, the subject and the body were lost.
    """
    inner = part.get_payload()
    inner = inner[0] if isinstance(inner, list) and inner else inner
    if hasattr(inner, "keys") and list(inner.keys()):
        return inner
    # get_payload(decode=True) returns nothing for a message/rfc822 part —
    # the encoded text sits in the header-less inner message Python built.
    raw = part.get_payload(decode=True)
    if not raw and hasattr(inner, "get_payload"):
        text = inner.get_payload()
        if isinstance(text, str):
            encoding = str(part.get("Content-Transfer-Encoding") or "").lower()
            try:
                if "base64" in encoding:
                    raw = base64.b64decode(text, validate=False)
                elif "quoted-printable" in encoding:
                    raw = quopri.decodestring(text.encode("ascii", "replace"))
                else:
                    raw = text.encode("utf-8", "replace")
            except Exception:                   # noqa: BLE001
                raw = b""
    if raw:
        try:
            reparsed = _parse(raw)
        except deadline.DeadlineExceeded:
            raise
        except Exception:                       # noqa: BLE001
            return inner if hasattr(inner, "get") else None
        if list(reparsed.keys()):
            return reparsed
    return inner if hasattr(inner, "get") else None


def _forwarded_messages(msg, depth: int = 0, bodies=()) -> list:
    """The text of each attached message, headed so it reads as quoted.

    Marked as quoted rather than merged into the body, because a forwarded
    message is someone else's text: the injection check reads an indented,
    headed block as quoted material, which is exactly what this is.
    """
    if depth > _MAX_NESTING:
        return []
    if msg.get_content_type() == "message/rfc822":
        # A mail that *is* a forwarded message, with nothing around it.
        # iter_attachments yields nothing for a message/* part, so the
        # original was lost whole: no sender, no subject, no text.
        return _forwarded_block(msg, depth)
    if not msg.is_multipart():
        return []
    out = []
    for part, _rule in _attachment_children(msg, bodies):
        deadline.check("the messages a mail forwards")
        if part.get_content_type() == "message/rfc822":
            with _element("a forwarded message"):
                out.extend(_forwarded_block(part, depth))
        elif part.is_multipart() and depth + 1 <= _MAX_NESTING:
            out.extend(_forwarded_messages(part, depth + 1, bodies))
    return out


def _forwarded_block(part, depth: int) -> list:
    """The quoted block of one message/rfc822 part, and those of the
    messages attached to it in turn."""
    inner = _inner_message(part)
    if inner is None:
        return []
    head = "   ".join(
        f"{name}: {_hval(inner, name)}"
        for name in ("From", "To", "Subject", "Date")
        if _hval(inner, name))
    inner_body, _, bodies = _read_body(inner)
    body_lines = inner_body.strip().splitlines()
    block = ["## Attached message", head,
             *(f"    {line}" for line in body_lines[:_ATTACHED_LINES])]
    if len(body_lines) > _ATTACHED_LINES:
        # Said, not swallowed. Every other truncation in this file
        # announces itself, and the attached original is the part the
        # commit that added this calls "usually the whole point".
        block.append(f"    … {len(body_lines) - _ATTACHED_LINES} further "
                     f"lines of the attached message are not shown")
    if depth + 1 > _MAX_NESTING:
        return ["\n".join(b for b in block if b)]
    return ["\n".join(b for b in block if b)] + _forwarded_messages(
        inner, depth + 1, bodies)


#: Figures worth comparing between the two halves of a multipart/alternative
#: message: amounts, account numbers, dates, reference numbers. A run of
#: three or more digits, or an IBAN.
_FIGURES = re.compile(r"[A-Z]{2}\d{2}[ ]?[A-Z0-9 ]{10,32}|\d[\d.,]{2,}")

#: The marker _body leaves in the body when the two alternatives
#: disagree, so the caller can raise it as a warning as well. It stays in
#: the body because the model should see it too.
_ALTERNATIVE_MISMATCH = re.compile(
    r"\[sift: the HTML version of this message states figures[^\]]*\]")


def _same_figures(plain: str, html: str) -> "tuple[bool, list[str]]":
    """Whether the two alternatives state the same numbers.

    multipart/alternative means "the same message, two renderings" — that is
    what the standard says and what every mail client assumes. Nothing
    enforces it. A message whose text part reads "19,90 EUR to DE02…" and
    whose HTML part reads "4.980,00 EUR to RO49…" is a valid mail, and every
    reader sees the HTML.

    sift reads the plain part, which is the right choice: it is the one that
    cannot carry a disguise. But reporting it as *the* message, with nothing
    said, is how a reader comes to pay the wrong amount into the wrong
    account.
    """
    def figures(text: str) -> dict:
        # Keyed by the digits alone. "1.240,00" and "1240.00" are the same
        # amount written by a German and an English formatter, and comparing
        # them as strings called every such pair a disagreement — a mail
        # whose text part said "Summe: 1240.00 EUR" and whose HTML said
        # "1.240,00 EUR" was reported as stating different figures.
        found = {}
        for match in _FIGURES.finditer(text):
            raw = "".join(match.group(0).split()).rstrip(".,")
            key = re.sub(r"[^0-9A-Z]", "", raw.upper())
            if key:
                found.setdefault(key, raw)
        return found

    theirs, ours = figures(html), figures(plain)
    in_html = [theirs[k] for k in theirs if k not in ours]
    # What is left is still a mixture: a substituted amount, and the figures
    # in a legal footer the plain part does not carry — an HRB number, a VAT
    # id, a telephone number. Measured on six ordinary German mails, one
    # trips on its footer. That is why this is a note and not a warning: it
    # says the HTML states figures the text does not, which is true of both,
    # and leaves the reading to a person.
    return (not in_html), sorted(in_html)[:6]


def _is_text(part) -> bool:
    """Whether a part is one a reader reads rather than opens: text/*, or a
    part with no Content-Type, which RFC 2045 makes text/plain."""
    return part.get_content_maintype() == "text"


def _read_body(msg) -> "tuple[str, str, tuple]":
    """(the body as text, what it was read from, the parts it was read
    from). Always a str; a body that cannot be read is a note, and ""."""
    chosen = []
    with _element("the body"):
        return _body(msg, chosen) + (tuple(chosen),)
    return "", "none", tuple(chosen)


def _body(msg, chosen: list) -> tuple[str, str]:
    """(the body as text, what it was read from). Always a str.

    A message that is one part, and that part is not text -- S/MIME's
    application/pkcs7-mime, a scan sent as image/png, a message/rfc822 --
    has no body: its one part is an attachment (_list_attachments). Read as
    a body it was bytes, and the first str operation on it raised
    TypeError; extract()'s boundary swallowed that, and the headers, the
    attachment list and all went with it.
    """
    if msg.is_multipart():
        plain = msg.get_body(preferencelist=("plain",))
        html = msg.get_body(preferencelist=("html",))
        chosen.extend(part for part in (plain, html) if part is not None)
        if plain is not None:
            body = _content(plain)
            if html is not None:
                agree, only_html = _same_figures(body, _strip_html(_content(html)))
                if not agree:
                    body += (
                        "\n\n[sift: the HTML version of this message states "
                        "figures the text version does not — "
                        + ", ".join(only_html) +
                        ". A mail reader shows the HTML one.]")
            return body, "text/plain"
        if html is not None:
            return _strip_html(_content(html)), "text/html→text"
        return "", "none"
    if not _is_text(msg):
        return "", "none"
    payload = _content(msg)
    if (msg.get_content_type() or "").endswith("html"):
        return _strip_html(payload), "text/html→text"
    return payload, "text/plain"


def _content(part: EmailMessage) -> str:
    """A text part's content, as a str whatever the part declares.

    get_content answers in the type the part names: bytes for a non-text
    type, a message for message/rfc822. Callers hand only text parts here
    (_body); anything else still comes back as a str, so a type
    nobody foresaw cannot turn into a TypeError further on.
    """
    try:
        content = part.get_content()
    except (LookupError, ValueError, KeyError):
        content = None
    if isinstance(content, str):
        return content
    payload = part.get_payload(decode=True)
    if isinstance(payload, bytes):
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, ValueError):
            # A charset Python does not know, or one it cannot even look up:
            # a NUL in its name is "embedded null character".
            return payload.decode("utf-8", errors="replace")
    payload = part.get_payload()
    return payload if isinstance(payload, str) else ""


class _Walk:
    """The attachments of one mail as the walk finds them, and how many
    parts it looked at (_MAX_PARTS)."""

    def __init__(self) -> None:
        self.out: list = []
        self.cut = False
        self.parts = 0

    def room(self) -> bool:
        """Whether one more part may be looked at."""
        if self.parts >= _MAX_PARTS:
            _said().more_parts = True
            return False
        self.parts += 1
        if self.parts % 256 == 0:
            deadline.check("the parts of a mail")
        return True


def _list_attachments(msg, bodies=()) -> list[dict]:
    """Every attachment, including the ones inside another part.

    iter_attachments() does not recurse, and get_payload(decode=True) is None
    for a container — so an attachment one level down was reported as
    "(unnamed), multipart/mixed, 0 B" and its real name, type and 140 bytes
    were all lost. A forward with the original attached came out the same
    way: "(unnamed), message/rfc822, 0 B". "0 B" is a stated falsehood about
    a file that is not empty, and the contradiction check permits a model to
    repeat it.

    ``bodies`` are the parts the body was read from (_read_body): every
    other part of an alternative is an attachment.
    """
    walk = _Walk()
    with _element("the attachments"):
        _message_attachments(msg, walk, depth=0, bodies=bodies)
    out = walk.out
    if walk.cut:
        out.append({"filename": "(nesting too deep to read)",
                    "content_type": "multipart", "size": 0,
                    "size_human": f"more than {_MAX_NESTING} levels deep"})
    _fill_sizes(out)
    return out


#: What a name and a Content-Type promise, in the vocabulary _magic answers
#: in. Only the formats a magic number can confirm — for anything else the
#: two cannot be compared and nothing is claimed.
_PROMISED = {
    "pdf": "pdf", "zip": "archive", "gz": "archive", "tgz": "archive",
    "7z": "archive", "rar": "archive", "png": "image", "jpg": "image",
    "jpeg": "image", "gif": "image", "bmp": "image", "webp": "image",
    "tif": "image", "tiff": "image", "heic": "image",
    "docx": "docx", "xlsx": "xlsx", "pptx": "pptx", "odt": "odt",
    "ods": "ods", "odp": "odp", "epub": "epub",
    "exe": "pe-executable", "dll": "pe-executable", "scr": "pe-executable",
    "so": "elf-binary", "doc": "office-legacy", "xls": "office-legacy",
    "ppt": "office-legacy", "msg": "office-legacy",
}
_MIME_PROMISED = {
    "application/pdf": "pdf", "application/zip": "archive",
    "application/gzip": "archive", "image/png": "image",
    "image/jpeg": "image", "image/gif": "image",
}
#: All of these really are zip files, so magic saying "docx" of a file named
#: .xlsx is a mislabelled office document, not a disguise.
_ZIP_FAMILY = frozenset({"archive", "docx", "xlsx", "pptx", "odt", "ods",
                         "odp", "epub"})


def _describe_bytes(head: bytes) -> str:
    """What to call bytes that are not what they claim but are not known."""
    if head.startswith(b"MZ"):
        return "a DOS or Windows executable image"
    if head.startswith(b"\x7fELF"):
        return "a Linux executable"
    if not head.strip():
        return "empty"
    if looks_like_html(head):
        # Named by what a browser would make of it: an HTML page runs its
        # scripts, and "text" said a reader could open it safely.
        return "an HTML page"
    printable = sum(1 for b in head[:512] if 32 <= b < 127 or b in (9, 10, 13))
    if printable >= len(head[:512]) * 0.9:
        return "text"
    return "something else"


def _suffix(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _promised_kind(filename: str, content_type: str) -> str:
    """The kind the name and the Content-Type claim, or "" if neither says."""
    suffix = _suffix(filename)
    return _PROMISED.get(suffix) or _MIME_PROMISED.get(
        (content_type or "").lower(), "")


#: The controls that turn what follows them right to left -- an override,
#: an embedding, an isolate -- and the ones that end them.
_RIGHT_TO_LEFT = "‮‫⁧"
_ENDS = {"‮": "‬", "‫": "‬", "⁧": "⁩"}


def _as_shown(name: str) -> str:
    """How a reader shows a name with a right-to-left control in it: what
    follows the control, to the end of its run, reversed; the controls
    themselves unseen. "invoice<RLO>fdp.exe" reads "invoiceexe.pdf".

    Not the whole Unicode bidi algorithm: the one reordering a name is
    built to hide its ending with."""
    out, index = [], 0
    while index < len(name):
        char = name[index]
        if char in _RIGHT_TO_LEFT:
            end = name.find(_ENDS[char], index + 1)
            end = len(name) if end < 0 else end
            out.append(_BIDI_CONTROL.sub("", name[index + 1:end])[::-1])
            index = end + 1
            continue
        if not _BIDI_CONTROL.match(char):
            out.append(char)
        index += 1
    return "".join(out)


#: Enough for every magic number sift knows; the rest of an attachment is
#: not read, since it is never sent to the model either.
_MAGIC_BYTES = 4096

#: Deep enough for a forward of a forward of a forward, bounded because a
#: crafted message can nest as far as it likes and each level costs a walk.
_MAX_NESTING = 8


def _chosen_parts(message) -> tuple:
    """The parts a message's body is read from, where _read_body would read
    it: get_body's plain and HTML choice."""
    chosen = []
    with _element("the body of an attached message"):
        if message.is_multipart():
            chosen = [part for part in (
                message.get_body(preferencelist=("plain",)),
                message.get_body(preferencelist=("html",)))
                if part is not None]
    return tuple(chosen)


def _message_attachments(message, walk: _Walk, depth: int, bodies=()) -> None:
    """The attachments of one message: the parts of a multipart, or the
    message's one part when that is not text.

    iter_attachments yields nothing for a message that is not multipart/*,
    so a mail that is a single application/pkcs7-mime, image/png or
    message/rfc822 part listed no attachment at all -- and that part is all
    there is of it.
    """
    maintype = message.get_content_maintype()
    if maintype == "multipart":
        _walk_attachments(message, walk, depth, bodies)
    elif maintype != "text" and walk.room():
        _add_part(message, walk, depth)


def _walk_attachments(part, walk: _Walk, depth: int, bodies=(),
                      inside: bool = False) -> None:
    if depth > _MAX_NESTING:
        # Said, not swallowed. The bound replaced a visibly wrong
        # "(unnamed), 0 B" with nothing at all, which reads as "this mail has
        # no attachments" — worse in the one dimension that matters.
        walk.cut = True
        return
    for child, rule in _attachment_children(part, bodies, inside):
        if not walk.room():
            return
        if rule in (_WALK, _WALK_INSIDE):
            if depth + 1 > _MAX_NESTING:
                walk.cut = True
                continue
            _walk_attachments(child, walk, depth + 1, bodies,
                              inside or rule == _WALK_INSIDE)
            continue
        _add_part(child, walk, depth, rule == _CHECK)


def _subparts(part) -> list:
    payload = part.get_payload()
    return list(payload) if isinstance(payload, list) else []


#: What _attachment_children says of a part: an attachment (_LIST); a part
#: of the body's own rendering, listed only where its bytes are not what
#: its name and type say (_CHECK); a container to walk, as the one it is
#: in is walked (_WALK) or as a part of the body (_WALK_INSIDE).
_LIST, _CHECK, _WALK, _WALK_INSIDE = "list", "check", "walk", "walk inside"


def _attachment_children(part, bodies=(), inside: bool = False):
    """(part, rule) for each part of a multipart that is an attachment or
    may hold one.

    iter_attachments() decides what is an attachment, and it passes over
    the body: it yields nothing for multipart/alternative, and in a mixed
    part it skips the first alternative and the first related part as
    candidate bodies. So a Windows program named invoice.pdf as a third
    alternative, or in a mixed part inside one, was no attachment and no
    warning. Those parts are walked now (``inside``): of an
    alternative, every part is an attachment but the ones the body was read
    from (``bodies``) and the other renderings of it -- text that is not
    declared an attachment and has no name, a calendar, a second HTML; of
    a related part inside the body, its images and styles are the
    rendering's own and are checked, not listed. Measured on 418 mails of
    every shape: the body's inline images and a calendar invite's calendar
    listed would have changed 43 of them.
    """
    if part.get_content_maintype() != "multipart":
        return
    _boundary_named_twice(part)
    subtype = part.get_content_subtype()
    chosen = {id(body) for body in bodies}
    children = _subparts(part)
    if subtype == "alternative":
        for child in children:
            if id(child) in chosen:
                continue
            if _holds_parts(child):
                yield child, _WALK_INSIDE
            elif not _renders_the_body(child):
                yield child, _LIST
        return
    attached = {id(child) for child in part.iter_attachments()}
    if subtype == "related":
        for child in children:
            if id(child) in attached:
                if _holds_parts(child):
                    yield child, _WALK
                else:
                    yield child, _CHECK if inside else _LIST
            elif _holds_parts(child):
                yield child, _WALK_INSIDE        # the root, a container
        return
    for child in children:
        if id(child) in attached:
            yield child, _WALK if _holds_parts(child) else _LIST
        elif child.get_content_type() in _CANDIDATE_BODIES:
            yield child, _WALK_INSIDE


#: The containers iter_attachments passes over as a candidate body.
_CANDIDATE_BODIES = frozenset({"multipart/alternative", "multipart/related"})


def _holds_parts(part) -> bool:
    """A container to walk: a multipart, not an attached message."""
    return part.is_multipart() and part.get_content_type() != "message/rfc822"


def _renders_the_body(part) -> bool:
    """A part of an alternative that is another rendering of the body, not
    a file: text, not declared an attachment, without a name."""
    return (part.get_content_maintype() == "text"
            and not part.is_attachment() and not part.get_filename())


def _boundary_named_twice(part) -> None:
    """Whether a multipart names its boundary twice. Python's parser splits
    it at the first, and its Content-Type keeps the first alone: the other
    is read from the header as the mail wrote it."""
    for key, raw in part.raw_items():
        if key.lower() == "content-type":
            named = sum(1 for parameter in _PARAMETER.findall(
                        str(raw)[:_MAX_HEADER_CHARS])
                        if parameter.strip().lower() in ("boundary",
                                                         "boundary*"))
            if named > 1:
                _said().boundary_twice = True
            return


#: The name of each parameter of a header as written: what follows a ";"
#: up to "=", a quoted string read past.
_PARAMETER = re.compile(r';([^;="]*)=(?:"(?:[^"\\]|\\.)*"|[^;]*)')


def _fill_sizes(out: list) -> None:
    for entry in out:
        if not entry["size_human"]:
            entry["size_human"] = human_size(entry["size"])


def _add_part(child, walk: _Walk, depth: int, checked: bool = False) -> None:
    """One attachment's entry -- a leaf part, or an attached message and
    what that message holds in turn. A part that cannot be read is listed
    as one, and a note says why. ``checked``: listed only where its bytes
    are not what its name and type say."""
    try:
        _add_entry(child, walk, depth, checked)
    except deadline.DeadlineExceeded:
        raise
    except Exception as exc:  # noqa: BLE001 - the sender's part
        _said().unreadable("one part", exc)
        walk.out.append({"filename": "(unreadable part)",
                         "content_type": _type_of(child), "size": 0,
                         "size_human": "?"})


def _type_of(part) -> str:
    try:
        return part.get_content_type()
    except Exception:  # noqa: BLE001
        return "application/octet-stream"


def _add_entry(child, walk: _Walk, depth: int, checked: bool = False) -> None:
    out = walk.out
    content_type = child.get_content_type()
    if content_type == "message/rfc822" and not checked:
        # An attached message. Its own name is usually absent, so the
        # subject is what a reader would call it.
        inner = child.get_payload()
        inner = inner[0] if isinstance(inner, list) and inner else inner
        subject = ""
        if hasattr(inner, "get"):
            subject = str(inner.get("subject") or "")
        out.append({
            "filename": child.get_filename()
                        or (f"{subject}.eml" if subject else "(attached message)"),
            "content_type": content_type,
            # Counted, never written: as_bytes() raised on what the sender
            # could put in a header Python rewrites (ValueError,
            # HeaderWriteError, UnicodeEncodeError), and recursed through a
            # forward of forwards past the interpreter's limit before the
            # nesting bound below was asked.
            "size": (len(bytes(child.get_payload(decode=False) or "",
                               "utf-8", "replace"))
                     if isinstance(child.get_payload(), str)
                     else _written_size(child)),
            "size_human": "",
        })
        if hasattr(inner, "get_content_maintype"):
            if depth + 1 > _MAX_NESTING:
                walk.cut = True
                return
            _message_attachments(inner, walk, depth + 1, _chosen_parts(inner))
        return
    payload = child.get_payload(decode=True) or b""
    if not isinstance(payload, bytes):
        payload = b""
    entry = {
        "filename": child.get_filename() or "(unnamed)",
        "content_type": content_type,
        "size": len(payload),
        "size_human": human_size(len(payload)),
    }
    # What the bytes are, against what the name and the Content-Type
    # claim. An attachment called Rechnung_4711.pdf whose first bytes are
    # MZ…PE is a Windows program, and sift reported it as
    # "Rechnung_4711.pdf (application/pdf, 334 B)" — the one thing about
    # the mail worth saying, not said. The name and the type both come
    # from whoever sent it; the bytes do not.
    head = payload[:_MAGIC_BYTES]
    actual = _magic(head, Path("attachment"))
    promised = _promised_kind(entry["filename"], content_type)
    if actual and promised and actual != promised and not (
            actual in _ZIP_FAMILY and promised in _ZIP_FAMILY):
        entry["contents_are"] = actual
    elif promised and contradicts_its_name(promised, head):
        # The weaker question, for when the stronger one has no answer.
        # _magic confirms a Windows executable only when its PE header
        # sits where the DOS stub says; a hand-built or truncated one
        # fails that and is recognised by nothing — while still not
        # beginning "%PDF-". A .pdf that does not begin "%PDF-" is wrong
        # whatever else it is.
        entry["contents_are"] = _describe_bytes(head)
    shown = _as_shown(entry["filename"])
    if shown != entry["filename"] and _suffix(shown) != _suffix(
            entry["filename"]):
        entry["name_reads_as"] = shown
    if checked and not ("contents_are" in entry or "name_reads_as" in entry):
        return
    out.append(entry)


def _lines_size(text: str) -> int:
    """What the generator writes of a text: each line break as one "\\n"."""
    return len(text) - text.count("\r\n")


def _headers_size(part, signed: bool) -> int:
    """The bytes the generator writes of a part's headers (policy.fold_binary),
    each header on its own: one that has to be refolded is folded, one
    Python cannot fold is counted as the mail wrote it."""
    total = 0
    policy = _SIGNED_POLICY if signed else _POLICY
    longest = sys.maxsize if signed else _POLICY.max_line_length
    for name, value in part.raw_items():
        if not isinstance(value, str):
            value = str(value)
        lines = linesep_splitter.split(value)
        refold = (lines and len(lines[0]) + len(name) + 2 > longest
                  or any(len(line) > longest for line in lines[1:])
                  or not value.isascii())
        if refold and len(value) <= _MAX_HEADER_CHARS:
            try:
                total += len(policy.fold_binary(name, value))
                continue
            except Exception:  # noqa: BLE001 - counted as written below
                pass
        total += len(name) + 2 + sum(map(len, lines)) + len(lines)
    return total


def _written_size(part) -> int:
    """The bytes part.as_bytes() writes, counted without writing any.

    Flattening a part to measure it raised on what the sender could put in
    a header Python rewrites -- 223 UnicodeEncodeErrors, 28 ValueErrors, 17
    HeaderWriteErrors of the review's fuzzed mails -- and recursed, level
    by level, through a forward of forwards: a RecursionError at 130
    levels. So this counts what email.generator would write, part by part
    off a stack rather than by recursion: the headers (_headers_size), a
    blank line, and a leaf's text with each line break one "\\n", a
    multipart's preamble, boundaries and epilogue, an attached message's
    own. The same number for the forwards ordinary mail is made of."""
    total, seen = 0, 0
    stack = [(part, False)]
    while stack:
        node, signed = stack.pop()
        seen += 1
        if seen % 256 == 0:
            deadline.check("the size of an attached message")
        if seen > _MAX_PARTS:
            break
        total += _headers_size(node, signed) + 1
        payload = node._payload
        maintype = node.get_content_maintype()
        if maintype == "multipart":
            if isinstance(payload, str):
                total += len(payload)
                continue
            parts = [] if payload is None else (
                payload if isinstance(payload, list) else [payload])
            boundary = len(node.get_boundary() or "=" * 36)
            if node.preamble is not None:
                total += _lines_size(node.preamble) + 1
            total += (boundary + 3) + max(len(parts) - 1, 0) * (boundary + 4) \
                + (boundary + 6)
            if node.epilogue is not None:
                total += _lines_size(node.epilogue)
            under = signed or node.get_content_type() == "multipart/signed"
            stack.extend((child, under) for child in reversed(parts))
        elif node.get_content_type() == "message/delivery-status" \
                and isinstance(payload, list):
            # Each block written without its last line break, the blocks
            # joined by one.
            total += max(len(payload) - 1, 0) - len(payload)
            stack.extend((child, signed) for child in reversed(payload))
        elif maintype == "message":
            if isinstance(payload, list) and payload:
                stack.append((payload[0], signed))
            elif isinstance(payload, str):
                total += len(payload)
        elif isinstance(payload, str):
            total += _lines_size(payload)
    return total


def _parts(msg):
    """Every part of a mail, the mail first, off a stack: msg.walk()
    recurses a level at a time. At most _MAX_PARTS."""
    stack, seen = [msg], 0
    while stack and seen < _MAX_PARTS:
        part = stack.pop()
        seen += 1
        if seen % 256 == 0:
            deadline.check("the parts of a mail")
        yield part
        payload = part.get_payload() if part.is_multipart() else None
        if isinstance(payload, list):
            stack.extend(reversed(payload))


def _strip_html(html: str) -> str:
    from .html_doc import html_to_text
    return html_to_text(html)


#: Names _hval already reads, so "other" does not repeat them.
_KNOWN = {"from", "to", "cc", "subject", "date", "message-id", "reply-to"}
#: Routing, signing and client bookkeeping. Real headers, and not what the
#: file is about; listing them would bury the one line that is.
_NOISE = re.compile(
    r"^(received|return-path|delivered-to|dkim-signature|domainkey-signature"
    r"|arc-[\w-]+|authentication-results|x-[\w-]+|mime-version|content-[\w-]+"
    r"|precedence|list-[\w-]+|user-agent|thread-\w+|auto-submitted"
    r"|received-spf|sender|errors-to|bcc|resent-[\w-]+)$", re.IGNORECASE)


def _render(headers: dict, body: str, attachments: list[dict]) -> str:
    lines = ["# Email"]
    for key in ("from", "to", "cc", "subject", "date"):
        if headers.get(key):
            lines.append(f"{key.capitalize()}: {headers[key]}")
    other = headers.get("other") or {}
    if other:
        # Indented under a heading that says whose they are. Rendered as bare
        # "Name: value" lines they were indistinguishable from sift's own,
        # and the *name* is chosen by the sender: a mail carrying a header
        # literally called "Attachments" wrote
        #   Attachments: signed-mandate.pdf (application/pdf, 812 KB)
        # into this block, one line above sift's real attachment list, with
        # the metadata four lines up saying attachment_count: 0. "Warning",
        # "Summary" and "Classification" — the field labels of sift's own
        # report — worked the same way. _hval closed this on the value side
        # and the door reopened on the name side.
        lines.append("")
        lines.append("## Headers the file set, as it wrote them")
        for key, value in list(other.items())[:12]:
            lines.append(f"    {_hval_name(key)}: {one_line(value)}")
        if len(other) > 12:
            lines.append(f"    … and {len(other) - 12} more")
    if attachments:
        # Through one_line, like every header value: an attachment filename
        # is a MIME parameter, and both routes into it carry newlines — a
        # percent-encoded filename*= and a nested message/rfc822 whose
        # RFC 2047 Subject decodes to line breaks. The header block itself is
        # neutralized by _hval; this line was the way around it.
        listed = attachments[:_MAX_LISTED]
        line = "Attachments: " + ", ".join(
            f"{one_line(a['filename'])} ({one_line(a['content_type'])},"
            f" {a['size_human']})" for a in listed)
        if len(attachments) > len(listed):
            line += f", … and {len(attachments) - len(listed)} more"
        lines.append(line)
    lines.append("")
    lines.append("## Body")
    lines.append(body.strip())
    return "\n".join(lines)
