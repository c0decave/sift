"""Rich Text Format — the words, without the formatting.

RTF is what Wordpad writes, what a great many German legal and insurance
templates are still stored as, and what a mail client produces when someone
pastes formatted text. Read as plain text it is unusable: the preview of a
two-line letter was ``{\\rtf1\\ansi\\deffo {\\fonttbl{\\f0 Arial;}}\\f0\\fs24 Sehr
geehrte``, which tells a reader nothing about the file and gives a model
nothing to summarize.

The format is a tree of ``{...}`` groups holding control words and text. This
reads the text and drops the rest, which is all a summary needs. It is
deliberately not a converter: tables lose their alignment and lists lose
their bullets, and both are said in the notes rather than pretended away.
"""
from __future__ import annotations

import codecs
import re

from .. import deadline
from ..detect import Detection
from .base import Extracted, clip, human_size

MAX_CHARS = 200_000

#: How deep the group nesting may go. Real documents reach ten or twenty;
#: this is a bound on work, not on documents.
_MAX_DEPTH = 256

#: Destinations whose content is markup rather than the document's words:
#: the font table, the colour table, style sheets, revision info, embedded
#: objects and pictures. Skipping the whole group is the point — a font
#: table read as text contributes "Times New Roman;Arial;Courier New".
_SKIP_DESTINATIONS = frozenset({
    "fonttbl", "colortbl", "stylesheet", "listtable", "listoverridetable",
    "info", "pict", "object", "themedata", "colorschememapping",
    "latentstyles", "datastore", "generator", "xmlnstbl", "rsidtbl",
    "mmathPr", "wgrffmtfilter", "fchars", "lchars", "filetbl", "revtbl",
    "upr", "author", "operator", "creatim", "revtim", "printim", "buptim",
})

#: Control words that stand for a character rather than for formatting.
_CHARACTERS = {
    "par": "\n", "line": "\n", "sect": "\n\n", "page": "\n\n",
    "tab": "\t", "cell": "\t", "row": "\n", "nestcell": "\t",
    "emdash": "—", "endash": "–", "emspace": " ", "enspace": " ",
    "qmspace": " ", "bullet": "•", "lquote": "‘", "rquote": "’",
    "ldblquote": "“", "rdblquote": "”", "~": " ", "-": "", "_": "-",
}

#: A control word: a backslash, letters, an optional signed number, and an
#: optional single space that belongs to the control word rather than to the
#: text after it.
_CONTROL = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?")
#: An escaped byte, written as two hex digits.
_HEX = re.compile(r"\\'([0-9a-fA-F]{2})")
#: A literal brace, backslash or other escaped character.
_ESCAPED = re.compile(r"\\([\\{}])")
#: A surrogate, which only a \u escape can write: one of a pair joins its
#: other half as it is read, and what is left stands alone. Looked for by the
#: engine: a loop of ord() over every character read was 2,400,000 calls for
#: an RTF of 2.4 MB of text (an earlier version; a review).
_SURROGATE = re.compile(r"[\ud800-\udfff]")


def extract(path, det: Detection, max_bytes: int) -> Extracted:
    raw = path.read_bytes()[:max_bytes]
    # The encoding a document declares, when it declares one. \ansicpg1252 is
    # what Word writes for a Western European document and is what turns
    # \'e4 back into "ä"; without it a German letter comes out in mojibake,
    # which then reads as a corrupt file rather than a foreign one.
    codepage = _declared_codepage(raw)
    source = raw.decode("latin-1", "replace")
    text, notes, unseen, was_hidden, was_deleted = _to_text(source, codepage)
    properties = _document_properties(source, codepage)
    body, truncated = clip(text, MAX_CHARS)
    meta = {
        "encoding": codepage,
        "words": len(body.split()),
        "paragraphs": sum(1 for line in body.splitlines() if line.strip()),
        **properties,
    }
    meta = {k: v for k, v in meta.items() if v}
    # rendered: the control words are gone and only the document's words are
    # left. Without the note a real model warned that the file "uses an .rtf
    # extension but contains only plain text without visible RTF markup" —
    # true of what sift showed it, false about the file, and stated as a
    # finding.
    ex = Extracted("rtf", det.mime, text=body, metadata=meta, rendered=True,
                   truncated=truncated, hidden_text=unseen[:20_000])
    if was_hidden:
        # Marked so summarize can raise it as a warning rather than a note.
        # A tracked deletion is an ordinary editing artefact; text marked
        # *hidden* is a deliberate act, and the example that prompted this
        # said the opposite of the visible line beside it — "bitte als
        # BEZAHLT anzugeben" under "Status: UNBEZAHLT".
        ex.metadata["hidden_characters"] = len(unseen)
    elif was_deleted:
        ex.note(f"the document holds {len(unseen)} characters marked as "
                f"deleted — a tracked change; they are not part of the text "
                f"above and were not sent to the model")
    for note in notes:
        ex.note(note)
    if truncated:
        ex.note(f"document longer than {human_size(MAX_CHARS)} of text; "
                f"the first part was read")
    return ex


#: The \info fields worth reporting. The group is skipped when reading the
#: text — it is not the document's words — but skipping it entirely meant an
#: author field was neither shown nor scanned, so a sentence addressed to an
#: assistant sitting in \author went out with no warning and no note. It is
#: also where the author actually is, which is worth a line of metadata.
_PROPERTIES = {"author": "author", "title": "title", "subject": "subject",
               "keywords": "keywords", "company": "company",
               "operator": "last_saved_by", "category": "category",
               "comment": "comment", "doccomm": "comment"}


def _document_properties(source: str, codepage: str) -> dict:
    r"""The \info group's fields, read on their own."""
    start = source.find("{\\info")
    if start < 0:
        return {}
    out = {}
    for word, key in _PROPERTIES.items():
        found = re.search(rf"\{{\\{word} ?([^{{}}]{{0,400}})\}}",
                          source[start:start + 20_000])
        if not found:
            continue
        value = _to_text("{" + found.group(1) + "}", codepage)[0]
        value = " ".join(value.split())
        if value:
            out[key] = value[:200]
    return out


def _declared_codepage(raw: bytes) -> str:
    """The document's own ``\\ansicpgNNNN``, mapped to a Python codec."""
    found = re.search(rb"\\ansicpg(\d{3,5})", raw[:4096])
    if not found:
        # \mac and \pc are the other two declarations; anything else is a
        # Word document, and Word writes cp1252 unless it says otherwise.
        return "cp1252"
    number = found.group(1).decode("ascii")
    try:
        # Normalised, so the report names the encoding a reader knows.
        # \ansicpg65001 is UTF-8; "cp65001" is a real Python alias for it and
        # a confident-looking answer nobody recognises.
        return codecs.lookup(f"cp{number}").name
    except LookupError:
        return "cp1252"


def _to_text(source: str, codepage: str,
             collect_hidden: bool = False) -> "tuple[str, list[str], str]":
    """The document's words, what was dropped to get them, and what Word
    does not display.

    Three kinds of text a reader never sees, and all three used to be read
    as the document's words:

    ``\\v`` … ``\\v0`` is hidden text. A 371-byte invoice reading "Status:
    UNBEZAHLT" carried, between two visible lines, a hidden sentence saying
    the invoice had been paid and instructing that it be reported as paid —
    and a real model duly summarized it as paid. Six characters.

    ``\\deleted`` is a tracked deletion. "Der Betrag beträgt 10.000 EUR" with
    the 10.000 struck through and 1.000 inserted came out as "10.000
    EUR1.000 EUR", which the model called a typographical error.

    ``\\*\\fldinst`` is a field *instruction*: MERGEFIELD, DATE, REF. What
    the reader sees is the field *result* beside it. Read as text, a real
    invoice became "Betrag: MERGEFIELD Rechnungsbetrag \\* MERGEFORMAT
    1.240,00 EUR" and was reported as a template with sample values.

    Hidden and deleted text is not thrown away: it goes where html_doc puts
    an HTML comment — into ``hidden_text``, which is scanned and reported
    but never sent to the model. Text a reader cannot see, saying something
    the visible text contradicts, is the most interesting thing about a
    file that has it.
    """
    out: list[str] = []
    hidden: list[str] = []
    notes: list[str] = []
    saw_hidden = saw_deleted = False
    skip_depth = 0                       # >0 while inside a skipped group
    hide_from = None                     # depth at which hiding began
    depth = 0
    position = 0
    length = len(source)
    saw_table = saw_picture = False
    # A \uNNNN character is followed by a fallback the reader must skip;
    # the count comes from \ucN and is 1 unless the document says otherwise.
    skip_after_unicode = 1
    pending_skip = 0

    while position < length:
        if not position % 8192:
            deadline.check("an RTF document")
        char = source[position]

        if char == "{":
            depth += 1
            if skip_depth:
                skip_depth += 1
            position += 1
            if depth > _MAX_DEPTH:
                notes.append("document nests deeper than sift will follow; "
                             "the rest was not read")
                break
            continue
        if char == "}":
            depth -= 1
            if skip_depth:
                skip_depth -= 1
            if hide_from is not None and depth < hide_from:
                hide_from = None         # the group that hid it has closed
            position += 1
            continue
        if char != "\\":
            if not skip_depth:
                if pending_skip > 0:
                    pending_skip -= 1
                elif char not in "\r\n":
                    (hidden if hide_from is not None else out).append(char)
                elif char == "\n":
                    pass                 # a real newline in RTF is nothing
            position += 1
            continue

        hexed = _HEX.match(source, position)
        if hexed:
            # A *run* of \'hh, decoded once. One byte at a time is right for
            # cp1252 and wrong for every multi-byte encoding: \ansicpg65001
            # is UTF-8, so "Grüße" arrives as five escapes making three
            # characters, and decoding each on its own produced "GrÃ¼ÃŸe" —
            # under a metadata line confidently naming the encoding it had
            # just failed to apply. Japanese and Chinese documents
            # (cp932, cp936, cp950) had the same fault.
            run = bytearray()
            while hexed:
                if skip_depth:
                    pass
                elif pending_skip > 0:
                    pending_skip -= 1
                else:
                    run.append(int(hexed.group(1), 16))
                position = hexed.end()
                hexed = _HEX.match(source, position)
            if run:
                (hidden if hide_from is not None else out).append(
                    bytes(run).decode(codepage, "replace"))
            continue

        escaped = _ESCAPED.match(source, position)
        if escaped:
            if not skip_depth and not pending_skip:
                (hidden if hide_from is not None else out).append(
                    escaped.group(1))
            elif pending_skip:
                pending_skip -= 1
            position = escaped.end()
            continue

        if source.startswith("\\*", position):
            # "\*" marks the group as ignorable: the format's own way of
            # saying "skip this whole thing if you do not understand it",
            # and every real RTF opens with {\*\generator …}. Matching
            # neither _CONTROL (which needs letters) nor _ESCAPED, the
            # backslash was skipped and the star was appended as text — so
            # essentially every document produced by Word or WordPad had a
            # spurious "*" as the first character of its summary.
            #
            # It is also what precedes \fldinst, the field *instruction*
            # (MERGEFIELD, DATE, REF) whose result is displayed beside it,
            # so honouring it removes that too.
            if not skip_depth:
                skip_depth = 1
            position += 2
            continue
        control = _CONTROL.match(source, position)
        if not control:
            position += 1                # a lone backslash; nothing to do
            continue
        word, argument = control.group(1), control.group(2)
        position = control.end()

        if word == "u" and argument is not None:
            if not skip_depth:
                code = int(argument)
                if code < 0:
                    code += 65536        # RTF writes these as signed 16-bit
                # A character above the basic plane — an emoji, 𝄞, CJK
                # extension B — is written by Word as a *pair* of \uNNNN
                # escapes, one high surrogate and one low. Emitting each on
                # its own left two lone surrogates in the text, which UTF-8
                # cannot encode: `--markdown -o` died on the whole run and
                # left a zero-byte report where every other summary in it
                # should have been.
                if (0xDC00 <= code <= 0xDFFF and out
                        and 0xD800 <= ord(out[-1]) <= 0xDBFF):
                    high = ord(out.pop())
                    code = 0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)
                (hidden if hide_from is not None else out).append(chr(code))
            pending_skip = skip_after_unicode
            continue
        if word == "uc" and argument is not None:
            skip_after_unicode = max(0, int(argument))
            continue
        if word in ("v", "deleted"):
            # \v and \deleted turn on; \v0 and \deleted0 turn off. Both are
            # character properties, so they last to the end of their group.
            if argument == "0":
                hide_from = None
            else:
                if hide_from is None:
                    hide_from = depth
                if word == "v":
                    saw_hidden = True
                else:
                    saw_deleted = True
            continue
        if word in _SKIP_DESTINATIONS:
            saw_picture |= word in ("pict", "object")
            if not skip_depth:
                skip_depth = 1
            continue
        if word in ("trowd", "cellx", "intbl"):
            saw_table = True
            continue
        if word in _CHARACTERS and not skip_depth:
            (hidden if hide_from is not None else out).append(
                _CHARACTERS[word])
            continue
        # Any other control word is formatting, and formatting is not the
        # summary's business.

    # Any surrogate that survives — a half pair at the end of a truncated
    # read, a file that writes them singly — is replaced rather than carried:
    # a string UTF-8 cannot encode poisons every writer downstream.
    joined = "".join(out)
    if _SURROGATE.search(joined):
        joined = _SURROGATE.sub("\ufffd", joined)
        notes.append("the document contains characters that are not valid on "
                     "their own; they are shown as \ufffd")
    text = re.sub(r"[ \t]+\n", "\n", joined)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if saw_table:
        notes.append("the document contains a table; its cells are shown as "
                     "tab-separated text and its layout is not preserved")
    if saw_picture:
        notes.append("the document contains an embedded picture or object, "
                     "which was not read")
    return (text, notes, " ".join("".join(hidden).split()),
            saw_hidden, saw_deleted)
