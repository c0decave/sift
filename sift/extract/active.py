"""Active content: what in a file would run, or load something from
elsewhere, if the file were shown -- found by reading its bytes.

A host application does not render a file that carries it. The operator's rule
(2026-09-28): no JavaScript in HTML, in a PDF, in a mail or anywhere else;
executable content in documents refused by default; rendered HTML with
simple tags only, and nothing downloaded -- not even a mail's images.

sift says what it found as ``metadata["active_content"] = {"classes":
[...], "checked": bool}``, for every file of a kind that can carry any
(KINDS). ``checked`` is False where the scan could not look everywhere: the
file is encrypted, damaged, cannot be listed, or a budget cut the scan
short. A reader must then treat the file as possibly active.

Nothing here runs, renders, decrypts or fetches anything. XML is read by
expat, which loads no external entity; a PDF is tokenized, not interpreted;
archives are listed and their members' first bytes read. The scan is bound
by what the file may cost sift already: its bytes up to --max-bytes, the
decompression budget below, and a share of the time left for the file.

Where two readings of the bytes are possible -- a browser with scripting on
or off, an element that is raw text in HTML but markup in SVG -- the scan
takes both and reports what either finds. It errs toward saying a file is
active; saying a file is not is the claim it has to be sure of.
"""
from __future__ import annotations

import base64
import bisect
import bz2
import codecs
import email
import email.header
import email.policy
import gzip
import html
import io
import lzma
import os
import re
import struct
import tarfile
import zipfile
import zlib
from pathlib import Path
from typing import NamedTuple
from xml.parsers import expat

from .. import deadline, zipguard

try:                                    # Python 3.14 and later
    from compression import zstd as _zstd  # type: ignore
except ImportError:                     # pragma: no cover - older Pythons
    _zstd = None


#: The kinds whose files can carry active content: each of them gets the
#: field, whatever the scan finds. (epub too: its pages are XHTML.)
KINDS = frozenset({"pdf", "docx", "xlsx", "pptx", "odt", "ods", "odp",
                   "epub", "rtf", "html", "xml", "email", "archive",
                   "office-legacy"})

#: Every class the scan can report; a host application is built against these names.
CLASSES = frozenset({
    "pdf-javascript", "pdf-open-action", "pdf-launch", "pdf-submit-form",
    "pdf-import-data", "pdf-embedded-file", "pdf-richmedia", "pdf-xfa",
    "pdf-goto-remote",
    "office-macros", "office-activex", "office-ole-object", "office-dde",
    "office-external-template", "odf-scripts", "rtf-object",
    "html-script", "html-event-handler", "html-javascript-url", "html-frame",
    "html-remote-resource",
    "mail-executable-attachment", "archive-executable", "archive-java"})

#: Bytes the scan of one file may decompress, decode or read out of an
#: archive, in all -- sift's other bound on what an archive may unpack
#: (fallback.MAX_TAR_SCAN_BYTES, office._MAX_TOTAL_INFLATED).
DECODE_BUDGET = 64 * 1024 * 1024

#: The share of the time left for the file that the scan may take. Running
#: out of it leaves ``checked`` False and the rest of the summary its time.
TIME_SHARE = 0.5

#: How deep one container may sit in another -- an attachment in a mail in
#: an archive -- before the scan stops following and says so.
MAX_DEPTH = 6

#: Members listed across every archive of one file (zipguard's bound on one).
MAX_MEMBERS = zipguard.MAX_ZIP_ENTRIES

#: What is read of a member to tell what it is.
HEAD = 4096

_PIECE = 1 << 16


class _Cut(Exception):
    """A budget of the scan is spent. What was found stands; the scan says
    it did not look everywhere."""


class Found:
    """What the scan of one file found, and where it fell short."""

    def __init__(self) -> None:
        self.classes: set = set()
        #: class -> {(what, how)}: which kind of executable, and how it was
        #: told ("pe", "by magic"), for the note.
        self.detail: dict = {}
        #: Why the scan could not look everywhere, in words for the note.
        self.cuts: list = []
        self.left = DECODE_BUDGET
        self.members = 0

    def add(self, cls: str, detail: "tuple | None" = None) -> None:
        if cls not in CLASSES:
            raise ValueError(f"not a class of active content: {cls!r}")
        self.classes.add(cls)
        if detail:
            self.detail.setdefault(cls, set()).add(detail)

    def cut(self, why: str) -> None:
        if why not in self.cuts:
            self.cuts.append(why)

    def spend(self, amount: int) -> None:
        """Charge ``amount`` decoded bytes to the budget."""
        self.left -= amount
        if self.left < 0:
            raise _Cut(f"it unpacks to more than "
                       f"{DECODE_BUDGET // (1 << 20)} MiB")

    def metadata(self) -> dict:
        """The interface: sorted unique class names, and whether the scan
        looked everywhere."""
        return {"classes": sorted(self.classes), "checked": not self.cuts}

    def note(self) -> "str | None":
        """What the note list says, or None when there is nothing to say."""
        cuts = "; ".join(self.cuts)
        if not self.classes:
            if not self.cuts:
                return None
            return (f"this file could not be checked for active content "
                    f"everywhere ({cuts}); a host application treats it as possibly "
                    f"active and does not render it")
        said = (f"this file carries active content "
                f"({', '.join(self._named(c) for c in sorted(self.classes))});"
                f" a host application does not render it")
        if self.cuts:
            said += f"; not all of it could be checked ({cuts})"
        return said

    def _named(self, cls: str) -> str:
        detail = self.detail.get(cls)
        if not detail:
            return cls
        by_how: dict = {}
        for what, how in sorted(detail):
            by_how.setdefault(how, []).append(what)
        return (f"{cls} [" + "; ".join(f"{', '.join(whats)} {how}"
                                        for how, whats in by_how.items())
                + "]")


def reach(max_bytes: int, active_bytes: int = 0) -> int:
    """How many bytes of a file the scan reads: ``max_bytes`` (0: all of
    it), or ``active_bytes`` where that is more (Config.active_max_bytes) --
    never less than the extractors read."""
    if max_bytes <= 0:
        return 0
    return max(max_bytes, active_bytes)


def scan(path: Path, det, max_bytes: int) -> Found:
    """What active content the file at ``path`` carries. Never raises for
    the file's sake: what the scan cannot read leaves ``checked`` False.
    DeadlineExceeded passes through only when the file's own budget is
    spent, as every extractor lets it."""
    found = Found()
    scanner = globals().get(_SCANNERS.get(det.kind, ""))
    if scanner is None:
        return found
    left = deadline.remaining()
    try:
        with deadline.limit(left * TIME_SHARE if left else None):
            scanner(Path(path), max_bytes, found)
    except deadline.DeadlineExceeded:
        if deadline.spent():
            raise
        found.cut("its share of the time budget ran out")
    except _Cut as exc:
        found.cut(str(exc))
    except RecursionError:
        found.cut("it nests deeper than the scan follows")
    except MemoryError:
        found.cut("it needs more memory than the scan may take")
    except Exception as exc:  # noqa: BLE001 - the file's, never the batch's
        # The type only: an exception's words may quote the file.
        found.cut(f"it could not be read ({type(exc).__name__})")
    return found


def _read(path: Path, max_bytes: int, found: Found) -> bytes:
    """The file, up to ``max_bytes`` -- what the extractors read of it."""
    with path.open("rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        data = fh.read(max_bytes) if max_bytes > 0 else fh.read()
    if size > len(data):
        found.cut(f"only the first {len(data)} of {size} bytes were read")
    return data


# ---- what a member or an attachment is ---------------------------------

_NATIVE_EXT = frozenset(
    ".exe .dll .scr .com .pif .cpl .sys .ocx .drv .so .dylib .elf .msi .msp "
    ".msix .appx .apk .dex .xll".split())
_SCRIPT_EXT = frozenset(
    ".js .jse .mjs .cjs .vbs .vbe .vb .wsf .wsh .wsc .sct .ps1 .psm1 .psd1 "
    ".bat .cmd .hta .lnk .url .scf .reg .sh .bash .zsh .ksh .csh .command "
    ".desktop .applescript .scpt".split())
_JAVA_EXT = frozenset(".jar .class .war .ear .jnlp".split())
_MACRO_EXT = frozenset(".docm .dotm .xlsm .xltm .xlam .xlsb .pptm .potm "
                       ".ppsm .ppam .sldm".split())
_DISK_IMAGE_EXT = frozenset(".iso .img .vhd .vhdx .dmg .udf".split())
_EXEC_TYPES = frozenset({
    "application/x-msdownload", "application/x-msdos-program",
    "application/x-dosexec", "application/x-executable",
    "application/vnd.microsoft.portable-executable",
    "application/x-ms-shortcut", "application/hta", "application/x-sh",
    "application/javascript", "text/javascript", "application/x-javascript",
    "text/vbscript", "application/x-vbscript", "application/java-archive",
    "application/x-java-archive", "application/x-ms-installer",
    "application/x-msi"})
_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _ext(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1].lower().rstrip(". ")
    return "." + base.rsplit(".", 1)[-1] if "." in base else ""


def _magic(head: bytes) -> "str | None":
    """The kind of program ``head`` begins, or None."""
    if head[:2] == b"MZ" and b"\x00" in head[2:64]:
        # A text that happens to begin "MZ" has no NUL in its first bytes;
        # a DOS or Windows image always does.
        return "pe"
    if head[:4] == b"\x7fELF":
        return "elf"
    if head[:4] in (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf",
                    b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe",
                    b"\xca\xfe\xba\xbf"):
        return "mach-o"
    if head[:4] == b"\xca\xfe\xba\xbe" and len(head) >= 8:
        # Java's class files and Mach-O's universal binaries share the
        # magic; a class file's major version is 45 or more, the count of
        # architectures of a universal binary a handful.
        return "java class" if struct.unpack(">H", head[6:8])[0] >= 45 \
            else "mach-o"
    if head[:4] == b"dex\n":
        return "dex"
    if head[:8] == b"L\x00\x00\x00\x01\x14\x02\x00":
        return "lnk"
    if head[:2] == b"#!":
        return "#!"
    return None


def _container(head: bytes) -> "str | None":
    """The archive format ``head`` begins, or None."""
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:3] == b"BZh" and head[3:4].isdigit() \
            and head[4:10] in (b"\x31\x41\x59\x26\x53\x59",
                               b"\x17\x72\x45\x38\x50\x90"):
        return "bz2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"
    if head[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    if head[:4] == b"Rar!":
        return "rar"
    if head[257:262] == b"ustar":
        return "tar"
    return None


def _judge_member(name: str, head: "bytes | None", found: Found) -> None:
    """A member of an archive: a native program by its magic, a script or
    a program by its extension, Java by either."""
    ext = _ext(name)
    if ext in _JAVA_EXT:
        found.add("archive-java", (ext, "by extension"))
    elif ext in _NATIVE_EXT or ext in _SCRIPT_EXT or ext in _MACRO_EXT:
        found.add("archive-executable", (ext, "by extension"))
    kind = _magic(head or b"")
    if kind == "java class":
        found.add("archive-java", (kind, "by magic"))
    elif kind:
        found.add("archive-executable", (kind, "by magic"))


def _judge_attachment(name: str, content_type: str, data: bytes,
                      found: Found, depth: int) -> None:
    """A mail's attachment: executable by its bytes, its name or its
    declared type; and whatever it holds, read as what its bytes say it
    is."""
    ext = _ext(name)
    kind = _magic(data[:HEAD])
    if kind:
        found.add("mail-executable-attachment", (kind, "by content"))
    if ext in _NATIVE_EXT or ext in _SCRIPT_EXT or ext in _JAVA_EXT \
            or ext in _MACRO_EXT:
        found.add("mail-executable-attachment", (ext, "by name"))
    if content_type in _EXEC_TYPES:
        found.add("mail-executable-attachment",
                  (content_type, "by content type"))
    if ext in _DISK_IMAGE_EXT:
        found.cut("a disk image attached to it cannot be looked into")
    had_macros = "office-macros" in found.classes
    _blob(data, name, found, depth + 1)
    if "office-macros" in found.classes and not had_macros:
        found.add("mail-executable-attachment", ("office macros", "by content"))


def _blob(data: bytes, name: str, found: Found, depth: int) -> None:
    """Bytes held in another file, read as what they begin with."""
    if depth > MAX_DEPTH:
        found.cut("it nests deeper than the scan follows")
        return
    head = data[:HEAD]
    kind = _container(head)
    ext = _ext(name)
    if head.startswith(b"%PDF-") or (ext == ".pdf" and b"%PDF-" in head[:1024]):
        _pdf_bytes(data, found, depth)
    elif kind == "zip":
        _zip_bytes(data, found, depth)
    elif kind in ("gzip", "bz2", "xz", "zstd", "tar"):
        _compressed(io.BytesIO(data), kind, name, found, depth)
    elif kind in ("7z", "rar"):
        found.cut(f"a {kind} archive in it is not looked into")
    elif head.startswith(_OLE):
        _cfb_bytes(data, found, depth)
    elif head.startswith(b"{\\rt"):
        _rtf_bytes(data, found)
    elif ext in _HTML_EXT:
        _html_bytes(data, found, depth)
    elif ext in (".eml", ".mht", ".mhtml"):
        _mail_bytes(data, found, depth)
    elif looks_like_html(head):
        # By its bytes, as a browser reads what it is not told the type
        # of: a page with a script, attached as invoice.pdf, ran nothing
        # here and was "text" to the mail extractor.
        _html_bytes(data, found, depth)


#: What a browser reads as HTML when nothing tells it the type: the HTML
#: patterns of the WHATWG MIME Sniffing Standard (7.1, "rules for
#: identifying an unknown MIME type"), each followed by a space or ">",
#: after whitespace.
_HTML_OPENINGS = (b"<!doctype html", b"<html", b"<head", b"<script",
                  b"<iframe", b"<h1", b"<div", b"<font", b"<table", b"<a",
                  b"<style", b"<title", b"<b", b"<body", b"<br", b"<p",
                  b"<!--")


def looks_like_html(head: bytes) -> bool:
    """Whether a browser that is not told what ``head`` is reads it as HTML
    (_HTML_OPENINGS); a UTF-8 byte order mark before it too."""
    start = head[:1024].lstrip(b"\t\n\x0c\r ")
    if start.startswith(b"\xef\xbb\xbf"):
        start = start[3:].lstrip(b"\t\n\x0c\r ")
    low = start[:16].lower()
    return any(low.startswith(opening)
               and low[len(opening):len(opening) + 1] in (b" ", b">")
               for opening in _HTML_OPENINGS)


# ---- HTML and CSS ------------------------------------------------------
#
# A tokenizer after the HTML standard's: tag names end at white space, "/"
# or ">"; attribute names at those or "="; values are quoted or end at
# white space or ">"; comments end at "-->" or "--!>", and "<!-->" is an
# empty one. Character references are decoded in values, names compared in
# lowercase. Raw-text elements (script, style, title, textarea, ...) end at
# the first "</name"; that pass is run, and where one of them held markup,
# a second pass that reads everything as markup -- an SVG's <style>, a
# <noscript> with scripting off. Either pass's findings count.

_HTML_EXT = frozenset(".html .htm .xhtml .xht .shtml .svg .hta".split())
_TAG_NAME = re.compile(r"[^\t\n\f\r />]*")
_BEFORE_ATTRIBUTE = re.compile(r"[\t\n\f\r /]*")
_SPACE = re.compile(r"[\t\n\f\r ]*")
_ATTRIBUTE_NAME = re.compile(r"[^\t\n\f\r />=]*")
_UNQUOTED = re.compile(r"[^\t\n\f\r >]*")
_COMMENT_END = re.compile(r"--!?>")
_RAWTEXT = frozenset({"script", "style", "xmp", "iframe", "noembed",
                      "noframes", "noscript", "textarea", "title",
                      "plaintext"})
_RAW_ENDS: dict = {}
_FRAMES = frozenset({"iframe", "frame", "frameset", "object", "embed",
                     "applet", "portal", "fencedframe"})
_URL_ATTRIBUTES = frozenset({
    "href", "src", "action", "formaction", "data", "background", "poster",
    "lowsrc", "dynsrc", "codebase", "cite", "longdesc", "icon", "manifest",
    "usemap", "profile", "classid", "code"})
_URL_LISTS = frozenset({"srcset", "imagesrcset"})
_SPACED_URLS = frozenset({"ping", "archive"})
_ANIMATED = frozenset({"values", "to", "from", "by"})
#: Elements whose URL is only ever shown as a picture or played: an SVG
#: there runs no script.
_IMAGE_ELEMENTS = frozenset({"img", "image", "source", "picture", "input",
                             "video", "audio", "track", "feimage"})
#: Attributes that begin "on" and are no event handler.
_NOT_HANDLERS = frozenset({"one", "only", "once", "online", "onset", "onto"})
#: Every attribute _judge_tag looks at besides the "on" ones.
_JUDGED = (_URL_ATTRIBUTES | _URL_LISTS | _SPACED_URLS | _ANIMATED
           | {"style", "srcdoc", "attributename"})
_SCRIPT_SCHEMES = ("javascript:", "vbscript:", "livescript:")
_ACTIVE_DATA = frozenset({
    "text/html", "application/xhtml+xml", "text/javascript",
    "application/javascript", "application/x-javascript", "text/ecmascript",
    "application/ecmascript", "text/vbscript"})
_MARKUP_DATA = frozenset({"image/svg+xml", "text/xml", "application/xml"})
_REMOTE_SCHEMES = frozenset({"http", "https", "ftp", "ftps", "file", "ws",
                             "wss", "smb", "sftp", "gopher"})
_SCHEME = re.compile(r"\A([a-z][a-z0-9+.\-]*):")
#: What a browser strips from a URL before it reads the scheme: C0
#: controls and spaces at either end, tabs and newlines anywhere.
_URL_EDGE = "".join(chr(c) for c in range(0x21))
_URL_INSIDE = str.maketrans("", "", "\t\n\r")


def _raw_end(name: str):
    pattern = _RAW_ENDS.get(name)
    if pattern is None:
        pattern = _RAW_ENDS[name] = re.compile(
            r"</" + re.escape(name) + r"(?=[\t\n\f\r />]|\Z)", re.IGNORECASE)
    return pattern


def _html_bytes(data: bytes, found: Found, depth: int = 0,
                charset: "str | None" = None) -> None:
    for text in _html_texts(data, charset):
        _html_text(text, found, depth)
    if b"<!ENTITY" in data:
        # An entity declared in the document's own DTD is expanded by an
        # XML reader (an .xhtml opened as XML); only expat expands it.
        _xml_bytes(data, found, depth, fallback=False)


def _html_texts(data: bytes, charset: "str | None"):
    """The page as a browser would decode it, and as UTF-8, where the two
    differ in the characters that make markup."""
    if data[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return [data.decode("utf-16", "replace")]
    if data[:3] == codecs.BOM_UTF8:
        return [data[3:].decode("utf-8", "replace")]
    texts = [data.decode("utf-8", "replace")]
    declared = (charset or _declared_charset(data[:4096]) or "").lower()
    if declared.startswith(("utf-16", "utf16", "utf-32", "utf32",
                            "iso-2022", "iso2022", "csiso2022")):
        try:
            texts.append(data.decode(declared, "replace"))
        except LookupError:
            pass
    return texts


_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""",
                           re.IGNORECASE)


def _declared_charset(head: bytes) -> "str | None":
    match = _META_CHARSET.search(head)
    return match.group(1).decode("ascii") if match else None


def _html_text(text: str, found: Found, depth: int = 0) -> None:
    if _html_pass(text, found, depth, rawtext=True):
        _html_pass(text, found, depth, rawtext=False)


def _html_pass(text: str, found: Found, depth: int, rawtext: bool) -> bool:
    """One pass over the page, judging every start tag. Returns whether a
    raw-text element held a "<" -- whether a pass reading it as markup
    could find more."""
    n, i, held = len(text), 0, False
    find = text.find
    while True:
        deadline.check("active content in HTML")
        lt = find("<", i)
        if lt < 0 or lt + 1 >= n:
            return held
        c = text[lt + 1]
        if c == "!":
            if text.startswith("--", lt + 2):
                k = lt + 4
                if text.startswith(">", k):
                    i = k + 1
                elif text.startswith("->", k):
                    i = k + 2
                else:
                    end = _COMMENT_END.search(text, k)
                    if end is None:
                        return held
                    i = end.end()
                continue
            gt = find(">", lt + 2)       # a doctype, CDATA: a bogus comment
            if gt < 0:
                return held
            i = gt + 1
            continue
        if c == "?":
            gt = find(">", lt + 2)
            if gt < 0:
                return held
            i = gt + 1
            continue
        closing = c == "/"
        at = lt + 2 if closing else lt + 1
        if at >= n:
            return held
        c = text[at]
        if not ("a" <= c <= "z" or "A" <= c <= "Z"):
            if closing:
                gt = find(">", at)
                if gt < 0:
                    return held
                i = gt + 1
            else:
                i = lt + 1
            continue
        end = _TAG_NAME.match(text, at).end()
        name = text[at:end].lower()
        attributes, i = _attributes(text, end)
        if closing:
            continue
        _judge_tag(name, attributes, found, depth)
        if name in _RAWTEXT:
            if name == "plaintext":
                close = n
            else:
                match = _raw_end(name).search(text, i)
                close = match.start() if match else n
            if rawtext:
                body = text[i:close]
                if name == "style":
                    _judge_css(body, found)
                if "<" in body:
                    held = True
                i = close


def _attributes(text: str, i: int):
    """A tag's attributes from ``i``, and where the tag ends."""
    if text.startswith(">", i):         # most tags: none at all
        return (), i + 1
    n, out = len(text), []
    while True:
        if len(out) % 1024 == 1023:
            deadline.check("active content in HTML")
        i = _BEFORE_ATTRIBUTE.match(text, i).end()
        if i >= n:
            return out, n
        if text[i] == ">":
            return out, i + 1
        start = i
        i = _ATTRIBUTE_NAME.match(text, i + 1).end()
        name = text[start:i]
        i = _SPACE.match(text, i).end()
        if i < n and text[i] == "=":
            i = _SPACE.match(text, i + 1).end()
            if i >= n:
                out.append((name, ""))
                return out, n
            quote = text[i]
            if quote in "\"'":
                close = text.find(quote, i + 1)
                if close < 0:
                    out.append((name, text[i + 1:]))
                    return out, n
                out.append((name, text[i + 1:close]))
                i = close + 1
            elif quote == ">":
                out.append((name, ""))
                return out, i + 1
            else:
                close = _UNQUOTED.match(text, i).end()
                out.append((name, text[i:close]))
                i = close
        else:
            out.append((name, ""))


_REFRESH_URL = re.compile(r"url\s*=\s*['\"]?([^'\"]*)", re.IGNORECASE)


def _judge_tag(name: str, attributes, found: Found, depth: int,
               decoded: bool = False) -> None:
    """One start tag. Names are compared by their local part, so an
    XHTML "svg:script" is a script whatever its prefix."""
    local = name.rpartition(":")[2]
    if local == "script":
        found.add("html-script")
    elif local in _FRAMES:
        found.add("html-frame")
    if not attributes:
        return
    image = local in _IMAGE_ELEMENTS
    meta = local == "meta"
    values = []
    refresh = False
    for raw, value in attributes:
        attribute = raw.lower()
        if ":" in attribute:
            attribute = attribute.rpartition(":")[2]
        if attribute not in _JUDGED and not attribute.startswith("on") \
                and not meta:
            continue
        if not decoded and "&" in value:
            value = html.unescape(value)
        values.append((attribute, value))
        if meta and attribute == "http-equiv" \
                and value.strip().lower() == "refresh":
            refresh = True
    for attribute, value in values:
        if attribute.startswith("on") and len(attribute) > 2 \
                and attribute not in _NOT_HANDLERS:
            found.add("html-event-handler")
        elif attribute in _URL_ATTRIBUTES:
            _judge_url(value, found, image)
        elif attribute in _URL_LISTS:
            for candidate in value.split(","):
                parts = candidate.split()
                if parts:
                    _judge_url(parts[0], found, image)
        elif attribute in _SPACED_URLS:
            for part in value.split():
                _judge_url(part, found, image)
        elif attribute == "style":
            _judge_css(value, found)
        elif attribute == "srcdoc":
            if depth >= MAX_DEPTH:
                found.cut("it nests deeper than the scan follows")
            else:
                _html_text(value, found, depth + 1)
        elif attribute in _ANIMATED:
            for part in value.split(";"):
                _judge_url(part, found, image, remote=False)
        elif attribute == "attributename":
            if value.strip().lower().rpartition(":")[2].startswith("on"):
                found.add("html-event-handler")
        elif refresh and attribute == "content":
            match = _REFRESH_URL.search(value)
            if match:
                _judge_url(match.group(1), found, image)


def _judge_url(value: str, found: Found, image: bool = False,
               remote: bool = True) -> None:
    url = value.strip(_URL_EDGE).translate(_URL_INSIDE).lower()
    if url.startswith(_SCRIPT_SCHEMES):
        found.add("html-javascript-url")
        return
    if url.startswith("data:"):
        kind = url[5:].split(",", 1)[0].split(";", 1)[0].strip()
        if kind in _ACTIVE_DATA or (kind in _MARKUP_DATA and not image):
            found.add("html-javascript-url")
        return
    if not remote:
        return
    url = url.replace("\\", "/")
    if url.startswith("//"):
        found.add("html-remote-resource")
        return
    scheme = _SCHEME.match(url)
    if scheme and scheme.group(1) in _REMOTE_SCHEMES:
        found.add("html-remote-resource")


_CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\n\r\f]?|(.))", re.DOTALL)
_CSS_URL = re.compile(r"url\(\s*(?:\"([^\"]*)\"|'([^']*)'|([^)\s]*))",
                      re.IGNORECASE)
_CSS_STRING = re.compile(r"\"([^\"\n]*)\"|'([^'\n]*)'")
_CSS_CODE = re.compile(r"expression\s*\(|behaviou?r\s*:|-moz-binding\s*:",
                       re.IGNORECASE)


def _css_character(match) -> str:
    digits, other = match.group(1), match.group(2)
    if digits:
        point = int(digits, 16)
        return chr(point) if 0 < point <= 0x10FFFF \
            and not 0xD800 <= point <= 0xDFFF else "�"
    return "" if other == "\n" else other


def _uncommented(css: str) -> str:
    """CSS without its comments, found by scanning, never by backtracking."""
    if "/*" not in css:
        return css
    out, at = [], 0
    while True:
        start = css.find("/*", at)
        if start < 0:
            out.append(css[at:])
            return "".join(out)
        out.append(css[at:start])
        end = css.find("*/", start + 2)
        if end < 0:
            return "".join(out)
        at = end + 2


def _judge_css(css: str, found: Found) -> None:
    """A style: what it loads (url(), @import) and what runs in it
    (expression(), behavior, -moz-binding), escapes decoded first."""
    if not css:
        return
    css = _uncommented(css)
    if "\\" in css:
        css = _CSS_ESCAPE.sub(_css_character, css)
    if _CSS_CODE.search(css):
        found.add("html-javascript-url")
    for match in _CSS_URL.finditer(css):
        _judge_url(match.group(1) or match.group(2) or match.group(3) or "",
                   found)
    for match in _CSS_STRING.finditer(css):
        _judge_url(match.group(1) or match.group(2) or "", found)


def _html(path: Path, max_bytes: int, found: Found) -> None:
    _html_bytes(_read(path, max_bytes, found), found)


# ---- XML: SVG, XHTML, flat OpenDocument, Office 2003 XML -----------------
#
# Read by expat with its namespaces resolved, so a prefix changes nothing:
# <s:script xmlns:s="http://www.w3.org/2000/svg"> is a script, and a
# <script> in a data file's own vocabulary is one too. HTML's rules for
# attributes apply to elements of the SVG and XHTML namespaces (and to a
# document whose root is an unqualified <svg> or <html>). Entities the
# document declares are expanded, as a browser's XML reader expands them;
# nothing external is loaded.

_SVG_NS = "http://www.w3.org/2000/svg"
_XHTML_NS = "http://www.w3.org/1999/xhtml"
_MARKUP_NS = frozenset({_SVG_NS, _XHTML_NS})
_ODF_OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_ODF_SCRIPT = "urn:oasis:names:tc:opendocument:xmlns:script:1.0"
_ODF_PRESENTATION = "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
_ODF_DRAW = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
_ODF_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_WORD_NS = frozenset({
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
    "http://schemas.microsoft.com/office/word/2003/wordml"})
_SHEET_NS = frozenset({
    "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "http://purl.oclc.org/ooxml/spreadsheetml/main"})
_SHEET_2003 = "urn:schemas-microsoft-com:office:spreadsheet"
_OFFICE_2003 = "urn:schemas-microsoft-com:office:office"
_ODF_CODE = ("vnd.sun.star.script:", "vnd.sun.star.basic:", "macro:")
_PP_CODE = ("ppaction://macro", "ppaction://program")
_ODF_DDE = frozenset({"dde-link", "dde-source", "dde-connection-decl",
                      "dde-connection"})
#: The first word of a DDE field's instruction.
_DDE_FIELD = re.compile(r"\A\s*DDE(?:AUTO)?(?![A-Za-z0-9])", re.IGNORECASE)
#: A formula that asks another program for its value: app|'topic'!item,
#: or the DDE() function.
_DDE_FORMULA = re.compile(r"\|\s*'[^']*'\s*!|[\w.]\|[^!|]{0,256}!"
                          r"|(?<![\w.])DDE(?:AUTO)?\s*\(", re.IGNORECASE)
_STYLESHEET_HREF = re.compile(r"""href\s*=\s*(?:"([^"]*)"|'([^']*)')""")


class _Xml:
    """The handlers of one expat pass."""

    def __init__(self, found: Found, depth: int) -> None:
        self.found, self.depth = found, depth
        self.markup_root: "bool | None" = None
        self.fields: list = []            # a DDE check per open field
        self.collect: "list | None" = None
        self.collecting = ""

    def start(self, name, attributes):
        uri, _, local = name.rpartition(" ")
        found = self.found
        if self.markup_root is None:
            self.markup_root = uri in _MARKUP_NS or (
                not uri and local.lower() in ("svg", "html"))
        markup = uri in _MARKUP_NS or (not uri and self.markup_root)
        if local == "script":
            found.add("odf-scripts" if uri == _ODF_OFFICE else "html-script")
        elif uri == _ODF_SCRIPT and local == "event-listener":
            found.add("odf-scripts")
        elif uri == _ODF_PRESENTATION and local == "event-listener":
            if attributes.get(_ODF_PRESENTATION + " action") in ("execute",
                                                                 "verb"):
                found.add("odf-scripts")
        elif uri == _ODF_DRAW and local == "object-ole":
            found.add("office-ole-object")
        elif uri == _ODF_DRAW and local in ("applet", "plugin"):
            found.add("odf-scripts")
        elif local in _ODF_DDE and uri.startswith("urn:oasis:names:tc:"
                                                  "opendocument:"):
            found.add("office-dde")
        elif uri in _WORD_NS:
            self._word(uri, local, attributes)
        elif uri in _SHEET_NS:
            if local == "ddeLink":
                found.add("office-dde")
            elif local == "oleLink":
                found.add("office-ole-object")
            elif local == "f":
                self.collect, self.collecting = [], "formula"
        elif uri == _OFFICE_2003 and local == "OLEObject":
            found.add("office-ole-object")
        elif markup and local == "style":
            self.collect, self.collecting = [], "style"
        judged = []
        for key, value in attributes.items():
            a_uri, _, a_local = key.rpartition(" ")
            low = value.strip().lower()
            if low.startswith(_ODF_CODE):
                found.add("odf-scripts")
            elif low.startswith(_PP_CODE):
                found.add("office-macros")
            if a_uri == _SHEET_2003 and a_local == "Formula" or \
                    a_uri == _ODF_TABLE and a_local == "formula":
                if _DDE_FORMULA.search(value):
                    found.add("office-dde")
            if markup:
                judged.append((a_local, value))
        if markup:
            _judge_tag(local, judged, found, self.depth, decoded=True)

    def _word(self, uri, local, attributes):
        found = self.found
        if local == "fldChar":
            kind = attributes.get(uri + " fldCharType")
            if kind == "begin":
                self.fields.append([[], False])
            elif kind in ("separate", "end") and self.fields:
                code, judged = self.fields[-1]
                if not judged:
                    self._field("".join(code))
                    self.fields[-1][1] = True
                if kind == "end":
                    self.fields.pop()
        elif local == "instrText":
            self.collect, self.collecting = [], "instr"
        elif local == "fldSimple":
            self._field(attributes.get(uri + " instr", ""))
        elif local == "binData":
            named = attributes.get(uri + " name", "").lower()
            if named.endswith("editdata.mso"):
                found.add("office-macros")
            elif named.endswith("oledata.mso"):
                found.add("office-ole-object")
        elif local == "docSuppData":
            found.add("office-macros")

    def _field(self, code):
        if _DDE_FIELD.match(code):
            self.found.add("office-dde")

    def chars(self, data):
        if self.collect is not None and len(self.collect) < 4096:
            self.collect.append(data)

    def end(self, name):
        if self.collect is None:
            return
        text, what = "".join(self.collect), self.collecting
        uri, _, local = name.rpartition(" ")
        if what == "instr" and local == "instrText":
            if self.fields:
                self.fields[-1][0].append(text)
            else:
                self._field(text)
        elif what == "formula" and local == "f":
            if _DDE_FORMULA.search(text):
                self.found.add("office-dde")
        elif what == "style" and local == "style":
            _judge_css(text, self.found)
        else:
            return
        self.collect, self.collecting = None, ""

    def instruction(self, target, data):
        if target.lower() == "xml-stylesheet":
            match = _STYLESHEET_HREF.search(data)
            if match:
                _judge_url(match.group(1) or match.group(2) or "",
                           self.found)

    def doctype(self, name, system, public, internal):
        # A DOCTYPE's system identifier: never loaded here, but a reader
        # that resolves it would load it.
        if system:
            _judge_url(system, self.found)

    def entity(self, name, is_parameter, value, base, system, public,
               notation):
        if system:
            _judge_url(system, self.found)


def _expat(handlers: _Xml):
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.buffer_text = True
    parser.StartElementHandler = handlers.start
    parser.EndElementHandler = handlers.end
    parser.CharacterDataHandler = handlers.chars
    parser.ProcessingInstructionHandler = handlers.instruction
    parser.StartDoctypeDeclHandler = handlers.doctype
    parser.EntityDeclHandler = handlers.entity
    return parser


class _Hint:
    """Words, any of which in a part's bytes (compared in lowercase) says
    the part is worth parsing. Substring tests of a lowercased copy, not a
    case-blind regex: an alternation of literals under IGNORECASE ran at
    10 MB/s."""

    def __init__(self, *words: bytes, pattern=None) -> None:
        self.words = tuple(word.lower() for word in words)
        self.pattern = pattern

    def search(self, data: bytes) -> bool:
        low = data.lower()
        return any(word in low for word in self.words) or bool(
            self.pattern and self.pattern.search(data))


#: Everything the XML pass can report needs one of these in the bytes: an
#: element or attribute name, a namespace or root that makes the document
#: markup, a declaration -- or a character reference or an entity, which
#: can spell any of them.
_XML_HINT = _Hint(
    b"script", b"event-listener", b"object-ole", b"applet", b"plugin",
    b"vnd.sun.star", b"macro:", b"ppaction", b"entity", b"stylesheet",
    b"doctype", b"dde", b"&#", b"instr", b"bindata", b"docsuppdata",
    b"oleobject", b"formula", b"svg", b"html")


def _xml_bytes(data: bytes, found: Found, depth: int = 0,
               fallback: bool = True) -> None:
    if fallback and data[:2] not in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE) \
            and b"\x00" not in data[:4] and not _XML_HINT.search(data):
        return
    handlers = _Xml(found, depth)
    parser = _expat(handlers)
    try:
        for at in range(0, len(data), _PIECE):
            deadline.check("active content in XML")
            parser.Parse(data[at:at + _PIECE], False)
        parser.Parse(b"", True)
    except (expat.ExpatError, ValueError, LookupError):
        # Not well-formed, or in an encoding expat does not know. A browser
        # shows what it read up to the error; an HTML reader of the same
        # bytes -- an SVG put inline -- reads on. So does this.
        if fallback:
            _html_bytes(data, found, depth)
        return


def _xml(path: Path, max_bytes: int, found: Found) -> None:
    _xml_bytes(_read(path, max_bytes, found), found)


# ---- PDF ---------------------------------------------------------------
#
# The file is tokenized from start to end: literal and hex strings and
# comments are stepped over, names decoded from their #xx escapes, and the
# body of a stream skipped -- unless the stream is an object stream, which
# is decoded (Flate, LZW, ASCIIHex, ASCII85, RunLength, PNG and TIFF
# predictors) and tokenized in turn, or its body holds an "N G obj" header,
# which a reader following the cross-reference table would parse from
# there. A string that runs into the next object's header ends there.
#
# No cross-reference table is needed and none is trusted: every object in
# the file is read, whichever of its versions a viewer would take. An open
# action or an additional action is parsed and followed through references
# (to every definition of the object); one that cannot be followed counts
# as active.

_REGULAR = rb"[^\x00\t\n\f\r ()<>\[\]{}/%]"
_PDF_WHITE = b"\x00\t\n\f\r "
_PDF_CLASSES = {
    b"JS": "pdf-javascript", b"JavaScript": "pdf-javascript",
    b"OnInstantiate": "pdf-javascript",
    b"Launch": "pdf-launch", b"SubmitForm": "pdf-submit-form",
    b"ImportData": "pdf-import-data",
    b"RichMedia": "pdf-richmedia", b"RichMediaContent": "pdf-richmedia",
    b"RichMediaExecute": "pdf-richmedia", b"RichMediaSettings": "pdf-richmedia",
    b"XFA": "pdf-xfa",
    b"EmbeddedFile": "pdf-embedded-file", b"EmbeddedFiles": "pdf-embedded-file",
    b"EF": "pdf-embedded-file", b"FileAttachment": "pdf-embedded-file"}
_PDF_FOLLOWED = (b"OpenAction", b"AA", b"Encrypt")
_PDF_TOKEN = re.compile(
    rb"(?P<dict><<)|(?P<close>>>)|(?P<str>\()|(?P<hex><)|(?P<com>%)"
    rb"|(?<![^\x00\t\n\f\r ()<>\[\]{}%])(?P<stream>stream)(?!" + _REGULAR + rb")"
    rb"|/(?P<name>" + b"|".join(sorted(list(_PDF_CLASSES) + list(_PDF_FOLLOWED),
                                       key=len, reverse=True))
    + rb")(?!" + _REGULAR + rb")"
    rb"|/(?P<esc>" + _REGULAR + rb"*#" + _REGULAR + rb"*)")
_PAREN = re.compile(rb"[()\\]")
_EOL = re.compile(rb"\r\n|\n|\r")
#: White space between two tokens, a comment included (one of a bounded
#: length: an unbounded one made "1 %1 %1 %..." quadratic).
_GAP = rb"(?:[\x00\t\n\f\r ]|%[^\r\n]{0,256}[\r\n])+"
_OBJ_HEADER = re.compile(
    rb"(?<![0-9])([0-9]{1,10})" + _GAP + rb"([0-9]{1,5})" + _GAP + rb"obj(?!"
    + _REGULAR + rb")")
#: Bytes that may not stand right before the keyword "obj" (a regular
#: character, or the "/" of a name), and right after it.
_NOT_BEFORE_OBJ = frozenset(set(range(256)) - set(b"\x00\t\n\f\r ()<>[]{}%"))
_NOT_AFTER_OBJ = frozenset(set(range(256)) - set(b"\x00\t\n\f\r ()<>[]{}/%"))


def _obj_marks(data: bytes) -> list:
    """Where the keyword "obj" stands as a token: where an object begins,
    whatever its header looks like. One pass over the file, stream bodies
    included, by find: a regex over every byte of a 6 MB image cost 90 ms,
    this a millisecond."""
    marks, at, n = [], data.find(b"obj"), len(data)
    while at >= 0:
        if (at == 0 or data[at - 1] not in _NOT_BEFORE_OBJ) and \
                (at + 3 >= n or data[at + 3] not in _NOT_AFTER_OBJ):
            marks.append(at)
        at = data.find(b"obj", at + 3)
    return marks
_DIRECT_LENGTH = re.compile(
    rb"/Length[\x00\t\n\f\r ]+([0-9]{1,12})(?![0-9]|[\x00\t\n\f\r ]+[0-9]+"
    rb"[\x00\t\n\f\r ]+R(?!" + _REGULAR + rb"))")
_NAME_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")
#: How many references an action is followed through.
_MAX_HOPS = 16


def _pdf_name(raw: bytes) -> bytes:
    return _NAME_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), raw)


class _Ref(NamedTuple):
    number: int
    generation: int


class _Name(str):
    """A PDF name, decoded."""


class _Word(str):
    """A PDF keyword or other bare token."""


_STRING = object()
_UNKNOWN = object()


class _Unparsed(Exception):
    """A value the mini parser could not read."""


_VALUE = re.compile(
    rb"(?:[\x00\t\n\f\r ]|%[^\r\n%]*)*(?:(<<)|(>>)|(\[)|(\])|(\()|(<)"
    rb"|/(" + _REGULAR + rb"*)|([+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+))(?!"
    + _REGULAR + rb")|(" + _REGULAR + rb"+)|(.))", re.DOTALL)


class _Pdf:
    """The tokenizing pass over one PDF's bytes, and what it keeps for
    following open actions afterwards."""

    def __init__(self, data: bytes, found: Found, depth: int) -> None:
        self.data, self.found, self.depth = data, found, depth
        #: Where each "obj" keyword stands: a literal string that has not
        #: closed by the next one ends there, and a stream body holding one
        #: holds an object a reader following the cross-reference table
        #: would parse.
        self.marks = _obj_marks(data)
        self.opens: list = []
        self.automatic: list = []
        self.encrypted = False
        self.packed: dict = {}            # number -> [(buffer, at)]
        self._index: "dict | None" = None

    def run(self) -> None:
        self._region(self.data, 0, len(self.data), self.marks, True, 0)
        for buf, at, end in self.opens:
            self._open_action(buf, at, end)
        for buf, at, end in self.automatic:
            self._additional(buf, at, end)
        if self.encrypted:
            self.found.cut("it is encrypted")

    # -- the pass --------------------------------------------------------

    def _region(self, buf, start, end, heads, streams, depth):
        if depth > MAX_DEPTH:
            self.found.cut("it nests deeper than the scan follows")
            return
        pos, search, found = start, _PDF_TOKEN.search, self.found
        # Where the outermost dictionary last opened: a stream's own
        # dictionary is the one that ends at its keyword.
        nesting, opened = 0, start
        while pos < end:
            deadline.check("active content in a PDF")
            m = search(buf, pos, end)
            if m is None:
                return
            kind = m.lastgroup
            if kind == "dict":
                if nesting == 0:
                    opened = m.start()
                nesting += 1
                pos = m.end()
            elif kind == "close":
                nesting = max(0, nesting - 1)
                pos = m.end()
            elif kind == "str":
                pos = _literal_end(buf, m.start(), _next(heads, m.start(), end))
            elif kind == "hex":
                limit = _next(heads, m.start(), end)
                close = buf.find(b">", m.end(), limit)
                pos = close + 1 if close >= 0 else limit
            elif kind == "com":
                eol = _EOL.search(buf, m.end(), end)
                pos = eol.end() if eol else end
            elif kind == "stream":
                if streams:
                    pos = self._stream(buf, opened, m.start(), m.end(), end,
                                       depth)
                    nesting, opened = 0, pos
                else:
                    pos = m.end()
            else:
                raw = m.group(kind)
                name = raw if kind == "name" else _pdf_name(raw)
                cls = _PDF_CLASSES.get(name)
                if cls:
                    found.add(cls)
                elif name == b"OpenAction":
                    self.opens.append((buf, m.end(), end))
                elif name == b"AA":
                    self.automatic.append((buf, m.end(), end))
                elif name == b"Encrypt":
                    self.encrypted = True
                pos = m.end()

    def _stream(self, buf, opened, keyword, after, end, depth):
        eol = _EOL.search(buf, after, end)
        body = eol.end() if eol else end
        # The stream's dictionary: from where the tokenizer last saw an
        # outermost "<<" -- strings and comments stepped over, so neither a
        # string nor a name holding "obj" or "<<" moves it.
        head = buf[opened:keyword]
        stop = resume = None
        length = _DIRECT_LENGTH.search(head)
        if length:
            close = body + int(length.group(1))
            if close <= end:
                k = close
                while k < end and buf[k] in _PDF_WHITE:
                    k += 1
                if buf.startswith(b"endstream", k):
                    stop, resume = close, k + 9
        if stop is None:
            close = buf.find(b"endstream", body, end)
            stop, resume = (close, close + 9) if close >= 0 else (end, end)
        if b"ObjStm" in head or b"First" in head or b"#" in head:
            if self._object_stream(head, buf[body:stop], depth):
                return resume
        if _holds_an_object(self.marks, body, stop):
            # A reader that follows the cross-reference table to an offset
            # inside this body parses an object there.
            self._region(buf, body, stop, self.marks, True, depth + 1)
        return resume

    def _object_stream(self, head, body, depth) -> bool:
        """Decode and read an object stream; False if the stream is none."""
        info = _stream_dict(head)
        if info is None:
            if b"ObjStm" in _pdf_name(head):
                self.found.cut("an object stream's dictionary cannot be read")
                return True
            return False
        if not isinstance(info, dict):
            return False
        typed = info.get("Type") == "ObjStm"
        count, first = info.get("N"), info.get("First")
        if not typed and not (isinstance(count, int) and isinstance(first, int)):
            return False
        try:
            data = _decode(body, info.get("Filter"), info.get("DecodeParms"),
                           self.found)
        except _Undecodable as exc:
            self.found.cut(str(exc))
            return True
        heads = []
        if isinstance(first, int) and first >= 0:
            numbers = [int(x) for x in re.findall(rb"[0-9]+", data[:first])]
            for number, offset in zip(numbers[0::2], numbers[1::2]):
                at = first + offset
                if at < len(data):
                    heads.append(at)
                    self.packed.setdefault(number, []).append((data, at))
            heads.sort()
        # Without /First no reader finds an object in it, but every byte of
        # it is read all the same: what it holds is said.
        self._region(data, 0, len(data), heads, False, depth + 1)
        return True

    # -- following actions -----------------------------------------------

    def _definitions(self, number):
        if self._index is None:
            self._index = {}
            for m in _OBJ_HEADER.finditer(self.data):
                deadline.check("active content in a PDF")
                self._index.setdefault(int(m.group(1)), []).append(m.end())
        places = [(self.data, at) for at in self._index.get(number, ())]
        return (places + self.packed.get(number, []))[:8]

    def _resolved(self, value, hops):
        """``value``, or every definition of the object it refers to."""
        if not isinstance(value, _Ref):
            yield value
            return
        if hops > _MAX_HOPS:
            yield _UNKNOWN
            return
        places = self._definitions(value.number)
        if not places:
            yield _UNKNOWN
        for buf, at in places:
            try:
                resolved, _ = _parse(buf, at, len(buf), 0, [0])
            except _Unparsed:
                yield _UNKNOWN
                continue
            yield from self._resolved(resolved, hops + 1)

    def _open_action(self, buf, at, end):
        try:
            value, _ = _parse(buf, at, end, 0, [0])
        except _Unparsed:
            self.found.add("pdf-open-action")
            return
        for action in self._resolved(value, 0):
            if action is _UNKNOWN:
                self.found.add("pdf-open-action")
            elif isinstance(action, dict):
                self._action(action, True, 0)
            # an array, a name or a string is a destination: a page to
            # open at, nothing that runs

    def _additional(self, buf, at, end):
        try:
            value, _ = _parse(buf, at, end, 0, [0])
        except _Unparsed:
            return
        for triggers in self._resolved(value, 0):
            if not isinstance(triggers, dict):
                continue
            for trigger in triggers.values():
                for action in self._resolved(trigger, 0):
                    if isinstance(action, dict):
                        self._action(action, False, 0)

    _ACTIONS = {"JavaScript": "pdf-javascript", "Launch": "pdf-launch",
                "SubmitForm": "pdf-submit-form",
                "ImportData": "pdf-import-data",
                "RichMediaExecute": "pdf-richmedia",
                "GoToR": "pdf-goto-remote", "GoToE": "pdf-goto-remote",
                "URI": "pdf-goto-remote"}

    def _action(self, action, opening, hops):
        """An action that runs by itself: on opening the document, or on
        one of the additional-action triggers."""
        if hops > _MAX_HOPS:
            if opening:
                self.found.add("pdf-open-action")
            return
        kinds = list(self._resolved(action.get("S"), hops))
        for kind in kinds:
            if kind == "GoTo":
                continue
            if opening:
                self.found.add("pdf-open-action")
            cls = self._ACTIONS.get(kind) if isinstance(kind, str) else None
            if cls:
                self.found.add(cls)
        following = action.get("Next")
        if following is None:
            return
        for nxt in self._resolved(following, hops + 1):
            for item in (nxt if isinstance(nxt, list) else [nxt]):
                for resolved in self._resolved(item, hops + 1):
                    if isinstance(resolved, dict):
                        self._action(resolved, opening, hops + 1)
                    elif resolved is _UNKNOWN and opening:
                        self.found.add("pdf-open-action")


def _next(heads, at, end):
    """The start of the first object header after ``at``, or ``end``."""
    index = bisect.bisect_right(heads, at)
    return min(heads[index], end) if index < len(heads) else end


def _literal_end(buf, start, limit):
    """The end of the literal string at ``start``: balanced parentheses,
    backslash escapes; at ``limit`` if it does not close before."""
    depth, pos, steps = 0, start, 0
    while True:
        steps += 1
        if steps % 4096 == 0:
            deadline.check("active content in a PDF")
        m = _PAREN.search(buf, pos, limit)
        if m is None:
            return limit
        c = buf[m.start()]
        if c == 0x5C:
            pos = m.start() + 2
            continue
        if c == 0x28:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
        pos = m.end()


def _holds_an_object(marks, start, end) -> bool:
    """Whether an "obj" keyword stands between ``start`` and ``end``."""
    index = bisect.bisect_left(marks, start)
    return index < len(marks) and marks[index] < end


def _stream_dict(head):
    """The dictionary that ends where ``head`` ends -- the stream's own --
    or None."""
    pos, last, count = 0, None, [0]
    while True:
        at = head.find(b"<<", pos)
        if at < 0:
            break
        try:
            value, after = _parse(head, at, len(head), 0, count)
        except _Unparsed:
            pos = at + 2
            continue
        last, pos = (value, after), after
    if last is None or head[last[1]:].strip(_PDF_WHITE) \
            or not isinstance(last[0], dict):
        return None
    return last[0]


def _parse(buf, pos, end, depth, count):
    """One PDF value at ``pos``: (value, where it ends). Bounded in depth
    and in the tokens it reads (``count`` is shared by the calls)."""
    if depth > 64:
        raise _Unparsed
    count[0] += 1
    if count[0] > 100_000:
        raise _Unparsed
    m = _VALUE.match(buf, pos, end)
    if m is None or m.lastindex is None:
        raise _Unparsed
    g = m.lastindex
    if g == 1:
        out = {}
        pos = m.end()
        while True:
            k = _VALUE.match(buf, pos, end)
            if k is None or k.lastindex is None:
                raise _Unparsed
            if k.lastindex == 2:
                return out, k.end()
            if k.lastindex != 7:
                raise _Unparsed
            value, pos = _parse(buf, k.end(), end, depth + 1, count)
            out[_pdf_name(k.group(7)).decode("latin-1")] = value
    if g == 3:
        out = []
        pos = m.end()
        while True:
            k = _VALUE.match(buf, pos, end)
            if k is None or k.lastindex is None:
                raise _Unparsed
            if k.lastindex == 4:
                return out, k.end()
            value, pos = _parse(buf, pos, end, depth + 1, count)
            out.append(value)
    if g == 5:
        return _STRING, _literal_end(buf, m.start(5), end)
    if g == 6:
        close = buf.find(b">", m.end(), end)
        if close < 0:
            raise _Unparsed
        return _STRING, close + 1
    if g == 7:
        return _Name(_pdf_name(m.group(7)).decode("latin-1")), m.end()
    if g == 8:
        text = m.group(8)
        if b"." in text:
            return float(text), m.end()
        number = int(text)
        second = _VALUE.match(buf, m.end(), end)
        if second is not None and second.lastindex == 8 \
                and b"." not in second.group(8):
            third = _VALUE.match(buf, second.end(), end)
            if third is not None and third.lastindex == 9 \
                    and third.group(9) == b"R":
                return _Ref(number, int(second.group(8))), third.end()
        return number, m.end()
    if g == 9:
        return _Word(m.group(9).decode("latin-1")), m.end()
    raise _Unparsed


class _Undecodable(Exception):
    """An object stream the scan cannot decode, in words for the note."""


_FILTERS = {"FlateDecode": "flate", "Fl": "flate", "LZWDecode": "lzw",
            "LZW": "lzw", "ASCIIHexDecode": "hex", "AHx": "hex",
            "ASCII85Decode": "a85", "A85": "a85", "RunLengthDecode": "rl",
            "RL": "rl"}


def _decode(data, filters, parameters, found):
    names = filters if isinstance(filters, list) else \
        ([] if filters is None else [filters])
    params = parameters if isinstance(parameters, list) else \
        [parameters] * len(names)
    for index, name in enumerate(names):
        how = _FILTERS.get(name) if isinstance(name, _Name) else None
        if how is None:
            raise _Undecodable(f"an object stream is encoded in a way the "
                               f"scan does not read ({str(name)[:32]})")
        p = params[index] if index < len(params) else None
        if p is not None and not isinstance(p, dict):
            raise _Undecodable("an object stream's decoding parameters "
                               "cannot be read")
        p = p or {}
        if how == "flate":
            data = _inflate(data, found)
            data = _predicted(data, p)
        elif how == "lzw":
            early = p.get("EarlyChange", 1)
            data = _lzw(data, 1 if early not in (0, 1) else early, found)
            data = _predicted(data, p)
        elif how == "hex":
            data = _ascii_hex(data)
        elif how == "a85":
            data = _ascii85(data)
        else:
            data = _run_length(data, found)
        found.spend(len(data))
    return data


def _inflate(data, found):
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(data, max(1, found.left + 1))
    except zlib.error:
        raise _Undecodable("an object stream is damaged") from None
    if len(out) > found.left:
        raise _Cut(f"it unpacks to more than {DECODE_BUDGET // (1 << 20)} MiB")
    if not out and data:
        raise _Undecodable("an object stream is damaged")
    return out


def _predicted(data, p):
    predictor = p.get("Predictor", 1)
    if not isinstance(predictor, int) or predictor == 1:
        return data
    colors, bits = p.get("Colors", 1), p.get("BitsPerComponent", 8)
    columns = p.get("Columns", 1)
    if not all(isinstance(v, int) and 0 < v <= 1 << 16
               for v in (colors, bits, columns)):
        raise _Undecodable("an object stream's predictor cannot be read")
    width = max(1, colors * bits // 8)
    row = (colors * bits * columns + 7) // 8
    if predictor == 2:
        if bits != 8:
            raise _Undecodable("an object stream's predictor cannot be read")
        out = bytearray(data)
        for start in range(0, len(out), row):
            for i in range(start + width, min(start + row, len(out))):
                out[i] = (out[i] + out[i - width]) & 0xFF
        return bytes(out)
    out, previous = bytearray(), bytearray(row)
    for start in range(0, len(data), row + 1):
        deadline.check("active content in a PDF")
        kind, line = data[start], bytearray(data[start + 1:start + 1 + row])
        line += bytes(row - len(line))
        for i in range(row):
            left = line[i - width] if i >= width else 0
            up = previous[i]
            if kind == 1:
                line[i] = (line[i] + left) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + up) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + (left + up) // 2) & 0xFF
            elif kind == 4:
                corner = previous[i - width] if i >= width else 0
                estimate = left + up - corner
                pa, pb, pc = abs(estimate - left), abs(estimate - up), \
                    abs(estimate - corner)
                line[i] = (line[i] + (left if pa <= pb and pa <= pc else
                                      up if pb <= pc else corner)) & 0xFF
        out += line
        previous = line
    return bytes(out)


def _ascii_hex(data):
    end = data.find(b">")
    digits = re.sub(rb"[^0-9A-Fa-f]", b"", data if end < 0 else data[:end])
    if len(digits) % 2:
        digits += b"0"
    return bytes.fromhex(digits.decode("ascii"))


def _ascii85(data):
    data = data.strip(_PDF_WHITE)
    if data.startswith(b"<~"):
        data = data[2:]
    end = data.find(b"~>")
    if end >= 0:
        data = data[:end]
    try:
        return base64.a85decode(data, ignorechars=_PDF_WHITE + b"\x0b")
    except ValueError:
        raise _Undecodable("an object stream is damaged") from None


def _run_length(data, found):
    out, i, n = bytearray(), 0, len(data)
    while i < n:
        length = data[i]
        if length == 128:
            break
        if length < 128:
            out += data[i + 1:i + 2 + length]
            i += 2 + length
        else:
            out += data[i + 1:i + 2] * (257 - length)
            i += 2
        if len(out) > found.left:
            raise _Cut(f"it unpacks to more than "
                       f"{DECODE_BUDGET // (1 << 20)} MiB")
    return bytes(out)


def _lzw(data, early, found):
    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    width, previous = 9, None
    buffer = bits = 0
    for byte in data:
        buffer, bits = (buffer << 8) | byte, bits + 8
        while bits >= width:
            bits -= width
            code = (buffer >> bits) & ((1 << width) - 1)
            buffer &= (1 << bits) - 1
            if code == 256:
                table, width, previous = table[:258], 9, None
                continue
            if code == 257:
                return bytes(out)
            if previous is None:
                if code >= len(table):
                    raise _Undecodable("an object stream is damaged")
                entry = table[code]
            elif code < len(table):
                entry = table[code]
                table.append(previous + entry[:1])
            elif code == len(table):
                entry = previous + previous[:1]
                table.append(entry)
            else:
                raise _Undecodable("an object stream is damaged")
            out += entry
            previous = entry
            if len(table) + early >= (1 << width) and width < 12:
                width += 1
            if len(out) > found.left:
                raise _Cut(f"it unpacks to more than "
                           f"{DECODE_BUDGET // (1 << 20)} MiB")
    return bytes(out)


def _pdf_bytes(data: bytes, found: Found, depth: int = 0) -> None:
    _Pdf(data, found, depth).run()


def _pdf(path: Path, max_bytes: int, found: Found) -> None:
    _pdf_bytes(_read(path, max_bytes, found), found)


# ---- zip: OOXML, OpenDocument, EPUB and plain archives -------------------

#: Parts read with expat only when their bytes hint at what they are read
#: for; a character reference could spell any of it, so it is a hint too.
#: A DDE field is an instrText element or an instr attribute.
_WORD_HINT = _Hint(b"instr", b"&#")
_SHEET_HINT = _Hint(b"|", b"dde", b"&#")
_SLIDE_HINT = _Hint(b"ppaction", b"&#")
_ODF_HINT = _Hint(b"event-listener", b"vnd.sun.star.", b"macro:",
                  b"object-ole", b"applet", b"plugin", b"dde", b"&#",
                  pattern=re.compile(rb"[<:]script[\t\n\r />]"))


def _zip_path(path: Path, max_bytes: int, found: Found) -> None:
    try:
        zipguard.check(path)
    except zipguard.ZipLimitError:
        found.cut("it holds more members than sift lists")
        return
    try:
        with zipfile.ZipFile(path) as zf:
            _zip(zf, found, 0)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError,
            NotImplementedError, ValueError, KeyError):
        found.cut("it is not a zip archive that can be listed")


def _zip_bytes(data: bytes, found: Found, depth: int) -> None:
    fh = io.BytesIO(data)
    try:
        zipguard.check_file(fh)
    except zipguard.ZipLimitError:
        found.cut("a zip archive in it holds more members than sift lists")
        return
    try:
        with zipfile.ZipFile(fh) as zf:
            _zip(zf, found, depth)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError,
            NotImplementedError, ValueError, KeyError):
        found.cut("a zip archive in it cannot be listed")


def _member(zf, info, limit, found) -> "bytes | None":
    """A member's bytes, the first ``limit`` or all; None, said, when it
    cannot be read."""
    want = info.file_size if limit is None else min(limit, info.file_size)
    try:
        with zf.open(info) as fh:
            data = fh.read(want)
    except (RuntimeError, zipfile.BadZipFile, zlib.error, OSError, EOFError,
            NotImplementedError, ValueError, lzma.LZMAError):
        found.cut("a member of it cannot be read")
        return None
    found.spend(len(data))
    return data


def _flavour(zf, names) -> str:
    if "[content_types].xml" in names:
        return "ooxml"
    info = names.get("mimetype")
    if info is not None and info.file_size < 256:
        try:
            declared = zf.read(info).decode("ascii", "replace").strip()
        except Exception:  # noqa: BLE001 - an unreadable mimetype is no flavour
            declared = ""
        if declared.startswith("application/vnd.oasis.opendocument"):
            return "odf"
        if declared == "application/epub+zip":
            return "epub"
    if "meta-inf/container.xml" in names:
        return "epub"
    return "plain"


def _zip(zf, found: Found, depth: int) -> None:
    if depth > MAX_DEPTH:
        found.cut("it nests deeper than the scan follows")
        return
    infos = zf.infolist()
    found.members += len(infos)
    if found.members > MAX_MEMBERS:
        raise _Cut("it holds more members than sift lists")
    names = {}
    for info in infos:
        names.setdefault(info.filename.lower(), info)
    flavour = _flavour(zf, names)
    for info in infos:
        deadline.check("active content in an archive")
        if info.is_dir():
            continue
        low = info.filename.lower().replace("\\", "/")
        if flavour == "ooxml":
            _ooxml_name(low, found)
        elif flavour == "odf":
            if low.startswith(("basic/", "scripts/")):
                found.add("odf-scripts")
        encrypted = bool(info.flag_bits & 0x1)
        if encrypted:
            _judge_member(low, None, found)
            found.cut("some of its members are encrypted")
            continue
        head = _member(zf, info, HEAD, found) if info.file_size else b""
        if head is None:
            _judge_member(low, None, found)
            continue
        _judge_member(low, head, found)
        kind = _container(head)
        if kind in ("7z", "rar"):
            found.cut(f"a {kind} archive in it is not looked into")
        elif kind:
            data = head if info.file_size <= len(head) \
                else _member(zf, info, None, found)
            if data is not None:
                if kind == "zip":
                    _zip_bytes(data, found, depth + 1)
                else:
                    _compressed(io.BytesIO(data), kind, low, found, depth + 1)
        if flavour == "ooxml":
            _ooxml_part(zf, info, low, found, depth)
        elif flavour == "odf":
            if low.rsplit("/", 1)[-1] in ("content.xml", "styles.xml"):
                _xml_member(zf, info, _ODF_HINT, found, depth)
        elif flavour == "epub":
            ext = _ext(low)
            if ext in _HTML_EXT:
                data = _member(zf, info, None, found)
                if data is not None:
                    _html_bytes(data, found, depth + 1)
            elif ext == ".css":
                data = _member(zf, info, None, found)
                if data is not None:
                    _judge_css(data.decode("utf-8", "replace"), found)


def _ooxml_name(low: str, found: Found) -> None:
    base = low.rsplit("/", 1)[-1]
    slashed = "/" + low
    if base in ("vbaproject.bin", "vbadata.xml", "vbaprojectsignature.bin",
                "vbaprojectsignatureagile.bin"):
        found.add("office-macros")
    if "/macrosheets/" in slashed or "/intlmacrosheets/" in slashed:
        found.add("office-macros")
    if "/activex/" in slashed:
        found.add("office-activex")
    if "/embeddings/" in slashed:
        if base.startswith("oleobject") or base.endswith(".bin"):
            found.add("office-ole-object")
        if _ext(base) in _MACRO_EXT:
            found.add("office-macros")


def _ooxml_part(zf, info, low, found, depth) -> None:
    if low == "[content_types].xml":
        _xml_member(zf, info, None, found, depth, _ContentTypes(found, depth))
    elif low.endswith(".rels"):
        _xml_member(zf, info, None, found, depth, _Relationships(found, depth))
    elif not low.endswith(".xml"):
        return
    elif low.startswith("word/"):
        _xml_member(zf, info, _WORD_HINT, found, depth)
    elif low.startswith("xl/"):
        _xml_member(zf, info, _SHEET_HINT, found, depth)
    elif low.startswith("ppt/"):
        _xml_member(zf, info, _SLIDE_HINT, found, depth)


class _ContentTypes(_Xml):
    def start(self, name, attributes):
        declared = (attributes.get("ContentType") or "").lower()
        if "macroenabled" in declared or "vbaproject" in declared \
                or "macrosheet" in declared:
            self.found.add("office-macros")
        elif "activex" in declared:
            self.found.add("office-activex")


class _Relationships(_Xml):
    def start(self, name, attributes):
        if name.rpartition(" ")[2] != "Relationship":
            return
        kind = (attributes.get("Type") or "").lower()
        mode = (attributes.get("TargetMode") or "").lower()
        if mode == "external" and not kind.endswith("/hyperlink"):
            self.found.add("office-external-template")
        if kind.endswith("/oleobject"):
            self.found.add("office-ole-object")
        elif kind.endswith(("/control", "/activexcontrol")):
            self.found.add("office-activex")
        elif kind.endswith(("/vbaproject", "/vbaprojectsignature")):
            self.found.add("office-macros")


def _xml_member(zf, info, hint, found, depth, handlers=None) -> None:
    """A part read with expat -- when ``hint`` finds something in its
    bytes, or always (``hint`` None). Streamed, never held whole."""
    try:
        if hint is not None and not _hinted(zf, info, hint, found):
            return
        handlers = handlers or _Xml(found, depth)
        parser = _expat(handlers)
        with zf.open(info) as fh:
            while True:
                deadline.check("active content in an office document")
                piece = fh.read(_PIECE)
                if not piece:
                    break
                found.spend(len(piece))
                parser.Parse(piece, False)
        parser.Parse(b"", True)
    except (expat.ExpatError, ValueError, LookupError):
        found.cut("a part of it is not well-formed XML")
    except (RuntimeError, zipfile.BadZipFile, zlib.error, OSError, EOFError,
            NotImplementedError, lzma.LZMAError):
        found.cut("a member of it cannot be read")


def _hinted(zf, info, hint, found) -> bool:
    tail = b""
    with zf.open(info) as fh:
        first = True
        while True:
            deadline.check("active content in an office document")
            piece = fh.read(_PIECE)
            if not piece:
                return False
            found.spend(len(piece))
            if first and (piece[:2] in (codecs.BOM_UTF16_LE,
                                        codecs.BOM_UTF16_BE)
                          or b"\x00" in piece[:4]):
                return True             # UTF-16: read it whole
            first = False
            window = tail + piece
            if hint.search(window):
                return True
            tail = window[-16:]


# ---- tar and compressed files --------------------------------------------

class _Budgeted:
    """A decompressing reader charged to the scan's budget."""

    def __init__(self, raw, found: Found) -> None:
        self.raw, self.found = raw, found

    def read(self, size=-1):
        deadline.check("active content in an archive")
        data = self.raw.read(size if size is not None and size >= 0
                             else _PIECE)
        self.found.spend(len(data))
        return data


def _decompressor(kind: str, fh):
    if kind == "gzip":
        return gzip.GzipFile(fileobj=fh, mode="rb")
    if kind == "bz2":
        return bz2.BZ2File(fh)
    if kind == "xz":
        return lzma.LZMAFile(fh)
    if kind == "zstd":
        if _zstd is None:
            raise _Cut("a zstd stream in it cannot be read on this Python")
        return _zstd.ZstdFile(fh)
    return fh


_ARCHIVE_ERRORS = (tarfile.TarError, EOFError, OSError, zlib.error,
                   lzma.LZMAError, ValueError, struct.error)


def _compressed(fh, kind: str, name: str, found: Found, depth: int) -> None:
    """A tar, compressed or not, or one compressed file."""
    if depth > MAX_DEPTH:
        found.cut("it nests deeper than the scan follows")
        return
    try:
        stream = _Budgeted(_decompressor(kind, fh), found)
        try:
            tar = tarfile.open(fileobj=stream, mode="r|")
        except tarfile.ReadError:
            tar = None
        if tar is not None:
            _tar(tar, found, depth)
            return
        if kind == "tar":
            found.cut("it is a damaged tar archive")
            return
        fh.seek(0)
        inner = _Budgeted(_decompressor(kind, fh), found)
        head = inner.read(HEAD)
        inner_name = re.sub(r"\.(?:gz|tgz|bz2|xz|zst)$", "", name.lower())
        _judge_member(inner_name, head, found)
        nested = _container(head)
        if nested in ("7z", "rar"):
            found.cut(f"a {nested} archive in it is not looked into")
        elif nested and nested != "tar":
            data = head + inner.read(found.left + 1)
            if nested == "zip":
                _zip_bytes(data, found, depth + 1)
            else:
                _compressed(io.BytesIO(data), nested, inner_name, found,
                            depth + 1)
    except _ARCHIVE_ERRORS:
        found.cut("it is a damaged archive")


def _tar(tar, found: Found, depth: int) -> None:
    for member in tar:
        deadline.check("active content in an archive")
        found.members += 1
        if found.members > MAX_MEMBERS:
            raise _Cut("it holds more members than sift lists")
        if not member.isfile():
            continue
        extracted = tar.extractfile(member)
        head = extracted.read(HEAD) if extracted is not None else b""
        low = member.name.lower()
        _judge_member(low, head, found)
        kind = _container(head)
        if kind in ("7z", "rar"):
            found.cut(f"a {kind} archive in it is not looked into")
        elif kind:
            found.spend(member.size)
            data = head + (extracted.read() if extracted is not None else b"")
            if kind == "zip":
                _zip_bytes(data, found, depth + 1)
            else:
                _compressed(io.BytesIO(data), kind, low, found, depth + 1)


def _archive(path: Path, max_bytes: int, found: Found) -> None:
    with path.open("rb") as fh:
        head = fh.read(HEAD)
        kind = _container(head)
        if kind == "zip":
            _zip_path(path, max_bytes, found)
        elif kind == "rar":
            found.cut("rar archives cannot be listed with the standard "
                      "library")
        elif kind == "7z":
            # sift lists a 7z (sevenzip); this scan does not look into it.
            found.cut("the active-content scan does not look into a 7z "
                      "archive")
        elif kind:
            fh.seek(0)
            _compressed(fh, kind, path.name, found, 0)
        else:
            # An ISO image, ar, cpio, cab: listed by fallback, and not
            # looked into here.
            found.cut("the active-content scan does not look into this "
                      "archive format")


# ---- RTF -------------------------------------------------------------------

_RTF_OBJECT = re.compile(
    rb"\\(object|objdata|objupdate|objemb|objlink|objautlink|objsub|objpub|"
    rb"objicemb|objhtml|objocx)(?![A-Za-z])", re.IGNORECASE)
_RTF_TEMPLATE = re.compile(rb"\\template(?![A-Za-z])", re.IGNORECASE)
_RTF_FIELD = re.compile(rb"\\fldinst(?![A-Za-z])", re.IGNORECASE)
_RTF_HEX = re.compile(rb"\\'([0-9A-Fa-f]{2})")
_RTF_CONTROL = re.compile(rb"\\\*|\\[A-Za-z]+-?[0-9]* ?|[{}]")


def _rtf_bytes(data: bytes, found: Found) -> None:
    for match in _RTF_OBJECT.finditer(data):
        found.add("rtf-object")
        if match.group(1).lower() == b"objocx":
            found.add("office-activex")
    if _RTF_TEMPLATE.search(data):
        found.add("office-external-template")
    for match in _RTF_FIELD.finditer(data):
        deadline.check("active content in RTF")
        code = data[match.end():match.end() + 512]
        code = _RTF_HEX.sub(lambda m: bytes([int(m.group(1), 16)]), code)
        code = _RTF_CONTROL.sub(b"", code)
        if _DDE_FIELD.match(code.decode("latin-1")):
            found.add("office-dde")


def _rtf(path: Path, max_bytes: int, found: Found) -> None:
    _rtf_bytes(_read(path, max_bytes, found), found)


# ---- compound files: .doc, .xls, .ppt, .msg ----------------------------

_FREE, _END = 0xFFFFFFFF, 0xFFFFFFFE


class _Damaged(Exception):
    pass


class _Cfb:
    """A compound file's directory and streams, read without trusting a
    single pointer in it: every chain is bounded by the file's sectors."""

    def __init__(self, data: bytes, found: Found) -> None:
        self.data, self.found = data, found
        if len(data) < 512 or data[:8] != _OLE:
            raise _Damaged
        shift, mini_shift = struct.unpack_from("<HH", data, 30)
        if shift not in (9, 12) or mini_shift != 6:
            raise _Damaged
        self.size = 1 << shift
        (_, n_fat, first_dir, _, self.cutoff, first_minifat, _,
         first_difat, n_difat) = struct.unpack_from("<9I", data, 40)
        self.sectors = max(0, (len(data) - self.size) // self.size + (
            1 if (len(data) - self.size) % self.size else 0))
        difat = list(struct.unpack_from("<109I", data, 76))
        seen, at = set(), first_difat
        while at < self.sectors and at not in seen and len(seen) < n_difat + 1:
            seen.add(at)
            block = self._sector(at)
            count = self.size // 4 - 1
            difat += struct.unpack_from("<%dI" % count, block, 0)
            at = struct.unpack_from("<I", block, count * 4)[0]
        fat_sectors = [s for s in difat if s < self.sectors][:max(n_fat, 0)]
        if len(fat_sectors) < n_fat:
            raise _Damaged
        self.fat = []
        for s in fat_sectors:
            block = self._sector(s)
            self.fat += struct.unpack_from("<%dI" % (self.size // 4), block, 0)
        entries = self._chain_bytes(first_dir, self.fat, self.size, None)
        self.entries = []
        for off in range(0, len(entries) - 127, 128):
            raw = entries[off:off + 128]
            length, kind = struct.unpack_from("<HB", raw, 64)
            left, right, child = struct.unpack_from("<III", raw, 68)
            start, size = struct.unpack_from("<IQ", raw, 116)
            name = raw[:max(0, min(length, 64) - 2)].decode("utf-16-le",
                                                           "replace")
            self.entries.append((name, kind, left, right, child, start, size))
        if not self.entries or self.entries[0][1] != 5:
            raise _Damaged
        self.minifat = None
        self.first_minifat = first_minifat
        self.parents = self._tree()

    def _sector(self, index: int) -> bytes:
        at = (index + 1) * self.size
        block = self.data[at:at + self.size]
        if len(block) < self.size:
            block += bytes(self.size - len(block))
        return block

    def _chain_bytes(self, start, fat, size, limit, base=None) -> bytes:
        out, seen, at = bytearray(), set(), start
        while at != _END and at != _FREE:
            if at in seen or at >= len(fat):
                raise _Damaged
            seen.add(at)
            if base is None:
                out += self._sector(at)
            else:
                out += base[at * size:(at + 1) * size]
            if limit is not None and len(out) >= limit:
                break
            if len(seen) % 4096 == 0:
                deadline.check("active content in an office document")
            at = fat[at]
        return bytes(out)

    def _tree(self) -> dict:
        """entry index -> parent index, walking from the root."""
        parents, stack, seen = {}, [(self.entries[0][4], 0)], {0}
        while stack:
            index, parent = stack.pop()
            if index == _FREE or index >= len(self.entries) or index in seen:
                continue
            seen.add(index)
            parents[index] = parent
            _, kind, left, right, child, _, _ = self.entries[index]
            stack += [(left, parent), (right, parent)]
            if kind in (1, 5):
                stack.append((child, index))
        return parents

    def children(self, index: int):
        return [i for i, p in self.parents.items() if p == index]

    def stream(self, index: int, limit: "int | None" = None) -> bytes:
        name, kind, _, _, _, start, size = self.entries[index]
        if self.size == 512:
            size &= 0xFFFFFFFF          # version 3: the high half is junk
        size = min(size, len(self.data) * 2)
        want = size if limit is None else min(size, limit)
        self.found.spend(want)
        if size < self.cutoff:
            if self.minifat is None:
                raw = self._chain_bytes(self.first_minifat, self.fat,
                                        self.size, None) \
                    if self.first_minifat < self.sectors else b""
                self.minifat = list(struct.unpack_from(
                    "<%dI" % (len(raw) // 4), raw, 0))
                root = self.entries[0]
                self.ministream = self._chain_bytes(root[5], self.fat,
                                                    self.size, root[6])
            data = self._chain_bytes(start, self.minifat, 64, want,
                                     self.ministream)
        else:
            data = self._chain_bytes(start, self.fat, self.size, want)
        return data[:want]


_ATTACH = "__attach_version1.0_#"
_MSG_NAME = ("__substg1.0_3707001f", "__substg1.0_3704001f",
             "__substg1.0_3707001e", "__substg1.0_3704001e")
_MSG_DATA = "__substg1.0_37010102"
_MSG_MIME = ("__substg1.0_370e001f", "__substg1.0_370e001e")
_MSG_HTML = ("__substg1.0_10130102", "__substg1.0_1013001f",
             "__substg1.0_1013001e")
_WORD_DDE = re.compile(rb"\x13[ \t\"]*DDE(?:AUTO)?(?![A-Za-z0-9])", re.I)
_WORD_DDE_WIDE = re.compile(
    rb"\x13\x00(?:[ \t\"]\x00)*[Dd]\x00[Dd]\x00[Ee]\x00(?![A-Za-z0-9]\x00)"
    rb"|\x13\x00(?:[ \t\"]\x00)*[Dd]\x00[Dd]\x00[Ee]\x00[Aa]\x00[Uu]\x00"
    rb"[Tt]\x00[Oo]\x00")
#: VBAInfoContainer and its VBAInfoAtom ([MS-PPT] 2.4.10), headers whole.
_PPT_VBA = re.compile(rb"\x0f\x00\xff\x03\x14\x00\x00\x00\x02\x00\x00\x04"
                      rb"\x0c\x00\x00\x00")
#: ExOleObjStg ([MS-PPT] 2.10.34): compressed, a size and a zlib header;
#: stored, a compound file.
_PPT_OLE = re.compile(rb"\x10\x00\x11\x10.{4}.{4}\x78[\x01\x5e\x9c\xda]"
                      rb"|\x00\x00\x11\x10.{4}\xd0\xcf\x11\xe0", re.DOTALL)


def _cfb_bytes(data: bytes, found: Found, depth: int = 0) -> None:
    try:
        cfb = _Cfb(data, found)
    except (_Damaged, struct.error):
        found.cut("it is a damaged compound file")
        return
    try:
        _judge_cfb(cfb, found, depth)
    except (_Damaged, struct.error):
        found.cut("it is a damaged compound file")


def _judge_cfb(cfb: _Cfb, found: Found, depth: int) -> None:
    for index in cfb.parents:
        deadline.check("active content in an office document")
        name, kind = cfb.entries[index][0], cfb.entries[index][1]
        low = name.lower()
        if kind == 1 and low in ("vba", "_vba_project_cur", "macros") \
                or kind == 2 and low == "_vba_project":
            found.add("office-macros")
        elif low == "\x01ole10native":
            found.add("office-ole-object")
        elif kind == 1 and low == "objectpool" and cfb.children(index):
            found.add("office-ole-object")
        elif kind == 1 and re.fullmatch(r"mbd[0-9a-f]{8}", low):
            found.add("office-ole-object")
        elif low in ("\x03ocxname", "ocxdata", "ctls"):
            found.add("office-activex")
        elif kind == 2 and low in ("workbook", "book"):
            _xls_globals(cfb.stream(index), found)
        elif kind == 2 and low == "worddocument":
            text = cfb.stream(index)
            if _WORD_DDE.search(text) or _WORD_DDE_WIDE.search(text):
                found.add("office-dde")
        elif kind == 2 and low == "powerpoint document":
            text = cfb.stream(index)
            if _PPT_VBA.search(text):
                found.add("office-macros")
            if _PPT_OLE.search(text):
                found.add("office-ole-object")
        elif kind == 2 and low in _MSG_HTML:
            raw = cfb.stream(index)
            if low.endswith("001f"):
                raw = raw.decode("utf-16-le", "replace").encode("utf-8")
            _html_bytes(raw, found, depth + 1)
        elif kind == 1 and name.startswith(_ATTACH):
            _msg_attachment(cfb, index, found, depth)


def _xls_globals(stream: bytes, found: Found) -> None:
    """BIFF records up to the end of the workbook globals: a sheet of type
    1 is an Excel 4.0 macro sheet, of type 6 a VBA module."""
    at = 0
    while at + 4 <= len(stream):
        kind, length = struct.unpack_from("<HH", stream, at)
        if kind == 0x000A:
            return
        if kind == 0x0085 and length >= 6 and at + 4 + length <= len(stream):
            if stream[at + 9] in (0x01, 0x06):
                found.add("office-macros")
        at += 4 + length


def _msg_attachment(cfb: _Cfb, index: int, found: Found, depth: int) -> None:
    name, mime, data = "", "", None
    for child in cfb.children(index):
        child_name = cfb.entries[child][0].lower()
        if cfb.entries[child][1] != 2:
            continue
        if child_name in _MSG_NAME and not name:
            raw = cfb.stream(child, 1024)
            name = raw.decode("utf-16-le" if child_name.endswith("001f")
                              else "latin-1", "replace").rstrip("\x00")
        elif child_name in _MSG_MIME:
            raw = cfb.stream(child, 256)
            mime = raw.decode("utf-16-le" if child_name.endswith("001f")
                              else "latin-1", "replace").rstrip(
                "\x00").strip().lower()
        elif child_name == _MSG_DATA:
            data = cfb.stream(child)
    if data is not None:
        _judge_attachment(name, mime, data, found, depth)
    elif name:
        _judge_attachment(name, mime, b"", found, depth)


def _cfb(path: Path, max_bytes: int, found: Found) -> None:
    _cfb_bytes(_read(path, max_bytes, found), found)


# ---- mail ------------------------------------------------------------------

_MBOX_SEP = re.compile(
    rb"(?:\A|\r?\n)(From \S+[^\r\n]*\r?\n)(?=[A-Za-z][A-Za-z0-9-]*:)")
_ENCRYPTED_TYPES = frozenset({"application/pkcs7-mime",
                              "application/x-pkcs7-mime",
                              "application/pgp-encrypted",
                              "application/ms-tnef",
                              "application/vnd.ms-tnef"})
_BROKEN_STRUCTURE = ("StartBoundaryNotFoundDefect",
                     "MultipartInvariantViolationDefect")


def _mail_bytes(data: bytes, found: Found, depth: int = 0) -> None:
    if data.startswith(b"From "):
        starts = [m.start(1) for m in _MBOX_SEP.finditer(data)] or [0]
        pieces = [data[a:b] for a, b in zip(starts, starts[1:] + [len(data)])]
        messages = [piece.partition(b"\n")[2] for piece in pieces]
    else:
        messages = [data]
    for raw in messages:
        deadline.check("active content in a mail")
        # compat32: the default policy parses every header it is asked
        # for into an object, which cost a mailbox of 300 messages half a
        # second; the names it would decode are decoded below.
        _message(email.message_from_bytes(raw, policy=email.policy.compat32),
                 found, depth)


def _message(msg, found: Found, depth: int) -> None:
    for part in msg.walk():
        deadline.check("active content in a mail")
        if any(type(d).__name__ in _BROKEN_STRUCTURE
               for d in getattr(part, "defects", ())):
            found.cut("its MIME structure cannot be read as declared")
        content_type = part.get_content_type()
        if part.is_multipart():
            if content_type == "multipart/encrypted":
                found.cut("it holds an encrypted part")
            continue
        if content_type in _ENCRYPTED_TYPES:
            found.cut("it holds an encrypted or packed part")
            continue
        name = _filename(part)
        try:
            payload = part.get_payload(decode=True) or b""
        except Exception:  # noqa: BLE001
            payload = b""
        if isinstance(payload, str):
            payload = payload.encode("utf-8", "replace")
        found.spend(len(payload))
        ext = _ext(name)
        if content_type in ("text/html", "application/xhtml+xml",
                            "image/svg+xml") or ext in _HTML_EXT:
            try:
                charset = part.get_content_charset()
            except Exception:  # noqa: BLE001
                charset = None
            _html_bytes(payload, found, depth + 1, charset)
        attached = name or part.get_content_disposition() == "attachment"
        if attached or not content_type.startswith(("text/", "multipart/",
                                                    "message/")):
            if content_type in ("text/html", "application/xhtml+xml",
                                "image/svg+xml") or ext in _HTML_EXT:
                _judge_attachment(name, content_type, b"", found, depth)
            else:
                _judge_attachment(name, content_type, payload, found, depth)


def _filename(part) -> str:
    """The attachment's name, RFC 2231 and RFC 2047 decoded: a name sent as
    =?utf-8?b?...?= must not hide its ".js"."""
    try:
        raw = part.get_filename() or ""
        return str(email.header.make_header(email.header.decode_header(raw)))
    except Exception:  # noqa: BLE001 - a header no parser agrees on
        try:
            return str(part.get_filename() or "")
        except Exception:  # noqa: BLE001
            return ""


def _mail(path: Path, max_bytes: int, found: Found) -> None:
    _mail_bytes(_read(path, max_bytes, found), found)


#: kind -> the name of its scanner, looked up when a file is scanned.
_SCANNERS = {
    "pdf": "_pdf", "html": "_html", "xml": "_xml", "email": "_mail",
    "rtf": "_rtf", "office-legacy": "_cfb", "archive": "_archive",
    "docx": "_zip_path", "xlsx": "_zip_path", "pptx": "_zip_path",
    "odt": "_zip_path", "ods": "_zip_path", "odp": "_zip_path",
    "epub": "_zip_path",
}
assert set(_SCANNERS) == KINDS
