"""Extractor for HTML — title, meta, and readable text (stdlib parser)."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ..detect import Detection
from .base import Extracted, clip, decode_text

MAX_CHARS = 150_000
_SKIP = {"script", "style", "head", "noscript", "template", "svg"}
_BLOCK = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4",
          "h5", "h6", "section", "article", "header", "footer", "table", "ul",
          "ol", "blockquote", "dt", "dd", "caption", "figcaption", "pre"}
# Without td/th here, adjacent cells were concatenated: a sales table reached
# the model as "RegionSales / North100 / South250", welding every label to its
# number.


#: Elements worth counting for a summary: they say what kind of page it is.
_COUNTED = ("p", "h1", "h2", "h3", "table", "ul", "ol", "li", "form",
            "input", "iframe", "script", "video", "audio")
_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str = ""
        self.description: str = ""
        self.links = 0
        self.images = 0
        # A page's shape. Reporting only "title, 64 characters, 0 links, 0
        # images" let a model conclude the file "lacks standard HTML
        # structure" — about a document with html, head, title, body, h1 and
        # p. It had no fact to the contrary because none was offered.
        self.headings: list[str] = []
        self.tags: dict = {}
        self._heading_level = 0
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _COUNTED:
            self.tags[tag] = self.tags.get(tag, 0) + 1
        if tag in _HEADINGS:
            self._heading_level = int(tag[1])
            self.headings.append("")
        if tag in _SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            self.links += 1
        if tag == "img":
            self.images += 1
        if tag == "meta":
            d = dict(attrs)
            if d.get("name", "").lower() == "description":
                self.description = d.get("content", "") or self.description
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _HEADINGS:
            self._heading_level = 0
        if tag in _SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:  # title lives inside <head>, which we otherwise skip
            self.title += data.strip() + " "
            return
        if self._skip_depth:
            return
        if data.strip():
            if self._heading_level and self.headings:
                self.headings[-1] += data.strip() + " "
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _Extractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML shouldn't crash us
        pass
    text = "".join(parser.parts)
    return _collapse(text)


_META_CHARSET = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.IGNORECASE)


_HTML_COMMENT = re.compile(rb"<!--.*?-->", re.DOTALL)
#: Declarations that are usually a copied boilerplate line rather than a fact.
_LATIN_ALIASES = {"iso88591", "latin1", "windows1252", "cp1252", "iso8859-1"}


def _declared_charset(data: bytes) -> str:
    """The encoding the document names for itself, if any.

    Ignoring it left every legacy non-Latin page as mojibake labelled cp1252 —
    a cp1251 Russian page came out as "Ïðèâåò ìèð". The declaration is in the
    first kilobytes by spec, and ASCII there regardless of the encoding.

    Two things outrank it, in HTML5's own order. A byte-order mark is decisive,
    and a declaration inside a comment is not a declaration — a commented-out
    windows-1251 line ahead of the real utf-8 one turned "Café" into "CafГ©".
    """
    if data[:3] == b"\xef\xbb\xbf":
        return ""                       # BOM wins over <meta>, per the spec
    head = _HTML_COMMENT.sub(b"", data[:4096])
    match = _META_CHARSET.search(head)
    if not match:
        return ""
    name = match.group(1).decode("ascii", "replace").lower()
    return "" if name in ("utf-8", "utf8", "us-ascii", "ascii") else name


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    with path.open("rb") as fh:
        data = fh.read(max_bytes)
    cut_short = det.size > len(data)
    declared = _declared_charset(data)
    html = encoding = None
    if declared:
        try:
            html, encoding = data.decode(declared), declared
        except (UnicodeDecodeError, LookupError):
            html = None
        else:
            # A page whose bytes are valid UTF-8 almost never really is
            # iso-8859-1, whatever its <meta> says — that declaration is the
            # most-copied line on the web. Believing it turned "Café — naïve"
            # into "CafÃ© â\x80\x94 naÃ¯ve" on a page that decoded cleanly.
            if declared.replace("-", "").lower() in _LATIN_ALIASES:
                try:
                    html, encoding = data.decode("utf-8"), "utf-8"
                except UnicodeDecodeError:
                    pass
    if html is None:
        html, encoding = decode_text(data, cut_short)

    parser = _Extractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001
        pass
    text = _collapse("".join(parser.parts))

    meta = {
        "encoding": encoding,
        "title": parser.title.strip(),
        "description": parser.description.strip(),
        "links": parser.links,
        "images": parser.images,
        "text_length": len(text),
        "headings": [h.strip()[:80] for h in parser.headings[:12] if h.strip()],
        "elements": {k: v for k, v in sorted(parser.tags.items()) if v},
    }
    # The title and the description are already in meta, and meta is already
    # in the prompt. Writing them again at the top of the text — the title as
    # a Markdown "# " line — bought nothing and cost accuracy: a real local
    # model twice reported "Markdown-style heading syntax within an HTML
    # document" as a property of a press release that contains no Markdown,
    # and went on saying it after being told in the prompt that the line was
    # sift's. The line that is not there cannot be misread.
    rendered = text

    # Comments never reach the model — the parser drops them — but a comment
    # addressed to a reading assistant is a fact about the file, and hiding it
    # there is more suspicious than writing it in the open. Carried separately
    # so it can be reported without being sent.
    hidden = " ".join(_comments(html.encode("utf-8", "replace")))
    body, truncated = clip(rendered, MAX_CHARS)
    return Extracted(det.kind, det.mime, text=body, metadata=meta,
                     # Still True with the heading gone. The flag controls a
                     # note with two branches, and dropping the heading only
                     # settled the first one ("these lines are sift's"). The
                     # second is the one a page needs: that the markup was
                     # taken out here and its absence says nothing about the
                     # file. Setting this False took that away too, and a
                     # model went straight back to reporting that a press
                     # release "lacks standard structural tags (<html>,
                     # <head>, <body>), resembling plain text" — of a file
                     # that has all three.
                     rendered=True, truncated=truncated,
                     hidden_text=hidden[:20_000])


#: How many comments are read, and how far a single unterminated one may
#: run. A page has a handful; these are bounds on work.
_MAX_COMMENTS = 4096
_MAX_COMMENT = 64 * 1024


def _comments(data: bytes):
    """Comment bodies, found by scanning rather than by backtracking.

    ``<!--.*?-->`` with DOTALL scans to the end of the buffer for every
    ``<!--`` with no ``-->`` after it, which is quadratic in a token that
    costs four bytes: 250,000 of them in a 1 MB page ran past two minutes.
    find() does the same job in one pass and cannot backtrack.
    """
    at, found = 0, 0
    while found < _MAX_COMMENTS:
        start = data.find(b"<!--", at)
        if start < 0:
            return
        end = data.find(b"-->", start + 4)
        if end < 0 or end - start > _MAX_COMMENT:
            return                       # unterminated, or not a comment
        yield data[start + 4:end].decode("utf-8", "replace")
        at, found = end + 3, found + 1


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
