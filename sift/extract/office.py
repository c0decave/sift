"""Extractors for OOXML office files — docx, xlsx, pptx, odt — pure stdlib.

OOXML files are ZIP archives of XML. We read them directly with ``zipfile`` and
``ElementTree`` so no third-party library is required. python-docx / openpyxl,
when installed, are not needed for text extraction.
"""
from __future__ import annotations

import collections
import itertools
import posixpath
import re
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

from ..detect import Detection
from ..zipguard import ZipLimitError, check as check_zip
from .tabular import _rows_look_headed
from .. import deadline
from .base import (CutRow, Extracted, TablePage, clip, fill_page,
                   human_size, one_line)

MAX_CHARS = 200_000
#: What one zip member may inflate to before it is read as truncated. This
#: is a *byte* budget for markup, not a character budget for text: XLSX
#: spends about 40 bytes of XML per cell, so a five-thousand-row sheet is
#: 2.4 MB of member for 200 rows of text the reader keeps. Refusing at ten
#: times MAX_CHARS therefore threw away ordinary spreadsheets, Word documents
#: containing one photograph, and epubs with a cover — a 136 KB file
#: summarized to nothing at all.
#: Sixteen megabytes of markup is about 90,000 rows of an eight-column
#: sheet — well past what the reader keeps (SHEET_ROW_CAP is 200) and
#: well past any document someone summarizes. Parsing costs roughly six
#: times the raw bytes, so this is the figure that bounds the peak.
_MAX_MEMBER = 16 * 1024 * 1024


def _read_member(zf, name: str) -> bytes:
    """A member's bytes, truncated rather than refused.

    The cap belongs on the read, not on a survey of the container: a member
    no extractor opens — a photograph, a font, a thumbnail — cannot cost
    anything, and one that is too large should lose its tail, not take the
    document with it. Reading one byte past the cap is how truncation is
    detected without inflating the rest.
    """
    with zf.open(name) as fh:
        data = fh.read(_MAX_MEMBER + 1)
    return data[:_MAX_MEMBER]


def _is_truncated(data: bytes) -> bool:
    return _fills_cap(len(data))


def _fills_cap(size: int) -> bool:
    """Whether ``size`` bytes of a member are as many as _read_member keeps:
    what it read may be cut there (_is_truncated)."""
    return size >= _MAX_MEMBER
#: Rows collected per worksheet. The sheet's real row count is reported
#: separately — the cap must never be mistaken for the size of the data.
SHEET_ROW_CAP = 200
#: Columns materialized per row. The true width is still reported.
MAX_RENDER_COLUMNS = 512
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CORE = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"
DC = "{http://purl.org/dc/elements/1.1/}"


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    try:
        check_zip(path)
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            props = _core_props(zf)
            if det.kind == "docx" or "word/document.xml" in names:
                return _docx(zf, det, props)
            if det.kind == "xlsx" or any(n.startswith("xl/") for n in names):
                return _xlsx(zf, det, props)
            if det.kind == "pptx" or any(n.startswith("ppt/slides/") for n in names):
                return _pptx(zf, det, props)
            if det.kind == "epub" or any(n.endswith(".opf") for n in names):
                return _epub(zf, det, props)
            if det.kind in ("odt", "ods", "odp") or "content.xml" in names:
                return _odf(zf, det, props)
    except ZipLimitError as exc:
        return Extracted(
            det.kind, det.mime, text="",
            metadata={"archive_member_count": exc.info.entries,
                      "archive_limit": str(exc)},
            notes=[_not_opened(exc)])
    except (zipfile.BadZipFile, OSError, ET.ParseError) as exc:
        return Extracted(det.kind, det.mime, text="",
                         metadata={"error": str(exc)},
                         notes=[f"could not open OOXML container: {exc}"])
    return Extracted(det.kind, det.mime, text="", notes=["unrecognized office container"])


def read_sheet_page(path: Path, det: Detection, sheet, offset: int,
                    limit: int, max_chars: int) -> TablePage:
    """One page of one sheet of an xlsx or ods, under the extractor's limits.

    ``sheet`` is a name from ``TablePage.sheets``, an index into it, or None
    for the first. The zip is checked the way extract() checks it and each
    member is read under the same cap; the row ceilings are gone, because
    they bound a rendering and a page has bounds of its own.

    An xlsx sheet is read up to the row after the page (_XlsxSheet); an ods
    is read to its end for its list of sheets, the rows of the one asked
    for handed to the page as they come (_OdsTables). Either way the page
    holds what 0.1.55's page held; see read_table for what a page that
    stops before its table's end leaves out.
    """
    page = TablePage(kind=det.kind, offset=offset)
    try:
        check_zip(path)
        with zipfile.ZipFile(path) as zf:
            if det.kind == "ods":
                _ods_page(zf, page, sheet, offset, limit, max_chars)
            else:
                _xlsx_page(zf, page, sheet, offset, limit, max_chars)
    except ZipLimitError as exc:
        page.error = _not_opened(exc)
    except _UNREADABLE as exc:
        page.error = unreadable(exc)
    return page


#: What read_sheet_page words as "could not read the spreadsheet".
_UNREADABLE = (zipfile.BadZipFile, OSError, KeyError, ET.ParseError)

#: What a page or a stream says of a file with no table in it.
NO_TABLE = "the file holds no table"


def unreadable(exc: Exception) -> str:
    """What a page or a stream says of a spreadsheet that raised ``exc``,
    one of _UNREADABLE."""
    return f"could not read the spreadsheet: {exc}"


def _not_opened(exc: Exception) -> str:
    """What a page or a stream says of a zip that zipguard refused."""
    return f"office container was not opened: {exc}"


def _sheet_cut_note(sink) -> None:
    """A sheet cut at _MAX_MEMBER, as 0.1.55 said it (_xlsx_sheet)."""
    sink.truncated = True
    sink.note(f"sheet larger than {human_size(_MAX_MEMBER)}; the rows after "
              f"the first part are not read")


def _no_rows(page) -> None:
    """``page`` as it was before its rows were read: a failure met while
    reading them costs the page every row, as it did in 0.1.55, which read
    the whole member before the first."""
    page.header, page.rows, page.total, page.has_more = [], [], None, False
    page.columns_truncated = page.cells_clipped = False


def _xlsx_page(zf, page: TablePage, sheet, offset: int, limit: int,
               max_chars: int) -> None:
    pairs = _sheets_in_order(zf)
    page.sheets = [one_line(name) for name, _ in pairs]
    index = _choose(page.sheets, sheet)
    if index is None:
        page.total = 0
        page.note(NO_TABLE)
        return
    page.sheet = page.sheets[index]
    source = _XlsxSheet(zf, pairs[index][1], page)
    try:
        _page_of_sheet(page, source.rows(), offset, limit, max_chars,
                       convert=source.line, count_rest=source.whole)
        if not page.has_more:
            source.finish()
    except _UNREADABLE:
        # 0.1.55 said the sheet was cut only once its cut tree parsed, so a
        # page with an error never said it.
        _no_rows(page)
        page.notes, page.truncated = source.notes_before, False
        raise
    finally:
        source.close()


def _ods_page(zf, page: TablePage, sheet, offset: int, limit: int,
              max_chars: int) -> None:
    tables = _OdsTables(zf, sheet, page)
    try:
        # The content is read to its end for the list of sheets: counting
        # the rows of the table asked for on the way costs next to nothing
        # (_OdsTables.count_rest).
        _page_of_sheet(page, tables.rows(), offset, limit, max_chars,
                       count_rest=tables.count_rest)
        index = tables.finish()
    except _UNREADABLE:
        # 0.1.55 parsed the whole content before it listed a sheet.
        _no_rows(page)
        page.notes, page.truncated = [], False
        raise
    finally:
        tables.close()
    page.sheets = tables.sheets
    if index is None:
        page.note(NO_TABLE)
        return
    page.sheet = page.sheets[index]


def open_sheet_rows(zf, det: Detection, sheet, sink):
    """(header, rows, close) of one sheet of an xlsx or ods, read as the rows
    are asked for (iter_table): the sheets, the sheet and the header set on
    ``sink`` as read_sheet_page sets them on a page, the notes and the
    truncation as they are met. ``rows`` hands out every data row once and
    reads, at its end, what a page that reaches the end reads (an xlsx's
    shared strings to their end, an ods's content to its end, which lists
    its sheets); ``close`` lets the members go. Raises what read_sheet_page
    raises or catches."""
    if det.kind == "ods":
        source = _OdsTables(zf, sheet, sink)
        convert = None
    else:
        pairs = _sheets_in_order(zf)
        sink.sheets = [one_line(name) for name, _ in pairs]
        index = _choose(sink.sheets, sheet)
        if index is None:
            sink.note(NO_TABLE)
            return None, iter(()), lambda: None
        sink.sheet = sink.sheets[index]
        source = _XlsxSheet(zf, pairs[index][1], sink)
        convert = source.line
    try:
        header, head, lines = _head(source.rows(), convert)
        if det.kind == "ods":
            if source.chosen is None:
                _finish_ods(source, sink)
                source.close()
                return None, iter(()), lambda: None
            sink.sheet = source.name
    except _UNREADABLE:
        source.close()
        # As read_sheet_page leaves a page (_ods_page, _xlsx_page).
        if det.kind == "ods":
            sink.notes, sink.truncated = [], False
        else:
            sink.notes, sink.truncated = source.notes_before, False
        raise
    except BaseException:
        source.close()
        raise

    def rows():
        yield from head
        for line in lines:
            yield line if convert is None else convert(line)
        if det.kind == "ods":
            _finish_ods(source, sink)
        else:
            source.finish()
    return header or None, rows(), source.close


def open_sheet_stream(path: Path, det: Detection, sheet, sink):
    """open_sheet_rows of the xlsx or ods at ``path``, the zip checked and
    opened as read_sheet_page opens it. A failure to open it, or to read
    the rows the header is decided on, is set on ``sink`` as read_sheet_page
    sets it on a page, and there are no rows; ``close`` closes the zip."""
    nothing = (None, iter(()), lambda: None)
    try:
        check_zip(path)
        zf = zipfile.ZipFile(path)
    except ZipLimitError as exc:
        sink.error = _not_opened(exc)
        return nothing
    except _UNREADABLE as exc:
        sink.error = unreadable(exc)
        return nothing
    try:
        header, rows, close = open_sheet_rows(zf, det, sheet, sink)
    except _UNREADABLE as exc:
        zf.close()
        sink.error = unreadable(exc)
        return nothing
    except BaseException:
        zf.close()
        raise

    def closing():
        close()
        zf.close()
    return header, rows, closing


def _finish_ods(tables, sink) -> None:
    index = tables.finish()
    sink.sheets = tables.sheets
    if index is None:
        sink.note(NO_TABLE)


class _Member:
    """A zip member's bytes as _read_member reads them -- at most
    _MAX_MEMBER, one byte more asked for, so that a member of exactly that
    size is read to its end and its CRC checked -- in pieces of _PIECE.

    ``cut`` says whether the zip declares at least _MAX_MEMBER bytes, so
    that what is read of it fills the cap and 0.1.55 would have called it
    truncated (_is_truncated; zipfile hands out no more than a member
    declares); ``whole`` whether it declares no more than _WHOLE_MEMBER,
    which is then parsed whole before its first row is handed out."""

    def __init__(self, zf, name: str):
        self.info = zf.getinfo(name)
        self._fh = zf.open(self.info)
        self.cut = _fills_cap(self.info.file_size)
        self.whole = self.info.file_size <= _WHOLE_MEMBER
        self.read = 0

    def pieces(self):
        cap = _MAX_MEMBER
        while self.read <= cap:
            data = self._fh.read(min(_PIECE, cap + 1 - self.read))
            if not data:
                break
            keep = data[:max(0, cap - self.read)]
            self.read += len(data)
            if keep:
                yield keep
        self.close()

    def truncated(self) -> bool:
        """_is_truncated of what was read (once it is all read)."""
        return _fills_cap(self.read)

    def close(self) -> None:
        self._fh.close()


#: A member whose zip entry declares at most this many bytes is read and
#: parsed whole before its first row is handed out, as 0.1.55 read every
#: member: a page of it is 0.1.55's page, damage anywhere in it included.
#: A larger one is parsed a piece at a time, as far as the page needs.
_WHOLE_MEMBER = 1 << 20
#: Inflated bytes parsed at a time.
_PIECE = 1 << 16


def _parsed(pieces):
    """The ("start", "end") events of an XMLPullParser fed ``pieces``, in
    batches, one per piece, parsed as they are asked for. A batch is read to
    its end before the next is asked for. The error messages are the
    parser's, the same as ET.fromstring's for the same bytes: a parser fed
    in pieces reports the position in the whole."""
    parser = ET.XMLPullParser(("start", "end"))
    for piece in pieces:
        deadline.check("a spreadsheet")
        parser.feed(piece)
        yield parser.read_events()
    parser.close()
    yield parser.read_events()


class _SharedStrings:
    """An xlsx's shared strings as _shared_strings reads them -- the text of
    every <si> that is a child of the root -- parsed as far as the cells
    read need them. ``get(index)`` is the string, "" past the last.

    One that declares no more than _WHOLE_MEMBER is read by 0.1.55's
    _shared_strings, at once, and so is one that declares more than
    _MAX_MEMBER: cut at the cap, it is no whole document, and 0.1.55 failed
    on it before reading a row. ``lookup`` is what _sheet_row is handed:
    the list itself then, else this."""

    def __init__(self, zf):
        self.strings = []
        self._whole = True
        self._more = None
        if "xl/sharedStrings.xml" not in zf.namelist():
            return
        size = zf.getinfo("xl/sharedStrings.xml").file_size
        if size <= _WHOLE_MEMBER or size > _MAX_MEMBER:
            self.strings = _shared_strings(zf)
            return
        self._whole = False
        self._more = _strings_read(_Member(zf, "xl/sharedStrings.xml"),
                                   self.strings)

    @property
    def lookup(self):
        # Not an attribute: one holding the reader itself was a cycle.
        return self.strings if self._whole else self

    def _step(self) -> None:
        try:
            next(self._more)
        except StopIteration:
            self._more = None
        except BaseException:
            self._more = None
            raise

    def get(self, index: int) -> str:
        while index >= len(self.strings) and self._more is not None:
            self._step()
        return self.strings[index] if index < len(self.strings) else ""

    def finish(self) -> None:
        while self._more is not None:
            self._step()

    def close(self) -> None:
        """Let the member go before its end is read."""
        more, self._more = self._more, None
        if more is not None:
            more.close()


def _strings_read(member, strings: list):
    """_SharedStrings' reading of ``member``: parse a piece at a time and,
    after each, append the strings it completed to ``strings``. A function,
    not a method: the generator holding the reader it fills made a cycle
    that held the member's file until the collector ran."""
    si, t = f"{SS}si", f"{SS}t"
    depth, root = 0, None
    try:
        for batch in _parsed(member.pieces()):
            for event, element in batch:
                if event == "start":
                    depth += 1
                    if root is None:
                        root = element
                    continue
                depth -= 1
                if depth != 1:
                    continue
                if element.tag == si:
                    only = element[0] if len(element) == 1 else None
                    if only is not None and only.tag == t and not len(only):
                        strings.append(only.text or "")  # <si><t>…</t>
                    else:
                        strings.append("".join(
                            node.text or "" for node in element.iter(t)))
                root.remove(element)
            yield
    finally:
        member.close()


def _closed_pieces(pieces, member):
    """The pieces of a sheet that fills the cap and does not parse, fed as
    0.1.55 parsed it (_closed): up to its last "</row>" and the closing
    tags after, or the empty sheet _closed makes when no row closed before
    the cap. Held back from the parser is only what follows the last
    "</row>" read so far. A member that ends before the cap after all is fed
    as it is, which is how 0.1.55 parsed it."""
    pending = bytearray()
    fed = False
    for piece in pieces:
        start = max(0, len(pending) - 5)
        pending += piece
        end = pending.rfind(b"</row>", start)
        if end >= 0:
            fed = True
            yield bytes(pending[:end + 6])
            del pending[:end + 6]
    if not member.truncated():
        if pending:
            yield bytes(pending)
    elif fed:
        yield b"</sheetData></worksheet>"
    else:
        yield b"<worksheet><sheetData></sheetData></worksheet>"


class _XlsxSheet:
    """One worksheet of an xlsx as _xlsx_sheet read it in 0.1.55, parsed as
    far as its rows are asked for.

    ``rows()`` hands out the <row> elements in the order root.iter() visits
    them; ``line(row)`` is the row as a line of cells.

    A member the zip declares no larger than _WHOLE_MEMBER is read and
    parsed as 0.1.55 did it, whole (``whole``): the member read, the shared
    strings parsed, the tree built -- cut back to its last row (_closed)
    when the member filled the cap -- before the first row. A larger one is
    read in that order too, but parsed as its rows are asked for. One the
    zip declares at least _MAX_MEMBER bytes fills the cap, and is cut where
    0.1.55 cut it (_at_the_cap): its first _MAX_MEMBER bytes are read at
    once, and what 0.1.55 made of them is known before the first row."""

    def __init__(self, zf, part: str, sink):
        member = _Member(zf, part)
        self._member = member
        self.shared = None
        try:
            if member.whole:
                raw = b"".join(member.pieces())
                self.shared = _SharedStrings(zf)
                self.notes_before = list(sink.notes)
                self._root = self._tree(raw, member, sink)
                return
            pieces = member.pieces()
            # Read before the shared strings, as 0.1.55 read them.
            raw = b"".join(pieces) if member.cut else None
            self.shared = _SharedStrings(zf)
            self.notes_before = list(sink.notes)
            if raw is not None and len(raw) <= _WHOLE_MEMBER:
                # The zip declared the cap and the member held no more than
                # a sheet parsed whole: parsed as 0.1.55 parsed every sheet,
                # on the bytes read.
                self._root = self._tree(raw, member, sink)
                return
            if raw is not None:
                pieces = _at_the_cap(raw, member, sink)
        except BaseException:
            self.close()
            raise
        self._root = None
        self._events = _parsed(pieces)

    @property
    def whole(self) -> bool:
        """Whether the sheet's tree is held whole: counting its rows after
        a page costs nothing but the count."""
        return self._root is not None

    @staticmethod
    def _tree(raw: bytes, member, sink):
        """0.1.55's parse of a whole sheet (_xlsx_sheet)."""
        try:
            return ET.fromstring(raw)
        except ET.ParseError:
            if not member.truncated():
                raise
            # Cut at the member cap, as in _xlsx: the rows before the cut
            # are real, the ones after are out of reach.
            root = ET.fromstring(_closed(raw))
            _sheet_cut_note(sink)
            return root

    def rows(self):
        if self._root is not None:
            return self._root.iter(f"{SS}row")
        return _row_elements(self._events)

    def line(self, row) -> list:
        line, width, lost = _sheet_row(row, self.shared.lookup)
        if width > MAX_RENDER_COLUMNS and lost:
            # Marked on the row, not the page: a wide row on page nine
            # must not set the flag on page one.
            line = CutRow(line)
        return line

    def finish(self) -> None:
        """What a page that reaches the sheet's end reads: the shared
        strings to their end, which 0.1.55 parsed whole for every page."""
        self.shared.finish()

    def close(self) -> None:
        self._member.close()
        if self.shared is not None:
            self.shared.close()


def _at_the_cap(raw: bytes, member, sink):
    """The pieces of ``raw``, the first _MAX_MEMBER bytes of a sheet the zip
    declares to fill the cap (_Member.cut), as 0.1.55 parsed them: whole
    where they parse -- a sheet followed by white space, a comment or a
    processing instruction up to the cap -- or where the bytes read did not
    fill the cap (_Member.truncated, 0.1.55's _is_truncated: a zip that
    declares more than the member holds is not a cut, a review), else
    cut back to their last row (_closed_pieces), which ``sink`` is told."""
    held = [raw[at:at + _PIECE] for at in range(0, len(raw), _PIECE)]
    if not member.truncated() or _parses_whole(raw):
        return iter(held)
    _sheet_cut_note(sink)
    return _closed_pieces(held, member)


#: XML's white space (the S production), which may follow the root element.
_XML_WHITE = b" \t\r\n"
#: The encoding an XML declaration names.
_DECLARED_ENCODING = re.compile(
    rb"<\?xml\s[^>]*?encoding\s*=\s*[\"']([^\"'>]{1,64})[\"']")


def _parses_whole(raw: bytes) -> bool:
    """Whether ET.fromstring(raw) parses: false at once where ``raw`` plainly
    ends mid-document (_plainly_cut), which is how nearly every sheet cut at
    the cap ends -- mid-row --; else parsed as ET.fromstring parses it (the
    same expat, the same handling of entities), keeping no tree."""
    if _plainly_cut(raw):
        return False
    parser = ET.XMLParser(target=_NoTree())
    try:
        for at in range(0, len(raw), _PIECE):
            deadline.check("a spreadsheet")
            parser.feed(raw[at:at + _PIECE])
        parser.close()
    except ET.ParseError:
        return False
    return True


def _plainly_cut(raw: bytes) -> bool:
    """Whether ``raw`` cannot be a whole document on its face: written in an
    encoding whose "<", ">" and white space are ASCII's -- no NUL at either
    end (UTF-16 and UTF-32 hold them), its first mark "<" (EBCDIC writes it
    as 0x4C), any encoding it declares one of those -- and not ending in ">"
    and white space, as every document ends: in its root's end tag, a
    comment's or a processing instruction's."""
    head = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    if b"\x00" in head[:4] or b"\x00" in raw[-64:] \
            or not head.lstrip(_XML_WHITE).startswith(b"<"):
        return False
    declared = _DECLARED_ENCODING.match(head)
    if declared is not None and not _ascii_marks(declared.group(1)):
        return False
    return not raw.rstrip(_XML_WHITE).endswith(b">")


def _ascii_marks(name: bytes) -> bool:
    """Whether the encoding ``name`` writes "<", ">" and XML's white space
    as ASCII does; False for one Python does not know."""
    try:
        marks = "<> \t\r\n".encode(name.decode("ascii"))
    except (LookupError, UnicodeError, ValueError):
        return False
    return marks == b"<> \t\r\n"


class _NoTree:
    """An XMLParser target that keeps nothing: the document is parsed for
    whether it parses."""

    def close(self):
        return None


def _row_elements(events):
    """The <row> elements of a worksheet's ("start", "end") ``events``, in
    the order root.iter() visits them -- a row inside another after the one
    holding it -- each complete when it is handed out and let go when the
    next is asked for. Nothing else of the tree is kept."""
    row = f"{SS}row"
    stack, rows_open = [], 0
    for batch in events:
        for event, element in batch:
            if event == "start":
                stack.append(element)
                if element.tag == row:
                    rows_open += 1
                continue
            stack.pop()
            if element.tag == row:
                rows_open -= 1
                if not rows_open:
                    yield from element.iter(row)
            if not rows_open and stack:
                stack[-1].remove(element)


class NoSuchSheet(ValueError):
    """The caller asked for a sheet the file does not have — an argument
    error, which read_table raises, unlike anything wrong with the file."""


def _choose(names: "list[str]", sheet) -> "int | None":
    """The index of the sheet asked for; None when there are none to ask."""
    if sheet is None and not names:
        return None
    index = _pick(names, sheet)
    if index is None:
        raise NoSuchSheet(f"no sheet {sheet!r}; the sheets are: "
                          f"{', '.join(map(repr, names)) or 'none'}")
    return index


def _pick(names: "list[str]", sheet) -> "int | None":
    """The index of the sheet ``sheet`` names among ``names`` -- the first
    for None, an index, a name --, None where it names none of them."""
    if sheet is None:
        return 0 if names else None
    if isinstance(sheet, int) and not isinstance(sheet, bool):
        return sheet if 0 <= sheet < len(names) else None
    if isinstance(sheet, str) and sheet in names:
        return names.index(sheet)
    return None


def _head(lines, convert=None):
    """(header, head, lines) of a sheet as _xlsx and _ods decide its header:
    leading blank rows are not the table, and the header is judged on the
    first nine rows that are. ``head`` holds those rows (the header taken
    off), ``lines`` the rest. With ``convert``, ``lines`` hands out what
    ``convert`` makes a line of, and the head is converted here."""
    lines = iter(lines)
    head = []
    for line in lines:
        if convert is not None:
            line = convert(line)
        if not head and not any(c.strip() for c in line):
            continue
        head.append(line)
        if len(head) >= 9:
            break
    header = ()
    if len(head) > 1 and _rows_look_headed(head):
        header = head.pop(0)
    return header, head, lines


def _page_of_sheet(page: TablePage, lines, offset: int, limit: int,
                   max_chars: int, convert=None, count_rest=False) -> None:
    """Header and page, decided the way _xlsx and _ods decide them (_head);
    with ``count_rest``, the rows after the page counted for the total
    (fill_page) -- where it is a function, by it: how many rows ``lines``
    has not handed out yet.

    With ``convert`` (an xlsx's rows as elements), a row fill_page does not
    look at -- one before ``offset``, one after the row past the page -- is
    counted and never converted: converting was most of what a far page of
    a large sheet cost, and would be most of what counting costs."""
    header, head, lines = _head(lines, convert)
    if convert is not None:
        lines = _converted(lines, convert, offset - len(head), limit + 1)
    rows = itertools.chain(head, lines)
    if callable(count_rest):
        rows = _Rows(head, lines, count_rest)
    fill_page(page, rows, header=header, offset=offset, limit=limit,
              max_columns=MAX_RENDER_COLUMNS, max_chars=max_chars,
              count_rest=bool(count_rest))


class _Rows:
    """The rows of ``head``, then those of ``lines``; ``rest()`` is how many
    are not handed out yet, ``count()`` of those ``lines`` has not handed
    out -- what fill_page counts after a page (count_rest)."""

    __slots__ = ("head", "at", "lines", "count")

    def __init__(self, head: list, lines, count):
        self.head, self.at, self.lines, self.count = head, 0, lines, count

    def __iter__(self):
        return self

    def __next__(self):
        if self.at < len(self.head):
            self.at += 1
            return self.head[self.at - 1]
        return next(self.lines)

    def rest(self) -> int:
        return len(self.head) - self.at + self.count()


def _converted(items, convert, skip: int, look: int):
    """``convert`` of the ``look`` of ``items`` after the first ``skip``,
    the rows fill_page looks at; for every other, a stand-in that it counts
    and never looks at."""
    for index, item in enumerate(items):
        yield convert(item) if skip <= index < skip + look else ()


def _cut_note(sink, times: int) -> None:
    """A row repeated past _ODS_MAX_ROWS. A page that counted 200 rows of a
    table holding a million and one said "200, all read"; the total is
    unknown, and the page says why."""
    sink.truncated = True
    sink.note(f"a row the file repeats {times} times is shown "
              f"{_ODS_MAX_ROWS} times; the repeats after that are not read")


class _OdsTables:
    """An ods's content read once, as _ods_sheet read it in 0.1.55: its
    tables listed -- the first _ODS_MAX_TABLES that hold a row, in the order
    root.iter() visits them, named as _ods_sheet named them -- and the rows
    of the one asked for (_ods_rows' rows) handed out as they are read.

    ``rows()`` ends with the table asked for; ``finish()`` reads the rest
    of the content for the list of sheets (``sheets``) and for what the
    rest of that table says (a row repeated past _ODS_MAX_ROWS), and
    returns the index _choose gives. Nothing of the tree is kept but the
    row being read and the elements around it.

    A table inside a cell of another is settled -- does it hold a row? --
    when its rows say, which may be before the table around it is; the
    rows of a table not yet known to be the one asked for are held until
    it is known."""

    def __init__(self, zf, sheet, sink):
        self._wanted, self._sink = sheet, sink
        self._member = _Member(zf, "content.xml")
        self._whole = None
        if self._member.whole:
            self._read_whole(sheet)
            return
        self._events = _parsed(self._member.pieces())
        self._tables = []           # [name, holds a row?] in document order
        self._open = {}             # id(element) -> index, tables open now
        self._listed = []           # (sheet name, index) listed so far
        self._settled = 0           # _tables[:_settled] are in _listed or out
        self._full = False          # _ODS_MAX_TABLES listed: no more count
        self._held = {}             # index -> [(cells, wanted)]
        self._ready = collections.deque()
        # The elements open now, and for each the table whose rows it
        # holds (_owner): its own index for a table, its parent's for a row
        # group, None for anything else.
        self._stack, self._owners = [], []
        self._rows_open, self._batch = 0, None
        self._finishing = self._done = self._chosen_done = False
        self._counting = False      # count_rest: rows counted, not made
        self._taken = self._handed = 0      # rows of the table asked for
        self.chosen = self.name = self.sheets = None

    def _read_whole(self, sheet) -> None:
        """A content the zip declares no larger than _WHOLE_MEMBER, read as
        0.1.55's _ods_sheet read it: parsed whole, its tables listed, the one
        asked for chosen (NoSuchSheet), before the first row."""
        root = ET.fromstring(b"".join(self._member.pieces()))
        tables = []
        for node in root.iter():
            if not _ods_is(node, "table"):
                continue
            if len(tables) >= _ODS_MAX_TABLES:
                break
            if next(_ods_rows(node), None) is None:
                continue
            name = _attr(node, "name") or f"Table {len(tables) + 1}"
            tables.append((one_line(name), node))
        self.sheets = [name for name, _ in tables]
        self._index = _choose(self.sheets, sheet)
        self.chosen = self.name = None
        self._whole = iter(())
        if self._index is not None:
            self.chosen = self._index
            self.name = self.sheets[self._index]
            sink = self._sink       # not self: a cycle held the member
            self._whole = _ods_rows(tables[self._index][1],
                                    lambda times: _cut_note(sink, times))

    def rows(self):
        if self._whole is not None:
            # Not "yield from": closing this generator when the page is done
            # would close the table's rows, which finish() reads on.
            for row in self._whole:
                yield row
            return
        while True:
            while self._ready:
                cells, repeat = self._ready.popleft()
                for _ in range(repeat):
                    self._handed += 1
                    yield cells
            if self._chosen_done or self._done:
                return
            self._advance()

    def count_rest(self) -> int:
        """How many rows of the table asked for rows() has not handed out:
        those it holds, and those of the rest of the table, read on for it
        and counted as _ods_rows would hand them out -- not made into cells
        (_ods_blank). rows() is not read after."""
        if self._whole is not None:
            count = 0
            for _ in self._whole:
                if not count % 512:
                    deadline.check("the rest of a table")
                count += 1
            return count
        self._counting = True
        self._ready.clear()
        while not (self._chosen_done or self._done):
            self._advance()
        return self._taken - self._handed

    def finish(self) -> "int | None":
        if self._whole is not None:
            for _ in self._whole:
                pass                # the rest of the table's repeat notes
            return self._index
        self._finishing = True
        self._ready.clear()
        self._held.clear()
        while not self._done:
            self._advance()
        self.sheets = [name for name, _ in self._listed]
        return _choose(self.sheets, self._wanted)

    def close(self) -> None:
        if self._whole is None:
            self._events.close()
        self._member.close()

    def _advance(self) -> None:
        """Read on until a row is ready (not while finishing), the table
        asked for has ended, or the content has."""
        table, table_row = f"{TABLE}table", f"{TABLE}table-row"
        stack, owners, groups = self._stack, self._owners, _ODS_ROW_GROUPS
        while True:
            if self._batch is None:
                self._batch = next(self._events, None)
                if self._batch is None:
                    self._done = True
                    return
            for event, element in self._batch:
                tag = element.tag
                if event == "start":
                    stack.append(element)
                    if tag == table:
                        owners.append(len(self._tables))
                        self._open[id(element)] = len(self._tables)
                        self._tables.append([_attr(element, "name"), None])
                    elif tag in groups:
                        owners.append(owners[-1] if owners else None)
                    else:
                        owners.append(None)
                        if tag == table_row:
                            self._rows_open += 1
                    continue
                stack.pop()
                owners.pop()
                if tag == table_row:
                    self._rows_open -= 1
                    self._row(element)
                elif tag == table:
                    self._table_end(self._open.pop(id(element)))
                if not self._rows_open and stack:
                    stack[-1].remove(element)
                if not self._finishing and (self._ready
                                            or self._chosen_done):
                    return              # the batch goes on from here
            self._batch = None

    def _owner(self) -> "int | None":
        """The table the row that just ended is a row of (_ods_row_elements:
        its own rows and those of its row groups), or None: its parent's
        entry in _owners, taken when the parent opened. Walking the row
        groups around the row at each row's end was quadratic in their
        depth: deep-20k.ods took 45.92 s for page 0 in 0.1.58, 0.42 s in
        0.1.55."""
        return self._owners[-1] if self._owners else None

    def _row(self, row) -> None:
        owner = self._owner()
        if owner is None:
            return
        entry = self._tables[owner]
        wanted = _int(_attr(row, "number-rows-repeated"), 1)
        chosen = owner == self.chosen
        if chosen and self._counting:
            if not _ods_blank(row):
                self._take(None, wanted)
            return
        undecided = self.chosen is None and owner >= self._settled
        if not ((entry[1] is None and not self._full) or undecided or (
                chosen and (not self._finishing
                            or wanted > _ODS_MAX_ROWS))):
            return
        cells = _ods_row(row)
        if not any(c.strip() for c in cells):
            return                      # a run of empty rows says nothing
        if entry[1] is None:
            entry[1] = True
            self._settle()
            chosen = owner == self.chosen
        if chosen:
            self._take(cells, wanted)
        elif self.chosen is None and owner >= self._settled:
            self._held.setdefault(owner, []).append((cells, wanted))

    def _table_end(self, index: int) -> None:
        if self._tables[index][1] is None:
            self._tables[index][1] = False
            self._settle()
        if index == self.chosen:
            self._chosen_done = True

    def _settle(self) -> None:
        """List the tables whose question is answered, in order, up to the
        first that is not; then choose, if the one asked for is listed."""
        tables, listed = self._tables, self._listed
        while not self._full and self._settled < len(tables):
            name, holds = tables[self._settled]
            if holds is None:
                break
            if holds:
                listed.append((one_line(name or f"Table {len(listed) + 1}"),
                               self._settled))
                self._full = len(listed) >= _ODS_MAX_TABLES
            self._settled += 1
        if self.chosen is not None:
            return
        pick = _pick([name for name, _ in listed], self._wanted)
        if pick is None:
            return
        self.name, self.chosen = listed[pick]
        self._chosen_done = self.chosen not in self._open.values()
        for cells, times in self._held.pop(self.chosen, ()):
            self._take(cells, times)
        self._held.clear()

    def _take(self, cells: list, wanted: int) -> None:
        """A row of the table asked for, as _ods_rows hands it out."""
        repeat = min(wanted, _ODS_MAX_ROWS)
        if wanted > repeat:
            _cut_note(self._sink, wanted)
        if not self._finishing:
            self._taken += repeat
            if not self._counting:
                self._ready.append((cells, repeat))


#: An epub is a zip of XHTML chapters with a manifest. Nothing read it, so a
#: book came back through the binary fallback as "epub. Preview: # Binary /
#: epub size: 514 B magic: 50 4b 03 04…" — a hex dump of a file sift knows the
#: format of.
_EPUB_SKIP = ("nav.xhtml", "toc.xhtml", "cover.xhtml")


def _looks_like_markup(raw: bytes) -> bool:
    """Whether a chapter member is actually markup."""
    head = raw[:1024].lstrip()
    return head.startswith((b"<", b"\xef\xbb\xbf<")) or b"<html" in raw[:4096].lower()


#: What one archive may yield in total, across every member. The per-member
#: ceiling was enforced; the member *count* was not, and 60 chapters of
#: 8 MB each fit in a 1 MB epub — 124 seconds of CPU for one upload. Worse
#: for epub than for the rest, because a text-free chapter produced no text,
#: so the "enough characters read" break never fired however many were read.
_MAX_TOTAL_INFLATED = 64 * 1024 * 1024


def _within_budget(zf, names, budget: "_Budget | None" = None):
    """Yield members until the archive's declared output exceeds the budget.

    Judged on the size the zip directory *declares*, which costs no
    decompression — a bomb has to declare its output to be a bomb. Yields a
    final ``None`` when it stops early, so the caller can say so rather than
    silently reporting a truncated book as a whole one. ``budget``: one
    already charged with the members read before these (_Budget).
    """
    budget = budget if budget is not None else _Budget(zf)
    for name in names:
        if budget.take(name):
            yield name
        elif budget.over:
            yield None
            return


class _Budget:
    """What the members read from one archive declare together, held to
    _MAX_TOTAL_INFLATED (_within_budget). ``take(name)`` charges a member
    and says whether it may be read: False for one the zip does not hold,
    and for every member once one has gone past the budget (``over``). The
    clock is looked at per member."""

    def __init__(self, zf, spent: int = 0):
        self._zf, self.spent, self.over = zf, spent, False

    def take(self, name: str) -> bool:
        deadline.check("an archive member")
        if self.over:
            return False
        try:
            self.spent += self._zf.getinfo(name).file_size
        except KeyError:
            return False
        if self.spent > _MAX_TOTAL_INFLATED:
            self.over = True
            return False
        return True


#: What reading one part besides a document's body can raise: a member
#: missing, damaged (BadZipFile is a CRC failure, zlib.error and EOFError a
#: broken stream), encrypted or in a method zipfile cannot read
#: (RuntimeError), or no XML. Such a part is passed over and counted; the
#: body read on its own terms (_docx, _xlsx, _pptx, _odf).
_PART_FAILURES = (KeyError, OSError, EOFError, RuntimeError,
                  zipfile.BadZipFile, zlib.error, ET.ParseError)


def _relations(zf, part: str, budget: _Budget) -> "list[tuple[str, str, str]]":
    """(Id, type, member) of each relationship of ``part`` to another member
    of the package, in the order its relationships part lists them: the
    type the last segment of the relationship's Type (header, comments,
    notesSlide -- the same in transitional and strict OOXML), the member its
    Target resolved as OPC resolves it (_resolved). An external target, one
    outside the package and a relationships part that is missing, over
    ``budget`` or unreadable give nothing."""
    folder, _, name = part.rpartition("/")
    rels = f"{folder}/_rels/{name}.rels" if folder else f"_rels/{name}.rels"
    if not budget.take(rels):
        return []
    try:
        root = ET.fromstring(_read_member(zf, rels))
    except _PART_FAILURES:
        return []
    found = []
    for rel in root:
        if rel.get("TargetMode", "").lower() == "external":
            continue
        member = _resolved(folder, rel.get("Target", ""))
        if member:
            found.append((rel.get("Id", ""),
                          rel.get("Type", "").rsplit("/", 1)[-1], member))
    return found


def _resolved(folder: str, target: str) -> str:
    """The member a relationship's ``target`` names: read against the
    folder of the part it belongs to, or from the package's root when it
    begins with "/". "" for one that climbs out of the package."""
    if not target:
        return ""
    if target.startswith("/"):
        path = target[1:]
    else:
        path = f"{folder}/{target}" if folder else target
    path = posixpath.normpath(path)
    if path in (".", "..") or path.startswith(("../", "/")):
        return ""
    return path


def _numbered(names, pattern) -> "list[str]":
    """The members of ``names`` ``pattern`` matches whole, in the order of
    the number its group holds -- header2 before header10 (_sheets_in_order
    says why not sorted()) --, none before 1."""
    found = []
    for name in names:
        match = pattern.fullmatch(name)
        if match:
            found.append((int(match.group(1) or 0), name))
    return [name for _, name in sorted(found)]


def _paragraph_lines(root, para: str, text: str, breaks=frozenset(),
                     tabs=frozenset(), skip=frozenset()) -> "list[str]":
    """The text of each ``para`` element under ``root``, one line each, in
    document order, blank ones left out: its ``text`` elements' text, "\n"
    for a ``breaks`` element and "\t" for a ``tabs`` one, as
    _docx_paragraph reads a paragraph; nothing of a ``skip`` element. A
    paragraph inside another -- a text box in a run -- is a line of its own
    and no part of the one around it, so every element is read once: linear
    in the part, where an iter() under each of a nest of paragraphs reads
    the nest once per level."""
    lines = []
    for count, top in enumerate(root.iter(para)):
        if not count % 1024:
            deadline.check("a document's parts")
        line = _paragraph_text(top, para, text, breaks, tabs, skip).strip()
        if line:
            lines.append(line)
    return lines


def _paragraph_text(top, para: str, text: str, breaks=frozenset(),
                    tabs=frozenset(), skip=frozenset()) -> str:
    """The text of the ``para`` element ``top`` as _paragraph_lines reads
    each: a paragraph inside it -- a text box's -- is no part of it."""
    out, stack = [], [iter(top)]
    while stack:
        for node in stack[-1]:
            tag = node.tag
            if tag == text:
                out.append(node.text or "")
            elif tag in breaks:
                out.append("\n")
            elif tag in tabs:
                out.append("\t")
            elif tag != para and tag not in skip and len(node):
                stack.append(iter(node))
                break
        else:
            stack.pop()
    return "".join(out)


#: A letter or a digit: a line of a header or a footer without one is the
#: separators left between its fields ("Page 1 / 99" less its fields is
#: "Page /", LibreOffice's report header "(???)" and ",").
_A_WORD = re.compile(r"[^\W_]")


def _sections(pairs) -> str:
    """(heading, lines) pairs as the parts after a document's body are
    written: each under its "## " heading, a blank line between."""
    return "\n\n".join(f"## {heading}\n" + "\n".join(lines)
                        for heading, lines in pairs)


def _spine_order(zf) -> "list[str]":
    """Chapters in the order the book is meant to be read.

    sorted() gives chapter1, chapter10, chapter11, chapter2 — the same
    lexicographic mistake _sheets_in_order twenty lines below exists to
    prevent, and the model was shown the book shuffled. The spine is the
    authoritative order and sits in the OPF that is already being opened for
    the title.
    """
    for name in zf.namelist():
        if not name.endswith(".opf"):
            continue
        try:
            root = ET.fromstring(_read_member(zf, name))
        except (ET.ParseError, KeyError, OSError):
            continue
        hrefs = {}
        for item in root.iter():
            if item.tag.endswith("}item") or item.tag == "item":
                if item.get("id"):
                    hrefs[item.get("id")] = item.get("href", "")
        order = []
        base = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
        for ref in root.iter():
            if not (ref.tag.endswith("}itemref") or ref.tag == "itemref"):
                continue
            href = hrefs.get(ref.get("idref", ""))
            if not href:
                continue
            member = base + href if base + href in zf.namelist() else href
            if member in zf.namelist():
                order.append(member)
        if order:
            return order
    return []


def _epub(zf, det, props) -> Extracted:
    from .html_doc import html_to_text
    title = props.get("title", "")
    author = props.get("author", "")
    if not title:
        for name in (n for n in zf.namelist() if n.endswith(".opf")):
            try:
                root = ET.fromstring(_read_member(zf, name))
            except ET.ParseError:
                continue
            for tag, key in ((f"{DC}title", "title"), (f"{DC}creator", "author")):
                node = root.find(f".//{tag}")
                if node is not None and (node.text or "").strip():
                    if key == "title":
                        title = node.text.strip()
                    else:
                        author = author or node.text.strip()
            break
    chapters = _spine_order(zf) or [
        n for n in sorted(zf.namelist())
        if n.lower().endswith((".xhtml", ".html", ".htm"))
        and not n.lower().endswith(_EPUB_SKIP)]
    blocks, read, over_budget = [], 0, False
    for name in _within_budget(zf, chapters):
        if name is None:
            over_budget = True
            break
        if read >= MAX_CHARS:
            break
        try:
            raw = _read_member(zf, name)
        except (KeyError, OSError, zipfile.BadZipFile, RuntimeError):
            # BadZipFile is a CRC failure and RuntimeError an encrypted
            # member; neither was caught, so one bad chapter lost the book.
            continue
        if b"\x00" in raw[:512] or not _looks_like_markup(raw):
            # A JPEG named .xhtml decoded to mojibake and NUL characters and
            # became the opening of the book as far as the model could tell.
            continue
        try:
            text = html_to_text(raw.decode("utf-8", "replace"))
        except ValueError:
            continue
        text = text.strip()
        if not text:
            continue
        blocks.append(text)
        read += len(text)
    body = "\n\n".join(blocks)
    meta = {k: v for k, v in props.items() if v}
    if over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of chapter text; "
            f"the rest of the book was not read")
    meta.update({"title": title or meta.get("title", ""),
                 "author": author or meta.get("author", ""),
                 "chapters": len(chapters),
                 "words": len(body.split())})
    meta = {k: v for k, v in meta.items() if v not in ("", None)}
    text, truncated = clip(body, MAX_CHARS)
    ex = Extracted("epub", det.mime, text=text, metadata=meta,
                   truncated=truncated)
    if len(blocks) < len(chapters):
        ex.note(f"read {len(blocks)} of {len(chapters)} chapters")
    return ex


# ---- docx -------------------------------------------------------------
def _docx(zf, det, props) -> Extracted:
    root = ET.fromstring(_read_member(zf, "word/document.xml"))
    # A table's cells contain paragraphs, and root.iter(w:p) descends into
    # them — so rendering tables separately printed every cell twice, once as
    # a stray unlabelled line and once in its table, and counted them twice.
    tables, in_tables = _docx_tables(root)
    paragraphs = []
    for count, para in enumerate(root.iter(f"{W}p")):
        if not count % _CHECK_EVERY:
            deadline.check("a document's paragraphs")
        if id(para) in in_tables:
            continue
        line = _docx_paragraph(para).strip()
        if line:
            paragraphs.append(line)
    text = "\n".join(paragraphs)
    if tables:
        # "tables: 1" while the cells arrive as six unlabelled lines implies a
        # structure the model never receives. Render them as tables instead.
        text += "\n\n" + "\n\n".join(
            "## Table\n" + "\n".join(" | ".join(cell for cell in row) for row in t)
            for t in tables)
    # The headers, footers, notes and comments after the body and its
    # tables, which stay where they were (_DocxParts).
    parts = _DocxParts(zf, root, MAX_CHARS - len(text))
    if parts.sections:
        text += ("\n\n" if text else "") + _sections(parts.sections)
    # Counted over everything delivered, tables included: reporting the
    # paragraph total while handing the model half again as much text is a
    # figure that does not describe the file.
    meta = {**props, "paragraphs": len(paragraphs), "words": len(text.split()),
            "characters": len(text), "tables": len(tables)}
    if tables:
        meta["table_shapes"] = [f"{len(t)}×{max(len(r) for r in t)}" for t in tables]
        meta["table_cells"] = sum(len(r) for t in tables for r in t)
    meta.update({kind: count for kind, count in parts.entries.items()
                 if count})
    if parts.over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of the "
            f"document's XML; later header, footer, note and comment parts "
            f"were not read")
    body, truncated = clip(text, MAX_CHARS)
    ex = Extracted("docx", det.mime, text=body, metadata=meta,
                   truncated=truncated)
    if parts.unread:
        ex.note(f"{parts.unread} of the document's header, footer, note and "
                f"comment parts could not be read")
    return ex


#: The parts of a Word document besides its body that hold text a reader
#: sees, by the type of the relationship that names them: the heading each
#: is written under, after the body in this order, and the member name Word
#: gives it, by which one no relationship names is found.
_DOCX_PARTS = (
    ("header", "Header", re.compile(r"word/header([0-9]{0,9})\.xml")),
    ("footer", "Footer", re.compile(r"word/footer([0-9]{0,9})\.xml")),
    ("footnotes", "Footnotes", re.compile(r"word/footnotes()\.xml")),
    ("endnotes", "Endnotes", re.compile(r"word/endnotes()\.xml")),
    ("comments", "Comments", re.compile(r"word/comments()\.xml")),
)
#: The entries of a notes or comments part: each is read, and counted in
#: the metadata, on its own. Word's separator notes hold no text.
_DOCX_ENTRIES = {"footnotes": f"{W}footnote", "endnotes": f"{W}endnote",
                 "comments": f"{W}comment"}


class _DocxParts:
    """A Word document's headers, footers, footnotes, endnotes and comments,
    which _docx read none of: a password in a footer or a comment was never
    in sift's text, so find_secrets never saw it and a host application, which shows
    that text as the document's, did not show it (a census of a host application's
    viewer: 5 of 18 real .docx files).

    The parts come in the document's order: headers and footers as the
    body's section properties name them (w:headerReference,
    w:footerReference, first naming first), then any other the document's
    relationships name, in their order, then a member Word would have named
    so (word/header2.xml) that nothing names, in the order of its number
    -- its text is in the file whether Word shows it or not. Each member is
    read once, under the first heading that claims it; a header or a footer
    that says word for word what one before it said (a first-page header
    as the default one) is written once.

    Read as the body is: every member through _read_member, parsed by the
    same ElementTree, a paragraph's text as _docx_paragraph reads it
    (_paragraph_lines). Bounded as an epub's chapters and a workbook's
    sheets are: what the body, the relationships and the parts declare
    together stays within _MAX_TOTAL_INFLATED (_Budget), the clock is
    looked at per member, zipguard lets no zip hold more than
    MAX_ZIP_ENTRIES of them, and reading stops once the text is past
    ``room``, the characters clip() keeps after the body. A part that is
    missing, damaged or no XML is counted in ``unread`` and the rest read.

    ``sections`` are (heading, lines), ``entries`` the footnotes, endnotes
    and comments that hold text, ``over_budget`` whether parts were left
    for the budget."""

    def __init__(self, zf, body, room: int):
        self.unread, self.over_budget = 0, False
        lines = {kind: [] for kind, _, _ in _DOCX_PARTS}
        self.entries = {kind: 0 for kind in _DOCX_ENTRIES}
        seen = {kind: set() for kind in lines}
        budget = _Budget(zf, _declared(zf, "word/document.xml"))
        kinds = self._members(zf, body, budget)
        for member in _within_budget(zf, list(kinds), budget):
            if member is None:
                self.over_budget = True
                break
            if room < 0:
                break
            kind = kinds[member]
            try:
                root = ET.fromstring(_read_member(zf, member))
            except _PART_FAILURES:
                self.unread += 1
                continue
            entry = _DOCX_ENTRIES.get(kind)
            items = [root] if entry is None else [
                child for child in root if child.tag == entry]
            for item in items:
                found = _paragraph_lines(item, f"{W}p", f"{W}t",
                                         breaks=(f"{W}br", f"{W}cr"),
                                         tabs=(f"{W}tab",),
                                         skip=(f"{W}pPr",))
                if not found:
                    continue
                if entry is None:
                    said = tuple(found)
                    if said in seen[kind]:
                        continue
                    seen[kind].add(said)
                else:
                    self.entries[kind] += 1
                lines[kind].extend(found)
                room -= sum(len(line) + 1 for line in found)
        self.sections = [(heading, lines[kind])
                         for kind, heading, _ in _DOCX_PARTS if lines[kind]]

    @staticmethod
    def _members(zf, body, budget: _Budget) -> dict:
        """{member: type} of every part to read, in the order it is read."""
        names = set(zf.namelist())
        named = {}
        for sect in body.iter(f"{W}sectPr"):
            for ref in sect:
                if ref.tag in (f"{W}headerReference", f"{W}footerReference"):
                    named.setdefault(ref.get(f"{R}id", ""), len(named))
        relations = _relations(zf, "word/document.xml", budget)
        kinds = {}
        for kind, _heading, pattern in _DOCX_PARTS:
            related = sorted(
                (named.get(rid, len(named)), index, member)
                for index, (rid, typ, member) in enumerate(relations)
                if typ == kind and member in names)
            for member in ([m for _, _, m in related]
                           + _numbered(names, pattern)):
                kinds.setdefault(member, kind)
        return kinds


def _declared(zf, name: str) -> int:
    """What the zip declares ``name`` inflates to; 0 for no such member."""
    try:
        return zf.getinfo(name).file_size
    except KeyError:
        return 0


_DOCX_BREAKS = frozenset({f"{W}br", f"{W}cr"})
_DOCX_TABS = frozenset({f"{W}tab"})


def _docx_paragraph(para) -> str:
    """Paragraph text with breaks and tabs kept.

    <w:br/> and <w:tab/> carry no text node, so dropping them welded "Line A"
    to "Line B" and "Col1" to "Col2" — changing both the reading and the word
    count.

    A paragraph inside this one -- a text box's, anchored in one of its
    runs -- is no part of it: root.iter() reaches it as a paragraph of its
    own (a table's walk as a piece of its cell). Read with every descendant,
    the box's "password: SECRET" and the run "Item" after it were one line,
    "password: SECRETItem", and a span "SECRETItem"; a nest of 5,000
    boxes read 12.5 million elements. Word writes a text box twice, its
    DrawingML shape and a VML fallback (mc:AlternateContent): both are
    read, each a line of its own.
    """
    return _paragraph_text(para, f"{W}p", f"{W}t", _DOCX_BREAKS, _DOCX_TABS)


#: Elements walked between two looks at the clock, in a walk of a part
#: that does little per element (_docx_tables, _ods).
_CHECK_EVERY = 1024


def _docx_tables(root) -> "tuple[list[list[list[str]]], set]":
    """(tables, in_tables): every table of the document, each rendered once,
    in the order root.iter() visits them, and the ids of the paragraphs
    inside any table, which the body's paragraphs leave out (_docx).

    A row is its nearest table's, a cell its nearest row's, and a paragraph
    is its nearest cell's text where no table lies between them: iter()
    descends, so a table nested in a cell had its rows counted as rows of
    the outer table as well -- a 2x2 inside a 2x2 was reported as 4x6 with
    16 cells for 8, the inner text delivered four times -- and _docx_paragraph
    iterates every descendant, so the outer cell absorbed the nested table's
    paragraphs.

    One walk of each table no other holds, each element of it visited once,
    the clock looked at every _CHECK_EVERY elements; what lies outside every
    table is passed over by root.iter(). Sets of the nested tables' rows and
    cells, rebuilt per table, were cubic in the nesting depth (1,600 nested
    <w:tbl> is a 1,016-byte file and took over ninety seconds); built once
    they still walked every table below each table -- 51x the CPU for 8x
    the depth, 6,400 tables 50 s against a 5 s budget, the clock looked at
    only once they were built -- and took the rows of every nested table out
    of every table that held one, the nested table's own among them: of
    three tables each nested in the one before, the middle one was not
    rendered at all (0.1.77).
    """
    tbl, tr, tc, para = f"{W}tbl", f"{W}tr", f"{W}tc", f"{W}p"
    tables, in_tables, walked, nested = [], set(), 0, set()
    for top in root.iter(tbl):
        if id(top) in nested:
            continue                    # walked with the table around it
        # Per open element: its children, and the table, the row and the
        # cell that what lies below it belongs to, None where there is none.
        stack = [(iter((top,)), None, None, None)]
        while stack:
            children, table, row, cell = stack[-1]
            for node in children:
                walked += 1
                if not walked % _CHECK_EVERY:
                    deadline.check("a document's tables")
                tag = node.tag
                below = (table, row, cell)
                if tag == tbl:
                    nested.add(id(node))
                    below = ([], None, None)
                    tables.append(below[0])
                elif tag == tr and table is not None:
                    below = (table, [], None)
                    table.append(below[1])
                elif tag == tc and row is not None:
                    below = (table, row, [])
                    row.append(below[2])
                elif tag == para:
                    in_tables.add(id(node))
                    if cell is not None:
                        cell.append(_docx_paragraph(node))
                if len(node):
                    stack.append((iter(node), *below))
                    break
            else:
                stack.pop()
    rendered = []
    for table in tables:
        rows = [[" ".join(" ".join(cell).split()) for cell in row]
                for row in table if row]
        if rows:
            rendered.append(rows)
    return rendered, in_tables


# ---- xlsx -------------------------------------------------------------
def _xlsx(zf, det, props) -> Extracted:
    shared = _shared_strings(zf)
    blocks, sheets_meta = [], []
    any_capped = any_wide = over_budget = False
    pairs = _sheets_in_order(zf)
    spent = _Budget(zf)
    budget = _within_budget(zf, [p for _, p in pairs], spent)
    # Each sheet reached, with its header and footer (_read_sheet), for the
    # parts after the sheets (_XlsxNotes).
    reached = []
    for (name, part), allowed in zip(pairs, budget):
        if allowed is None:
            over_budget = True
            break
        furniture = {}
        reached.append((name, part, furniture))
        try:
            raw = _read_member(zf, part)
        except (KeyError, OSError):
            sheets_meta.append({"name": name, "error": "unreadable sheet"})
            continue
        try:
            rows, total_rows, n_cols, wide, formulas = _read_sheet(
                raw, shared, furniture=furniture)
        except ET.ParseError:
            if _is_truncated(raw):
                # Cut at the member cap, so the XML ends mid-element. The rows
                # before the cut are real and worth reporting; saying only
                # "unreadable sheet" about a sheet sift read 200 rows of would
                # be the truncation lying about itself.
                rows, total_rows, n_cols, wide, formulas = _read_sheet(
                    _closed(raw), shared)
                sheets_meta.append({
                    "name": name, "rows": total_rows, "columns": n_cols,
                    "rows_read": len(rows),
                    "note": f"sheet larger than {human_size(_MAX_MEMBER)}; "
                            f"read the first part"})
                any_capped = True
                blocks.append(f"## Sheet: {name} (first {total_rows} rows read "
                              f"of a larger sheet × {n_cols} cols)")
                for row in rows[:12]:
                    blocks.append("\t".join(row))
                blocks.append("")
                continue
            sheets_meta.append({"name": name, "error": "unreadable sheet"})
            continue
        # A blank spacer row above the table is one of the commonest
        # spreadsheet shapes, and the CSV reader drops leading blank lines.
        # Keeping them here made the same table report one row more as .xlsx
        # than as .csv — the very inconsistency this was meant to remove.
        leading_blank = 0
        while rows and not any(c.strip() for c in rows[0]):
            rows.pop(0)
            leading_blank += 1
        total_rows -= leading_blank
        capped = total_rows > len(rows)
        any_capped |= capped
        # "rows" has to mean the same thing it means for a CSV: records, not
        # lines. Counting the header among them made the same spreadsheet
        # report 9 rows as .xlsx and 8 as .csv — one word, two meanings, in
        # the one place a reader compares two exports of the same table.
        # Two rows minimum. _rows_look_headed calls a lone textual row a
        # header — right for a one-line CSV, absurd for a sheet, where it
        # turned a single record into "rows: 0".
        headed = len(rows) > 1 and _rows_look_headed(rows[:9])
        entry = {"name": name, "rows": total_rows - (1 if headed else 0),
                 "columns": n_cols, "has_header": headed}
        if formulas:
            # A formula cell shows whatever the last program to save the file
            # cached there. Nothing recalculates it on the way here, so a
            # SUM over 1000 and 2000 can arrive reading 9, and every summary
            # downstream states the 9 as a fact about the business.
            entry["formula_cells"] = formulas
        if wide:
            # Losing columns 513+ with nothing said is the same silent
            # truncation the row cap is careful to announce.
            entry["columns_read"] = MAX_RENDER_COLUMNS
            any_wide = True
        if capped:
            # The cap is a reading limit, not the sheet's size. Reporting the
            # capped number as the row count turned a 501-row sheet into a
            # 200-row one, silently and with no note.
            entry["rows_read"] = len(rows)
        sheets_meta.append(entry)
        shown = (f"{entry['rows']} data rows" if headed else f"{total_rows} rows")
        shown += f", first {len(rows)} read" if capped else ""
        blocks.append(f"## Sheet: {name} ({shown} × {n_cols} cols)")
        for row in rows[:12]:
            blocks.append("\t".join(row))
        blocks.append("")

    meta = {**props, "sheets": sheets_meta, "sheet_count": len(sheets_meta)}
    computed = sum(s.get("formula_cells", 0) for s in sheets_meta)
    if computed:
        meta["computed_cells"] = (
            f"{computed} cell(s) hold a formula; the values shown are the "
            f"ones cached by the program that saved the file and are not "
            f"recalculated here")
    if over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of sheet XML; "
            f"later sheets were not read")
    text = "# Spreadsheet\n" + "\n".join(blocks)
    notes = _XlsxNotes(zf, reached, spent, MAX_CHARS - len(text))
    if notes.blocks:
        text += "\n" + "\n".join(notes.blocks)
    if notes.comments:
        meta["comments"] = notes.comments
    if notes.over_budget and not over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of sheet XML "
            f"and comments; later comments were not read")
    body, truncated = clip(text, MAX_CHARS)
    ex = Extracted("xlsx", det.mime, text=body, metadata=meta, rendered=True,
                   truncated=truncated or any_capped or any_wide)
    if notes.unread:
        ex.note(f"{notes.unread} comments part"
                f"{'s' if notes.unread != 1 else ''} could not be read")
    if any_capped:
        ex.note(f"read the first {SHEET_ROW_CAP} rows of the larger sheets")
    if any_wide:
        ex.note(f"showed the first {MAX_RENDER_COLUMNS} columns of the wider "
                f"sheets; the column counts are the widest row that was read, "
                f"which for a capped sheet may not be the widest row it has")
    return ex


#: Where a sheet's header and footer are written (headerFooter): odd and
#: even pages and the first, each a header or a footer.
_XLSX_HEADER_FOOTER = {f"{SS}{name}": name[-6:].lower() for name in (
    "oddHeader", "evenHeader", "firstHeader", "oddFooter", "evenFooter",
    "firstFooter")}
#: A comments part as Excel names it, by which one no sheet relates is
#: found (xl/comments1.xml).
_XLSX_COMMENTS = re.compile(r"xl/comments([0-9]{0,9})\.xml")
#: A cell's address as a comment's ``ref`` gives it (A1 to XFD1048576);
#: any other ref is not written.
_CELL_REF = re.compile(r"[A-Za-z]{1,3}[0-9]{1,7}")
_DIGITS = frozenset("0123456789")


def _sheet_header_footer(root) -> dict:
    """{"header": lines, "footer": lines} of a worksheet's headerFooter,
    each line once, in the order written (_header_footer_lines)."""
    found = {"header": {}, "footer": {}}
    block = root.find(f"{SS}headerFooter")
    for child in (block if block is not None else ()):
        which = _XLSX_HEADER_FOOTER.get(child.tag)
        if which:
            for line in _header_footer_lines(child.text or ""):
                found[which].setdefault(line)
    return {which: list(lines) for which, lines in found.items()}


def _header_footer_lines(value: str) -> "list[str]":
    """The text of a sheet's header or footer as Excel prints it, one line
    per section and per line break, blank space folded.

    The value is text with Excel's codes in it: "&L", "&C" and "&R" begin
    the left, centre and right section (a line each here); "&&" is an
    ampersand; a font ("&"Arial,Bold""), a size ("&12") and a colour
    ("&KFF0000", a theme's "&K01+000") are formatting; every other code is
    a field (the page, the pages, the date, the time, the file's and the
    sheet's name, a picture) or a switch (bold, italic, ...), none of it
    text of the file. A line with no letter or digit is what separated its
    fields (_A_WORD): LibreOffice writes '&C&"DejaVu Serif,Book"&12&A'
    into every sheet, which is no line at all. One pass, no backtracking.
    """
    sections, out, at, end = [], [], 0, len(value)
    while at < end:
        amp = value.find("&", at)
        if amp < 0:
            out.append(value[at:])
            break
        out.append(value[at:amp])
        code = value[amp + 1:amp + 2]
        at = amp + 2
        if code == "&":
            out.append("&")
        elif code in ("L", "C", "R"):
            sections.append("".join(out))
            out = []
        elif code == '"':
            close = value.find('"', at)
            at = end if close < 0 else close + 1
        elif code in _DIGITS:
            while at < end and value[at] in _DIGITS:
                at += 1
        elif code == "K":
            at += 6
    sections.append("".join(out))
    lines = []
    for section in sections:
        for line in section.splitlines():
            line = " ".join(line.split())
            if _A_WORD.search(line):
                lines.append(line)
    return lines


class _XlsxNotes:
    """What an xlsx says besides its cells, which _xlsx read none of: each
    sheet's header and footer (headerFooter, read with the sheet) and its
    cell comments, the comments part the sheet's relationships name
    (xl/comments1.xml). A comments part nothing relates is read after them,
    in the order of its number: its text is in the file.

    ``blocks`` are written after every sheet -- the sheets' rendering stays
    as it was --, per sheet in the workbook's order its header, its footer
    and its comments, each comment as "A1: text", under headings that begin
    "## Sheet", as the sheets' own do: the model's note names them sift's
    ("# Sheet" is in them, _rendering_note), and _own_text keeps them as the
    sheets' headings, whose lines begin no entry of _OUR_RENDERING.
    ``comments`` counts the comments that hold text.

    Bounded as the sheets are: the relationships and the comments parts go
    on ``budget``, the sheets' (_Budget), reading stops once the text is
    past ``room``, what clip() keeps, and a part that is missing, damaged or
    no XML is counted in ``unread``."""

    def __init__(self, zf, sheets, budget: _Budget, room: int):
        self.blocks, self.comments, self.unread = [], 0, 0
        names = set(zf.namelist())
        owner = {}
        for index, (_name, part, _furniture) in enumerate(sheets):
            for _rid, kind, member in _relations(zf, part, budget):
                if kind == "comments" and member in names:
                    owner.setdefault(member, index)
        for member in _numbered(names, _XLSX_COMMENTS):
            owner.setdefault(member, None)
        said = collections.defaultdict(list)
        for member in _within_budget(zf, list(owner), budget):
            if member is None or room < 0:
                break
            try:
                root = ET.fromstring(_read_member(zf, member))
            except _PART_FAILURES:
                self.unread += 1
                continue
            lines = self._comments(root)
            said[owner[member]].extend(lines)
            room -= sum(len(line) + 1 for line in lines)
        self.over_budget = budget.over
        for index, (name, _part, furniture) in enumerate(sheets):
            name = one_line(name)
            self._section(f"Sheet header: {name}", furniture.get("header"))
            self._section(f"Sheet footer: {name}", furniture.get("footer"))
            self._section(f"Sheet comments: {name}", said.get(index))
        self._section("Sheet comments", said.get(None))

    def _comments(self, root) -> "list[str]":
        """The lines of every comment of a comments part that holds text,
        its text as a shared string's (every <t>), its cell's address
        before it."""
        lines = []
        for comment in root.iter(f"{SS}comment"):
            body = comment.find(f"{SS}text")
            text = "" if body is None else "".join(
                t.text or "" for t in body.iter(f"{SS}t"))
            found = [line.rstrip() for line in text.splitlines()
                     if line.strip()]
            if not found:
                continue
            self.comments += 1
            ref = comment.get("ref", "")
            if _CELL_REF.fullmatch(ref):
                found[0] = f"{ref}: {found[0].strip()}"
            lines.extend(found)
        return lines

    def _section(self, heading: str, lines) -> None:
        if lines:
            self.blocks.append(f"## {heading}")
            self.blocks.extend(lines)
            self.blocks.append("")


def _sheets_in_order(zf) -> list[tuple[str, str]]:
    """Pair each sheet name with its part, via the workbook relationships.

    Matching sorted(sheet1.xml, sheet10.xml, …) against the name list by index
    is lexicographic, so from ten sheets onward almost every sheet's content
    was filed under someone else's name — eleven of twelve, in the audit.
    The r:id in workbook.xml is the authoritative link.
    """
    names = zf.namelist()
    parts = [n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)]
    # Numeric, not lexicographic: sheet10 belongs after sheet9, not after sheet1.
    parts.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)))

    book = None
    try:
        book = ET.fromstring(_read_member(zf, "xl/workbook.xml"))
    except (KeyError, ET.ParseError):
        pass
    book_names = [s.get("name", "") for s in book.iter(f"{SS}sheet")] if book else []

    if book is not None:
        try:
            rels = ET.fromstring(_read_member(zf, "xl/_rels/workbook.xml.rels"))
        except (KeyError, ET.ParseError):
            rels = None
        if rels is not None:
            target_by_id = {r.get("Id"): r.get("Target", "") for r in rels}
            pairs: list[tuple[str, str]] = []
            for sheet in book.iter(f"{SS}sheet"):
                target = target_by_id.get(sheet.get(f"{R}id"), "")
                if not target:
                    continue
                part = target.lstrip("/")
                if not part.startswith("xl/"):
                    part = "xl/" + part.lstrip("./")
                # Only worksheets. A chartsheet is listed among the sheets but
                # its part lives elsewhere; counting it made the pair count
                # differ from the worksheet count, which threw away the correct
                # mapping and reinstated the very bug this replaced — one
                # workbook lost a sheet entirely.
                if part in names and part.startswith("xl/worksheets/"):
                    pairs.append((sheet.get("name", part), part))
            if {p for _, p in pairs} == set(parts):
                return pairs
    # No usable relationships: pair by position, which is right as long as the
    # workbook lists its sheets in file order — true for every writer we have
    # seen, and now at least not defeated by sorting.
    return [(book_names[i] if i < len(book_names) and book_names[i]
             else f"Sheet{i + 1}", p) for i, p in enumerate(parts)]


def _shared_strings(zf) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(_read_member(zf, "xl/sharedStrings.xml"))
    out = []
    for si in root.findall(f"{SS}si"):
        out.append("".join(t.text or "" for t in si.iter(f"{SS}t")))
    return out


def _sheet_names(zf) -> list[str]:
    if "xl/workbook.xml" not in zf.namelist():
        return []
    root = ET.fromstring(_read_member(zf, "xl/workbook.xml"))
    return [s.get("name", "") for s in root.iter(f"{SS}sheet")]


def _closed(raw: bytes) -> bytes:
    """Truncated sheet XML, cut back to the last complete row and closed.

    ElementTree needs a well-formed document; a member cut at a byte boundary
    is not one. Dropping the partial row and appending the closing tags gives
    the parser the rows that did arrive.
    """
    end = raw.rfind(b"</row>")
    if end < 0:
        return b"<worksheet><sheetData></sheetData></worksheet>"
    return raw[:end + 6] + b"</sheetData></worksheet>"


def _read_sheet(data: bytes, shared: list[str], max_rows: int = None,
                furniture: "dict | None" = None,
                ) -> tuple[list[list[str]], int, int, bool, int]:
    """Rows read, and how many the sheet actually has.

    Cell positions come from the ``r`` reference, not from document order. A
    row that skips a column — the normal shape of a real spreadsheet — used to
    shift every later value one place left, so a value in column D was reported
    under B, and a lone B5 landed in column A.

    ``furniture``, where the caller wants it, is a dict it owns, given the
    sheet's header and footer lines (_sheet_header_footer).
    """
    cap = max_rows if max_rows is not None else SHEET_ROW_CAP
    formulas = [0]
    root = ET.fromstring(data)
    if furniture is not None:
        furniture.update(_sheet_header_footer(root))
    rows: list[list[str]] = []
    widths: list[int] = []
    clipped: list[int] = []
    total = 0
    for row in root.iter(f"{SS}row"):
        total += 1
        if len(rows) >= cap:
            continue                    # keep counting, stop collecting
        line, width, lost = _sheet_row(row, shared, formulas)
        rows.append(line)
        widths.append(width)
        if width > MAX_RENDER_COLUMNS and lost:
            # Only when values were actually dropped. A sparse row labelled by
            # column loses nothing, and claiming otherwise is the same
            # over-report in the opposite direction.
            clipped.append(width)
    return (rows, total, max(widths, default=0), bool(clipped),
            formulas[0])


def _sheet_row(row, shared: list,
               formulas: "list | None" = None) -> "tuple[list[str], int, bool]":
    """One <row>: its cells as text, its width, and whether values were lost.

    ``formulas``, where the caller counts them, is a one-element counter it
    owns, bumped per cell that holds a formula.
    """
    cells: dict[int, str] = {}
    next_col = 0
    for c in row.findall(f"{SS}c"):
        col = _column_index(c.get("r")) if c.get("r") else next_col
        cells[col] = _cell_value(c, shared)
        if formulas is not None and c.find(f"{SS}f") is not None:
            # A cell holding a formula shows the value the last program
            # to save the file happened to cache. Nothing recalculates it
            # — not sift, not a diff tool, not a mail gateway — so a
            # SUM(B2:B3) over 1000 and 2000 can arrive reading 9, and a
            # summary states the 9. Counted so the sheet can say how many
            # of its numbers are somebody else's arithmetic.
            formulas[0] += 1
        next_col = col + 1
    # Width is measured after dropping empty cells. Formatting a whole row
    # in Excel writes a styled but valueless cell at XFD, and taking the
    # raw maximum reported a two-column sheet as 16384 columns wide.
    filled = [i for i, v in cells.items() if v != ""]
    width = max(filled) + 1 if filled else 0
    # Materialize only what anyone will read. A sheet may legitimately be
    # 16384 columns wide; building that list for every row costs memory
    # nobody spends to see twelve sample rows.
    lost = True
    if width <= MAX_RENDER_COLUMNS:
        line = [cells.get(i, "") for i in range(width)]
        lost = False
    else:
        # A row wider than the cap is sparse in practice — one note far to
        # the right. Filling the gap produced 511 tab characters and then
        # dropped the value that made the row wide. Name the columns
        # instead, so nothing is lost and nothing is invented.
        filled_cells = [(i, v) for i, v in sorted(cells.items()) if v != ""]
        line = [f"{_column_name(i)}={v}"
                for i, v in filled_cells[:MAX_RENDER_COLUMNS]]
        if len(filled_cells) <= MAX_RENDER_COLUMNS:
            lost = False
    while line and line[-1] == "":
        line.pop()
    return line, width, lost


#: The last column a spreadsheet can have is XFD. A reference is a string out
#: of an untrusted file, and "AAAAAAAAAA1" decodes to 5.6 trillion — enough to
#: turn a row into an out-of-memory kill while building the list.
MAX_COLUMN = 16384


def _column_name(index: int) -> str:
    """3 -> 'D'. The inverse of _column_index, for labelling a sparse row."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(65 + rest) + name
    return name


def _column_index(ref: str) -> int:
    """'D4' -> 3. Spreadsheet columns are base-26 with no zero digit.

    A to Z only. isalpha() also let "ß" through, whose upper case is the two
    letters "SS", and ord() refused it: one such cell reference made the
    whole workbook fail to extract.
    """
    index = 0
    for ch in ref[:8]:                  # XFD is three letters; eight is slack
        if not ("A" <= ch <= "Z" or "a" <= ch <= "z"):
            break
        index = index * 26 + (ord(ch.upper()) - 64)
        if index > MAX_COLUMN:
            return MAX_COLUMN - 1
    return min(max(0, index - 1), MAX_COLUMN - 1)


#: A shared-string index as the schema writes it (xsd:unsignedInt): ASCII
#: digits, at most ten. str.isdigit() also took "²", which int() refuses --
#: read_sheet_page raised a ValueError for such a cell past the 200 rows
#: extract() reads -- and "٣", which int() reads as 3; int() refuses more
#: than 4,300 digits. The host application
#: xlsx codec reads an index by the same rule, and locks a cell whose index
#: names no string.
_SHARED_INDEX = re.compile(r"[0-9]{1,10}")


def _cell_value(c, shared) -> str:
    t = c.get("t")
    v = c.find(f"{SS}v")
    if t == "s":  # shared string index
        if v is not None and v.text and _SHARED_INDEX.fullmatch(v.text):
            idx = int(v.text)
            if isinstance(shared, list):
                return shared[idx] if idx < len(shared) else ""
            return shared.get(idx)          # _SharedStrings, read as needed
        return ""
    if t == "inlineStr":
        return "".join(node.text or "" for node in c.iter(f"{SS}t"))
    return v.text if v is not None and v.text else ""


# ---- pptx -------------------------------------------------------------
def _pptx(zf, det, props) -> Extracted:
    slide_files = sorted(
        (n for n in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
        key=lambda n: int(re.search(r"(\d+)", n).group(1)))
    blocks, total_text, over_budget = [], 0, False
    budget, reached = _Budget(zf), []
    for i, sf in enumerate(_within_budget(zf, slide_files, budget), 1):
        if sf is None:
            over_budget = True
            break
        root = ET.fromstring(_read_member(zf, sf))
        reached.append(sf)
        texts = [t.text for t in root.iter(f"{A}t") if t.text and t.text.strip()]
        total_text += sum(len(t) for t in texts)
        blocks.append(f"## Slide {i}\n" + "\n".join(texts))
    meta = {**props, "slides": len(slide_files), "characters": total_text}
    if over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of slide XML; "
            f"later slides were not read")
    text = "# Presentation\n" + "\n\n".join(blocks)
    # The speaker notes after every slide, whose text and offsets stay as
    # they were (_PptxNotes).
    notes = _PptxNotes(zf, reached, budget, MAX_CHARS - len(text))
    if notes.sections:
        text += "\n\n" + _sections(notes.sections)
    if notes.count:
        meta["notes"] = notes.count
    # The comments after the notes, as the notes after the slides
    # (_PptxComments).
    comments = _PptxComments(zf, notes.commented, budget,
                             MAX_CHARS - len(text))
    if comments.sections:
        text += "\n\n" + _sections(comments.sections)
    if comments.count:
        meta["comments"] = comments.count
    if notes.over_budget and not over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of slide XML "
            f"and notes; later notes were not read, and no comments")
    elif comments.over_budget and not over_budget:
        meta["read_limit"] = (
            f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of slide XML, "
            f"notes and comments; later comments were not read")
    body, truncated = clip(text, MAX_CHARS)
    ex = Extracted("pptx", det.mime, text=body, metadata=meta, rendered=True,
                   truncated=truncated)
    for unread, what in ((notes.unread, "notes"),
                         (comments.unread, "comments")):
        if unread:
            ex.note(f"{unread} {what} part{'s' if unread != 1 else ''} "
                    f"could not be read")
    return ex


#: A notes slide as PowerPoint names it, by which one no slide relates is
#: found (ppt/notesSlides/notesSlide1.xml).
_PPTX_NOTES = re.compile(r"ppt/notesSlides/notesSlide([0-9]{0,9})\.xml")


class _PptxNotes:
    """A presentation's speaker notes, which _pptx read none of: a password
    in the notes under a slide was never in sift's text, so find_secrets
    never saw it and a host application, which shows that text as the file's, did not
    show it.

    A slide's notes are the notes slide its relationships name (notesSlide);
    a notes part no slide relates is read after them, in the order of its
    number (_numbered): its text is in the file. ``sections`` are written
    after every slide -- the slides' rendering stays as it was --, a
    slide's notes under "## Slide N notes", N the number its "## Slide N"
    has, in the slides' order, then the others under "## Slide notes":
    headings that begin "## Slide", sift's own (_OUR_RENDERING). A note is
    read a paragraph at a time (_paragraph_lines): its runs one line, a
    break a line of its own; a field -- the slide number on the notes page,
    a date -- is what PowerPoint put there, no note. ``count`` counts the
    notes slides that hold text.

    Bounded as the slides are: the slides' relationships and the notes go
    on ``budget``, the slides' (_Budget), the clock is looked at per member,
    reading stops once the text is past ``room``, what clip() keeps, and a
    part that is missing, damaged or no XML is counted in ``unread``.

    ``commented``: the comments parts the slides' relationships name, each
    the number of the first slide that names it -- read in the same pass,
    so a slide's relationships are read, and charged, once
    (_PptxComments)."""

    def __init__(self, zf, slides, budget: _Budget, room: int):
        self.sections, self.count, self.unread = [], 0, 0
        self.commented = {}
        self.over_budget = budget.over
        if room < 0:
            return
        names = set(zf.namelist())
        owner = {}
        for number, part in enumerate(slides, 1):
            for _rid, kind, member in _relations(zf, part, budget):
                if kind == "notesSlide" and member in names:
                    owner.setdefault(member, number)
                elif kind == "comments" and member in names:
                    self.commented.setdefault(member, number)
        for member in _numbered(names, _PPTX_NOTES):
            owner.setdefault(member, None)
        said = collections.defaultdict(list)
        for member in _within_budget(zf, list(owner), budget):
            if member is None or room < 0:
                break
            try:
                root = ET.fromstring(_read_member(zf, member))
            except _PART_FAILURES:
                self.unread += 1
                continue
            lines = _paragraph_lines(root, f"{A}p", f"{A}t",
                                     breaks=(f"{A}br",), skip=(f"{A}fld",))
            if not lines:
                continue
            self.count += 1
            said[owner[member]].extend(lines)
            room -= sum(len(line) + 1 for line in lines)
        self.over_budget = budget.over
        self.sections = [(f"Slide {number} notes", said[number])
                         for number in sorted(n for n in said if n)]
        if said.get(None):
            self.sections.append(("Slide notes", said[None]))


#: A comments part as PowerPoint names it, by which one no slide relates
#: is found: ppt/comments/comment1.xml (2007 to 2019), and Microsoft 365's
#: threads, ppt/comments/modernComment_<slide id>_<hex>.xml.
_PPTX_COMMENTS = re.compile(r"ppt/comments/comment([0-9]{0,9})\.xml")
_PPTX_MODERN_COMMENTS = re.compile(r"ppt/comments/modernComment_[^/]{1,64}"
                                   r"\.xml")


class _PptxComments:
    """A presentation's comments, which _pptx read none of: a
    password in a review comment was never in sift's text, so find_secrets
    never saw it and a host application did not show it.

    A slide's comments are the parts its relationships name (comments --
    the type's last segment, the same for the old parts and the modern
    threads; ``related``, from _PptxNotes' pass); a part no slide relates is
    read after them, the old ones in the order of their number, then the
    modern ones by name. ``sections`` come after the notes: a slide's under
    "## Slide N comments", in the slides' order, the others under "## Slide
    comments" -- sift's own headings (_OUR_RENDERING). A comment's text is
    its lines (an old comment's p:text; a modern one's paragraphs,
    _paragraph_lines), then each reply's; no author, no date. ``count``
    counts the comments and replies that hold text.

    Bounded as the notes are: on ``budget``, the slides', the clock looked
    at per member and per comment, reading stopped once the text is past
    ``room``, a part that is missing, damaged or no XML counted in
    ``unread``."""

    def __init__(self, zf, related: dict, budget: _Budget, room: int):
        self.sections, self.count, self.unread = [], 0, 0
        self.over_budget = budget.over
        if room < 0 or budget.over:
            return
        names = set(zf.namelist())
        owner = dict(related)
        for member in (_numbered(names, _PPTX_COMMENTS) + sorted(
                n for n in names if _PPTX_MODERN_COMMENTS.fullmatch(n))):
            owner.setdefault(member, None)
        said = collections.defaultdict(list)
        for member in _within_budget(zf, list(owner), budget):
            if member is None or room < 0:
                break
            try:
                root = ET.fromstring(_read_member(zf, member))
            except _PART_FAILURES:
                self.unread += 1
                continue
            for lines in _comment_threads(root):
                self.count += 1
                said[owner[member]].extend(lines)
                room -= sum(len(line) + 1 for line in lines)
        self.over_budget = budget.over
        self.sections = [(f"Slide {number} comments", said[number])
                         for number in sorted(n for n in said if n)]
        if said.get(None):
            self.sections.append(("Slide comments", said[None]))


def _local(tag) -> str:
    """An element's name without its namespace: the old comments and the
    modern ones, transitional and strict, name theirs alike."""
    return tag.rpartition("}")[2] if isinstance(tag, str) else ""


def _comment_threads(root) -> "list[list[str]]":
    """The lines of each comment and each reply of a comments part that
    hold text, in the order a reader meets them: a comment, then its
    replies (the file puts the replies first)."""
    found = []
    for count, cm in enumerate(root.iter()):
        if not count % _CHECK_EVERY:
            deadline.check("a presentation's comments")
        if _local(cm.tag) != "cm":
            continue
        # Direct children only, at the depths the schema puts them
        # (cm/text, cm/txBody, cm/replyLst/reply/txBody): a walk under each
        # comment would read a nest of them once per level.
        bodies = [child for child in cm if _local(child.tag) in ("text",
                                                                 "txBody")]
        replies = [body for group in cm if _local(group.tag) == "replyLst"
                   for reply in group if _local(reply.tag) == "reply"
                   for body in reply if _local(body.tag) == "txBody"]
        for body in bodies + replies:
            if _local(body.tag) == "text":
                lines = [line.strip() for line in (body.text or "").splitlines()
                         if line.strip()]
            else:
                lines = _paragraph_lines(body, f"{A}p", f"{A}t",
                                         breaks=(f"{A}br",),
                                         skip=(f"{A}fld",))
            if lines:
                found.append(lines)
    return found


# ---- OpenDocument -----------------------------------------------------
#: Bounds for a spreadsheet's tables, matching the OOXML side: a sheet is
#: described, not exported.
_ODS_MAX_TABLES = 32
_ODS_MAX_ROWS = 200
_ODS_MAX_COLUMNS = 512
#: table:number-columns-repeated is how ODF collapses runs of identical
#: cells, and a saved sheet routinely ends a row with "repeated 16384". The
#: repeat is honoured up to the column cap and no further; without a cap a
#: three-cell row inflates into sixteen thousand.
_ODS_MAX_REPEAT = 4096
#: The one namespace a sheet's tables, rows, cells and counts are read in.
#: A table:table-row, table:table-cell or table:table of another namespace,
#: or a number-rows-repeated of another namespace, is no part of the sheet
#: to LibreOffice 25.2 (measured with the host application oracle, 2026-09-27);
#: read by local name, such a row was one more line here, and a count
#: "foo:number-rows-repeated" repeated it, where the host application ods codec,
#: which reads the table namespace only, saw neither (a review
#: 0.2.9, finding 3).
TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
STYLE = "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}"
DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"
PRESENTATION = "{urn:oasis:names:tc:opendocument:xmlns:presentation:1.0}"


def _odf(zf, det, props) -> Extracted:
    """odt as paragraphs, ods as tables, odp as slides.

    All three were read as one flat list of paragraphs, which is right for a
    text document and loses the shape of the other two: a spreadsheet came
    back as "OpenDocument spreadsheet, ~6 words" where the xlsx beside it
    said "1 sheet: Quartal (3×2)".

    The page headers and footers after the body, whose text stays where it
    was (_OdfParts): an odt's and an ods's master pages', an odp's header
    and footer declarations.
    """
    root = ET.fromstring(_read_member(zf, "content.xml"))
    parts = _OdfParts(zf, root, det.kind)
    if det.kind == "ods":
        tabular = _ods(root, det, props, parts)
        if tabular is not None:
            return tabular
        # No tables in it. A spreadsheet whose content sits outside any
        # table:table is unusual, and reading it as nothing is worse than
        # reading it as text — losing the file's words to gain its shape is
        # not a trade a reader wants.
    paragraphs, slides = [], 0
    for node in root.iter():
        if node.tag.endswith("}page") or node.tag.endswith("}drawing-page"):
            slides += 1
            continue
        if not _odf_paragraph(node.tag):
            continue
        # A paragraph inside this one -- a footnote's, a text box's, a
        # comment's -- is a line of its own here, as root.iter() reaches it.
        line = _odf_line(node)
        if line:
            paragraphs.append(line)
    text = "\n".join(paragraphs)
    sections = [(f"Sheet {heading.lower()}" if det.kind == "ods" else heading,
                 lines) for heading, lines in parts.read(MAX_CHARS - len(text))]
    if sections:
        text += ("\n\n" if text else "") + _sections(sections)
    meta = {**props, "paragraphs": len(paragraphs), "characters": len(text),
            "words": len(text.split())}
    if slides:
        meta["slides"] = slides
    body, truncated = clip(text, MAX_CHARS)
    ex = Extracted(det.kind, det.mime, text=body, metadata=meta,
                   truncated=truncated)
    parts.say(ex)
    return ex


def _ods(root, det, props, parts: "_OdfParts") -> "Extracted | None":
    """The tables of a spreadsheet, named, with their shape.

    ``None`` when the document has none — the caller falls back to reading
    it as paragraphs, because a reader that returns nothing has not
    described the file, it has hidden it.

    After the tables, as an xlsx's are (_XlsxNotes), under headings that
    begin "## Sheet": the page header and footer (_OdfParts), then per
    table its cells' comments and shapes (_ods_aside), which are no part of
    a cell's value. The clock is looked at every _CHECK_EVERY elements;
    nothing looked at it here before.
    """
    tables, listed, blocks, seen, aside = [], [], [], 0, []
    table = f"{TABLE}table"
    for count, node in enumerate(root.iter()):
        if not count % _CHECK_EVERY:
            deadline.check("a spreadsheet's content")
        tag = node.tag
        if tag == table:
            seen += 1
            if len(tables) >= _ODS_MAX_TABLES:
                continue
            entry = _ods_table(node, len(tables), blocks)
            if entry is not None:
                tables.append(entry)
                listed.append(node)
        elif tag in _ODF_ASIDE or tag.startswith(DRAW):
            aside.append(node)

    if not tables:
        return None
    meta = {**props, "sheets": tables, "sheet_count": len(tables)}
    if seen > len(tables):
        meta["read_limit"] = (f"{seen} tables; the first {len(tables)} "
                              f"are described")
    text = "\n".join(blocks)
    after = []
    for heading, lines in parts.read(MAX_CHARS - len(text)):
        after += [f"## Sheet {heading.lower()}", *lines, ""]
    more, comments = _ods_aside(tables, listed, aside, MAX_CHARS - len(text)
                                - sum(len(line) + 1 for line in after))
    after += more
    if after:
        text += "\n" + "\n".join(after)
    if comments:
        meta["comments"] = comments
    body, truncated = clip(text, MAX_CHARS)
    ex = Extracted(det.kind, det.mime, text=body, metadata=meta,
                   rendered=True, truncated=truncated)
    parts.say(ex)
    return ex


def _ods_table(node, index: int, blocks: list) -> "dict | None":
    """The entry of the table ``node`` for the metadata, its rendering
    appended to ``blocks``; None for a table that holds no row (_ods)."""
    name = _attr(node, "name") or f"Table {index + 1}"
    rows = []
    for cells in _ods_rows(node):
        rows.append(cells)
        if len(rows) >= _ODS_MAX_ROWS:
            break
    if not rows:
        return None
    columns = max(len(r) for r in rows)
    headed = len(rows) > 1 and _rows_look_headed(rows[:9])
    entry = {"name": one_line(name),
             "rows": len(rows) - (1 if headed else 0),
             "columns": columns, "has_header": headed}
    if len(rows) >= _ODS_MAX_ROWS:
        entry["rows_read"] = _ODS_MAX_ROWS
    shown = (f"{entry['rows']} data rows" if headed
             else f"{len(rows)} rows")
    blocks.append(f"## Sheet: {one_line(name)} ({shown} × {columns} cols)")
    for row in rows[:12]:
        blocks.append("\t".join(row))
    blocks.append("")
    return entry


def _ods_aside(tables: list, listed: list, aside: list,
               room: int) -> "tuple[list[str], int]":
    """(lines, comments): what an ods's cells hold beside their values --
    comments (office:annotation) and shapes (the drawing namespace), each
    with its paragraphs -- as the lines after its sheets, and how many
    comments hold text.

    A host application's ods codec leaves both out of a cell's value
    (BESIDE_THE_VALUE, and anything of the drawing namespace), and so does
    LibreOffice; read into it, a comment was glued to its author, its date
    and the cell's text -- "A2026-09-29T10:00:00password: Kx9v...TzItem" --,
    find_secrets learned the secret with the cell's value on its end or
    found nothing at all, and a model's sentence that repeated the bare
    secret was published unhidden.

    ``aside`` holds them in document order, ``listed`` the table elements of
    ``tables``, the entries _ods lists. Per table in their order its
    comments under "## Sheet comments: X" and its shapes under "## Sheet
    shapes: X", X the innermost listed table one lies in; then those of none
    under "## Sheet comments" and "## Sheet shapes". Each paragraph is a
    line (_odf_lines), the author and the date no part of it; one inside
    another taken -- a comment in a shape -- is among that one's lines.
    Reading stops once the lines are past ``room``. Which table holds one
    is looked up only for a file that holds one: a walk of each listed
    table. The clock is looked at every _CHECK_EVERY elements of those
    walks and of ``aside``."""
    if not aside:
        return [], 0
    wanted = {id(node): at for at, node in enumerate(aside)}
    owners = [None] * len(aside)
    for index, element in enumerate(listed):        # an inner one after
        for count, node in enumerate(element.iter()):
            if not count % _CHECK_EVERY:
                deadline.check("a spreadsheet's comments")
            at = wanted.get(id(node))
            if at is not None:
                owners[at] = index
    said, covered, comments = collections.defaultdict(list), set(), 0
    for count, (owner, node) in enumerate(zip(owners, aside)):
        if not count % _CHECK_EVERY:
            deadline.check("a spreadsheet's comments")
        if room < 0:
            break
        if id(node) in covered:
            continue
        covered.update(id(inner) for inner in node.iter())
        comment = node.tag in _ODF_ASIDE
        lines = []
        for line in _odf_lines(node):
            lines.append(line)
            room -= len(line) + 1
            if room < 0:
                break
        if lines:
            comments += comment
            said[owner, comment].extend(lines)
    out = []
    for owner in [*range(len(tables)), None]:
        name = "" if owner is None else f": {tables[owner]['name']}"
        for comment, heading in ((True, "comments"), (False, "shapes")):
            lines = said.get((owner, comment))
            if lines:
                out += [f"## Sheet {heading}{name}", *lines, ""]
    return out, comments


#: The elements a sheet's rows stand in besides the table itself, in the
#: table namespace only: header rows, a rows block and row groups, nested
#: in each other too. LibreOffice shows their rows in document order, and
#: the host application ods codec numbers them so (ods.ROW_GROUPS, _rows); read
#: as nothing, a header row "Name, Qty" was missing from the page and its
#: cells were never checked.
_ODS_ROW_GROUPS = frozenset(f"{TABLE}{name}" for name in (
    "table-header-rows", "table-rows", "table-row-group"))


def _ods_row_elements(table):
    """The table:table-row elements of ``table`` in document order: its own
    and those inside its row groups (_ODS_ROW_GROUPS), walked with a stack
    -- groups nest as deep as the XML does, past Python's recursion limit.
    Anything else, a nested table among it, holds no row of this one."""
    stack = [iter(table)]
    while stack:
        for child in stack[-1]:
            if _ods_is(child, "table-row"):
                yield child
            elif child.tag in _ODS_ROW_GROUPS:
                stack.append(iter(child))
                break
        else:
            stack.pop()


def _ods_rows(table, on_cut=None):
    """A table's rows as lists of cell text, repeats expanded.

    The rows are the table's own and those of its row groups
    (_ods_row_elements). A repeated row is expanded at most _ODS_MAX_ROWS
    times, so one element of the file cannot become a million rows.
    ``on_cut(times)`` is called when a row with content is repeated more
    often than that: the rows counted after it are then not all the rows
    there are.
    """
    for row in _ods_row_elements(table):
        wanted = _int(_attr(row, "number-rows-repeated"), 1)
        repeat = min(wanted, _ODS_MAX_ROWS)
        cells = _ods_row(row)
        if not any(c.strip() for c in cells):
            continue                    # a run of empty rows says nothing
        if wanted > repeat and on_cut is not None:
            on_cut(wanted)
        for _ in range(repeat):
            yield cells


def _ods_blank(row) -> bool:
    """Whether _ods_row(row) holds no cell that is not white space -- a row
    _ods_rows passes over -- found without making its cells: a cell's text
    is white space exactly when each piece of it is, and _ods_row keeps the
    first _ODS_MAX_COLUMNS columns (a test holds the two alike). A cell
    whose every piece of text is white space is so without what it holds
    beside its value too; one that is not is read as its value is
    (_odt_text) -- a cell holding only a comment is empty."""
    columns = 0
    for cell in row:
        if not _ods_is(cell, "table-cell") and not _ods_is(
                cell, "covered-table-cell"):
            continue
        if any(text and not text.isspace() for text in cell.itertext()) \
                and _odt_text(cell).strip():
            return False
        columns += min(_int(_attr(cell, "number-columns-repeated"), 1),
                       _ODS_MAX_REPEAT)
        if columns >= _ODS_MAX_COLUMNS:
            return True
    return True


def _ods_row(row) -> list:
    cells = []
    for cell in row:
        if not _ods_is(cell, "table-cell") and not _ods_is(
                cell, "covered-table-cell"):
            continue
        value = " ".join(_odt_text(cell).split())
        repeat = min(_int(_attr(cell, "number-columns-repeated"), 1),
                     _ODS_MAX_REPEAT)
        for _ in range(repeat):
            cells.append(value)
            if len(cells) >= _ODS_MAX_COLUMNS:
                return cells
    while cells and not cells[-1].strip():
        cells.pop()                     # the trailing run ODF pads rows with
    return cells


def _ods_is(node, name: str) -> bool:
    """Whether ``node`` is the table namespace's element ``name`` (TABLE)."""
    return node.tag == f"{TABLE}{name}"


def _attr(node, name: str) -> str:
    """A table-namespace attribute (TABLE), "" without it."""
    return node.get(f"{TABLE}{name}", "")


#: A count as LibreOffice 25.2 reads number-rows-repeated and
#: number-columns-repeated (measured with the host application oracle, 2026-09-27):
#: XML white space, an optional "+", then the ASCII digits up to the first
#: other character -- "1_0" is 1, "3x" 3, " 3 " and "+3" 3. int() read
#: "1_0" as 10 and "٣" as 3.
_COUNT = re.compile(r"\+?([0-9]+)")
#: The XML white space LibreOffice skips before a count. Stripped before
#: _COUNT matches: "[ \t\n\r]*" ahead of the digits made a search of a
#: long run of white space quadratic (test_backtracking: 9.2 s).
_XML_SPACE = " \t\n\r"
#: The most digits a count is read with: more is no count LibreOffice reads
#: (eleven nines were one row), and int() refuses more than 4,300.
_COUNT_DIGITS = 10


def _int(value: str, default: int) -> int:
    """The count ``value`` holds (_COUNT); ``default`` for none, for 0 and
    for more than _COUNT_DIGITS digits -- "²", "٣", "-2", "0" and eleven
    nines were one row to LibreOffice."""
    found = _COUNT.match((value or "").lstrip(_XML_SPACE))
    if found is None or len(found.group(1)) > _COUNT_DIGITS:
        return default
    return int(found.group(1)) or default


def _odt_text(node) -> str:
    """A cell's text as its value: all text in it, including what follows
    inline spans, its paragraphs apart, and nothing of what it holds beside
    its value -- a comment, a shape (_odf_text).

    Reading only ``node.text`` drops every XML *tail* — the text after a
    </text:span>. "Hello <b>brave</b> new world" came out as "Hello brave",
    losing " new world" from any document with bold, italics or a link.
    """
    return _odf_text(node, drawings=True)


#: What a paragraph or a cell holds beside its own text: a comment, with
#: its author (dc:creator), its date and its paragraphs. Read into it, the
#: comment was glued to the words around it (_ods_aside).
_ODF_ASIDE = frozenset({f"{OFFICE}annotation"})
#: The white space ODF writes as an element -- a run of spaces (text:s), a
#: tab, a line break -- holds no text: read as nothing, the words either
#: side were one ("password:<text:tab/>Kx9v...<text:line-break/>User" was
#: learned as "Kx9v...User: bob", and the bare secret published unhidden).
_ODF_WHITE = {f"{TEXT}s": " ", f"{TEXT}tab": "\t", f"{TEXT}line-break": "\n"}
#: What _odf_text writes for a line break where the caller keeps it a
#: line break (_odf_line): NUL, which no XML text holds, and so no other
#: piece -- the text's own new lines are white space it folds.
_BREAK = "\x00"
_LINE_BREAK = f"{TEXT}line-break"


def _odf_paragraph(tag: str) -> bool:
    """Whether ``tag`` is a paragraph or a heading as _odf has always read
    them: a p or an h of any namespace."""
    return tag.endswith("}p") or tag.endswith("}h")


def _odf_text(node, lines: bool = False, drawings: bool = False,
              drop=frozenset(), breaks: bool = False) -> str:
    """The text of ``node`` in document order -- its own, then each child's
    and each child's tail -- as a reader shows it: a text:s, text:tab or
    text:line-break the white space it stands for (with ``breaks`` a line
    break _BREAK), and nothing but the tail of a comment (_ODF_ASIDE), of an
    element ``drop`` names (a header's fields, _ODF_FIELDS) and, with
    ``drawings``, of one of the drawing namespace. A paragraph inside
    ``node`` is, with ``lines``, no part of it (_odf_line: it is a line of
    its own), and else the white space between two lines (a cell's
    paragraphs, which its value folds into one).

    Walked with a stack, not recursively: spans nest as deep as the XML
    does, and a paragraph of 50,000 nested spans lost the document to
    "nesting too deep". Whatever it adds is white space the callers fold."""
    out = [node.text or ""]
    stack = [(iter(node), None)]
    while stack:
        children, holder = stack[-1]
        for child in children:
            tag = child.tag
            if tag in _ODF_ASIDE or tag in drop or (
                    drawings and tag.startswith(DRAW)):
                if child.tail:
                    out.append(child.tail)
                continue
            if tag.endswith("}p") or tag.endswith("}h"):   # _odf_paragraph
                if lines:
                    if child.tail:
                        out.append(child.tail)
                    continue
                out.append("\n")
            elif tag in _ODF_WHITE:
                out.append(_BREAK if breaks and tag == _LINE_BREAK
                           else _ODF_WHITE[tag])
            if child.text:
                out.append(child.text)
            if len(child):
                stack.append((iter(child), child))
                break
            if child.tail:
                out.append(child.tail)
        else:
            stack.pop()
            if holder is not None and holder.tail:
                out.append(holder.tail)
    return "".join(out)


def _odf_line(para, drop=frozenset()) -> str:
    """A paragraph's or a heading's text, its white space folded, a line
    break (text:line-break) a line break, as a docx's w:br is
    (_docx_paragraph), blank lines left out (_odf_text). A paragraph
    inside it is a line of its own, as root.iter() reaches it. Folded into
    the one line, the words after a break were the end of a password's
    value: "password: Kx9v... User bob" (find_secrets reads a value to the
    end of its line)."""
    text = _odf_text(para, lines=True, drop=drop, breaks=True)
    return "\n".join(line for line in (" ".join(part.split())
                                       for part in text.split(_BREAK)) if line)


def _odf_lines(node, drop=frozenset()):
    """The lines of every paragraph and heading in ``node``, itself among
    them, in document order, blank ones left out (_odf_line), a line at a
    time as they are asked for; the clock looked at every 1024
    paragraphs."""
    count = 0
    for para in node.iter():
        if not _odf_paragraph(para.tag):
            continue
        count += 1
        if not count % 1024:
            deadline.check("a document's parts")
        line = _odf_line(para, drop)
        if line:
            yield from line.split("\n")


#: The regions of a master page (styles.xml) that print on its pages, by
#: the heading their lines are written under.
_ODF_REGIONS = {f"{STYLE}{name}": name.split("-")[0].title() for name in (
    "header", "header-left", "header-first", "footer", "footer-left",
    "footer-first")}
#: The fields a header or a footer holds: what the last layout put there,
#: no text of the file's. LibreOffice writes "???" for the sheet's name and
#: the title, "00/00/0000" for the date and "99" for the page count into
#: every spreadsheet it saves (6 of 6 ods in a census); in a text document
#: they hold the page, the date, the file's name or the chapter of the
#: last layout.
_ODF_FIELDS = frozenset(f"{TEXT}{name}" for name in (
    "page-number", "page-count", "page-continuation", "sheet-name", "title",
    "subject", "description", "keywords", "date", "time", "file-name",
    "template-name", "chapter", "author-name", "author-initials", "creator",
    "initial-creator", "creation-date", "creation-time", "modification-date",
    "modification-time", "print-date", "print-time", "printed-by",
    "editing-cycles", "editing-duration", "page-variable-get",
    "character-count", "word-count", "paragraph-count", "table-count",
    "image-count", "object-count"))


class _OdfParts:
    """An OpenDocument file's page headers and footers, which _odf read none
    of: a password in them was never in sift's text, so find_secrets never
    saw it and a host application, which shows that text as the file's, did not show
    it.

    An odt's and an ods's are the regions of the master pages in styles.xml
    (style:header, style:footer and their -left and -first pages), every
    master page in its order, used by a page or not: their text is in the
    file. An odp's are its header and footer declarations
    (presentation:header-decl, presentation:footer-decl in content.xml);
    its master pages hold the layout's placeholder text, not the file's.

    ``read(room)`` gives (heading, lines): "Header", then "Footer", each
    written after the body under its heading -- "## Header", "## Sheet
    header" --, the body's text and offsets as they were. A paragraph is a
    line (_odf_line), its fields dropped (_ODF_FIELDS); a line without a
    letter or a digit is what separated its fields and is dropped as an
    xlsx's is (_A_WORD); a region or a declaration that says word for word
    what one before it said -- the same header on every master page -- is
    written once.

    Bounded as a docx's parts are: styles.xml read through _read_member,
    what content.xml and it declare together within _MAX_TOTAL_INFLATED
    (_Budget), the clock looked at per member, per region and every 1024
    paragraphs, reading stopped once the lines are past ``room``, what
    clip() keeps. A
    styles.xml that is damaged or no XML is said in a note (``say``); the
    body is read regardless."""

    def __init__(self, zf, root, kind: str):
        self._zf, self._root, self._kind = zf, root, kind
        self.unread = self.over_budget = False

    def read(self, room: int) -> "list[tuple[str, list[str]]]":
        found = {"Header": [], "Footer": []}
        seen = set()
        for heading, lines in self._regions():
            # Per region too: a million regions or declarations that say the
            # same, or nothing, add no line and read no member.
            deadline.check("a document's parts")
            kept, left = [], room
            for line in lines:
                if left < 0:
                    break
                if _A_WORD.search(line):
                    kept.append(line)
                    left -= len(line) + 1
            said = (heading, tuple(kept))
            if kept and said not in seen:
                seen.add(said)
                found[heading] += kept
                room = left
            if room < 0:
                break
        return [(heading, lines) for heading, lines in found.items() if lines]

    def _regions(self):
        """(heading, lines) of each region or declaration in the file's
        order, the lines read as they are asked for."""
        if self._kind == "odp":
            for heading in ("Header", "Footer"):
                tag = f"{PRESENTATION}{heading.lower()}-decl"
                for decl in self._root.iter(tag):
                    yield heading, _odf_line(decl).split("\n")
            return
        budget = _Budget(self._zf, _declared(self._zf, "content.xml"))
        if not budget.take("styles.xml"):
            self.over_budget = budget.over
            return
        try:
            styles = ET.fromstring(_read_member(self._zf, "styles.xml"))
        except _PART_FAILURES:
            self.unread = True
            return
        for master in styles.iter(f"{STYLE}master-page"):
            for region in master:
                heading = _ODF_REGIONS.get(region.tag)
                if heading is not None:
                    yield heading, _odf_lines(region, drop=_ODF_FIELDS)

    def say(self, ex: Extracted) -> None:
        """Tell ``ex`` what was not read."""
        if self.over_budget:
            limit = (f"stopped after {human_size(_MAX_TOTAL_INFLATED)} of "
                     f"the document's XML; the page headers and footers in "
                     f"styles.xml were not read")
            said = ex.metadata.get("read_limit")
            ex.metadata["read_limit"] = f"{said}; {limit}" if said else limit
        if self.unread:
            ex.note("styles.xml could not be read: the page headers and "
                    "footers are not in the text")


# ---- shared -----------------------------------------------------------
def _core_props(zf) -> dict:
    if "docProps/core.xml" not in zf.namelist():
        return {}
    try:
        root = ET.fromstring(_read_member(zf, "docProps/core.xml"))
    except ET.ParseError:
        return {}
    fields = {
        "title": f"{DC}title", "creator": f"{DC}creator", "subject": f"{DC}subject",
        "created": f"{CORE}created", "modified": f"{CORE}modified",
    }
    out = {}
    for label, tag in fields.items():
        node = root.find(tag)
        if node is not None and node.text:
            out[label] = node.text.strip()
    return out
