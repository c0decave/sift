"""The classic binary Office formats -- .xls, .doc and .ppt -- read for their text.

A compound file (``D0 CF 11 E0``) holds one of three streams at its root:
``Workbook`` (``Book`` before Excel 97) for a spreadsheet, ``WordDocument``
for a Word document, ``PowerPoint Document`` for a presentation. sift calls
every one of them ``office-legacy`` and, up to 0.1.90, read none: a .xls full
of cells was a hex dump with a size. This reads the text out of them with the
standard library alone and the compound reader sift already trusts for its
active-content scan (``active._Cfb``: every pointer bounded by the file's own
sectors).

What is read, and what is deliberately not:

- .xls: BIFF8 (Excel 97-2003) and BIFF5/7 (Excel 5-95) worksheets -- every
  cell's cached value (a formula is not recalculated), shared strings,
  dates by their cell format, the text of cell notes and text boxes. Not
  read: charts, macro sheets, pivot tables, a workbook that is encrypted.
- .doc: Word 97 and later through the piece table, Word 6/95 as one run
  of text; the body, footnotes, endnotes, comments, text boxes and headers.
  Field codes are dropped and a field's shown result kept. Not read:
  formatting, pictures, an encrypted document.
- .ppt: the text atoms of the slides and the notes. Not read: masters,
  embedded objects, pictures.

Nothing here executes or follows anything: a macro, an embedded object or a
link is not looked at (the active-content scan reports those), and every
size, count and chain in a file is bounded before it is used.
"""
from __future__ import annotations

import datetime
import decimal
import re
import struct
from pathlib import Path

from .. import deadline
from ..detect import Detection
from . import fallback, office
from .active import Found, _Cfb, _Cut, _Damaged
from .base import Extracted, TablePage, clip, human_size, one_line

#: A compound file larger than this is not read: the whole of it is held.
MAX_FILE = 64 * 1024 * 1024
#: Sheets, and cells' worth of text a workbook's shared strings may hold.
MAX_SHEETS = 255
#: Characters of a document or a presentation kept before it is cut.
MAX_TEXT = 4_000_000
#: Text pieces of a Word document's piece table.
MAX_PIECES = 500_000
#: Records of a PowerPoint stream.
MAX_PPT_RECORDS = 2_000_000
#: Rows of the rendering of a sheet (the page reader has its own bounds).
ROWS_SHOWN = 12

DOC, XLS, PPT = "doc", "xls", "ppt"
_STREAMS = (("WordDocument", DOC), ("Workbook", XLS), ("Book", XLS),
            ("PowerPoint Document", PPT))

NO_TABLE = office.NO_TABLE


class Unreadable(Exception):
    """The file is a compound file this module cannot read; the words are
    for a note or a page's ``error``."""


class Encrypted(Unreadable):
    pass


# ---- the compound file ----------------------------------------------------

def _compound(path) -> _Cfb:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_FILE:
        raise Unreadable(f"a compound file larger than {human_size(MAX_FILE)}"
                         f" is not read")
    data = path.read_bytes()
    try:
        return _Cfb(data, Found())
    except (_Damaged, struct.error, IndexError, ValueError, OverflowError):
        raise Unreadable("the compound file's structure cannot be followed")
    except _Cut as exc:
        raise Unreadable(str(exc))


def _root_streams(cfb: _Cfb) -> dict:
    names = {}
    for index in cfb.children(0):
        name, kind = cfb.entries[index][0], cfb.entries[index][1]
        if kind == 2 and name not in names:
            names[name] = index
    return names


def _read(cfb: _Cfb, index: int) -> bytes:
    try:
        return cfb.stream(index)
    except (_Damaged, struct.error, IndexError, ValueError, OverflowError):
        raise Unreadable("a stream of the compound file cannot be followed")
    except _Cut as exc:
        raise Unreadable(str(exc))


def format_of(cfb: _Cfb) -> "str | None":
    """DOC, XLS or PPT by the stream the root holds; None for any other
    compound file (a .msg, a Visio drawing, an unknown storage)."""
    streams = _root_streams(cfb)
    for name, kind in _STREAMS:
        if name in streams:
            return kind
    return None


def stream_kind(path) -> "str | None":
    """DOC, XLS, PPT, or None where ``path`` is not one of them (or not a
    compound file sift can follow)."""
    try:
        return format_of(_compound(path))
    except (Unreadable, OSError):
        return None


# ---- byte helpers ---------------------------------------------------------

def _u16(data: bytes, at: int) -> int:
    return struct.unpack_from("<H", data, at)[0]


def _u32(data: bytes, at: int) -> int:
    return struct.unpack_from("<I", data, at)[0]


def _codec(codepage: int) -> str:
    if codepage == 10000:
        return "mac_roman"
    if codepage == 65001:
        return "utf-8"
    if codepage in (1200, 1201):
        return "utf-16-le" if codepage == 1200 else "utf-16-be"
    return f"cp{codepage}"


def _decode(raw: bytes, codec: str) -> str:
    try:
        return raw.decode(codec, "replace")
    except LookupError:
        return raw.decode("latin-1")


# ---- BIFF records ---------------------------------------------------------

_BOF, _EOF, _CONTINUE = 0x0809, 0x000A, 0x003C
_FILEPASS, _CODEPAGE, _DATEMODE = 0x002F, 0x0042, 0x0022
_BOUNDSHEET, _SST, _FORMAT, _XF = 0x0085, 0x00FC, 0x041E, 0x00E0
_LABELSST, _LABEL, _RSTRING, _NUMBER = 0x00FD, 0x0204, 0x00D6, 0x0203
_RK, _MULRK, _BOOLERR, _FORMULA, _STRING = (0x027E, 0x00BD, 0x0205, 0x0006,
                                            0x0207)
_TXO = 0x01B6
_CELL_RECORDS = frozenset({_LABELSST, _LABEL, _RSTRING, _NUMBER, _RK, _MULRK,
                           _BOOLERR, _FORMULA})

_ERRORS = {0x00: "#NULL!", 0x07: "#DIV/0!", 0x0F: "#VALUE!", 0x17: "#REF!",
           0x1D: "#NAME?", 0x24: "#NUM!", 0x2A: "#N/A"}
#: Built-in number formats that show a date or a time.
_DATE_FORMATS = frozenset(range(14, 23)) | frozenset({45, 46, 47})
_QUOTED = re.compile(r'"[^"]*"|\\.|\[[^\]]*\]|_.|\*.')
_DATE_TOKEN = re.compile(r"[ymdhsYMDHS]|AM/PM|A/P", re.I)


def _records(data: bytes, pos: int = 0):
    """(record id, [segments]) from ``pos``: the bytes of every CONTINUE that
    follows a record are its further segments, in order."""
    size = len(data)
    last = None
    count = 0
    while pos + 4 <= size:
        rid, length = struct.unpack_from("<HH", data, pos)
        pos += 4
        segment = data[pos:pos + length]
        pos += length
        count += 1
        if count % 8192 == 0:
            deadline.check("a legacy Excel workbook")
        if rid == _CONTINUE and last is not None:
            last[1].append(segment)
            continue
        if last is not None:
            yield last[0], last[1]
        last = [rid, [segment]]
    if last is not None:
        yield last[0], last[1]


class _Segments:
    """Fields read across the segments of one record (CONTINUE)."""

    def __init__(self, segments):
        self.segments = segments
        self.index = 0
        self.at = 0

    def _fix(self) -> bool:
        while (self.index < len(self.segments)
               and self.at >= len(self.segments[self.index])):
            self.index += 1
            self.at = 0
        return self.index < len(self.segments)

    def empty(self) -> bool:
        return not self._fix()

    def take(self, count: int) -> bytes:
        out = bytearray()
        while count > 0 and self._fix():
            segment = self.segments[self.index]
            piece = segment[self.at:self.at + count]
            out += piece
            self.at += len(piece)
            count -= len(piece)
        if count > 0:
            raise Unreadable("a record ends inside a string")
        return bytes(out)

    def skip(self, count: int) -> None:
        while count > 0 and self._fix():
            have = len(self.segments[self.index]) - self.at
            step = min(have, count)
            self.at += step
            count -= step

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def characters(self, count: int, high: bool) -> str:
        """``count`` characters: a CONTINUE that begins inside them starts
        with one byte, the flag of the rest of the characters."""
        out = []
        started = False
        while count > 0:
            if not self._fix():
                raise Unreadable("a record ends inside a string")
            if started and self.at == 0:
                high = bool(self.segments[self.index][0] & 1)
                self.at = 1
                if not self._fix():
                    raise Unreadable("a record ends inside a string")
            started = True
            have = len(self.segments[self.index]) - self.at
            width = 2 if high else 1
            n = min(count, have // width)
            if n == 0:
                raise Unreadable("a character is split between records")
            raw = self.take(n * width)
            out.append(raw.decode("utf-16-le", "replace") if high
                       else raw.decode("latin-1"))
            count -= n
        return "".join(out)


def _unicode_string(reader: _Segments, wide_length: bool = True) -> str:
    """An XLUnicodeString (BIFF8): a 16-bit length (8-bit for the short
    form), a flag byte, the characters."""
    cch = reader.u16() if wide_length else reader.u8()
    flags = reader.u8()
    high = bool(flags & 1)
    if flags & 8:
        runs = reader.u16()
    else:
        runs = 0
    ext = reader.u32() if flags & 4 else 0
    text = reader.characters(cch, high)
    reader.skip(runs * 4 + ext)
    return text


def _number_text(value: float) -> str:
    """A number as Excel's General format shows it: fifteen significant
    digits, no exponent between 1e-9 and 1e15."""
    if value != value or value in (float("inf"), float("-inf")):
        return "#NUM!"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    text = f"{value:.15g}"
    if "e" in text and 1e-9 <= abs(value) < 1e15:
        text = format(decimal.Decimal(text), "f")
    return text.replace("e", "E")


def _rk(raw: int) -> float:
    if raw & 2:
        value = float(struct.unpack("<i", struct.pack("<I", raw))[0] >> 2)
    else:
        value = struct.unpack("<d", struct.pack("<Q",
                                                (raw & 0xFFFFFFFC) << 32))[0]
    return value / 100 if raw & 1 else value


def _date_text(value: float, date1904: bool) -> str:
    """A serial date as ISO text; the number where it names no date."""
    if not 0 <= value < 2958466:
        return _number_text(value)
    days = int(value)
    seconds = round((value - days) * 86400)
    if seconds >= 86400:
        days, seconds = days + 1, seconds - 86400
    clock = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    if date1904:
        day = datetime.date(1904, 1, 1) + datetime.timedelta(days=days)
        text = day.isoformat()
    elif days == 0:
        return clock
    elif days == 60:
        text = "1900-02-29"
    elif days < 60:
        text = (datetime.date(1899, 12, 31)
                + datetime.timedelta(days=days)).isoformat()
    else:
        text = (datetime.date(1899, 12, 30)
                + datetime.timedelta(days=days)).isoformat()
    return text if seconds == 0 else f"{text} {clock}"


def _is_date_code(code: str) -> bool:
    return bool(_DATE_TOKEN.search(_QUOTED.sub("", code)))


class _Sheet:
    __slots__ = ("name", "offset", "state", "kind")

    def __init__(self, name, offset, state, kind):
        self.name, self.offset, self.state, self.kind = (name, offset, state,
                                                         kind)

    @property
    def is_table(self) -> bool:
        return self.kind == 0

    @property
    def hidden(self) -> bool:
        return self.state in (1, 2)


class Workbook:
    """The globals of a BIFF workbook stream and what reads its sheets."""

    def __init__(self, data: bytes):
        self.data = data
        self.biff = 8
        self.codec = "cp1252"
        self.date1904 = False
        self.sheets: list = []
        self.strings: list = []
        self.formats: dict = {}
        self.xf: list = []
        self._date_xf: dict = {}
        self.notes: list = []
        self._globals()

    # -- globals ------------------------------------------------------------

    def _globals(self) -> None:
        data = self.data
        if len(data) >= 4 and _u16(data, 0) in (0x0009, 0x0209, 0x0409):
            raise Unreadable("a BIFF2-4 workbook is not read")
        if len(data) < 8 or _u16(data, 0) != _BOF:
            raise Unreadable("the workbook stream does not begin with a BOF "
                             "record")
        version = _u16(data, 4)
        if version < 0x0500:
            raise Unreadable("a BIFF2-4 workbook is not read")
        self.biff = 8 if version >= 0x0600 else 5
        depth = 0
        for rid, segments in _records(data):
            head = segments[0]
            if rid == _BOF:
                depth += 1
                continue
            if rid == _EOF:
                depth -= 1
                if depth <= 0:
                    break
                continue
            if depth != 1:
                continue
            if rid == _FILEPASS:
                raise Encrypted("the workbook is encrypted")
            if rid == _CODEPAGE and len(head) >= 2:
                self.codec = _codec(_u16(head, 0))
            elif rid == _DATEMODE and len(head) >= 2:
                self.date1904 = _u16(head, 0) == 1
            elif rid == _BOUNDSHEET:
                self._boundsheet(head)
            elif rid == _SST:
                self._sst(segments)
            elif rid == _FORMAT:
                self._format(segments)
            elif rid == _XF and len(head) >= 4:
                self.xf.append(_u16(head, 2))
        if not self.sheets:
            raise Unreadable("the workbook lists no sheet")

    def _boundsheet(self, head: bytes) -> None:
        if len(head) < 8 or len(self.sheets) >= MAX_SHEETS:
            return
        offset, state, kind = struct.unpack_from("<IBB", head, 0)
        cch = head[6]
        if self.biff >= 8:
            flags = head[7] if len(head) > 7 else 0
            raw = head[8:8 + cch * (2 if flags & 1 else 1)]
            name = (raw.decode("utf-16-le", "replace") if flags & 1
                    else raw.decode("latin-1"))
        else:
            name = _decode(head[7:7 + cch], self.codec)
        self.sheets.append(_Sheet(one_line(name), offset, state & 3,
                                  kind))

    def _sst(self, segments) -> None:
        reader = _Segments(segments)
        try:
            reader.take(8)
            while not reader.empty():
                self.strings.append(_unicode_string(reader))
        except Unreadable:
            # The strings read stand; the rest of the table is damaged.
            pass

    def _format(self, segments) -> None:
        try:
            reader = _Segments(segments)
            index = reader.u16()
            if self.biff >= 8:
                code = _unicode_string(reader)
            else:
                cch = reader.u8()
                code = _decode(reader.take(cch), self.codec)
        except Unreadable:
            return
        self.formats[index] = code

    def is_date(self, ixfe: int) -> bool:
        cached = self._date_xf.get(ixfe)
        if cached is None:
            fmt = self.xf[ixfe] if 0 <= ixfe < len(self.xf) else 0
            if fmt in self.formats:
                cached = _is_date_code(self.formats[fmt])
            else:
                cached = fmt in _DATE_FORMATS
            self._date_xf[ixfe] = cached
        return cached

    # -- cells --------------------------------------------------------------

    def _shown(self, value: float, ixfe: int) -> str:
        if self.is_date(ixfe):
            return _date_text(value, self.date1904)
        return _number_text(value)

    def cells(self, sheet: _Sheet, counts: "dict | None" = None,
              collect: bool = True):
        """(row, column, text) of every cell of ``sheet`` in file order. A
        cell with a formula shows the value the program that saved the file
        cached (``counts["formulas"]`` counts them); with ``collect`` the
        text of notes and text boxes is added to ``self.notes``."""
        data = self.data
        if sheet.offset >= len(data) or _u16(data, sheet.offset) != _BOF:
            raise Unreadable(f"the sheet {sheet.name!r} does not begin with a "
                             f"BOF record")
        depth = 0
        waiting = None            # a formula whose string a STRING record gives
        for rid, segments in _records(data, sheet.offset):
            head = segments[0]
            if rid == _BOF:
                depth += 1
                continue
            if rid == _EOF:
                depth -= 1
                if depth <= 0:
                    return
                continue
            if depth != 1:
                continue
            try:
                if rid == _LABELSST and len(head) >= 10:
                    row, col, _x, index = struct.unpack_from("<HHHI", head, 0)
                    if index < len(self.strings):
                        yield row, col, self.strings[index]
                elif rid in (_LABEL, _RSTRING) and len(head) >= 8:
                    row, col, _x = struct.unpack_from("<HHH", head, 0)
                    reader = _Segments(segments)
                    reader.skip(6)
                    if self.biff >= 8 and rid == _LABEL:
                        yield row, col, _unicode_string(reader)
                    else:
                        cch = reader.u16()
                        yield row, col, _decode(reader.take(cch), self.codec)
                elif rid == _NUMBER and len(head) >= 14:
                    row, col, ixfe = struct.unpack_from("<HHH", head, 0)
                    value = struct.unpack_from("<d", head, 6)[0]
                    yield row, col, self._shown(value, ixfe)
                elif rid == _RK and len(head) >= 10:
                    row, col, ixfe, raw = struct.unpack_from("<HHHI", head, 0)
                    yield row, col, self._shown(_rk(raw), ixfe)
                elif rid == _MULRK and len(head) >= 6:
                    row, first = struct.unpack_from("<HH", head, 0)
                    count = (len(head) - 6) // 6
                    for k in range(count):
                        ixfe, raw = struct.unpack_from("<HI", head, 4 + k * 6)
                        yield row, first + k, self._shown(_rk(raw), ixfe)
                elif rid == _BOOLERR and len(head) >= 8:
                    row, col, _x, value, is_error = struct.unpack_from(
                        "<HHHBB", head, 0)
                    yield row, col, (_ERRORS.get(value, "#ERR!") if is_error
                                     else ("TRUE" if value else "FALSE"))
                elif rid == _FORMULA and len(head) >= 14:
                    if counts is not None:
                        counts["formulas"] = counts.get("formulas", 0) + 1
                    row, col, ixfe = struct.unpack_from("<HHH", head, 0)
                    result = head[6:14]
                    if result[6:8] == b"\xff\xff":
                        kind = result[0]
                        if kind == 0:
                            waiting = (row, col)
                        elif kind == 1:
                            yield row, col, "TRUE" if result[2] else "FALSE"
                        elif kind == 2:
                            yield row, col, _ERRORS.get(result[2], "#ERR!")
                    else:
                        value = struct.unpack("<d", result)[0]
                        yield row, col, self._shown(value, ixfe)
                elif rid == _STRING and waiting is not None:
                    reader = _Segments(segments)
                    if self.biff >= 8:
                        text = _unicode_string(reader)
                    else:
                        text = _decode(reader.take(reader.u16()), self.codec)
                    yield waiting[0], waiting[1], text
                    waiting = None
                elif rid == _TXO and collect and len(segments) > 1:
                    self._note(head, segments[1:])
            except (Unreadable, struct.error):
                continue

    def _note(self, head: bytes, rest) -> None:
        if len(head) < 12 or len(self.notes) >= 100_000:
            return
        cch = _u16(head, 10)
        if cch == 0 or not rest or not rest[0]:
            return
        reader = _Segments([rest[0]])
        try:
            flag = reader.u8()
            text = reader.characters(cch, bool(flag & 1))
        except Unreadable:
            return
        if text.strip():
            self.notes.append(one_line(text))

    def ordered(self, sheet: _Sheet) -> bool:
        """True where the cells of ``sheet`` come in row order, so its rows
        can be handed out as they are read."""
        last = -1
        for row, _col, _text in self.cells(sheet, collect=False):
            if row < last:
                return False
            last = row
        return True

    def rows(self, sheet: _Sheet, counts: "dict | None" = None):
        """The rows of ``sheet`` as lists of text, blank rows between two
        rows with cells included, columns past the last cell left out."""
        if self.ordered(sheet):
            yield from self._ordered_rows(sheet, counts)
            return
        collected: dict = {}
        total = 0
        for row, col, text in self.cells(sheet, counts):
            total += 1
            if total > MAX_CELLS:
                break
            collected.setdefault(row, {})[col] = text
        yield from _dense(sorted(collected.items()))

    def _ordered_rows(self, sheet: _Sheet, counts):
        current = None
        cells: dict = {}
        for row, col, text in self.cells(sheet, counts):
            if current is None:
                current = row
            if row != current:
                yield _row(cells)
                for _ in range(row - current - 1):
                    yield []
                current, cells = row, {}
            cells[col] = text
        if current is not None:
            yield _row(cells)


#: Cells held at once when a sheet's cells are out of order.
MAX_CELLS = 2_000_000


def _row(cells: dict) -> list:
    width = max(cells) + 1
    return [cells.get(col, "") for col in range(width)]


def _dense(items):
    last = None
    for row, cells in items:
        if last is not None:
            for _ in range(row - last - 1):
                yield []
        yield _row(cells)
        last = row


# ---- .xls: extract, pages and streams -------------------------------------

def _workbook(cfb: _Cfb) -> Workbook:
    streams = _root_streams(cfb)
    name = "Workbook" if "Workbook" in streams else "Book"
    if name not in streams:
        raise Unreadable(NO_TABLE)
    return Workbook(_read(cfb, streams[name]))


def _extract_xls(cfb: _Cfb, det: Detection) -> Extracted:
    book = _workbook(cfb)
    blocks, meta, formulas_total, capped_any, wide_any = [], [], 0, False, \
        False
    for sheet in book.sheets:
        if not sheet.is_table:
            meta.append({"name": sheet.name, "error": "not a worksheet "
                         "(chart, macro or module sheet)"})
            continue
        counts: dict = {}
        rows, total = [], 0
        leading = 0
        try:
            for row in book.rows(sheet, counts):
                total += 1
                if not rows and not any(c.strip() for c in row):
                    leading += 1
                    continue
                if len(rows) < office.SHEET_ROW_CAP:
                    rows.append(row[:office.MAX_RENDER_COLUMNS])
                if len(row) > office.MAX_RENDER_COLUMNS:
                    wide_any = True
        except Unreadable as exc:
            meta.append({"name": sheet.name, "error": str(exc)})
            continue
        total -= leading
        columns = max((len(r) for r in rows), default=0)
        capped = total > len(rows)
        capped_any |= capped
        headed = len(rows) > 1 and office._rows_look_headed(rows[:9])
        entry = {"name": sheet.name, "rows": total - (1 if headed else 0),
                 "columns": columns, "has_header": headed}
        if counts.get("formulas"):
            entry["formula_cells"] = counts["formulas"]
            formulas_total += counts["formulas"]
        if capped:
            entry["rows_read"] = len(rows)
        if sheet.hidden:
            entry["hidden"] = True
        meta.append(entry)
        shown = (f"{entry['rows']} data rows" if headed
                 else f"{total} rows")
        shown += f", first {len(rows)} read" if capped else ""
        label = " [hidden]" if sheet.hidden else ""
        blocks.append(f"## Sheet: {sheet.name}{label} ({shown} × "
                      f"{columns} cols)")
        for row in rows[:ROWS_SHOWN]:
            blocks.append("\t".join(row))
        blocks.append("")
    if book.notes:
        blocks.append("## Notes and text boxes")
        blocks.extend(book.notes)
        blocks.append("")
    metadata = {"format": f"Excel BIFF{book.biff}", "sheets": meta,
                "sheet_count": len(meta)}
    if formulas_total:
        metadata["computed_cells"] = (
            f"{formulas_total} cell(s) hold a formula; the values shown are "
            f"the ones cached by the program that saved the file and are not "
            f"recalculated here")
    hidden = [s.name for s in book.sheets if s.hidden]
    if hidden:
        metadata["hidden_sheets"] = hidden
    text = "# Spreadsheet\n" + "\n".join(blocks)
    body, truncated = clip(text, office.MAX_CHARS)
    ex = Extracted(det.kind, det.mime, text=body, metadata=metadata,
                   rendered=True, truncated=truncated or capped_any
                   or wide_any)
    if capped_any:
        ex.note(f"read the first {office.SHEET_ROW_CAP} rows of the larger "
                f"sheets")
    if wide_any:
        ex.note(f"showed the first {office.MAX_RENDER_COLUMNS} columns of "
                f"the wider sheets")
    return ex


def read_sheet_page(path, det: Detection, sheet, offset: int, limit: int,
                    max_chars: int) -> TablePage:
    """One page of one sheet of a .xls (office.read_sheet_page's contract)."""
    page = TablePage(kind=det.kind, offset=offset)
    try:
        book = _workbook(_compound(path))
    except Unreadable as exc:
        page.error = f"could not read the spreadsheet: {exc}"
        return page
    tables = [s for s in book.sheets if s.is_table]
    page.sheets = [s.name for s in tables]
    index = office._choose(page.sheets, sheet)
    if index is None:
        page.total = 0
        page.note(NO_TABLE)
        return page
    page.sheet = page.sheets[index]
    try:
        office._page_of_sheet(page, book.rows(tables[index]), offset, limit,
                              max_chars, count_rest=True)
    except Unreadable as exc:
        office._no_rows(page)
        page.error = f"could not read the spreadsheet: {exc}"
    return page


def open_sheet_stream(path, det: Detection, sheet, sink):
    """(header, rows, close) as office.open_sheet_stream gives them."""
    nothing = (None, iter(()), lambda: None)
    try:
        book = _workbook(_compound(path))
    except Unreadable as exc:
        sink.error = f"could not read the spreadsheet: {exc}"
        return nothing
    tables = [s for s in book.sheets if s.is_table]
    sink.sheets = [s.name for s in tables]
    index = office._choose(sink.sheets, sheet)
    if index is None:
        sink.note(NO_TABLE)
        return nothing
    sink.sheet = sink.sheets[index]
    try:
        header, head, lines = office._head(book.rows(tables[index]))
    except Unreadable as exc:
        sink.error = f"could not read the spreadsheet: {exc}"
        return nothing

    def rows():
        yield from head
        yield from lines
    return header or None, rows(), lambda: None


# ---- .doc ------------------------------------------------------------------

_FIELD_BEGIN, _FIELD_SEP, _FIELD_END = "\x13", "\x14", "\x15"
_DROPPED = frozenset("\x00\x01\x02\x03\x04\x05\x06\x08\x0f\x10\x11\x12\x16"
                     "\x17\x18\x19\x1a\x1b\x1c\x1d\x1f")
_NEWLINES = frozenset("\r\x0b\x0c\x0e")


def clean_word_text(text: str) -> str:
    """Word's text as plain text: paragraph, line and page marks become line
    breaks, a cell mark a tab (two in a row end a table row), a field its
    shown result only (the instruction is dropped), the markers of pictures
    and notes nothing."""
    out = []
    stack: list = []          # one entry per open field: True once its result
    after_cell = False
    for char in text:
        if char == _FIELD_BEGIN:
            stack.append(False)
            continue
        if char == _FIELD_SEP:
            if stack:
                stack[-1] = True
            continue
        if char == _FIELD_END:
            if stack:
                stack.pop()
            continue
        if stack and not all(stack):
            continue
        if char == "\x07":
            if after_cell and out:
                out[-1] = "\n"
                after_cell = False
            else:
                out.append("\t")
                after_cell = True
            continue
        after_cell = False
        if char in _NEWLINES:
            out.append("\n")
        elif char == "\x1e":
            out.append("-")
        elif char in _DROPPED or (char < " " and char not in "\t\n"):
            continue
        else:
            out.append(char)
    return "".join(out)


_DOC_SECTIONS = (("main", 3), ("footnotes", 4), ("headers", 5),
                 ("comments", 7), ("endnotes", 8), ("text boxes", 9),
                 ("header text boxes", 10))


def _doc_sections(cfb: _Cfb) -> dict:
    streams = _root_streams(cfb)
    if "WordDocument" not in streams:
        raise Unreadable("no WordDocument stream")
    word = _read(cfb, streams["WordDocument"])
    if len(word) < 0x20 or _u16(word, 0) != 0xA5EC:
        raise Unreadable("the WordDocument stream has no Word header")
    nfib, flags = _u16(word, 2), _u16(word, 0x0A)
    if flags & 0x0100:
        raise Encrypted("the document is encrypted")
    if nfib < 0x00C1:
        # Word 6.0 / 95: one run of 8-bit text between two offsets.
        first, last = _u32(word, 0x18), _u32(word, 0x1C)
        if not first <= last <= len(word):
            raise Unreadable("the text of a Word 6/95 document lies outside "
                             "its stream")
        return {"main": _decode(word[first:last], "cp1252")}
    csw = _u16(word, 32)
    lw = 34 + csw * 2
    cslw = _u16(word, lw)
    lw += 2
    fc = lw + cslw * 4
    n_pairs = _u16(word, fc)
    fc += 2
    if cslw < 11 or n_pairs < 34:
        raise Unreadable("the FIB is too short for a Word 97 document")
    lengths = struct.unpack_from(f"<{cslw}I", word, lw)
    fc_clx, lcb_clx = struct.unpack_from("<II", word, fc + 33 * 8)
    table_name = "1Table" if flags & 0x0200 else "0Table"
    if table_name not in streams:
        raise Unreadable(f"no {table_name} stream")
    table = _read(cfb, streams[table_name])
    clx = table[fc_clx:fc_clx + lcb_clx]
    pos = 0
    while pos + 3 <= len(clx) and clx[pos] == 1:
        pos += 3 + _u16(clx, pos + 1)
    if pos + 5 > len(clx) or clx[pos] != 2:
        raise Unreadable("the piece table is not where the FIB says")
    size = _u32(clx, pos + 1)
    plc = clx[pos + 5:pos + 5 + size]
    pieces = (len(plc) - 4) // 12
    if pieces < 1 or pieces > MAX_PIECES:
        raise Unreadable("the piece table holds no usable piece")
    cps = struct.unpack_from(f"<{pieces + 1}I", plc, 0)
    parts, total = [], 0
    for i in range(pieces):
        if cps[i + 1] < cps[i]:
            raise Unreadable("the piece table is not in order")
        chars = cps[i + 1] - cps[i]
        fc_piece = _u32(plc, 4 * (pieces + 1) + i * 8 + 2)
        compressed = bool(fc_piece & 0x40000000)
        offset = (fc_piece & 0x3FFFFFFF) // 2 if compressed else fc_piece
        raw = word[offset:offset + chars * (1 if compressed else 2)]
        total += chars
        if total > MAX_TEXT:
            parts.append(_decode(raw, "cp1252") if compressed
                         else raw.decode("utf-16-le", "replace"))
            break
        parts.append(_decode(raw, "cp1252") if compressed
                     else raw.decode("utf-16-le", "replace"))
        deadline.check("a Word document's text")
    full = "".join(parts)
    sections, at = {}, 0
    for name, index in _DOC_SECTIONS:
        count = lengths[index]
        chunk = full[at:at + count]
        at += count
        if chunk:
            sections[name] = chunk
    if not sections:
        sections["main"] = full
    return sections


def _extract_doc(cfb: _Cfb, det: Detection) -> Extracted:
    sections = _doc_sections(cfb)
    blocks = []
    meta = {"format": "Word binary document"}
    for name, raw in sections.items():
        text = clean_word_text(raw).strip("\n")
        if not text.strip():
            continue
        blocks.append(text if name == "main" else f"## {name.capitalize()}"
                      f"\n{text}")
    text = "# Document\n" + "\n\n".join(blocks)
    body, truncated = clip(text, office.MAX_CHARS)
    return Extracted(det.kind, det.mime, text=body, metadata=meta,
                     rendered=True, truncated=truncated)


# ---- .ppt ------------------------------------------------------------------

_PPT_TEXT_CHARS, _PPT_TEXT_BYTES = 0x0FA0, 0x0FA8
_PPT_SLIDE, _PPT_NOTES, _PPT_MAIN_MASTER = 0x03EE, 0x03F0, 0x03F8
_PPT_SLWT, _PPT_PERSIST = 0x0FF0, 0x03F3


def _ppt_blocks(data: bytes):
    """(section, slide number, text) of the text atoms of a PowerPoint
    stream in file order. ``section`` is "slides" (a slide's placeholders),
    "shapes" (a text box on a slide, numbered by the slides seen so far) or
    "notes"; master slides are left out."""
    out = []
    stack: list = []                 # (limit, section, slide number)
    section, slide_no, slides_seen = "shapes", 0, 0
    at, records = 0, 0
    while at + 8 <= len(data):
        while stack and at >= stack[-1][0]:
            _limit, section, slide_no = stack.pop()
        ver_inst, rtype, length = struct.unpack_from("<HHI", data, at)
        body = at + 8
        records += 1
        if records > MAX_PPT_RECORDS:
            break
        if records % 8192 == 0:
            deadline.check("a PowerPoint stream")
        if ver_inst & 0xF == 0xF:
            stop = min(body + length, len(data))
            instance = ver_inst >> 4
            if rtype == _PPT_MAIN_MASTER or (rtype == _PPT_SLWT
                                             and instance == 1):
                at = stop
                continue
            stack.append((stop, section, slide_no))
            if rtype == _PPT_SLWT:
                section, slide_no = ("notes" if instance == 2
                                     else "slides"), 0
            elif rtype == _PPT_SLIDE:
                slides_seen += 1
                section, slide_no = "shapes", slides_seen
            elif rtype == _PPT_NOTES:
                section = "notes"
            at = body
            continue
        if rtype == _PPT_PERSIST and section == "slides":
            slide_no += 1
        elif rtype in (_PPT_TEXT_CHARS, _PPT_TEXT_BYTES):
            raw = data[body:body + length]
            text = (raw.decode("utf-16-le", "replace")
                    if rtype == _PPT_TEXT_CHARS
                    else raw.decode("cp1252", "replace"))
            out.append((section, slide_no, text))
        at = body + length
    return out


def _extract_ppt(cfb: _Cfb, det: Detection) -> Extracted:
    streams = _root_streams(cfb)
    data = _read(cfb, streams["PowerPoint Document"])
    slides: dict = {}
    notes: list = []
    loose: list = []
    for section, number, text in _ppt_blocks(data):
        cleaned = re.sub(r"[\x00-\x08\x0e-\x1f]", "",
                         text.replace("\r", "\n").replace("\x0b", "\n"))
        cleaned = cleaned.strip()
        if not cleaned:
            continue
        if section != "slides" and not any(c.isalnum() for c in cleaned):
            continue
        if section == "notes":
            notes.append(cleaned)
        elif number:
            slides.setdefault(number, []).append(cleaned)
        else:
            loose.append(cleaned)
    blocks = []
    for number in sorted(slides):
        blocks.append(f"## Slide {number}\n" + "\n".join(slides[number]))
    if loose:
        blocks.append("## Text boxes\n" + "\n".join(loose))
    if notes:
        blocks.append("## Notes\n" + "\n".join(notes))
    text = "# Presentation\n" + "\n\n".join(blocks)
    body, truncated = clip(text, office.MAX_CHARS)
    return Extracted(det.kind, det.mime, text=body,
                     metadata={"format": "PowerPoint binary presentation",
                               "slide_count": len(slides)},
                     rendered=True, truncated=truncated)


# ---- the extractor ---------------------------------------------------------

def extract(path, det: Detection, max_bytes: int) -> Extracted:
    """The text of a .xls, .doc or .ppt. Any other compound file (a .msg, a
    Visio drawing) is read as it was before this module: by the fallback."""
    found = _extract(path, det)
    return found if found is not None else fallback.extract(path, det,
                                                            max_bytes)


def _extract(path, det: Detection) -> "Extracted | None":
    try:
        cfb = _compound(path)
        kind = format_of(cfb)
        if kind is None:
            return None
        if kind == XLS:
            return _extract_xls(cfb, det)
        if kind == DOC:
            return _extract_doc(cfb, det)
        return _extract_ppt(cfb, det)
    except Encrypted as exc:
        return Extracted(det.kind, det.mime, text="",
                         metadata={"encrypted": True},
                         notes=[f"{exc}; its text is not read"])
    except Unreadable as exc:
        return Extracted(det.kind, det.mime, text="",
                         metadata={"error": str(exc)},
                         notes=[f"could not read the legacy Office file: "
                                f"{exc}"])


__all__ = ["extract", "format_of", "stream_kind", "read_sheet_page",
           "open_sheet_stream", "clean_word_text", "Workbook", "Unreadable",
           "Encrypted", "DOC", "XLS", "PPT"]
