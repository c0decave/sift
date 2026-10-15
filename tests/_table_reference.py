"""read_table as sift 0.1.55 read it, kept frozen for the parity tests.

0.1.55 read the whole file (a csv's text up to max_bytes, an xlsx sheet's or
an ods's whole XML tree) for every page, and counted every row after the
page for ``total``. The streaming readers must hand out the same pages, and
this is what "the same" is measured against: the functions below are
0.1.55's read_page, read_sheet_page and fill_page statement for statement,
over the helpers 0.1.57 did not change (_prepare, _take_preamble,
_rows_look_headed, _sheets_in_order, _shared_strings, _read_member,
_closed, _sheet_row, _ods_rows, _choose). _records and _body are copied as
0.1.55 wrote them, because 0.1.57 rewrote their loops. One change of 0.1.59
reaches it through _prepare, on purpose: a csv whose lines end in carriage
returns alone and which opens with a sep= line kept its rows.
"""
from __future__ import annotations

import csv
import io
import itertools
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from sift import deadline
from sift.config import Config
from sift.extract import office, tabular
from sift.extract.base import CutRow, TablePage, _clip_cells
from sift.zipguard import ZipLimitError, check as check_zip

MAX_PAGE_ROWS = 1000
MAX_PAGE_CHARS = tabular.MAX_TEXT_CHARS
_TABLE_KINDS = ("csv", "tsv", "xlsx", "ods")


def read_table(path, det, sheet=None, offset=0, limit=100, *,
               max_bytes=None, time_budget=None) -> TablePage:
    if det.kind not in _TABLE_KINDS:
        raise ValueError(det.kind)
    for name, value, least in (("offset", offset, 0), ("limit", limit, 1)):
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < least:
            raise ValueError(name)
    defaults = Config()
    max_bytes = defaults.max_bytes if max_bytes is None else max_bytes
    time_budget = defaults.time_budget if time_budget is None else time_budget
    path = Path(path)
    if det.kind in ("csv", "tsv") and sheet is not None:
        raise ValueError("sheet")
    with deadline.limit(time_budget):
        try:
            if det.kind in ("csv", "tsv"):
                page = read_page(path, det, offset, min(limit, MAX_PAGE_ROWS),
                                 max_bytes, MAX_PAGE_CHARS)
            else:
                page = read_sheet_page(path, det, sheet, offset,
                                       min(limit, MAX_PAGE_ROWS),
                                       MAX_PAGE_CHARS)
        except (deadline.DeadlineExceeded, office.NoSuchSheet):
            raise
        except Exception as exc:  # noqa: BLE001 - as 0.1.55's boundary
            page = TablePage(kind=det.kind, offset=offset)
            page.error = f"could not read the table: {exc}"
    if limit > MAX_PAGE_ROWS:
        page.note(f"a page holds at most {MAX_PAGE_ROWS} rows")
    return page


def fill_page(page, rows, *, header=(), offset, limit, max_columns,
              max_chars) -> None:
    used = 0
    if header:
        page.columns_truncated = (len(header) > max_columns
                                  or isinstance(header, CutRow))
        page.header = list(header[:max_columns])
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
            page.has_more = True
            break
        if size > max_chars - used:
            row = _clip_cells(row, max_chars - used)
            size = sum(len(cell) for cell in row)
            page.cells_clipped = True
        page.columns_truncated |= wide
        page.rows.append(list(row))
        used += size
    if page.has_more:
        try:
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


# -- csv, tsv ---------------------------------------------------------------
def read_page(path, det, offset, limit, max_bytes, max_chars) -> TablePage:
    page = TablePage(kind=det.kind, offset=offset)
    text, _encoding, delimiter, byte_capped = tabular._prepare(
        path, det, max_bytes)
    if byte_capped:
        page.truncated = True
        page.note(f"file larger than max_bytes ({det.size} bytes); rows "
                  f"after the first {max_bytes} bytes are not read")
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, 16 * 1024 * 1024))
    try:
        with tabular._numbers_written_with(delimiter):
            provisional = tabular._has_header(text, delimiter)
            records = tabular._until_unparseable(_records(text, delimiter),
                                                 page)
            head = list(itertools.islice(
                records, tabular.MAX_ROWS_SCAN + (1 if provisional else 0)))
            page.preamble = tabular._take_preamble(head, delimiter)
            header = None
            if head and tabular._rows_look_headed(head):
                header = [c.strip() for c in head.pop(0)]
            fill_page(page, _body(head, records, header),
                      header=header or (), offset=offset, limit=limit,
                      max_columns=tabular._MAX_COLUMNS, max_chars=max_chars)
    finally:
        csv.field_size_limit(previous_limit)
    return page


def _body(first, rest, header):
    held = []
    for row in itertools.chain(first, rest):
        if header is not None and [c.strip() for c in row] == header:
            continue
        if not any(c.strip() for c in row):
            held.append(row)
            continue
        yield from held
        held.clear()
        yield row


def _records(text, delimiter):
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    started = False
    for line_number, row in enumerate(reader):
        if not line_number % 512:
            deadline.check("a table")
        if len(row) <= 1 and not (row and row[0].strip()):
            continue
        if not any(c.strip() for c in row) and not started:
            continue
        started = True
        yield row


# -- xlsx, ods --------------------------------------------------------------
def read_sheet_page(path, det, sheet, offset, limit, max_chars) -> TablePage:
    page = TablePage(kind=det.kind, offset=offset)
    try:
        check_zip(path)
        with zipfile.ZipFile(path) as zf:
            if det.kind == "ods":
                lines = _ods_sheet(zf, page, sheet)
            else:
                lines = _xlsx_sheet(zf, page, sheet)
            if lines is None:
                page.total = 0
                page.note("the file holds no table")
                return page
            _page_of_sheet(page, lines, offset, limit, max_chars)
    except ZipLimitError as exc:
        page.error = f"office container was not opened: {exc}"
    except (zipfile.BadZipFile, OSError, KeyError, ET.ParseError) as exc:
        page.error = f"could not read the spreadsheet: {exc}"
    return page


def _xlsx_sheet(zf, page, sheet):
    pairs = office._sheets_in_order(zf)
    page.sheets = [office.one_line(name) for name, _ in pairs]
    index = office._choose(page.sheets, sheet)
    if index is None:
        return None
    page.sheet = page.sheets[index]
    raw = office._read_member(zf, pairs[index][1])
    shared = office._shared_strings(zf)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        if not office._is_truncated(raw):
            raise
        root = ET.fromstring(office._closed(raw))
        page.truncated = True
        page.note(f"sheet larger than "
                  f"{office.human_size(office._MAX_MEMBER)}; the rows "
                  f"after the first part are not read")
    formulas = [0]

    def lines():
        for row in root.iter(f"{office.SS}row"):
            line, width, lost = office._sheet_row(row, shared, formulas)
            if width > office.MAX_RENDER_COLUMNS and lost:
                line = CutRow(line)
            yield line
    return lines()


def _ods_sheet(zf, page, sheet):
    root = ET.fromstring(office._read_member(zf, "content.xml"))
    tables = []
    for node in root.iter():
        if not office._ods_is(node, "table"):
            continue
        if len(tables) >= office._ODS_MAX_TABLES:
            break
        if next(office._ods_rows(node), None) is None:
            continue
        name = office._attr(node, "name") or f"Table {len(tables) + 1}"
        tables.append((office.one_line(name), node))
    page.sheets = [name for name, _ in tables]
    index = office._choose(page.sheets, sheet)
    if index is None:
        return None
    page.sheet = page.sheets[index]

    def cut(times):
        page.truncated = True
        page.note(f"a row the file repeats {times} times is shown "
                  f"{office._ODS_MAX_ROWS} times; the repeats after that "
                  f"are not read")
    return office._ods_rows(tables[index][1], cut)


def _page_of_sheet(page, lines, offset, limit, max_chars):
    lines = iter(lines)
    head = []
    for line in lines:
        if not head and not any(c.strip() for c in line):
            continue
        head.append(line)
        if len(head) >= 9:
            break
    header = ()
    if len(head) > 1 and tabular._rows_look_headed(head):
        header = head.pop(0)
    fill_page(page, itertools.chain(head, lines), header=header,
              offset=offset, limit=limit,
              max_columns=office.MAX_RENDER_COLUMNS, max_chars=max_chars)
