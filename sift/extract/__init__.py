"""Extractor registry — maps a detected kind to the right extractor."""
from __future__ import annotations

import contextlib
from pathlib import Path

from .. import deadline
from ..config import Config
from ..detect import Detection
from . import (active, email_msg, fallback, html_doc, image, legacy, office,
               pdf, rtf, sqlite_db, structured, tabular, text)
from .base import (ERROR_CHARS, SECRET_SCAN_LIMIT, CutRow, Extracted, Span,
                   TablePage, clip, decode_text, find_secrets, human_size,
                   one_line, repeatable, scrub)

# kind -> module exposing extract(path, det, max_bytes) -> Extracted
_REGISTRY = {
    "text": text, "markdown": text, "log": text, "source-code": text,
    "batch": text, "shell": text, "powershell": text,
    "csv": tabular, "tsv": tabular,
    "email": email_msg,
    "json": structured, "jsonl": structured, "yaml": structured,
    "toml": structured, "ini": structured, "xml": structured,
    "html": html_doc,
    "pdf": pdf,
    "docx": office, "xlsx": office, "pptx": office,
    "odt": office, "ods": office, "odp": office, "epub": office,
    "image": image,
    "rtf": rtf,
    "sqlite": sqlite_db,
    # .xls, .doc, .ppt; any other compound file (.msg, Visio) is the
    # fallback's, as before (legacy.extract)
    "office-legacy": legacy,
}

SUPPORTED_KINDS = sorted(set(_REGISTRY) | {
    "archive", "binary", "pe-executable", "elf-binary", "office-legacy", "empty"})


def _one_line_deep(obj):
    """Fold line breaks in every string a metadata value contains.

    Metadata is rendered one fact per line into the model prompt, and for an
    image that block is the *whole* untrusted input — there is no content to
    fence, so a PNG tEXt chunk with newlines in it was writing top-level
    prompt lines. Same for an HTML <title>, a <meta description>, a docx
    dc:title and a parser's complaint: all of them are the file talking.
    """
    if isinstance(obj, str):
        return one_line(obj)
    if isinstance(obj, dict):
        return {k: _one_line_deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_one_line_deep(v) for v in obj)
    return obj


def extract(path: Path, det: Detection, max_bytes: int,
            active_bytes: int = 0) -> Extracted:
    """Run the extractor registered for ``det.kind`` (fallback for the rest).
    ``active_bytes``: how far the active-content scan reads where that is
    further than ``max_bytes`` (Config.active_max_bytes, active.reach)."""
    reach = active.reach(max_bytes, active_bytes)
    module = _REGISTRY.get(det.kind, fallback)
    try:
        ex = module.extract(path, det, max_bytes)
    except deadline.DeadlineExceeded:
        # Not "one bad file": the caller turns this into a summary that says
        # the file ran out of time. Swallowing it here would have reported a
        # sixty-second timeout as an ordinary extractor error and cached it.
        raise
    except RecursionError:
        # RecursionError is not an Exception subclass path anyone remembers:
        # it inherits from RuntimeError, so `except Exception` does catch it —
        # but a *second* recursion while unwinding does not, and the parsers
        # that can raise it are the ones that recurse per nesting level.
        # Named here so the message says what happened.
        return _with_active(Extracted(
            det.kind, det.mime, text="",
            metadata={"extractor_error": "nesting too deep"},
            notes=[f"{det.kind} extractor gave up: the file nests deeper "
                   f"than the parser allows"]), path, det, reach)
    except Exception as exc:  # noqa: BLE001 - never let one bad file abort a batch
        # The exception's words may quote the file -- a UnicodeDecodeError's
        # repr holds every byte it was decoding -- and this record left
        # before the boundary below: said as sift says an error (repeatable),
        # redacted, one line, at most ERROR_CHARS.
        return _with_active(Extracted(
            det.kind, det.mime, text="",
            metadata={"extractor_error": repeatable(repr(exc), ERROR_CHARS)},
            notes=[f"{det.kind} extractor failed: "
                   f"{repeatable(exc, ERROR_CHARS)}"]), path, det, reach)
    # A text kind can hold a stray NUL now -- WordPad ends an RTF with one,
    # and detect lets rare NULs through (_stray_nuls_in_text). It is not a
    # character of the text: dropped from what sift renders, once, here.
    # (Not from decode_text, whose offsets a caller maps back to the file.)
    if ex.text and "\x00" in ex.text:
        ex.text = ex.text.replace("\x00", "")
    # One boundary rather than a guard per call site. Metadata travels into
    # --json, the on-disk cache, the model prompt and a browser card, and
    # every round of this project has found one more field in it that nobody
    # had guarded: a text preview, an INI parse error, a ragged-row note, a
    # CSV column's examples, an email's headers. There is no counting the
    # routes; there is only the boundary they all cross.
    #
    # The extracted *text* is not scrubbed: it is the file, the model has to
    # read it to summarize it, and a local model reading a local file is the
    # arrangement the user chose. What must not travel is the part sift
    # repeats about the file afterwards.
    _with_active(ex, path, det, reach)
    ex.metadata = {k: _one_line_deep(v) for k, v in scrub(ex.metadata).items()}
    ex.notes = [scrub(n) for n in ex.notes]
    # And the warnings, which quote the file too: a mail's masquerade
    # warning names the attachment as its sender did, and a file name
    # "password: Kx9vQ2mZ7pL4.pdf" left extract() whole there while the same
    # name was hidden in the metadata beside it.
    ex.warnings = [scrub(w) for w in ex.warnings]
    # Keyed by the message, so the keys go through the same scrub the notes
    # just did, or a note that scrub changed would lose its kind.
    ex.finding_kinds = {scrub(message): kind
                        for message, kind in ex.finding_kinds.items()}
    return ex


def _with_active(ex: Extracted, path: Path, det: Detection,
                 reach: int) -> Extracted:
    """The active content of a file of a kind that can carry any
    (active.KINDS): in the metadata for a host application, which does not render
    such a file, and in the notes for a reader. Whatever the extractor
    made of the file, the scan reads it on its own -- a PDF pypdf could
    not parse still has its bytes."""
    if det.kind in active.KINDS:
        found = active.scan(path, det, reach)
        ex.metadata["active_content"] = found.metadata()
        ex.note(found.note())
    return ex


#: Rows one page may hold. A viewer asks for a screenful; a request for a
#: million rows is an export, and an export is not what a page is for.
MAX_PAGE_ROWS = 1000
#: Characters one page may hold: the budget the extractor gives the whole
#: rendering of a table, so a page never costs more than the summary did.
MAX_PAGE_CHARS = tabular.MAX_TEXT_CHARS

_TABLE_KINDS = ("csv", "tsv", "xlsx", "ods", "office-legacy")


def read_table(path: Path, det: Detection, sheet=None, offset: int = 0,
               limit: int = 100, *, max_bytes: "int | None" = None,
               time_budget: "float | None" = None) -> TablePage:
    """One page of the rows of a csv, tsv, xlsx, ods or legacy .xls — any page, not only
    the twelve rows the rendering shows.

    ``sheet`` is a name from ``TablePage.sheets``, an index into it, or None
    for the first; a csv or tsv has none. ``offset`` counts data rows, the
    header excluded. ``max_bytes`` and ``time_budget`` default to the
    Config defaults, so a page is read under the limits a summary is read
    under; running out of time before the page is complete raises
    ``deadline.DeadlineExceeded``, as it does for extract().

    A page costs what it shows: the table is read up to the row after the
    page (which says there is one, ``has_more``) and no further -- an ods to
    its end, for its list of sheets. Where the rest costs next to nothing to
    count -- a csv or tsv read whole (at most 1 MiB) or whose records all
    fit the header scan (tabular.MAX_ROWS_SCAN), an xlsx sheet parsed whole
    (a member of at most 1 MiB), every ods -- it is counted, as 0.1.55
    counted it. Its header, rows, sheets, flags, notes, total and error are
    what sift 0.1.55 gave, which read the whole file for every page, with
    two exceptions for a page that stops before the end of a table it does
    not count: its ``total`` is None, and what only the rest of the file
    holds -- a later line csv cannot parse, damage further into a large xlsx
    member -- is said by the page that reaches it. A csv whose lines end in
    carriage returns alone and which opens with a sep= line keeps its rows,
    which 0.1.55 lost.

    A file that cannot be read as a table comes back with ``error`` set and
    no rows. An argument that cannot name a page raises ValueError.
    """
    _check_table(det, sheet)
    for name, value, least in (("offset", offset, 0), ("limit", limit, 1)):
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < least:
            raise ValueError(f"{name} must be an integer of at least "
                             f"{least}, not {value!r}")
    defaults = Config()
    max_bytes = defaults.max_bytes if max_bytes is None else max_bytes
    time_budget = defaults.time_budget if time_budget is None else time_budget
    path = Path(path)
    with deadline.limit(time_budget):
        try:
            if det.kind in ("csv", "tsv"):
                page = tabular.read_page(path, det, offset,
                                         min(limit, MAX_PAGE_ROWS), max_bytes,
                                         MAX_PAGE_CHARS)
            elif det.kind == "office-legacy":
                page = legacy.read_sheet_page(path, det, sheet, offset,
                                              min(limit, MAX_PAGE_ROWS),
                                              MAX_PAGE_CHARS)
            else:
                page = office.read_sheet_page(path, det, sheet, offset,
                                              min(limit, MAX_PAGE_ROWS),
                                              MAX_PAGE_CHARS)
        except (deadline.DeadlineExceeded, office.NoSuchSheet):
            raise
        except Exception as exc:  # noqa: BLE001 - the file, not the caller
            # One boundary, as extract() has one. The readers name the
            # failures they expect; the rest — zlib's own error for a damaged
            # member, a TypeError from a cell reference nobody imagined, a
            # file that vanished before the csv reader opened it — are just
            # as much the file's fault, and the contract is an error on the
            # page, not a traceback in the viewer.
            page = TablePage(kind=det.kind, offset=offset)
            page.error = _table_error(exc)
    if limit > MAX_PAGE_ROWS:
        page.note(f"a page holds at most {MAX_PAGE_ROWS} rows")
    return page


def _check_table(det: Detection, sheet) -> None:
    if det.kind not in _TABLE_KINDS:
        raise ValueError(f"read_table reads {', '.join(_TABLE_KINDS)}; "
                         f"this file is {det.kind!r}")
    if det.kind in ("csv", "tsv") and sheet is not None:
        raise ValueError(f"a {det.kind} file has no sheets; "
                         f"sheet must be None, not {sheet!r}")


def _table_error(exc: Exception) -> str:
    """What a page says of a failure its reader did not name."""
    return f"could not read the table: {exc}"


class _Found:
    """What reading a table found, which a TableStream shows: the readers
    set it as they set a page (sheets, sheet, preamble, notes, truncation,
    error), the stream its header and flags. A thing of its own, so that
    what reads the rows holds it and not the stream: holding the stream,
    they were a cycle that kept the file open until the collector ran."""

    def __init__(self):
        self.sheets: list = []
        self.sheet: "str | None" = None
        self.header: list = []
        self.preamble: list = []
        self.notes: list = []
        self.truncated = False
        self.columns_truncated = False
        self.error: "str | None" = None

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)


def _shown(name: str):
    return property(lambda self: getattr(self._found, name),
                    lambda self, value: setattr(self._found, name, value),
                    doc=f"The table's {name}, as reading it found it.")


class TableStream:
    """The rows of one table -- a csv or tsv, or one sheet of an xlsx or
    ods -- handed out as they are read (iter_table).

    Iterating it yields every data row once, as a list of cells: the rows
    read_table's pages hold, cut to the column cap as a page cuts them, and
    never clipped to a page's character budget. ``header`` (empty when the
    first row is data), ``preamble``, ``sheet`` and, for a csv or an xlsx,
    ``sheets`` are known when iter_table returns. ``notes``,
    ``truncated``, ``columns_truncated``, ``error`` and an ods's ``sheets``
    are complete once the rows are exhausted: they say what reading the
    rows found. A failure while the rows are read ends them and sets
    ``error``, worded as read_table words it; the rows before it have been
    handed out, where read_table's page that reaches it holds none.

    ``close()``, or leaving a ``with`` block, lets the file go before the
    rows are exhausted; exhausting them does too, and so does letting go
    of the stream: nothing that reads its rows holds it.
    """

    #: Rows read at a time, under the stream's clock and csv's field limit;
    #: between two batches the caller's own are in force.
    BATCH = 256

    sheets = _shown("sheets")
    sheet = _shown("sheet")
    header = _shown("header")
    preamble = _shown("preamble")
    notes = _shown("notes")
    truncated = _shown("truncated")
    columns_truncated = _shown("columns_truncated")
    error = _shown("error")

    def __init__(self, kind: str):
        self.kind = kind
        self._found = _Found()
        self._rows = iter(())

    def note(self, message: str) -> None:
        self._found.note(message)

    def __iter__(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        rows, self._rows = self._rows, iter(())
        if hasattr(rows, "close"):
            rows.close()

    def _start(self, header, rows, close, budget, max_columns: int,
               field_limit) -> None:
        found = self._found
        if header:
            found.columns_truncated = (len(header) > max_columns
                                       or isinstance(header, CutRow))
            found.header = list(header[:max_columns])
        self._rows = _batches(found, rows, close, budget, max_columns,
                              field_limit, self.BATCH)


def _batches(found, rows, close, budget, max_columns, field_limit, size):
    """TableStream's rows: read ``size`` at a time under ``budget`` and
    ``field_limit``, what they say set on ``found``; ``close`` called once
    they end or are let go."""
    try:
        while True:
            batch, more = [], False
            with deadline.resumed(budget), field_limit():
                try:
                    for row in rows:
                        if len(row) > max_columns \
                                or isinstance(row, CutRow):
                            found.columns_truncated = True
                        batch.append(list(row[:max_columns]))
                        if len(batch) >= size:
                            more = True
                            break
                except deadline.DeadlineExceeded:
                    raise
                except office._UNREADABLE as exc:
                    found.error = office.unreadable(exc)
                except Exception as exc:  # noqa: BLE001 - the file's
                    found.error = _table_error(exc)
            yield from batch
            if not more or found.error:
                return
    finally:
        close()


def iter_table(path: Path, det: Detection, sheet=None, *,
               max_bytes: "int | None" = None,
               time_budget: "float | None" = None) -> TableStream:
    """Every row of a csv, tsv, or one sheet of an xlsx or ods, read as the
    caller asks for it: one read of the table, however long, holding a
    row at a time -- where read_table reads the table from its start for
    each page.

    ``sheet`` and ``max_bytes`` are read_table's. The header is decided as
    read_table decides it, on the table's first rows, which are read before
    this returns. ``time_budget`` is a budget of wall-clock seconds that
    starts when iter_table is called: the rows are read under it, a batch
    of TableStream.BATCH at a time as the caller asks for them, and the
    clock does not stop while the caller works on a row -- the time the
    caller takes between two rows is spent of the budget as well, though
    the caller's own code never runs under it and is never cut short by
    it. None (the default) gives the stream no clock of its own: the
    deadline in force where the caller asks for the next row, if any,
    bounds the reading of the batch it starts -- a table read row by row
    has no size a fixed budget fits. Running out of time raises
    ``deadline.DeadlineExceeded``, here or while the rows are read.

    A file that cannot be read as a table comes back with ``error`` set and
    no rows. An argument that cannot name a table raises ValueError.
    """
    _check_table(det, sheet)
    max_bytes = Config().max_bytes if max_bytes is None else max_bytes
    budget = deadline.started(time_budget)
    path = Path(path)
    table = TableStream(det.kind)
    found = table._found
    with deadline.resumed(budget):
        try:
            if det.kind in ("csv", "tsv"):
                header, rows, close = tabular.open_rows(path, det, max_bytes,
                                                        found)
                limits, width = tabular._field_limit, tabular._MAX_COLUMNS
            elif det.kind == "office-legacy":
                header, rows, close = legacy.open_sheet_stream(path, det,
                                                               sheet, found)
                limits, width = contextlib.nullcontext, \
                    office.MAX_RENDER_COLUMNS
            else:
                header, rows, close = office.open_sheet_stream(path, det,
                                                               sheet, found)
                limits, width = contextlib.nullcontext, \
                    office.MAX_RENDER_COLUMNS
        except (deadline.DeadlineExceeded, office.NoSuchSheet):
            raise
        except Exception as exc:  # noqa: BLE001 - the file, not the caller
            found.error = found.error or _table_error(exc)
            return table
    table._start(header, rows, close, budget, width, limits)
    return table


__all__ = ["extract", "read_table", "iter_table", "find_secrets",
           "Extracted", "Span", "TablePage", "TableStream", "clip",
           "decode_text", "human_size", "MAX_PAGE_CHARS", "MAX_PAGE_ROWS",
           "SECRET_SCAN_LIMIT", "SUPPORTED_KINDS"]
