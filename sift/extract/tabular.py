"""Extractor for CSV / TSV — recovers structure, column types and samples."""
from __future__ import annotations

import codecs
import contextlib
import contextvars
import csv
import io
import itertools
import math
import re
from pathlib import Path

from ..detect import Detection, sep_directive, utf16_encoding
from .. import deadline
from .base import (Extracted, TablePage, clip, decode_text, fill_page,
                   one_line, redact_secrets, redact_value)

#: The ceiling on the rendered table, matching the other extractors'.
MAX_TEXT_CHARS = 200_000
#: And on one example value, which travels everywhere metadata does.
_MAX_EXAMPLE = 200
#: And on a column name, which travels the same roads plus the summary.
_MAX_NAME = 120

MAX_ROWS_SCAN = 5000
#: A ceiling on how many columns a file can claim to have. A grid of 60,001
#: uniform fields is not a table anyone is reading; profiling and rendering
#: every column of it took twenty seconds and produced four megabytes of
#: "extracted text" from a 937 KB upload.
_MAX_COLUMNS = 512
SAMPLE_ROWS = 8


def _prepare(path: Path, det: Detection,
             max_bytes: int) -> "tuple[str, str, str, bool]":
    """The file's text, its encoding, its delimiter and whether it was cut.

    Shared by the extractor and by read_table (which reads a file larger
    than _WHOLE_BYTES as a stream, _Text), so that paging through a table
    reads it exactly as the summary did.
    """
    with path.open("rb") as fh:
        data = fh.read(max_bytes)
    byte_capped = det.size > max_bytes > 0
    text, encoding, delimiter = _prepare_text(data, det, byte_capped)
    return text, encoding, delimiter, byte_capped


def _prepare_text(data: bytes, det: Detection,
                  byte_capped: bool) -> "tuple[str, str, str]":
    """_prepare's text, encoding and delimiter of the bytes it read."""
    text, encoding = decode_text(data, byte_capped)
    if byte_capped:
        # The last line is almost certainly cut in half. Counting it as a row
        # invents a record and, with it, empty-cell statistics for columns the
        # file never left blank.
        text = text[:text.rfind("\n") + 1] if "\n" in text else ""
    text, delimiter = _shaped(text, det)
    return text, encoding, delimiter


def _shaped(text: str, det: Detection) -> "tuple[str, str]":
    """``text`` without its sep= line, read with carriage returns as line
    ends where it has no "\\n", and the delimiter it is read with -- as
    _prepare shapes a csv's whole text, and _open_text the head of one it
    streams, which is the whole text where it holds no "\\n"."""
    # The declared separator wins outright, and its line is not data: Excel
    # writes "sep=;" so the spreadsheet opens correctly, and reading it as the
    # header row cost the file its delimiter, its column count and its
    # header.
    declared = sep_directive(text)
    if declared:
        if "\r" in text and "\n" not in text:
            # Classic-Mac line ends (below), read so before the line is cut:
            # cut at "\n", the one line such a text has was the directive's,
            # and every row went with it.
            text = text.replace("\r", "\n")
        # Cut the line the directive is actually on. sep_directive skips
        # blank lines to find it and this cut the first *physical* line, so a
        # file that opened with a blank line lost the blank and kept the
        # directive — which then became the header row, which is the whole
        # bug the directive handling was added to fix.
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if line.strip():
                text = "\n".join(lines[index + 1:])
                break
    delimiter = (declared or det.extra.get("delimiter")
                 or ("\t" if det.kind == "tsv" else ","))
    if not declared and not det.extra.get("delimiter"):
        # Only when detection had nothing to say. Re-sniffing over the top of
        # a delimiter that was chosen by scoring threw that work away and
        # reinstated csv.Sniffer's preference for the comma — which is how a
        # headerless European export came out with every amount split down
        # its decimal comma.
        try:
            delimiter = csv.Sniffer().sniff(
                text[:_SNIFF_CHARS], delimiters=",;\t|").delimiter
        except csv.Error:
            pass
    if "\r" in text and "\n" not in text:
        # Classic-Mac line endings. io.StringIO does not treat a lone \r as a
        # line break, so csv.reader saw one enormous line and raised — and the
        # file was reported as "? rows and ? columns".
        text = text.replace("\r", "\n")
    return text, delimiter


#: A csv whose bytes (up to max_bytes) are no more than this is read and
#: decoded whole, as 0.1.55 read every csv; a larger one is read in pieces
#: of _PIECE_BYTES, as far as its rows are asked for (_Text). Page 0 of a
#: 50 MiB csv read and decoded 25 MB and parsed every row in 0.1.55: 0.71 s.
_WHOLE_BYTES = 1 << 20
_PIECE_BYTES = 1 << 20
#: The characters of the text the header decision is taken on (_has_header)
#: and the delimiter sniffed on (_shaped), typed here once; a test holds
#: every reading to them.
_HEAD_CHARS = 65_536
_SNIFF_CHARS = 4_096
#: The bytes cp1252 has no character for. decode_text reads a file that
#: holds one as latin-1.
_NOT_CP1252 = re.compile(rb"[\x81\x8d\x8f\x90\x9d]")


class _Text:
    """A csv's text as _prepare makes it, read from the file only as far as
    its lines are asked for.

    The bytes are _prepare's (``fh.read(max_bytes)``), decoded as
    decode_text decodes them. decode_text's choice depends on every byte --
    one byte past the first 25 MB that is not UTF-8 makes the whole file
    cp1252 -- but a byte that is ASCII reads the same in every encoding it
    can choose (UTF-8, cp1252, latin-1). So the pieces are decoded as ASCII
    up to the first one with a byte that is not, and there the rest of the
    bytes is read once, without being kept, for decode_text's choice and for
    the bytes it leaves off a file cut at max_bytes (``_verdict``); the text
    goes on in that encoding. A file of no more than _WHOLE_BYTES, and one
    that decode_text reads as UTF-16, is read whole (``whole``); ``small``
    says it was the former.
    """

    def __init__(self, path: Path, det: Detection, max_bytes: int):
        self.byte_capped = det.size > max_bytes > 0
        self.whole = None
        self.small = False
        self._fh = path.open("rb")
        # fh.read(max_bytes) reads nothing for 0 and everything for less.
        self._left = max_bytes if max_bytes >= 0 else None
        self._at = 0
        try:
            first = self._read(max(_WHOLE_BYTES + 1, 4096))
            if len(first) <= _WHOLE_BYTES or utf16_encoding(first[:4096]):
                rest = iter(lambda: self._read(_PIECE_BYTES), b"")
                data = b"".join([first, *rest])
                self.small = len(data) <= _WHOLE_BYTES
                self.whole = _prepare_text(data, det, self.byte_capped)
                self.close()
            else:
                self._first = first
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._fh.close()

    def _read(self, size: int) -> bytes:
        """The next ``size`` bytes of what _prepare reads, fewer at its end."""
        if self._left is not None:
            size = min(size, self._left)
            if size <= 0:
                return b""
        data = self._fh.read(size)
        self._at += len(data)
        if self._left is not None:
            self._left -= len(data)
        return data

    def _pieces(self):
        """The text, in pieces, as decode_text decodes the bytes."""
        chunk, self._first = self._first, None
        while chunk and chunk.isascii():
            deadline.check("a table")
            yield chunk.decode("ascii")
            chunk = self._read(_PIECE_BYTES)
        if not chunk:
            return
        start = self._at - len(chunk)
        encoding, keep = self._verdict(chunk)
        if encoding == "utf-8" and start == 0:
            encoding = "utf-8-sig"              # a byte order mark left out
        decoder = codecs.getincrementaldecoder(encoding)()
        while chunk:
            deadline.check("a table")
            if keep is not None:
                chunk = chunk[:keep]
                keep -= len(chunk)
            text = decoder.decode(chunk)
            if text:
                yield text
            if keep is not None and keep <= 0:
                break
            chunk = self._read(_PIECE_BYTES)
        tail = decoder.decode(b"", True)
        if tail:
            yield tail

    def _verdict(self, chunk: bytes) -> "tuple[str, int | None]":
        """(encoding, bytes to decode from ``chunk`` on, None for all) as
        decode_text chooses them for the bytes from ``chunk`` -- the first
        piece with a byte past ASCII -- to the end; every byte before
        ``chunk`` is ASCII.

        decode_text tries UTF-8 with 0 to 3 bytes left off a cut file, then
        cp1252, then latin-1. The bytes are read to the end once, and the
        file is put back where it was. All but the last three are judged by
        an incremental decoder, and the last three once per trim: UTF-8 is
        valid across a character boundary exactly when it is valid on both
        sides of it, so data[:n - trim] is valid when the bytes before the
        last three are, and the bytes the decoder still holds (a character
        begun) plus the last three less the trim are.
        """
        back, left, at = self._fh.tell(), self._left, self._at
        try:
            utf8 = codecs.getincrementaldecoder("utf-8")()
            valid, cp1252, size, held = True, True, 0, b""
            while chunk:
                deadline.check("a table's encoding")
                size += len(chunk)
                if cp1252 and _NOT_CP1252.search(chunk):
                    cp1252 = False
                if valid:
                    held += chunk
                    if len(held) > 3:
                        try:
                            utf8.decode(held[:-3])
                        except UnicodeDecodeError:
                            valid = False
                        held = held[-3:]
                chunk = self._read(_PIECE_BYTES)
            if valid:
                pending = utf8.getstate()[0]
                # Leaving off every byte from ``chunk`` on leaves ASCII, so
                # a trim no longer than ``held`` decodes if a shorter did not.
                for trim in (0, 1, 2, 3) if self.byte_capped else (0,):
                    if trim > len(held):
                        break
                    try:
                        (pending + held[:len(held) - trim]).decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    return "utf-8", size - trim
            return ("cp1252" if cp1252 else "latin-1"), None
        finally:
            self._fh.seek(back)
            self._left, self._at = left, at

    def lines(self):
        """The text's lines as io.StringIO hands them to csv.reader: split
        at "\\n" only, ends kept. A file cut at max_bytes loses its last line,
        as _prepare cuts the text after its last "\\n"."""
        held = []
        try:
            for piece in self._pieces():
                parts = piece.split("\n")
                if len(parts) == 1:
                    held.append(piece)
                    continue
                held.append(parts[0])
                yield "".join(held) + "\n"
                for part in parts[1:-1]:
                    yield part + "\n"
                held = [parts[-1]] if parts[-1] else []
            if held and not self.byte_capped:
                yield "".join(held)
        finally:
            self.close()


def _open_text(path: Path, det: Detection, max_bytes: int):
    """(lines, head, delimiter, byte_capped, close, small): the text
    _prepare makes of the file, as the iterator of its lines io.StringIO
    would be, read as far as the lines are asked for; ``head``, its first
    _HEAD_CHARS characters (or all), is what the header decision reads;
    ``close`` lets the file go; ``small``, whether the text was read whole
    from no more than _WHOLE_BYTES.

    _prepare takes the sep= line off by the first line that is not white
    space, sniffs the delimiter on the first 4,096 characters after it, and
    reads a text without any "\\n" with its carriage returns as line ends.
    All three are decided here on the lines read up to _HEAD_CHARS
    characters past the sep= line; a line held here ends in "\\n" unless it
    is the text's last, so each is decided as on the whole text.
    """
    source = _Text(path, det, max_bytes)
    if source.whole is not None:
        text, _encoding, delimiter = source.whole
        return (io.StringIO(text), text[:_HEAD_CHARS], delimiter,
                source.byte_capped, source.close, source.small)
    lines = source.lines()
    held, size = [], 0
    for line in lines:
        held.append(line)
        size += len(line)
        if not line.isspace():
            break
    if held and not held[-1].endswith("\n"):
        # One line, and the text's last: the whole text, without a "\n"
        # in it, shaped as _prepare shapes it.
        text, delimiter = _shaped("".join(held), det)
        return (io.StringIO(text), text[:_HEAD_CHARS], delimiter,
                source.byte_capped, lines.close, False)
    declared = sep_directive("".join(held))
    if declared:
        for index, line in enumerate(held):
            if line.strip():
                del held[:index + 1]
                break
        size = sum(len(line) for line in held)
    while size < _HEAD_CHARS:
        line = next(lines, None)
        if line is None:
            break
        held.append(line)
        size += len(line)
    text = "".join(held)
    delimiter = (declared or det.extra.get("delimiter")
                 or ("\t" if det.kind == "tsv" else ","))
    if not declared and not det.extra.get("delimiter"):
        try:
            delimiter = csv.Sniffer().sniff(
                text[:_SNIFF_CHARS], delimiters=",;\t|").delimiter
        except csv.Error:
            pass
    # Held lines end in "\n" but the text's last: without one among them,
    # they are the whole text.
    if "\r" in text and "\n" not in text:
        text = text.replace("\r", "\n")
        return (io.StringIO(text), text[:_HEAD_CHARS], delimiter,
                source.byte_capped, lines.close, False)
    return (itertools.chain(held, lines), text[:_HEAD_CHARS], delimiter,
            source.byte_capped, lines.close, False)


@contextlib.contextmanager
def _field_limit():
    """csv's field limit raised for the file's rows: a description column
    longer than 128 KB is normal in real exports. The limit is csv's, for
    the whole process, and goes back when the rows are read."""
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 16 * 1024 * 1024))
    try:
        yield
    finally:
        csv.field_size_limit(previous)


def open_rows(path: Path, det: Detection, max_bytes: int, sink):
    """(header, rows, close) of a csv or tsv, read as the rows are asked for.

    The header (None when the first record is data) and the preamble are
    decided as read_page decided them in 0.1.55, on the first MAX_ROWS_SCAN
    records; the preamble goes on ``sink``, and so do the notes and the
    truncation (max_bytes, a line csv cannot parse) as they are met.
    ``rows`` hands out the data records -- every one, each once -- and is
    read with csv's field limit raised (``_field_limit``); ``close`` lets
    the file go before they are exhausted.
    """
    return _open_records(path, det, max_bytes, sink)[:3]


def _open_records(path: Path, det: Detection, max_bytes: int, sink):
    """open_rows' (header, rows, close), and whether counting the rows is
    cheap: the text was read whole from no more than _WHOLE_BYTES, or its
    records all fit the first MAX_ROWS_SCAN read for the header."""
    lines, head, delimiter, byte_capped, close, small = _open_text(
        path, det, max_bytes)
    try:
        if byte_capped:
            sink.truncated = True
            sink.note(f"file larger than max_bytes ({det.size} bytes); rows "
                      f"after the first {max_bytes} bytes are not read")
        with _field_limit(), _numbers_written_with(delimiter):
            provisional = _has_header(head, delimiter)
            records = _until_unparseable(
                _records_of_lines(lines, delimiter), sink)
            wanted = MAX_ROWS_SCAN + (1 if provisional else 0)
            first = list(itertools.islice(records, wanted))
            cheap = small or len(first) < wanted
            sink.preamble = _take_preamble(first, delimiter)
            header = None
            if first and _rows_look_headed(first):
                header = [c.strip() for c in first.pop(0)]
    except BaseException:
        close()
        raise
    return header, _body(first, records, header), close, cheap


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    text, encoding, delimiter, byte_capped = _prepare(path, det, max_bytes)
    # From here on every number is read in this file's own convention. The
    # header test calls _is_number too, so the frame has to open before it.
    stack = contextlib.ExitStack()
    stack.enter_context(_numbers_written_with(delimiter))
    # Provisional, and used for one thing only: the row cap below has to know
    # whether to allow one line for a header while it is still reading. The
    # verdict that decides the table is taken further down, on the rows that
    # are actually the table.
    has_header = _has_header(text, delimiter)

    # A description column longer than 128 KB is normal in real exports, and
    # the default limit turns the whole file into an extractor_error.
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, 16 * 1024 * 1024))
    try:
        rows: list[list[str]] = []
        scanned_all = True
        for row in _records(text, delimiter):
            rows.append(row)
            if len(rows) > MAX_ROWS_SCAN + (1 if has_header else 0):
                rows.pop()
                scanned_all = False
                break
    except csv.Error:
        # A \r-only file makes csv.reader raise mid-iteration ("new-line
        # character seen in unquoted field"). Losing the rows read so far
        # turned the whole file into "? rows and ? columns".
        scanned_all = False
    finally:
        csv.field_size_limit(previous_limit)

    # Lines above the table: a report's title and "Generated …" line.
    preamble = _take_preamble(rows, delimiter)
    if not rows:
        stack.close()
        return Extracted(det.kind, det.mime, text=text[:2000],
                         metadata={"encoding": encoding, "rows": 0}, notes=["empty table"])

    # An unnamed header cell is the index column pandas writes. Leaving it
    # blank makes the profile read as if a column had gone missing.
    # Decided here, on the rows that are actually the table — not on the
    # raw text before the preamble came off. Deciding first and cutting
    # afterwards meant the verdict belonged to a different first row, so a
    # header could be filed as a preamble and a record promoted to names.
    has_header = _rows_look_headed(rows)
    # A row identical to the header is the header again: two exports
    # concatenated, which is how a lot of CSV arrives. Counted as data it
    # inflated the row count by one per join and put the column names into
    # every column's examples — "900 rows" for 899 records, silently.
    repeated_header = []
    if has_header and rows:
        first = [c.strip() for c in rows[0]]
        for i, row in enumerate(rows[1:], 2):
            if [c.strip() for c in row] == first:
                repeated_header.append(i)
        if repeated_header:
            keep = set(repeated_header)
            rows = [rows[0]] + [r for i, r in enumerate(rows[1:], 2)
                                if i not in keep]

    header = ([c.strip() or f"col{i+1}" for i, c in enumerate(rows[0])]
              if has_header else [f"col{i+1}" for i in range(len(rows[0]))])
    body_rows = rows[1:] if has_header else rows
    # The table's own width is the count most of its rows have, not the
    # widest row's. One line of a million commas made n_cols a million: the
    # profile and the renderer both loop over it, so a 1 MB upload took 70
    # seconds, 754 MB of memory and emitted 42 MB of "extracted text". And
    # on the ordinary case it was simply wrong — "3,Smith, John,Rome" in a
    # three-column file was reported as four columns with the fourth 80%
    # empty, while the note two lines below said the row had a field too
    # many. The output contradicted itself.
    declared = min(declared_width(rows), _MAX_COLUMNS)
    # A header wider than its rows declares the table: "id,name,city,notes,
    # extra" over three-field rows has five columns, two of them empty, and
    # reporting three deleted two column names outright.
    if has_header and len(rows[0]) > declared:
        # Trailing empty cells are a hand-edit and an Excel artefact, not
        # columns: "id,name,city,," declared five and reported every data row
        # as missing two fields.
        named = len(rows[0])
        while named > declared and not rows[0][named - 1].strip():
            named -= 1
        declared = min(max(named, declared), _MAX_COLUMNS)
    n_cols = declared
    # A row with more fields than that is a broken line, not a new column.
    # "3,Smith, John,Rome" in a three-column file was reported as four
    # columns with the fourth 80% empty — a plausible-sounding description
    # that points the reader away from the one line that is actually wrong.
    # Raggedness is judged against the file's real width, not the capped one.
    # Capping the *profile* at 512 columns is right — a 600-column survey
    # export is an ordinary instrument and profiling all of it is waste — but
    # suppressing the ragged test along with it made a genuinely short row in
    # such a file silent, which is the one thing about it worth saying.
    wide = declared_width(rows)
    capped = wide > _MAX_COLUMNS
    against = wide if capped else declared

    def _ragged(index: int, row: list) -> bool:
        """Whether this row's width contradicts the table's.

        Trailing empty cells on the header are a hand-edit and an Excel
        artefact — but only when the body does not have them too. Stripping
        them unconditionally made "id,name,stadt,," over rows that also end
        in ",," report the *header* as the one line missing two fields: the
        file accusing its own first line, in a table that is perfectly
        regular.
        """
        if len(row) == against:
            return False
        if index == 1 and has_header:
            trimmed = list(row)
            while trimmed and not trimmed[-1].strip():
                trimmed.pop()
            if len(trimmed) == against:
                return False
        return True

    ragged = [i for i, r in enumerate(rows, 1) if r and _ragged(i, r)]
    # What the extra fields held. Capping the column count stops one line of
    # a million commas from defining the table, but it must not make the
    # content vanish: "3,Smith, John,Rome" lost "Rome" from the profile, the
    # rendered text and the prompt, and the note said only that a row had a
    # field too many.
    overflow = [c for r in rows if len(r) > against for c in r[against:]
                if c.strip()][:5]
    header = (header + [f"col{i+1}" for i in range(len(header), n_cols)])[:n_cols]

    # Trailing delimiter-only lines are an Excel export artefact, not records
    # with 100% missing data.
    while body_rows and not any(c.strip() for c in body_rows[-1]):
        body_rows.pop()

    columns = _profile_columns(header, body_rows, n_cols)
    # Truncated means "the numbers below describe part of the file". The header
    # must not count against the row budget, or a file of exactly MAX_ROWS_SCAN
    # rows reports one row fewer and claims it was cut short.
    truncated = not scanned_all or byte_capped
    total_rows = len(body_rows)

    meta = {
        "encoding": encoding,
        "delimiter": delimiter if delimiter != "\t" else "\\t",
        "has_header": has_header,
        "columns": n_cols,
        "rows": total_rows,
        "rows_truncated": truncated,
        "column_profile": columns,
    }
    if truncated:
        meta["rows_note"] = "row count and column statistics cover the part read"
    if preamble:
        meta["preamble"] = preamble
    if capped:
        # A 600-column export is an ordinary survey instrument or feature
        # matrix. Capping the profile is right; reporting all twenty of its
        # rows as broken, with the *names* of the dropped columns quoted as
        # stray data, is not.
        meta["columns_note"] = (
            f"{declared_width(rows)} columns; the first {_MAX_COLUMNS} are "
            f"profiled")
    if repeated_header:
        shown = ", ".join(str(i) for i in repeated_header[:5])
        more = (f" and {len(repeated_header) - 5} more"
                if len(repeated_header) > 5 else "")
        meta["repeated_header"] = (
            f"the header row appears again at line{'s' if len(repeated_header) > 1 else ''} "
            f"{shown}{more} — counted as headers, not as records; the file "
            f"looks like several exports joined together")
    if ragged:
        meta["ragged_note"] = ""
        shown = ", ".join(str(i) for i in ragged[:5])
        more = f" and {len(ragged) - 5} more" if len(ragged) > 5 else ""
        meta["ragged_rows"] = len(ragged)
        # Both against the real width, not the capped one. The note said "do
        # not have 512 fields" of a 600-column file, and diagnosed a short
        # row as an unquoted separator because 88 columns of every *intact*
        # row sat past the cap and read as an overflow.
        why = ("an unquoted separator, most likely"
               if any(len(r) > against for r in rows)
               else "a missing field, most likely")
        note = (f"{len(ragged)} row(s) do not have {against} fields "
                f"(line{'s' if len(ragged) > 1 else ''} {shown}{more}) — {why}")
        if overflow:
            # Redacted like every other value sift repeats. This note travels
            # into --json, the terminal, the on-disk cache and the model
            # prompt, and the field it quotes came from a data row: a token
            # column with one unquoted comma in it put a live credential in
            # all four.
            shown = [redact_secrets(v[:40]) for v in overflow]
            note += "; the extra field(s) held: " + ", ".join(
                repr(v) for v in shown)
        meta["ragged_note"] = note

    # Clipped like every other extractor. This was the only one that was not:
    # text 200k, html 150k, office 200k, structured 120k, email 100k — and
    # tabular uncapped, so one 2 MB quoted cell became 22 MB of ex.text and
    # then 207 seconds of CPU downstream, from a single unauthenticated POST.
    # The row and column scans are already bounded; the *rendering* was not.
    text_repr, over = clip(
        _render(header, columns, body_rows, delimiter, total_rows, truncated),
        MAX_TEXT_CHARS)
    stack.close()
    ex = Extracted(det.kind, det.mime, text=text_repr, metadata=meta, rendered=True,
                   truncated=truncated or over, text_covers=("column_profile",))
    if over:
        ex.note(f"table rendering longer than {MAX_TEXT_CHARS} characters; "
                f"the model was shown the first part")
    if ragged:
        ex.note(note)
    if not scanned_all:
        ex.note(f"scanned the first {MAX_ROWS_SCAN} data rows of a longer file")
    if byte_capped:
        ex.note(f"file larger than max_bytes ({det.size} bytes); read first "
                f"{max_bytes}")
    return ex


def read_page(path: Path, det: Detection, offset: int, limit: int,
              max_bytes: int, max_chars: int) -> TablePage:
    """One page of a CSV or TSV's records, read the way extract() reads them.

    The same text, delimiter, preamble and header decision as the summary —
    the header is judged on the same first rows the extractor profiles — so
    the rows a reader pages through are the rows the summary counted. Only
    the extractor's row ceiling is gone: that bounds a profile, and a page
    has bounds of its own. The file is read up to the row after the page;
    where the rows are in hand or nearly (_open_records), the rest is
    counted for the total, as 0.1.55 counted it.
    """
    page = TablePage(kind=det.kind, offset=offset)
    header, rows, close, cheap = _open_records(path, det, max_bytes, page)
    try:
        with _field_limit():
            fill_page(page, rows, header=header or (), offset=offset,
                      limit=limit, max_columns=_MAX_COLUMNS,
                      max_chars=max_chars, count_rest=cheap)
    finally:
        close()
    return page


def _until_unparseable(records, page: TablePage):
    """The records up to the first line csv cannot parse, which ends them.

    The extractor treats that line the same way: what came before it is
    real, what comes after is not reachable.
    """
    try:
        yield from records
    except csv.Error as exc:
        page.truncated = True
        page.note(f"the rest of the file is not valid CSV ({exc}); the rows "
                  f"after that point are not shown")


def _body(first: list, rest, header: "list | None"):
    """The data rows, without what the extractor does not count as records:
    the header repeated where two exports were joined, and delimiter-only
    lines at the very end, which are an Excel artefact.

    Delimiter-only lines anywhere else are records — a respondent who
    answered nothing — so they are held back only until a real row shows
    they were not the end. (A row is blank when the join of its cells is:
    a cell holds something other than white space exactly when the join
    does, and the join is one call where the cells were one each.)
    """
    held = []
    width = -1 if header is None else len(header)
    for row in itertools.chain(first, rest):
        if len(row) == width and [c.strip() for c in row] == header:
            continue
        if not "".join(row).strip():
            held.append(row)
            continue
        if held:
            yield from held
            held.clear()
        yield row


def _records(text: str, delimiter: str):
    """The file's records as csv.reader reads them from ``text``
    (_records_of_lines)."""
    return _records_of_lines(io.StringIO(text), delimiter)


def _records_of_lines(lines, delimiter: str):
    """The file's records as csv.reader reads them from ``lines`` (the
    text's lines, as io.StringIO hands them), minus the lines that are
    formatting rather than records.

    The caller raises csv.field_size_limit around the iteration and catches
    csv.Error, which a lone carriage return inside an unquoted field
    raises mid-file.
    """
    reader = csv.reader(lines, delimiter=delimiter)
    started = False
    for line_number, row in enumerate(reader):
        if not line_number % 512:
            deadline.check("a table")
        # csv.reader yields [] for a genuinely blank line and ['', '', '']
        # for ",,". A blank line has no fields, so it is formatting
        # wherever it appears — a paragraph break between records inflated
        # the row count and invented a missing value in every column. A
        # delimiter-only line does have fields: between records that is a
        # record, a survey respondent who answered nothing, and dropping it
        # reported zero missing data. Leading ones are still formatting,
        # because keeping them makes row zero empty and the real header
        # becomes data, turning named typed columns into col1/col2 strings.
        if len(row) <= 1 and not (row and row[0].strip()):
            # No fields, or one blank field: csv.reader gives [] for "\n"
            # and ['   '] for a line of spaces, and neither is a record.
            # Testing only for [] left the whitespace line a phantom.
            continue
        if not started:
            if not "".join(row).strip():
                continue
            started = True
        yield row


def _take_preamble(rows: list, delimiter: str) -> "list[str]":
    """Remove the lines above the table from ``rows``, and return them.

    Reporting tools write a title and a "Generated …" line before the
    header, and read as data they became the column names. Kept in the
    metadata, not thrown away: they are the file's own text and often say
    what the table is.

    A title line is *narrower* than the table by a wide margin — one field
    where the table has several. A pandas or R index column is narrower by
    exactly one, and that is a header, not a preamble; taking both made the
    header itself the preamble and promoted the first record to column
    names.
    """
    preamble = []
    while (len(rows) > 2 and len(preamble) < 5
           and len(rows[0]) < declared_width(rows[1:])
           and not _index_column_header(rows)
           # A short *record* is not a preamble. "2026-01-01,GET,/a" over
           # four-field log rows is a line that lost a field, and filing it
           # as a title destroyed it with no trace.
           # rows[1:9], not the whole table: passing every row of a
           # 5000x200 file made this loop five seconds of work.
           and not _looks_like_the_body(rows[0], rows[1:9])):
        preamble.append(delimiter.join(rows.pop(0)).strip())
    return preamble


def _index_column_header(rows: list) -> bool:
    """True when row one is a header over a table with a row-label column.

    pandas writes ",name,value"; R writes '"x","y"' over '"a",1,4'. Both are
    one field narrower than their rows, and so is a two-field report title
    over a three-column table — "Monthly Sales,2026-01". Treating every
    narrow first row as a header gave the table to the title; treating every
    one as a preamble threw the header away. What tells them apart is what is
    underneath: an index column is a column of short distinct labels.
    """
    if len(rows) < 3:
        return False
    first, rest = rows[0], rows[1:9]
    if len(first) + 1 != declared_width(rest):
        return False
    if not first[0].strip():
        return True                      # pandas: the label column is unnamed
    labels = [r[0].strip() for r in rest if r and r[0].strip()]
    if len(labels) < 2:
        return False
    # Real row labels are gene symbols, sample ids, city names — a ceiling
    # of three characters recognised almost none of them, so an R matrix
    # header was filed as a preamble and the columns lost their names. What
    # makes a label column is that its values are distinct and unlike the
    # numbers beside them, not that they are short.
    if len(set(labels)) != len(labels):
        return False
    if all(_is_number(v) for v in labels):
        return False                     # a numeric first column is data
    # Either shape of row label: short ones (a,b,c — pandas and hand-written
    # tables) or long ones over a body of numbers (gene symbols, sample ids —
    # what R's write.csv produces from a matrix). Requiring only the second
    # cost the first: a header of "name,city" over rows with a trailing
    # comma was filed as a preamble again.
    if all(len(v) <= 4 for v in labels):
        return True
    others = [c for r in rest for c in r[1:] if c.strip()]
    return bool(others) and all(_is_number(c) or _DATE.match(c.strip())
                                for c in others)


def declared_width(rows: list) -> int:
    """The field count most rows have — the table's real width."""
    widths = [len(r) for r in rows if r]
    return max(set(widths), key=widths.count) if widths else 0


def _has_header(text: str, delimiter: str) -> bool:
    """Decide whether row one is a header, without trusting the Sniffer alone.

    csv.Sniffer().has_header misfires in both directions on real files: a TSV
    whose header words contain commas is called headerless (so the header words
    become data examples), and a header of years — 2020,2021,2022 — is called
    data (so 2020 lands in the column statistics and drags the mean to 535).
    The structural test is simpler and steadier: a header row is one whose
    cells look unlike the rows beneath it.
    """
    try:
        rows = list(csv.reader(io.StringIO(text[:_HEAD_CHARS]),
                               delimiter=delimiter))
    except csv.Error:
        rows = []
    return _rows_look_headed(rows)


def _strict_signature(value: str):
    """A signature a column's values share only if they are the same *kind*.

    Exact length, not a bucket. A bucket of four was too coarse in one
    direction — "Name" and "Anna" fall in the same one — and too fine in the
    other, because Berlin, Hamburg, Muenchen and Koeln fall in three, so an
    ordinary word column looked non-uniform and said nothing at all.

    Exact length says nothing about word columns either, and that is the
    point: it abstains there instead of guessing. Where it does speak — a
    column of two-letter country codes, of seven-digit populations, of
    fixed-width identifiers, of dates — it is right, and those are the
    columns a headerless file has.
    """
    v = value.strip()
    if not v:
        return None
    if _DATE.match(v):
        return ("date", len(v))
    if _is_number(v):
        return ("number", len(v.lstrip("+-")))
    return ("text", len(v), v.isupper(), v.islower(),
            v[:1].isupper() and not v.isupper(),
            any(ch.isdigit() for ch in v))


def _looks_like_the_body(first: list, rest: list) -> bool:
    """True when row one is, in some column, exactly the kind the body is.

    One column is enough, because a coincidence in one column of a headed
    file is rare and a match in every column of a headerless one is not
    always available: "Ann,Berlin,DE" gives it in the first and the third,
    and "Berlin,3664088" only in the second.
    """
    agree = 0
    for index, cell in enumerate(first):
        column = [r[index] for r in rest if index < len(r) and r[index].strip()]
        if len(column) < 2:
            # Two body values are enough to see a shape; three meant the test
            # could never speak on a three-line table, and the fallback
            # favours "header" — 23 of 24 three-line files lost their first
            # record, against none before. A three-line CSV is a filtered
            # query result, a receipt, a config dump.
            continue
        shapes = {_strict_signature(c) for c in column}
        if len(shapes) != 1 or None in shapes:
            continue                     # the column is not uniform
        signature = next(iter(shapes))
        if not _informative(signature):
            # A column of four-letter Title-case words agrees with a
            # four-letter Title-case header by coincidence, and the
            # coincidence is common: Name over Anna, Beth, Cara. Dates,
            # numbers, codes and identifiers do not agree by coincidence,
            # and a long word agrees rarely enough to be worth believing.
            continue
        mine = _strict_signature(cell)
        if mine and signature[0] == "number" and mine[0] == "number":
            # Numbers over numbers. Where the widths differ, or where row one
            # reads as a year and the column below it does not, the question
            # belongs to the year-header test further down: "2022" over
            # 1388/1390/1395 is a pivot table's header and "1000" over 1/2/3
            # is a file sorted descending, and nothing about their shape
            # separates them. Where neither holds — 31 over 42/25/38, 3664088
            # over seven-digit populations — the agreement is real.
            if mine[1] != signature[1] or (_looks_like_year(cell)
                                           and not any(_looks_like_year(c)
                                                       for c in column)):
                continue
        if mine != signature:
            # One column where row one is *not* the kind the body is settles
            # it the other way: a header. "ICAO,Name,Country" agrees in
            # column one and disagrees in column three, and it is a header.
            return False
        agree += 1
    return agree > 0


def _looks_like_year(value: str) -> bool:
    """A four-digit number in the range a spreadsheet header uses."""
    text = value.strip()
    return len(text) == 4 and text.isdigit() and 1900 <= int(text) <= 2100


def _informative(signature) -> bool:
    """True when agreement with this signature means something."""
    kind = signature[0]
    if kind in ("date", "number"):
        return True
    _, length, upper, _lower, _title, has_digit = signature
    # Length alone is not evidence about words. "Country" is seven letters
    # and so are Austria, Finland and Denmark; "userid" is six and so is
    # every Active Directory login beside it; three real Swedish surnames
    # are nine. A column of uppercase codes or of strings carrying digits
    # does not agree with a header by accident — an ordinary word column
    # does, constantly, and the file where it does is a headed one.
    return bool(upper or has_digit) and length >= 2


def _rows_look_headed(rows: list) -> bool:
    """The structural half of the test, shared with the spreadsheet reader.

    A sheet's first row is a header on exactly the same evidence as a CSV's,
    and having only one of the two formats know that made "rows" mean data
    records in one and every line in the other.
    """
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return False
    if len(rows) < 2:
        # A single line is a header only if it reads like names. Calling a lone
        # data row a header reported the file as having zero rows.
        return not any(_is_number(c) or _DATE.match(c.strip()) for c in rows[0])
    first, rest = rows[0], rows[1:9]
    if not first:
        return False
    named = [c.strip() for c in first if c.strip()]
    if not named:
        return False                    # an entirely blank row names nothing
    if len(named) < len(first) * 0.5:
        return False                    # mostly blank: data, not names
    # A single empty leading cell is the index column pandas.to_csv() writes —
    # ",name,value" is a header, and calling it data cost the file a row and
    # every column its statistics.
    # A repeated name is not evidence against a header: "date,value,date,
    # value" is what a joined or pivoted export looks like, and reading it as
    # data cost the file a record, its column names and its column types all
    # at once. Whether row one is a header is decided below, on whether it
    # fits the shape of the column beneath it.
    # Vocabulary first: it is the only signal that separates a header of
    # ordinary words from a record of ordinary words, and it never says
    # "data", so consulting it first cannot cost a headerless file anything.
    # A record that is exactly the kind its column holds is a record,
    # whatever its words happen to be.
    #
    # A list of ~200 English and German column names used to sit after this,
    # added because shape analysis cannot tell "Name" from "Anna". It was
    # measured against a corpus written after it and scored 34 of 34; an
    # independently written one put it at rescuing nothing and costing six
    # headerless files their first record — "Total" is a petrol station,
    # "mail" is a service name, "no" is one half of a glossary. With
    # _looks_like_the_body in front of it it earns exactly nothing: the same
    # numbers with it and without it, on both sides. So it is gone. A
    # mechanism that costs nothing and buys nothing is still worth deleting,
    # because the next reader would otherwise have to work out what it is for.
    if _looks_like_the_body(first, rest):
        return False
    # A row one narrower than the body is the index column pandas and R
    # write: ",name,value" and '"x","y"' over three-field rows. The missing
    # cell is the row label, and the row is a header.
    body_width = declared_width(rest)
    if body_width and len(first) < body_width:
        return True
    # _shape_verdict used to sit here, asking whether row one had the same
    # coarse *form* as the column below it. It could not tell "Name" from
    # "Anna" — both four-letter Title-case words — and with a length bucket
    # added to tell them apart it stopped speaking about ordinary word
    # columns at all, because Berlin, Hamburg, Muenchen and Koeln fall in
    # three different buckets. Measured on an independently written corpus it
    # was wrong on 19% of headed files and 35% of headerless ones, worse on
    # both sides than the version it replaced. _looks_like_the_body above
    # asks the sharper question and abstains where it has nothing to say.
    numeric_first = sum(1 for c in first if _is_number(c) or _DATE.match(c.strip()))
    if numeric_first == len(first):
        # Every cell of row one is a number. It is still a header when those
        # numbers are years — the pivot-table shape, where 2020 in the
        # statistics moved a mean from 40 to 535. Anything looser than that
        # promoted an ordinary outlier row into column names.
        below_numeric = sum(
            1 for r in rest for c in r if _is_number(c) or _DATE.match(c.strip()))
        below_cells = sum(len(r) for r in rest) or 1
        if below_numeric / below_cells < 0.5:
            return True                 # numbers over text: a header
        return _looks_like_year_header(first, rest)
    # Row one is at least partly textual: a header when the body is less so.
    body_text = sum(1 for r in rest for c in r
                    if c.strip() and not _is_number(c) and not _DATE.match(c.strip()))
    body_cells = sum(1 for r in rest for c in r if c.strip()) or 1
    return (body_text / body_cells) < 0.9 or numeric_first == 0


def _profile_columns(header, body_rows, n_cols) -> list[dict]:
    """Profile every scanned value, not a sample of them.

    The type, the range and the mean were taken from the first 200 values but
    reported as facts about the column. On a 5000-row file that put the maximum
    at 199 instead of 4999 and the mean off by 25x — and a column that turns to
    text after row 200 was still declared numeric. Scanning the rows we already
    hold in memory costs nothing; being wrong about them costs the whole point.
    """
    cols = []
    stack = contextlib.ExitStack()
    for c in range(n_cols):
        stack.close()                    # the previous column's convention
        raw = [r[c] for r in body_rows if c < len(r) and r[c] != ""]
        missing = _missing_markers(raw)
        values = [v for v in raw if not missing.match(v)]
        # Decided per column, not per file. "12,500" in a tab-separated file
        # is five hundred grams in a German parts list and twelve and a half
        # thousand people in an American one, and the file-level rule turned
        # the second into 12.5 while "8,336,817" in the *same column* stayed
        # whole — the column contradicting itself, which is exactly the
        # failure the file-level rule was written to remove.
        stack.enter_context(_numbers_written_with_commas(_comma_means(values)))
        # Clipped: the header is a row like any other, so the 2 MB cell that
        # this clip exists for can just as easily be a column *name* — and a
        # name travels into the schema list, the summary, every key point,
        # the metadata, the cache and the browser card. Measured at 6 MB of
        # --json for one 2 MB cell, all of it the same string four times.
        name = header[c] if c < len(header) else f"col{c+1}"
        if len(name) > _MAX_NAME:
            name = name[:_MAX_NAME] + "…"
        # A column of NaN and Inf is not an empty column. Both are treated as
        # missing markers, which is right — NaN means "no measurement" — but
        # reporting the result as "wert (empty), 4 empty" says the cells are
        # blank, and they are not: they carry four statements that the value
        # is unavailable. A reader chasing a data-loading bug is sent to the
        # wrong place.
        marked = [v for v in raw if missing.match(v)]
        kind = _infer_type(values)
        if not values and marked:
            kind = "missing"
        col = {
            "name": name,
            "type": kind,
            "non_empty": len(values),
            # "empty" stays the total of cells with no usable value, which
            # is what every reader and every earlier test means by it;
            # "marked_missing" says how many of those were an explicit
            # marker rather than a blank.
            "empty": len(body_rows) - len(values),
            "marked_missing": len(marked),
            "distinct": len(set(values)),
            # The column's own name is the key its values are judged by, and
            # this is the only place both are in hand: the metadata scrub
            # walks a *list* of profiles, so it sees the key "column_profile"
            # and never the name "password". A CSV of API keys published
            # three of them per column into --json, the cache, the prompt and
            # the browser card — and "hunter2" under a header saying so.
            # Clipped, like the rendered text. A 2 MB single cell put two
            # megabytes into --json, into a cache entry, into a JSON.stringify
            # in the browser card and through the folding scan twice. An
            # example is for recognising the column, and 200 characters of one
            # is as much of that as anybody reads.
            "examples": [redact_value(v[:_MAX_EXAMPLE], name)
                         for v in values[:3]],
        }
        if col["type"] in ("integer", "float"):
            nums = [n for n in (_as_number(v) for v in values) if n is not None]
            if nums:
                # An integer column reports integers. "min: 1.0" is a value the
                # file does not contain, and it reaches the model as well as
                # the reader — one duly wrote "IDs run from 1.0 to 8.0".
                exact = col["type"] == "integer" and all(n.is_integer() for n in nums)
                col["min"] = int(min(nums)) if exact else min(nums)
                col["max"] = int(max(nums)) if exact else max(nums)
                col["mean"] = round(sum(nums) / len(nums), 4)
        cols.append(col)
    return cols


_INT = re.compile(r"^[-+]?\d+$")
#: A decimal comma ("1,5") and a thousands separator ("1,234") are written the
#: same way. Distinguished by shape: exactly three digits after a comma, with
#: more groups possible, is a separator; anything else is a decimal mark.
_GROUPED = re.compile(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$")
#: European grouping needs proof, because "3.141" is also just a decimal — and
#: reading it as 3141 is the same thousandfold error this whole block exists to
#: prevent, merely inverted. Proof is two groups ("1.234.567") or a decimal
#: comma after them ("1.234,56"). A lone "1.234" stays a decimal.
_GROUPED_EU = re.compile(r"^[-+]?\d{1,3}(\.\d{3}){2,}(,\d+)?$"
                         r"|^[-+]?\d{1,3}(\.\d{3})+,\d+$")
_DECIMAL_COMMA = re.compile(r"^[-+]?\d+,\d*$")
#: Exactly one comma in the whole number.
_ONE_COMMA = re.compile(r"^[-+]?\d+,\d+$")
_FLOAT = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")
_DATE = re.compile(
    r"^\d{4}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])"
    r"([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
    r"|^(0?[1-9]|[12]\d|3[01])[./](0?[1-9]|1[0-2])[./]\d{2,4}$")
_BOOL = {"true", "false", "yes", "no", "0", "1", "y", "n"}


def _looks_like_year_header(first: list[str], rest: list[list[str]]) -> bool:
    """True only for the one all-numeric header shape worth rescuing: years.

    "2020,2021,2022" over measurements is a real and common pivot-table layout,
    and reading those years as data drags every statistic. But a general
    "row one is an outlier" test is far too eager — a headerless file sorted
    descending has an outlier first row too, and promoting it to column names
    silently drops a record. So the rescue is narrow on purpose: every cell a
    plausible year, ascending or equal, and the body not years itself.
    """
    values = [_as_number(c) for c in first]
    if any(v is None or v != int(v) or not (1900 <= v <= 2100) for v in values):
        return False
    # Descending is the standard financial-comparatives ordering — 2024, 2023,
    # 2022 — and rejecting it put 2024 in the statistics. The 1900-2100 range
    # and "the body is not years" carry the discrimination on their own.
    below = [n for r in rest for n in (_as_number(c) for c in r) if n is not None]
    if not below:
        return False
    years_below = sum(1 for n in below if 1900 <= n <= 2100)
    return years_below < len(below) * 0.5


#: What a tool writes where a value is absent. pandas and numpy write "nan"
#: into a float column and R writes "NA" — but "NA" is also Namibia's country
#: code, "Na" is sodium, and "none" is a real priority level. A word is a
#: missing-value marker only where the rest of the column says so: among
#: numbers. Treating it as one everywhere erased Namibia from a country
#: column while the rows shown to the model still contained it.
_MISSING = re.compile(r"^\s*(?:nan|-?inf(?:inity)?|na|n/a|null|none|nil)\s*$",
                      re.IGNORECASE)
#: The unambiguous ones, which no ordinary value collides with.
_ALWAYS_MISSING = re.compile(r"^\s*(?:nan|-?inf(?:inity)?|n/a)\s*$",
                             re.IGNORECASE)


def _missing_markers(values: "list[str]") -> "re.Pattern":
    """Which markers count as missing in this column.

    A column whose other values are numbers is a numeric column with holes;
    one whose other values are words is a column of words, and "NA" is one of
    them.
    """
    real = [v for v in values if v.strip() and not _MISSING.match(v)]
    if not real and values:
        # Every value is a marker. That is the strongest evidence there is
        # that they are markers here — and the guard read it as the weakest,
        # so a column of nothing but "NA" was reported as three values of
        # text while a column of "nan" was reported as empty: two spellings
        # of "never collected", reported as opposites.
        return _MISSING
    numeric = sum(1 for v in real if _is_number(v))
    if real and numeric >= len(real) * 0.8:
        return _MISSING
    return _ALWAYS_MISSING


#: True while reading a file whose delimiter is not the comma. In such a file
#: a comma inside a field cannot be separating fields, and in every European
#: export it is the decimal mark — so "0,500" is half a kilo, not five
#: hundred. Read as a thousands separator it was, and a parts list reported
#: screws weighing 500 kg with a mean of 4,375. The neighbouring column in
#: the same file, whose values had two decimals, was read correctly, so the
#: file contradicted itself in sift's own output.
#:
#: A ContextVar rather than a module global: `sift serve` reads uploads on
#: one thread each, and a global would let a German price list change how an
#: American one two milliseconds later is read.
_decimal_comma_file: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "sift_decimal_comma_file", default=False)


@contextlib.contextmanager
def _numbers_written_with(delimiter: str):
    token = _decimal_comma_file.set(delimiter not in (",", ""))
    try:
        yield
    finally:
        _decimal_comma_file.reset(token)


@contextlib.contextmanager
def _numbers_written_with_commas(decimal: "bool"):
    token = _decimal_comma_file.set(decimal)
    try:
        yield
    finally:
        _decimal_comma_file.reset(token)


def _comma_means(values: "list") -> bool:
    """True when this column's commas are decimal marks.

    Asked of the column, because the evidence is in the column. A group of
    other than three digits ("0,45") can only be a decimal mark; two groups
    ("8,336,817") can only be grouping. Where every value is a single group
    of three the two readings are indistinguishable — "0,500" is half a kilo
    and "12,500" is twelve thousand five hundred — and only then does the
    file's delimiter decide, which is the best evidence left.
    """
    grouped = decimal = False
    for value in values:
        parts = value.strip().lstrip("+-").split(",")
        if len(parts) < 2:
            continue
        if len(parts) > 2:
            grouped = True                # 8,336,817 — two groups is grouping
        elif "." in parts[1]:
            grouped = True                # 1,234.50 — the dot is the decimal
        elif len(parts[1]) != 3 or not parts[1].isdigit():
            decimal = True                # 0,45 — two digits is a decimal
    # Proof of grouping outranks proof of a decimal mark, because one price
    # written with cents is enough to prove the first and a whole column of
    # three-digit groups proves neither. Tested the other way round, a single
    # "1,234.50" marked a column decimal and "12,000" beside it became 12.0
    # while "8,336,817" stayed whole — the column contradicting itself, which
    # is the failure this function exists to remove.
    # A decimal mark outranks grouping, not the other way round. One
    # US-formatted price in a German weights column re-read every value in
    # it: 0,500 became 500 and the mean went from 494 to 844, which is the
    # thousandfold error this function exists to remove, inverted. The
    # "1,234.50 is grouping" clause above is what the reorder was for, and it
    # does that job on its own.
    if decimal:
        return True
    if grouped:
        return False
    return _decimal_comma_file.get()


def _as_number(value: str):
    """Parse a number written in either convention, or return None."""
    v = value.strip()
    if not v:
        return None
    if _decimal_comma_file.get() and _ONE_COMMA.match(v):
        # One comma and nothing else: a decimal mark. Two or more groups
        # ("1,250,000") is still grouping — nobody writes two decimal marks.
        v = v.replace(",", ".")
    elif _GROUPED.match(v):
        v = v.replace(",", "")
    elif _GROUPED_EU.match(v):
        v = v.replace(".", "").replace(",", ".")
    elif _DECIMAL_COMMA.match(v):
        v = v.replace(",", ".")
    try:
        number = float(v)
    except ValueError:
        return None
    # "nan" and "inf" are what pandas, numpy and R write for a missing or
    # overflowed value, and float() accepts them — so a column holding one
    # produced a mean of nan, which json.dumps emits as the bare token NaN.
    # That is not JSON: jq, node, Go and every strict parser reject the whole
    # document, and the browser's res.json() throws. A missing value is
    # missing, which the profile already knows how to say.
    return number if math.isfinite(number) else None


def _is_number(v: str) -> bool:
    return _as_number(v) is not None


def _is_whole(v: str) -> bool:
    """True when the value is a whole number however it is written.

    Asked as a question about the value rather than as a ban on characters.
    Banning "," typed a quoted American "8,336,817" as a float; banning "."
    did the same to a German "3.664.088". Both are populations, and neither
    has a fractional part.
    """
    if _INT.match(v):
        return True
    if _GROUPED_EU.match(v) and "," not in v:
        return True                      # 1.234.567 — grouping, proven
    if not _decimal_comma_file.get() and _GROUPED.match(v) and "." not in v:
        return True                      # 8,336,817 — grouping, comma file
    return False


#: A number whose written form does not survive being read as one. A postal
#: code "01067" becomes 1067 and an article number "007" becomes 7 — and the
#: profile then reports a range ("1,067–10,115") that no value in the file
#: has, with no note that anything was lost.
#:
#: A leading zero only. The first version of this also called a column with
#: trailing zeros after the point a code, to catch versions like "2.10" — and
#: "1.50" and "14.00" are prices, which is the commonest numeric column
#: there is. A version and a price are not distinguishable by their digits,
#: so that half bought a real defect for a speculative one.
#:
#: This is a test against something measured rather than against a wordlist:
#: the question is not what the column is called, it is whether the digits
#: mean a quantity or a code, and a leading zero settles that.
def _survives_as_a_number(text: str) -> bool:
    body = text.strip().lstrip("+-")
    if not body:
        return True
    whole = body.split(".")[0].split(",")[0]
    return not (len(whole) > 1 and whole[0] == "0")


def _infer_type(values: list[str]) -> str:
    if not values:
        return "empty"
    s = [v.strip() for v in values]
    if any(not _survives_as_a_number(v) for v in s if v):
        # At least one value would be changed by reading it as a number, so
        # the column is codes. Reporting "integer, range 1,067–10,115" for
        # 01067/04103/10115 is a statement about numbers the file does not
        # contain.
        return "code"
    # In a file whose numbers use a decimal comma, "0,500" is not a grouped
    # integer — it is half. Calling the column "integer" while the values it
    # reports are 0.5 and 12.75 is a contradiction inside one line of output.
    if all(_is_whole(v) for v in s):
        return "integer"
    if all(_is_number(v) for v in s):
        return "float"
    if all(v.lower() in _BOOL for v in s):
        return "boolean"
    dates = sum(bool(_DATE.match(v)) for v in s)
    if dates == len(s):
        return "date"
    if dates >= len(s) * 0.8:
        # Mostly dates, but not all. Saying "date" hides the exceptions, which
        # are exactly what someone profiling a column needs to know about.
        return "date-mixed"
    return "string"


def _quote_example(value) -> str:
    text = " ".join(str(value).split())
    return f'"{text}"' if ("," in text or ";" in text or "\t" in text) else text


def _render(header, columns, body_rows, delimiter, total_rows, truncated) -> str:
    lines = ["# Tabular data"]
    lines.append(f"rows: {total_rows}{'+' if truncated else ''}, columns: {len(header)}")
    lines.append("")
    lines.append("## Schema")
    for col in columns:
        stat = ""
        if "min" in col:
            stat = f" [min={col['min']}, max={col['max']}, mean={col['mean']}]"
        # Quoted like the rows below: an example containing the
        # delimiter — "Doe, Jane" — otherwise reads as two examples,
        # and an embedded newline broke the schema list apart.
        ex = ", ".join(_quote_example(e) for e in col["examples"][:3])
        # Deliberately not "e.g.": that is a cue the injection check reads
        # as the file describing an example, and the renderer must not be
        # able to disarm a payload it is quoting out of a cell.
        # one_line on the name: csv lets a quoted header cell span lines, and
        # a two-line column name wrote its own "## First rows" and "## Note
        # from sift" blocks into the schema list — into the prompt, into
        # summ.summary, into the cache and the browser card, all with
        # warnings: []. _RENDERING_NOTE tells the model that ## headings are
        # sift's own layout, so the forged block inherits sift's authority.
        lines.append(f"- {one_line(col['name'])} ({col['type']}){stat}"
                     f" — sample: {ex}")
    lines.append("")
    lines.append("## First rows")
    # Re-quoted through csv, not joined by hand. A field containing the
    # delimiter — "Doe, Jane" — was written back out raw, so a 2-column row
    # reached the model as 5 fields, contradicting the "columns: 2" three
    # lines above; an embedded newline split one record across two lines.
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    for row in body_rows[:SAMPLE_ROWS]:
        writer.writerow(row)
    lines.append(buffer.getvalue().rstrip("\n"))
    return "\n".join(lines)
