"""SQLite — the tables, their shape, and how much is in them.

A .db or .sqlite file is a data file, which is what this tool is for, and it
came back as "Binary file, 96 B, entropy 1.194" — true of the bytes and
useless about the contents. Python ships the reader; not using it was the
only reason.

Read strictly read-only, through a URI connection with ``mode=ro`` and
``immutable=0``: sift must never write to a file it was asked to summarize,
and a journal or WAL replay on open would do exactly that. Extensions are
not loaded, so nothing in the file can execute; the risk in an untrusted
SQLite file is a slow query, which the per-file deadline bounds.

A database in WAL mode on a directory sift cannot write to is the one case
``mode=ro`` cannot open: its reader needs the ``-shm`` index and creates it
when it is missing. That file is read once more with ``immutable=1``, which
reads the main file alone, and says so.

Nothing is SELECTed from a user table beyond a row count and a handful of
sample values, because a summary is a description and not an export.
"""
from __future__ import annotations

import os
import re
import sqlite3

from .. import deadline
from ..detect import Detection
from .base import Extracted, clip, one_line, redact_value

MAX_CHARS = 120_000

#: How many tables are described, and how many rows are counted in each.
#: A schema with hundreds of tables is a real application database; the
#: point is to say what is in it, not to inventory it.
_MAX_TABLES = 64
_MAX_SAMPLE_ROWS = 3
#: COUNT(*) walks the table. Above this many rows the count is reported as a
#: lower bound rather than paid for.
_COUNT_LIMIT = 200_000


def extract(path, det: Detection, max_bytes: int) -> Extracted:
    # resolve() first: as_uri() raises on a relative path, the registry's
    # blanket except turned that into "extractor failed", and the heuristic
    # then stated "SQLite database with 0 tables" as a fact about the file.
    # Every test used a tempfile, which is absolute, so the extractor was
    # dead for `sift my.db` — the ordinary way a CLI is invoked — and no
    # test could see it.
    location = f"file:{path.resolve().as_uri()[7:]}"
    ex, error = _attempt(location + "?mode=ro", det)
    if error is None or not (_WAL_CANNOT_OPEN.search(str(error))
                             and _in_wal_mode(path)):
        return ex
    # A WAL database on a directory sift may not write to. Measured: a
    # cleanly closed one fails with "attempt to write a readonly database",
    # one whose -wal was left without its -shm with "unable to open database
    # file", and the whole file was reported unreadable. The main file is
    # intact; only the log is out of reach, and saying that is the truth.
    # If this attempt fails too, its reason is the one reported: it got past
    # the log, so what stopped it is about the file.
    ex, error = _attempt(location + "?mode=ro&immutable=1", det)
    if error is None and _log_has_content(path):
        ex.note(_WITHOUT_THE_LOG)
    return ex


def _log_has_content(path) -> bool:
    """Whether a -wal beside the database holds anything to leave out.

    A cleanly closed database has none: the last close checkpoints the log
    and removes it, and "changes not yet checkpointed are not included"
    said something was missing from a file that was complete. A log that
    cannot be looked at is counted as present — the note then errs on the
    side of saying too much.
    """
    try:
        return os.stat(f"{path}-wal").st_size > 0
    except FileNotFoundError:
        return False
    except OSError:
        return True


#: The two ways mode=ro fails on a WAL database whose -shm it cannot create.
_WAL_CANNOT_OPEN = re.compile(r"unable to open database file|"
                              r"attempt to write a readonly database")
_WITHOUT_THE_LOG = ("read without the write-ahead log; changes not yet "
                    "checkpointed are not included")


def _in_wal_mode(path) -> bool:
    """Whether the header says WAL: bytes 18 and 19, the write and read
    format versions, are 2 in WAL mode and 1 with a rollback journal.

    Only then is immutable=1 an honest fallback. A rollback database that
    fails the same way may have a hot journal, and reading past that is not
    "without the log" — it is reading a half-written transaction.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    return len(header) == 20 and header[18] == 2 and header[19] == 2


def _attempt(uri: str, det: Detection) -> "tuple[Extracted, object]":
    """One read: the description and None, or the failure and its error.

    Opening is lazy — connect() succeeds on a WAL database it cannot read
    and the refusal arrives with the first query — so the caller needs the
    error from either step to decide whether another attempt makes sense.
    """
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error as exc:
        return Extracted("sqlite", det.mime, text="",
                         metadata={"parse_error": str(exc)[:200]},
                         notes=[f"the database could not be opened: {exc}"]), exc
    try:
        with connection:
            return _describe(connection, det), None
    except sqlite3.DatabaseError as exc:
        # A truncated or corrupt file, or one that is not really SQLite past
        # its header. Saying so is more use than a byte count.
        return Extracted("sqlite", det.mime, text="",
                         metadata={"parse_error": str(exc)[:200]},
                         notes=[f"the database is unreadable past its header: "
                                f"{exc}"]), exc
    finally:
        connection.close()


def _describe(connection, det: Detection) -> Extracted:
    connection.text_factory = lambda b: b.decode("utf-8", "replace")
    cursor = connection.cursor()
    objects = cursor.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()

    tables = [(name, sql) for kind, name, sql in objects if kind == "table"]
    views = [name for kind, name, _ in objects if kind == "view"]
    indexes = [name for kind, name, _ in objects if kind == "index"]
    triggers = [name for kind, name, _ in objects if kind == "trigger"]

    lines = ["# SQLite database"]
    described = []
    # Per-table failures were swallowed: a table whose pages are corrupt
    # counted as 0 rows and sampled as nothing, and the summary said "0+
    # rows" in the same voice it uses for an empty table. Those are opposite
    # facts. Collected here and reported once, so a database that is partly
    # unreadable says so.
    unreadable: "list[str]" = []
    for name, _sql in tables[:_MAX_TABLES]:
        deadline.check("a database")
        columns = _columns(cursor, name)
        rows = _row_count(cursor, name)
        if rows.get("failed") and name not in unreadable:
            unreadable.append(name)
        entry = {"name": name, "columns": [c["name"] for c in columns],
                 "column_types": {c["name"]: c["type"] for c in columns},
                 "rows": rows["value"], "rows_exact": rows["exact"]}
        sample, sample_failed = _sample(cursor, name, columns)
        if sample_failed and name not in unreadable:
            unreadable.append(name)
        if sample:
            entry["examples"] = sample
        described.append(entry)

        shape = (f"{rows['value']}{'+' if not rows['exact'] else ''} row"
                 f"{'s' if rows['value'] != 1 else ''}")
        # one_line on every name. SQLite quotes identifiers, so a table may
        # be called anything at all — including several lines — and a name
        # with newlines in it wrote its own "## Table geheim (999 rows, 2
        # columns)" block, with rows sift never read and a column called
        # "freigabe" holding "genehmigt".
        lines.append(f"\n## Table {one_line(name)} ({shape}, "
                     f"{len(columns)} columns)")
        lines.append(", ".join(
            f"{one_line(c['name'])} ({one_line(c['type']) or 'untyped'})"
            for c in columns) or "(no columns)")
        for row in sample:
            lines.append(" | ".join(one_line(cell) for cell in row))

    if views:
        # In the rendered text as well as in the metadata. The model read the
        # metadata, saw a view named there and nothing about it in the
        # content, and said so — which was a fair complaint about a file
        # description that mentioned something it never showed.
        lines.append("\n## Views")
        for kind, name, sql in objects:
            if kind != "view":
                continue
            definition = " ".join((sql or "").split())[:200]
            lines.append(f"{one_line(name)}: {definition}"
                         if definition else one_line(name))

    meta = {
        "tables": described,
        "table_count": len(tables),
        "view_count": len(views),
        "index_count": len(indexes),
        "trigger_count": len(triggers),
    }
    if views:
        meta["views"] = views[:20]
    meta = {k: v for k, v in meta.items() if v}

    body, truncated = clip("\n".join(lines), MAX_CHARS)
    ex = Extracted("sqlite", det.mime, text=body, metadata=meta,
                   rendered=True, truncated=truncated, reads_whole_file=True)
    if len(tables) > _MAX_TABLES:
        ex.note(f"{len(tables)} tables; the first {_MAX_TABLES} are described")
    if unreadable:
        shown = ", ".join(one_line(n) for n in unreadable[:8])
        ex.metadata["unreadable_tables"] = unreadable[:32]
        ex.note(f"{len(unreadable)} table(s) could not be read — {shown}"
                f"{' …' if len(unreadable) > 8 else ''}; any row count shown "
                f"for them is not a count of their rows")
    if triggers:
        # Worth saying: a trigger is code the database runs, and a reader
        # deciding whether a file is inert should know it is not.
        ex.note(f"the database defines {len(triggers)} trigger(s), which are "
                f"statements it runs on its own when rows change")
    return ex


def _columns(cursor, table: str) -> "list[dict]":
    """Column names and declared types, via PRAGMA rather than a SELECT."""
    try:
        info = cursor.execute(
            f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")'
        ).fetchall()
    except sqlite3.DatabaseError:
        return []
    return [{"name": row[1], "type": (row[2] or "").upper()} for row in info]


def _row_count(cursor, table: str) -> dict:
    """The row count, or a lower bound when counting would cost too much.

    COUNT(*) walks the table. On a multi-gigabyte log table that is the whole
    file read for one number, and the number is not worth it — but silently
    reporting a capped count as the count would be a stated falsehood, so the
    cap is reported as one.
    """
    quoted = table.replace('"', '""')
    try:
        capped = cursor.execute(
            f'SELECT COUNT(*) FROM (SELECT 1 FROM "{quoted}" '
            f'LIMIT {_COUNT_LIMIT + 1})').fetchone()[0]
    except sqlite3.DatabaseError:
        return {"value": 0, "exact": False, "failed": True}
    if capped > _COUNT_LIMIT:
        return {"value": _COUNT_LIMIT, "exact": False}
    return {"value": capped, "exact": True}


def _sample(cursor, table: str,
            columns: "list[dict]") -> "tuple[list[list[str]], bool]":
    """A few rows, as text, with credentials hidden by their column name.

    The same rule the CSV profile uses: the column's own name is the key its
    values are judged by, and this is the only place both are in hand. A
    "users" table has a "password_hash" column more often than not.
    """
    if not columns:
        return [], False
    quoted = table.replace('"', '""')
    try:
        rows = cursor.execute(
            f'SELECT * FROM "{quoted}" LIMIT {_MAX_SAMPLE_ROWS}').fetchall()
    except sqlite3.DatabaseError:
        return [], True
    out = []
    for row in rows:
        cells = []
        for column, value in zip(columns, row):
            if value is None:
                cells.append("NULL")
                continue
            text = str(value)
            if len(text) > 60:
                text = text[:60] + "…"
            cells.append(str(redact_value(text, column["name"])))
        out.append(cells)
    return out, False
