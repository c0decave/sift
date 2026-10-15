"""Extractor for PDF.

Uses ``pypdf`` when installed (reliable). Falls back to a pure-stdlib best-effort
text recovery that inflates FlateDecode streams and pulls text-showing operators
— good enough for many simple PDFs, and honest about its limits via a note.
"""
from __future__ import annotations

import collections
import contextlib
import contextvars
import gc
import hashlib
import io
import itertools
import logging
import operator
import re
import sys
import threading
import zlib
from itertools import repeat
from pathlib import Path

from ..detect import Detection
from .. import deadline
from .base import Extracted, clip

MAX_CHARS = 200_000


def extract(path: Path, det: Detection, max_bytes: int) -> Extracted:
    # Honour the cap: --max-bytes was silently ignored here, so a 2 GB PDF was
    # read whole however small the limit said.
    with path.open("rb") as fh:
        data = fh.read(max_bytes) if max_bytes > 0 else fh.read()
    byte_capped = det.size > len(data)
    text, meta, notes, pages_cut = _with_pypdf(data)
    if text is None:
        # Why pypdf could not read it is said beside the fallback's own note,
        # not replaced by it.
        text, meta, note = _best_effort(data, det.size,
                                        pypdf_failed=bool(notes))
        notes = notes + [note] if note else notes
    body, truncated = clip(text or "", MAX_CHARS)
    # rendered: page structure, fonts and positioning are gone; what is left
    # is the text. Same reason as html and rtf.
    ex = Extracted("pdf", det.mime, text=body, metadata=meta, rendered=True,
                   truncated=truncated or byte_capped or pages_cut)
    if byte_capped:
        # A PDF's cross-reference table is at the end. Cut it off and the page
        # count and text are of a fragment, not of the document.
        ex.note(f"read only the first {len(data)} of {det.size} bytes; page "
                f"count and text cover that part")
    for note in notes:
        ex.note(note)
    if not body.strip():
        ex.note("no extractable text — this PDF may be scanned images (OCR needed)")
    return ex


def _with_pypdf(data: bytes):
    """(text, metadata, notes, whether a page's or a form's content was cut
    short).

    text is None when pypdf is missing or failed; the notes then say why.
    """
    try:
        import pypdf  # type: ignore
    except ModuleNotFoundError:
        return None, {}, [], False
    del pypdf                   # whether it imports: _open reads with it
    reader = None
    structure = _Structure(_structure_budget(len(data)),
                           _entry_budget(len(data)))
    try:
        with _counting_pypdf_log() as logged:
            structure.logged = logged
            reader = _open(data, structure)
            text, meta, notes, cut = _read_pages(reader, len(data), logged)
            return text, meta, _structure_notes(structure) + notes, cut
    except deadline.DeadlineExceeded:
        # The budget ran out; the caller turns this into a summary that says
        # so, not an ordinary "pypdf failed" that would fall to the stdlib
        # reader and run just as long again.
        raise
    except _HeldBound:
        # Walking the page tree, before a page was read (_read_pages).
        return None, {}, [f"{_held_said(structure)}, walking its page tree; "
                          f"used fallback", *_structure_notes(structure)], False
    except _ReaderBound:
        # A read of the file once the time budget was spent (_Alarm), while
        # a page was read: pypdf was parsing it a byte at a time.
        raise deadline.DeadlineExceeded(deadline.seconds() or 0.0,
                                        "a PDF") from None
    except _Unsupported as exc:
        return None, {}, [f"{exc}; used fallback"], False
    except Exception as exc:  # noqa: BLE001 - pypdf raises varied errors
        return None, {}, [f"pypdf failed ({exc}); used fallback",
                          *_structure_notes(structure)], False
    finally:
        structure.disarm()
        structure.logged = None
        structure.opened = []           # what refers back to it (_Entries)
        for number in list(structure.kept):
            structure.let_go(number)
        _close(reader)


def _open(data, structure):
    """A reader of ``data``: pypdf's own, as sift reads a document with it
    (_reader_class), counting what it decodes, lists and searches of the
    document's structure in ``structure`` (a _Structure)."""
    import pypdf  # type: ignore
    return _reader_class(pypdf)(data, structure)


def _close(reader):
    """Let go of what ``reader`` holds once its document is read: what it
    parsed and what it knows of the file (PdfReader.close in 6.19.0), and
    what it points to itself through -- its pages and its root, whose
    references hold the reader. Kept, the reader was a reference cycle that
    only a full collection freed, and it held every stream pypdf decoded of
    the document's structure: 892 MB after sift returned, 52 MB once
    collected."""
    close = getattr(reader, "close", None)
    if close is None:
        return
    structure = getattr(reader, "structure", None)
    if structure is not None:
        structure.closed = True     # what close() empties is not charged
    try:
        close()
    except Exception:  # noqa: BLE001 - what is left goes with the collector
        pass
    for name in ("_validated_root", "_page_id2num",
                 "_named_destinations_cache"):
        if name in vars(reader):
            setattr(reader, name, None)


class _Structure:
    """What pypdf does with one document's structure, against what sift lets
    it do: what its cross-reference streams and object streams decode to,
    against ``budget`` bytes (_structure_budget) -- ``spent`` so far, whether
    a stream would have decoded past what was left (``over``), the error
    pypdf raised there (``fired``), and the object streams whose parse passed
    OBJECT_STREAM_READS_A_BYTE (``refused``, by number); the entries of
    objects in use its cross-reference sections make it hold, against
    ``entries`` (_entry_budget) -- ``listed`` so far, each object once, and
    whether one was left unread (``listed_over``); the file's bytes
    (``data``) and its index of "N G obj" (_Found), which answers pypdf's
    searches of the whole file for an object its tables lack or misplace;
    the copies of the whole file pypdf still takes -- to recover a table's
    line, to search past the index -- how many (``searched``) and their
    bytes (``searched_bytes``), ``searches_over`` past MAX_SEARCHED, and the
    objects searched for so and not found (``absent``, (generation,
    number)); the _LogCount of what pypdf logs meanwhile (``logged``); and
    the streams pypdf reads the file from, which are tripped once the time
    budget is spent (_Alarm), or, while the reader is made, the log bound
    passed (``tripped``, why); and what the reader holds of the document,
    ``held`` against ``bound`` (MAX_DOCUMENT_HELD) -- its entries, the
    objects it parses and keeps, the pages it walks, the index, the object
    streams kept decoded (``kept``) --, the most it held (``most``), and the
    entries a rebuild of a lost table wrote (``written``, ``written_over``)."""

    __slots__ = ("budget", "spent", "over", "fired", "refused", "entries",
                 "listed", "listed_over", "building", "data", "found",
                 "searched", "searched_bytes", "searches_over", "absent",
                 "logged", "streams", "tripped", "alarm", "lock", "bound",
                 "held", "_most", "kept", "kept_bytes", "written",
                 "written_over", "depth", "root", "reads", "room",
                 "pages_charged", "closed", "opened")

    def __init__(self, budget, entries):
        self.budget, self.spent, self.over, self.fired = budget, 0, False, None
        self.refused, self.entries, self.listed = set(), entries, 0
        self.listed_over, self.building = False, False
        self.data, self.found = b"", None
        self.searched, self.searched_bytes, self.searches_over = 0, 0, False
        self.absent, self.logged = set(), None
        self.streams, self.tripped, self.alarm = [], None, None
        self.lock = threading.Lock()
        # What the reader holds of the document (hold), against ``bound``,
        # MAX_DOCUMENT_HELD; the most it held (``most``); the object streams
        # kept decoded (keep), and what they are charged; the entries a
        # rebuild of a lost table wrote (write), and whether one was not.
        self.bound, self.held, self._most = MAX_DOCUMENT_HELD, 0, 0
        self.kept, self.kept_bytes = collections.OrderedDict(), 0
        self.written, self.written_over = 0, False
        # The parse under way (begin): how deep, the frame it began in, what
        # its reads would hold at the worst case (READ_HELD), and up to where
        # they may before it is walked (in_flight); the pages walked so far
        # that are charged (walked); whether the reader is let go of (_close).
        self.depth, self.root, self.reads, self.room = 0, None, 0, 0
        self.pages_charged, self.closed, self.opened = 0, False, []

    @property
    def most(self):
        """The most the reader held of the document at any time, in bytes as
        charged."""
        return max(self._most, self.held)

    def hold(self, amount):
        """Charge ``amount`` bytes the reader holds of the document from now
        on: past ``bound`` _HeldBound ends pypdf's reading of it."""
        self.held += amount
        if self.held > self.bound:
            raise _HeldBound()

    def give_back(self, amount):
        """What was charged of something the reader no longer holds."""
        self._most = max(self._most, self.held)
        self.held -= amount

    def keep(self, number, kept):
        """Keep object stream ``number`` decoded (_Kept), charged already
        (_kept_stream), and let go of those used longest ago while what is
        kept passes MAX_OBJECT_STREAMS_HELD: an object of one is decoded again
        (and charged again to ``budget``) when it is asked for next."""
        self.kept[number] = kept
        self.kept_bytes += kept.charged
        while self.kept_bytes > MAX_OBJECT_STREAMS_HELD and len(self.kept) > 1:
            self.let_go(next(iter(self.kept)))

    def let_go(self, number):
        """Let go of object stream ``number``, where it is kept."""
        kept = self.kept.pop(number, None)
        if kept is not None:
            self.kept_bytes -= kept.charged
            self.give_back(kept.charged)

    def begin(self, root):
        """A parse begins in frame ``root`` whose reads are counted (counting,
        _ObjectStreamBytes): the outermost one sets where the frames of the
        parse under way end (in_flight), and what its reads may hold at the
        worst case before it is walked: what is left of ``bound``."""
        if not self.depth:
            self.root, self.reads = root, 0
            self.room = self.bound - self.held
        self.depth += 1

    def end(self):
        """The parse begun last is done."""
        self.depth -= 1
        if not self.depth:
            self.root = None

    def in_flight(self):
        """The parse under way read what it would hold at the worst case up to
        ``room``: what it holds is walked (_in_flight_held), and past ``bound``
        with what is held, it ends (_HeldBound); else it may read what is left
        of ``bound`` at the worst case before it is walked again, and at least
        IN_FLIGHT_READS reads: each walk costs what the parse holds, and one
        near the bound would be walked at every read. Between two walks the
        parse passes the bound by READ_HELD x IN_FLIGHT_READS at most."""
        left = self.bound - self.held - _in_flight_held(self.root)
        if left < 0:
            raise _HeldBound()
        self.room = self.reads + max(left, READ_HELD * IN_FLIGHT_READS)

    def walked(self, pages):
        """Charge the pages pypdf walked of the page tree since last asked
        (``pages``, its flattened_pages): what each copy of a page's
        dictionary holds (_page_held)."""
        pages = pages or ()
        for page in pages[self.pages_charged:]:
            self.hold(_page_held(page))
        self.pages_charged = len(pages)

    def counting(self, stream):
        """Count each read of ``stream`` from now on (_CountedFile) as what the
        parse under way (begin) may hold of it at the worst case: True where
        this call made it so. A read counted costs pypdf half as much again as
        one of io.BytesIO, and only reads of a parse are counted:
        an object resolved, a trailer, a cross-reference stream, an object a
        rebuild of a lost table reads."""
        with self.lock:
            counted = _COUNTED.get(type(stream))
            if counted is None:
                return False            # counted already, or tripped
            stream.__class__ = counted
            return True

    def uncounting(self, stream):
        """``stream``'s reads io.BytesIO's again, where it was not tripped
        meanwhile."""
        with self.lock:
            plain = _UNCOUNTED.get(type(stream))
            if plain is not None:
                stream.__class__ = plain

    def write(self):
        """Count an entry a rebuild of a lost table writes of an object stream,
        new or not, against ``entries``, after the objects the rebuild found
        in the file: past it _EntriesFull ends the stream's listing. Objects
        held are charged once (allot); what the rebuild does is counted each
        time, as 0.1.58 counted it -- "0 0 0 ..." in an object stream wrote
        entry 0 over and over, charged once, and an 11 MB file ran to the 60
        s budget where 0.1.58 took 3.05 s."""
        if self.written >= self.entries:
            self.written_over = True
            raise _EntriesFull()
        self.written += 1

    @contextlib.contextmanager
    def decoding(self):
        """pypdf's configuration with what one stream may decode to lowered
        to what is left of the budget -- never raised above what the
        configuration in force allows, 0 being no limit to pypdf. Past it
        pypdf raises LimitReachedError decoding the stream, as at its own
        limit, and the budget is spent: the next stream stops at its first
        byte rather than at what was left again.

        Only past a limit sift lowered: pypdf raises LimitReachedError at
        each of its own limits, and one of those -- /Columns past
        flate_maximum_columns in an older cross-reference stream -- spent the
        budget, and the document pypdf alone reads went to the fallback."""
        from pypdf import apply_configuration, get_configuration  # type: ignore
        from pypdf.errors import LimitReachedError  # type: ignore
        left = max(self.budget - self.spent, 1)
        current = get_configuration()
        lowered = {name: min(left, getattr(current, name) or left)
                   for name in _DECODED_LIMITS}
        try:
            with apply_configuration(**lowered):
                yield
        except LimitReachedError as exc:
            if str(exc).startswith(_OUTPUT_LIMITS) and any(
                    lowered[name] < (getattr(current, name) or left + 1)
                    for name in _DECODED_LIMITS):
                self.over, self.spent, self.fired = True, self.budget, exc
            raise

    def charge(self, stream):
        """Charge what ``stream`` decoded to and drop pypdf's decoded copy of
        it: pypdf has read what it needs of a cross-reference stream, and
        caches every object of an object stream it parsed."""
        decoded = getattr(stream, "decoded_self", None)
        if decoded is not None:
            self.spent += len(decoded.get_data())
            stream.decoded_self = None

    def allot(self):
        """Count an entry pypdf is about to make of an object in use while the
        reader is made (_Entries): past ``entries`` it is not made, and
        _EntriesFull says so. What it holds is charged as it is made."""
        if not self.building:
            return
        if self.listed >= self.entries:
            self.listed_over = True
            raise _EntriesFull()
        self.listed += 1

    def watch(self):
        """While the reader is made: end it once the time budget is spent,
        or what pypdf logged passed MAX_OP_LOG_PER_DOCUMENT (_ReaderBound)."""
        if deadline.spent():
            raise _ReaderBound("deadline")
        logged = self.logged
        if logged is not None and logged.over:
            raise _ReaderBound("log")

    def search(self, size, building):
        """Whether pypdf may copy and search the whole file, of ``size``
        bytes, once more, and count it: within MAX_SEARCHED bytes in all,
        each past the time budget ending it -- the reader being made
        (``building``), or the page."""
        if building:
            self.watch()
        else:
            _page_check()
        if self.searched_bytes + size > MAX_SEARCHED:
            self.searches_over = True
            return False
        self.searched += 1
        self.searched_bytes += size
        return True

    def find(self, number, generation):
        """(start, end) of the first "number generation obj" of the file as
        pypdf's search of it finds it -- re.search(rf"\\s{number}\\s+"
        rf"{generation}\\s+obj", file) in PdfReader.get_object -- or None:
        answered from the document's index (_Found), made at the first such
        search; past what the index holds, and for a number no index key can
        be, searched for so, within MAX_SEARCHED."""
        if self.building:
            self.watch()
        else:
            _page_check()
        if self.found is None:
            # At most what is left of ``bound`` at its peak (INDEX_KEY_PEAK a
            # key), and charged what it holds once made (_index_held): at its
            # cap the index held 112 MB, uncounted. Past it,
            # pypdf's search answers, within MAX_SEARCHED.
            cap = min(_index_budget(len(self.data)),
                      max(self.bound - self.held, 0) // INDEX_KEY_PEAK)
            self.found = _Found(self.data, cap,
                                self.watch if self.building else _page_check)
            self.hold(_index_held(self.found))
        answer = self.found.get(number, generation)
        if answer is not _Found.UNKNOWN:
            return answer
        key = (generation, number)
        if key in self.absent:
            return None
        start = self.found.upto if _Found.keyed(number, generation) else 0
        if not self.search(len(self.data) - start, self.building):
            self.absent.add(key)
            return None
        match = re.compile(rf"\s{number}\s+{generation}\s+obj".encode()
                           ).search(self.data, start)
        if match is None:
            self.absent.add(key)
            return None
        return match.start(0), match.end(0)

    def register(self, stream):
        """``stream`` is what pypdf reads the file from now: tripped at once
        where the budget or the log bound already was (trip)."""
        with self.lock:
            self.streams.append(stream)
            if self.tripped is not None:
                _trip(stream)

    def trip(self, why):
        """From now on each read pypdf makes of the file raises _ReaderBound
        (``why``): called by the _Alarm's thread once the time budget is
        spent, or by the _LogCount once what pypdf logged passed the bound
        while the reader is made."""
        with self.lock:
            if self.tripped is None:
                self.tripped = why
            for stream in self.streams:
                _trip(stream)

    def arm(self):
        """Trip the streams once the time budget in force is spent, where
        one is (_Alarm)."""
        left = deadline.remaining()
        if left is not None and self.alarm is None:
            self.alarm = _Alarm(left, self.trip)

    def disarm(self):
        alarm, self.alarm = self.alarm, None
        if alarm is not None:
            alarm.cancel()


def _page_check():
    """What a page's reading looks at before sift searches for pypdf: the
    time budget, raised past as a read of the file then raises (_trip) --
    not as DeadlineExceeded, an Exception pypdf would catch around a page's
    parts and read on."""
    if deadline.spent():
        raise _ReaderBound("deadline")


class _Alarm:
    """A timer that trips a document's streams (_Structure.trip) when its
    time budget is spent. pypdf reads the file a byte at a time -- the
    /Annots of 4,000,000 references read from the file for 21 s past a 3 s
    budget once the reader was made, where nothing watched a read -- and a read that looked at the clock cost the reader's making
    twice what pypdf alone took: 3.09 to 3.79 s for cl20.pdf, 1.65 to 2.14 s
    unwatched. Tripped, a stream's class is one whose read
    raises; until then its reads are io.BytesIO's own."""

    __slots__ = ("timer",)

    def __init__(self, seconds, trip):
        self.timer = threading.Timer(seconds, trip, ("deadline",))
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        self.timer.cancel()


class _ReaderBound(BaseException):
    """Making the reader passed the time budget, the log bound, or the
    entries a classic cross-reference table may list (``why``); or a read
    of the file came after the time budget was spent (_Alarm), making the
    reader or reading a page. A BaseException, as _ParseBound is one: pypdf
    catches every Exception reading a cross-reference section it cannot
    read, logs it and reads on (PdfReader._read_xref_tables_and_trailers in
    6.19.0)."""

    def __init__(self, why):
        super().__init__(why)
        self.why = why


class _EntriesFull(_ReaderBound):
    """pypdf was about to make an entry past the entry bound (_Structure.
    allot): a cross-reference stream's rest is not read (_read_xref_
    subsections), a rebuild lists no more, and a classic table, which
    cannot be read in part, ends the reader."""

    def __init__(self):
        super().__init__("entries")


class _HeldBound(_ReaderBound):
    """What the reader holds of the document passed MAX_DOCUMENT_HELD
    (_Structure.hold), as pypdf made it: opening the document, the reader is
    not made (_Unread); walking its page tree, sift does not read with pypdf;
    reading a page, the document ends there (_read_pages)."""

    where = None        # reading which page (_read_each)

    def __init__(self):
        super().__init__("memory")


def _held_said(structure):
    """What sift says of a document whose reader passed ``structure.bound``,
    before where."""
    return (f"pypdf would hold more than {structure.bound} bytes of this "
            f"document's objects -- half of the {WORKER_MEMORY} bytes a worker "
            f"reads a document in --")


class _Unsupported(Exception):
    """The pypdf installed lacks what sift bounds its reading with."""


#: pypdf's configuration fields that bound what one stream decodes to
#: (pypdf/_configuration.py in 6.19.0), those of the filters a document's
#: structure may be written with.
_DECODED_LIMITS = ("zlib_maximum_output_length", "lzw_maximum_output_length",
                   "run_length_maximum_output_length",
                   "array_based_stream_maximum_output_length")

#: How pypdf's LimitReachedError begins where one of _DECODED_LIMITS fired
#: (pypdf/filters.py:97, :164, :492, _codecs/_codecs.py:264 and
#: generic/_data_structures.py:1260 in 6.19.0 and 6.18.1), and no other of
#: its limits: "Recovery limit reached", "Number of columns", "Row length",
#: "Declared stream length" begin otherwise.
_OUTPUT_LIMITS = ("Limit reached while decompressing",
                  "Array-based stream has at least")


def _structure_budget(size):
    """What the cross-reference and object streams of a document of ``size``
    bytes may decode to in all (STRUCTURE_BYTES_A_FILE_BYTE)."""
    return max(MAX_INFLATED, STRUCTURE_BYTES_A_FILE_BYTE * size)


def _entry_budget(size):
    """How many entries the cross-reference sections of a document of
    ``size`` bytes may list in all (XREF_FILE_BYTES_AN_ENTRY,
    MAX_XREF_ENTRIES)."""
    return min(MAX_XREF_ENTRIES, size // XREF_FILE_BYTES_AN_ENTRY)


def _index_budget(size):
    """How many objects the index of "N G obj" of a document of ``size``
    bytes may hold (_Found): as many as its cross-reference sections may
    make pypdf hold (_entry_budget), each less than XREF_ENTRY_HELD."""
    return _entry_budget(size)


def _structure_notes(structure):
    """What sift says of what it did not let pypdf decode or parse of a
    document's structure, counted in ``structure`` (_Structure)."""
    notes = []
    if structure.over:
        notes.append(
            f"pypdf's cross-reference and object streams of this document "
            f"decode to more than {structure.budget} bytes, "
            f"{STRUCTURE_BYTES_A_FILE_BYTE} times its size; what is past "
            f"that was not read")
    if structure.refused:
        count = len(structure.refused)
        notes.append(
            f"an object stream pypdf parses past {OBJECT_STREAM_READS_A_BYTE}"
            f" reads a byte was not read further"
            if count == 1 else
            f"{count} object streams pypdf parses past "
            f"{OBJECT_STREAM_READS_A_BYTE} reads a byte were not read further")
    if structure.listed_over:
        notes.append(
            f"pypdf's cross-reference sections of this document list more "
            f"than {structure.entries} objects in use -- one a "
            f"{XREF_FILE_BYTES_AN_ENTRY} bytes of it, {MAX_XREF_ENTRIES} at "
            f"most --; those past that were not read")
    if structure.written_over:
        notes.append(
            f"pypdf rebuilt this document's lost cross-reference table "
            f"writing more than {structure.entries} entries -- one a "
            f"{XREF_FILE_BYTES_AN_ENTRY} bytes of it --; what is past that "
            f"was not read")
    if structure.searches_over:
        notes.append(
            f"pypdf searched this document's whole file {structure.searched} "
            f"times, {MAX_SEARCHED} bytes at most, for objects its "
            f"cross-reference tables lack; objects it looked for after that "
            f"were taken as absent")
    return notes


#: The reader class sift reads with, or what the pypdf lacks that sift
#: needs of it, per (pypdf, its PdfReader).
_READERS = {}

#: What sift reads with of pypdf's module and of its PdfReader: all of it is
#: in pypdf 6.18.0 and after, the configuration API only there (read in the
#: sources of 6.8.0, 6.14.2, 6.16.2, 6.17.0, 6.18.0, 6.18.1 and 6.19.0).
_PYPDF_NEEDED = ("apply_configuration", "get_configuration")
_READER_NEEDED = ("_read_pdf15_xref_stream", "_read_xref",
                  "_read_xref_subsections", "_get_object_from_stream",
                  "_rebuild_xref_table", "_find_pdf_objects",
                  "_find_pdf_trailers", "_XREF_STREAM_TRAILER_KEYS",
                  "get_object", "cache_get_indirect_object",
                  "cache_indirect_object", "read_object_header")


def _reader_class(pypdf):
    """pypdf.PdfReader, with what it does with a document's structure
    bounded (_Structure): each cross-reference stream decodes within what is
    left of the budget and is charged, an object stream the same, and its
    objects are parsed from bytes that watch the deadline and
    OBJECT_STREAM_READS_A_BYTE (_ObjectStreamBytes); the entries of objects
    in use the cross-reference sections make it hold are counted as it makes
    them, within the entry bound (_Entries); each read of the file raises
    once the time budget is spent (_Alarm), and, while the reader is made,
    once the log bound is passed; a table pypdf rebuilds is rebuilt under
    all of these (_rebuild); and an object the tables lack or misplace is
    looked up in the document's index of "N G obj" (_Found), made once,
    where pypdf copied and searched the whole file each time (_get_object).

    pypdf reads every cross-reference stream while the reader is made,
    before sift reads a page, decoded to its own limit of 75,000,000 bytes
    and kept (pypdf/_reader.py:1227-1273 in 6.19.0): 12 of them in an 839 KB
    file held 949 MB, 48 of them 3.5 GB. It reads an object
    stream whole the first time an object of it is asked for, and parses
    every object in it in one call (_reader.py:357-454), skipping the white
    space after each a byte at a time: 20 objects at one offset, followed by
    73 MB of zeros, ran 47 s past a 5 s budget. It makes an
    entry of about 170 bytes of every two bytes a cross-reference stream
    decodes to (_reader.py:1454-1487), and looks at the offset of every
    entry of type 1 while the reader is made, logging each that holds no
    object (:733-751): 12,000,000 entries took 2,090 MB, 399,000 of them in
    a 1 KB file 6.73 s past a 5 s budget and 19.8 M characters of log, with
    nothing watching the time or the log until the first page.

    Where pypdf lacks what sift bounds it with (_PYPDF_NEEDED,
    _READER_NEEDED), sift does not read with it, and says so (_Unsupported):
    through 0.1.55 a PdfReader lacking one of the methods was read with
    plain, bounding none of the above, and pypdf 6.14.2, which has the
    methods but not the configuration API, lost 62 of 111 real documents to
    the fallback, each noted as "pypdf failed (... cannot import name
    'apply_configuration' ...)"."""
    base = pypdf.PdfReader
    kind = _READERS.get((pypdf, base))
    if kind is None:
        kind = _READERS[(pypdf, base)] = _bounded_reader(pypdf, base)
    if isinstance(kind, str):
        raise _Unsupported(kind)
    return kind


def _older_than_the_floor(version) -> bool:
    """Whether the pypdf ``version`` names is older than PYPDF_FLOOR, as PEP
    440 orders versions (_version_key): a pre-release or a development
    release of the floor is older than it, as pip orders it; False for a
    version PEP 440 cannot read. 0.1.59 read the leading numbers and a
    pre-release written without a separator: "6.18.1-rc1", "6.18.1-dev0"
    and "v6.18.0" were read with."""
    have = _version_key(version)
    return have is not None and have < _version_key(PYPDF_FLOOR)


#: A version as PEP 440 writes it, any case, surrounding white space aside
#: (PEP 440, "Appendix B: Parsing version strings with regular
#: expressions").
_PEP_440 = re.compile(
    r"v?(?:(?P<epoch>[0-9]+)!)?(?P<release>[0-9]+(?:\.[0-9]+)*)"
    r"(?:[-_.]?(?P<pre>alpha|a|beta|b|preview|pre|c|rc)[-_.]?"
    r"(?P<pre_n>[0-9]+)?)?"
    r"(?:-(?P<post_n1>[0-9]+)|[-_.]?(?P<post>post|rev|r)[-_.]?"
    r"(?P<post_n2>[0-9]+)?)?"
    r"(?:[-_.]?(?P<dev>dev)[-_.]?(?P<dev_n>[0-9]+)?)?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?", re.IGNORECASE)
_PRE = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2,
        "preview": 2}


def _version_key(text):
    """A key that orders versions as PEP 440 does, or None where ``text``
    names none: epoch, release (its trailing zeros dropped), pre-release (a
    development release of a final release before its pre-releases),
    post-release, development release. A local version is not looked at: it
    orders after its public one, and the floor has none."""
    match = _PEP_440.fullmatch(str(text).strip())
    if match is None:
        return None
    release = [int(part) for part in match["release"].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    posted = match["post_n1"] is not None or match["post"] is not None
    if match["pre"] is not None:
        pre = (0, _PRE[match["pre"].lower()], int(match["pre_n"] or 0))
    elif match["dev"] is not None and not posted:
        pre = (-1,)
    else:
        pre = (1,)
    post = (int(match["post_n1"] or match["post_n2"] or 0),) if posted \
        else (-1,)
    dev = (int(match["dev_n"] or 0),) if match["dev"] is not None \
        else (float("inf"),)
    return int(match["epoch"] or 0), tuple(release), pre, post, dev


def _bounded_reader(pypdf, base):
    """_reader_class's reader of ``base``, pypdf's PdfReader, or what the
    pypdf lacks that sift needs of it, said."""
    version = getattr(pypdf, "__version__", "(of no version)")
    lacking = [name for name in _PYPDF_NEEDED if not hasattr(pypdf, name)]
    lacking += [name for name in _READER_NEEDED if not hasattr(base, name)]
    if lacking:
        names = " and ".join((", ".join(lacking[:-1]), lacking[-1])) \
            if len(lacking) > 1 else lacking[0]
        return (f"pypdf {version} lacks {names}, with which sift bounds what "
                f"it reads; pypdf {PYPDF_FLOOR} and later have them")
    if _older_than_the_floor(version):
        # 6.18.0 has all of the above, and sift's tests fail on it: it was
        # read with, silently.
        return (f"pypdf {version} is older than {PYPDF_FLOOR}, the oldest "
                f"sift reads with, which is tested with it")

    class Reader(base):
        def __setattr__(self, name, value):
            # Where pypdf keeps the entries of the objects in use its
            # cross-reference sections list: counted as it makes them while
            # the reader is made (_Entries). Caught as pypdf sets them, not
            # as it reads them: a property cost every read of them, three a
            # line of a classic table.
            if self.__dict__.get("structure") is None \
                    or self.structure.closed:
                pass            # made, or let go of (close): not charged
            elif name == "xref" and not isinstance(value, _Generations):
                value = _Generations(self.structure, value)
            elif name == "xref_free_entry" \
                    and not isinstance(value, _Generations):
                value = _Generations(self.structure, value, counted=False)
            elif name == "xref_objStm" and not isinstance(value, _Entries):
                value = _Entries(self.structure, value)
            super().__setattr__(name, value)

        def __init__(self, data, structure):
            self.structure = structure
            structure.data = data
            deadline.check("a PDF")
            structure.arm()
            building = _BuildingBytes(data, structure)
            structure.register(building)
            structure.building = True
            logged = structure.logged
            if logged is not None:
                logged.trip = structure.trip
            try:
                super().__init__(building)
            except _ReaderBound as bound:
                _close(self)
                if bound.why == "deadline":
                    deadline.check("a PDF")
                    raise deadline.DeadlineExceeded(
                        deadline.seconds() or 0.0, "a PDF") from None
                raise _Unread(bound.why, structure) from None
            except BaseException:
                # The reader sift never gets holds what it read so far, round
                # the same cycle: a reader whose newest cross-reference stream
                # decoded past the budget outlived its document.
                _close(self)
                raise
            finally:
                structure.building = False
                if logged is not None:
                    logged.trip = None
            # What it kept of the trailers it read, charged: each was parsed
            # in flight (_read_xref, _read_pdf15_xref_stream, _rebuild); and
            # the entries made since last charged (_Entries.settle).
            try:
                opened, structure.opened = structure.opened, []
                for entries in opened:
                    entries.settle()
                    entries.every = 1           # from now on, as made
                structure.hold(_held_by(getattr(self, "trailer", None)))
            except _HeldBound:
                _close(self)
                raise _Unread("memory", structure) from None
            # Read from now on for pages: each search of it counted, each
            # read tripped with the rest once the time budget is spent.
            self.stream = _FileBytes(data, structure)
            structure.register(self.stream)

        def _read_pdf15_xref_stream(self, stream):
            # Its dictionary and data parsed counted, in flight: pypdf keeps
            # what it lists of the trailer's keys, charged with the trailer
            # once the reader is made, and the stream itself in its cache
            # (_reader.py:1234 in 6.19.0), charged here, what it decoded let
            # go of: 216 KB of 100,000 entries, uncharged.
            structure = self.structure
            structure.begin(sys._getframe())
            counted = structure.counting(stream)
            before = _cache_size(self)
            try:
                with structure.decoding():
                    xref = super()._read_pdf15_xref_stream(stream)
            finally:
                if counted:
                    structure.uncounting(stream)
                structure.end()
            structure.charge(xref)
            structure.hold(_held_by(xref)
                           + _cached_held(self, before, xref))
            return xref

        def _read_standard_xref_table(self, stream):
            # The table's lines are io.BytesIO's reads -- a counted read
            # costs pypdf more than twice as much there --, and
            # the trailer pypdf parses after them counted, in flight, until
            # _read_xref is done: its frame, pypdf's _read_xref, is where the
            # parse began.
            super()._read_standard_xref_table(stream)
            structure = self.structure
            structure.begin(sys._getframe(1))
            if not structure.counting(stream):
                structure.end()

        def _flatten(self, *args, **kwargs):
            # The copies pypdf makes of a page's dictionary, and what it
            # inherits, charged as it makes them (_Structure.walked): one for
            # each kid that names the page -- 20,000 kids naming one page of
            # 2,000 keys, a 137 KB file, took 0.1.59 past 1 GiB.
            # Each kid is walked through here.
            walked = super()._flatten(*args, **kwargs)
            self.structure.walked(self.flattened_pages)
            return walked

        def _read_xref_subsections(self, idx_pairs, get_entry, used_before):
            # As pypdf reads them (_pairs), in parts of XREF_ENTRIES_A_CHECK,
            # the time budget and the log bound looked at before each: pypdf
            # reads them from the stream's decoded bytes, which no _Alarm
            # trips. An entry past the bound is not made, and the rest of
            # the stream is not read (_EntriesFull).
            structure, read = self.structure, super()._read_xref_subsections
            try:
                for i in range(0, len(idx_pairs) - 1, 2):
                    start, count = idx_pairs[i], idx_pairs[i + 1]
                    if not (isinstance(start, int)
                            and isinstance(count, int)):
                        read([start, count], get_entry, used_before)
                        continue
                    done = 0
                    while done < count:
                        structure.watch()
                        step = min(count - done, XREF_ENTRIES_A_CHECK)
                        read([start + done, step], get_entry, used_before)
                        done += step
            except _EntriesFull:
                pass

        def _read_xref(self, stream):
            # A classic table cannot be read in part: one that starts where
            # the sections read so far reached the entry bound is not read,
            # and the older ones with it; one that crosses it ends the reader
            # (_EntriesFull).
            structure = self.structure
            if structure.listed >= structure.entries:
                structure.listed_over = True
                return None
            depth = structure.depth
            try:
                return super()._read_xref(stream)
            finally:
                if structure.depth > depth:     # _read_standard_xref_table
                    structure.uncounting(stream)
                    structure.end()

        def _rebuild_xref_table(self, stream):
            _rebuild(self, stream)

        def _get_object_from_stream(self, indirect_reference):
            return _object_from_stream(self, indirect_reference)

        def get_object(self, indirect_reference):
            return _get_object(self, indirect_reference)
    return Reader


class _Unread(Exception):
    """The reader was not made: making it passed the log bound or
    MAX_DOCUMENT_HELD, or a classic cross-reference table crossed the entry
    bound (_ReaderBound)."""

    def __init__(self, why, structure):
        if why == "log":
            said = (f"pypdf logged more than {MAX_OP_LOG_PER_DOCUMENT} "
                    f"characters opening this document")
        elif why == "memory":
            said = f"{_held_said(structure)}, opening it"
        else:
            structure.listed_over = True
            said = (f"a cross-reference table of this document lists past "
                    f"the {structure.entries} objects in use sift lets pypdf "
                    f"hold")
        super().__init__(said)


def _get_object(reader, indirect_reference):
    """PdfReader.get_object of pypdf 6.19.0 (and 6.18.1, whose _reader.py is
    the same), statement for statement -- a test holds what it makes, caches
    and logs equal to pypdf's -- but that where pypdf copies the whole file
    and searches it for "N G obj" (an object its tables lack, or one not
    where they say), the document's index answers (_Structure.find). pypdf
    searched the file again each time it resolved such a reference: 2,000
    pages of a 658 KB file whose offsets all missed took 14.4 s in 0.1.55,
    and 0.1.58, which let pypdf search 256 MiB in all, read 407 of them
    (insfix7.pdf, a review)."""
    (pypdf_reader, FileNotDecryptedError, LimitReachedError, PdfReadError,
     IndirectObject, NullObject, read_object,
     skip_over_whitespace) = _GET_OBJECT_NAMES or _get_object_names()
    if isinstance(indirect_reference, int):
        indirect_reference = IndirectObject(indirect_reference, 0, reader)
    retval = reader.cache_get_indirect_object(
        indirect_reference.generation, indirect_reference.idnum
    )
    if retval is not None:
        return retval
    if (
        indirect_reference.generation == 0
        and indirect_reference.idnum in reader.xref_objStm
    ):
        retval = reader._get_object_from_stream(indirect_reference)
    elif (
        indirect_reference.generation in reader.xref
        and indirect_reference.idnum
        in reader.xref[indirect_reference.generation]
    ):
        if reader.xref_free_entry.get(indirect_reference.generation, {}).get(
            indirect_reference.idnum, False
        ):
            return NullObject()
        start = reader.xref[indirect_reference.generation][
            indirect_reference.idnum]
        reader.stream.seek(start, 0)
        try:
            idnum, generation = reader.read_object_header(reader.stream)
            if (
                idnum != indirect_reference.idnum
                or generation != indirect_reference.generation
            ):
                raise PdfReadError("Not matching, we parse the file for it")
        except Exception:
            m = reader.structure.find(indirect_reference.idnum,
                                      indirect_reference.generation)
            if m is not None:
                pypdf_reader.logger_warning(
                    "Object ID %(idnum)d,%(generation)d ref repaired",
                    source="pypdf._reader",
                    idnum=indirect_reference.idnum,
                    generation=indirect_reference.generation,
                )
                reader.xref[indirect_reference.generation][
                    indirect_reference.idnum
                ] = (m[0] + 1)
                reader.stream.seek(m[0] + 1)
                idnum, generation = reader.read_object_header(reader.stream)
            else:
                idnum = -1
                generation = -1  # exception will be raised below
        if idnum != indirect_reference.idnum and reader.xref_index:
            # xref table probably had bad indexes due to not being
            # zero-indexed
            if reader.strict:  # pragma: no cover - sift reads leniently
                raise PdfReadError(
                    f"Expected object ID ({indirect_reference.idnum} "
                    f"{indirect_reference.generation}) does not match actual "
                    f"({idnum} {generation}); xref table not zero-indexed."
                )
            # xref table is corrected in non-strict mode
        elif idnum != indirect_reference.idnum and reader.strict:
            # some other problem
            raise PdfReadError(  # pragma: no cover - sift reads leniently
                f"Expected object ID ({indirect_reference.idnum} "
                f"{indirect_reference.generation}) does not match actual "
                f"({idnum} {generation})."
            )
        if reader.strict:  # pragma: no cover - sift reads leniently
            assert generation == indirect_reference.generation

        current_object = (indirect_reference.idnum,
                          indirect_reference.generation)
        if current_object in reader._known_objects:
            raise LimitReachedError(f"Detected loop with self reference for "
                                    f"{indirect_reference!r}.")
        reader._known_objects.add(current_object)
        retval = _counted_parse(reader, read_object)
        reader._known_objects.remove(current_object)

        # override encryption is used for the /Encrypt dictionary
        if not reader._override_encryption and reader._encryption is not None:
            # if we don't have the encryption key:
            if not reader._encryption.is_decrypted():
                raise FileNotDecryptedError("File has not been decrypted")
            # otherwise, decrypt here...
            retval = reader._encryption.decrypt_object(
                retval, indirect_reference.idnum,
                indirect_reference.generation, strict=reader.strict,
            )
    else:
        m = reader.structure.find(indirect_reference.idnum,
                                  indirect_reference.generation)
        if m is not None:
            pypdf_reader.logger_warning(
                "Object %(idnum)d %(generation)d found",
                source="pypdf._reader",
                idnum=indirect_reference.idnum,
                generation=indirect_reference.generation,
            )
            if indirect_reference.generation not in reader.xref:
                reader.xref[indirect_reference.generation] = {}
            reader.xref[indirect_reference.generation][
                indirect_reference.idnum] = (m[0] + 1)
            reader.stream.seek(m[1] + 1)
            skip_over_whitespace(reader.stream)
            reader.stream.seek(-1, 1)
            retval = _counted_parse(reader, read_object)

            # override encryption is used for the /Encrypt dictionary
            if not reader._override_encryption \
                    and reader._encryption is not None:
                # if we don't have the encryption key:
                if not reader._encryption.is_decrypted():
                    raise FileNotDecryptedError("File has not been decrypted")
                # otherwise, decrypt here...
                retval = reader._encryption.decrypt_object(
                    retval, indirect_reference.idnum,
                    indirect_reference.generation, strict=reader.strict,
                )
        else:
            pypdf_reader.logger_warning(
                "Object %(idnum)d %(generation)d not defined.",
                source="pypdf._reader",
                idnum=indirect_reference.idnum,
                generation=indirect_reference.generation,
            )
            if reader.strict:  # pragma: no cover - sift reads leniently
                raise PdfReadError("Could not find object.")
    # For ObjStm objects, _get_object_from_stream already cached
    # the result during batch parsing; skip the redundant cache write
    # to avoid "Overwriting cache" warnings. For non-ObjStm objects
    # (including encrypted ones that need decrypted values cached),
    # always write.
    if not (
        indirect_reference.generation == 0
        and indirect_reference.idnum in reader.xref_objStm
    ):
        new = (indirect_reference.generation, indirect_reference.idnum) \
            not in reader.resolved_objects
        before = _cache_size(reader)
        reader.cache_indirect_object(
            indirect_reference.generation, indirect_reference.idnum, retval
        )
        if new:
            reader.structure.hold(_cached_held(reader, before, retval))
    return retval


def _counted_parse(reader, read_object):
    """read_object(reader.stream, reader), each read of the file counted as
    the parse goes (_Structure.counting, in_flight), and the object charged
    what it holds once parsed (_held_by): pypdf parses and keeps it, a byte
    at a time -- a page whose /Annots held 300,000 numbers held 20 MB, and
    one object of a file of 25 MB could hold more than a worker's memory
    before it was counted."""
    structure, stream = reader.structure, reader.stream
    structure.begin(sys._getframe())
    counted = structure.counting(stream)
    try:
        obj = read_object(stream, reader)
    finally:
        if counted:
            structure.uncounting(stream)
        structure.end()
    structure.hold(_held_by(obj))
    return obj


#: What _get_object reads with of pypdf, looked up once (_get_object_names)
#: rather than imported at each call, as the other copies of pypdf's code
#: here import theirs: get_object is called for every object pypdf
#: resolves.
_GET_OBJECT_NAMES = None


def _get_object_names():
    global _GET_OBJECT_NAMES
    import pypdf._reader as pypdf_reader  # type: ignore
    from pypdf.errors import FileNotDecryptedError  # type: ignore
    from pypdf.errors import LimitReachedError, PdfReadError  # type: ignore
    from pypdf.generic import IndirectObject, NullObject  # type: ignore
    from pypdf.generic import read_object  # type: ignore
    from pypdf._utils import skip_over_whitespace  # type: ignore
    _GET_OBJECT_NAMES = (pypdf_reader, FileNotDecryptedError,
                         LimitReachedError, PdfReadError, IndirectObject,
                         NullObject, read_object, skip_over_whitespace)
    return _GET_OBJECT_NAMES


#: What an object of each of pypdf's classes holds past what sys.getsizeof
#: says of it and past the attributes _held_by counts, in bytes, as
#: tracemalloc traces it (pypdf 6.19.0, CPython 3.14, arrays of 20,000 of
#: each, a review): 24 for an object keeping its attributes inline and 8
#: an attribute its class keeps room for -- IndirectObject three,
#: TextStringObject four, BooleanObject one, a stream two, and
#: indirect_reference where the class came to keep room for it, which
#: depends on what the process did before (8 bytes more, measured caching
#: one object of each class first; none after reading 31 PDFs, as a worker
#: does) --, what a page's copy keeps besides (PageObject, 104 to 112), and 4
#: to 8 for an int's or a bytes' subclass, which keeps them in a dictionary
#: of its own. The most measured is written here; a test measures each
#: again, and holds it within 8 bytes under. A pypdf class not written here
#: is charged the most any is (_RESIDUAL_ELSE), a class of Python's own
#: nothing.
_HELD_PAST_SIZEOF = {
    "NumberObject": 8, "ByteStringObject": 4, "FloatObject": 32,
    "NameObject": 32, "NullObject": 32, "ArrayObject": 32,
    "DictionaryObject": 32, "BooleanObject": 40, "IndirectObject": 52,
    "TextStringObject": 64, "DecodedStreamObject": 48,
    "EncodedStreamObject": 48, "PageObject": 112}
#: What a dictionary pypdf's cache makes of an int's or a bytes' subclass --
#: setting its indirect_reference, which it has nowhere else to keep --
#: holds past what sys.getsizeof says of it (556.4 bytes traced a cached
#: NumberObject, 520.4 counted without it; a review).
_ATTRIBUTE_DICT_PAST_SIZEOF = 40
#: What a kept object stream holds besides its bytes, its stream and where its
#: objects begin (_Kept): the _Kept, its entry of the kept streams, what its
#: stream keeps of its own -- 664 bytes traced past what is counted, 10 to
#: 20,000 objects a stream.
_KEPT_HELD = 672
#: What an int holds past sys.getsizeof: an int made by arithmetic keeps the
#: digits it was made with, one more than it needs where a sum carried
#: nothing -- the numbers of a table's lines, of a stream's entries, the
#: index of a kept object stream's pairs: 4 to 12 bytes an entry traced past
#: sys.getsizeof, a digit an int.
_INT_PAST_SIZEOF = sys.int_info.sizeof_digit
#: What a key of the index of "N G obj" (_Found) holds past that: a product,
#: number x 10 ** 20, and a sum; 7.1 to 8.0 bytes more a key traced past
#: sys.getsizeof (1,000 to 300,000 keys, a review), a digit more.
_INDEX_KEY_PAST_SIZEOF = sys.int_info.sizeof_digit
_RESIDUAL_ELSE = max(_HELD_PAST_SIZEOF.values())
#: The residual of each class met so far (_residual).
_RESIDUALS = {}
#: What a 2-tuple holds: a key of pypdf's cache, a pair of a kept object
#: stream's header.
_PAIR_TUPLE = sys.getsizeof((None, None))
#: The bytes objects Python makes once and shares: the empty one and each of
#: one byte, as slicing and b"x" make them.
_SHARED_BYTES = frozenset(id(bytes([byte])) for byte in range(256)) \
    | {id(b"")}


def _residual(kind):
    """What an object of class ``kind`` holds past sys.getsizeof and the
    attributes _held_by counts (_HELD_PAST_SIZEOF)."""
    residual = _RESIDUALS.get(kind)
    if residual is None:
        residual = _HELD_PAST_SIZEOF.get(
            kind.__name__, _RESIDUAL_ELSE
            if kind.__module__.startswith("pypdf") else 0)
        _RESIDUALS[kind] = residual
    return residual


def _int_held(value):
    """What an int a reader keeps holds: nothing for one Python shares -- a
    small one, True, False --, else sys.getsizeof and a digit
    (_INT_PAST_SIZEOF)."""
    if type(value) is bool or type(value) is int and -5 <= value <= 256:
        return 0
    return sys.getsizeof(value) + _INT_PAST_SIZEOF \
        if isinstance(value, int) else 0


def _bytes_held(value):
    """What bytes an object keeps hold: nothing for bytes Python shares."""
    if not isinstance(value, (bytes, bytearray)) or id(value) in _SHARED_BYTES:
        return 0
    return sys.getsizeof(value)


def _entry_held(value):
    """What the value of an entry of the reader's tables holds: an offset, a
    pair (object stream, index), a flag, as pypdf makes them."""
    if isinstance(value, tuple):
        return sys.getsizeof(value) + sum(_int_held(part) for part in value)
    if isinstance(value, int):
        return _int_held(value)
    return 0


def _held_by(obj):
    """What ``obj``, as pypdf parsed it, holds, in bytes: each object it is
    made of -- sys.getsizeof, and what each class holds past it
    (_HELD_PAST_SIZEOF) --, the ints an IndirectObject keeps, the bytes a
    TextStringObject was read from and a stream's data. Walked without a
    set of what was seen -- what pypdf parses is a tree, and a set of 6,000,000
    elements' ids held 300 MB --, and with a stack of iterators, not of the
    elements themselves. Measured against tracemalloc within 5.5 % over,
    never under, on every shape of 10 to 20,000 elements (a test holds it
    within TIGHT)."""
    getsizeof, walks, walk = sys.getsizeof, _WALKS, _walk
    chain = itertools.chain.from_iterable
    total, stack = 0, [iter((obj,))]
    while stack:
        for node in stack[-1]:
            kind = type(node)
            how = walks.get(kind) or walk(kind)
            total += getsizeof(node) + how[0]
            how = how[1]
            if how is _LEAF:
                continue
            if how is _MAPPING:
                data = getattr(node, "_data", None)
                if data is not None:
                    total += _bytes_held(data)
                if node:
                    stack.append(chain(node.items()))
                    break
            elif how is _SEQUENCE:
                if node:
                    stack.append(iter(node))
                    break
            elif how is _REFERENCE:
                total += _int_held(node.idnum) + _int_held(node.generation)
            else:
                total += _bytes_held(getattr(node, "_original_bytes", None))
        else:
            stack.pop()
    return total


#: How _held_by walks an object of a class, besides what the class holds past
#: sys.getsizeof (_residual): not at all, into a dictionary's items and its
#: stream's data, into a sequence, to a reference's ints, to the bytes a
#: string was read from.
_LEAF, _MAPPING, _SEQUENCE, _REFERENCE, _TEXT = (
    "leaf", "mapping", "sequence", "reference", "text")
#: The class of each object _held_by walked, and how it walks one (_walk).
_WALKS = {}


def _walk(kind):
    """(what an object of class ``kind`` holds past sys.getsizeof, how
    _held_by walks it), kept for the next object of the class (_WALKS): what
    a class is is looked up once, not at each of the objects it made."""
    if issubclass(kind, dict):
        how = _MAPPING
    elif issubclass(kind, (list, tuple)):
        how = _SEQUENCE
    elif kind.__name__ == "IndirectObject":
        how = _REFERENCE
    elif issubclass(kind, str):
        how = _TEXT
    else:
        how = _LEAF
    walked = _WALKS[kind] = (_residual(kind), how)
    return walked


def _cache_size(reader):
    """What pypdf's cache of the reader's objects holds of its own table."""
    return reader.resolved_objects.__sizeof__()


def _cached_held(reader, before, obj):
    """What pypdf's cache of ``reader`` holds of ``obj`` past the object,
    once it cached it -- its cache having held ``before`` of its own table
    (_cache_size): the table's growth, its key, the reference pypdf sets on
    the object, and the dictionary that makes of an int's or a bytes'
    subclass. The key's ints are the reference's own, held already."""
    held = _cache_size(reader) - before + _PAIR_TUPLE
    reference = getattr(obj, "indirect_reference", None) \
        if obj is not None else None
    if reference is not None:
        held += sys.getsizeof(reference) + _residual(type(reference))
        if type(obj).__itemsize__:
            held += sys.getsizeof(obj.__dict__) + _ATTRIBUTE_DICT_PAST_SIZEOF
    return held


def _page_held(page):
    """What the copy pypdf makes of a page's dictionary as it walks the page
    tree holds past what it copies (PageObject): its table of keys, what it
    keeps of its own."""
    return sys.getsizeof(page) + _residual(type(page))


def _index_held(found):
    """What the index of "N G obj" (_Found) holds: its table, its keys, its
    offsets."""
    index = found.index
    return sys.getsizeof(index) + sum(map(_int_held, index)) \
        + len(index) * _INDEX_KEY_PAST_SIZEOF \
        + sum(map(_int_held, index.values()))


def _in_flight_held(root):
    """What the parse under way holds so far, in bytes: what the frames
    between the read and ``root``, where the parse began, hold of it --
    pypdf's array as it grows, its dictionary, a stream's data, a name's
    bytes --, each once (_held_by). pypdf parses an object in one call, and
    until it returns nothing but its frames holds what it made: a page whose
    /Annots holds 3,000,000 references holds 312 MB, which a worst case a
    read charged 960 MB. Without ``root`` nothing is walked, and
    what the reads would hold at the worst case is what it holds."""
    if root is None:
        return MAX_DOCUMENT_HELD
    total, seen = 0, set()
    frame = sys._getframe(2)
    while frame is not None and frame is not root:
        for value in frame.f_locals.values():
            if isinstance(value, _IN_FLIGHT) and id(value) not in seen:
                seen.add(id(value))
                total += _held_by(value)
        frame = frame.f_back
    return total


#: What a frame of a parse under way may hold of what it makes.
_IN_FLIGHT = (dict, list, tuple, bytes, bytearray, str)


class _Growth:
    """What the dictionary ``of`` holds as it grows -- its table, its keys and
    values since last asked (_int_held, _entry_held) --, charged to
    ``structure`` each time it is called; ``held`` in all."""

    __slots__ = ("of", "structure", "size", "count", "held")

    def __init__(self, of, structure):
        self.of, self.structure = of, structure
        self.size, self.count, self.held = sys.getsizeof(of), 0, 0

    def __call__(self):
        of = self.of
        added = itertools.islice(reversed(of.items()), len(of) - self.count)
        size = sys.getsizeof(of)
        amount = size - self.size + sum(_int_held(key) + _entry_held(value)
                                        for key, value in added)
        self.size, self.count = size, len(of)
        self.held += amount
        self.structure.hold(amount)


#: What pypdf's search for an object in the whole file finds, for any
#: number and generation (PdfReader.get_object: \s{idnum}\s+{generation}
#: \s+obj). Its matches do not overlap one another's start, so finditer
#: finds each place the search for one object would.
_OBJ_AT = re.compile(rb"\s(\d+)\s+(\d+)\s+obj")
#: The most digits a number of the index is kept with: past it no int pypdf
#: holds is written so in a file sift reads, and it is searched for.
_OBJ_DIGITS = 19


class _Found:
    """A document's index of "N G obj": for each number and generation, where
    its first match of _OBJ_AT starts -- where pypdf's search of the whole
    file for that object finds it -- made in one pass, under the time budget
    (``check``), of at most ``cap`` objects, the document's entry bound;
    ``upto`` is where the index ends (the file's end where it holds all).
    ``get`` answers (start, end), None, or UNKNOWN for an object past
    ``upto`` or one written with digits no key has (keyed)."""

    UNKNOWN = object()
    __slots__ = ("data", "index", "upto")

    def __init__(self, data, cap, check):
        self.data, self.index, self.upto = data, {}, len(data)
        index, scale = self.index, 10 ** (_OBJ_DIGITS + 1)
        for count, match in enumerate(_OBJ_AT.finditer(data)):
            if not count % 4096:
                check()
            number, generation = match.group(1), match.group(2)
            if not (self._plain(number) and self._plain(generation)):
                continue        # what no search for an int's digits finds
            key = int(number) * scale + int(generation)
            if key in index:
                continue
            if len(index) >= cap:
                self.upto = match.start()
                break
            index[key] = match.start()

    @staticmethod
    def _plain(digits: bytes) -> bool:
        """Whether ``digits`` are how str() writes a number: no leading 0
        but in 0 itself, at most _OBJ_DIGITS of them."""
        return len(digits) <= _OBJ_DIGITS and (digits[:1] != b"0"
                                               or len(digits) == 1)

    @staticmethod
    def keyed(number, generation) -> bool:
        """Whether the index can hold (``number``, ``generation``)."""
        return all(isinstance(value, int) and not isinstance(value, bool)
                   and 0 <= value < 10 ** _OBJ_DIGITS
                   for value in (number, generation))

    def get(self, number, generation):
        if not self.keyed(number, generation):
            return self.UNKNOWN
        start = self.index.get(number * 10 ** (_OBJ_DIGITS + 1) + generation)
        if start is not None:
            return start, _OBJ_AT.match(self.data, start).end()
        return None if self.upto == len(self.data) else self.UNKNOWN


class _Entries(dict):
    """xref_objStm, one generation's entries of xref, or -- ``counted``
    False -- of xref_free_entry, as pypdf keeps them: each object in use it
    enters while the reader is made is counted against the entry bound
    (_Structure.allot), once however often pypdf writes it, and what the
    entries hold is charged -- the table's growth, their keys, their values
    -- ENTRIES_A_CHARGE at a time while the reader is made (settle), each as
    it is made after: charged one by one, they cost a document of 800,000
    entries 1.3 s more. A classic table's line makes two
    entries of xref_free_entry too, used or free -- of its generation and of
    65535 --, which share its number with xref: charged there where the line
    is free, in its generation's (``generation``). 0.1.59 counted none of
    them, and 0.1.60's first cut charged XREF_ENTRY_HELD an entry of an
    object in use, the most one was measured to hold, 1.3 to 2.7 times what
    they hold."""

    __slots__ = ("structure", "counted", "generation", "size", "added",
                 "every")

    def __init__(self, structure, items=(), counted=True, generation=None):
        super().__init__(items)
        self.structure, self.counted = structure, counted
        self.generation, self.size, self.added = generation, 0, len(self)
        structure.hold(sys.getsizeof(self) - dict.__sizeof__(self))
        if structure.building:
            structure.opened.append(self)       # settled once it is made
            self.every = ENTRIES_A_CHARGE
        else:
            self.every = 1
        self.settle()

    def __setitem__(self, key, value):
        if key in self:
            old = dict.__getitem__(self, key)
            dict.__setitem__(self, key, value)
            self.structure.hold(_entry_held(value) - _entry_held(old))
            return
        if self.counted:
            self.structure.allot()
        dict.__setitem__(self, key, value)
        self.added += 1
        if self.added >= self.every:
            self.settle()

    def __delitem__(self, key):
        """pypdf drops an entry that points where no object is, and one it
        renumbers, while it makes the reader (6.19.0 _reader.py:750, :726):
        the batch pending is charged first -- it is counted from the table's
        end --, then what the entry held is given back; the table keeps its
        size until it grows again."""
        value = dict.__getitem__(self, key)
        if self.added:
            self.settle()
        dict.__delitem__(self, key)
        self.structure.give_back(self._held(itertools.repeat(key, 1), [value]))

    def settle(self):
        """Charge what the entries made since last charged hold: the table's
        growth, and their keys and values, counted in C (_ints_held). The
        batch is the table's last entries: pypdf drops them with del alone
        (__delitem__), and an entry removed otherwise leaves the batch
        counted over the entries left -- over what is held, never under."""
        added, self.added = min(self.added, len(self)), 0
        size = dict.__sizeof__(self)
        held, self.size = size - self.size, size
        if added:
            held += self._held(
                itertools.islice(reversed(self), added),
                list(itertools.islice(reversed(self.values()), added)))
        self.structure.hold(held)

    def _held(self, keys, values):
        """What the entries of ``keys`` and ``values`` (a list, in the same
        order) hold past the table: a counted table's keys and values -- an
        offset, a pair (object stream, index) --, of a free line's
        generation the numbers of the free lines."""
        if self.counted:
            if type(values[0]) is tuple:        # (object stream, index)
                return _ints_held(keys) + sum(map(sys.getsizeof, values)) \
                    + _ints_held(itertools.chain.from_iterable(values))
            return _ints_held(keys) + _ints_held(values)        # offsets
        if self.generation != 65535:
            # A free line's number: the line's entry of its generation holds
            # True (free); a used line's is xref's.
            return _ints_held(itertools.compress(
                keys, map(True.__eq__, values)))
        return 0


def _ints_held(ints):
    """What the ints of ``ints`` a reader keeps hold (_int_held), counted in
    C: those past 256, which Python does not share, at sys.getsizeof and a
    digit each."""
    sizes = list(map(sys.getsizeof, filter((256).__lt__, ints)))
    return sum(sizes) + len(sizes) * _INT_PAST_SIZEOF


class _Generations(dict):
    """xref or xref_free_entry: its generations' entries kept as _Entries,
    counted against the entry bound where ``counted``, its own table charged
    as it grows."""

    __slots__ = ("structure", "counted")

    def __init__(self, structure, items=(), counted=True):
        super().__init__()
        self.structure, self.counted = structure, counted
        structure.hold(sys.getsizeof(self))
        for key, value in dict(items).items():
            self[key] = value

    def __setitem__(self, key, value):
        before = dict.__sizeof__(self)
        dict.__setitem__(self, key, value if isinstance(value, _Entries)
                         else _Entries(self.structure, value, self.counted,
                                       key))
        self.structure.hold(dict.__sizeof__(self) - before + _int_held(key))


class _FileBytes(io.BytesIO):
    """The file as pypdf's reader holds it. Every copy of it whole pypdf
    takes -- to recover a line of a table it cannot read
    (_read_standard_xref_table), or where the index does not answer
    (_Structure.find) -- looks at the time budget and is counted
    (_Structure.search); past MAX_SEARCHED pypdf is handed nothing, and
    finds nothing. pypdf copied and searched the whole file each time it
    resolved a reference to an object the file lacks: 100 of them in a 4.9
    MB file ran 9.81 s past a 5 s budget, 10,000 of them 921 s. Its reads are io.BytesIO's, until the document's budget is spent
    (_trip)."""

    building = False

    def __init__(self, data, structure):
        super().__init__(data)
        self.structure, self.size = structure, len(data)

    def getbuffer(self):
        if self.structure.search(self.size, self.building):
            return super().getbuffer()
        return memoryview(b"")


class _BuildingBytes(_FileBytes):
    """The file as pypdf reads it making the reader."""

    building = True


class _TrippedFile(_FileBytes):
    """A _FileBytes tripped (_trip): each read raises _ReaderBound."""

    def read(self, size=-1):
        raise _ReaderBound(self.structure.tripped)


class _TrippedBuilding(_BuildingBytes):
    """A _BuildingBytes tripped (_trip): each read raises _ReaderBound."""

    def read(self, size=-1):
        raise _ReaderBound(self.structure.tripped)


class _CountedFile(_FileBytes):
    """A _FileBytes whose reads are counted (_Structure.counting) for the
    parse under way: each at what what pypdf parses of it may hold at the
    worst case -- READ_HELD, or its bytes where it read more --, and past
    what is left of the bound, what the parse holds is walked
    (_Structure.in_flight): as pypdf parses, not once it has parsed."""

    def read(self, size=-1):
        data = io.BytesIO.read(self, size)
        structure = self.structure
        structure.reads += READ_HELD if len(data) <= READ_HELD else len(data)
        if structure.reads > structure.room:
            structure.in_flight()
        return data


class _CountedBuilding(_BuildingBytes):
    """A _BuildingBytes whose reads are counted, as _CountedFile's are."""

    read = _CountedFile.read


_TRIPPED = {_FileBytes: _TrippedFile, _BuildingBytes: _TrippedBuilding,
            _CountedFile: _TrippedFile, _CountedBuilding: _TrippedBuilding}
#: A stream of the file, and the same stream counted (_Structure.counting).
_COUNTED = {_FileBytes: _CountedFile, _BuildingBytes: _CountedBuilding}
_UNCOUNTED = {counted: plain for plain, counted in _COUNTED.items()}


def _trip(stream):
    """Make each read of ``stream`` raise _ReaderBound from now on: its class
    changed, between two reads pypdf makes of it, to one whose read raises.
    A read that looked at a flag instead cost every read of every document
    (_Alarm)."""
    tripped = _TRIPPED.get(type(stream))
    if tripped is not None:
        stream.__class__ = tripped


class _ParseBound(BaseException):
    """Parsing an object stream passed OBJECT_STREAM_READS_A_BYTE, or the
    deadline: ends the parse. A BaseException, as _LogBound is one: pypdf
    catches every Exception reading a dictionary's value, logs it and reads
    on (DictionaryObject.read_from_stream in 6.19.0)."""


class _ObjectStreamBytes(io.BytesIO):
    """The bytes of an object stream as pypdf parses them, counting its
    reads: past ``reads`` of them, or once the deadline is spent -- looked
    at every 1,024 reads -- the next read ends the parse (_ParseBound).
    pypdf reads most of what it parses a byte at a time. Where it is kept
    (``structure``, _kept_stream), the reads of a parse of an object to be
    kept count as those of the file do (_CountedFile), 1,024 at a time."""

    def __init__(self, data, reads, structure=None):
        super().__init__(data)
        self.reads, self.timed_out = reads, False
        self.structure = structure

    def read(self, size=-1):
        self.reads -= 1
        if not self.reads & 1023:
            if self.reads < 0:
                raise _ParseBound()
            structure = self.structure
            if structure is not None and structure.depth:
                structure.reads += 1024 * READ_HELD
                if structure.reads > structure.room:
                    structure.in_flight()
            if deadline.spent():
                self.timed_out = True
                raise _ParseBound()
        return super().read(size)


class _Kept:
    """An object stream kept decoded while its document is read
    (_Structure.keep): its bytes as pypdf parses them (_ObjectStreamBytes,
    whose read bound spans every object parsed of it), where its objects
    begin (/First), each number's first pair of its header (index, offset),
    and what it is charged: what it holds (_kept_stream)."""

    __slots__ = ("stream", "first", "at", "charged")

    def __init__(self, stream, first, at, charged):
        self.stream, self.first, self.at = stream, first, at
        self.charged = charged


def _object_from_stream(reader, indirect_reference):
    """PdfReader._get_object_from_stream of pypdf 6.19.0, but that only the
    object asked for is parsed and kept -- a test holds what it makes and
    logs equal to pypdf's, the objects asked for in any order.

    pypdf parses every object of the stream the first time one of them is
    asked for, and keeps them all: walking the page tree of a tagged report
    whose pages sit in object streams beside their struct elements parsed
    all 402,003 of them, 1,015 MB, and t2100p.pdf, 21 MB, was OOM-killed at
    --memory=1g. sift keeps the stream decoded instead
    (_kept_stream), charged, within MAX_OBJECT_STREAMS_HELD, and parses an
    object of it where pypdf's own parse would begin it -- at the first pair
    of the stream's header that names it -- when it is asked for; what it
    reads to parse and keep it is charged as it reads (_ObjectStreamBytes),
    and what pypdf logs of it is logged then. Where pypdf's parse of the
    whole stream raised at another object than the one asked for, sift's
    reads the one asked for.

    The stream decodes within the document's budget and is charged
    (_Structure), and its objects are parsed from _ObjectStreamBytes, whose
    read bound spans every object parsed of it: a stream whose parse passes
    it is refused -- the objects parsed so far are kept, as pypdf keeps
    them, the rest are null, and the notes say so. Where the deadline ended
    the parse, DeadlineExceeded is raised."""
    import pypdf._reader as pypdf_reader  # type: ignore
    from pypdf.errors import PdfReadError, PdfStreamError  # type: ignore
    from pypdf.generic import NullObject, read_object  # type: ignore
    from pypdf._utils import read_non_whitespace  # type: ignore
    structure = reader.structure
    stmnum, _idx = reader.xref_objStm[indirect_reference.idnum]
    deadline.check("a PDF")
    if stmnum in structure.refused:
        return NullObject()
    kept = structure.kept.get(stmnum)
    if kept is None:
        kept = _kept_stream(reader, stmnum)
        if kept is None:
            return NullObject()
    else:
        structure.kept.move_to_end(stmnum)
    at = kept.at.get(indirect_reference.idnum)
    if at is None:
        if reader.strict:  # pragma: no cover - sift reads leniently
            raise PdfReadError("This is a fatal error in strict mode.")
        return NullObject()
    index, offset = at
    stream_data = kept.stream
    structure.begin(sys._getframe())
    try:
        stream_data.seek(kept.first + offset, 0)
        read_non_whitespace(stream_data)
        stream_data.seek(-1, 1)
        try:
            obj = read_object(stream_data, reader)
        except PdfStreamError as exc:
            pypdf_reader.logger_warning(
                "Invalid stream (index %(index)d) within object "
                "%(obj_num)d 0: %(exc)s",
                source="pypdf._reader", index=index,
                obj_num=indirect_reference.idnum, exc=exc)
            if reader.strict:  # pragma: no cover
                raise PdfReadError(f"Cannot read object stream: {exc}")
            obj = NullObject()
    except _ParseBound:
        if stream_data.timed_out:
            deadline.check("a PDF")
        structure.refused.add(stmnum)
        structure.let_go(stmnum)
        return NullObject()
    finally:
        structure.end()
    structure.hold(_held_by(obj))
    before = _cache_size(reader)
    reader.cache_indirect_object(0, indirect_reference.idnum, obj)
    structure.hold(_cached_held(reader, before, obj))
    return obj


def _kept_stream(reader, stmnum):
    """Object stream ``stmnum`` decoded and its header read, as pypdf reads
    them (PdfReader._get_object_from_stream of pypdf 6.19.0), and kept
    (_Structure.keep); None where its header's parse passed its read bound
    (refused) -- or raises what pypdf's reading of it raises, and is decoded
    and read again the next time an object of it is asked for, as pypdf
    does. /N is taken at most a pair a three bytes of the stream, as pypdf
    takes it, counted without pypdf's copy of the stream (getbuffer)."""
    import pypdf._reader as pypdf_reader  # type: ignore
    from pypdf.errors import LimitReachedError  # type: ignore
    from pypdf.generic import IndirectObject, NumberObject  # type: ignore
    from pypdf._utils import read_non_whitespace  # type: ignore
    structure = reader.structure
    obj_stm = IndirectObject(stmnum, 0, reader).get_object()
    assert obj_stm["/Type"] == "/ObjStm"
    with structure.decoding():
        data = obj_stm.get_data()
    structure.charge(obj_stm)
    stream_data = _ObjectStreamBytes(
        data, OBJECT_STREAM_READS_A_BYTE * len(data) + OBJECT_STREAM_READS,
        structure)
    n = int(obj_stm["/N"])
    first_offset = int(obj_stm["/First"])
    max_n = len(data) // 3
    if n > max_n:
        if reader.strict:
            raise LimitReachedError(f"Value /N {n} for object {stmnum} "
                                    f"exceeds maximum allowed value {max_n}.")
        pypdf_reader.logger_warning(
            "Value /N %(n)d for object %(stmnum)d exceeds maximum allowed "
            "value %(max_n)d. Limiting to %(max_n)d.",
            source="pypdf._reader", n=n, stmnum=stmnum, max_n=max_n)
        n = max_n
    # What it holds, charged as it is read, a thousand and twenty-four pairs
    # at a time: a header of "1 0 " pairs holds 64 times its bytes. Given back
    # where it is not kept.
    at = {}
    base = sys.getsizeof(data) + sys.getsizeof(stream_data) + _KEPT_HELD \
        + sys.getsizeof(at)
    structure.hold(base)
    grown = _Growth(at, structure)
    try:
        for i in range(n):
            if not i & 1023 and i:
                grown()
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            objnum = NumberObject.read_from_stream(stream_data)
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            offset = NumberObject.read_from_stream(stream_data)
            read_non_whitespace(stream_data)
            stream_data.seek(-1, 1)
            at.setdefault(int(objnum), (i, int(offset)))
        grown()
    except _ParseBound:
        structure.give_back(base + grown.held)
        if stream_data.timed_out:
            deadline.check("a PDF")
        structure.refused.add(stmnum)
        return None
    except BaseException:
        structure.give_back(base + grown.held)
        raise
    kept = _Kept(stream_data, first_offset, at, base + grown.held)
    structure.keep(stmnum, kept)
    return kept


def _rebuild(reader, stream):
    """PdfReader._rebuild_xref_table of pypdf 6.19.0 (and 6.18.1), statement
    for statement -- a test holds what it makes and logs equal to pypdf's --
    but that each object it finds is listed within the entry bound
    (_Structure.allot) and each entry its object streams write counted
    against it (_Structure.write), each trailer it reads is charged as it is read
    (_Structure.counting), each object and trailer it reads looks at the time
    budget and the log bound first (_Structure.watch), and each object
    stream decodes within the budget, is charged, and is parsed from
    _ObjectStreamBytes. pypdf decoded every object stream of the file within
    what was left of the budget, charged none, and parsed each a byte at a
    time: 200 of 790,000 bytes, a 174 KB file whose table was lost, ran 44 s
    past a 5 s budget. An object stream past the budget, or
    whose parse passed OBJECT_STREAM_READS_A_BYTE, is not read further --
    pypdf's own rebuild gave up on the file at the first limit -- and the
    notes say so; the objects of the file past the entry bound are not
    listed (_EntriesFull)."""
    _rebuilding(reader, stream, reader.structure)


def _rebuilding(reader, stream, structure):
    """_rebuild's rebuilding of ``reader``'s table from ``stream``."""
    import pypdf._reader as pypdf_reader  # type: ignore
    from pypdf.errors import LimitReachedError  # type: ignore
    from pypdf.generic import DictionaryObject, IndirectObject  # type: ignore
    from pypdf.generic import NameObject, read_object  # type: ignore
    from pypdf._utils import read_until_whitespace  # type: ignore
    from pypdf._utils import skip_over_whitespace  # type: ignore
    reader.xref = {}
    stream.seek(0, 0)
    stream_data = stream.read(-1)

    try:
        for object_number, generation_number, object_start in \
                reader._find_pdf_objects(stream_data):
            structure.watch()
            if generation_number not in reader.xref:
                reader.xref[generation_number] = {}
            reader.xref[generation_number][object_number] = object_start
    except _EntriesFull:
        pass

    pypdf_reader.logger_warning("parsing for Object Streams",
                                source="pypdf._reader")
    # Each entry the object streams make it write counts from here on, with
    # the objects it found (_Structure.write): the file's own, each once as
    # before, bounded by the file's marks of "N G obj".
    structure.written = structure.listed
    xref_stream_trailers = []
    for generation_number in reader.xref:
        for object_number in reader.xref[generation_number]:
            structure.watch()
            object_start = reader.xref[generation_number][object_number]
            stream.seek(object_start, 0)
            try:
                _ = reader.read_object_header(stream)
                obj = _parsed_in_flight(structure, stream, read_object, reader)
                object_type = obj.get("/Type", "")
                if object_type == "/XRef":
                    trailer = DictionaryObject()
                    for key in reader._XREF_STREAM_TRAILER_KEYS:
                        if key in obj:
                            trailer[NameObject(key)] = obj.raw_get(key)
                    xref_stream_trailers.append((object_start, trailer))
                    continue
                if object_type != "/ObjStm":
                    continue
                try:
                    with structure.decoding():
                        data = obj.get_data()
                except LimitReachedError as exc:
                    if exc is structure.fired:
                        continue            # past the budget: said
                    raise
                structure.charge(obj)
                object_stream = _ObjectStreamBytes(
                    data, OBJECT_STREAM_READS_A_BYTE * len(data)
                    + OBJECT_STREAM_READS)
                actual_count = 0
                try:
                    while True:
                        current = read_until_whitespace(
                            object_stream,
                            max_bytes=IndirectObject._MAXIMUM_PART_LENGTH,
                            strict=reader.strict)
                        if not current.isdigit():
                            break
                        inner_object_number = int(current)
                        skip_over_whitespace(object_stream)
                        object_stream.seek(-1, 1)
                        current = read_until_whitespace(
                            object_stream,
                            max_bytes=IndirectObject._MAXIMUM_PART_LENGTH,
                            strict=reader.strict)
                        if not current.isdigit():  # pragma: no cover
                            break  # pragma: no cover
                        inner_offset = int(current)
                        structure.write()
                        reader.xref_objStm[inner_object_number] = (
                            object_number, inner_offset)
                        actual_count += 1
                except _EntriesFull:
                    pass
                except _ParseBound:
                    if object_stream.timed_out:
                        raise _ReaderBound("deadline") from None
                    structure.refused.add(object_number)
                    continue
                expected_count = obj["/N"]
                if actual_count != expected_count:  # pragma: no cover
                    pypdf_reader.logger_warning(  # pragma: no cover
                        ("found %(actual_count)d objects within "
                         "Object(%(object_number)d,%(generation_number)d) "
                         "whereas %(expected)d expected"),
                        source="pypdf._reader",
                        actual_count=actual_count,
                        object_number=object_number,
                        generation_number=generation_number,
                        expected=expected_count,
                    )
            except LimitReachedError:
                # Do not let the broad recovery below bypass the token-length
                # limit when strict parsing is enabled.
                raise
            except Exception:  # noqa: BLE001 - could be multiple causes
                pass

    stream.seek(0, 0)
    trailers = list(xref_stream_trailers)
    listed = 0              # what the trailers listed hold, charged meanwhile
    try:
        for position in reader._find_pdf_trailers(stream_data):
            structure.watch()
            stream.seek(position, 0)
            trailer = _parsed_in_flight(structure, stream, read_object, reader)
            amount = _held_by(trailer)
            listed += amount
            structure.hold(amount)
            trailers.append((position, trailer))
        # Here, we are parsing the file from start to end, the new data have
        # to erase the existing.
        for _, new_trailer in sorted(trailers, key=operator.itemgetter(0)):
            for key, value in new_trailer.items():
                reader.trailer[key] = value
    finally:
        structure.give_back(listed)     # what is kept: with the trailer


def _parsed_in_flight(structure, stream, read_object, reader):
    """read_object(stream, reader) of what the reader does not keep -- an
    object a rebuild of a lost table reads, a trailer --, its reads counted
    as the parse goes (_Structure.in_flight), and nothing charged once
    parsed: pypdf's rebuild parsed every object of the file, a page of
    3,000,000 references too, uncounted."""
    structure.begin(sys._getframe())
    counted = structure.counting(stream)
    try:
        return read_object(stream, reader)
    finally:
        if counted:
            structure.uncounting(stream)
        structure.end()


def _read_pages(reader, size, logged):
    """(text, metadata, notes, whether a page's or a form's content was cut
    short) of ``reader``, a document of ``size`` bytes, whose pypdf log is
    counted in ``logged`` (_LogCount)."""
    bounds = _Bounds(reader, size, logged)
    # The page tree walked: past MAX_DOCUMENT_HELD there (_HeldBound), no
    # page is read, and the caller does not read with pypdf.
    pages = reader.pages
    count = len(pages)
    # clip() keeps the first MAX_CHARS characters. Once the pages joined so
    # far are longer, the rest only lengthens what is cut off: a 63 MB text
    # PDF held 60 million characters for a 200,000-character body.
    parts, cut = [], []
    held = None        # where the reader passed MAX_DOCUMENT_HELD, if it did
    try:
        _read_each(pages, bounds, logged, parts, cut)
    except _HeldBound as bound:
        held = bound.where
    try:
        info = (reader.metadata or {}) if held is None else {}
    except _HeldBound:
        info, held = {}, "reading its metadata"
    meta = {
        "pages": count,
        "title": _clean(info.get("/Title")),
        "author": _clean(info.get("/Author")),
        "producer": _clean(info.get("/Producer")),
        "engine": "pypdf",
    }
    meta = {k: v for k, v in meta.items() if v not in (None, "")}
    notes = [note for note in (_cut_note(cut, forms=False), *bounds.notes(),
                               _log_note(logged.page),
                               _held_note(reader.structure, held)) if note]
    return ("\n\n".join(parts), meta, notes,
            bool(cut) or bounds.truncated or bool(logged.page)
            or held is not None)


def _held_note(structure, where):
    """What sift says where the reader passed MAX_DOCUMENT_HELD reading the
    document's pages (``where``), else None."""
    if where is None:
        return None
    if where == "reading its metadata":
        return f"{_held_said(structure)}, {where}; it is not said"
    return (f"{_held_said(structure)}, {where}; the text of page "
            f"{where.rsplit(' ', 1)[1]} and after is not in the body")


def _read_each(pages, bounds, logged, parts, cut):
    """Read ``pages`` into ``parts``, what their content cut into ``cut``;
    past MAX_DOCUMENT_HELD, _HeldBound says on which page."""
    joined = -2
    for i, page in enumerate(pages):
        try:
            if not _read_one(i, page, bounds, logged, parts, cut):
                break
        except _HeldBound as bound:
            bound.where = f"reading page {i + 1}"
            raise
        joined += len(parts[-1]) + 2
        if joined > MAX_CHARS:
            break


def _read_one(i, page, bounds, logged, parts, cut):
    """Read page ``i`` (from 0) into ``parts``: False where the document ends
    before it."""
    # A per-file budget across pages: a document of many pages, each
    # of which pypdf reads whole, was bounded only by the request's
    # own timeout (a host application's editor runs this in the web request).
    # The budget is inert unless a caller set deadline.limit().
    deadline.check("a PDF")
    if logged.over:
        logged.page = i + 1
        return False
    own = _own_entries(page)
    try:
        sizes = _bound_content(page)
        if sizes:
            cut.append((i + 1, *sizes))
        text = bounds.read(page, i + 1)
    except _LogBound:
        logged.page = i + 1
        return False
    finally:
        # The reader lives as long as the document is read, and what
        # was decoded for a page stayed on it: 1,000 pages of 790 KB
        # of content held 800 MB, 47 MB dropped after each page.
        _restore(page, own)
        bounds.forget()
    # pypdf catches what a form raises and reads on: a budget spent
    # inside a form ends only that form, and the page came back short.
    # pypdf logs that, and whatever else it gives up on, through its
    # own logger, which sift neither configures nor mutes on purpose:
    # a library does not change a dependency's global logging, and
    # A host application's PDF codec gate listens for exactly those records.
    # sift counts them (_LogWatch), and ends the document past
    # MAX_OP_LOG_PER_DOCUMENT.
    deadline.check("a page of a PDF")
    parts.append(f"[page {i+1}]\n{text or ''}".strip())
    return True


def _log_note(page):
    if not page:
        return None
    return (f"pypdf logged more than {MAX_OP_LOG_PER_DOCUMENT} characters "
            f"reading this document, past them on page {page}; the text of "
            f"page {page} and after is not in the body")


def _own_entries(page):
    """The page's /Contents and /Resources as the file has them (None where
    it has none), to be put back once the page is read -- or None for the
    test's stand-in for pypdf, whose pages have nothing to put back."""
    try:
        return {key: page.get(key) for key in ("/Contents", "/Resources")}
    except AttributeError:
        return None


def _restore(page, own):
    """Put back what sift replaced on the reader's own page for pypdf -- a
    cut copy of its content, 800,000 bytes, held 433 MB over 500 pages -- and
    drop what pypdf decoded of its content streams."""
    if own is None:
        return
    for key, value in own.items():
        if page.get(key) is value:
            continue
        if value is None:
            del page[key]
        else:
            from pypdf.generic import NameObject  # type: ignore
            page[NameObject(key)] = value
    try:
        contents = page["/Contents"] if "/Contents" in page else None
    except Exception:  # noqa: BLE001 - pypdf met the same and decided
        return
    for stream in contents if isinstance(contents, list) else [contents]:
        if stream is not None:
            _forget_decoded(_held(stream))


def _bound_content(page):
    """Hand pypdf the page's content, cut to MAX_INFLATED bytes, as a
    ContentStream it parses as its own (_page_content), before extract_text.

    Stopping between pages bounds nothing when one page holds the text:
    pypdf parses a page's content whole, every operator into an object, and
    builds its text a line at a time, in more than linear time. 160,000
    lines on one page -- 12 MB of content in a 42 KB file -- took 243 s and
    131 MB, and 810,000 lines had not finished after 600 s, at 407 MB.

    Returns (bytes kept, bytes the content holds) when it cut, else None.
    """
    try:
        contents = page.get_contents()
        data = contents.get_data() if contents is not None else b""
    except deadline.DeadlineExceeded:
        raise
    except Exception:  # noqa: BLE001
        # extract_text meets the same content and decides about it, as before.
        return None
    if contents is None:
        return None         # pypdf reads nothing of it (_extract_text)
    from pypdf.generic import NameObject  # type: ignore
    original = page["/Contents"]
    kept = data[:_content_cut(data, MAX_INFLATED)]
    # The reader's own page, for as long as it is read: _restore puts the
    # page's own content back.
    page[NameObject("/Contents")] = _page_content(kept, page.pdf)
    for stream in original if isinstance(original, list) else [original]:
        _forget_decoded(_held(stream))
    return (len(kept), len(data)) if len(kept) < len(data) else None


def _held(value):
    """What the reader resolved ``value`` to, without resolving it again:
    get_contents resolved each element of the page's /Contents, and sift
    resolved each twice more to drop what pypdf decoded of it -- for an
    object the file lacks, three searches of the whole file."""
    from pypdf.generic import IndirectObject  # type: ignore
    if isinstance(value, IndirectObject):
        return value.pdf.cache_get_indirect_object(value.generation,
                                                   value.idnum)
    return value


def _forget_decoded(stream):
    """Drop the copy of ``stream``'s content that pypdf decoded and keeps on
    the reader's own object.

    Kept, it outlived the cut: eight forms of 70 MB, each cut to 800,000
    bytes, grew the process by 67 MB a form, to 593 MB. pypdf decodes the
    stream again should anything ask for it.
    """
    if getattr(stream, "decoded_self", None) is not None:
        stream.decoded_self = None


def _operator_check(logged):
    """What pypdf calls before every operator, the forms' own included, for
    a document whose pypdf log is counted in ``logged`` (_LogCount, or
    None): pypdf reads a page and every form it draws in one call, so a
    check between pages bounds nothing inside one -- a 972-byte page that
    drew one 200-line form 5,000 times took 74.7 s past a 60 s budget. Past
    MAX_OP_LOG_PER_DOCUMENT it ends the page (_LogBound): 3,000 draws of a
    form the page lacks log a record each (_page.py:1983 in 6.19.0). One
    per document, holding its _LogCount: looked up on every operator
    instead, it cost 2.7 % of reading pm.pdf."""
    check = deadline.check
    if logged is None:
        def each_operator(*_):
            check("a page of a PDF")
    else:
        def each_operator(*_):
            check("a page of a PDF")
            if logged.over:
                raise _LogBound()
    return each_operator


class _LogBound(BaseException):
    """What pypdf logged reading a document passed MAX_OP_LOG_PER_DOCUMENT:
    ends the page being read, and _read_pages the document. A BaseException,
    as KeyboardInterrupt is one: pypdf catches every Exception a form's
    reading raises, logs it and reads on (_page.py:1980-1987 in 6.19.0)."""


class _LogCount:
    """What pypdf logged reading one document, counted as it logs it
    (_LogWatch): its records and their characters, each record's message and
    LOG_RECORD_PREFIX; whether those passed MAX_OP_LOG_PER_DOCUMENT; the
    bytes of the content pypdf parses meanwhile, closed once they did
    (_operations); and the page the document ended on for it."""

    __slots__ = ("records", "characters", "over", "parsing", "page", "trip")

    def __init__(self):
        self.records = self.characters = self.page = 0
        self.over, self.parsing, self.trip = False, None, None

    def add(self, record):
        try:
            said = len(record.getMessage())
        except Exception:  # noqa: BLE001 - logging says so itself, and goes on
            said = 0
        self.records += 1
        self.characters += said + LOG_RECORD_PREFIX
        if self.characters > MAX_OP_LOG_PER_DOCUMENT:
            self.over = True
            if self.parsing is not None:
                self.parsing.close()
            if self.trip is not None:
                self.trip("log")        # the reader being made (_trip)


#: The _LogCount of the document the current thread reads, else None.
_log_count = contextvars.ContextVar("sift_pdf_log", default=None)


class _LogWatch(logging.Handler):
    """Counts each record pypdf logs into the _LogCount of the document the
    thread that logs it reads (_log_count): records of threads reading none
    are passed over. It formats nothing for anyone, filters nothing, and
    lets every record go on to whatever handles it -- where nothing else
    does, Python's last resort, as logging hands it one where no handler at
    all is found (Logger.callHandlers), and this one would be found.

    It takes no lock of logging's: not its own handler lock -- what it
    changes is the count of the thread's own document -- and not logging's
    module lock, which logging.getLogger takes. Taken under the handler lock
    (Handler.handle), the module lock was the reverse of logging's own order:
    dictConfig holds it and takes every handler's lock to shut it down, and
    a thread reading a noisy PDF while the host configured logging hung the
    process."""

    def handle(self, record):
        self.emit(record)
        return True

    def emit(self, record):
        logged = _log_count.get()
        if logged is not None:
            logged.add(record)
        logger = _logger_of(record.name)
        if self._alone(logger):
            _last_resort(record, logger)

    def _alone(self, logger):
        """Whether this is the one handler a record of ``logger`` reaches."""
        while logger:
            if any(handler is not self for handler in logger.handlers):
                return False
            if not logger.propagate:
                return True
            logger = logger.parent
        return True


def _logger_of(name):
    """The logger that logged a record of ``name``, looked up without
    logging's module lock: that logger, else -- a record made by hand --
    the logger logging.getLogger would make its parent, its nearest ancestor
    logging made, else the root. A dictionary's get is atomic."""
    loggers = logging.root.manager.loggerDict
    while name:
        logger = loggers.get(name)
        if isinstance(logger, logging.Logger):
            return logger
        name = name.rpartition(".")[0]
    return logging.root


def _last_resort(record, logger):
    """What Logger.callHandlers does with ``record`` of ``logger`` where no
    handler is found (Python 3.9 to 3.14)."""
    resort = logging.lastResort
    if resort is not None:
        if record.levelno >= resort.level:
            resort.handle(record)
        return
    manager = logger.manager
    if logging.raiseExceptions and not manager.emittedNoHandlerWarning:
        import sys
        sys.stderr.write(f'No handlers could be found for logger '
                         f'"{record.name}"\n')
        manager.emittedNoHandlerWarning = True


_LOG_WATCH = _LogWatch()
_watching = 0           # the documents read now, in all threads
_watching_lock = threading.Lock()


@contextlib.contextmanager
def _counting_pypdf_log():
    """A _LogCount of what pypdf logs in this context from now on: sift's
    _LogWatch is on pypdf's logger while any thread reads a document, and
    off once none does."""
    global _watching
    logged = _LogCount()
    token = _log_count.set(logged)
    logger = logging.getLogger("pypdf")
    with _watching_lock:
        _watching += 1
        logger.addHandler(_LOG_WATCH)       # again, where it was taken off
    try:
        yield logged
    finally:
        with _watching_lock:
            _watching -= 1
            if not _watching:
                logger.removeHandler(_LOG_WATCH)
        _log_count.reset(token)


def _operations(stream):
    """The operations of ``stream``, a ContentStream, parsed as pypdf parses
    them (ContentStream.operations in 6.19.0), where a document's pypdf log
    is counted and pypdf's parser is where it was -- from bytes that
    _LogCount.add closes once what pypdf logged of the document passes
    MAX_OP_LOG_PER_DOCUMENT: pypdf's next read of them raises, and the parse
    ends (_LogBound). pypdf logs as it parses -- a number it cannot read, a
    name, a dictionary -- and a page's content of 800,000 bytes logged
    200,000 records before its first operator. A check of the count on
    every read cost the parse 8 to 22 %, bytes of a class of sift's own 4
    %."""
    from pypdf.generic import ContentStream  # type: ignore
    logged = _log_count.get()
    parse = getattr(stream, "_parse_content_stream", None)
    data = getattr(stream, "_data", None)
    if logged is None or parse is None or not isinstance(data, bytes) \
            or getattr(stream, "_operations", None) != []:
        return ContentStream.operations.fget(stream)
    if logged.over:
        raise _LogBound()
    if data:
        source = io.BytesIO(data)
        logged.parsing = source
        try:
            parse(source)
        except Exception:  # noqa: BLE001 - what reading closed bytes raises
            if source.closed:
                raise _LogBound() from None
            raise
        finally:
            logged.parsing = None
        stream._data = b""
    return stream._operations


def _page_content(data, pdf):
    """A ContentStream of ``data`` pypdf reads as a page's content: its
    operations parsed where pypdf parses them -- after the page's fonts, and
    not at all where the page has no resources (_extract_text in 6.19.0) --
    by _operations."""
    from pypdf.generic import ContentStream  # type: ignore
    global _PageContent
    if _PageContent is None or _PageContent.__base__ is not ContentStream:

        class PageContent(ContentStream):
            @property
            def operations(self):
                return _operations(self)

            @operations.setter
            def operations(self, operations):
                ContentStream.operations.fset(self, operations)
        _PageContent = PageContent
    content = _PageContent(None, pdf, "bytes")
    content.set_data(data)
    return content


_PageContent = None


class _Form:
    """What sift keeps of a form XObject for the reader's life: its
    content's length as pypdf is handed it, and its cut (bytes kept, bytes
    held) until that is said -- once, of the page that first reads it."""

    __slots__ = ("size", "cut")

    def __init__(self, size, cut):
        self.size, self.cut = size, cut


class _Bounds:
    """What pypdf is handed in place of the forms and fonts of one reader.

    pypdf reads a page and every form it draws in one call to extract_text,
    and at the start of each -- the page, and every single draw -- it builds
    a Font for every name the resources' /Font holds, reading whatever the
    font holds. sift bounds both where pypdf meets them: the page's fonts
    before extract_text, a form and its fonts when pypdf draws it, through
    PageObject.extract_xform_text, which pypdf calls for every form it draws,
    a form's forms included (read in the source of pypdf 6.8.0, 6.14.2,
    6.16.2 and 6.19.0). Cutting forms before the page is read would decode
    forms the page never draws.

    Each form and each font is measured once per reader and said once. What
    pypdf decoded for a page, and sift's cut copies, go when the page is
    read: a letterhead is decoded again on every page that draws it. What
    pypdf parsed of a form is kept for the reader's life, up to
    MAX_FORM_CONTENT bytes of the forms' content in all, and past that for
    one page: the operations of 800,000 bytes of path content hold 35 MB,
    and a document whose every page has a form of its own would hold all of
    them. A letterhead of that size on 5 pages took 4.2 s parsed on every
    page, 1.3 s parsed once.
    """

    def __init__(self, pdf, size=0, logged=None):
        self.pdf = pdf
        self.fonts = {}         # id -> (font, its _FontCost)
        self.forms = {}         # id -> (form, a _Form, or None for a form
                                #        pypdf cannot read)
        self.sizes = {}         # (id, cap) -> (stream, its decoded size)
        self.readings = {}      # id -> (map, its _MapReading, or None
                                #        where pypdf cannot decode it, what
                                #        counting it cost sift)
        self.contents = {}      # (length, digest) of a map's decoded bytes
                                #        -> its _MapReading
        self.walks = {}         # id -> (a /W, its _WidthsWalk)
        self.named = {}         # id -> (a /Font that is no dictionary,
                                #        (what _named makes of it, what
                                #        walking it raised))
        # What each resolve pypdf makes of a reference to an object the file
        # lacks costs: a search of the whole file, and two records (_absent).
        self.missing = (size // MISSING_REF_BYTES_A_UNIT
                        + 2 * LOG_RECORD_COST + MISSING_REF_LOGGED)
        self.parsed = {}        # id -> a form's ContentStream, parsed
        self.held = 0           # the content bytes of those
        self.cut = []           # (page, bytes kept, bytes held) a form cut
        self.unreadable = []    # the page first drawing a form not read
        self.maps = 0           # fonts whose /ToUnicode map is left out
        self.skipped = []       # (page, form bytes read) a page with draws
                                # skipped
        self.spared = []        # (page, fonts without their parts, fonts
                                # left out) a page past its font budget
        self.hushed = [0, 0, 0] # past MAX_FONT_LOG_PER_DOCUMENT: fonts
                                # without their parts, fonts left out,
                                # the page of the first
        self.logged = 0         # what pypdf logs of the fonts handed so far
        self.load = 0           # the font load of the page being read
        self.fonts_held = 0     # what pypdf holds of the fonts built for it
        self.check = _operator_check(logged)   # before every operator
        self.promoted = False   # whether Python's collector ran over its
                                # older generations while it was read
        self.made = {}          # id -> a form's stand-in, for this page
        self.touched = {}       # id -> a stream decoded for this page
        self.rewritten = 0      # the bytes of the maps written as ranges
                                # kept for the reader's life
        self.passing = {}       # id -> a _MapReading whose map written as
                                # ranges is kept for this page alone
        self.dropping = []      # (dictionary, key) of what holds those

    @property
    def truncated(self):
        return bool(self.cut or self.unreadable or self.maps or self.skipped
                    or self.spared or self.hushed[0] or self.hushed[1])

    def read(self, page, number):
        """page.extract_text, with the fonts of the page and of every form it
        draws within the page's font budget, and every form cut, parsed once
        and counted.

        pypdf.generic is imported only where there is something to bound, as
        before: the test's stand-in for pypdf has pages and nothing else.

        ``spared`` counts the fonts handed without their parts and those left
        out, past the page's font budget and past the document's log bound."""
        self.load, self.fonts_held, spared = 0, 0, [0, 0, 0, 0]
        try:
            resources = page.get_inherited("/Resources")
        except Exception:  # noqa: BLE001
            resources = None    # pypdf meets the same and decides about it
        handed = self._hand(resources, spared)
        if handed is not resources:
            from pypdf.generic import NameObject  # type: ignore
            # The reader's own page, for as long as it is read.
            page[NameObject("/Resources")] = handed
        real, parsed, spent, over = page.extract_xform_text, {}, 0, False

        def extract_xform_text(xform, *args, **kwargs):
            nonlocal spent, over
            from pypdf.generic import StreamObject  # type: ignore
            if not isinstance(xform, StreamObject):
                return real(xform, *args, **kwargs)     # no content at all
            if over:
                # Nothing past the budget is decoded: each form decoded only
                # to learn its length, one page drawing 1,000 forms of
                # 790,000 bytes took 797 MB.
                return ""
            self.touched[id(xform)] = xform
            form = self._form(xform, number)
            if form is None:
                return ""       # not read: said once, and pypdf logged why
            try:
                own = _resources_of(xform)
            except Exception:  # noqa: BLE001 - pypdf meets the same, or not
                # Parents past MAX_INHERITED, or round a cycle -- pypdf gives
                # up on a cycle too -- or one pypdf cannot walk: not read,
                # said once, and pypdf logs why.
                self._unreadable(xform, number)
                raise
            if spent + form.size > MAX_FORM_CONTENT:
                self.skipped.append((number, spent))
                over = True
                return ""
            spent += form.size
            # A ContentStream parsed once, however often the form is drawn:
            # handed a stream, pypdf wraps and parses it on every draw. pypdf
            # keys its own cycle check on the form it resolved, before this
            # is called.
            stream = self.parsed.get(id(xform), parsed.get(id(xform)))
            if stream is None:
                stream = self._parse(xform, number)
                if self.held + form.size <= MAX_FORM_CONTENT:
                    self.held += form.size
                    self.parsed[id(xform)] = stream
                else:
                    parsed[id(xform)] = stream
            if form.cut is not None:
                # Said of the page that reads it: a form cut and then
                # skipped was said to be read in part, and not a byte of it
                # was.
                self.cut.append((number, *form.cut))
                form.cut = None
            return real(self._handing(stream, own, self._hand(own, spared)),
                        *args, **kwargs)
        page.extract_xform_text = extract_xform_text
        aged = _older_collections()
        try:
            return page.extract_text(visitor_operand_before=self.check)
        finally:
            del page.extract_xform_text     # and what it parsed goes with it
            # What pypdf built for the page may have been moved to Python's
            # oldest generation, where only a full collection finds it, if
            # the collector ran over the older generations meanwhile.
            self.promoted = aged is None or _older_collections() != aged
            if spared[0] or spared[1]:
                self.spared.append((number, *spared[:2]))
            if (spared[2] or spared[3]) and not self.hushed[2]:
                self.hushed[2] = number
            self.hushed[0] += spared[2]
            self.hushed[1] += spared[3]

    def forget(self):
        """Drop what pypdf decoded for the page just read, and sift's
        stand-ins for its forms: 500 pages with a form of their own held
        421 MB, 45 MB so; 300 with a font of their own and a 3.1 MB map
        936 MB, 53 MB so. Drop the maps written as ranges kept for the page
        alone, with the readings and the fonts that hold them: kept for the
        reader's life, 300 pages with a font of their own took 196 MB, 0.40
        MB a page. And count what pypdf holds of the fonts it
        built for it, which only the collector frees (_collect)."""
        for stream in self.touched.values():
            _forget_decoded(stream)
        self.touched, self.made = {}, {}
        for owner, key in self.dropping:
            owner.pop(key, None)
        self.passing, self.dropping = {}, []
        _collect(self.fonts_held, self.promoted)
        self.load, self.fonts_held = 0, 0

    def _form(self, xform, number):
        """The _Form of ``xform``, or None for a form pypdf cannot read.
        Raises what pypdf raised the first time, so that pypdf logs it as it
        did."""
        key = id(xform)
        if key not in self.forms:
            stand_in, cut = self._stand_in(xform, number)
            self.forms[key] = (xform, _Form(len(stand_in.get_data()), cut))
        return self.forms[key][1]

    def _stand_in(self, xform, number):
        """(what pypdf reads for ``xform`` on this page, its cut or None):
        the form itself, or a copy cut to MAX_INFLATED."""
        key = id(xform)
        if key not in self.made:
            try:
                self.made[key] = _bound_form(xform)
            except deadline.DeadlineExceeded:
                raise
            except Exception:  # noqa: BLE001 - pypdf raises varied errors
                # Decoded again on every draw, a form that inflates past
                # pypdf's 75,000,000-byte limit took 0.39 s drawn once and
                # 6.16 s drawn 100 times, and its text was left out without
                # a word.
                self._unreadable(xform, number)
                raise
        return self.made[key]

    def _unreadable(self, xform, number):
        self.forms[id(xform)] = (xform, None)
        self.unreadable.append(number)

    def _parse(self, xform, number):
        """A ContentStream of what pypdf reads for ``xform``, parsed as pypdf
        parses a form, with the form's own keys."""
        from pypdf.generic import ContentStream, NameObject  # type: ignore
        stand_in, _ = self._stand_in(xform, number)
        stream = ContentStream(stand_in, self.pdf, "bytes")
        for key, value in xform.items():
            if key not in ("/Filter", "/DecodeParms", "/Length"):
                stream[NameObject(key)] = value
        try:
            _operations(stream)
        except deadline.DeadlineExceeded:
            raise
        except Exception:  # noqa: BLE001 - pypdf raises varied errors
            # pypdf would give up on it again on every draw -- and a stream
            # that failed half-way would keep the operations read so far.
            self._unreadable(xform, number)
            raise
        return stream

    def _handing(self, stream, own, resources):
        """``stream``, or where the form's fonts are handed otherwise than
        its ``own`` resources hold them, the same parsed operations with
        ``resources``."""
        if resources is own:
            return stream
        from pypdf.generic import ContentStream, NameObject  # type: ignore
        handed = ContentStream(None, self.pdf, "bytes")
        for key, value in stream.items():
            handed[NameObject(key)] = value
        handed[NameObject("/Resources")] = resources
        handed.operations = stream.operations
        return handed

    def _hand(self, resources, spared):
        """``resources``, or a copy whose /Font holds what fits the page's
        font budget and the document's log bound: each name is charged
        FONT_COST, what pypdf does with its font whatever it is handed of it
        (_FontCost.kept) and what it reads of it beyond the dictionary
        (_FontCost.parts), in the order pypdf builds them. A name that does
        not fit whole is handed its font without those parts, and one that
        does not fit even so is left out; ``spared`` counts both, past the
        page's budget and past the document's log bound apart. What counting
        a font's map cost sift, the first time it measures the font, is
        charged first, whatever the font is handed: a page may name thousands
        of fonts with maps of their own, and charged only where a font was
        not handed whole, a page with fresh fonts took 1.85 times its budget. A name of an object the file lacks costs a search of the
        file and what pypdf logs of it (_absent), and a /Font that is no
        dictionary is handed as the dictionary pypdf reads it as
        (_named)."""
        try:
            fonts = resources["/Font"]
        except Exception:  # noqa: BLE001 - None, no /Font, a broken one
            return resources
        from pypdf.generic import DictionaryObject, NameObject  # type: ignore
        named = self._named(fonts)
        if not isinstance(named, DictionaryObject):
            return resources
        kept = None
        for at, name in enumerate(named):
            # 2,000 fonts with double-Flate maps of 70 MB took 117 s past a
            # 60 s budget, measured here before pypdf read a thing.
            deadline.check("a page of a PDF")
            if self.load + FONT_COST > MAX_FONT_LOAD_PER_PAGE:
                spared[1] += len(named) - at
                kept = kept if kept is not None else _first(named, at)
                break
            try:
                font, absent = self._resolve(named.raw_get(name))
            except deadline.DeadlineExceeded:
                raise
            except Exception:  # noqa: BLE001
                font, absent = None, False  # pypdf meets the same, decides
            # A name of an object the file lacks: pypdf looks for it in the
            # whole file on every build, and gives up on the name.
            cost, measured = (_FontCost(None, None, kept=self.missing,
                                        logged_kept=MISSING_REF_LOGGED), 0) \
                if absent else self._font(font)
            self.touched.update(cost.streams)
            self.load += measured
            bare = FONT_COST + cost.kept
            over = self.load + bare > MAX_FONT_LOAD_PER_PAGE
            if over or (self.logged + cost.logged_kept
                        > MAX_FONT_LOG_PER_DOCUMENT):
                # What it costs handed at all does not fit, and a font
                # without its parts still costs it: the names after it may.
                spared[1 if over else 3] += 1
                kept = kept if kept is not None else _first(named, at)
                continue
            fits = self.load + bare + cost.parts <= MAX_FONT_LOAD_PER_PAGE
            quiet = (self.logged + cost.logged_kept + cost.logged_parts
                     <= MAX_FONT_LOG_PER_DOCUMENT)
            if fits and quiet:
                handed = cost.whole
                self.load += bare + cost.parts
                self.logged += cost.logged_kept + cost.logged_parts
                self.fonts_held += cost.held_bare + cost.held_parts
            else:
                handed = cost.bare
                self.load += bare
                self.logged += cost.logged_kept
                self.fonts_held += cost.held_bare
                spared[0 if not fits else 2] += 1
            if handed is not font:
                kept = kept if kept is not None else _first(named, at)
                kept[name] = handed
            elif kept is not None:
                kept[name] = named.raw_get(name)
        if kept is None:
            if named is fonts:
                return resources
            kept = named
        copy = DictionaryObject(resources)
        copy[NameObject("/Font")] = kept
        return copy

    def _named(self, fonts):
        """``fonts``, or where it is an array or a string, which pypdf walks
        as it walks a /Font dictionary, the dictionary pypdf reads it as:
        ``fonts[entry]`` for every entry that indexes it, keyed by the entry
        (pypdf/_page.py:1882-1891 in 6.19.0). An array [1 4 0 R 1 1 ...]
        builds the font at index 1 once for every 1 in it, uncharged: 201
        builds of a font of 99,999 widths from a 1,132-byte file took 2.8 s,
        and a string of 10,000,000 characters walked 4.6 s.
        Built once per reader, and what pypdf raises walking it -- an entry
        indexing past the end -- raised again each time."""
        if not isinstance(fonts, (list, str, bytes)):
            return fonts        # a dictionary, or nothing pypdf builds from
        key = id(fonts)
        if key not in self.named:
            from pypdf.generic import DictionaryObject  # type: ignore
            named, raised = DictionaryObject(), None
            # A string's entries index nothing (str[str]); a byte string's
            # index bytes, which pypdf passes over (int.get_object()).
            entries = fonts if isinstance(fonts, list) else \
                [max(fonts)] if isinstance(fonts, bytes) and fonts else []
            for entry in entries:
                try:
                    value = fonts[entry]
                    value.get_object()
                except (AttributeError, TypeError):
                    continue            # as pypdf passes over it
                except Exception as exc:  # noqa: BLE001 - as pypdf raises it
                    raised = (type(exc), exc.args)
                    break
                named[entry] = value
            self.named[key] = (fonts, (named, raised))
        named, raised = self.named[key][1]
        if raised is not None:
            raise raised[0](*raised[1])
        return named

    def _font(self, font):
        """(the _FontCost of ``font``, measured once per reader; what counting
        its map cost sift now (_MapReading.read), or nothing where it had
        measured it before)."""
        from pypdf.generic import DictionaryObject  # type: ignore
        if not isinstance(font, DictionaryObject):
            return _FontCost(font, font), 0
        if id(font) not in self.fonts:
            cost, measured = self._measure(font)
            self.fonts[id(font)] = (font, cost)
            return cost, measured
        return self.fonts[id(font)][1], 0

    def _measure(self, font):
        """(the _FontCost of ``font``, what counting its map cost sift). Whole,
        it is still without a /ToUnicode map that costs more than
        MAX_TO_UNICODE. A font whose references to objects the file lacks
        alone cost more than a page's budget is not read further: it would
        be left out on every page, and each of them is a search of the whole
        file for sift as well (_absent)."""
        lacking = self._lacking(font) * self.missing
        if FONT_COST + lacking > MAX_FONT_LOAD_PER_PAGE:
            return _FontCost(font, font, kept=lacking), 0
        whole, streams, measured = _without_long_boxes(font), {}, 0
        parts = held = logged = 0
        cmap = _map_of(whole)
        if cmap is not None:
            streams[id(cmap)] = cmap
            fresh = id(cmap) not in self.readings
            reading = self._reading(cmap)
            measured += self.readings[id(cmap)][2] if fresh else 0
            if reading is not None and reading.units > MAX_TO_UNICODE:
                # A copy: the reader's own font stays as the file has it.
                # Without the map pypdf reads a Type1 font's font file
                # instead, charged below.
                whole = _without(whole, "/ToUnicode")
                self.maps += 1
            elif reading is not None:
                if reading.ranges is not None:
                    # A copy, as above, with the map pypdf reads for less --
                    # measured again, where the map is kept for the page
                    # alone, on the next page that names the font.
                    whole = _with_map(whole, reading.ranges[0])
                    self._for_the_page(reading, self.fonts, id(font))
                    reading = reading.ranges[1]
                parts += reading.units
                held += MAP_ENTRY_HELD * reading.entries
                logged += reading.logged
        found = _type1_program(whole)
        program = _font_file(found)
        if program is not None:
            streams[id(program)] = program
            size = self._size(program, MAX_FONT_LOAD_PER_PAGE) or 0
            parts += size
            held += FONT_FILE_HELD * size
        for walk in self._widths(whole):
            parts += (WIDTH_COST * max(walk.widths, walk.elements)
                      + LOG_RECORD_COST * walk.records + walk.logged
                      + self.missing * walk.missing)
            held += WIDTH_HELD * walk.widths
            logged += walk.logged + MISSING_REF_LOGGED * walk.missing
        differences, said = _differences(whole)
        parts += differences + (LOG_RECORD_COST + said if said else 0)
        logged += said
        descendants = max(len(_descendants(whole)) - 1, 0)
        parts += _glyph_procedures(whole) + FONT_COST * descendants
        held += FONT_HELD * descendants
        said = _type1c_logged(whole, found)
        parts += LOG_RECORD_COST + said if said else 0
        logged += said
        strings, said = _strings(whole), _encoding_logged(whole)
        return _FontCost(
            whole, _stripped(whole) if parts else whole, parts,
            strings + (LOG_RECORD_COST + said if said else 0) + lacking,
            FONT_HELD + strings, held,
            said + MISSING_REF_LOGGED * (lacking // self.missing), logged,
            streams), measured

    def _reading(self, cmap):
        """The _MapReading of the map ``cmap``, once per reader, or None
        where pypdf cannot decode it: counted from its decoded bytes where
        they are within MAX_TO_UNICODE -- pypdf decodes it to read it, and
        the copy goes with the page (forget) --, else past the bound by
        their number alone. A map whose bytes sift counted before, in
        another stream, is not counted again: producers write the same map
        into a font of every page, and counting it is what sift spends most
        on such a page. What counting it cost sift is kept beside it: its
        digest a unit per MAP_HASH_BYTES_A_UNIT bytes, and its reading.

        A map written as ranges (_ranged_map) is kept with its reading as
        long as the reader while those kept fit MAX_MAPS_REWRITTEN bytes,
        else for the page alone (forget): counted again on a later page, and
        charged to it, as the first time."""
        key = id(cmap)
        if key not in self.readings:
            size = self._size(cmap, MAX_TO_UNICODE)
            reading, spent = None if size is None else _MapReading(size), 0
            if size is not None and size <= MAX_TO_UNICODE:
                try:
                    data = cmap.get_data()
                    digest = (len(data), hashlib.blake2b(
                        data, digest_size=16).digest())
                    spent = len(data) // MAP_HASH_BYTES_A_UNIT
                    reading = self.contents.get(digest)
                    if reading is None:
                        fours = []
                        reading = _map_reading(data, MAX_TO_UNICODE,
                                               fours=fours)
                        self.contents[digest] = reading
                        spent += reading.read + _ranged_map(reading, fours)
                        self._keep(reading, digest)
                except deadline.DeadlineExceeded:
                    raise
                except Exception:  # noqa: BLE001 - pypdf raises varied errors
                    reading = None
            self.readings[key] = (cmap, reading, spent)
            self._for_the_page(reading, self.readings, key)
        return self.readings[key][1]

    def _keep(self, reading, digest):
        """Keep the map ``reading`` wrote as ranges, if any, for the reader's
        life where it fits MAX_MAPS_REWRITTEN with those kept so far, else
        for the page alone."""
        if reading.ranges is None:
            return
        size = len(reading.ranges[0])
        if self.rewritten + size <= MAX_MAPS_REWRITTEN:
            self.rewritten += size
        else:
            self.passing[id(reading)] = reading
            self._for_the_page(reading, self.contents, digest)

    def _for_the_page(self, reading, owner, key):
        """Where the map ``reading`` wrote as ranges is kept for the page
        alone (_keep), drop ``owner[key]`` -- what holds it or what it was
        measured for -- with it when the page is read (forget): kept past
        the page, it would keep the map. Every cache that can hold such a
        map registers here."""
        if id(reading) in self.passing:
            self.dropping.append((owner, key))

    def _size(self, stream, cap):
        """_decoded_size, once per stream and cap: fonts share maps and font
        files."""
        key = (id(stream), cap)
        if key not in self.sizes:
            self.sizes[key] = (stream, _decoded_size(stream, cap))
        return self.sizes[key][1]

    def _widths(self, font):
        """The _WidthsWalk of each /W pypdf walks for ``font``: a composite
        font's descendants', each walked once per reader."""
        found = []
        for descendant in _descendants(font):
            try:
                raw = dict.get(self._resolve(descendant)[0], "/W")
                widths, absent = self._resolve(raw)
            except deadline.DeadlineExceeded:
                raise
            except Exception:  # noqa: BLE001 - none, or pypdf decides
                continue
            if widths is None and not absent:
                continue
            key = id(raw) if absent else id(widths)
            if key not in self.walks:
                walk = _WidthsWalk()
                if absent:
                    walk.missing = 1        # pypdf resolves it, and stops
                else:
                    walk = _walk_widths(widths, self._resolve)
                self.walks[key] = (raw if absent else widths, walk)
            found.append(self.walks[key][1])
        return found

    def _resolve(self, value):
        """(``value`` resolved, whether it is a reference to an object the
        file lacks, left unresolved, None)."""
        from pypdf.generic import IndirectObject  # type: ignore
        if not isinstance(value, IndirectObject):
            return value, False
        if self._absent(value):
            return None, True
        return value.get_object(), False

    def _absent(self, reference):
        """Whether pypdf looks for ``reference`` in the whole file whenever it
        resolves it: an object the cross-reference tables lack, and that it
        did not find so far -- PdfReader.get_object in 6.19.0 caches what it
        did not find as None, searches the file for it again next time, and
        logs it twice. pypdf resolves every /W element on every Font it
        builds, and sift resolved each walking it: 10,000 such elements in
        one font of a 71 KB file, named once, took 22 s (found fixing a review); 0.13 s now."""
        pdf = self.pdf
        try:
            if pdf.resolved_objects.get(
                    (reference.generation, reference.idnum)) is not None:
                return False
            if reference.generation == 0 and \
                    reference.idnum in pdf.xref_objStm:
                return False
            return reference.idnum not in pdf.xref.get(reference.generation,
                                                       ())
        except (AttributeError, TypeError):
            return False        # a reader of another shape: resolved

    def _lacking(self, font):
        """How often pypdf resolves a reference to an object the file lacks
        building a Font of ``font``, whatever it is handed of it, at most:
        twice for every such value of the font, of its /FontDescriptor and
        /Encoding, of each descendant and its descriptor, and once for an
        element of its /Widths (pypdf gives up on the font there)."""
        from pypdf.generic import ArrayObject, DictionaryObject  # type: ignore
        from pypdf.generic import IndirectObject  # type: ignore

        def absent_in(owner):
            values = dict.values(owner) if isinstance(owner, DictionaryObject) \
                else list.__iter__(owner) if isinstance(owner, ArrayObject) \
                else ()
            return sum(1 for value in values
                       if isinstance(value, IndirectObject)
                       and self._absent(value))

        def at(owner, key):
            try:
                return self._resolve(dict.get(owner, key))[0]
            except deadline.DeadlineExceeded:
                raise
            except Exception:  # noqa: BLE001 - pypdf meets the same, decides
                return None
        count = 2 * absent_in(font)
        count += 2 * absent_in(at(font, "/FontDescriptor"))
        count += 2 * absent_in(at(font, "/Encoding"))
        count += min(absent_in(at(font, "/Widths")), 1)
        descendants = at(font, "/DescendantFonts")
        for raw in (list.__iter__(descendants)
                    if isinstance(descendants, ArrayObject) else ()):
            try:
                descendant, absent = self._resolve(raw)
            except deadline.DeadlineExceeded:
                raise
            except Exception:  # noqa: BLE001 - pypdf meets the same, decides
                continue
            if isinstance(descendant, DictionaryObject):
                count += 2 * absent_in(descendant)
                count += 2 * absent_in(at(descendant, "/FontDescriptor"))
            count += 2 * absent
        return count

    def notes(self):
        return [_cut_note(self.cut, forms=True), self._unreadable_note(),
                self._skipped_note(), self._maps_note(),
                self._spared_note(), self._hushed_note()]

    def _hushed_note(self):
        stripped, left_out, first = self.hushed
        if not (stripped or left_out):
            return None
        what = _spared_fonts(stripped, left_out)
        return (f"pypdf would log more than {MAX_FONT_LOG_PER_DOCUMENT} "
                f"characters building the fonts of this document, the first "
                f"past it on page {first}: {what}; their text may be wrong")

    def _spared_note(self):
        if not self.spared:
            return None
        what = _spared_fonts(sum(page[1] for page in self.spared),
                             sum(page[2] for page in self.spared))
        if len(self.spared) == 1:
            return (f"page {self.spared[0][0]} loads fonts past a page's "
                    f"font budget of {MAX_FONT_LOAD_PER_PAGE}: {what}; their "
                    f"text may be wrong")
        return (f"{len(self.spared)} pages load fonts past a page's font "
                f"budget of {MAX_FONT_LOAD_PER_PAGE}, the first of them page "
                f"{self.spared[0][0]}: {what}; their text may be wrong")

    def _unreadable_note(self):
        if not self.unreadable:
            return None
        if len(self.unreadable) == 1:
            return (f"could not read a form XObject page {self.unreadable[0]}"
                    f" draws; its text is not in the body")
        return (f"could not read {len(self.unreadable)} form XObjects, the "
                f"first of them drawn on page {self.unreadable[0]}; their "
                f"text is not in the body")

    def _skipped_note(self):
        if not self.skipped:
            return None
        if len(self.skipped) == 1:
            number, spent = self.skipped[0]
            return (f"skipped the form XObjects page {number} draws after "
                    f"the first {spent} bytes of their content; their text "
                    f"is not in the body")
        return (f"skipped the form XObjects {len(self.skipped)} pages draw "
                f"past the first {MAX_FORM_CONTENT} bytes of their content "
                f"on each, the first of them page {self.skipped[0][0]}; "
                f"their text is not in the body")

    def _maps_note(self):
        if not self.maps:
            return None
        if self.maps == 1:
            return (f"ignored the ToUnicode map of a font past "
                    f"{MAX_TO_UNICODE} bytes' worth of reading; its text may "
                    f"be wrong")
        return (f"ignored the ToUnicode maps of {self.maps} fonts past "
                f"{MAX_TO_UNICODE} bytes' worth of reading each; their text "
                f"may be wrong")


def _spared_fonts(stripped, left_out):
    """What the notes say of fonts handed to pypdf without their parts,
    ``stripped``, and of those ``left_out``."""
    return ", ".join(part for part in (
        stripped and f"{stripped} of them without their ToUnicode maps, "
                     f"widths, encoding differences, glyph procedures and "
                     f"font files",
        left_out and f"{left_out} left out altogether") if part)


def _first(fonts, count):
    """A /Font dictionary of the first ``count`` names in ``fonts``, as the
    file has them."""
    from itertools import islice
    from pypdf.generic import DictionaryObject, NameObject  # type: ignore
    return DictionaryObject({NameObject(name): fonts.raw_get(name)
                             for name in islice(fonts, count)})


def _resources_of(form):
    """A form's /Resources as pypdf finds them, resolved, or None: its own,
    or where it has none those it inherits through /Parent, walked as pypdf
    walks it (pypdf._page._get_page_resources, get_inherited in
    pypdf/generic/_data_structures.py:442-470 in 6.19.0). sift read a form's
    own only, and pypdf built every font a parent names unbudgeted: 2.2 KB
    took 1,021 MB, drawn 20 times 7,641 MB. Raises past
    MAX_INHERITED parents, which a cycle among them is, as pypdf raises on
    the cycle."""
    current = form
    for _ in range(MAX_INHERITED + 1):
        if "/Resources" in current:
            resources = current["/Resources"]
            return resources.get_object() if resources is not None else None
        if "/Parent" not in current:
            return None
        current = current["/Parent"].get_object()
    from pypdf.errors import LimitReachedError  # type: ignore
    raise LimitReachedError(f"a form XObject's /Resources inherited through "
                            f"more than {MAX_INHERITED} /Parent entries")


def _strings(font):
    """How many characters the strings pypdf copies of ``font`` on every
    Font it builds hold: its /BaseFont and /Subtype, copied into the Font
    (Font.from_font_resource in 6.19.0) -- an /Encoding it logs is
    _encoding_logged's. As names, pypdf reads them to 4,096 characters
    (NameObject._LENGTH_LIMIT), 8 KB a Font at most, within the 25 KB peak
    FONT_COST was measured at; a string has no bound. 2,000 names of a font
    whose /BaseFont was a string of 1,000,000 characters held 1,968 MB. A character is charged as a map byte: it costs pypdf less,
    one to four bytes copied; and it is held as long as the Font is."""
    from pypdf.generic import NameObject  # type: ignore
    count = 0
    for key in ("/BaseFont", "/Subtype"):
        try:
            value = font[key] if key in font else None
        except Exception:  # noqa: BLE001 - pypdf meets the same and decides
            continue
        if isinstance(value, str) and not isinstance(value, NameObject):
            count += len(value)
    return count


class _FontCost:
    """What pypdf does with one font on every Font it builds of it, measured
    once per reader (_Bounds._measure): ``whole`` and ``bare``, the font as
    pypdf is handed it whole and without its parts (_stripped); ``parts``,
    what reading those parts costs pypdf, in map bytes' worth; ``kept``, what
    the font costs past FONT_COST whatever it is handed of it; the bytes a
    Font of it holds bare, and those its parts add; the characters pypdf logs
    of it whatever it is handed, and those its parts add; and the streams
    pypdf decodes of it."""

    __slots__ = ("whole", "bare", "parts", "kept", "held_bare", "held_parts",
                 "logged_kept", "logged_parts", "streams")

    def __init__(self, whole, bare, parts=0, kept=0, held_bare=0,
                 held_parts=0, logged_kept=0, logged_parts=0, streams=None):
        self.whole, self.bare, self.parts, self.kept = whole, bare, parts, kept
        self.held_bare, self.held_parts = held_bare, held_parts
        self.logged_kept, self.logged_parts = logged_kept, logged_parts
        self.streams = streams or {}


#: What pypdf's log records say around what they format (_cmap and _font in
#: 6.19.0), in characters.
_BROKEN_LINE = len("Skipping broken line : ") + 80     # and the error's text
_ODD_LINE = len("Ignoring final token of odd-length line .")
_BAD_HEX = len("Got invalid hex string: Non-hexadecimal digit found ()")
_NOT_AN_ARRAY = len("Font encoding differences are not an array: ")
_ADVANCED = len("Advanced encoding  not implemented yet")
_NO_FONTTOOLS = len(
    "fontTools is required to fully parse the encoding of a CFF Type1 font "
    "in font dictionary , but is not installed. Consider installing "
    "fontTools if you encounter encoding problems.")
_NOT_A_WIDTH = len("Expected numeric value for width, got . Ignoring it.")
_NOT_A_DEFINITION = len("Invalid font width definition. Last element: .")

#: What becomes of a line pypdf reads of a map: it logs it and reads on, or
#: gives up on the map and the Font with it.
_LOGGED, _RAISED = "logged", "raised"
_HEX_DIGITS = re.compile(rb"[0-9A-Fa-f]*")


class _MapReading:
    """What pypdf does reading one /ToUnicode map on every Font it builds
    (_map_reading): ``units``, what it costs, in map bytes' worth;
    ``entries``, those pypdf builds; ``records`` and ``logged``, the records
    it logs and their characters; ``raised``, whether it gives up on the map,
    and the Font with it; ``read``, what counting it cost sift, in the same
    units (_map_reading); and ``ranges``, the same map in fewer units where
    pypdf is handed that instead (_Bounds._reading)."""

    __slots__ = ("units", "entries", "records", "logged", "raised", "read",
                 "ranges")

    def __init__(self, units):
        self.units, self.entries, self.records, self.logged = units, 0, 0, 0
        self.raised, self.read = False, 0
        self.ranges = None      # (the map as _as_ranges writes it, its
                                #  _MapReading), where pypdf is handed that


def _map_reading(data, cap, runs=True, fours=None):
    """The _MapReading of the /ToUnicode map ``data``, decoded, counted as
    pypdf reads it and without building a thing, no further than one unit
    past ``cap``.

    pypdf prepares the map (_cmap.prepare_cm in 6.19.0), splits it into
    lines and reads each (process_cm_line): a line of a bfrange section adds
    an entry for every code it spans (parse_bfrange), a line of a bfchar
    section one for each pair (parse_bfchar), a broken line is logged and
    passed over, and past 100,000 entries, at a code of more than 16 digits
    and the like pypdf gives up on the map and on the Font. 191 bytes of two
    bfrange lines build 98,304 entries: sift charged their 191 bytes, and 60
    names of the font took 10.7 s and 674 MB.

    Charged: every byte -- preparing it, whose costliest byte is the unit --,
    every line read past the first check MAP_LINE_COST, every entry of a
    bfrange MAP_RANGE_ENTRY_COST and of a bfchar MAP_CHAR_ENTRY_COST, the
    list elements a bfchar line of many pairs copies -- pypdf drops each
    pair by copying the rest -- a unit per MAP_COPIES_A_UNIT, and every
    record logged LOG_RECORD_COST and a unit a character.

    Counted in runs (``runs``): sift prepares the map as pypdf does, in steps
    that run in C (_prepared), and counts a run of well-formed lines in one
    match (_MapCount.char_run, range_run), walking one by one only the lines
    a run does not decide -- what it counts is what walking every line
    counts (a test holds them equal). Walking every line cost sift 0.82 to
    0.90 of what pypdf's reading costs, and 150 pages each with a font of
    its own of 20,902 bfchar entries took 27.1 s against 14.2 s.
    ``read``, what that cost sift, is a unit per MAP_PREPARE_BYTES_A_UNIT
    bytes, MAP_RUN_CHAR_COST a bfchar line of a run and MAP_RUN_RANGE_COST a
    bfrange line of one, and what pypdf's reading costs of every line
    walked, but a bfrange's entries. A list ``fours`` receives the map as
    _prepared leaves it, and then (start, end) of each run of bfchar lines
    of four digits each in a bfchar section alone (_as_ranges)."""
    reading = _MapReading(len(data))
    if reading.units > cap:
        reading.units = cap + 1
        return reading
    try:
        import pypdf._cmap as cmaps  # type: ignore
    except ImportError:
        return reading          # a pypdf of another shape: its bytes alone
    prepared = _prepared(data)
    count = _MapCount(getattr(cmaps, "MAPPING_DICTIONARY_SIZE_LIMIT", None),
                      getattr(cmaps, "MAX_CMAP_CODE_BYTES_LIMIT", None),
                      getattr(cmaps, "MAX_CMAP_STRING_BYTES_LIMIT", None))
    runs = runs and count.runs()
    if fours is not None:
        fours.append(prepared)
        count.fours = fours
    at, end = 0, len(prepared)
    while at <= end and not count.raised \
            and count.units(len(data)) <= cap:
        if runs:
            ran = count.run(prepared, at, len(data), cap)
            if ran != at:
                at = ran
                continue
        stop = prepared.find(b"\n", at)
        stop = end if stop < 0 else stop
        line, at = prepared[at:stop].strip(b" \t"), stop + 1
        if line and line[0] != 37:              # "%": a comment
            count.cost += MAP_LINE_COST
            count.read(line.replace(b"\t", b" "))
    reading.units = min(count.units(len(data)), cap + 1)
    reading.read = min(len(data) // MAP_PREPARE_BYTES_A_UNIT + count.cost
                       + count.counted, cap + 1)
    reading.entries, reading.raised = count.entries, count.raised
    reading.records, reading.logged = count.records, count.logged
    return reading


#: pypdf's prepare_cm in steps that run in C: a "<...>" of the map becomes
#: " ... ", its spaces left out and "." where it is empty, a "<" without a
#: ">" before the next "<" a " ". _MAP_STRING_ODD finds one whose spaces or
#: emptiness a template cannot write, _MAP_CLOSED_TWICE a ">" after the one
#: that closes a "<", which stays.
_MAP_STRING = re.compile(rb"<([^<>]*)>")
_MAP_STRING_ODD = re.compile(rb"<[^<>]* [^<>]*>|<>")
_MAP_CLOSED_TWICE = re.compile(rb">[^<]*>")
_HEX_DIGIT_BYTES = b"0123456789ABCDEFabcdef"


def _prepared(data):
    """pypdf's _cmap.prepare_cm of the map ``data`` (6.19.0), byte for byte --
    a test holds them equal -- without its loop in Python over every "<" of
    the map, which is 11.7 ms of the 77.6 ms pypdf spends reading a map of
    20,902 bfchar entries."""
    cm = (data.strip()
          .replace(b"beginbfchar", b"\nbeginbfchar\n")
          .replace(b"endbfchar", b"\nendbfchar\n")
          .replace(b"beginbfrange", b"\nbeginbfrange\n")
          .replace(b"endbfrange", b"\nendbfrange\n")
          .replace(b"<<", b"\n{\n")
          .replace(b">>", b"\n}\n"))
    first = cm.find(b"<")
    head, rest = (cm, b"") if first < 0 else (cm[:first], cm[first:])
    close = head.find(b">")         # pypdf reads the text before any "<" too
    if close >= 0:
        head = (head[:close].replace(b" ", b"") if close else b".") + b" " \
            + head[close + 1:]
    # Where every "<" holds hex digits and a ">" before the next, the
    # digits left out leave "<>" alone: nothing for a template to write.
    plain = b"<>" not in rest and not any(
        mark in rest.translate(None, _HEX_DIGIT_BYTES).replace(b"<>", b"")
        for mark in (b"<", b">"))
    if plain:
        rest = rest.replace(b"<", b" ").replace(b">", b" ")
    elif _MAP_STRING_ODD.search(rest):
        rest = _MAP_STRING.sub(_map_string, rest).replace(b"<", b" ")
    elif _MAP_CLOSED_TWICE.search(rest):
        rest = _MAP_STRING.sub(rb" \1 ", rest).replace(b"<", b" ")
    else:
        rest = rest.replace(b"<", b" ").replace(b">", b" ")
    return ((head + rest)
            .replace(b"[", b" [ ")
            .replace(b"]", b" ]\n ")
            .replace(b"\r", b"\n"))


def _as_ranges(data, cap):
    """The /ToUnicode map ``data`` as _prepared leaves it, with every
    MAP_RANGE_FROM or more bfchar lines of four digits each, in a bfchar
    section alone, that map consecutive codes onto consecutive strings
    written as one bfrange line between their own section's lines: pypdf
    builds the same entries in the same order (_cmap.parse_bfrange and
    parse_bfchar in 6.19.0 -- a test holds what it builds of both equal),
    at MAP_RANGE_ENTRY_COST an entry rather than a line's, an entry's and
    the line's bytes. Or None where no such lines are, where pypdf logs or
    gives up reading the map, or where it costs more than ``cap``.

    Written as _prepared leaves a map, and so as pypdf leaves it: without a
    "<" or ">" -- a map with either is left as it is -- pypdf prepares it to
    the same lines, blank ones and spaces aside. A full map of two-byte
    codes onto themselves costs pypdf 1.39 million units so, 2.93 million as
    written."""
    fours = []
    reading = _map_reading(data, cap, fours=fours)
    if reading.units > cap or reading.raised or reading.records:
        return None
    return _ranges_of(fours)[0]


def _ranges_of(fours):
    """(_as_ranges of the map ``fours`` holds as _map_reading leaves it --
    the map as _prepared leaves it, and its runs of four digits -- or None;
    the lines of those runs, each a line to rewrite)."""
    if len(fours) < 2 or b"<" in fours[0] or b">" in fours[0]:
        return None, 0
    from array import array
    import sys
    prepared, runs = fours[0], fours[1:]
    parts, done, merged, examined = [], 0, 0, 0
    for at, end in runs:
        codes = array("H", bytes.fromhex(
            prepared[at:end].translate(None, b" \t\n").decode("ascii")))
        if sys.byteorder == "little":
            codes.byteswap()
        sources, strings = codes[0::2], codes[1::2]
        examined += len(sources)
        breaks = [at_line for at_line, a, b, c, d in zip(
            range(1, len(sources)), sources[1:], sources, strings[1:], strings)
            if a - b != 1 or c - d != 1]
        lines = prepared[at:end].split(b"\n")
        written, ranges = [], []
        for first, stop in zip([0, *breaks], [*breaks, len(sources)]):
            if stop - first >= MAP_RANGE_FROM:
                ranges.append(b"%04X %04X %04X" % (
                    sources[first], sources[stop - 1], strings[first]))
                merged += stop - first
                continue
            if ranges:
                written.append(b"beginbfrange\n" + b"\n".join(ranges)
                               + b"\nendbfrange")
                ranges = []
            written.extend(lines[first:stop])
        if ranges:
            written.append(b"beginbfrange\n" + b"\n".join(ranges)
                           + b"\nendbfrange")
        parts += [prepared[done:at], b"\n".join(written), b"\n"]
        done = end
    if not merged:
        return None, examined
    parts.append(prepared[done:])
    return b"".join(parts), examined


def _ranged_map(reading, fours):
    """What rewriting a map as _as_ranges writes it cost sift, where
    pypdf's reading of it, ``reading``, costs MAP_RANGES_FROM units or more,
    and logs and raises nothing; ``fours`` is what _map_reading left of it.
    ``reading.ranges`` is set where pypdf's reading of the rewritten map
    costs a fourth less and counts the same entries, and no record."""
    if not MAP_RANGES_FROM <= reading.units <= MAX_TO_UNICODE \
            or reading.raised or reading.records:
        return 0
    ranges, examined = _ranges_of(fours)
    spent = MAP_RANGES_LINE_COST * examined
    if ranges is None:
        return spent
    shorter = _map_reading(ranges, MAX_TO_UNICODE)
    if shorter.units <= reading.units - reading.units // 4 \
            and (shorter.entries, shorter.records, shorter.raised) \
            == (reading.entries, 0, False):
        reading.ranges = (ranges, shorter)
    return spent + shorter.read


def _map_string(match):
    inner = match.group(1)
    return b" " + (inner.replace(b" ", b"") if inner else b".") + b" "


#: A run of lines of a bfchar section, as _prepared leaves them, that
#: _MapCount.char_line counts an entry each and nothing else: one pair of a
#: code of one byte or of two to eight in pairs, and a string of one byte,
#: of whole UTF-16 code units, or empty. The group is the last code.
#: _CHAR_RUN_OF_FOUR is the same of four digits each, as producers write
#: them, in a fourth of the time.
_CHAR_RUN = re.compile(
    rb"(?:[ \t]*((?:[0-9A-Fa-f]{4}){1,4}|[0-9A-Fa-f]{2})[ \t]+"
    rb"(?:(?:[0-9A-Fa-f]{4}){1,256}|[0-9A-Fa-f]{2}|\.)[ \t]*\n)+")
_CHAR_RUN_OF_FOUR = re.compile(
    rb"(?:[ \t]*[0-9A-Fa-f]{4}[ \t]+[0-9A-Fa-f]{4}[ \t]*\n)+")
#: A line of a bfrange section as _prepared leaves it -- a range of codes
#: of one byte or two onto a string of one UTF-16 code unit -- and a run of
#: them.
_RANGE_LINE = re.compile(
    rb"[ \t]*([0-9A-Fa-f]{4}|[0-9A-Fa-f]{2})[ \t]+"
    rb"([0-9A-Fa-f]{4}|[0-9A-Fa-f]{2})[ \t]+([0-9A-Fa-f]{4})[ \t]*\n")
_RANGE_RUN = re.compile(
    rb"(?:%s)+" % _RANGE_LINE.pattern.replace(b"(", b"(?:"))
#: Blank lines, and the lines that end a section and begin the next of its
#: kind -- a count between them or not -- as _prepared leaves them. The
#: group is the count's line.
_BLANK_LINES = re.compile(rb"(?:[ \t]*\n)+")
_TURNS = {kind: re.compile(
    rb"[ \t]*end%s[ \t]*\n(?:[ \t]*\n)*(?:([ \t]*[0-9]+[ \t]*\n)"
    rb"(?:[ \t]*\n)*)?[ \t]*begin%s[ \t]*\n" % (kind, kind))
    for kind in (b"bfchar", b"bfrange")}


class _MapCount:
    """pypdf's reading of a map, line by line, counted (_map_reading): its
    state as pypdf keeps it -- the section, the byte width of its codes, a
    list going on past its line -- and what it did so far. ``limit`` is the
    entries pypdf builds at most, ``code`` and ``string`` the longest code
    and string it takes, None where it takes any."""

    def __init__(self, limit, code, string):
        self.limit = 10 ** 18 if limit is None else limit
        self.code, self.string = code, string
        self.ranges = self.chars = False
        self.width = None       # the bytes of a code: map_dict[-1]
        self.multi = None       # (next code, last code) of a list going on
        self.entries = self.cost = self.copies = 0
        self.records = self.logged = 0
        self.raised = False
        self.counted = 0        # the part of cost sift pays counting it:
                                # the entries of a bfrange it does not, nor
                                # the lines it counts in runs
        self.walk = 0           # where read() counts lines one by one up to
                                # (run): a run past the cap or pypdf's entries,
                                # or with a line a run does not decide
        self.fours = None       # where runs of four digits go (_as_ranges)

    def units(self, size):
        """What pypdf's reading of the map costs so far, its ``size`` bytes
        with it."""
        return size + self.cost + self.copies // MAP_COPIES_A_UNIT

    def runs(self):
        """Whether pypdf takes every code and string _CHAR_RUN and
        _RANGE_LINE match, as limits the pypdf in use has go."""
        return (self.code is None or self.code >= 16) \
            and (self.string is None or self.string >= 1024)

    def run(self, prepared, at, size, cap):
        """Where a run of lines of ``prepared`` from ``at`` -- the start of a
        line -- ends that this counts as read() would count them, or ``at``
        where read() is to count the next line. ``size`` and ``cap`` are
        _map_reading's: the run ends where read() would stop at the cap."""
        if at < self.walk:
            return at
        blank = _BLANK_LINES.match(prepared, at)
        if blank is not None:
            return blank.end()          # read() passes over them
        ran = at
        if self.ranges and self.multi is None:
            ran = self.range_run(prepared, at, size, cap)
        elif self.chars and not self.ranges:
            ran = self.char_run(prepared, at, size, cap)
        if ran == at and self.ranges != self.chars:
            ran = self.turn(prepared, at, size, cap)
        return ran

    def turn(self, prepared, at, size, cap):
        """The lines that end the one section open and begin the next of its
        kind (_TURNS): MAP_LINE_COST each, a count between them read in no
        section, at what they cost pypdf to sift."""
        turn = _TURNS[b"bfrange" if self.ranges else b"bfchar"].match(
            prepared, at)
        if turn is None:
            return at
        lines = 2 if turn.group(1) is None else 3
        if self.units(size) + MAP_LINE_COST * (lines - 1) > cap:
            self.walk = turn.end()      # read() stops within it
            return at
        self.cost += MAP_LINE_COST * lines
        return turn.end()

    def char_run(self, prepared, at, size, cap):
        """A run of bfchar lines of one well-formed pair each (_CHAR_RUN):
        an entry each, and no copies, record or failure (char_line), at
        MAP_RUN_CHAR_COST a line to sift."""
        run, width = _CHAR_RUN_OF_FOUR.match(prepared, at), 2
        four = run is not None
        if run is None:
            run = _CHAR_RUN.match(prepared, at)
            if run is None:
                return at
            width = len(run.group(1)) // 2
        lines = prepared.count(b"\n", at, run.end())
        each = MAP_LINE_COST + MAP_CHAR_ENTRY_COST
        if self.entries + lines > self.limit \
                or self.units(size) + each * (lines - 1) > cap:
            self.walk = run.end()   # read() gives up, or stops, within it
            return at
        self.entries += lines
        self.cost += each * lines
        self.counted += (MAP_RUN_CHAR_COST - each) * lines
        self.width = width
        if four and self.fours is not None and self.multi is None:
            self.fours.append((at, run.end()))
        return run.end()

    def range_run(self, prepared, at, size, cap):
        """A run of bfrange lines of one range within four digits each
        (_RANGE_RUN): its codes' entries each, and no record or failure
        (range_line), at MAP_RUN_RANGE_COST a line to sift. Where a line of
        it is past its four digits, or the run past the cap or pypdf's
        entries, read() counts the run."""
        run = _RANGE_RUN.match(prepared, at)
        if run is None:
            return at
        firsts, lasts, ontos = zip(*_RANGE_LINE.findall(prepared, at,
                                                         run.end()))
        steps = list(map(operator.sub, map(int, lasts, repeat(16)),
                         map(int, firsts, repeat(16))))
        lines = len(steps)
        entries = sum(map(max, steps, repeat(-1))) + lines
        each = MAP_LINE_COST
        before_last = self.units(size) + each * (lines - 1) \
            + MAP_RANGE_ENTRY_COST * (entries - max(steps[-1] + 1, 0))
        if max(map(operator.add, map(int, ontos, repeat(16)), steps)) \
                > 0xFFFF or self.entries + entries > self.limit \
                or before_last > cap:
            self.walk = run.end()   # read() logs, gives up or stops within it
            return at
        self.width = (max(len(firsts[-1]), len(lasts[-1])) + 1) // 2
        self.entries += entries
        self.cost += each * lines + MAP_RANGE_ENTRY_COST * entries
        self.counted += (MAP_RUN_RANGE_COST - each) * lines \
            - MAP_RANGE_ENTRY_COST * entries
        self.multi = None
        return run.end()

    def read(self, line):
        """process_cm_line, past its first check."""
        if b"beginbfrange" in line:
            self.ranges = True
        elif b"endbfrange" in line:
            self.ranges = False
        elif b"beginbfchar" in line:
            self.chars = True
        elif b"endbfchar" in line:
            self.chars = False
        elif self.ranges:
            self.range_line(line)
        elif self.chars:
            self.char_line(line)

    def record(self, logged):
        self.records += 1
        self.logged += logged
        self.cost += LOG_RECORD_COST + logged

    def fail(self, how, line=b""):
        if how == _RAISED:
            self.raised = True
        else:
            self.record(_BROKEN_LINE + 2 * len(repr(line)))

    def long(self, token, limit):
        return limit is not None and len(token) > limit

    def ranged(self, added):
        self.entries += added
        self.cost += MAP_RANGE_ENTRY_COST * added
        self.counted -= MAP_RANGE_ENTRY_COST * added

    def range_line(self, line):
        """parse_bfrange."""
        tokens = line.split()
        if self.multi is not None:
            first, last = self.multi
            added, closed, how = self.listed(tokens, first)
            self.ranged(added)
            if how:
                return self.fail(how, line)     # self.multi as it was
            self.multi = None if closed else (first + added, last)
            return None
        if not tokens:
            return self.fail(_LOGGED, line)
        if self.long(tokens[0], self.code):
            return self.fail(_RAISED)
        if len(tokens) < 2:
            return self.fail(_LOGGED, line)
        if self.long(tokens[1], self.code):
            return self.fail(_RAISED)
        try:
            first, last = int(tokens[0], 16), int(tokens[1], 16)
        except ValueError:
            return self.fail(_LOGGED, line)
        self.width = (max(len(tokens[0]), len(tokens[1])) + 1) // 2
        if len(tokens) < 3:
            return self.fail(_LOGGED, line)
        if tokens[2] == b"[":
            added, closed, how = self.listed(tokens[3:], first)
            self.ranged(added)
            if how:
                return self.fail(how, line)
            self.multi = None if closed else (first + added, last)
            return None
        if self.long(tokens[2], self.string):
            return self.fail(_RAISED)
        try:
            onto = int(tokens[2], 16)
        except ValueError:
            return self.fail(_LOGGED, line)
        if self.entries + max(0, last - first + 1) > self.limit:
            return self.fail(_RAISED)
        added, how = _ranged(first, last, onto, len(tokens[2]), self.width,
                             self.code)
        self.ranged(added)
        if how:
            return self.fail(how, line)
        self.multi = None
        return None

    def listed(self, tokens, first):
        """(entries added, whether the list closed, how it failed or None):
        the codes from ``first`` onto ``tokens``, a bfrange's list."""
        added = 0
        for token in tokens:
            if token == b"]":
                return added, True, None
            if self.long(token, self.string) \
                    or self.entries + added + 1 > self.limit:
                return added, False, _RAISED
            if not _unhexlifies(token) or (len(token) // 2) % 2:
                return added, False, _LOGGED    # UTF-16 of whole code units
            how = _key_error(first + added, self.width)
            if how:
                return added, False, how
            added += 1
        return added, False, None

    def char_line(self, line):
        """parse_bfchar: each pair an entry, and each copies the rest."""
        tokens = [token for token in line.split(b" ") if token]
        if not tokens:
            return self.record(len("Skipping broken line : Line is empty.")
                               + len(repr(line)))
        if len(tokens) % 2:
            self.record(_ODD_LINE + len(repr(line)))
        pairs, length = len(tokens) // 2, len(tokens)
        if self.entries + pairs > self.limit:
            return self.fail(_RAISED)
        self.width = len(tokens[0]) // 2
        added, how = 0, None
        for pair in range(pairs):
            source, target = tokens[2 * pair], tokens[2 * pair + 1]
            if self.long(source, self.code):
                how = _RAISED
                break
            if target != b".":
                if self.long(target, self.string):
                    how = _RAISED
                    break
                if not _unhexlifies(target):
                    self.record(_BAD_HEX + len(repr(target)))
                elif len(target) >= 4 and (len(target) // 2) % 2:
                    how = _LOGGED               # UTF-16 of an odd byte
                    break
            if not _unhexlifies(source) \
                    or (self.width != 1 and (len(source) // 2) % 2):
                how = _LOGGED
                break
            added += 1
            self.copies += length - 2 * (pair + 1)
        self.entries += added
        self.cost += MAP_CHAR_ENTRY_COST * added
        return self.fail(how, line) if how else None


def _ranged(first, last, onto, digits, width, limit):
    """(the entries pypdf builds of a bfrange line from code ``first`` to
    ``last`` onto ``onto``, written in ``digits`` digits, before it logs the
    line or gives up on it; which of the two, or None). pypdf writes each
    destination in at least max(4, digits) digits, as many more as it needs
    -- an odd number is no hex, and one of two bytes past whole UTF-16 code
    units does not decode -- and a code of ``width`` bytes."""
    count = last - first + 1
    if count <= 0:
        return 0, None
    if onto < 0:
        return 0, _LOGGED               # "-" is no hex digit
    least, done = max(4, digits), 0
    while done < count:
        written = max(least, len(b"%X" % (onto + done)))
        if written % 2:
            return done, _LOGGED
        if limit is not None and written // 2 > limit:
            return done, _RAISED
        if written % 4:
            return done, _LOGGED
        if done == 0:
            how = _key_error(first, width)
            if how:
                return 0, how
        done = 16 ** written - onto     # the first written with more digits
    return count, None


def _key_error(code, width):
    """How pypdf fails on code ``code`` of a map whose codes are ``width``
    bytes, or None: a code that does not fit, as an OverflowError it gives up
    on, or UTF-16 of an odd byte, logged."""
    if width is None or code < 0 or code >> (8 * width):
        return _RAISED
    return _LOGGED if width != 1 and width % 2 else None


def _unhexlifies(token):
    """Whether binascii.unhexlify reads ``token``."""
    return not len(token) % 2 and _HEX_DIGITS.fullmatch(token) is not None


def _logged_size(value, cap):
    """len("%s" % value) -- what a log record says of ``value`` -- counted no
    further than one past ``cap``, without formatting it: a dictionary or an
    array is walked as repr walks it, a reference in it counted as its repr,
    which resolves nothing; anything else is formatted, as small as it
    is in the file."""
    from pypdf.generic import IndirectObject  # type: ignore
    if not _walked(value):
        try:
            text = repr(value) if isinstance(value, IndirectObject) \
                else "%s" % (value,)
        except Exception:  # noqa: BLE001 - logging says the same and goes on
            return 0
        return min(len(text), cap + 1)
    size, todo, path = 0, [(value, False)], set()
    while todo and size <= cap:
        item, leaving = todo.pop()
        if leaving:
            path.discard(id(item))
        elif not _walked(item):
            size += len(repr(item))
        elif id(item) in path:
            size += 5                   # "{...}" or "[...]": repr's own guard
        else:
            path.add(id(item))
            todo.append((item, True))
            if isinstance(item, dict):
                size += 4 * len(item) if item else 2      # "{", ": ", ", ", "}"
                for key, entry in dict.items(item):
                    todo.append((key, False))
                    todo.append((entry, False))
            else:
                size += 2 * len(item) if item else 2      # "[", ", ", "]"
                todo.extend((entry, False) for entry in list.__iter__(item))
    return min(size, cap + 1)


def _walked(value):
    """Whether repr of ``value`` is a dictionary's or a list's own."""
    return (isinstance(value, dict) and type(value).__repr__ is dict.__repr__
            or isinstance(value, list)
            and type(value).__repr__ is list.__repr__)


def _encoding_logged(font):
    """The characters pypdf logs of ``font``'s /Encoding on every Font it
    builds, where it knows no such encoding: a name or a string, whole, or a
    dictionary whose /BaseEncoding it does not know (_cmap._parse_encoding in
    6.19.0). An unknown name of 4,000 characters, named on many pages, wrote
    6.8 GB from a 227 KB file; a string of 1,000,000 characters
    named 2,000 times 2,000 MB."""
    from pypdf.generic import DictionaryObject  # type: ignore
    try:
        from pypdf._cmap import _predefined_cmap  # type: ignore
        from pypdf._codecs import charset_encoding  # type: ignore
    except ImportError:
        _predefined_cmap, charset_encoding = {}, {}
    try:
        encoding = font["/Encoding"] if "/Encoding" in font else None
        if isinstance(encoding, str):
            if encoding in charset_encoding or encoding in _predefined_cmap \
                    or "-UCS2-" in encoding:
                return 0
            return _ADVANCED + _logged_size(encoding,
                                            MAX_FONT_LOG_PER_DOCUMENT)
        if isinstance(encoding, DictionaryObject) \
                and "/BaseEncoding" in encoding:
            try:
                if encoding["/BaseEncoding"] in charset_encoding:
                    return 0
            except Exception:  # noqa: BLE001 - unhashable: pypdf logs it
                pass
            return _ADVANCED + len("[]")    # pypdf logs its empty list
    except Exception:  # noqa: BLE001 - pypdf meets the same and decides
        return 0
    return 0


def _type1_program(font):
    """(the key, the font file) of the font file pypdf looks at for a Type1
    font's encoding where the font has no /ToUnicode entry -- the first of
    its descriptor's /FontFile and /FontFile3 that is a stream -- else None
    (_cmap._parse_to_unicode in 6.19.0). One walk of the descriptor for
    what pypdf reads of it (_font_file) and what it logs of it
    (_type1c_logged)."""
    from pypdf.generic import StreamObject  # type: ignore
    try:
        if "/ToUnicode" in font or font.get("/Subtype", "") != "/Type1":
            return None
        descriptor = font.get("/FontDescriptor")
        if not descriptor:
            return None
        descriptor = descriptor.get_object()
        for key in ("/FontFile", "/FontFile3"):
            program = descriptor[key] if key in descriptor else None
            if isinstance(program, StreamObject):
                return key, program
    except Exception:  # noqa: BLE001 - pypdf meets the same and decides
        return None
    return None


def _is_type1c(found):
    """Whether ``found`` (_type1_program) is a /FontFile3 of /Subtype
    /Type1C, which pypdf reads with fontTools, or logs that it cannot."""
    if found is None or found[0] != "/FontFile3":
        return False
    try:
        return found[1].get("/Subtype") == "/Type1C"
    except Exception:  # noqa: BLE001 - pypdf meets the same and decides
        return False


def _has_fonttools():
    try:
        from pypdf._font import HAS_FONTTOOLS  # type: ignore
    except ImportError:
        return False
    return HAS_FONTTOOLS


def _type1c_logged(font, found):
    """The characters pypdf logs of a Type1 font without a /ToUnicode entry
    whose font file is CFF -- /FontFile3 of /Subtype /Type1C, ``found`` by
    _type1_program -- where fontTools is not installed: the font's whole
    dictionary as pypdf is handed it, on every Font it builds
    (_cmap._parse_to_unicode in 6.19.0)."""
    if not _is_type1c(found) or _has_fonttools():
        return 0
    return _NO_FONTTOOLS + _logged_size(font, MAX_FONT_LOG_PER_DOCUMENT)


def _map_of(font):
    """A font's /ToUnicode map when it is a stream, else None -- a name such
    as /Identity-H costs pypdf nothing to read."""
    from pypdf.generic import StreamObject  # type: ignore
    try:
        cmap = font["/ToUnicode"] if "/ToUnicode" in font else None
    except Exception:  # noqa: BLE001
        return None
    return cmap if isinstance(cmap, StreamObject) else None


def _font_file(found):
    """The font file pypdf decodes and reads for a Type1 font's encoding,
    ``found`` by _type1_program, else None: /FontFile, or /FontFile3 of
    /Subtype /Type1C where fontTools is installed
    (pypdf._cmap._parse_to_unicode in 6.19.0)."""
    if found is None:
        return None
    if found[0] == "/FontFile":
        return found[1]
    return found[1] if _is_type1c(found) and _has_fonttools() else None


def _descendants(font):
    """The /DescendantFonts pypdf walks for ``font``: a composite font's,
    else none."""
    from pypdf.generic import ArrayObject  # type: ignore
    try:
        if font.get("/Subtype") in ("/Type1", "/MMType1", "/TrueType",
                                    "/Type3"):
            return ()
        descendants = font["/DescendantFonts"]
    except Exception:  # noqa: BLE001 - pypdf meets the same and decides
        return ()
    return descendants if isinstance(descendants, ArrayObject) else ()


class _WidthsWalk:
    """What pypdf does with one /W walking it for a composite font's widths,
    on every Font it builds (Font._collect_cid_character_widths in 6.19.0):
    the widths it builds and the elements it walks, the records it logs and
    their characters, and how often it resolves a reference to an object
    the file lacks, each a search of the whole file (_Bounds._absent)."""

    __slots__ = ("widths", "elements", "records", "logged", "missing")

    def __init__(self):
        self.widths = self.elements = self.records = 0
        self.logged = self.missing = 0


def _walk_widths(widths, resolve):
    """The _WidthsWalk of ``widths``, a /W, walked as pypdf walks it: every
    element that is no number pypdf logs, whatever it is -- /W elements
    naming one name of 4,000 characters, 3 units each, wrote 16.2 GB from
    606 KB -- and a dictionary it walks key by key, logging
    each. ``resolve`` is _Bounds._resolve. Nothing where pypdf gives up on
    the font."""
    from collections.abc import Mapping, Sequence
    walk = _WidthsWalk()

    def record(template, value):
        walk.records += 1
        walk.logged += template + _logged_size(value, MAX_FONT_LOG_PER_DOCUMENT)

    def look(at):
        value, absent = resolve(list.__getitem__(widths, at))
        walk.missing += absent
        if not absent and not hasattr(value, "get_object"):
            raise TypeError("no PDF object")     # pypdf's .get_object() fails
        return value

    def number(value):
        return isinstance(value, (int, float))
    try:
        if isinstance(widths, Mapping):
            # pypdf walks a dictionary key by key, logging each: a /W of
            # 100,000 keys named 2,000 times ran 134 s past a 60 s deadline
            #.
            for key in dict.keys(widths):
                record(_NOT_A_WIDTH, key)
            walk.elements = len(widths)
            return walk
        if not isinstance(widths, list):
            return _WidthsWalk()        # pypdf gives up on the font
        length, skip, count = len(widths), 0, 0
        walk.elements = length
        for at in range(length):
            if skip:
                skip -= 1
                continue
            first = look(at)
            if not number(first):
                record(_NOT_A_WIDTH, first)
                continue
            after = look(at + 1) if at + 1 < length else None
            if isinstance(after, Sequence):
                count, skip = count + len(after), 1
            elif number(after) and at + 2 < length and number(look(at + 2)):
                count, skip = count + max(int(after) - int(first) + 1, 0), 2
            else:
                record(_NOT_A_DEFINITION, first)
        walk.widths = count
        return walk
    except deadline.DeadlineExceeded:
        raise
    except Exception:  # noqa: BLE001 - pypdf meets the same and decides
        return _WidthsWalk()


def _differences(font):
    """(the length of the /Differences array pypdf walks for ``font``, the
    characters pypdf logs of a /Differences that is no array: all of it, on
    every Font it builds (_cmap._parse_encoding in 6.19.0) -- 3,000 names of
    a font whose /Differences was a dictionary ran 79 s past a 20 s deadline
    and wrote 4.17 GB of log)."""
    from pypdf.generic import ArrayObject, DictionaryObject  # type: ignore
    try:
        encoding = font["/Encoding"] if "/Encoding" in font else None
        if not isinstance(encoding, DictionaryObject) \
                or "/Differences" not in encoding:
            return 0, 0
        differences = encoding["/Differences"]
    except Exception:  # noqa: BLE001 - none, or pypdf decides
        return 0, 0
    if isinstance(differences, ArrayObject):
        return len(differences), 0
    if differences is None:
        return 0, 0     # pypdf gives up on the font: .get_object() of None
    return 0, _NOT_AN_ARRAY + _logged_size(differences,
                                           MAX_FONT_LOG_PER_DOCUMENT)


def _glyph_procedures(font):
    """The /CharProcs pypdf walks for ``font``: a Type3 font's without a
    /ToUnicode entry, for whether each is an Adobe glyph name -- an array as
    well as a dictionary: 2,000 names of a font with an array of 100,000
    names took 6.9 s. A string's first character is none (every
    name begins with "/", and "/" is none), and pypdf stops there."""
    from pypdf.generic import ArrayObject, DictionaryObject  # type: ignore
    try:
        if font.get("/Subtype") != "/Type3" or "/ToUnicode" in font:
            return 0
        procedures = font["/CharProcs"] if "/CharProcs" in font else None
    except Exception:  # noqa: BLE001
        return 0
    return len(procedures) \
        if isinstance(procedures, (ArrayObject, DictionaryObject)) else 0


def _with_map(font, data):
    """A copy of the font dictionary ``font`` whose /ToUnicode map is
    ``data``."""
    from pypdf.generic import DecodedStreamObject  # type: ignore
    from pypdf.generic import DictionaryObject, NameObject  # type: ignore
    cmap = DecodedStreamObject()
    cmap.set_data(data)
    copy = DictionaryObject(font)
    copy[NameObject("/ToUnicode")] = cmap
    return copy


def _without(owner, *keys):
    """A copy of the dictionary ``owner`` without ``keys``."""
    from pypdf.generic import DictionaryObject  # type: ignore
    copy = DictionaryObject(owner)
    for key in keys:
        copy.pop(key, None)
    return copy


def _without_long_boxes(font):
    """``font`` -- or a copy of it -- without the /FontBBox entries of other
    than four elements that pypdf would read and drop: the font's own (a
    Type3 font's), its descriptor's and each descendant's descriptor's.
    pypdf floats every element of a box once per name before it keeps only
    one of four (Font._parse_bbox): 2,000 names of a font with a box of
    1,000,000 elements ran 77 s past a deadline of 60 before the first
    operator (review of 0.1.48), and a string of 1,000,000 digits, floated
    character by character, 110 s. Nothing pypdf keeps is left
    out: a string of four digits it reads as a box."""
    from collections.abc import Sized
    from pypdf.generic import ArrayObject, DictionaryObject  # type: ignore
    from pypdf.generic import NameObject  # type: ignore

    def long_box(owner):
        try:
            box = owner["/FontBBox"] if "/FontBBox" in owner else None
        except Exception:  # noqa: BLE001 - pypdf meets the same and decides
            return False
        return isinstance(box, Sized) and len(box) != 4

    def with_descriptor(owner):
        try:
            descriptor = (owner["/FontDescriptor"]
                          if "/FontDescriptor" in owner else None)
        except Exception:  # noqa: BLE001 - none, or pypdf decides
            return owner
        if not isinstance(descriptor, DictionaryObject) \
                or not long_box(descriptor):
            return owner
        copy = DictionaryObject(owner)
        copy[NameObject("/FontDescriptor")] = _without(descriptor, "/FontBBox")
        return copy

    copy = _without(font, "/FontBBox") if long_box(font) else font
    copy = with_descriptor(copy)
    descendants = _descendants(copy)
    if descendants:
        handed = [with_descriptor(d.get_object())
                  if isinstance(d.get_object(), DictionaryObject) else d
                  for d in descendants]
        if any(h is not d.get_object() for h, d in zip(handed, descendants)):
            copy = DictionaryObject(copy) if copy is font else copy
            copy[NameObject("/DescendantFonts")] = ArrayObject(handed)
    return copy


def _stripped(font):
    """A copy of ``font`` without what pypdf reads of it beyond the
    dictionary, and logs of it: its /ToUnicode map, its descendant's /W and
    every descendant past the first, its /Differences, an array or not, its
    /CharProcs and its font files."""
    from pypdf.generic import ArrayObject, DictionaryObject  # type: ignore
    from pypdf.generic import NameObject  # type: ignore
    copy = _without(font, "/ToUnicode") if _map_of(font) is not None \
        else DictionaryObject(font)
    descendants = _descendants(font)
    if len(descendants):
        first = descendants[0].get_object()
        if isinstance(first, DictionaryObject):
            copy[NameObject("/DescendantFonts")] = ArrayObject(
                [_without(first, "/W")])
    if any(_differences(font)):
        copy[NameObject("/Encoding")] = _without(font["/Encoding"],
                                                 "/Differences")
    if _glyph_procedures(copy):
        copy[NameObject("/CharProcs")] = DictionaryObject()
    try:
        descriptor = font["/FontDescriptor"]
    except Exception:  # noqa: BLE001 - none, or pypdf decides
        descriptor = None
    if isinstance(descriptor, DictionaryObject):
        copy[NameObject("/FontDescriptor")] = _without(
            descriptor, "/FontFile", "/FontFile3")
    return copy


#: The filters sift measures a stream's decoded length through itself.
_FLATE = ("/FlateDecode", "/Fl")
_HEX = ("/ASCIIHexDecode", "/AHx")
_A85 = ("/ASCII85Decode", "/A85")
#: White space as pypdf's ASCII85 decoder skips it; bytes.split() for hex.
_PDF_WHITE = b"\x00\t\n\f\r "
_HEX_WHITE = (b" ", b"\t", b"\n", b"\r", b"\x0b", b"\x0c")


def _decoded_size(stream, cap):
    """How many bytes ``stream`` decodes to -- counted no further than one
    past ``cap`` where sift can count it itself -- or None where pypdf
    cannot decode it.

    Without /DecodeParms, a stream whose filters are all /FlateDecode is
    inflated here stage by stage, to one byte past ``cap`` at most, never
    whole; one whose one filter is /ASCIIHexDecode or /ASCII85Decode is
    measured from its encoded bytes. Any other is decoded by pypdf, and
    pypdf's decoded copy is dropped again when it is past ``cap``, so that
    the reader does not keep what pypdf will not read.
    """
    from pypdf.generic import EncodedStreamObject  # type: ignore
    if (isinstance(stream, EncodedStreamObject)
            and stream.decoded_self is None
            and "/DecodeParms" not in stream):
        filters = stream.get("/Filter")
        filters = filters.get_object() if filters is not None else None
        filters = [f.get_object() for f in filters] \
            if isinstance(filters, list) else [filters]
        # The encoded bytes, where pypdf's own decoder reads them
        # (pypdf.filters.decode_stream_data in 6.8.0, 6.14.2, 6.16.2 and
        # 6.19.0).
        data = stream._data
        try:
            if all(f in _FLATE for f in filters):
                return _inflated_size(data, len(filters), cap)
        except zlib.error:
            pass                # pypdf recovers some damaged streams
        if len(filters) == 1 and filters[0] in _HEX:
            return _hex_size(data)
        if len(filters) == 1 and filters[0] in _A85:
            return _a85_size(data)
    try:
        size = len(stream.get_data())
    except Exception:  # noqa: BLE001 - pypdf raises varied errors
        return None
    if size > cap:
        _forget_decoded(stream)
    return size


def _inflated_size(data, stages, cap):
    """The length of ``data`` inflated ``stages`` times over, counted no
    further than one past ``cap``: 2,000 maps of 70 MB inflated twice took
    117 s decoded whole by pypdf."""
    chunks = (data,)
    for _ in range(stages):
        chunks = _inflating(chunks)
    size = 0
    for chunk in chunks:
        size += len(chunk)
        if size > cap:
            return cap + 1
    return size


def _inflating(chunks, step=1 << 16):
    """The chunks of ``chunks`` inflated, ``step`` bytes at most at a time."""
    inflate = zlib.decompressobj()
    for chunk in chunks:
        while chunk and not inflate.eof:
            out = inflate.decompress(chunk, step)
            chunk = inflate.unconsumed_tail
            if out:
                yield out
        if inflate.eof:
            return
    out = inflate.flush()
    if out:
        yield out


def _hex_size(data):
    """What pypdf's ASCIIHexDecode makes of ``data``: its digits up to the
    first ">", white space left out, two to a byte."""
    end = data.find(b">")
    end = len(data) if end < 0 else end
    digits = end - sum(data.count(white, 0, end) for white in _HEX_WHITE)
    return (digits + 1) // 2


def _a85_size(data):
    """What pypdf's ASCII85Decode makes of ``data``: four bytes a "z", four
    a group of five characters, one fewer than its characters a last group
    of two to four, white space and the "<~" and "~>" around left out."""
    data = data.translate(None, _PDF_WHITE)
    if data.endswith(b"~>"):
        data = data[2:-2] if data.startswith(b"<~") else data[:-2]
    zeros = data.count(b"z")
    rest = len(data) - zeros
    return 4 * zeros + rest // 5 * 4 + max(rest % 5 - 1, 0)


def _bound_form(xform):
    """(the form itself when its content fits MAX_INFLATED, else a copy of it
    whose content is cut there; (bytes kept, bytes held) when cut, else
    None). Raises what pypdf raises decoding it."""
    data = xform.get_data()
    if len(data) <= MAX_INFLATED:
        return xform, None
    from pypdf.generic import DecodedStreamObject, NameObject  # type: ignore
    kept = DecodedStreamObject()
    for key, value in xform.items():
        if key not in ("/Filter", "/DecodeParms", "/Length"):
            kept[NameObject(key)] = value
    kept.set_data(data[:_content_cut(data, MAX_INFLATED)])
    _forget_decoded(xform)
    return kept, (len(kept.get_data()), len(data))


def _cut_note(cut, forms):
    """What the notes say of the pages' own content cut at MAX_INFLATED, or
    where ``forms``, of the forms' content: ``cut`` holds (page, bytes kept,
    bytes held) of each."""
    if not cut:
        return None
    first = cut[0][0]
    if len(cut) == 1:
        number, kept, total = cut[0]
        of = (f"a form XObject page {number} draws" if forms
              else f"page {number}'s content")
        return (f"read only the first {kept} of {total} bytes of {of}; its "
                f"text after that is not in the body")
    of = (f"{len(cut)} form XObjects, the first of them drawn on page {first}"
          if forms else f"{len(cut)} pages, the first of them page {first}")
    return (f"read at most the first {MAX_INFLATED} bytes of the content of "
            f"{of}; their text after that is not in the body")


#: One token of a content stream outside a literal string: white space, a
#: comment, a dictionary or array bracket, a hex string, a name, or a word --
#: a number or an operator. A literal string nests, so it is walked apart.
_TOKEN = re.compile(rb"[\x00\t\n\f\r ]+|%[^\r\n]*|<<|>>|<[^<>]*>|[\[\]{}]"
                    rb"|/[^\x00\t\n\f\r ()<>\[\]{}/%]*"
                    rb"|[^\x00\t\n\f\r ()<>\[\]{}/%]+")
_PARENTHESIS = re.compile(rb"[()\\]")
_WHITE = b"\x00\t\n\f\r "
#: The end of an inline image's data as pypdf looks for it: "EI" between
#: white space.
_INLINE_IMAGE_END = re.compile(rb"[\x00\t\n\f\r ]EI(?=[\x00\t\n\f\r ]|$)")


def _content_cut(data: bytes, limit: int) -> int:
    """How much of a content stream to keep: all of it within ``limit``, else
    up to the last whole token before ``limit`` outside any string, array,
    dictionary or inline image.

    pypdf gives up on a content stream that ends inside one of those -- and a
    page it gives up on sends the whole document to the fallback. It reads
    one that ends between two tokens, with operands left over or not. A token
    counts as whole once white space follows it within the limit.
    """
    if len(data) <= limit:
        return len(data)
    at, depth, cut = 0, 0, 0
    while at < limit:
        if data[at] == 0x28:        # "(": parentheses nest, "\" escapes a byte
            nesting, at = 1, at + 1
            while nesting:
                hit = _PARENTHESIS.search(data, at, limit)
                if hit is None:
                    return cut
                at = hit.end() + (hit.group() == b"\\")
                nesting += {b"(": 1, b")": -1}.get(hit.group(), 0)
            continue
        token = _TOKEN.match(data, at, limit)
        if token is None:           # an unterminated hex string, a stray ")"
            return cut
        word, at = token.group(), token.end()
        if word in (b"[", b"<<"):
            depth += 1
        elif word in (b"]", b">>"):
            depth = max(depth - 1, 0)
        elif word == b"BI":         # an inline image: binary data up to "EI"
            end = _INLINE_IMAGE_END.search(data, at, limit)
            if end is None:
                return cut
            at = end.end()
        elif word[0] in _WHITE and depth == 0:
            cut = token.start()
    return cut


#: What the cross-reference streams and object streams of one document may
#: decode to in all, in bytes a byte of the file (_structure_budget), and at
#: least MAX_INFLATED: 12 times the most any of 111 real documents decoded of
#: them against its size, 0.66 (5,982 bytes of 9,081, testPopupAnnotation.pdf;
#: a cross-reference stream 0.018 at most, 335,790 bytes of soa.pdf's 18 MB,
#: an object stream 0.32), none of whose 2,635 object streams and 150
#: cross-reference streams comes near it. pypdf decoded each to its own limit
#: of 75,000,000 bytes and kept every one.
STRUCTURE_BYTES_A_FILE_BYTE = 8

#: How many reads parsing one object stream may take (_ObjectStreamBytes):
#: OBJECT_STREAM_READS_A_BYTE a byte of the stream and OBJECT_STREAM_READS
#: besides. pypdf reads each byte of an object stream once or twice as it
#: parses: the 231 object streams sift's reading of 111 real documents parses
#: took 1.92 reads a byte at most (587 bytes of soa.pdf), and
#: OBJECT_STREAM_READS_A_BYTE is twice that, rounded up (a test holds it).
#: Objects sharing bytes are each parsed from where they begin, though: 20
#: objects at one offset, followed by 73 MB of zeros, read the zeros 20 times
#: over.
OBJECT_STREAM_READS_A_BYTE = 4
OBJECT_STREAM_READS = 65_536

#: The memory a host application's workers read a document in, 1 GiB (a host application
#: deploy/sdctl.py APP_MEMORY), the largest memory sift is run in.
WORKER_MEMORY = 1 << 30
#: What pypdf's reader may hold of one document in all, in bytes, as sift
#: charges it (_Structure.hold): half of WORKER_MEMORY. The other half is the
#: process's own, the file and pypdf's copies of it, what one page's reading
#: builds -- its content cut to MAX_INFLATED, its fonts, the forms it draws
#: --, and a stream as it decodes. 0.1.59 bounded the entries alone, and
#: pypdf parsed and kept every object of each object stream the page tree
#: touched: 402,003 of them, 1,015 MB, for t2000p.pdf, and t2100p.pdf, 21 MB,
#: was OOM-killed at --memory=1g. Charged, each as it is made,
#: what it holds -- sys.getsizeof, and what each of pypdf's classes holds past
#: it (_HELD_PAST_SIZEOF): each entry of its tables (_Entries), each object
#: it parses (_held_by) and keeps (_cached_held), each copy of a page it
#: walks (_page_held), the index of "N G obj" (_index_held), each object
#: stream kept decoded (_kept_stream), its trailer; the parse under way at
#: the worst case (READ_HELD) until it is walked (_in_flight_held). A test
#: holds each within 15 % over what tracemalloc traces, never under (pypdf
#: 6.19.0, Python 3.14): the first cut charged every read at the worst case
#: and each object at the most measured, 1.4 to 1.9 times what the reader
#: held, and pages100k.pdf, 100,000 pages 0.1.59 read in 440 MB, fell back
#:. Past it, opening the document the reader is not made,
#: walking its page tree pypdf is not read with, reading a page the document
#: ends there; the notes say so.
MAX_DOCUMENT_HELD = WORKER_MEMORY // 2
#: What sift lets the entries pypdf's reader makes of one document's objects
#: in use hold, in bytes: half of MAX_DOCUMENT_HELD, a quarter of
#: WORKER_MEMORY.
MAX_XREF_HELD = MAX_DOCUMENT_HELD // 2
#: What pypdf's reader holds of an entry of an object in use, in bytes
#: resident, the most measured (pypdf 6.19.0, Python 3.14, 200,000 and
#: 1,000,000 entries): a classic table's line 189 to 244, a cross-reference
#: stream's entry of type 1 140 to 162, of type 2 201 to 208; an object of a
#: hybrid file, a free line of the table and an entry of its /XRefStm, 211 to
#: 242. A free line alone holds 74 to 89 and is not counted (_Entries).
XREF_ENTRY_HELD = 256

#: How many entries of objects in use the cross-reference sections of one
#: document may make pypdf hold (_entry_budget): one a
#: XREF_FILE_BYTES_AN_ENTRY bytes of the file, at most MAX_XREF_ENTRIES --
#: MAX_XREF_HELD of XREF_ENTRY_HELD each (a test holds it). Counted as pypdf
#: makes them (_Entries), each object once: a line a classic table reads
#: twice (21 bytes, ending "\r\n"), an object a hybrid file lists in its table
#: and its /XRefStm, and a free entry no longer count. 0.1.58 counted reads
#: and 12 times what 111 real documents list, and cut legitimate files: a
#: classic table of 341,000 objects in 22 MB lost 34 % of its text, a tagged
#: report of 890,000 objects in 18 MB went from 2,000 pages to the fallback
#:. The fewest bytes a file spent on an object in use were
#: 19.7 of the reviewer's generated files (t2000_a.pdf, 890,426 objects in
#: 17.6 MB, the most of any), 123 of the 111 real ones (9 in 1,111 bytes;
#: soa.pdf lists the most, 56,050): XREF_FILE_BYTES_AN_ENTRY is less than a
#: third of it, MAX_XREF_ENTRIES more than all of them (a test holds both).
#: pypdf made an entry of about 170 bytes of every two bytes a
#: cross-reference stream decoded to: 12,000,000 of them, from 2.9 MB, took
#: 2,090 MB and 13.3 s.
XREF_FILE_BYTES_AN_ENTRY = 6
MAX_XREF_ENTRIES = MAX_XREF_HELD // XREF_ENTRY_HELD

#: How many entries of a cross-reference stream pypdf lists between two looks
#: at the time budget and the log bound (_Structure.watch): it reads them
#: from the stream's decoded bytes, which no _Alarm trips.
XREF_ENTRIES_A_CHECK = 4096
#: How many entries of one table pypdf makes, while the reader is made,
#: between two charges of what they hold (_Entries.settle): at about 150 to
#: 250 bytes an entry, the reader passes the bound by 1 MB at most before it
#: is charged.
ENTRIES_A_CHARGE = 4096

#: What what pypdf parses of one read holds at most, in bytes (_CountedFile,
#: _ObjectStreamBytes): the costliest shapes traced, each an array of
#: 50,000, took 28.4 bytes a read (hex strings "<a>"), 27.2 (strings), 27.1
#: (one-key dictionaries), a page's dictionary 20.9, numbers 12.1 (pypdf
#: 6.19.0, Python 3.14, a review). A read of more bytes than this counts
#: its bytes: a stream's data, read whole. The worst case of the parse under
#: way only, until it is walked (_Structure.in_flight): a parsed object is
#: charged what it holds (_held_by).
READ_HELD = 32
#: How many reads the parse under way reads at least between two walks of it
#: (_Structure.in_flight): each walk costs what the parse holds, and near the
#: bound it passes the bound by 32 KiB at most, READ_HELD a read, before it
#: is walked and ends.
IN_FLIGHT_READS = 1024
#: The most the index of "N G obj" (_Found) holds a key at its peak, as it
#: is made, in bytes: 141.6 traced (1,000,000 keys; 124.4 once made, a review; a test holds it). It is made of at most what is left of the
#: bound at this a key, and charged what it holds (_index_held).
INDEX_KEY_PEAK = 160
#: What the object streams sift keeps decoded may be charged in all
#: (_Structure.keep), a sixteenth of MAX_DOCUMENT_HELD: past it the stream
#: used longest ago is let go of, and decoded again -- charged again to the
#: structure budget -- when an object of it is asked for next. pypdf parsed
#: an object stream once and kept every object of it; sift parses the
#: objects asked for, from the stream kept.
MAX_OBJECT_STREAMS_HELD = MAX_DOCUMENT_HELD // 16

#: How many bytes pypdf may copy and search in all, the whole file each
#: time, to recover a line of a table it cannot read, or for an object the
#: document's index does not answer (_Structure.find: one past the index's
#: MAX_XREF_ENTRIES objects, one written with more digits than a key has)
#: (_FileBytes): about 1.6 s, a search taking 6.1 to 6.3 ms a MB measured
#: (5 MB and 60 MB of random bytes, Python 3.14) -- 51 searches of a 5 MB
#: file, 4 of one of 60 MB, a host application's largest upload. None of 111 real
#: documents made pypdf search at all. Past it, what pypdf would search for
#: is taken as absent, and each search looks at the time budget first. The
#: searches for objects the tables lack or misplace are answered from the
#: index, made in one pass: 0.1.58 let them copy and search the file until
#: the bound, and read 407 of the 2,000 pages of a 658 KB file whose
#: objects all sat past where its table said (insfix7.pdf, a review).
MAX_SEARCHED = 256 << 20

#: The oldest pypdf sift reads with and is tested on. 6.18.0 brought the
#: configuration API sift bounds what pypdf decodes with (pypdf's CHANGELOG:
#: "Rework configuration value handling (#4044)"; apply_configuration is not
#: in 6.8.0, 6.14.2, 6.16.2 or 6.17.0). sift's tests fail on 6.18.0 -- 2
#: failures and 1 error, all in what sift counts of pypdf's reading of fonts
#: and maps, whose _cmap.py, _font.py and _page.py 6.18.1 changed -- and
#: pass on 6.18.1 and 6.19.0, whose _reader.py, _cmap.py, _font.py, _page.py
#: and filters.py are the same. pyproject.toml, requirements-optional.txt
#: and the READMEs say the same (a test holds it). A pypdf whose version is
#: older is not read with (_older_than_the_floor): 6.18.0 has all sift
#: bounds pypdf with, and was read with, silently.
PYPDF_FLOOR = "6.18.1"

#: What one stream may inflate to. A 189 KB file whose single stream held
#: 200 MB of zeros allocated 381 MB, and it scales linearly — `sift serve`
#: accepts 60 MB uploads. The text extractor only ever keeps MAX_CHARS, so
#: there is no reason to materialize more than a small multiple of it.
#: It is also as much of one page's content as pypdf is given to parse
#: (_bound_content). There, 770 KB of plain text took 1.0 s and 5 MB, and
#: the largest page in four real documents, 449 pages, was 281 KB. A page
#: cut there still yields MAX_CHARS when its content spends under four bytes
#: a character: one operator per line of text spends about one; those four
#: documents spent 1.8 to 27 (the median of each).
MAX_INFLATED = 4 * MAX_CHARS

#: How much form content pypdf is handed for one page, counted on every
#: draw. The bound on a page's own content, for the same reason: pypdf parses
#: what it is handed and runs every operator in it, whether the page holds it
#: or a form the page draws. Without it a form within MAX_INFLATED was parsed
#: and run once per draw, up to pypdf's own 5,000 draws a page, bounded only
#: by the clock where a caller set one. Text drawn through forms still yields
#: MAX_CHARS within it, as a page's own does (see MAX_INFLATED).
MAX_FORM_CONTENT = MAX_INFLATED

#: What reading one font's /ToUnicode map may cost pypdf, in map bytes'
#: worth (_map_reading), before it is spared the map. pypdf reads the map of
#: every font the resources name, at the start of the page and of every form
#: draw: a 30 KB file whose map inflated to 30 MB took 2.99 s and 1400 MB
#: drawn once. Counted in entries as well as bytes: pypdf builds up to
#: 100,000 of them, and two bfrange lines of 191 bytes built 98,304. At the bound, pypdf spends about 0.29 s on a map (91 ns a unit); its
#: largest, 100,000 entries of one bfrange line, costs 2.1 million, and a
#: full map of two-byte codes, 65,536 bfchar lines, 2.9 million. The costliest
#: of the 853 maps of 111 real documents cost 11,967 (3,794 bytes, 256
#: entries).
MAX_TO_UNICODE = 4 * MAX_INFLATED

#: What the fonts pypdf loads for one page may cost, counted on the page and
#: on every form draw: pypdf builds a Font for every name in /Font each time,
#: before the first operator, reading what the font holds -- its /ToUnicode
#: map, a composite font's /W widths (up to 100,000), a /Differences of any
#: length, a Type3 font's /CharProcs, a Type1 font's font file where it has
#: no map -- and logging what it cannot read. One /W font named 170 times, a
#: 4,807-byte file, took 3.3 s and 1,034 MB; a 3 MB map in a form a 4 KB file
#: draws 50 times 13.6 s, now 0.6 s. A name that does not fit is handed its
#: font without those parts, and one that does not fit even so is left out
#: (_Bounds._hand).
#:
#: Counted in bytes of map, every other part at what it costs pypdf 6.19.0
#: against the costliest map byte measured -- "<a>" repeated, 91 ns and a
#: 42-byte peak a byte (Python 3.14) -- in time or peak memory, whichever is
#: more: a byte of a font file (54 ns, 5.5 bytes), an element of
#: /Differences (63 ns) and an entry of /CharProcs (81 ns) count one, a width
#: WIDTH_COST, every name FONT_COST, a map as _map_reading counts it, a log
#: record LOG_RECORD_COST and a character of it one, and what counting a
#: font's map costs sift itself (_MapReading.read) on the page that first
#: names the font. The budget is one map at MAX_TO_UNICODE and a fourth of
#: it, what counting a well-formed map costs sift at most: 23 times the most
#: any page of 111 real documents loaded, 170,212 (soa.pdf's first page,
#: 161,273 without sift's counting); one read a page of pypdf's largest map
#: (100,000 entries, 2.1 million) or of a full map of two-byte codes (2.9
#: million, and 0.46 million counting it), where 0.1.51 let three of the
#: latter through. Without sift's counting charged, a page of fresh fonts
#: took 1.85 times the budget of 0.1.52, one map at MAX_TO_UNICODE.
MAX_FONT_LOAD_PER_PAGE = MAX_TO_UNICODE + MAX_TO_UNICODE // 4

#: What every name pypdf builds a Font for costs, however little its font
#: holds: the costliest bare font measured, TrueType with 256 /Widths, took
#: 34 us and a 24,945-byte peak -- 592 map bytes' worth. It bounds the names one
#: page loads at MAX_FONT_LOAD_PER_PAGE // FONT_COST, 6,250 (a test holds the
#: number); 111 real documents loaded 24 at most, 24 of them from one /Font.
FONT_COST = 640

#: What one /W width costs: 126 ns and a 118.5-byte peak, 2.8 map bytes'
#: worth. pypdf walks every element of /W for them, and where the elements
#: are more, each is charged as a width (_Bounds._measure).
WIDTH_COST = 3

#: What pypdf's reading of a map costs past its bytes (_map_reading), each
#: measured on pypdf 6.19.0 against the map byte (91 ns): a line it reads
#: past its first check, 1.5 us over the bytes of "0 0 0" lines; an entry of
#: a bfrange, 1.9 us at most (codes of eight digits onto 32); an entry of a
#: bfchar, one pair a line, 1.0 us past its line and bytes; and the list
#: elements a bfchar line copies to drop each pair, 3 ns each -- a line of
#: 32,000 pairs, 192 KB, took 3.1 s a build (found fixing a review).
MAP_LINE_COST = 17
MAP_RANGE_ENTRY_COST = 21
MAP_CHAR_ENTRY_COST = 13
MAP_COPIES_A_UNIT = 30

#: What counting a map costs sift itself (_map_reading's ``read``), in the
#: same units: preparing it, a unit per MAP_PREPARE_BYTES_A_UNIT of its
#: bytes, a bfchar line of a run MAP_RUN_CHAR_COST and a bfrange line of a
#: run MAP_RUN_RANGE_COST; each line walked one by one what it costs pypdf,
#: but a bfrange's entries. Measured on pypdf 6.19.0 and Python 3.14, 80 to
#: 86 ns a unit of pypdf's own reading: preparing 11.7 ns a byte (a map of
#: 298,386 bytes, 3.5 ms); a bfchar line of a run 0.17 us of four digits
#: each and 0.38 us onto eight; a bfrange line of a run 0.9 us. 20,902
#: bfchar lines cost sift 7.1 ms against pypdf's 77.0, charged 0.21 of what
#: pypdf's reading is; onto eight digits 12.1 against 80.4, charged 0.22;
#: 65,536 bfrange lines of one code 76 against 323, charged 0.38.
MAP_PREPARE_BYTES_A_UNIT = 4
MAP_RUN_CHAR_COST = 6
MAP_RUN_RANGE_COST = MAP_LINE_COST

#: What knowing a map's bytes again costs sift (_Bounds._reading): its
#: BLAKE2b digest, a unit per MAP_HASH_BYTES_A_UNIT bytes -- 1.1 ns a byte
#: measured, 0.013 units.
MAP_HASH_BYTES_A_UNIT = 16

#: A map pypdf's reading of which costs MAP_RANGES_FROM units or more is
#: handed to pypdf with its runs of consecutive bfchar lines written as
#: bfrange lines (_as_ranges) where that costs a fourth less: 16 times the
#: costliest map of 111 real documents, 11,967 units, none of which is
#: rewritten. MAP_RANGE_FROM lines at least make a bfrange line: its own
#: line and those that open and close its section cost what four bfchar
#: lines do. Rewriting costs sift MAP_RANGES_LINE_COST a line of its runs,
#: and counting the rewritten map.
MAP_RANGES_FROM = MAX_TO_UNICODE // 16
MAP_RANGE_FROM = 8
MAP_RANGES_LINE_COST = 4

#: The bytes of maps written as ranges (_as_ranges) sift keeps for the
#: reader's life (_Bounds._reading); one past them is kept only while the
#: page that hands it is read. Derived from MAX_TO_UNICODE, and equal to it:
#: a map written so costs pypdf a fourth less than one of MAX_TO_UNICODE
#: units, and its bytes no more than its units, so one always fits (a test
#: holds it).
MAX_MAPS_REWRITTEN = MAX_TO_UNICODE

#: What a record pypdf logs costs past the characters it says, with
#: logging on as a host application runs it: 6.6 to 8.2 us over the same broken map
#: line unlogged, 73 to 90 map bytes' worth -- charged with room to spare.
#: A character of what it says costs 13 to 20 ns and is charged one unit:
#: what pypdf logs is output, whose volume is its own cost (see
#: MAX_FONT_LOG_PER_DOCUMENT).
LOG_RECORD_COST = 128

#: A reference to an object the file lacks, each time pypdf resolves it:
#: PdfReader.get_object in 6.19.0 copies the whole file and searches it
#: (about 15 ns a byte, charged at one unit per MISSING_REF_BYTES_A_UNIT
#: bytes of the file), and logs it twice, some 70 characters. sift's reader
#: searches once for each such object (_reader_class), within MAX_SEARCHED;
#: each resolve is charged as a search even so, the most it may cost.
MISSING_REF_BYTES_A_UNIT = 4
MISSING_REF_LOGGED = 80

#: What pypdf may log building the fonts of one document, in characters,
#: counted as each font is handed (_Bounds._hand): past it a font is handed
#: without the parts that log, or where it logs even so, left out. What it
#: logs is bounded by nothing else but the deadline: /W elements naming one
#: name of 4,000 characters wrote 16.2 GB to stderr from 606 KB, an unknown
#: /Encoding name 6.8 GB from 227 KB, and a host application's PDF codec
#: gate keeps every record. Derived from MAX_INFLATED, and equal to it --
#: as many characters as one page's content may hold bytes --, which is 126
#: times the most any of 111 real documents logged building fonts, 6,341
#: characters (speexdsp-manual.pdf); 8 of them logged any. A test holds the
#: ratio: should MAX_INFLATED move, it says whether this still holds.
MAX_FONT_LOG_PER_DOCUMENT = MAX_INFLATED

#: What pypdf may log reading one document, in characters: each record's
#: message and LOG_RECORD_PREFIX, what a log line says before it --
#: "2026-09-28 01:35:48,932 WARNING pypdf.generic._data_structures " as
#: A host application formats it --, counted as pypdf logs them (_LogWatch). Past it
#: the page being read ends, mid-parse or at its next operator, and the
#: document with it: 12 pages sharing one content stream of 800,000 bytes of
#: numbers pypdf cannot read, a 2,772-byte file, wrote 230 MB in 2.4 million
#: records, bounded by the deadline alone. Derived from
#: MAX_INFLATED, four times it -- what one font's map may cost pypdf,
#: MAX_TO_UNICODE --, which is 446 times the most any of 111 real documents
#: logged, 7,173 characters (speexdsp-manual.pdf); 9 of them logged any. A
#: test holds the ratio, as for MAX_FONT_LOG_PER_DOCUMENT. It holds what pypdf may log of a document's fonts within
#: MAX_FONT_LOG_PER_DOCUMENT, whose records say 40 characters at least:
#: 20,000 records, 2,080,000 characters with their prefixes.
MAX_OP_LOG_PER_DOCUMENT = 4 * MAX_INFLATED
LOG_RECORD_PREFIX = 64

#: What a Font pypdf built holds, in bytes, for the collector (_collect):
#: bare, 24,569 bytes at most measured (TrueType with 256 /Widths, a core
#: font's metrics 17,224); an entry of a map, 106 to 188 bytes; a width, 89;
#: a byte of a Type1 font file, 5.8 at most (its "dup" lines).
FONT_HELD = 25_600
MAP_ENTRY_HELD = 192
WIDTH_HELD = 96
FONT_FILE_HELD = 8

#: What the Fonts pypdf built may hold before sift collects them, in bytes --
#: across pages and documents, since the process holds them (_collect): 32
#: MiB, a little more than the 26 MB of widths 0.1.51's bound came to, now
#: counted in what the Fonts hold rather than in what they cost to build: a
#: page of the reviewer's map font held 10.5 MB, counted 1,164 units. Seven times the most any page of 111 real documents held, 4,750,240
#: bytes (soa.pdf); their 1,620 pages held 340 MB in all and collected 11
#: times, 9 of them the young generations only, in 0.03 s (0.1.51: 16 full
#: collections, 0.19 s).
MAX_FONT_HELD_UNCOLLECTED = 32 << 20

#: How many /Parent entries a form XObject's /Resources are looked for
#: through where it has none of its own (_resources_of): eight times the
#: deepest page tree of 111 real documents (4). None of their 116 forms has
#: a /Parent; 94 have no /Resources, and pypdf reads no text of them.
MAX_INHERITED = 32

#: What pypdf holds of the fonts built since sift last collected, of those
#: that may have grown old since it last collected everything (_collect), and
#: their lock: a caller may read documents in several threads at once.
_young = 0
_old = 0
_uncollected_lock = threading.Lock()


def _older_collections():
    """How often Python's collector has collected each of its older
    generations so far, or None where it does not say (gc.get_stats)."""
    try:
        stats = gc.get_stats()
        return tuple(generation["collections"] for generation in stats[1:]) \
            if len(stats) >= 3 else None
    except Exception:  # noqa: BLE001 - another Python: collect everything
        return None


def _collect(held, promoted=True):
    """Count ``held``, what pypdf holds of the Fonts it built for a page, and
    collect once what was counted since the last collection reaches
    MAX_FONT_HELD_UNCOLLECTED: the young generations (gc.collect(1)), unless
    what may have been moved to the oldest -- ``promoted``, the pages read
    while Python's collector ran over the older generations -- reaches it,
    then everything.

    What pypdf builds to read a page, and each form it draws, is a reference
    cycle: its TextExtraction's operation_handlers hold methods bound to it
    (pypdf/_text_extraction/_text_extractor.py:97-111 in 6.19.0), and it
    holds the page's Fonts (95, 124), their widths and maps. Only the
    collector frees a cycle, and it runs as the objects it tracks are
    allocated; a dict of numbers and strings is not tracked, so what makes a
    Font large never counts. 50 pages of a 12.8 KB file, each within its
    font budget, held 3,171 MB, 141 MB collected after every page. A full collection costs what the process holds -- 240 to 310 ms with
    3,000,000 objects, a server's --: 100 pages each just past the bound took
    38.9 s, 9.5 s not collected; the young generations, which
    hold what a page built unless the collector moved it on while the page
    was read, took 11 ms.
    """
    global _young, _old
    with _uncollected_lock:
        _young += held
        _old += held if promoted else 0
        if _old >= MAX_FONT_HELD_UNCOLLECTED:
            generation, _young, _old = 2, 0, 0
        elif _young >= MAX_FONT_HELD_UNCOLLECTED:
            generation, _young = 1, 0
        else:
            return
    gc.collect(generation)


def _inflate(blob: bytes) -> bytes:
    """Decompress a stream, bounded. Returns the input on a zlib error."""
    try:
        return zlib.decompressobj().decompress(blob, MAX_INFLATED)
    except zlib.error:
        return blob


#: How many stream objects the fallback will read. A real PDF has one or a
#: few per page; this is a bound on work, not on documents.
_MAX_STREAMS = 4096


def _streams(data: bytes):
    """Every ``stream … endstream`` body, found by scanning rather than by
    backtracking.

    The regex this replaces was ``stream\r?\n(.{0,20000000}?)\r?\nendstream``
    with DOTALL: a lazy twenty-million-character run that, for every
    ``stream`` with no ``endstream`` after it, scanned to the end of the
    buffer before giving up. Quadratic in the number of unterminated
    ``stream`` keywords, which cost an attacker seven bytes each — 200 KB of
    them took 68 seconds, 280 KB took 127, and the default max_bytes is
    25 MB.

    find() does the same job in one pass and cannot backtrack. A ``stream``
    with no ``endstream`` after it simply ends the scan, which is also the
    truthful reading of a malformed file.
    """
    at, found = 0, 0
    while found < _MAX_STREAMS:
        start = data.find(b"stream", at)
        if start < 0:
            return
        body = start + 6
        if data[body:body + 2] == b"\r\n":
            body += 2
        elif data[body:body + 1] == b"\n":
            body += 1
        else:
            at = start + 6              # "endstream" itself, or a longer word
            continue
        end = data.find(b"endstream", body)
        if end < 0:
            return                      # unterminated: nothing more to read
        yield data[body:end].rstrip(b"\r\n")
        at, found = end + 9, found + 1


def _best_effort(data: bytes, det_size: int = 0, pypdf_failed: bool = False):
    chunks: list[str] = []
    for blob in _streams(data):
        deadline.check("a PDF")
        chunks.append(_text_ops(_inflate(blob)))
    text = _collapse("\n".join(c for c in chunks if c))
    pages = _page_count(data, det_size)
    title = _meta_field(data, b"/Title")
    author = _meta_field(data, b"/Author")
    meta = {"pages": pages or None, "title": title, "author": author,
            "engine": "stdlib-fallback"}
    meta = {k: v for k, v in meta.items() if v}
    # Installing pypdf is advice only where it is missing: where it failed,
    # it is installed already.
    if pypdf_failed:
        note = "extracted with the stdlib fallback, less reliable than pypdf"
    else:
        note = ("extracted with the stdlib fallback; install 'pypdf' for "
                "reliable output")
    return text, meta, note


#: The smallest a page object can plausibly be. A page needs a dictionary
#: naming its type, its parent and its media box; nothing real gets under this.
#: Used only as a sanity bound on the number the *file* states about itself.
_MIN_BYTES_PER_PAGE = 20


def _page_count(data: bytes, size: int = 0) -> int:
    """Number of pages, without counting the page *tree*.

    Counting the substring '/Page' also matches '/Pages', the node that holds
    them — and in a modern PDF the page dictionaries live inside a compressed
    object stream where the substring does not appear at all, so a 7-page file
    reported 2. The /Count on the page tree is the number the document itself
    states.
    """
    # Only when the file's size is known. Called with a fragment — as the
    # unit tests do — there is no document to weigh the claim against.
    limit = max(1, size // _MIN_BYTES_PER_PAGE) if size else None
    stated = _count_in(data)
    inflated = b""
    if not stated:
        # Modern producers put the catalog and the page tree inside a
        # compressed object stream, where none of those bytes appear in the
        # file at all — a 7-page document reported 2.
        inflated = b"\n".join(_inflate(blob) for blob in _streams(data))
        stated = _count_in(inflated)
    if not stated:
        return _observed_pages(data, inflated)
    # The /Count is the document describing itself, and a document can lie:
    # "/Type /Pages /Count 999999999" in a hundred-byte file was reported as
    # nine hundred million pages, into --json and into the prompt. It cannot
    # exceed what the bytes could hold.
    if limit is None or stated <= limit:
        return stated
    # Refusing the claim must leave a smaller number, not none at all: the
    # fallback counted page objects in the *raw* bytes, which is zero for
    # exactly the object-stream documents this function was rewritten to
    # handle, so `pages` silently disappeared from the metadata.
    if not inflated:
        inflated = b"\n".join(_inflate(blob) for blob in _streams(data))
    return _observed_pages(data, inflated)


def _observed_pages(*blobs: bytes) -> int:
    """Page objects actually present, for when the stated count is not real."""
    return max((len(re.findall(rb"/Type\s*/Page(?![s])", blob))
                for blob in blobs if blob), default=0)


def _count_in(data: bytes) -> int:
    # Bounded gaps. "[^>]*?" runs to the end of the buffer for every
    # "/Type /Pages" with no ">" after it — 100,000 of them is 1.3 MB and
    # over two minutes, forty bytes from the loop that was fixed for exactly
    # this reason. A page-tree dictionary is tens of bytes, not megabytes.
    counts = [int(m.group(1)) for m in
              re.finditer(rb"/Type\s*/Pages\b[^>]{0,400}?/Count\s+(\d+)", data)]
    counts += [int(m.group(1)) for m in
               re.finditer(rb"/Count\s+(\d+)[^>]{0,400}?/Type\s*/Pages\b", data)]
    return max(counts) if counts else 0


def _text_ops(raw: bytes) -> str:
    out: list[str] = []
    for m in re.finditer(rb"\((?:[^()\\]|\\.){0,4000}\)\s*Tj", raw):
        out.append(_unescape(m.group(0)))
    # "[^\[\]]", not ".*?" and not "[^\]]". The lazy form restarts the hunt
    # for "] TJ" at every "[" in the buffer, and so does a class that allows
    # "[" through — both quadratic, and a 118-byte file whose stream inflated
    # to 20 KB of "[" took 1.4 seconds. A TJ array holds strings and kerns and
    # never another "[", so excluding it makes every failed start give up at
    # once. The length bound is belt and braces: an array is a line of glyphs.
    for m in re.finditer(rb"\[([^\[\]]{0,4000})\]\s*TJ", raw):
        out.append(_kerned(m.group(1)))
    return " ".join(s for s in out if s)


#: In a TJ array a number is a kern: positive values pull the next glyph left,
#: negative ones push it right. A large enough negative number is how PDF
#: writes a space, so concatenating the strings glued "Page two here" into
#: "Pagetwohere". The threshold is the usual one — a fifth of an em.
_KERN_SPACE = -100.0
_TJ_TOKEN = re.compile(rb"\((?:[^()\\]|\\.)*\)|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _kerned(array: bytes) -> str:
    out: list[str] = []
    for token in _TJ_TOKEN.finditer(array):
        raw = token.group(0)
        if raw.startswith(b"("):
            out.append(_unescape(raw))
            continue
        try:
            kern = float(raw)
        except ValueError:
            continue
        if kern <= _KERN_SPACE and out and not out[-1].endswith(" "):
            out.append(" ")
    return "".join(out)


def _unescape(token: bytes) -> str:
    inner = re.sub(rb"\)\s*(Tj)?$", b"", token)
    inner = re.sub(rb"^\(", b"", inner)
    inner = inner.replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
    inner = inner.replace(b"\\n", b"\n").replace(b"\\r", b"\r").replace(b"\\t", b"\t")
    return inner.decode("latin-1", errors="replace")


def _meta_field(data: bytes, key: bytes):
    m = re.search(re.escape(key) + rb"\s*\((.*?)\)", data)
    if m:
        return m.group(1).decode("latin-1", errors="replace").strip() or None
    return None


def _clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _collapse(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
