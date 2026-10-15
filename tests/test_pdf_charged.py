"""What sift charges of a PDF reader's memory is what the reader holds, within a tight factor.

0.1.60's first cut charged each read pypdf makes of an object at the most any shape holds a read (32 bytes, hex
strings measured 28.4), each object kept at 640 bytes, each page walked at 512 and 48 a key: 1.4 to 1.9 times what
tracemalloc traced, and pages100k.pdf (100,000 pages), which 0.1.59 read in 440 MB, fell back at the 512 MiB bound.
The charges are now what the objects hold, measured as they are made (sys.getsizeof, and what an object of each of
pypdf's classes holds past it, _HELD_PAST_SIZEOF); a test holds each within TIGHT of tracemalloc, both ways, on small,
medium and large shapes. Only the parse under way is charged a worst case, and only until it is walked
(_Structure.in_flight).
"""
import gc
import io
import sys
import tracemalloc
import unittest
from unittest import mock

import test_extract_formats as formats
import test_pdf_held as held

#: How much more than what tracemalloc traces a charge may be.
TIGHT = 1.15
#: What tracemalloc traces a measurement allocating once, whatever it measures: the smallest tables a dictionary
#: grows past, which Python keeps on its freelist (280 bytes measured caching 10 objects), a code path's first
#: cache, what a reader holds of its own besides what one of the smallest documents does (796 bytes at the most).
ONCE = 1024
#: The shapes' sizes: small, medium, large.
SIZES = (10, 1_000, 20_000)

PAGE = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 8 0 R /Resources << /Font << /F1 4 0 R >> "
        b">> /StructParents 7 >>")
#: One element of each shape an array is made of: every class read_object makes, and the dictionaries of real files.
ITEMS = {"numbers": b"0 ", "big numbers": b"12345678901 ", "floats": b"0.5 ", "references": b"5 0 R ",
         "big references": b"123456 0 R ", "names": b"/a ", "long names": b"/" + b"a" * 60 + b" ",
         "strings": b"(a)", "strings of two": b"(ab)", "long strings": b"(" + b"a" * 200 + b")", "hex strings": b"<a>",
         "long hex strings": b"<" + b"ab" * 100 + b">", "nulls": b"null ", "booleans": b"true ",
         "byte strings": b"(\xff\xfe\x00)", "empty dictionaries": b"<<>>", "empty arrays": b"[]",
         "one-key dictionaries": b"<</a 0>>", "pages": PAGE,
         "struct elements": b"<< /Type /StructElem /S /TD /P 123 0 R /Pg 45 0 R /K 7 >>",
         "fonts": b"<< /Type /Font /Subtype /TrueType /BaseFont /ABCDEF+Arial /FirstChar 32 /LastChar 255 "
                  b"/Widths 9 0 R /FontDescriptor 10 0 R /Encoding /WinAnsiEncoding >>",
         "streams": b"<< /Length 50 >>\nstream\n" + b"x" * 50 + b"\nendstream\n",
         "large streams": b"<< /Length 5000 >>\nstream\n" + b"x" * 5000 + b"\nendstream\n",
         "nested arrays": b"[[0 1] [2 [3 4]]]"}


def _traced(make):
    """(what ``make`` returned, the bytes tracemalloc traced it holding once made)."""
    gc.collect()
    tracemalloc.start()
    try:
        made = make()
        gc.collect()
        return made, tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()


def _opened_traced(data, bound=10 ** 10, walk=False):
    """(the reader sift opens of ``data``, its _Structure, the bytes tracemalloc traced it holding) -- the page tree
    walked where ``walk``."""
    from sift.extract import pdf
    structure = pdf._Structure(bound, 10 ** 7)

    def opened():
        reader = pdf._open(data, structure)
        if walk:
            len(reader.pages)
        return reader
    with formats._PypdfLog():
        reader, traced = _traced(opened)
    return reader, structure, traced


def _reader():
    """A reader of a small document, one object of each of pypdf's classes cached by it: as reading any document
    does, pypdf has set indirect_reference on an object of each, which each class keeps room for from then on."""
    import pypdf
    from pypdf.generic import read_object
    with formats._PypdfLog():
        reader = pypdf.PdfReader(io.BytesIO(formats.PdfReaderBuildTest._pages_pdf(1)))
    for number, item in enumerate((b"12345678901", b"0.5", b"/a", b"(ab)", b"null", b"true", b"[]", b"<<>>",
                                   b"(\xff\xfe\x00)", b"<< /Length 1 >>\nstream\nx\nendstream\n")):
        reader.cache_indirect_object(0, 10 ** 8 + number, read_object(io.BytesIO(item), reader))
    return reader


class ChargedTightTest(unittest.TestCase):

    def setUp(self):
        import importlib.util
        if importlib.util.find_spec("pypdf") is None:
            self.skipTest("no pypdf")

    def assertTight(self, charged, traced, what):
        self.assertGreaterEqual(charged, traced - ONCE, f"{what}: charged {charged} under traced {traced}")
        self.assertLessEqual(charged, TIGHT * traced, f"{what}: charged {charged} over {TIGHT} x traced {traced}")

    def test_a_parsed_object_is_charged_what_it_holds(self):
        """Every shape read_object makes, at each size: what _held_by charges of the parsed array against what
        tracemalloc traces it holding."""
        from pypdf.generic import read_object
        from sift.extract import pdf
        reader = _reader()
        for item in ITEMS.values():             # what a class's first objects allocate once, freelists filled
            read_object(io.BytesIO(b"[" + item * 1_000 + b"]"), reader)
        for name, item in ITEMS.items():
            for size in SIZES:
                with self.subTest(name, size=size):
                    obj, traced = _traced(lambda: read_object(io.BytesIO(b"[" + item * size + b"]"), reader))
                    self.assertTight(pdf._held_by(obj), traced, name)

    def test_a_cached_object_is_charged_what_caching_it_holds(self):
        """What pypdf's cache holds of an object past the object: its entry, the reference it sets on it, and the
        dictionary that makes of an int's or a bytes' subclass."""
        from pypdf.generic import (ArrayObject, BooleanObject, ByteStringObject, DictionaryObject, FloatObject,
                                   NameObject, NullObject, NumberObject, TextStringObject)
        from sift.extract import pdf
        kinds = {"numbers": lambda i: NumberObject(1_000 + i), "byte strings": lambda i: ByteStringObject(b"x%d" % i),
                 "names": lambda i: NameObject("/n%d" % i), "strings": lambda i: TextStringObject("t%d" % i),
                 "dictionaries": lambda i: DictionaryObject({NameObject("/A"): NumberObject(1)}),
                 "arrays": lambda i: ArrayObject([1]), "floats": lambda i: FloatObject(0.5),
                 "nulls": lambda i: NullObject(), "booleans": lambda i: BooleanObject(True)}
        warm = _reader()
        for name, make in kinds.items():             # what caching an object of a class first allocates, once
            for i in range(1_000):
                before = pdf._cache_size(warm)
                obj = make(i)
                warm.cache_indirect_object(0, 10 ** 7 + len(warm.resolved_objects), obj)
                pdf._cached_held(warm, before, obj)
        for name, make in kinds.items():
            for size in SIZES:
                with self.subTest(name, size=size):
                    reader = _reader()
                    objects = [make(i) for i in range(size)]
                    numbers = [10 ** 6 + i for i in range(size)]

                    def cache():
                        charged = 0
                        for number, obj in zip(numbers, objects):
                            before = pdf._cache_size(reader)
                            reader.cache_indirect_object(0, number, obj)
                            charged += pdf._cached_held(reader, before, obj)
                        return charged
                    charged, traced = _traced(cache)
                    self.assertTight(charged, traced, name)

    def test_a_walked_page_is_charged_what_its_copy_holds(self):
        """The PageObject pypdf makes of each kid of the page tree: a copy of the page's dictionary."""
        from pypdf import PageObject
        from pypdf.generic import IndirectObject, read_object
        from sift.extract import pdf
        for keys in (6, 100, 2_000):
            for size in (10, 1_000, max(10, 200_000 // keys)):
                with self.subTest(keys=keys, size=size):
                    reader = _reader()
                    page = read_object(io.BytesIO(b"<< /Type /Page " + b" ".join(b"/K%d 0" % k for k in range(keys - 1))
                                                  + b" >>"), reader)
                    reader.cache_indirect_object(0, 999_999, page)
                    reference = IndirectObject(999_999, 0, reader)
                    PageObject(reader, reference)       # the class's first allocations
                    pages = []

                    def walk():
                        charged = 0
                        for _ in range(size):
                            copy = PageObject(reader, reference)
                            copy.update(page)
                            pages.append(copy)
                            charged += pdf._page_held(copy)
                        return charged
                    charged, traced = _traced(walk)
                    traced -= sys.getsizeof(pages)
                    self.assertTight(charged, traced, f"{keys} keys")

    def test_entries_are_charged_what_the_reader_holds_of_them(self):
        """A classic table's lines (an entry and a free-entry record each), a stream's entries of type 1 and of
        type 2: what the reader is charged opening a document of them against what it traces holding, less the
        file's bytes."""
        from sift.extract import pdf
        shapes = {"classic": lambda n: formats.PdfReaderBuildTest._pages_pdf(n // 2),
                  "type 1": lambda n: formats.PdfReaderBuildTest._streamed(n, 1),
                  "type 2": lambda n: formats.PdfReaderBuildTest._streamed(n, 2)}
        for name, make in shapes.items():
            # What a reader holds whatever its entries -- the reader, its streams, its trailer --, of one of the
            # smallest documents of the shape, once what opening one first allocates is: charged or not, it is not
            # what is measured here.
            pdf._close(_opened_traced(make(2))[0])
            first, structure, traced = _opened_traced(make(2))
            fixed = traced - structure.held
            pdf._close(first)
            for size in (100, 10_000, 100_000):
                with self.subTest(name, size=size):
                    reader, structure, traced = _opened_traced(make(size))
                    self.assertTight(structure.held, traced - fixed, name)
                    pdf._close(reader)

    def test_the_index_is_charged_what_it_holds_and_capped_at_its_peak(self):
        """The index of "N G obj" (_Found) is charged what it holds once made; how many keys it may hold is what
        is left of the bound at INDEX_KEY_PEAK a key, the most its making was traced holding a key at its peak."""
        from sift.extract import pdf
        for keys in (1_000, 50_000, 300_000):
            marks = b"%PDF-1.7\n" + b"".join(b"\n%d 0 obj" % n for n in range(1, keys + 1))
            with self.subTest(keys=keys):
                gc.collect()
                tracemalloc.start()
                try:
                    found = pdf._Found(marks, keys, lambda: None)
                    gc.collect()
                    traced, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertTight(pdf._index_held(found), traced, "index")
                self.assertLessEqual(peak / keys, pdf.INDEX_KEY_PEAK)

    def test_a_kept_object_stream_is_charged_what_it_holds(self):
        """An object stream kept decoded (_Kept): its bytes, and where each of its objects begins."""
        from sift.extract import pdf
        for count in (10, 1_000, 20_000):
            bodies = [formats.PdfStructureTest.PAGE] * count
            data = formats.PdfStructureTest._object_streams_pdf(count, bodies=bodies)
            with self.subTest(count=count), formats._PypdfLog():
                structure = pdf._Structure(10 ** 9, 10 ** 7)
                reader = pdf._open(data, structure)
                stmnum = next(iter(reader.xref_objStm.values()))[0]
                reader.get_object(stmnum)               # the stream itself, parsed and cached
                before = structure.held
                kept, traced = _traced(lambda: pdf._kept_stream(reader, stmnum))
                self.assertTight(structure.held - before, traced, "kept stream")
                pdf._close(reader)

    def test_the_residuals_are_what_tracemalloc_measures(self):
        """_HELD_PAST_SIZEOF is what an object of each class traces past sys.getsizeof and the attributes _held_by
        counts, measured: each within 8 bytes under what the table says -- the room a class keeps for
        indirect_reference depends on what the process did before, and the table says the most (ONCE, what a
        measurement allocates once, spread over its 20,000 objects)."""
        from pypdf.generic import read_object
        from sift.extract import pdf
        reader = _reader()
        shapes = {"NumberObject": b"12345678901 ", "FloatObject": b"0.5 ", "IndirectObject": b"123456 0 R ",
                  "NameObject": b"/a ", "TextStringObject": b"(ab)", "NullObject": b"null ", "BooleanObject": b"true ",
                  "ArrayObject": b"[]", "DictionaryObject": b"<<>>", "ByteStringObject": b"(\xff\xfe\x00)",
                  "DecodedStreamObject": b"<< /Length 5 >>\nstream\nxxxxx\nendstream\n"}
        for name, item in shapes.items():
            with self.subTest(name):
                read_object(io.BytesIO(b"[" + item * 1_000 + b"]"), reader)
                obj, traced = _traced(lambda: read_object(io.BytesIO(b"[" + item * 20_000 + b"]"), reader))
                self.assertEqual(type(obj[0]).__name__, name)
                with mock.patch.dict(pdf._HELD_PAST_SIZEOF, {name: 0}):
                    pdf._RESIDUALS.clear(), pdf._WALKS.clear()
                    bare = pdf._held_by(obj)
                pdf._RESIDUALS.clear(), pdf._WALKS.clear()
                measured, spread = (traced - bare) / 20_000, ONCE / 20_000
                self.assertLessEqual(measured, pdf._HELD_PAST_SIZEOF[name] + spread)
                self.assertGreaterEqual(measured, pdf._HELD_PAST_SIZEOF[name] - 8 - spread)

    # ---- entries dropped ----------------------------------------------------------------------------------------

    @staticmethod
    def _entries(structure, counted=True):
        """An _Entries of ``structure`` made while the reader is (the entries charged in batches)."""
        from sift.extract import pdf
        structure.building = True
        return pdf._Entries(structure, counted=counted, generation=None if counted else 0)

    @staticmethod
    def _modelled(entries, counted=True):
        """What ``entries`` holds by the size model: its table, its keys and values past what Python shares, and of
        a free line's generation only the numbers of the lines that are free."""
        from sift.extract import pdf
        if not counted:
            return sys.getsizeof(entries) + sum(pdf._int_held(key) for key, free in entries.items() if free is True)
        return sys.getsizeof(entries) + sum(pdf._int_held(key) + pdf._entry_held(value)
                                            for key, value in entries.items())

    def test_entries_pypdf_drops_while_the_reader_is_made_are_given_back(self):
        """pypdf drops each entry that points where no object is while it makes the reader (pypdf 6.19.0
        _reader.py:750, and :726 renumbering one): the batch not yet charged was counted from the table's end, over
        entries it no longer held -- a table they all left raised IndexError, and the reader fell back --, and
        those charged stayed charged. What is charged once they are dropped is what the table holds."""
        from sift.extract import pdf
        values = {"offsets": lambda number: 10 ** 6 + number, "pairs": lambda number: (10 ** 6 + number, number)}
        for kind, value in values.items():
            for dropped in ("every pending one", "charged ones", "charged and pending ones"):
                with self.subTest(kind, dropped=dropped):
                    structure = pdf._Structure(10 ** 10, 10 ** 7)
                    entries = self._entries(structure)
                    for number in range(1_000, 2_000):
                        entries[number] = value(number)
                    if dropped == "every pending one":
                        for number in range(1_000, 2_000):
                            del entries[number]
                    else:
                        entries.settle()
                        for number in range(3_000, 3_500 if dropped != "charged ones" else 3_000):
                            entries[number] = value(number)
                        for number in list(range(1_000, 1_600)) + list(range(3_000, 3_200)):
                            if number in entries:
                                del entries[number]
                    entries.settle()
                    self.assertEqual(structure.held, self._modelled(entries))

    def test_a_free_lines_entries_dropped_are_given_back(self):
        """Edge: a free line's generation holds True or False a line; dropping its entries gives back the numbers
        of the free ones only, those alone charged."""
        from sift.extract import pdf
        structure = pdf._Structure(10 ** 10, 10 ** 7)
        entries = self._entries(structure, counted=False)
        for number in range(1_000, 2_000):
            entries[number] = number % 2 == 0
        entries.settle()
        for number in range(1_000, 1_500):
            del entries[number]
        entries.settle()
        self.assertEqual(structure.held, self._modelled(entries, counted=False))

    def test_dropping_an_entry_the_table_lacks_changes_nothing(self):
        """Sad path: del of a number the table does not list raises KeyError, as a dictionary's del does, and
        charges nothing, the batch pending included."""
        from sift.extract import pdf
        structure = pdf._Structure(10 ** 10, 10 ** 7)
        entries = self._entries(structure)
        for number in range(1_000, 1_100):
            entries[number] = 10 ** 6 + number
        with self.assertRaises(KeyError):
            del entries[5_000]
        entries.settle()
        self.assertEqual(structure.held, self._modelled(entries))

    def test_entries_removed_behind_the_tables_back_are_charged_over_never_under(self):
        """Edge: an entry removed without del -- dict.pop, which pypdf 6.19.0 never uses on its tables -- is not
        given back, and the batch pending is counted over the entries left: over what is held, and never a
        failure."""
        from sift.extract import pdf
        structure = pdf._Structure(10 ** 10, 10 ** 7)
        entries = self._entries(structure)
        for number in range(1_000, 1_100):
            entries[number] = 10 ** 6 + number
        for number in range(1_000, 1_100):
            dict.pop(entries, number)
        entries.settle()
        self.assertGreaterEqual(structure.held, self._modelled(entries))

    def test_a_document_whose_entries_pypdf_drops_is_charged_what_its_reader_holds(self):
        """The reader of a document whose cross-reference stream lists 1,000 to 100,000 entries where no object is,
        which pypdf drops as it makes the reader: charged against what tracemalloc traces it holding. pypdf's
        record of each dropped entry is not made (logging.disable) -- a test's log keeps them, the reader does
        not."""
        import logging
        from sift.extract import pdf

        def make(count):
            return formats.PdfReaderBuildTest._entries_pdf(1, count)
        logging.disable(logging.CRITICAL)
        try:
            pdf._close(_opened_traced(make(1))[0])
            first, structure, traced = _opened_traced(make(1))
            fixed = traced - structure.held
            pdf._close(first)
            for count in (1_000, 20_000, 100_000):
                with self.subTest(count=count):
                    reader, structure, traced = _opened_traced(make(count))
                    self.assertNotIn(1_000, reader.xref[0])         # the premise: pypdf dropped them
                    self.assertTight(structure.held, traced - fixed, "dropped entries")
                    pdf._close(reader)
        finally:
            logging.disable(logging.NOTSET)

    # ---- whole documents ----------------------------------------------------------------------------------------

    def test_a_document_is_charged_what_its_reader_holds(self):
        """The reader of a document once its page tree is walked -- its entries, the page dictionaries parsed and
        cached, their copies, the tree's arrays -- charged against what tracemalloc traces it holding, for pages in
        a classic table and in object streams."""
        from sift.extract import pdf
        shapes = {"classic": lambda n: formats.PdfReaderBuildTest._pages_pdf(n),
                  "object streams": lambda n: held._tagged_pdf(n, 2, 2, per=40)}
        for name, make in shapes.items():
            pdf._close(_opened_traced(make(1), walk=True)[0])
            first, structure, traced = _opened_traced(make(1), walk=True)
            fixed = traced - structure.held
            pdf._close(first)
            for pages in (20, 400, 4_000):
                with self.subTest(name, pages=pages):
                    reader, structure, traced = _opened_traced(make(pages), walk=True)
                    self.assertTight(structure.held, traced - fixed, name)
                    pdf._close(reader)

    def test_pages_a_worst_case_charge_cut_off_are_read(self):
        """pages100k.pdf, scaled: a document whose reader holds less than the bound is read whole, where 0.1.60's
        first cut charged it past the bound. The bound is set between what it holds and 1.4 times that."""
        from sift.extract import pdf
        data = formats.PdfReaderBuildTest._pages_pdf(3_000)
        with formats._PypdfLog():
            (text, _, notes, _), readers = formats.PdfReaderBuildTest._opened(self, data)
        most = readers[0].structure.most
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", int(most * 1.02)):
            got, _, said, _ = pdf._with_pypdf(data)
        self.assertEqual((got, said), (text, notes))
        self.assertIn("[page 3000]\npage 3000", text[-40:] + got[-40:])

    # ---- the parse under way ------------------------------------------------------------------------------------

    def test_one_object_that_fits_is_read_though_its_worst_case_does_not(self):
        """annots3m.pdf, scaled: a page whose /Annots holds 60,000 references holds about 104 bytes a reference, and
        pypdf reads it with ten reads a reference, which a worst case charges 32 each. What the parse under way
        holds is walked before the bound ends it: the page is read."""
        from sift.extract import pdf
        data = held._array_page_pdf(b"5 0 R ", 60_000, pages=2)
        with formats._PypdfLog():
            (text, _, notes, _), readers = formats.PdfReaderBuildTest._opened(self, data)
        most = readers[0].structure.most
        self.assertGreater(60_000 * 10 * pdf.READ_HELD, 2 * most)          # the premise
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", int(most * 1.05)):
            got, _, said, _ = pdf._with_pypdf(data)
        self.assertEqual((got, said), (text, notes))
        self.assertEqual(text, "[page 1]\npage 1\n\n[page 2]\npage 2")

    def test_one_object_that_does_not_fit_ends_as_it_is_parsed(self):
        """Sad path: the same page past the bound ends as pypdf parses it, and what is held stays near the bound."""
        from sift.extract import pdf
        data = held._array_page_pdf(b"0 ", 400_000, pages=2)
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", 4_000_000):
            (text, _, notes, _), peak = formats.PdfStructureTest._peak(lambda: pdf._with_pypdf(data))
        self.assertIsNone(text)
        self.assertIn("walking its page tree", " ".join(notes))
        self.assertLess(peak, 4_000_000 + 2_000_000, f"{peak} bytes held")

    def test_the_parse_under_way_is_walked_rarely(self):
        """Each walk of the parse under way costs what the parse holds: after one, the parse may read what is left
        of the bound at the worst case before the next, so a page of 400,000 numbers under a bound it fits is walked
        a few times, not at every read."""
        from sift.extract import pdf
        data = held._array_page_pdf(b"0 ", 400_000, pages=1)
        walks, real = [], pdf._in_flight_held

        def counting(*args):
            walks.append(1)
            return real(*args)
        with formats._PypdfLog():
            (text, _, _, _), readers = formats.PdfReaderBuildTest._opened(self, data)
        most = readers[0].structure.most
        with formats._PypdfLog(), mock.patch.object(pdf, "MAX_DOCUMENT_HELD", int(most * 1.1)), \
                mock.patch.object(pdf, "_in_flight_held", counting):
            got, _, _, _ = pdf._with_pypdf(data)
        self.assertEqual(got, text)
        self.assertLessEqual(len(walks), 12)
        self.assertGreater(len(walks), 0)


if __name__ == "__main__":
    unittest.main()
