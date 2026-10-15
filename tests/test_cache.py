"""The cache, tested mostly for when it must NOT answer.

A cache that returns a stale summary is worse than no cache: the answer looks
fresh, carries a model name, and is wrong. So the interesting cases here are
the misses — changed content, a different model, different settings — not the
hits. Measured benefit for the hit case, against the operator's LM Studio: a
two-file run went from 158.7 s to 0.07 s with identical summaries.
"""
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from sift.cache import CACHE_VERSION, SummaryCache, cache_key, file_digest
from sift.config import Config
from sift.report import render_json
from sift.summarize import summarize_path


class KeyTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(no_llm=True)
        self.base = cache_key("abc123", self.cfg, "lmstudio", "qwen")

    def test_the_same_inputs_give_the_same_key(self):
        self.assertEqual(cache_key("abc123", self.cfg, "lmstudio", "qwen"),
                         self.base)

    def test_different_content_is_a_different_key(self):
        self.assertNotEqual(cache_key("def456", self.cfg, "lmstudio", "qwen"),
                            self.base)

    def test_a_different_model_is_a_different_key(self):
        """Two models do not write the same summary; the cache must not pretend."""
        self.assertNotEqual(cache_key("abc123", self.cfg, "lmstudio", "llama3"),
                            self.base)

    def test_a_different_provider_is_a_different_key(self):
        self.assertNotEqual(cache_key("abc123", self.cfg, "ollama", "qwen"),
                            self.base)

    def test_every_setting_that_changes_the_answer_changes_the_key(self):
        for field, value in (("context_tokens", 16384), ("max_output_tokens", 500),
                             ("temperature", 0.9), ("max_bytes", 1000),
                             ("vision", False), ("no_llm", not self.cfg.no_llm)):
            altered = replace(self.cfg, **{field: value})
            self.assertNotEqual(cache_key("abc123", altered, "lmstudio", "qwen"),
                                self.base, field)

    def test_the_active_scan_reach_is_in_the_key_only_where_it_reads_further(self):
        """A summary holds what the active-content scan found, and a scan that
        read further finds more; a reach that reads no further than max_bytes
        is the default run's key."""
        for same in (0, self.cfg.max_bytes, self.cfg.max_bytes - 1):
            with self.subTest(same=same):
                self.assertEqual(cache_key("abc123", replace(self.cfg, active_max_bytes=same),
                                           "lmstudio", "qwen"), self.base)
        further = replace(self.cfg, active_max_bytes=self.cfg.max_bytes + 1)
        self.assertNotEqual(cache_key("abc123", further, "lmstudio", "qwen"), self.base)


class DigestTest(unittest.TestCase):
    def test_identical_bytes_give_identical_digests_regardless_of_name_or_time(self):
        """mtime is not the key: a restored backup has new times, same bytes."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.txt", Path(d) / "b.txt"
            a.write_bytes(b"same content")
            b.write_bytes(b"same content")
            import os
            os.utime(b, (0, 0))
            self.assertEqual(file_digest(a), file_digest(b))

    def test_one_changed_byte_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_bytes(b"content A")
            first = file_digest(p)
            p.write_bytes(b"content B")
            self.assertNotEqual(file_digest(p), first)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = SummaryCache(True, Path(self._tmp.name))

    def test_a_stored_summary_comes_back(self):
        self.cache.put("k1", {"filename": "x.csv", "summary": "eight rows"})
        self.assertEqual(self.cache.get("k1")["summary"], "eight rows")

    def test_an_unknown_key_is_a_miss(self):
        self.assertIsNone(self.cache.get("nope"))

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self):
        self.cache.put("k1", {"summary": "fine"})
        path = next(Path(self._tmp.name).rglob("*.json"))
        path.write_text("{ this is not json")
        self.assertIsNone(self.cache.get("k1"))

    def test_a_foreign_json_value_at_the_expected_path_is_a_miss(self):
        self.cache.put("k1", {"summary": "fine"})
        path = next(Path(self._tmp.name).rglob("*.json"))
        path.write_text('["not", "a", "cache envelope"]', encoding="utf-8")
        self.assertIsNone(self.cache.get("k1"))

    def test_an_entry_from_an_older_version_is_a_miss(self):
        """The version is how a changed summariser invalidates the past."""
        self.cache.put("k1", {"summary": "old"})
        path = next(Path(self._tmp.name).rglob("*.json"))
        entry = json.loads(path.read_text())
        entry["cache_version"] = str(CACHE_VERSION) + "-stale"
        path.write_text(json.dumps(entry))
        self.assertIsNone(self.cache.get("k1"))

    def test_the_version_tracks_the_code_that_writes_summaries(self):
        """Nobody has to remember to bump it — editing a summariser does.

        Twice in one session the heuristics were rewritten and the old answers
        kept being served, which is indistinguishable from the rewrite not
        working.
        """
        from sift import cache as cache_module
        first = cache_module._code_fingerprint()
        target = Path(cache_module.__file__).parent / "summarize.py"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# a change that alters answers\n")
            self.assertNotEqual(cache_module._code_fingerprint(), first)
        finally:
            target.write_bytes(original)
        self.assertEqual(cache_module._code_fingerprint(), first)

    def test_a_disabled_cache_neither_stores_nor_answers(self):
        off = SummaryCache(False, Path(self._tmp.name))
        off.put("k1", {"summary": "x"})
        self.assertIsNone(off.get("k1"))
        self.assertEqual(list(Path(self._tmp.name).rglob("*.json")), [])

    def test_an_unwritable_directory_does_not_break_the_run(self):
        blocked = SummaryCache(True, Path(self._tmp.name) / "file-not-dir")
        Path(self._tmp.name, "file-not-dir").write_text("x")
        blocked.put("k1", {"summary": "x"})      # must not raise
        self.assertIsNone(blocked.get("k1"))

    def test_clear_removes_everything_and_counts_it(self):
        keys = [f"{i:064x}" for i in range(3)]
        for i, key in enumerate(keys):
            self.cache.put(key, {"summary": str(i)})
        self.assertEqual(self.cache.clear(), 3)
        self.assertIsNone(self.cache.get(keys[0]))

    def test_clear_preserves_everything_that_is_not_a_cache_entry(self):
        key = "ab" + "0" * 62
        self.cache.put(key, {"summary": "cached"})
        root = Path(self._tmp.name)
        sentinel = root / "do-not-delete.txt"
        sentinel.write_text("operator data", encoding="utf-8")
        lookalike = root / "ab" / "ab-not-a-cache-entry.json"
        lookalike.write_text('{"summary": "user document"}', encoding="utf-8")
        nested = root / "project" / "nested.json"
        nested.parent.mkdir()
        nested.write_text('{"important": true}', encoding="utf-8")

        self.assertEqual(self.cache.clear(), 1)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "operator data")
        self.assertTrue(lookalike.is_file())
        self.assertTrue(nested.is_file())
        self.assertTrue(root.is_dir())

    def test_clear_preserves_a_foreign_file_with_the_old_cache_envelope(self):
        """Shape alone is not ownership in an operator-selected directory."""
        root = Path(self._tmp.name)
        key = "ab" + "1" * 62
        foreign = root / "ab" / f"{key}.json"
        foreign.parent.mkdir()
        foreign.write_text(json.dumps({
            "cache_version": CACHE_VERSION,
            "stored_at": 1,
            "summary": {"summary": "operator document"},
        }), encoding="utf-8")

        self.assertEqual(self.cache.clear(), 0)
        self.assertTrue(foreign.is_file())

    def test_eviction_never_counts_or_deletes_foreign_json(self):
        root = Path(self._tmp.name)
        foreign = root / "project" / "nested.json"
        foreign.parent.mkdir()
        foreign.write_text('{"operator": "data"}' * 100, encoding="utf-8")

        with mock.patch("sift.cache.MAX_CACHE_BYTES", 1):
            self.cache.put("cd" + "2" * 62, {"summary": "cached"})

        self.assertTrue(foreign.is_file())
        self.assertEqual(self.cache.evicted, 1)
        self.assertEqual(foreign.read_text(encoding="utf-8"),
                         '{"operator": "data"}' * 100)

    def test_clear_does_not_follow_a_cache_shaped_symlink(self):
        root = Path(self._tmp.name)
        outside = root.parent / f"{root.name}-outside.json"
        outside.write_text(json.dumps({
            "cache_version": CACHE_VERSION,
            "stored_at": 1,
            "summary": {"summary": "outside"},
        }), encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        key = "aa" + "3" * 62
        bucket = root / "aa"
        bucket.mkdir()
        link = bucket / f"{key}.json"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available")

        self.assertEqual(self.cache.clear(), 0)
        self.assertTrue(outside.is_file())
        self.assertTrue(link.is_symlink())


class EndToEndTest(unittest.TestCase):
    """Through summarize_path, where the cache actually earns its keep."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cache = SummaryCache(True, self.dir / "cache")
        self.file = self.dir / "t.csv"
        self.file.write_text("id,name\n1,Ada\n2,Bob\n", encoding="utf-8")
        self.cfg = Config(no_llm=True)

    def test_the_second_run_is_served_from_cache(self):
        first = summarize_path(self.file, self.cfg, None, self.cache)
        second = summarize_path(self.file, self.cfg, None, self.cache)
        self.assertNotIn("from cache", first.notes)
        self.assertIn("from cache", second.notes)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.metadata, second.metadata)

    def test_editing_the_file_invalidates_it(self):
        summarize_path(self.file, self.cfg, None, self.cache)
        self.file.write_text("id,name\n1,Ada\n2,Bob\n3,Cy\n", encoding="utf-8")
        again = summarize_path(self.file, self.cfg, None, self.cache)
        self.assertNotIn("from cache", again.notes)
        self.assertIn("3 rows", again.summary)

    def test_changed_settings_invalidate_it(self):
        summarize_path(self.file, self.cfg, None, self.cache)
        other = replace(self.cfg, max_bytes=12)
        again = summarize_path(self.file, other, None, self.cache)
        self.assertNotIn("from cache", again.notes)

    def test_an_unreadable_file_is_not_cached_as_a_failure(self):
        """A transport error must not be served instantly forever after."""
        missing = self.dir / "gone.csv"
        summ = summarize_path(missing, self.cfg, None, self.cache)
        self.assertIsNotNone(summ.error)
        self.assertEqual(list((self.dir / "cache").rglob("*.json"))
                         if (self.dir / "cache").exists() else [], [])

class FilenameKeyTest(unittest.TestCase):
    """The name is part of the answer, so it is part of the key.

    Identical bytes named t.txt and t.csv are a text file and a table — the
    extension steers detection, and the name goes into the prompt. Without it
    the cache answered "Plain text with 3 lines" for a CSV.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.cache = SummaryCache(True, self.dir / "cache")

    def test_same_bytes_different_extension_is_a_different_summary(self):
        body = "id,name\n1,Ada\n2,Bob\n"
        as_text = self.dir / "t.txt"
        as_csv = self.dir / "t.csv"
        as_text.write_text(body, encoding="utf-8")
        as_csv.write_text(body, encoding="utf-8")
        first = summarize_path(as_text, Config(no_llm=True), None, self.cache)
        second = summarize_path(as_csv, Config(no_llm=True), None, self.cache)
        self.assertEqual(first.kind, "text")
        self.assertEqual(second.kind, "csv")
        self.assertNotIn("from cache", second.notes)

    def test_the_filename_alone_changes_the_key(self):
        cfg = Config(no_llm=True)
        self.assertNotEqual(cache_key("h", cfg, "p", "m", "e", "a.csv"),
                            cache_key("h", cfg, "p", "m", "e", "b.csv"))

class UnwritableCacheTest(unittest.TestCase):
    """A cache that cannot be written must not end the run.

    The whole point of the cache is that it is an optimisation. Every failure
    on the way in is swallowed — but the swallowing lives in `except (OSError,
    ValueError)`, and the `finally` that cleans up the temp file runs *first*.
    If the temp file was never created, `temp` is None, and a cleanup that
    touched it anyway would raise AttributeError — which that except does not
    catch, and which would take down a run over hundreds of files because one
    directory turned read-only.
    """

    def test_a_failure_before_the_temp_file_exists_is_swallowed(self):
        import shutil
        import tempfile as _tempfile
        from pathlib import Path
        from unittest import mock

        from sift.cache import SummaryCache

        directory = Path(_tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        cache = SummaryCache(True, directory)

        def refuse(*args, **kwargs):
            raise OSError(13, "Permission denied")

        with mock.patch("tempfile.NamedTemporaryFile", refuse):
            cache.put("k", {"summary": "x"})       # must not raise

        self.assertIsNone(cache.get("k"))
        # And nothing half-written was left where a later read could find it.
        self.assertEqual([p for p in directory.rglob("*") if p.is_file()], [])


if __name__ == "__main__":
    unittest.main()


class CachedRedactionTest(unittest.TestCase):
    """A cache hit skips the reading. It must not skip the hiding.

    What gets hidden is learned while reading the file, and a hit does not
    read the file — so if the redaction happened anywhere downstream of the
    cache, the second run would publish what the first one hid. It does not,
    because what is stored is the summary after redaction; this is the test
    that says so, and the one that checks the credential is not sitting in
    a file under ~/.cache either.
    """

    SECRETS = ("Winterreise2026", "884213")
    KEPT = ("buchhaltung_svc", "4471")
    BODY = ("Uebergabe Buchhaltung\n"
            "Der Dienstnutzer heisst buchhaltung_svc.\n"
            "Das Kennwort lautet Winterreise2026!\n"
            "Die TAN ist 884213.\n"
            "Rechnung 4471 ist offen.\n")

    def test_every_run_hides_it_including_the_cached_ones(self):
        for run in (1, 2, 3):
            summ = summarize_path(self.file, self.cfg, None, self.cache)
            rendered = render_json([summ])
            for secret in self.SECRETS:
                self.assertNotIn(secret, rendered, f"run {run}")
            for kept in self.KEPT:
                self.assertIn(kept, rendered, f"run {run}")

    def test_the_credential_is_not_written_to_disk(self):
        summarize_path(self.file, self.cfg, None, self.cache)
        on_disk = []
        for path in Path(self.cache_dir).rglob("*"):
            if path.is_file():
                on_disk.append(path.read_bytes())
        blob = b"\n".join(on_disk)
        for secret in self.SECRETS:
            self.assertNotIn(secret.encode(), blob, secret)

    def test_editing_the_file_replaces_the_answer(self):
        summarize_path(self.file, self.cfg, None, self.cache)
        self.file.write_text("Uebergabe Buchhaltung\n"
                             "Neues Kennwort: Sommernacht2027!\n"
                             "Rechnung 4472 ist offen.\n", encoding="utf-8")
        rendered = render_json([summarize_path(self.file, self.cfg, None,
                                               self.cache)])
        self.assertNotIn("Sommernacht2027", rendered)
        self.assertIn("4472", rendered)
        self.assertNotIn("4471", rendered)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.cache_dir, ignore_errors=True)
        self.file = Path(self.dir) / "uebergabe.txt"
        self.file.write_text(self.BODY, encoding="utf-8")
        self.cache = SummaryCache(Path(self.cache_dir))
        self.cfg = Config(no_llm=True, quiet=True, cache=True)


class KeepTextTest(unittest.TestCase):
    """keep_text hands the caller the text sift read; from_cache says where
    the summary came from. Both are for the caller of summarize_path, and
    neither may reach the JSON, the cache, or a key.
    """

    #: The keys of a result in schema version 2. A new runtime field that
    #: leaked into to_dict() would change this, and every consumer pinned on
    #: the schema with it.
    SCHEMA_2 = ("category", "category_origin", "category_set_hash",
                "classification", "detection_confidence", "elapsed_seconds",
                "entities", "error", "filename", "findings", "key_points",
                "kind", "language", "llm_used", "metadata", "mime", "model",
                "natural_language", "notes", "path", "provider", "size_bytes",
                "size_human", "summary", "tags", "tags_suspect", "truncated",
                "warnings")

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.cache = SummaryCache(True, self.dir / "cache")
        self.file = self.dir / "t.csv"
        self.file.write_text("id,name,password\n1,Ada,Winterreise2026\n"
                             "2,Bob,Sommernacht2027\n", encoding="utf-8")
        self.cfg = Config(no_llm=True)

    def _rendered(self):
        from sift.detect import detect
        from sift.extract import extract
        return extract(self.file, detect(self.file), self.cfg.max_bytes).text

    def _entries(self):
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in (self.dir / "cache").rglob("*.json")]

    # -- happy -------------------------------------------------------------
    def test_a_miss_keeps_the_text_it_read(self):
        summ = summarize_path(self.file, self.cfg, None, self.cache,
                              keep_text=True)
        self.assertEqual(self._rendered(), summ.text)
        self.assertIn("# Tabular data", summ.text)
        self.assertFalse(summ.text_truncated)
        self.assertFalse(summ.from_cache)

    def test_the_text_is_not_redacted_though_the_summary_is(self):
        summ = summarize_path(self.file, self.cfg, None, self.cache,
                              keep_text=True)
        self.assertIn("Winterreise2026", summ.text)
        self.assertNotIn("Winterreise2026", render_json([summ]))

    def test_a_hit_reads_the_text_again_and_says_it_is_a_hit(self):
        first = summarize_path(self.file, self.cfg, None, self.cache)
        second = summarize_path(self.file, self.cfg, None, self.cache,
                                keep_text=True)
        self.assertFalse(first.from_cache)
        self.assertIsNone(first.text)
        self.assertTrue(second.from_cache)
        self.assertEqual(self._rendered(), second.text)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.metadata, second.metadata)

    def test_from_cache_does_not_depend_on_disclosing_it(self):
        quiet = replace(self.cfg, disclose_cache=False)
        summarize_path(self.file, quiet, None, self.cache)
        hit = summarize_path(self.file, quiet, None, self.cache)
        self.assertTrue(hit.from_cache)
        self.assertNotIn("from cache", hit.notes)
        self.assertIsNone(hit.text)

    def test_the_text_is_not_in_the_repr(self):
        """A Summary in a log line or a traceback is printed by repr(), and
        the text is the file, unredacted."""
        summ = summarize_path(self.file, self.cfg, None, self.cache,
                              keep_text=True)
        self.assertIn("Winterreise2026", summ.text)
        self.assertNotIn("Winterreise2026", repr(summ))
        self.assertNotIn("# Tabular data", repr(summ))
        self.assertIn("summary=", repr(summ))

    # -- sad ---------------------------------------------------------------
    def test_running_out_of_time_on_the_reread_keeps_the_cached_summary(self):
        from sift import deadline
        cached = summarize_path(self.file, self.cfg, None, self.cache)

        def out_of_time(*args, **kwargs):
            raise deadline.DeadlineExceeded(60.0, "a table")

        with mock.patch("sift.summarize.extract", out_of_time):
            hit = summarize_path(self.file, self.cfg, None, self.cache,
                                 keep_text=True)
        self.assertIsNone(hit.error)
        self.assertTrue(hit.from_cache)
        self.assertIsNone(hit.text)
        self.assertIn("text not re-read within the time budget", hit.notes)
        self.assertEqual(cached.summary, hit.summary)

    def test_a_file_that_cannot_be_read_has_no_text(self):
        summ = summarize_path(self.dir / "gone.csv", self.cfg, None,
                              self.cache, keep_text=True)
        self.assertIsNotNone(summ.error)
        self.assertIsNone(summ.text)
        self.assertFalse(summ.from_cache)

    # -- edge --------------------------------------------------------------
    def test_none_of_it_is_written_to_the_cache(self):
        summarize_path(self.file, self.cfg, None, self.cache, keep_text=True)
        summarize_path(self.file, self.cfg, None, self.cache, keep_text=True)
        entries = self._entries()
        self.assertEqual(1, len(entries))
        stored = entries[0]["summary"]
        for key in ("from_cache", "text", "text_truncated"):
            self.assertNotIn(key, stored)
        blob = json.dumps(entries)
        self.assertNotIn("Winterreise2026", blob)
        self.assertNotIn("# Tabular data", blob)

    def test_runtime_fields_in_a_cache_entry_are_not_believed(self):
        """sift never writes them, so an entry that has them was written by
        something else — an older build, a hand, another program sharing the
        directory — and a hit must not hand them to the caller: here a text
        the file does not contain, without keep_text, and a flag saying the
        model's answer was cut off."""
        summarize_path(self.file, self.cfg, None, self.cache)
        (path,) = (self.dir / "cache").rglob("*.json")
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["summary"].update(text="STALE TEXT", text_truncated=True,
                                truncated_answer=True, from_cache=False)
        path.write_text(json.dumps(entry), encoding="utf-8")
        hit = summarize_path(self.file, self.cfg, None, self.cache)
        self.assertTrue(hit.from_cache)
        self.assertIsNone(hit.text)
        self.assertFalse(hit.text_truncated)
        self.assertFalse(hit.truncated_answer)
        kept = summarize_path(self.file, self.cfg, None, self.cache,
                              keep_text=True)
        self.assertEqual(self._rendered(), kept.text)

    def test_an_entry_whose_summary_is_not_an_object_is_a_miss(self):
        summarize_path(self.file, self.cfg, None, self.cache)
        (path,) = (self.dir / "cache").rglob("*.json")
        entry = json.loads(path.read_text(encoding="utf-8"))
        for shape in (["not", "a", "summary"], "text", 42):
            with self.subTest(shape=shape):
                entry["summary"] = shape
                path.write_text(json.dumps(entry), encoding="utf-8")
                summ = summarize_path(self.file, self.cfg, None, self.cache)
                self.assertFalse(summ.from_cache)
                self.assertIsNone(summ.error)

    def test_the_json_keeps_the_schema_whatever_is_kept(self):
        plain = summarize_path(self.file, self.cfg, None, None)
        kept = summarize_path(self.file, self.cfg, None, None, keep_text=True)
        self.assertIsNotNone(kept.text)
        kept.from_cache = plain.from_cache = True
        for summ in (plain, kept):
            self.assertEqual(sorted(self.SCHEMA_2), sorted(summ.to_dict()))
        normal = [dict(s.to_dict(), elapsed_seconds=0) for s in (plain, kept)]
        self.assertEqual(normal[0], normal[1])

    def test_the_cli_json_is_the_schema(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from sift.cli import main
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            self.assertEqual(0, main(["--no-llm", "--no-cache", "--json",
                                      str(self.file)]))
        result = json.loads(out.getvalue())["results"][0]
        self.assertEqual(sorted(self.SCHEMA_2), sorted(result))

    def test_a_text_cut_at_max_bytes_says_so(self):
        big = self.dir / "big.txt"
        big.write_text("line of prose\n" * 2000, encoding="utf-8")
        small = replace(self.cfg, max_bytes=1000)
        for run in ("miss", "hit"):
            with self.subTest(run):
                summ = summarize_path(big, small, None, self.cache,
                                      keep_text=True)
                self.assertTrue(summ.text_truncated)
                self.assertLessEqual(len(summ.text), 1000)
                self.assertEqual(run == "hit", summ.from_cache)
