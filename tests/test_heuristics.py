"""The no-LLM summariser, the decoder, and the client factory.

The heuristic path is the one sift guarantees: it runs when no model is
reachable, when none is loaded, when the model contradicts the file, and under
--no-llm in CI. It is also the only summary that is true by construction, so
it is the last place that should be taken on trust.
"""
import tempfile
import unittest
from pathlib import Path

from sift.config import Config
from sift.detect import detect
from sift.extract import extract
from sift.extract.base import decode_text
from sift.llm import build_client
from sift.llm.ollama import OllamaClient
from sift.llm.openai_compat import OpenAICompatClient
from sift.summarize import summarize_path


class HeuristicSummaryTest(unittest.TestCase):
    """Each file kind, summarized without a model. Facts must be checkable."""

    def summarize(self, name: str, body, binary: bool = False):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_bytes(body) if binary else p.write_text(body, encoding="utf-8")
            return summarize_path(p, Config(no_llm=True))

    def test_csv_states_its_shape_and_column_types(self):
        s = self.summarize("t.csv", "id,name,salary\n1,Ada,90000\n2,Bob,49000\n")
        self.assertIn("2 rows", s.summary)
        self.assertIn("3 columns", s.summary)
        self.assertIn("salary (integer)", s.summary)
        self.assertIsNone(s.error)

    def test_a_batch_script_names_its_risky_operations(self):
        s = self.summarize("go.bat", "@echo off\ndel /s /q C:\\temp\\*\n"
                                     "shutdown /r /t 0\n")
        flags = " ".join(s.warnings).lower()
        self.assertIn("delete", flags)
        self.assertIn("shutdown", flags)

    def test_a_log_counts_severities_and_shows_the_worst_lines(self):
        s = self.summarize("s.log",
                           "2026-01-01 INFO started\n"
                           "2026-01-01 ERROR disk full\n"
                           "2026-01-01 ERROR retry failed\n"
                           "2026-01-01 CRITICAL paged on-call\n")
        self.assertIn("2 errors", s.summary)
        self.assertIn("1 critical", s.summary)
        self.assertNotIn("{", s.summary, "no dict literals in prose")
        self.assertTrue(any("disk full" in p for p in s.key_points))

    def test_an_email_names_sender_recipient_and_attachments(self):
        s = self.summarize("m.eml",
                           "From: a@x.com\nTo: b@y.com\nSubject: Hello\n"
                           "Date: Mon, 1 Jan 2024 00:00:00 +0000\n\nBody.\n")
        self.assertIn("a@x.com", s.summary)
        self.assertIn("b@y.com", s.summary)
        self.assertIn("Hello", s.summary)

    def test_json_lists_its_top_level_keys(self):
        s = self.summarize("c.json", '{"service": "api", "port": 8080}')
        self.assertIn("service", s.summary)
        self.assertIn("port", s.summary)

    def test_markdown_counts_headings(self):
        s = self.summarize("n.md", "# One\n\ntext\n\n## Two\n\n## Three\n")
        self.assertIn("3 headings", s.summary)
        self.assertIn("One", " ".join(s.key_points))

    def test_an_empty_file_says_so_and_is_not_an_error(self):
        s = self.summarize("e.txt", "")
        self.assertEqual(s.kind, "empty")
        self.assertIsNone(s.error)

    def test_a_binary_reports_entropy_and_strings_with_their_scope(self):
        """The window the entropy was measured over, not a fixed 64 KB.

        An 11-byte file was described as "entropy 3.278 of the first 64 KB",
        and the number of bytes actually read was in the same metadata dict
        the sentence is built from.
        """
        small = self.summarize("x.bin", bytes(range(256)) * 8 + b"readable",
                               binary=True)
        self.assertIn("entropy", small.summary)
        self.assertIn("over the whole file", small.summary)
        # A file larger than the extractor's own window says which window.
        big = self.summarize("y.bin", bytes(range(256)) * 6000, binary=True)
        self.assertIn("entropy", big.summary)
        self.assertIn("over the first", big.summary)

    def test_an_unreadable_file_is_an_error_summary_not_an_exception(self):
        with tempfile.TemporaryDirectory() as d:
            s = summarize_path(Path(d) / "missing.txt", Config(no_llm=True))
        self.assertIsNotNone(s.error)
        self.assertEqual(s.kind, "error")

    def test_every_heuristic_produces_a_non_empty_summary(self):
        """A blank summary is the one output shape that helps nobody."""
        cases = [
            ("t.csv", "a,b\n1,2\n"), ("t.txt", "hello world"),
            ("t.md", "# H\n"), ("t.json", "{}"), ("t.log", "INFO ok\n"),
            ("t.eml", "From: a@x\n\nbody\n"), ("t.sh", "#!/bin/sh\necho hi\n"),
            ("t.xml", "<r><a/></r>"), ("t.ini", "[s]\nk = v\n"),
            ("t.html", "<html><body>hi</body></html>"),
            ("t.py", "def f():\n    return 1\n"),
        ]
        for name, body in cases:
            with self.subTest(name):
                s = self.summarize(name, body)
                self.assertTrue(s.summary.strip(), name)
                self.assertTrue(s.classification.strip(), name)


class DecodeTest(unittest.TestCase):
    def test_plain_utf8(self):
        self.assertEqual(decode_text("Größe".encode("utf-8")), ("Größe", "utf-8"))

    def test_a_bom_is_stripped(self):
        text, enc = decode_text("hello".encode("utf-8-sig"))
        self.assertEqual(text, "hello")
        self.assertEqual(enc, "utf-8")

    def test_utf16_with_and_without_a_bom(self):
        self.assertEqual(decode_text("hallo".encode("utf-16"))[0], "hallo")
        text, enc = decode_text("hallo welt".encode("utf-16-le"))
        self.assertEqual(text, "hallo welt")
        self.assertIn("utf-16", enc)

    def test_a_character_cut_in_half_does_not_condemn_the_whole_file(self):
        """Otherwise cp1252 accepts everything and every umlaut is mangled."""
        blob = "Grüße an alle Kolleginnen und Kollegen".encode("utf-8")
        text, enc = decode_text(blob[:-1], cut_short=True)   # cut mid-character
        self.assertEqual(enc, "utf-8")
        self.assertIn("Grüße", text)

    def test_real_latin1_is_not_silently_shortened(self):
        """b"caf\\xe9" is latin-1 "café" or UTF-8 cut mid-character.

        Trimming without being told the data was cut drops the é from every
        complete latin-1 file — the caller knows which case it is, the bytes
        do not.
        """
        text, enc = decode_text("café".encode("latin-1"))
        self.assertIn(enc, ("cp1252", "latin-1"))
        self.assertEqual(text, "café")


class ClientFactoryTest(unittest.TestCase):
    def test_no_llm_yields_no_client(self):
        self.assertIsNone(build_client(Config(no_llm=True)))

    def test_provider_none_yields_no_client(self):
        self.assertIsNone(build_client(Config(provider="none")))

    def test_explicit_providers_build_the_right_client(self):
        self.assertIsInstance(build_client(Config(provider="ollama")), OllamaClient)
        for name in ("lmstudio", "openai"):
            client = build_client(Config(provider=name))
            self.assertIsInstance(client, OpenAICompatClient)
            self.assertEqual(client.provider, name)

    def test_an_unknown_provider_is_an_error(self):
        from sift.llm import LLMError
        with self.assertRaises(LLMError):
            build_client(Config(provider="chatgpt5"))

    def test_settings_reach_the_client(self):
        cfg = Config(provider="lmstudio", endpoint="http://h:1234/v1",
                     model="m", api_key="k", timeout=42.0,
                     allow_model_load=True, stream=False)
        client = build_client(cfg)
        self.assertEqual(client.endpoint, "http://h:1234/v1")
        self.assertEqual(client.model, "m")
        self.assertEqual(client.api_key, "k")
        self.assertEqual(client.timeout, 42.0)
        self.assertTrue(client.allow_model_load)
        self.assertFalse(client.stream)

    def test_auto_returns_none_when_nothing_answers(self):
        """Both defaults point at loopback ports nothing is listening on."""
        cfg = Config(provider="auto", timeout=1.0,
                     endpoint="http://127.0.0.1:1")
        self.assertIsNone(build_client(cfg))


if __name__ == "__main__":
    unittest.main()


class StructuredSummaryTest(unittest.TestCase):
    """INI, TOML, XML and YAML describe their structure, not our rendering.

    The generic fallback previewed ``ex.text``, which for these kinds is the
    representation sift builds for the model — so an INI file was summarized
    as "Preview: # INI/Config sections: s [s] k = v", quoting a header sift had
    written itself.
    """

    def summarize(self, name: str, body: str):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            p.write_text(body, encoding="utf-8")
            return summarize_path(p, Config(no_llm=True))

    def test_ini_names_its_sections_and_keys(self):
        s = self.summarize("app.ini", "[server]\nhost = a\nport = 80\n"
                                      "[client]\nretries = 3\n")
        self.assertIn("2 sections", s.summary)
        self.assertIn("server", s.summary)
        self.assertNotIn("Preview:", s.summary)
        self.assertTrue(any("host" in p for p in s.key_points))

    def test_toml_names_its_tables(self):
        s = self.summarize("c.toml", '[tool]\nname = "x"\n[build]\nflags = []\n')
        self.assertIn("tool", s.summary)
        self.assertNotIn("Preview:", s.summary)

    def test_a_config_of_many_sections_is_counted_and_listed_in_part(self):
        # 300,000 sections gave a 30 MB summary: every name in the sentence and
        # in the metadata, past the worker's reply limit.
        import json
        from sift.extract.structured import MAX_LISTED
        for n in (MAX_LISTED, MAX_LISTED + 1, 5000):
            with self.subTest(sections=n):
                s = self.summarize("big.ini", "".join(f"[s{i}]\nk = v\n" for i in range(n)))
                self.assertIn(f"{n} sections", s.summary)
                self.assertEqual(n > MAX_LISTED, "…" in s.summary)
                self.assertLess(len(json.dumps(s.to_dict(), default=str)), 64_000)

    def test_a_config_of_many_bare_settings_is_counted_and_listed_in_part(self):
        import json
        s = self.summarize("big.env", "".join(f"KEY_{i}=v\n" for i in range(20000)))
        self.assertIn("20000 settings", s.summary)
        self.assertLess(len(json.dumps(s.to_dict(), default=str)), 64_000)

    def test_toml_of_many_tables_is_counted_and_listed_in_part(self):
        import json
        s = self.summarize("big.toml", "".join(f"[t{i}]\nk = 1\n" for i in range(5000)))
        self.assertIn("5000 top-level tables", s.summary)
        self.assertLess(len(json.dumps(s.to_dict(), default=str)), 64_000)

    def test_xml_names_its_root_and_children(self):
        s = self.summarize("d.xml", "<catalog><book/><book/><cd/></catalog>")
        self.assertIn("catalog", s.summary)
        self.assertIn("book ×2", " ".join(s.key_points))

    def test_an_unparseable_document_says_so_instead_of_previewing_itself(self):
        s = self.summarize("bad.xml", "<catalog><book></catalog>")
        self.assertIn("could not be parsed", s.summary)

    def test_an_empty_file_is_described_once(self):
        s = self.summarize("e.txt", "")
        self.assertEqual(s.summary, "Empty file — 0 bytes, nothing to summarize.")

    def test_an_email_without_a_date_does_not_claim_one(self):
        s = self.summarize("m.eml", "From: a@x.com\nTo: b@y.com\n"
                                    "Subject: Test\n\nHello.\n")
        self.assertNotIn("unknown", s.summary)
        self.assertIn("0 attachments", s.summary)
