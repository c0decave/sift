"""End-to-end over a real socket: client, CLI and streaming against a server.

The other LLM tests replace ``urlopen`` and so prove only what we intended to
send. These run the whole path — sockets, chunked transfer encoding, SSE
framing, the CLI's own output — against a stand-in that behaves like LM Studio,
including punishing the one thing the operator forbade: evicting a loaded model.
"""
import json
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path

from sift.config import Config
from sift.llm import build_client
from sift.llm.base import LLMError, NoVisionModel
from sift.llm.openai_compat import OpenAICompatClient

from fake_lmstudio import DEFAULT_MODELS, FakeLMStudio

REPO = Path(__file__).resolve().parent.parent


def client(server: FakeLMStudio, **kw) -> OpenAICompatClient:
    return OpenAICompatClient(server.base_url, provider="lmstudio", **kw)


class StreamOverSocketTest(unittest.TestCase):
    def test_chunked_sse_reassembles_exactly(self):
        text = "TYPE: csv\nSUMMARY: eight rows, six columns."
        with FakeLMStudio(reply=text, chunk_size=3) as srv:
            self.assertEqual(client(srv).generate("s", "u").text, text)

    def test_multibyte_characters_survive_chunk_boundaries(self):
        """A chunk size of 1 splits every UTF-8 sequence it can.

        Decoding per read with errors="replace" would turn these into U+FFFD;
        the summary of a German or Japanese document would come back shredded.
        """
        text = "TYPE: txt\nSUMMARY: Größenänderung, 日本語, emoji 🚀 — done"
        with FakeLMStudio(reply=text, chunk_size=1) as srv:
            got = client(srv).generate("s", "u").text
        self.assertEqual(got, text)
        self.assertNotIn("�", got)

    def test_non_streaming_path_over_a_socket(self):
        with FakeLMStudio(reply="TYPE: x\nSUMMARY: y") as srv:
            self.assertEqual(client(srv, stream=False).generate("s", "u").text,
                             "TYPE: x\nSUMMARY: y")

    def test_http_500_is_reported_not_swallowed(self):
        with FakeLMStudio(fail_times=1) as srv:
            with self.assertRaises(LLMError) as ctx:
                client(srv, stream=False).generate("s", "u")
        self.assertIn("500", str(ctx.exception))


class ResidencyOverSocketTest(unittest.TestCase):
    def test_a_full_run_never_evicts_a_loaded_model(self):
        with FakeLMStudio() as srv:
            c = client(srv)
            for _ in range(3):
                c.generate("s", "u")
            self.assertEqual(srv.evictions, [])
            self.assertEqual(set(srv.models_used), {"qwen2.5-7b-instruct"})

    def test_cold_model_is_refused_before_any_request_is_sent(self):
        with FakeLMStudio() as srv:
            with self.assertRaises(LLMError):
                client(srv, model="mistral-nemo-instruct-2407").generate("s", "u")
            self.assertEqual(srv.evictions, [],
                             "the refusal must happen client-side, not by "
                             "letting the server reject it")
            self.assertEqual(srv.requests, [])

    def test_allow_model_load_reaches_the_cold_model(self):
        with FakeLMStudio() as srv:
            with self.assertRaises(LLMError):   # the fake refuses, loudly
                client(srv, model="mistral-nemo-instruct-2407",
                       allow_model_load=True).generate("s", "u")
            self.assertEqual(srv.evictions, ["mistral-nemo-instruct-2407"])

    def test_image_goes_to_the_vision_model_after_text_traffic(self):
        png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        with FakeLMStudio() as srv:
            c = client(srv)
            c.generate("s", "a text file")
            c.generate("s", "a picture", images=[png])
            self.assertEqual(srv.models_used,
                             ["qwen2.5-7b-instruct", "llava-v1.6-mistral-7b"])
            self.assertEqual(srv.evictions, [])

    def test_no_loaded_vision_model_raises_the_recoverable_error(self):
        models = [dict(m) for m in DEFAULT_MODELS]
        for m in models:
            if m["type"] == "vlm":
                m["state"] = "not-loaded"
        with FakeLMStudio(models=models) as srv:
            with self.assertRaises(NoVisionModel):
                client(srv).generate("s", "u", images=[b"\x89PNG\r\n\x1a\n"])
            self.assertEqual(srv.evictions, [])


class ErrorInA200OverSocketTest(unittest.TestCase):
    """LM Studio says "no such path" with HTTP 200 and an error body."""

    def test_auto_does_not_take_lmstudio_for_ollama(self):
        with FakeLMStudio() as srv:
            chosen = build_client(Config(
                endpoint=f"http://127.0.0.1:{srv.port}"))
        self.assertIsInstance(chosen, OpenAICompatClient)
        self.assertEqual(chosen.provider, "lmstudio")

    def test_a_wrong_base_path_is_an_error_not_an_idle_server(self):
        """/x/v1 makes the residency query hit /x/api/v0/models, which the
        server does not serve. Read as "nothing loaded", that licensed a
        load of a cold model."""
        with FakeLMStudio() as srv:
            wrong = OpenAICompatClient(
                f"http://127.0.0.1:{srv.port}/x/v1", provider="lmstudio")
            with self.assertRaises(LLMError) as caught:
                wrong.generate("s", "u")
            self.assertEqual(srv.evictions, [])
            self.assertEqual(srv.requests, [])
        self.assertIn("Unexpected endpoint or method", str(caught.exception))


class CliOverSocketTest(unittest.TestCase):
    """The CLI as a user runs it — separate process, real argv, real stdout."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "sift", *args],
                              cwd=REPO, capture_output=True, text=True,
                              timeout=120)

    def test_json_output_is_parseable_and_names_the_model(self):
        with FakeLMStudio(reply="TYPE: csv\nSUMMARY: eight employees") as srv:
            r = self._run("examples/employees.csv", "--provider", "lmstudio",
                          "--endpoint", srv.base_url, "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)          # stdout must be JSON and nothing else
        results = data if isinstance(data, list) else data.get("results", [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model"], "qwen2.5-7b-instruct")
        self.assertIn("eight employees", results[0]["summary"])

    def test_models_subcommand_reports_load_state(self):
        with FakeLMStudio() as srv:
            r = self._run("--models", "--provider", "lmstudio",
                          "--endpoint", srv.base_url)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("qwen2.5-7b-instruct", r.stdout)
        self.assertIn("not loaded", r.stdout)      # the cold one is marked
        self.assertIn("already loaded", r.stdout)  # and the pick is explained

    def test_a_dead_backend_still_produces_a_summary(self):
        """Deterministic extraction must never be lost to a network failure."""
        r = self._run("examples/employees.csv", "--provider", "lmstudio",
                      "--endpoint", "http://127.0.0.1:1/v1", "--timeout", "5",
                      "--json")
        data = json.loads(r.stdout)
        results = data if isinstance(data, list) else data.get("results", [])
        self.assertIn("8 rows", results[0]["summary"])
        self.assertIsNone(results[0]["model"])

# ---------------------------------------------------------------------------
# Ollama over a real socket. Not reachable from this environment, so its path
# had only ever been exercised with a replaced urlopen — which proves the
# payload we build and nothing about the bytes on the wire.
# ---------------------------------------------------------------------------
from fake_ollama import FakeOllama  # noqa: E402

from sift.llm.ollama import OllamaClient  # noqa: E402


def ollama(server: FakeOllama, **kw) -> OllamaClient:
    return OllamaClient(server.base_url, **kw)


class OllamaOverSocketTest(unittest.TestCase):
    def test_ndjson_stream_reassembles(self):
        text = "TYPE: log\nSUMMARY: two errors and one critical."
        with FakeOllama(reply=text, chunk_size=3) as srv:
            self.assertEqual(ollama(srv).generate("s", "u").text, text)

    def test_multibyte_characters_survive_chunk_boundaries(self):
        text = "SUMMARY: Größenänderung, 日本語, emoji 🚀"
        with FakeOllama(reply=text, chunk_size=1) as srv:
            got = ollama(srv).generate("s", "u").text
        self.assertEqual(got, text)
        self.assertNotIn("�", got)

    def test_the_resident_model_is_used_and_nothing_is_evicted(self):
        with FakeOllama() as srv:
            client = ollama(srv)
            for _ in range(3):
                client.generate("s", "u")
            self.assertEqual(srv.models_used, ["llama3.1:latest"] * 3)
            self.assertEqual(srv.evictions, [])

    def test_a_cold_model_is_refused_before_the_request_is_sent(self):
        with FakeOllama() as srv:
            with self.assertRaises(LLMError):
                ollama(srv, model="mistral:7b").generate("s", "u")
            self.assertEqual(srv.evictions, [])
            self.assertEqual(srv.requests, [])

    def test_the_implicit_latest_tag_is_refused_too(self):
        """/api/tags only ever returns tagged names; users type the bare one."""
        with FakeOllama() as srv:
            with self.assertRaises(LLMError):
                ollama(srv, model="mistral").generate("s", "u")
            self.assertEqual(srv.evictions, [])

    def test_an_embedding_model_is_never_chosen_for_chat(self):
        with FakeOllama(resident=["nomic-embed-text:latest",
                                  "llama3.1:latest"]) as srv:
            ollama(srv).generate("s", "u")
            self.assertEqual(srv.models_used, ["llama3.1:latest"])

    def test_images_need_a_resident_projector_model(self):
        png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        with FakeOllama() as srv:                      # only llama3.1 resident
            with self.assertRaises(NoVisionModel):
                ollama(srv).generate("s", "u", images=[png])
            self.assertEqual(srv.evictions, [])
        with FakeOllama(resident=["llama3.1:latest", "llava:13b"]) as srv:
            ollama(srv).generate("s", "u", images=[png])
            self.assertEqual(srv.models_used, ["llava:13b"])

    def test_an_idle_server_may_load_since_nothing_is_evicted(self):
        with FakeOllama(resident=[]) as srv:
            with self.assertRaises(LLMError):          # the fake refuses loudly
                ollama(srv).generate("s", "u")
            self.assertTrue(srv.evictions, "an idle server should be allowed")
class ProgressTest(unittest.TestCase):
    """The answer arrives in pieces; a still line looks like a hang.

    A local model takes a minute or more per file. Against the operator's 40B
    this produced 668 updates in 90 seconds — and made visible that the model
    spends its first 450 tokens thinking, which is the phenomenon that had
    silently broken map-reduce.
    """

    def test_the_client_reports_each_piece_as_it_arrives(self):
        seen = []
        with FakeLMStudio(reply="TYPE: csv\nSUMMARY: eight rows", chunk_size=4) as srv:
            c = client(srv)
            c.on_progress = lambda content, reasoning: seen.append(content)
            text = c.generate("s", "u").text
        self.assertEqual(text, "TYPE: csv\nSUMMARY: eight rows")
        self.assertGreater(len(seen), 3, "one callback per streamed piece")
        self.assertEqual(sum(seen), len(text))

    def test_reasoning_is_reported_separately_from_the_answer(self):
        raw = "".join(
            "data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n"
            for d in ({"reasoning_content": "thinking hard"},
                      {"content": "SUMMARY: ok"})) + "data: [DONE]\n\n"
        content, reasoning = [], []
        with FakeLMStudio() as srv:
            srv.reply = raw          # unused; the route below serves it
            c = client(srv)
            c.on_progress = lambda a, b: (content.append(a), reasoning.append(b))
            with mock.patch.object(c, "_stream_lines",
                                   return_value=iter(raw.splitlines())):
                c.generate("s", "u")
        self.assertEqual(sum(reasoning), len("thinking hard"))
        self.assertEqual(sum(content), len("SUMMARY: ok"))

    def test_ollama_reports_progress_too(self):
        seen = []
        with FakeOllama(reply="SUMMARY: two errors", chunk_size=3) as srv:
            c = ollama(srv)
            c.on_progress = lambda content, reasoning: seen.append(content)
            c.generate("s", "u")
        self.assertGreater(len(seen), 3)
        self.assertEqual(sum(seen), len("SUMMARY: two errors"))

if __name__ == "__main__":
    unittest.main()
