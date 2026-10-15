"""What sift repeats of a model server's failure is redacted and bounded.

A server that fails the chat call and quotes the user message back -- a validation error that echoes its input --
put the file's DB_PASSWORD into Summary.error, and from there into the terminal, the JSON and the markdown report:
_redact_what_sift_says passed over the error, an HTTP body was cut at 500 characters and an Ollama stream's error not
at all. The server's words are now redacted and bounded where they enter an error
(llm.base.repeatable), and the error again where sift says it.
"""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SECRET = "Xq7vR2mTz9LpW4ks"
NOTES = f"deploy notes for the staging box\nDB_PASSWORD={SECRET}\nthe rest is harmless text about the rollout plan.\n"


class _Echo:
    """A model server that fails the chat call and quotes the user message back, in ``mode``:
    http400 (OpenAI dialect, HTTP 400), complaint (HTTP 200 whose body is an error), sse (a stream event that is an
    error), ndjson (an Ollama stream line that is an error), http500 (Ollama, HTTP 500), notjson (an Ollama stream
    line that is not JSON), embed (an OpenAI-dialect embeddings answer whose vector holds the text)."""

    def __init__(self, mode: str, pad: int = 0, spaced: bool = False):
        self.mode, self.pad, self.spaced = mode, pad, spaced

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj, ctype="application/json"):
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    return self._send(200, {"data": [{"id": "m"}], "object": "list"})
                if self.path in ("/api/tags", "/api/ps"):
                    return self._send(200, {"models": [{"name": "m", "model": "m"}]})
                return self._send(404, {"error": "nope"})

            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/api/show":
                    return self._send(200, {"model_info": {"llama.context_length": 8192}})
                users = [m for m in request.get("messages", []) if m.get("role") == "user"]
                echo = users[-1]["content"] if users else " ".join(map(str, request.get("input", [])))
                at = echo.find("DB_PASSWORD")
                pad = "lorem " * (outer.pad // 6) if outer.spaced else "x" * outer.pad
                echo = pad + (echo[max(0, at - 20):] if at >= 0 else echo)
                mode = outer.mode
                if mode == "http400":
                    return self._send(400, {"error": {"message": "invalid input: " + echo}})
                if mode == "complaint":
                    return self._send(200, {"error": "input rejected: " + echo})
                if mode == "sse":
                    event = json.dumps({"error": "input rejected: " + echo})
                    return self._send(200, f"data: {event}\n\n".encode(), "text/event-stream")
                if mode == "ndjson":
                    return self._send(200, (json.dumps({"error": "input rejected: " + echo}) + "\n").encode(),
                                      "application/x-ndjson")
                if mode == "http500":
                    return self._send(500, {"error": echo})
                if mode == "notjson":
                    return self._send(200, ("not json: " + echo.replace("\n", " ") + "\n").encode(),
                                      "application/x-ndjson")
                if mode == "embed":
                    return self._send(200, {"data": [{"index": 0, "embedding": [0.5, echo]}]})
                return self._send(404, {"error": "nope"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _client(kind: str, url: str, stream: bool = True):
    from sift.llm.ollama import OllamaClient
    from sift.llm.openai_compat import OpenAICompatClient
    if kind == "ollama":
        return OllamaClient(url, "m", "", 30.0, True, stream)
    return OpenAICompatClient(url + "/v1", "m", "", 30.0, True, stream, provider="lmstudio")


class ErrorSaidTest(unittest.TestCase):
    """Every client's failure, echoing the file, said without the file's secret, within ERROR_CHARS."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "notes.env"
        self.path.write_text(NOTES)

    def tearDown(self):
        self.dir.cleanup()

    def _summarized(self, kind: str, mode: str, pad: int = 0, stream: bool = True):
        from sift.config import Config
        from sift.summarize import summarize_path
        events = []
        with _Echo(mode, pad) as server:
            summ = summarize_path(self.path, Config(), _client(kind, server.url, stream), cache=None,
                                  on_event=events.append)
        return summ, events

    def _said_everywhere(self, summ, events) -> list:
        from sift.report import render_json, render_markdown, render_text
        return [summ.error or "", render_json([summ]), render_markdown([summ]), render_text(summ, color=False),
                *(json.dumps(event.data, default=str) for event in events),
                *(call.error or "" for call in summ.llm_calls)]

    def test_each_clients_failure_is_said_without_the_files_secret(self):
        from sift.llm.base import ERROR_CHARS
        for kind, mode, stream, words in (("lmstudio", "http400", True, "HTTP 400"),
                                          ("lmstudio", "http400", False, "HTTP 400"),
                                          ("lmstudio", "complaint", False, "reported an error"),
                                          ("lmstudio", "complaint", True, "the server said"),
                                          ("lmstudio", "sse", True, "stream error"),
                                          ("ollama", "ndjson", True, "stream error"),
                                          ("ollama", "http500", True, "HTTP 500"),
                                          ("ollama", "notjson", True, "not JSON")):
            with self.subTest(kind=kind, mode=mode, stream=stream):
                summ, events = self._summarized(kind, mode, stream=stream)
                self.assertIn(words, summ.error)
                self.assertLessEqual(len(summ.error), ERROR_CHARS)
                self.assertNotIn("\n", summ.error)
                for said in self._said_everywhere(summ, events):
                    self.assertNotIn(SECRET, said)
                    self.assertNotIn(SECRET[:6], said)

    def test_every_clients_error_is_within_the_bound_as_it_is_raised(self):
        """Each client cuts what it quotes of the server at SERVER_WORDS or less, and says at most as much again
        around it: every error it raises is within ERROR_CHARS before summarize cuts it."""
        from sift.llm.base import ERROR_CHARS, LLMError
        for kind, mode, stream in (("lmstudio", "http400", True), ("lmstudio", "http400", False),
                                   ("lmstudio", "complaint", False), ("lmstudio", "complaint", True),
                                   ("lmstudio", "sse", True), ("ollama", "ndjson", True),
                                   ("ollama", "http500", True), ("ollama", "notjson", True)):
            with self.subTest(kind=kind, mode=mode, stream=stream), _Echo(mode, 10_000, spaced=True) as server:
                with self.assertRaises(LLMError) as caught:
                    _client(kind, server.url, stream).generate("system", f"notes DB_PASSWORD={SECRET}")
                said = str(caught.exception)
                self.assertLessEqual(len(said), ERROR_CHARS)
                self.assertIn("lorem lorem", said)
                self.assertIn("…", said)
                self.assertNotIn(SECRET, said)

    def test_a_secret_where_a_bound_cuts_is_not_said_in_part(self):
        """Edge: the server's words cut where the secret stands: not a character of it is said, and the cut is
        marked."""
        from sift.llm.base import SERVER_WORDS
        for kind, mode in (("lmstudio", "http400"), ("ollama", "ndjson")):
            for pad in (SERVER_WORDS - 60, SERVER_WORDS - 40, 4 * SERVER_WORDS - 30):
                with self.subTest(kind=kind, mode=mode, pad=pad):
                    summ, events = self._summarized(kind, mode, pad=pad)
                    self.assertIn("…", summ.error)
                    for said in self._said_everywhere(summ, events):
                        self.assertNotIn(SECRET[:4], said)

    def test_a_failure_that_quotes_nothing_secret_keeps_its_words(self):
        """Happy path: what the server said is still said, where there is nothing to hide in it."""
        self.path.write_text("just a harmless line of notes\n")
        summ, _ = self._summarized("ollama", "ndjson")
        self.assertIn("ollama stream error: input rejected: Filename: notes.env", summ.error)

    def test_the_embeddings_answer_is_said_without_the_text(self):
        """A vector that holds the text it was asked for: the error names the value, redacted and bounded."""
        from sift.llm.base import ERROR_CHARS, LLMError, recording_calls
        with _Echo("embed") as server, recording_calls() as calls:
            client = _client("lmstudio", server.url)
            client.embedding_model = "m"
            with self.assertRaises(LLMError) as caught:
                client.embed([NOTES])
        said = str(caught.exception)
        self.assertNotIn(SECRET, said)
        self.assertLessEqual(len(said), ERROR_CHARS)
        self.assertTrue(all(SECRET not in (call.error or "") for call in calls))

    def test_reclassifying_says_the_failure_without_the_secret(self):
        """categories.classify keeps the model's failure for the caller: redacted and bounded the same way, however
        much the server says."""
        from sift.categories import classify
        from sift.llm.base import ERROR_CHARS
        stored = {"kind": "env", "summary": f"notes; DB_PASSWORD={SECRET}", "classification": "notes",
                  "key_points": [], "tags": []}
        for kind, mode in (("lmstudio", "http400"), ("ollama", "ndjson")):
            with self.subTest(kind=kind), _Echo(mode, pad=20_000) as server:
                result = classify(stored, ["Notes", "Code"], _client(kind, server.url))
                self.assertIsNotNone(result.error)
                self.assertNotIn(SECRET, result.error)
                self.assertLessEqual(len(result.error), ERROR_CHARS)

    def test_a_file_that_could_not_be_identified_is_said_without_what_it_quotes(self):
        """An exception's text about the file -- a parser quoting what it choked on -- is redacted and bounded
        in the error record sift makes of the file."""
        from unittest import mock
        from sift.config import Config
        from sift.llm.base import ERROR_CHARS
        from sift.summarize import summarize_path
        with mock.patch("sift.summarize.detect", side_effect=ValueError(f"bad bytes DB_PASSWORD={SECRET} " * 200)):
            summ = summarize_path(self.path, Config(), None, cache=None)
        self.assertEqual(summ.kind, "error")
        self.assertNotIn(SECRET, summ.error + summ.summary)
        self.assertLessEqual(len(summ.error), ERROR_CHARS)


class EscapedWordsTest(unittest.TestCase):
    """A review, still at 0.1.69: a server's JSON error body quotes the file with JSON's escapes --
    Kx9v\\"Q2mZ7pL4Wq for a quote, Kx9v\\tQ2... for a tab --, which is no spelling hide_known knows, and the value
    ran on into the fence after it, so scrub held back too. The strings of such a body are read as their JSON says,
    as a 200's error value always was (complaint_text)."""

    #: The review's two files, whose value scrub reads by its key since 0.1.71 (a marker is no longer taken for
    #: "<" in the value), and four whose key scrub cannot read it by -- only what the file taught hide_known hides
    #: them, in the spelling the file writes. The echo starts at "DB_PASSWORD" (_Echo).
    CASES = (("quote.env", 'DEBUG=true\nTARGET_DB_PASSWORD=Kx9v"Q2mZ7pL4Wq\nMODE=x\n'),
             ("mail.txt", "Hi team,\nthe new DB_PASSWORD: Kx9v\tQ2mZ7pL4Wq8rTz\nregards\n"),
             ("note-quote.env", '# DB_PASSWORD is not here\nthe notes follow here\nNOTE=Kx9v"Q2mZ7pL4Wq\nMODE=x\n'),
             ("note-tab.env", '# DB_PASSWORD is not here\nthe notes follow here\nNOTE="Kx9v\tQ2mZ7pL4Wq8rTz"\nMODE=x\n'),
             ("note-umlaut.env", "# DB_PASSWORD is not here\nthe notes follow here\nNOTE=Kx9vQ2mZ7pL4Wq\u00e4\nMODE=x\n"),
             ("note-bslash.env", "# DB_PASSWORD is not here\nthe notes follow here\nNOTE=Kx9v\\Q2mZ7pL4Wq\nMODE=x\n"))

    def test_an_escaped_quote_of_the_file_is_not_said(self):
        from sift.config import Config
        from sift.summarize import summarize_path
        failures = []
        with tempfile.TemporaryDirectory() as folder:
            for name, body in self.CASES:
                path = Path(folder) / name
                path.write_text(body, encoding="utf-8")
                for kind, mode in (("lmstudio", "http400"), ("ollama", "http500")):
                    events = []
                    with _Echo(mode) as server:
                        summ = summarize_path(path, Config(), _client(kind, server.url), cache=None,
                                              on_event=events.append)
                    said = ErrorSaidTest._said_everywhere(self, summ, events)
                    if any("Q2mZ7pL4Wq" in text for text in said):
                        failures.append((name, mode, summ.error))
        self.assertEqual([], failures)

    def said(self, body: bytes, code: int = 400) -> str:
        from sift.llm.base import _http_message

        class Failure:
            def __init__(self):
                self.code = code

            def read(self, size):
                return body[:size]

        return _http_message("lmstudio", "/v1/chat/completions", Failure())

    # -- sad ---------------------------------------------------------------
    def test_a_body_with_nothing_escaped_is_said_as_it_was(self):
        self.assertEqual("lmstudio HTTP 400 at /v1/chat/completions: {\"error\": \"model not found\"}",
                         self.said(b'{"error": "model not found"}'))

    def test_a_body_that_is_no_json_keeps_its_backslashes(self):
        said = self.said(b"cannot open C:\\new\\table.txt", 500)
        self.assertIn("C:\\new\\table.txt", said)

    # -- edge --------------------------------------------------------------
    def test_escapes_are_read_as_json_reads_them(self):
        said = self.said(b'{"error": "model \\"m\\" \\u00e4\\u00df at \\ud83d\\ude00 / \\ud83d \\x"}')
        self.assertIn('model "m" \u00e4\u00df at \U0001f600 / \ufffd \\x', said)

    def test_a_body_cut_inside_a_string_is_read_too(self):
        body = b'{"error": {"message": "invalid input: DB_PASSWORD: Kx9v\\tQ2mZ7pL4Wq8rTz ' + b"x" * 3000 + b'"}}'
        said = self.said(body)
        self.assertNotIn("Q2mZ7pL4Wq8rTz", said)
        self.assertIn("\u2026", said)


class RepeatableTest(unittest.TestCase):
    """llm.base.repeatable: the server's words as sift may repeat them."""

    def test_short_words_are_kept_on_one_line(self):
        from sift.llm.base import repeatable
        self.assertEqual(repeatable("model  'm'\nnot found", 100), "model 'm' not found")

    def test_a_secret_is_redacted(self):
        from sift.llm.base import repeatable
        said = repeatable(f"invalid input: password={SECRET} and more", 100)
        self.assertNotIn(SECRET, said)
        self.assertIn("invalid input", said)

    def test_words_past_the_bound_are_cut_at_a_space_and_marked(self):
        from sift.llm.base import repeatable
        for limit in (10, 11, 30, 500):
            with self.subTest(limit=limit):
                said = repeatable("alpha beta gamma delta " * 100, limit)
                self.assertLessEqual(len(said), limit)
                self.assertTrue(said.endswith("…"))
                self.assertTrue(all(word in ("alpha", "beta", "gamma", "delta") for word in said[:-1].split()))
        self.assertEqual(repeatable("x" * 1000, 50), "…")
        self.assertEqual(repeatable("", 50), "")
        self.assertEqual(repeatable("exactly ten", 11), "exactly ten")

    def test_a_gap_in_a_value_is_redacted_before_it_is_folded(self):
        """A review: the words were folded onto one line first, and the tab inside "Kx9v<TAB>Q2mZ..."
        became the one blank that ends a value in prose -- "Kx9v" was hidden and the rest said."""
        from sift.llm.base import repeatable
        said = repeatable("invalid input: the new vault password: Kx9v\tQ2mZ7pL4Wq8rTz\nregards", 200)
        self.assertNotIn("Q2mZ7pL4Wq8rTz", said)
        self.assertNotIn("\t", said)
        self.assertTrue(said.endswith(" regards"), said)

    def test_what_the_file_taught_is_hidden_too(self):
        """repeatable says the server's words as everything else sift says (hide_said): a value the file taught
        hide_known is hidden where no rule of scrub's could see it."""
        from sift.extract.base import remember_hidden
        from sift.llm.base import repeatable
        with remember_hidden() as seen:
            seen.add("Kx9vQ2mZ7pL4Wq8r")
            said = repeatable("the server choked on Kx9vQ2mZ7pL4Wq8r here", 200)
        self.assertEqual("the server choked on <16 chars — not shown> here", said)


if __name__ == "__main__":
    unittest.main()
