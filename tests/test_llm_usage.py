"""What each call to the model cost, reported for every call.

LLMResult exists only when a call succeeds, and the expensive calls are the
ones that do not: a reasoning model that thinks through its whole budget and
answers nothing, a retry that loses to the answer it was meant to improve, a
part of a large file that fails. So every HTTP call to a chat endpoint is
reported through LLMClient.on_call(CallUsage), whatever became of it, and a
summary collects all of its own in Summary.llm_calls.

Against the fakes over a real socket, and against a scripted transport where
a reply has to be shaped exactly.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.config import Config
from sift.llm import CallUsage, LLMError
from sift.llm.ollama import OllamaClient
from sift.llm.openai_compat import OpenAICompatClient
from sift.summarize import summarize_path
from tests import _fixtures as fx

from fake_lmstudio import DEFAULT_USAGE, FakeLMStudio
from fake_ollama import DEFAULT_COUNTS, FakeOllama

REPLY = ("TYPE: memo\nSUMMARY: A short memo.\nKEY POINTS:\n- one\n"
         "ENTITIES: ACME\nNOTES: none")


def lmstudio(server, **kw) -> OpenAICompatClient:
    return OpenAICompatClient(server.base_url, model="qwen2.5-7b-instruct",
                              **kw)


def ollama(server, **kw) -> OllamaClient:
    return OllamaClient(server.base_url, model="llama3.1:latest", **kw)


def watched(client):
    seen = []
    client.on_call = seen.append
    return seen


class Scripted:
    """A transport answering each completion with the next scripted body:
    a str is an event stream, a dict a JSON body, an int an HTTP status."""

    def __init__(self, *replies, models=None):
        self.replies = list(replies)
        self.payloads = []
        self.models = models or {"object": "list", "data": [
            {"id": "m", "state": "loaded", "type": "llm"}]}

    def __call__(self, req, timeout=None):
        import io
        import urllib.error
        from test_llm_clients import FakeResp
        if "/api/v0/models" in req.full_url:
            return FakeResp(self.models)
        self.payloads.append(json.loads(req.data))
        reply = self.replies.pop(0)
        if isinstance(reply, tuple):
            status, body = reply
            raise urllib.error.HTTPError(req.full_url, status, "no", {},
                                         io.BytesIO(json.dumps(body).encode()))
        if isinstance(reply, int):
            raise urllib.error.HTTPError(req.full_url, reply, "no", {},
                                         io.BytesIO(b""))
        if isinstance(reply, str):
            return FakeResp(None, raw=reply)
        return FakeResp(reply)


def events(*frames, done=True) -> str:
    return "".join("data: " + json.dumps(f) + "\n\n" for f in frames) + (
        "data: [DONE]\n\n" if done else "")


def say(text, finish="stop"):
    return {"choices": [{"delta": {"content": text},
                         "finish_reason": finish}]}


def think(text):
    return {"choices": [{"delta": {"reasoning_content": text}}]}


class OpenAICompatUsageTest(unittest.TestCase):

    # -- happy -------------------------------------------------------------
    def test_a_stream_asks_for_usage_and_reports_the_servers_figures(self):
        with FakeLMStudio(reply=REPLY) as server:
            client = lmstudio(server)
            seen = watched(client)
            result = client.generate("s", "u")
        sent = server.requests[0]
        self.assertEqual({"include_usage": True}, sent["stream_options"])
        self.assertEqual(1, len(seen))
        (call,) = seen
        self.assertIsInstance(call, CallUsage)
        self.assertEqual(("chat", "ok", "server"),
                         (call.kind, call.outcome, call.source))
        self.assertEqual((10, 5, 2), (call.prompt_tokens,
                                      call.completion_tokens,
                                      call.reasoning_tokens))
        self.assertEqual(("lmstudio", "qwen2.5-7b-instruct"),
                         (call.provider, call.model))
        self.assertIsNotNone(call.ttft_seconds)
        self.assertGreaterEqual(call.total_seconds, call.ttft_seconds)
        self.assertEqual("stop", call.stop_reason)
        self.assertEqual([call], result.calls)
        self.assertEqual(REPLY, result.text)

    def test_a_single_answer_carries_its_usage(self):
        with FakeLMStudio(reply=REPLY) as server:
            client = lmstudio(server, stream=False)
            seen = watched(client)
            client.generate("s", "u")
        self.assertNotIn("stream_options", server.requests[0])
        (call,) = seen
        self.assertEqual((10, 5, 2, "server"),
                         (call.prompt_tokens, call.completion_tokens,
                          call.reasoning_tokens, call.source))
        self.assertIsNone(call.ttft_seconds, "not measurable without a "
                                             "stream")

    # -- sad ---------------------------------------------------------------
    def test_a_server_that_refuses_the_option_is_asked_once_without_it(self):
        with FakeLMStudio(reply=REPLY, refuse_stream_options=True) as server:
            client = lmstudio(server)
            seen = watched(client)
            first = client.generate("s", "u")
            second = client.generate("s", "u")
        self.assertEqual(REPLY, first.text)
        self.assertEqual(REPLY, second.text)
        # One refusal, then never again for this client.
        self.assertEqual(1, len(server.refused))
        self.assertEqual(2, len(server.requests))
        self.assertTrue(all("stream_options" not in r
                            for r in server.requests))
        self.assertEqual(["error", "ok", "ok"], [c.outcome for c in seen])
        self.assertIn("stream_options", seen[0].error)
        self.assertIsNone(seen[0].prompt_tokens, "the server read nothing")
        self.assertEqual("estimated", seen[1].source)
        self.assertGreater(seen[1].prompt_tokens, 0)
        self.assertGreater(seen[1].completion_tokens, 0)

    def test_another_bad_request_is_not_retried(self):
        transport = Scripted((400, {"error": {"message": "context length "
                                              "exceeded"}}))
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OpenAICompatClient("http://h:1234/v1")
            seen = watched(client)
            with self.assertRaises(LLMError):
                client.generate("s", "u")
        self.assertEqual(1, len(transport.payloads))
        self.assertEqual(["error"], [c.outcome for c in seen])
        self.assertIn("context length", seen[0].error)

    def test_a_server_that_reports_no_usage_is_estimated(self):
        with FakeLMStudio(reply=REPLY, usage=None) as server:
            client = lmstudio(server)
            seen = watched(client)
            client.generate("system words " * 20, "user words " * 50)
        (call,) = seen
        self.assertEqual("estimated", call.source)
        self.assertGreater(call.prompt_tokens, 50)
        self.assertGreater(call.completion_tokens, 5)
        self.assertEqual(0, call.reasoning_tokens)

    def test_a_failing_server_is_still_reported(self):
        with FakeLMStudio(reply=REPLY, fail_times=5) as server:
            client = lmstudio(server)
            seen = watched(client)
            with self.assertRaises(LLMError):
                client.generate("s", "u")
        self.assertEqual(["error"], [c.outcome for c in seen])
        self.assertGreaterEqual(seen[0].total_seconds, 0.0)

    def test_a_hook_that_raises_costs_the_answer_nothing(self):
        with FakeLMStudio(reply=REPLY) as server:
            client = lmstudio(server)

            def broken(call):
                raise RuntimeError("the caller's database is down")
            client.on_call = broken
            self.assertEqual(REPLY, client.generate("s", "u").text)

    # -- edge --------------------------------------------------------------
    def test_thinking_without_an_answer_is_an_empty_call_then_a_retry(self):
        transport = Scripted(events(think("hmm " * 200)),
                             events(say("SUMMARY: done")))
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OpenAICompatClient("http://h:1234/v1")
            seen = watched(client)
            result = client.generate("s", "u", max_tokens=400)
        self.assertEqual("SUMMARY: done", result.text)
        self.assertEqual([("chat", "empty"), ("retry", "ok")],
                         [(c.kind, c.outcome) for c in seen])
        self.assertEqual(seen, result.calls)
        self.assertGreater(seen[0].reasoning_tokens, 0)
        self.assertEqual("estimated", seen[0].source)

    def test_a_retry_that_loses_the_comparison_is_discarded(self):
        first = events(think("x" * 800), say("SUMMARY: eight rows and six "
                                             "col", finish="length"))
        worse = events(say("SUM", finish="length"))
        transport = Scripted(first, worse)
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OpenAICompatClient("http://h:1234/v1")
            seen = watched(client)
            result = client.generate("s", "u", max_tokens=400)
        self.assertEqual("SUMMARY: eight rows and six col", result.text)
        self.assertEqual([("chat", "ok"), ("retry", "discarded")],
                         [(c.kind, c.outcome) for c in seen])
        self.assertEqual(["length", "length"], [c.stop_reason for c in seen])

    def test_a_retry_that_wins_discards_the_first_answer(self):
        first = events(think("x" * 800), say("SUMMARY: half", finish="length"))
        better = events(say("SUMMARY: the whole answer"))
        transport = Scripted(first, better)
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OpenAICompatClient("http://h:1234/v1")
            seen = watched(client)
            client.generate("s", "u", max_tokens=400)
        self.assertEqual([("chat", "discarded"), ("retry", "ok")],
                         [(c.kind, c.outcome) for c in seen])

    def test_nothing_at_all_twice_is_two_empty_calls_and_an_error(self):
        transport = Scripted(events(think("y" * 400)),
                             events(think("z" * 400)))
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OpenAICompatClient("http://h:1234/v1")
            seen = watched(client)
            with self.assertRaises(LLMError):
                client.generate("s", "u", max_tokens=400)
        self.assertEqual(["empty", "empty"], [c.outcome for c in seen])

    def test_the_usage_chunk_has_no_choices_and_is_not_content(self):
        stream = events(say("SUMMARY: ok"),
                        {"choices": [], "usage": dict(DEFAULT_USAGE)})
        transport = Scripted(stream)
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OpenAICompatClient("http://h:1234/v1")
            seen = watched(client)
            result = client.generate("s", "u")
        self.assertEqual("SUMMARY: ok", result.text)
        self.assertEqual((10, 5, "server"), (seen[0].prompt_tokens,
                                             seen[0].completion_tokens,
                                             seen[0].source))

    def test_tokens_per_second(self):
        call = CallUsage(completion_tokens=100, ttft_seconds=1.0,
                         total_seconds=3.0)
        self.assertEqual(50.0, call.tokens_per_second)
        self.assertIsNone(CallUsage(completion_tokens=None,
                                    total_seconds=3.0).tokens_per_second)
        self.assertIsNone(CallUsage(completion_tokens=10, ttft_seconds=2.0,
                                    total_seconds=2.0).tokens_per_second)
        timed = CallUsage(completion_tokens=12, eval_seconds=0.48,
                          total_seconds=9.0)
        self.assertEqual(25.0, timed.tokens_per_second)


class OllamaUsageTest(unittest.TestCase):

    def test_the_last_line_of_a_stream_says_what_it_cost(self):
        with FakeOllama(reply=REPLY) as server:
            client = ollama(server)
            seen = watched(client)
            result = client.generate("s", "u")
        (call,) = seen
        self.assertEqual(("chat", "ok", "server", "ollama"),
                         (call.kind, call.outcome, call.source, call.provider))
        self.assertEqual((26, 12), (call.prompt_tokens,
                                    call.completion_tokens))
        self.assertIsNone(call.reasoning_tokens)
        self.assertEqual((0.13, 0.48), (call.prompt_eval_seconds,
                                        call.eval_seconds))
        self.assertIsNotNone(call.ttft_seconds)
        self.assertEqual([call], result.calls)
        self.assertEqual(25.0, call.tokens_per_second)

    def test_a_single_answer_says_it_too(self):
        with FakeOllama(reply=REPLY) as server:
            client = ollama(server, stream=False)
            seen = watched(client)
            client.generate("s", "u")
        self.assertEqual((26, 12, "server"), (seen[0].prompt_tokens,
                                              seen[0].completion_tokens,
                                              seen[0].source))

    def test_without_counts_the_tokens_are_estimated(self):
        with FakeOllama(reply=REPLY, usage=False) as server:
            client = ollama(server)
            seen = watched(client)
            client.generate("s", "u")
        self.assertEqual("estimated", seen[0].source)
        self.assertGreater(seen[0].completion_tokens, 0)
        self.assertIsNone(seen[0].eval_seconds)

    def test_an_error_line_is_reported(self):
        stream = json.dumps({"error": "model runner has crashed"}) + "\n"

        def transport(req, timeout=None):
            from test_llm_clients import FakeResp
            if "/api/" in req.full_url and "chat" not in req.full_url:
                return FakeResp({"models": [{"name": "m"}]})
            return FakeResp(None, raw=stream)
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            client = OllamaClient("http://h:11434", model="m")
            seen = watched(client)
            with self.assertRaises(LLMError):
                client.generate("s", "u")
        self.assertEqual(["error"], [c.outcome for c in seen])
        self.assertIn("crashed", seen[0].error)


class SummaryLLMCallsTest(unittest.TestCase):
    """Summary.llm_calls: every call made for the file, and never a fact
    about the file — kept out of the JSON, the cache and the repr."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_one_call_for_a_small_file(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        with FakeLMStudio(reply=REPLY) as server:
            client = lmstudio(server)
            outside = watched(client)
            summ = summarize_path(path, Config(cache=False), client)
        self.assertTrue(summ.llm_used)
        self.assertEqual(["chat"], [c.kind for c in summ.llm_calls])
        self.assertEqual(outside, summ.llm_calls, "the caller's hook sees "
                                                  "the same calls")
        data = summ.to_dict()
        self.assertNotIn("llm_calls", data)
        self.assertNotIn("llm_calls", repr(summ))

    def test_map_and_reduce_are_labelled(self):
        path = fx.write(self.dir / "big.txt",
                        "This is sentence number many. " * 400)
        with FakeLMStudio(reply=REPLY) as server:
            summ = summarize_path(path, Config(cache=False, context_tokens=900,
                                               max_output_tokens=100),
                                  lmstudio(server))
        kinds = [c.kind for c in summ.llm_calls]
        self.assertGreaterEqual(kinds.count("map"), 2)
        self.assertEqual("reduce", kinds[-1])
        self.assertEqual(len(server.requests), len(kinds))

    def test_calls_are_kept_when_the_heuristic_takes_over(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        with FakeLMStudio(reply=REPLY, fail_times=10) as server:
            summ = summarize_path(path, Config(cache=False), lmstudio(server))
        self.assertFalse(summ.llm_used)
        self.assertIsNotNone(summ.error)
        self.assertEqual(["error"], [c.outcome for c in summ.llm_calls])

    def test_a_cache_hit_made_no_calls(self):
        from sift.cache import SummaryCache
        cache = SummaryCache(True, self.dir / "cache")
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        with FakeLMStudio(reply=REPLY) as server:
            client = lmstudio(server)
            first = summarize_path(path, Config(), client, cache)
            again = summarize_path(path, Config(), client, cache)
        self.assertEqual(1, len(first.llm_calls))
        self.assertTrue(again.from_cache)
        self.assertEqual([], again.llm_calls)
        entry = json.dumps([json.loads(p.read_text())
                            for p in (self.dir / "cache").rglob("*.json")])
        self.assertNotIn("llm_calls", entry)

    def test_the_vision_fallback_is_labelled(self):
        from sift.llm.base import NoVisionModel
        path = fx.make_png(self.dir / "i.png")
        calls = []

        class Blind(OpenAICompatClient):
            def generate(self, system, user, *, images=None, **kw):
                if images:
                    raise NoVisionModel("no vision model")
                return super().generate(system, user, images=None, **kw)

        with FakeLMStudio(reply=REPLY) as server:
            client = Blind(server.base_url, model="qwen2.5-7b-instruct")
            client.on_call = calls.append
            summ = summarize_path(path, Config(cache=False), client)
        self.assertEqual(["vision_fallback"],
                         [c.kind for c in summ.llm_calls])

    def test_running_out_of_time_keeps_the_calls_made(self):
        from sift import deadline
        path = fx.write(self.dir / "big.txt",
                        "This is sentence number many. " * 400)
        real = deadline.check
        state = {"parts": 0}

        def expire_on_the_second_part(where=""):
            if where == "a long file, part by part":
                state["parts"] += 1
                if state["parts"] == 2:
                    raise deadline.DeadlineExceeded(1.0, where)
            return real(where)

        with FakeLMStudio(reply=REPLY) as server, mock.patch.object(
                deadline, "check", expire_on_the_second_part):
            summ = summarize_path(path, Config(cache=False, context_tokens=900,
                                               max_output_tokens=100),
                                  lmstudio(server))
        self.assertIn("budget", " ".join(summ.warnings))
        self.assertEqual(["map"], [c.kind for c in summ.llm_calls])

    def test_a_heuristic_run_made_no_calls(self):
        path = fx.write(self.dir / "d.csv", fx.CSV_TEXT)
        summ = summarize_path(path, Config(no_llm=True, cache=False))
        self.assertEqual([], summ.llm_calls)


if __name__ == "__main__":
    unittest.main()
