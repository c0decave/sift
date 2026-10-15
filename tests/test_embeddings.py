"""Embeddings: vectors from the model server, under the rule that governs
every model sift uses — never make the server load a cold one.

LLMClient.embed(texts) -> EmbeddingResult(vectors, model, dims, usage):
Ollama's POST /api/embed, the OpenAI dialect's POST /v1/embeddings (LM
Studio documents it). The model is one of kind "embeddings" that is not a
reranker — the name hints behind kind lump the two together, and a
reranker scores pairs, it does not embed — and like a chat model it is
used only if it is resident, or if loading it can evict nothing.

embeddings.embed_input(summary, text) says what gets embedded: with a
model's summary its summary, key points and tags; without one the head of
the text; either way within the embedding model's context.

Against the fakes over a real socket for the path that works, and against
a scripted transport where a body has to be shaped exactly wrong.
"""
import io
import json
import math
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.chunk import estimate_tokens
from sift.embeddings import embed_input
from sift.llm import CallUsage, EmbeddingResult, LLMError
from sift.llm.base import is_reranker
from sift.llm.ollama import OllamaClient
from sift.llm.openai_compat import OpenAICompatClient

from corpus_secrets import GITHUB_TOKEN
from fake_lmstudio import DEFAULT_MODELS, FakeLMStudio, fake_vector
from fake_ollama import FakeOllama

EMBEDDER = "text-embedding-nomic-embed-text-v1.5"
RERANKER = {"id": "bge-reranker-v2-m3", "type": "embeddings",
            "state": "loaded", "max_context_length": 8192}


def lmstudio(server, **kw) -> OpenAICompatClient:
    return OpenAICompatClient(server.base_url, **kw)


def watched(client):
    seen = []
    client.on_call = seen.append
    return seen


class LMStudioTest(unittest.TestCase):

    # -- happy -------------------------------------------------------------
    def test_vectors_for_every_text_from_the_resident_embedder(self):
        with FakeLMStudio() as server:
            client = lmstudio(server)
            seen = watched(client)
            result = client.embed(["an invoice", "a contract"])
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(EMBEDDER, result.model)
        self.assertEqual(8, result.dims)
        self.assertEqual([fake_vector("an invoice"),
                          fake_vector("a contract")], result.vectors)
        self.assertEqual([{"model": EMBEDDER,
                           "input": ["an invoice", "a contract"]}],
                         server.embed_requests)
        self.assertEqual([], server.evictions)
        usage = result.usage
        self.assertIsInstance(usage, CallUsage)
        self.assertEqual(("embed", "ok", "server", 4),
                         (usage.kind, usage.outcome, usage.source,
                          usage.prompt_tokens))
        self.assertIsNone(usage.completion_tokens)
        self.assertEqual([usage], seen)

    def test_the_vectors_follow_the_index_not_the_order(self):
        with FakeLMStudio(embed_shuffled=True) as server:
            result = lmstudio(server).embed(["one", "two", "three"])
        self.assertEqual([fake_vector(t) for t in ("one", "two", "three")],
                         result.vectors)

    def test_a_named_resident_embedder_is_used(self):
        models = DEFAULT_MODELS + [{"id": "mxbai-embed-large-v1",
                                    "type": "embeddings", "state": "loaded"}]
        with FakeLMStudio(models=models) as server:
            client = lmstudio(server, embedding_model="mxbai-embed-large-v1")
            result = client.embed(["x"])
        self.assertEqual("mxbai-embed-large-v1", result.model)

    def test_the_chat_model_is_not_disturbed(self):
        with FakeLMStudio(reply="TYPE: t\nSUMMARY: s") as server:
            client = lmstudio(server)
            client.embed(["x"])
            client.generate("system", "user")
        self.assertEqual(["qwen2.5-7b-instruct"], server.models_used)

    # -- sad ---------------------------------------------------------------
    def test_a_reranker_is_never_the_embedder(self):
        with FakeLMStudio(models=DEFAULT_MODELS + [RERANKER]) as server:
            result = lmstudio(server).embed(["x"])
        self.assertEqual(EMBEDDER, result.model)
        only = [m for m in DEFAULT_MODELS if m["type"] != "embeddings"]
        with FakeLMStudio(models=only + [RERANKER]) as server:
            with self.assertRaises(LLMError) as caught:
                lmstudio(server).embed(["x"])
        self.assertIn("embedding", str(caught.exception))
        self.assertEqual([], server.embed_requests)

    def test_a_cold_embedder_is_not_loaded_over_a_resident_model(self):
        models = [dict(m, state="not-loaded") if m["type"] == "embeddings"
                  else m for m in DEFAULT_MODELS]
        with FakeLMStudio(models=models) as server:
            with self.assertRaises(LLMError) as caught:
                lmstudio(server).embed(["x"])
            with self.assertRaises(LLMError):
                lmstudio(server, embedding_model=EMBEDDER).embed(["x"])
        self.assertEqual([], server.evictions)
        self.assertEqual([], server.embed_requests)
        self.assertIn(EMBEDDER, str(caught.exception))
        self.assertIn("allow-model-load", str(caught.exception))

    def test_allow_model_load_is_the_one_way_to_load_it(self):
        models = [dict(m, state="not-loaded") if m["type"] == "embeddings"
                  else m for m in DEFAULT_MODELS]
        with FakeLMStudio(models=models) as server:
            with self.assertRaises(LLMError):
                lmstudio(server, allow_model_load=True).embed(["x"])
        # The fake refuses the load it records; a real server would load.
        self.assertEqual([EMBEDDER], server.evictions)

    def test_an_idle_server_may_load_the_embedder(self):
        idle = [dict(m, state="not-loaded") for m in DEFAULT_MODELS]
        with FakeLMStudio(models=idle) as server:
            with self.assertRaises(LLMError):
                lmstudio(server).embed(["x"])
        # Nothing was resident, so nothing could be evicted: sift asked.
        self.assertEqual([EMBEDDER], server.evictions)

    def test_the_arguments_are_checked_before_anything_is_sent(self):
        with FakeLMStudio() as server:
            client = lmstudio(server)
            for bad in ("one text", ["ok", ""], ["ok", "   "], ["ok", 3],
                        None):
                with self.subTest(bad):
                    with self.assertRaises(ValueError):
                        client.embed(bad)
            self.assertEqual([], server.embed_requests)

    def test_without_usage_the_tokens_are_estimated(self):
        with FakeLMStudio(usage=None) as server:
            result = lmstudio(server).embed(["an invoice from ACME"])
        self.assertEqual("estimated", result.usage.source)
        self.assertEqual(estimate_tokens("an invoice from ACME"),
                         result.usage.prompt_tokens)

    def test_a_server_that_counts_nothing_has_not_counted(self):
        # LM Studio answers "prompt_tokens": 0 for texts it did read; a
        # non-blank text is at least one token, so 0 is no count at all.
        with FakeLMStudio(embed_usage_zero=True) as server:
            result = lmstudio(server).embed(["an invoice from ACME"])
        self.assertEqual(("estimated", estimate_tokens("an invoice from ACME")),
                         (result.usage.source, result.usage.prompt_tokens))

    # -- edge --------------------------------------------------------------
    def test_nothing_to_embed_asks_nothing(self):
        with FakeLMStudio() as server:
            result = lmstudio(server).embed([])
        self.assertEqual(([], 0, None), (result.vectors, result.dims,
                                         result.usage))
        self.assertEqual([], server.embed_requests)

    def test_a_single_text_in_a_tuple(self):
        with FakeLMStudio(embed_dims=768) as server:
            result = lmstudio(server).embed(("ä" * 3,))
        self.assertEqual(768, result.dims)
        self.assertEqual([fake_vector("äää", 768)], result.vectors)


class OllamaTest(unittest.TestCase):

    def test_vectors_from_api_embed(self):
        with FakeOllama(resident=["llama3.1:latest",
                                  "nomic-embed-text:latest"]) as server:
            client = OllamaClient(server.base_url)
            seen = watched(client)
            result = client.embed(["an invoice", "a contract"])
        self.assertEqual("nomic-embed-text:latest", result.model)
        self.assertEqual([fake_vector("an invoice"),
                          fake_vector("a contract")], result.vectors)
        self.assertEqual([{"model": "nomic-embed-text:latest",
                           "input": ["an invoice", "a contract"]}],
                         server.embed_requests)
        self.assertEqual(("embed", "ok", "server", 4),
                         (result.usage.kind, result.usage.outcome,
                          result.usage.source, result.usage.prompt_tokens))
        self.assertEqual([result.usage], seen)

    def test_a_cold_embedder_is_refused(self):
        with FakeOllama() as server:            # only llama3.1 resident
            with self.assertRaises(LLMError) as caught:
                OllamaClient(server.base_url).embed(["x"])
        self.assertEqual([], server.evictions)
        self.assertEqual([], server.embed_requests)
        self.assertIn("nomic-embed-text:latest", str(caught.exception))

    def test_a_reranker_named_so_is_not_chosen(self):
        installed = [
            {"name": "llama3.1:latest", "details": {"families": ["llama"]}},
            {"name": "bge-reranker-v2-m3:latest",
             "details": {"families": ["bert"]}},
        ]
        with FakeOllama(installed=installed,
                        resident=["llama3.1:latest",
                                  "bge-reranker-v2-m3:latest"]) as server:
            with self.assertRaises(LLMError):
                OllamaClient(server.base_url).embed(["x"])
        self.assertEqual([], server.embed_requests)


class Scripted:
    """urlopen answering each POST with the next scripted body."""

    def __init__(self, *replies, models=None):
        self.replies = list(replies)
        self.models = models or {"object": "list", "data": [
            {"id": "embedder", "state": "loaded", "type": "embeddings"}]}

    def __call__(self, req, timeout=None):
        from test_llm_clients import FakeResp
        if "/api/v0/models" in req.full_url:
            return FakeResp(self.models)
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(req.full_url, reply, "no", {},
                                         io.BytesIO(b""))
        if isinstance(reply, str):
            return FakeResp(None, raw=reply)
        return FakeResp(reply)


def body(*vectors, usage=True):
    answer = {"object": "list", "model": "embedder",
              "data": [{"object": "embedding", "index": i, "embedding": v}
                       for i, v in enumerate(vectors)]}
    if usage:
        answer["usage"] = {"prompt_tokens": 3, "total_tokens": 3}
    return answer


class MalformedAnswerTest(unittest.TestCase):
    """A body that is not what was asked for is an LLMError that says
    what was wrong — never vectors that belong to the wrong text."""

    def embed(self, reply, texts=("a", "b")):
        client = OpenAICompatClient("http://x/v1")
        seen = watched(client)
        with mock.patch("urllib.request.urlopen", Scripted(reply)):
            try:
                return client.embed(list(texts)), seen
            except LLMError as exc:
                return exc, seen

    def assertRefused(self, reply, says, texts=("a", "b")):
        got, seen = self.embed(reply, texts)
        self.assertIsInstance(got, LLMError, got)
        self.assertIn(says, str(got))
        self.assertEqual(["error"], [c.outcome for c in seen])
        self.assertIn(says, seen[0].error)

    def test_the_wrong_number_of_vectors(self):
        self.assertRefused(body([1.0, 2.0]), "1 vector for 2 texts")

    def test_vectors_of_different_lengths(self):
        self.assertRefused(body([1.0, 2.0], [1.0]), "dimensions")

    def test_a_vector_that_is_not_numbers(self):
        for bad in (["x", 1.0], [True, 1.0], [], None, {"0": 1.0}):
            with self.subTest(bad):
                self.assertRefused(body([1.0, 2.0], bad), "vector")

    def test_a_vector_with_nan_or_infinity(self):
        # json.loads reads NaN and Infinity; a vector holding one poisons
        # every distance computed with it.
        raw = ('{"data": [{"index": 0, "embedding": [1.0, NaN]}, '
               '{"index": 1, "embedding": [1.0, Infinity]}]}')
        self.assertRefused(raw, "not a finite number")

    def test_indices_that_do_not_cover_the_texts(self):
        answer = body([1.0], [2.0])
        answer["data"][1]["index"] = 0
        self.assertRefused(answer, "index")
        answer = body([1.0], [2.0])
        answer["data"][1]["index"] = 5
        self.assertRefused(answer, "index")

    def test_no_data_at_all(self):
        self.assertRefused({"object": "list", "model": "embedder"}, "data")

    def test_an_error_in_a_successful_response(self):
        self.assertRefused({"error": "model is not an embedding model"},
                           "not an embedding model")

    def test_a_server_error(self):
        got, seen = self.embed(500)
        self.assertIsInstance(got, LLMError)
        self.assertEqual(["error"], [c.outcome for c in seen])
        self.assertIsNone(seen[0].prompt_tokens)

    def test_a_well_formed_answer_passes(self):
        got, seen = self.embed(body([1, 2.5], [0, -1]))
        self.assertIsInstance(got, EmbeddingResult)
        self.assertEqual([[1.0, 2.5], [0.0, -1.0]], got.vectors)
        self.assertTrue(all(isinstance(x, float)
                            for v in got.vectors for x in v))
        self.assertEqual(["ok"], [c.outcome for c in seen])


class RerankerNameTest(unittest.TestCase):

    def test_rerankers_by_name(self):
        for name in ("bge-reranker-v2-m3", "jina-reranker-v1-turbo-en",
                     "BAAI/bge-reranker-large", "mxbai-rerank-large-v1",
                     "ms-marco-MiniLM-L-6-v2-cross-encoder"):
            with self.subTest(name):
                self.assertTrue(is_reranker(name))
        for name in ("nomic-embed-text", "bge-m3", "mxbai-embed-large",
                     "text-embedding-nomic-embed-text-v1.5", "all-minilm",
                     "qwen2.5-7b-instruct", ""):
            with self.subTest(name):
                self.assertFalse(is_reranker(name))


class EmbedInputTest(unittest.TestCase):

    SUMMARY = {"llm_used": True, "classification": "invoice",
               "summary": "An invoice from ACME for 1,200 EUR.",
               "key_points": ["due 2026-10-01", "net 30"],
               "tags": ["invoice", "finance"],
               "tags_suspect": ["approved-for-payment"]}

    # -- happy -------------------------------------------------------------
    def test_with_a_model_the_summary_key_points_and_tags(self):
        said = embed_input(self.SUMMARY, "the file text, never used here")
        for part in ("An invoice from ACME for 1,200 EUR.", "due 2026-10-01",
                     "net 30", "invoice, finance"):
            self.assertIn(part, said)
        self.assertNotIn("the file text", said)
        self.assertNotIn("approved-for-payment", said)

    def test_without_a_model_the_head_of_the_text(self):
        heuristic = dict(self.SUMMARY, llm_used=False)
        said = embed_input(heuristic, "Rechnung 4711 der ACME GmbH.\n" * 3)
        self.assertTrue(said.startswith("Rechnung 4711 der ACME GmbH."))
        self.assertNotIn("An invoice from ACME", said)

    def test_a_summary_object_works_like_its_dict(self):
        from sift.summarize import Summary
        summ = Summary(path="p", filename="f", size_bytes=1, size_human="1 B",
                       kind="text", mime="text/plain",
                       detection_confidence=1.0, language=None,
                       classification="invoice", summary="An invoice.",
                       llm_used=True, key_points=["due soon"],
                       tags=["invoice"])
        self.assertEqual(embed_input(summ.to_dict(), None),
                         embed_input(summ, None))
        self.assertIn("due soon", embed_input(summ, None))

    # -- sad ---------------------------------------------------------------
    def test_without_a_model_and_without_text_the_summary_stands_in(self):
        heuristic = dict(self.SUMMARY, llm_used=False)
        for text in (None, "", "   \n"):
            with self.subTest(text):
                self.assertIn("An invoice from ACME",
                              embed_input(heuristic, text))

    def test_a_summary_that_is_not_one_is_refused(self):
        for bad in (None, "text", ["summary"]):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    embed_input(bad, "text")
        with self.assertRaises(ValueError):
            embed_input(self.SUMMARY, "text", context_tokens=0)

    def test_the_head_is_never_cut_through_a_secret(self):
        """The head is cut before it is scrubbed, and a piece of a token is
        a shape the patterns no longer know. Wherever the cut falls — in a
        line, in a text without line breaks, in one without blanks — no
        piece of the token comes out of it.

        On _head itself: through embed_input the fit cuts deeper than the
        head does (no text packs six characters into a token), so the cut
        is guarded twice, and this is the guard that holds on its own."""
        from sift.embeddings import _HEAD_CHARS_PER_TOKEN, _head
        token = GITHUB_TOKEN
        pieces = [token[i:i + 8] for i in range(0, len(token) - 8, 4)]
        budget = 20
        limit = budget * _HEAD_CHARS_PER_TOKEN
        leaked, kept = [], 0
        for filler, before in (("Zeile mit Text\n", "\n"), ("wort ", " "),
                               ("指令", "")):
            for length in range(limit - len(token) - 4, limit + 2):
                text = (filler * 100)[:length] + before + token + " end"
                said = _head(text, budget)
                kept += bool(said)
                leaked += [(filler, length, p) for p in pieces if p in said]
        self.assertEqual([], leaked)
        self.assertGreater(kept, 100)       # not passing by keeping nothing

    def test_the_head_is_read_wide_enough_for_the_fit(self):
        from sift import chunk, embeddings
        widest = max(chunk._CHARS_PER_LETTER_TOKEN,
                     chunk._CHARS_PER_DIGIT_TOKEN,
                     chunk._CHARS_PER_SYMBOL_TOKEN,
                     chunk._CHARS_PER_SPACE_TOKEN)
        self.assertGreater(embeddings._HEAD_CHARS_PER_TOKEN, widest)

    def test_secret_shapes_in_the_text_are_not_embedded(self):
        token = GITHUB_TOKEN
        said = embed_input({"llm_used": False},
                           f"deploy notes\nGITHUB_TOKEN={token}\n")
        self.assertIn("deploy notes", said)
        self.assertNotIn(token, said)

    # -- edge --------------------------------------------------------------
    def test_everything_fits_the_embedding_models_context(self):
        long_text = "Ein Satz über Rechnungen und Verträge, Nummer 4711. " * 5000
        cjk = "指令和合同的摘要" * 20000
        long_summary = dict(self.SUMMARY,
                            summary="word " * 50000,
                            key_points=[f"point {i}" for i in range(3000)])
        for context in (64, 512, 8192):
            for summary, text in ((dict(self.SUMMARY, llm_used=False),
                                   long_text),
                                  (dict(self.SUMMARY, llm_used=False), cjk),
                                  (long_summary, None)):
                with self.subTest(context=context, text=(text or "")[:10]):
                    said = embed_input(summary, text, context_tokens=context)
                    self.assertTrue(said)
                    self.assertLessEqual(estimate_tokens(said), context)
                    if text:
                        self.assertEqual(text[:5], said[:5])
        short = embed_input(self.SUMMARY, None, context_tokens=8192)
        self.assertIn("net 30", short)       # nothing cut that fits

    def test_nothing_to_embed_is_the_empty_string(self):
        empty = {"llm_used": True, "summary": "", "key_points": [],
                 "tags": []}
        self.assertEqual("", embed_input(empty, None))
        self.assertEqual("", embed_input({"llm_used": False}, None))


if __name__ == "__main__":
    unittest.main()
