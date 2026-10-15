"""Transport-level tests for the LLM clients — mock urllib, assert the wire
format. This covers the one seam that cannot run live in the sandbox (no Ollama
or LM Studio server), verifying request payloads and response parsing.
"""
import io
import json
import unittest
import urllib.error
from unittest import mock

from sift.llm.base import LLMError, NoVisionModel
from sift.llm.ollama import OllamaClient
from sift.llm.openai_compat import OpenAICompatClient

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF"
GIF = b"GIF89a\x01\x00"
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 "


class FakeResp:
    """A urlopen result that can be read whole or iterated line by line."""

    def __init__(self, payload, raw: str = ""):
        self._b = raw.encode("utf-8") if raw else json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __iter__(self):
        return iter(self._b.splitlines(keepends=True))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sse(*chunks: str, done: bool = True) -> str:
    """OpenAI-style event stream carrying content deltas."""
    out = "".join(
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n\n"
        for c in chunks)
    return out + ("data: [DONE]\n\n" if done else "")


def ndjson(*chunks: str, done: bool = True) -> str:
    """Ollama-style newline-delimited stream."""
    out = "".join(json.dumps({"message": {"content": c}, "done": False}) + "\n"
                  for c in chunks)
    return out + (json.dumps({"message": {"content": ""}, "done": True}) + "\n"
                  if done else "")


def _streamed(payload: dict, is_ollama: bool) -> str:
    """Re-express a non-streaming route body as the stream the server would send.

    Lets every route in these tests stay written as the plain response shape
    while the client under test streams by default.
    """
    if is_ollama:
        return ndjson((payload.get("message") or {}).get("content", ""))
    choices = payload.get("choices") or []
    text = (choices[0].get("message") or {}).get("content", "") if choices else ""
    return sse(text)


def router(routes, captured):
    def _fake(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["method"] = req.get_method()
        wants_stream = bool(req.data) and b'"stream": true' in req.data
        # Longest key first: "/models" is a substring of "/api/v0/models", so
        # insertion order decided which route answered — and a fake that
        # served the LM Studio endpoint from a plain-OpenAI fixture is a fake
        # of a server that does not exist.
        for key in sorted(routes, key=len, reverse=True):
            payload = routes[key]
            if key in req.full_url:
                if isinstance(payload, int):      # a status, not a body
                    raise urllib.error.HTTPError(
                        req.full_url, payload, "no", {}, io.BytesIO(b""))
                if isinstance(payload, str):      # a raw stream body
                    return FakeResp(None, raw=payload)
                if wants_stream and key in ("/chat/completions", "/api/chat"):
                    return FakeResp(None,
                                    raw=_streamed(payload, key == "/api/chat"))
                return FakeResp(payload)
        return FakeResp({})
    return _fake


def patched(routes, captured):
    return mock.patch("sift.llm.base.urllib.request.urlopen",
                      router(routes, captured))


class OllamaClientTest(unittest.TestCase):
    def test_available_and_list_models(self):
        cap = {}
        routes = {"/api/tags": {"models": [{"name": "llama3.1"}, {"name": "llava"}]}}
        with patched(routes, cap):
            c = OllamaClient("http://localhost:11434")
            self.assertTrue(c.available())
            self.assertEqual(c.list_models(), ["llama3.1", "llava"])
            self.assertEqual(cap["method"], "GET")

    def test_generate_payload_and_parse(self):
        cap = {}
        routes = {"/api/chat": {"message": {"content": "TYPE: x\nSUMMARY: y"}}}
        with patched(routes, cap):
            c = OllamaClient("http://localhost:11434", model="llama3.1")
            r = c.generate("sys", "hello", temperature=0.1, max_tokens=42)
        self.assertTrue(r.text.startswith("TYPE: x"))
        self.assertEqual(r.model, "llama3.1")
        body = json.loads(cap["data"])
        self.assertIn("/api/chat", cap["url"])
        self.assertEqual(body["model"], "llama3.1")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["options"]["num_predict"], 42)
        self.assertTrue(body["stream"], "streaming is the default; a silent "
                                        "connection gets closed by proxies")

    def test_vision_images_base64(self):
        cap = {}
        routes = {"/api/chat": {"message": {"content": "ok"}}}
        with patched(routes, cap):
            c = OllamaClient("http://localhost:11434", model="llava")
            c.generate("sys", "describe", images=[b"\x89PNGdata"])
        body = json.loads(cap["data"])
        self.assertIn("images", body["messages"][1])
        self.assertTrue(body["messages"][1]["images"][0])  # non-empty b64

    def test_auto_model_skips_embeddings(self):
        cap = {}
        routes = {"/api/tags": {"models": [{"name": "nomic-embed-text"},
                                           {"name": "qwen2.5"}]},
                  "/api/chat": {"message": {"content": "ok"}}}
        with patched(routes, cap):
            c = OllamaClient("http://localhost:11434")  # no model -> auto
            c.generate("sys", "u")
        self.assertEqual(json.loads(cap["data"])["model"], "qwen2.5")

    def test_urlerror_becomes_llmerror(self):
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")
        with mock.patch("sift.llm.base.urllib.request.urlopen", boom):
            c = OllamaClient("http://localhost:11434", model="m")
            with self.assertRaises(LLMError):
                c.generate("s", "u")


class OpenAICompatClientTest(unittest.TestCase):
    def test_mlx_lm_provider_uses_the_openai_compatible_client(self):
        from sift.config import Config
        from sift.llm import build_client

        client = build_client(Config(
            provider="mlx-lm", endpoint="http://203.0.113.10:8081"))
        self.assertIsInstance(client, OpenAICompatClient)
        self.assertEqual(client.provider, "mlx-lm")
        self.assertEqual(client.endpoint, "http://203.0.113.10:8081/v1")

    def test_auto_detection_reaches_the_default_mlx_lm_server(self):
        from sift.config import Config, DEFAULT_MLX_LM_ENDPOINT
        from sift.llm import build_client

        def openai_available(client):
            return client.endpoint == DEFAULT_MLX_LM_ENDPOINT

        with mock.patch.object(OllamaClient, "available", return_value=False), \
             mock.patch.object(OpenAICompatClient, "available",
                               autospec=True,
                               side_effect=openai_available) as available:
            client = build_client(Config())

        self.assertEqual(client.provider, "mlx-lm")
        self.assertEqual(client.endpoint, DEFAULT_MLX_LM_ENDPOINT)
        self.assertEqual(available.call_count, 2)

    def test_auto_detection_adds_v1_to_a_custom_openai_server(self):
        from sift.config import Config
        from sift.llm import build_client

        wanted = "http://203.0.113.10:8081/v1"

        def openai_available(client):
            return client.endpoint == wanted

        with mock.patch.object(OllamaClient, "available", return_value=False), \
             mock.patch.object(OpenAICompatClient, "available",
                               autospec=True,
                               side_effect=openai_available):
            client = build_client(Config(
                endpoint="http://203.0.113.10:8081"))

        self.assertEqual(client.endpoint, wanted)

    def test_generate_payload_and_parse(self):
        cap = {}
        routes = {"/chat/completions": {"choices": [{"message": {"content": "hi"}}]}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://localhost:1234/v1", model="qwen2.5",
                                   provider="lmstudio")
            r = c.generate("sys", "hello")
        self.assertEqual(r.text, "hi")
        body = json.loads(cap["data"])
        self.assertIn("/chat/completions", cap["url"])
        self.assertEqual(body["model"], "qwen2.5")
        self.assertEqual(body["max_tokens"], 800)

    def test_vision_content_parts(self):
        cap = {}
        routes = {"/chat/completions": {"choices": [{"message": {"content": "x"}}]}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://localhost:1234/v1", model="vlm")
            c.generate("sys", "describe", images=[PNG])
        content = json.loads(cap["data"])["messages"][1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_data_uri_states_the_real_format(self):
        """A JPEG must not be announced as PNG — servers decode by this label."""
        for blob, mime in ((PNG, "image/png"), (JPEG, "image/jpeg"),
                           (GIF, "image/gif"), (WEBP, "image/webp")):
            cap = {}
            routes = {"/chat/completions": {"choices": [{"message": {"content": "x"}}]}}
            with patched(routes, cap):
                OpenAICompatClient("http://localhost:1234/v1",
                                   model="vlm").generate("s", "u", images=[blob])
            url = json.loads(cap["data"])["messages"][1]["content"][1]["image_url"]["url"]
            self.assertTrue(url.startswith(f"data:{mime};base64,"),
                            f"{mime} was announced as {url[:30]!r}")

    def test_list_models(self):
        cap = {}
        routes = {"/models": {"data": [{"id": "qwen2.5"}, {"id": "phi3"}]}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://localhost:1234/v1")
            self.assertEqual(c.list_models(), ["qwen2.5", "phi3"])


# ---------------------------------------------------------------------------
# Model residency. The operator's rule: use what is loaded, never evict it.
# Getting this wrong is invisible in the output — the summary looks fine while
# the user's model has been swapped out underneath them — so it is tested from
# every side.
# ---------------------------------------------------------------------------
def v0(*models) -> dict:
    """An /api/v0/models body from (id, state, type) triples."""
    # loaded_context_length is what LM Studio reports for a model that is
    # actually resident, and it is the only number sift may size a prompt
    # from. Emitting max_context_length alone made every test exercise the
    # fallback branch and none the real one — so a change that dropped the
    # correct field entirely still passed.
    return {"object": "list", "data": [
        {"id": i, "state": s, "type": t, "max_context_length": 32768,
         **({"loaded_context_length": 4096} if s == "loaded" else {})}
        for i, s, t in models]}


def ps(*names) -> dict:
    return {"models": [{"name": n, "model": n} for n in names]}


def tags(*names) -> dict:
    return {"models": [{"name": n, "model": n, "details": {}} for n in names]}


LOADED_CHAT = "qwen2.5-7b-instruct"
COLD_CHAT = "mistral-nemo"
LOADED_VLM = "llava-v1.6"
LOADED_EMBED = "text-embedding-nomic"


class LMStudioResidencyTest(unittest.TestCase):
    def _client(self, body, **kw):
        cap = {}
        routes = {"/api/v0/models": body,
                  "/chat/completions": {"choices": [{"message": {"content": "ok"}}]}}
        return OpenAICompatClient("http://localhost:1234/v1", **kw), routes, cap

    def test_inventory_reads_state_and_kind(self):
        c, routes, cap = self._client(v0((LOADED_CHAT, "loaded", "llm"),
                                         (COLD_CHAT, "not-loaded", "llm"),
                                         (LOADED_EMBED, "loaded", "embeddings")))
        with patched(routes, cap):
            inv = {m.id: m for m in c.inventory()}
        self.assertTrue(inv[LOADED_CHAT].loaded)
        self.assertFalse(inv[COLD_CHAT].loaded)
        self.assertEqual(inv[LOADED_EMBED].kind, "embeddings")
        self.assertEqual(inv[LOADED_CHAT].context, 4096)

    def test_picks_the_loaded_model_not_the_first_listed(self):
        """The cold model is listed first — order must not decide this."""
        c, routes, cap = self._client(v0((COLD_CHAT, "not-loaded", "llm"),
                                         (LOADED_CHAT, "loaded", "llm")))
        with patched(routes, cap):
            c.generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], LOADED_CHAT)

    def test_never_picks_a_loaded_embedding_model(self):
        c, routes, cap = self._client(v0((LOADED_EMBED, "loaded", "embeddings"),
                                         (LOADED_CHAT, "loaded", "llm")))
        with patched(routes, cap):
            c.generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], LOADED_CHAT)

    def test_prefers_a_plain_chat_model_over_a_vision_one_for_text(self):
        c, routes, cap = self._client(v0((LOADED_VLM, "loaded", "vlm"),
                                         (LOADED_CHAT, "loaded", "llm")))
        with patched(routes, cap):
            c.generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], LOADED_CHAT)

    def test_explicit_cold_model_is_refused_while_another_is_loaded(self):
        c, routes, cap = self._client(v0((COLD_CHAT, "not-loaded", "llm"),
                                         (LOADED_CHAT, "loaded", "llm")),
                                      model=COLD_CHAT)
        with patched(routes, cap):
            with self.assertRaises(LLMError) as ctx:
                c.generate("s", "u")
        self.assertIn(LOADED_CHAT, str(ctx.exception))     # names the alternative
        self.assertNotIn("/chat/completions", cap.get("url", ""))  # never sent

    def test_allow_model_load_permits_the_swap(self):
        c, routes, cap = self._client(v0((COLD_CHAT, "not-loaded", "llm"),
                                         (LOADED_CHAT, "loaded", "llm")),
                                      model=COLD_CHAT, allow_model_load=True)
        with patched(routes, cap):
            c.generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], COLD_CHAT)

    def test_idle_server_may_load_since_nothing_is_evicted(self):
        """Refusing here would protect nothing and only produce a worse summary."""
        c, routes, cap = self._client(v0((COLD_CHAT, "not-loaded", "llm")))
        with patched(routes, cap):
            c.generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], COLD_CHAT)

    def test_images_need_a_loaded_vision_model(self):
        c, routes, cap = self._client(v0((LOADED_CHAT, "loaded", "llm"),
                                         (LOADED_VLM, "not-loaded", "vlm")))
        with patched(routes, cap):
            with self.assertRaises(NoVisionModel):
                c.generate("s", "u", images=[PNG])

    def test_images_use_the_loaded_vision_model(self):
        c, routes, cap = self._client(v0((LOADED_CHAT, "loaded", "llm"),
                                         (LOADED_VLM, "loaded", "vlm")))
        with patched(routes, cap):
            c.generate("s", "u", images=[PNG])
        self.assertEqual(json.loads(cap["data"])["model"], LOADED_VLM)

    def test_text_first_then_image_still_reaches_the_vision_model(self):
        """Regression: the text pick must not be reused for the picture.

        Summarizing a folder means text files first, an image later. Caching
        one model for the whole client sent the image to the text model, which
        answers about the filename and never says it could not see anything.
        """
        c, routes, cap = self._client(v0((LOADED_CHAT, "loaded", "llm"),
                                         (LOADED_VLM, "loaded", "vlm")))
        with patched(routes, cap):
            c.generate("s", "a text file")
            self.assertEqual(json.loads(cap["data"])["model"], LOADED_CHAT)
            c.generate("s", "a picture", images=[PNG])
        self.assertEqual(json.loads(cap["data"])["model"], LOADED_VLM)

    def test_plain_openai_server_without_api_v0_still_works(self):
        """llama.cpp/vLLM have no /api/v0 — unknown state must not mean 'cold'."""
        cap = {}
        routes = {"/api/v0/models": 404,   # llama.cpp really does 404 here
                  "/models": {"data": [{"id": "some-model"}]},
                  "/chat/completions": {"choices": [{"message": {"content": "ok"}}]}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://localhost:8080/v1")
            c.generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], "some-model")

    def test_server_root_strips_the_v1_suffix(self):
        c = OpenAICompatClient("http://host:1234/v1")
        self.assertEqual(c.server_root(), "http://host:1234")
        self.assertEqual(
            OpenAICompatClient("http://host:1234/").server_root(),
            "http://host:1234")

    def test_inventory_is_fetched_once_per_client(self):
        """N files must not mean N model-list round trips."""
        calls = []
        body = v0((LOADED_CHAT, "loaded", "llm"))

        def counting(req, timeout=None):
            calls.append(req.full_url)
            if "/api/v0/models" in req.full_url:
                return FakeResp(body)
            return FakeResp(None, raw=sse("ok"))

        with mock.patch("sift.llm.base.urllib.request.urlopen", counting):
            c = OpenAICompatClient("http://localhost:1234/v1")
            for _ in range(5):
                c.generate("s", "u")
        self.assertEqual(sum("/api/v0/models" in u for u in calls), 1)


class OllamaResidencyTest(unittest.TestCase):
    def test_api_ps_marks_resident_models(self):
        cap = {}
        routes = {"/api/tags": tags(COLD_CHAT, LOADED_CHAT),
                  "/api/ps": ps(LOADED_CHAT),
                  "/api/chat": {"message": {"content": "ok"}}}
        with patched(routes, cap):
            OllamaClient("http://localhost:11434").generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], LOADED_CHAT)

    def test_idle_ollama_may_load(self):
        cap = {}
        routes = {"/api/tags": tags(COLD_CHAT),
                  "/api/ps": {"models": []},
                  "/api/chat": {"message": {"content": "ok"}}}
        with patched(routes, cap):
            OllamaClient("http://localhost:11434").generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], COLD_CHAT)

    def test_vision_family_detected_from_details(self):
        cap = {}
        routes = {"/api/tags": {"models": [
                      {"name": "gemma3:12b", "details": {"families": ["gemma3"]}}]},
                  "/api/ps": ps("gemma3:12b"),
                  "/api/chat": {"message": {"content": "ok"}}}
        with patched(routes, cap):
            c = OllamaClient("http://localhost:11434")
            self.assertEqual(c.inventory()[0].kind, "vlm")
            c.generate("s", "u", images=[PNG])
        self.assertIn("images", json.loads(cap["data"])["messages"][1])


# ---------------------------------------------------------------------------
# Streaming. Measured against this cube's proxy: a non-streaming completion is
# closed at 20.0 s while nothing flows, the same request streamed finishes at
# 46.7 s. So streaming is the default, and its parsing is load-bearing.
# ---------------------------------------------------------------------------
class StreamingTest(unittest.TestCase):
    def _lmstudio(self, raw: str, **kw):
        cap = {}
        routes = {"/api/v0/models": v0((LOADED_CHAT, "loaded", "llm")),
                  "/chat/completions": raw}
        return OpenAICompatClient("http://h:1234/v1", **kw), routes, cap

    def test_deltas_are_concatenated_in_order(self):
        c, routes, cap = self._lmstudio(sse("TYPE: csv\n", "SUMMARY: ", "eight rows"))
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text,
                             "TYPE: csv\nSUMMARY: eight rows")

    def test_reasoning_content_is_not_part_of_the_answer(self):
        """A thinking model streams its scratchpad beside the answer.

        Real shape from qwen3.6-40b via LM Studio: reasoning_content arrives
        first, content afterwards. Concatenating both would print the model's
        deliberation as the file summary.
        """
        raw = "".join(
            "data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n"
            for d in ({"role": "assistant", "reasoning_content": "The user "},
                      {"reasoning_content": "wants a summary."},
                      {"content": "SUMMARY: eight employees"})
        ) + "data: [DONE]\n\n"
        c, routes, cap = self._lmstudio(raw)
        with patched(routes, cap):
            text = c.generate("s", "u").text
        self.assertEqual(text, "SUMMARY: eight employees")
        self.assertNotIn("user", text)

    def test_inline_think_tags_are_stripped(self):
        """llama.cpp and Ollama inline the reasoning instead of separating it."""
        c, routes, cap = self._lmstudio(
            sse("<think>Let me count the rows.</think>", "\nSUMMARY: ok"))
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text, "SUMMARY: ok")

    def test_unclosed_think_tag_yields_no_false_answer(self):
        c, routes, cap = self._lmstudio(sse("SUMMARY: ok\n", "<think>hmm, but"))
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text, "SUMMARY: ok")

    def test_done_marker_ends_the_stream(self):
        c, routes, cap = self._lmstudio(sse("kept") + sse("after done"))
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text, "kept")

    def test_keepalive_comments_and_blank_lines_are_ignored(self):
        raw = ": ping\n\n" + sse("a", "b")
        c, routes, cap = self._lmstudio(raw)
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text, "ab")

    def test_a_corrupt_frame_does_not_lose_the_rest(self):
        raw = "data: {not json\n\n" + sse("still here")
        c, routes, cap = self._lmstudio(raw)
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text, "still here")

    def test_error_event_mid_stream_is_raised(self):
        raw = ('data: {"error": {"message": "context overflow"}}\n\n')
        c, routes, cap = self._lmstudio(raw)
        with patched(routes, cap):
            with self.assertRaises(LLMError) as ctx:
                c.generate("s", "u")
        self.assertIn("context overflow", str(ctx.exception))

    def test_empty_stream_is_an_error_not_an_empty_summary(self):
        """Silence must not be reported as a successful empty summary."""
        c, routes, cap = self._lmstudio("")
        with patched(routes, cap):
            with self.assertRaises(LLMError):
                c.generate("s", "u")

    def test_reasoning_only_answer_is_an_error(self):
        raw = ("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}) +
            "\n\ndata: [DONE]\n\n")
        c, routes, cap = self._lmstudio(raw)
        with patched(routes, cap):
            with self.assertRaises(LLMError) as ctx:
                c.generate("s", "u")
        self.assertIn("reasoning", str(ctx.exception))

    def test_no_stream_falls_back_to_a_single_response(self):
        cap = {}
        routes = {"/api/v0/models": v0((LOADED_CHAT, "loaded", "llm")),
                  "/chat/completions": {"choices": [{"message": {"content": "hi"}}]}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://h:1234/v1", stream=False)
            self.assertEqual(c.generate("s", "u").text, "hi")
        self.assertFalse(json.loads(cap["data"])["stream"])

    def test_ollama_ndjson_stream(self):
        cap = {}
        routes = {"/api/tags": tags(LOADED_CHAT), "/api/ps": ps(LOADED_CHAT),
                  "/api/chat": ndjson("SUM", "MARY: ok")}
        with patched(routes, cap):
            c = OllamaClient("http://h:11434")
            self.assertEqual(c.generate("s", "u").text, "SUMMARY: ok")

    def test_ollama_stream_error_is_raised(self):
        cap = {}
        routes = {"/api/tags": tags(LOADED_CHAT), "/api/ps": ps(LOADED_CHAT),
                  "/api/chat": json.dumps({"error": "model not found"}) + "\n"}
        with patched(routes, cap):
            with self.assertRaises(LLMError) as ctx:
                OllamaClient("http://h:11434").generate("s", "u")
        self.assertIn("model not found", str(ctx.exception))


class EmptyErrorBodyTest(unittest.TestCase):
    """A refusal with nothing in it has to name where the fault probably is.

    Observed against this project's own proxy: it answers 403 with zero
    bytes. The message read "lmstudio HTTP 403 at /models: " — a dangling
    colon and no information — and the natural next move for whoever read it
    was to go and check the model server, which was the one place the fault
    was not.
    """

    def _error(self, code, body=b""):
        import io
        import urllib.error
        from unittest import mock
        from sift.config import Config
        from sift.llm import build_client
        from sift.llm.base import LLMError

        client = build_client(Config(provider="lmstudio",
                                     endpoint="http://h:1234/v1"))

        def raise_it(req, *a, **kw):
            raise urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body))

        with mock.patch("urllib.request.urlopen", raise_it):
            with self.assertRaises(LLMError) as caught:
                client.list_models()
        return str(caught.exception)

    def test_a_bare_403_points_at_the_middlebox(self):
        message = self._error(403)
        self.assertIn("proxy", message)
        self.assertNotIn(":  ", message)
        self.assertFalse(message.rstrip().endswith(":"), message)

    def test_a_bare_407_says_authentication(self):
        self.assertIn("authentication", self._error(407))

    def test_a_status_with_no_such_cause_just_says_so(self):
        message = self._error(404)
        self.assertIn("no message in the body", message)
        self.assertNotIn("proxy", message)

    def test_a_body_that_says_why_is_preferred_over_the_guess(self):
        message = self._error(403, b'{"error": "api key invalid"}')
        self.assertIn("api key invalid", message)
        self.assertNotIn("proxy", message)


class TwelfthAuditLLMTest(unittest.TestCase):
    """The residency rule, and an answer that stops without saying so."""

    def _client(self, routes):
        cap = {}
        return OpenAICompatClient("http://h:1234/v1", provider="lmstudio"), routes, cap

    def test_a_stream_that_just_stops_is_not_a_complete_answer(self):
        """The worst failure this tool can have: half a summary, presented
        whole, cached forever.

        A connection closed cleanly mid-stream — which is what a proxy does
        on an idle timeout, and the one in front of this machine has a
        20-second one — sent no [DONE] and no finish_reason. `saw_data` was
        set by the first content frame and never distinguished "the server
        finished" from "the bytes stopped", so an invoice summary missing its
        total and the word UNPAID came back with truncated=False.
        """
        body = ('data: {"choices": [{"delta": {"content": '
                '"TYPE: invoice. SUMMARY: total due is"}}]}\n\n')
        c, routes, cap = self._client({
            "/api/v0/models": v0((LOADED_CHAT, "loaded", "llm")),
            "/chat/completions": body})
        with patched(routes, cap):
            result = c.generate("s", "u")
        self.assertTrue(result.truncated)
        self.assertEqual(result.stop_reason, "cut")

    def test_a_stream_that_ends_properly_is_complete(self):
        """Both endings, because a check that fires on everything is not one."""
        for ending, label in (("data: [DONE]\n\n", "[DONE]"),
                              ('data: {"choices": [{"delta": {}, '
                               '"finish_reason": "stop"}]}\n\n', "finish_reason")):
            body = ('data: {"choices": [{"delta": {"content": '
                    '"TYPE: csv. SUMMARY: fine."}}]}\n\n') + ending
            c, routes, cap = self._client({
                "/api/v0/models": v0((LOADED_CHAT, "loaded", "llm")),
                "/chat/completions": body})
            with patched(routes, cap):
                result = c.generate("s", "u")
            self.assertFalse(result.truncated, label)
            self.assertEqual(result.stop_reason, "", label)

    def test_a_hiccup_on_the_model_list_is_not_permission_to_evict(self):
        """inventory() re-raises on a 500 so a hiccup cannot become
        permission to evict; ensure_model caught that very exception eleven
        lines below and returned the model anyway. One 500 and an explicit
        --model naming a cold model evicted the resident one."""
        from sift.llm.base import _Refusal

        c, routes, cap = self._client({"/api/v0/models": 500,
                                       "/chat/completions": {"choices": []}})
        c.model = COLD_CHAT
        with patched(routes, cap):
            with self.assertRaises(_Refusal):
                c.generate("s", "u")
        self.assertIsNone(cap.get("data"), "a request was sent anyway")

    def test_a_server_that_answers_but_will_not_say_is_not_an_idle_server(self):
        """ModelInfo.loaded has three states and _may_load collapsed None
        into "yes, load". An LM Studio build that omits `state` was read as
        idle, and _pick's context tiebreak then chose the largest model on
        the box — the one least likely to be resident."""
        silent = {"object": "list", "data": [
            {"id": LOADED_CHAT, "type": "llm", "max_context_length": 32768},
            {"id": COLD_CHAT, "type": "llm", "max_context_length": 1024000}]}
        c, routes, cap = self._client({"/api/v0/models": silent,
                                       "/chat/completions": {"choices": []}})
        with patched(routes, cap):
            with self.assertRaises(LLMError) as caught:
                c.generate("s", "u")
        self.assertIn("did not report which models are loaded",
                      str(caught.exception))
        self.assertIsNone(cap.get("data"), "a request was sent anyway")

    def test_a_server_with_no_such_endpoint_is_still_usable(self):
        """The other side of the same rule. llama.cpp and vLLM have no
        residency endpoint at all; refusing there would cost the user every
        summary to protect a model that does not exist."""
        cap = {}
        routes = {"/api/v0/models": 404,
                  "/models": {"data": [{"id": "some-model"}]},
                  "/chat/completions": {"choices": [{"message": {"content": "ok"}}]}}
        with patched(routes, cap):
            OpenAICompatClient("http://h:8080/v1").generate("s", "u")
        self.assertEqual(json.loads(cap["data"])["model"], "some-model")

    def test_the_window_is_the_loaded_one_or_nothing(self):
        """max_context_length is a property of a model that is *downloaded*.
        Sizing a prompt from it overflows a model loaded smaller — the file's
        own comment says so, and the `or` fallback did it anyway whenever the
        real number was absent, then reported the fiction to the user as
        "reported by the server for this model"."""
        with_loaded = v0((LOADED_CHAT, "loaded", "llm"))
        without = {"object": "list", "data": [
            {"id": LOADED_CHAT, "state": "loaded", "type": "llm",
             "max_context_length": 1024000}]}
        cap = {}
        with patched({"/api/v0/models": with_loaded}, cap):
            c = OpenAICompatClient("http://h:1234/v1", provider="lmstudio")
            self.assertEqual(c.inventory()[0].context, 4096)
        with patched({"/api/v0/models": without}, cap):
            c = OpenAICompatClient("http://h:1234/v1", provider="lmstudio")
            model = c.inventory()[0]
            self.assertIsNone(model.context)
            self.assertEqual(model.max_context, 1024000)

    def test_an_unexpected_shape_does_not_end_the_run(self):
        """A 200 whose body is valid JSON of the wrong shape. Three of these
        raised AttributeError out of summarize_path, which promises never to
        raise — and the CLI loops on that promise, so one such response
        discarded every file already summarized."""
        shapes = [
            ('data: {"choices": "backend crashed"}\n\ndata: [DONE]\n\n',
             "choices is a string"),
            ('data: [1, 2, 3]\n\ndata: [DONE]\n\n', "event is a list"),
        ]
        for body, label in shapes:
            c, routes, cap = self._client({
                "/api/v0/models": v0((LOADED_CHAT, "loaded", "llm")),
                "/chat/completions": body})
            with patched(routes, cap):
                with self.assertRaises(LLMError, msg=label):
                    c.generate("s", "u")

    def test_an_address_with_no_scheme_is_an_llm_error(self):
        """Request() itself raises ValueError, and it was built outside the
        try — so endpoint = "/v1", a plausible typo, ended the batch."""
        c = OpenAICompatClient("/v1", model="m", provider="lmstudio")
        with self.assertRaises(LLMError):
            c.list_models()


if __name__ == "__main__":
    unittest.main()


class HardeningTest(unittest.TestCase):
    """Each case here is a defect an adversarial audit reproduced."""

    def _lm(self, models_body, **kw):
        cap = {}
        routes = {"/api/v0/models": models_body,
                  "/chat/completions": sse("TYPE: x\nSUMMARY: y")}
        return OpenAICompatClient("http://h:1234/v1", **kw), routes, cap

    # -- residency ---------------------------------------------------
    def test_a_differently_spelled_model_id_is_still_refused(self):
        """LM Studio resolves ids case-insensitively; an exact-match check
        let --model QWEN... past the refusal and the server loaded it."""
        c, routes, cap = self._lm(v0((COLD_CHAT, "not-loaded", "llm"),
                                     (LOADED_CHAT, "loaded", "llm")),
                                  model=COLD_CHAT.upper())
        with patched(routes, cap):
            with self.assertRaises(LLMError):
                c.generate("s", "u")
        self.assertNotIn("/chat/completions", cap.get("url", ""))

    def test_ollamas_implicit_latest_tag_is_still_refused(self):
        cap = {}
        routes = {"/api/tags": tags("llama3:latest", "hot:latest"),
                  "/api/ps": ps("hot:latest"),
                  "/api/chat": ndjson("ok")}
        with patched(routes, cap):
            with self.assertRaises(LLMError):
                OllamaClient("http://h:11434", model="llama3").generate("s", "u")
        self.assertNotIn("/api/chat", cap.get("url", ""))

    def test_a_failed_residency_query_does_not_read_as_an_idle_server(self):
        """Falling back to names alone made every model 'unknown', which the
        idle-server rule then treated as permission to load."""
        def failing(req, timeout=None):
            if "/api/v0/models" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
            return FakeResp(None, raw=sse("ok"))
        with mock.patch("sift.llm.base.urllib.request.urlopen", failing):
            with self.assertRaises(LLMError):
                OpenAICompatClient("http://h:1234/v1").generate("s", "u")

    def test_a_server_without_api_v0_still_falls_back(self):
        """404 means 'not LM Studio', which is not the same as 'broken'."""
        def not_found(req, timeout=None):
            if "/api/v0/models" in req.full_url:
                raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, None)
            if "/models" in req.full_url:
                return FakeResp({"data": [{"id": "solo"}]})
            return FakeResp(None, raw=sse("ok"))
        with mock.patch("sift.llm.base.urllib.request.urlopen", not_found):
            self.assertEqual(
                OpenAICompatClient("http://h:1234/v1").generate("s", "u").text,
                "ok")

    def test_loading_once_does_not_license_loading_again(self):
        """The inventory is read once. Without marking the model we caused to
        load as resident, every file in a batch evicted the previous one."""
        c, routes, cap = self._lm(v0((COLD_CHAT, "not-loaded", "llm"),
                                     ("cold-vlm", "not-loaded", "vlm")))
        with patched(routes, cap):
            c.generate("s", "u")                       # idle server: may load
            self.assertEqual(json.loads(cap["data"])["model"], COLD_CHAT)
            with self.assertRaises(NoVisionModel):     # now something is loaded
                c.generate("s", "u", images=[PNG])

    # -- transport ---------------------------------------------------
    def test_a_stream_cut_mid_frame_raises_llmerror_not_incompleteread(self):
        """IncompleteRead is not an OSError; uncaught it took down the whole
        batch, including files that had already been summarized."""
        import http.client

        class Dying:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"part"}}]}\n'
                raise http.client.IncompleteRead(b"", 42)

        def dying(req, timeout=None):
            if "/api/v0/models" in req.full_url:
                return FakeResp(v0((LOADED_CHAT, "loaded", "llm")))
            return Dying()
        with mock.patch("sift.llm.base.urllib.request.urlopen", dying):
            with self.assertRaises(LLMError):
                OpenAICompatClient("http://h:1234/v1").generate("s", "u")

    def test_one_json_object_split_across_two_data_lines(self):
        raw = ('data: {"choices":[{"delta":\n'
               'data: {"content":"MULTILINE"}}]}\n\n' + sse("TAIL"))
        c, routes, cap = self._lm(v0((LOADED_CHAT, "loaded", "llm")))
        routes["/chat/completions"] = raw
        with patched(routes, cap):
            self.assertEqual(c.generate("s", "u").text, "MULTILINETAIL")

    def test_finish_reason_length_is_reported(self):
        raw = ('data: {"choices":[{"delta":{"content":"cut off mid-"},'
               '"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
        c, routes, cap = self._lm(v0((LOADED_CHAT, "loaded", "llm")))
        routes["/chat/completions"] = raw
        with patched(routes, cap):
            self.assertTrue(c.generate("s", "u").truncated)

    def test_a_non_dict_error_event_is_still_an_error(self):
        raw = 'data: {"error": ["overloaded"]}\n\n'
        c, routes, cap = self._lm(v0((LOADED_CHAT, "loaded", "llm")))
        routes["/chat/completions"] = raw
        with patched(routes, cap):
            with self.assertRaises(LLMError) as ctx:
                c.generate("s", "u")
        self.assertIn("overloaded", str(ctx.exception))

    # -- reasoning ---------------------------------------------------
    def test_a_mentioned_think_tag_does_not_truncate_the_answer(self):
        """Summarizing a file that talks about <think> tags is not thinking."""
        from sift.llm.base import strip_reasoning
        reply = ("TYPE: Vue component\n"
                 "SUMMARY: The template renders <think> blocks for the UI.\n"
                 "KEY POINTS:\n- uses <think> tags\nENTITIES: none")
        self.assertEqual(strip_reasoning(reply), reply)

    def test_an_unfinished_reasoning_block_is_dropped(self):
        from sift.llm.base import strip_reasoning
        self.assertEqual(
            strip_reasoning("SUMMARY: ok\n<think>hmm, but wait"), "SUMMARY: ok")


class ReasoningBudgetTest(unittest.TestCase):
    """A model that thinks past its budget gets one more chance.

    Eleven map-reduce parts in a row returned reasoning and no answer, five
    minutes of GPU time for nothing, because the map step capped output at 400
    tokens and the model spent all of them thinking.
    """

    def _routes(self, first_reply: str, second_reply: str):
        state = {"calls": 0}
        models = v0((LOADED_CHAT, "loaded", "llm"))

        def transport(req, timeout=None):
            if "/api/v0/models" in req.full_url:
                return FakeResp(models)
            state["calls"] += 1
            body = json.loads(req.data)
            state[f"max_tokens_{state['calls']}"] = body["max_tokens"]
            raw = first_reply if state["calls"] == 1 else second_reply
            return FakeResp(None, raw=raw)

        return transport, state

    def test_a_reasoning_only_reply_is_retried_with_more_room(self):
        thinking = ("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": "let me think" * 20}}]})
            + "\n\ndata: [DONE]\n\n")
        transport, state = self._routes(thinking, sse("SUMMARY: eight rows"))
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            result = OpenAICompatClient("http://h:1234/v1").generate(
                "s", "u", max_tokens=400)
        self.assertEqual(result.text, "SUMMARY: eight rows")
        self.assertEqual(state["max_tokens_1"], 400)
        self.assertEqual(state["max_tokens_2"], 1600)

    def test_an_answer_cut_off_after_thinking_is_asked_again(self):
        """The field case: 2,015 tokens of thought under a 2,000-token cap."""
        started = ("data: " + json.dumps({"choices": [
            {"delta": {"reasoning_content": "x" * 800}}]}) + "\n\n"
            + "data: " + json.dumps({"choices": [
                {"delta": {"content": "SUMMARY: eight ro"},
                 "finish_reason": "length"}]}) + "\n\ndata: [DONE]\n\n")
        transport, state = self._routes(started, sse("SUMMARY: eight rows"))
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            result = OpenAICompatClient("http://h:1234/v1").generate(
                "s", "u", max_tokens=400)
        self.assertEqual(result.text, "SUMMARY: eight rows")
        self.assertFalse(result.truncated)
        # 800 chars of thought ≈ 200 tokens, plus the 400 the answer needed.
        self.assertEqual(state["max_tokens_2"], 600)

    def test_a_long_answer_without_thinking_is_not_asked_again(self):
        """Re-asking would just produce another long answer."""
        long_cut = ("data: " + json.dumps({"choices": [
            {"delta": {"content": "SUMMARY: " + "word " * 50},
             "finish_reason": "length"}]}) + "\n\ndata: [DONE]\n\n")
        transport, state = self._routes(long_cut, sse("SUMMARY: short"))
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            result = OpenAICompatClient("http://h:1234/v1").generate(
                "s", "u", max_tokens=400)
        self.assertEqual(state["calls"], 1)
        self.assertTrue(result.truncated)

    def test_a_worse_second_answer_is_discarded(self):
        """A retry that stops even earlier is not an improvement."""
        first = ("data: " + json.dumps({"choices": [
            {"delta": {"reasoning_content": "y" * 800}}]}) + "\n\n"
            + "data: " + json.dumps({"choices": [
                {"delta": {"content": "SUMMARY: eight rows and six col"},
                 "finish_reason": "length"}]}) + "\n\ndata: [DONE]\n\n")
        worse = ("data: " + json.dumps({"choices": [
            {"delta": {"content": "SUM"}, "finish_reason": "length"}]})
            + "\n\ndata: [DONE]\n\n")
        transport, state = self._routes(first, worse)
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            result = OpenAICompatClient("http://h:1234/v1").generate(
                "s", "u", max_tokens=400)
        self.assertEqual(state["calls"], 2)
        self.assertEqual(result.text, "SUMMARY: eight rows and six col")

    def test_a_genuinely_empty_reply_is_not_retried_forever(self):
        empty = "data: " + json.dumps({"choices": [{"delta": {"content": ""}}]}) \
            + "\n\ndata: [DONE]\n\n"
        transport, state = self._routes(empty, empty)
        with mock.patch("sift.llm.base.urllib.request.urlopen", transport):
            with self.assertRaises(LLMError):
                OpenAICompatClient("http://h:1234/v1").generate("s", "u",
                                                               max_tokens=400)
        self.assertEqual(state["calls"], 1, "no reasoning, so no retry")


# ---------------------------------------------------------------------------
# An error reported in a 200. LM Studio answers a path it does not serve with
# HTTP 200 and {"error": "Unexpected endpoint or method. (GET /api/tags)"} —
# measured with curl against a live LM Studio. A transport that trusts the
# status reads that as an answer: provider=auto took LM Studio for Ollama, and
# an empty model list on the wrong base path read as "nothing is resident".
# ---------------------------------------------------------------------------
LMSTUDIO_TAGS = {"error": "Unexpected endpoint or method. (GET /api/tags)"}
LMSTUDIO_V0 = {"error": "Unexpected endpoint or method. (GET /api/v0/models)"}


class ServerComplaintTest(unittest.TestCase):

    def test_a_normal_answer_is_still_an_answer(self):
        from sift.llm.base import ServerComplaint  # noqa: F401 - must exist
        cap = {}
        routes = {"/models": {"data": [{"id": LOADED_CHAT}]}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://h:1234/v1")
            self.assertTrue(c.probe("/models"))
            self.assertEqual(c.list_models(), [LOADED_CHAT])

    def test_lmstudio_is_not_taken_for_ollama(self):
        """The body LM Studio really sends for GET /api/tags."""
        from sift.config import Config
        from sift.llm import build_client
        from sift.llm.base import ServerComplaint

        cap = {}
        routes = {"/api/tags": LMSTUDIO_TAGS,
                  "/v1/models": {"data": [{"id": LOADED_CHAT}]}}
        with patched(routes, cap):
            self.assertFalse(OllamaClient("http://h:1234").available())
            with self.assertRaises(ServerComplaint) as caught:
                OllamaClient("http://h:1234").list_models()
            client = build_client(Config(endpoint="http://h:1234"))
        self.assertIn("Unexpected endpoint or method", str(caught.exception))
        self.assertIn("/api/tags", str(caught.exception))
        self.assertIsInstance(client, OpenAICompatClient)
        self.assertEqual(client.provider, "lmstudio")

    def test_a_complaint_on_the_residency_list_is_not_an_idle_server(self):
        """The body LM Studio sends for /api/v0/models on a wrong base path.

        Falling back to names alone reported every model as "residency
        unknown", _may_load said yes, and ensure_model picked a cold model.
        """
        from sift.llm.base import ServerComplaint

        routes = {"/api/v0/models": LMSTUDIO_V0,
                  "/models": {"data": [{"id": COLD_CHAT},
                                       {"id": "gemma-4-e4b"}]},
                  "/chat/completions": {"choices": [
                      {"message": {"content": "ok"}}]}}
        cap = {}
        with patched(routes, cap):
            c = OpenAICompatClient("http://h:1234/v1")
            with self.assertRaises(ServerComplaint):
                c.inventory()
            with self.assertRaises(LLMError):
                OpenAICompatClient("http://h:1234/v1").generate("s", "u")
            explicit = OpenAICompatClient("http://h:1234/v1", model=COLD_CHAT)
            with self.assertRaises(LLMError):
                explicit.generate("s", "u")
        self.assertIsNone(cap.get("data"), "a completion was sent anyway")

    def test_a_complaint_instead_of_a_completion_names_the_reason(self):
        """Non-streaming: the server's words, not "returned no choices"."""
        from sift.llm.base import ServerComplaint

        cap = {}
        routes = {"/api/v0/models": v0((LOADED_CHAT, "loaded", "llm")),
                  "/chat/completions": {"error": {
                      "message": "Model unloaded while generating"}}}
        with patched(routes, cap):
            c = OpenAICompatClient("http://h:1234/v1", stream=False)
            with self.assertRaises(ServerComplaint) as caught:
                c.generate("s", "u")
        self.assertIn("Model unloaded while generating", str(caught.exception))

    def test_ollama_non_streaming_complaint_is_raised(self):
        from sift.llm.base import ServerComplaint

        cap = {}
        routes = {"/api/tags": tags(LOADED_CHAT), "/api/ps": ps(LOADED_CHAT),
                  "/api/chat": {"error": "model runner has crashed"}}
        with patched(routes, cap):
            c = OllamaClient("http://h:11434", stream=False)
            with self.assertRaises(ServerComplaint) as caught:
                c.generate("s", "u")
        self.assertIn("model runner has crashed", str(caught.exception))

    def test_a_null_error_field_is_not_a_complaint(self):
        cap = {}
        body = dict(v0((LOADED_CHAT, "loaded", "llm")), error=None)
        with patched({"/api/v0/models": body}, cap):
            c = OpenAICompatClient("http://h:1234/v1")
            self.assertTrue(c.probe("/api/v0/models"))
            self.assertTrue(c.inventory()[0].loaded)

    def test_an_empty_or_false_error_value_is_a_complaint_that_says_so(self):
        """The plan's rule is "not null", and it stays: a server that puts
        "error" at the top level with any value has said the call failed.
        What changes is the sentence — "reported an error in a successful
        response: " with nothing after it named neither the value nor what
        else the body said."""
        from sift.llm.base import ServerComplaint
        for value, shown in (("", '""'), (0, "0"), (False, "false"),
                             ([], "[]"), ({}, "{}")):
            with self.subTest(value=value):
                cap = {}
                body = {"error": value, "data": [{"id": LOADED_CHAT}]}
                with patched({"/models": body}, cap):
                    c = OpenAICompatClient("http://h:1234/v1")
                    self.assertFalse(c.probe("/models"))
                    with self.assertRaises(ServerComplaint) as caught:
                        c.list_models()
                message = str(caught.exception)
                self.assertIn(f'"error": {shown}', message)
                self.assertIn(LOADED_CHAT, message)
                self.assertFalse(message.rstrip().endswith(":"), message)

    def test_a_complaint_with_words_is_quoted_as_before(self):
        from sift.llm.base import ServerComplaint
        cap = {}
        with patched({"/models": {"error": {"message": "Model gone"}}}, cap):
            with self.assertRaises(ServerComplaint) as caught:
                OpenAICompatClient("http://h:1234/v1").list_models()
        self.assertTrue(str(caught.exception).endswith(": Model gone"),
                        str(caught.exception))

    def test_an_error_field_below_the_top_level_is_not_a_complaint(self):
        """Only the top level speaks for the response. A model entry or a
        usage block carrying its own "error" is data, not a failed call."""
        cap = {}
        body = {"object": "list", "data": [
            {"id": LOADED_CHAT, "state": "loaded", "type": "llm",
             "error": "last load failed"}]}
        answer = {"choices": [{"message": {"content": "TYPE: x"}}],
                  "usage": {"error": "not counted"}}
        with patched({"/api/v0/models": body, "/chat/completions": answer}, cap):
            c = OpenAICompatClient("http://h:1234/v1", stream=False)
            self.assertTrue(c.inventory()[0].loaded)
            self.assertEqual(c.generate("s", "u").text, "TYPE: x")

    def test_a_list_body_is_still_the_wrong_shape_not_a_complaint(self):
        from sift.llm.base import ServerComplaint

        cap = {}
        with patched({"/models": [{"error": "x"}]}, cap):
            c = OpenAICompatClient("http://h:1234/v1")
            self.assertFalse(c.probe("/models"))
            with self.assertRaises(LLMError) as caught:
                c.list_models()
        self.assertNotIsInstance(caught.exception, ServerComplaint)
        self.assertIn("not an object", str(caught.exception))
