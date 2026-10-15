"""A faithful LM Studio stand-in, served over a real socket.

Mocking ``urlopen`` proves the payload we build; it does not prove the bytes
survive a socket, a chunked stream or a proxy. This server closes that gap and
adds the one thing a mock cannot express: it makes an eviction *visible*. On a
real LM Studio, asking for a model whose state is ``not-loaded`` silently
JIT-loads it and drops the resident one. Here that request fails loudly and is
recorded, so a test can assert the app never made it.

Endpoints mirror LM Studio 0.3.x:
    GET  /v1/models             OpenAI list — everything downloaded, no state
    GET  /api/v0/models         native list — with "state" and "type"
    POST /v1/chat/completions   completion, streaming or not
    POST /v1/embeddings         vectors, for a loaded embeddings model
    GET  /__probe__             test-only: what did the client actually do?

Token usage as LM Studio reports it (measured 2026-09-24 against
qwen3.6-35b-a3b-mlx@4bit): a non-streaming answer carries "usage"; a
stream carries it only when asked with stream_options.include_usage, in
one last chunk whose "choices" is an empty list. ``usage=None`` plays a
server that never reports it; ``refuse_stream_options`` one that answers
the option with HTTP 400 naming it, as older OpenAI-compatible servers do.

Any other path gets what LM Studio really sends for it: HTTP *200* with
``{"error": "Unexpected endpoint or method. (GET /api/tags)"}`` (measured with
curl on 2026-09-24). A 404 here made every client that trusts the status look
correct against a server that does not exist.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODELS = [
    {"id": "qwen2.5-7b-instruct", "type": "llm", "state": "loaded",
     "max_context_length": 32768},
    {"id": "llava-v1.6-mistral-7b", "type": "vlm", "state": "loaded",
     "max_context_length": 32768},
    {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings",
     "state": "loaded", "max_context_length": 2048},
    {"id": "mistral-nemo-instruct-2407", "type": "llm", "state": "not-loaded",
     "max_context_length": 1024000},
]


#: What the server says a completion cost. The fields LM Studio sends.
DEFAULT_USAGE = {"prompt_tokens": 10, "completion_tokens": 5,
                 "total_tokens": 15,
                 "completion_tokens_details": {"reasoning_tokens": 2}}


def fake_vector(text: str, dims: int = 8) -> list:
    """The vector both fakes give ``text``: fixed, so a test can tell which
    text a vector belongs to after the server shuffled them."""
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    return [round((raw[i % len(raw)] - 127.5) / 127.5, 6) for i in range(dims)]


class FakeLMStudio:
    """Context manager yielding a running server. ``base_url`` points at /v1."""

    def __init__(self, models=None, reply: str = "TYPE: test\nSUMMARY: ok",
                 chunk_size: int = 7, fail_times: int = 0,
                 usage="default", refuse_stream_options: bool = False,
                 embed_dims: int = 8, embed_shuffled: bool = False,
                 embed_usage_zero: bool = False):
        self.models = [dict(m) for m in (models or DEFAULT_MODELS)]
        self.reply = reply
        self.chunk_size = chunk_size
        self.fail_times = fail_times          # first N completions return 500
        self.usage = dict(DEFAULT_USAGE) if usage == "default" else usage
        self.refuse_stream_options = refuse_stream_options
        self.embed_dims = embed_dims
        #: Answer the vectors in reverse order, each with its "index" — the
        #: order the OpenAI dialect promises nothing about.
        self.embed_shuffled = embed_shuffled
        #: Report "prompt_tokens": 0 for embeddings, as LM Studio does
        #: (measured 2026-09-24 with text-embedding-nomic-embed-text-v1.5).
        self.embed_usage_zero = embed_usage_zero
        self.embed_requests: list[dict] = []  # every embeddings payload
        self.requests: list[dict] = []        # every completion payload
        self.refused: list[dict] = []         # payloads answered with a 400
        self.evictions: list[str] = []        # cold models the client asked for
        self._srv = None
        self._thread = None

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "FakeLMStudio":
        outer = self

        class Handler(_BaseHandler):
            state = outer

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)
        return False

    @property
    def port(self) -> int:
        return self._srv.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    # -- assertions -----------------------------------------------------
    @property
    def models_used(self) -> list[str]:
        return [r["model"] for r in self.requests]

    def by_state(self, state: str) -> set[str]:
        return {m["id"] for m in self.models if m["state"] == state}


class _BaseHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: FakeLMStudio = None

    def log_message(self, *a):
        pass

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.rstrip("/")
        if path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": m["id"], "object": "model",
                 "owned_by": "organization_owner"} for m in self.state.models]})
        elif path == "/api/v0/models":
            self._send(200, {"object": "list", "data": [
                dict(m, object="model") for m in self.state.models]})
        elif path == "/__probe__":
            self._send(200, {"requests": self.state.requests,
                             "evictions": self.state.evictions})
        else:
            self._unexpected("GET")

    def _unexpected(self, method: str) -> None:
        self._send(200, {"error": f"Unexpected endpoint or method. "
                                  f"({method} {self.path})"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            self._send(400, {"error": "invalid json"})
            return
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/embeddings"):
            self._unexpected("POST")
            return

        known = {m["id"]: m for m in self.state.models}
        wanted = payload.get("model", "")
        if wanted not in known:
            self._send(404, {"error": {"message": f"Model '{wanted}' not found",
                                       "type": "invalid_request_error"}})
            return
        if known[wanted]["state"] != "loaded":
            # The real server would load it here and evict the resident model.
            self.state.evictions.append(wanted)
            self._send(400, {"error": {
                "message": f"EVICTION: loading '{wanted}' would unload the "
                           f"resident model",
                "type": "invalid_request_error"}})
            return
        if path == "/v1/embeddings":
            self._embeddings(payload, known[wanted])
            return

        if payload.get("stream_options") is not None \
                and self.state.refuse_stream_options:
            self.state.refused.append(payload)
            self._send(400, {"error": {
                "message": "Unrecognized request argument supplied: "
                           "stream_options",
                "type": "invalid_request_error"}})
            return

        self.state.requests.append(payload)
        if len(self.state.requests) <= self.state.fail_times:
            self._send(500, {"error": {"message": "simulated backend failure"}})
            return

        messages = payload.get("messages", [])
        last = messages[-1] if messages else {}
        content = last.get("content", "")
        images = [p for p in content
                  if isinstance(p, dict) and p.get("type") == "image_url"] \
            if isinstance(content, list) else []
        if images and known[wanted]["type"] != "vlm":
            self._send(400, {"error": {
                "message": f"Model '{wanted}' cannot see images",
                "type": "invalid_request_error"}})
            return

        if payload.get("stream"):
            options = payload.get("stream_options") or {}
            self._stream(wanted, bool(options.get("include_usage")))
        else:
            answer = {
                "id": "chatcmpl-fake", "object": "chat.completion",
                "model": wanted,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": self.state.reply}}]}
            if self.state.usage is not None:
                answer["usage"] = self.state.usage
            self._send(200, answer)

    def _embeddings(self, payload: dict, model: dict) -> None:
        if model["type"] != "embeddings":
            self._send(400, {"error": {
                "message": f"Model '{model['id']}' is not an embedding model",
                "type": "invalid_request_error"}})
            return
        self.state.embed_requests.append(payload)
        texts = payload.get("input")
        texts = [texts] if isinstance(texts, str) else list(texts or [])
        data = [{"object": "embedding", "index": i,
                 "embedding": fake_vector(t, self.state.embed_dims)}
                for i, t in enumerate(texts)]
        if self.state.embed_shuffled:
            data.reverse()
        answer = {"object": "list", "data": data, "model": model["id"]}
        if self.state.usage is not None:
            words = 0 if self.state.embed_usage_zero else sum(len(t.split()) for t in texts)
            answer["usage"] = {"prompt_tokens": words, "total_tokens": words}
        self._send(200, answer)

    def _stream(self, model: str, include_usage: bool = False) -> None:
        """Chunked SSE, split mid-word and mid-multibyte on purpose."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        raw = self.state.reply
        size = max(1, self.state.chunk_size)
        pieces = [raw[i:i + size] for i in range(0, len(raw), size)] or [""]
        for piece in pieces:
            frame = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                     "model": model,
                     "choices": [{"index": 0, "delta": {"content": piece},
                                  "finish_reason": None}]}
            self._chunk(f"data: {json.dumps(frame)}\n\n")
        finish = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                  "model": model,
                  "choices": [{"index": 0, "delta": {},
                               "finish_reason": "stop"}]}
        self._chunk(f"data: {json.dumps(finish)}\n\n")
        if include_usage and self.state.usage is not None:
            # The measured shape: no choices, only the usage.
            self._chunk("data: " + json.dumps({
                "id": "chatcmpl-fake", "object": "chat.completion.chunk",
                "model": model, "choices": [],
                "usage": self.state.usage}) + "\n\n")
        self._chunk("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _chunk(self, text: str) -> None:
        body = text.encode("utf-8")
        self.wfile.write(f"{len(body):X}\r\n".encode("ascii") + body + b"\r\n")
        self.wfile.flush()
