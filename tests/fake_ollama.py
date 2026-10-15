"""An Ollama stand-in, served over a real socket.

Ollama is not reachable from this environment, so its path was only ever
exercised with a replaced ``urlopen`` — which proves the payload we build and
nothing about the bytes on the wire. This server closes that gap, and like the
LM Studio stand-in it makes an eviction visible: asking for a model that
``/api/ps`` does not list is recorded and refused, so a test can assert sift
never did it.

Endpoints mirror Ollama's:
    GET  /api/tags   every installed model
    GET  /api/ps     the models currently held in memory
    POST /api/chat   newline-delimited JSON stream
    POST /api/embed  vectors, for a resident model
    GET  /__probe__  test-only: what did the client actually do?

The last line of a chat carries what it cost, as Ollama's API documents it:
prompt_eval_count and eval_count in tokens, prompt_eval_duration,
eval_duration, load_duration and total_duration in nanoseconds.
``usage=False`` plays a server that leaves them out.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fake_lmstudio import fake_vector

DEFAULT_INSTALLED = [
    {"name": "llama3.1:latest", "model": "llama3.1:latest",
     "details": {"families": ["llama"]}, "context_length": 131072},
    {"name": "llava:13b", "model": "llava:13b",
     "details": {"families": ["clip", "llama"]}},
    {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest",
     "details": {"families": ["nomic-bert"]}},
    {"name": "mistral:7b", "model": "mistral:7b",
     "details": {"families": ["llama"]}},
]
DEFAULT_RESIDENT = ["llama3.1:latest"]
#: What the last line of a chat reports. Durations are nanoseconds.
DEFAULT_COUNTS = {"prompt_eval_count": 26, "eval_count": 12,
                  "prompt_eval_duration": 130_000_000,
                  "eval_duration": 480_000_000,
                  "load_duration": 5_000_000, "total_duration": 640_000_000}


class FakeOllama:
    """Context manager yielding a running server. ``base_url`` is its root."""

    def __init__(self, installed=None, resident=None,
                 reply: str = "TYPE: test\nSUMMARY: ok", chunk_size: int = 5,
                 usage: bool = True, embed_dims: int = 8):
        self.installed = [dict(m) for m in (installed or DEFAULT_INSTALLED)]
        self.resident = list(DEFAULT_RESIDENT if resident is None else resident)
        self.reply = reply
        self.chunk_size = chunk_size
        self.usage = usage
        self.embed_dims = embed_dims
        self.embed_requests: list[dict] = []
        self.requests: list[dict] = []
        self.evictions: list[str] = []
        self._srv = None
        self._thread = None

    def __enter__(self) -> "FakeOllama":
        outer = self

        class Handler(_Handler):
            state = outer

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)
        return False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    @property
    def models_used(self) -> list[str]:
        return [r["model"] for r in self.requests]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: FakeOllama = None

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
        if path == "/api/tags":
            self._send(200, {"models": self.state.installed})
        elif path == "/api/ps":
            self._send(200, {"models": [
                {"name": n, "model": n} for n in self.state.resident]})
        elif path == "/__probe__":
            self._send(200, {"requests": self.state.requests,
                             "evictions": self.state.evictions})
        else:
            self._send(404, {"error": f"no such endpoint: {self.path}"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            self._send(400, {"error": "invalid json"})
            return
        path = self.path.rstrip("/")
        if path not in ("/api/chat", "/api/embed"):
            self._send(404, {"error": f"no such endpoint: {self.path}"})
            return

        wanted = payload.get("model", "")
        known = {m["name"] for m in self.state.installed}
        if wanted not in known:
            self._send(404, {"error": f"model '{wanted}' not found"})
            return
        if wanted not in self.state.resident:
            # The real server would load it here, evicting what is resident.
            self.state.evictions.append(wanted)
            self._send(400, {"error": f"EVICTION: loading '{wanted}' would "
                                      f"unload the resident model"})
            return

        if path == "/api/embed":
            self._embed(wanted, payload)
            return
        self.state.requests.append(payload)
        last = (payload.get("messages") or [{}])[-1]
        if last.get("images") and "clip" not in _families(self.state, wanted):
            self._send(400, {"error": f"model '{wanted}' cannot see images"})
            return

        if payload.get("stream"):
            self._stream(wanted)
        else:
            self._send(200, dict({"model": wanted, "done": True,
                                  "done_reason": "stop",
                                  "message": {"role": "assistant",
                                              "content": self.state.reply}},
                                 **self._counts()))

    def _embed(self, model: str, payload: dict) -> None:
        """The shape Ollama's API documents for /api/embed."""
        self.state.embed_requests.append(payload)
        texts = payload.get("input")
        texts = [texts] if isinstance(texts, str) else list(texts or [])
        answer = {"model": model,
                  "embeddings": [fake_vector(t, self.state.embed_dims)
                                 for t in texts],
                  "total_duration": 14_143_917, "load_duration": 1_019_500}
        if self.state.usage:
            answer["prompt_eval_count"] = sum(len(t.split()) for t in texts)
        self._send(200, answer)

    def _counts(self) -> dict:
        return dict(DEFAULT_COUNTS) if self.state.usage else {}

    def _stream(self, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        raw = self.state.reply
        size = max(1, self.state.chunk_size)
        for i in range(0, len(raw), size):
            self._chunk(json.dumps({
                "model": model, "done": False,
                "message": {"role": "assistant", "content": raw[i:i + size]}}) + "\n")
        self._chunk(json.dumps(dict({"model": model, "done": True,
                                     "done_reason": "stop",
                                     "message": {"role": "assistant",
                                                 "content": ""}},
                                    **self._counts())) + "\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _chunk(self, text: str) -> None:
        body = text.encode("utf-8")
        self.wfile.write(f"{len(body):X}\r\n".encode("ascii") + body + b"\r\n")
        self.wfile.flush()


def _families(state: FakeOllama, name: str) -> list[str]:
    for m in state.installed:
        if m["name"] == name:
            return (m.get("details") or {}).get("families", [])
    return []
