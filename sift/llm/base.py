"""LLM client base: shared HTTP (stdlib urllib) and result/error types."""
from __future__ import annotations

import contextlib
import contextvars
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

from ..chunk import estimate_tokens
from ..extract.base import ERROR_CHARS, SERVER_WORDS, repeatable


class LLMError(Exception):
    """Raised when an LLM backend is unreachable or returns an error."""


#: The sentence inside an error value a server answers with (complaint_text).
COMPLAINT_CHARS = 200


def _body_of(exc) -> str:
    """Best-effort error body, what repeatable reads of it at most. Reporting
    a failure must not itself fail."""
    reach = 4 * SERVER_WORDS
    try:
        raw = exc.read(reach + 1)
    except Exception:  # noqa: BLE001 - the body is a nicety, not the error
        return ""
    if isinstance(raw, (bytes, bytearray)):
        cut = len(raw) > reach
        raw = raw.decode("utf-8", "replace")
        if cut:             # the word the read ended in is not repeated
            head = raw.rsplit(None, 1)
            raw = (head[0] if len(head) == 2 else "") + " …"
    return str(raw or "")


#: Statuses that, with nothing in the body, almost never come from the model
#: server: it would say why. A bare one is something in between — a proxy with
#: an allow-list, a gateway, a corporate filter — refusing the address. Told
#: only "HTTP 403 at /models:" a reader goes looking at the model server,
#: which is the one place the fault is not. Observed exactly this against a
#: proxy that answered 403 with zero bytes.
_MIDDLEBOX = {403: "a proxy or gateway", 407: "a proxy requiring "
              "authentication", 502: "a gateway", 504: "a gateway"}


#: A string of a JSON text: closed, or cut off where the body was (_body_of).
_JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)("|\Z)', re.DOTALL)
#: An escape JSON knows, inside such a string.
_JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt]))')
_JSON_ESCAPED = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                 "n": "\n", "r": "\r", "t": "\t"}


def _unescaped(body: str) -> str:
    """``body`` with the strings of a JSON text read as JSON reads them.

    A server's error body quotes what it was sent, and quotes it escaped --
    Kx9v\\"Q2mZ7pL4Wq for a quote, Kx9v\\tQ2... for a tab, \\u00e4 for an
    umlaut --, which is no spelling hide_known knows: a value the file
    taught it was said whole. A 200's error value was
    read decoded all along (complaint_text). A body that is no JSON text is
    left as it is; a string cut off where the body was is read to its end,
    an escape JSON does not know stays as written, and a surrogate that is
    half of no pair becomes U+FFFD."""
    if body.lstrip()[:1] not in ("{", "["):
        return body

    def one(match) -> str:
        code, char = match.groups()
        return chr(int(code, 16)) if code else _JSON_ESCAPED[char]

    def string(match) -> str:
        text = _JSON_ESCAPE.sub(one, match.group(1))
        text = text.encode("utf-16", "surrogatepass").decode("utf-16",
                                                            "replace")
        return f'"{text}{match.group(2)}'

    return _JSON_STRING.sub(string, body)


def _http_message(provider: str, path: str, exc) -> str:
    body = repeatable(_unescaped(_body_of(exc)))
    if body:
        return f"{provider} HTTP {exc.code} at {path}: {body}"
    hint = _MIDDLEBOX.get(exc.code)
    if hint:
        return (f"{provider} HTTP {exc.code} at {path} with an empty body — "
                f"typically {hint} between sift and the server, not the "
                f"server itself; check the proxy settings and whether the "
                f"address is allowed")
    return f"{provider} HTTP {exc.code} at {path} (no message in the body)"


def complaint_text(problem) -> str:
    """The sentence inside an error value, one line and bounded.

    Servers put it either straight in ``"error"`` or one level down in
    ``"error": {"message": ...}`` — the OpenAI dialect does the second.
    """
    if isinstance(problem, dict):
        problem = problem.get("message", problem)
    return repeatable(problem, COMPLAINT_CHARS)


class HTTPStatusError(LLMError):
    """An HTTP error whose status code the caller needs to branch on."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class ServerComplaint(LLMError):
    """The server answered, and the answer was an error.

    Its own type because the status line said otherwise: LM Studio answers
    a path it does not serve with HTTP 200 and ``{"error": "Unexpected
    endpoint or method. (GET /api/tags)"}``. Read by status, that is a
    success, and both places that ask "is anyone there, and what is
    loaded" drew the wrong conclusion from it.
    """


class _Refusal(LLMError):
    """We could talk to the server but chose not to, to protect a loaded model.

    Separate from a transport failure so the caller does not "helpfully" retry
    the very request the refusal exists to prevent.
    """


class NoVisionModel(LLMError):
    """No image-capable model is available without evicting a loaded one.

    Its own type so the caller can retry text-only instead of giving up on the
    model entirely — a distinction a string match on the message would lose.
    """


@dataclass
class CallUsage:
    """One HTTP call to the model, and what it cost — whatever became of it.

    LLMResult exists only when a call succeeds, and the expensive calls are
    the ones that do not: a reasoning model that spends its whole budget
    thinking and answers nothing, a retry that loses to the answer it was
    meant to improve, a part of a large file that fails. So each call is
    reported on its own, through LLMClient.on_call.

    ``kind``: chat, or what the caller was doing — map, reduce,
    vision_fallback, reclassify — retry for the client's own second
    attempt after a reasoning model ran out of room, and embed for a call
    to the embeddings endpoint (which has no completion tokens). ``outcome``: ok (the
    answer used), empty (no answer, only thinking), error (the call
    failed), discarded (an answer, but not the one used). ``source``:
    server when the server counted the tokens, estimated when sift did —
    from what it sent and what came back. A call that failed before the
    server answered at all has no tokens: nothing is known to have been
    read.
    """

    kind: str = "chat"
    outcome: str = "ok"
    provider: str = ""
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    #: Measured by the client: from sending the request to the first piece
    #: of the answer, thinking included. None without a stream.
    ttft_seconds: Optional[float] = None
    total_seconds: float = 0.0
    source: str = "estimated"
    stop_reason: str = ""
    #: Ollama's own timings, which it reports in nanoseconds: reading the
    #: prompt, and writing the answer.
    prompt_eval_seconds: Optional[float] = None
    eval_seconds: Optional[float] = None
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Answer tokens per second of writing them: by the server's own
        clock where it has one, else from the first token to the last."""
        if not self.completion_tokens:
            return None
        if self.eval_seconds:
            return round(self.completion_tokens / self.eval_seconds, 2)
        if self.ttft_seconds is None:
            return None
        writing = self.total_seconds - self.ttft_seconds
        return (round(self.completion_tokens / writing, 2)
                if writing > 0 else None)


#: Where the calls of one summary are collected, and what the caller is
#: doing at the moment (see call_kind). ContextVars, because `sift serve`
#: summarizes each upload on its own thread with one shared client.
_RECORDED: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "sift_llm_calls", default=None)
_KIND: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "sift_llm_call_kind", default="chat")


@contextlib.contextmanager
def recording_calls():
    """Collect every call reported in this context, in order."""
    calls: list = []
    token = _RECORDED.set(calls)
    try:
        yield calls
    finally:
        _RECORDED.reset(token)


@contextlib.contextmanager
def call_kind(kind: str):
    """Label the calls made in this context: "map", "reduce", …"""
    token = _KIND.set(kind)
    try:
        yield
    finally:
        _KIND.reset(token)


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    #: True when the server stopped because the token limit was reached. The
    #: answer is then cut off mid-sentence, and saying so is the difference
    #: between a short summary and a wrong one.
    truncated: bool = False
    #: Characters the model spent thinking before answering. Reasoning counts
    #: against max_tokens, so a model that deliberates at length runs out of
    #: budget mid-answer — and the fix is a bigger budget, not a shorter file.
    reasoning_chars: int = 0
    #: Why the answer stopped, when the server said: "length" (the token
    #: budget ran out), "stop" (the model finished), or "cut" (the bytes
    #: stopped arriving and nothing said the answer was over). The three call
    #: for three different sentences to the user, and collapsing them into one
    #: boolean made the third indistinguishable from success.
    stop_reason: str = ""
    #: The calls this answer took: one, or two when a reasoning model was
    #: asked again with more room. on_call has seen them too.
    calls: list = field(default_factory=list)


@dataclass
class EmbeddingResult:
    """Vectors for the texts LLMClient.embed() was given, in their order."""
    vectors: list
    model: str
    #: The length of every vector — embed() refuses an answer where they
    #: differ. 0 for no texts.
    dims: int
    #: The one call it took (kind "embed"), as on_call saw it. None when
    #: there was nothing to embed and nothing was asked.
    usage: Optional[CallUsage] = None


@dataclass
class ModelInfo:
    """One model the server knows about.

    ``loaded`` distinguishes three states on purpose:
      True  — resident in memory right now, free to use;
      False — downloaded but cold, using it forces a load;
      None  — the server does not report load state, so we cannot tell.
    """

    id: str
    loaded: Optional[bool] = None
    kind: str = "unknown"            # llm | vlm | embeddings | unknown
    #: The window actually available: what the model was loaded with, which may
    #: be far below what it supports.
    context: Optional[int] = None
    max_context: Optional[int] = None

    @property
    def usable_for_chat(self) -> bool:
        return self.kind in ("llm", "vlm", "unknown")

    @property
    def sees_images(self) -> bool:
        return self.kind == "vlm"


class LLMClient(ABC):
    """Common interface for local LLM backends."""

    provider: str = "base"

    def __init__(self, endpoint: str, model: str = "", api_key: str = "",
                 timeout: float = 180.0, allow_model_load: bool = False,
                 stream: bool = True, embedding_model: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # False (the default) means: never make the server load a cold model.
        # Loading one evicts whatever the user already has resident.
        self.allow_model_load = allow_model_load
        self.stream = stream
        #: Called with each piece of the answer as it arrives. A local model
        #: takes a minute or more per file, and a still line of output is
        #: indistinguishable from a hung one.
        self.on_progress = None
        #: Called with a CallUsage once for every call to the chat endpoint,
        #: after it is over, whatever became of it. An exception it raises
        #: is swallowed: counting must not cost the answer.
        self.on_call = None
        self._inventory_cache: Optional[list["ModelInfo"]] = None
        #: True when the server has a residency endpoint, it answered, and it
        #: still did not say which models are loaded. That is different from
        #: a server with no such endpoint at all — llama.cpp and vLLM have
        #: none, and refusing to summarize against those would cost the user
        #: everything for nothing. Here the server could have told us and did
        #: not, so "unknown" has to count as "something may be resident".
        self._residency_withheld = False
        self._auto_choice: dict[bool, str] = {}   # need_vision -> model id
        #: The model embed() uses. Empty: the resident embeddings model that
        #: is not a reranker, chosen like a chat model is.
        self.embedding_model = embedding_model
        self._embed_choice = ""

    # -- lifecycle ------------------------------------------------------
    @abstractmethod
    def available(self) -> bool:
        """True if the backend responds (server up)."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Model identifiers the server offers."""

    def inventory(self) -> list[ModelInfo]:
        """Models with load state and capability, best effort.

        Backends that can report residency override this. The base version
        knows only the names, so it reports ``loaded=None`` — "cannot tell" —
        and infers the kind from the name.
        """
        return [ModelInfo(id=name, loaded=None, kind=classify_model(name))
                for name in self.list_models()]

    @abstractmethod
    def generate(self, system: str, user: str, *,
                 images: Optional[list[bytes]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 800) -> LLMResult:
        """Run one chat completion and return the assistant text."""

    # -- call accounting ------------------------------------------------
    def _new_call(self, calls: list, model: str, kind: str = "chat"
                  ) -> CallUsage:
        usage = CallUsage(kind=kind, provider=self.provider, model=model)
        calls.append(usage)
        return usage

    def _report_calls(self, calls: list) -> None:
        """Hand each call over, once its outcome is final.

        Not as it happens: the first answer of a pair is "ok" or "discarded"
        depending on a retry that comes after it, and a caller writing rows
        as they arrive must not be told one outcome and then another.
        """
        kind = _KIND.get()
        recorded = _RECORDED.get()
        for usage in calls:
            if usage.kind == "chat":
                usage.kind = kind
            if recorded is not None:
                recorded.append(usage)
            if self.on_call is not None:
                try:
                    self.on_call(usage)
                except Exception:  # noqa: BLE001 - the caller's hook
                    pass

    # -- embeddings -----------------------------------------------------
    def embed(self, texts) -> EmbeddingResult:
        """One vector per text, from an embeddings model, in one call.

        The model follows the rule every model sift uses follows: resident,
        or loadable without evicting anything (ensure_embedding_model). The
        call is reported like a chat call — CallUsage kind "embed", through
        on_call and recording_calls — whatever became of it.

        Raises ValueError for texts that cannot be embedded (not a list of
        non-blank strings: a bare string would be embedded a character at a
        time), and LLMError for everything the server does or answers.
        """
        texts = _embeddable(texts)
        if not texts:
            return EmbeddingResult(vectors=[], model="", dims=0)
        model = self.ensure_embedding_model()
        calls: list = []
        usage = self._new_call(calls, model, kind="embed")
        started = time.monotonic()
        try:
            vectors, counted = self._embed_request(model, texts)
            vectors = _checked_vectors(self.provider, vectors, len(texts))
        except Exception as exc:
            usage.outcome = "error"
            usage.error = repeatable(str(exc) or type(exc).__name__,
                                     ERROR_CHARS)
            raise
        else:
            # Every text here is non-blank, so it is at least one token: a
            # server that says 0 (LM Studio does, for embeddings) has not
            # counted, and the estimate is the better figure.
            if counted:
                usage.prompt_tokens, usage.source = counted, "server"
            else:
                usage.prompt_tokens = sum(estimate_tokens(t) for t in texts)
            usage.stop_reason = "stop"
        finally:
            usage.total_seconds = round(time.monotonic() - started, 3)
            self._report_calls(calls)
        return EmbeddingResult(vectors=vectors, model=model,
                               dims=len(vectors[0]), usage=usage)

    def _embed_request(self, model: str, texts: list) -> tuple:
        """(vectors, prompt tokens the server counted or None) — the
        provider's own endpoint. Unchecked: embed() checks the answer."""
        raise LLMError(f"{self.provider}: sift does not know this server's "
                       f"embeddings endpoint")

    def ensure_embedding_model(self) -> str:
        """The embeddings model to use, under the rule ensure_model follows.

        Only models of kind "embeddings" that are not rerankers: the name
        hints behind classify_model put both under "embeddings", and a
        reranker scores a pair of texts — it has no vector to give. A named
        embedding_model is used if it is resident, or if loading it cannot
        evict anything; otherwise refused, as a named chat model is.
        """
        if self.embedding_model:
            if self.allow_model_load:
                return self.embedding_model
            try:
                models = self.cached_inventory()
            except LLMError as exc:
                raise _Refusal(
                    f"{self.provider}: cannot check whether "
                    f"{self.embedding_model!r} is already loaded ({exc}). "
                    f"Refusing rather than risk evicting the resident "
                    f"model — pass --allow-model-load if that is what you "
                    f"want.") from exc
            return self._verify_embedder(models)
        if self._embed_choice:
            return self._embed_choice
        models = self.cached_inventory()
        pool = sorted((m for m in models if m.kind == "embeddings"
                       and not is_reranker(m.id)),
                      key=lambda m: (m.loaded is not True, m.id))
        trust_unknown = not self._residency_withheld
        resident = [m for m in pool if m.loaded is True
                    or (trust_unknown and m.loaded is None)]
        choice = resident[0] if resident else None
        if choice is None and pool and self._may_load(models):
            choice = pool[0]
            choice.loaded = True        # as ensure_model does: loaded now
        if choice is None:
            cold = [m.id for m in pool]
            installed = (f" Installed but not loaded: {', '.join(cold)}."
                         if cold else " No embedding model is installed "
                                      "(rerankers do not count).")
            raise LLMError(
                f"{self.provider}: no embedding model is loaded on "
                f"{self.endpoint}.{installed} Load one, or pass "
                f"--allow-model-load to let sift load it — which evicts "
                f"what is resident.")
        self._embed_choice = choice.id
        return choice.id

    def _verify_embedder(self, models: list) -> str:
        known = {m.id: m for m in models}
        info = known.get(self.embedding_model) \
            or _loose_match(self.embedding_model, models)
        # The steps of _verify_requested, for the other model slot.
        if info is None or info.loaded is True:
            return self.embedding_model
        if info.loaded is None and self._residency_withheld:
            raise _Refusal(
                f"{self.provider}: {self.endpoint} did not report which "
                f"models are loaded, so sift cannot tell whether using "
                f"{self.embedding_model!r} would evict one. Load it "
                f"yourself, or pass --allow-model-load to accept the swap.")
        if self._may_load(models):
            return self.embedding_model     # nothing would be evicted
        raise _Refusal(
            f"{self.provider}: embedding model {self.embedding_model!r} is "
            f"not loaded, and using it would make {self._ui_name()} load it "
            f"and evict the resident model. Load it yourself, or pass "
            f"--allow-model-load to accept the swap.")

    # -- helpers --------------------------------------------------------
    def ensure_model(self, need_vision: bool = False) -> str:
        """Pick the model to use, preferring one that is already resident.

        Asking a local server for a cold model makes it load that model, and
        both LM Studio and Ollama evict the previously resident one to make
        room. A summarizer must not silently reorganize the user's GPU, so an
        already-loaded model always wins, and a cold one is only used when
        ``allow_model_load`` says it may be.
        """
        if self.model:                     # explicit choice always wins
            if self.allow_model_load:
                return self.model
            try:
                return self._verify_requested(self.cached_inventory())
            except _Refusal:
                raise
            except LLMError as exc:
                # Not "the server did not answer; just try it". inventory()
                # re-raises on any status outside 404/405/501 precisely so a
                # hiccup cannot become permission to evict — and this handler,
                # eleven lines below it, turned every 500, 502 and dropped
                # connection into exactly that permission. Measured: one 500
                # on /api/v0/models and `--model mistral-nemo` evicted the
                # resident model.
                #
                # A server that has no inventory endpoint at all is a
                # different case: it cannot report residency, so it cannot be
                # asked, and refusing there would make sift useless against
                # a plain OpenAI-compatible endpoint. That case arrives as
                # an empty list, not as an error.
                raise _Refusal(
                    f"{self.provider}: cannot check whether "
                    f"{self.model!r} is already loaded ({exc}). Refusing "
                    f"rather than risk evicting the resident model — pass "
                    f"--allow-model-load if that is what you want."
                ) from exc

        cached = self._auto_choice.get(need_vision)
        if cached:
            return cached

        models = self.cached_inventory()
        if not models:
            raise LLMError(
                f"{self.provider}: no model available on {self.endpoint}. "
                f"Load a model in {self._ui_name()} or pass --model.")

        trust_unknown = not self._residency_withheld
        choice = _pick(models, need_vision=need_vision, loaded_only=True,
                       trust_unknown=trust_unknown)
        if choice is None and self._may_load(models):
            # Nothing resident to protect, so loading evicts nothing.
            choice = _pick(models, need_vision=need_vision, loaded_only=False)
            if choice is not None:
                # Record it as resident. The inventory is read once per run, so
                # without this the server still looks idle on the next file and
                # sift keeps loading — evicting the model it loaded a moment
                # ago, once per file, for the whole batch.
                choice.loaded = True
        if choice is None:
            err = NoVisionModel if need_vision else LLMError
            raise err(self._nothing_loaded_message(models, need_vision))
        # Cached per job kind, not written back to self.model: a run that
        # summarizes text and images needs a different model for each, and one
        # shared slot would send the picture to whichever came first.
        self._auto_choice[need_vision] = choice.id
        return choice.id

    def context_for(self, model_id: str) -> Optional[int]:
        """The usable context window of a model, if the server reports one.

        Guessing 8192 wastes most of a 262144-token window; guessing from
        max_context_length overflows one loaded at 16384. The server knows.
        """
        try:
            for model in self.cached_inventory():
                if model.id == model_id:
                    return model.context
        except LLMError:
            return None
        return None

    def cached_inventory(self) -> list[ModelInfo]:
        """inventory() once per client — summarizing N files must not poll N times."""
        if self._inventory_cache is None:
            self._inventory_cache = self.inventory()
        return self._inventory_cache

    def _may_load(self, models: list[ModelInfo]) -> bool:
        """True when loading a model cannot cost the user a loaded one.

        The rule the operator actually asked for is "do not unload what is
        loaded" — not "never load". With an idle server there is nothing to
        evict, so refusing there would only produce worse summaries for no gain.

        But "nobody reports as loaded" is not the same as "nothing is loaded".
        ModelInfo.loaded has three states on purpose, and this read None —
        "the server does not say" — as an idle server: an LM Studio build
        whose /api/v0/models omits `state`, or an Ollama whose /api/ps
        answers 404, and sift would load the largest model on the box over
        the top of whatever was resident. Both are older-build shapes the
        code elsewhere handles explicitly, so they are not hypothetical.

        A server that never reports residency at all — a plain
        OpenAI-compatible endpoint — has nothing to protect, and refusing
        there would cost the user a summary for no gain. So the silence only
        counts against the two providers that run models on the operator's
        own machine and normally do say.
        """
        if self.allow_model_load:
            return True
        if any(m.loaded for m in models):
            return False                # something is resident: do not evict
        if self._residency_withheld:
            return not any(m.loaded is None for m in models)
        return True

    def _verify_requested(self, models: list[ModelInfo]) -> str:
        """Honour an explicit --model, but refuse to cold-load it by default.

        The lookup is deliberately forgiving. Both servers resolve a model id
        loosely — LM Studio matches case-insensitively, Ollama fills in the
        ':latest' tag — so an exact-match check lets 'LLAMA3' or 'llama3' sail
        past a refusal that 'llama3:latest' would have triggered, and the
        server then loads it anyway.
        """
        known = {m.id: m for m in models}
        info = known.get(self.model) or _loose_match(self.model, models)
        if info is None or info.loaded is True:
            # Not a model this server lists — it may still resolve it — or
            # one that is already resident.
            return self.model
        if info.loaded is None and self._residency_withheld:
            raise _Refusal(
                f"{self.provider}: {self.endpoint} did not report which "
                f"models are loaded, so sift cannot tell whether using "
                f"{self.model!r} would evict one. Load it yourself, or pass "
                f"--allow-model-load to accept the swap.")
        if self._may_load(models):
            return self.model               # nothing would be evicted
        loaded = sorted(m.id for m in models if m.loaded and m.usable_for_chat)
        # What is resident may be a model this run cannot use — an
        # embeddings model is the ordinary case, and it is still a model
        # that loading would evict. Naming only the chat ones printed
        # "Already loaded: ." and then "Pick one of those", which is an
        # empty list and an impossible instruction.
        others = sorted(m.id for m in models if m.loaded and not m.usable_for_chat)
        if loaded:
            resident = (f"Already loaded: {', '.join(loaded)}. Pick one of "
                        f"those, load {self.model!r} yourself, or pass ")
        elif others:
            resident = (f"What is loaded is {', '.join(others)}, which this "
                        f"run cannot use and would be the model evicted. "
                        f"Load {self.model!r} yourself, or pass ")
        else:
            resident = f"Load {self.model!r} yourself, or pass "
        raise _Refusal(
            f"{self.provider}: model {self.model!r} is not loaded. Using it "
            f"would make {self._ui_name()} load it and evict the resident "
            f"model. {resident}--allow-model-load to accept the swap.")

    def _nothing_loaded_message(self, models: list[ModelInfo],
                                need_vision: bool) -> str:
        loaded = sorted(m.id for m in models if m.loaded)
        if need_vision:
            cold = sorted(m.id for m in models if m.sees_images)
            extra = (f" Installed but not loaded: {', '.join(cold)}."
                     if cold else " No vision model is installed.")
            return (f"{self.provider}: no vision model is loaded on "
                    f"{self.endpoint}.{extra} Load one, or accept a text-only "
                    f"description of the image.")
        if (self._residency_withheld and models
                and all(m.loaded is None for m in models)):
            # Different problem, different sentence: not "load something" but
            # "your server did not say what is loaded". Told the first, the
            # operator goes and looks at a server that already has a model
            # resident, and concludes sift is broken.
            return (f"{self.provider}: {self.endpoint} did not report which "
                    f"models are loaded — an older build, or a proxy that "
                    f"drops the field. sift will not guess, because guessing "
                    f"wrong evicts whatever is resident. Name one with "
                    f"--model, or pass --allow-model-load.")
        return (f"{self.provider}: nothing usable for chat is loaded on "
                f"{self.endpoint} (loaded: {', '.join(loaded) or 'none'}). "
                f"Load a chat model, or pass --allow-model-load to let sift "
                f"load one — which evicts what is resident.")

    def _ui_name(self) -> str:
        return {"ollama": "Ollama", "lmstudio": "LM Studio",
                "mlx-lm": "MLX-LM"}.get(
            self.provider, self.provider)

    def _request(self, path: str, payload: Optional[dict] = None,
                 method: str = "POST", base: Optional[str] = None) -> dict:
        url = f"{base.rstrip('/') if base else self.endpoint}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Inside the try, not before it: Request() itself raises ValueError
        # for a url with no scheme, and an endpoint of "/v1" — a plausible
        # typo in a config file — escaped summarize_path as a bare
        # ValueError and ended the whole batch, discarding files already
        # summarized.
        try:
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method=method)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise HTTPStatusError(
                _http_message(self.provider, path, exc),
                exc.code) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"{self.provider} unreachable at {url}: "
                           f"{repeatable(exc.reason)}") from exc
        except ValueError as exc:
            raise LLMError(f"{self.provider}: {url!r} is not a usable "
                           f"address ({exc})") from exc
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            # IncompleteRead is an HTTPException, not an OSError. Uncaught it
            # escapes summarize_path — which promises never to raise — and
            # takes the whole batch down, including files already summarized.
            raise LLMError(f"{self.provider} request failed: "
                           f"{type(exc).__name__}: {repeatable(exc)}") from exc
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise LLMError(f"{self.provider} returned non-JSON: "
                           f"{repeatable(body, COMPLAINT_CHARS)}") from exc
        if not isinstance(parsed, dict):
            # Valid JSON of the wrong shape. `[{"id": "m"}]` from a model list
            # endpoint reached `.get` and raised AttributeError out of
            # summarize_path, which promises never to raise — one such
            # response ended a whole batch.
            raise LLMError(f"{self.provider} returned "
                           f"{type(parsed).__name__}, not an object: "
                           f"{repeatable(body, COMPLAINT_CHARS)}")
        if parsed.get("error") is not None:
            # A 200 that says it failed. LM Studio answers every path it does
            # not serve this way, so returning the body let probe() report
            # an Ollama on LM Studio's port, and let inventory() read an
            # empty model list as an idle server — the one reading that
            # licenses a load. Only the top level: a model entry or a usage
            # block with its own "error" key is data, not a failed call.
            #
            # Any value but null counts, as the rule says — "error": "" or
            # false is still a server putting the word at the top of its
            # answer. Those have nothing to quote, so the value and the body
            # are shown instead of a sentence ending in a colon.
            problem = parsed["error"]
            said = complaint_text(problem) if problem else ""
            if not said:
                said = (f'no reason given ("error": '
                        f'{json.dumps(problem)}); the body was: '
                        f'{repeatable(body, COMPLAINT_CHARS)}')
            raise ServerComplaint(
                f"{self.provider} at {path} reported an error in a "
                f"successful response: {said}")
        return parsed

    def _stream_lines(self, path: str, payload: dict,
                      base: Optional[str] = None) -> Iterator[str]:
        """Yield response lines as the server produces them.

        Streaming is not only about progress. A non-streaming request sends no
        bytes at all while the model thinks, and any proxy or load balancer in
        between reads that silence as a dead connection and closes it. Measured
        against this cube's proxy: a 2000-token completion is killed at 20.0 s
        with stream=false and completes in 46.7 s with stream=true.
        """
        url = f"{base.rstrip('/') if base else self.endpoint}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(url, data=data, headers=headers,
                                         method="POST")
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise HTTPStatusError(
                _http_message(self.provider, path, exc),
                exc.code) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"{self.provider} unreachable at {url}: "
                           f"{repeatable(exc.reason)}") from exc
        except ValueError as exc:
            raise LLMError(f"{self.provider}: {url!r} is not a usable "
                           f"address ({exc})") from exc
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            # IncompleteRead is an HTTPException, not an OSError. Uncaught it
            # escapes summarize_path — which promises never to raise — and
            # takes the whole batch down, including files already summarized.
            raise LLMError(f"{self.provider} request failed: "
                           f"{type(exc).__name__}: {repeatable(exc)}") from exc
        try:
            with resp:
                for raw in resp:
                    yield raw.decode("utf-8", "replace").rstrip("\r\n")
        except ValueError as exc:
            raise LLMError(f"{self.provider}: {url!r} is not a usable "
                           f"address ({exc})") from exc
        except (TimeoutError, OSError, http.client.HTTPException) as exc:
            raise LLMError(f"{self.provider} stream broke off mid-answer: "
                           f"{type(exc).__name__}: {repeatable(exc)}") from exc

    def probe(self, path: str) -> bool:
        try:
            self._request(path, method="GET")
            return True
        except LLMError:
            return False


class _Meter:
    """The measurements of one call while it runs: when it started, when the
    first piece of the answer came, and what came.

    ``settle`` fills the CallUsage at the end, from the server's own count
    where it gave one and by estimate where it did not — an estimate of
    what was sent only once the server is known to have answered, since a
    refused request read nothing.
    """

    def __init__(self, usage: CallUsage, prompt: str) -> None:
        self.usage = usage
        self.prompt = prompt
        self.started = time.monotonic()
        self.answered = False
        self.content: list = []
        self.reasoning: list = []
        self.counted: Optional[dict] = None

    def heard(self) -> None:
        """The server answered: some response arrived."""
        self.answered = True

    def piece(self, content: str, reasoning: str = "") -> None:
        self.answered = True
        if (content or reasoning) and self.usage.ttft_seconds is None:
            self.usage.ttft_seconds = round(time.monotonic() - self.started, 3)
        if content:
            self.content.append(content)
        if reasoning:
            self.reasoning.append(reasoning)

    def failed(self, exc: BaseException) -> None:
        self.usage.outcome = "error"
        self.usage.error = repeatable(exc, ERROR_CHARS)

    def settle(self) -> None:
        usage = self.usage
        usage.total_seconds = round(time.monotonic() - self.started, 3)
        counted = self.counted
        if counted and (counted.get("prompt_tokens") is not None
                        or counted.get("completion_tokens") is not None):
            usage.source = "server"
            usage.prompt_tokens = _count(counted.get("prompt_tokens"))
            usage.completion_tokens = _count(counted.get("completion_tokens"))
            if "reasoning_tokens" in counted:
                usage.reasoning_tokens = _count(counted["reasoning_tokens"])
            return
        if not self.answered:
            return                      # nothing is known to have been read
        usage.source = "estimated"
        thought = estimate_tokens("".join(self.reasoning))
        usage.prompt_tokens = estimate_tokens(self.prompt)
        usage.completion_tokens = estimate_tokens("".join(self.content)) \
            + thought
        usage.reasoning_tokens = thought


def _count(value) -> Optional[int]:
    """A token count from the server, or None for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def prompt_text(payload: dict) -> str:
    """The text of a chat payload's messages, for estimating what it cost."""
    parts = []
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content
                               if isinstance(part, dict))
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


_EMBED_HINTS = ("embed", "bge-", "nomic", "minilm", "gte-", "-e5-", "rerank",
                "sentence-transformers")
_VISION_HINTS = ("vision", "llava", "-vl-", "-vl:", "vlm", "moondream",
                 "minicpm-v", "pixtral", "internvl", "cogvlm", "bakllava",
                 "gemma-3", "gemma3", "qwen2-vl", "qwen2.5-vl", "granite-vision")


#: An innermost block: an opening tag, no further opening tag inside it, then
#: its close. Removing those repeatedly unwraps nesting from the inside out —
#: the lazy form alone leaves "c</think>" behind on
#: "<think>a<think>b</think>c</think>", and the greedy form eats the text
#: between two separate blocks.
_THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>"
    r"(?:(?!<(?:think|thinking|reasoning)\b).)*?"
    r"</\1\s*>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_LINE = re.compile(r"^[ \t]*<(think|thinking|reasoning)\b",
                              re.IGNORECASE | re.MULTILINE)
_CLOSE_THINK = re.compile(r"</(think|thinking|reasoning)\s*>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove inline chain-of-thought from an answer.

    LM Studio reports reasoning in its own ``reasoning_content`` field, but
    llama.cpp and Ollama hand back a single string with the thinking inlined in
    <think> tags. Left in place it becomes the summary: the parser reads the
    model's musings as the answer.
    """
    # Repeatedly: "<think>a<think>b</think>c</think>" leaves "c</think>"
    # after one pass, because the lazy close matches the inner tag. Models do
    # nest them, and a stray "</think>" in the summary is the model's
    # scratchpad leaking into the answer.
    cleaned = text
    for _ in range(4):
        shorter = _THINK_BLOCK.sub("", cleaned)
        if shorter == cleaned:
            break
        cleaned = shorter
    # An unclosed tag that opens its own line, with no closing tag after it, is
    # a reasoning block the model never finished — drop it and everything
    # after. A tag inside a sentence ("the template renders <think> blocks") is
    # the file's own content being described, and cutting there would throw
    # away a correct answer. That distinction is the whole rule.
    match = _OPEN_THINK_LINE.search(cleaned)
    if match and not _CLOSE_THINK.search(cleaned, match.end()):
        cleaned = cleaned[:match.start()]
    return cleaned.strip()


_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def image_mime(data: bytes) -> str:
    """MIME type from the image's own bytes.

    The data: URI must state the real format. Labelling a JPEG as image/png
    makes some servers decode garbage or reject the request outright, so the
    extension is never consulted here.
    """
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    head = data[:512].lstrip()
    if head.startswith(b"<?xml") or head.startswith(b"<svg"):
        return "image/svg+xml"
    return "application/octet-stream"


#: Rerankers, which the name hints above file under "embeddings" too. A
#: reranker scores a (query, document) pair; asked for a vector it has none
#: to give — or, served through an embeddings endpoint, gives one that
#: means nothing. Kept apart so embed() never picks one.
_RERANK_HINTS = ("rerank", "cross-encoder")


def is_reranker(name: str) -> bool:
    """True when the model's name says it is a reranker."""
    low = (name or "").lower()
    return any(hint in low for hint in _RERANK_HINTS)


def _embeddable(texts) -> list:
    """The texts as a list, or ValueError for what cannot be embedded."""
    if isinstance(texts, (str, bytes)) or not isinstance(texts, (list, tuple)):
        raise ValueError(f"embed() takes a list of texts, not "
                         f"{type(texts).__name__}")
    for number, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"text {number} is {type(text).__name__}, not "
                             f"a string")
        if not text.strip():
            raise ValueError(f"text {number} is blank: there is nothing in "
                             f"it to embed")
    return list(texts)


def _checked_vectors(provider: str, vectors, count: int) -> list:
    """The vectors, as floats, or LLMError naming what is wrong with them.

    A vector that belongs to another text, or holds NaN — which json.loads
    reads without complaint — does not fail loudly later: it quietly makes
    every distance computed with it wrong. So the answer is checked whole
    before anyone stores it.
    """
    if not isinstance(vectors, list):
        raise LLMError(f"{provider}: the embeddings answer holds no "
                       f"list of vectors")
    if len(vectors) != count:
        raise LLMError(
            f"{provider}: {len(vectors)} vector{'s' * (len(vectors) != 1)} "
            f"for {count} text{'s' * (count != 1)}")
    out = []
    for number, vector in enumerate(vectors):
        if not isinstance(vector, list) or not vector:
            raise LLMError(f"{provider}: vector {number} is not a "
                           f"list of numbers")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LLMError(f"{provider}: vector {number} holds "
                               f"{repeatable(repr(value), 80)}, not a number")
            if not math.isfinite(value):
                raise LLMError(f"{provider}: vector {number} holds "
                               f"{value!r}, not a finite number")
        out.append([float(value) for value in vector])
    dims = {len(vector) for vector in out}
    if len(dims) > 1:
        raise LLMError(f"{provider}: the vectors have different "
                       f"dimensions ({', '.join(map(str, sorted(dims)))})")
    return out


def classify_model(name: str) -> str:
    """Guess a model's kind from its name, for servers that do not say."""
    low = name.lower()
    if any(h in low for h in _EMBED_HINTS):
        return "embeddings"
    if any(h in low for h in _VISION_HINTS):
        return "vlm"
    return "llm"


def _pick(models: list[ModelInfo], *, need_vision: bool,
          loaded_only: bool, trust_unknown: bool = True) -> Optional[ModelInfo]:
    """Best model for the job, or None if the constraints cannot be met.

    ``loaded is None`` counts as eligible under ``loaded_only`` when
    ``trust_unknown`` — a server that never reports residency has nothing to
    protect, and refusing there would cost a summary for no gain.

    It does *not* when the provider is one that normally reports residency
    and this time did not. There "unknown" is not "resident", and the
    tiebreak below makes the wrong guess actively: it orders by advertised
    context, which is a property of a model that is *downloaded*, so on a
    silent server it selects the largest model on the box — the one least
    likely to be the resident one, and the most expensive to load.
    """
    pool = [m for m in models if m.usable_for_chat]
    if loaded_only:
        pool = [m for m in pool
                if m.loaded is True or (trust_unknown and m.loaded is None)]
    if not pool:
        return None
    if need_vision:
        pool = [m for m in pool if m.sees_images]
        if not pool:
            return None
    # Order: confirmed-resident first; then the model built for the job — a
    # plain chat model for text, since a VLM carries a vision tower it will not
    # use; then the roomier context; then the name, so runs are reproducible.
    pool.sort(key=lambda m: (m.loaded is not True,
                             not _kind_fits(m, need_vision),
                             -(m.context or 0),
                             m.id))
    return pool[0]


def _kind_fits(model: ModelInfo, need_vision: bool) -> bool:
    return model.sees_images if need_vision else model.kind == "llm"


def _loose_match(wanted: str, models: list[ModelInfo]) -> Optional[ModelInfo]:
    """Find the model a server would resolve this name to.

    Mirrors what Ollama and LM Studio actually accept: case folding, and
    Ollama's implicit ':latest' tag. Used only to decide whether a request
    would cause a load — a name we cannot place is still sent as typed.
    """
    target = wanted.strip().casefold()
    if not target:
        return None
    for model in models:
        mid = model.id.casefold()
        if mid == target:
            return model
        if mid.split(":", 1)[0] == target.split(":", 1)[0] and (
                ":" not in target or ":" not in mid):
            return model
    return None
