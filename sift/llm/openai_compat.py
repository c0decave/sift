"""OpenAI-compatible backend — LM Studio, llama.cpp server, vLLM, etc.

Default endpoint targets LM Studio (http://localhost:1234/v1). The /chat/
completions dialect and multimodal ``image_url`` content parts are shared by all
these servers.
"""
from __future__ import annotations

import base64
import json
from typing import Optional

from ..chunk import CHARS_PER_TOKEN
from .base import (COMPLAINT_CHARS, HTTPStatusError, LLMClient, LLMError,
                   LLMResult, ModelInfo, _Meter, _count, classify_model,
                   complaint_text, image_mime, prompt_text, repeatable,
                   strip_reasoning)


#: Upper bound for the one retry given to a model that spent its whole budget
#: reasoning. Large enough for any summary, small enough to stay a retry.
_REASONING_RETRY_CEILING = 8000


def _retry_budget(content: str, cut: bool, thought: int, max_tokens: int) -> int:
    """The budget worth asking again with, or 0 for "do not bother".

    Sized from the reasoning we just watched go by rather than doubled blindly:
    a model that spent 2,000 tokens thinking needs those *plus* room to answer,
    and one that merely wrote a long summary needs no retry at all — asking
    again would just produce another long summary.
    """
    if max_tokens >= _REASONING_RETRY_CEILING or not thought:
        return 0
    if content and not cut:
        return 0                        # a complete answer; nothing to fix
    spent = max(1, thought // CHARS_PER_TOKEN)
    if content:
        # The answer began, so the thinking finished and we watched all of it.
        # Its cost is known: give it back, plus the room the answer needs.
        wanted = spent + max_tokens
    else:
        # Nothing came back but thought, so the thinking was itself cut off.
        # What we measured is a floor, not the requirement.
        wanted = max(spent * 2 + max_tokens, max_tokens * 4)
    wanted = min(wanted, _REASONING_RETRY_CEILING)
    return wanted if wanted > max_tokens * 1.25 else 0


class OpenAICompatClient(LLMClient):
    provider = "lmstudio"

    def __init__(self, *args, provider: str = "lmstudio", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.provider = provider
        #: Whether to ask a stream for its token usage. A server that answers
        #: the option with a 400 naming it is asked once more without it,
        #: and never with it again.
        self._stream_usage = True

    def available(self) -> bool:
        return self.probe("/models")

    def list_models(self) -> list[str]:
        data = self._request("/models", method="GET")
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]

    def server_root(self) -> str:
        """Endpoint with the OpenAI ``/v1`` suffix removed.

        LM Studio's own REST API lives beside the compatibility layer, at
        ``/api/v0``, not under ``/v1``.
        """
        root = self.endpoint
        for suffix in ("/v1", "/v1/openai", "/openai/v1"):
            if root.endswith(suffix):
                return root[: -len(suffix)] or "/"
        return root

    def inventory(self) -> list[ModelInfo]:
        """Models with load state, via LM Studio's native /api/v0/models.

        ``/v1/models`` lists everything downloaded and says nothing about what
        is resident, so it cannot answer the only question that matters here:
        which model can we use without evicting another. Servers that are
        merely OpenAI-compatible (llama.cpp, vLLM) have no such endpoint; for
        them we fall back to names alone and report load state as unknown.
        """
        try:
            data = self._request("/api/v0/models", method="GET",
                                 base=self.server_root())
        except HTTPStatusError as exc:
            if exc.status not in (404, 405, 501):
                # The endpoint exists but would not answer. Falling back to
                # names alone reports every model as "residency unknown",
                # which reads as "the server is idle" — and idle means sift
                # may load. A hiccup must not become permission to evict.
                raise
            return super().inventory()      # not LM Studio; no such concept
        except LLMError:
            # A ServerComplaint too: LM Studio on a wrong base path answers
            # 200 {"error": "Unexpected endpoint..."}, and falling back to
            # names there is the same hiccup-as-permission refused above.
            raise
        entries = data.get("data", [])
        # The endpoint answered. If it still carries no load state, that is a
        # different situation from a server that has no such endpoint: this
        # one could have told us. An LM Studio build that omits `state` was
        # read as an idle server, and sift loaded the largest model on the
        # box over the top of whatever was resident.
        self._residency_withheld = bool(entries) and not any(
            m.get("state") is not None for m in entries)
        out: list[ModelInfo] = []
        for m in entries:
            mid = m.get("id")
            if not mid:
                continue
            state = m.get("state")
            kind = m.get("type") or classify_model(mid)
            out.append(ModelInfo(
                id=mid,
                loaded=(state == "loaded") if state is not None else None,
                kind=kind if kind in ("llm", "vlm", "embeddings") else "unknown",
                # What the model was loaded with, not what it could do. LM
                # Studio reports both, and a 262144-capable model loaded at
                # 16384 rejected a prompt sized from the larger number.
                # loaded_context_length only. The comment above says a
                # 262144-capable model loaded at 16384 rejects a prompt sized
                # from the larger number — and then the `or` used the larger
                # number anyway whenever the smaller was absent, while
                # _context_tokens reported it to the user as "reported by the
                # server for this model". None means "not reported", which
                # the caller already knows how to handle.
                context=m.get("loaded_context_length"),
                max_context=m.get("max_context_length"),
            ))
        return out or super().inventory()

    def _embed_request(self, model: str, texts: list) -> tuple:
        """POST /embeddings under the /v1 endpoint — the OpenAI dialect,
        which LM Studio documents. Each vector comes with the ``index`` of
        its text, and the dialect promises nothing about their order, so
        the index places it; one that places nothing, or a place twice, is
        an answer that cannot be trusted to match vectors to texts."""
        data = self._request("/embeddings", {"model": model, "input": texts})
        entries = data.get("data")
        if not isinstance(entries, list):
            raise LLMError(f"{self.provider}: the embeddings answer has no "
                           f"data list")
        vectors: list = [None] * len(entries)
        placed: set = set()
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise LLMError(f"{self.provider}: embeddings entry "
                               f"{position} is not an object")
            index = entry.get("index", position)
            if isinstance(index, bool) or not isinstance(index, int) \
                    or not 0 <= index < len(entries) or index in placed:
                raise LLMError(f"{self.provider}: embeddings entry "
                               f"{position} has the index "
                               f"{repeatable(repr(index), 80)}, which "
                               f"places no vector of this answer")
            placed.add(index)
            vectors[index] = entry.get("embedding")
        usage = data.get("usage")
        counted = _count(usage.get("prompt_tokens")) \
            if isinstance(usage, dict) else None
        return vectors, counted

    def generate(self, system: str, user: str, *,
                 images: Optional[list[bytes]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 800) -> LLMResult:
        model = self.ensure_model(need_vision=bool(images))
        if images:
            content: list = [{"type": "text", "text": user}]
            for img in images:
                b64 = base64.b64encode(img).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime(img)};base64,{b64}"},
                })
            user_content: object = content
        else:
            user_content = user
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.stream,
        }
        calls: list = []
        try:
            content, cut, thought = self._collect(payload, calls)
            content = strip_reasoning(content)
            first = calls[-1]
            first.outcome = "ok" if content else "empty"
            retry = _retry_budget(content, cut, thought, max_tokens)
            if retry:
                # The thinking ate the budget — either all of it, leaving no
                # answer at all, or enough of it that the answer stopped
                # mid-sentence. Without the first case, eleven map-reduce
                # parts in a row produced five minutes of reasoning and no
                # text; without the second, the operator's 40B spent 2,015
                # tokens thinking under a 2,000-token cap and delivered half
                # a summary.
                payload["max_tokens"] = retry
                try:
                    again, cut_again, thought_again = self._collect(
                        payload, calls, kind="retry")
                except BaseException:
                    if content:
                        first.outcome = "discarded"   # the answer is lost
                    raise
                again = strip_reasoning(again)
                second = calls[-1]
                # Only if it is actually better: a second truncation that
                # stops even earlier is not an improvement worth shipping.
                if again and (not cut_again or len(again) > len(content)):
                    if content:
                        first.outcome = "discarded"
                    content, cut, thought = again, cut_again, thought_again
                    second.outcome = "ok"
                else:
                    second.outcome = "discarded" if again else "empty"
            if not content:
                raise LLMError(f"{self.provider} returned empty content for "
                               f"model {model} (only reasoning, no answer)")
            return LLMResult(text=content, model=model,
                             provider=self.provider, truncated=bool(cut),
                             reasoning_chars=thought, stop_reason=cut,
                             calls=list(calls))
        finally:
            self._report_calls(calls)

    def _collect(self, payload: dict, calls: list,
                 kind: str = "chat") -> "tuple[str, str, int]":
        """One call, measured, and its CallUsage appended to ``calls``.

        A stream asks for its usage (stream_options.include_usage) — the
        only way an OpenAI-compatible stream reports one. A server that
        refuses the option with a 400 naming it is asked once more without
        it; any other 400 is the request's fault and is not repeated.
        """
        if not self.stream:
            return self._measured(self._collect_once, payload, calls, kind)
        if not self._stream_usage:
            return self._measured(self._collect_stream, payload, calls, kind)
        asking = dict(payload, stream_options={"include_usage": True})
        try:
            return self._measured(self._collect_stream, asking, calls, kind)
        except HTTPStatusError as exc:
            if exc.status != 400 or "stream_options" not in str(exc):
                raise
            self._stream_usage = False
            return self._measured(self._collect_stream, payload, calls, kind)

    def _measured(self, collect, payload: dict, calls: list,
                  kind: str) -> "tuple[str, str, int]":
        usage = self._new_call(calls, payload["model"], kind)
        meter = _Meter(usage, prompt_text(payload))
        try:
            content, cut, thought = collect(payload, meter)
            usage.stop_reason = cut or "stop"
            return content, cut, thought
        except BaseException as exc:
            meter.failed(exc)
            raise
        finally:
            meter.settle()

    def _collect_once(self, payload: dict,
                      meter: "_Meter") -> tuple[str, str, int]:
        data = self._request("/chat/completions", payload)
        meter.heard()
        meter.counted = _usage_of(data.get("usage"))
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.provider} returned no choices for model "
                           f"{payload['model']}")
        message = choices[0].get("message") or {}
        cut = "length" if choices[0].get("finish_reason") == "length" else ""
        reasoning = _as_text(message.get("reasoning_content", ""))
        content = _as_text(message.get("content", ""))
        meter.content.append(content)
        meter.reasoning.append(reasoning)
        return content, cut, len(reasoning)

    def _collect_stream(self, payload: dict,
                        meter: "_Meter") -> tuple[str, str, int]:
        """Assemble the answer from server-sent events.

        Only ``content`` is kept. Reasoning models stream their scratchpad in
        ``reasoning_content``; concatenating both would hand the parser the
        model thinking out loud instead of its answer.
        """
        parts: list[str] = []
        thought = 0
        pending = ""            # a data: value continued on the next line
        saw_data = False
        # Two different questions, and one flag used to answer both. `cut`
        # says the answer is incomplete; `finished` says the server told us
        # the answer was over. A connection closed cleanly mid-stream — which
        # is what a proxy does on its idle timeout, and this one's is 20
        # seconds — set neither, so half a summary came back with
        # truncated=False and was cached under a key that looks fresh
        # forever. The invoice's total and its UNPAID status were simply
        # gone, with nothing to say so.
        finished = False
        cut = ""
        # Lines that are not SSE at all, kept for the error message. A server
        # that answers a bad request with a plain JSON body says exactly what
        # is wrong, and throwing that away left the user with "no events".
        non_events: list[str] = []
        for line in self._stream_lines("/chat/completions", payload):
            meter.heard()
            end_of_event = not line.strip()
            if line.startswith("data:"):
                # Per the SSE spec consecutive data: lines form one event, and
                # a server may split a JSON object across them. Dropping the
                # halves loses content with no error at all.
                chunk = line[5:].strip()
                pending = f"{pending}\n{chunk}" if pending else chunk
            elif not end_of_event:
                if not saw_data and len(non_events) < 4:
                    non_events.append(line)
                continue                      # SSE comment or another field
            if not pending:
                continue
            if pending.strip() == "[DONE]":
                saw_data = True
                finished = True
                break
            try:
                event = json.loads(pending)
                if not isinstance(event, dict):
                    raise LLMError(
                        f"{self.provider} sent a stream event that is "
                        f"{type(event).__name__}, not an object: "
                        f"{repeatable(pending, 80)}")
            except ValueError:
                if not end_of_event:
                    continue                  # incomplete; more lines follow
                # A blank line ends the event, so this one really is corrupt.
                # Drop it rather than let it swallow every frame after it.
                pending = ""
                continue
            pending = ""
            saw_data = True
            if event.get("error") is not None:
                raise LLMError(f"{self.provider} stream error: "
                               f"{complaint_text(event['error'])}")
            # The usage comes in a last chunk of its own, with "choices" an
            # empty list — measured against LM Studio with include_usage.
            counted = _usage_of(event.get("usage"))
            if counted:
                meter.counted = counted
            choices = event.get("choices") if isinstance(event, dict) else None
            if choices is not None and not isinstance(choices, list):
                raise LLMError(
                    f"{self.provider} sent a stream event whose 'choices' is "
                    f"{type(choices).__name__}, not a list: "
                    f"{repeatable(str(choices), 80)}")
            for choice in choices or []:
                if not isinstance(choice, dict):
                    raise LLMError(
                        f"{self.provider} sent a stream choice that is "
                        f"{type(choice).__name__}, not an object")
                delta = choice.get("delta") or choice.get("message") or {}
                piece = _as_text(delta.get("content", ""))
                parts.append(piece)
                reasoning = _as_text(delta.get("reasoning_content", ""))
                thought += len(reasoning)
                meter.piece(piece, reasoning)
                if self.on_progress:
                    self.on_progress(len(piece), len(reasoning))
                reason = choice.get("finish_reason")
                if reason:
                    finished = True
                    if reason == "length":
                        cut = "length"
        if pending and pending != "[DONE]":
            raise LLMError(f"{self.provider} stream ended on an incomplete "
                           f"event: {repeatable(pending, 120)}")
        if not saw_data:
            # The server's own words, when it gave any. LM Studio answers a
            # wrong path with HTTP *200* and {"error": "Unexpected endpoint or
            # method. (POST /chat/completions)"} — so no status check fires,
            # the body is not an SSE frame, and the message a user got was
            # "sent no stream events", with the actual reason sitting unread
            # in the response. Every summary failed while --models kept
            # working, because discovery uses a different base path.
            said = _server_complaint(non_events)
            raise LLMError(
                f"{self.provider} sent no stream events for model "
                f"{payload['model']}"
                + (f" — the server said: {said}" if said else ""))
        if not finished:
            # Neither [DONE] nor a finish_reason: the bytes stopped and
            # nothing said the answer had. Reported rather than raised,
            # because a server that sends neither on purpose exists and
            # refusing its answers outright would be worse than saying the
            # answer may be short.
            cut = cut or "cut"
        return "".join(parts), cut, thought


def _usage_of(block) -> "Optional[dict]":
    """The token counts of an OpenAI-dialect "usage" block, or None."""
    if not isinstance(block, dict):
        return None
    counted = {key: block.get(key)
               for key in ("prompt_tokens", "completion_tokens")}
    details = block.get("completion_tokens_details")
    if isinstance(details, dict) and "reasoning_tokens" in details:
        counted["reasoning_tokens"] = details.get("reasoning_tokens")
    return counted


def _server_complaint(lines: "list[str]") -> str:
    """The server's own reason, out of a body that was not a stream.

    JSON first, because that is what every OpenAI-dialect server sends; then
    the raw text, bounded, because an HTML error page from something in
    between is still more use than silence.
    """
    body = "\n".join(lines).strip()
    if not body:
        return ""
    try:
        parsed = json.loads(body)
    except ValueError:
        return repeatable(body, COMPLAINT_CHARS)
    if isinstance(parsed, dict):
        return complaint_text(parsed.get("error", parsed.get("message", parsed)))
    return complaint_text(parsed)


def _as_text(content) -> str:
    """Content may be a string or a list of parts, depending on the server."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict))
    return content if isinstance(content, str) else ""
