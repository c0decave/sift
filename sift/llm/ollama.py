"""Ollama backend (http://localhost:11434) via its native /api/chat endpoint."""
from __future__ import annotations

import base64
import json
from typing import Optional

from .openai_compat import _retry_budget
from .base import (LLMClient, LLMError, LLMResult, ModelInfo, _Meter,
                   _count, classify_model, complaint_text, prompt_text,
                   repeatable, strip_reasoning)

# Projector families Ollama reports for multimodal models.
_VISION_FAMILIES = {"clip", "mllama", "qwen2vl", "qwen25vl", "gemma3",
                    "mistral3", "llama4", "siglip"}


def _counts_of(event: dict, meter: "_Meter") -> "Optional[dict]":
    """The token counts and timings of Ollama's last line, or None.

    Durations arrive in nanoseconds, per Ollama's API documentation.
    """
    if event.get("prompt_eval_count") is None \
            and event.get("eval_count") is None:
        return None
    for key, field in (("prompt_eval_duration", "prompt_eval_seconds"),
                       ("eval_duration", "eval_seconds")):
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and value > 0:
            setattr(meter.usage, field, round(value / 1e9, 3))
    return {"prompt_tokens": event.get("prompt_eval_count"),
            "completion_tokens": event.get("eval_count")}


def _ollama_kind(name: str, details: dict) -> str:
    """Model kind from /api/tags details, falling back to the name."""
    families = {str(f).lower() for f in (details.get("families") or [])}
    if details.get("family"):
        families.add(str(details["family"]).lower())
    if families & _VISION_FAMILIES:
        return "vlm"
    # No projector family listed: fall back to the name, which still catches
    # llava/moondream/*-vl tags that older Ollama builds do not annotate.
    return classify_model(name)


class OllamaClient(LLMClient):
    provider = "ollama"

    def available(self) -> bool:
        return self.probe("/api/tags")

    def list_models(self) -> list[str]:
        data = self._request("/api/tags", method="GET")
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def inventory(self) -> list[ModelInfo]:
        """Installed models, marking the ones currently held in memory.

        ``/api/tags`` lists what is installed; ``/api/ps`` lists what is
        resident. Requesting an installed-but-cold model makes Ollama load it
        and drop the resident one, so the two lists must not be confused.
        """
        tags = self._request("/api/tags", method="GET").get("models", [])
        try:
            running = self._request("/api/ps", method="GET").get("models", [])
        except LLMError:
            running = []                       # older Ollama: no /api/ps
            resident: Optional[set] = None
        else:
            resident = {m.get("name") or m.get("model") for m in running}
            resident |= {m.get("model") for m in running if m.get("model")}
            resident.discard(None)

        out: list[ModelInfo] = []
        for m in tags:
            name = m.get("name") or m.get("model")
            if not name:
                continue
            details = m.get("details") or {}
            out.append(ModelInfo(
                id=name,
                loaded=(name in resident) if resident is not None else None,
                kind=_ollama_kind(name, details),
                context=(m.get("context_length")
                         or details.get("context_length")),
            ))
        return out

    def _embed_request(self, model: str, texts: list) -> tuple:
        """POST /api/embed, as Ollama's API documents it: ``input`` a list,
        ``embeddings`` one vector per text in the same order, and
        ``prompt_eval_count`` the tokens it read."""
        data = self._request("/api/embed", {"model": model, "input": texts})
        return data.get("embeddings"), _count(data.get("prompt_eval_count"))

    def generate(self, system: str, user: str, *,
                 images: Optional[list[bytes]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 800) -> LLMResult:
        model = self.ensure_model(need_vision=bool(images))
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(img).decode("ascii") for img in images]
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, user_msg],
            "stream": self.stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        calls: list = []
        try:
            content, cut, thought = self._once(payload, calls)
            first = calls[-1]
            first.outcome = "ok" if content else "empty"
            # The same retry the OpenAI-compatible client has had since a
            # reasoning model spent its whole budget thinking and returned
            # nothing. Ollama is the *default* provider, and without this it
            # fell back to the heuristic exactly where LM Studio succeeded.
            retry = _retry_budget(content, cut, thought, max_tokens)
            if retry:
                payload["options"]["num_predict"] = retry
                try:
                    again, cut_again, thought_again = self._once(
                        payload, calls, kind="retry")
                except BaseException:
                    if content:
                        first.outcome = "discarded"
                    raise
                second = calls[-1]
                if again and (not cut_again or len(again) > len(content)):
                    if content:
                        first.outcome = "discarded"
                    content, cut, thought = again, cut_again, thought_again
                    second.outcome = "ok"
                else:
                    second.outcome = "discarded" if again else "empty"
            if not content:
                raise LLMError(f"ollama returned empty content for model "
                               f"{model}")
            return LLMResult(text=content, model=model,
                             provider=self.provider, truncated=bool(cut),
                             reasoning_chars=thought, stop_reason=cut,
                             calls=list(calls))
        finally:
            self._report_calls(calls)

    def _once(self, payload: dict, calls: list,
              kind: str = "chat") -> "tuple[str, bool, int]":
        usage = self._new_call(calls, payload["model"], kind)
        meter = _Meter(usage, prompt_text(payload))
        try:
            if self.stream:
                content, cut, thought = self._collect_stream(payload, meter)
            else:
                data = self._request("/api/chat", payload)
                meter.heard()
                meter.counted = _counts_of(data, meter)
                content = (data.get("message") or {}).get("content", "")
                cut = "length" if data.get("done_reason") == "length" else ""
                thinking = (data.get("message") or {}).get("thinking") or ""
                thought = len(thinking)
                meter.content.append(content if isinstance(content, str)
                                     else "")
                meter.reasoning.append(thinking)
            usage.stop_reason = cut or "stop"
        except BaseException as exc:
            meter.failed(exc)
            raise
        finally:
            meter.settle()
        # What strip_reasoning removed counts as thinking. `message.thinking`
        # is populated only when the request carries "think": true, which this
        # one does not — so on the default path a reasoning model inlines its
        # scratchpad in <think> tags, strip_reasoning deletes it, and the
        # retry saw thought == 0 and never fired. Ollama is the default
        # provider; measuring what was removed is the only figure available.
        stripped = strip_reasoning(content)
        return stripped, cut, max(thought, len(content) - len(stripped))

    def _collect_stream(self, payload: dict,
                        meter: "_Meter") -> "tuple[str, str, int]":
        """Assemble the answer from Ollama's newline-delimited JSON stream.

        Each line is a whole object, not an SSE frame, and reasoning models put
        their scratchpad in ``thinking`` — which is dropped, like LM Studio's
        ``reasoning_content``.
        """
        parts: list[str] = []
        saw_data = False
        # Same distinction as the SSE collector: `cut` is "the answer is
        # incomplete", `finished` is "the server said the answer was over".
        # Without the second, an NDJSON stream whose `done:true` line never
        # arrives — a proxy closing on its idle timeout — came back as a
        # complete summary.
        finished = False
        cut = ""
        thought = 0
        for line in self._stream_lines("/api/chat", payload):
            meter.heard()
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                raise LLMError(f"ollama sent a line that is not JSON: "
                               f"{repeatable(line, 120)}") from None
            if not isinstance(event, dict):
                raise LLMError(f"ollama sent a stream line that is "
                               f"{type(event).__name__}, not an object: "
                               f"{repeatable(line, 80)}")
            saw_data = True
            if event.get("error"):
                raise LLMError(f"ollama stream error: "
                               f"{complaint_text(event['error'])}")
            message = event.get("message")
            if message is not None and not isinstance(message, dict):
                raise LLMError(f"ollama sent a 'message' that is "
                               f"{type(message).__name__}, not an object")
            chunk = (message or {}).get("content", "")
            if isinstance(chunk, str):
                parts.append(chunk)
                thinking = (message or {}).get("thinking") or ""
                thought += len(thinking)
                meter.piece(chunk, thinking)
                if self.on_progress:
                    self.on_progress(len(chunk), len(thinking))
            if event.get("done"):
                finished = True
                meter.counted = _counts_of(event, meter)
                if event.get("done_reason") == "length":
                    cut = "length"
                break
        if not saw_data:
            raise LLMError(f"ollama sent no stream events for model "
                           f"{payload['model']}")
        if not finished:
            cut = cut or "cut"
        return "".join(parts), cut, thought
