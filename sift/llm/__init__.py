"""LLM backend factory and exports."""
from __future__ import annotations

from typing import Optional

from ..config import (Config, DEFAULT_LMSTUDIO_ENDPOINT,
                      DEFAULT_MLX_LM_ENDPOINT, DEFAULT_OLLAMA_ENDPOINT)
from .base import (CallUsage, EmbeddingResult, LLMClient, LLMError,
                   LLMResult, ModelInfo, NoVisionModel, ServerComplaint,
                   call_kind, recording_calls)
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient
from . import prompts

__all__ = ["CallUsage", "EmbeddingResult", "LLMClient", "LLMError",
           "LLMResult", "ModelInfo", "NoVisionModel", "ServerComplaint",
           "build_client", "call_kind", "prompts", "recording_calls"]


def build_client(config: Config) -> Optional[LLMClient]:
    """Construct an LLM client from config, or None if LLM use is disabled.

    ``provider='auto'`` probes Ollama, LM Studio, then MLX-LM and returns the
    first that answers. Returns None when nothing is reachable so the caller
    can fall back to heuristic summaries.
    """
    if config.no_llm or config.provider == "none":
        return None

    provider = config.provider
    if provider == "auto":
        return _auto(config)

    if provider == "ollama":
        return OllamaClient(config.resolved_endpoint("ollama"), config.model,
                            config.api_key, config.timeout,
                            config.allow_model_load, config.stream)
    if provider in ("lmstudio", "mlx-lm", "openai"):
        return OpenAICompatClient(config.resolved_endpoint(provider), config.model,
                                  config.api_key, config.timeout,
                                  config.allow_model_load, config.stream,
                                  provider=provider)
    raise LLMError(f"unknown provider: {provider!r}")


def _auto(config: Config) -> Optional[LLMClient]:
    ollama = OllamaClient(
        config.resolved_endpoint("ollama") if config.endpoint
        else DEFAULT_OLLAMA_ENDPOINT,
        config.model, config.api_key, config.timeout, config.allow_model_load,
        config.stream)
    if ollama.available():
        return ollama
    lmstudio = OpenAICompatClient(
        config.resolved_endpoint("lmstudio") if config.endpoint
        else DEFAULT_LMSTUDIO_ENDPOINT,
        config.model, config.api_key, config.timeout, config.allow_model_load,
        config.stream, provider="lmstudio")
    if lmstudio.available():
        return lmstudio
    mlx_lm = OpenAICompatClient(
        config.resolved_endpoint("mlx-lm") if config.endpoint
        else DEFAULT_MLX_LM_ENDPOINT,
        config.model, config.api_key, config.timeout, config.allow_model_load,
        config.stream, provider="mlx-lm")
    if mlx_lm.available():
        return mlx_lm
    return None
