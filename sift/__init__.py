"""sift — recognize & summarize files with a local LLM (offline-first).

sift detects a file's true type (by extension *and* content), extracts its
text/structure with a purpose-built extractor, and asks a locally-hosted LLM
(Ollama, LM Studio, or MLX-LM) to classify and summarize it. Everything runs
on your machine; no data leaves the host.

The core has **zero third-party dependencies** — csv, eml, json, html, docx,
xlsx, pptx and image metadata are all parsed with the standard library. Optional
packages (pypdf, Pillow, PyYAML, rich, …) are used automatically when present to
raise fidelity, but nothing breaks when they are absent.
"""
from __future__ import annotations

__version__ = "0.1.93"
__all__ = ["__version__"]
