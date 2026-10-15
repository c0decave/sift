"""Token estimation and text chunking for map-reduce summarization."""
from __future__ import annotations

from typing import Iterable

# A flat four characters per token is right for English prose and badly wrong
# for anything dense. Measured against the operator's tokenizer: prose came out
# at 4.48 characters per token, log lines with timestamps at 1.62 — so a budget
# sized by the flat rule was two and a half times too large, and the server
# rejected the request. These weights are fitted to those two measurements and
# deliberately biased to over-count: over-estimating costs a smaller chunk,
# under-estimating costs the whole request.
_CHARS_PER_LETTER_TOKEN = 4.5
_CHARS_PER_DIGIT_TOKEN = 0.8
_CHARS_PER_SYMBOL_TOKEN = 1.0
_CHARS_PER_SPACE_TOKEN = 5.0

#: Kept for callers that only need a rough figure for English text.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Token count estimated from what the text is made of.

    Digits and punctuation tokenize far worse than letters — an ISO timestamp
    is most of a dozen tokens — so the composition matters more than the
    length.
    """
    if not text:
        return 0
    letters = digits = spaces = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
        elif ch.isdigit():
            digits += 1
        elif ch.isspace():
            spaces += 1
    symbols = len(text) - letters - digits - spaces
    estimate = (letters / _CHARS_PER_LETTER_TOKEN
                + digits / _CHARS_PER_DIGIT_TOKEN
                + symbols / _CHARS_PER_SYMBOL_TOKEN
                + spaces / _CHARS_PER_SPACE_TOKEN)
    return max(1, int(estimate + 0.5))


def chars_per_token(text: str) -> float:
    """How many characters of *this* text fit in a token."""
    return len(text) / estimate_tokens(text) if text else CHARS_PER_TOKEN


def input_token_budget(
    context_tokens: int,
    max_output_tokens: int,
    prompt_overhead_tokens: int = 700,
) -> int:
    """Tokens of *content* that fit, leaving room for the prompt and answer.

    Compare token counts against this, not character counts against the char
    budget: the two are related by a ratio that depends on the text, so mixing
    them silently applies that ratio twice.
    """
    return max(128, context_tokens - max_output_tokens - prompt_overhead_tokens)


def input_char_budget(
    context_tokens: int,
    max_output_tokens: int,
    prompt_overhead_tokens: int = 700,
    text: str = "",
) -> int:
    """Characters of *content* we can include, leaving room for prompt+answer.

    Sized by the density of the text we are about to send, not by an average
    of English.
    """
    usable = context_tokens - max_output_tokens - prompt_overhead_tokens
    usable = max(512, usable)
    return int(usable * chars_per_token(text))


def chunk_text(text: str, max_chars: int, overlap: int = 200) -> list[str]:
    """Split *text* into chunks of at most *max_chars*, preferring clean breaks.

    Packs paragraphs greedily; paragraphs longer than a chunk are hard-split on
    line, then word, then character boundaries. A small overlap keeps context
    continuous across chunk edges.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text] if text else []

    overlap = max(0, min(overlap, max_chars // 4))
    units = _split_units(text, max_chars)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + len(unit) <= max_chars:
            current += unit
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + unit) if len(tail) + len(unit) <= max_chars else unit
    if current:
        chunks.append(current)
    return chunks


def _split_units(text: str, max_chars: int) -> Iterable[str]:
    """Yield paragraph-ish units, hard-splitting any that exceed *max_chars*."""
    for para in _keepends_split(text, "\n\n"):
        if len(para) <= max_chars:
            yield para
            continue
        for line in _keepends_split(para, "\n"):
            if len(line) <= max_chars:
                yield line
                continue
            yield from _hard_split(line, max_chars)


def _keepends_split(text: str, sep: str) -> list[str]:
    """Split on *sep* but keep the separators attached to the left piece."""
    parts = text.split(sep)
    out = [p + sep for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _hard_split(text: str, max_chars: int) -> Iterable[str]:
    words = text.split(" ")
    if len(words) > 1:
        buf = ""
        for word in words:
            piece = (word + " ")
            if len(piece) > max_chars:  # a single monster token
                if buf:
                    yield buf
                    buf = ""
                yield from _char_split(piece, max_chars)
            elif len(buf) + len(piece) <= max_chars:
                buf += piece
            else:
                yield buf
                buf = piece
        if buf:
            yield buf
    else:
        yield from _char_split(text, max_chars)


def _char_split(text: str, max_chars: int) -> Iterable[str]:
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]
