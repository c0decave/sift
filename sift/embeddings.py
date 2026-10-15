"""What gets embedded for a file.

A vector is only as good as the text it was made from. With a model's
summary that is the summary — SUMMARY, KEY POINTS and TAGS: the densest
statement of what the file is about, and about as long for a one-page note
as for a 50 MB log. Without one it is the head of the text: a heuristic
summary describes what sift measured ("A CSV table with 3 rows and 4
columns"), which is the same sentence for every table, whatever it holds.

Either way it has to fit the embedding model's context. Past it a server
truncates without saying so, or refuses the whole batch.

The vectors come from LLMClient.embed(); this module only decides the text.
"""
from __future__ import annotations

import re

from .chunk import estimate_tokens
from .extract.base import scrub

#: The context most embedding models of the BERT family have, and so the
#: assumption when the server does not report one.
DEFAULT_CONTEXT = 512

#: The share of the context the estimate may fill. estimate_tokens is
#: calibrated against chat-model tokenizers; the WordPiece tokenizers most
#: embedding models use split digits, compounds and non-Latin text finer.
_CONTEXT_SHARE = 0.8

#: How many characters of the text are looked at per token of budget: more
#: than any character class of estimate_tokens packs into one token, so the
#: fit below, not this cut, decides how much stays. A test holds the
#: relation to chunk's constants.
_HEAD_CHARS_PER_TOKEN = 6

#: What a cut without a blank or a line break to end at drops from its end:
#: a run of printable ASCII, which is what a key, a token or a password
#: consists of — Chinese or Japanese text has no blanks to cut at.
_TRAILING_WORD = re.compile(r"[\x21-\x7e]+$")


def embed_input(summary, text=None, context_tokens: int = DEFAULT_CONTEXT
                ) -> str:
    """The text to embed for a file: from its model summary when there is
    one, else from the head of ``text``, within ``context_tokens``.

    ``summary`` is a Summary or its to_dict(). ``text`` is what sift read —
    Summary.text, which is not redacted, so it is scrubbed here: a vector
    is not a place a secret can be read back from, but the request that
    makes it goes to a server, and the rule for what leaves sift is the
    same for every request. "" when there is nothing to embed.
    """
    if hasattr(summary, "to_dict"):
        summary = summary.to_dict()
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a Summary or its dict, not "
                         f"{type(summary).__name__}")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) \
            or context_tokens < 1:
        raise ValueError(f"context_tokens must be a positive whole number, "
                         f"not {context_tokens!r}")
    budget = max(1, int(context_tokens * _CONTEXT_SHARE))
    said = _head(_from_summary(summary), budget) \
        if summary.get("llm_used") else ""
    if not said:
        said = _head(text, budget)
    if not said:
        # No text either — an image, an archive, a file that could not be
        # read. What sift said about it is better than no vector at all.
        said = _head(_from_summary(summary), budget)
    return _fit(said, budget)


def _from_summary(summary: dict) -> str:
    """SUMMARY, KEY POINTS and TAGS, one per line, for _head to cut and
    scrub. tags_suspect stay out: the file may have asked for them."""
    parts = []
    text = summary.get("summary")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    for point in summary.get("key_points") or []:
        if isinstance(point, str) and point.strip():
            parts.append(point.strip())
    tags = [tag for tag in summary.get("tags") or []
            if isinstance(tag, str) and tag.strip()
            and scrub(tag, "tags") == tag]
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    return "\n".join(parts)


def _head(text, budget: int) -> str:
    """The start of ``text``, scrubbed, ending at a line break — no more
    of it than could fit ``budget``, so neither the scrub nor the fit
    works through a text they would throw away.

    Cut at a line break before scrubbing, not after: a cut through a
    ``TOKEN=value`` line leaves a piece of the value the patterns no longer
    recognise, and it would go out as it stands. Without a line break, at a
    blank; without either, before the run of ASCII the cut went through.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    head = text.lstrip()
    limit = budget * _HEAD_CHARS_PER_TOKEN
    if len(head) > limit:
        cut = head[:limit]
        end = cut.rfind("\n")
        if end <= 0:
            end = max(cut.rfind(" "), cut.rfind("\t"))
        head = cut[:end] if end > 0 else _TRAILING_WORD.sub("", cut)
    return scrub(head, "").strip()


def _fit(text: str, budget: int) -> str:
    """The longest prefix of ``text`` the estimate puts within ``budget``,
    ending at a blank where one is near. estimate_tokens only grows as a
    prefix does, so the search can halve."""
    if estimate_tokens(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    prefix = text[:low]
    blank = max(prefix.rfind(" "), prefix.rfind("\n"))
    if blank > low * 0.8:
        prefix = prefix[:blank]
    return prefix.rstrip()
