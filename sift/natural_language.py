"""Which natural language a text is written in: German, English, or neither.

Summary.language is the *programming* language — "python" for a .py file.
Full-text search needs the other one: a German text indexed with English
stemming rules does not find "Rechnungen" under "Rechnung". A wrong answer
picks the wrong rules; no answer picks the neutral configuration, which still
finds every word as written. So this says "de" or "en" only when the text
says so, and None for everything else — another language, code, a table,
three words.

Function words, not vocabulary. "der", "und", "nicht", "the", "and",
"which" make up a third or more of any prose in their language and almost
none of anything else, and they do not depend on the subject: an invoice and
a recipe share them. Words both languages spell alike ("in", "was", "also",
"will", "die") count for neither, and so do the ones Dutch, Danish, French
or Spanish use just as often ("er" is Dutch as well, "her" Danish, "on"
French) — those languages share enough with the two answered here to be
called one of them otherwise.
"""
from __future__ import annotations

import re
from typing import Optional

#: Reads this much and no more. A language does not change after the first
#: hundred kilobytes of a file, and the scan is a regex over all of it.
SCAN_LIMIT = 100_000

GERMAN_WORDS = frozenset("""
der das und ist nicht ein eine einen einem einer eines zu dem mit sich auf
für fuer sie es auch dass daß wir ihr ich du nach bei aus wie noch nur oder
aber vor zur zum bis wird werden wurde wurden sind hat haben hatte hatten
kann können koennen muss müssen muessen soll sollte sollten über ueber durch
ohne um wenn weil dann dort hier schon sehr mehr doch jetzt immer sein seine
seinen ihre ihren ihrem unsere unser unseren diese dieser dieses diesem
diesen welche welcher keine kein keinen nichts alle allen beim vom im ins
sowie sondern damit zwischen während waehrend ob habe gibt mich mir dich dir
uns euch ihm ihn ihnen jede jeder jedes bereits bitte worden
wäre waere würde wuerde gewesen seit etwas dazu dafür dafuer darauf daher
deshalb jedoch sodass zwar sowohl
""".split())

ENGLISH_WORDS = frozenset("""
the and of to is that it for with as are be this by not or from have has
had but which they you we she his their been were would there what can could
if all when more about other some into than then them these those its our
your my should may any only because after before while where who whom how
why just very does did yes such each both being again out up down off most
many much same own too here shall must might whether until
upon within without
""".split())

#: Tokens: runs of letters. Digits and punctuation are not words, and a
#: table of numbers is not prose in any language.
_WORD = re.compile(r"[^\W\d_]+")

#: How much of the text the winning language's function words must make up.
#: Prose in either language runs at 30 to 50 percent; Dutch and Danish, the
#: nearest neighbours, reach 5 to 12 percent on German's list, and code,
#: tables and logs stay under 10 percent on both.
MIN_SHARE = 0.18
#: Distinct function words. One word repeated — a column called "the",
#: Danish "der" over and over — is not a language, and neither are three:
#: "with", "as" and "for" are Python's keywords, and a five-line function
#: made of them read as English.
MIN_DISTINCT = 4
#: Of the function words found, how many must belong to the winner. A German
#: text quoting an English sentence is German; half and half is neither.
MIN_LEAD = 0.75


def detect_natural_language(text: Optional[str]) -> "tuple[Optional[str], float]":
    """("de" | "en" | None, confidence between 0 and 1).

    None comes with 0.0. The confidence says how clearly the text reads as
    the language named: how far past MIN_SHARE its function words go, and
    how few of the other language's there are.
    """
    if not text:
        return None, 0.0
    sample = text[:SCAN_LIMIT]
    found_words = _WORD.findall(sample)
    if any(ch.islower() for ch in sample):
        # A word in capitals among lowercase ones is a keyword or an
        # acronym — SQL's SELECT, FROM, WHERE and ORDER BY — not a word of
        # the prose around it. A text written entirely in capitals keeps
        # them: there, they are the prose.
        found_words = [w for w in found_words
                       if len(w) < 2 or not w.isupper()]
    words = [w.lower() for w in found_words]
    if not words:
        return None, 0.0
    german = [w for w in words if w in GERMAN_WORDS]
    english = [w for w in words if w in ENGLISH_WORDS]
    if len(german) >= len(english):
        code, found, other = "de", german, len(english)
    else:
        code, found, other = "en", english, len(german)
    share = len(found) / len(words)
    lead = len(found) / (len(found) + other) if found else 0.0
    if (share < MIN_SHARE or len(set(found)) < MIN_DISTINCT
            or lead < MIN_LEAD):
        return None, 0.0
    # Two parts, each in 0..1: the share scaled so ordinary prose (a third
    # of it function words) reaches the top, and the lead over the other
    # language.
    confidence = min(1.0, share / 0.33) * lead
    return code, round(confidence, 2)
