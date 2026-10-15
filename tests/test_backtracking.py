"""Every compiled pattern in the project, against the inputs that blow up.

Four times now a pattern in this project has taken seconds or minutes on an
input a few kilobytes long, and every one was the same shape: two unbounded
whitespace runs in a row, which can split one run of spaces between them in
combinatorially many ways. Each was found by accident, after it shipped:

  _BASE64_WRAPPED   200 kB without spaces        441 s
  _SECRET_IN_SENTENCE   3 kB of spaces            85 s
  _BARE_DIRECTIVE   "Say" + 60,000 spaces        148 s
  (and a quadratic line lookup before those)

The cure each time was to bound the runs. The cure for *the next one* is
this file: it finds every ``re.Pattern`` reachable from the package and runs
each against inputs designed to make a backtracking engine work — long runs
of one character, of two alternating, of words, and of the literal prefixes
the patterns themselves look for.

The threshold is deliberately loose. This is not a performance test; a
pattern that takes half a second on 30 kB is doing linear work with a large
constant, which is a different conversation. A search above a second on this
input is timed again against an eighth of the same bait, the two in turn:
super-linear is what grows faster than linear_bound(GROWTH), and
super-linear on attacker-supplied text is the bug this catches. A second
alone no longer tells: the slowest honest pattern takes half of one on a
quiet core, and the slowest search took 0.87 s of CPU at a host load of 40.
"""
import importlib
import pkgutil
import re
import time
import unittest

import sift

from _timing import cpu_ratio, linear_bound

#: Long enough that super-linear behaviour is unmistakable and linear
#: behaviour is still fast. The known failures were seconds-to-minutes here.
LENGTH = 30_000

#: A search past a second of CPU on 30 kB is timed again for its growth.
#: The slowest honest pattern, _EMPTY_VOCABULARY, takes 0.49 s of it on a
#: quiet core (0.1.74, linear: 8.6x the CPU for 8x the bait) and 0.67 s at a
#: host load of 24 -- a second a search is the screen, not the verdict.
BUDGET = 1.0

#: How many times the bait grows between the two searches timed against each
#: other: 30 kB is eight times 3,750 characters.
GROWTH = 8


def _bait(n=None):
    """Inputs shaped to make a backtracking engine explore, ``n`` characters
    of each run (LENGTH)."""
    n = LENGTH if n is None else n
    cases = [
        ("spaces", " " * n),
        ("tabs", "\t" * n),
        ("lowercase run", "a" * n),
        ("uppercase run", "A" * n),
        ("digits", "1" * n),
        ("hex", "abcdef0123" * (n // 10)),
        ("base64", "QUJDREVGR0g" * (n // 11)),
        ("words and spaces", "wort " * (n // 5)),
        ("underscores", "_" * n),
        ("dots", "." * n),
        ("hyphens", "-" * n),
        ("open quote", '"' + "x" * n),
        # A key in quotes, a subscript, "=>".
        ("open single quote", "'" + "x" * n),
        ("single quotes", "'" * n),
        ("quoted keys", "'a': " * (n // 5)),
        ("subscripts", "['" * (n // 2)),
        ("arrows", "=>" * (n // 2)),
        ("parens", "(" * (n // 2)),
        ("colons", ":" * n),
        ("equals", "=" * n),
        ("slashes", "/" * n),
        ("mixed word chars", "a1_ " * (n // 4)),
        ("umlauts", "äöü" * (n // 3)),
        ("cjk", "指令" * (n // 2)),
        ("newlines", "zeile\n" * (n // 6)),
    ]
    # And the same runs behind the literals the patterns look for: a pattern
    # only backtracks once its first alternative has matched, so the bait has
    # to get past it. Each of the four known failures needed this.
    for prefix in ("password", "secret", "ignore", "Say", "Never", "Report",
                   "token", "key", "Antworte", "erwähne", "do not",
                   "'password'", "cfg['token'] =", "'password' =>"):
        cases.append((f"{prefix} + spaces", prefix + " " * n))
        cases.append((f"{prefix} + words", prefix + " " + "wort " * (n // 5)))
    return cases


def _patterns():
    """Every compiled pattern reachable from the package, deduplicated."""
    modules = [sift]
    for info in pkgutil.walk_packages(sift.__path__, "sift."):
        try:
            modules.append(importlib.import_module(info.name))
        except Exception:                # an optional dependency is absent
            continue
    seen, found = set(), []
    for module in modules:
        for name in dir(module):
            obj = getattr(module, name, None)
            if isinstance(obj, re.Pattern) and id(obj) not in seen:
                seen.add(id(obj))
                found.append((f"{module.__name__}.{name}", obj))
    return found


class BacktrackingTest(unittest.TestCase):

    def test_no_pattern_is_super_linear_on_hostile_input(self):
        patterns = _patterns()
        # If this ever collapses to a handful, the discovery broke and the
        # test is passing by not looking.
        self.assertGreater(len(patterns), 50,
                           f"only found {len(patterns)} patterns to check")
        slow = []
        shorter = dict(_bait(LENGTH // GROWTH))
        for name, pattern in patterns:
            for how, text in _bait():
                started = time.process_time()
                try:
                    pattern.search(text)
                except Exception:
                    continue
                spent = time.process_time() - started
                if spent <= BUDGET:
                    continue
                short = shorter[how]
                ratio = cpu_ratio(lambda: pattern.search(text),
                                  lambda: pattern.search(short))
                if ratio >= linear_bound(GROWTH):
                    slow.append(f"{name} took {spent:.1f}s on {how}, "
                                f"{ratio:.1f}x the CPU for {GROWTH}x the text")
                    break               # one report per pattern is enough
        self.assertEqual([], slow, "\n".join(slow))


if __name__ == "__main__":
    unittest.main()
