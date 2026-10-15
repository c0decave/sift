"""A ceiling on the time one file may take, checked cooperatively.

Every pathological input this project has measured was a different mechanism
— a 2 MB quoted CSV cell, a character that NFKC expands eighteenfold, sixty
text-free epub chapters, a quadratic line lookup — and each was fixed where
it lived. That is the right fix for a bug you have found. It is not a bound
on the ones you have not: nothing in the tool said how long a file was
*allowed* to take, so every new pathology was unbounded until someone
measured it.

This says it once. A file gets a budget; the loops that can run long look at
the clock; when it is spent the summary says so instead of the process
disappearing into a header cell.

Cooperative, because it has to be. ``signal.setitimer`` fires only on the
main thread and the web server answers each request on its own, so the one
mechanism that could interrupt CPU-bound Python from outside is unavailable
exactly where the untrusted input arrives. The check points are therefore in
the loops themselves, which is also where the reason for stopping is known.

The budget is per *file*, not per run: a batch of a thousand files should
lose the one that is pathological, not the nine hundred and ninety-nine
after it.
"""
from __future__ import annotations

import contextlib
import contextvars
import time

#: No file the tool is meant for comes close to this. The measured worst
#: cases after the bounds in tabular, summarize and office are 0.2 s for a
#: 2 MB single-cell CSV, 5.6 s for a 3 MB NFKC bomb and 16 s for a 1 MB epub
#: bomb; an ordinary document is milliseconds. A minute is room for a slow
#: disk and a genuinely large spreadsheet, and still bounds an attack.
DEFAULT_SECONDS = 60.0


class DeadlineExceeded(Exception):
    """The time allowed for one file ran out at the next check point."""

    def __init__(self, seconds: float, where: str = ""):
        self.seconds = seconds
        self.where = where
        super().__init__(
            f"gave up after {seconds:g}s"
            + (f" while reading {where}" if where else ""))


_state: "contextvars.ContextVar[tuple[float, float] | None]" = (
    contextvars.ContextVar("sift_deadline", default=None))


@contextlib.contextmanager
def limit(seconds: "float | None"):
    """Give the enclosed work ``seconds``; ``None`` or 0 means no limit.

    A ContextVar rather than a module global: the web server handles each
    request on its own thread, and one upload's budget must not be another's.
    """
    if not seconds or seconds <= 0:
        yield
        return
    token = _state.set((time.monotonic() + seconds, seconds))
    try:
        yield
    finally:
        _state.reset(token)


@contextlib.contextmanager
def paused():
    """Stop the clock for work that is not sift's to bound.

    A model takes as long as it takes — thirty to ninety seconds a file is
    ordinary for a local one that reasons before answering — and it has its
    own ceiling in --timeout. Counting that against the per-file budget made
    a 174-byte CSV "exceed the 60s budget", which is a true sentence about
    the clock and a false one about the file.

    The budget exists to bound *sift's* loops, which are the ones nobody has
    measured yet. The model's time is measured, configured and visible.
    """
    state = _state.get()
    if state is None:
        yield
        return
    started = time.monotonic()
    try:
        yield
    finally:
        spent_here = time.monotonic() - started
        token = _state.set((state[0] + spent_here, state[1]))
        del token                       # the shift stays for this context


def started(seconds: "float | None"):
    """A budget of ``seconds`` from now, to be put in force later with
    ``resumed``; ``None`` or 0 is no budget of its own.

    For work handed out a piece at a time -- the rows of a table, read as
    the caller asks for them. ``limit`` cannot bound it: the budget it sets
    lives in the caller's context while the work waits between two pieces,
    so the caller's own code would run under it, and the work would run
    under whatever the caller set when it asks for the next piece.
    """
    if not seconds or seconds <= 0:
        return None
    return (time.monotonic() + seconds, seconds)


@contextlib.contextmanager
def resumed(budget):
    """Run the enclosed piece of work under ``budget`` (from ``started``),
    its clock running since it was started; ``None`` leaves the budget in
    force as it is."""
    if budget is None:
        yield
        return
    token = _state.set(budget)
    try:
        yield
    finally:
        _state.reset(token)


def check(where: str = "") -> None:
    """Raise if the budget is spent. Cheap enough for a per-row loop.

    ``where`` names the work in progress, so the message can say what the
    file was doing rather than only that it stopped.
    """
    state = _state.get()
    if state is not None and time.monotonic() > state[0]:
        raise DeadlineExceeded(state[1], where)


def spent() -> bool:
    """Whether the budget is gone, without raising — for loop conditions."""
    state = _state.get()
    return state is not None and time.monotonic() > state[0]


def seconds() -> "float | None":
    """The budget in force, for a message that has to quote it."""
    state = _state.get()
    return state[1] if state else None


def remaining() -> "float | None":
    """The seconds left of the budget in force, 0 once it is spent; None
    where there is none -- for work that watches the clock from another
    thread, as a timer."""
    state = _state.get()
    if state is None:
        return None
    return max(0.0, state[0] - time.monotonic())
