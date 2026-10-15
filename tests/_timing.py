"""What a timing test measures with: CPU time of runs taken in turn, and the bound a cost of a given growth keeps.

A test that reads the wall clock against a fixed number of seconds reads the host as much as the code. Under a
load of 27 to 49 on 20 cores, five such tests of sift failed while the code they time was the same (0.1.73).
Process time does not rescue a fixed number of seconds on this host either -- a virtual machine, whose /proc/stat
counts no time as stolen: a fixed loop of 3,000,000 additions took 0.114 s of CPU at one moment and 0.734 s at
another, best of three each. What holds is a ratio of two runs taken in turn -- a slow moment slows both alike --
so a test of growth compares a short input with a long one (cpu_ratio) against the most its cost may grow:
linear_bound for a linear cost, flat_bound past a cut, quadratic_bound where the square is known and the cube is
what is guarded against.

least_cpus, cpu_ratio and linear_bound are a host application's core.testing's, where they were measured first.
"""
import contextlib
import gc
import itertools
import time
from unittest import mock


def least_cpus(*fns, runs=3, clock=None):
    """The least process time of each of ``fns``, in their order: in each of ``runs`` rounds every one runs once, in
    turn, with the garbage collector off while it runs. A load that comes and goes on the host then falls on all of
    them alike, where the runs of one after all the runs of the other let a burst fall on one side only (a host application
    0.2.85: an ods save whose cost does not grow came out 3.28 times against a bound of 3). A collection inside one
    run costs in proportion to the whole heap, not to the function's work. ``clock`` is time.process_time but for
    tests."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if not fns:
        raise ValueError("no function to time")
    clock = clock or time.process_time
    best = [None] * len(fns)
    for _ in range(runs):
        for i, fn in enumerate(fns):
            gc.collect()
            gc.disable()
            try:
                began = clock()
                fn()
                took = clock() - began
            finally:
                gc.enable()
            best[i] = took if best[i] is None else min(best[i], took)
    return best


def cpu_ratio(long_fn, short_fn, runs=3, clock=None, floor=1e-6):
    """``long_fn``'s least process time over ``short_fn``'s (at least ``floor`` seconds), the two run in turn
    ``runs`` times each (least_cpus). ``clock`` is time.process_time but for tests."""
    long_took, short_took = least_cpus(long_fn, short_fn, runs=runs, clock=clock)
    return long_took / max(short_took, floor)


def linear_bound(growth):
    """The most a linear cost may grow when its input grows ``growth`` times: growth ** 1.5, the geometric mean of
    linear (``growth``) and quadratic (``growth`` squared) growth, as far from either on a log scale. At eight times
    the input it is 22.6: linear 8, quadratic 64. A linear reading measures above ``growth`` under load -- a host application
    measured 6.4 at a low host load and 9.3 at a high one for four times the input, what the cache makes of the
    larger one -- so the bound is not ``growth`` itself."""
    if growth <= 1:
        raise ValueError("growth must be more than 1")
    return growth ** 1.5


def flat_bound(growth):
    """The most a cost may grow when its input grows ``growth`` times past a cut that makes it read no more of it:
    growth ** 0.5, the geometric mean of no growth (1) and linear growth (``growth``), as far from either on a log
    scale. Past the cut a longer input costs no more; read past it, where that reading is what costs, a linear one
    grows ``growth`` times. At eight times the input it is 2.83."""
    if growth <= 1:
        raise ValueError("growth must be more than 1")
    return growth ** 0.5


def quadratic_bound(growth):
    """The most a cost known to grow with the square of its input may grow when the input grows ``growth`` times:
    growth ** 2.5, the geometric mean of quadratic and cubic growth -- for a guard against the cube where the square
    is what the code does. At eight times the input it is 181: quadratic 64, cubic 512."""
    if growth <= 1:
        raise ValueError("growth must be more than 1")
    return growth ** 2.5


@contextlib.contextmanager
def given_up_then_given_time(module, name):
    """sift.deadline's clock for a file given up on whose check past the budget has its time: a second a reading, so
    any budget runs out at its first look, until ``module.name`` is called -- summarize._head_of, which that check
    reads the file with -- and standing where it was last read from then on, so the check's own budget
    (_check_past_the_budget), as long as the file's, never runs out. A tiny --time-budget was how a test sent a file
    down that path; the check past it now runs under the same budget and says "not checked" where it runs out."""
    ticks, read, standing = itertools.count(), [0.0], []
    real = getattr(module, name)

    def clock():
        if not standing:
            read[0] = float(next(ticks))
        return read[0]

    def reached(*args, **kwargs):
        standing.append(True)
        return real(*args, **kwargs)
    with mock.patch("sift.deadline.time.monotonic", clock), mock.patch.object(module, name, reached):
        yield
