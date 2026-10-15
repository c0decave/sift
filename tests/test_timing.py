"""The helpers the timing tests measure with (_timing).

sift's timing tests read the wall clock against a fixed number of seconds,
and under a host load of 27 to 49 on 20 cores five of them failed on 0.1.73
while the code they time had not changed. On this host a process's CPU time
is no steadier than the wall clock: a fixed loop of 3,000,000 additions took
0.114 s of CPU at one moment and 0.734 s at another (best of three each) --
a virtual machine, whose /proc/stat counts no time as stolen. What holds is a
ratio of two runs taken in turn, which a slow moment slows alike -- so a
growth test compares a short input with a long one, and these are the helpers
it does it with -- least_cpus, cpu_ratio and linear_bound as a host application's
core.testing has them.
"""
import gc
import unittest

from _timing import (cpu_ratio, flat_bound, least_cpus, linear_bound,
                     quadratic_bound)


class _Clock:
    """A clock that moves only when a timed function says it spent time."""

    def __init__(self):
        self.now = 0.0
        self.calls = []

    def __call__(self):
        return self.now

    def spending(self, name, *costs):
        """A function that spends ``costs`` in turn, one a call (the last
        one again once they run out), and records that it ran."""
        left = list(costs)

        def run():
            self.calls.append(name)
            self.now += left.pop(0) if len(left) > 1 else left[0]
        return run


class LeastCpusTest(unittest.TestCase):

    def test_each_function_gets_its_least_time_in_its_order(self):
        clock = _Clock()
        best = least_cpus(clock.spending("a", 3.0, 1.0, 2.0),
                          clock.spending("b", 0.5, 0.75, 0.25),
                          runs=3, clock=clock)
        self.assertEqual([1.0, 0.25], best)

    def test_the_functions_take_turns(self):
        """A burst of load then falls on both sides alike: the long side's
        runs one after the other let one fall on all of them."""
        clock = _Clock()
        least_cpus(clock.spending("long", 1.0), clock.spending("short", 1.0),
                   runs=3, clock=clock)
        self.assertEqual(["long", "short"] * 3, clock.calls)

    def test_the_collector_is_off_while_a_function_runs_and_on_after(self):
        seen = []
        least_cpus(lambda: seen.append(gc.isenabled()), runs=2)
        self.assertEqual([False, False], seen)
        self.assertTrue(gc.isenabled())

    def test_a_function_that_raises_leaves_the_collector_on(self):
        def boom():
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            least_cpus(boom)
        self.assertTrue(gc.isenabled())

    def test_no_run_and_no_function_are_refused(self):
        with self.assertRaises(ValueError):
            least_cpus(lambda: None, runs=0)
        with self.assertRaises(ValueError):
            least_cpus()

    def test_one_run_is_that_runs_time(self):
        clock = _Clock()
        self.assertEqual([2.5], least_cpus(clock.spending("a", 2.5), runs=1,
                                           clock=clock))


class CpuRatioTest(unittest.TestCase):

    def test_the_ratio_is_the_long_sides_least_over_the_short_sides(self):
        clock = _Clock()
        ratio = cpu_ratio(clock.spending("long", 9.0, 8.0, 8.5),
                          clock.spending("short", 1.0, 2.0, 1.5),
                          runs=3, clock=clock)
        self.assertEqual(8.0, ratio)

    def test_a_short_side_the_clock_cannot_see_is_the_floor(self):
        clock = _Clock()
        ratio = cpu_ratio(clock.spending("long", 1e-3),
                          clock.spending("short", 0.0), clock=clock,
                          floor=1e-4)
        self.assertAlmostEqual(10.0, ratio)

    def test_the_ratio_is_of_process_time_by_default(self):
        """A sleep costs the process nothing: a clock that counted it would
        count every other process on the host too."""
        import time
        ratio = cpu_ratio(lambda: time.sleep(0.05), lambda: None, runs=1,
                          floor=1e-3)
        self.assertLess(ratio, 25)


class LinearBoundTest(unittest.TestCase):

    def test_the_bound_is_halfway_between_linear_and_quadratic(self):
        """On a log scale: growth ** 1.5, as far from growth as from
        growth ** 2."""
        for growth in (2, 4, 8):
            with self.subTest(growth=growth):
                bound = linear_bound(growth)
                self.assertAlmostEqual(bound / growth, growth ** 2 / bound)
        self.assertAlmostEqual(22.627, linear_bound(8), places=3)

    def test_a_growth_of_one_or_less_has_no_bound(self):
        for growth in (1, 0.5, 0, -2):
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                linear_bound(growth)


class FlatBoundTest(unittest.TestCase):
    """For work cut at a bound: past the cut, a longer input must not cost
    more, and one read past the cut costs growth times as much."""

    def test_the_bound_is_halfway_between_flat_and_linear(self):
        for growth in (2, 4, 8):
            with self.subTest(growth=growth):
                bound = flat_bound(growth)
                self.assertAlmostEqual(bound / 1, growth / bound)
        self.assertAlmostEqual(2.828, flat_bound(8), places=3)

    def test_a_growth_of_one_or_less_has_no_bound(self):
        for growth in (1, 0.5, 0, -2):
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                flat_bound(growth)


class QuadraticBoundTest(unittest.TestCase):
    """For work known to grow with the square of its input, guarded against
    growing with the cube."""

    def test_the_bound_is_halfway_between_quadratic_and_cubic(self):
        for growth in (2, 4, 8):
            with self.subTest(growth=growth):
                bound = quadratic_bound(growth)
                self.assertAlmostEqual(bound / growth ** 2, growth ** 3 / bound)
        self.assertAlmostEqual(181.019, quadratic_bound(8), places=3)

    def test_a_growth_of_one_or_less_has_no_bound(self):
        for growth in (1, 0.5, 0, -2):
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                quadratic_bound(growth)


if __name__ == "__main__":
    unittest.main()
