"""The few statistics needed to avoid over-reading a small eval. Plain formulas, no dependencies."""

from __future__ import annotations

import math


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a pass rate. Honest at small n and at 0% / 100% (unlike mean +- 2 sd)."""
    if total == 0:
        return (0.0, 1.0)
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sign_test_p(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value for 'A and B are equally likely to win a scenario they disagree on'.

    Only the scenarios where exactly one model passed carry information; ties say nothing about
    which is better. With few disagreements the p-value is necessarily large, which is the point.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. The 95th of 29 runs is nearly the maximum: read it that way."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
