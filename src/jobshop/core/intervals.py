"""Small helpers for half-open integer intervals ``(start, end)`` meaning [start, end).

Used by the solver (to turn availability minus downtime into blocked time) and by
KPIs (to measure available time). The validator deliberately does NOT use these:
it must stay independent of the code it checks.
"""

from __future__ import annotations

from collections.abc import Iterable

Interval = tuple[int, int]


def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Sort, drop empty intervals, and merge overlapping or touching ones."""
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract(base: Iterable[Interval], cut: Iterable[Interval]) -> list[Interval]:
    """Return the parts of ``base`` not covered by ``cut`` (both merged first)."""
    cuts = merge(cut)
    result: list[Interval] = []
    for start, end in merge(base):
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor or cut_start >= end:
                continue
            if cut_start > cursor:
                result.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if cursor < end:
            result.append((cursor, end))
    return result


def complement(free: Iterable[Interval], lo: int, hi: int) -> list[Interval]:
    """Return the parts of [lo, hi) that are NOT in ``free``."""
    return subtract([(lo, hi)], free)


def clip(intervals: Iterable[Interval], lo: int, hi: int) -> list[Interval]:
    """Restrict intervals to [lo, hi), dropping any that fall outside."""
    clipped = ((max(s, lo), min(e, hi)) for s, e in intervals)
    return [(s, e) for s, e in clipped if e > s]


def overlap_length(a: Interval, b: Interval) -> int:
    """Length of the intersection of two intervals (0 if they don't overlap)."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def total_length(intervals: Iterable[Interval]) -> int:
    return sum(end - start for start, end in intervals)
