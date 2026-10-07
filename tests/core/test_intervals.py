from jobshop.core import intervals


def test_merge_sorts_merges_overlaps_and_touching_and_drops_empty():
    assert intervals.merge([(5, 10), (0, 3), (3, 4), (9, 12), (7, 7)]) == [(0, 4), (5, 12)]


def test_subtract_removes_cut_parts_including_overlapping_cuts():
    base = [(0, 100)]
    cut = [(10, 20), (15, 30), (90, 120)]
    assert intervals.subtract(base, cut) == [(0, 10), (30, 90)]


def test_subtract_everything_leaves_nothing():
    assert intervals.subtract([(10, 20)], [(0, 100)]) == []


def test_subtract_with_no_cuts_returns_merged_base():
    assert intervals.subtract([(0, 5), (5, 10)], []) == [(0, 10)]


def test_complement_is_the_closed_time():
    assert intervals.complement([(10, 20)], 0, 30) == [(0, 10), (20, 30)]
    assert intervals.complement([], 0, 30) == [(0, 30)]


def test_clip_restricts_and_drops_outside():
    assert intervals.clip([(0, 10), (20, 30), (50, 60)], 5, 25) == [(5, 10), (20, 25)]


def test_overlap_length():
    assert intervals.overlap_length((0, 10), (5, 20)) == 5
    assert intervals.overlap_length((0, 10), (10, 20)) == 0  # half-open: touching is no overlap
    assert intervals.overlap_length((0, 10), (30, 40)) == 0


def test_total_length():
    assert intervals.total_length([(0, 10), (20, 25)]) == 15
