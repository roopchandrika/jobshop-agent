import pytest

from jobshop.evals.stats import mean, percentile, sign_test_p, wilson_interval


@pytest.mark.parametrize("passed, total, low, high", [
    (5, 10, 0.237, 0.763),
    (0, 10, 0.0, 0.278),
    (10, 10, 0.722, 1.0),
    (29, 29, 0.883, 1.0),
    (1, 2, 0.095, 0.905),
])
def test_wilson_interval_matches_the_textbook_values(passed, total, low, high):
    lo, hi = wilson_interval(passed, total)
    assert lo == pytest.approx(low, abs=0.002) and hi == pytest.approx(high, abs=0.002)


def test_wilson_with_no_data_is_everything_and_never_leaves_0_to_1():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    for k in range(0, 30):
        lo, hi = wilson_interval(k, 29)
        assert 0.0 <= lo <= k / 29 <= hi <= 1.0


def test_more_data_narrows_the_interval():
    narrow, wide = wilson_interval(50, 100), wilson_interval(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


@pytest.mark.parametrize("a, b, p", [
    (0, 0, 1.0),          # no disagreements: no evidence either way
    (4, 0, 0.125),        # 4 wins to none is still plausible by chance
    (10, 0, 2 / 1024),    # 10 to none is not
    (5, 5, 1.0),
    (3, 1, 0.625),
])
def test_sign_test_p_values(a, b, p):
    assert sign_test_p(a, b) == pytest.approx(p)
    assert sign_test_p(b, a) == pytest.approx(p)  # symmetric: it does not matter which is "A"


def test_percentiles_interpolate():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([7], 95) == 7 and percentile([], 50) == 0.0
    assert percentile([4, 1, 3, 2], 0) == 1 and percentile([4, 1, 3, 2], 100) == 4  # input order does not matter


def test_mean_of_nothing_is_zero_not_an_error():
    assert mean([]) == 0.0 and mean([1, 2, 6]) == 3
