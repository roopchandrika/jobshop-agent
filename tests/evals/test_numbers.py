import pytest

from jobshop.evals.numbers import extract


def numbers(text):
    return sorted(extract(text).numbers)


def test_plain_quantities():
    assert numbers("3 orders are late by 45 minutes") == [3, 45]
    assert numbers("1,430 weighted minutes and 92.5% utilization") == [92.5, 1430]


@pytest.mark.parametrize("text", ["O-101 and O-112", "M4 and D1", "RUSH-1-op2 on M3", "order O-103-op1"])
def test_identifiers_are_names_not_quantities(text):
    assert numbers(text) == []


def test_clock_times_are_times_not_numbers():
    f = extract("M4 is down 14:00-17:00, then 6:05 tomorrow")
    assert f.times == {"14:00", "17:00", "06:05"} and f.numbers == set()


def test_dates_are_dates_and_their_digits_are_not_numbers():
    f = extract("done by 2026-01-05 14:00")
    assert f.dates == {"2026-01-05"} and f.times == {"14:00"} and f.numbers == set()


@pytest.mark.parametrize("text, time", [("at 2pm", "14:00"), ("at 2 PM", "14:00"), ("12 am", "00:00"), ("12:30 p.m.", "12:30"), ("9:15am", "09:15")])
def test_am_pm_is_read_as_a_24_hour_time(text, time):
    assert extract(text).times == {time}


def test_words_are_not_numbers():
    assert numbers("two orders, about an hour") == []


def test_missing_from_lists_only_unsupported_claims():
    claimed = extract("5 orders late at 15:30 on 2026-01-06, also 3")
    seen = extract("late orders: 3. Done 2026-01-05 15:30")
    assert claimed.missing_from(seen) == ["5", "2026-01-06"]


def test_equal_values_match_however_they_are_written():
    assert extract("45").missing_from(extract("45.0 min")) == []
    assert extract("1430").missing_from(extract("1,430")) == []
