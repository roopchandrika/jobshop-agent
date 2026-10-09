"""A limit on agent turns per minute and per hour, so a runaway client cannot spend the model budget without end."""

import threading

import pytest

from jobshop.api import server
from jobshop.api.ratelimit import DEFAULT_PER_HOUR, DEFAULT_PER_MINUTE, Limit, RateLimiter, limit_from_env
from tests.api.conftest import Web, proposal_script
from tests.fake_llm import submit


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def limiter(per_minute=0, per_hour=0):
    clock = Clock()
    return RateLimiter(Limit(per_minute, per_hour), clock), clock


# -- the limiter itself ---------------------------------------------------------------------------------------------------------------


def test_requests_within_the_minute_budget_pass_and_the_next_is_told_how_long_to_wait():
    rl, clock = limiter(per_minute=3)
    assert [rl.try_acquire() for _ in range(3)] == [0, 0, 0]
    clock.now += 20
    assert rl.try_acquire() == 40                     # the oldest of the three leaves the window in 40 s


def test_the_budget_comes_back_as_the_window_slides():
    rl, clock = limiter(per_minute=2)
    rl.try_acquire()
    clock.now += 30
    rl.try_acquire()
    assert rl.try_acquire() == 30
    clock.now += 30
    assert rl.try_acquire() == 0                      # the first has now aged out
    assert rl.try_acquire() == 30                     # the second (at +30) is still in the window


def test_a_refused_request_is_not_counted_so_waiting_the_stated_time_is_enough():
    rl, clock = limiter(per_minute=1)
    assert rl.try_acquire() == 0
    for _ in range(5):
        assert rl.try_acquire() == 60
    clock.now += 60
    assert rl.try_acquire() == 0


def test_the_hourly_budget_applies_even_when_the_minute_budget_has_room():
    rl, clock = limiter(per_minute=2, per_hour=3)
    for _ in range(3):
        assert rl.try_acquire() == 0
        clock.now += 61                                # spaced out, so the minute window never fills
    assert rl.try_acquire() == 3600 - 183              # the first of the three leaves the hour window at t = 3600
    clock.now += 3600 - 183
    assert rl.try_acquire() == 0


def test_the_wait_is_rounded_up_to_whole_seconds_and_is_at_least_one():
    rl, clock = limiter(per_minute=1)
    rl.try_acquire()
    clock.now += 59.2
    assert rl.try_acquire() == 1
    clock.now += 0.5
    assert rl.try_acquire() == 1


def test_a_fractional_wait_is_rounded_up_not_down():
    rl, clock = limiter(per_minute=1)
    rl.try_acquire()
    clock.now += 30.5
    assert rl.try_acquire() == 30                     # 29.5 s remain; waiting 29 would be refused again


def test_old_entries_are_forgotten_so_memory_does_not_grow_without_bound():
    rl, clock = limiter(per_minute=5, per_hour=100)
    for _ in range(50):
        rl.try_acquire()
        clock.now += 100
    assert len(rl._starts) <= 37                       # about an hour of entries at one per 100 s


def test_zero_means_no_limit_in_that_window():
    rl, _ = limiter(per_minute=0, per_hour=0)
    assert all(rl.try_acquire() == 0 for _ in range(1000))
    rl, clock = limiter(per_minute=0, per_hour=2)
    assert [rl.try_acquire() for _ in range(2)] == [0, 0] and rl.try_acquire() > 0


def test_the_limiter_is_safe_under_threads():
    rl = RateLimiter(Limit(per_minute=50, per_hour=0), Clock())
    results: list[float] = []
    lock = threading.Lock()

    def worker():
        for _ in range(20):
            r = rl.try_acquire()
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(0) == 50 and len(results) == 160


@pytest.mark.parametrize("args", [(-1, 0), (0, -1)])
def test_negative_limits_are_rejected(args):
    with pytest.raises(ValueError, match="cannot be negative"):
        Limit(*args)


def test_an_hourly_limit_below_the_minute_limit_is_a_mistake_not_a_silent_surprise():
    with pytest.raises(ValueError, match="could never be reached"):
        Limit(per_minute=10, per_hour=5)
    Limit(per_minute=0, per_hour=5)


def test_limits_come_from_the_environment_with_safe_defaults():
    assert limit_from_env({}) == Limit(DEFAULT_PER_MINUTE, DEFAULT_PER_HOUR)
    assert limit_from_env({"JOBSHOP_RATE_LIMIT_PER_MIN": "3", "JOBSHOP_RATE_LIMIT_PER_HOUR": "0"}) == Limit(3, 0)
    with pytest.raises(ValueError, match="JOBSHOP_RATE_LIMIT_PER_MIN must be a whole number"):
        limit_from_env({"JOBSHOP_RATE_LIMIT_PER_MIN": "many"})


# -- in the web app ------------------------------------------------------------------------------------------------------------------


def test_the_chat_endpoint_answers_429_with_retry_after_once_the_budget_is_spent_and_never_calls_the_model(ctx):
    w = Web(ctx, [submit(summary="One."), submit(summary="Two.")], rate_limit=Limit(per_minute=2, per_hour=0))
    assert w.chat("first").status_code == 202 and w.chat("second").status_code == 202
    calls_before = len(w.fake.requests)
    refused = w.chat("third", wait=False)
    assert refused.status_code == 429 and 1 <= int(refused.headers["Retry-After"]) <= 60
    assert "Try again in" in refused.json()["detail"] and len(w.fake.requests) == calls_before
    state = w.state()
    assert state["busy"] is False and all(m.get("text") != "third" for m in state["transcript"])      # nothing recorded, nothing stuck


def test_a_refused_message_does_not_leave_the_assistant_locked(ctx):
    w = Web(ctx, [submit(summary="One.")], rate_limit=Limit(per_minute=1, per_hour=0))
    w.chat("first")
    assert w.chat("again", wait=False).status_code == 429
    assert w.chat("and again", wait=False).status_code == 429        # 429 again, not 409: the lock was released
    assert w.http.get("/api/gantt").status_code == 200


def test_a_message_refused_because_the_assistant_is_busy_does_not_use_up_the_budget(ctx):
    started, release = threading.Event(), threading.Event()

    def hold(kwargs):
        started.set()
        assert release.wait(10)
        return submit(summary="Done waiting.")

    w = Web(ctx, [hold, submit(summary="Second.")], rate_limit=Limit(per_minute=2, per_hour=0))
    first = w.chat("first", wait=False).json()["turn_id"]
    assert started.wait(5)
    try:
        for _ in range(5):
            assert w.post("/api/chat", {"message": "busy"}).status_code == 409
    finally:
        release.set()
    w.wait(first)
    assert w.chat("second").status_code == 202                         # the five busy refusals cost nothing
    assert w.chat("third", wait=False).status_code == 429


def test_the_app_has_no_limit_unless_the_server_sets_one(ctx):
    w = Web(ctx, [submit(summary=str(i)) for i in range(15)])
    assert all(w.chat(f"m{i}").status_code == 202 for i in range(15))


def test_the_server_turns_a_bad_rate_limit_setting_into_a_configuration_error(monkeypatch, capsys):
    monkeypatch.setattr(server, "load_dotenv", lambda: None)
    monkeypatch.setenv("JOBSHOP_RATE_LIMIT_PER_MIN", "lots")
    monkeypatch.setenv("JOBSHOP_MEMORY", "off")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert server.main(["--port", "0", "--solve-seconds", "1"]) == 2
    assert "JOBSHOP_RATE_LIMIT_PER_MIN must be a whole number" in capsys.readouterr().err


def test_proposal_scripts_still_work_with_a_limit_in_place(ctx):
    w = Web(ctx, proposal_script(), rate_limit=Limit(per_minute=5, per_hour=50))
    assert w.chat().status_code == 202 and w.state()["proposal"] is not None
