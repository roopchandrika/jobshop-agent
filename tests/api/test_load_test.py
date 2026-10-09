"""The load-test script itself: its arithmetic, its verdict, and a tiny real run against the scripted demo."""

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("load_test", ROOT / "scripts" / "load_test.py")
lt = importlib.util.module_from_spec(spec)
sys.modules["load_test"] = lt          # dataclasses look their module up here
spec.loader.exec_module(lt)


@pytest.mark.parametrize("values, p, expected", [([], 50, 0.0), ([5], 95, 5), ([1, 2, 3, 4, 5], 50, 3), ([1, 2, 3, 4, 5], 100, 5), ([5, 1, 3], 0, 1)])
def test_percentiles(values, p, expected):
    assert lt.percentile(values, p) == expected


def test_a_summary_counts_failures_and_5xx_but_not_a_409_which_is_an_answer():
    s = lt.summarise([(200, 0.010), (200, 0.030), (409, 0.020), (500, 0.5), (0, 1.0)])
    assert s.count == 5 and s.errors == 2 and s.statuses == {200: 2, 409: 1, 500: 1, 0: 1}
    assert s.max_ms == 30.0                                         # failures do not distort the latency of real answers


def contention(**kw):
    base = dict(users=10, accepted=1, refused_busy=9, server_errors=0, turn_finished=True,
                state_during_turn=lt.Latency(), busy_answers=0)
    return lt.Contention(**{**base, **kw})


def test_the_verdict_is_empty_when_exactly_one_chat_is_accepted_and_the_rest_are_told_to_wait():
    assert lt.verdict({"/api/state": lt.Latency(count=5)}, contention()) == []


@pytest.mark.parametrize("change, expected", [
    ({"accepted": 2, "refused_busy": 8}, "exactly 1 chat message accepted, got 2"),
    ({"accepted": 0, "refused_busy": 10, "turn_finished": False}, "exactly 1 chat message accepted, got 0"),
    ({"refused_busy": 5}, "neither 202 nor 409"),
    ({"server_errors": 3, "refused_busy": 6}, "3 chat requests failed or returned 5xx"),
    ({"turn_finished": False}, "never finished"),
    ({"state_during_turn": lt.Latency(errors=2)}, "reads failed while a turn was running"),
])
def test_each_way_the_web_layer_can_misbehave_is_reported(change, expected):
    assert any(expected in p for p in lt.verdict({"/api/state": lt.Latency()}, contention(**change)))


def test_read_errors_are_reported_per_endpoint():
    assert lt.verdict({"/healthz": lt.Latency(errors=4)}, contention()) == ["/healthz: 4 failed or 5xx responses"]


def test_a_refused_connection_is_a_zero_status_not_an_exception():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                                    # nothing listens here once the socket closes
    status, took, body = lt.request(f"http://127.0.0.1:{port}", "GET", "/healthz", timeout=1)
    assert status == 0 and body == b"" and took >= 0


def test_a_real_run_against_the_scripted_demo_passes_and_reports_numbers(capsys):
    assert lt.main(["--spawn", "--users", "4", "--requests", "3", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["problems"] == [] and out["contention"]["accepted"] == 1 and out["contention"]["refused_busy"] == 3
    assert out["reads"]["/healthz"]["count"] == 12 and out["reads"]["/healthz"]["errors"] == 0


def test_the_text_report_names_each_section(capsys):
    assert lt.main(["--spawn", "--users", "3", "--requests", "2"]) == 0
    text = capsys.readouterr().out
    for section in ["Reads under concurrency", "Chat contention (3 users", "Reads while that turn ran", "OK: the web layer behaved."]:
        assert section in text


def test_bad_arguments_are_rejected(capsys):
    for argv in (["--spawn", "--users", "1"], ["--spawn", "--requests", "0"], []):
        with pytest.raises(SystemExit):
            lt.main(argv)
