"""Grading through the Message Batches API: one submission, a poll, and a safe fallback to one-by-one grading."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.providers import forget_unsupported_models
from jobshop.evals import cli
from jobshop.evals.judge import judge_batch, judge_request
from jobshop.evals.oracle import OracleClient
from jobshop.evals.runner import run_suite
from tests.evals.conftest import EVALS
from tests.evals.test_judge import all_scores, grade
from tests.evals.test_judge import run as run_fixture  # noqa: F401  (the fixture, re-exported for this module)
from tests.helpers import FAST


@pytest.fixture(autouse=True)
def fresh():
    forget_unsupported_models()
    yield
    forget_unsupported_models()


def grade_for(params, score=5):
    """A grade covering exactly the criteria this request asks for (they differ between scenarios)."""
    enum = params["tools"][0]["input_schema"]["properties"]["grades"]["items"]["properties"]["criterion"]["enum"]
    return grade({c: score for c in enum})


class FakeBatchClient:
    """A messages client with batches. ``entries`` maps a custom_id index to a Message, or to a result type string for a failure."""

    def __init__(self, grades, polls=2, fail=(), sync_grade=None, create_error=None):
        self.grades, self.polls, self.fail = list(grades), polls, set(fail)
        self.sync_grade, self.create_error = sync_grade, create_error
        self.submitted: list[dict] = []
        self.cancelled = False
        self.sync_calls: list[dict] = []
        self.retrieves = 0
        self.messages = self
        self.batches = self

    # -- batches
    def create(self, *, requests=None, **kw):
        if self.create_error:
            raise self.create_error
        self.submitted = requests
        return SimpleNamespace(id="batch_1", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.retrieves += 1
        return SimpleNamespace(id=batch_id, processing_status="ended" if self.retrieves >= self.polls else "in_progress")

    def cancel(self, batch_id):
        self.cancelled = True

    def results(self, batch_id):
        for i, req in enumerate(self.submitted):
            if i in self.fail:
                yield SimpleNamespace(custom_id=req["custom_id"], result=SimpleNamespace(type="errored", message=None))
            else:
                message = self.grades[i % len(self.grades)] if self.grades else grade_for(req["params"])
                yield SimpleNamespace(custom_id=req["custom_id"], result=SimpleNamespace(type="succeeded", message=message))

    # -- the ordinary API, used for the fallback
    def create_sync(self, **kwargs):
        self.sync_calls.append(kwargs)
        return self.sync_grade or grade_for(kwargs)


class WithSync(FakeBatchClient):
    """``messages.create`` is the ordinary call; ``messages.batches.create`` is the batch one."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.messages = SimpleNamespace(create=self.create_sync, batches=SimpleNamespace(
            create=self.create, retrieve=self.retrieve, cancel=self.cancel, results=self.results))


def instant(_seconds):
    return None


def test_all_runs_are_graded_in_one_submission_and_polled_until_it_ends(run_fixture):
    client = WithSync([grade(all_scores(5))], polls=3)
    results = judge_batch(client, "j", [run_fixture, run_fixture, run_fixture], sleep=instant)
    assert len(client.submitted) == 3 and [r["custom_id"] for r in client.submitted] == ["judge-0", "judge-1", "judge-2"]
    assert client.retrieves == 3 and client.sync_calls == []
    assert [r.passed for r in results] == [True, True, True] and all(r.scores for r in results)


def test_each_request_is_the_same_request_the_one_by_one_judge_would_send(run_fixture):
    client = WithSync([grade(all_scores(5))], polls=1)
    judge_batch(client, "the-judge", [run_fixture], sleep=instant)
    params = client.submitted[0]["params"]
    assert params == judge_request(run_fixture, "the-judge") and params["tool_choice"] == {"type": "tool", "name": "submit_grade"}


def test_an_entry_the_batch_could_not_complete_is_graded_again_on_the_ordinary_api(run_fixture):
    client = WithSync([grade(all_scores(5))], polls=1, fail={1}, sync_grade=grade(all_scores(2)))
    results = judge_batch(client, "j", [run_fixture, run_fixture], sleep=instant)
    assert [r.passed for r in results] == [True, False]       # the second came back from the fallback with a different grade
    assert len(client.sync_calls) == 1


def test_a_batch_that_never_ends_is_cancelled_and_everything_is_graded_one_by_one(run_fixture):
    ticks = iter(range(0, 10_000, 100))
    client = WithSync([grade(all_scores(5))], polls=10**9, sync_grade=grade(all_scores(4)))
    results = judge_batch(client, "j", [run_fixture, run_fixture], timeout_s=250, sleep=instant, clock=lambda: next(ticks))
    assert client.cancelled and len(client.sync_calls) == 2 and all(r.passed for r in results)


def test_a_rejected_submission_falls_back_to_one_by_one_grading(run_fixture):
    response = httpx2.Response(400, request=httpx2.Request("POST", "http://x"))
    error = anthropic.BadRequestError("bad batch", response=response, body=None)
    client = WithSync([grade(all_scores(5))], create_error=error, sync_grade=grade(all_scores(5)))
    results = judge_batch(client, "j", [run_fixture], sleep=instant)
    assert len(client.sync_calls) == 1 and results[0].passed


def test_nothing_to_grade_makes_no_call(run_fixture):
    client = WithSync([grade(all_scores(5))])
    assert judge_batch(client, "j", []) == [] and client.submitted == []


def test_a_model_known_to_refuse_forced_tools_is_not_forced_in_the_batch(run_fixture):
    from jobshop.agent import providers

    providers._NO_FORCED_TOOL.add("picky")
    request = judge_request(run_fixture, "picky")
    assert "tool_choice" not in request and [t["name"] for t in request["tools"]] == ["submit_grade"]
    assert "calling the tool 'submit_grade'" in request["system"]


# -- through the runner ------------------------------------------------------------------------------------------------------------------


def test_the_runner_grades_every_run_in_one_batch_and_reports_the_same_results_in_the_same_order(scenarios, shops, kb):
    chosen = [scenarios["sd-01"], scenarios["q-04"], scenarios["sd-02"]]
    client = WithSync([], polls=1)
    seen = []
    results = run_suite(chosen, OracleClient, AgentConfig(model="oracle"), shops, FAST, judge=(client, "j"), batch_judge=True,
                        knowledge=kb, progress=lambda r: seen.append(r.id))
    assert [r.id for r in results] == ["sd-01", "q-04", "sd-02"] and seen == ["sd-01", "q-04", "sd-02"]
    assert len(client.submitted) == 3 and client.sync_calls == []
    assert all(r.checks["judge"]["passed"] is True for r in results)


def test_batch_and_one_by_one_grading_give_identical_results(scenarios, shops, kb):
    chosen = [scenarios["sd-01"], scenarios["q-04"]]
    batch_client = WithSync([], polls=1)
    one_by_one = FakeSyncOnly()
    a = run_suite(chosen, OracleClient, AgentConfig(model="oracle"), shops, FAST, judge=(batch_client, "j"), batch_judge=True, knowledge=kb)
    b = run_suite(chosen, OracleClient, AgentConfig(model="oracle"), shops, FAST, judge=(one_by_one, "j"), knowledge=kb)
    assert [(r.id, r.passed, r.checks["judge"]) for r in a] == [(r.id, r.passed, r.checks["judge"]) for r in b]


class FakeSyncOnly:
    def __init__(self):
        self.messages = self

    def create(self, **kw):
        return grade_for(kw)


def test_without_a_judge_the_batch_flag_does_nothing(scenarios, shops, kb):
    results = run_suite([scenarios["sd-01"]], OracleClient, AgentConfig(model="oracle"), shops, FAST, judge=None, batch_judge=True, knowledge=kb)
    assert results[0].passed and "judge" not in results[0].checks


def test_the_run_command_accepts_the_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert cli.main(["--evals-dir", str(EVALS), "run", "--oracle", "--batch-judge", "--only", "sd-01", "--out", str(tmp_path)]) == 0
