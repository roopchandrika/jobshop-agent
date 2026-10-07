"""Check branches that a scripted agent cannot trigger by behaving badly, so the tests alter the store directly."""

from jobshop.core.models import Schedule
from jobshop.evals.checks import check_outcome, check_validator
from tests.evals.conftest import failed
from tests.evals.test_checks import D, sd01
from tests.fake_llm import submit


def test_a_change_to_the_live_plan_during_a_run_fails_outcome(play):
    _, run = play("sd-01", sd01())
    assert check_outcome(run).passed
    run.ctx.store.set_clock(run.ctx.store.committed.instance.now + 30)  # anything that bumps the live version
    result = check_outcome(run)
    assert result.passed is False and "the live plan changed during the run" in result.details


def test_the_validator_check_catches_a_solved_but_invalid_schedule(play):
    _, run = play("sd-01", sd01())
    assert check_validator(run).passed
    draft = run.ctx.store.draft(D)
    data = draft.schedule.model_dump()
    machine = data["assignments"][0]["machine_id"]
    for a in data["assignments"]:  # pile every operation onto one machine
        a["machine_id"] = machine
    draft.schedule = Schedule.model_validate(data)
    result = check_validator(run)
    assert result.passed is False and result.details


def test_an_invented_date_fails_numbers(play):
    result, _ = play("sd-01", sd01(summary="Everything finishes on 2026-01-09."))
    assert failed(result) == {"numbers"} and "2026-01-09" in result.checks["numbers"]["details"][0]


def test_an_answer_with_no_numbers_has_nothing_to_check(play):
    result, _ = play("am-01", [submit(summary="Which machine?", clarifying_question="Which machine, and from when?")])
    assert result.checks["numbers"]["passed"] is True
