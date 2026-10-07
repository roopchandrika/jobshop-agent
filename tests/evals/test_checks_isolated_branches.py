"""One test per check branch that the broader bad-agent tests did not isolate (found by mutation testing)."""

from jobshop.agent.loop import AgentConfig
from jobshop.evals.runner import execute, score
from tests.evals.conftest import failed
from tests.evals.test_checks import D, M2_OUTAGE, proposal
from tests.fake_llm import FakeClient, message, submit, tool
from tests.helpers import FAST


def test_a_summary_without_a_question_does_not_count_as_clarifying(play):
    result, _ = play("am-01", [submit(summary="Which machine is going down?")])
    assert failed(result) == {"outcome"}
    assert result.checks["outcome"]["details"] == ["expected a clarifying question"]


def test_an_unrequested_edit_fails_outcome_even_when_the_tool_was_not_forbidden(play):
    # im-01 only forbids reschedule, so the tools check allows this edit; the outcome check must not.
    script = [message(tool("create_draft", "c1")),
              message(tool("simulate_downtime", "e1", draft_id=D, machine_id="M1", start="2026-01-05 11:00", end="2026-01-05 13:00")),
              submit(summary="Done.")]
    result, _ = play("im-01", script)
    assert failed(result) == {"outcome"} and "nothing should have been changed" in result.checks["outcome"]["details"][0]


def test_solving_a_feasible_draft_is_not_the_infeasible_outcome(play):
    # Only M2 goes down, which can be absorbed. The agent solves it, then submits without naming the draft,
    # so needs_approval is false and only the "was it really infeasible" branch can catch it.
    script = [message(tool("create_draft", "c1")),
              message(tool("simulate_downtime", "e1", draft_id=D, machine_id="M2", start="2026-01-05 10:00", end="2026-01-05 22:00")),
              message(tool("reschedule", "r1", draft_id=D)),
              submit(summary="No valid schedule exists.")]
    result, _ = play("im-04", script)
    assert any("a reschedule found a schedule" in d for d in result.checks["outcome"]["details"])


def test_a_rush_order_with_the_wrong_due_time_fails_changes(play):
    result, _ = play("ro-01", proposal(("add_rush_order", dict(draft_id=D, family="gear", due="2026-01-05 17:00", priority=5))))
    assert failed(result) == {"changes"}
    assert any("missing rush order" in d for d in result.checks["changes"]["details"])
    assert any("unexpected rush order" in d for d in result.checks["changes"]["details"])


def test_a_proposal_whose_solve_found_nothing_has_no_schedule_to_validate(shop, scenarios):
    # Reuse the impossible rush order, but expect a proposal: the draft exists and was solved, with no schedule found.
    scenario = scenarios["im-05"].model_copy(update={"expect": scenarios["im-05"].expect.model_copy(update={"outcome": "proposal"})})
    script = [message(tool("create_draft", "c1")),
              message(tool("add_rush_order", "e1", draft_id=D, family="shaft", due="2026-01-05 21:30", priority=5)),
              message(tool("reschedule", "r1", draft_id=D)), submit(summary="Done.", draft_id=D)]
    result = score(execute(scenario, FakeClient(script), AgentConfig(model="fake"), shop, FAST), None)
    assert result.checks["validator"] == {"passed": False, "details": ["no solved schedule to validate"]}


def test_a_live_plan_that_changes_during_the_run_is_noticed(shop, scenarios, monkeypatch):
    from jobshop.evals import runner

    contexts = []
    real = runner.fresh_context
    monkeypatch.setattr(runner, "fresh_context", lambda *a, **k: contexts.append(real(*a, **k)) or contexts[-1])

    def sabotage(kwargs):  # something commits behind the agent's back while it works
        contexts[0].store.set_clock(contexts[0].store.committed.instance.now + 30)
        return submit(summary="Done.")

    result = score(execute(scenarios["q-01"], FakeClient([sabotage]), AgentConfig(model="fake"), shop, FAST), None)
    assert "the live plan changed during the run" in result.checks["outcome"]["details"]


def test_numbers_that_appear_only_in_the_planners_request_may_be_repeated(play):
    # "11:00 to 13:00" is in the request but in no tool result (M9 does not exist, so nothing was applied).
    script = [message(tool("get_machine_status", "g1")),
              submit(summary="There is no M9, so I did not apply the 11:00 to 13:00 outage.")]
    result, _ = play("im-01", script)
    assert result.checks["numbers"]["passed"] is True, result.checks["numbers"]
