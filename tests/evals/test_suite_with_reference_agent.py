"""The scripted reference agent must pass every scenario.

If it cannot, the scenario (or a check) is wrong, which is cheaper to find here than in a paid run.
This says nothing about any model.
"""

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.evals.oracle import OracleClient
from jobshop.evals.runner import execute, score
from tests.evals.conftest import EVALS, failed
from jobshop.evals.scenario import load_scenarios
from tests.helpers import FAST

IDS = [s.id for s in load_scenarios(EVALS / "scenarios")]


@pytest.mark.parametrize("scenario_id", IDS)
def test_the_reference_agent_passes(scenario_id, scenarios, shop):
    scenario = scenarios[scenario_id]
    run = execute(scenario, OracleClient(scenario), AgentConfig(model="oracle"), shop, FAST)
    result = score(run, judge=None)
    assert result.passed, {name: c["details"] for name, c in result.checks.items() if c["passed"] is False}


@pytest.mark.parametrize("scenario_id", [s.id for s in load_scenarios(EVALS / "scenarios") if s.expect.outcome == "infeasible"])
def test_infeasible_scenarios_are_proven_infeasible_not_just_unsolved(scenario_id, scenarios, shop):
    scenario = scenarios[scenario_id]
    run = execute(scenario, OracleClient(scenario), AgentConfig(model="oracle"), shop, FAST)
    [reschedule] = [c for c in run.calls if c.name == "reschedule"]
    assert reschedule.result["solve"]["status"] == "INFEASIBLE"  # a timeout (UNKNOWN) would make the scenario depend on speed


def test_the_reference_agent_is_run_through_the_real_loop_and_registry(scenarios, shop):
    scenario = scenarios["sd-01"]
    run = execute(scenario, OracleClient(scenario), AgentConfig(model="oracle"), shop, FAST)
    assert [c.name for c in run.calls] == ["create_draft", "simulate_downtime", "reschedule", "compare_schedules"]
    assert run.turn.final.needs_approval and run.turn.final.changes_made  # filled by the harness, from the store
    assert failed(score(run, None)) == set()
