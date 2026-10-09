"""A scenario's standing preferences reach the system prompt the model is given."""

from jobshop.agent.loop import AgentConfig
from jobshop.evals.runner import execute
from tests.fake_llm import FakeClient, submit
from tests.helpers import FAST


def system_prompt_for(scenario, shops):
    client = FakeClient([submit(summary="ok")])
    execute(scenario, client, AgentConfig(model="fake"), shops[scenario.shop], FAST)
    return " ".join(client.requests[0]["system"].split())


def test_a_scenarios_preferences_are_in_the_prompt(scenarios, shops):
    scenario = scenarios["mem-01"]
    system = system_prompt_for(scenario, shops)
    assert "Standing preferences" in system and f"1. {scenario.preferences[0]}" in system


def test_a_scenario_without_preferences_has_no_such_section(scenarios, shops):
    assert "Standing preferences" not in system_prompt_for(scenarios["q-01"], shops)


def test_the_reference_agent_passes_each_memory_scenario_and_the_goal_check_sees_the_preference_at_work(scenarios, shops, kb):
    from jobshop.evals.oracle import OracleClient
    from jobshop.evals.runner import score

    for sid, goal in (("mem-01", "earliest_finish"), ("mem-02", "fewest_moves"), ("mem-03", "earliest_finish")):
        scenario = scenarios[sid]
        assert scenario.expect.reschedule_goal == goal
        result = score(execute(scenario, OracleClient(scenario), AgentConfig(model="oracle"), shops[scenario.shop], FAST, knowledge=kb), None)
        assert result.passed, (sid, result.checks)
