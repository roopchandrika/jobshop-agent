from pathlib import Path

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.evals.runner import execute, score
from jobshop.evals.scenario import load_scenarios
from jobshop.evals.shop import load_shop
from tests.fake_llm import FakeClient
from tests.helpers import FAST

EVALS = Path(__file__).resolve().parents[2] / "evals"


@pytest.fixture(scope="session")
def shop():
    return load_shop(EVALS / "shop.json")


@pytest.fixture(scope="session")
def scenarios():
    return {s.id: s for s in load_scenarios(EVALS / "scenarios")}


@pytest.fixture
def play(shop, scenarios):
    """play("sd-01", script) runs a scripted agent through the real harness and scores it (no judge)."""

    def run(scenario_id, script, judge=None):
        run_ = execute(scenarios[scenario_id], FakeClient(script), AgentConfig(model="fake"), shop, FAST)
        return score(run_, judge), run_

    return run


def failed(result):
    """The names of the checks that failed (not the ones that did not apply)."""
    return {name for name, c in result.checks.items() if c["passed"] is False}
