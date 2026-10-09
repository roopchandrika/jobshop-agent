from pathlib import Path

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.evals.runner import execute, score
from jobshop.evals.scenario import load_scenarios
from jobshop.evals.shop import load_shops
from jobshop.knowledge import KnowledgeBase
from tests.fake_llm import FakeClient
from tests.helpers import FAST

EVALS = Path(__file__).resolve().parents[2] / "evals"


@pytest.fixture(scope="session")
def shops():
    return load_shops(EVALS, {"default", "tight"})


@pytest.fixture(scope="session")
def shop(shops):
    """The default shop, for tests that do not care which one."""
    return shops["default"]


@pytest.fixture(scope="session")
def kb():
    """The plant documents shipped with the project (what the knowledge scenarios search)."""
    return KnowledgeBase.from_directory(EVALS.parent / "knowledge")


@pytest.fixture(scope="session")
def scenarios():
    return {s.id: s for s in load_scenarios(EVALS / "scenarios")}


@pytest.fixture
def play(shops, scenarios, kb):
    """play("sd-01", script) runs a scripted agent through the real harness and scores it (no judge)."""

    def run(scenario_id, script, judge=None):
        scenario = scenarios[scenario_id]
        run_ = execute(scenario, FakeClient(script), AgentConfig(model="fake"), shops[scenario.shop], FAST, knowledge=kb)
        return score(run_, judge), run_

    return run


def failed(result):
    """The names of the checks that failed (not the ones that did not apply)."""
    return {name for name, c in result.checks.items() if c["passed"] is False}
