"""A person approving a plan is told which goal it was solved for, from the draft, not from the model."""

import json

import pytest

from jobshop.mcp_server.admin import main as admin_main
from jobshop.api import views as web_views
from jobshop.api.app import STATIC
from jobshop.core.generator import generate_instance
from jobshop.core.solver import solve
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.functions import ToolContext
from jobshop.tools.outcome import draft_outcome
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.store import Store
from jobshop.tools.views import GOAL_LABELS
from tests.agent.test_cli import Session
from tests.fake_llm import message, submit, tool
from tests.helpers import FAST, SMALL


def solved_draft(registry, goal=None):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    registry.call("reschedule", {"draft_id": d, **({"goal": goal} if goal else {})})
    return d


def test_a_draft_remembers_the_goal_of_its_last_solve(ctx, registry):
    d = solved_draft(registry)
    assert ctx.store.draft(d).goal == "fewest_moves"
    registry.call("reschedule", {"draft_id": d, "goal": "earliest_finish"})
    assert ctx.store.draft(d).goal == "earliest_finish"
    assert draft_outcome(ctx, d).goal == "earliest_finish"


def test_the_goal_survives_the_state_file_and_old_state_files_still_load(ctx, registry):
    d = solved_draft(registry, "earliest_finish")
    data = json.loads(ctx.store._dump())
    assert data["drafts"][d]["goal"] == "earliest_finish"
    ctx.store._apply(data)
    assert ctx.store.draft(d).goal == "earliest_finish"

    del data["drafts"][d]["goal"]      # a file written before the goal existed
    ctx.store._apply(data)
    assert ctx.store.draft(d).goal == "fewest_moves"


def test_every_goal_has_wording_for_people():
    assert set(GOAL_LABELS) == {"fewest_moves", "earliest_finish"}
    assert "earliest finish" in GOAL_LABELS["earliest_finish"].split(",")[0]
    assert "fewest" in GOAL_LABELS["fewest_moves"].split(",")[0]


@pytest.mark.parametrize("goal", ["fewest_moves", "earliest_finish"])
def test_the_chat_approval_screen_names_the_goal(ctx, goal):
    script = [message(tool("create_draft", "a1")),
              message(tool("change_priority", "a2", draft_id="D1", order_id="O-101", priority=5),
                      tool("reschedule", "a3", draft_id="D1", goal=goal)),
              submit(summary="Done.", draft_id="D1")]
    s = Session(ctx, script, answers=["n"])
    s.chat.handle("Make O-101 urgent.")
    assert f"Solved for (recorded by the system): {GOAL_LABELS[goal]}" in s.output
    assert s.output.index("Solved for") < s.output.index("KPIs (from the solver")   # shown with the changes, before the numbers


def test_no_goal_line_when_nothing_is_proposed(ctx):
    s = Session(ctx, [submit(summary="Which order?", clarifying_question="Which order?")])
    s.chat.handle("Bump one.")
    assert "Solved for" not in s.output


@pytest.mark.parametrize("goal", ["fewest_moves", "earliest_finish"])
def test_the_admin_review_names_the_goal(tmp_path, goal):
    path = tmp_path / "state.json"
    inst = generate_instance(SMALL)
    Store.create(path, inst, solve(inst, config=FAST))
    store = Store.open(path)
    registry = ToolRegistry(ToolContext(store, ApprovalAuthority(), FAST), surface="mcp")
    with store.transaction():
        d = solved_draft(registry, goal)
        registry.call("request_commit", {"draft_id": d})

    lines = []
    admin_main(["--state", str(path), "approve"], out=lines.append, ask=lambda prompt: "n")
    assert f"Solved for: {GOAL_LABELS[goal]}" in "\n".join(lines)


def test_the_web_proposal_carries_the_goal_and_its_wording(ctx, registry):
    d = solved_draft(registry, "earliest_finish")
    p = web_views.proposal(ctx, d)
    assert p["goal"] == "earliest_finish" and p["goal_label"] == GOAL_LABELS["earliest_finish"]


def test_the_page_only_claims_fewest_moves_were_proven_when_that_was_the_goal():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'p.goal === "fewest_moves"' in script and "Solved for: ${p.goal_label}" in script


def test_the_goal_names_are_defined_once_and_every_user_of_them_agrees():
    from typing import get_args

    from jobshop.evals.scenario import Expect
    from jobshop.tools.functions import RescheduleInput
    from jobshop.tools.store import Draft
    from jobshop.tools.views import Goal

    names = set(get_args(Goal))
    assert names == set(GOAL_LABELS) == {"fewest_moves", "earliest_finish"}
    for model, field in ((RescheduleInput, "goal"), (Expect, "reschedule_goal")):
        assert names <= set(model.model_json_schema()["properties"][field].get("enum", []) or
                            [v for a in model.model_json_schema()["properties"][field].get("anyOf", []) for v in a.get("enum", [])])
    assert Draft.__dataclass_fields__["goal"].default in names
