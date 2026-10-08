"""The reschedule tool's `goal` argument: one of two exact spellings, defaulting to the old behaviour."""

import pytest

from jobshop.tools.errors import ToolError


def draft_with_rush(registry):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("add_rush_order", {"draft_id": d, "family": "gear", "due": "2026-01-05 18:00", "priority": 5})
    return d


def test_the_default_goal_is_fewest_moves_and_is_reported_back(registry):
    d = draft_with_rush(registry)
    out = registry.call("reschedule", {"draft_id": d})
    assert out["goal"] == "fewest_moves" and out["feasible"]
    assert out["solve"]["stability_proven_optimal"] is not None or out["solve"]["status"] == "FEASIBLE"


def test_earliest_finish_is_accepted_and_never_finishes_later(registry):
    slow = draft_with_rush(registry)
    fast = draft_with_rush(registry)
    a = registry.call("reschedule", {"draft_id": slow})
    b = registry.call("reschedule", {"draft_id": fast, "goal": "earliest_finish"})
    assert b["goal"] == "earliest_finish" and b["feasible"]
    assert b["solve"]["stability_proven_optimal"] is None  # fewest moves is no longer the second goal
    assert b["kpis"]["weighted_tardiness"] == a["kpis"]["weighted_tardiness"]
    assert b["kpis"]["makespan_min"] <= a["kpis"]["makespan_min"]

    # a person can compare them: the faster plan moves at least as many operations
    moved = lambda d: registry.call("compare_schedules", {"after": d})["diff"]["moved_operation_count"]  # noqa: E731
    assert moved(fast) >= moved(slow)


@pytest.mark.parametrize("goal", ["fastest", "EARLIEST_FINISH", "earliest finish", "", None, 1, True])
def test_any_other_spelling_is_rejected_with_the_allowed_values(registry, goal):
    d = registry.call("create_draft", {})["draft_id"]
    with pytest.raises(ToolError, match="fewest_moves|earliest_finish"):
        registry.call("reschedule", {"draft_id": d, "goal": goal})


def test_the_model_is_shown_both_values_in_the_schema(registry):
    spec = next(s for s in registry.api_specs() if s["name"] == "reschedule")
    goal = spec["input_schema"]["properties"]["goal"]
    assert goal["enum"] == ["fewest_moves", "earliest_finish"] and goal["default"] == "fewest_moves"
