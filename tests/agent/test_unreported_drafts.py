"""Edits a turn leaves in a draft the answer does not name are shown beside the answer, so nothing waits there unseen."""

from jobshop.agent.loop import AgentConfig
from jobshop.tools.outcome import draft_outcome, unreported_drafts
from tests.agent.test_loop import proposal_script, turn
from tests.fake_llm import message, submit, tool


def two_drafts_name_the_first():
    return [
        message(tool("create_draft", "c1")),
        message(tool("change_priority", "e1", draft_id="D1", order_id="O-101", priority=5), tool("reschedule", "r1", draft_id="D1")),
        message(tool("create_draft", "c2")),
        message(tool("change_priority", "e2", draft_id="D2", order_id="O-102", priority=1)),
        message(tool("compare_schedules", "k", after="D1")),
        submit(summary="O-101 goes first.", draft_id="D1"),
    ]


def test_a_second_draft_with_edits_that_the_answer_does_not_name_is_reported(ctx, registry):
    result, _, _ = turn(ctx, registry, two_drafts_name_the_first())
    [warning] = [w for w in result.final.warnings if "D2" in w]
    assert "does not mention it" in warning and "O-102 priority" in warning
    assert not any("D1" in w for w in result.final.warnings)           # the named draft is not reported


def test_a_turn_with_one_named_draft_has_no_such_warning(ctx, registry):
    result, _, _ = turn(ctx, registry, proposal_script())
    assert not [w for w in result.final.warnings if "does not mention" in w]


def test_a_draft_left_from_an_earlier_turn_and_untouched_is_not_reported_again(ctx, registry):
    first, _, messages = turn(ctx, registry, proposal_script())
    assert first.final.draft_id == "D1"
    script = [message(tool("get_schedule", "g")), submit(summary="All is as before.")]
    second, _, _ = turn(ctx, registry, script, messages=messages, user="Anything else?")
    assert not [w for w in second.final.warnings if "does not mention" in w]


def test_a_draft_edited_again_in_a_later_turn_is_reported_if_the_answer_does_not_name_it(ctx, registry):
    _, _, messages = turn(ctx, registry, proposal_script())
    script = [message(tool("change_priority", "e", draft_id="D1", order_id="O-103", priority=2)), submit(summary="Noted.")]
    second, _, _ = turn(ctx, registry, script, messages=messages, user="Tweak it.")
    assert any("D1" in w and "O-103 priority" in w for w in second.final.warnings)


def test_the_guard_can_be_switched_off_with_the_other_answer_guards(ctx, registry):
    result, _, _ = turn(ctx, registry, two_drafts_name_the_first(), config=AgentConfig(model="m", answer_guards=False))
    assert not [w for w in result.final.warnings if "does not mention" in w]


def test_an_unsolved_named_draft_lists_what_it_holds(ctx, registry):
    registry.call("create_draft", {})
    registry.call("change_priority", {"draft_id": "D1", "order_id": "O-101", "priority": 5})
    [warning] = draft_outcome(ctx, "D1").warnings
    assert "no solved schedule" in warning and "holds these unsolved edits: O-101 priority" in warning


def test_long_lists_of_edits_are_cut_with_a_count():
    from jobshop.tools.outcome import _listed

    items = [f"edit {i}" for i in range(11)]
    assert _listed(items) == "; ".join(items[:8]) + "; and 3 more"
    assert _listed(items[:3]) == "edit 0; edit 1; edit 2"


def test_unreported_drafts_ignores_drafts_without_changes(ctx, registry):
    from jobshop.tools.outcome import snapshot_drafts

    before = snapshot_drafts(ctx)
    registry.call("create_draft", {})
    assert unreported_drafts(ctx, before, None) == []
