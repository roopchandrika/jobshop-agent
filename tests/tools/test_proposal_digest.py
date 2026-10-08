"""Approval must bind to the draft's EDITS, not just its schedule.

Found while building the web UI: a draft edited after review can solve to the identical schedule
(an order's priority raised where nothing moves), so a schedule-only fingerprint still matched and the
commit carried an edit nobody had seen.
"""

import pytest

from jobshop.tools import human
from jobshop.tools.approval import ApprovalError, proposal_digest, schedule_digest
from jobshop.tools.errors import ToolError
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.store import Store


def reviewed_draft(registry):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    registry.call("reschedule", {"draft_id": d})
    return d


def edit_without_moving_anything(registry, ctx, d):
    """Change another order's priority and re-solve. Asserts the schedule is unchanged, so the test is not vacuous."""
    before = schedule_digest(ctx.store.draft(d).schedule)
    registry.call("change_priority", {"draft_id": d, "order_id": "O-102", "priority": 5})
    registry.call("reschedule", {"draft_id": d})
    assert schedule_digest(ctx.store.draft(d).schedule) == before, "precondition: the edit must not move any operation"


def test_the_digest_changes_with_the_edits_even_when_the_schedule_is_identical(registry, ctx):
    d = reviewed_draft(registry)
    draft = ctx.store.draft(d)
    first = proposal_digest(draft.instance, draft.schedule)
    edit_without_moving_anything(registry, ctx, d)
    assert proposal_digest(draft.instance, draft.schedule) != first


def test_the_digest_is_stable_for_identical_content_and_across_a_file_round_trip(registry, ctx, tmp_path):
    d = reviewed_draft(registry)
    draft = ctx.store.draft(d)
    assert proposal_digest(draft.instance, draft.schedule) == proposal_digest(draft.instance, draft.schedule)

    shared = Store.create(tmp_path / "state.json", ctx.store.committed.instance, ctx.store.committed.schedule)
    with shared.transaction():
        copy = shared.create_draft()
        copy.edited(draft.instance, "x")
        copy.schedule = draft.schedule
        expected = proposal_digest(copy.instance, copy.schedule)
    reopened = Store.open(tmp_path / "state.json")
    with reopened.transaction():
        again = reopened.draft(copy.id)
        assert proposal_digest(again.instance, again.schedule) == expected   # the MCP processes see the same value


def test_a_token_for_the_reviewed_draft_is_useless_after_an_unseen_edit(registry, ctx):
    d = reviewed_draft(registry)
    draft = ctx.store.draft(d)
    token = ctx.authority.issue(draft_id=d, base_version=draft.base_version, schedule_digest=proposal_digest(draft.instance, draft.schedule))
    edit_without_moving_anything(registry, ctx, d)
    with pytest.raises(ToolError, match="approval rejected"):
        registry.call("commit_schedule", {"draft_id": d, "approval_token": token}, allow_hidden=True)
    assert ctx.store.committed.version == 1


def test_the_web_style_commit_refuses_a_draft_edited_after_review(registry, ctx):
    d = reviewed_draft(registry)
    draft = ctx.store.draft(d)
    reviewed = proposal_digest(draft.instance, draft.schedule)
    edit_without_moving_anything(registry, ctx, d)
    with pytest.raises(ToolError, match="not the proposal you reviewed"):
        human.commit_draft(ctx, d, reviewed_digest=reviewed)
    assert ctx.store.committed.version == 1
    human.commit_draft(ctx, d)           # without a reviewed digest (the chat prompt, nobody can edit meanwhile) it works
    assert ctx.store.committed.version == 2


def test_an_mcp_approval_request_goes_stale_when_the_draft_is_edited_after_it(ctx):
    mcp = ToolRegistry(ctx, surface="mcp")
    d = reviewed_draft(mcp)
    request_id = mcp.call("request_commit", {"draft_id": d})["request_id"]
    edit_without_moving_anything(mcp, ctx, d)
    with pytest.raises(ToolError, match="changed after approval was requested"):
        human.approve(ctx, request_id)
    assert ctx.store.committed.version == 1
    again = mcp.call("request_commit", {"draft_id": d})                  # asking again creates a fresh request
    assert again["request_id"] != request_id
    assert human.approve(ctx, again["request_id"])["new_version"] == 2
