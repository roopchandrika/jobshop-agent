"""Blast-radius limits: what a confused or manipulated model can pile up before it is stopped."""

import pytest

from jobshop.tools.errors import ToolError
from jobshop.tools.store import MAX_CHANGES_PER_DRAFT, MAX_OPEN_DRAFTS, MAX_PENDING_REQUESTS
from tests.tools.test_tools import new_draft


def test_open_drafts_are_capped_and_discarding_frees_a_slot(registry):
    ids = [new_draft(registry) for _ in range(MAX_OPEN_DRAFTS)]
    with pytest.raises(ToolError, match="open drafts, which is the limit"):
        registry.call("create_draft", {})
    registry.call("discard_draft", {"draft_id": ids[0]})
    registry.call("create_draft", {})


def test_changes_per_draft_are_capped_and_a_refused_change_leaves_the_draft_unchanged(ctx, registry):
    d = new_draft(registry)
    for k in range(MAX_CHANGES_PER_DRAFT):
        registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 1 + k % 5})
    before = list(ctx.store.draft(d).changes)

    with pytest.raises(ToolError, match="already has 20 changes"):
        registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    assert ctx.store.draft(d).changes == before


def test_pending_approval_requests_are_capped(ctx):
    from jobshop.tools.registry import ToolRegistry

    mcp = ToolRegistry(ctx, surface="mcp")
    # A draft with a solved (cheap) schedule: reuse the committed one so no solving is needed.
    for k in range(MAX_PENDING_REQUESTS + 1):
        draft = ctx.store.create_draft()
        draft.edited(draft.instance, f"change {k}")
        draft.schedule = ctx.store.committed.schedule
        if k < MAX_PENDING_REQUESTS:
            assert mcp.call("request_commit", {"draft_id": draft.id})["status"] == "pending"
        else:
            with pytest.raises(ToolError, match="approval requests are already waiting"):
                mcp.call("request_commit", {"draft_id": draft.id})


def test_asking_again_for_the_same_draft_does_not_use_up_the_cap(ctx):
    from jobshop.tools.registry import ToolRegistry

    mcp = ToolRegistry(ctx, surface="mcp")
    draft = ctx.store.create_draft()
    draft.edited(draft.instance, "change")
    draft.schedule = ctx.store.committed.schedule
    first = mcp.call("request_commit", {"draft_id": draft.id})["request_id"]
    for _ in range(MAX_PENDING_REQUESTS + 2):
        assert mcp.call("request_commit", {"draft_id": draft.id})["request_id"] == first
