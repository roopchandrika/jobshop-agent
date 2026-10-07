"""The MCP route to approval: a model can ask, only a human can decide."""

import pytest

from jobshop.tools import human
from jobshop.tools.errors import ToolError
from jobshop.tools.registry import ToolRegistry


@pytest.fixture
def mcp(ctx):
    return ToolRegistry(ctx, surface="mcp")


def solved_draft(mcp, priority=5):
    d = mcp.call("create_draft", {})["draft_id"]
    mcp.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": priority})
    mcp.call("reschedule", {"draft_id": d})
    return d


# --- which front end sees which tools -------------------------------------------------------


def test_the_chat_agent_does_not_get_the_mcp_only_tools(registry):
    assert {"request_commit", "get_approval_status"}.isdisjoint(registry.names())
    with pytest.raises(ToolError, match="unknown tool 'request_commit'"):
        registry.call("request_commit", {"draft_id": "D1"})


def test_the_mcp_surface_has_the_approval_request_tools_but_never_commit(mcp):
    names = mcp.names()
    assert {"request_commit", "get_approval_status"} <= set(names)
    assert "commit_schedule" not in names and len(names) == 13
    with pytest.raises(ToolError, match="unknown tool 'commit_schedule'"):
        mcp.call("commit_schedule", {"draft_id": "D1", "approval_token": "x"})
    assert "commit_schedule" not in [s["name"] for s in mcp.api_specs()]


# --- request_commit --------------------------------------------------------------------------


def test_request_commit_records_a_pending_request_and_commits_nothing(ctx, mcp):
    d = solved_draft(mcp)
    out = mcp.call("request_commit", {"draft_id": d})
    assert out["status"] == "pending" and out["request_id"] == "R1" and out["draft_id"] == d
    assert "NOTHING HAS BEEN COMMITTED" in out["message"]
    assert ctx.store.committed.version == 1


def test_requesting_the_same_schedule_twice_reuses_the_request(mcp):
    d = solved_draft(mcp)
    first = mcp.call("request_commit", {"draft_id": d})["request_id"]
    assert mcp.call("request_commit", {"draft_id": d})["request_id"] == first


def test_request_commit_refuses_unsolved_unchanged_and_stale_drafts(ctx, mcp):
    unsolved = mcp.call("create_draft", {})["draft_id"]
    with pytest.raises(ToolError, match="Call reschedule first"):
        mcp.call("request_commit", {"draft_id": unsolved})

    no_changes = mcp.call("create_draft", {})["draft_id"]
    mcp.call("reschedule", {"draft_id": no_changes})
    with pytest.raises(ToolError, match="nothing to commit"):
        mcp.call("request_commit", {"draft_id": no_changes})

    stale = solved_draft(mcp)
    ctx.store.set_clock(30)
    with pytest.raises(ToolError, match="stale"):
        mcp.call("request_commit", {"draft_id": stale})


def test_get_approval_status_for_each_state(ctx, mcp, clock):
    rid = mcp.call("request_commit", {"draft_id": solved_draft(mcp)})["request_id"]

    def status():
        return mcp.call("get_approval_status", {"request_id": rid})

    assert status()["status"] == "pending" and status()["live_plan_version"] == 1

    clock.advance(ctx.store.request_ttl_s + 1)
    assert status()["status"] == "expired"
    clock.t -= ctx.store.request_ttl_s + 1

    ctx.store.set_clock(30)
    assert status()["status"] == "stale" and "can no longer be approved" in status()["message"]

    with pytest.raises(ToolError, match="unknown approval request 'R99'"):
        mcp.call("get_approval_status", {"request_id": "R99"})


# --- the human side ---------------------------------------------------------------------------


def test_a_human_approval_commits_exactly_what_was_requested(ctx, mcp):
    d = solved_draft(mcp)
    rid = mcp.call("request_commit", {"draft_id": d})["request_id"]
    requested_schedule = ctx.store.draft(d).schedule

    result = human.approve(ctx, rid)
    assert result["new_version"] == 2
    assert ctx.store.committed.schedule is requested_schedule
    assert ctx.store.committed.instance.order("O-101").priority == 5
    status = mcp.call("get_approval_status", {"request_id": rid})
    assert status["status"] == "approved" and status["live_plan_version"] == 2


def test_describe_shows_the_comparison_computed_from_the_store(ctx, mcp):
    d = solved_draft(mcp)
    rid = mcp.call("request_commit", {"draft_id": d})["request_id"]
    info = human.describe(ctx, rid)
    assert info["status"] == "pending" and info["changes"] == ctx.store.draft(d).changes
    assert info["comparison"]["diff"]["kpi_before"]["late_orders"] >= 0


def test_approval_is_refused_if_the_draft_changed_after_the_request(ctx, mcp):
    d = solved_draft(mcp)
    rid = mcp.call("request_commit", {"draft_id": d})["request_id"]
    mcp.call("change_priority", {"draft_id": d, "order_id": "O-102", "priority": 5})  # edits the reviewed draft
    with pytest.raises(ToolError, match="was changed after approval was requested"):
        human.approve(ctx, rid)
    assert ctx.store.committed.version == 1


def test_a_re_solved_draft_cannot_ride_on_an_old_request(ctx, mcp):
    d = solved_draft(mcp)
    rid = mcp.call("request_commit", {"draft_id": d})["request_id"]
    mcp.call("simulate_downtime", {"draft_id": d, "machine_id": "M3", "start": "2026-01-05 07:00", "end": "2026-01-05 09:00"})
    mcp.call("reschedule", {"draft_id": d})  # a different schedule than the one that was requested
    with pytest.raises(ToolError, match="was changed after approval was requested"):
        human.approve(ctx, rid)


def test_requests_cannot_be_approved_twice_or_after_a_denial_or_when_stale(ctx, mcp):
    d1 = solved_draft(mcp)
    r1 = mcp.call("request_commit", {"draft_id": d1})["request_id"]
    d2 = solved_draft(mcp, priority=4)
    r2 = mcp.call("request_commit", {"draft_id": d2})["request_id"]

    human.approve(ctx, r1)
    with pytest.raises(ToolError, match="is approved, not pending"):
        human.approve(ctx, r1)
    with pytest.raises(ToolError, match="is stale, not pending"):  # r2 was made against version 1
        human.approve(ctx, r2)

    d3 = solved_draft(mcp, priority=3)
    r3 = mcp.call("request_commit", {"draft_id": d3})["request_id"]
    human.deny(ctx, r3)
    assert mcp.call("get_approval_status", {"request_id": r3})["status"] == "denied"
    with pytest.raises(ToolError, match="is denied, not pending"):
        human.approve(ctx, r3)
    assert ctx.store.committed.version == 2  # only r1 ever committed


def test_an_expired_request_cannot_be_approved(ctx, mcp, clock):
    rid = mcp.call("request_commit", {"draft_id": solved_draft(mcp)})["request_id"]
    clock.advance(ctx.store.request_ttl_s + 1)
    with pytest.raises(ToolError, match="is expired, not pending"):
        human.approve(ctx, rid)


def test_pending_requests_lists_only_pending_ones(ctx, mcp):
    r1 = mcp.call("request_commit", {"draft_id": solved_draft(mcp)})["request_id"]
    r2 = mcp.call("request_commit", {"draft_id": solved_draft(mcp, priority=4)})["request_id"]
    human.deny(ctx, r2)
    assert [r.id for r in human.pending_requests(ctx)] == [r1]
