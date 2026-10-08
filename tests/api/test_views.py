"""The data behind the UI, checked against independent recomputation from the store."""

import pytest

from jobshop.api import views as web
from jobshop.core.kpis import compute_kpis, diff_schedules
from jobshop.tools.approval import proposal_digest
from jobshop.tools.errors import ToolError


def solved_draft(registry, priority=5):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": priority})
    registry.call("reschedule", {"draft_id": d})
    return d


def test_the_live_gantt_has_every_operation_exactly_once_grouped_by_machine(ctx):
    g = web.gantt(ctx, "committed")
    committed = ctx.store.committed
    bars = [op for m in g.machines for op in m.ops]
    assert sorted(op.op_id for op in bars) == sorted(a.op_id for a in committed.schedule.assignments)
    for m in g.machines:
        assert all(op.machine_id == m.machine_id for op in m.ops)
        assert [op.start for op in m.ops] == sorted(op.start for op in m.ops)
    assert all(op.changed is None and op.was is None for op in bars)   # nothing to compare the live plan with
    assert g.moved_count == 0 and g.now == committed.instance.now


def test_the_axis_and_windows_come_from_the_instance(ctx):
    g = web.gantt(ctx, "committed")
    inst = ctx.store.committed.instance
    assert g.axis_end == inst.horizon and g.axis_start == 0 and g.t0 == "2026-01-05 06:00"
    for row, machine in zip(g.machines, inst.machines):
        assert [(w.start, w.end) for w in row.open] == [(w.start, w.end) for w in machine.availability]


def test_late_orders_are_marked_on_their_last_operation_only(registry, ctx):
    d = registry.call("create_draft", {})["draft_id"]
    family = next(iter(ctx.store.committed.instance.routing_templates))
    registry.call("add_rush_order", {"draft_id": d, "family": family, "due": "2026-01-05 06:30"})  # cannot be met: it is late for sure
    registry.call("reschedule", {"draft_id": d})
    g = web.gantt(ctx, d)
    inst, schedule = ctx.store.draft(d).instance, ctx.store.draft(d).schedule
    kpis = compute_kpis(inst, schedule)
    assert g.late_orders == kpis.late_order_ids and "RUSH-1" in g.late_orders
    bars = [op for m in g.machines for op in m.ops]
    for k in kpis.orders:
        mine = [op for op in bars if op.order_id == k.order_id]
        assert {op.late_min for op in mine} == {k.tardiness}
        assert sum(op.last_of_order for op in mine) == 1
        assert max(mine, key=lambda op: op.end).last_of_order
    assert any(op.late_min > 0 for op in bars)   # the scenario really produced lateness


def test_outages_in_a_draft_are_drawn_as_windows_on_their_machine(registry, ctx):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("simulate_downtime", {"draft_id": d, "machine_id": "M1", "start": "2026-01-05 08:00", "end": "2026-01-05 10:00"})
    registry.call("reschedule", {"draft_id": d})
    g = web.gantt(ctx, d)
    m1 = next(m for m in g.machines if m.machine_id == "M1")
    assert [(w.start_at, w.end_at) for w in m1.downtime] == [("2026-01-05 08:00", "2026-01-05 10:00")]
    assert all(not m.downtime for m in g.machines if m.machine_id != "M1")
    assert not [op for op in m1.ops if op.start < 240 and op.end > 120]   # nothing runs inside the outage


def test_a_draft_chart_flags_moved_operations_exactly_as_the_independent_diff_does(registry, ctx):
    d = solved_draft(registry)
    g = web.gantt(ctx, d)
    committed, draft = ctx.store.committed, ctx.store.draft(d)
    expected = {m.op_id for m in diff_schedules(committed.instance, committed.schedule, draft.instance, draft.schedule).moved_operations}
    flagged = {op.op_id for m in g.machines for op in m.ops if op.changed == "moved"}
    assert flagged == expected and g.moved_count == len(expected)
    live = committed.schedule.by_op()
    for op in (op for m in g.machines for op in m.ops if op.changed == "moved"):
        assert op.was.startswith(live[op.op_id].machine_id)


def test_an_operation_that_only_shifts_in_time_on_the_same_machine_is_still_moved(ctx):
    from jobshop.core.models import Schedule, SolveInfo, SolveStatus

    committed = ctx.store.committed
    data = committed.schedule.model_dump()
    victim = data["assignments"][0]
    victim["start"] += 5
    victim["end"] += 5                                          # same machine, five minutes later
    draft = ctx.store.create_draft()
    draft.edited(committed.instance, "shifted one operation")
    draft.schedule = Schedule(assignments=data["assignments"], solve_info=SolveInfo(status=SolveStatus.FEASIBLE))

    g = web.gantt(ctx, draft.id)
    moved = [op for m in g.machines for op in m.ops if op.changed == "moved"]
    assert [op.op_id for op in moved] == [victim["op_id"]] and g.moved_count == 1
    assert moved[0].machine_id == victim["machine_id"] and moved[0].was.startswith(victim["machine_id"])


def test_operations_of_a_new_rush_order_are_marked_new(registry, ctx):
    d = registry.call("create_draft", {})["draft_id"]
    family = next(iter(ctx.store.committed.instance.routing_templates))
    registry.call("add_rush_order", {"draft_id": d, "family": family, "due": "2026-01-05 20:00"})
    registry.call("reschedule", {"draft_id": d})
    new = [op for m in web.gantt(ctx, d).machines for op in m.ops if op.changed == "new"]
    assert new and all(op.order_id.startswith("RUSH-") for op in new)


@pytest.mark.parametrize("source, message", [("D9", "unknown draft")])
def test_an_unknown_draft_is_an_error(ctx, source, message):
    with pytest.raises(ToolError, match=message):
        web.gantt(ctx, source)


def test_an_unsolved_or_stale_draft_cannot_be_drawn(registry, ctx):
    d = registry.call("create_draft", {})["draft_id"]
    with pytest.raises(ToolError, match="no solved schedule"):
        web.gantt(ctx, d)
    registry.call("reschedule", {"draft_id": d})
    ctx.store.set_clock(ctx.store.committed.instance.now + 30)
    with pytest.raises(ToolError, match="stale"):
        web.gantt(ctx, d)


# -- the proposal -----------------------------------------------------------------------------------------


def test_a_proposal_carries_the_digest_of_the_exact_schedule_and_the_systems_comparison(registry, ctx):
    d = solved_draft(registry)
    p = web.proposal(ctx, d)
    assert p["draft_id"] == d and p["digest"] == proposal_digest(ctx.store.draft(d).instance, ctx.store.draft(d).schedule)
    assert p["changes"] == ctx.store.draft(d).changes
    assert p["comparison"]["diff"]["moved_operation_count"] == web.gantt(ctx, d).moved_count


@pytest.mark.parametrize("make", ["none", "unknown", "unsolved", "no_changes", "stale"])
def test_nothing_is_proposed_unless_the_draft_can_still_be_approved(registry, ctx, make):
    if make == "none":
        assert web.proposal(ctx, None) is None
        return
    if make == "unknown":
        assert web.proposal(ctx, "D9") is None
        return
    d = registry.call("create_draft", {})["draft_id"]
    if make == "unsolved":
        registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    else:
        registry.call("reschedule", {"draft_id": d})            # solved, but no changes to approve
        if make == "stale":
            registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
            registry.call("reschedule", {"draft_id": d})
            ctx.store.set_clock(ctx.store.committed.instance.now + 30)
    assert web.proposal(ctx, d) is None


def test_draft_summaries_say_which_drafts_are_solved_and_stale(registry, ctx):
    d = solved_draft(registry)
    assert web.draft_summaries(ctx) == [{"id": d, "changes": ctx.store.draft(d).changes, "solved": True, "stale": False}]
    ctx.store.set_clock(ctx.store.committed.instance.now + 30)
    assert web.draft_summaries(ctx)[0]["stale"] is True
