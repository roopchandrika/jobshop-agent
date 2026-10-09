"""Tool results carry slack and order counts, so the model quotes them instead of doing arithmetic
(a first real run showed it subtracting times and counting orders, which the numbers check flags)."""

from jobshop.agent.prompts import build_system_prompt, server_instructions
from jobshop.core.kpis import compute_kpis
from jobshop.tools import views
from tests.helpers import assign, instance, machine, order, schedule


def two_orders():
    inst = instance(
        [machine("M1", windows=((0, 1000),))],
        [order("A", 150, [(100, "cut")]), order("B", 120, [(100, "cut")])],
    )
    # A finishes at 100 (due 150: 50 minutes early); B finishes at 200 (due 120: 80 minutes late).
    sched = schedule([assign("A-op1", "A", "M1", 0, 100), assign("B-op1", "B", "M1", 100, 200)])
    return inst, sched


def test_slack_is_minutes_before_the_due_time_and_negative_when_late():
    inst, sched = two_orders()
    rows = {r.order_id: r for r in views.kpi_view(inst, compute_kpis(inst, sched)).orders}
    assert (rows["A"].slack_min, rows["A"].tardiness_min) == (50, 0)
    assert (rows["B"].slack_min, rows["B"].tardiness_min) == (-80, 80)   # late: negative, never confused with "exactly on time"


def test_an_order_finishing_exactly_on_its_due_time_has_zero_slack_and_a_late_one_does_not():
    inst = instance([machine("M1", windows=((0, 1000),))], [order("A", 100, [(100, "cut")]), order("B", 100, [(100, "cut")])])
    sched = schedule([assign("A-op1", "A", "M1", 0, 100), assign("B-op1", "B", "M1", 100, 200)])
    rows = {r.order_id: r.slack_min for r in views.kpi_view(inst, compute_kpis(inst, sched)).orders}
    assert rows == {"A": 0, "B": -100}


def test_an_order_without_a_schedule_has_no_slack_figure():
    inst, _ = two_orders()
    assert views.order_row(inst, inst.orders[0], None).slack_min is None


def test_kpis_count_orders_so_nobody_has_to():
    inst, sched = two_orders()
    kpi = views.kpi_view(inst, compute_kpis(inst, sched))
    assert (kpi.total_orders, kpi.late_orders, kpi.on_time_orders) == (2, 1, 1)
    # the compact comparison view (no per-order rows) still has the counts
    compact = views.kpi_view(inst, compute_kpis(inst, sched), include_orders=False)
    assert (compact.total_orders, compact.on_time_orders) == (2, 1) and compact.orders == []


def test_the_tools_return_them(registry):
    schedule_out = registry.call("get_schedule", {})
    kpis = schedule_out["kpis"]
    assert kpis["total_orders"] == len(kpis["orders"]) == kpis["on_time_orders"] + kpis["late_orders"]
    assert all(isinstance(o["slack_min"], int) and o["slack_min"] >= 0 for o in kpis["orders"])   # the live shop starts on time
    assert all(isinstance(o["slack_min"], int) for o in registry.call("list_orders", {})["orders"])


def test_the_prompt_tells_the_model_to_quote_them_and_to_refuse_commit_plainly(ctx):
    for prompt in (build_system_prompt(ctx), server_instructions()):
        assert "slack_min" in prompt and "do not subtract" in prompt
        assert "say plainly in your first sentence that you cannot" in prompt.replace("\\\n", "")


def test_list_orders_counts_its_rows_and_the_whole_plan_separately(registry):
    everything = registry.call("list_orders", {})
    assert everything["order_count"] == len(everything["orders"]) == everything["total_orders"] == everything["on_time_orders"]  # starts on time

    d = registry.call("create_draft", {})["draft_id"]
    registry.call("add_rush_order", {"draft_id": d, "family": "gear", "due": "2026-01-05 06:30", "priority": 5})  # cannot be on time
    registry.call("reschedule", {"draft_id": d})
    after = registry.call("list_orders", {"source": d})
    late = [o for o in after["orders"] if o["tardiness_min"]]
    assert after["total_orders"] == after["order_count"] == everything["total_orders"] + 1
    assert after["on_time_orders"] == after["total_orders"] - len(late) < after["total_orders"]
    assert all(o["slack_min"] < 0 for o in late)


def test_a_filter_changes_the_rows_but_not_the_whole_plan_counts(registry):
    """A late-only list must not read as 'no orders are on time' (the counts are about the plan, not the rows)."""
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("add_rush_order", {"draft_id": d, "family": "gear", "due": "2026-01-05 06:30", "priority": 5})
    registry.call("reschedule", {"draft_id": d})
    whole = registry.call("list_orders", {"source": d})
    late_only = registry.call("list_orders", {"source": d, "late_only": True})

    assert 1 <= late_only["order_count"] == len(late_only["orders"]) < whole["order_count"]
    assert (late_only["total_orders"], late_only["on_time_orders"]) == (whole["total_orders"], whole["on_time_orders"])
    assert late_only["on_time_orders"] > 0

    one_family = registry.call("list_orders", {"source": d, "family": "gear"})
    assert one_family["order_count"] < one_family["total_orders"] == whole["total_orders"]


def test_an_unsolved_draft_has_no_on_time_figure_but_still_a_total(registry):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    out = registry.call("list_orders", {"source": d})
    assert out["order_count"] == len(out["orders"]) == out["total_orders"] and out["on_time_orders"] is None
