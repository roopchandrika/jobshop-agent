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


def test_each_order_row_says_how_many_minutes_early_it_finishes_and_never_negative():
    inst, sched = two_orders()
    rows = {r.order_id: r for r in views.kpi_view(inst, compute_kpis(inst, sched)).orders}
    assert (rows["A"].slack_min, rows["A"].tardiness_min) == (50, 0)
    assert (rows["B"].slack_min, rows["B"].tardiness_min) == (0, 80)  # late: no negative slack


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
    assert all(isinstance(o["slack_min"], int) and o["slack_min"] >= 0 for o in kpis["orders"])
    assert all(isinstance(o["slack_min"], int) for o in registry.call("list_orders", {})["orders"])


def test_the_prompt_tells_the_model_to_quote_them_and_to_refuse_commit_plainly(ctx):
    for prompt in (build_system_prompt(ctx), server_instructions()):
        assert "slack_min" in prompt and "do not subtract" in prompt
        assert "say plainly in your first sentence that you cannot" in prompt.replace("\\\n", "")


def test_list_orders_counts_its_rows_and_the_on_time_ones(registry):
    everything = registry.call("list_orders", {})
    assert everything["order_count"] == len(everything["orders"]) and everything["on_time_count"] == everything["order_count"]  # the shop starts on time

    d = registry.call("create_draft", {})["draft_id"]
    registry.call("add_rush_order", {"draft_id": d, "family": "gear", "due": "2026-01-05 06:30", "priority": 5})  # cannot be on time
    registry.call("reschedule", {"draft_id": d})
    after = registry.call("list_orders", {"source": d})
    assert after["order_count"] == everything["order_count"] + 1
    assert after["on_time_count"] == after["order_count"] - len([o for o in after["orders"] if o["tardiness_min"]]) < after["order_count"]

    late_only = registry.call("list_orders", {"source": d, "late_only": True})
    assert late_only["order_count"] == len(late_only["orders"]) >= 1 and late_only["on_time_count"] == 0


def test_an_unsolved_draft_has_no_on_time_count(registry):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    out = registry.call("list_orders", {"source": d})
    assert out["order_count"] == len(out["orders"]) and out["on_time_count"] is None
