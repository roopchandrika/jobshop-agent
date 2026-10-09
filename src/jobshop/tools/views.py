"""Display-ready output models for the tools.

Design rule: the model may only quote numbers that tools returned, so tools return numbers
that are already final: times as plant-local "YYYY-MM-DD HH:MM" strings, durations in whole
minutes, utilization as a percentage rounded to one decimal. There is no raw float for the
model to re-round differently from what the planner sees elsewhere.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from jobshop.core.kpis import KPIs, ScheduleDiff
from jobshop.core.models import Assignment, Instance, Order, SolveInfo


class View(BaseModel):
    model_config = ConfigDict(extra="forbid")


# What a re-plan optimises after avoiding late orders. The one place the names are defined: the tool's
# argument, the draft, the answer and the eval scenarios all use this type.
Goal = Literal["fewest_moves", "earliest_finish"]
GOAL_LABELS: dict[Goal, str] = {
    "fewest_moves": "fewest operations moved, then earliest finish",
    "earliest_finish": "earliest finish, then fewest operations moved",
}


def fmt(instance: Instance, minute: int) -> str:
    return instance.to_datetime(minute).strftime("%Y-%m-%d %H:%M")


def pct(fraction: float) -> float:
    return round(fraction * 100, 1)


class OrderRowView(View):
    order_id: str
    family: str
    priority: int
    due_at: str
    completion_at: str | None = None  # None when there is no solved schedule to read from
    tardiness_min: int | None = None
    # Due time minus finish, in minutes: negative when the order is late, so late and exactly-on-time
    # are never confused. Given so nobody has to subtract.
    slack_min: int | None = None


class KPIView(View):
    total_tardiness_min: int
    weighted_tardiness: int
    total_orders: int
    late_orders: int
    on_time_orders: int
    late_order_ids: list[str]
    # Minutes from the plan start (t0) until the last operation finishes.
    makespan_min: int
    all_orders_done_at: str
    mean_utilization_pct: float
    machine_utilization_pct: dict[str, float]
    orders: list[OrderRowView]


class SolveView(View):
    status: str  # OPTIMAL, FEASIBLE, INFEASIBLE or UNKNOWN
    tardiness_proven_optimal: bool
    makespan_proven_optimal: bool
    # Whether "fewest operations moved from the live plan" was proven minimal; null if not applicable.
    stability_proven_optimal: bool | None = None
    solve_seconds: float


class AssignmentView(View):
    op_id: str
    order_id: str
    machine_id: str
    start_at: str
    end_at: str


class WindowView(View):
    start_at: str
    end_at: str


def order_row(instance: Instance, order: Order, kpis: KPIs | None) -> OrderRowView:
    row = OrderRowView(
        order_id=order.id,
        family=order.family,
        priority=order.priority,
        due_at=fmt(instance, order.due),
    )
    if kpis is not None:
        match = next(k for k in kpis.orders if k.order_id == order.id)
        row = row.model_copy(
            update={
                "completion_at": fmt(instance, match.completion),
                "tardiness_min": match.tardiness,
                "slack_min": order.due - match.completion,
            }
        )
    return row


def kpi_view(instance: Instance, kpis: KPIs, include_orders: bool = True) -> KPIView:
    orders = {o.id: o for o in instance.orders}
    return KPIView(
        total_tardiness_min=kpis.total_tardiness,
        weighted_tardiness=kpis.weighted_tardiness,
        total_orders=len(kpis.orders),
        late_orders=kpis.late_orders,
        on_time_orders=len(kpis.orders) - kpis.late_orders,
        late_order_ids=kpis.late_order_ids,
        makespan_min=kpis.makespan,
        all_orders_done_at=fmt(instance, kpis.makespan),
        mean_utilization_pct=pct(kpis.mean_utilization),
        machine_utilization_pct={m: pct(u) for m, u in kpis.machine_utilization.items()},
        orders=[order_row(instance, orders[k.order_id], kpis) for k in kpis.orders]
        if include_orders
        else [],
    )


def solve_view(info: SolveInfo) -> SolveView:
    return SolveView(
        status=info.status.value,
        tardiness_proven_optimal=info.tardiness_optimal,
        makespan_proven_optimal=info.makespan_optimal,
        stability_proven_optimal=info.stability_optimal,
        solve_seconds=round(info.wall_time_s, 1),
    )


def assignment_view(instance: Instance, a: Assignment) -> AssignmentView:
    return AssignmentView(
        op_id=a.op_id,
        order_id=a.order_id,
        machine_id=a.machine_id,
        start_at=fmt(instance, a.start),
        end_at=fmt(instance, a.end),
    )


class MovedOperationView(View):
    op_id: str
    order_id: str
    from_machine: str
    to_machine: str
    from_start_at: str
    to_start_at: str


class OrderChangeView(View):
    order_id: str
    priority: int
    tardiness_before_min: int
    tardiness_after_min: int
    change_min: int  # positive means the order now finishes later relative to its due date


class DiffView(View):
    kpi_before: KPIView
    kpi_after: KPIView
    delta_total_tardiness_min: int
    delta_weighted_tardiness: int
    delta_late_orders: int
    delta_makespan_min: int
    delta_mean_utilization_pct: float
    newly_late_orders: list[str]
    no_longer_late_orders: list[str]
    order_changes: list[OrderChangeView]  # only orders whose tardiness changed
    moved_operation_count: int
    machine_change_count: int  # moved operations that switched machine, not just time
    moved_operations: list[MovedOperationView]  # first MAX_MOVED_LISTED only
    moved_operations_truncated: bool
    added_operations: list[str]


MAX_MOVED_LISTED = 20


def diff_view(
    before_instance: Instance, after_instance: Instance, diff: ScheduleDiff
) -> DiffView:
    priorities = {o.id: o.priority for o in after_instance.orders}
    moved = diff.moved_operations
    return DiffView(
        kpi_before=kpi_view(before_instance, diff.kpi_before, include_orders=False),
        kpi_after=kpi_view(after_instance, diff.kpi_after, include_orders=False),
        delta_total_tardiness_min=diff.delta_total_tardiness,
        delta_weighted_tardiness=diff.delta_weighted_tardiness,
        delta_late_orders=diff.delta_late_orders,
        delta_makespan_min=diff.delta_makespan,
        delta_mean_utilization_pct=round(diff.delta_mean_utilization * 100, 1),
        newly_late_orders=diff.newly_late_orders,
        no_longer_late_orders=diff.no_longer_late_orders,
        order_changes=[
            OrderChangeView(
                order_id=d.order_id,
                priority=priorities[d.order_id],
                tardiness_before_min=d.tardiness_before,
                tardiness_after_min=d.tardiness_after,
                change_min=d.change,
            )
            for d in diff.order_deltas
        ],
        moved_operation_count=len(moved),
        machine_change_count=diff.machine_changes,
        moved_operations=[
            MovedOperationView(
                op_id=m.op_id,
                order_id=m.order_id,
                from_machine=m.from_machine,
                to_machine=m.to_machine,
                from_start_at=fmt(before_instance, m.from_start),
                to_start_at=fmt(after_instance, m.to_start),
            )
            for m in moved[:MAX_MOVED_LISTED]
        ],
        moved_operations_truncated=len(moved) > MAX_MOVED_LISTED,
        added_operations=diff.added_operations,
    )
