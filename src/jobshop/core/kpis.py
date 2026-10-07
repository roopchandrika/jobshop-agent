"""KPIs and before/after comparison.

Definitions (these are what the agent is allowed to quote, so they must be exact):

* completion(order)  = end of its last operation.
* tardiness(order)   = max(0, completion - due), in minutes.
* weighted tardiness = sum of tardiness * priority weight over all orders.
* late order         = an order with tardiness > 0.
* makespan           = latest end over all operations, in minutes since t0.
* utilization(m)     = busy minutes / open minutes for machine m, both measured in the
                       window [instance.now, makespan]. "Open" means availability minus
                       downtime. A machine with no open time in the window reports 0.0.
* mean utilization   = average over machines that have open time in the window.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from jobshop.core import intervals
from jobshop.core.models import Instance, Schedule


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OrderKPI(_Model):
    order_id: str
    due: int
    completion: int
    tardiness: int
    weighted_tardiness: int


class KPIs(_Model):
    total_tardiness: int
    weighted_tardiness: int
    late_orders: int
    late_order_ids: list[str]
    makespan: int
    machine_utilization: dict[str, float]
    mean_utilization: float
    orders: list[OrderKPI]


def compute_kpis(instance: Instance, schedule: Schedule) -> KPIs:
    by_op = schedule.by_op()
    missing = [
        op.id for order in instance.orders for op in order.operations if op.id not in by_op
    ]
    if missing:
        raise ValueError(f"schedule is missing operations: {missing[:5]}")

    order_kpis = []
    for order in instance.orders:
        completion = by_op[order.operations[-1].id].end
        tardiness = max(0, completion - order.due)
        order_kpis.append(
            OrderKPI(
                order_id=order.id,
                due=order.due,
                completion=completion,
                tardiness=tardiness,
                weighted_tardiness=tardiness * order.weight,
            )
        )

    makespan = max((a.end for a in schedule.assignments), default=0)
    utilization = _utilization(instance, schedule, instance.now, makespan)
    measured = [u for u, has_time in utilization.values() if has_time]

    return KPIs(
        total_tardiness=sum(k.tardiness for k in order_kpis),
        weighted_tardiness=sum(k.weighted_tardiness for k in order_kpis),
        late_orders=sum(1 for k in order_kpis if k.tardiness > 0),
        late_order_ids=[k.order_id for k in order_kpis if k.tardiness > 0],
        makespan=makespan,
        machine_utilization={mid: u for mid, (u, _) in utilization.items()},
        mean_utilization=sum(measured) / len(measured) if measured else 0.0,
        orders=order_kpis,
    )


def _utilization(
    instance: Instance, schedule: Schedule, lo: int, hi: int
) -> dict[str, tuple[float, bool]]:
    """machine id -> (utilization, machine has open time in the window)."""
    result: dict[str, tuple[float, bool]] = {}
    for machine in instance.machines:
        open_minutes = intervals.total_length(intervals.clip(machine.free_windows(), lo, hi))
        busy = sum(
            intervals.overlap_length((a.start, a.end), (lo, hi))
            for a in schedule.assignments
            if a.machine_id == machine.id
        )
        result[machine.id] = (busy / open_minutes if open_minutes else 0.0, open_minutes > 0)
    return result


class MovedOperation(_Model):
    op_id: str
    order_id: str
    from_machine: str
    to_machine: str
    from_start: int
    to_start: int


class OrderDelta(_Model):
    order_id: str
    tardiness_before: int
    tardiness_after: int
    change: int  # after - before; positive means later


class ScheduleDiff(_Model):
    kpi_before: KPIs
    kpi_after: KPIs
    delta_total_tardiness: int
    delta_weighted_tardiness: int
    delta_late_orders: int
    delta_makespan: int
    delta_mean_utilization: float
    newly_late_orders: list[str]
    no_longer_late_orders: list[str]
    moved_operations: list[MovedOperation]
    machine_changes: int  # moved operations whose machine changed (not just their time)
    added_operations: list[str]
    removed_operations: list[str]
    order_deltas: list[OrderDelta]  # only orders whose tardiness changed


def diff_schedules(
    before_instance: Instance,
    before: Schedule,
    after_instance: Instance,
    after: Schedule,
) -> ScheduleDiff:
    """Compare two schedules. Each side brings its own instance because a draft may have
    different priorities, downtime, or extra orders than the committed instance."""
    kpi_before = compute_kpis(before_instance, before)
    kpi_after = compute_kpis(after_instance, after)

    ops_before, ops_after = before.by_op(), after.by_op()
    moved = [
        MovedOperation(
            op_id=op_id,
            order_id=new.order_id,
            from_machine=ops_before[op_id].machine_id,
            to_machine=new.machine_id,
            from_start=ops_before[op_id].start,
            to_start=new.start,
        )
        for op_id, new in ops_after.items()
        if op_id in ops_before
        and (ops_before[op_id].machine_id, ops_before[op_id].start) != (new.machine_id, new.start)
    ]

    late_before, late_after = set(kpi_before.late_order_ids), set(kpi_after.late_order_ids)
    tardiness_before = {k.order_id: k.tardiness for k in kpi_before.orders}
    tardiness_after = {k.order_id: k.tardiness for k in kpi_after.orders}
    deltas = [
        OrderDelta(
            order_id=oid,
            tardiness_before=tardiness_before[oid],
            tardiness_after=tardiness_after[oid],
            change=tardiness_after[oid] - tardiness_before[oid],
        )
        for oid in tardiness_after
        if oid in tardiness_before and tardiness_after[oid] != tardiness_before[oid]
    ]

    return ScheduleDiff(
        kpi_before=kpi_before,
        kpi_after=kpi_after,
        delta_total_tardiness=kpi_after.total_tardiness - kpi_before.total_tardiness,
        delta_weighted_tardiness=kpi_after.weighted_tardiness - kpi_before.weighted_tardiness,
        delta_late_orders=kpi_after.late_orders - kpi_before.late_orders,
        delta_makespan=kpi_after.makespan - kpi_before.makespan,
        delta_mean_utilization=kpi_after.mean_utilization - kpi_before.mean_utilization,
        newly_late_orders=sorted(late_after - late_before),
        no_longer_late_orders=sorted(late_before - late_after),
        moved_operations=moved,
        machine_changes=sum(1 for m in moved if m.from_machine != m.to_machine),
        added_operations=sorted(set(ops_after) - set(ops_before)),
        removed_operations=sorted(set(ops_before) - set(ops_after)),
        order_deltas=deltas,
    )
