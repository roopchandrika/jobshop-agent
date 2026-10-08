"""What the web UI is shown, built from the store. Pure functions: no HTTP, no model.

Everything here is computed from the stored schedules, never from text the model wrote, for the
same reason the CLI's approval screen is: the planner decides on facts the harness produced.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from jobshop.core.kpis import compute_kpis
from jobshop.core.models import Instance, Schedule
from jobshop.tools import views
from jobshop.tools.approval import proposal_digest
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import CompareInput, ToolContext, compare_schedules


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpBar(_View):
    op_id: str
    order_id: str
    machine_id: str
    start: int  # minutes since t0
    end: int
    start_at: str
    end_at: str
    priority: int
    family: str
    late_min: int  # tardiness of this op's order (0 = on time)
    last_of_order: bool
    changed: Literal["moved", "new"] | None = None  # versus the live plan; draft charts only
    was: str | None = None  # for a moved op: where the live plan had it, e.g. "M2 2026-01-05 11:00"


class Window(_View):
    start: int
    end: int
    start_at: str
    end_at: str


class MachineRow(_View):
    machine_id: str
    type: str
    open: list[Window]
    downtime: list[Window]
    ops: list[OpBar]


class Gantt(_View):
    source: str
    t0: str
    now: int
    now_at: str
    axis_start: int
    axis_end: int
    machines: list[MachineRow]
    late_orders: list[str]
    moved_count: int


def _window(inst: Instance, start: int, end: int) -> Window:
    return Window(start=start, end=end, start_at=views.fmt(inst, start), end_at=views.fmt(inst, end))


def _resolve(ctx: ToolContext, source: str) -> tuple[Instance, Schedule]:
    if source == "committed":
        c = ctx.store.committed
        return c.instance, c.schedule
    draft = ctx.store.draft(source)
    if ctx.store.is_stale(draft):
        raise ToolError(f"draft {source} is stale: the live plan or the clock changed after it was made")
    if not draft.solved or draft.schedule is None:
        raise ToolError(f"draft {source} has no solved schedule to draw yet")
    return draft.instance, draft.schedule


def gantt(ctx: ToolContext, source: str) -> Gantt:
    inst, schedule = _resolve(ctx, source)
    live = ctx.store.committed.schedule.by_op()
    kpis = compute_kpis(inst, schedule)
    late = {k.order_id: k.tardiness for k in kpis.orders}
    orders = {o.id: o for o in inst.orders}
    last_end: dict[str, int] = {}
    for a in schedule.assignments:
        last_end[a.order_id] = max(last_end.get(a.order_id, 0), a.end)

    rows, moved = [], 0
    for m in inst.machines:
        bars = []
        for a in sorted((x for x in schedule.assignments if x.machine_id == m.id), key=lambda x: x.start):
            changed, was = None, None
            if source != "committed":
                before = live.get(a.op_id)
                if before is None:
                    changed = "new"
                elif (before.machine_id, before.start) != (a.machine_id, a.start):
                    changed, was = "moved", f"{before.machine_id} {views.fmt(inst, before.start)}"
            moved += changed == "moved"
            bars.append(OpBar(
                op_id=a.op_id, order_id=a.order_id, machine_id=a.machine_id, start=a.start, end=a.end,
                start_at=views.fmt(inst, a.start), end_at=views.fmt(inst, a.end),
                priority=orders[a.order_id].priority, family=orders[a.order_id].family,
                late_min=late[a.order_id], last_of_order=a.end == last_end[a.order_id], changed=changed, was=was,
            ))
        rows.append(MachineRow(
            machine_id=m.id, type=m.type,
            open=[_window(inst, w.start, w.end) for w in m.availability],
            downtime=[_window(inst, w.start, w.end) for w in m.downtime],
            ops=bars,
        ))
    return Gantt(
        source=source, t0=views.fmt(inst, 0), now=inst.now, now_at=views.fmt(inst, inst.now),
        axis_start=min((w.start for m in inst.machines for w in m.availability), default=0), axis_end=inst.horizon,
        machines=rows, late_orders=kpis.late_order_ids, moved_count=moved,
    )


def draft_summaries(ctx: ToolContext) -> list[dict[str, Any]]:
    return [
        {"id": d.id, "changes": d.changes, "solved": d.solved, "stale": ctx.store.is_stale(d)}
        for d in ctx.store.drafts()
    ]


def proposal(ctx: ToolContext, draft_id: str | None) -> dict[str, Any] | None:
    """The draft the planner is being asked to approve, or None if it is no longer approvable.

    ``digest`` fingerprints the exact schedule shown here; the Approve button sends it back so the
    server can refuse if the draft has changed since the planner looked.
    """
    if draft_id is None:
        return None
    try:
        draft = ctx.store.draft(draft_id)
    except ToolError:
        return None
    if ctx.store.is_stale(draft) or not draft.solved or draft.schedule is None or not draft.changes:
        return None
    comparison = compare_schedules(ctx, CompareInput(before="committed", after=draft.id)).model_dump(mode="json")
    return {
        "draft_id": draft.id,
        "digest": proposal_digest(draft.instance, draft.schedule),
        "changes": list(draft.changes),
        "goal": draft.goal,
        "goal_label": views.GOAL_LABELS[draft.goal],
        "comparison": comparison,
    }
