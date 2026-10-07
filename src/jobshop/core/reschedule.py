"""Rules for rescheduling around "now".

History is history: any operation that has already started (start < now) stays exactly
where it was. The one exception is an operation that is *running right now* on a machine
that is about to go down: the work is interrupted, its partial progress is lost, and it
must restart from ``now`` or later. That is a deliberately conservative model; the tools
layer reports every interrupted operation so the planner is told.
"""

from __future__ import annotations

from dataclasses import dataclass

from jobshop.core.models import Assignment, Instance, Schedule


@dataclass(frozen=True)
class ReschedulePlan:
    frozen: list[Assignment]  # keep these exactly as committed
    interrupted: list[Assignment]  # were running, hit by downtime: restart from now


def started_assignments(instance: Instance, schedule: Schedule) -> list[Assignment]:
    """Operations in ``schedule`` that began before ``instance.now`` (done or in progress)."""
    return [a for a in schedule.assignments if a.start < instance.now]


def plan_reschedule(draft_instance: Instance, committed: Schedule) -> ReschedulePlan:
    """Decide which committed assignments stay frozen when solving ``draft_instance``.

    ``draft_instance`` must have the same ``now`` as the schedule it is based on.
    Operations that do not exist in the draft (none today) are ignored.
    """
    now = draft_instance.now
    machines = {m.id: m for m in draft_instance.machines}
    known_ops = {op.id for o in draft_instance.orders for op in o.operations}

    frozen: list[Assignment] = []
    interrupted: list[Assignment] = []
    for a in started_assignments(draft_instance, committed):
        if a.op_id not in known_ops:
            continue
        running = a.end > now
        downtime_hits_it = running and any(
            d.start < a.end and a.start < d.end and d.end > now
            for d in machines[a.machine_id].downtime
        )
        (interrupted if downtime_hits_it else frozen).append(a)
    return ReschedulePlan(frozen=frozen, interrupted=interrupted)
