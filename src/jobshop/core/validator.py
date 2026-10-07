"""Independent schedule checker. This is the ground truth for tests and evals.

It looks only at the raw instance data and the assignments. It does not import the
solver, does not use ``intervals`` or ``Machine.free_windows``, and does not trust
``schedule.solve_info`` (a schedule claiming OPTIMAL is checked like any other). A bug
in the solver therefore cannot hide itself here.

All violations are collected, not just the first, so a test can assert exactly which
rules a deliberately broken schedule breaks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict

from jobshop.core.models import Assignment, Instance, Schedule


class ViolationKind(str, Enum):
    MISSING_OPERATION = "MISSING_OPERATION"
    UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
    DUPLICATE_ASSIGNMENT = "DUPLICATE_ASSIGNMENT"
    ORDER_MISMATCH = "ORDER_MISMATCH"
    NEGATIVE_START = "NEGATIVE_START"
    BAD_DURATION = "BAD_DURATION"
    UNKNOWN_MACHINE = "UNKNOWN_MACHINE"
    INELIGIBLE_MACHINE = "INELIGIBLE_MACHINE"
    OUTSIDE_AVAILABILITY = "OUTSIDE_AVAILABILITY"
    DOWNTIME_OVERLAP = "DOWNTIME_OVERLAP"
    MACHINE_OVERLAP = "MACHINE_OVERLAP"
    PRECEDENCE = "PRECEDENCE"
    BEFORE_NOW = "BEFORE_NOW"
    FROZEN_MOVED = "FROZEN_MOVED"


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ViolationKind
    message: str
    op_id: str | None = None
    machine_id: str | None = None


class ValidationReport(BaseModel):
    violations: list[Violation]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def kinds(self) -> set[ViolationKind]:
        return {v.kind for v in self.violations}


def validate_schedule(
    instance: Instance,
    schedule: Schedule,
    frozen: Sequence[Assignment] = (),
) -> ValidationReport:
    found: list[Violation] = []

    def report(kind: ViolationKind, message: str, op_id: str | None = None,
               machine_id: str | None = None) -> None:
        found.append(Violation(kind=kind, message=message, op_id=op_id, machine_id=machine_id))

    known_ops = {
        op.id: (order.id, op) for order in instance.orders for op in order.operations
    }
    machines = {m.id: m for m in instance.machines}
    frozen_by_op = {a.op_id: a for a in frozen}

    # --- every operation scheduled exactly once -------------------------------------
    placed: dict[str, Assignment] = {}
    for a in schedule.assignments:
        if a.op_id not in known_ops:
            report(ViolationKind.UNKNOWN_OPERATION, f"{a.op_id} is not in the instance", a.op_id)
        elif a.op_id in placed:
            report(ViolationKind.DUPLICATE_ASSIGNMENT, f"{a.op_id} is assigned twice", a.op_id)
        else:
            placed[a.op_id] = a
    for op_id in known_ops:
        if op_id not in placed:
            report(ViolationKind.MISSING_OPERATION, f"{op_id} has no assignment", op_id)

    # --- per-assignment rules --------------------------------------------------------
    for op_id, a in placed.items():
        order_id, op = known_ops[op_id]
        if a.order_id != order_id:
            report(ViolationKind.ORDER_MISMATCH,
                   f"{op_id} belongs to {order_id}, assignment says {a.order_id}", op_id)
        if a.start < 0:
            report(ViolationKind.NEGATIVE_START, f"{op_id} starts at {a.start}", op_id)
        if a.end - a.start != op.duration:
            report(ViolationKind.BAD_DURATION,
                   f"{op_id} spans {a.end - a.start} min, needs {op.duration}", op_id)
        if op_id not in frozen_by_op and a.start < instance.now:
            report(ViolationKind.BEFORE_NOW,
                   f"{op_id} starts at {a.start}, before now ({instance.now})", op_id)

        machine = machines.get(a.machine_id)
        if machine is None:
            report(ViolationKind.UNKNOWN_MACHINE,
                   f"{op_id} uses unknown machine {a.machine_id}", op_id, a.machine_id)
            continue
        if op.required_capability not in machine.capabilities:
            report(ViolationKind.INELIGIBLE_MACHINE,
                   f"{a.machine_id} cannot do '{op.required_capability}' needed by {op_id}",
                   op_id, a.machine_id)
        if not _inside_open_time(machine.availability, a.start, a.end):
            report(ViolationKind.OUTSIDE_AVAILABILITY,
                   f"{op_id} [{a.start},{a.end}) is not inside {a.machine_id}'s open windows",
                   op_id, a.machine_id)
        for outage in machine.downtime:
            if a.start < outage.end and outage.start < a.end:
                report(ViolationKind.DOWNTIME_OVERLAP,
                       f"{op_id} [{a.start},{a.end}) overlaps downtime "
                       f"[{outage.start},{outage.end}) on {a.machine_id}", op_id, a.machine_id)

    # --- frozen operations must not move --------------------------------------------
    for fa in frozen:
        current = placed.get(fa.op_id)
        if current and (current.machine_id, current.start) != (fa.machine_id, fa.start):
            report(ViolationKind.FROZEN_MOVED,
                   f"{fa.op_id} was frozen on {fa.machine_id}@{fa.start} but is on "
                   f"{current.machine_id}@{current.start}", fa.op_id)

    # --- no two operations share a machine at the same time -------------------------
    per_machine: dict[str, list[Assignment]] = defaultdict(list)
    for a in placed.values():
        per_machine[a.machine_id].append(a)
    for machine_id, items in per_machine.items():
        items.sort(key=lambda a: (a.start, a.end))
        for earlier, later in zip(items, items[1:]):
            if later.start < earlier.end:
                report(ViolationKind.MACHINE_OVERLAP,
                       f"{earlier.op_id} [{earlier.start},{earlier.end}) overlaps "
                       f"{later.op_id} [{later.start},{later.end}) on {machine_id}",
                       later.op_id, machine_id)

    # --- operations of an order run in sequence -------------------------------------
    for order in instance.orders:
        for first, second in zip(order.operations, order.operations[1:]):
            a1, a2 = placed.get(first.id), placed.get(second.id)
            if a1 and a2 and a2.start < a1.end:
                report(ViolationKind.PRECEDENCE,
                       f"{second.id} starts at {a2.start} before {first.id} ends at {a1.end}",
                       second.id)

    return ValidationReport(violations=found)


def _inside_open_time(windows, start: int, end: int) -> bool:
    """True if [start, end) lies inside the machine's open time.

    Windows that overlap or touch count as one continuous stretch, so an operation may
    legitimately span two back-to-back windows.
    """
    stretches: list[list[int]] = []
    for w in sorted(windows, key=lambda w: w.start):
        if stretches and w.start <= stretches[-1][1]:
            stretches[-1][1] = max(stretches[-1][1], w.end)
        else:
            stretches.append([w.start, w.end])
    return any(s <= start and end <= e for s, e in stretches)
