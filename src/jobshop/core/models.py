"""Domain models for the flexible job shop.

Conventions that every other module relies on:

* Time is an integer number of minutes since ``Instance.t0``. ``Instance.now`` is the
  current minute. Datetimes only appear at the edges (``to_datetime``/``to_minutes``).
* Intervals are half-open: an operation on [10, 20) and another starting at 20 do not
  overlap.
* Priority 5 is the most urgent; ``PRIORITY_WEIGHTS`` turns it into a tardiness weight.
* Models are frozen (no attribute reassignment) and reject unknown fields, so a typo in
  a tool argument fails loudly instead of being silently ignored. Note that
  ``model_copy(update=...)`` skips validation; build changed copies with
  ``Model.model_validate({...})`` when the result must be re-checked.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jobshop.core import intervals

PRIORITY_WEIGHTS: dict[int, int] = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeWindow(_Model):
    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _end_after_start(self) -> TimeWindow:
        if self.end <= self.start:
            raise ValueError(f"window end ({self.end}) must be after start ({self.start})")
        return self


class Machine(_Model):
    id: str
    type: str
    capabilities: list[str] = Field(min_length=1)
    # When the machine can work (e.g. shifts). Outside these windows it is closed.
    availability: list[TimeWindow]
    # Outages inside the availability windows (planned or unplanned).
    downtime: list[TimeWindow] = Field(default_factory=list)

    def free_windows(self) -> list[intervals.Interval]:
        """Availability minus downtime, merged: the time the machine can actually work."""
        return intervals.subtract(
            [(w.start, w.end) for w in self.availability],
            [(w.start, w.end) for w in self.downtime],
        )


class Operation(_Model):
    id: str
    duration: int = Field(gt=0)
    # Eligibility is derived: any machine whose capabilities include this one can run it.
    required_capability: str


class RoutingStep(_Model):
    capability: str
    duration: int = Field(gt=0)


class Order(_Model):
    id: str
    family: str = "generic"
    due: int = Field(ge=0)
    priority: int = Field(ge=1, le=5)
    # Operations must run in this order, each starting after the previous one ends.
    operations: list[Operation] = Field(min_length=1)
    # Untrusted free text. It may contain anything, including text that looks like
    # instructions. It is data to display, never something to obey.
    notes: str = ""

    @property
    def weight(self) -> int:
        return PRIORITY_WEIGHTS[self.priority]


class Instance(_Model):
    t0: datetime
    now: int = Field(default=0, ge=0)
    machines: list[Machine]
    orders: list[Order]
    # family name -> ordered steps; used later to build rush orders from just a family.
    routing_templates: dict[str, list[RoutingStep]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> Instance:
        if self.t0.tzinfo is not None:
            raise ValueError("t0 must be a naive (plant-local) datetime")

        machine_ids = [m.id for m in self.machines]
        _require_unique(machine_ids, "machine")
        _require_unique([o.id for o in self.orders], "order")
        _require_unique([op.id for o in self.orders for op in o.operations], "operation")

        capabilities = {c for m in self.machines for c in m.capabilities}
        for order in self.orders:
            for op in order.operations:
                if op.required_capability not in capabilities:
                    raise ValueError(
                        f"operation {op.id} needs '{op.required_capability}' "
                        "but no machine offers it"
                    )
        for family, steps in self.routing_templates.items():
            for step in steps:
                if step.capability not in capabilities:
                    raise ValueError(
                        f"routing template '{family}' needs '{step.capability}' "
                        "but no machine offers it"
                    )
        return self

    @property
    def horizon(self) -> int:
        """Last minute any machine is open. Nothing can be scheduled after this."""
        return max((w.end for m in self.machines for w in m.availability), default=0)

    def machine(self, machine_id: str) -> Machine:
        for m in self.machines:
            if m.id == machine_id:
                return m
        raise KeyError(f"unknown machine {machine_id}")

    def order(self, order_id: str) -> Order:
        for o in self.orders:
            if o.id == order_id:
                return o
        raise KeyError(f"unknown order {order_id}")

    def eligible_machines(self, op: Operation) -> list[Machine]:
        return [m for m in self.machines if op.required_capability in m.capabilities]

    def to_datetime(self, minute: int) -> datetime:
        return self.t0 + timedelta(minutes=minute)

    def to_minutes(self, moment: datetime) -> int:
        if moment.tzinfo is not None:
            raise ValueError("expected a naive (plant-local) datetime")
        return int((moment - self.t0).total_seconds() // 60)


def _require_unique(ids: list[str], what: str) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise ValueError(f"duplicate {what} id: {item}")
        seen.add(item)


class Assignment(_Model):
    op_id: str
    order_id: str
    machine_id: str
    start: int
    end: int


class SolveStatus(str, Enum):
    OPTIMAL = "OPTIMAL"  # both objectives proven optimal
    FEASIBLE = "FEASIBLE"  # valid schedule, but not proven optimal (time limit hit)
    INFEASIBLE = "INFEASIBLE"  # proven: no schedule satisfies the constraints
    UNKNOWN = "UNKNOWN"  # ran out of time without finding any schedule


class SolveInfo(_Model):
    status: SolveStatus
    tardiness_optimal: bool = False
    makespan_optimal: bool = False
    # Best proven lower bound on weighted tardiness (None if no solve happened).
    tardiness_bound: float | None = None
    # What the solver itself reported for the schedule it returned. Tests compare these
    # with values recomputed independently by ``kpis``.
    reported_weighted_tardiness: int | None = None
    reported_makespan: int | None = None
    wall_time_s: float = 0.0
    num_workers: int = 1
    seed: int = 0


class Schedule(_Model):
    assignments: list[Assignment]
    solve_info: SolveInfo

    def by_op(self) -> dict[str, Assignment]:
        return {a.op_id: a for a in self.assignments}
