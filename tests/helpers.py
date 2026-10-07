"""Tiny builders so each test reads like a small scenario instead of model plumbing."""

from __future__ import annotations

from datetime import datetime

from jobshop.core.models import (
    Assignment,
    Instance,
    Machine,
    Operation,
    Order,
    Schedule,
    SolveInfo,
    SolveStatus,
    TimeWindow,
)
from jobshop.core.solver import SolverConfig

T0 = datetime(2026, 1, 5, 6, 0)

# One worker + fixed seed + short limit: deterministic and fast on tiny instances.
FAST = SolverConfig(time_limit_s=5.0, num_workers=1, seed=0)


def machine(machine_id, caps=("cut",), windows=((0, 1000),), downtime=()):
    return Machine(
        id=machine_id,
        type="test",
        capabilities=list(caps),
        availability=[TimeWindow(start=a, end=b) for a, b in windows],
        downtime=[TimeWindow(start=a, end=b) for a, b in downtime],
    )


def order(order_id, due, ops, priority=3, family="generic", notes=""):
    """ops is a list of (duration, capability); op ids become '<order>-op1', '-op2', ..."""
    return Order(
        id=order_id,
        family=family,
        due=due,
        priority=priority,
        operations=[
            Operation(id=f"{order_id}-op{k + 1}", duration=d, required_capability=cap)
            for k, (d, cap) in enumerate(ops)
        ],
        notes=notes,
    )


def instance(machines, orders, now=0, templates=None):
    return Instance(
        t0=T0, now=now, machines=machines, orders=orders, routing_templates=templates or {}
    )


def assign(op_id, order_id, machine_id, start, end):
    return Assignment(op_id=op_id, order_id=order_id, machine_id=machine_id, start=start, end=end)


def schedule(assignments, status=SolveStatus.OPTIMAL):
    return Schedule(assignments=list(assignments), solve_info=SolveInfo(status=status))
