"""CP-SAT model for the flexible job shop.

How the model works (read this before the code):

* Every operation gets one start variable and one end variable (end = start + duration).
* For each machine that is *eligible* for the operation there is an **optional interval**
  tied to those same start/end variables and to a Boolean "this machine is chosen".
  Exactly one machine's Boolean is true per operation. That is the "flexible" part.
* Each machine gets a ``NoOverlap`` constraint over its optional intervals **plus fixed
  intervals covering every minute it is closed** (outside shifts, or in downtime). A
  non-preemptive operation therefore cannot overlap closed time, so it has to fit
  entirely inside one open window.
* Operations of one order run in sequence: each starts after the previous one ends.
* Objective is lexicographic and solved in two stages:
    1. minimize weighted tardiness (weight comes from order priority);
    2. fix that tardiness value as a constraint, then minimize makespan.
  Two stages keep the priorities strict (no weighting trade-off between tardiness and
  makespan) and let us report honestly which stage was proven optimal.

Rescheduling: ``instance.now`` is the current minute. Operations in ``frozen`` keep their
machine and start; every other operation must start at or after ``now``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from jobshop.core import intervals
from jobshop.core.models import (
    Assignment,
    Instance,
    Schedule,
    SolveInfo,
    SolveStatus,
)


@dataclass(frozen=True)
class SolverConfig:
    time_limit_s: float = 10.0
    # CP-SAT with several workers is not deterministic; tests use 1 worker.
    num_workers: int = 8
    seed: int = 0
    # Share of the time budget stage 1 may use. Stage 2 gets whatever is left.
    stage1_fraction: float = 0.6


@dataclass
class _Built:
    model: cp_model.CpModel
    start: dict[str, cp_model.IntVar]
    presence: dict[tuple[str, str], cp_model.IntVar]  # (op_id, machine_id) -> chosen?
    weighted_tardiness: cp_model.LinearExpr
    makespan: cp_model.IntVar
    domains: dict[str, tuple[int, int]]  # op_id -> (earliest start, latest start)


def solve(
    instance: Instance,
    frozen: Sequence[Assignment] = (),
    config: SolverConfig = SolverConfig(),
    hint: Schedule | None = None,
) -> Schedule:
    """Solve ``instance``. ``hint`` (typically the committed schedule) warm-starts the search.

    A hint only guides the search; it never constrains the answer. Starting from the
    current plan makes a re-solve far less noisy, because CP-SAT begins from a good
    solution instead of from scratch.
    """
    started = time.monotonic()
    frozen_by_op = _check_frozen(instance, frozen)

    if not any(order.operations for order in instance.orders):
        info = _info(config, started, SolveStatus.OPTIMAL, True, True, 0.0, 0, 0)
        return Schedule(assignments=[], solve_info=info)

    built = _build(instance, frozen_by_op)
    if built is None:  # some operation cannot fit before the horizon at all
        return _no_schedule(config, started, SolveStatus.INFEASIBLE)

    # Stage 1: weighted tardiness.
    if hint is not None:
        _apply_hint(built, hint)
    built.model.minimize(built.weighted_tardiness)
    solver1 = _make_solver(config, config.time_limit_s * config.stage1_fraction)
    status1 = solver1.solve(built.model)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _no_schedule(config, started, _map_failure(status1))

    best_tardiness = round(solver1.objective_value)
    tardiness_optimal = status1 == cp_model.OPTIMAL
    tardiness_bound = solver1.best_objective_bound
    assignments = _extract(instance, built, solver1)
    reported_tardiness = best_tardiness
    reported_makespan = solver1.value(built.makespan)
    makespan_optimal = False

    # Stage 2: keep tardiness no worse, minimize makespan. Rebuilding is cheap and keeps
    # stage 1's model untouched; the stage 1 solution is passed in as a hint so stage 2
    # always has a valid starting point even with little time left.
    remaining = config.time_limit_s - (time.monotonic() - started)
    if remaining >= 0.2:
        built2 = _build(instance, frozen_by_op)
        assert built2 is not None  # same instance as stage 1, which succeeded
        built2.model.add(built2.weighted_tardiness <= best_tardiness)
        _copy_hint(built, solver1, built2)
        built2.model.minimize(built2.makespan)
        solver2 = _make_solver(config, remaining)
        status2 = solver2.solve(built2.model)
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignments = _extract(instance, built2, solver2)
            reported_tardiness = round(solver2.value(built2.weighted_tardiness))
            reported_makespan = solver2.value(built2.makespan)
            makespan_optimal = status2 == cp_model.OPTIMAL

    status = (
        SolveStatus.OPTIMAL if tardiness_optimal and makespan_optimal else SolveStatus.FEASIBLE
    )
    info = _info(
        config, started, status, tardiness_optimal, makespan_optimal,
        tardiness_bound, reported_tardiness, reported_makespan,
    )
    return Schedule(assignments=assignments, solve_info=info)


def _build(instance: Instance, frozen_by_op: dict[str, Assignment]) -> _Built | None:
    horizon = instance.horizon
    model = cp_model.CpModel()
    start: dict[str, cp_model.IntVar] = {}
    end: dict[str, cp_model.IntVar] = {}
    presence: dict[tuple[str, str], cp_model.IntVar] = {}
    machine_intervals: dict[str, list[cp_model.IntervalVar]] = {m.id: [] for m in instance.machines}
    domains: dict[str, tuple[int, int]] = {}

    for order in instance.orders:
        previous_end: cp_model.IntVar | None = None
        for op in order.operations:
            frozen = frozen_by_op.get(op.id)
            # Frozen operations may sit before ``now`` (they already started).
            earliest = 0 if frozen else instance.now
            latest_start = horizon - op.duration
            if latest_start < earliest:
                return None

            s = model.new_int_var(earliest, latest_start, f"s_{op.id}")
            e = model.new_int_var(earliest + op.duration, horizon, f"e_{op.id}")
            model.add(e == s + op.duration)
            start[op.id], end[op.id] = s, e
            domains[op.id] = (earliest, latest_start)

            chosen = []
            for machine in instance.eligible_machines(op):
                p = model.new_bool_var(f"on_{op.id}_{machine.id}")
                interval = model.new_optional_interval_var(
                    s, op.duration, e, p, f"iv_{op.id}_{machine.id}"
                )
                presence[(op.id, machine.id)] = p
                machine_intervals[machine.id].append(interval)
                chosen.append(p)
            model.add_exactly_one(chosen)

            if frozen is not None:
                model.add(s == frozen.start)
                model.add(presence[(op.id, frozen.machine_id)] == 1)
            if previous_end is not None:
                model.add(s >= previous_end)
            previous_end = e

    for machine in instance.machines:
        closed = intervals.complement(machine.free_windows(), 0, horizon)
        blocks = [
            model.new_fixed_size_interval_var(a, b - a, f"closed_{machine.id}_{i}")
            for i, (a, b) in enumerate(closed)
        ]
        model.add_no_overlap(machine_intervals[machine.id] + blocks)

    terms = []
    for order in instance.orders:
        # Equality (not ">=") so tardiness never carries slack: the objective value the
        # solver reports is then exactly the weighted tardiness of the schedule it returns.
        tardiness = model.new_int_var(0, horizon, f"late_{order.id}")
        model.add_max_equality(tardiness, [0, end[order.operations[-1].id] - order.due])
        terms.append(order.weight * tardiness)
    weighted_tardiness = cp_model.LinearExpr.sum(terms)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, list(end.values()))

    return _Built(model, start, presence, weighted_tardiness, makespan, domains)


def _check_frozen(instance: Instance, frozen: Sequence[Assignment]) -> dict[str, Assignment]:
    ops = {op.id: (order.id, op) for order in instance.orders for op in order.operations}
    machines = {m.id: m for m in instance.machines}
    by_op: dict[str, Assignment] = {}
    for a in frozen:
        if a.op_id not in ops:
            raise ValueError(f"frozen assignment for unknown operation {a.op_id}")
        order_id, op = ops[a.op_id]
        if a.order_id != order_id:
            raise ValueError(f"frozen assignment {a.op_id} names the wrong order")
        if a.machine_id not in machines:
            raise ValueError(f"frozen assignment {a.op_id} uses unknown machine {a.machine_id}")
        if op.required_capability not in machines[a.machine_id].capabilities:
            raise ValueError(f"frozen assignment {a.op_id} uses ineligible machine {a.machine_id}")
        if a.end - a.start != op.duration:
            raise ValueError(f"frozen assignment {a.op_id} has the wrong duration")
        if a.op_id in by_op:
            raise ValueError(f"operation {a.op_id} is frozen twice")
        by_op[a.op_id] = a
    return by_op


def _make_solver(config: SolverConfig, time_limit_s: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(time_limit_s, 0.05)
    solver.parameters.num_workers = config.num_workers
    solver.parameters.random_seed = config.seed
    return solver


def _apply_hint(built: _Built, hint: Schedule) -> None:
    """Suggest the hinted machine and start for every operation the model still has."""
    for a in hint.assignments:
        domain = built.domains.get(a.op_id)
        if domain is None or not domain[0] <= a.start <= domain[1]:
            continue  # unknown operation, or the hinted start is no longer allowed
        if (a.op_id, a.machine_id) not in built.presence:
            continue
        built.model.add_hint(built.start[a.op_id], a.start)
        for (op_id, machine_id), var in built.presence.items():
            if op_id == a.op_id:
                built.model.add_hint(var, 1 if machine_id == a.machine_id else 0)


def _copy_hint(source: _Built, solver: cp_model.CpSolver, target: _Built) -> None:
    for op_id, var in source.start.items():
        target.model.add_hint(target.start[op_id], solver.value(var))
    for key, var in source.presence.items():
        target.model.add_hint(target.presence[key], solver.value(var))


def _extract(instance: Instance, built: _Built, solver: cp_model.CpSolver) -> list[Assignment]:
    assignments = []
    for order in instance.orders:
        for op in order.operations:
            machine_id = next(
                m.id
                for m in instance.eligible_machines(op)
                if solver.value(built.presence[(op.id, m.id)]) == 1
            )
            begin = solver.value(built.start[op.id])
            assignments.append(
                Assignment(
                    op_id=op.id,
                    order_id=order.id,
                    machine_id=machine_id,
                    start=begin,
                    end=begin + op.duration,
                )
            )
    return assignments


def _map_failure(status: int) -> SolveStatus:
    if status == cp_model.INFEASIBLE:
        return SolveStatus.INFEASIBLE
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError("CP-SAT rejected the model as invalid (this is a bug in the solver code)")
    return SolveStatus.UNKNOWN


def _no_schedule(config: SolverConfig, started: float, status: SolveStatus) -> Schedule:
    info = _info(config, started, status, False, False, None, None, None)
    return Schedule(assignments=[], solve_info=info)


def _info(
    config: SolverConfig,
    started: float,
    status: SolveStatus,
    tardiness_optimal: bool,
    makespan_optimal: bool,
    tardiness_bound: float | None,
    reported_tardiness: int | None,
    reported_makespan: int | None,
) -> SolveInfo:
    return SolveInfo(
        status=status,
        tardiness_optimal=tardiness_optimal,
        makespan_optimal=makespan_optimal,
        tardiness_bound=tardiness_bound,
        reported_weighted_tardiness=reported_tardiness,
        reported_makespan=reported_makespan,
        wall_time_s=round(time.monotonic() - started, 3),
        num_workers=config.num_workers,
        seed=config.seed,
    )
