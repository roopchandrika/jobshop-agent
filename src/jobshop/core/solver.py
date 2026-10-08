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
* Objective is lexicographic, solved in two stages. After stage 1 its optimum is fixed as
  a constraint, so stage 2 can never undo it:
    1. minimize weighted tardiness (weight comes from order priority) and, as a strict
       tie-break, the number of operations that differ from a reference plan (different
       machine or different start) when ``stay_close_to`` is given;
    2. minimize makespan.
  Strict priorities (no weighting trade-off) let us report which part was proven optimal.
  ``earliest_finish=True`` is the planner's alternative order: tardiness, then makespan, then
  the number of operations moved (the same folding trick, applied to stage 2).

Why the stability tie-break exists: tardiness and makespan do not care where unaffected
work sits, so a re-solve reshuffles most of the plan for no reason (an empty change moved
58 of 81 operations in measurement). The tie-break makes a change move only what it must.

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
    # Share of the whole budget the tardiness stage may use.
    stage1_fraction: float = 0.6


@dataclass
class _Built:
    model: cp_model.CpModel
    start: dict[str, cp_model.IntVar]
    presence: dict[tuple[str, str], cp_model.IntVar]  # (op_id, machine_id) -> chosen?
    weighted_tardiness: cp_model.LinearExpr
    makespan: cp_model.IntVar
    domains: dict[str, tuple[int, int]]  # op_id -> (earliest start, latest start)


@dataclass
class _Solved:
    """A finished stage: its model, its solver, and (if it had one) the 'moved' expression."""

    built: _Built
    solver: cp_model.CpSolver
    moved: cp_model.LinearExpr | None = None


def solve(
    instance: Instance,
    frozen: Sequence[Assignment] = (),
    config: SolverConfig = SolverConfig(),
    hint: Schedule | None = None,
    stay_close_to: Schedule | None = None,
    earliest_finish: bool = False,
) -> Schedule:
    """Solve ``instance``.

    ``hint`` (typically the committed schedule) only warm-starts the search; it never
    constrains the answer. ``stay_close_to`` adds the stability tie-break: among schedules
    with the best tardiness, prefer the one that differs from this reference in the fewest
    operations. Frozen operations are never counted (they cannot move).

    ``earliest_finish`` swaps the last two goals when there is a reference: tardiness, then the
    earliest finish, then fewest moves. The default order can leave the shop finishing hours later
    than necessary just to avoid moving a few operations; this is the planner's alternative.
    ``stability_optimal`` is then left unset: "fewest moves" is no longer the second goal.
    """
    started = time.monotonic()
    frozen_by_op = _check_frozen(instance, frozen)

    if not any(order.operations for order in instance.orders):
        info = _info(config, started, SolveStatus.OPTIMAL, tardiness_optimal=True,
                     makespan_optimal=True, tardiness_bound=0.0,
                     reported_tardiness=0, reported_makespan=0)
        return Schedule(assignments=[], solve_info=info)

    built = _build(instance, frozen_by_op)
    if built is None:  # some operation cannot fit before the horizon at all
        return _no_schedule(config, started, SolveStatus.INFEASIBLE)

    def remaining() -> float:
        return config.time_limit_s - (time.monotonic() - started)

    # ---- Stage 1: weighted tardiness (then fewest moved operations, as a tie-break) -----------
    # With a reference, the objective is  (K + 1) * weighted_tardiness + moved  where K is the
    # most operations that could be counted as moved. Because 0 <= moved <= K, one unit of
    # tardiness always outweighs any number of moves, so minimizing this single number is
    # exactly "tardiness first, then fewest moves". Folding the tie-break into the first solve
    # (rather than a separate later stage) matters: it steers the search toward the reference
    # from the start instead of letting it wander far away and then trying to walk back.
    if hint is not None:
        _apply_hint(built, hint)
    moved1, comparable = (None, 0)
    finish_first = earliest_finish and stay_close_to is not None  # moves are decided in stage 2 instead
    if stay_close_to is not None and not finish_first:
        moved1, comparable = _add_stability(built, stay_close_to, frozen_by_op)
    scale = comparable + 1
    if moved1 is not None and comparable:
        built.model.minimize(scale * built.weighted_tardiness + moved1)
    else:
        moved1 = None
        built.model.minimize(built.weighted_tardiness)

    solver1 = _make_solver(config, config.time_limit_s * config.stage1_fraction)
    status1 = solver1.solve(built.model)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _no_schedule(config, started, _map_failure(status1))

    best = _Solved(built, solver1, moved1)
    best_tardiness = round(solver1.value(built.weighted_tardiness))
    best_moved = round(solver1.value(moved1)) if moved1 is not None else None
    stage1_optimal = status1 == cp_model.OPTIMAL
    # A proof of the combined objective proves both parts, in that priority order.
    tardiness_optimal = stage1_optimal
    stability_optimal: bool | None = stage1_optimal if moved1 is not None else None
    # Lower bound on weighted tardiness implied by the bound on the combined objective.
    tardiness_bound = max(0.0, (solver1.best_objective_bound - comparable) / scale)

    # ---- Stage 2: makespan, with the earlier optima held fixed -------------------------------
    # Rebuilding is cheap; the best solution so far is passed in as a hint so this stage always
    # has a valid starting point even when little time is left.
    makespan_optimal = False
    if remaining() >= 0.2:
        built2 = _build(instance, frozen_by_op)
        assert built2 is not None  # same instance as stage 1, which succeeded
        built2.model.add(built2.weighted_tardiness <= best_tardiness)
        moved2 = None
        objective2 = built2.makespan
        if finish_first:
            assert stay_close_to is not None
            moved2, counted = _add_stability(built2, stay_close_to, frozen_by_op)
            if counted:  # one minute of makespan outweighs any number of moves
                objective2 = (counted + 1) * built2.makespan + moved2
            else:
                moved2 = None
        elif best_moved is not None:
            assert stay_close_to is not None
            moved2, _ = _add_stability(built2, stay_close_to, frozen_by_op)
            built2.model.add(moved2 <= best_moved)
        _copy_hint(best.built, best.solver, built2)
        built2.model.minimize(objective2)
        solver2 = _make_solver(config, remaining())
        status2 = solver2.solve(built2.model)
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            best = _Solved(built2, solver2, moved2)
            makespan_optimal = status2 == cp_model.OPTIMAL

    proven = tardiness_optimal and makespan_optimal and stability_optimal is not False
    info = _info(
        config, started,
        SolveStatus.OPTIMAL if proven else SolveStatus.FEASIBLE,
        tardiness_optimal=tardiness_optimal,
        makespan_optimal=makespan_optimal,
        stability_optimal=stability_optimal,
        tardiness_bound=tardiness_bound,
        reported_tardiness=round(best.solver.value(best.built.weighted_tardiness)),
        reported_makespan=best.solver.value(best.built.makespan),
        reported_moved=round(best.solver.value(best.moved)) if best.moved is not None else None,
    )
    return Schedule(assignments=_extract(instance, best.built, best.solver), solve_info=info)


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


def _add_stability(
    built: _Built, reference: Schedule, frozen_by_op: dict[str, Assignment]
) -> tuple[cp_model.LinearExpr, int]:
    """Add 'unchanged' indicators and return (number of changed operations, how many were counted).

    An operation counts as unchanged only if it is on the same machine at the same start as in
    ``reference``. Each indicator is tied to that fact in BOTH directions (no slack), so the
    count the solver reports is exactly the count you would get by comparing the schedules.

    Not counted: frozen operations (they cannot move), operations missing from the reference,
    and operations whose reference start is no longer allowed (e.g. they must restart after
    ``now``). Those are neither a choice nor a cost the solver can influence.
    """
    model = built.model
    unchanged = []
    for a in reference.assignments:
        domain = built.domains.get(a.op_id)
        on_ref_machine = built.presence.get((a.op_id, a.machine_id))
        if a.op_id in frozen_by_op or domain is None or on_ref_machine is None:
            continue
        if not domain[0] <= a.start <= domain[1]:
            continue

        same_start = model.new_bool_var(f"same_start_{a.op_id}")
        model.add(built.start[a.op_id] == a.start).only_enforce_if(same_start)
        model.add(built.start[a.op_id] != a.start).only_enforce_if(~same_start)

        same = model.new_bool_var(f"same_{a.op_id}")
        model.add_bool_and([same_start, on_ref_machine]).only_enforce_if(same)
        model.add_bool_or([~same_start, ~on_ref_machine]).only_enforce_if(~same)
        unchanged.append(same)
    return len(unchanged) - cp_model.LinearExpr.sum(unchanged), len(unchanged)


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
    return Schedule(assignments=[], solve_info=_info(config, started, status))


def _info(
    config: SolverConfig,
    started: float,
    status: SolveStatus,
    *,
    tardiness_optimal: bool = False,
    makespan_optimal: bool = False,
    stability_optimal: bool | None = None,
    tardiness_bound: float | None = None,
    reported_tardiness: int | None = None,
    reported_makespan: int | None = None,
    reported_moved: int | None = None,
) -> SolveInfo:
    return SolveInfo(
        status=status,
        tardiness_optimal=tardiness_optimal,
        makespan_optimal=makespan_optimal,
        stability_optimal=stability_optimal,
        tardiness_bound=tardiness_bound,
        reported_weighted_tardiness=reported_tardiness,
        reported_makespan=reported_makespan,
        reported_moved_operations=reported_moved,
        wall_time_s=round(time.monotonic() - started, 3),
        num_workers=config.num_workers,
        seed=config.seed,
    )
