import pytest

from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.kpis import compute_kpis
from jobshop.core.models import SolveStatus
from jobshop.core.solver import SolverConfig, solve
from jobshop.core.validator import validate_schedule
from tests.helpers import FAST, assign, instance, machine, order


def solved(inst, **kwargs):
    sched = solve(inst, config=FAST, **kwargs)
    return sched, sched.by_op()


def test_higher_priority_order_is_scheduled_first():
    # One machine, two 10-minute jobs both due at 10: only one can be on time.
    # Delaying the priority-1 job (weight 1) costs 10; delaying priority 5 (weight 16) costs 160.
    inst = instance(
        [machine("M")],
        [order("A", 10, [(10, "cut")], priority=1), order("B", 10, [(10, "cut")], priority=5)],
    )
    sched, ops = solved(inst)
    assert sched.solve_info.status == SolveStatus.OPTIMAL
    assert ops["B-op1"].end == 10 and ops["A-op1"].end == 20
    assert compute_kpis(inst, sched).weighted_tardiness == 10
    assert sched.solve_info.reported_weighted_tardiness == 10


def test_flexible_orders_run_in_parallel_on_different_machines():
    inst = instance(
        [machine("M1"), machine("M2")],
        [order("X", 1000, [(30, "cut")]), order("Y", 1000, [(30, "cut")])],
    )
    sched, ops = solved(inst)
    assert compute_kpis(inst, sched).makespan == 30
    assert ops["X-op1"].machine_id != ops["Y-op1"].machine_id


def test_operation_never_straddles_a_closed_gap():
    # Machine open [0,10) and [20,60). A 15-minute job cannot start in the first window.
    inst = instance([machine("M", windows=[(0, 10), (20, 60)])], [order("A", 1000, [(15, "cut")])])
    _, ops = solved(inst)
    assert ops["A-op1"].start == 20


def test_operation_waits_out_downtime():
    inst = instance(
        [machine("M", windows=[(0, 100)], downtime=[(0, 30)])], [order("A", 1000, [(20, "cut")])]
    )
    _, ops = solved(inst)
    assert ops["A-op1"].start == 30


def test_operation_moves_to_another_machine_when_one_is_down():
    inst = instance(
        [machine("M1", windows=[(0, 100)], downtime=[(0, 50)]), machine("M2", windows=[(0, 100)])],
        [order("A", 1000, [(20, "cut")])],
    )
    _, ops = solved(inst)
    assert ops["A-op1"].machine_id == "M2" and ops["A-op1"].start == 0


def test_machine_closed_for_the_whole_horizon_is_infeasible():
    inst = instance([machine("M", windows=[(0, 100)], downtime=[(0, 100)])], [order("A", 1000, [(10, "cut")])])
    sched, _ = solved(inst)
    assert sched.solve_info.status == SolveStatus.INFEASIBLE
    assert sched.assignments == []


def test_operation_longer_than_the_horizon_is_infeasible():
    inst = instance([machine("M", windows=[(0, 50)])], [order("A", 1000, [(60, "cut")])])
    sched, _ = solved(inst)
    assert sched.solve_info.status == SolveStatus.INFEASIBLE


def test_operations_of_an_order_run_in_sequence_across_machines():
    inst = instance(
        [machine("M1", caps=("cut",)), machine("M2", caps=("drill",))],
        [order("A", 1000, [(10, "cut"), (10, "drill")])],
    )
    _, ops = solved(inst)
    assert ops["A-op2"].start >= ops["A-op1"].end
    assert (ops["A-op1"].machine_id, ops["A-op2"].machine_id) == ("M1", "M2")


def test_second_stage_compacts_the_schedule_without_hurting_tardiness():
    # Tardiness is 0 however the jobs are placed; stage 2 must still pack them tightly.
    inst = instance(
        [machine("M", windows=[(0, 500)])],
        [order("A", 10, [(10, "cut")], priority=5), order("B", 1000, [(10, "cut")], priority=1)],
    )
    sched, _ = solved(inst)
    kpis = compute_kpis(inst, sched)
    assert kpis.makespan == 20 and kpis.weighted_tardiness == 0
    assert sched.solve_info.status == SolveStatus.OPTIMAL
    assert sched.solve_info.tardiness_optimal and sched.solve_info.makespan_optimal


def test_unfrozen_operations_start_at_or_after_now():
    inst = instance([machine("M", windows=[(0, 500)])], [order("A", 1000, [(10, "cut"), (10, "cut")])], now=100)
    sched, _ = solved(inst)
    assert all(a.start >= 100 for a in sched.assignments)
    assert validate_schedule(inst, sched).ok


def test_frozen_operation_stays_put_and_the_rest_works_around_it():
    # M1 is busy with A-op1 from 40 to 70 (already started: now=50). Everything else is >= 50.
    inst = instance(
        [machine("M1", windows=[(0, 300)]), machine("M2", windows=[(0, 300)])],
        [order("A", 200, [(30, "cut"), (20, "cut")]), order("B", 100, [(20, "cut")])],
        now=50,
    )
    frozen = [assign("A-op1", "A", "M1", 40, 70)]
    sched, ops = solved(inst, frozen=frozen)
    assert (ops["A-op1"].machine_id, ops["A-op1"].start) == ("M1", 40)
    assert ops["A-op2"].start >= 70
    assert ops["B-op1"].start >= 50
    assert validate_schedule(inst, sched, frozen).ok


@pytest.mark.parametrize(
    "bad, message",
    [
        (assign("NOPE", "A", "M1", 0, 10), "unknown operation"),
        (assign("A-op1", "A", "M9", 0, 10), "unknown machine"),
        (assign("A-op1", "A", "M2", 0, 10), "ineligible"),
        (assign("A-op1", "A", "M1", 0, 99), "wrong duration"),
        (assign("A-op1", "Z", "M1", 0, 10), "wrong order"),
    ],
)
def test_inconsistent_frozen_input_is_rejected(bad, message):
    inst = instance(
        [machine("M1", caps=("cut",)), machine("M2", caps=("drill",))], [order("A", 100, [(10, "cut")])]
    )
    with pytest.raises(ValueError, match=message):
        solve(inst, frozen=[bad], config=FAST)


def test_instance_without_orders_is_trivially_optimal():
    sched = solve(instance([machine("M")], []), config=FAST)
    assert sched.solve_info.status == SolveStatus.OPTIMAL and sched.assignments == []


def test_single_worker_runs_are_repeatable():
    inst = generate_instance(GeneratorSettings(seed=2, n_machines=3, n_orders=4, ops_min=2, ops_max=3, n_days=2))
    first, second = solve(inst, config=FAST), solve(inst, config=FAST)
    assert first.assignments == second.assignments


@pytest.mark.parametrize("seed", range(5))
def test_generated_small_instances_solve_validly_and_match_recomputed_kpis(seed):
    settings = GeneratorSettings(seed=seed, n_machines=4, n_orders=6, ops_min=2, ops_max=3, n_days=2)
    inst = generate_instance(settings)
    sched = solve(inst, config=FAST)
    assert sched.solve_info.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)
    report = validate_schedule(inst, sched)
    assert report.ok, report.violations
    # Two independent code paths (solver internals vs. kpis) must agree.
    kpis = compute_kpis(inst, sched)
    assert sched.solve_info.reported_weighted_tardiness == kpis.weighted_tardiness
    assert sched.solve_info.reported_makespan == kpis.makespan


@pytest.mark.slow
def test_default_size_instance_returns_a_valid_schedule():
    inst = generate_instance(GeneratorSettings(seed=0))
    sched = solve(inst, config=SolverConfig(time_limit_s=10.0, num_workers=8, seed=0))
    assert sched.solve_info.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)
    assert validate_schedule(inst, sched).ok
    kpis = compute_kpis(inst, sched)
    assert sched.solve_info.reported_weighted_tardiness == kpis.weighted_tardiness
    assert sched.solve_info.reported_makespan == kpis.makespan


# --- warm start -------------------------------------------------------------------------------


def _small_instance(seed=2):
    return generate_instance(GeneratorSettings(seed=seed, n_machines=4, n_orders=6, ops_min=2, ops_max=3, n_days=2))


def test_warm_start_from_an_optimal_schedule_gives_an_equally_good_valid_schedule():
    inst = _small_instance()
    first = solve(inst, config=FAST)
    again = solve(inst, config=FAST, hint=first)
    assert validate_schedule(inst, again).ok
    assert again.solve_info.reported_weighted_tardiness == first.solve_info.reported_weighted_tardiness
    assert again.solve_info.reported_makespan == first.solve_info.reported_makespan


def test_hint_never_forces_the_answer_when_it_is_no_longer_valid():
    # The hint puts every operation at start 0 on one machine: wildly infeasible. The solver
    # must ignore the bad parts and still return a correct schedule.
    inst = _small_instance()
    first = solve(inst, config=FAST)
    bad = first.model_copy(update={"assignments": [
        a.model_copy(update={"start": 0, "end": a.end - a.start, "machine_id": "M1"})
        for a in first.assignments
    ]})
    sched = solve(inst, config=FAST, hint=bad)
    assert validate_schedule(inst, sched).ok


def test_hint_entries_for_unknown_operations_or_out_of_range_starts_are_ignored():
    inst = _small_instance()
    first = solve(inst, config=FAST)
    junk = first.model_copy(update={"assignments": first.assignments + [
        assign("NOPE-op1", "NOPE", "M1", 0, 10),
        first.assignments[0].model_copy(update={"start": 10**6, "end": 10**6 + 5}),
    ]})
    assert validate_schedule(inst, solve(inst, config=FAST, hint=junk)).ok


# --- stability stage (fewest operations moved from a reference plan) ---------------------------

from jobshop.core import changes  # noqa: E402
from jobshop.core.kpis import diff_schedules  # noqa: E402
from tests.helpers import schedule  # noqa: E402


def _as_tuples(sched):
    return sorted((a.op_id, a.machine_id, a.start, a.end) for a in sched.assignments)


def test_nothing_moves_when_nothing_needs_to_change():
    inst = instance(
        [machine("M1"), machine("M2")],
        [order(x, 1000, [(10, "cut")]) for x in "ABC"],
    )
    # A deliberately wasteful but valid plan: gaps and unbalanced machines.
    ref = schedule([assign("A-op1", "A", "M1", 0, 10), assign("B-op1", "B", "M2", 50, 60), assign("C-op1", "C", "M1", 100, 110)])
    sched = solve(inst, config=FAST, stay_close_to=ref)

    assert _as_tuples(sched) == _as_tuples(ref)
    assert sched.solve_info.reported_moved_operations == 0 and sched.solve_info.stability_optimal is True
    assert sched.solve_info.status == SolveStatus.OPTIMAL
    # Stability outranks makespan: the plan is kept even though it is not compact...
    assert compute_kpis(inst, sched).makespan == 110
    # ...whereas without a reference the solver packs it tightly.
    assert compute_kpis(inst, solve(inst, config=FAST)).makespan == 20


def test_tardiness_outranks_stability():
    # The reference runs the low-priority job first. Staying put would make the priority-5
    # job late (cost 160); the best tardiness (10) requires swapping them, so both move.
    inst = instance(
        [machine("M")],
        [order("A", 10, [(10, "cut")], priority=5), order("B", 10, [(10, "cut")], priority=1)],
    )
    ref = schedule([assign("B-op1", "B", "M", 0, 10), assign("A-op1", "A", "M", 10, 20)])
    sched = solve(inst, config=FAST, stay_close_to=ref)
    ops = sched.by_op()
    assert ops["A-op1"].start == 0 and ops["B-op1"].start == 10
    assert compute_kpis(inst, sched).weighted_tardiness == 10
    assert sched.solve_info.reported_moved_operations == 2


def test_a_disruption_moves_only_the_operations_it_forces():
    # M1: A,B   M2: C,D,E (10 minutes each, loose due dates). M2 goes down 10-20, hitting D.
    base = instance(
        [machine("M1"), machine("M2")],
        [order(x, 1000, [(10, "cut")]) for x in "ABCDE"],
    )
    ref = schedule([
        assign("A-op1", "A", "M1", 0, 10), assign("B-op1", "B", "M1", 10, 20),
        assign("C-op1", "C", "M2", 0, 10), assign("D-op1", "D", "M2", 10, 20), assign("E-op1", "E", "M2", 20, 30),
    ])
    disrupted, _ = changes.add_downtime(base, "M2", 10, 20)
    sched = solve(disrupted, config=FAST, hint=ref, stay_close_to=ref)

    changed = {a.op_id for a in sched.assignments} - {a.op_id for a in sched.assignments if a in ref.assignments}
    assert changed == {"D-op1"}  # everything else stayed exactly where it was
    assert sched.by_op()["D-op1"].start >= 20 or sched.by_op()["D-op1"].machine_id == "M1"
    assert sched.solve_info.reported_moved_operations == 1 and sched.solve_info.stability_optimal is True
    assert validate_schedule(disrupted, sched).ok


def test_without_a_reference_the_stability_stage_does_not_run():
    inst = _small_instance()
    info = solve(inst, config=FAST).solve_info
    assert info.stability_optimal is None and info.reported_moved_operations is None


@pytest.mark.parametrize("seed", range(3))
def test_reported_moved_count_matches_an_independent_comparison(seed):
    inst = _small_instance(seed)
    base = solve(inst, config=FAST)
    disrupted, _ = changes.add_downtime(inst, "M2", 60, 200)
    new = solve(disrupted, config=FAST, hint=base, stay_close_to=base)

    assert validate_schedule(disrupted, new).ok
    diff = diff_schedules(inst, base, disrupted, new)
    assert new.solve_info.reported_moved_operations == len(diff.moved_operations)


def test_new_operations_are_not_counted_as_moved():
    inst = _small_instance()
    base = solve(inst, config=FAST)
    family = next(iter(inst.routing_templates))
    with_rush = changes.add_rush_order(inst, "RUSH-1", family, due=900)
    new = solve(with_rush, config=FAST, hint=base, stay_close_to=base)
    diff = diff_schedules(inst, base, with_rush, new)
    assert diff.added_operations and new.solve_info.reported_moved_operations == len(diff.moved_operations)
    assert validate_schedule(with_rush, new).ok


def test_frozen_operations_are_excluded_from_the_stability_count():
    inst = instance(
        [machine("M1", windows=[(0, 300)]), machine("M2", windows=[(0, 300)])],
        [order("A", 200, [(30, "cut"), (20, "cut")]), order("B", 100, [(20, "cut")])],
        now=50,
    )
    ref = schedule([assign("A-op1", "A", "M1", 40, 70), assign("A-op2", "A", "M1", 70, 90), assign("B-op1", "B", "M2", 50, 70)])
    frozen = [ref.assignments[0]]
    sched = solve(inst, frozen=frozen, config=FAST, hint=ref, stay_close_to=ref)
    assert _as_tuples(sched) == _as_tuples(ref)
    assert sched.solve_info.reported_moved_operations == 0
    assert validate_schedule(inst, sched, frozen).ok


def test_one_unit_of_tardiness_outweighs_any_number_of_moves():
    # One priority-1 job (due 29) sits last, finishing at 30: just 1 minute late (cost 1).
    # Fixing that means swapping it with an earlier job: 2 operations move to save 1 unit.
    # Strict lexicographic order says fix the lateness anyway; a too-small stability weight
    # (cost 1 to stay vs 2 to move) would keep the plan.
    inst = instance(
        [machine("M")],
        [order("P", 29, [(10, "cut")], priority=1), order("Q", 1000, [(10, "cut")]), order("R", 1000, [(10, "cut")])],
    )
    ref = schedule([assign("Q-op1", "Q", "M", 0, 10), assign("R-op1", "R", "M", 10, 20), assign("P-op1", "P", "M", 20, 30)])
    sched = solve(inst, config=FAST, stay_close_to=ref)
    assert compute_kpis(inst, sched).weighted_tardiness == 0
    assert sched.solve_info.reported_moved_operations == 2


def test_makespan_stage_never_gives_up_tardiness():
    # The machine is open [0,15) and [100,300). Only one job fits in the first window.
    #   A (priority 5, due 10, 10 min) first  -> A on time, D at 100-112: makespan 112
    #   D (12 min) first                      -> A at 100-110, 100 min late: makespan 110
    # The shorter makespan is the worse plan. Tardiness is fixed first, so makespan must not win.
    inst = instance(
        [machine("M", windows=[(0, 15), (100, 300)])],
        [order("A", 10, [(10, "cut")], priority=5), order("D", 1000, [(12, "cut")], priority=1)],
    )
    sched = solve(inst, config=FAST)
    ops = sched.by_op()
    assert ops["A-op1"].start == 0 and ops["D-op1"].start == 100
    kpis = compute_kpis(inst, sched)
    assert (kpis.weighted_tardiness, kpis.makespan) == (0, 112)
    assert sched.solve_info.status == SolveStatus.OPTIMAL
