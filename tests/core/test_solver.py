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
