"""The planner's alternative goal order: tardiness, then the earliest finish, then fewest moves.

The default order (tardiness, fewest moves, finish) can leave the shop finishing much later than it
has to, just to avoid moving a few operations. ``earliest_finish`` trades the other way.
"""

import pytest

from jobshop.core import changes
from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.kpis import compute_kpis, diff_schedules
from jobshop.core.models import SolveStatus
from jobshop.core.solver import solve
from jobshop.core.validator import validate_schedule
from tests.helpers import FAST, assign, instance, machine, order, schedule


def wasteful_plan():
    """Three 10-minute jobs on two machines in a valid but slow plan: gaps and an idle machine."""
    inst = instance([machine("M1"), machine("M2")], [order(x, 1000, [(10, "cut")]) for x in "ABC"])
    ref = schedule([assign("A-op1", "A", "M1", 0, 10), assign("B-op1", "B", "M2", 50, 60), assign("C-op1", "C", "M1", 100, 110)])
    return inst, ref


def test_the_default_keeps_the_plan_and_earliest_finish_compacts_it():
    inst, ref = wasteful_plan()
    default = solve(inst, config=FAST, stay_close_to=ref)
    fast = solve(inst, config=FAST, stay_close_to=ref, earliest_finish=True)

    assert compute_kpis(inst, default).makespan == 110 and default.solve_info.reported_moved_operations == 0
    # 30 minutes of work on two machines finishes at minute 20; only B and C have to move for that.
    assert compute_kpis(inst, fast).makespan == 20
    assert fast.solve_info.reported_moved_operations == 2
    assert {m.op_id for m in diff_schedules(inst, ref, inst, fast).moved_operations} == {"B-op1", "C-op1"}
    assert validate_schedule(inst, fast).ok


def test_among_the_earliest_finishes_it_still_moves_as_little_as_possible():
    inst, ref = wasteful_plan()
    fast = solve(inst, config=FAST, stay_close_to=ref, earliest_finish=True)
    # A already sits at minute 0 on M1 and is not needed elsewhere: it must not be shuffled for nothing.
    assert fast.by_op()["A-op1"].start == 0 and fast.by_op()["A-op1"].machine_id == "M1"
    assert fast.solve_info.status == SolveStatus.OPTIMAL and fast.solve_info.makespan_optimal


def test_fewest_moves_is_not_claimed_for_the_earliest_finish_plan():
    inst, ref = wasteful_plan()
    assert solve(inst, config=FAST, stay_close_to=ref).solve_info.stability_optimal is True
    assert solve(inst, config=FAST, stay_close_to=ref, earliest_finish=True).solve_info.stability_optimal is None


def test_tardiness_still_comes_first():
    # Priority 5 finishes late whichever way; the earliest-finish plan must not give that up for speed.
    inst = instance(
        [machine("M")],
        [order("A", 10, [(10, "cut")], priority=5), order("B", 10, [(10, "cut")], priority=1)],
    )
    ref = schedule([assign("B-op1", "B", "M", 0, 10), assign("A-op1", "A", "M", 10, 20)])
    fast = solve(inst, config=FAST, stay_close_to=ref, earliest_finish=True)
    assert fast.by_op()["A-op1"].start == 0
    assert compute_kpis(inst, fast).weighted_tardiness == 10


def test_without_a_reference_the_flag_changes_nothing():
    inst, _ = wasteful_plan()
    plain, flagged = solve(inst, config=FAST), solve(inst, config=FAST, earliest_finish=True)
    assert compute_kpis(inst, plain).makespan == compute_kpis(inst, flagged).makespan == 20
    assert flagged.solve_info.stability_optimal is None and flagged.solve_info.reported_moved_operations is None


def test_frozen_work_is_respected_and_not_counted_as_moved():
    inst, ref = wasteful_plan()
    frozen = [ref.by_op()["A-op1"]]
    fast = solve(inst, frozen=frozen, config=FAST, stay_close_to=ref, earliest_finish=True)
    assert fast.by_op()["A-op1"] == frozen[0]
    assert validate_schedule(inst, fast, frozen).ok
    assert fast.solve_info.reported_moved_operations == 2


@pytest.mark.parametrize("seed", range(3))
def test_on_generated_shops_the_first_goal_is_unchanged_and_the_second_can_only_improve(seed):
    base = generate_instance(GeneratorSettings(seed=seed, n_machines=4, n_orders=6, ops_min=2, ops_max=3, n_days=2))
    ref = solve(base, config=FAST)
    disrupted, _ = changes.add_downtime(base, "M1", 60, 240)

    default = solve(disrupted, config=FAST, hint=ref, stay_close_to=ref)
    fast = solve(disrupted, config=FAST, hint=ref, stay_close_to=ref, earliest_finish=True)

    for sched in (default, fast):
        assert validate_schedule(disrupted, sched).ok
    d, f = compute_kpis(disrupted, default), compute_kpis(disrupted, fast)
    if default.solve_info.status == fast.solve_info.status == SolveStatus.OPTIMAL:
        assert f.weighted_tardiness == d.weighted_tardiness  # the first goal is the same in both orders
        assert f.makespan <= d.makespan                       # the second goal can only be better
        assert fast.solve_info.reported_moved_operations >= default.solve_info.reported_moved_operations
    # the numbers it reports about itself match an independent comparison
    moved = len(diff_schedules(disrupted, ref, disrupted, fast).moved_operations)
    assert fast.solve_info.reported_moved_operations == moved


def test_if_the_second_stage_gets_no_time_the_fallback_is_the_stable_plan_not_an_arbitrary_one():
    from jobshop.core.solver import SolverConfig

    inst, ref = wasteful_plan()
    out_of_time = SolverConfig(time_limit_s=0.1, num_workers=1, seed=0)   # stage 2 needs 0.2 s left, so it is skipped
    fast = solve(inst, config=out_of_time, stay_close_to=ref, earliest_finish=True)

    assert fast.solve_info.makespan_optimal is False and fast.solve_info.status == SolveStatus.FEASIBLE
    assert fast.solve_info.reported_moved_operations == 0                   # the live plan, untouched
    assert compute_kpis(inst, fast).makespan == 110 and validate_schedule(inst, fast).ok
    assert fast.solve_info.stability_optimal is None                        # still never claimed for this goal
