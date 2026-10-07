from jobshop.core import changes
from jobshop.core.reschedule import plan_reschedule, started_assignments
from jobshop.core.solver import solve
from jobshop.core.validator import validate_schedule
from tests.helpers import FAST, assign, instance, machine, order, schedule

# One machine. Order A: op1 on [0,40) then op2 on [40,80). Order B: op1 on [80,120).
BASE = instance(
    [machine("M1", windows=[(0, 400)]), machine("M2", windows=[(0, 400)])],
    [order("A", 500, [(40, "cut"), (40, "cut")]), order("B", 500, [(40, "cut")])],
)
COMMITTED = schedule(
    [
        assign("A-op1", "A", "M1", 0, 40),
        assign("A-op2", "A", "M1", 40, 80),
        assign("B-op1", "B", "M1", 80, 120),
    ]
)


def at(now, inst=BASE):
    return inst.model_validate({**inst.model_dump(), "now": now})


def test_started_assignments_are_those_that_began_before_now():
    ids = lambda now: [a.op_id for a in started_assignments(at(now), COMMITTED)]
    assert ids(0) == []
    assert ids(40) == ["A-op1"]  # starts exactly at now count as not started yet
    assert ids(41) == ["A-op1", "A-op2"]
    assert ids(500) == ["A-op1", "A-op2", "B-op1"]


def test_started_operations_are_frozen_when_nothing_conflicts():
    plan = plan_reschedule(at(60), COMMITTED)
    assert [a.op_id for a in plan.frozen] == ["A-op1", "A-op2"]  # A-op2 is running (40-80)
    assert plan.interrupted == []


def test_running_operation_hit_by_new_downtime_is_interrupted_not_frozen():
    inst = at(60)
    down, _ = changes.add_downtime(inst, "M1", 70, 200)  # A-op2 runs 40-80, so 70-80 overlaps
    plan = plan_reschedule(down, COMMITTED)
    assert [a.op_id for a in plan.interrupted] == ["A-op2"]
    assert [a.op_id for a in plan.frozen] == ["A-op1"]  # finished work is never undone


def test_downtime_on_another_machine_does_not_interrupt():
    down, _ = changes.add_downtime(at(60), "M2", 70, 200)
    plan = plan_reschedule(down, COMMITTED)
    assert plan.interrupted == []


def test_completed_operation_is_never_interrupted():
    # now=100: A-op1 (0-40) and A-op2 (40-80) are finished; B-op1 (80-120) is running.
    # The downtime is clipped to start at 100, so it can only reach the running operation.
    down, _ = changes.add_downtime(at(100), "M1", 50, 300)
    plan = plan_reschedule(down, COMMITTED)
    assert [a.op_id for a in plan.interrupted] == ["B-op1"]
    assert {a.op_id for a in plan.frozen} == {"A-op1", "A-op2"}


def test_downtime_after_a_running_operation_ends_does_not_interrupt_it():
    # B-op1 runs 80-120; downtime starts at 120 exactly (half-open: no overlap).
    down, _ = changes.add_downtime(at(100), "M1", 120, 300)
    assert plan_reschedule(down, COMMITTED).interrupted == []


def test_interrupted_operation_restarts_after_now_and_the_result_is_valid():
    inst = at(60)
    down, _ = changes.add_downtime(inst, "M1", 70, 200)
    plan = plan_reschedule(down, COMMITTED)
    sched = solve(down, frozen=plan.frozen, config=FAST, hint=COMMITTED)
    ops = sched.by_op()
    assert (ops["A-op1"].machine_id, ops["A-op1"].start) == ("M1", 0)  # frozen history
    assert ops["A-op2"].start >= 60  # restarted from now or later
    assert not (ops["A-op2"].machine_id == "M1" and ops["A-op2"].start < 200 and ops["A-op2"].end > 70)
    assert validate_schedule(down, sched, plan.frozen).ok
