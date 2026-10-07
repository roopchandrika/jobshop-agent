"""The validator is the ground truth for evals, so each rule gets a test that breaks it."""

import pytest

from jobshop.core.models import Assignment, SolveStatus
from jobshop.core.validator import ViolationKind as K
from jobshop.core.validator import validate_schedule
from tests.helpers import assign, instance, machine, order, schedule

#   M1: cut,        open [0,100), down [40,50)
#   M2: cut, drill, open [0,50) and [60,200)
#   A (due 100): cut 20 -> drill 30      B (due 50): cut 30
BASE = instance(
    [
        machine("M1", caps=("cut",), windows=[(0, 100)], downtime=[(40, 50)]),
        machine("M2", caps=("cut", "drill"), windows=[(0, 50), (60, 200)]),
    ],
    [order("A", 100, [(20, "cut"), (30, "drill")]), order("B", 50, [(30, "cut")], priority=5)],
)
VALID = [
    assign("A-op1", "A", "M1", 0, 20),
    assign("A-op2", "A", "M2", 20, 50),  # starts exactly when A-op1 ends; ends at window close
    assign("B-op1", "B", "M1", 50, 80),  # starts exactly when the downtime ends
]


def variant(op_id, **changes):
    """VALID with one assignment altered."""
    return [a.model_copy(update=changes) if a.op_id == op_id else a for a in VALID]


def kinds(assignments, inst=BASE, frozen=()):
    return validate_schedule(inst, schedule(assignments), frozen).kinds


def test_a_correct_schedule_passes_including_touching_edges():
    report = validate_schedule(BASE, schedule(VALID))
    assert report.ok and report.violations == []


def test_operation_may_span_two_back_to_back_windows():
    inst = instance([machine("M", windows=[(0, 50), (50, 100)])], [order("A", 100, [(20, "cut")])])
    assert validate_schedule(inst, schedule([assign("A-op1", "A", "M", 40, 60)])).ok


BROKEN = [
    (K.MISSING_OPERATION, [a for a in VALID if a.op_id != "B-op1"]),
    (K.UNKNOWN_OPERATION, VALID + [assign("ZZZ", "A", "M1", 80, 90)]),
    (K.DUPLICATE_ASSIGNMENT, VALID + [assign("A-op1", "A", "M1", 0, 20)]),
    (K.ORDER_MISMATCH, variant("B-op1", order_id="A")),
    (K.NEGATIVE_START, variant("A-op1", start=-5, end=15)),
    (K.BAD_DURATION, variant("A-op1", end=25)),
    (K.UNKNOWN_MACHINE, variant("A-op1", machine_id="M9")),
    (K.INELIGIBLE_MACHINE, variant("A-op2", machine_id="M1")),  # M1 cannot drill
    (K.OUTSIDE_AVAILABILITY, variant("A-op1", machine_id="M2", start=45, end=65)),  # spans 50-60 gap
    (K.DOWNTIME_OVERLAP, variant("B-op1", start=30, end=60)),  # overlaps downtime 40-50
    (K.MACHINE_OVERLAP, variant("B-op1", start=10, end=40)),  # overlaps A-op1 0-20 on M1
    (K.PRECEDENCE, variant("A-op2", start=10, end=40)),  # starts before A-op1 ends at 20
]


@pytest.mark.parametrize("expected, assignments", BROKEN, ids=[k.value for k, _ in BROKEN])
def test_each_rule_is_caught(expected, assignments):
    assert expected in kinds(assignments)


def test_overlap_check_is_precise():
    # [10,40) clashes with A-op1 but ends exactly where downtime [40,50) begins.
    found = kinds(variant("B-op1", start=10, end=40))
    assert K.MACHINE_OVERLAP in found
    assert K.DOWNTIME_OVERLAP not in found


def test_operations_before_now_are_flagged_unless_frozen():
    # now=30: A-op1 (starts 0) and A-op2 (starts 20) began in the past; B-op1 (starts 50) did not.
    inst = BASE.model_validate({**BASE.model_dump(), "now": 30})

    def flagged(frozen):
        report = validate_schedule(inst, schedule(VALID), frozen)
        return {v.op_id for v in report.violations if v.kind == K.BEFORE_NOW}

    assert flagged([]) == {"A-op1", "A-op2"}
    assert flagged([VALID[0]]) == {"A-op2"}  # freezing A-op1 excuses only A-op1
    assert flagged([VALID[0], VALID[1]]) == set()


def test_moving_a_frozen_operation_is_caught():
    frozen = [assign("A-op1", "A", "M1", 0, 20)]
    assert K.FROZEN_MOVED in kinds(variant("A-op1", machine_id="M2"), frozen=frozen)
    assert K.FROZEN_MOVED in kinds(variant("A-op1", start=5, end=25), frozen=frozen)
    assert K.FROZEN_MOVED not in kinds(VALID, frozen=frozen)


def test_validator_does_not_trust_the_solver_status():
    broken = schedule(variant("B-op1", start=10, end=40), status=SolveStatus.OPTIMAL)
    assert broken.solve_info.status == SolveStatus.OPTIMAL
    assert not validate_schedule(BASE, broken).ok


def test_all_violations_are_reported_not_just_the_first():
    messy = [
        assign("A-op1", "A", "M9", 0, 20),  # unknown machine
        assign("A-op2", "A", "M1", 20, 50),  # M1 cannot drill
    ]  # and B-op1 is missing
    assert {K.UNKNOWN_MACHINE, K.INELIGIBLE_MACHINE, K.MISSING_OPERATION} <= kinds(messy)
