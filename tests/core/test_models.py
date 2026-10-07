from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from jobshop.core.models import (
    Instance,
    Machine,
    Operation,
    Order,
    RoutingStep,
    TimeWindow,
)
from tests.helpers import T0, instance, machine, order


def test_window_must_end_after_it_starts():
    with pytest.raises(ValidationError):
        TimeWindow(start=10, end=10)
    with pytest.raises(ValidationError):
        TimeWindow(start=10, end=5)


def test_window_cannot_start_before_t0():
    with pytest.raises(ValidationError):
        TimeWindow(start=-1, end=5)


@pytest.mark.parametrize("priority", [0, 6])
def test_priority_must_be_1_to_5(priority):
    with pytest.raises(ValidationError):
        order("A", 10, [(5, "cut")], priority=priority)


def test_operation_duration_must_be_positive():
    with pytest.raises(ValidationError):
        Operation(id="x", duration=0, required_capability="cut")


def test_unknown_fields_are_rejected_so_typos_fail_loudly():
    with pytest.raises(ValidationError):
        Machine(id="M1", type="t", capabilities=["cut"], availability=[], colour="red")


def test_models_are_frozen():
    m = machine("M1")
    with pytest.raises(ValidationError):
        m.id = "other"


def test_weight_follows_priority():
    assert order("A", 10, [(5, "cut")], priority=5).weight == 16
    assert order("A", 10, [(5, "cut")], priority=1).weight == 1


def test_duplicate_machine_order_and_operation_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate machine"):
        instance([machine("M1"), machine("M1")], [])
    with pytest.raises(ValueError, match="duplicate order"):
        instance([machine("M1")], [order("A", 10, [(5, "cut")]), order("A", 10, [(5, "cut")])])
    clash = Order(
        id="B", due=10, priority=3,
        operations=[Operation(id="A-op1", duration=5, required_capability="cut")],
    )
    with pytest.raises(ValueError, match="duplicate operation"):
        instance([machine("M1")], [order("A", 10, [(5, "cut")]), clash])


def test_every_operation_needs_a_capable_machine():
    with pytest.raises(ValueError, match="no machine offers it"):
        instance([machine("M1", caps=("cut",))], [order("A", 10, [(5, "weld")])])


def test_routing_template_needs_a_capable_machine():
    with pytest.raises(ValueError, match="routing template"):
        instance([machine("M1")], [], templates={"f": [RoutingStep(capability="weld", duration=5)]})


def test_t0_must_be_naive():
    with pytest.raises(ValueError, match="naive"):
        Instance(t0=datetime(2026, 1, 5, tzinfo=timezone.utc), machines=[], orders=[])


def test_free_windows_subtract_downtime_and_merge():
    m = machine("M1", windows=[(0, 100)], downtime=[(20, 30), (25, 40)])
    assert m.free_windows() == [(0, 20), (40, 100)]


def test_horizon_is_the_last_open_minute():
    inst = instance([machine("M1", windows=[(0, 50), (100, 300)]), machine("M2", windows=[(0, 80)])], [])
    assert inst.horizon == 300
    assert instance([], []).horizon == 0


def test_eligible_machines_come_from_capabilities():
    m1, m2 = machine("M1", caps=("cut",)), machine("M2", caps=("cut", "drill"))
    o = order("A", 10, [(5, "drill")])
    inst = instance([m1, m2], [o])
    assert [m.id for m in inst.eligible_machines(o.operations[0])] == ["M2"]


def test_minutes_and_datetimes_round_trip():
    inst = instance([machine("M1")], [])
    assert inst.to_datetime(90) == T0 + timedelta(minutes=90)
    assert inst.to_minutes(T0 + timedelta(minutes=90)) == 90
    with pytest.raises(ValueError):
        inst.to_minutes(datetime(2026, 1, 5, tzinfo=timezone.utc))


def test_instance_survives_a_json_round_trip():
    inst = instance([machine("M1", downtime=[(5, 10)])], [order("A", 10, [(5, "cut")], notes="hi")])
    assert Instance.model_validate_json(inst.model_dump_json()) == inst
