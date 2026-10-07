import pytest

from jobshop.core import changes
from jobshop.core.changes import ChangeError
from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.models import RoutingStep
from tests.helpers import instance, machine, order


@pytest.fixture
def inst():
    return instance(
        [machine("M1"), machine("M2", caps=("cut", "drill"))],
        [order("A", 100, [(10, "cut")], priority=2), order("B", 200, [(20, "cut"), (5, "drill")])],
        templates={"bracket": [RoutingStep(capability="cut", duration=15), RoutingStep(capability="drill", duration=25)]},
    )


# --- add_downtime ---------------------------------------------------------------------------


def test_downtime_is_added_to_the_right_machine_without_mutating_the_original(inst):
    new, notes = changes.add_downtime(inst, "M1", 50, 80)
    assert [(w.start, w.end) for w in new.machine("M1").downtime] == [(50, 80)]
    assert new.machine("M2").downtime == []
    assert inst.machine("M1").downtime == []  # original untouched
    assert notes == []


def test_downtime_that_started_in_the_past_is_clipped_to_now(inst):
    at_100 = inst.model_validate({**inst.model_dump(), "now": 100})
    new, notes = changes.add_downtime(at_100, "M1", 60, 180)
    assert [(w.start, w.end) for w in new.machine("M1").downtime] == [(100, 180)]
    assert len(notes) == 1 and "clipped" in notes[0]


def test_downtime_entirely_in_the_past_is_rejected(inst):
    at_100 = inst.model_validate({**inst.model_dump(), "now": 100})
    with pytest.raises(ChangeError, match="not after the current time"):
        changes.add_downtime(at_100, "M1", 20, 100)


@pytest.mark.parametrize("machine_id, start, end, message", [
    ("M9", 10, 20, "unknown machine"),
    ("M1", 20, 20, "must end after it starts"),
    ("M1", 30, 20, "must end after it starts"),
])
def test_invalid_downtime_is_rejected(inst, machine_id, start, end, message):
    with pytest.raises(ChangeError, match=message):
        changes.add_downtime(inst, machine_id, start, end)


# --- change_priority ------------------------------------------------------------------------


def test_priority_is_changed_on_one_order_only(inst):
    new = changes.change_priority(inst, "A", 5)
    assert new.order("A").priority == 5 and new.order("B").priority == 3
    assert inst.order("A").priority == 2


@pytest.mark.parametrize("order_id, priority, message", [
    ("Z", 5, "unknown order"),
    ("A", 0, "between 1"),
    ("A", 6, "between 1"),
])
def test_invalid_priority_changes_are_rejected(inst, order_id, priority, message):
    with pytest.raises(ChangeError, match=message):
        changes.change_priority(inst, order_id, priority)


# --- add_rush_order -------------------------------------------------------------------------


def test_rush_order_is_built_from_the_family_template(inst):
    new = changes.add_rush_order(inst, "RUSH-1", "bracket", due=300, priority=5)
    rush = new.order("RUSH-1")
    assert [(op.duration, op.required_capability) for op in rush.operations] == [(15, "cut"), (25, "drill")]
    assert [op.id for op in rush.operations] == ["RUSH-1-op1", "RUSH-1-op2"]
    assert (rush.due, rush.priority, rush.family, rush.notes) == (300, 5, "bracket", "")
    assert len(inst.orders) == 2  # original untouched


def test_next_rush_id_skips_taken_ids(inst):
    assert changes.next_rush_id(inst) == "RUSH-1"
    with_rush = changes.add_rush_order(inst, "RUSH-1", "bracket", due=300)
    assert changes.next_rush_id(with_rush) == "RUSH-2"


@pytest.mark.parametrize("kwargs, message", [
    (dict(order_id="RUSH-1", family="nope", due=300), "unknown product family 'nope' \\(known families: bracket\\)"),
    (dict(order_id="A", family="bracket", due=300), "already exists"),
    (dict(order_id="RUSH-1", family="bracket", due=0), "after the current time"),
    (dict(order_id="RUSH-1", family="bracket", due=300, priority=9), "between 1"),
])
def test_invalid_rush_orders_are_rejected(inst, kwargs, message):
    with pytest.raises(ChangeError, match=message):
        changes.add_rush_order(inst, **kwargs)


def test_changes_work_on_a_generated_instance():
    inst = generate_instance(GeneratorSettings(seed=0))
    family = next(iter(inst.routing_templates))
    new = changes.add_rush_order(inst, changes.next_rush_id(inst), family, due=900)
    new, _ = changes.add_downtime(new, "M4", 480, 660)
    new = changes.change_priority(new, "O-112", 5)
    assert len(new.orders) == len(inst.orders) + 1
    assert new.order("O-112").priority == 5
