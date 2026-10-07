import pytest

from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.models import Instance


def test_same_seed_gives_the_same_instance():
    assert generate_instance(GeneratorSettings(seed=7)) == generate_instance(GeneratorSettings(seed=7))


def test_different_seeds_give_different_instances():
    assert generate_instance(GeneratorSettings(seed=1)) != generate_instance(GeneratorSettings(seed=2))


def test_size_settings_are_honoured():
    s = GeneratorSettings(seed=3, n_machines=5, n_orders=9, ops_min=2, ops_max=4, n_days=2)
    inst = generate_instance(s)
    assert len(inst.machines) == 5
    assert len(inst.orders) == 9
    assert all(2 <= len(o.operations) <= 4 for o in inst.orders)
    assert all(len(m.availability) == 2 for m in inst.machines)


def test_default_instance_matches_the_spec_example_names():
    inst = generate_instance()
    assert len(inst.machines) == 6 and len(inst.orders) == 20
    assert inst.machine("M4") and inst.order("O-112")  # "machine M4", "order O-112"


def test_shifts_repeat_daily():
    inst = generate_instance(GeneratorSettings(n_days=3, shift_minutes=960))
    windows = [(w.start, w.end) for w in inst.machines[0].availability]
    assert windows == [(0, 960), (1440, 2400), (2880, 3840)]


def test_every_operation_has_a_choice_of_machines():
    inst = generate_instance()
    for order in inst.orders:
        for op in order.operations:
            assert len(inst.eligible_machines(op)) >= 2


def test_orders_follow_their_family_routing_template():
    inst = generate_instance(GeneratorSettings(seed=4))
    for order in inst.orders:
        template = inst.routing_templates[order.family]
        assert [op.required_capability for op in order.operations] == [s.capability for s in template]


def test_generated_notes_are_harmless():
    inst = generate_instance(GeneratorSettings(seed=5, n_orders=40))
    assert any(o.notes for o in inst.orders)
    assert not any("ignore" in o.notes.lower() or "commit" in o.notes.lower() for o in inst.orders)


def test_generated_instance_survives_a_json_round_trip():
    inst = generate_instance()
    assert Instance.model_validate_json(inst.model_dump_json()) == inst


@pytest.mark.parametrize(
    "bad",
    [
        {"n_machines": 0},
        {"n_orders": 0},
        {"n_days": 0},
        {"ops_min": 0},
        {"ops_min": 5, "ops_max": 3},
        {"shift_minutes": 0},
        {"shift_minutes": 1500},
        {"n_families": 0},
        {"n_families": 99},
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        GeneratorSettings(**bad)
