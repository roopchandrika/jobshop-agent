import pytest
import yaml
from pydantic import ValidationError

from jobshop.evals.scenario import CATEGORIES, EDIT_TOOLS, Expect, Scenario, load_scenarios
from tests.evals.conftest import EVALS


def test_the_suite_has_25_to_30_scenarios_covering_every_category(scenarios):
    assert 25 <= len(scenarios) <= 30
    by_category = {c: [s for s in scenarios.values() if s.category == c] for c in CATEGORIES}
    assert all(len(v) >= 3 for v in by_category.values()), {c: len(v) for c, v in by_category.items()}


def test_scenarios_use_the_plant_the_fixture_describes(scenarios, shop):
    instance, _ = shop
    machines, orders = {m.id for m in instance.machines}, {o.id for o in instance.orders}
    for s in scenarios.values():
        for d in (s.expect.changes.downtimes if s.expect.changes else []):
            assert d.machine in machines, s.id
        for order_id in [*(s.expect.changes.priorities if s.expect.changes else {}), *s.poison]:
            assert order_id in orders, s.id
        for r in (s.expect.changes.rush_orders if s.expect.changes else []):
            assert r.family in instance.routing_templates, s.id


def test_every_injection_scenario_plants_hostile_text_and_is_judged_for_it(scenarios):
    injections = [s for s in scenarios.values() if s.category == "injection"]
    assert injections
    for s in injections:
        assert s.poison and "injection_handling" in s.judge.extra_criteria, s.id


def test_non_proposal_scenarios_forbid_changing_anything_unless_they_must_solve(scenarios):
    for s in scenarios.values():
        if s.expect.outcome in ("clarify",) or s.category == "read_only":
            assert set(EDIT_TOOLS) <= set(s.expect.tools_forbidden), s.id


def test_the_groups_expand_so_scenarios_can_say_edits():
    expect = Expect(outcome="no_action", tools_forbidden=["@edits"])
    assert expect.tools_forbidden == EDIT_TOOLS


@pytest.mark.parametrize("bad, message", [
    ({"outcome": "proposal"}, "needs the expected changes"),
    ({"outcome": "infeasible", "changes": {}}, "needs the expected changes"),
    ({"outcome": "clarify", "changes": {"priorities": {"O-101": 5}}}, "must stay empty"),
    ({"outcome": "no_action", "tools_forbidden": ["fly_to_moon"]}, "unknown tool"),
    ({"outcome": "no_action", "tools_any": [["list_orders", "nope"]]}, "unknown tool"),
    ({"outcome": "maybe"}, "outcome"),
])
def test_an_inconsistent_expectation_is_rejected(bad, message):
    with pytest.raises(ValidationError, match=message):
        Expect.model_validate(bad)


def test_times_must_be_plant_local_stamps():
    base = {"id": "x", "category": "simple_downtime", "description": "d", "request": "r",
            "expect": {"outcome": "proposal", "changes": {"downtimes": [{"machine": "M1", "start": "11:00", "end": "2026-01-05 12:00"}]}}}
    with pytest.raises(ValidationError, match="YYYY-MM-DD HH:MM"):
        Scenario.model_validate(base)


def test_unknown_fields_are_rejected_so_typos_do_not_silently_weaken_a_scenario():
    with pytest.raises(ValidationError):
        Expect.model_validate({"outcome": "no_action", "tools_forbiden": ["reschedule"]})


def test_loading_rejects_duplicate_ids_and_bad_files(tmp_path):
    one = {"id": "a", "category": "read_only", "description": "d", "request": "r", "expect": {"outcome": "no_action"}}
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"scenarios": [one]}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"scenarios": [one]}))
    with pytest.raises(ValueError, match="duplicate scenario ids"):
        load_scenarios(tmp_path)

    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"scenario": [one]}))
    with pytest.raises(ValueError, match="top level"):
        load_scenarios(tmp_path)

    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"scenarios": [{**one, "id": "b", "expect": {"outcome": "proposal"}}]}))
    with pytest.raises(ValueError, match="b.yaml, scenario 'b'"):
        load_scenarios(tmp_path)


def test_the_real_suite_loads_from_disk():
    assert len(load_scenarios(EVALS / "scenarios")) >= 25
