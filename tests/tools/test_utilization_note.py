"""Utilization falls when the last finish moves later, even for the same work. The note must say so, and be true."""

from jobshop.api.app import STATIC
from jobshop.core.kpis import compute_kpis
from jobshop.tools.functions import UTILIZATION_NOTE
from tests.helpers import assign, instance, machine, order, schedule


def test_the_note_is_true_the_same_work_finishing_later_gives_lower_utilization():
    inst = instance([machine("M1", windows=((0, 1000),))], [order("A", 1000, [(100, "cut")])])
    early = compute_kpis(inst, schedule([assign("A-op1", "A", "M1", 0, 100)]))
    later = compute_kpis(inst, schedule([assign("A-op1", "A", "M1", 50, 150)]))   # identical work, ends 50 min later
    assert early.mean_utilization == 1.0
    assert later.mean_utilization == 100 / 150 < early.mean_utilization
    assert "even if the same work gets done" in UTILIZATION_NOTE


def test_schedule_and_comparison_results_carry_the_note(registry):
    d = registry.call("create_draft", {})["draft_id"]
    registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": 5})
    registry.call("reschedule", {"draft_id": d})
    for result in (registry.call("get_schedule", {}), registry.call("compare_schedules", {"after": d})):
        note = result["utilization_note"]
        assert note == UTILIZATION_NOTE and "does not mean idle machines or spare capacity" in note


def test_other_results_do_not_carry_it(registry):
    assert "utilization_note" not in registry.call("list_orders", {})
    assert "utilization_note" not in registry.call("get_order", {"order_id": "O-101"})


def test_the_web_page_says_what_the_utilization_tile_measures():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "now to last finish; falls if the finish moves later" in script
    assert ".tile .hint" in (STATIC / "style.css").read_text(encoding="utf-8")
