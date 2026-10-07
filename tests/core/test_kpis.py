import pytest

from jobshop.core.kpis import compute_kpis, diff_schedules
from tests.helpers import assign, instance, machine, order, schedule

# One busy machine, one idle machine.
#   A (due 20, priority 3 -> weight 4): two 10-min ops   B (due 25, priority 5 -> weight 16): 15 min
MACHINES = [machine("M1", windows=[(0, 100)]), machine("M2", windows=[(0, 100)])]


def base_instance(b_priority=5, extra_orders=()):
    return instance(
        MACHINES,
        [
            order("A", 20, [(10, "cut"), (10, "cut")], priority=3),
            order("B", 25, [(15, "cut")], priority=b_priority),
            *extra_orders,
        ],
    )


ALL_ON_M1 = schedule(
    [
        assign("A-op1", "A", "M1", 0, 10),
        assign("A-op2", "A", "M1", 10, 20),
        assign("B-op1", "B", "M1", 20, 35),
    ]
)


def test_tardiness_numbers_are_exact():
    k = compute_kpis(base_instance(), ALL_ON_M1)
    assert k.makespan == 35
    assert k.total_tardiness == 10  # B finishes at 35, due 25
    assert k.weighted_tardiness == 160  # 10 * weight 16
    assert k.late_orders == 1 and k.late_order_ids == ["B"]
    by_id = {o.order_id: o for o in k.orders}
    assert (by_id["A"].completion, by_id["A"].tardiness) == (20, 0)
    assert (by_id["B"].completion, by_id["B"].tardiness) == (35, 10)


def test_utilization_is_busy_over_open_time_in_the_window():
    k = compute_kpis(base_instance(), ALL_ON_M1)
    assert k.machine_utilization == {"M1": pytest.approx(1.0), "M2": pytest.approx(0.0)}
    assert k.mean_utilization == pytest.approx(0.5)


def test_downtime_is_not_counted_as_open_time():
    # M open [0,100) but down [10,20). Window is [0,30): open = 20, busy = 20 -> 100%, not 20/30.
    inst = instance(
        [machine("M", windows=[(0, 100)], downtime=[(10, 20)])],
        [order("A", 100, [(10, "cut")]), order("B", 100, [(10, "cut")])],
    )
    sched = schedule([assign("A-op1", "A", "M", 0, 10), assign("B-op1", "B", "M", 20, 30)])
    assert compute_kpis(inst, sched).machine_utilization["M"] == pytest.approx(1.0)


def test_utilization_window_starts_at_now():
    inst = instance([machine("M", windows=[(0, 100)])], [order("A", 100, [(20, "cut"), (10, "cut")])], now=10)
    sched = schedule([assign("A-op1", "A", "M", 0, 20), assign("A-op2", "A", "M", 20, 30)])
    # Window [10,30): busy = 10 (tail of op1) + 10 = 20, open = 20.
    assert compute_kpis(inst, sched).machine_utilization["M"] == pytest.approx(1.0)


def test_machine_with_no_open_time_in_window_is_left_out_of_the_mean():
    machines = MACHINES + [machine("M3", windows=[(500, 600)])]
    inst = instance(
        machines, [order("A", 20, [(10, "cut"), (10, "cut")]), order("B", 25, [(15, "cut")], priority=5)]
    )
    k = compute_kpis(inst, ALL_ON_M1)
    assert k.machine_utilization["M3"] == 0.0
    assert k.mean_utilization == pytest.approx(0.5)  # mean of M1 and M2 only


def test_incomplete_schedule_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="missing operations"):
        compute_kpis(base_instance(), schedule([assign("A-op1", "A", "M1", 0, 10)]))


def test_no_orders_gives_zero_kpis():
    k = compute_kpis(instance([machine("M")], []), schedule([]))
    assert (k.makespan, k.total_tardiness, k.late_orders, k.mean_utilization) == (0, 0, 0, 0.0)


# --- diff -------------------------------------------------------------------------------


def test_diff_of_identical_schedules_is_empty():
    d = diff_schedules(base_instance(), ALL_ON_M1, base_instance(), ALL_ON_M1)
    assert d.moved_operations == [] and d.order_deltas == []
    assert (d.delta_total_tardiness, d.delta_makespan, d.delta_late_orders) == (0, 0, 0)
    assert d.newly_late_orders == [] and d.added_operations == []


def test_diff_reports_moves_additions_and_kpi_changes():
    rush = order("R", 40, [(10, "cut")], priority=5)
    after_instance = base_instance(extra_orders=[rush])
    after = schedule(
        [
            assign("A-op1", "A", "M1", 0, 10),  # unchanged
            assign("A-op2", "A", "M1", 10, 20),  # unchanged
            assign("B-op1", "B", "M2", 0, 15),  # moved to the idle machine, now on time
            assign("R-op1", "R", "M1", 20, 30),  # new rush order
        ]
    )
    d = diff_schedules(base_instance(), ALL_ON_M1, after_instance, after)

    assert [m.op_id for m in d.moved_operations] == ["B-op1"]
    moved = d.moved_operations[0]
    assert (moved.from_machine, moved.to_machine, moved.from_start, moved.to_start) == ("M1", "M2", 20, 0)
    assert d.machine_changes == 1
    assert d.added_operations == ["R-op1"] and d.removed_operations == []

    assert d.delta_total_tardiness == -10
    assert d.delta_weighted_tardiness == -160
    assert d.delta_late_orders == -1
    assert d.delta_makespan == -5  # 35 -> 30
    assert d.no_longer_late_orders == ["B"] and d.newly_late_orders == []
    assert [(o.order_id, o.change) for o in d.order_deltas] == [("B", -10)]


def test_diff_flags_orders_that_become_late():
    # Push A's second op back so A misses its due date of 20.
    after = schedule(
        [
            assign("A-op1", "A", "M1", 0, 10),
            assign("A-op2", "A", "M1", 15, 25),
            assign("B-op1", "B", "M2", 0, 15),
        ]
    )
    d = diff_schedules(base_instance(), ALL_ON_M1, base_instance(), after)
    assert d.newly_late_orders == ["A"] and d.no_longer_late_orders == ["B"]
    assert d.machine_changes == 1  # only B changed machine; A-op2 just moved in time
    assert len(d.moved_operations) == 2


def test_priority_change_moves_weighted_but_not_total_tardiness():
    d = diff_schedules(base_instance(b_priority=5), ALL_ON_M1, base_instance(b_priority=1), ALL_ON_M1)
    assert d.delta_total_tardiness == 0
    assert d.delta_weighted_tardiness == 10 - 160
