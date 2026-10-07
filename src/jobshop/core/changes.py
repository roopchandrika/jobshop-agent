"""Planner-requested changes, applied to an Instance to produce a new Instance.

These are pure functions: they never touch a schedule and never solve. A "draft" in the
tools layer is just the committed instance plus a list of these changes.

Every function returns a *new* validated instance (via ``Instance.model_validate``, not
``model_copy``, so all the consistency checks run again) plus human-readable notes about
anything that was adjusted, so the agent can tell the planner.
"""

from __future__ import annotations

from jobshop.core.models import Instance, Operation, Order, TimeWindow


MAX_DUE_AHEAD_MIN = 365 * 24 * 60  # a typo like year 2062 should be an error, not a huge due date


class ChangeError(ValueError):
    """The requested change is invalid (unknown id, bad window, ...). Message is user-facing."""


def add_downtime(
    instance: Instance, machine_id: str, start: int, end: int
) -> tuple[Instance, list[str]]:
    """Add an outage on a machine. Times are minutes since ``instance.t0``.

    History cannot be changed, so an outage that is already over is rejected, and one that
    began in the past is clipped to start at ``now``.
    """
    if machine_id not in {m.id for m in instance.machines}:
        raise ChangeError(f"unknown machine '{machine_id}'")
    if end <= start:
        raise ChangeError("downtime must end after it starts")
    if end <= instance.now:
        raise ChangeError(
            f"downtime ends at {instance.to_datetime(end):%Y-%m-%d %H:%M}, "
            f"which is not after the current time {instance.to_datetime(instance.now):%Y-%m-%d %H:%M}"
        )

    # Judge the effective start (the past is clipped to now below): once the clock is past the end
    # of the plan, no outage can matter, and clipping both ends would leave an empty window.
    horizon = instance.horizon
    if max(start, instance.now) >= horizon:
        raise ChangeError(
            f"downtime would start at {instance.to_datetime(max(start, instance.now)):%Y-%m-%d %H:%M}, "
            f"at or after the end of the plan ({instance.to_datetime(horizon):%Y-%m-%d %H:%M}); "
            "nothing is scheduled then. Check the date."
        )

    notes: list[str] = []
    if start < instance.now:
        notes.append(
            f"start clipped from {instance.to_datetime(start):%H:%M} to the current time "
            f"{instance.to_datetime(instance.now):%H:%M} (the past cannot change)"
        )
        start = instance.now
    if end > horizon:
        notes.append(f"end clipped to the end of the plan, {instance.to_datetime(horizon):%Y-%m-%d %H:%M}")
        end = horizon

    data = instance.model_dump()
    for machine in data["machines"]:
        if machine["id"] == machine_id:
            machine["downtime"].append(TimeWindow(start=start, end=end).model_dump())
    return Instance.model_validate(data), notes


def change_priority(instance: Instance, order_id: str, priority: int) -> Instance:
    if order_id not in {o.id for o in instance.orders}:
        raise ChangeError(f"unknown order '{order_id}'")
    if not 1 <= priority <= 5:
        raise ChangeError("priority must be between 1 (lowest) and 5 (most urgent)")
    data = instance.model_dump()
    for order in data["orders"]:
        if order["id"] == order_id:
            order["priority"] = priority
    return Instance.model_validate(data)


def next_rush_id(instance: Instance) -> str:
    taken = {o.id for o in instance.orders}
    n = 1
    while f"RUSH-{n}" in taken:
        n += 1
    return f"RUSH-{n}"


def add_rush_order(
    instance: Instance, order_id: str, family: str, due: int, priority: int = 5
) -> Instance:
    """Add an order built from the family's routing template (template durations, no jitter)."""
    template = instance.routing_templates.get(family)
    if template is None:
        known = ", ".join(sorted(instance.routing_templates)) or "none"
        raise ChangeError(f"unknown product family '{family}' (known families: {known})")
    if order_id in {o.id for o in instance.orders}:
        raise ChangeError(f"order '{order_id}' already exists")
    if due <= instance.now:
        raise ChangeError("due time must be after the current time")
    if due - instance.now > MAX_DUE_AHEAD_MIN:
        raise ChangeError("due time is more than a year ahead; check the date")
    if not 1 <= priority <= 5:
        raise ChangeError("priority must be between 1 (lowest) and 5 (most urgent)")

    order = Order(
        id=order_id,
        family=family,
        due=due,
        priority=priority,
        operations=[
            Operation(id=f"{order_id}-op{k + 1}", duration=step.duration, required_capability=step.capability)
            for k, step in enumerate(template)
        ],
    )
    data = instance.model_dump()
    data["orders"].append(order.model_dump())
    return Instance.model_validate(data)
