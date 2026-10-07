"""Synthetic job-shop data. Entirely invented; not modelled on any real plant.

The same seed and settings always produce the same instance, so tests and evals can
refer to "seed 0" and mean the same problem every time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from jobshop.core.models import (
    Instance,
    Machine,
    Operation,
    Order,
    RoutingStep,
    TimeWindow,
)

CAPABILITIES = [
    "turning", "milling", "drilling", "grinding",
    "welding", "assembly", "painting", "inspection",
]
FAMILIES = ["bracket", "housing", "shaft", "gear", "plate", "valve"]
# Harmless notes. Phase 4 injects hostile text separately; the generator never does.
BENIGN_NOTES = [
    "Customer pickup scheduled for Friday.",
    "Packaging needs extra care.",
    "Repeat order, no changes from the last batch.",
    "Confirm tolerances with quality before shipping.",
]
MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class GeneratorSettings:
    seed: int = 0
    n_machines: int = 6
    n_orders: int = 20
    ops_min: int = 3
    ops_max: int = 5
    n_days: int = 3
    shift_minutes: int = 960  # one 16h working window per day, e.g. 06:00-22:00
    n_families: int = 4
    t0: datetime = datetime(2026, 1, 5, 6, 0)  # "today 06:00", plant-local

    def __post_init__(self) -> None:
        if self.n_machines < 1 or self.n_orders < 1 or self.n_days < 1:
            raise ValueError("need at least one machine, one order and one day")
        if not 1 <= self.ops_min <= self.ops_max:
            raise ValueError("need 1 <= ops_min <= ops_max")
        if not 1 <= self.shift_minutes <= MINUTES_PER_DAY:
            raise ValueError("shift_minutes must be between 1 and 1440")
        if not 1 <= self.n_families <= len(FAMILIES):
            raise ValueError(f"n_families must be between 1 and {len(FAMILIES)}")


def generate_instance(settings: GeneratorSettings = GeneratorSettings()) -> Instance:
    rng = random.Random(settings.seed)

    n_caps = min(settings.n_machines, len(CAPABILITIES))
    pool = CAPABILITIES[:n_caps]
    width = min(3, n_caps)  # each machine offers 3 capabilities, so each is offered by 3 machines

    machines = [
        Machine(
            id=f"M{i + 1}",
            type=f"{pool[i % n_caps]} cell",
            capabilities=[pool[(i + j) % n_caps] for j in range(width)],
            availability=[
                TimeWindow(
                    start=day * MINUTES_PER_DAY,
                    end=day * MINUTES_PER_DAY + settings.shift_minutes,
                )
                for day in range(settings.n_days)
            ],
        )
        for i in range(settings.n_machines)
    ]

    templates = {
        family: _make_template(rng, pool, settings)
        for family in FAMILIES[: settings.n_families]
    }
    family_names = list(templates)

    orders = []
    for i in range(settings.n_orders):
        order_id = f"O-{101 + i}"
        family = rng.choice(family_names)
        operations = [
            Operation(
                id=f"{order_id}-op{k + 1}",
                duration=_round5(step.duration * rng.uniform(0.8, 1.2)),
                required_capability=step.capability,
            )
            for k, step in enumerate(templates[family])
        ]
        total = sum(op.duration for op in operations)
        # Due dates are tight enough that a loaded shop runs some orders late, so the
        # priority weights actually have something to trade off.
        due = _round5(total * rng.uniform(1.2, 1.9)) + rng.randrange(0, 450, 15)
        orders.append(
            Order(
                id=order_id,
                family=family,
                due=due,
                priority=rng.choices([1, 2, 3, 4, 5], weights=[15, 30, 30, 15, 10])[0],
                operations=operations,
                notes=rng.choice(BENIGN_NOTES) if rng.random() < 0.25 else "",
            )
        )

    return Instance(
        t0=settings.t0, now=0, machines=machines, orders=orders, routing_templates=templates
    )


def _make_template(
    rng: random.Random, pool: list[str], settings: GeneratorSettings
) -> list[RoutingStep]:
    steps: list[RoutingStep] = []
    previous = None
    for _ in range(rng.randint(settings.ops_min, settings.ops_max)):
        choices = [c for c in pool if c != previous] or pool  # avoid the same step twice in a row
        capability = rng.choice(choices)
        steps.append(RoutingStep(capability=capability, duration=rng.randrange(20, 95, 5)))
        previous = capability
    return steps


def _round5(value: float) -> int:
    return max(5, int(round(value / 5)) * 5)
