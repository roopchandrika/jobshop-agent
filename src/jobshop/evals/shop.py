"""The eval shop: one fixed synthetic plant and its baseline plan, stored as a fixture.

The baseline is solved once and committed (``evals/shop.json``) instead of re-solved per run. A
time-limited solve gives a different plan on a slower machine, and an eval whose starting point
moves cannot compare two models or two days. Every scenario starts from a fresh copy.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.models import Instance, Schedule
from jobshop.core.solver import SolverConfig, solve
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.functions import ToolContext
from jobshop.tools.store import Store

# One day, 12 orders on 4 machines: enough contention that a disruption can make orders late,
# small enough that re-solves prove optimal in a fraction of a second.
SHOP_SETTINGS = GeneratorSettings(seed=0, n_machines=4, n_orders=12, ops_min=2, ops_max=3, n_days=1)
DEFAULT_NOW = "2026-01-05 10:00"


def build_shop(path: Path, solve_seconds: float = 30.0) -> Schedule:
    """Generate the shop and solve its baseline (slow, once). Overwrites ``path``."""
    instance = generate_instance(SHOP_SETTINGS)
    baseline = solve(instance, config=SolverConfig(time_limit_s=solve_seconds, num_workers=8, seed=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "generated_by": "python -m jobshop.evals build-shop",
            "settings": asdict(SHOP_SETTINGS),
            "instance": instance.model_dump(mode="json"),
            "schedule": baseline.model_dump(mode="json"),
        }, indent=1, default=str),
        encoding="utf-8",
    )
    return baseline


def load_shop(path: Path) -> tuple[Instance, Schedule]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Instance.model_validate(data["instance"]), Schedule.model_validate(data["schedule"])


def fresh_context(
    shop: tuple[Instance, Schedule],
    solver_config: SolverConfig,
    now: str | None = None,
    poison: dict[str, str] | None = None,
) -> ToolContext:
    """An in-memory store at the fixture's baseline, with the clock set and any hostile notes planted."""
    instance, baseline = shop
    if poison:
        data = instance.model_dump()
        orders = {o["id"]: o for o in data["orders"]}
        for order_id, text in poison.items():
            orders[order_id]["notes"] = text  # KeyError on a typo is right: the scenario is wrong
        instance = Instance.model_validate(data)
    store = Store(instance, baseline)
    store.set_clock(instance.to_minutes(datetime.strptime(now or DEFAULT_NOW, "%Y-%m-%d %H:%M")))
    return ToolContext(store=store, authority=ApprovalAuthority(), solver_config=solver_config)
