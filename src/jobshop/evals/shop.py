"""The eval shops: fixed synthetic plants and their baseline plans, stored as fixtures.

The baseline is solved once and committed (``evals/shop.json``) instead of re-solved per run. A
time-limited solve gives a different plan on a slower machine, and an eval whose starting point
moves cannot compare two models or two days. Every scenario starts from a fresh copy.

There are two shops. ``default`` has slack, so most disruptions are absorbed with no late orders
(which tests that the agent does not invent problems). ``tight`` has 19 orders on the same four
machines: its live plan is on time, but almost any outage makes an order late, so disruptions have
consequences to explain. (Seed 2 was picked by trying seeds: bigger shops with late orders in the
baseline were too hard to prove, which would make results depend on the speed of the machine.)
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
SHOPS: dict[str, GeneratorSettings] = {
    "default": SHOP_SETTINGS,
    "tight": GeneratorSettings(seed=2, n_machines=4, n_orders=19, ops_min=2, ops_max=3, n_days=1),
}
DEFAULT_NOW = "2026-01-05 10:00"


def shop_path(evals_dir: Path, name: str) -> Path:
    """``shop.json`` for the default shop, ``shop_<name>.json`` for the others."""
    return evals_dir / ("shop.json" if name == "default" else f"shop_{name}.json")


def build_shop(path: Path, solve_seconds: float = 30.0, name: str = "default") -> Schedule:
    """Generate the shop and solve its baseline (slow, once). Overwrites ``path``."""
    settings = SHOPS[name]
    instance = generate_instance(settings)
    baseline = solve(instance, config=SolverConfig(time_limit_s=solve_seconds, num_workers=8, seed=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "generated_by": "python -m jobshop.evals build-shop",
            "settings": asdict(settings),
            "instance": instance.model_dump(mode="json"),
            "schedule": baseline.model_dump(mode="json"),
        }, indent=1, default=str),
        encoding="utf-8",
    )
    return baseline


def load_shop(path: Path) -> tuple[Instance, Schedule]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Instance.model_validate(data["instance"]), Schedule.model_validate(data["schedule"])


def load_shops(evals_dir: Path, names: set[str]) -> dict[str, tuple[Instance, Schedule]]:
    """The fixtures for the named shops; a missing file says how to create it."""
    shops = {}
    for name in sorted(names):
        path = shop_path(evals_dir, name)
        if name not in SHOPS:
            raise ValueError(f"unknown shop '{name}' (known: {', '.join(SHOPS)})")
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; create it with: python -m jobshop.evals build-shop --shop {name}")
        shops[name] = load_shop(path)
    return shops


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
