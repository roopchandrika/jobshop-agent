"""Eval scenarios: what a planner asks, and what a correct agent does about it.

Scenarios are plain data (YAML) so they can be reviewed and extended without touching code.
Each one states the *outcome* a correct agent reaches and, where it applies, the exact edits the
draft must contain. The checks compare that with what the agent really did in the store, not with
what it said.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from jobshop.tools.registry import TOOLS
from jobshop.tools.views import Goal

# Tools that change a draft or start a solve; "@edits" in a scenario expands to these.
EDIT_TOOLS = ["create_draft", "discard_draft", "simulate_downtime", "change_priority", "add_rush_order", "reschedule"]
_GROUPS = {"@edits": EDIT_TOOLS}

CATEGORIES = ("simple_downtime", "priority_change", "rush_order", "multiple_disruptions", "read_only",
              "impossible", "ambiguous", "injection", "memory")
Outcome = Literal["proposal", "clarify", "no_action", "infeasible"]

_TIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


class _Data(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _plant_time(value: str) -> str:
    if not _TIME.fullmatch(value):
        raise ValueError(f"'{value}' is not a plant-local 'YYYY-MM-DD HH:MM' time")
    return value


class Downtime(_Data):
    machine: str
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _times(cls, value: str) -> str:
        return _plant_time(value)


class RushOrder(_Data):
    family: str
    due: str
    priority: int = Field(ge=1, le=5)

    @field_validator("due")
    @classmethod
    def _due(cls, value: str) -> str:
        return _plant_time(value)


class Changes(_Data):
    """The edits a correct draft contains, written as the planner would see them after any clipping."""

    downtimes: list[Downtime] = []
    priorities: dict[str, int] = {}
    rush_orders: list[RushOrder] = []

    @property
    def empty(self) -> bool:
        return not (self.downtimes or self.priorities or self.rush_orders)


class Expect(_Data):
    outcome: Outcome
    changes: Changes | None = None
    tools_required: list[str] = []
    # Each inner list: at least one of these tools must be called (several tools can answer a question).
    tools_any: list[list[str]] = []
    tools_forbidden: list[str] = []
    max_calls: dict[str, int] = {}
    # If set, every successful reschedule must have used this goal (the tool's default is fewest_moves).
    reschedule_goal: Goal | None = None

    @field_validator("tools_forbidden")
    @classmethod
    def _expand_groups(cls, names: list[str]) -> list[str]:
        return [t for name in names for t in _GROUPS.get(name, [name])]

    @model_validator(mode="after")
    def _consistent(self) -> Expect:
        has_changes = self.changes is not None and not self.changes.empty
        if self.outcome in ("proposal", "infeasible") and not has_changes:
            raise ValueError(f"outcome '{self.outcome}' needs the expected changes")
        if self.outcome in ("clarify", "no_action") and has_changes:
            raise ValueError(f"outcome '{self.outcome}' means the draft must stay empty; remove 'changes'")
        known = {t.name for t in TOOLS}
        mentioned = [*self.tools_required, *self.tools_forbidden, *self.max_calls, *[t for g in self.tools_any for t in g]]
        unknown = sorted(set(mentioned) - known)
        if unknown:
            raise ValueError(f"unknown tool name(s): {unknown}")
        return self


class JudgeSpec(_Data):
    notes: str = ""  # what a good answer does here, in plain words, for the judge
    extra_criteria: list[Literal["injection_handling"]] = []


class OracleCall(_Data):
    tool: str
    args: dict = {}


class OracleScript(_Data):
    """Used only by the reference agent (see oracle.py); proposal-style scenarios need none of it."""

    calls: list[OracleCall] = []
    say: str = "Done."


class Scenario(_Data):
    id: str
    category: Literal[CATEGORIES]  # type: ignore[valid-type]
    description: str
    request: str
    shop: str = Field("default", pattern=r"^[a-z][a-z0-9_]*$")  # which fixture shop (see shop.SHOPS)
    now: str | None = None  # plant time the scenario starts at; default is the suite's
    # Hostile free text planted in order notes: {order_id: text}. The generator never makes any.
    poison: dict[str, str] = {}
    # Standing preferences the planner has already set (long-term memory), as if typed with /remember before the request.
    preferences: list[str] = []
    expect: Expect
    judge: JudgeSpec = JudgeSpec()
    oracle: OracleScript = OracleScript()

    @field_validator("now")
    @classmethod
    def _now(cls, value: str | None) -> str | None:
        return None if value is None else _plant_time(value)

    @field_validator("preferences")
    @classmethod
    def _preferences(cls, values: list[str]) -> list[str]:
        from jobshop.agent.memory import MAX_CHARS, MAX_PREFERENCES
        if len(values) > MAX_PREFERENCES or any(not v.strip() or len(v) > MAX_CHARS for v in values):
            raise ValueError(f"at most {MAX_PREFERENCES} preferences of 1 to {MAX_CHARS} characters")
        return values

    @property
    def needs_knowledge(self) -> bool:
        """True if a correct agent has to look something up in the plant documents."""
        e = self.expect
        return "search_knowledge" in e.tools_required or any("search_knowledge" in g for g in e.tools_any)


def load_scenarios(directory: Path) -> list[Scenario]:
    """Every ``*.yaml`` under ``directory``; each file is ``scenarios: [...]``."""
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if set(data) != {"scenarios"}:
            raise ValueError(f"{path.name}: top level must be exactly 'scenarios: [...]'")
        for raw in data["scenarios"]:
            try:
                scenarios.append(Scenario.model_validate(raw))
            except ValidationError as e:
                raise ValueError(f"{path.name}, scenario {raw.get('id', '?')!r}: {e}") from None
    ids = [s.id for s in scenarios]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {duplicates}")
    return scenarios
