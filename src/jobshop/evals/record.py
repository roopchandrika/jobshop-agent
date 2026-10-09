"""What happened in one scenario run, in the shape the checks and the judge read."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobshop.agent.loop import SUBMIT, TurnResult
from jobshop.evals.scenario import Scenario
from jobshop.tools.functions import ToolContext


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any  # parsed JSON when possible, else the raw text
    is_error: bool


def tool_calls(messages: list[dict[str, Any]]) -> list[ToolCall]:
    """Every tool the model called, in order, paired with what it got back (``submit_response`` excluded)."""
    answers: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    answers[block["tool_use_id"]] = block

    calls: list[ToolCall] = []
    for m in messages:
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        for block in m["content"]:
            if block["type"] != "tool_use" or block["name"] == SUBMIT:
                continue
            answer = answers.get(block["id"], {})
            raw = answer.get("content", "")
            try:
                result: Any = json.loads(raw)
            except (TypeError, ValueError):
                result = raw
            calls.append(ToolCall(block["name"], block["input"], result, bool(answer.get("is_error"))))
    return calls


@dataclass
class Run:
    scenario: Scenario
    attempt: int
    ctx: ToolContext
    messages: list[dict[str, Any]]
    turn: TurnResult
    version_at_start: int
    wall_s: float
    trace_path: Path | None = None
    model: str | None = None
    pattern: str = "react"
    calls: list[ToolCall] = field(init=False)

    def __post_init__(self) -> None:
        self.calls = tool_calls(self.messages)


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None: does not apply to this scenario (or could not be evaluated)
    details: list[str] = field(default_factory=list)
