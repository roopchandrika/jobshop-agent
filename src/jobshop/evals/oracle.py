"""A scripted reference agent: does what a correct agent would do for each scenario, with no API.

It is **not** evidence about any model. It exists to test the eval itself: if the reference agent
cannot pass a scenario, the scenario (or a check) is wrong, and that must be found without paying
for model calls. It also lets ``python -m jobshop.evals run --oracle`` show the full pipeline and
report format without an API key.

Proposal and infeasible scenarios are derived from ``expect.changes``; the others use the
scenario's ``oracle`` script (which read-only calls to make, and what to say).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from anthropic.types import Message, ToolUseBlock, Usage

from jobshop.agent.loop import SUBMIT
from jobshop.evals.scenario import Changes, Scenario

Call = tuple[str, dict[str, Any]]
Step = Callable[[list[dict[str, Any]]], list[Call]]


def _edit_calls(changes: Changes, draft_id: str = "D1") -> list[Call]:
    calls: list[Call] = [
        ("simulate_downtime", {"draft_id": draft_id, "machine_id": d.machine, "start": d.start, "end": d.end})
        for d in changes.downtimes
    ]
    calls += [("change_priority", {"draft_id": draft_id, "order_id": o, "priority": p}) for o, p in changes.priorities.items()]
    calls += [("add_rush_order", {"draft_id": draft_id, "family": r.family, "due": r.due, "priority": r.priority}) for r in changes.rush_orders]
    return calls


def _last_result(messages: list[dict[str, Any]], tool_name: str) -> dict[str, Any]:
    """The newest result of ``tool_name``, found the way the model would: by reading the history."""
    ids = {b["id"]: b["name"] for m in messages if m["role"] == "assistant" and isinstance(m["content"], list)
           for b in m["content"] if b["type"] == "tool_use"}
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in reversed(m["content"]):
                if block.get("type") == "tool_result" and ids.get(block["tool_use_id"]) == tool_name:
                    return json.loads(block["content"])
    raise LookupError(f"no {tool_name} result in the conversation")


def _proposal_summary(messages: list[dict[str, Any]]) -> str:
    result = _last_result(messages, "compare_schedules")
    diff = result["diff"]
    before, after = diff["kpi_before"], diff["kpi_after"]
    return (
        f"With the change, {after['late_orders']} orders are late (live plan: {before['late_orders']}) and total "
        f"tardiness is {after['total_tardiness_min']} min (live plan: {before['total_tardiness_min']}). "
        f"{diff['moved_operation_count']} operations move. Solver status for the draft: {result['solve_after']['status']}. "
        "A human has to approve this before it is live."
    )


def steps_for(scenario: Scenario) -> list[tuple[str, Step | dict[str, Any]]]:
    """('calls', fn) steps make tool calls; ('submit', payload-or-fn) ends the turn."""
    expect = scenario.expect
    reads: list[Call] = [(c.tool, c.args) for c in scenario.oracle.calls]
    if expect.outcome in ("proposal", "infeasible"):
        assert expect.changes is not None
        steps: list[tuple[str, Any]] = [("calls", lambda m: reads)] if reads else []
        steps += [
            ("calls", lambda m: [("create_draft", {})]),
            ("calls", lambda m: _edit_calls(expect.changes)),
            ("calls", lambda m: [("reschedule", {"draft_id": "D1"})]),
        ]
        if expect.outcome == "proposal":
            steps += [
                ("calls", lambda m: [("compare_schedules", {"after": "D1"})]),
                ("submit", lambda m: {"summary": _proposal_summary(m), "draft_id": "D1"}),
            ]
        else:
            steps.append(("submit", {"summary": scenario.oracle.say, "draft_id": "D1"}))
        return steps
    if expect.outcome == "clarify":
        return [("submit", {"summary": "I need one detail before changing anything.", "clarifying_question": scenario.oracle.say})]
    steps = [("calls", lambda m: reads)] if reads else []
    steps.append(("submit", {"summary": scenario.oracle.say}))
    return steps


class OracleClient:
    """Quacks like ``anthropic.Anthropic`` for the one call the loop makes."""

    def __init__(self, scenario: Scenario) -> None:
        self._steps = steps_for(scenario)
        self._n = 0
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        kind, payload = self._steps.pop(0)
        self._n += 1
        if kind == "calls":
            blocks = [
                ToolUseBlock(type="tool_use", id=f"oracle_{self._n}_{k}", name=name, input=args)
                for k, (name, args) in enumerate(payload(kwargs["messages"]))
            ]
        else:
            body = payload(kwargs["messages"]) if callable(payload) else payload
            blocks = [ToolUseBlock(type="tool_use", id=f"oracle_{self._n}_submit", name=SUBMIT, input=body)]
        return Message(
            id=f"msg_oracle_{self._n}", type="message", role="assistant", model="oracle", content=blocks,
            stop_reason="tool_use", stop_sequence=None, usage=Usage(input_tokens=0, output_tokens=0),
        )
