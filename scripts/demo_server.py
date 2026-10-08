"""Try the web UI without an API key: the REAL app and solver with a SCRIPTED stand-in for the model.

    uv run python scripts/demo_server.py          # then open http://127.0.0.1:8765

What is real: the shop, the solver, every plan and KPI, the Gantt charts, the approval flow.
What is scripted: the model. Whatever you type, the next canned turn plays, in this order:

  1. an outage on M1 until 16:00 (a proposal with a late order, to approve or reject)
  2. O-103 becomes top priority (a second proposal)
  3. the assistant asks a clarifying question instead of guessing
  4. then "nothing further to change", forever

Every answer is labelled so it cannot be mistaken for real model output. Restart to play it again.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import uvicorn
from anthropic.types import Message, ToolUseBlock, Usage
from fastapi import FastAPI

from jobshop.agent.loop import SUBMIT, AgentConfig
from jobshop.api.app import create_app
from jobshop.core.solver import SolverConfig
from jobshop.evals.shop import fresh_context, load_shop

REPO = Path(__file__).resolve().parents[1]
TAG = "[Scripted demo: this answer is canned, whatever you typed.] "
Step = list[tuple[str, dict[str, Any]]]  # the tool calls the "model" makes in one reply


def build_steps() -> list[Step]:
    answer = lambda **payload: [(SUBMIT, payload)]  # noqa: E731
    return [
        # 1. an outage that makes an order late
        [("create_draft", {})],
        [("simulate_downtime", {"draft_id": "D1", "machine_id": "M1", "start": "2026-01-05 10:00", "end": "2026-01-05 16:00"})],
        [("reschedule", {"draft_id": "D1"})],
        [("compare_schedules", {"after": "D1"})],
        answer(summary=TAG + "With M1 down until 16:00, work shifts to the other machines. "
                             "Review the comparison on the right, then approve or reject.", draft_id="D1"),
        # 2. a priority change
        [("create_draft", {})],
        [("change_priority", {"draft_id": "D2", "order_id": "O-103", "priority": 5}), ("reschedule", {"draft_id": "D2"})],
        [("compare_schedules", {"after": "D2"})],
        answer(summary=TAG + "O-103 is now top priority in the draft. Compare it with the live plan on the right.", draft_id="D2"),
        # 3. asking instead of guessing
        answer(summary=TAG + "I need one detail before I change anything.",
               clarifying_question="Which machine is going down, and from what time to what time?"),
    ]


class ScriptedModel:
    """Quacks like ``anthropic.Anthropic`` for the one call the agent loop makes."""

    def __init__(self, steps: list[Step]) -> None:
        self._steps = list(steps)
        self._ids = itertools.count(1)
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        n = next(self._ids)
        step = self._steps.pop(0) if self._steps else [(SUBMIT, {"summary": TAG + "Nothing further to change."})]
        blocks = [ToolUseBlock(type="tool_use", id=f"demo_{n}_{k}", name=name, input=args) for k, (name, args) in enumerate(step)]
        return Message(
            id=f"msg_demo_{n}", type="message", role="assistant", model="scripted-demo", content=blocks,
            stop_reason="tool_use", stop_sequence=None, usage=Usage(input_tokens=0, output_tokens=0),
        )


def build_app(solve_seconds: float = 5.0, now: str = "2026-01-05 10:00") -> FastAPI:
    shop = load_shop(REPO / "evals" / "shop.json")
    ctx = fresh_context(shop, SolverConfig(time_limit_s=solve_seconds, num_workers=1, seed=0), now)
    return create_app(ctx, ScriptedModel(build_steps()), AgentConfig(model="scripted-demo"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--solve-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    print(f"SCRIPTED DEMO: the model is canned, the solver and plans are real.\nOpen http://127.0.0.1:{args.port}   (Ctrl+C to stop)", flush=True)
    uvicorn.run(build_app(args.solve_seconds), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
