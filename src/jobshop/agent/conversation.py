"""One planner's conversation with the agent: history, system prompt, and system notices.

Shared by the chat CLI and the web API so they cannot drift apart. It deliberately knows
nothing about how answers are shown or how approval is asked for: those belong to the
human-facing layer of each front end.
"""

from __future__ import annotations

from typing import Any

from jobshop.agent.loop import AgentConfig, TurnResult, run_turn
from jobshop.agent.prompts import build_system_prompt
from jobshop.agent.trace import Tracer
from jobshop.tools.functions import ToolContext
from jobshop.tools.registry import ToolRegistry


class Conversation:
    def __init__(self, client: Any, ctx: ToolContext, config: AgentConfig, tracer: Tracer | None = None) -> None:
        self.client, self.ctx, self.config = client, ctx, config
        self.registry = ToolRegistry(ctx)
        self.tracer = tracer or Tracer()
        self.messages: list[dict[str, Any]] = []
        self._notices: list[str] = []

    def notify(self, text: str) -> None:
        """Record a fact from this program (not from the planner) to tell the model on its next turn,
        e.g. that the planner approved or declined a draft, or that the clock moved."""
        self._notices.append(text)

    def say(self, text: str) -> TurnResult:
        """Send the planner's words, prefixed by any pending notices, and run one turn."""
        if self._notices:
            note = " ".join(self._notices)
            text = f"[Notice from the scheduling system, not from the planner: {note}]\n\n{text}"
            self._notices.clear()
        return run_turn(self.client, self.registry, build_system_prompt(self.ctx), self.messages, text, self.config, self.tracer)
