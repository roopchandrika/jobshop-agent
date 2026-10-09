"""One planner's conversation with the agent: history, system prompt, and system notices.

Shared by the chat CLI and the web API so they cannot drift apart. It deliberately knows
nothing about how answers are shown or how approval is asked for: those belong to the
human-facing layer of each front end.
"""

from __future__ import annotations

from typing import Any

from jobshop.agent.history import clear_old_results, drop_oldest_turns, estimate_tokens
from jobshop.agent.loop import AgentConfig, TurnResult, run_turn
from jobshop.agent.memory import PreferenceStore
from jobshop.agent.prompts import build_system_prompt
from jobshop.agent.trace import Tracer
from jobshop.tools.functions import ToolContext
from jobshop.tools.registry import ToolRegistry


class Conversation:
    def __init__(
        self, client: Any, ctx: ToolContext, config: AgentConfig, tracer: Tracer | None = None,
        preferences: PreferenceStore | None = None,
    ) -> None:
        self.client, self.ctx, self.config = client, ctx, config
        # Long-term memory. Only the planner writes to it (a command or a button); the model only reads it.
        self.preferences = preferences if preferences is not None else PreferenceStore()
        self.registry = ToolRegistry(ctx)
        self.tracer = tracer or Tracer()
        self.messages: list[dict[str, Any]] = []
        self._notices: list[str] = []

    def notify(self, text: str) -> None:
        """Record a fact from this program (not from the planner) to tell the model on its next turn,
        e.g. that the planner approved or declined a draft, or that the clock moved."""
        self._notices.append(text)

    def _tidy_history(self) -> None:
        """Keep the conversation inside its budget (short-term memory): clear old tool results, then drop old turns."""
        limit = self.config.history_token_limit
        before = estimate_tokens(self.messages)
        if limit is None or before <= limit:
            return
        keep = self.config.history_keep_turns
        tidied, cleared = clear_old_results(self.messages, keep)
        tidied, dropped = drop_oldest_turns(tidied, keep, limit)
        if not (cleared or dropped):
            return   # nothing older than the protected turns to remove
        self.messages[:] = tidied
        self.tracer.event("history_compacted", before_tokens=before, after_tokens=estimate_tokens(self.messages),
                          results_cleared=cleared, turns_dropped=dropped)
        if dropped:
            self.notify(f"The oldest {dropped} turn(s) of this conversation were removed to save space; "
                        "ask the planner if you need something from them.")

    def say(self, text: str) -> TurnResult:
        """Send the planner's words, prefixed by any pending notices, and run one turn."""
        self._tidy_history()
        if self._notices:
            note = " ".join(self._notices)
            text = f"[Notice from the scheduling system, not from the planner: {note}]\n\n{text}"
            self._notices.clear()
        prompt = build_system_prompt(self.ctx, self.preferences.texts())
        return run_turn(self.client, self.registry, prompt, self.messages, text, self.config, self.tracer)
