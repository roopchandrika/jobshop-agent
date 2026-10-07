"""JSONL step logging: one JSON object per line, one file per session.

Every record has ``ts``, ``session`` and ``event``. Events, and the fields that matter:

* ``turn_start``  trace_version, model, user_text
* ``llm_call``    step, model, response_id, stop_reason, input_tokens, output_tokens,
                  cache_read_tokens, cache_write_tokens, latency_ms, step_cost_usd,
                  total_cost_usd (running total for the turn), text, tool_calls
* ``tool_call``   step, tool, arguments, is_error, result, latency_ms
* ``turn_end``    status, steps, input_tokens, output_tokens, cost_usd, llm_ms, tool_ms, wall_ms, text
* ``api_error``, ``tool_exception``, ``dropped_block`` for the unhappy paths

Costs are ``null`` when no prices were supplied: the system never guesses a price. Read a trace
with ``python -m jobshop.agent.trace_report FILE``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_VERSION = 2  # 1 was Phase 2 (cost_usd on llm_call was a running total); 2 adds step_cost_usd and timing totals


class Tracer:
    def __init__(
        self,
        path: Path | None = None,
        session_id: str | None = None,
        echo: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.path = path
        self._echo = echo
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session": self.session_id,
            "event": name,
            **fields,
        }
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        if self._echo is not None:
            self._echo(record)
