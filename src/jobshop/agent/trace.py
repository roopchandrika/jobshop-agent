"""JSONL step logging: one JSON object per line, one file per session.

Every record has ``ts``, ``session`` and ``event``. The loop emits ``turn_start``, ``llm_call``
(tokens and latency), ``tool_call`` (arguments, full result, latency) and ``turn_end``. The
token, latency and cost fields are present from the start so Phase 6 extends this format
instead of replacing it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
