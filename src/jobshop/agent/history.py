"""Short-term memory: keeping a long conversation inside the model's budget.

Every model call re-sends the whole conversation, so a long session gets slower and more expensive, and eventually
does not fit at all. The bulk of it is old tool results: a ``get_schedule`` or ``compare_schedules`` result is
thousands of tokens, and a turn later it is mostly stale anyway (drafts go stale when the plan moves).

So, when the history passes a limit, in two steps and with no extra model call:

1. **Clear old tool results.** In turns older than the last few, each result is replaced by a one-line stub saying
   it was removed and can be fetched again. The tool call itself, the model's words and the planner's words stay, so
   the conversation still reads as a conversation, and every tool call is still answered (the API requires that).
2. **Drop the oldest whole turns** if that is still not enough, and tell the model that it happened.

This is deliberately mechanical. Summarising with a model would keep more meaning but costs a call and can drift or
be steered by what it summarises; it can be added behind the same interface. Compaction rewrites the start of the
conversation, which makes the prompt cache miss once; that is the price of staying within budget.

Token counts here are an estimate (characters / 4), good enough to decide when to compact, not to bill.
"""

from __future__ import annotations

import json
from typing import Any

STUB_MARK = "removed to save space"


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, default=str)) // 4


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    content = message["content"]
    return message["role"] == "user" and isinstance(content, list) and any(b.get("type") == "tool_result" for b in content)


NUDGE = "Please finish by calling"   # the loop's reminder to submit; it sits inside a turn and does not start one


def turn_starts(messages: list[dict[str, Any]]) -> list[int]:
    """Indexes where a planner's message begins a turn (a user message that is neither tool results nor the loop's nudge)."""
    return [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and not _is_tool_result_message(m)
        and not (isinstance(m["content"], str) and m["content"].startswith(NUDGE))
    ]


def clear_old_results(messages: list[dict[str, Any]], keep_turns: int = 2) -> tuple[list[dict[str, Any]], int]:
    """Copy of ``messages`` with tool results older than the last ``keep_turns`` turns replaced by stubs.

    Returns the new list and how many results were replaced. The input is not modified. Running it again changes
    nothing further.
    """
    starts = turn_starts(messages)
    if len(starts) <= keep_turns:
        return list(messages), 0
    cutoff = starts[-keep_turns] if keep_turns > 0 else len(messages)
    names = {
        b["id"]: b["name"]
        for m in messages[:cutoff] if m["role"] == "assistant" and isinstance(m["content"], list)
        for b in m["content"] if b.get("type") == "tool_use"
    }
    replaced = 0
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i < cutoff and _is_tool_result_message(m):
            blocks = []
            for b in m["content"]:
                if b.get("type") == "tool_result" and STUB_MARK not in str(b.get("content", ""))[:200]:
                    stub = {"type": "tool_result", "tool_use_id": b["tool_use_id"],
                            "content": f"[{names.get(b['tool_use_id'], 'tool')} result {STUB_MARK}; call the tool again if you still need it]"}
                    if b.get("is_error"):
                        stub["is_error"] = True
                    blocks.append(stub)
                    replaced += 1
                else:
                    blocks.append(b)
            out.append({**m, "content": blocks})
        else:
            out.append(m)
    return out, replaced


def drop_oldest_turns(messages: list[dict[str, Any]], keep_turns: int, limit_tokens: int) -> tuple[list[dict[str, Any]], int]:
    """Remove whole oldest turns until the estimate fits, never going below ``keep_turns``. Returns (messages, turns dropped)."""
    messages = list(messages)
    dropped = 0
    while estimate_tokens(messages) > limit_tokens:
        starts = turn_starts(messages)
        if len(starts) <= max(keep_turns, 1):
            break
        del messages[starts[0]:starts[1]]
        dropped += 1
    return messages, dropped
