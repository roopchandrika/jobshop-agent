"""Streaming a model call, so the planner can watch the answer being written instead of waiting for all of it.

The answer is not ordinary text: the model delivers it by calling ``submit_response``, so what the planner reads arrives as the
``summary`` field of a tool call's JSON input, which the API streams as partial JSON. The SDK parses that partial JSON for us
(``input_json`` events carry the object so far), so this module only has to notice when ``summary`` grows and pass on the new
part. Plain text blocks (the model thinking aloud between tool calls) are passed on too, as a different kind.

Only unforced calls are streamed: the triage, plan and review calls are internal and never shown.

``on_delta(kind, text)`` is called with:

* ``"reset"``: a new streamed call began; discard what was shown so far (an answer can be sent back and rewritten);
* ``"answer"``: the next piece of the answer the planner will read;
* ``"text"``: the next piece of ordinary text.

The final message is still the complete, validated response the loop works on; streaming only adds a view of it as it forms.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DeltaCallback = Callable[[str, str], None]


def can_stream(client: Any) -> bool:
    return callable(getattr(getattr(client, "messages", None), "stream", None))


def stream_message(client: Any, on_delta: DeltaCallback, **kwargs: Any) -> Any:
    """``client.messages.create`` made as a stream. Returns the complete final message; raises what the SDK raises."""
    on_delta("reset", "")
    shown = ""
    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            kind = getattr(event, "type", None)
            if kind == "text":
                on_delta("text", event.text)
            elif kind == "input_json":
                snapshot = getattr(event, "snapshot", None)
                summary = snapshot.get("summary") if isinstance(snapshot, dict) else None
                if isinstance(summary, str) and len(summary) > len(shown) and summary.startswith(shown):
                    on_delta("answer", summary[len(shown):])
                    shown = summary
        return stream.get_final_message()
