"""A scripted stand-in for ``anthropic.Anthropic``.

It returns real ``anthropic.types.Message`` objects (the same classes the SDK returns), so the
loop's handling of SDK types is exercised; only the network is faked. Each script entry is a
Message, an exception to raise, or a callable that receives the request kwargs and returns one.
"""

from __future__ import annotations

import copy
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def text(content: str) -> TextBlock:
    return TextBlock(type="text", text=content)


def tool(name: str, tool_id: str, **arguments: Any) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=tool_id, name=name, input=arguments)


def message(
    *blocks, stop_reason: str | None = None, tokens_in: int = 100, tokens_out: int = 50,
    cache_read: int = 0, cache_write: int = 0,
) -> Message:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return Message(
        id="msg_fake", type="message", role="assistant", model="fake-model",
        content=list(blocks), stop_reason=stop_reason, stop_sequence=None,
        usage=Usage(
            input_tokens=tokens_in, output_tokens=tokens_out,
            cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write,
        ),
    )


def submit(tool_id: str = "toolu_submit", **payload: Any) -> Message:
    payload.setdefault("summary", "Done.")
    return message(tool("submit_response", tool_id, **payload))


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.messages = self  # so `client.messages.create(...)` works

    def create(self, **kwargs: Any) -> Message:
        # Snapshot the request: the loop keeps mutating the same messages list afterwards.
        self.requests.append(copy.deepcopy(kwargs))
        if not self.script:
            raise AssertionError("the model was called more times than the test scripted")
        item = self.script.pop(0)
        if callable(item):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        return item


def last_text(request: dict[str, Any]) -> str:
    """The text of the last message in a request, whether it was sent as a string or as blocks."""
    content = request["messages"][-1]["content"]
    return content if isinstance(content, str) else " ".join(b["text"] for b in content if b["type"] == "text")


def last_tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool_result blocks the model was shown most recently in a request."""
    last = request["messages"][-1]
    assert last["role"] == "user" and isinstance(last["content"], list)
    return last["content"]
