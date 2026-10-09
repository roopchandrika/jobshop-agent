"""Open models: the same agent loop on a model served by Ollama, next to Anthropic models.

The loop only ever calls ``client.messages.create(...)`` and reads ``content``, ``usage`` and ``stop_reason`` from
what comes back. So a second provider is an adapter that takes the same request, talks to Ollama's ``/api/chat``,
and returns the same shape. A model whose name starts with ``ollama:`` goes there (``ollama:llama3.1``); every
other name goes to Anthropic, so one run can mix them (a small local model for the triage, a large one for the plan).

What the adapter cannot do, and says so rather than pretending:

* Forced tool choice. Ollama has no ``tool_choice``, so a forced call is approximated by offering only that tool and
  saying in the system prompt that it must be called. A model may still answer in prose; the loop treats that like
  any other malformed answer (the triage falls back to the full agent, the plan stage is skipped).
* Prompt caching. ``cache_control`` is dropped; no cache tokens are reported.
* The context window. Ollama silently cuts a prompt that does not fit ``num_ctx``, which would drop the start of the
  system prompt, the part with the safety rules. The adapter sets a large ``num_ctx`` and refuses a reply whose prompt
  filled the window instead of letting the model answer from a truncated prompt.

This adapter is tested against a stub server that behaves as Ollama's documentation says. It has not been run against a
real Ollama (none is installed on the machine it was written on), so the first real run is the real test.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import anthropic
import httpx2
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

OLLAMA_PREFIX = "ollama:"
DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 16384
DEFAULT_TIMEOUT_S = 300.0     # a local model on a laptop can be slow, especially on the first call (it loads the weights)


def is_open_model(model: str | None) -> bool:
    return bool(model) and model.startswith(OLLAMA_PREFIX)


def needs_anthropic(*models: str | None) -> bool:
    """True if any of these (non-empty) model names is served by Anthropic, so an API key is required."""
    return any(m and not is_open_model(m) for m in models)


class OllamaClient:
    """Quacks like ``anthropic.Anthropic`` for ``messages.create``, backed by a local Ollama server."""

    def __init__(self, base_url: str = DEFAULT_URL, *, num_ctx: int = DEFAULT_NUM_CTX, timeout_s: float = DEFAULT_TIMEOUT_S,
                 http: httpx2.Client | None = None) -> None:
        if num_ctx < 2048:
            raise ValueError("num_ctx below 2048 cannot hold the system prompt and tool descriptions")
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self._http = http or httpx2.Client(timeout=timeout_s)
        self._n = 0
        self.messages = self

    def create(self, *, model: str, max_tokens: int, system: Any, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None = None, tool_choice: dict[str, Any] | None = None, **_ignored: Any) -> Message:
        self._n += 1
        name = model.removeprefix(OLLAMA_PREFIX)
        tools = list(tools or [])
        system_text = _system_text(system)
        if tool_choice and tool_choice.get("type") == "tool":
            forced = tool_choice["name"]
            tools = [t for t in tools if t["name"] == forced]
            system_text += f"\n\nYou must respond now by calling the tool '{forced}'. Do not answer in prose."
        body: dict[str, Any] = {
            "model": name, "stream": False,
            "messages": [{"role": "system", "content": system_text}, *_convert_messages(messages)],
            "options": {"num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                                "parameters": t["input_schema"]}} for t in tools]
        url = f"{self.base_url}/api/chat"
        request = httpx2.Request("POST", url)
        try:
            response = self._http.post(url, json=body)
        except httpx2.TransportError as e:
            raise anthropic.APIConnectionError(message=f"cannot reach Ollama at {self.base_url}: {e}", request=request) from e
        if response.status_code != 200:
            raise anthropic.APIStatusError(f"Ollama answered {response.status_code}: {response.text[:300]}", response=response, body=None)
        try:
            data = response.json()
            reply = data["message"]
        except (ValueError, KeyError, TypeError) as e:
            raise anthropic.APIConnectionError(message=f"Ollama sent a reply that is not a chat message: {response.text[:200]}", request=request) from e
        prompt_tokens, output_tokens = int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)
        if prompt_tokens >= self.num_ctx - 1:
            raise anthropic.APIConnectionError(
                message=f"the prompt filled Ollama's whole {self.num_ctx}-token window, so the start of it (the rules) was probably cut off; "
                        "raise JOBSHOP_OLLAMA_NUM_CTX or use a shorter conversation", request=request)
        blocks: list[Any] = []
        if text := (reply.get("content") or "").strip():
            blocks.append(TextBlock(type="text", text=text))
        for k, call in enumerate(reply.get("tool_calls") or []):
            function = call.get("function") or {}
            blocks.append(ToolUseBlock(type="tool_use", id=f"ollama_{self._n}_{k}", name=function.get("name", ""),
                                       input=_arguments(function.get("arguments"))))
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "max_tokens" if data.get("done_reason") == "length" else "end_turn"
        return Message(id=f"msg_ollama_{self._n}", type="message", role="assistant", model=model, content=blocks, stop_reason=stop,
                       stop_sequence=None, usage=Usage(input_tokens=prompt_tokens, output_tokens=output_tokens))


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system if isinstance(b, Mapping))


def _arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict. Some models send them as a JSON string, some as garbage; garbage becomes an argument the
    tool rejects by name, so the model is told what went wrong instead of the call silently running with no arguments."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"unparsed_arguments": raw}
        return parsed if isinstance(parsed, dict) else {"unparsed_arguments": raw}
    return {}


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-format history to Ollama chat messages. Tool results become 'tool' messages that name the tool."""
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        if message["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = []
            for b in content:
                if b["type"] == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b["input"]}})
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        text_parts = []
        for b in content:
            if b["type"] == "tool_result":
                result = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                out.append({"role": "tool", "tool_name": names.get(b["tool_use_id"], ""), "content": ("ERROR: " if b.get("is_error") else "") + result})
            elif b["type"] == "text":
                text_parts.append(b["text"])
        if text_parts:
            out.append({"role": "user", "content": "\n\n".join(text_parts)})
    return out


class MultiProviderClient:
    """Sends each call to Ollama or Anthropic by the model's name. The Anthropic client is made on first use, so a run that
    only uses local models never needs a key."""

    def __init__(self, ollama: OllamaClient, anthropic_client: Any | None = None) -> None:
        self._ollama = ollama
        self._anthropic = anthropic_client
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        if is_open_model(kwargs.get("model")):
            return self._ollama.create(**kwargs)
        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic()
        return self._anthropic.messages.create(**kwargs)


def ollama_from_env(env: Mapping[str, str]) -> OllamaClient:
    def number(name: str, default: float, cast: type) -> Any:
        raw = env.get(name, "").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            raise ValueError(f"{name} must be a number, got {raw!r}") from None

    return OllamaClient(env.get("JOBSHOP_OLLAMA_URL", "").strip() or DEFAULT_URL,
                        num_ctx=number("JOBSHOP_OLLAMA_NUM_CTX", DEFAULT_NUM_CTX, int),
                        timeout_s=number("JOBSHOP_OLLAMA_TIMEOUT_S", DEFAULT_TIMEOUT_S, float))


def build_client(env: Mapping[str, str] | None = None, *models: str | None) -> Any:
    """The client for a run that will use these models: plain Anthropic if none is open, else one that routes by name."""
    if not any(is_open_model(m) for m in models):
        return anthropic.Anthropic()
    return MultiProviderClient(ollama_from_env(os.environ if env is None else env))
