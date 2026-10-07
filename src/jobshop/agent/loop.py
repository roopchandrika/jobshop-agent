"""The tool-use loop, written by hand on the Anthropic Messages API.

One *turn* is: the planner says something, then we repeat {ask the model, run the tools it
asks for, send the results back} until it calls ``submit_response`` (or a limit is hit).

What the loop guarantees, independent of how well the model behaves:

* It stops: at most ``max_steps`` model calls and ``max_total_tokens`` tokens per turn.
* A failing tool never crashes the chat: the error goes back to the model as an error result.
* The model never commits: it is only shown the registry's visible tools, and any attempt to
  call a hidden one is answered as "unknown tool".
* The final answer's KPIs and ``needs_approval`` come from the store (see ``tools.outcome``),
  never from text the model typed.
* ``messages`` stays valid for the next turn: every tool_use is answered, and roles alternate.

The client is injected (``client.messages.create(...)``), so tests drive this loop with a
scripted fake and never need the network.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jobshop.agent.trace import Tracer
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import DraftId
from jobshop.tools.outcome import draft_outcome
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.views import KPIView

SUBMIT = "submit_response"


class AnswerPayload(BaseModel):
    """What the model supplies when it finishes: words and a draft id. Nothing numeric."""

    model_config = ConfigDict(extra="forbid")

    # No `changes_made` here on purpose: the list a human sees beside the approval prompt comes
    # from the draft itself (see FinalResponse), so the model cannot misdescribe what it changed.
    summary: str = Field(
        min_length=1,
        max_length=4000,
        description="The outcome and trade-offs in plain language for the planner. Quote only numbers from tool results.",
    )
    draft_id: DraftId | None = Field(
        None, description="The draft that holds your proposal, if you made one."
    )
    clarifying_question: str | None = Field(
        None,
        max_length=1000,
        description="Set ONLY if the request is too ambiguous to act on. Then make no changes.",
    )


SUBMIT_SPEC = {
    "name": SUBMIT,
    "description": (
        "Give your final answer to the planner. Call this exactly once, on its own, after you "
        "have all the tool results you need. KPIs and the approval prompt are added "
        "automatically from the tool data, so do not repeat them in a table."
    ),
    "input_schema": AnswerPayload.model_json_schema(),
}


class FinalResponse(AnswerPayload):
    """The answer plus the parts the harness computes itself."""

    changes_made: list[str] = Field(default_factory=list)  # recorded by the draft, not typed by the model
    kpi_before: KPIView | None = None
    kpi_after: KPIView | None = None
    needs_approval: bool = False
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentConfig:
    model: str
    max_steps: int = 12
    max_total_tokens: int = 200_000
    max_output_tokens: int = 2048
    max_nudges: int = 2
    max_cost_usd: float | None = None
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None

    def __post_init__(self) -> None:
        if self.max_cost_usd is not None and None in (self.price_input_per_mtok, self.price_output_per_mtok):
            raise ValueError("a cost budget needs both token prices (input and output per million tokens)")


Status = Literal["answered", "no_final_response", "step_limit", "budget_exceeded", "api_error"]


@dataclass
class TurnResult:
    status: Status
    final: FinalResponse | None
    text: str | None  # explanation when there is no final answer
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def run_turn(
    client: Any,
    registry: ToolRegistry,
    system_prompt: str,
    messages: list[dict[str, Any]],
    user_text: str,
    config: AgentConfig,
    tracer: Tracer | None = None,
) -> TurnResult:
    tracer = tracer or Tracer()
    start_len = len(messages)
    messages.append({"role": "user", "content": user_text})
    tracer.event("turn_start", user_text=user_text)

    tools = registry.api_specs() + [SUBMIT_SPEC]
    steps = nudges = in_tokens = out_tokens = 0
    last_text = ""

    def cost() -> float | None:
        if config.price_input_per_mtok is None or config.price_output_per_mtok is None:
            return None
        return (in_tokens * config.price_input_per_mtok + out_tokens * config.price_output_per_mtok) / 1e6

    def finish(status: Status, final: FinalResponse | None = None, text: str | None = None) -> TurnResult:
        result = TurnResult(status, final, text, steps, in_tokens, out_tokens, cost())
        tracer.event("turn_end", status=status, steps=steps, input_tokens=in_tokens,
                     output_tokens=out_tokens, cost_usd=result.cost_usd, text=text)
        return result

    def stop_politely(status: Status, why: str) -> TurnResult:
        # Keep the history valid for the next turn: end with an assistant message.
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"(I stopped: {why}.)"}]})
        return finish(status, text=why)

    while True:
        if steps >= config.max_steps:
            return stop_politely("step_limit", f"reached the limit of {config.max_steps} model calls")
        spent = cost()
        if in_tokens + out_tokens >= config.max_total_tokens or (
            config.max_cost_usd is not None and spent is not None and spent >= config.max_cost_usd
        ):
            return stop_politely("budget_exceeded", "reached the token/cost budget for this request")

        steps += 1
        began = time.perf_counter()
        try:
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_output_tokens,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as e:
            del messages[start_len:]  # nothing from this turn happened as far as the model knows
            tracer.event("api_error", step=steps, error=f"{type(e).__name__}: {e}")
            return finish("api_error", text=f"The model API failed: {type(e).__name__}: {e}")
        latency_ms = round((time.perf_counter() - began) * 1000)

        in_tokens += response.usage.input_tokens
        out_tokens += response.usage.output_tokens
        content = _blocks_to_params(response.content, tracer)
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b["type"] == "tool_use"]
        texts = [b["text"] for b in content if b["type"] == "text"]
        last_text = "\n".join(texts) or last_text
        tracer.event(
            "llm_call", step=steps, model=config.model, stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms, cost_usd=cost(), text=texts, tool_calls=[b["name"] for b in tool_uses],
        )

        if not tool_uses:
            if nudges >= config.max_nudges:
                return finish("no_final_response", text=last_text or "The model gave no answer.")
            nudges += 1
            messages.append({"role": "user", "content": (
                f"Please finish by calling {SUBMIT}; it is the only way your answer reaches the planner."
            )})
            continue

        truncated = response.stop_reason == "max_tokens"
        results: list[dict[str, Any]] = []
        payload: AnswerPayload | None = None
        for block in tool_uses:
            if truncated:
                results.append(_result(block, "Your response was cut off, so this call was not run. Retry with less text.", True))
            elif block["name"] == SUBMIT:
                payload, result = _handle_submit(block, alone=len(tool_uses) == 1)
                results.append(result)
            else:
                results.append(_run_tool(registry, block, tracer, steps))
        messages.append({"role": "user", "content": results})

        if payload is not None:
            outcome = draft_outcome(registry.ctx, payload.draft_id)
            final = FinalResponse(
                **payload.model_dump(),
                changes_made=outcome.changes,
                kpi_before=outcome.kpi_before,
                kpi_after=outcome.kpi_after,
                needs_approval=outcome.needs_approval,
                warnings=outcome.warnings,
            )
            recap = final.clarifying_question or final.summary
            if payload.draft_id:
                recap += f" (Proposal is in draft {payload.draft_id}.)"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": recap}]})
            return finish("answered", final=final)


def _blocks_to_params(blocks: list[Any], tracer: Tracer) -> list[dict[str, Any]]:
    """Echo back only what the API accepts as input. Real SDK blocks carry extra fields."""
    params: list[dict[str, Any]] = []
    for block in blocks:
        if block.type == "text":
            params.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            params.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        else:
            tracer.event("dropped_block", block_type=block.type)
    return params


def _result(block: dict[str, Any], content: Any, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": block["id"],
        "content": content if isinstance(content, str) else json.dumps(content),
    }
    if is_error:
        result["is_error"] = True
    return result


def _handle_submit(block: dict[str, Any], alone: bool) -> tuple[AnswerPayload | None, dict[str, Any]]:
    if not alone:
        return None, _result(block, f"Call {SUBMIT} on its own, after you have seen all tool results.", True)
    try:
        payload = AnswerPayload.model_validate(block["input"])
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return None, _result(block, f"Invalid {SUBMIT} arguments ({problems}). Call it again with a valid payload.", True)
    return payload, _result(block, "Answer delivered to the planner.")


def _run_tool(registry: ToolRegistry, block: dict[str, Any], tracer: Tracer, step: int) -> dict[str, Any]:
    began = time.perf_counter()
    is_error = False
    try:
        # allow_hidden stays False: the model can only use the tools it was shown.
        content: Any = registry.call(block["name"], block["input"])
    except ToolError as e:
        content, is_error = {"error": str(e)}, True
    except Exception as e:  # a bug in a tool must not kill the chat
        tracer.event("tool_exception", step=step, tool=block["name"], traceback=traceback.format_exc())
        content, is_error = {"error": f"internal error in {block['name']} ({type(e).__name__})"}, True
    tracer.event(
        "tool_call", step=step, tool=block["name"], arguments=block["input"], is_error=is_error,
        result=content, latency_ms=round((time.perf_counter() - began) * 1000),
    )
    return _result(block, content, is_error)
