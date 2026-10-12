"""The agent loop re-expressed as a LangGraph state graph, for a side-by-side comparison with the hand-written loop.

This is a comparison, not part of the product: nothing else imports it, the dependency is an optional extra
(``uv sync --extra frameworks``), and the project's rule of "no agent framework" still holds for the product itself. It exists to
answer an interview question honestly: what does a framework give you for this loop, and what does it cost?

What is the same: the tools and their validation (the same ``ToolRegistry``), the system prompt, the terminal ``submit_response``
tool, the system-computed answer (changes, KPIs, approval, warnings) and the scoring (the whole eval suite runs against it:
``python -m jobshop.evals run --oracle --loop langgraph``).

What is deliberately not ported, because it is where the hand-written loop's code is: the agent patterns (plan, verify, reflect,
route), prompt caching, history compaction, streaming, per-call pricing for two models, and the trace events. The graph is the
plain ReAct loop: a ``model`` node and a ``tools`` node with conditional edges between them.

The loop is one function so it can be swapped in anywhere ``run_turn`` is used; it has the same signature and returns the same
``TurnResult``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TypedDict

import anthropic

try:
    from langgraph.graph import END, StateGraph
except ImportError as e:  # pragma: no cover - exercised only where the extra is missing
    raise ImportError("this comparison needs LangGraph: uv sync --extra frameworks") from e

from jobshop.agent.loop import (
    SUBMIT, SUBMIT_SPEC, AgentConfig, AnswerPayload, FinalResponse, TurnResult, _blocks_to_params, _handle_submit, _result, _run_tool,
    guard_warnings,
)
from jobshop.agent.providers import create_message
from jobshop.agent.trace import Tracer
from jobshop.tools.outcome import draft_outcome
from jobshop.tools.registry import ToolRegistry


class State(TypedDict, total=False):
    status: str            # set when the turn is over: answered, step_limit, api_error, no_final_response, budget_exceeded
    pending: list[dict[str, Any]]   # tool_use blocks from the last model reply, waiting to run


@dataclass
class _Totals:
    steps: int = 0
    nudges: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    llm_ms: int = 0
    tool_ms: int = 0
    last_text: str = ""
    final: FinalResponse | None = None
    error_text: str = ""


def run_turn_langgraph(
    client: Any,
    registry: ToolRegistry,
    system_prompt: str,
    messages: list[dict[str, Any]],
    user_text: str,
    config: AgentConfig,
    tracer: Tracer | None = None,
    on_delta: Any = None,
) -> TurnResult:
    tracer = tracer or Tracer()
    start_len = len(messages)
    messages.append({"role": "user", "content": user_text})
    tracer.event("turn_start", model=config.model, user_text=user_text, pattern="langgraph")
    totals = _Totals()
    tools = registry.api_specs() + [SUBMIT_SPEC]

    def model_node(state: State) -> State:
        if totals.steps >= config.max_steps:
            return {"status": "step_limit", "pending": []}
        if totals.in_tokens + totals.out_tokens + totals.cache_read + totals.cache_write >= config.max_total_tokens:
            return {"status": "budget_exceeded", "pending": []}
        totals.steps += 1
        began = time.perf_counter()
        try:
            response = create_message(client, model=config.model, max_tokens=config.max_output_tokens, system=system_prompt,
                                      tools=tools, messages=messages)
        except anthropic.APIError as e:
            del messages[start_len:]
            totals.error_text = f"The model API failed: {type(e).__name__}: {e}"
            return {"status": "api_error", "pending": []}
        totals.llm_ms += round((time.perf_counter() - began) * 1000)
        usage = response.usage
        totals.in_tokens += usage.input_tokens
        totals.out_tokens += usage.output_tokens
        totals.cache_read += getattr(usage, "cache_read_input_tokens", None) or 0
        totals.cache_write += getattr(usage, "cache_creation_input_tokens", None) or 0
        content = _blocks_to_params(response.content, tracer)
        messages.append({"role": "assistant", "content": content})
        calls = [b for b in content if b["type"] == "tool_use"]
        totals.last_text = "\n".join(b["text"] for b in content if b["type"] == "text") or totals.last_text
        if calls:
            if response.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": [
                    _result(b, "Your response was cut off, so this call was not run. Retry with less text.", True) for b in calls]})
                return {"pending": []}
            return {"pending": calls}
        if totals.nudges >= config.max_nudges:
            return {"status": "no_final_response", "pending": []}
        totals.nudges += 1
        messages.append({"role": "user", "content": f"Please finish by calling {SUBMIT}; it is the only way your answer reaches the planner."})
        return {"pending": []}

    def tools_node(state: State) -> State:
        results: list[dict[str, Any]] = []
        payload: AnswerPayload | None = None
        calls = state.get("pending", [])
        for block in calls:
            if block["name"] == SUBMIT:
                got, result = _handle_submit(block, alone=len(calls) == 1)
                payload = got or payload
                results.append(result)
            else:
                result, took_ms = _run_tool(registry, block, tracer, totals.steps)
                totals.tool_ms += took_ms
                results.append(result)
        messages.append({"role": "user", "content": results})
        if payload is None:
            return {"pending": []}
        outcome = draft_outcome(registry.ctx, payload.draft_id)
        totals.final = FinalResponse(
            **payload.model_dump(), changes_made=outcome.changes, kpi_before=outcome.kpi_before, kpi_after=outcome.kpi_after,
            needs_approval=outcome.needs_approval,
            warnings=[*outcome.warnings, *(guard_warnings(payload.summary, outcome) if config.answer_guards else [])], goal=outcome.goal,
        )
        recap = totals.final.clarifying_question or totals.final.summary
        if payload.draft_id:
            recap += f" (Proposal is in draft {payload.draft_id}.)"
        messages.append({"role": "assistant", "content": [{"type": "text", "text": recap}]})
        return {"status": "answered", "pending": []}

    def after_model(state: State) -> str:
        if state.get("status"):
            return END
        return "tools" if state.get("pending") else "model"

    def after_tools(state: State) -> str:
        return END if state.get("status") else "model"

    graph = StateGraph(State)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", after_model, {"tools": "tools", "model": "model", END: END})
    graph.add_conditional_edges("tools", after_tools, {"model": "model", END: END})
    final_state = graph.compile().invoke({"pending": []}, {"recursion_limit": 4 * config.max_steps + 20})

    status = final_state.get("status") or "no_final_response"
    text = totals.error_text if status == "api_error" else ("" if totals.final else totals.last_text)
    result = TurnResult(status, totals.final, text, totals.steps, totals.in_tokens, totals.out_tokens, None, totals.llm_ms,
                        totals.tool_ms, totals.cache_read, totals.cache_write, None)
    tracer.event("turn_end", status=status, steps=totals.steps, input_tokens=totals.in_tokens, output_tokens=totals.out_tokens)
    return result
