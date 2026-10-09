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

from jobshop.agent.claims import LIVE_CLAIM_WARNING, claims_nothing_is_late, claims_plan_is_live
from jobshop.agent.patterns import (
    COMMIT_REFUSAL, PATTERNS, PLAN_RULES, PLAN_TOOL, READER_ROLE, READ_ONLY_TOOLS, REVIEW_TOOL, TRIAGE_SPEC,
    TRIAGE_SYSTEM, TRIAGE_TOOL, Plan, critic_request, evidence_facts, harness_facts, parse_plan, parse_review,
    parse_triage, plan_adherence, plan_block, plan_spec, planner_context, review_message, unverified_figures,
    verify_message,
)
from jobshop.agent.pricing import Prices
from jobshop.agent.trace import TRACE_VERSION, Tracer
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import DraftId
from jobshop.tools.outcome import DraftOutcome, draft_outcome
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.views import Goal, KPIView, fmt

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
    goal: Goal | None = None  # what the proposal was solved for, recorded by the draft


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
    # Mark the conversation so far as cacheable: each model call then re-reads the system prompt, the
    # tool definitions and earlier steps at a fraction of the price instead of paying for them again.
    # Deterministic checks on the words of the answer against what the harness knows (see guard_warnings). On unless a red-team
    # run is measuring what they are worth.
    answer_guards: bool = True
    prompt_caching: bool = True
    # Short-term memory: when the conversation's estimated size passes this, old tool results are cleared (and, if need be,
    # the oldest turns dropped). None switches it off. See agent/history.py.
    history_token_limit: int | None = 40_000
    history_keep_turns: int = 2   # this many latest turns are never touched
    # How the agent is organised around the model: react (default), plan, verify, reflect, or several joined with '+'.
    # See agent/patterns.py. 'verify' and 'reflect' may send an answer back this many times before delivering it with a warning.
    pattern: str = "react"
    max_revisions: int = 1
    # With the 'route' pattern: the model that does the triage (a small cheap one is the point), and its own prices.
    # None means the main model. Every call is costed at its own model's prices.
    triage_model: str | None = None
    triage_price_input_per_mtok: float | None = None
    triage_price_output_per_mtok: float | None = None

    def __post_init__(self) -> None:
        unknown = set(self.patterns) - set(PATTERNS)
        if unknown or not self.patterns:
            raise ValueError(f"unknown pattern {self.pattern!r}; choose from {', '.join(PATTERNS)}, joined with + to combine")
        if self.triage_model and self.triage_model != self.model and self.prices is not None and self.triage_prices is None:
            raise ValueError("a separate triage model needs its own prices (triage_price_*), or its cost would be counted at the main model's rate")
        if self.max_cost_usd is not None and None in (self.price_input_per_mtok, self.price_output_per_mtok):
            raise ValueError("a cost budget needs both token prices (input and output per million tokens)")

    @property
    def patterns(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.pattern.lower().split("+") if p.strip())

    @property
    def triage_prices(self) -> Prices | None:
        if self.triage_price_input_per_mtok is None or self.triage_price_output_per_mtok is None:
            return None
        return Prices(self.triage_price_input_per_mtok, self.triage_price_output_per_mtok)

    @property
    def prices(self) -> Prices | None:
        if self.price_input_per_mtok is None or self.price_output_per_mtok is None:
            return None
        return Prices(self.price_input_per_mtok, self.price_output_per_mtok)


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
    llm_ms: int = 0   # time spent waiting for the model, summed over its calls
    tool_ms: int = 0  # time spent running tools (the solver dominates this)
    # input_tokens above counts only uncached input, as the API reports it; cached input is counted here.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    route: str | None = None   # with the 'route' pattern: where the request was sent (see agent/patterns.py)

    @property
    def total_tokens(self) -> int:
        """Everything the model processed, cached or not (what a person means by "tokens used")."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


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
    tracer.event("turn_start", trace_version=TRACE_VERSION, model=config.model, user_text=user_text, pattern=config.pattern)

    patterns = config.patterns
    route: str | None = None
    spent = 0.0
    unpriced_calls = 0
    steps = nudges = revisions = in_tokens = out_tokens = llm_ms = tool_ms = cache_read = cache_write = 0
    last_text = api_error_text = ""
    turn_began = time.perf_counter()
    system = system_prompt
    plan: Plan | None = None
    extra_warnings: list[str] = []

    def prices_for(model: str) -> Prices | None:
        return config.triage_prices if model == config.triage_model and model != config.model else config.prices

    def cost() -> float | None:
        """What the turn has cost so far: each call at its own model's prices; None if any call could not be priced."""
        return None if unpriced_calls or config.prices is None else spent

    def finish(status: Status, final: FinalResponse | None = None, text: str | None = None) -> TurnResult:
        result = TurnResult(status, final, text, steps, in_tokens, out_tokens, cost(), llm_ms, tool_ms, cache_read, cache_write, route)
        tracer.event("turn_end", status=status, steps=steps, input_tokens=in_tokens,
                     output_tokens=out_tokens, cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                     cost_usd=result.cost_usd, llm_ms=llm_ms, tool_ms=tool_ms, route=route,
                     wall_ms=round((time.perf_counter() - turn_began) * 1000), text=text)
        return result

    def stop_politely(status: Status, why: str) -> TurnResult:
        # Keep the history valid for the next turn: end with an assistant message.
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"(I stopped: {why}.)"}]})
        return finish(status, text=why)

    def out_of_budget() -> TurnResult | None:
        if steps >= config.max_steps:
            return stop_politely("step_limit", f"reached the limit of {config.max_steps} model calls")
        spent = cost()
        if in_tokens + out_tokens + cache_read + cache_write >= config.max_total_tokens or (
            config.max_cost_usd is not None and spent is not None and spent >= config.max_cost_usd
        ):
            return stop_politely("budget_exceeded", "reached the token/cost budget for this request")
        return None

    def ask(*, system: str, tools: list[dict[str, Any]], history: list[dict[str, Any]], purpose: str,
            tool_choice: dict[str, Any] | None = None, cache: bool = True, max_tokens: int | None = None,
            model: str | None = None):
        """One model call with all the accounting (steps, tokens, cost, time, trace). Returns (response, content),
        or None if the API failed (the failure is traced and ``api_error_text`` is set)."""
        nonlocal steps, in_tokens, out_tokens, llm_ms, cache_read, cache_write, api_error_text, spent, unpriced_calls
        model = model or config.model
        steps += 1
        began = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens or config.max_output_tokens, system=system,
            tools=_cacheable_tools(tools) if cache and config.prompt_caching else tools,
            messages=_cacheable_messages(history) if cache and config.prompt_caching else history,
        )
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        try:
            response = client.messages.create(**kwargs)
        except anthropic.APIError as e:
            del messages[start_len:]  # nothing from this turn happened as far as the model knows
            api_error_text = f"The model API failed: {type(e).__name__}: {e}"
            tracer.event("api_error", step=steps, error=f"{type(e).__name__}: {e}",
                         latency_ms=round((time.perf_counter() - began) * 1000))
            return None
        latency_ms = round((time.perf_counter() - began) * 1000)
        llm_ms += latency_ms
        step_read = getattr(response.usage, "cache_read_input_tokens", None) or 0
        step_write = getattr(response.usage, "cache_creation_input_tokens", None) or 0
        in_tokens += response.usage.input_tokens
        out_tokens += response.usage.output_tokens
        cache_read += step_read
        cache_write += step_write
        content = _blocks_to_params(response.content, tracer)
        calls = [b for b in content if b["type"] == "tool_use"]
        call_prices = prices_for(model)
        step_cost = None if call_prices is None else call_prices.cost(response.usage.input_tokens, response.usage.output_tokens, step_read, step_write)
        if step_cost is None:
            unpriced_calls += 1
        else:
            spent += step_cost
        tracer.event(
            "llm_call", step=steps, model=model, response_id=getattr(response, "id", None),
            stop_reason=response.stop_reason, purpose=purpose,
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            cache_read_tokens=step_read, cache_write_tokens=step_write,
            latency_ms=latency_ms,
            step_cost_usd=step_cost,
            total_cost_usd=cost(), text=[b["text"] for b in content if b["type"] == "text"], tool_calls=[b["name"] for b in calls],
        )
        return response, content

    # -- pattern: route (multi-agent). A triage call that sees only the planner's words picks a specialist. ------------------------
    if "route" in patterns:
        triaged = ask(model=config.triage_model or config.model, system=TRIAGE_SYSTEM, tools=[TRIAGE_SPEC],
                      history=[{"role": "user", "content": planner_context(messages[:start_len], user_text)}], purpose="triage",
                      tool_choice={"type": "tool", "name": TRIAGE_TOOL}, cache=False, max_tokens=300)
        if triaged is None:
            return finish("api_error", text=api_error_text)
        call = next((b for b in triaged[1] if b["type"] == "tool_use" and b["name"] == TRIAGE_TOOL), None)
        decision = parse_triage(call["input"]) if call else None
        if decision is None:
            tracer.event("route_invalid")              # carry on as the full agent: a bad triage must not block the planner
            route = "plan"
        else:
            route = decision.route
            tracer.event("route", route=route, reason=decision.reason)
        if route in ("clarify", "decline_commit"):
            # Answered without the agent: a question the triage wrote, or a fixed refusal. No tools, no further model call.
            assert decision is not None
            final = FinalResponse(
                summary="I need one detail before I change anything." if route == "clarify" else COMMIT_REFUSAL,
                clarifying_question=decision.question if route == "clarify" else None,
            )
            messages.append({"role": "assistant", "content": [{"type": "text", "text": final.clarifying_question or final.summary}]})
            return finish("answered", final=final)
        if route == "read":
            registry = registry.scoped(READ_ONLY_TOOLS)
            system_prompt = system_prompt + READER_ROLE
            system = system_prompt
    tools = registry.api_specs() + [SUBMIT_SPEC]

    # -- pattern: plan. One forced call before acting; the plan guides this turn only and is not stored in the history. ---------
    if "plan" in patterns:
        planned = ask(system=system_prompt + PLAN_RULES, tools=[plan_spec(registry.names() + [SUBMIT])], history=messages,
                      purpose="plan", tool_choice={"type": "tool", "name": PLAN_TOOL})
        if planned is None:
            return finish("api_error", text=api_error_text)
        call = next((b for b in planned[1] if b["type"] == "tool_use" and b["name"] == PLAN_TOOL), None)
        plan = parse_plan(call["input"]) if call else None
        if plan is None:
            tracer.event("plan_invalid")           # carry on without one: a plan is a help, not a gate
        else:
            system = system_prompt + plan_block(plan)
            tracer.event("plan", steps=[s.model_dump() for s in plan.steps])

    while True:
        if (stop := out_of_budget()) is not None:
            return stop

        got = ask(system=system, tools=tools, history=messages, purpose="agent")
        if got is None:
            return finish("api_error", text=api_error_text)
        response, content = got
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b["type"] == "tool_use"]
        last_text = "\n".join(b["text"] for b in content if b["type"] == "text") or last_text

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
        submit_block: dict[str, Any] | None = None
        submit_at = -1
        for block in tool_uses:
            if truncated:
                results.append(_result(block, "Your response was cut off, so this call was not run. Retry with less text.", True))
            elif block["name"] == SUBMIT:
                payload, result = _handle_submit(block, alone=len(tool_uses) == 1)
                if payload is not None:
                    submit_block, submit_at = block, len(results)
                results.append(result)
            else:
                result, took_ms = _run_tool(registry, block, tracer, steps)
                results.append(result)
                tool_ms += took_ms

        outcome = draft_outcome(registry.ctx, payload.draft_id) if payload is not None else None
        if payload is not None and outcome is not None and ("verify" in patterns or "reflect" in patterns):
            # -- patterns: verify, reflect. Check the answer before the planner sees it; send it back at most ``max_revisions`` times.
            committed = registry.ctx.store.committed
            problem: str | None = None
            kept: list[str] = []
            if "verify" in patterns:
                figures = unverified_figures(payload.summary, evidence_facts(messages, fmt(committed.instance, committed.instance.now)))
                tracer.event("claims_check", figures=figures, revision=revisions)
                if figures:
                    problem = verify_message(figures, revisions + 1, config.max_revisions)
                    kept.append(f"The answer quotes figure(s) that no tool returned: {', '.join(figures)}.")
            if problem is None and "reflect" in patterns and steps < config.max_steps:
                request = critic_request(payload.summary, harness_facts(messages[start_len:], outcome))
                reviewed = ask(**{k: request[k] for k in ("system", "tools", "tool_choice")}, history=request["messages"],
                               purpose="critic", cache=False, max_tokens=min(config.max_output_tokens, 600))
                if reviewed is None:
                    return finish("api_error", text=api_error_text)
                call = next((b for b in reviewed[1] if b["type"] == "tool_use" and b["name"] == REVIEW_TOOL), None)
                review = parse_review(call["input"]) if call else None
                tracer.event("critique", ok=None if review is None else review.ok,
                             problems=[] if review is None else review.problems, revision=revisions)
                if review is not None and not review.ok:
                    problem = review_message(review.problems, revisions + 1, config.max_revisions)
                    kept.append("A reviewer flagged: " + "; ".join(review.problems))
            if problem is not None:
                if revisions < config.max_revisions:
                    revisions += 1
                    assert submit_block is not None
                    results[submit_at] = _result(submit_block, problem, True)
                    payload = None
                else:
                    extra_warnings += kept              # out of revisions: deliver it, but say so beside the answer

        messages.append({"role": "user", "content": results})

        if payload is not None:
            assert outcome is not None
            final = FinalResponse(
                **payload.model_dump(),
                changes_made=outcome.changes,
                kpi_before=outcome.kpi_before,
                kpi_after=outcome.kpi_after,
                needs_approval=outcome.needs_approval,
                warnings=[*outcome.warnings, *extra_warnings, *(guard_warnings(payload.summary, outcome) if config.answer_guards else [])],
                goal=outcome.goal,
            )
            recap = final.clarifying_question or final.summary
            if payload.draft_id:
                recap += f" (Proposal is in draft {payload.draft_id}.)"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": recap}]})
            if plan is not None:
                actual = [b["name"] for m in messages[start_len:] if m["role"] == "assistant" and isinstance(m["content"], list)
                          for b in m["content"] if b.get("type") == "tool_use" and b["name"] != SUBMIT]
                tracer.event("plan_adherence", **plan_adherence([s.tool for s in plan.steps if s.tool != SUBMIT], actual))
            return finish("answered", final=final)


def guard_warnings(summary: str, outcome: DraftOutcome) -> list[str]:
    """Warnings shown beside an answer whose words contradict what the harness knows. They never change or block the answer:
    the model's text stays as written, and the planner sees the contradiction next to it."""
    found: list[str] = []
    if claims_plan_is_live(summary):
        found.append(LIVE_CLAIM_WARNING)
    late = outcome.kpi_after.late_order_ids if outcome.kpi_after is not None else []
    if late and claims_nothing_is_late(summary):
        found.append(f"The answer says no order is late, but the solver's result for this draft shows {len(late)} late: {', '.join(late)}.")
    return found


_CACHE = {"type": "ephemeral"}


def _cacheable_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of the tool list with a cache breakpoint on the last tool (the tools come first in the prompt)."""
    return [*tools[:-1], {**tools[-1], "cache_control": _CACHE}]


def _cacheable_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of the history with a cache breakpoint on its last block.

    The next model call starts with exactly this prefix (system prompt, tools, every earlier step), so
    it is read back from the cache instead of being processed and billed again. ``messages`` itself is
    left alone: it is the stored conversation and must stay free of request-only markers.
    """
    last = messages[-1]
    blocks = [{"type": "text", "text": last["content"]}] if isinstance(last["content"], str) else list(last["content"])
    blocks[-1] = {**blocks[-1], "cache_control": _CACHE}
    return [*messages[:-1], {**last, "content": blocks}]


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


def _run_tool(registry: ToolRegistry, block: dict[str, Any], tracer: Tracer, step: int) -> tuple[dict[str, Any], int]:
    """Run one tool call; returns the tool_result block and how long the tool took (ms)."""
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
    latency_ms = round((time.perf_counter() - began) * 1000)
    tracer.event(
        "tool_call", step=step, tool=block["name"], arguments=block["input"], is_error=is_error,
        result=content, latency_ms=latency_ms,
    )
    return _result(block, content, is_error), latency_ms
