import json

import anthropic
import httpx2
import pytest

from jobshop.agent.loop import SUBMIT, AgentConfig, run_turn
from jobshop.agent.prompts import build_system_prompt
from jobshop.agent.trace import Tracer
from jobshop.core.kpis import compute_kpis
from jobshop.tools import views
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import CreateDraftInput
from jobshop.tools.registry import Tool, ToolRegistry
from tests.fake_llm import FakeClient, last_text, last_tool_results, message, submit, text, tool

CONFIG = AgentConfig(model="fake-model")


def turn(ctx, registry, script, user="Make O-101 urgent.", messages=None, config=CONFIG, tracer=None):
    client = FakeClient(script)
    messages = [] if messages is None else messages
    result = run_turn(client, registry, build_system_prompt(ctx), messages, user, config, tracer)
    return result, client, messages


def proposal_script():
    """A model that behaves well: draft, edit + solve, compare, answer."""
    return [
        message(tool("create_draft", "t1")),
        message(
            tool("change_priority", "t2", draft_id="D1", order_id="O-101", priority=5),
            tool("reschedule", "t3", draft_id="D1"),
        ),
        message(tool("compare_schedules", "t4", after="D1")),
        submit(summary="O-101 is now first in line.", draft_id="D1"),
    ]


def assert_history_is_valid(messages):
    """Roles alternate starting with user, and every tool_use is answered in the next message."""
    assert messages[0]["role"] == "user"
    for earlier, later in zip(messages, messages[1:]):
        assert earlier["role"] != later["role"], "consecutive messages with the same role"
        if earlier["role"] == "assistant" and isinstance(earlier["content"], list):
            wanted = {b["id"] for b in earlier["content"] if b["type"] == "tool_use"}
            if wanted:  # a plain-text user message (nudge or new request) has nothing to answer
                assert isinstance(later["content"], list)
                answered = {b["tool_use_id"] for b in later["content"] if b["type"] == "tool_result"}
                assert wanted == answered
    assert messages[-1]["role"] == "assistant"


# --- the happy path -------------------------------------------------------------------------


def test_full_turn_produces_an_answer_with_harness_computed_numbers(ctx, registry):
    result, client, messages = turn(ctx, registry, proposal_script())

    assert result.status == "answered" and result.steps == 4
    final = result.final
    assert final.summary == "O-101 is now first in line." and final.draft_id == "D1"

    # KPIs and needs_approval come from the store, not from anything the model typed.
    draft, committed = ctx.store.draft("D1"), ctx.store.committed
    assert final.kpi_before == views.kpi_view(committed.instance, compute_kpis(committed.instance, committed.schedule))
    assert final.kpi_after == views.kpi_view(draft.instance, compute_kpis(draft.instance, draft.schedule))
    assert final.needs_approval is True and final.warnings == []

    assert result.input_tokens == 400 and result.output_tokens == 200
    assert_history_is_valid(messages)
    assert messages[-1]["content"][0]["text"] == "O-101 is now first in line. (Proposal is in draft D1.)"


def test_parallel_tool_calls_are_all_answered_in_order_in_one_message(ctx, registry):
    _, client, _ = turn(ctx, registry, proposal_script())
    results = last_tool_results(client.requests[2])  # request made after the two-tool message
    assert [r["tool_use_id"] for r in results] == ["t2", "t3"]
    assert json.loads(results[1]["content"])["feasible"] is True


def test_model_sees_visible_tools_plus_submit_but_never_commit(ctx, registry):
    _, client, _ = turn(ctx, registry, [submit()])
    request = client.requests[0]
    names = [t["name"] for t in request["tools"]]
    assert SUBMIT in names and "commit_schedule" not in names and "reschedule" in names
    assert request["model"] == "fake-model" and request["max_tokens"] == CONFIG.max_output_tokens
    assert "Plant clock: 2026-01-05 06:00" in request["system"]


def test_sdk_blocks_are_echoed_back_as_plain_minimal_dicts(ctx, registry):
    _, client, messages = turn(ctx, registry, [message(text("Let me look."), tool("get_schedule", "t1")), submit()])
    assistant = messages[1]["content"]
    assert assistant[0] == {"type": "text", "text": "Let me look."}
    assert set(assistant[1]) == {"type", "id", "name", "input"}  # no SDK-only fields like caller
    json.dumps(messages)  # fully serializable


# --- errors go back to the model, not up the stack ------------------------------------------


def test_tool_errors_become_error_results_and_the_loop_continues(ctx, registry):
    result, client, _ = turn(ctx, registry, [message(tool("get_schedule", "t1", order_id="NOPE")), submit()])
    (shown,) = last_tool_results(client.requests[1])
    assert shown["is_error"] is True and "unknown order" in json.loads(shown["content"])["error"]
    assert result.status == "answered"


def test_invalid_arguments_are_reported_back(ctx, registry):
    _, client, _ = turn(ctx, registry, [message(tool("change_priority", "t1", draft_id="D1")), submit()])
    (shown,) = last_tool_results(client.requests[1])
    assert shown["is_error"] and "invalid arguments for change_priority" in json.loads(shown["content"])["error"]


def test_a_crashing_tool_is_contained_and_traced(ctx, tmp_path):
    boom = Tool("boom", "always fails", CreateDraftInput, lambda c, a: 1 / 0)
    reg = ToolRegistry(ctx, tools=[boom])
    tracer = Tracer(tmp_path / "t.jsonl")
    result, client, _ = turn(ctx, reg, [message(tool("boom", "t1")), submit()], tracer=tracer)
    (shown,) = last_tool_results(client.requests[1])
    assert shown["is_error"] and "internal error in boom (ZeroDivisionError)" in json.loads(shown["content"])["error"]
    assert result.status == "answered"
    events = [json.loads(line)["event"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "tool_exception" in events


def test_the_model_cannot_commit_even_by_name(ctx, registry):
    version = ctx.store.committed.version
    script = [
        message(tool("create_draft", "t1")),
        message(tool("commit_schedule", "t2", draft_id="D1", approval_token="forged")),
        submit(),
    ]
    result, client, _ = turn(ctx, registry, script)
    (shown,) = last_tool_results(client.requests[2])
    assert shown["is_error"] and "unknown tool 'commit_schedule'" in json.loads(shown["content"])["error"]
    assert ctx.store.committed.version == version


def test_truncated_tool_calls_are_not_run(ctx, registry):
    script = [message(tool("create_draft", "t1"), stop_reason="max_tokens"), submit()]
    _, client, _ = turn(ctx, registry, script)
    (shown,) = last_tool_results(client.requests[1])
    assert shown["is_error"] and "cut off" in shown["content"]
    with pytest.raises(ToolError):
        ctx.store.draft("D1")  # the draft was never created


# --- submit_response handling ---------------------------------------------------------------


def test_submit_must_stand_alone(ctx, registry):
    script = [message(tool("create_draft", "t1"), tool(SUBMIT, "t2", summary="too early")), submit(summary="Now.")]
    result, client, _ = turn(ctx, registry, script)
    first, second = last_tool_results(client.requests[1])
    assert "draft_id" in first["content"]  # create_draft did run
    assert second["is_error"] and "on its own" in second["content"]
    assert result.status == "answered" and result.final.summary == "Now."


def test_invalid_submit_payloads_are_rejected_and_retried(ctx, registry):
    bad_extra = message(tool(SUBMIT, "t1", summary="x", needs_approval=True))  # the model may not set this
    empty = message(tool(SUBMIT, "t2", summary=""))
    result, client, _ = turn(ctx, registry, [bad_extra, empty, submit(summary="Fine.")])
    assert "Extra inputs are not permitted" in last_tool_results(client.requests[1])[0]["content"]
    assert last_tool_results(client.requests[2])[0]["is_error"]
    assert result.status == "answered" and result.final.needs_approval is False


def test_a_text_only_reply_is_nudged_to_submit(ctx, registry):
    result, client, messages = turn(ctx, registry, [message(text("Sure!")), submit(summary="Real answer.")])
    assert "calling submit_response" in last_text(client.requests[1])
    assert result.final.summary == "Real answer."
    assert_history_is_valid(messages)


def test_giving_up_after_the_nudges_keeps_history_valid(ctx, registry):
    result, _, messages = turn(ctx, registry, [message(text(f"chat {i}")) for i in range(3)])
    assert result.status == "no_final_response" and result.text == "chat 2"
    assert_history_is_valid(messages)


def test_clarifying_question_makes_no_proposal(ctx, registry):
    result, _, _ = turn(ctx, registry, [submit(summary="Need more info.", clarifying_question="Which machine do you mean?")])
    assert result.final.clarifying_question == "Which machine do you mean?"
    assert (result.final.kpi_before, result.final.kpi_after, result.final.needs_approval) == (None, None, False)


def test_the_models_claims_about_drafts_are_checked_against_the_store(ctx, registry):
    ghost, _, _ = turn(ctx, registry, [submit(draft_id="D9")])
    assert ghost.final.needs_approval is False and "does not exist" in ghost.final.warnings[0]

    unsolved = [message(tool("create_draft", "t1")), submit(draft_id="D1")]
    result, _, _ = turn(ctx, registry, unsolved)
    assert result.final.needs_approval is False and "no solved schedule" in result.final.warnings[0]


def test_a_stale_draft_cannot_need_approval(ctx, registry):
    script = proposal_script()
    result, _, _ = turn(ctx, registry, script[:3] + [lambda kw: (ctx.store.set_clock(30), submit(draft_id="D1"))[1]])
    assert result.final.needs_approval is False and "stale" in result.final.warnings[0]


# --- limits ---------------------------------------------------------------------------------


def test_step_limit_stops_the_turn_and_leaves_valid_history(ctx, registry):
    looping = [message(tool("get_schedule", f"t{i}")) for i in range(10)]
    result, client, messages = turn(ctx, registry, looping, config=AgentConfig(model="m", max_steps=3))
    assert result.status == "step_limit" and result.steps == 3 and len(client.requests) == 3
    assert_history_is_valid(messages)


def test_token_budget_stops_before_another_model_call(ctx, registry):
    script = [message(tool("get_schedule", "t1"), tokens_in=100, tokens_out=50), submit()]
    result, client, messages = turn(ctx, registry, script, config=AgentConfig(model="m", max_total_tokens=120))
    assert result.status == "budget_exceeded" and len(client.requests) == 1
    assert_history_is_valid(messages)


def test_cost_is_tracked_and_can_stop_the_turn(ctx, registry):
    config = AgentConfig(model="m", price_input_per_mtok=3.0, price_output_per_mtok=15.0, max_cost_usd=0.01)
    script = [message(tool("get_schedule", "t1"), tokens_in=1000, tokens_out=500), submit()]
    result, _, _ = turn(ctx, registry, script, config=config)
    assert result.cost_usd == pytest.approx(0.0105)  # 1000*3/1e6 + 500*15/1e6
    assert result.status == "budget_exceeded"


def test_cost_budget_requires_prices():
    with pytest.raises(ValueError, match="needs both token prices"):
        AgentConfig(model="m", max_cost_usd=1.0)


def test_cost_is_none_without_prices(ctx, registry):
    result, _, _ = turn(ctx, registry, [submit()])
    assert result.cost_usd is None


def test_api_failure_rolls_back_the_turn(ctx, registry):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://example.invalid"))
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
    before = json.dumps(history)
    result, _, messages = turn(ctx, registry, [error], messages=history)
    assert result.status == "api_error" and "APIConnectionError" in result.text
    assert json.dumps(messages) == before  # as if the failed request never happened


# --- conversation and tracing ---------------------------------------------------------------


def test_a_second_turn_continues_the_same_conversation(ctx, registry):
    _, _, messages = turn(ctx, registry, proposal_script())
    result, client, messages = turn(ctx, registry, [submit(summary="Follow-up answer.")], user="And why?", messages=messages)
    assert result.final.summary == "Follow-up answer."
    sent = client.requests[0]["messages"]
    assert sent[0]["content"] == "Make O-101 urgent." and "Proposal is in draft D1" in sent[-2]["content"][0]["text"]
    assert_history_is_valid(messages)


def test_trace_records_every_step_with_tokens_latency_and_results(ctx, registry, tmp_path):
    path = tmp_path / "trace.jsonl"
    turn(ctx, registry, proposal_script(), tracer=Tracer(path, session_id="s1"))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["session"] for r in records} == {"s1"}
    kinds = [r["event"] for r in records]
    assert kinds[0] == "turn_start" and kinds[-1] == "turn_end"
    assert kinds.count("llm_call") == 4 and kinds.count("tool_call") == 4  # submit is not a tool_call
    llm = next(r for r in records if r["event"] == "llm_call")
    assert llm["input_tokens"] == 100 and llm["output_tokens"] == 50 and "latency_ms" in llm
    reschedule = next(r for r in records if r["event"] == "tool_call" and r["tool"] == "reschedule")
    assert reschedule["arguments"] == {"draft_id": "D1"} and reschedule["result"]["feasible"] is True
    assert records[-1]["status"] == "answered"


def test_a_solved_draft_with_no_changes_has_nothing_to_approve(ctx, registry):
    script = [message(tool("create_draft", "t1")), message(tool("reschedule", "t2", draft_id="D1")), submit(draft_id="D1")]
    result, _, _ = turn(ctx, registry, script)
    assert result.final.kpi_after is not None  # it did solve
    assert result.final.needs_approval is False  # but there is no proposal to commit
