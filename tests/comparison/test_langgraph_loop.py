"""The LangGraph port of the loop (comparison only). Skipped where the optional extra is not installed."""

import anthropic
import httpx2
import pytest

pytest.importorskip("langgraph")

from jobshop.agent.loop import AgentConfig, run_turn  # noqa: E402
from jobshop.agent.prompts import build_system_prompt  # noqa: E402
from jobshop.comparison.langgraph_loop import run_turn_langgraph  # noqa: E402
from jobshop.evals.oracle import OracleClient  # noqa: E402
from jobshop.evals.runner import run_suite  # noqa: E402
from tests.agent.test_loop import assert_history_is_valid, proposal_script  # noqa: E402
from tests.fake_llm import FakeClient, message, submit, text, tool  # noqa: E402
from tests.helpers import FAST  # noqa: E402
from tests.evals.conftest import kb, scenarios, shops  # noqa: E402,F401  (fixtures: the shipped shops, scenarios and documents)


def play(loop, ctx, registry, script, config=None, user="Make O-101 urgent."):
    messages: list = []
    result = loop(FakeClient(script), registry, build_system_prompt(ctx), messages, user, config or AgentConfig(model="m"))
    return result, messages


def test_the_port_answers_a_proposal_exactly_as_the_handwritten_loop_does(ctx, registry):
    a, messages = play(run_turn_langgraph, ctx, registry, proposal_script())
    assert a.status == "answered" and a.final.needs_approval and a.final.draft_id == "D1" and a.steps == 4
    assert a.final.changes_made and a.final.kpi_after is not None
    assert_history_is_valid(messages)


def test_both_loops_give_the_same_answer_on_every_scenario_with_the_reference_agent(scenarios, shops, kb):
    config = AgentConfig(model="oracle")
    mine = run_suite(list(scenarios.values()), OracleClient, config, shops, FAST, knowledge=kb, loop=run_turn)
    theirs = run_suite(list(scenarios.values()), OracleClient, config, shops, FAST, knowledge=kb, loop=run_turn_langgraph)
    assert all(r.passed for r in mine) and all(r.passed for r in theirs)
    key = lambda r: (r.id, r.status, r.steps, r.tools_called, r.answer["summary"] if r.answer else None,   # noqa: E731
                     r.answer["changes_made"] if r.answer else None, r.answer["needs_approval"] if r.answer else None)
    assert [key(r) for r in theirs] == [key(r) for r in mine]


def test_prose_without_a_tool_call_is_nudged_then_stopped_like_the_original(ctx, registry):
    script = [message(text("I think it is fine."))] * 6
    config = AgentConfig(model="m", max_nudges=2)
    a, _ = play(run_turn_langgraph, ctx, registry, script, config)
    b, _ = play(run_turn, ctx, registry, script, config)
    assert (a.status, a.steps) == (b.status, b.steps) == ("no_final_response", 3)


def test_the_step_limit_ends_the_turn(ctx, registry):
    script = [message(tool("get_schedule", f"g{i}")) for i in range(10)]
    a, _ = play(run_turn_langgraph, ctx, registry, script, AgentConfig(model="m", max_steps=3))
    assert a.status == "step_limit" and a.steps == 3 and a.final is None


def test_an_api_failure_ends_the_turn_and_leaves_the_history_alone(ctx, registry):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    a, messages = play(run_turn_langgraph, ctx, registry, [error])
    assert a.status == "api_error" and "APIConnectionError" in a.text and messages == []


def test_an_invalid_submit_is_sent_back_and_a_valid_one_then_ends_the_turn(ctx, registry):
    script = [message(tool("submit_response", "s1", summary="")), submit(summary="Fine.")]
    a, messages = play(run_turn_langgraph, ctx, registry, script)
    assert a.status == "answered" and a.final.summary == "Fine." and a.steps == 2
    assert_history_is_valid(messages)


def test_a_model_cannot_use_a_tool_it_was_not_offered(ctx, registry):
    script = [message(tool("commit_schedule", "c1", draft_id="D1", approval_token="x")), submit(summary="Done.")]
    a, messages = play(run_turn_langgraph, ctx, registry, script)
    results = [b for m in messages if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]
    assert results[0]["is_error"] and "unknown tool" in results[0]["content"] and ctx.store.committed.version == 1


def test_the_answer_guards_apply_in_the_port_too(ctx, registry):
    from jobshop.agent.claims import LIVE_CLAIM_WARNING

    a, _ = play(run_turn_langgraph, ctx, registry, [submit(summary="The schedule is live.")])
    assert LIVE_CLAIM_WARNING in a.final.warnings
