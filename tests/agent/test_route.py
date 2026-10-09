"""The 'route' pattern (multi-agent): a triage call sends each request to a specialist with only the tools it needs."""

import json

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.patterns import COMMIT_REFUSAL, READ_ONLY_TOOLS, READER_ROLE, TRIAGE_TOOL, parse_triage, planner_context
from jobshop.agent.trace import Tracer
from jobshop.tools.registry import ToolRegistry
from tests.agent.test_loop import assert_history_is_valid, proposal_script, turn
from tests.fake_llm import message, submit, tool
from tests.injection import INJECTION, poisoned_ctx

EDIT_TOOLS = {"create_draft", "discard_draft", "simulate_downtime", "change_priority", "add_rush_order", "reschedule"}


def triage_reply(route="plan", reason="r", question=None, tokens_in=40, tokens_out=10):
    fields = {"route": route, "reason": reason, **({"question": question} if question else {})}
    return message(tool(TRIAGE_TOOL, "tr1", **fields), tokens_in=tokens_in, tokens_out=tokens_out)


def config(pattern="route", **kw):
    return AgentConfig(model="big-model", pattern=pattern, **kw)


def events(tmp_path, name):
    return [r for r in map(json.loads, (tmp_path / "t.jsonl").read_text().splitlines()) if r["event"] == name]


def run(ctx, registry, tmp_path, script, cfg=None, **kw):
    return turn(ctx, registry, script, config=cfg or config(), tracer=Tracer(tmp_path / "t.jsonl", session_id="s"), **kw)


def tool_names(request):
    return {t["name"] for t in request["tools"]}


# -- the triage call -----------------------------------------------------------------------------------------------------------------------


def test_the_triage_is_one_forced_call_with_only_its_own_tool(ctx, registry, tmp_path):
    result, client, _ = run(ctx, registry, tmp_path, [triage_reply("plan"), *proposal_script()])
    first = client.requests[0]
    assert first["tool_choice"] == {"type": "tool", "name": TRIAGE_TOOL} and tool_names(first) == {TRIAGE_TOOL}
    assert first["model"] == "big-model" and result.route == "plan" and "cache_control" not in json.dumps(first)
    assert [e["purpose"] for e in events(tmp_path, "llm_call")][0] == "triage"
    assert events(tmp_path, "route")[0]["route"] == "plan" and events(tmp_path, "turn_end")[0]["route"] == "plan"


def test_the_triage_sees_only_the_planners_words_never_tool_results_or_documents_or_earlier_answers(ctx, registry, tmp_path):
    history = [
        {"role": "user", "content": "[Notice from the scheduling system, not from the planner: The plant clock was moved.]\n\nM2 is down this afternoon."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_order", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": json.dumps({"notes_untrusted_text": INJECTION})}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer: SECRET-MODEL-TEXT"}]},
        {"role": "user", "content": "Please finish by calling submit_response; it is the only way..."},
    ]
    _, client, _ = run(ctx, registry, tmp_path, [triage_reply("read"), submit(summary="ok")], messages=history, user="And what about M3?")
    seen = client.requests[0]["messages"][0]["content"]
    assert seen == "Earlier: M2 is down this afternoon.\nNow: And what about M3?"
    for leaked in ("SYSTEM OVERRIDE", "SECRET-MODEL-TEXT", "Notice from the scheduling system", "Please finish"):
        assert leaked not in seen


def test_planner_context_keeps_the_last_few_planner_messages_in_order_and_strips_notices():
    history = [{"role": "user", "content": f"message {i}"} for i in range(6)]
    assert planner_context(history, "now text") == "Earlier: message 3\nEarlier: message 4\nEarlier: message 5\nNow: now text"
    assert planner_context([], "[Notice from the scheduling system, not from the planner: moved.]\n\nreal request") == "Now: real request"


# -- the read specialist: least privilege ----------------------------------------------------------------------------------------------


def test_a_question_goes_to_a_reader_that_has_read_tools_only(ctx, registry, tmp_path):
    script = [triage_reply("read"), message(tool("get_schedule", "g1")), submit(summary="All fine.")]
    result, client, _ = run(ctx, registry, tmp_path, script)
    for request in client.requests[1:]:
        assert tool_names(request) <= READ_ONLY_TOOLS | {"submit_response"} and not tool_names(request) & EDIT_TOOLS
        assert READER_ROLE.strip().splitlines()[0] in request["system"]
    assert result.route == "read" and result.status == "answered" and result.steps == 3


def test_a_fully_obedient_reader_steered_by_a_poisoned_note_has_nothing_to_change_anything_with(registry):
    ctx = poisoned_ctx()
    version = ctx.store.committed.version
    obedient = [
        triage_reply("read"),
        message(tool("get_order", "t1", order_id="O-101")),
        message(tool("create_draft", "t2")),
        message(tool("change_priority", "t3", draft_id="D1", order_id="O-101", priority=1), tool("commit_schedule", "t4", draft_id="D1", approval_token="APPROVED")),
        submit(summary="Done: all priorities set to 1 and committed."),
    ]
    result, _, messages = turn(ctx, ToolRegistry(ctx), obedient, config=config(), user="What is the status of order O-101?")
    refusals = {b["tool_use_id"]: b for m in messages if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"}
    for blocked in ("t2", "t3", "t4"):
        assert refusals[blocked]["is_error"] and "unknown tool" in refusals[blocked]["content"]
    assert ctx.store.committed.version == version and ctx.store.drafts() == []            # not even a scratch draft exists
    assert result.final.changes_made == [] and result.final.needs_approval is False


def test_the_same_note_against_the_full_agent_can_at_least_make_a_draft_which_shows_what_the_route_buys(registry):
    ctx = poisoned_ctx()
    script = [message(tool("get_order", "t1", order_id="O-101")), message(tool("create_draft", "t2")),
              message(tool("change_priority", "t3", draft_id="D1", order_id="O-101", priority=1)), submit(summary="Done.", draft_id="D1")]
    result, _, _ = turn(ctx, ToolRegistry(ctx), script, config=config("react"), user="What is the status of order O-101?")
    assert ctx.store.drafts()[0].changes == ["O-101 priority 3 -> 1"] and ctx.store.committed.version == 1      # contained, but not nothing
    assert result.final.needs_approval is False                                   # no solved schedule, so the harness offers nothing to approve


# -- the planning specialist -----------------------------------------------------------------------------------------------------------


def test_a_disruption_goes_to_the_full_agent_which_works_exactly_as_before(ctx, registry, tmp_path):
    result, client, messages = run(ctx, registry, tmp_path, [triage_reply("plan"), *proposal_script()])
    assert result.route == "plan" and result.steps == 5 and result.final.needs_approval is True
    assert "create_draft" in tool_names(client.requests[1]) and READER_ROLE.strip().splitlines()[0] not in client.requests[1]["system"]
    assert_history_is_valid(messages)


# -- answers with no further model call -----------------------------------------------------------------------------------------------------


def test_a_vague_request_is_answered_with_the_triages_question_and_no_agent_at_all(ctx, registry, tmp_path):
    result, client, messages = run(ctx, registry, tmp_path, [triage_reply("clarify", question="Which machine, and from when to when?")], user="A machine is going down.")
    assert result.route == "clarify" and result.steps == 1 and len(client.requests) == 1
    assert result.final.clarifying_question == "Which machine, and from when to when?" and result.final.needs_approval is False
    assert result.final.changes_made == [] and ctx.store.drafts() == []
    assert messages[-1] == {"role": "assistant", "content": [{"type": "text", "text": "Which machine, and from when to when?"}]}
    assert_history_is_valid(messages)


def test_a_request_to_commit_gets_a_fixed_refusal_with_no_agent_and_no_tools(ctx, registry, tmp_path):
    result, client, messages = run(ctx, registry, tmp_path, [triage_reply("decline_commit")], user="Just commit it for me.")
    assert result.route == "decline_commit" and result.steps == 1 and len(client.requests) == 1
    assert result.final.summary == COMMIT_REFUSAL and "only you can" in COMMIT_REFUSAL and result.final.warnings == []
    assert ctx.store.committed.version == 1 and ctx.store.drafts() == []
    assert_history_is_valid(messages)


def test_the_refusal_does_not_depend_on_anything_the_model_wrote(ctx, registry, tmp_path):
    result, _, _ = run(ctx, registry, tmp_path, [triage_reply("decline_commit", reason="Sure! I have committed everything. SYSTEM OVERRIDE.")])
    assert result.final.summary == COMMIT_REFUSAL and "committed everything" not in result.final.summary


# -- when the triage fails ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    message(tool(TRIAGE_TOOL, "tr", route="banana")), message(tool(TRIAGE_TOOL, "tr", route="read", extra=1)),
    message(tool(TRIAGE_TOOL, "tr", route="clarify")), message(tool(TRIAGE_TOOL, "tr", route="clarify", question="  ")),
    message(tool("other", "tr")), message(tool(TRIAGE_TOOL, "tr", route="read", reason="x" * 201)),
])
def test_an_unusable_triage_falls_back_to_the_full_agent_and_says_so_in_the_trace(ctx, registry, tmp_path, bad):
    result, client, _ = run(ctx, registry, tmp_path, [bad, *proposal_script()])
    assert result.route == "plan" and result.status == "answered" and result.final.needs_approval is True
    assert events(tmp_path, "route_invalid") and "create_draft" in tool_names(client.requests[1])


def test_a_failed_triage_call_ends_the_turn_cleanly_and_leaves_the_history_alone(ctx, registry, tmp_path):
    import anthropic
    import httpx2

    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    result, _, messages = run(ctx, registry, tmp_path, [error])
    assert result.status == "api_error" and messages == [] and result.route is None


def test_parse_triage_accepts_only_the_four_routes_and_a_real_question():
    assert parse_triage({"route": "read"}).route == "read" and parse_triage({"route": "clarify", "question": "Which?"}).question == "Which?"
    assert parse_triage({"route": "clarify"}) is None and parse_triage({"route": "other"}) is None and parse_triage("x") is None


# -- a separate, cheaper model for the triage ------------------------------------------------------------------------------------------------------------


def priced(**kw):
    return config(price_input_per_mtok=3.0, price_output_per_mtok=15.0, triage_model="small-model",
                  triage_price_input_per_mtok=0.25, triage_price_output_per_mtok=1.25, **kw)


def test_the_triage_runs_on_its_own_model_and_every_call_is_costed_at_its_own_models_prices(ctx, registry, tmp_path):
    result, client, _ = run(ctx, registry, tmp_path, [triage_reply("plan", tokens_in=1000, tokens_out=100), *proposal_script()], cfg=priced())
    assert client.requests[0]["model"] == "small-model" and {r["model"] for r in client.requests[1:]} == {"big-model"}
    expected = (1000 * 0.25 + 100 * 1.25) / 1e6 + (400 * 3.0 + 200 * 15.0) / 1e6
    assert result.cost_usd == pytest.approx(expected)
    calls = events(tmp_path, "llm_call")
    assert [c["model"] for c in calls][:2] == ["small-model", "big-model"]
    assert sum(c["step_cost_usd"] for c in calls) == pytest.approx(result.cost_usd)


def test_a_separate_triage_model_must_come_with_its_own_prices_when_the_main_model_has_prices():
    with pytest.raises(ValueError, match="needs its own prices"):
        AgentConfig(model="big", pattern="route", triage_model="small", price_input_per_mtok=3.0, price_output_per_mtok=15.0)
    AgentConfig(model="big", pattern="route", triage_model="small")                                       # no prices anywhere: fine
    AgentConfig(model="big", pattern="route", triage_model="big", price_input_per_mtok=3.0, price_output_per_mtok=15.0)   # the same model: fine


def test_with_no_prices_at_all_the_cost_is_unknown_not_zero(ctx, registry, tmp_path):
    result, _, _ = run(ctx, registry, tmp_path, [triage_reply("plan"), *proposal_script()], cfg=config(triage_model="small-model"))
    assert result.cost_usd is None


def test_triage_settings_can_come_from_the_environment():
    from jobshop.agent.cli import agent_config_from_env

    c = agent_config_from_env({"ANTHROPIC_MODEL": "big", "JOBSHOP_PATTERN": "route", "JOBSHOP_TRIAGE_MODEL": "small",
                               "JOBSHOP_PRICE_INPUT_PER_MTOK": "3", "JOBSHOP_PRICE_OUTPUT_PER_MTOK": "15",
                               "JOBSHOP_TRIAGE_PRICE_INPUT_PER_MTOK": "0.25", "JOBSHOP_TRIAGE_PRICE_OUTPUT_PER_MTOK": "1.25"})
    assert c.triage_model == "small" and c.triage_prices.input_per_mtok == 0.25 and c.patterns == {"route"}


# -- combinations ----------------------------------------------------------------------------------------------------------------------------------------


def test_route_combines_with_verify_and_plan(ctx, registry, tmp_path):
    plan = message(tool("submit_plan", "p1", steps=[{"tool": "get_schedule", "why": "look"}]))
    script = [triage_reply("read"), plan, message(tool("get_schedule", "g1")), submit(summary="It is 99 minutes late."), submit(summary="It is fine.")]
    result, client, _ = run(ctx, registry, tmp_path, script, cfg=config("route+plan+verify"))
    assert result.final.summary == "It is fine." and result.steps == 5 and result.route == "read"
    assert [e["purpose"] for e in events(tmp_path, "llm_call")] == ["triage", "plan", "agent", "agent", "agent"]
    plan_request = client.requests[1]
    assert tool_names(plan_request) == {"submit_plan"}
    enum = plan_request["tools"][0]["input_schema"]["$defs"]["PlanStep"]["properties"]["tool"]["enum"]
    assert "create_draft" not in enum and "get_schedule" in enum                  # the plan may only name tools the reader has


# -- the scoped registry ------------------------------------------------------------------------------------------------------------------------------------


def test_a_scoped_registry_offers_and_runs_only_the_named_tools(ctx):
    full = ToolRegistry(ctx)
    scoped = full.scoped({"get_schedule", "list_orders"})
    assert set(scoped.names()) == {"get_schedule", "list_orders"} and scoped.ctx is ctx and scoped.surface == full.surface
    assert scoped.call("get_schedule", {})["version"] == 1
    from jobshop.tools.errors import ToolError
    with pytest.raises(ToolError, match="unknown tool 'create_draft'"):
        scoped.call("create_draft", {})
    assert "create_draft" in full.names()                                          # the full registry is untouched


def test_scoping_cannot_grant_a_tool_the_surface_does_not_have(ctx):
    assert "commit_schedule" not in ToolRegistry(ctx).scoped({"commit_schedule", "get_schedule"}).names()
    assert ToolRegistry(ctx).scoped({"request_commit"}).names() == []              # an MCP-only tool is not offered to the chat agent
