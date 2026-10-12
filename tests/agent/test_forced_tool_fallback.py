"""A model that refuses a forced tool call must still work.

Found on the first live run: claude-sonnet-5-5 answers 400 'tool_choice: type "tool" and "any" are not supported for this model',
so every forced call (the triage, the plan, the review, the judge, the extraction) failed and, in the red team, was scored as
'contained'. Scripted clients never refuse, so only a real call could show it."""

import anthropic
import httpx2
import pytest

from jobshop.agent import providers
from jobshop.agent.loop import AgentConfig
from jobshop.agent.patterns import TRIAGE_SPEC, TRIAGE_TOOL
from jobshop.agent.providers import create_message, forget_unsupported_models
from tests.agent.test_loop import assert_history_is_valid, proposal_script, turn
from tests.agent.test_route import triage_reply
from tests.fake_llm import FakeClient, message, submit, tool

REFUSAL = 'tool_choice: type "tool" and "any" are not supported for this model.'


def refusal(text=REFUSAL):
    response = httpx2.Response(400, request=httpx2.Request("POST", "http://x"))
    return anthropic.BadRequestError(f"Error code: 400 - {text}", response=response, body={"error": {"message": text}})


@pytest.fixture(autouse=True)
def fresh():
    forget_unsupported_models()
    yield
    forget_unsupported_models()


def forced_kwargs(**kw):
    other = {"name": "other", "description": "d", "input_schema": {"type": "object"}}
    return dict(model="m1", max_tokens=100, system="SYS", messages=[{"role": "user", "content": "hi"}],
                tools=[other, TRIAGE_SPEC], tool_choice={"type": "tool", "name": TRIAGE_TOOL}, **kw)


def test_a_model_that_accepts_a_forced_tool_gets_the_request_unchanged():
    client = FakeClient([triage_reply("read")])
    create_message(client, **forced_kwargs())
    [sent] = client.requests
    assert sent["tool_choice"] == {"type": "tool", "name": TRIAGE_TOOL} and sent["system"] == "SYS" and len(sent["tools"]) == 2


def test_a_refusal_is_retried_with_only_the_forced_tool_offered_and_an_instruction_to_call_it():
    client = FakeClient([refusal(), triage_reply("read")])
    reply = create_message(client, **forced_kwargs())
    first, second = client.requests
    assert "tool_choice" in first and "tool_choice" not in second
    assert [t["name"] for t in second["tools"]] == [TRIAGE_TOOL]
    assert second["system"].startswith("SYS") and f"calling the tool '{TRIAGE_TOOL}'" in second["system"]
    assert reply.content[0].name == TRIAGE_TOOL


def test_a_model_known_to_refuse_goes_straight_to_the_fallback_next_time():
    client = FakeClient([refusal(), triage_reply("read"), triage_reply("plan")])
    create_message(client, **forced_kwargs())
    create_message(client, **forced_kwargs())
    assert len(client.requests) == 3 and "tool_choice" not in client.requests[2]


def test_one_model_refusing_does_not_change_how_another_is_called():
    client = FakeClient([refusal(), triage_reply("read"), triage_reply("plan")])
    create_message(client, **forced_kwargs())
    create_message(client, **{**forced_kwargs(), "model": "m2"})
    assert client.requests[2]["tool_choice"] == {"type": "tool", "name": TRIAGE_TOOL}


def test_a_list_system_prompt_gets_the_instruction_as_an_extra_block():
    client = FakeClient([refusal(), triage_reply("read")])
    create_message(client, **{**forced_kwargs(), "system": [{"type": "text", "text": "A"}]})
    assert client.requests[1]["system"][-1]["text"].startswith("You must respond now by calling the tool")


def test_other_bad_requests_are_not_swallowed():
    client = FakeClient([refusal("messages: roles must alternate")])
    with pytest.raises(anthropic.BadRequestError):
        create_message(client, **forced_kwargs())
    assert len(client.requests) == 1


def test_a_request_with_no_forced_tool_is_never_touched():
    client = FakeClient([submit(summary="x")])
    create_message(client, model="m1", max_tokens=5, system="s", messages=[], tools=[TRIAGE_SPEC])
    assert "tool_choice" not in client.requests[0] and client.requests[0]["system"] == "s"


def test_an_unrelated_api_error_on_a_forced_call_is_raised():
    client = FakeClient([anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))])
    with pytest.raises(anthropic.APIConnectionError):
        create_message(client, **forced_kwargs())


# -- through the real loop ------------------------------------------------------------------------------------------------------------------


def test_the_route_pattern_works_on_a_model_that_refuses_forced_tools(ctx, registry):
    script = [refusal(), triage_reply("plan"), *proposal_script()]
    result, client, messages = turn(ctx, registry, script, config=AgentConfig(model="m1", pattern="route"))
    assert result.status == "answered" and result.route == "plan" and result.final.needs_approval is True
    triage_requests = [r for r in client.requests if TRIAGE_TOOL in [t["name"] for t in r["tools"]]]
    assert len(triage_requests) == 2 and "tool_choice" in triage_requests[0] and "tool_choice" not in triage_requests[1]
    assert_history_is_valid(messages)


def test_a_model_that_ignores_the_instruction_and_talks_still_gets_the_full_agent(ctx, registry):
    script = [refusal(), message(tool("get_schedule", "g")), submit(summary="All fine.")]   # prose-ish: no triage call at all
    result, _, _ = turn(ctx, registry, script, config=AgentConfig(model="m1", pattern="route"))
    assert result.route == "plan" and result.status == "answered"


def test_the_judge_and_the_extractor_use_the_same_fallback():
    import inspect

    from jobshop.evals import judge
    from jobshop.extraction import llm

    assert "create_message(" in inspect.getsource(judge.judge_run) and "create_message(" in inspect.getsource(llm.extract_with_llm)
    assert providers.create_message is create_message
