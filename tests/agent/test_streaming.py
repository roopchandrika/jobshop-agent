"""Streaming the answer as the model writes it. Tested against SDK-shaped stream events from a scripted client."""

import anthropic
import httpx2
import pytest

from jobshop.agent.loop import AgentConfig, run_turn
from jobshop.agent.prompts import build_system_prompt
from jobshop.agent.streaming import can_stream, stream_message
from tests.agent.test_loop import assert_history_is_valid, proposal_script
from tests.fake_llm import FakeClient, StreamingFakeClient, message, submit, text, tool

ANSWER = "O-101 goes first and nothing else is affected, because it has slack."


def collect():
    events: list[tuple[str, str]] = []
    return events, lambda kind, chunk: events.append((kind, chunk))


def test_the_answer_arrives_in_pieces_that_add_up_to_the_summary():
    client = StreamingFakeClient([submit(summary=ANSWER)], chunk=9)
    events, on_delta = collect()
    final = stream_message(client, on_delta, model="m", max_tokens=10, system="s", tools=[], messages=[])
    assert events[0] == ("reset", "")
    pieces = [t for k, t in events if k == "answer"]
    assert len(pieces) > 3 and "".join(pieces) == ANSWER
    assert final.content[0].input["summary"] == ANSWER                         # the complete message is what the loop uses


def test_ordinary_text_is_passed_on_as_text_and_other_tools_stream_no_answer():
    client = StreamingFakeClient([message(text("Let me look."), tool("get_schedule", "g"))])
    events, on_delta = collect()
    stream_message(client, on_delta, model="m", max_tokens=10, system="s", tools=[], messages=[])
    assert events == [("reset", ""), ("text", "Let me look.")]


def test_a_summary_that_is_rewritten_never_emits_a_broken_continuation():
    from types import SimpleNamespace

    class Rewriting:
        messages = SimpleNamespace()

        def stream(self, **kw):
            events = [SimpleNamespace(type="input_json", snapshot={"summary": "Draf"}),
                      SimpleNamespace(type="input_json", snapshot={"summary": "Different start"})]
            return type("S", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None, "__iter__": lambda s: iter(events),
                                  "get_final_message": lambda s: submit(summary="Different start")})()

    client = Rewriting()
    client.messages = client
    events, on_delta = collect()
    stream_message(client, on_delta, model="m", max_tokens=1, system="", tools=[], messages=[])
    assert [t for k, t in events if k == "answer"] == ["Draf"]           # the rewritten text does not extend what was shown


def test_a_client_that_cannot_stream_is_detected():
    assert can_stream(StreamingFakeClient([])) and not can_stream(FakeClient([]))
    assert not can_stream(object())


# -- through the loop ---------------------------------------------------------------------------------------------------------------------


def run(ctx, registry, client, **kw):
    messages: list = []
    result = run_turn(client, registry, build_system_prompt(ctx), messages, "Make O-101 urgent.", AgentConfig(model="m"), **kw)
    return result, messages


def test_the_loop_streams_unforced_calls_and_the_result_is_the_same_as_without_streaming(ctx, registry):
    events, on_delta = collect()
    streaming, _ = run(ctx, registry, StreamingFakeClient(proposal_script()), on_delta=on_delta)
    plain, _ = run(ctx, registry, FakeClient(proposal_script()))
    assert streaming.status == plain.status == "answered"
    assert streaming.final.summary == plain.final.summary and streaming.steps == plain.steps
    shown = "".join(t for k, t in events if k == "answer")
    assert shown == "O-101 is now first in line." == streaming.final.summary
    assert [k for k, _ in events].count("reset") == streaming.steps          # one reset per streamed call


def test_without_a_callback_nothing_streams(ctx, registry):
    client = StreamingFakeClient(proposal_script())
    result, _ = run(ctx, registry, client)
    assert result.status == "answered" and client.streamed == 0


def test_forced_internal_calls_are_never_streamed(ctx, registry):
    from tests.agent.test_route import triage_reply

    client = StreamingFakeClient([triage_reply("plan"), *proposal_script()])
    events, on_delta = collect()
    messages: list = []
    result = run_turn(client, registry, build_system_prompt(ctx), messages, "Make O-101 urgent.", AgentConfig(model="m", pattern="route"),
                      on_delta=on_delta)
    assert result.route == "plan" and client.streamed == result.steps - 1       # every call but the forced triage


def test_a_bounced_answer_is_reset_so_the_planner_never_sees_two_answers_joined(ctx, registry):
    script = [message(tool("create_draft", "c")), message(tool("change_priority", "e", draft_id="D1", order_id="O-101", priority=5),
                                                           tool("reschedule", "r", draft_id="D1")),
              message(tool("compare_schedules", "k", after="D1")),
              submit(summary="It will save 4817 minutes overall.", draft_id="D1"),     # a figure no tool returned
              submit(summary="O-101 is now first in line.", draft_id="D1")]
    events, on_delta = collect()
    messages: list = []
    result = run_turn(StreamingFakeClient(script), registry, build_system_prompt(ctx), messages, "Make O-101 urgent.",
                      AgentConfig(model="m", pattern="verify"), on_delta=on_delta)
    assert result.final.summary == "O-101 is now first in line."
    since_reset = []
    for kind, chunk in events:
        since_reset = [] if kind == "reset" else since_reset + ([chunk] if kind == "answer" else [])
    assert "".join(since_reset) == "O-101 is now first in line."                 # only the last answer is on show


def test_a_stream_that_fails_part_way_ends_the_turn_cleanly_and_leaves_history_alone(ctx, registry):
    from types import SimpleNamespace

    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))

    class Breaking:
        def __init__(self):
            self.messages = self

        def stream(self, **kw):
            def events():
                yield SimpleNamespace(type="input_json", snapshot={"summary": "Partial"})
                raise error
            return type("S", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None, "__iter__": lambda s: events()})()

    events, on_delta = collect()
    result, messages = run(ctx, registry, Breaking(), on_delta=on_delta)
    assert result.status == "api_error" and messages == [] and ("answer", "Partial") in events
