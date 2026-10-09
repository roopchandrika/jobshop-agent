"""Short-term memory: a long conversation is kept inside its budget by clearing old tool results, then old turns."""

import copy
import json

import pytest

from jobshop.agent.history import (
    NUDGE, STUB_MARK, clear_old_results, drop_oldest_turns, estimate_tokens, turn_starts,
)
from jobshop.agent.loop import AgentConfig
from jobshop.agent.trace import Tracer
from jobshop.agent.conversation import Conversation
from tests.agent.test_loop import assert_history_is_valid
from tests.fake_llm import message, submit, tool

BIG = json.dumps({"rows": ["x" * 40] * 200})                                  # a few thousand tokens of tool output


def one_turn(n, big=BIG, error=False):
    """A planner message, one tool call with a large result, and the assistant's closing text."""
    return [
        {"role": "user", "content": f"request {n}"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{n}", "name": "get_schedule", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": big, **({"is_error": True} if error else {})}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {n}"}]},
    ]


def conversation(turns, **kw):
    return [m for n in range(1, turns + 1) for m in one_turn(n, **kw)]


# -- finding turns ---------------------------------------------------------------------------------------------------------


def test_a_turn_starts_at_each_planner_message_not_at_tool_results_or_the_loops_nudge():
    messages = conversation(2)
    messages[3:3] = [{"role": "user", "content": NUDGE + " submit_response; it is the only way..."}]   # inside turn 1
    assert [messages[i]["content"] for i in turn_starts(messages)] == ["request 1", "request 2"]


def test_the_size_estimate_grows_with_the_conversation():
    assert estimate_tokens(conversation(1)) < estimate_tokens(conversation(3)) and estimate_tokens(conversation(3)) > 1000


# -- clearing old results ---------------------------------------------------------------------------------------------------------


def test_only_results_older_than_the_last_turns_are_replaced_and_the_rest_is_untouched():
    messages = conversation(5)
    cleared, count = clear_old_results(messages, keep_turns=2)
    assert count == 3
    for n in (1, 2, 3):
        [block] = cleared[(n - 1) * 4 + 2]["content"]
        assert STUB_MARK in block["content"] and "get_schedule" in block["content"] and "call the tool again" in block["content"]
    assert cleared[-8:] == messages[-8:]                                       # the last two turns, byte for byte
    assert estimate_tokens(cleared) < estimate_tokens(messages) / 2


def test_the_words_of_the_conversation_and_every_tool_call_stay_so_it_remains_valid():
    cleared, _ = clear_old_results(conversation(4), keep_turns=1)
    assert_history_is_valid(cleared)
    assert [m["content"] for m in cleared if m["role"] == "user" and isinstance(m["content"], str)] == [f"request {n}" for n in range(1, 5)]
    uses = {b["id"] for m in cleared if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"}
    results = {b["tool_use_id"] for m in cleared if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result"}
    assert uses == results


def test_the_original_list_is_not_modified_and_clearing_twice_changes_nothing_more():
    messages = conversation(4)
    before = copy.deepcopy(messages)
    once, count = clear_old_results(messages, 1)
    assert messages == before
    again, more = clear_old_results(once, 1)
    assert more == 0 and again == once


def test_an_error_result_stays_marked_as_an_error():
    cleared, _ = clear_old_results(conversation(3, error=True), 1)
    assert cleared[2]["content"][0]["is_error"] is True and STUB_MARK in cleared[2]["content"][0]["content"]


def test_nothing_is_cleared_when_there_are_not_more_turns_than_are_protected():
    messages = conversation(2)
    assert clear_old_results(messages, keep_turns=2) == (messages, 0) and clear_old_results(messages, keep_turns=5)[1] == 0


def test_protecting_no_turns_clears_every_result():
    assert clear_old_results(conversation(3), keep_turns=0)[1] == 3


# -- dropping old turns ---------------------------------------------------------------------------------------------------------------


def test_whole_oldest_turns_are_dropped_until_the_conversation_fits():
    messages = conversation(6)
    limit = estimate_tokens(conversation(3)) + 10
    kept, dropped = drop_oldest_turns(messages, keep_turns=2, limit_tokens=limit)
    assert dropped == 3 and estimate_tokens(kept) <= limit
    assert kept[0] == {"role": "user", "content": "request 4"} and kept == messages[12:]
    assert_history_is_valid(kept)


def test_the_protected_turns_are_never_dropped_even_if_that_leaves_it_over_the_limit():
    kept, dropped = drop_oldest_turns(conversation(4), keep_turns=2, limit_tokens=1)
    assert dropped == 2 and [m["content"] for m in kept if isinstance(m["content"], str)] == ["request 3", "request 4"]


def test_a_conversation_that_already_fits_is_left_alone():
    messages = conversation(3)
    assert drop_oldest_turns(messages, 1, 10**9) == (messages, 0)


# -- in a real conversation -------------------------------------------------------------------------------------------------------------------


def chat(ctx, tmp_path, limit, turns, keep=2):
    script = []
    for _ in range(turns):
        script += [message(tool("get_schedule", f"s{len(script)}")), submit(summary="ok")]
    from tests.fake_llm import FakeClient
    client = FakeClient(script)
    config = AgentConfig(model="fake", history_token_limit=limit, history_keep_turns=keep)
    convo = Conversation(client, ctx, config, Tracer(tmp_path / "t.jsonl", session_id="s"))
    return convo, client


def stubbed_results(messages):
    """Tool results that were replaced by a stub (the notice to the model also contains the words, so look only here)."""
    return [b for m in messages if isinstance(m["content"], list) for b in m["content"]
            if b.get("type") == "tool_result" and STUB_MARK in str(b["content"])[:200]]


def trace_events(tmp_path, name):
    return [r for r in map(json.loads, (tmp_path / "t.jsonl").read_text().splitlines()) if r["event"] == name]


def test_a_long_conversation_has_its_old_results_cleared_before_the_next_message_and_says_so_in_the_trace(ctx, tmp_path):
    convo, client = chat(ctx, tmp_path, limit=2500, turns=6)
    for i in range(6):
        convo.say(f"question {i}")
    [first, *_] = trace_events(tmp_path, "history_compacted")
    assert first["results_cleared"] >= 1 and first["after_tokens"] < first["before_tokens"]
    sent = client.requests[-1]["messages"]
    assert stubbed_results(sent[:6]) and "get_schedule" in stubbed_results(sent[:6])[0]["content"]
    assert_history_is_valid(convo.messages)


def test_with_no_limit_nothing_is_ever_touched(ctx, tmp_path):
    convo, _ = chat(ctx, tmp_path, limit=None, turns=4)
    for i in range(4):
        convo.say(f"question {i}")
    assert not stubbed_results(convo.messages) and not (tmp_path / "t.jsonl").read_text().count("history_compacted")


def test_below_the_limit_nothing_happens(ctx, tmp_path):
    convo, _ = chat(ctx, tmp_path, limit=10**9, turns=3)
    for i in range(3):
        convo.say(f"question {i}")
    assert not stubbed_results(convo.messages)


def test_if_clearing_is_not_enough_old_turns_go_and_the_model_is_told(ctx, tmp_path):
    convo, client = chat(ctx, tmp_path, limit=1200, turns=8)
    for i in range(8):
        convo.say(f"question {i}")
    dropped_events = [e for e in trace_events(tmp_path, "history_compacted") if e["turns_dropped"]]
    assert dropped_events
    last_user_texts = [m["content"] for m in client.requests[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], str)]
    assert any("were removed to save space" in t for t in " ".join(last_user_texts).split("\n\n")) or any(
        "removed to save space" in t for t in last_user_texts)
    assert_history_is_valid(convo.messages)


def test_the_latest_turns_are_always_kept_whole(ctx, tmp_path):
    convo, client = chat(ctx, tmp_path, limit=1, turns=5, keep=2)
    for i in range(5):
        convo.say(f"question {i}")
    sent = client.requests[-1]["messages"]
    assert not stubbed_results(sent[-6:])                                      # the turn in progress and the one before it


def test_the_limit_can_be_set_from_the_environment():
    from jobshop.agent.cli import agent_config_from_env
    assert agent_config_from_env({"ANTHROPIC_MODEL": "m", "JOBSHOP_HISTORY_TOKENS": "1234"}).history_token_limit == 1234
    assert agent_config_from_env({"ANTHROPIC_MODEL": "m"}).history_token_limit == 40_000


def test_even_when_no_turn_is_protected_the_last_turn_is_never_dropped_and_nothing_crashes():
    kept, dropped = drop_oldest_turns(conversation(3), keep_turns=0, limit_tokens=1)
    assert dropped == 2 and [m["content"] for m in kept if isinstance(m["content"], str)] == ["request 3"]
    assert_history_is_valid(kept)


def test_exactly_at_the_limit_nothing_is_compacted(ctx, tmp_path):
    convo, _ = chat(ctx, tmp_path, limit=None, turns=4)
    for i in range(3):
        convo.say(f"question {i}")
    convo.config = AgentConfig(model="fake", history_token_limit=estimate_tokens(convo.messages), history_keep_turns=1)
    convo.client.script += [message(tool("get_schedule", "z0")), submit(summary="ok")]
    convo._tidy_history()
    assert not stubbed_results(convo.messages)                                   # equal to the limit is not over it
    convo.config = AgentConfig(model="fake", history_token_limit=estimate_tokens(convo.messages) - 1, history_keep_turns=1)
    convo._tidy_history()
    assert stubbed_results(convo.messages)                                       # one token over is


def test_compaction_updates_the_shared_list_in_place_so_other_holders_of_it_see_it(ctx, tmp_path):
    convo, _ = chat(ctx, tmp_path, limit=2500, turns=6)
    held = convo.messages                                                        # the chat CLI and the web app keep this reference
    for i in range(6):
        convo.say(f"question {i}")
    assert convo.messages is held and stubbed_results(held)


def test_a_conversation_over_the_limit_with_every_turn_protected_changes_nothing_and_says_nothing(ctx, tmp_path):
    convo, _ = chat(ctx, tmp_path, limit=1, turns=2, keep=2)
    convo.say("one")
    convo.say("two")
    assert not stubbed_results(convo.messages) and not (tmp_path / "t.jsonl").read_text().count("history_compacted")
    assert not convo._notices
