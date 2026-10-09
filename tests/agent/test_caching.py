"""Prompt caching: the request carries cache breakpoints, the stored history does not, and the
cost, token budget and reports account for cached tokens (which the API reports separately)."""

import copy
import json

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.pricing import Prices
from jobshop.agent.trace import Tracer
from jobshop.agent.trace_report import read_trace, render, summarize
from tests.agent.test_loop import proposal_script, turn
from tests.fake_llm import message, submit, tool

MARK = {"type": "ephemeral"}


def marked_blocks(request):
    """Every place in a request that carries a cache marker."""
    found = [("tool", t["name"]) for t in request["tools"] if "cache_control" in t]
    for m in request["messages"]:
        if isinstance(m["content"], list):
            found += [("message", b["type"]) for b in m["content"] if "cache_control" in b]
    return found


def test_the_last_tool_and_the_last_message_block_carry_a_cache_marker(ctx, registry):
    _, client, _ = turn(ctx, registry, proposal_script())
    for request in client.requests:
        marked = marked_blocks(request)
        assert marked == [("tool", "submit_response"), ("message", marked[1][1])]
        assert request["tools"][-1]["cache_control"] == MARK
        assert request["messages"][-1]["content"][-1]["cache_control"] == MARK
    # exactly one marker among the messages, always on the newest block
    assert sum(("cache_control" in b) for m in client.requests[2]["messages"] if isinstance(m["content"], list) for b in m["content"]) == 1


def test_a_plain_text_request_is_sent_as_a_block_so_it_can_carry_the_marker(ctx, registry):
    _, client, _ = turn(ctx, registry, [submit()], user="Make O-101 urgent.")
    assert client.requests[0]["messages"][-1]["content"] == [
        {"type": "text", "text": "Make O-101 urgent.", "cache_control": MARK}
    ]


def test_the_stored_conversation_never_contains_a_cache_marker(ctx, registry):
    _, _, messages = turn(ctx, registry, proposal_script())
    assert "cache_control" not in json.dumps(messages)
    assert messages[0]["content"] == "Make O-101 urgent."  # still the plain string the user typed


def test_the_tool_definitions_themselves_are_not_modified(ctx, registry):
    before = copy.deepcopy(registry.api_specs())
    turn(ctx, registry, [submit()])
    assert registry.api_specs() == before


def test_caching_can_be_switched_off(ctx, registry):
    _, client, _ = turn(ctx, registry, proposal_script(), config=AgentConfig(model="m", prompt_caching=False))
    assert all(marked_blocks(r) == [] for r in client.requests)
    assert client.requests[0]["messages"][-1]["content"] == "Make O-101 urgent."


# -- cost: the API reports cached tokens separately from input_tokens -------------------------------------


def test_cached_tokens_are_priced_at_their_own_rates():
    p = Prices(3.0, 15.0)
    assert p.cache_read == pytest.approx(0.3) and p.cache_write == pytest.approx(3.75)
    assert p.cost(1000, 100, cache_read_tokens=10_000, cache_write_tokens=2_000) == pytest.approx(
        (1000 * 3 + 100 * 15 + 10_000 * 0.3 + 2_000 * 3.75) / 1e6)
    assert Prices(3.0, 15.0, cache_read_per_mtok=0.5, cache_write_per_mtok=4.0).cost(0, 0, 1_000_000, 1_000_000) == pytest.approx(4.5)
    with pytest.raises(ValueError, match="negative"):
        Prices(3.0, 15.0, cache_read_per_mtok=-1)


def test_a_turn_counts_and_prices_its_cache_reads_and_writes(ctx, registry, tmp_path):
    script = [message(tool("get_schedule", "t1"), tokens_in=100, tokens_out=10, cache_write=5000),
              message(tool("get_schedule", "t2"), tokens_in=200, tokens_out=10, cache_read=5000, cache_write=300),
              submit()]
    path = tmp_path / "t.jsonl"
    config = AgentConfig(model="m", price_input_per_mtok=3.0, price_output_per_mtok=15.0)
    result, _, _ = turn(ctx, registry, script, config=config, tracer=Tracer(path, session_id="s"))

    assert (result.cache_read_tokens, result.cache_write_tokens) == (5000, 5300)
    expected = Prices(3.0, 15.0).cost(400, 70, 5000, 5300)
    assert result.cost_usd == pytest.approx(expected)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    calls = [r for r in records if r["event"] == "llm_call"]
    assert [(c["cache_read_tokens"], c["cache_write_tokens"]) for c in calls] == [(0, 5000), (5000, 300), (0, 0)]
    assert sum(c["step_cost_usd"] for c in calls) == pytest.approx(expected)
    end = records[-1]
    assert (end["cache_read_tokens"], end["cache_write_tokens"]) == (5000, 5300)


def test_cached_tokens_count_towards_the_token_budget(ctx, registry):
    # 100 in + 10 out per step is far below the limit; the 5000 cache-read tokens are not.
    script = [message(tool("get_schedule", "t1"), tokens_in=100, tokens_out=10, cache_read=5000), submit()]
    result, _, _ = turn(ctx, registry, script, config=AgentConfig(model="m", max_total_tokens=5000))
    assert result.status == "budget_exceeded" and result.steps == 1


def test_the_trace_report_shows_cached_tokens(ctx, registry, tmp_path):
    path = tmp_path / "t.jsonl"
    script = [message(tool("get_schedule", "t1"), cache_write=4000), message(tool("get_schedule", "t2"), cache_read=4000), submit()]
    turn(ctx, registry, script, tracer=Tracer(path, session_id="s"))
    summary = summarize(read_trace(path))
    assert (summary.cache_read_tokens, summary.cache_write_tokens) == (4000, 4000)
    text = render(summary)
    assert "cached" in text and "4000 read from cache, 4000 written to it" in text


def test_an_old_trace_without_cache_fields_still_reads(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps({"event": "turn_start"}) + "\n"
        + json.dumps({"event": "llm_call", "step": 1, "model": "m", "input_tokens": 5, "output_tokens": 2,
                      "latency_ms": 1, "tool_calls": []}) + "\n"
    )
    assert summarize(read_trace(path)).cache_read_tokens == 0


# -- what a person is shown: tokens used include the cached ones --------------------------------------------


def cached_script():
    return [message(tool("get_schedule", "t1"), tokens_in=10, tokens_out=20, cache_write=4000),
            message(tool("get_schedule", "t2"), tokens_in=10, tokens_out=20, cache_read=4000),
            submit(summary="Done.")]            # submit() adds 100 in, 50 out


def test_a_turn_reports_everything_the_model_processed_as_its_token_total(ctx, registry):
    result, _, _ = turn(ctx, registry, cached_script())
    assert result.input_tokens == 120 and result.output_tokens == 90          # uncached input, output
    assert result.total_tokens == 120 + 90 + 4000 + 4000


def test_the_chat_terminal_shows_the_total_and_how_much_came_from_the_cache(ctx):
    from tests.agent.test_cli import Session

    s = Session(ctx, cached_script())
    s.chat.handle("Show me the plan.")
    assert "3 model calls, 8210 tokens, 4000 read from cache" in s.output


def test_the_terminal_omits_the_cache_note_when_nothing_was_cached(ctx):
    from tests.agent.test_cli import Session

    s = Session(ctx, [submit(summary="Done.")])
    s.chat.handle("Hello.")
    assert "1 model calls, 150 tokens)" in s.output and "from cache" not in s.output


def test_the_eval_report_says_how_many_tokens_were_cached():
    from jobshop.evals.report import render_markdown, summarize
    from jobshop.evals.runner import ScenarioResult

    results = [ScenarioResult("a", "read_only", 1, "answered", True, input_tokens=100, output_tokens=50,
                              cache_read_tokens=900, cache_write_tokens=50)]
    text = render_markdown({"run_id": "r", "model": "m", "judge_model": None, "scenarios": 1, "repeat": 1, "solve_seconds": 5},
                           summarize(results), results)
    assert "agent tokens: 1100 (900 of them read from the prompt cache)" in text
