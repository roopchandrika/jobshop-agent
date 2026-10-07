"""Traces must let you answer "what did this turn cost, and where did the time go?" from the file alone."""

import json

import anthropic
import httpx2
import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.pricing import Prices, parse_prices
from jobshop.agent.trace import TRACE_VERSION, Tracer
from jobshop.agent.trace_report import main as report_main
from jobshop.agent.trace_report import read_trace, render, summarize
from tests.agent.test_loop import proposal_script, turn
from tests.fake_llm import message, submit, tool

PRICED = AgentConfig(model="m", price_input_per_mtok=3.0, price_output_per_mtok=15.0)


def records_of(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_traced(ctx, registry, tmp_path, script, config=PRICED, **kw):
    path = tmp_path / "t.jsonl"
    result, _, _ = turn(ctx, registry, script, config=config, tracer=Tracer(path, session_id="s"), **kw)
    return result, records_of(path)


# -- pricing -----------------------------------------------------------------------------------------


def test_prices_turn_tokens_into_dollars():
    assert Prices(3.0, 15.0).cost(1000, 500) == pytest.approx(0.0105)
    assert Prices(3.0, 15.0).cost(0, 0) == 0


@pytest.mark.parametrize("text, expected", [("3,15", (3.0, 15.0)), (" 0.8 , 4 ", (0.8, 4.0))])
def test_prices_parse_as_input_then_output(text, expected):
    assert (parse_prices(text).input_per_mtok, parse_prices(text).output_per_mtok) == expected


@pytest.mark.parametrize("text", ["3", "3,15,1", "a,b", "", "-1,5"])
def test_bad_prices_are_rejected_with_the_expected_form(text):
    with pytest.raises(ValueError, match="(input,output|negative)"):
        parse_prices(text)


# -- per-step cost, tokens and latency -----------------------------------------------------------------


def test_every_model_call_records_its_own_cost_and_a_running_total(ctx, registry, tmp_path):
    script = [message(tool("get_schedule", "t1"), tokens_in=1000, tokens_out=100),
              message(tool("get_schedule", "t2"), tokens_in=2000, tokens_out=300), submit()]
    result, records = run_traced(ctx, registry, tmp_path, script)
    calls = [r for r in records if r["event"] == "llm_call"]

    # 1000*3/1e6 + 100*15/1e6, 2000*3/1e6 + 300*15/1e6, and submit()'s default 100 in / 50 out.
    assert [c["step_cost_usd"] for c in calls] == pytest.approx([0.0045, 0.0105, 0.00105])
    assert [c["total_cost_usd"] for c in calls] == pytest.approx([0.0045, 0.0150, 0.01605])
    assert sum(c["step_cost_usd"] for c in calls) == pytest.approx(result.cost_usd) == pytest.approx(records[-1]["cost_usd"])
    assert sum(c["input_tokens"] for c in calls) == result.input_tokens == records[-1]["input_tokens"]


def test_without_prices_costs_are_null_not_zero(ctx, registry, tmp_path):
    result, records = run_traced(ctx, registry, tmp_path, [submit()], config=AgentConfig(model="m"))
    call = next(r for r in records if r["event"] == "llm_call")
    assert call["step_cost_usd"] is None and call["total_cost_usd"] is None and result.cost_usd is None


def test_turn_end_totals_equal_the_sum_of_the_step_latencies(ctx, registry, tmp_path):
    result, records = run_traced(ctx, registry, tmp_path, proposal_script())
    end = records[-1]
    assert end["llm_ms"] == sum(r["latency_ms"] for r in records if r["event"] == "llm_call") == result.llm_ms
    assert end["tool_ms"] == sum(r["latency_ms"] for r in records if r["event"] == "tool_call") == result.tool_ms
    assert end["wall_ms"] >= end["llm_ms"] + end["tool_ms"] - 5  # the turn took at least as long as its parts


def test_the_solver_shows_up_as_tool_time_not_model_time(ctx, registry, tmp_path):
    _, records = run_traced(ctx, registry, tmp_path, proposal_script())
    by_tool = {r["tool"]: r["latency_ms"] for r in records if r["event"] == "tool_call"}
    assert by_tool["reschedule"] == max(by_tool.values()) and by_tool["reschedule"] > 0


def test_failed_tool_calls_are_traced_and_timed_too(ctx, registry, tmp_path):
    script = [message(tool("get_order", "t1", order_id="O-999")), submit()]
    result, records = run_traced(ctx, registry, tmp_path, script)
    failed = next(r for r in records if r["event"] == "tool_call")
    assert failed["is_error"] is True and "unknown order" in failed["result"]["error"]
    assert result.tool_ms == failed["latency_ms"]


def test_the_trace_names_the_model_and_the_format_version(ctx, registry, tmp_path):
    _, records = run_traced(ctx, registry, tmp_path, [submit()])
    assert records[0]["trace_version"] == TRACE_VERSION == 2 and records[0]["model"] == "m"
    assert all(r["model"] == "m" for r in records if r["event"] == "llm_call")


def test_an_api_failure_is_traced_with_how_long_it_took(ctx, registry, tmp_path):
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://example.invalid"))
    result, records = run_traced(ctx, registry, tmp_path, [error])
    failure = next(r for r in records if r["event"] == "api_error")
    assert result.status == "api_error" and "latency_ms" in failure and records[-1]["status"] == "api_error"


# -- reading traces ----------------------------------------------------------------------------------


def test_the_report_has_one_row_per_model_call_with_its_tools_and_totals(ctx, registry, tmp_path):
    _, records = run_traced(ctx, registry, tmp_path, proposal_script())
    summary = summarize(records)
    assert [s.step for s in summary.steps] == [1, 2, 3, 4]
    assert summary.steps[0].tools == ["create_draft"]
    reschedule_step = next(s for s in summary.steps if "reschedule" in s.tools)
    assert reschedule_step.tool_ms == max(s.tool_ms for s in summary.steps)  # tool time is attributed to its own step
    assert summary.input_tokens == 400 and summary.output_tokens == 200 and summary.statuses == ["answered"]
    assert summary.cost_usd == pytest.approx(Prices(3.0, 15.0).cost(400, 200))

    text = render(summary)
    assert "create_draft" in text and "4 model call(s): 400 in + 200 out tokens" in text and "turn status: answered" in text


def test_a_failed_tool_call_is_flagged_in_its_step(ctx, registry, tmp_path):
    _, records = run_traced(ctx, registry, tmp_path, [message(tool("get_order", "t1", order_id="O-999")), submit()])
    assert "(1 failed)" in render(summarize(records))


def test_costs_in_a_report_are_n_a_without_prices(ctx, registry, tmp_path):
    _, records = run_traced(ctx, registry, tmp_path, [submit()], config=AgentConfig(model="m"))
    summary = summarize(records)
    assert summary.cost_usd is None and "cost n/a" in render(summary)


def test_turns_in_one_session_are_numbered_separately(ctx, registry, tmp_path):
    path = tmp_path / "t.jsonl"
    tracer = Tracer(path, session_id="s")
    _, _, messages = turn(ctx, registry, [submit(summary="one")], config=PRICED, tracer=tracer)
    turn(ctx, registry, [submit(summary="two")], user="again", messages=messages, config=PRICED, tracer=tracer)
    summary = summarize(read_trace(path))
    assert summary.turns == 2 and [(s.turn, s.step) for s in summary.steps] == [(1, 1), (2, 1)]


def test_a_corrupt_trace_names_the_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"event": "turn_start"}\nnot json\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl, line 2"):
        read_trace(path)


def test_the_report_command_prints_and_fails_cleanly(ctx, registry, tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    turn(ctx, registry, proposal_script(), config=PRICED, tracer=Tracer(path))
    assert report_main([str(path)]) == 0 and "model call(s)" in capsys.readouterr().out
    assert report_main([str(tmp_path / "missing.jsonl")]) == 1 and "cannot read" in capsys.readouterr().out


# -- what a trace must never contain --------------------------------------------------------------------


def test_an_approval_token_never_lands_in_the_trace(ctx, tmp_path):
    from jobshop.agent.cli import ChatSession
    from tests.fake_llm import FakeClient

    minted = []
    real_issue = ctx.authority.issue
    ctx.authority.issue = lambda **kw: minted.append(real_issue(**kw)) or minted[-1]
    path = tmp_path / "t.jsonl"
    script = [message(tool("create_draft", "a1")),
              message(tool("change_priority", "a2", draft_id="D1", order_id="O-101", priority=5), tool("reschedule", "a3", draft_id="D1")),
              submit(summary="Done.", draft_id="D1")]
    ChatSession(FakeClient(script), ctx, PRICED, Tracer(path), out=lambda s: None, ask=lambda p: "y").handle("Make O-101 urgent.")

    assert len(minted) == 1 and ctx.store.committed.version == 2  # the approval really happened
    assert minted[0] not in path.read_text()
