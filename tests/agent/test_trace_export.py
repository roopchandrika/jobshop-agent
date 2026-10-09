"""OpenTelemetry export of the JSONL traces: structure, timing, no content, and the failure cases."""

import json
import re
from datetime import datetime

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.trace import Tracer
from jobshop.agent.trace_export import KIND_CLIENT, STATUS_ERROR, STATUS_OK, main, to_otlp, turns
from jobshop.agent.trace_report import read_trace
from tests.agent.test_loop import proposal_script, turn
from tests.fake_llm import message, submit, tool

PRICED = AgentConfig(model="fake-model", price_input_per_mtok=3.0, price_output_per_mtok=15.0)


def traced(ctx, registry, tmp_path, script, config=PRICED, name="t.jsonl", **kw):
    path = tmp_path / name
    result, client, messages = turn(ctx, registry, script, config=config, tracer=Tracer(path, session_id="sess1"), **kw)
    return result, read_trace(path)


def spans_of(records):
    return to_otlp(records)["resourceSpans"][0]["scopeSpans"][0]["spans"]


def attrs(span):
    return {a["key"]: a["value"] for a in span["attributes"]}


def test_the_document_has_the_otlp_json_shape_and_names_the_service(ctx, registry, tmp_path):
    _, records = traced(ctx, registry, tmp_path, proposal_script())
    doc = to_otlp(records)
    [resource] = doc["resourceSpans"]
    assert resource["resource"]["attributes"] == [{"key": "service.name", "value": {"stringValue": "jobshop-agent"}}]
    [scope] = resource["scopeSpans"]
    assert scope["scope"]["name"] == "jobshop.agent.trace_export" and scope["spans"]
    json.dumps(doc)       # serialisable as is


def test_one_turn_is_one_trace_with_a_root_and_a_child_for_every_model_and_tool_call(ctx, registry, tmp_path):
    _, records = traced(ctx, registry, tmp_path, proposal_script())
    spans = spans_of(records)
    root, *children = spans
    assert root["name"] == "agent turn" and "parentSpanId" not in root
    assert len({s["traceId"] for s in spans}) == 1 and re.fullmatch(r"[0-9a-f]{32}", root["traceId"])
    assert all(re.fullmatch(r"[0-9a-f]{16}", s["spanId"]) for s in spans) and len({s["spanId"] for s in spans}) == len(spans)
    assert all(c["parentSpanId"] == root["spanId"] for c in children)
    assert [c["name"] for c in children if c["kind"] == KIND_CLIENT] == ["chat fake-model"] * 4
    tools = [c["name"] for c in children if c["name"].startswith("execute_tool")]
    assert tools == ["execute_tool create_draft", "execute_tool change_priority", "execute_tool reschedule", "execute_tool compare_schedules"]


def test_spans_have_real_durations_inside_the_turn_and_the_call_latencies(ctx, registry, tmp_path):
    result, records = traced(ctx, registry, tmp_path, proposal_script())
    spans = spans_of(records)
    root = spans[0]
    start, end = int(root["startTimeUnixNano"]), int(root["endTimeUnixNano"])
    assert end >= start
    latencies = [r["latency_ms"] for r in records if r["event"] == "llm_call"]
    chat = [s for s in spans if s["kind"] == KIND_CLIENT]
    assert [round((int(s["endTimeUnixNano"]) - int(s["startTimeUnixNano"])) / 1e6) for s in chat] == latencies
    assert all(int(s["startTimeUnixNano"]) <= int(s["endTimeUnixNano"]) for s in spans)


def test_attributes_use_otlp_encodings_and_carry_tokens_model_and_cost(ctx, registry, tmp_path):
    result, records = traced(ctx, registry, tmp_path, proposal_script())
    spans = spans_of(records)
    chat = attrs(next(s for s in spans if s["kind"] == KIND_CLIENT))
    assert chat["gen_ai.operation.name"] == {"stringValue": "chat"} and chat["gen_ai.request.model"] == {"stringValue": "fake-model"}
    assert chat["gen_ai.usage.input_tokens"] == {"intValue": "100"} and chat["gen_ai.usage.output_tokens"] == {"intValue": "50"}
    assert chat["jobshop.purpose"] == {"stringValue": "agent"} and isinstance(chat["jobshop.cost_usd"]["doubleValue"], float)
    root = attrs(spans[0])
    assert root["jobshop.status"] == {"stringValue": "answered"} and root["jobshop.steps"] == {"intValue": "4"}
    assert root["gen_ai.usage.input_tokens"] == {"intValue": "400"} and root["jobshop.pattern"] == {"stringValue": "react"}
    assert abs(root["jobshop.cost_usd"]["doubleValue"] - result.cost_usd) < 1e-12


def test_without_prices_there_is_no_cost_attribute_rather_than_a_zero(ctx, registry, tmp_path):
    _, records = traced(ctx, registry, tmp_path, proposal_script(), config=AgentConfig(model="fake-model"))
    assert all("jobshop.cost_usd" not in attrs(s) for s in spans_of(records))


def test_no_content_leaves_the_machine_only_metadata(ctx, registry, tmp_path):
    _, records = traced(ctx, registry, tmp_path, proposal_script(), user="Make O-101 urgent.")
    text = json.dumps(to_otlp(records))
    for private in ["Make O-101 urgent", "O-101 is now first in line", "approval", "notes_untrusted_text", "arguments", "result"]:
        assert private not in text


def test_a_tool_that_failed_is_an_error_span(ctx, registry, tmp_path):
    script = [message(tool("get_order", "t1", order_id="NOPE")), submit()]
    _, records = traced(ctx, registry, tmp_path, script)
    failed = next(s for s in spans_of(records) if s["name"] == "execute_tool get_order")
    assert failed["status"]["code"] == STATUS_ERROR and "error" in failed["status"]["message"]
    assert next(s for s in spans_of(records) if s["name"] == "agent turn")["status"]["code"] == STATUS_OK


def test_a_failed_api_call_is_an_error_span_and_marks_the_turn(ctx, registry, tmp_path):
    import anthropic
    import httpx2

    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://x"))
    _, records = traced(ctx, registry, tmp_path, [error])
    spans = spans_of(records)
    assert spans[0]["status"]["code"] == STATUS_ERROR and "api_error" in spans[0]["status"]["message"]
    assert any(s["name"] == "chat (failed)" and s["status"]["code"] == STATUS_ERROR for s in spans)


def test_a_turn_that_never_finished_is_marked_so(ctx, registry, tmp_path):
    _, records = traced(ctx, registry, tmp_path, proposal_script())
    cut = [r for r in records if r["event"] != "turn_end"]
    root = spans_of(cut)[0]
    assert root["status"] == {"code": STATUS_ERROR, "message": "the turn did not finish"}
    assert int(root["endTimeUnixNano"]) >= int(root["startTimeUnixNano"])


def test_each_turn_in_a_session_is_its_own_trace_and_ids_are_repeatable(ctx, registry, tmp_path):
    _, first = traced(ctx, registry, tmp_path, proposal_script())
    path = tmp_path / "t.jsonl"
    result, client, _ = turn(ctx, registry, [submit(summary="Second.")], config=PRICED, tracer=Tracer(path, session_id="sess1"))
    records = read_trace(path)
    assert len(turns(records)) == 2
    roots = [s for s in spans_of(records) if s["name"] == "agent turn"]
    assert len({r["traceId"] for r in roots}) == 2
    assert [s["spanId"] for s in spans_of(records)] == [s["spanId"] for s in spans_of(records)]       # same input, same ids
    assert attrs(roots[1])["jobshop.turn"] == {"intValue": "2"}


def test_planner_and_critic_calls_are_labelled_by_purpose(ctx, registry, tmp_path):
    from jobshop.agent.patterns import PLAN_TOOL, REVIEW_TOOL
    steps = [{"tool": "create_draft", "why": "x"}]
    script = [message(tool(PLAN_TOOL, "p", steps=steps)), *proposal_script(), message(tool(REVIEW_TOOL, "r", ok=True, problems=[]))]
    _, records = traced(ctx, registry, tmp_path, script, config=AgentConfig(model="fake-model", pattern="plan+reflect"))
    purposes = [attrs(s)["jobshop.purpose"]["stringValue"] for s in spans_of(records) if s["kind"] == KIND_CLIENT]
    assert purposes == ["plan", "agent", "agent", "agent", "agent", "critic"]


def test_records_outside_any_turn_are_ignored():
    assert turns([{"event": "api_error", "ts": datetime.now().isoformat()}]) == [] and spans_of([]) == []


def test_the_command_writes_a_file_or_prints_and_reports_unreadable_input(ctx, registry, tmp_path, capsys):
    traced(ctx, registry, tmp_path, proposal_script(), name="in.jsonl")
    out = tmp_path / "spans.json"
    assert main([str(tmp_path / "in.jsonl"), "-o", str(out)]) == 0 and json.loads(out.read_text())["resourceSpans"]
    assert "wrote" in capsys.readouterr().out
    assert main([str(tmp_path / "in.jsonl")]) == 0 and '"resourceSpans"' in capsys.readouterr().out
    (tmp_path / "bad.jsonl").write_text("{broken")
    assert main([str(tmp_path / "bad.jsonl")]) == 1 and "cannot read" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.jsonl")]) == 1
