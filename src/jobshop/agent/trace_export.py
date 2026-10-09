"""Export a trace as OpenTelemetry spans (OTLP/JSON), so it can be looked at in a tracing tool.

    python -m jobshop.agent.trace_export logs/traces/20260105-120000.jsonl -o spans.json

The JSONL traces are this project's own format. Tracing tools (Jaeger, Grafana Tempo, Honeycomb, ...) speak
OpenTelemetry; this converts one to the other without adding a dependency: it writes the JSON that an OTLP/HTTP
endpoint accepts (``POST /v1/traces``, content type application/json).

Each turn becomes one trace. Its root span is the turn; under it, one span per model call and one per tool call, with
the real start time and duration, token counts, model, cost and the tool's name.

Left out on purpose: the planner's text, the model's answers, tool arguments and tool results. A trace file holds full
tool results (order notes, plans), and a tracing backend is usually a shared service. Metadata only.

Honest limits: the attribute names follow OpenTelemetry's GenAI conventions as I know them, which were still evolving, so
a newer backend may expect slightly different names; and the output has not been loaded into a real collector here,
only checked against the OTLP JSON structure by tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jobshop.agent.trace_report import read_trace

SERVICE = "jobshop-agent"
KIND_INTERNAL, KIND_CLIENT = 1, 3
STATUS_OK, STATUS_ERROR = 1, 2


def _attr(key: str, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}            # OTLP/JSON carries 64-bit integers as strings
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _attrs(**kv: Any) -> list[dict[str, Any]]:
    return [a for k, v in kv.items() if (a := _attr(k.replace("__", "."), v)) is not None]


def _hex(*parts: object, length: int) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:length]


def _nanos(moment: datetime) -> str:
    return str(int(moment.timestamp() * 1_000_000_000))


def _when(record: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(record["ts"]).astimezone(timezone.utc)


def _span(trace_id: str, span_id: str, parent: str | None, name: str, kind: int, start: datetime, end: datetime,
          attributes: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
    span: dict[str, Any] = {
        "traceId": trace_id, "spanId": span_id, "name": name, "kind": kind,
        "startTimeUnixNano": _nanos(start), "endTimeUnixNano": _nanos(end), "attributes": attributes,
        "status": {"code": STATUS_ERROR, "message": error} if error else {"code": STATUS_OK},
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def turns(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group records into turns: each starts at ``turn_start`` and runs up to the next one."""
    out: list[list[dict[str, Any]]] = []
    for r in records:
        if r["event"] == "turn_start":
            out.append([])
        if out:
            out[-1].append(r)
    return out


def turn_spans(turn: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    start_record = turn[0]
    session = start_record.get("session", "")
    trace_id = _hex(session, index, length=32)
    root_id = _hex(trace_id, "turn", length=16)
    end_record = next((r for r in reversed(turn) if r["event"] == "turn_end"), None)
    started = _when(start_record)
    ended = _when(end_record) if end_record else max(_when(r) for r in turn)

    spans: list[dict[str, Any]] = []
    for n, r in enumerate(turn):
        event = r["event"]
        if event == "llm_call":
            end = _when(r)
            spans.append(_span(
                trace_id, _hex(trace_id, "llm", n, length=16), root_id, f"chat {r.get('model', '')}".strip(), KIND_CLIENT,
                end - timedelta(milliseconds=r["latency_ms"]), end,
                _attrs(gen_ai__operation__name="chat", gen_ai__system="anthropic", gen_ai__request__model=r.get("model"),
                       gen_ai__response__id=r.get("response_id"), gen_ai__response__finish_reasons=r.get("stop_reason"),
                       gen_ai__usage__input_tokens=r["input_tokens"], gen_ai__usage__output_tokens=r["output_tokens"],
                       jobshop__step=r.get("step"), jobshop__purpose=r.get("purpose"),
                       jobshop__cache_read_tokens=r.get("cache_read_tokens"), jobshop__cache_write_tokens=r.get("cache_write_tokens"),
                       jobshop__cost_usd=r.get("step_cost_usd"), jobshop__tool_calls=",".join(r.get("tool_calls", [])) or None)))
        elif event == "tool_call":
            end = _when(r)
            spans.append(_span(
                trace_id, _hex(trace_id, "tool", n, length=16), root_id, f"execute_tool {r['tool']}", KIND_INTERNAL,
                end - timedelta(milliseconds=r["latency_ms"]), end,
                _attrs(gen_ai__operation__name="execute_tool", gen_ai__tool__name=r["tool"], jobshop__step=r.get("step")),
                "the tool returned an error" if r.get("is_error") else None))
        elif event == "api_error":
            end = _when(r)
            spans.append(_span(
                trace_id, _hex(trace_id, "apierror", n, length=16), root_id, "chat (failed)", KIND_CLIENT,
                end - timedelta(milliseconds=r.get("latency_ms", 0)), end,
                _attrs(gen_ai__operation__name="chat", gen_ai__system="anthropic", jobshop__step=r.get("step")), r.get("error", "API error")))

    status = end_record.get("status") if end_record else None
    failed = status not in (None, "answered")
    root = _span(
        trace_id, root_id, None, "agent turn", KIND_INTERNAL, started, max(ended, started),
        _attrs(gen_ai__system="anthropic", gen_ai__request__model=start_record.get("model"), jobshop__pattern=start_record.get("pattern"),
               jobshop__session=session, jobshop__turn=index, jobshop__status=status or "unfinished",
               jobshop__steps=end_record.get("steps") if end_record else None,
               gen_ai__usage__input_tokens=end_record.get("input_tokens") if end_record else None,
               gen_ai__usage__output_tokens=end_record.get("output_tokens") if end_record else None,
               jobshop__cost_usd=end_record.get("cost_usd") if end_record else None,
               jobshop__llm_ms=end_record.get("llm_ms") if end_record else None, jobshop__tool_ms=end_record.get("tool_ms") if end_record else None),
        f"turn ended with status {status}" if failed else ("the turn did not finish" if end_record is None else None))
    return [root, *spans]


def to_otlp(records: list[dict[str, Any]]) -> dict[str, Any]:
    spans = [s for i, t in enumerate(turns(records), 1) for s in turn_spans(t, i)]
    return {"resourceSpans": [{
        "resource": {"attributes": _attrs(service__name=SERVICE)},
        "scopeSpans": [{"scope": {"name": "jobshop.agent.trace_export"}, "spans": spans}],
    }]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.agent.trace_export", description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="write here instead of printing")
    args = parser.parse_args(argv)
    records: list[dict[str, Any]] = []
    for path in args.traces:
        try:
            records += read_trace(path)
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: cannot read ({type(e).__name__}: {e})", file=sys.stderr)
            return 1
    text = json.dumps(to_otlp(records), indent=1)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
