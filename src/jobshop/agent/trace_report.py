"""Read JSONL traces: ``python -m jobshop.agent.trace_report logs/traces/20260105-120000.jsonl``.

Prints one line per model call (tokens, latency, cost, the tools it asked for and how long they
took) and the totals, so "why was that turn slow or expensive?" has an answer without reading JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Step:
    turn: int
    step: int
    model: str | None
    input_tokens: int
    output_tokens: int
    llm_ms: int
    cost_usd: float | None
    tools: list[str]
    tool_ms: int = 0
    tool_errors: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class TraceSummary:
    turns: int = 0
    steps: list[Step] = field(default_factory=list)
    api_errors: int = 0
    statuses: list[str] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def cache_read_tokens(self) -> int:
        return sum(s.cache_read_tokens for s in self.steps)

    @property
    def cache_write_tokens(self) -> int:
        return sum(s.cache_write_tokens for s in self.steps)

    @property
    def llm_ms(self) -> int:
        return sum(s.llm_ms for s in self.steps)

    @property
    def tool_ms(self) -> int:
        return sum(s.tool_ms for s in self.steps)

    @property
    def cost_usd(self) -> float | None:
        costs = [s.cost_usd for s in self.steps]
        return None if not costs or any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]


def read_trace(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name}, line {number}: not valid JSON ({e.msg})") from None
    return records


def summarize(records: list[dict[str, Any]]) -> TraceSummary:
    summary = TraceSummary()
    current: dict[tuple[int, int], Step] = {}
    for r in records:
        event = r["event"]
        if event == "turn_start":
            summary.turns += 1
        elif event == "turn_end":
            summary.statuses.append(r["status"])
        elif event == "api_error":
            summary.api_errors += 1
        elif event == "llm_call":
            step = Step(summary.turns, r["step"], r.get("model"), r["input_tokens"], r["output_tokens"],
                        r["latency_ms"], r.get("step_cost_usd"), list(r["tool_calls"]),
                        cache_read_tokens=r.get("cache_read_tokens") or 0,
                        cache_write_tokens=r.get("cache_write_tokens") or 0)
            current[(summary.turns, r["step"])] = step
            summary.steps.append(step)
        elif event == "tool_call":
            step = current.get((summary.turns, r["step"]))
            if step is not None:
                step.tool_ms += r["latency_ms"]
                step.tool_errors += bool(r["is_error"])
    return summary


def render(summary: TraceSummary) -> str:
    def money(value: float | None) -> str:
        return "n/a" if value is None else f"${value:.4f}"

    # "cached" is input read back from the prompt cache; "in tok" counts only input that was not cached.
    rows = [("turn", "step", "in tok", "cached", "out tok", "llm ms", "cost", "tool ms", "tools")]
    for s in summary.steps:
        tools = ", ".join(s.tools) + (f"  ({s.tool_errors} failed)" if s.tool_errors else "")
        rows.append((str(s.turn), str(s.step), str(s.input_tokens), str(s.cache_read_tokens), str(s.output_tokens),
                     str(s.llm_ms), money(s.cost_usd), str(s.tool_ms) if s.tools else "-", tools))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(r[:-1], widths)) + "  " + r[-1] for r in rows]
    lines.insert(1, "-" * len(lines[0]))
    lines.append(
        f"\n{summary.turns} turn(s), {len(summary.steps)} model call(s): {summary.input_tokens} in + {summary.output_tokens} out tokens "
        f"(+ {summary.cache_read_tokens} read from cache, {summary.cache_write_tokens} written to it), "
        f"cost {money(summary.cost_usd)}, model time {summary.llm_ms / 1000:.1f} s, tool time {summary.tool_ms / 1000:.1f} s"
        + (f", {summary.api_errors} API error(s)" if summary.api_errors else "")
        + f"\nturn status: {', '.join(summary.statuses) or 'none recorded'}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.agent.trace_report", description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for path in args.traces:
        try:
            print(f"== {path}\n{render(summarize(read_trace(path)))}\n")
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: cannot read ({type(e).__name__}: {e})")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
