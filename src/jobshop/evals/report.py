"""Turn a list of scenario results into the summary table, a markdown report and a JSON file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jobshop.evals.runner import ScenarioResult, result_dicts

CHECKS = ("outcome", "tools", "changes", "validator", "numbers", "judge")


def tally(results: list[ScenarioResult], check: str) -> tuple[int, int]:
    """(passed, applicable) for one check. A check that did not apply (or could not run) is left out."""
    applicable = [r.checks[check]["passed"] for r in results if check in r.checks and r.checks[check]["passed"] is not None]
    return sum(applicable), len(applicable)


def summarize(results: list[ScenarioResult]) -> dict[str, Any]:
    by_category: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    def block(rs: list[ScenarioResult]) -> dict[str, Any]:
        return {
            "runs": len(rs),
            "passed": sum(r.passed for r in rs),
            "checks": {c: tally(rs, c) for c in CHECKS},
        }

    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    return {
        "overall": block(results),
        "by_category": {c: block(rs) for c, rs in by_category.items()},
        "judge_errors": sum(1 for r in results if r.checks.get("judge", {}).get("passed") is None and "judge" in r.checks),
        "crashed": [r.id for r in results if r.status == "crashed"],
        "agent_tokens": sum(r.input_tokens + r.output_tokens + r.cache_read_tokens + r.cache_write_tokens for r in results),
        "agent_cached_tokens": sum(r.cache_read_tokens for r in results),
        "judge_tokens": sum(sum(r.judge_tokens) for r in results),
        "agent_cost_usd": round(sum(costs), 4) if costs else None,
        "mean_wall_s": round(sum(r.wall_s for r in results) / len(results), 2) if results else 0,
    }


def _cell(passed: int, total: int) -> str:
    return f"{passed}/{total}" if total else "-"


def render_table(summary: dict[str, Any]) -> str:
    rows = [("category", "runs", *CHECKS, "all")]
    for name, b in [*sorted(summary["by_category"].items()), ("TOTAL", summary["overall"])]:
        rows.append((name, str(b["runs"]), *(_cell(*b["checks"][c]) for c in CHECKS), _cell(b["passed"], b["runs"])))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths))) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.insert(-1, lines[1])
    return "\n".join(lines)


def render_markdown(meta: dict[str, Any], summary: dict[str, Any], results: list[ScenarioResult]) -> str:
    out = [f"# Eval run {meta['run_id']}", ""]
    out += [f"- agent model: `{meta['model']}`", f"- judge model: `{meta['judge_model'] or 'none (judge skipped)'}`",
            f"- scenarios: {meta['scenarios']} x {meta['repeat']} run(s); solver limit {meta['solve_seconds']:g} s",
            f"- agent tokens: {summary['agent_tokens']} ({summary['agent_cached_tokens']} of them read from the prompt cache), judge tokens: {summary['judge_tokens']}, "
            f"agent cost: {'n/a (no prices set)' if summary['agent_cost_usd'] is None else '$' + str(summary['agent_cost_usd'])}, "
            f"mean wall time per run: {summary['mean_wall_s']} s", ""]
    if summary["judge_errors"]:
        out += [f"**{summary['judge_errors']} judge call(s) failed and are excluded from the judge column.**", ""]
    if summary["crashed"]:
        out += [f"**Crashed runs (harness bug, not a model failure): {', '.join(summary['crashed'])}**", ""]
    out += ["```", render_table(summary), "```", "",
            "Each cell is passed/applicable. `-` means the check did not apply (for example, no schedule to validate).", ""]

    failed = [r for r in results if not r.passed]
    out += ["## Failures", ""] if failed else ["## Failures", "", "None.", ""]
    for r in failed:
        out.append(f"### {r.id} (attempt {r.attempt}) - {r.category}")
        if r.error:
            out.append(f"- crashed: `{r.error}`")
        for name, c in r.checks.items():
            if c["passed"] is False:
                out += [f"- **{name}**: " + "; ".join(c["details"] or ["failed"])]
        if r.answer:
            out.append(f"- answer: {r.answer['summary'][:400]}")
        out.append("")
    return "\n".join(out)


def write_results(out_dir: Path, meta: dict[str, Any], results: list[ScenarioResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "results": result_dicts(results)}, indent=1, default=str),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_markdown(meta, summary, results), encoding="utf-8")
    return out_dir
