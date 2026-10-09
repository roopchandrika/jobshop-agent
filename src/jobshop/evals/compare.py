"""Run the same scenarios against several models and compare quality, cost and latency.

Fairness rules, all enforced here or in the CLI:

* Every model gets the same scenarios, the same fixture shop, the same solver limits, the same
  agent limits (steps, tokens) and the same judge.
* The judge is not one of the models being compared.
* Prices are per model and explicit. A model with no price shows cost as n/a; nothing is guessed,
  and one model's prices are never applied to another.
* Quality differences are reported with a confidence interval and a paired test on the scenarios
  where the models disagree, because 29 scenarios cannot rank models that are close.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jobshop.agent.loop import AgentConfig
from jobshop.agent.pricing import Prices
from jobshop.core.models import Instance, Schedule
from jobshop.core.solver import SolverConfig
from jobshop.knowledge import Retriever
from jobshop.evals.report import CHECKS, tally, write_results
from jobshop.evals.runner import ScenarioResult, run_suite
from jobshop.evals.scenario import Scenario
from jobshop.evals.stats import mean, percentile, sign_test_p, wilson_interval


@dataclass
class ModelSpec:
    name: str                       # the label in the report; also the model id unless ``model`` says otherwise
    client_for: Callable[[Scenario], Any]
    prices: Prices | None = None
    model: str | None = None        # the real model id, when the label is something like "claude-x [verify]"
    pattern: str | None = None      # run this model with a different agent pattern (see agent/patterns.py)


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in ".-" else "_" for c in name)[:40]


def run_comparison(
    scenarios: list[Scenario],
    models: list[ModelSpec],
    base_config: AgentConfig,
    shop: tuple[Instance, Schedule] | dict[str, tuple[Instance, Schedule]],
    solver_config: SolverConfig,
    *,
    knowledge: Retriever | None = None,
    judge: tuple[Any, str] | None = None,
    repeat: int = 1,
    out_dir: Path | None = None,
    progress: Callable[[str, ScenarioResult], None] = lambda model, r: None,
) -> dict[str, list[ScenarioResult]]:
    names = [m.name for m in models]
    if len(set(names)) != len(names) or len(models) < 2:
        raise ValueError("compare needs at least two distinct models")
    if judge is not None and judge[1] in {m.model or m.name for m in models}:
        raise ValueError(f"the judge ({judge[1]}) must not be one of the models being compared")

    results: dict[str, list[ScenarioResult]] = {}
    for spec in models:
        config = replace(
            base_config, model=spec.model or spec.name, pattern=spec.pattern or base_config.pattern, max_cost_usd=None,
            price_input_per_mtok=spec.prices.input_per_mtok if spec.prices else None,
            price_output_per_mtok=spec.prices.output_per_mtok if spec.prices else None,
        )
        results[spec.name] = run_suite(
            scenarios, spec.client_for, config, shop, solver_config, judge=judge, repeat=repeat,
            trace_dir=out_dir / slug(spec.name) / "traces" if out_dir else None,
            progress=lambda r, model=spec.name: progress(model, r), knowledge=knowledge,
        )
    return results


# -- metrics -----------------------------------------------------------------------------------------


def model_metrics(results: list[ScenarioResult]) -> dict[str, Any]:
    passed = sum(r.passed for r in results)
    low, high = wilson_interval(passed, len(results))
    costs = [r.cost_usd for r in results]
    priced = bool(results) and all(c is not None for c in costs)
    total_cost = sum(costs) if priced else None  # type: ignore[arg-type]
    judge_scores = [s for r in results if r.judge for s in r.judge["scores"].values()]
    walls = [r.wall_s for r in results]
    return {
        "runs": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "pass_ci": (low, high),
        "checks": {c: tally(results, c) for c in CHECKS},
        "judge_mean_score": mean(judge_scores) if judge_scores else None,
        "cost_total_usd": total_cost,
        "cost_per_run_usd": total_cost / len(results) if priced and results else None,
        "cost_per_pass_usd": total_cost / passed if priced and passed else None,
        # All input the model processed, cached or not (cached input is billed at a lower rate).
        "input_tokens_per_run": mean([r.input_tokens + r.cache_read_tokens + r.cache_write_tokens for r in results]),
        "output_tokens_per_run": mean([r.output_tokens for r in results]),
        "steps_per_run": mean([r.steps for r in results]),
        "tool_errors_per_run": mean([r.tool_errors for r in results]),
        "wall_s_mean": mean(walls),
        "wall_s_p50": percentile(walls, 50),
        "wall_s_p95": percentile(walls, 95),
        "llm_s_per_run": mean([r.llm_ms for r in results]) / 1000,
        "tool_s_per_run": mean([r.tool_ms for r in results]) / 1000,
        "statuses": dict(Counter(r.status for r in results)),
    }


def pair_stats(a: list[ScenarioResult], b: list[ScenarioResult]) -> dict[str, Any]:
    """Compare two models run by run. Runs are paired by (scenario id, attempt number)."""
    by_b = {(r.id, r.attempt): r for r in b}
    only_a, only_b, both, neither = [], [], 0, 0
    for ra in a:
        rb = by_b.get((ra.id, ra.attempt))
        if rb is None:
            continue
        if ra.passed and rb.passed:
            both += 1
        elif not ra.passed and not rb.passed:
            neither += 1
        elif ra.passed:
            only_a.append((ra.id, ra.attempt, [n for n, c in rb.checks.items() if c["passed"] is False] or [rb.status]))
        else:
            only_b.append((ra.id, ra.attempt, [n for n, c in ra.checks.items() if c["passed"] is False] or [ra.status]))
    return {"both": both, "neither": neither, "only_a": only_a, "only_b": only_b,
            "p_value": sign_test_p(len(only_a), len(only_b))}


# -- rendering ---------------------------------------------------------------------------------------


def _money(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"${value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"  # noqa: E731
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, rows)])


def render_comparison(meta: dict[str, Any], results: dict[str, list[ScenarioResult]]) -> str:
    names = list(results)
    metrics = {n: model_metrics(r) for n, r in results.items()}

    def pct(m: dict[str, Any]) -> str:
        lo, hi = m["pass_ci"]
        return f"{m['pass_rate']:.0%} ({m['passed']}/{m['runs']}), 95% CI {lo:.0%}-{hi:.0%}"

    rows = [
        ["**Quality**", *[""] * len(names)],
        ["scenarios passed", *[pct(metrics[n]) for n in names]],
        *[[f"  {c} check", *[("-" if not metrics[n]["checks"][c][1] else f"{metrics[n]['checks'][c][0]}/{metrics[n]['checks'][c][1]}") for n in names]] for c in CHECKS],
        ["judge mean score (1-5)", *["n/a" if metrics[n]["judge_mean_score"] is None else f"{metrics[n]['judge_mean_score']:.2f}" for n in names]],
        ["**Cost**", *[""] * len(names)],
        ["total", *[_money(metrics[n]["cost_total_usd"]) for n in names]],
        ["per run", *[_money(metrics[n]["cost_per_run_usd"], 5) for n in names]],
        ["per passing run", *[_money(metrics[n]["cost_per_pass_usd"], 5) for n in names]],
        ["tokens per run (in / out)", *[f"{metrics[n]['input_tokens_per_run']:.0f} / {metrics[n]['output_tokens_per_run']:.0f}" for n in names]],
        ["model calls per run", *[f"{metrics[n]['steps_per_run']:.1f}" for n in names]],
        ["**Latency** (per scenario run)", *[""] * len(names)],
        ["mean / median / p95", *[f"{metrics[n]['wall_s_mean']:.1f} / {metrics[n]['wall_s_p50']:.1f} / {metrics[n]['wall_s_p95']:.1f} s" for n in names]],
        ["of which waiting for the model", *[f"{metrics[n]['llm_s_per_run']:.1f} s" for n in names]],
        ["of which running tools (solver)", *[f"{metrics[n]['tool_s_per_run']:.1f} s" for n in names]],
        ["**Behaviour**", *[""] * len(names)],
        ["failed tool calls per run", *[f"{metrics[n]['tool_errors_per_run']:.2f}" for n in names]],
        ["runs ending any way but 'answered'", *[str(sum(v for k, v in metrics[n]["statuses"].items() if k != "answered")) for n in names]],
    ]
    out = [f"# Model comparison {meta['run_id']}", "",
           f"- models: {', '.join(f'`{n}`' for n in names)}",
           f"- judge: `{meta['judge_model'] or 'none'}`",
           f"- {meta['scenarios']} scenarios x {meta['repeat']} run(s) per model; solver limit {meta['solve_seconds']:g} s; same shop, limits and judge for all",
           "", _table(["", *names], rows), ""]

    base = names[0]
    for other in names[1:]:
        p = pair_stats(results[base], results[other])
        n_diff = len(p["only_a"]) + len(p["only_b"])
        out += [f"## `{base}` vs `{other}`", "",
                f"- both pass: {p['both']}, neither passes: {p['neither']}, only `{base}`: {len(p['only_a'])}, only `{other}`: {len(p['only_b'])}"]
        if n_diff == 0:
            out.append("- The models passed and failed exactly the same runs.")
        else:
            verdict = ("a difference this size is plausible by chance" if p["p_value"] >= 0.05
                       else "unlikely to be chance")
            out.append(f"- Sign test on the {n_diff} run(s) where they disagree: p = {p['p_value']:.3f} ({verdict}).")
        for items, winner, loser in ((p["only_a"], base, other), (p["only_b"], other, base)):
            for scenario_id, attempt, failed_checks in items:
                out.append(f"  - {scenario_id} (run {attempt}): only `{winner}` passed; `{loser}` failed: {', '.join(failed_checks)}")
        out.append("")

    out += ["## How to read this", "",
            "- A confidence interval that overlaps another model's means the pass rates are not clearly different. "
            "29 scenarios cannot rank close models; use `--repeat` and more scenarios before concluding.",
            "- Cost per passing run is the number to compare, not cost per run: a cheap model that fails more "
            "is not cheaper per solved task. It is only as good as the prices you supplied.",
            "- Latency includes the solver (identical work for both models) and the model's thinking time. "
            "Compare the 'waiting for the model' line to compare the models themselves.",
            "- Judge scores are from one judge model; compare them across models, not against an absolute standard.", ""]
    return "\n".join(out)


def write_comparison(out_dir: Path, meta: dict[str, Any], results: dict[str, list[ScenarioResult]]) -> Path:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rs in results.items():
        write_results(out_dir / slug(name), {**meta, "model": name}, rs)
    (out_dir / "comparison.md").write_text(render_comparison(meta, results), encoding="utf-8")
    (out_dir / "comparison.json").write_text(
        json.dumps({"meta": meta, "metrics": {n: model_metrics(r) for n, r in results.items()}}, indent=1, default=str),
        encoding="utf-8",
    )
    return out_dir
