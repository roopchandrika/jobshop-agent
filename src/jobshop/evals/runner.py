"""Run scenarios against an agent and score them.

The agent under test runs exactly as in production: ``run_turn`` with the real system prompt and
the real tool registry over a fresh in-memory store. Only the starting point (the fixture shop)
and the clock are set by the eval.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jobshop.agent.loop import AgentConfig, run_turn
from jobshop.agent.prompts import build_system_prompt
from jobshop.agent.trace import Tracer
from jobshop.core.models import Instance, Schedule
from jobshop.core.solver import SolverConfig
from jobshop.evals.checks import evaluate
from jobshop.evals.judge import JudgeResult, judge_run
from jobshop.evals.record import CheckResult, Run
from jobshop.evals.scenario import Scenario
from jobshop.evals.shop import fresh_context
from jobshop.tools.registry import ToolRegistry


@dataclass
class ScenarioResult:
    id: str
    category: str
    attempt: int
    status: str  # the turn's status: answered, step_limit, ... or "crashed"
    passed: bool
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    wall_s: float = 0.0
    model: str | None = None
    llm_ms: int = 0   # waiting for the model
    tool_ms: int = 0  # running tools (mostly the solver)
    tools_called: list[str] = field(default_factory=list)
    tool_errors: int = 0
    answer: dict[str, Any] | None = None
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    judge_tokens: tuple[int, int] = (0, 0)
    trace: str | None = None
    error: str | None = None
    cache_read_tokens: int = 0   # input read back from the prompt cache (input_tokens excludes these)
    cache_write_tokens: int = 0


def execute(
    scenario: Scenario,
    client: Any,
    config: AgentConfig,
    shop: tuple[Instance, Schedule],
    solver_config: SolverConfig,
    attempt: int = 1,
    trace_path: Path | None = None,
) -> Run:
    ctx = fresh_context(shop, solver_config, scenario.now, scenario.poison)
    version_at_start = ctx.store.committed.version  # read BEFORE the run: that is the point of comparing
    started = time.perf_counter()
    messages: list[dict[str, Any]] = []
    turn = run_turn(
        client, ToolRegistry(ctx), build_system_prompt(ctx), messages, scenario.request, config,
        Tracer(path=trace_path, session_id=f"{scenario.id}#{attempt}"),
    )
    return Run(scenario, attempt, ctx, messages, turn, version_at_start, time.perf_counter() - started, trace_path, config.model)


def _checks_to_dict(checks: dict[str, CheckResult]) -> dict[str, dict[str, Any]]:
    return {name: {"passed": c.passed, "details": c.details} for name, c in checks.items()}


def score(run: Run, judge: tuple[Any, str] | None) -> ScenarioResult:
    checks = evaluate(run)
    judged: JudgeResult | None = judge_run(judge[0], judge[1], run) if judge else None
    if judged is not None:
        checks["judge"] = CheckResult("judge", judged.passed, [judged.error] if judged.error else [
            f"{name}: {s} - {judged.reasons[name]}" for name, s in judged.scores.items()
        ])
    final = run.turn.final
    return ScenarioResult(
        id=run.scenario.id, category=run.scenario.category, attempt=run.attempt, status=run.turn.status,
        passed=all(c.passed for c in checks.values() if c.passed is not None),
        steps=run.turn.steps, input_tokens=run.turn.input_tokens, output_tokens=run.turn.output_tokens,
        cache_read_tokens=run.turn.cache_read_tokens, cache_write_tokens=run.turn.cache_write_tokens,
        cost_usd=run.turn.cost_usd, wall_s=round(run.wall_s, 2),
        model=run.model, llm_ms=run.turn.llm_ms, tool_ms=run.turn.tool_ms,
        tools_called=[c.name for c in run.calls], tool_errors=sum(c.is_error for c in run.calls),
        answer=None if final is None else {
            "summary": final.summary, "clarifying_question": final.clarifying_question,
            "changes_made": final.changes_made, "needs_approval": final.needs_approval, "warnings": final.warnings,
        },
        checks=_checks_to_dict(checks),
        judge=None if judged is None else {"scores": judged.scores, "reasons": judged.reasons, "error": judged.error},
        judge_tokens=(judged.input_tokens, judged.output_tokens) if judged else (0, 0),
        trace=str(run.trace_path) if run.trace_path else None,
    )


def run_suite(
    scenarios: list[Scenario],
    client_for: Callable[[Scenario], Any],
    config: AgentConfig,
    shop: tuple[Instance, Schedule] | dict[str, tuple[Instance, Schedule]],
    solver_config: SolverConfig,
    *,
    judge: tuple[Any, str] | None = None,
    repeat: int = 1,
    trace_dir: Path | None = None,
    progress: Callable[[ScenarioResult], None] = lambda r: None,
) -> list[ScenarioResult]:
    """Run every scenario ``repeat`` times. ``shop`` is one fixture (the "default" shop) or a dict by name."""
    shops = shop if isinstance(shop, dict) else {"default": shop}
    missing = sorted({s.shop for s in scenarios} - set(shops))
    if missing:  # a setup mistake, found before any (paid) run starts
        raise ValueError(f"scenarios use shop(s) that were not loaded: {missing}")
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        for attempt in range(1, repeat + 1):
            trace_path = trace_dir / f"{scenario.id}-{attempt}.jsonl" if trace_dir else None
            try:
                run = execute(scenario, client_for(scenario), config, shops[scenario.shop], solver_config, attempt, trace_path)
                result = score(run, judge)
            except Exception as e:  # a bug must cost one scenario, not the whole (paid) run
                result = ScenarioResult(
                    scenario.id, scenario.category, attempt, "crashed", False,
                    error="".join(traceback.format_exception_only(type(e), e)).strip(),
                    trace=str(trace_path) if trace_path else None,
                )
            results.append(result)
            progress(result)
    return results


def result_dicts(results: list[ScenarioResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
