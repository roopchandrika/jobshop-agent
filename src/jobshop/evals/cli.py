"""``python -m jobshop.evals run|build-shop``."""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from jobshop.agent.cli import ConfigError, agent_config_from_env
from jobshop.core.solver import SolverConfig
from jobshop.evals.oracle import OracleClient
from jobshop.evals.report import render_table, summarize, write_results
from jobshop.evals.runner import ScenarioResult, run_suite
from jobshop.evals.scenario import load_scenarios
from jobshop.evals.shop import build_shop, load_shop


def _progress(total: int):
    done = 0

    def show(r: ScenarioResult) -> None:
        nonlocal done
        done += 1
        failed = [name for name, c in r.checks.items() if c["passed"] is False]
        verdict = "PASS" if r.passed else "FAIL " + ",".join(failed or [r.status])
        print(f"[{done:>3}/{total}] {r.id:<8} {verdict}", flush=True)

    return show


def _run(args: argparse.Namespace) -> int:
    load_dotenv()
    evals_dir: Path = args.evals_dir
    try:
        scenarios = load_scenarios(evals_dir / "scenarios")
        if args.only:
            wanted = set(args.only)
            scenarios = [s for s in scenarios if s.id in wanted or s.category in wanted]
            if not scenarios:
                raise ConfigError(f"--only {args.only} matches no scenario id or category")
        if not (evals_dir / "shop.json").exists():
            raise ConfigError(f"{evals_dir / 'shop.json'} is missing; create it with: python -m jobshop.evals build-shop")

        model = args.model or os.environ.get("ANTHROPIC_MODEL", "")
        judge_model = None if (args.no_judge or args.oracle) else (args.judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", ""))
        if args.oracle:
            model = "oracle (scripted reference agent, not a model)"
            config = agent_config_from_env({"ANTHROPIC_MODEL": "oracle"})
        else:
            config = agent_config_from_env({**os.environ, "ANTHROPIC_MODEL": model})
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ConfigError("ANTHROPIC_API_KEY is not set (put it in .env or the environment).")
            if judge_model is not None and not judge_model:
                raise ConfigError("no judge model: set ANTHROPIC_JUDGE_MODEL or pass --judge-model (or --no-judge)")
            if judge_model == model:
                raise ConfigError("the judge must be a different model from the one under test")
    except (ConfigError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    client = None if args.oracle else anthropic.Anthropic()
    judge = (client, judge_model) if judge_model else None
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9.-]+', '_', model)[:40]}"
    out_dir = args.out / run_id
    print(f"Running {len(scenarios)} scenario(s) x {args.repeat} against {model}; judge: {judge_model or 'off'}")

    results = run_suite(
        scenarios, OracleClient if args.oracle else (lambda s: client), config,
        load_shop(evals_dir / "shop.json"), SolverConfig(time_limit_s=args.solve_seconds, num_workers=1, seed=0),
        judge=judge, repeat=args.repeat, trace_dir=out_dir / "traces", progress=_progress(len(scenarios) * args.repeat),
    )
    meta = {"run_id": run_id, "model": model, "judge_model": judge_model, "scenarios": len(scenarios),
            "repeat": args.repeat, "solve_seconds": args.solve_seconds}
    write_results(out_dir, meta, results)
    print("\n" + render_table(summarize(results)) + f"\n\nReport: {out_dir / 'report.md'}")
    return 0 if all(r.passed for r in results) else 1


def _build(args: argparse.Namespace) -> int:
    path = args.evals_dir / "shop.json"
    print(f"Solving the baseline plan (up to {args.solve_seconds:g} s) and writing {path} ...")
    baseline = build_shop(path, args.solve_seconds)
    print(f"Done: baseline status {baseline.solve_info.status.value}. Commit {path} so every run starts from the same plan.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.evals", description=__doc__.strip())
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"), help="folder with scenarios/, shop.json (default ./evals)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scenarios against a model and score them")
    run.add_argument("--model", help="model under test (default: $ANTHROPIC_MODEL)")
    run.add_argument("--judge-model", help="grader model, must differ from --model (default: $ANTHROPIC_JUDGE_MODEL)")
    run.add_argument("--no-judge", action="store_true", help="skip the LLM judge (deterministic checks only)")
    run.add_argument("--oracle", action="store_true", help="use the scripted reference agent: no API calls, tests the eval itself")
    run.add_argument("--only", action="append", help="a scenario id or category; repeatable")
    run.add_argument("--repeat", type=int, default=1, help="runs per scenario (models vary between runs)")
    run.add_argument("--solve-seconds", type=float, default=5.0, help="solver time limit per reschedule")
    run.add_argument("--out", type=Path, default=Path("evals/results"))
    run.set_defaults(func=_run)

    build = sub.add_parser("build-shop", help="regenerate the fixture shop and its baseline plan (slow)")
    build.add_argument("--solve-seconds", type=float, default=30.0)
    build.set_defaults(func=_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
