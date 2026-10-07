"""``python -m jobshop.evals run | compare | build-shop``."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from jobshop.agent.cli import ConfigError, agent_config_from_env
from jobshop.agent.loop import AgentConfig
from jobshop.agent.pricing import Prices, parse_prices
from jobshop.core.solver import SolverConfig
from jobshop.evals.compare import ModelSpec, render_comparison, run_comparison, write_comparison
from jobshop.evals.oracle import OracleClient
from jobshop.evals.report import render_table, summarize, write_results
from jobshop.evals.runner import ScenarioResult, run_suite
from jobshop.evals.scenario import Scenario, load_scenarios
from jobshop.evals.shop import build_shop, load_shop

# Per-model money settings must never leak from one model to another through the environment.
_ENV_MONEY = ("JOBSHOP_PRICE_INPUT_PER_MTOK", "JOBSHOP_PRICE_OUTPUT_PER_MTOK", "JOBSHOP_MAX_COST_USD")


def _progress(total: int, prefix: str = ""):
    done = 0

    def show(r: ScenarioResult) -> None:
        nonlocal done
        done += 1
        failed = [name for name, c in r.checks.items() if c["passed"] is False]
        verdict = "PASS" if r.passed else "FAIL " + ",".join(failed or [r.status])
        print(f"{prefix}[{done:>3}/{total}] {r.id:<8} {verdict}", flush=True)

    return show


def _scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios = load_scenarios(args.evals_dir / "scenarios")
    if args.only:
        wanted = set(args.only)
        scenarios = [s for s in scenarios if s.id in wanted or s.category in wanted]
        if not scenarios:
            raise ConfigError(f"--only {args.only} matches no scenario id or category")
    if not (args.evals_dir / "shop.json").exists():
        raise ConfigError(f"{args.evals_dir / 'shop.json'} is missing; create it with: python -m jobshop.evals build-shop")
    return scenarios


def _run_id(label: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9.-]+', '_', label)[:40]}"


def _need_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError("ANTHROPIC_API_KEY is not set (put it in .env or the environment).")


def _prices(text: str | None) -> Prices | None:
    return None if text is None else parse_prices(text)


def _with_prices(config: AgentConfig, prices: Prices | None) -> AgentConfig:
    if prices is None:
        return config
    return replace(config, price_input_per_mtok=prices.input_per_mtok, price_output_per_mtok=prices.output_per_mtok)


# -- run ---------------------------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    load_dotenv()
    try:
        scenarios = _scenarios(args)
        model = args.model or os.environ.get("ANTHROPIC_MODEL", "")
        judge_model = None if (args.no_judge or args.oracle) else (args.judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", ""))
        if args.oracle:
            model = "oracle (scripted reference agent, not a model)"
            config = agent_config_from_env({"ANTHROPIC_MODEL": "oracle"})
        else:
            config = _with_prices(agent_config_from_env({**os.environ, "ANTHROPIC_MODEL": model}), _prices(args.price))
            _need_key()
            if judge_model is not None and not judge_model:
                raise ConfigError("no judge model: set ANTHROPIC_JUDGE_MODEL or pass --judge-model (or --no-judge)")
            if judge_model == model:
                raise ConfigError("the judge must be a different model from the one under test")
    except (ConfigError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    client = None if args.oracle else anthropic.Anthropic()
    judge = (client, judge_model) if judge_model else None
    run_id = _run_id(model)
    out_dir = args.out / run_id
    print(f"Running {len(scenarios)} scenario(s) x {args.repeat} against {model}; judge: {judge_model or 'off'}")

    results = run_suite(
        scenarios, OracleClient if args.oracle else (lambda s: client), config,
        load_shop(args.evals_dir / "shop.json"), SolverConfig(time_limit_s=args.solve_seconds, num_workers=1, seed=0),
        judge=judge, repeat=args.repeat, trace_dir=out_dir / "traces", progress=_progress(len(scenarios) * args.repeat),
    )
    meta = {"run_id": run_id, "model": model, "judge_model": judge_model, "scenarios": len(scenarios),
            "repeat": args.repeat, "solve_seconds": args.solve_seconds}
    write_results(out_dir, meta, results)
    print("\n" + render_table(summarize(results)) + f"\n\nReport: {out_dir / 'report.md'}")
    return 0 if all(r.passed for r in results) else 1


# -- compare -----------------------------------------------------------------------------------------


def _demo_models() -> list[ModelSpec]:
    return [
        ModelSpec("demo-careful", lambda s: OracleClient(s, usage=(1800, 250), delay_s=0.01), Prices(3.0, 15.0)),
        ModelSpec("demo-sloppy", lambda s: OracleClient(s, sloppy=True, usage=(1000, 120), delay_s=0.005), Prices(1.0, 5.0)),
    ]


def _compare(args: argparse.Namespace) -> int:
    load_dotenv()
    client = None
    try:
        scenarios = _scenarios(args)
        env = {k: v for k, v in os.environ.items() if k not in _ENV_MONEY}
        if args.demo:
            models, judge_model = _demo_models(), None
            base = agent_config_from_env({**env, "ANTHROPIC_MODEL": "demo"})
        else:
            names = args.model or [m for m in (os.environ.get("ANTHROPIC_MODEL"), os.environ.get("ANTHROPIC_COMPARE_MODEL")) if m]
            if len(set(names)) < 2:
                raise ConfigError("compare needs two different models: pass --model A --model B (or set ANTHROPIC_MODEL and ANTHROPIC_COMPARE_MODEL)")
            if len(set(names)) != len(names):
                raise ConfigError("the same model was given twice")
            _need_key()
            prices: dict[str, Prices] = {}
            for item in args.price or []:
                name, _, text = item.rpartition("=")
                if name not in names:
                    raise ConfigError(f"--price {item!r}: '{name}' is not one of the models being compared")
                prices[name] = parse_prices(text)
            judge_model = None if args.no_judge else (args.judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", ""))
            if judge_model is not None and not judge_model:
                raise ConfigError("no judge model: set ANTHROPIC_JUDGE_MODEL or pass --judge-model (or --no-judge)")
            if judge_model in names:
                raise ConfigError(f"the judge ({judge_model}) must not be one of the models being compared")
            client = anthropic.Anthropic()
            models = [ModelSpec(n, lambda s: client, prices.get(n)) for n in names]
            base = agent_config_from_env({**env, "ANTHROPIC_MODEL": names[0]})
    except (ConfigError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    names = [m.name for m in models]
    run_id = _run_id("-vs-".join(names))
    out_dir = args.out / run_id
    judge = (client, judge_model) if judge_model else None
    if args.demo:
        print("DEMO: two scripted agents with made-up prices. The numbers show the report, not any model.")
    print(f"Comparing {', '.join(names)} on {len(scenarios)} scenario(s) x {args.repeat}; judge: {judge_model or 'off'}")

    results = run_comparison(
        scenarios, models, base, load_shop(args.evals_dir / "shop.json"),
        SolverConfig(time_limit_s=args.solve_seconds, num_workers=1, seed=0),
        judge=judge, repeat=args.repeat, out_dir=out_dir,
        progress=lambda model, r: print(f"[{model}] {r.id:<8} {'PASS' if r.passed else 'FAIL'}", flush=True),
    )
    meta = {"run_id": run_id, "judge_model": judge_model, "scenarios": len(scenarios), "repeat": args.repeat,
            "solve_seconds": args.solve_seconds, "demo": args.demo}
    write_comparison(out_dir, meta, results)
    print("\n" + render_comparison(meta, results) + f"\nWritten to {out_dir}")
    return 0


def _build(args: argparse.Namespace) -> int:
    path = args.evals_dir / "shop.json"
    print(f"Solving the baseline plan (up to {args.solve_seconds:g} s) and writing {path} ...")
    baseline = build_shop(path, args.solve_seconds)
    print(f"Done: baseline status {baseline.solve_info.status.value}. Commit {path} so every run starts from the same plan.")
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", help="grader model; must not be a model under test (default: $ANTHROPIC_JUDGE_MODEL)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (deterministic checks only)")
    p.add_argument("--only", action="append", help="a scenario id or category; repeatable")
    p.add_argument("--repeat", type=int, default=1, help="runs per scenario (models vary between runs)")
    p.add_argument("--solve-seconds", type=float, default=5.0, help="solver time limit per reschedule")
    p.add_argument("--out", type=Path, default=Path("evals/results"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.evals", description=__doc__.strip())
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"), help="folder with scenarios/, shop.json (default ./evals)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scenarios against one model and score them")
    _common(run)
    run.add_argument("--model", help="model under test (default: $ANTHROPIC_MODEL)")
    run.add_argument("--price", help="USD per million tokens as 'input,output', for cost numbers (default: $JOBSHOP_PRICE_*)")
    run.add_argument("--oracle", action="store_true", help="use the scripted reference agent: no API calls, tests the eval itself")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", help="run the same scenarios against two models and compare quality, cost and latency")
    _common(compare)
    compare.add_argument("--model", action="append", help="a model to compare; give it twice (default: $ANTHROPIC_MODEL and $ANTHROPIC_COMPARE_MODEL)")
    compare.add_argument("--price", action="append", help="'MODEL=input,output' USD per million tokens; once per model, or cost shows n/a")
    compare.add_argument("--demo", action="store_true", help="compare two scripted agents (no API): shows the report, says nothing about models")
    compare.set_defaults(func=_compare)

    build = sub.add_parser("build-shop", help="regenerate the fixture shop and its baseline plan (slow)")
    build.add_argument("--solve-seconds", type=float, default=30.0)
    build.set_defaults(func=_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
