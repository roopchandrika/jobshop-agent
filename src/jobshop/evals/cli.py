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
from jobshop.knowledge import KnowledgeBase
from jobshop.evals.compare import ModelSpec, render_comparison, run_comparison, write_comparison
from jobshop.evals.oracle import OracleClient
from jobshop.evals import replay, retrieval
from jobshop.evals.report import render_table, summarize, write_results
from jobshop.evals.runner import ScenarioResult, run_suite
from jobshop.evals.scenario import Scenario, load_scenarios
from jobshop.evals.shop import SHOPS, build_shop, load_shops, shop_path

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
    return scenarios


def _shops(args: argparse.Namespace, scenarios: list[Scenario]):
    """The fixtures the chosen scenarios need (and no others)."""
    try:
        return load_shops(args.evals_dir, {s.shop for s in scenarios})
    except (ValueError, FileNotFoundError) as e:
        raise ConfigError(str(e)) from None


def _knowledge(args: argparse.Namespace, scenarios: list[Scenario]):
    """The plant documents, if the folder exists. Scenarios that need them make a missing folder an error."""
    folder = args.knowledge_dir if args.knowledge_dir is not None else args.evals_dir.parent / "knowledge"
    if folder.is_dir():
        return KnowledgeBase.from_directory(folder)
    needing = sorted(s.id for s in scenarios if s.needs_knowledge)
    if needing:
        raise ConfigError(f"{', '.join(needing)} need the plant documents, but {folder} is not a folder (see --knowledge-dir)")
    return None


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
        replaying = args.replay is not None
        if replaying and (args.oracle or args.record):
            raise ConfigError("--replay plays back a recording; it cannot be combined with --oracle or --record")
        offline = args.oracle or replaying   # no model is called, so no key, no price and no judge
        judge_model = None if (args.no_judge or offline) else (args.judge_model or os.environ.get("ANTHROPIC_JUDGE_MODEL", ""))
        if args.oracle:
            model = "oracle (scripted reference agent, not a model)"
            config = agent_config_from_env({"ANTHROPIC_MODEL": "oracle"})
        elif replaying:
            have = replay.available(args.replay) if args.replay.is_dir() else {}
            missing = [sc.id for sc in scenarios if have.get(sc.id, 0) < args.repeat]
            if missing:
                raise ConfigError(f"{args.replay} has no recording of {', '.join(missing[:5])}"
                                  f"{'...' if len(missing) > 5 else ''} for {args.repeat} run(s); replay what was recorded (--only)")
            model = f"replay of {args.replay.name}"
            config = agent_config_from_env({"ANTHROPIC_MODEL": "replay"})
        else:
            config = _with_prices(agent_config_from_env({**os.environ, "ANTHROPIC_MODEL": model}), _prices(args.price))
            _need_key()
            if judge_model is not None and not judge_model:
                raise ConfigError("no judge model: set ANTHROPIC_JUDGE_MODEL or pass --judge-model (or --no-judge)")
            if judge_model == model:
                raise ConfigError("the judge must be a different model from the one under test")
        shops = _shops(args, scenarios)
        knowledge = _knowledge(args, scenarios)
    except (ConfigError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    client = None if offline else anthropic.Anthropic()
    judge = (client, judge_model) if judge_model else None
    if args.oracle:
        client_for = OracleClient
    elif replaying:
        client_for = replay.replay_factory(args.replay)
    else:
        client_for = lambda s: client  # noqa: E731
    if args.record:
        client_for = replay.recording_factory(args.record, client_for, model)
        print(f"Recording every model response to {args.record} (replay them later with --replay, at no cost)")
    run_id = _run_id(model)
    out_dir = args.out / run_id
    print(f"Running {len(scenarios)} scenario(s) x {args.repeat} against {model}; judge: {judge_model or 'off'}")

    results = run_suite(
        scenarios, client_for, config,
        shops, SolverConfig(time_limit_s=args.solve_seconds, num_workers=1, seed=0),
        judge=judge, repeat=args.repeat, trace_dir=out_dir / "traces", progress=_progress(len(scenarios) * args.repeat),
        knowledge=knowledge,
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
        shops = _shops(args, scenarios)
        knowledge = _knowledge(args, scenarios)
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
        scenarios, models, base, shops,
        SolverConfig(time_limit_s=args.solve_seconds, num_workers=1, seed=0),
        judge=judge, repeat=args.repeat, out_dir=out_dir, knowledge=knowledge,
        progress=lambda model, r: print(f"[{model}] {r.id:<8} {'PASS' if r.passed else 'FAIL'}", flush=True),
    )
    meta = {"run_id": run_id, "judge_model": judge_model, "scenarios": len(scenarios), "repeat": args.repeat,
            "solve_seconds": args.solve_seconds, "demo": args.demo}
    write_comparison(out_dir, meta, results)
    print("\n" + render_comparison(meta, results) + f"\nWritten to {out_dir}")
    return 0


def _retrieval(args: argparse.Namespace) -> int:
    """Score the document search on labelled questions: no model, no cost."""
    folder = args.knowledge_dir if args.knowledge_dir is not None else args.evals_dir.parent / "knowledge"
    questions_file = args.questions if args.questions is not None else args.evals_dir / "retrieval.yaml"
    try:
        kb = KnowledgeBase.from_directory(folder)
        outcomes = retrieval.evaluate(kb, retrieval.load_questions(questions_file), args.k)
    except (ValueError, OSError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    print(f"{len(kb.sources)} documents, {len(kb.chunks)} passages; questions from {questions_file}")
    print()
    print(retrieval.render(outcomes, args.k))
    ordinary = retrieval.score([o for o in outcomes if not o.question.hard], args.k)
    if args.min_hit is not None and ordinary.hit_at_k < args.min_hit:
        print(f"FAIL: ordinary-question hit@{args.k} {ordinary.hit_at_k:.2f} is below {args.min_hit:.2f}", file=sys.stderr)
        return 1
    return 0


def _build(args: argparse.Namespace) -> int:
    for name in args.shop or ["default"]:
        path = shop_path(args.evals_dir, name)
        print(f"Solving the baseline plan for the '{name}' shop (up to {args.solve_seconds:g} s) and writing {path} ...")
        baseline = build_shop(path, args.solve_seconds, name)
        print(f"Done: baseline status {baseline.solve_info.status.value}. Commit {path} so every run starts from the same plan.")
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", help="grader model; must not be a model under test (default: $ANTHROPIC_JUDGE_MODEL)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (deterministic checks only)")
    p.add_argument("--only", action="append", help="a scenario id or category; repeatable")
    p.add_argument("--repeat", type=int, default=1, help="runs per scenario (models vary between runs)")
    p.add_argument("--solve-seconds", type=float, default=5.0, help="solver time limit per reschedule")
    p.add_argument("--out", type=Path, default=Path("evals/results"))
    p.add_argument("--knowledge-dir", type=Path, help="plant documents for the search_knowledge tool (default: <evals-dir>/../knowledge)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.evals", description=__doc__.strip())
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"), help="folder with scenarios/, shop.json (default ./evals)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scenarios against one model and score them")
    _common(run)
    run.add_argument("--model", help="model under test (default: $ANTHROPIC_MODEL)")
    run.add_argument("--price", help="USD per million tokens as 'input,output', for cost numbers (default: $JOBSHOP_PRICE_*)")
    run.add_argument("--oracle", action="store_true", help="use the scripted reference agent: no API calls, tests the eval itself")
    run.add_argument("--record", type=Path, metavar="DIR", help="save every model response under DIR so the run can be replayed for free")
    run.add_argument("--replay", type=Path, metavar="DIR", help="re-run from a recording made with --record: no API calls, no cost; "
                     "a scenario whose prompt, tools or tool results changed since is reported as a stale recording")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", help="run the same scenarios against two models and compare quality, cost and latency")
    _common(compare)
    compare.add_argument("--model", action="append", help="a model to compare; give it twice (default: $ANTHROPIC_MODEL and $ANTHROPIC_COMPARE_MODEL)")
    compare.add_argument("--price", action="append", help="'MODEL=input,output' USD per million tokens; once per model, or cost shows n/a")
    compare.add_argument("--demo", action="store_true", help="compare two scripted agents (no API): shows the report, says nothing about models")
    compare.set_defaults(func=_compare)

    build = sub.add_parser("build-shop", help="regenerate the fixture shop and its baseline plan (slow)")
    build.add_argument("--solve-seconds", type=float, default=30.0)
    build.add_argument("--shop", action="append", choices=list(SHOPS),
                       help="which shop to rebuild; repeatable (default: the default shop). Rebuilding replaces a "
                            "committed fixture, so results from before are no longer comparable")
    build.set_defaults(func=_build)

    ret = sub.add_parser("retrieval", help="score the plant-document search on labelled questions (no model, no cost)")
    ret.add_argument("--k", type=int, default=3, help="how many passages count as 'found' (default 3)")
    ret.add_argument("--questions", type=Path, help="question file (default: <evals-dir>/retrieval.yaml)")
    ret.add_argument("--knowledge-dir", type=Path, help="plant documents (default: <evals-dir>/../knowledge)")
    ret.add_argument("--min-hit", type=float, help="exit 1 if the share of ordinary questions found in the top k is below this")
    ret.set_defaults(func=_retrieval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
