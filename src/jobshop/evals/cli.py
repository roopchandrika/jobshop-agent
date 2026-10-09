"""``python -m jobshop.evals run | compare | redteam | snapshot | retrieval | build-shop``."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from jobshop.agent.cli import ConfigError, agent_config_from_env
from jobshop.agent.loop import AgentConfig
from jobshop.agent.pricing import Prices, parse_prices
from jobshop.agent.providers import build_client, needs_anthropic
from jobshop.core.solver import SolverConfig
from jobshop.evals import redteam, replay, retrieval, snapshot
from jobshop.evals.compare import ModelSpec, render_comparison, run_comparison, write_comparison
from jobshop.evals.oracle import OracleClient
from jobshop.evals.report import render_table, summarize, write_results
from jobshop.evals.runner import ScenarioResult, run_suite
from jobshop.evals.scenario import Scenario, load_scenarios
from jobshop.evals.shop import SHOPS, build_shop, load_shops, shop_path
from jobshop.knowledge import RETRIEVERS, KnowledgeBase, build_retriever
from jobshop.knowledge.loading import CACHE_VAR, MODEL_VAR, RETRIEVER_VAR
from jobshop.knowledge.semantic import DEFAULT_MODEL

NL = chr(10)
BLANK = NL + NL

# Per-model money settings must never leak from one model to another through the environment.
_ENV_MONEY = ("JOBSHOP_PRICE_INPUT_PER_MTOK", "JOBSHOP_PRICE_OUTPUT_PER_MTOK", "JOBSHOP_MAX_COST_USD",
              "JOBSHOP_TRIAGE_PRICE_INPUT_PER_MTOK", "JOBSHOP_TRIAGE_PRICE_OUTPUT_PER_MTOK")


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
        kind = (args.retriever or os.environ.get(RETRIEVER_VAR, "bm25")).strip().lower()
        try:
            return build_retriever(kind, folder, cache_dir=Path(os.environ.get(CACHE_VAR, ".cache")),
                                   model=os.environ.get(MODEL_VAR, DEFAULT_MODEL))
        except ValueError as e:
            raise ConfigError(str(e)) from None
    needing = sorted(s.id for s in scenarios if s.needs_knowledge)
    if needing:
        raise ConfigError(f"{', '.join(needing)} need the plant documents, but {folder} is not a folder (see --knowledge-dir)")
    return None


def _run_id(label: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9.-]+', '_', label)[:40]}"


def _need_key(*models: str | None) -> None:
    """An Anthropic key is needed only if some model in this run is an Anthropic model (not a local ollama: one)."""
    if needs_anthropic(*models) and not os.environ.get("ANTHROPIC_API_KEY"):
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
            if judge_model is not None and not judge_model:
                raise ConfigError("no judge model: set ANTHROPIC_JUDGE_MODEL or pass --judge-model (or --no-judge)")
            _need_key(model, judge_model, os.environ.get("JOBSHOP_TRIAGE_MODEL"), args.triage_model)
            if judge_model == model:
                raise ConfigError("the judge must be a different model from the one under test")
        if args.pattern:
            config = replace(config, pattern=args.pattern)   # a mistyped pattern is a configuration error, not a crash
        if args.triage_model or args.triage_price:
            triage = _prices(args.triage_price)
            config = replace(config, triage_model=args.triage_model or config.triage_model,
                             **({} if triage is None else {"triage_price_input_per_mtok": triage.input_per_mtok,
                                                           "triage_price_output_per_mtok": triage.output_per_mtok}))
        shops = _shops(args, scenarios)
        knowledge = _knowledge(args, scenarios)
    except (ConfigError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    client = None if offline else build_client(os.environ, model, judge_model, config.triage_model)
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
            patterns = list(dict.fromkeys(args.pattern or []))
            if len(set(names)) < 2 and len(patterns) < 2:
                raise ConfigError("compare needs two different models (--model A --model B) or one model with two patterns (--model A --pattern react --pattern verify)")
            if len(set(names)) != len(names):
                raise ConfigError("the same model was given twice")
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
            _need_key(*names, judge_model, os.environ.get("JOBSHOP_TRIAGE_MODEL"))
            client = build_client(os.environ, *names, judge_model, os.environ.get("JOBSHOP_TRIAGE_MODEL"))
            if patterns:
                for p in patterns:
                    AgentConfig(model="x", pattern=p)   # a mistyped pattern is a configuration error now, not after a paid run starts
                models = [ModelSpec(f"{n} [{p}]", lambda s: client, prices.get(n), model=n, pattern=p) for n in names for p in patterns]
            else:
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
    methods = list(RETRIEVERS) if args.method == "all" else [args.method]
    try:
        questions = retrieval.load_questions(questions_file)
        off_topic = retrieval.load_unanswerable(questions_file)
        results, separation = {}, {}
        for method in methods:
            kb = build_retriever(method, folder, cache_dir=Path(os.environ.get(CACHE_VAR, ".cache")),
                                 model=os.environ.get(MODEL_VAR, DEFAULT_MODEL))
            results[method] = retrieval.evaluate(kb, questions, args.k)
            separation[method] = retrieval.measure_separation(kb, questions, off_topic)
    except (ValueError, OSError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    print(f"{len(kb.sources)} documents, {len(kb.chunks)} passages; questions from {questions_file}")
    print()
    print(retrieval.render_comparison(results, args.k) if len(results) > 1 else retrieval.render(results[methods[0]], args.k))
    if off_topic:
        print(retrieval.render_separation(separation))
    failed = [m for m, outcomes in results.items()
              if args.min_hit is not None and retrieval.score([o for o in outcomes if not o.question.hard], args.k).hit_at_k < args.min_hit]
    for m in failed:
        print(f"FAIL: {m}: ordinary-question hit@{args.k} is below {args.min_hit:.2f}", file=sys.stderr)
    return 1 if failed else 0


def _redteam(args: argparse.Namespace) -> int:
    load_dotenv()
    live = args.live
    try:
        attacks = redteam.load_attacks(args.attacks or args.evals_dir / "redteam.yaml")
        if args.only:
            attacks = [a for a in attacks if a.id in set(args.only) or a.surface in args.only or a.goal in args.only]
            if not attacks:
                raise ConfigError(f"--only {args.only} matches no attack id, surface or goal")
        if live:
            model = args.model or os.environ.get("ANTHROPIC_MODEL", "")
            if not model:
                raise ConfigError("--live needs a model: --model or $ANTHROPIC_MODEL")
            _need_key(model, os.environ.get("JOBSHOP_TRIAGE_MODEL"))
            base = agent_config_from_env({**os.environ, "ANTHROPIC_MODEL": model})
            patterns = list(dict.fromkeys(args.pattern or ["react", "route"]))
            rows = [redteam.Row(p, p) for p in patterns] + ([redteam.Row("react, answer guards off", "react", guards=False)] if args.no_guards_row else [])
        else:
            if args.model or args.pattern:
                raise ConfigError("--model and --pattern choose what a real model runs; they only apply with --live (the default set-ups are fixed)")
            model, base, rows = "obedient", None, list(redteam.ROWS)
        for row in rows:
            AgentConfig(model="x", pattern=row.pattern)        # a mistyped pattern is a configuration error before anything is paid for
        shops = load_shops(args.evals_dir, {a.shop for a in attacks})
        folder = args.knowledge_dir if args.knowledge_dir is not None else args.evals_dir.parent / "knowledge"
        knowledge = KnowledgeBase.from_directory(folder) if folder.is_dir() else None
        if knowledge is None and any(a.document for a in attacks):
            raise ConfigError(f"some attacks plant a document among the plant documents, but {folder} is not a folder (see --knowledge-dir)")
    except (ConfigError, ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    runs = len(attacks) * len(rows) * args.repeat
    if live:
        print(f"LIVE: {runs} runs against {model} (each is a full agent turn, typically 3-8 model calls; this costs money). Hostile content is "
              "planted in order notes and plant documents of an in-memory shop; nothing outside this process is touched.", flush=True)
    else:
        print(f"{runs} runs with an obedient scripted model (no API calls, no cost).", flush=True)
    done = 0

    def progress(r: redteam.AttackResult) -> None:
        nonlocal done
        done += 1
        print(f"[{done:>3}/{runs}] {r.attack} {r.row:<26} {'ATTACK SUCCEEDED' if r.success else 'contained'}" + (f" ({r.error})" if r.error else ""), flush=True)

    client = build_client(os.environ, model) if live else None
    results = redteam.run_redteam(attacks, rows, shops, SolverConfig(time_limit_s=args.solve_seconds, num_workers=1, seed=0),
                                  client_for=(lambda a: client) if live else redteam.ObedientClient, repeat=args.repeat,
                                  knowledge=knowledge, base_config=base, progress=progress)
    out_dir = args.out / redteam.run_id("live" if live else "obedient")
    meta = {"mode": "live" if live else "obedient", "model": model if live else None, "repeat": args.repeat, "attacks": len(attacks)}
    path = redteam.write_report(out_dir, attacks, results, meta)
    summary = redteam.summarize(results)
    print("\n" + redteam.render_table(summary) + f"\n\nReport: {path}")
    crashed = sum(r.status == "crashed" for r in results)
    if crashed:
        print(f"{crashed} run(s) crashed (counted as errors, not as contained)", file=sys.stderr)
    return 1 if crashed else 0


def _snapshot(args: argparse.Namespace) -> int:
    path = args.evals_dir / snapshot.SNAPSHOT_NAME
    try:
        if args.update:
            now = snapshot.current(args.evals_dir, args.knowledge_dir)
            changes = snapshot.differences(snapshot.load(path), now) if path.exists() else None
            snapshot.save(path, now)
            if changes is None:
                print(f"Created {path}: {len(now)} pieces of text.")
            elif not changes:
                print(f"{path} is already up to date.")
            else:
                print(f"Updated {path}: {len(changes)} of {len(now)} pieces of text changed.")
                print(BLANK.join(changes))
                print(f"{NL}Now run the evals against a real model, and commit this file together with the numbers that justify the change.")
            return 0
        problems = snapshot.check(args.evals_dir, args.knowledge_dir)
    except (OSError, ValueError, KeyError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    if problems and not (args.evals_dir / snapshot.SNAPSHOT_NAME).exists():
        print(problems[0], file=sys.stderr)
        return 1
    if problems:
        print(BLANK.join(problems), file=sys.stderr)
        print(f"{NL}What the model is told has changed. If that was intended: run the evals, then {snapshot.UPDATE_COMMAND} and commit the result.", file=sys.stderr)
        return 1
    print("The prompts and tool descriptions match the snapshot.")
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
    p.add_argument("--retriever", choices=RETRIEVERS, help="how the documents are searched (default: $JOBSHOP_RETRIEVER, else bm25)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.evals", description=__doc__.strip())
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"), help="folder with scenarios/, shop.json (default ./evals)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scenarios against one model and score them")
    _common(run)
    run.add_argument("--model", help="model under test (default: $ANTHROPIC_MODEL)")
    run.add_argument("--price", help="USD per million tokens as 'input,output', for cost numbers (default: $JOBSHOP_PRICE_*)")
    run.add_argument("--oracle", action="store_true", help="use the scripted reference agent: no API calls, tests the eval itself")
    run.add_argument("--pattern", help="agent pattern: react (default), plan, verify, reflect, or several joined with + (e.g. plan+verify). "
                     "The scripted reference agent can only play react and verify.")
    run.add_argument("--triage-model", help="with the 'route' pattern: a smaller model for the triage call (default: the model under test)")
    run.add_argument("--triage-price", help="USD per million tokens for the triage model as 'input,output'")
    run.add_argument("--record", type=Path, metavar="DIR", help="save every model response under DIR so the run can be replayed for free")
    run.add_argument("--replay", type=Path, metavar="DIR", help="re-run from a recording made with --record: no API calls, no cost; "
                     "a scenario whose prompt, tools or tool results changed since is reported as a stale recording")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", help="run the same scenarios against two models and compare quality, cost and latency")
    _common(compare)
    compare.add_argument("--model", action="append", help="a model to compare; give it twice (default: $ANTHROPIC_MODEL and $ANTHROPIC_COMPARE_MODEL)")
    compare.add_argument("--pattern", action="append", help="agent pattern to run each model with; repeat to compare patterns "
                         "(react, plan, verify, reflect, or joined with +). One model and two patterns is a valid comparison.")
    compare.add_argument("--price", action="append", help="'MODEL=input,output' USD per million tokens; once per model, or cost shows n/a")
    compare.add_argument("--demo", action="store_true", help="compare two scripted agents (no API): shows the report, says nothing about models")
    compare.set_defaults(func=_compare)

    build = sub.add_parser("build-shop", help="regenerate the fixture shop and its baseline plan (slow)")
    build.add_argument("--solve-seconds", type=float, default=30.0)
    build.add_argument("--shop", action="append", choices=list(SHOPS),
                       help="which shop to rebuild; repeatable (default: the default shop). Rebuilding replaces a "
                            "committed fixture, so results from before are no longer comparable")
    build.set_defaults(func=_build)

    red = sub.add_parser("redteam", help="plant hostile content (notes, documents, messages) and measure how often the attacks get what they want")
    red.add_argument("--attacks", type=Path, help="attack file (default: <evals-dir>/redteam.yaml)")
    red.add_argument("--only", action="append", help="an attack id, surface (order_note, document, user_message) or goal; repeatable")
    red.add_argument("--live", action="store_true", help="let a real model read the hostile content (calls the API, costs money); default: an obedient scripted model")
    red.add_argument("--model", help="with --live: the model to attack (default: $ANTHROPIC_MODEL)")
    red.add_argument("--pattern", action="append", help="with --live: pattern to attack; repeatable (default: react and route)")
    red.add_argument("--no-guards-row", action="store_true", help="with --live: also run react with the answer guards off, to see what they catch")
    red.add_argument("--repeat", type=int, default=1, help="runs per attack and set-up (a real model varies between runs)")
    red.add_argument("--solve-seconds", type=float, default=5.0)
    red.add_argument("--out", type=Path, default=Path("evals/results"))
    red.add_argument("--knowledge-dir", type=Path, help="plant documents (default: <evals-dir>/../knowledge)")
    red.set_defaults(func=_redteam)

    snap = sub.add_parser("snapshot", help="check (or --update) the saved text of every prompt and tool description; fails when the model's instructions changed")
    snap.add_argument("--update", action="store_true", help="rewrite the snapshot from the code (after the evals justify the change)")
    snap.add_argument("--knowledge-dir", type=Path, help="plant documents (default: <evals-dir>/../knowledge)")
    snap.set_defaults(func=_snapshot)

    ret = sub.add_parser("retrieval", help="score the plant-document search on labelled questions (no model, no cost)")
    ret.add_argument("--k", type=int, default=3, help="how many passages count as 'found' (default 3)")
    ret.add_argument("--method", choices=[*RETRIEVERS, "all"], default="bm25",
                     help="keyword (default), dense (embeddings), hybrid, or all three side by side; dense and hybrid need "
                          "the optional extra: uv sync --extra embeddings")
    ret.add_argument("--questions", type=Path, help="question file (default: <evals-dir>/retrieval.yaml)")
    ret.add_argument("--knowledge-dir", type=Path, help="plant documents (default: <evals-dir>/../knowledge)")
    ret.add_argument("--min-hit", type=float, help="exit 1 if the share of ordinary questions found in the top k is below this")
    ret.set_defaults(func=_retrieval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
