"""``python -m jobshop.api``: build the shop, connect the model if configured, and serve the UI."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import uvicorn
from dotenv import load_dotenv

from jobshop.agent.cli import ConfigError, agent_config_from_env, build_context
from jobshop.agent.memory import PreferenceError, preferences_from_env
from jobshop.api.app import LOOPBACK_HOSTS, create_app
from jobshop.core.generator import GeneratorSettings
from jobshop.core.solver import SolverConfig
from jobshop.evals.shop import DEFAULT_NOW, fresh_context, load_shop
from jobshop.knowledge import load_knowledge
from jobshop.tools.functions import ToolContext

DEFAULT_FIXTURE = Path("evals/shop.json")


def build_shop_context(args: argparse.Namespace, solver_config: SolverConfig) -> ToolContext:
    ctx = _shop_context(args, solver_config)
    ctx.knowledge = load_knowledge(os.environ)   # ./knowledge unless JOBSHOP_KNOWLEDGE_DIR says otherwise
    if ctx.knowledge is not None:
        print(f"Plant documents loaded: {len(ctx.knowledge.sources)} files, {len(ctx.knowledge.chunks)} passages.")
    return ctx


def _shop_context(args: argparse.Namespace, solver_config: SolverConfig) -> ToolContext:
    """The committed fixture shop if there is one (instant start), else generate and solve a fresh one."""
    now = args.now or DEFAULT_NOW
    if args.fixture.exists() and not args.generate:
        print(f"Using the fixture shop {args.fixture} (pass --generate for a fresh random one).")
        return fresh_context(load_shop(args.fixture), solver_config, now)
    print(f"Generating a shop (seed {args.seed}) and solving its baseline plan (up to {solver_config.time_limit_s:g} s)...")
    return build_context(
        GeneratorSettings(seed=args.seed, n_orders=args.orders, n_machines=args.machines), solver_config,
        datetime.strptime(now, "%Y-%m-%d %H:%M"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.api", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="must be a loopback address: there is no login")
    parser.add_argument("--container", action="store_true",
                        help="allow --host 0.0.0.0, for use INSIDE a container only; publish the port to loopback on the host: "
                             "docker run -p 127.0.0.1:8000:8000 ...")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--generate", action="store_true", help="ignore the fixture and generate a new random shop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orders", type=int, default=20)
    parser.add_argument("--machines", type=int, default=6)
    parser.add_argument("--now", help=f'plant time to start at (default "{DEFAULT_NOW}")')
    parser.add_argument("--solve-seconds", type=float, help="solver time limit per reschedule (default 10, or JOBSHOP_SOLVE_SECONDS)")
    parser.add_argument("--trace-dir", type=Path, default=Path("logs/traces"))
    args = parser.parse_args(argv)

    if args.host not in LOOPBACK_HOSTS and not (args.container and args.host == "0.0.0.0"):
        print(f"Refusing to listen on {args.host}: this app has no login, so it only runs on 127.0.0.1 / localhost "
              "(inside a container, --container --host 0.0.0.0 is allowed, with the port published to loopback only).", file=sys.stderr)
        return 2

    load_dotenv()
    seconds = args.solve_seconds or float(os.environ.get("JOBSHOP_SOLVE_SECONDS") or 10)
    solver_config = SolverConfig(time_limit_s=seconds)
    try:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ConfigError("ANTHROPIC_API_KEY is not set")
        config = agent_config_from_env(os.environ)
        client = anthropic.Anthropic()
    except ConfigError as e:
        print(f"No model configured ({e}). The page will show the plan, but chat is disabled.", file=sys.stderr)
        config, client = agent_config_from_env({"ANTHROPIC_MODEL": "none"}), None

    try:
        ctx = build_shop_context(args, solver_config)
    except ValueError as e:   # e.g. JOBSHOP_KNOWLEDGE_DIR names a folder that does not exist
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    trace = args.trace_dir / f"api-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    try:
        preferences = preferences_from_env(os.environ)
    except PreferenceError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    app = create_app(ctx, client, config, tracer_path=trace, preferences=preferences)
    print(f"Open http://{args.host}:{args.port}   (trace: {trace})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
