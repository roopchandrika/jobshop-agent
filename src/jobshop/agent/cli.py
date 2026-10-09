"""Chat CLI: ``python -m jobshop.agent.cli``.

This file is the *human-facing layer*. It is the only code that holds the ApprovalAuthority
and can mint an approval token, and it does so only after the planner types "y". The token
goes straight to ``commit_schedule``; it never appears in anything the model sees.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from jobshop.agent.conversation import Conversation
from jobshop.agent.loop import AgentConfig, FinalResponse, TurnResult
from jobshop.agent.trace import Tracer
from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.kpis import compute_kpis
from jobshop.core.solver import SolverConfig, solve
from jobshop.knowledge import Retriever, load_knowledge
from jobshop.tools import views
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.errors import ToolError
from jobshop.tools.human import commit_draft, kpi_lines
from jobshop.tools.functions import ToolContext
from jobshop.tools.store import Store
from jobshop.tools.text import terminal_safe

HELP = """Type a request, e.g. "Machine M4 is down from 14:00 to 17:00 today and order O-112 is now urgent."
Commands:  /state  show the live plan   /clock YYYY-MM-DD HH:MM  advance the shop clock
           /help   this message         /quit  leave"""


class ConfigError(Exception):
    pass


def agent_config_from_env(env: Mapping[str, str]) -> AgentConfig:
    model = env.get("ANTHROPIC_MODEL", "").strip()
    if not model:
        raise ConfigError("ANTHROPIC_MODEL is not set (put it in .env or the environment).")

    def number(name: str, cast: Callable[[str], Any]) -> Any:
        raw = env.get(name, "").strip()
        try:
            return cast(raw) if raw else None
        except ValueError:
            raise ConfigError(f"{name} must be a number, got '{raw}'") from None

    kwargs: dict[str, Any] = {}
    for key, name, cast in [
        ("max_steps", "JOBSHOP_MAX_STEPS", int),
        ("max_total_tokens", "JOBSHOP_MAX_TOKENS", int),
        ("max_cost_usd", "JOBSHOP_MAX_COST_USD", float),
        ("price_input_per_mtok", "JOBSHOP_PRICE_INPUT_PER_MTOK", float),
        ("price_output_per_mtok", "JOBSHOP_PRICE_OUTPUT_PER_MTOK", float),
    ]:
        value = number(name, cast)
        if value is not None:
            kwargs[key] = value
    try:
        return AgentConfig(model=model, **kwargs)
    except ValueError as e:
        raise ConfigError(str(e)) from None


def build_context(
    settings: GeneratorSettings, solver_config: SolverConfig, now: datetime | None = None,
    knowledge: Retriever | None = None,
) -> ToolContext:
    """Generate a shop, solve its baseline plan once, and make that the committed schedule."""
    instance = generate_instance(settings)
    baseline = solve(instance, config=solver_config)
    store = Store(instance, baseline)
    if now is not None:
        store.set_clock(instance.to_minutes(now))
    return ToolContext(store=store, authority=ApprovalAuthority(), solver_config=solver_config, knowledge=knowledge)


class ChatSession:
    def __init__(
        self,
        client: Any,
        ctx: ToolContext,
        config: AgentConfig,
        tracer: Tracer | None = None,
        out: Callable[[str], None] = print,
        ask: Callable[[str], str] = input,
    ) -> None:
        self.ctx = ctx
        self.conversation = Conversation(client, ctx, config, tracer)
        # Everything shown to the planner passes through terminal_safe, whatever its source: a
        # model's summary (or a note it echoed) must not be able to redraw the approval screen.
        self.out: Callable[[str], None] = lambda text: out(terminal_safe(text))
        self.ask = ask

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.conversation.messages

    # -- one line of input --------------------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            return self._command(line)

        result = self.conversation.say(line)
        self._show(result)
        if result.final is not None and result.final.needs_approval:
            self._offer_commit(result.final)
        return True

    # -- rendering ----------------------------------------------------------------------------

    def _show(self, result: TurnResult) -> None:
        final = result.final
        if final is None:
            self.out(f"\n[{result.status}] {result.text}")
            return
        self.out(f"\n{final.clarifying_question or final.summary}")
        if final.changes_made:
            self.out("\nChanges in the draft (recorded by the system):")
            self.out("\n".join(f"  - {c}" for c in final.changes_made))
        if final.goal and final.needs_approval:
            self.out(f"\nSolved for (recorded by the system): {views.GOAL_LABELS[final.goal]}")
        if final.kpi_before and final.kpi_after:
            self.out(f"\nKPIs (from the solver, not from the model):\n" + "\n".join(kpi_lines(final.kpi_before, final.kpi_after)))
        for warning in final.warnings:
            self.out(f"\n! {warning}")
        cost = f", ${result.cost_usd:.4f}" if result.cost_usd is not None else ""
        cached = f", {result.cache_read_tokens} read from cache" if result.cache_read_tokens else ""
        self.out(f"\n({result.steps} model calls, {result.total_tokens} tokens{cached}{cost})")

    # -- the human approval gate --------------------------------------------------------------

    def _offer_commit(self, final: FinalResponse) -> None:
        assert final.draft_id is not None
        draft = self.ctx.store.draft(final.draft_id)
        answer = self.ask(f"\nCommit draft {draft.id} to the live schedule? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            self.out(f"Not committed. Draft {draft.id} is kept; the live plan is unchanged.")
            self.conversation.notify(f"The planner chose NOT to commit draft {draft.id}; the live plan is unchanged.")
            return

        try:
            out = commit_draft(self.ctx, draft.id)
        except ToolError as e:
            self.out(f"Commit failed: {e}")
            self.conversation.notify(f"Committing draft {draft.id} failed: {e}")
            return
        self.out(f"Committed. The live plan is now version {out['new_version']}.")
        self.conversation.notify(f"The planner approved and committed draft {draft.id}; the live plan is now version {out['new_version']}.")

    # -- slash commands -----------------------------------------------------------------------

    def _command(self, line: str) -> bool:
        name, _, arg = line.partition(" ")
        if name in ("/quit", "/exit"):
            return False
        if name == "/help":
            self.out(HELP)
        elif name == "/state":
            c = self.ctx.store.committed
            kpis = views.kpi_view(c.instance, compute_kpis(c.instance, c.schedule), include_orders=False)
            self.out(f"Live plan version {c.version}, plant time {views.fmt(c.instance, c.instance.now)}")
            self.out(f"  late orders {kpis.late_orders}, total tardiness {kpis.total_tardiness_min} min, "
                     f"all orders done at {kpis.all_orders_done_at}, mean utilization {kpis.mean_utilization_pct}%")
        elif name == "/clock":
            self._set_clock(arg.strip())
        else:
            self.out(f"Unknown command {name}. Type /help.")
        return True

    def _set_clock(self, text: str) -> None:
        inst = self.ctx.store.committed.instance
        try:
            moment = datetime.strptime(text, "%Y-%m-%d %H:%M")
            version = self.ctx.store.set_clock(inst.to_minutes(moment)).version
        except ValueError as e:
            self.out(f"Could not set the clock: {e}. Use /clock YYYY-MM-DD HH:MM, moving forward only.")
            return
        self.out(f"Clock is now {text}. Live plan is version {version}; earlier drafts are stale.")
        self.conversation.notify(f"The plant clock was moved to {text}; any earlier drafts are now stale.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.agent.cli", description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="synthetic shop seed (default 0)")
    parser.add_argument("--orders", type=int, default=20)
    parser.add_argument("--machines", type=int, default=6)
    parser.add_argument("--now", help='plant time to start at, e.g. "2026-01-05 12:00" (default: plan start)')
    parser.add_argument("--solve-seconds", type=float, help="time limit per solve (default 30, or JOBSHOP_SOLVE_SECONDS)")
    parser.add_argument("--trace-dir", default="logs/traces")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every step as it happens")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        config = agent_config_from_env(os.environ)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ConfigError("ANTHROPIC_API_KEY is not set (put it in .env or the environment).")
        seconds = args.solve_seconds or float(os.environ.get("JOBSHOP_SOLVE_SECONDS") or 30)
        now = datetime.strptime(args.now, "%Y-%m-%d %H:%M") if args.now else None
    except (ConfigError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    solver_config = SolverConfig(time_limit_s=seconds)
    print(f"Generating shop (seed {args.seed}) and solving the baseline plan (up to {seconds:g} s)...")
    try:
        knowledge = load_knowledge(os.environ)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    ctx = build_context(GeneratorSettings(seed=args.seed, n_orders=args.orders, n_machines=args.machines), solver_config, now, knowledge)

    tracer = Tracer(
        path=Path(args.trace_dir) / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl",
        echo=(lambda r: print(f"  [trace] {r['event']} " + " ".join(f"{k}={v}" for k, v in r.items() if k in ('step', 'tool', 'stop_reason', 'tool_calls', 'is_error', 'latency_ms')))) if args.verbose else None,
    )
    session = ChatSession(anthropic.Anthropic(), ctx, config, tracer)
    print(f"Trace: {tracer.path}\n{HELP}\n")
    session.handle("/state")
    try:
        while session.handle(input("\nplanner> ")):
            pass
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
