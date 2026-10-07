"""The human-facing command for the MCP setup: ``python -m jobshop.mcp_server.admin <command>``.

  init      create the shared state (generates a synthetic shop and solves its baseline plan)
  status    show the live plan, drafts and pending approval requests
  approve   review a pending request and, if you say yes, commit it
  deny      decline a pending request
  clock     move the shop clock forward

Approval is deliberately a separate step from the chat: the model can ask (``request_commit``)
but only a person running ``approve`` can commit. The review screen is built from the stored
schedules, not from anything the model wrote.

``approve`` never holds the state lock while it waits for you to decide. It re-checks
everything (still pending, plan unchanged, draft unchanged) at the moment you say yes.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from jobshop.core.generator import GeneratorSettings, generate_instance
from jobshop.core.kpis import compute_kpis
from jobshop.core.solver import SolverConfig, solve
from jobshop.mcp_server.server import state_path_from_env
from jobshop.tools import human, views
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import ToolContext
from jobshop.tools.store import Store
from jobshop.tools.text import terminal_safe
from jobshop.tools.views import KPIView


def _context(path: Path) -> tuple[Store, ToolContext]:
    store = Store.open(path)
    # This process's own authority: it issues a token and consumes it immediately, here.
    return store, ToolContext(store=store, authority=ApprovalAuthority(), solver_config=SolverConfig())


def cmd_init(args: argparse.Namespace, out: Callable[[str], None], ask: Callable[[str], str]) -> int:
    path: Path = args.state
    if path.exists():
        if not args.force:
            out(f"{path} already exists. Use --force to replace it (this discards its drafts and requests).")
            return 1
        path.unlink()
        path.with_name(path.name + ".lock").unlink(missing_ok=True)

    instance = generate_instance(GeneratorSettings(seed=args.seed, n_orders=args.orders, n_machines=args.machines))
    ops = sum(len(o.operations) for o in instance.orders)
    out(f"Solving the baseline plan for {len(instance.orders)} orders / {ops} operations (up to {args.solve_seconds:g} s)...")
    baseline = solve(instance, config=SolverConfig(time_limit_s=args.solve_seconds, num_workers=args.workers))
    store = Store.create(path, instance, baseline)
    if args.now:
        with store.transaction():
            store.set_clock(instance.to_minutes(datetime.strptime(args.now, "%Y-%m-%d %H:%M")))
    out(f"Created {path} (baseline status {baseline.solve_info.status.value}).")
    return 0


def cmd_status(args: argparse.Namespace, out: Callable[[str], None], ask: Callable[[str], str]) -> int:
    store, ctx = _context(args.state)
    with store.transaction():
        c = store.committed
        k = views.kpi_view(c.instance, compute_kpis(c.instance, c.schedule), include_orders=False)
        out(f"Live plan version {c.version}, plant time {views.fmt(c.instance, c.instance.now)}")
        out(f"  late orders {k.late_orders}, total tardiness {k.total_tardiness_min} min, all orders done at {k.all_orders_done_at}")
        out(f"Drafts: {', '.join(d.id for d in store.drafts()) or 'none'}")
        pending = human.pending_requests(ctx)
        out(f"Pending approval requests: {', '.join(r.id + ' (draft ' + r.draft_id + ')' for r in pending) or 'none'}")
    return 0


def _pick_request(ctx: ToolContext, requested: str | None, out: Callable[[str], None]) -> str | None:
    pending = human.pending_requests(ctx)
    if requested:
        return requested
    if not pending:
        out("No pending approval requests.")
        return None
    if len(pending) > 1:
        out("Several requests are pending; name one: " + ", ".join(r.id for r in pending))
        return None
    return pending[0].id


def _review(info: dict, out: Callable[[str], None]) -> None:
    req, cmp_ = info["request"], info["comparison"]
    diff = cmp_["diff"]
    out(f"\nApproval request {req['id']} for draft {req['draft_id']}")
    out("Changes in the draft:\n" + "\n".join(f"  - {c}" for c in info["changes"]))
    out("\nKPIs (computed from the stored schedules, not from the model):")
    out("\n".join(human.kpi_lines(KPIView.model_validate(diff["kpi_before"]), KPIView.model_validate(diff["kpi_after"]))))
    out(f"\nOperations moved: {diff['moved_operation_count']} ({diff['machine_change_count']} changed machine)")
    out(f"Orders newly late: {', '.join(diff['newly_late_orders']) or 'none'}   "
        f"No longer late: {', '.join(diff['no_longer_late_orders']) or 'none'}")
    after = cmp_["solve_after"]
    out(f"Solver status for the draft: {after['status']} "
        f"(tardiness proven: {after['tardiness_proven_optimal']}, minimal change proven: {after['stability_proven_optimal']})")
    if cmp_["confidence_note"]:
        out(f"Note: {cmp_['confidence_note']}")


def cmd_approve(args: argparse.Namespace, out: Callable[[str], None], ask: Callable[[str], str]) -> int:
    store, ctx = _context(args.state)
    with store.transaction():  # read-only: show what would be committed
        request_id = _pick_request(ctx, args.request_id, out)
        if request_id is None:
            return 0 if not human.pending_requests(ctx) else 1
        info = human.describe(ctx, request_id)
    if info["status"] != "pending":
        out(f"Request {request_id} is {info['status']}, not pending.")
        return 1

    _review(info, out)
    if ask("\nCommit this draft as the live schedule? [y/N] ").strip().lower() not in ("y", "yes"):
        out("Not committed. The request stays pending; run `deny` to decline it.")
        return 0
    try:
        with store.transaction():  # re-validates everything at the moment of approval
            result = human.approve(ctx, request_id)
    except ToolError as e:
        out(f"Could not commit: {e}")
        return 1
    out(f"Committed. The live plan is now version {result['new_version']}.")
    return 0


def cmd_deny(args: argparse.Namespace, out: Callable[[str], None], ask: Callable[[str], str]) -> int:
    store, ctx = _context(args.state)
    try:
        with store.transaction():
            request_id = _pick_request(ctx, args.request_id, out)
            if request_id is None:
                return 1
            human.deny(ctx, request_id)
    except ToolError as e:
        out(f"Could not deny: {e}")
        return 1
    out(f"Request {request_id} denied. The live plan is unchanged.")
    return 0


def cmd_clock(args: argparse.Namespace, out: Callable[[str], None], ask: Callable[[str], str]) -> int:
    store, _ = _context(args.state)
    try:
        moment = datetime.strptime(args.time, "%Y-%m-%d %H:%M")
        with store.transaction():
            version = store.set_clock(store.committed.instance.to_minutes(moment)).version
    except ValueError as e:
        out(f"Could not set the clock: {e}. Use YYYY-MM-DD HH:MM, moving forward only.")
        return 1
    out(f"Clock is now {args.time}. Live plan is version {version}; existing drafts and requests are stale.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m jobshop.mcp_server.admin", description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=None, help="state file (default: $JOBSHOP_STATE or ~/.jobshop/state.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create the shared state")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--orders", type=int, default=20)
    init.add_argument("--machines", type=int, default=6)
    init.add_argument("--now", help='start the clock at e.g. "2026-01-05 12:00"')
    init.add_argument("--solve-seconds", type=float, default=30.0)
    init.add_argument("--workers", type=int, default=8)
    init.add_argument("--force", action="store_true", help="replace an existing state file")
    init.set_defaults(func=cmd_init)

    sub.add_parser("status", help="show the live plan and pending requests").set_defaults(func=cmd_status)
    for name, func, help_ in (("approve", cmd_approve, "review and commit a request"), ("deny", cmd_deny, "decline a request")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("request_id", nargs="?", help="e.g. R1 (optional if only one is pending)")
        p.set_defaults(func=func)
    clock = sub.add_parser("clock", help="move the shop clock forward")
    clock.add_argument("time", help='"YYYY-MM-DD HH:MM"')
    clock.set_defaults(func=cmd_clock)
    return parser


def main(argv: list[str] | None = None, out: Callable[[str], None] = print, ask: Callable[[str], str] = input) -> int:
    args = build_parser().parse_args(argv)

    def safe_out(text: str) -> None:  # same reason as in the chat CLI: nothing can redraw the review screen
        out(terminal_safe(text))

    args.state = args.state or state_path_from_env(os.environ)
    if args.command != "init" and not args.state.exists():
        safe_out(f"No state at {args.state}. Run `init` first.")
        return 2
    return args.func(args, safe_out, ask)


if __name__ == "__main__":
    raise SystemExit(main())
