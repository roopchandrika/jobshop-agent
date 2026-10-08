"""HUMAN-FACING operations: review, approve or deny an approval request.

Nothing in this module may ever be registered as a model tool. It is called only by code a
person drives (the approval command today; the web Approve button in Phase 7). It is the one
place outside the chat CLI that mints approval tokens, and it mints one only after a person has
decided, for the exact schedule that was requested.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from jobshop.tools.approval import proposal_digest
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import CompareInput, ToolContext, compare_schedules
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.store import ApprovalRequest
from jobshop.tools.views import KPIView


def kpi_lines(before: KPIView, after: KPIView) -> list[str]:
    rows = [
        ("late orders", before.late_orders, after.late_orders),
        ("total tardiness (min)", before.total_tardiness_min, after.total_tardiness_min),
        ("weighted tardiness", before.weighted_tardiness, after.weighted_tardiness),
        ("all orders done at", before.all_orders_done_at, after.all_orders_done_at),
        ("mean utilization (%)", before.mean_utilization_pct, after.mean_utilization_pct),
    ]
    lines = [f"  {'':24}{'live plan':>18}{'draft':>18}"]
    lines += [f"  {name:24}{str(b):>18}{str(a):>18}" for name, b, a in rows]
    return lines


def commit_draft(ctx: ToolContext, draft_id: str, reviewed_digest: str | None = None) -> dict[str, Any]:
    """Commit a draft because a person just said yes to it (chat prompt or web Approve button).

    A token is minted for the draft's schedule *as it is now*. If ``reviewed_digest`` is given (the web
    UI sends the fingerprint of what was on screen), a draft that has changed since is refused, so a
    person can only ever approve what they were shown.
    """
    draft = ctx.store.draft(draft_id)
    if draft.schedule is None:
        raise ToolError(f"draft {draft_id} has no schedule to commit")
    digest = proposal_digest(draft.instance, draft.schedule)
    if reviewed_digest is not None and digest != reviewed_digest:
        raise ToolError(f"draft {draft_id} is not the proposal you reviewed (it has changed); review it again before approving")
    token = ctx.authority.issue(draft_id=draft.id, base_version=draft.base_version, schedule_digest=digest)
    return ToolRegistry(ctx).call("commit_schedule", {"draft_id": draft.id, "approval_token": token}, allow_hidden=True)


def pending_requests(ctx: ToolContext) -> list[ApprovalRequest]:
    return [r for r in ctx.store.requests() if ctx.store.request_status(r) == "pending"]


def describe(ctx: ToolContext, request_id: str) -> dict[str, Any]:
    """Everything a person needs to decide, computed from the store (not from the model's words)."""
    request = ctx.store.request(request_id)
    status = ctx.store.request_status(request)
    info: dict[str, Any] = {"request": asdict(request), "status": status}
    if status == "pending":
        draft = ctx.store.draft(request.draft_id)
        info["changes"] = draft.changes
        info["comparison"] = compare_schedules(
            ctx, CompareInput(before="committed", after=draft.id)
        ).model_dump(mode="json")
    return info


def approve(ctx: ToolContext, request_id: str) -> dict[str, Any]:
    """Commit the requested draft. Call this only after a person has said yes."""
    request = ctx.store.request(request_id)
    status = ctx.store.request_status(request)
    if status != "pending":
        raise ToolError(f"request {request_id} is {status}, not pending")

    draft = ctx.store.draft(request.draft_id)
    if draft.schedule is None or proposal_digest(draft.instance, draft.schedule) != request.schedule_digest:
        raise ToolError(
            f"draft {draft.id} was changed after approval was requested; "
            "what you reviewed is no longer what would be committed. Ask for approval again."
        )

    token = ctx.authority.issue(
        draft_id=draft.id, base_version=draft.base_version, schedule_digest=request.schedule_digest
    )
    result = ToolRegistry(ctx).call(
        "commit_schedule", {"draft_id": draft.id, "approval_token": token}, allow_hidden=True
    )
    ctx.store.decide(request_id, "approved")
    return result


def deny(ctx: ToolContext, request_id: str) -> None:
    request = ctx.store.request(request_id)
    status = ctx.store.request_status(request)
    if status != "pending":
        raise ToolError(f"request {request_id} is {status}, not pending")
    ctx.store.decide(request_id, "denied")
