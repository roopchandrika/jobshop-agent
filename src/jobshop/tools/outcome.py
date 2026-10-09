"""What the harness (not the model) reports about a draft once the agent is done.

The agent's final answer includes before/after KPIs and ``needs_approval``. Letting the model
type those would let it misstate them, so the harness computes them here from the store, using
the same views the tools return. The model only supplies words and a draft id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jobshop.core.kpis import compute_kpis
from jobshop.tools import views
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import ToolContext
from jobshop.tools.views import Goal, KPIView


@dataclass
class DraftOutcome:
    changes: list[str] = field(default_factory=list)  # what the draft actually contains
    kpi_before: KPIView | None = None
    kpi_after: KPIView | None = None
    needs_approval: bool = False
    warnings: list[str] = field(default_factory=list)
    goal: Goal | None = None  # what the draft was solved for, from the draft, not from the model


def draft_outcome(ctx: ToolContext, draft_id: str | None) -> DraftOutcome:
    if draft_id is None:
        return DraftOutcome()

    try:
        draft = ctx.store.draft(draft_id)
    except ToolError:
        return DraftOutcome(warnings=[f"The answer refers to draft '{draft_id}', which does not exist."])

    committed = ctx.store.committed
    if ctx.store.is_stale(draft):
        return DraftOutcome(warnings=[f"Draft {draft_id} is stale (the plan changed since it was made); it cannot be approved."])
    if not draft.solved or draft.schedule is None:
        return DraftOutcome(warnings=[f"Draft {draft_id} has no solved schedule, so there are no after-KPIs."])

    return DraftOutcome(
        changes=list(draft.changes),
        kpi_before=views.kpi_view(committed.instance, compute_kpis(committed.instance, committed.schedule)),
        kpi_after=views.kpi_view(draft.instance, compute_kpis(draft.instance, draft.schedule)),
        needs_approval=bool(draft.changes),
        goal=draft.goal,
    )
