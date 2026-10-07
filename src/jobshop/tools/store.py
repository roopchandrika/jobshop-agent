"""In-memory state: the committed schedule, the shop clock, and draft scenarios.

Only this class touches that state, so Phase 3 (MCP server in a separate process) can swap in
a file-backed implementation without changing any tool.

``version`` increments whenever the committed state changes (a commit, or the clock moving).
A draft remembers the version it was based on; if the version has moved on, the draft is
*stale* and refuses to be solved, compared or committed. That is how a planner can never
approve a schedule that was computed against facts that are no longer true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jobshop.core.models import Assignment, Instance, Schedule, SolveStatus
from jobshop.tools.errors import ToolError


@dataclass(frozen=True)
class Committed:
    instance: Instance
    schedule: Schedule
    version: int


@dataclass
class Draft:
    id: str
    base_version: int
    instance: Instance
    changes: list[str] = field(default_factory=list)  # human-readable, for display
    schedule: Schedule | None = None  # None until `reschedule` runs
    interrupted: list[Assignment] = field(default_factory=list)  # from the last reschedule

    @property
    def solved(self) -> bool:
        return self.schedule is not None and self.schedule.solve_info.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.FEASIBLE,
        )

    def edited(self, instance: Instance, description: str) -> None:
        """Record a change. Any earlier solution no longer matches the instance, so drop it."""
        self.instance = instance
        self.changes.append(description)
        self.schedule = None
        self.interrupted = []


class Store:
    def __init__(self, instance: Instance, schedule: Schedule) -> None:
        self._committed = Committed(instance, schedule, version=1)
        self._drafts: dict[str, Draft] = {}
        self._draft_counter = 0

    @property
    def committed(self) -> Committed:
        return self._committed

    def set_clock(self, now: int) -> Committed:
        """Move the shop clock forward. Done by the human-facing layer, never by the model."""
        current = self._committed
        if now < current.instance.now:
            raise ValueError("the clock can only move forward")
        instance = Instance.model_validate({**current.instance.model_dump(), "now": now})
        self._committed = Committed(instance, current.schedule, current.version + 1)
        return self._committed

    def create_draft(self) -> Draft:
        self._draft_counter += 1
        draft = Draft(
            id=f"D{self._draft_counter}",
            base_version=self._committed.version,
            instance=self._committed.instance,
        )
        self._drafts[draft.id] = draft
        return draft

    def draft(self, draft_id: str) -> Draft:
        try:
            return self._drafts[draft_id]
        except KeyError:
            known = ", ".join(self._drafts) or "none"
            raise ToolError(
                f"unknown draft '{draft_id}' (existing drafts: {known}). Call create_draft first."
            ) from None

    def discard(self, draft_id: str) -> None:
        self.draft(draft_id)
        del self._drafts[draft_id]

    def is_stale(self, draft: Draft) -> bool:
        return draft.base_version != self._committed.version

    def commit(self, draft: Draft) -> Committed:
        """Make a draft the committed state. Callers must have validated it and checked approval."""
        assert draft.schedule is not None and not self.is_stale(draft)
        self._committed = Committed(draft.instance, draft.schedule, self._committed.version + 1)
        return self._committed
