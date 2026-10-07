"""State: the committed plan, the shop clock, draft scenarios, and approval requests.

Only this class touches that state, so every tool works unchanged whether the store lives in
memory (the chat CLI) or in a shared file (the MCP server and the human approval command, which
are separate processes).

``version`` increments whenever the committed state changes (a commit, or the clock moving).
A draft remembers the version it was based on; if the version has moved on, the draft is
*stale* and refuses to be solved, compared or committed. That is how a planner can never
approve a schedule that was computed against facts that are no longer true.

File-backed stores are used through ``transaction()``: take the lock, reload the file, run the
code, write back only if something changed. If the code raises, nothing is written, which is
the rollback. Touching a file-backed store outside a transaction raises, because its in-memory
copy may be stale.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jobshop.core.models import Assignment, Instance, Schedule, SolveStatus
from jobshop.tools.errors import ToolError

STATE_FORMAT = 1


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


@dataclass
class ApprovalRequest:
    """A model's request that a human review a draft. It commits nothing by itself."""

    id: str
    draft_id: str
    base_version: int
    schedule_digest: str  # what the human will be shown and approve: this exact schedule
    created_at: float
    status: str = "pending"  # pending | approved | denied (expired/stale are derived)
    decided_at: float | None = None


class Store:
    def __init__(
        self,
        instance: Instance,
        schedule: Schedule,
        *,
        clock: Callable[[], float] = time.time,
        request_ttl_s: float = 1800.0,
    ) -> None:
        self._committed = Committed(instance, schedule, version=1)
        self._drafts: dict[str, Draft] = {}
        self._draft_counter = 0
        self._requests: dict[str, ApprovalRequest] = {}
        self._request_counter = 0
        self._clock = clock
        self.request_ttl_s = request_ttl_s
        self.path: Path | None = None
        self._rlock = threading.RLock()  # serializes threads in this process
        self._depth = 0

    # -- file-backed stores ---------------------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path, instance: Instance, schedule: Schedule, **kwargs: Any) -> Store:
        """Start a new shared state file. Refuses to overwrite an existing one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"{path} already exists; delete it or choose another path")
        store = cls(instance, schedule, **kwargs)
        store.path = path
        with _FileLock(store._lock_path):
            store._write_atomic(store._dump())
        return store

    @classmethod
    def open(cls, path: str | Path, **kwargs: Any) -> Store:
        """Attach to an existing state file. Use ``transaction()`` to read or change it."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        committed = data["committed"]
        store = cls(
            Instance.model_validate(committed["instance"]),
            Schedule.model_validate(committed["schedule"]),
            **kwargs,
        )
        store.path = path
        store._apply(data)
        return store

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a block with exclusive access to the shared file (no-op for in-memory stores)."""
        with self._rlock:
            if self.path is None or self._depth > 0:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            with _FileLock(self._lock_path):
                before = self.path.read_text(encoding="utf-8")
                self._apply(json.loads(before))
                self._depth = 1
                try:
                    yield
                    after = self._dump()
                    if after != before:
                        self._write_atomic(after)
                finally:
                    self._depth = 0

    @property
    def _lock_path(self) -> Path:
        assert self.path is not None
        return self.path.with_name(self.path.name + ".lock")

    def _require_transaction(self) -> None:
        if self.path is not None and self._depth == 0:
            raise RuntimeError("this store is file-backed: wrap access in `with store.transaction():`")

    def _write_atomic(self, text: str) -> None:
        assert self.path is not None
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, self.path)

    def _dump(self) -> str:
        c = self._committed
        state = {
            "format": STATE_FORMAT,
            "committed": {
                "instance": c.instance.model_dump(mode="json"),
                "schedule": c.schedule.model_dump(mode="json"),
                "version": c.version,
            },
            "drafts": {
                d.id: {
                    "id": d.id,
                    "base_version": d.base_version,
                    "instance": d.instance.model_dump(mode="json"),
                    "changes": d.changes,
                    "schedule": d.schedule.model_dump(mode="json") if d.schedule else None,
                    "interrupted": [a.model_dump(mode="json") for a in d.interrupted],
                }
                for d in self._drafts.values()
            },
            "draft_counter": self._draft_counter,
            "requests": {r.id: asdict(r) for r in self._requests.values()},
            "request_counter": self._request_counter,
        }
        return json.dumps(state, sort_keys=True, separators=(",", ":"))

    def _apply(self, data: dict[str, Any]) -> None:
        if data.get("format") != STATE_FORMAT:
            raise ValueError(f"unsupported state file format {data.get('format')!r}")
        c = data["committed"]
        self._committed = Committed(
            Instance.model_validate(c["instance"]), Schedule.model_validate(c["schedule"]), c["version"]
        )
        self._drafts = {
            did: Draft(
                id=d["id"],
                base_version=d["base_version"],
                instance=Instance.model_validate(d["instance"]),
                changes=list(d["changes"]),
                schedule=Schedule.model_validate(d["schedule"]) if d["schedule"] else None,
                interrupted=[Assignment.model_validate(a) for a in d["interrupted"]],
            )
            for did, d in data["drafts"].items()
        }
        self._draft_counter = data["draft_counter"]
        self._requests = {rid: ApprovalRequest(**r) for rid, r in data["requests"].items()}
        self._request_counter = data["request_counter"]

    # -- committed plan and clock ---------------------------------------------------------------

    @property
    def committed(self) -> Committed:
        self._require_transaction()
        return self._committed

    def set_clock(self, now: int) -> Committed:
        """Move the shop clock forward. Done by the human-facing layer, never by the model."""
        self._require_transaction()
        current = self._committed
        if now < current.instance.now:
            raise ValueError("the clock can only move forward")
        instance = Instance.model_validate({**current.instance.model_dump(), "now": now})
        self._committed = Committed(instance, current.schedule, current.version + 1)
        return self._committed

    # -- drafts ---------------------------------------------------------------------------------

    def create_draft(self) -> Draft:
        self._require_transaction()
        self._draft_counter += 1
        draft = Draft(
            id=f"D{self._draft_counter}",
            base_version=self._committed.version,
            instance=self._committed.instance,
        )
        self._drafts[draft.id] = draft
        return draft

    def draft(self, draft_id: str) -> Draft:
        self._require_transaction()
        try:
            return self._drafts[draft_id]
        except KeyError:
            known = ", ".join(self._drafts) or "none"
            raise ToolError(
                f"unknown draft '{draft_id}' (existing drafts: {known}). Call create_draft first."
            ) from None

    def drafts(self) -> list[Draft]:
        self._require_transaction()
        return list(self._drafts.values())

    def discard(self, draft_id: str) -> None:
        self.draft(draft_id)
        del self._drafts[draft_id]

    def is_stale(self, draft: Draft) -> bool:
        self._require_transaction()
        return draft.base_version != self._committed.version

    def commit(self, draft: Draft) -> Committed:
        """Make a draft the committed state. Callers must have validated it and checked approval."""
        self._require_transaction()
        assert draft.schedule is not None and not self.is_stale(draft)
        self._committed = Committed(draft.instance, draft.schedule, self._committed.version + 1)
        return self._committed

    # -- approval requests ----------------------------------------------------------------------

    def add_request(self, draft: Draft, schedule_digest: str) -> ApprovalRequest:
        self._require_transaction()
        self._request_counter += 1
        request = ApprovalRequest(
            id=f"R{self._request_counter}",
            draft_id=draft.id,
            base_version=draft.base_version,
            schedule_digest=schedule_digest,
            created_at=self._clock(),
        )
        self._requests[request.id] = request
        return request

    def request(self, request_id: str) -> ApprovalRequest:
        self._require_transaction()
        try:
            return self._requests[request_id]
        except KeyError:
            known = ", ".join(self._requests) or "none"
            raise ToolError(f"unknown approval request '{request_id}' (existing: {known})") from None

    def requests(self) -> list[ApprovalRequest]:
        self._require_transaction()
        return list(self._requests.values())

    def request_status(self, request: ApprovalRequest) -> str:
        """pending, approved, denied, stale (the plan moved on) or expired (too old)."""
        self._require_transaction()
        if request.status != "pending":
            return request.status
        if request.base_version != self._committed.version:
            return "stale"
        if self._clock() - request.created_at > self.request_ttl_s:
            return "expired"
        return "pending"

    def decide(self, request_id: str, status: str) -> None:
        assert status in ("approved", "denied")
        request = self.request(request_id)
        request.status = status
        request.decided_at = self._clock()


class _FileLock:
    """Cross-process exclusive lock on a file. The OS releases it if the holder dies."""

    def __init__(self, path: Path, timeout_s: float = 120.0) -> None:
        self._path = path
        self._timeout_s = timeout_s
        self._handle: Any = None

    def __enter__(self) -> _FileLock:
        self._handle = open(self._path, "a+b")
        deadline = time.monotonic() + self._timeout_s
        while True:
            try:
                self._acquire()
                return self
            except OSError:
                if time.monotonic() > deadline:
                    self._handle.close()
                    raise TimeoutError(f"could not lock {self._path} within {self._timeout_s:g}s") from None
                time.sleep(0.05)

    def __exit__(self, *exc: object) -> None:
        try:
            self._release()
        finally:
            self._handle.close()

    def _acquire(self) -> None:
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(self) -> None:
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle, fcntl.LOCK_UN)
