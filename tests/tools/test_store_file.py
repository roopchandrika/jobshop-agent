"""The file-backed store: shared state between the MCP server and the human approval command."""

import json
import threading
import time

import pytest

from jobshop.core.generator import generate_instance
from jobshop.core.solver import solve
from jobshop.tools.errors import ToolError
from jobshop.tools.store import Store
from tests.helpers import FAST, SMALL, FakeClock


@pytest.fixture
def path(tmp_path):
    return tmp_path / "state.json"


@pytest.fixture
def created(path):
    inst = generate_instance(SMALL)
    return Store.create(path, inst, solve(inst, config=FAST))


def test_create_writes_a_file_and_refuses_to_overwrite_one(path):
    inst = generate_instance(SMALL)
    sched = solve(inst, config=FAST)
    Store.create(path, inst, sched)
    assert json.loads(path.read_text())["committed"]["version"] == 1
    with pytest.raises(FileExistsError, match="already exists"):
        Store.create(path, inst, sched)


def test_open_round_trips_the_committed_plan(path, created):
    with created.transaction():
        original = created.committed
    reopened = Store.open(path)
    with reopened.transaction():
        again = reopened.committed
    assert (again.instance, again.schedule, again.version) == (original.instance, original.schedule, 1)


def test_using_a_file_backed_store_outside_a_transaction_is_an_error(created):
    with pytest.raises(RuntimeError, match="transaction"):
        created.committed
    with pytest.raises(RuntimeError, match="transaction"):
        created.create_draft()


def test_changes_made_in_one_store_are_seen_by_another_over_the_same_file(path, created):
    other = Store.open(path)
    with created.transaction():
        draft = created.create_draft()
        draft.edited(draft.instance, "a change")
    with other.transaction():
        seen = other.draft("D1")
        assert seen.changes == ["a change"] and seen.base_version == 1


def test_a_failed_transaction_writes_nothing(path, created):
    before = path.read_text()
    with pytest.raises(ZeroDivisionError):
        with created.transaction():
            created.create_draft()
            1 / 0
    assert path.read_text() == before
    with created.transaction():  # the half-made draft is gone, not just hidden
        assert created.drafts() == []


def test_a_read_only_transaction_does_not_rewrite_the_file(path, created):
    before = path.stat().st_mtime_ns
    with created.transaction():
        created.committed
    assert path.stat().st_mtime_ns == before


def test_nested_transactions_are_allowed(created):
    with created.transaction():
        with created.transaction():
            created.create_draft()
        assert len(created.drafts()) == 1


def test_clock_version_drafts_and_solutions_survive_a_reload(path, created):
    with created.transaction():
        draft = created.create_draft()
        draft.edited(draft.instance, "x")
        draft.schedule = solve(draft.instance, config=FAST)
        created.set_clock(60)
    again = Store.open(path)
    with again.transaction():
        assert again.committed.version == 2 and again.committed.instance.now == 60
        d = again.draft("D1")
        assert d.solved and d.base_version == 1 and again.is_stale(d)
        assert again.create_draft().id == "D2"  # the counter persisted


def test_approval_requests_persist_and_their_status_is_derived(path):
    inst = generate_instance(SMALL)
    clock = FakeClock()
    store = Store.create(path, inst, solve(inst, config=FAST), clock=clock, request_ttl_s=100)
    with store.transaction():
        draft = store.create_draft()
        request = store.add_request(draft, "digest")
        assert (request.id, store.request_status(request)) == ("R1", "pending")
    clock.advance(101)
    reopened = Store.open(path, clock=clock, request_ttl_s=100)
    with reopened.transaction():
        assert reopened.request_status(reopened.request("R1")) == "expired"
    clock.t -= 101
    with reopened.transaction():
        reopened.set_clock(30)  # the plan moves on
        assert reopened.request_status(reopened.request("R1")) == "stale"


def test_decided_requests_keep_their_decision(path, created):
    with created.transaction():
        draft = created.create_draft()
        created.add_request(draft, "d")
        created.decide("R1", "approved")
    reopened = Store.open(path)
    with reopened.transaction():
        assert reopened.request_status(reopened.request("R1")) == "approved"
        with pytest.raises(ToolError, match="unknown approval request 'R9'"):
            reopened.request("R9")


def test_concurrent_threads_never_lose_updates(path, created):
    """Many threads each add a draft through their own Store object: all must survive."""
    stores = [Store.open(path) for _ in range(8)]
    ids: list[str] = []

    def add(store):
        with store.transaction():
            ids.append(store.create_draft().id)
            time.sleep(0.02)  # widen the read-modify-write window so a missing lock cannot go unnoticed

    threads = [threading.Thread(target=add, args=(s,)) for s in stores]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert sorted(ids) == sorted(f"D{i}" for i in range(1, 9))  # unique ids: no two shared a counter value
    check = Store.open(path)
    with check.transaction():
        assert len(check.drafts()) == 8


def test_threads_sharing_one_store_object_are_serialized_too(path, created):
    ids: list[str] = []

    def add():
        with created.transaction():
            ids.append(created.create_draft().id)

    threads = [threading.Thread(target=add) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(ids) == sorted(f"D{i}" for i in range(1, 9))


def test_unsupported_state_formats_are_rejected(path, created):
    data = json.loads(path.read_text())
    data["format"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsupported state file format"):
        Store.open(path)
    with pytest.raises(ValueError, match="unsupported state file format"):
        with created.transaction():
            pass


def test_no_temp_file_is_left_behind(path, created):
    with created.transaction():
        created.create_draft()
    assert not path.with_name(path.name + ".tmp").exists()
