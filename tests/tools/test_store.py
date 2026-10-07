import pytest

from jobshop.core.solver import solve
from jobshop.tools.errors import ToolError
from tests.helpers import FAST, build_ctx


@pytest.fixture
def store():
    return build_ctx().store


def test_store_starts_at_version_one(store):
    assert store.committed.version == 1


def test_drafts_get_sequential_ids_and_remember_their_base_version(store):
    d1, d2 = store.create_draft(), store.create_draft()
    assert (d1.id, d2.id) == ("D1", "D2")
    assert d1.base_version == 1 and d1.instance is store.committed.instance


def test_unknown_draft_names_the_ones_that_exist(store):
    store.create_draft()
    with pytest.raises(ToolError, match=r"unknown draft 'D9' \(existing drafts: D1\)"):
        store.draft("D9")


def test_discard_removes_a_draft(store):
    d = store.create_draft()
    store.discard(d.id)
    with pytest.raises(ToolError):
        store.draft(d.id)
    with pytest.raises(ToolError):
        store.discard(d.id)


def test_moving_the_clock_bumps_the_version_and_makes_drafts_stale(store):
    d = store.create_draft()
    assert not store.is_stale(d)
    store.set_clock(60)
    assert store.committed.version == 2 and store.committed.instance.now == 60
    assert store.is_stale(d)


def test_the_clock_cannot_move_backwards(store):
    store.set_clock(60)
    with pytest.raises(ValueError, match="forward"):
        store.set_clock(30)


def test_commit_replaces_the_committed_state_and_stales_other_drafts(store):
    d1, d2 = store.create_draft(), store.create_draft()
    d1.schedule = solve(d1.instance, config=FAST)
    new = store.commit(d1)
    assert new.version == 2 and store.committed.schedule is d1.schedule
    assert store.is_stale(d2) and store.is_stale(d1)


def test_editing_a_draft_discards_its_old_solution(store):
    d = store.create_draft()
    d.schedule = solve(d.instance, config=FAST)
    assert d.solved
    d.edited(d.instance, "some change")
    assert d.schedule is None and not d.solved and d.changes == ["some change"]
