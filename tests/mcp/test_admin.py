"""The human-facing admin command: init, status, approve, deny, clock."""

import threading

import pytest

from jobshop.core.generator import generate_instance
from jobshop.core.solver import solve
from jobshop.mcp_server.admin import main
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.functions import ToolContext
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.store import Store
from tests.helpers import FAST, SMALL


class Run:
    """Calls the admin CLI in-process with captured output and scripted answers."""

    def __init__(self, path, answers=()):
        self.path, self.answers, self.prompts, self.lines = path, list(answers), [], []

    def __call__(self, *argv, answers=None):
        if answers is not None:
            self.answers = list(answers)
        self.lines.clear()
        code = main(["--state", str(self.path), *argv], out=self.lines.append, ask=self._ask)
        return code, "\n".join(self.lines)

    def _ask(self, prompt):
        self.prompts.append(prompt)
        assert self.answers, f"unexpected prompt {prompt!r}"
        return self.answers.pop(0)


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "state.json"
    inst = generate_instance(SMALL)
    Store.create(p, inst, solve(inst, config=FAST))
    return p


def ask_for_approval(path, priority=5):
    """What a model does through MCP: draft, change, solve, request. Returns the request id."""
    store = Store.open(path)
    registry = ToolRegistry(ToolContext(store, ApprovalAuthority(), FAST), surface="mcp")
    with store.transaction():
        d = registry.call("create_draft", {})["draft_id"]
        registry.call("change_priority", {"draft_id": d, "order_id": "O-101", "priority": priority})
        registry.call("reschedule", {"draft_id": d})
        return registry.call("request_commit", {"draft_id": d})["request_id"], d


def live_version(path):
    store = Store.open(path)
    with store.transaction():
        return store.committed.version


# --- init / status / clock ---------------------------------------------------------------------


def test_init_creates_state_refuses_to_overwrite_and_can_force(tmp_path):
    p = tmp_path / "fresh" / "state.json"
    run = Run(p)
    args = ("init", "--seed", "1", "--orders", "5", "--machines", "4", "--solve-seconds", "2", "--workers", "1")
    code, text = run(*args)
    assert code == 0 and p.exists() and "baseline status" in text

    code, text = run(*args)
    assert code == 1 and "already exists" in text and "--force" in text
    code, _ = run(*args, "--force")
    assert code == 0


def test_init_can_start_the_clock_mid_shift(tmp_path):
    p = tmp_path / "state.json"
    code, _ = Run(p)("init", "--seed", "1", "--orders", "4", "--machines", "4", "--solve-seconds", "2", "--workers", "1", "--now", "2026-01-05 12:00")
    store = Store.open(p)
    with store.transaction():
        assert code == 0 and store.committed.instance.now == 360 and store.committed.version == 2


def test_commands_other_than_init_need_an_existing_state(tmp_path):
    code, text = Run(tmp_path / "missing.json")("status")
    assert code == 2 and "Run `init` first" in text


def test_status_summarizes_the_plan_drafts_and_pending_requests(path):
    run = Run(path)
    _, text = run("status")
    assert "Live plan version 1" in text and "Drafts: none" in text and "Pending approval requests: none" in text
    rid, draft = ask_for_approval(path)
    _, text = run("status")
    assert f"Drafts: {draft}" in text and f"{rid} (draft {draft})" in text


def test_clock_moves_forward_only_and_stales_requests(path):
    run = Run(path)
    rid, _ = ask_for_approval(path)
    code, text = run("clock", "2026-01-05 12:00")
    assert code == 0 and "version 2" in text
    code, text = run("clock", "2026-01-05 08:00")
    assert code == 1 and "Could not set the clock" in text
    code, text = run("clock", "noon")
    assert code == 1
    code, text = run("approve", rid, answers=[])
    assert code == 1 and "is stale, not pending" in text


# --- approve / deny ----------------------------------------------------------------------------


def test_approve_shows_a_review_then_commits_on_yes(path):
    rid, _ = ask_for_approval(path)
    run = Run(path, answers=["y"])
    code, text = run("approve")  # no id needed: exactly one request is pending
    assert code == 0 and "Committed. The live plan is now version 2." in text
    assert "O-101 priority" in text and "live plan" in text and "late orders" in text
    assert "computed from the stored schedules, not from the model" in text
    assert live_version(path) == 2


@pytest.mark.parametrize("reply", ["", "n", "no", "maybe"])
def test_anything_but_yes_commits_nothing_and_leaves_the_request_pending(path, reply):
    rid, _ = ask_for_approval(path)
    code, text = Run(path, answers=[reply])("approve", rid)
    assert code == 0 and "Not committed" in text and live_version(path) == 1
    _, status = Run(path)("status")
    assert f"{rid} (draft" in status


def test_with_several_pending_requests_you_must_name_one(path):
    r1, _ = ask_for_approval(path, priority=5)
    r2, _ = ask_for_approval(path, priority=4)
    code, text = Run(path)("approve")
    assert code == 1 and r1 in text and r2 in text and live_version(path) == 1


def test_nothing_pending_is_not_an_error(path):
    code, text = Run(path)("approve")
    assert code == 0 and "No pending approval requests" in text


def test_deny_declines_and_the_request_cannot_then_be_approved(path):
    rid, _ = ask_for_approval(path)
    code, text = Run(path)("deny", rid)
    assert code == 0 and "denied" in text
    code, text = Run(path, answers=["y"])("approve", rid)
    assert code == 1 and "is denied, not pending" in text and live_version(path) == 1


def test_the_state_lock_is_not_held_while_a_person_decides(path):
    """If it were, the MCP server would freeze for as long as the human deliberated."""
    rid, _ = ask_for_approval(path)
    outcome = []

    def other_process_touches_state():
        store = Store.open(path)
        with store.transaction():
            outcome.append(store.committed.version)

    def deliberate(prompt):
        worker = threading.Thread(target=other_process_touches_state, daemon=True)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive(), "the approve command held the state lock while waiting for input"
        return "n"

    code = main(["--state", str(path), "approve", rid], out=lambda _: None, ask=deliberate)
    assert code == 0 and outcome == [1]


def test_if_the_draft_changes_while_you_deliberate_approval_fails(path):
    rid, draft = ask_for_approval(path)

    def tamper_then_say_yes(prompt):
        store = Store.open(path)
        registry = ToolRegistry(ToolContext(store, ApprovalAuthority(), FAST), surface="mcp")
        with store.transaction():  # the model edits the draft the human is reviewing
            registry.call("change_priority", {"draft_id": draft, "order_id": "O-102", "priority": 5})
        return "y"

    lines = []
    code = main(["--state", str(path), "approve", rid], out=lines.append, ask=tamper_then_say_yes)
    assert code == 1 and "was changed after approval was requested" in "\n".join(lines)
    assert live_version(path) == 1


def test_if_the_plan_moves_while_you_deliberate_approval_fails(path):
    rid, _ = ask_for_approval(path)

    def move_the_clock_then_say_yes(prompt):
        store = Store.open(path)
        with store.transaction():
            store.set_clock(30)
        return "y"

    lines = []
    code = main(["--state", str(path), "approve", rid], out=lines.append, ask=move_the_clock_then_say_yes)
    assert code == 1 and "is stale, not pending" in "\n".join(lines)
    assert live_version(path) == 2  # only the clock move


# --- nothing printed to the reviewer can redraw the screen -----------------------------------------


def test_admin_output_cannot_carry_terminal_escape_sequences(path, monkeypatch):
    from jobshop.mcp_server import admin

    def hostile_status(args, out, ask):
        out("Pending: none\x1b[2J\x1b[1;1H All orders on time, safe to approve.\r")
        return 0

    monkeypatch.setattr(admin, "cmd_status", hostile_status)
    code, text = Run(path)("status")
    assert code == 0 and "\x1b" not in text and "\r" not in text and "safe to approve" in text
