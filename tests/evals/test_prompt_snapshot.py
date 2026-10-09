"""The prompt-change gate. The first test is the gate itself: it fails when anything the model is told differs from the saved text."""

import json
import shutil

import pytest

from jobshop.agent import patterns
from jobshop.evals import cli, snapshot
from tests.evals.conftest import EVALS


def test_what_the_model_is_told_matches_the_saved_snapshot():
    problems = snapshot.check(EVALS)
    assert not problems, (
        "The prompts or tool descriptions changed.\n\n" + "\n\n".join(problems) +
        f"\n\nIf that was intended, run the evals against a real model, then: {snapshot.UPDATE_COMMAND}"
    )


def test_the_snapshot_covers_every_piece_of_text_the_model_receives():
    saved = snapshot.load(EVALS / snapshot.SNAPSHOT_NAME)
    assert set(saved) == {
        "system_prompt", "system_prompt_with_preferences", "tools_agent", "tools_reader", "submit_response_tool", "plan_rules", "plan_tool",
        "critic_system", "review_tool", "triage_system", "triage_tool", "reader_role", "commit_refusal", "live_claim_warning",
    }
    assert all(len(v) > 50 for v in saved.values())
    assert "commit_schedule" not in saved["tools_agent"], "the chat agent must not be offered a commit tool"
    for name in ("create_draft", "reschedule", "change_priority"):
        assert name in saved["tools_agent"] and name not in saved["tools_reader"]
    assert "get_schedule" in saved["tools_reader"] and "search_knowledge" in saved["tools_reader"]


def test_the_text_is_built_the_same_way_every_time():
    assert snapshot.current(EVALS) == snapshot.current(EVALS)


@pytest.mark.parametrize("change", [
    lambda monkeypatch: monkeypatch.setattr(patterns, "TRIAGE_SYSTEM", patterns.TRIAGE_SYSTEM + " Be generous."),
    lambda monkeypatch: monkeypatch.setattr(patterns, "COMMIT_REFUSAL", "Sure, committed."),
    lambda monkeypatch: monkeypatch.setattr(patterns, "READ_ONLY_TOOLS", patterns.READ_ONLY_TOOLS | {"create_draft"}),
])
def test_a_changed_prompt_or_a_wider_tool_scope_fails_the_gate_and_the_diff_names_it(monkeypatch, change):
    change(monkeypatch)
    problems = snapshot.check(EVALS)
    assert problems and any(p.startswith("~ ") for p in problems)


def test_a_reworded_tool_description_shows_as_a_line_diff():
    saved = snapshot.load(EVALS / snapshot.SNAPSHOT_NAME)
    changed = dict(saved, system_prompt=saved["system_prompt"].replace("never", "rarely", 1))
    [block] = snapshot.differences(saved, changed)
    assert block.startswith("~ system_prompt") and "-" in block and "rarely" in block and "@@" in block


def test_added_and_removed_pieces_are_reported():
    assert snapshot.differences({"a": "x"}, {"a": "x", "b": "y"}) == ["+ b: new, not in the snapshot"]
    assert snapshot.differences({"a": "x", "b": "y"}, {"a": "x"}) == ["- b: in the snapshot, no longer produced"]
    assert snapshot.differences({"a": "x"}, {"a": "x"}) == []


def test_a_long_diff_is_cut_short_with_a_count():
    old = {"a": "\n".join(f"line {i}" for i in range(100))}
    new = {"a": "\n".join(f"changed {i}" for i in range(100))}
    [block] = snapshot.differences(old, new)
    assert "more diff lines" in block and len(block.splitlines()) <= 43      # the heading, 40 diff lines, the note


def test_windows_line_endings_in_the_saved_file_do_not_count_as_a_change(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"a": "one\r\ntwo"}), encoding="utf-8")
    assert snapshot.load(path) == {"a": "one\ntwo"}


# -- the command ------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def evals_copy(tmp_path):
    """A copy of the fixture folder (shop and snapshot) and the documents, so the command can change files freely."""
    target = tmp_path / "evals"
    target.mkdir()
    for name in ("shop.json", snapshot.SNAPSHOT_NAME):
        shutil.copy(EVALS / name, target / name)
    shutil.copytree(EVALS.parent / "knowledge", tmp_path / "knowledge")
    return target


def run_cli(evals, *args):
    return cli.main(["--evals-dir", str(evals), "snapshot", *args])


def test_the_command_passes_when_nothing_changed(evals_copy, capsys):
    assert run_cli(evals_copy) == 0
    assert "match the snapshot" in capsys.readouterr().out


def test_the_command_fails_with_a_diff_and_the_way_forward_when_a_prompt_changed(evals_copy, capsys, monkeypatch):
    monkeypatch.setattr(patterns, "TRIAGE_SYSTEM", patterns.TRIAGE_SYSTEM + " Extra line.")
    assert run_cli(evals_copy) == 1
    err = capsys.readouterr().err
    assert "~ triage_system" in err and "Extra line." in err
    assert "snapshot --update" in err and "run the evals" in err


def test_update_rewrites_the_file_and_the_check_then_passes(evals_copy, capsys, monkeypatch):
    monkeypatch.setattr(patterns, "COMMIT_REFUSAL", patterns.COMMIT_REFUSAL + " Changed.")
    assert run_cli(evals_copy, "--update") == 0
    out = capsys.readouterr().out
    assert "1 of 14 pieces of text changed" in out and "commit_refusal" in out and "Now run the evals" in out
    assert run_cli(evals_copy) == 0
    assert run_cli(evals_copy, "--update") == 0 and "already up to date" in capsys.readouterr().out


def test_a_missing_snapshot_is_reported_with_the_command_that_creates_it(evals_copy, capsys):
    (evals_copy / snapshot.SNAPSHOT_NAME).unlink()
    assert run_cli(evals_copy) == 1
    err = capsys.readouterr().err
    assert "does not exist" in err and snapshot.UPDATE_COMMAND in err and "has changed" not in err
    assert run_cli(evals_copy, "--update") == 0 and "Created" in capsys.readouterr().out
    assert (evals_copy / snapshot.SNAPSHOT_NAME).exists()


def test_a_damaged_snapshot_is_a_configuration_error(evals_copy, capsys):
    (evals_copy / snapshot.SNAPSHOT_NAME).write_text("not json", encoding="utf-8")
    assert run_cli(evals_copy) == 2
    assert "Configuration error" in capsys.readouterr().err
