"""The fine-tuning dataset export: only conversations that passed, rebuilt faithfully, in a format that replays."""

import json
import shutil

import pytest

from jobshop.evals import cli, export_sft
from tests.evals.conftest import EVALS


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """A real (scripted reference agent) run of four scenarios: a proposal, a question, a clarification and an injection."""
    out = tmp_path_factory.mktemp("run")
    code = cli.main(["--evals-dir", str(EVALS), "run", "--oracle", "--only", "sd-01", "--only", "q-04", "--only", "am-01",
                     "--only", "in-02", "--out", str(out)])
    assert code == 0
    return next(out.iterdir())


@pytest.fixture
def edited(results_dir, tmp_path):
    """A writable copy of the run, so a test can change results.json."""
    target = tmp_path / "run"
    shutil.copytree(results_dir, target)
    return target


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_every_passing_conversation_is_exported_with_its_system_prompt_and_tools(results_dir, tmp_path):
    out = tmp_path / "sft.jsonl"
    summary = export_sft.export(results_dir, EVALS, out)
    assert (summary.considered, summary.kept) == (4, 4) and not summary.skipped
    rows = lines(out)
    assert sorted(r["scenario"] for r in rows) == ["am-01", "in-02", "q-04", "sd-01"]
    for r in rows:
        assert "production planner" in r["system"] or "scheduling" in r["system"].lower()
        names = [t["name"] for t in r["tools"]]
        assert "submit_response" in names and "commit_schedule" not in names and "create_draft" in names
        assert r["checks"] and all(v in (True, None) for v in r["checks"].values())


def test_each_conversation_is_valid_and_ends_with_the_submitted_answer(results_dir, tmp_path):
    out = tmp_path / "sft.jsonl"
    export_sft.export(results_dir, EVALS, out)
    for r in lines(out):
        m = r["messages"]
        assert m[0]["role"] == "user" and isinstance(m[0]["content"], str)
        assert [x["role"] for x in m] == ["user", "assistant"] * (len(m) // 2)           # strictly alternating, ending on the answer
        called = [b["id"] for x in m[:-1] if x["role"] == "assistant" for b in x["content"] if b["type"] == "tool_use"]
        answered = [b["tool_use_id"] for x in m if x["role"] == "user" and isinstance(x["content"], list) for b in x["content"]]
        assert called == answered                                           # every call has its result, in order
        final = m[-1]
        assert final["role"] == "assistant" and final["content"][0]["name"] == "submit_response"
        assert final["content"][0]["input"]["summary"]


def test_a_proposal_names_its_draft_and_a_question_has_no_draft(results_dir, tmp_path):
    out = tmp_path / "sft.jsonl"
    export_sft.export(results_dir, EVALS, out)
    by = {r["scenario"]: r["messages"][-1]["content"][0]["input"] for r in lines(out)}
    assert by["sd-01"]["draft_id"] == "D1"
    assert "draft_id" not in by["q-04"] and by["am-01"]["clarifying_question"]


def test_the_request_and_tool_arguments_and_results_are_those_of_the_run(results_dir, tmp_path):
    out = tmp_path / "sft.jsonl"
    export_sft.export(results_dir, EVALS, out)
    row = next(r for r in lines(out) if r["scenario"] == "sd-01")
    assert "M2" in row["messages"][0]["content"]
    calls = [b for x in row["messages"][:-1] if x["role"] == "assistant" for b in x["content"] if b["type"] == "tool_use"]
    assert [c["name"] for c in calls][:2] == ["create_draft", "simulate_downtime"] and calls[1]["input"]["machine_id"] == "M2"
    results = [b for x in row["messages"] if x["role"] == "user" and isinstance(x["content"], list) for b in x["content"]]
    assert json.loads(results[0]["content"])["draft_id"] == "D1"


def test_a_run_that_failed_a_check_is_left_out_unless_asked_for(edited, tmp_path):
    data = json.loads((edited / "results.json").read_text(encoding="utf-8"))
    data["results"][0]["passed"] = False
    (edited / "results.json").write_text(json.dumps(data), encoding="utf-8")
    summary = export_sft.export(edited, EVALS, tmp_path / "a.jsonl")
    assert summary.kept == 3 and summary.skipped["failed a check"] == 1
    again = export_sft.export(edited, EVALS, tmp_path / "b.jsonl", include_failed=True)
    assert again.kept == 4


def test_a_turn_that_never_answered_is_never_exported_even_when_asked_for_failures(edited, tmp_path):
    data = json.loads((edited / "results.json").read_text(encoding="utf-8"))
    data["results"][1].update(status="api_error", passed=False, answer=None)
    (edited / "results.json").write_text(json.dumps(data), encoding="utf-8")
    summary = export_sft.export(edited, EVALS, tmp_path / "a.jsonl", include_failed=True)
    assert summary.kept == 3 and summary.skipped["turn did not answer (api_error)"] == 1


def test_a_judge_threshold_keeps_only_runs_the_judge_graded_that_high(edited, tmp_path):
    data = json.loads((edited / "results.json").read_text(encoding="utf-8"))
    data["results"][0]["judge"] = {"scores": {"accuracy": 5, "honesty": 5}, "reasons": {}, "error": None}
    data["results"][1]["judge"] = {"scores": {"accuracy": 3, "honesty": 5}, "reasons": {}, "error": None}
    (edited / "results.json").write_text(json.dumps(data), encoding="utf-8")
    summary = export_sft.export(edited, EVALS, tmp_path / "a.jsonl", min_judge=4)
    assert summary.kept == 1 and summary.skipped["judge score below 4"] == 1 and summary.skipped["not graded by the judge"] == 2
    assert lines(tmp_path / "a.jsonl")[0]["judge_scores"] == {"accuracy": 5, "honesty": 5}


def test_a_missing_trace_is_reported_not_silently_dropped(edited, tmp_path):
    for f in (edited / "traces").glob("sd-01-*.jsonl"):
        f.unlink()
    data = json.loads((edited / "results.json").read_text(encoding="utf-8"))
    data["results"][0]["trace"] = None
    (edited / "results.json").write_text(json.dumps(data), encoding="utf-8")
    summary = export_sft.export(edited, EVALS, tmp_path / "a.jsonl")
    assert summary.skipped["no trace file"] == 1 and summary.kept == 3


def test_the_command_writes_the_file_and_says_what_it_skipped(results_dir, tmp_path, capsys):
    out = tmp_path / "x.jsonl"
    assert cli.main(["--evals-dir", str(EVALS), "export-sft", "--results", str(results_dir), "--out", str(out)]) == 0
    assert "4 of 4 conversations written" in capsys.readouterr().out and len(lines(out)) == 4


def test_the_command_fails_when_nothing_qualifies_and_reports_a_bad_folder(results_dir, tmp_path, capsys):
    assert cli.main(["--evals-dir", str(EVALS), "export-sft", "--results", str(results_dir), "--out", str(tmp_path / "x.jsonl"), "--min-judge", "5"]) == 1
    assert "skipped 4: not graded by the judge" in capsys.readouterr().out
    assert cli.main(["--evals-dir", str(EVALS), "export-sft", "--results", str(tmp_path / "nowhere"), "--out", str(tmp_path / "y.jsonl")]) == 2
    assert "Configuration error" in capsys.readouterr().err
