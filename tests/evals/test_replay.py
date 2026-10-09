"""Record once, replay for free; a changed prompt, tool or tool result makes the recording stale, loudly."""

import json

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.evals import cli, replay
from jobshop.evals.oracle import OracleClient
from jobshop.evals.runner import run_suite
from tests.evals.conftest import EVALS
from tests.fake_llm import FakeClient, message, submit, tool
from tests.helpers import FAST

IDS = ("sd-01", "q-04", "ts-01")


@pytest.fixture
def chosen(scenarios):
    return [scenarios[i] for i in IDS]


def record(chosen, shops, directory, repeat=1):
    return run_suite(chosen, replay.recording_factory(directory, OracleClient, "oracle"), AgentConfig(model="m"),
                     shops, FAST, repeat=repeat)


def play_back(chosen, shops, directory, repeat=1):
    return run_suite(chosen, replay.replay_factory(directory), AgentConfig(model="replay"), shops, FAST, repeat=repeat)


# -- the fingerprint ------------------------------------------------------------------------------------------


def request(system="S", tools=None, content="hello", **extra):
    return {"system": system, "tools": tools or [{"name": "t"}], "messages": [{"role": "user", "content": content}], **extra}


def test_equal_requests_have_equal_fingerprints_whatever_the_model_or_output_limit():
    assert replay.fingerprint(request(model="a", max_tokens=1)) == replay.fingerprint(request(model="b", max_tokens=9))


@pytest.mark.parametrize("changed", [request(system="S2"), request(tools=[{"name": "u"}]), request(content="bye")])
def test_the_prompt_the_tools_and_the_conversation_each_change_it(changed):
    assert replay.fingerprint(changed) != replay.fingerprint(request())


def test_cache_markers_do_not_change_it_so_caching_can_be_switched_without_re_recording():
    marked = request(tools=[{"name": "t", "cache_control": {"type": "ephemeral"}}])
    marked["messages"][0]["content"] = [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]
    plain = request()
    plain["messages"][0]["content"] = [{"type": "text", "text": "hello"}]
    assert replay.fingerprint(marked) == replay.fingerprint(plain)


def test_solver_wall_clock_time_inside_a_tool_result_does_not_change_it_but_other_values_do():
    def with_result(seconds, status="OPTIMAL"):
        body = json.dumps({"solve": {"status": status, "solve_seconds": seconds}})
        return request(content=[{"type": "tool_result", "tool_use_id": "x", "content": body}])

    assert replay.fingerprint(with_result(0.3)) == replay.fingerprint(with_result(4.9))
    assert replay.fingerprint(with_result(0.3)) != replay.fingerprint(with_result(0.3, status="FEASIBLE"))


# -- recording and replaying --------------------------------------------------------------------------------------


def test_a_recorded_run_replays_to_the_same_results_without_any_client(chosen, shops, tmp_path):
    first = record(chosen, shops, tmp_path)
    again = play_back(chosen, shops, tmp_path)
    assert [(r.id, r.passed, r.steps, r.tools_called) for r in again] == [(r.id, r.passed, r.steps, r.tools_called) for r in first]
    assert [(r.input_tokens, r.output_tokens) for r in again] == [(r.input_tokens, r.output_tokens) for r in first]
    assert [r.answer for r in again] == [r.answer for r in first] and all(r.passed for r in again)


def test_one_file_per_scenario_and_attempt_and_the_available_runs_are_counted(chosen, shops, tmp_path):
    record(chosen, shops, tmp_path, repeat=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{i}-{n}.json" for i in IDS for n in (1, 2))
    assert replay.available(tmp_path) == {i: 2 for i in IDS}


def test_a_recording_keeps_the_model_name_and_when_it_was_made(chosen, shops, tmp_path):
    record(chosen[:1], shops, tmp_path)
    data = json.loads((tmp_path / "sd-01-1.json").read_text())
    assert data["format"] == replay.FORMAT and data["model"] == "oracle" and data["recorded_at"] and data["steps"]


def test_every_response_is_saved_as_it_arrives_so_a_stopped_run_keeps_what_was_paid_for(tmp_path):
    path = tmp_path / "x-1.json"
    boom = RuntimeError("network died")
    recorder = replay.Recorder(FakeClient([message(tool("get_schedule", "t1")), boom]), path, "m")
    recorder.create(system="S", tools=[], messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(RuntimeError):
        recorder.create(system="S", tools=[], messages=[{"role": "user", "content": "hi again"}])
    assert len(json.loads(path.read_text())["steps"]) == 1


# -- staleness: the point of the fingerprint ---------------------------------------------------------------------


def test_a_changed_request_makes_that_scenario_fail_as_stale_and_leaves_the_others_alone(chosen, shops, tmp_path):
    record(chosen, shops, tmp_path)
    edited = [s.model_copy(update={"request": s.request + " Please hurry."}) if s.id == "sd-01" else s for s in chosen]
    results = play_back(edited, shops, tmp_path)

    stale = next(r for r in results if r.id == "sd-01")
    assert stale.status == "crashed" and not stale.passed
    assert "call 1 is a different request" in stale.error and "Re-record with --record" in stale.error
    assert all(r.passed for r in results if r.id != "sd-01")


def test_a_changed_system_prompt_is_detected(chosen, shops, tmp_path, monkeypatch):
    record(chosen[:1], shops, tmp_path)
    from jobshop.evals import runner

    original = runner.build_system_prompt
    monkeypatch.setattr(runner, "build_system_prompt", lambda ctx: original(ctx) + "\nAlso be brief.")
    [result] = play_back(chosen[:1], shops, tmp_path)
    assert result.status == "crashed" and "different request" in result.error


def test_a_recording_that_ends_early_says_so(chosen, shops, tmp_path):
    record(chosen[:1], shops, tmp_path)
    path = tmp_path / "sd-01-1.json"
    data = json.loads(path.read_text())
    data["steps"] = data["steps"][:2]
    path.write_text(json.dumps(data))
    [result] = play_back(chosen[:1], shops, tmp_path)
    assert result.status == "crashed" and "has only 2" in result.error


def test_a_missing_recording_names_the_file(chosen, shops, tmp_path):
    [result] = play_back(chosen[:1], shops, tmp_path)
    assert result.status == "crashed" and "no recording for sd-01 run 1" in result.error


def test_an_unknown_recording_format_is_refused(tmp_path):
    path = tmp_path / "a-1.json"
    path.write_text(json.dumps({"format": 99, "model": "m", "steps": []}))
    with pytest.raises(ValueError, match="unsupported recording format"):
        replay.Replayer(path)


# -- the command line ------------------------------------------------------------------------------------------------


def run_cli(*args):
    return cli.main(["--evals-dir", str(EVALS), "run", *args])


def test_record_then_replay_from_the_command_line_needs_no_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    only = ["--only", "sd-01", "--only", "q-04"]
    assert run_cli("--oracle", "--record", str(tmp_path / "rec"), "--out", str(tmp_path / "o1"), *only) == 0
    capsys.readouterr()
    assert run_cli("--replay", str(tmp_path / "rec"), "--out", str(tmp_path / "o2"), *only) == 0
    out = capsys.readouterr().out
    assert "2/2" in out and "replay of rec" in out and "judge: off" in out
    [run_dir] = list((tmp_path / "o2").iterdir())
    assert (run_dir / "report.md").exists()


def test_replaying_what_was_not_recorded_is_a_clear_configuration_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    (tmp_path / "rec").mkdir()
    assert run_cli("--replay", str(tmp_path / "rec"), "--only", "sd-01") == 2
    assert "no recording of sd-01" in capsys.readouterr().err
    assert run_cli("--replay", str(tmp_path / "nowhere"), "--only", "sd-01") == 2


def test_replay_cannot_be_combined_with_oracle_or_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert run_cli("--replay", str(tmp_path), "--oracle") == 2
    assert run_cli("--replay", str(tmp_path), "--record", str(tmp_path / "x")) == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_replay_needs_enough_recorded_runs_for_the_repeat_count(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert run_cli("--oracle", "--record", str(tmp_path / "rec"), "--only", "q-04", "--out", str(tmp_path / "o")) == 0
    assert run_cli("--replay", str(tmp_path / "rec"), "--only", "q-04", "--repeat", "2", "--out", str(tmp_path / "o")) == 2
    assert "no recording of q-04 for 2 run(s)" in capsys.readouterr().err
