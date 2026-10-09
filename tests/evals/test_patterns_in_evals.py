"""Agent patterns in the evals: recorded on every result, selectable on the command line, comparable in one report."""

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.evals import cli
from jobshop.evals.compare import ModelSpec, run_comparison
from jobshop.evals.oracle import OracleClient
from jobshop.evals.runner import run_suite
from tests.evals.conftest import EVALS
from tests.helpers import FAST


def test_every_result_says_which_pattern_produced_it(scenarios, shops):
    chosen = [scenarios["sd-01"]]
    for pattern in ("react", "verify"):
        [result] = run_suite(chosen, OracleClient, AgentConfig(model="m", pattern=pattern), shops, FAST)
        assert result.pattern == pattern


def test_the_scripted_reference_agent_gets_the_same_results_under_verify_so_the_check_has_no_false_alarms(scenarios, shops, kb):
    everything = list(scenarios.values())
    plain = run_suite(everything, OracleClient, AgentConfig(model="oracle"), shops, FAST, knowledge=kb)
    checked = run_suite(everything, OracleClient, AgentConfig(model="oracle", pattern="verify"), shops, FAST, knowledge=kb)
    assert all(r.passed for r in checked)
    assert [(r.id, r.steps, r.tools_called) for r in checked] == [(r.id, r.steps, r.tools_called) for r in plain]   # no bounced answers
    assert all(r.answer["warnings"] == p.answer["warnings"] for r, p in zip(checked, plain))


def test_one_model_can_be_compared_with_itself_under_different_patterns(scenarios, shops):
    subset = [scenarios[i] for i in ("sd-01", "q-01")]
    models = [ModelSpec("m [react]", OracleClient, model="m", pattern="react"), ModelSpec("m [verify]", OracleClient, model="m", pattern="verify")]
    out = run_comparison(subset, models, AgentConfig(model="x"), shops, FAST)
    assert {r.pattern for r in out["m [react]"]} == {"react"} and {r.pattern for r in out["m [verify]"]} == {"verify"}
    assert {r.model for rs in out.values() for r in rs} == {"m"}                  # the real model id, not the label


def test_the_judge_may_not_be_the_model_behind_a_labelled_entry(scenarios, shops):
    models = [ModelSpec("m [react]", OracleClient, model="m", pattern="react"), ModelSpec("m [verify]", OracleClient, model="m", pattern="verify")]
    with pytest.raises(ValueError, match="judge"):
        run_comparison([scenarios["q-01"]], models, AgentConfig(model="x"), shops, FAST, judge=(object(), "m"))


def test_a_comparison_without_two_distinct_entries_is_still_refused(scenarios, shops):
    same = ModelSpec("m [react]", OracleClient, model="m", pattern="react")
    with pytest.raises(ValueError, match="at least two distinct"):
        run_comparison([scenarios["q-01"]], [same, same], AgentConfig(model="x"), shops, FAST)


# -- the command line ------------------------------------------------------------------------------------------------------------------------------


def run_cli(*args):
    return cli.main(["--evals-dir", str(EVALS), *args])


def test_run_takes_a_pattern_and_the_report_is_unchanged_for_the_reference_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert run_cli("run", "--oracle", "--pattern", "verify", "--only", "sd-01", "--out", str(tmp_path)) == 0


def test_a_mistyped_pattern_is_a_configuration_error_before_anything_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert run_cli("run", "--oracle", "--pattern", "nonsense", "--only", "sd-01", "--out", str(tmp_path)) == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err and "unknown pattern 'nonsense'" in err and not list(tmp_path.iterdir())


def test_compare_needs_two_models_or_two_patterns(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert run_cli("compare", "--model", "a", "--only", "q-01") == 2
    err = capsys.readouterr().err
    assert "two different models" in err and "two patterns" in err
    assert run_cli("compare", "--model", "a", "--pattern", "react", "--only", "q-01") == 2          # one model, one pattern: nothing to compare


def test_compare_rejects_a_mistyped_pattern_and_a_judge_that_is_a_model_under_test(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert run_cli("compare", "--model", "a", "--pattern", "react", "--pattern", "bogus", "--no-judge", "--only", "q-01") == 2
    assert "unknown pattern 'bogus'" in capsys.readouterr().err
    assert run_cli("compare", "--model", "a", "--pattern", "react", "--pattern", "verify", "--judge-model", "a", "--only", "q-01") == 2
    assert "must not be one of the models" in capsys.readouterr().err


# -- the route pattern in the evals ------------------------------------------------------------------------------------------------------------


def test_the_reference_agent_is_routed_where_each_scenario_says_and_passes_everything(scenarios, shops, kb):
    everything = list(scenarios.values())
    results = run_suite(everything, OracleClient, AgentConfig(model="oracle", pattern="route"), shops, FAST, knowledge=kb)
    assert all(r.passed for r in results)
    asked = {r.id: r.route for r in results}
    assert {s.id: s.expect.route for s in everything if s.expect.route} == {i: asked[i] for i in asked if scenarios[i].expect.route}
    assert asked["im-03"] == "decline_commit" and asked["am-01"] == "clarify" and asked["q-01"] == "read" and asked["sd-01"] == "plan"


def test_a_wrong_route_fails_the_scenario_but_only_under_the_route_pattern(scenarios, shops, monkeypatch):
    from jobshop.evals import oracle

    monkeypatch.setattr(oracle, "triage_for", lambda scenario: {"route": "plan", "reason": "misjudged"})   # a triage that gets q-01 wrong
    [routed] = run_suite([scenarios["q-01"]], OracleClient, AgentConfig(model="oracle", pattern="route"), shops, FAST)
    [plain] = run_suite([scenarios["q-01"]], OracleClient, AgentConfig(model="oracle"), shops, FAST)
    assert not routed.passed and routed.route == "plan" and "routed to 'plan', expected 'read'" in str(routed.checks["tools"])
    assert plain.passed and plain.route is None


def test_run_takes_a_triage_model_and_its_prices(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert run_cli("run", "--oracle", "--pattern", "route", "--triage-model", "small", "--triage-price", "0.25,1.25", "--only", "q-01", "--out", str(tmp_path)) == 0
    assert run_cli("run", "--oracle", "--pattern", "route", "--triage-price", "nonsense", "--only", "q-01", "--out", str(tmp_path)) == 2
    assert "Configuration error" in capsys.readouterr().err
