import json

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.pricing import Prices
from jobshop.evals import cli
from jobshop.evals.compare import ModelSpec, model_metrics, pair_stats, render_comparison, run_comparison, write_comparison
from jobshop.evals.oracle import OracleClient, flaw_for
from jobshop.evals.runner import ScenarioResult
from tests.evals.conftest import EVALS
from tests.helpers import FAST

META = dict(run_id="r1", judge_model="the-judge", scenarios=4, repeat=1, solve_seconds=5)


def result(id_, passed, attempt=1, cost=0.01, wall=2.0, failed_check="tools", **extra):
    checks = {"outcome": {"passed": True, "details": []}, "tools": {"passed": passed, "details": [] if passed else ["x"]}}
    if not passed:
        checks = {**checks, failed_check: {"passed": False, "details": ["x"]}}
    return ScenarioResult(id_, "simple_downtime", attempt, "answered", passed, steps=4, input_tokens=1000, output_tokens=100,
                          cost_usd=cost, wall_s=wall, llm_ms=1500, tool_ms=500, checks=checks, **extra)


# -- metrics -----------------------------------------------------------------------------------------


def test_model_metrics_summarise_quality_cost_and_latency():
    rs = [result("a", True, cost=0.01, wall=1.0), result("b", True, cost=0.03, wall=3.0), result("c", False, cost=0.02, wall=5.0)]
    m = model_metrics(rs)
    assert (m["passed"], m["runs"]) == (2, 3) and m["pass_rate"] == pytest.approx(2 / 3)
    assert m["pass_ci"][0] < 2 / 3 < m["pass_ci"][1]
    assert m["cost_total_usd"] == pytest.approx(0.06) and m["cost_per_run_usd"] == pytest.approx(0.02)
    assert m["cost_per_pass_usd"] == pytest.approx(0.03)  # a failed run's money still counts against the passes
    assert (m["wall_s_mean"], m["wall_s_p50"]) == (3.0, 3.0) and m["wall_s_p95"] == pytest.approx(4.8)
    assert (m["llm_s_per_run"], m["tool_s_per_run"]) == (1.5, 0.5)
    assert (m["input_tokens_per_run"], m["output_tokens_per_run"], m["steps_per_run"]) == (1000, 100, 4)


def test_without_prices_cost_is_none_everywhere_not_zero():
    m = model_metrics([result("a", True, cost=None), result("b", True, cost=None)])
    assert m["cost_total_usd"] is None and m["cost_per_run_usd"] is None and m["cost_per_pass_usd"] is None


def test_one_unpriced_run_makes_the_total_unknown_rather_than_understated():
    assert model_metrics([result("a", True, cost=0.01), result("b", True, cost=None)])["cost_total_usd"] is None


def test_cost_per_passing_run_is_undefined_when_nothing_passes():
    m = model_metrics([result("a", False), result("b", False)])
    assert m["cost_total_usd"] == pytest.approx(0.02) and m["cost_per_pass_usd"] is None


def test_the_judge_mean_score_averages_every_criterion_of_every_run():
    judged = lambda scores: dict(judge={"scores": scores, "reasons": {}, "error": None})  # noqa: E731
    rs = [result("a", True, **judged({"accuracy": 5, "honesty": 3})), result("b", True, **judged({"accuracy": 4, "honesty": 4}))]
    assert model_metrics(rs)["judge_mean_score"] == pytest.approx(4.0)
    assert model_metrics([result("a", True)])["judge_mean_score"] is None


# -- paired comparison ------------------------------------------------------------------------------------


def test_pairing_is_by_scenario_and_attempt_and_names_the_failing_checks():
    a = [result("s1", True), result("s2", True), result("s3", False), result("s4", False), result("s1", True, attempt=2)]
    b = [result("s1", True), result("s2", False, failed_check="numbers"), result("s3", True), result("s4", False), result("only-in-b", True)]
    p = pair_stats(a, b)
    assert (p["both"], p["neither"]) == (1, 1)
    assert [(i, n) for i, n, _ in p["only_a"]] == [("s2", 1)] and "numbers" in p["only_a"][0][2]
    assert [(i, n) for i, n, _ in p["only_b"]] == [("s3", 1)]
    assert p["p_value"] == 1.0  # one win each: no evidence either way


def test_repeat_runs_of_one_scenario_are_paired_attempt_by_attempt():
    # With --repeat the same scenario appears several times; run 2 must be compared with run 2, not run 1.
    a = [result("s1", True, attempt=1), result("s1", True, attempt=2)]
    b = [result("s1", True, attempt=1), result("s1", False, attempt=2)]
    p = pair_stats(a, b)
    assert p["both"] == 1 and [(i, n) for i, n, _ in p["only_a"]] == [("s1", 2)]


def test_a_clean_sweep_on_few_disagreements_is_still_not_significant():
    a = [result(f"s{i}", True) for i in range(4)]
    b = [result(f"s{i}", False) for i in range(4)]
    assert pair_stats(a, b)["p_value"] == pytest.approx(0.125)


# -- the rendered report ----------------------------------------------------------------------------------------


def test_the_report_puts_quality_cost_and_latency_side_by_side():
    results = {"model-a": [result(f"s{i}", True, cost=0.02) for i in range(4)],
               "model-b": [result(f"s{i}", i < 2, cost=0.01) for i in range(4)]}
    text = render_comparison(META, results)
    assert "| model-a" in text and "model-b" in text and "the-judge" in text
    assert "100% (4/4)" in text and "50% (2/4)" in text
    assert "per passing run" in text and "$0.02000" in text   # model-a: 0.02 per run, every run passes
    assert "$0.02000 | $0.02000" not in text and "$0.01000" in text and "$0.02000" in text.split("per passing run")[1]
    assert "mean / median / p95" in text and "waiting for the model" in text and "running tools" in text


def test_cost_per_passing_run_can_reverse_the_cheaper_per_run_ranking():
    # b is half the price per run but passes a quarter as often: it costs MORE per solved scenario.
    results = {"a": [result(f"s{i}", True, cost=0.02) for i in range(4)],
               "b": [result(f"s{i}", i == 0, cost=0.01) for i in range(4)]}
    ma, mb = (model_metrics(results[k]) for k in "ab")
    assert mb["cost_per_run_usd"] < ma["cost_per_run_usd"] and mb["cost_per_pass_usd"] > ma["cost_per_pass_usd"]


def test_the_report_says_unpriced_models_have_no_cost_instead_of_zero():
    text = render_comparison(META, {"a": [result("s", True, cost=None)], "b": [result("s", True, cost=0.01)]})
    assert "n/a" in text.split("**Cost**")[1].split("**Latency**")[0]


def test_the_verdict_wording_follows_the_evidence():
    four_to_none = {"a": [result(f"s{i}", True) for i in range(4)], "b": [result(f"s{i}", False) for i in range(4)]}
    assert "plausible by chance" in render_comparison(META, four_to_none)
    ten_to_none = {"a": [result(f"s{i}", True) for i in range(10)], "b": [result(f"s{i}", False) for i in range(10)]}
    assert "unlikely to be chance" in render_comparison(META, ten_to_none)
    same = {"a": [result("s", True)], "b": [result("s", True)]}
    assert "exactly the same runs" in render_comparison(META, same)


def test_scenarios_one_model_alone_passed_are_listed_with_the_check_the_other_failed():
    text = render_comparison(META, {"a": [result("s1", True)], "b": [result("s1", False, failed_check="numbers")]})
    assert "s1 (run 1): only `a` passed; `b` failed: tools, numbers" in text


# -- running a comparison -----------------------------------------------------------------------------------


def models(prices_a=Prices(3.0, 15.0), prices_b=None):
    return [ModelSpec("careful", lambda s: OracleClient(s, usage=(1000, 200)), prices_a),
            ModelSpec("sloppy", lambda s: OracleClient(s, sloppy=True, usage=(500, 100)), prices_b)]


@pytest.fixture
def subset(scenarios):
    return [scenarios[i] for i in ("sd-01", "pc-01", "ro-01", "md-02", "q-01", "am-01")]


def test_each_model_is_run_with_its_own_prices_and_tokens(subset, shop):
    out = run_comparison(subset, models(), AgentConfig(model="x", max_cost_usd=None), shop, FAST)
    careful, sloppy = out["careful"], out["sloppy"]
    assert len(careful) == len(sloppy) == len(subset)
    assert [r.model for r in careful] == ["careful"] * len(subset)
    sd01 = careful[0]
    assert sd01.cost_usd == pytest.approx(sd01.steps * (1000 * 3 + 200 * 15) / 1e6)
    assert all(r.cost_usd is None for r in sloppy)   # no price was given for it, and the other's was not borrowed
    assert careful[0].input_tokens == careful[0].steps * 1000


def test_two_priced_models_are_each_costed_at_their_own_prices(subset, shop):
    out = run_comparison(subset, models(prices_b=Prices(1.0, 5.0)), AgentConfig(model="x"), shop, FAST)
    careful, sloppy = out["careful"][0], out["sloppy"][0]
    assert careful.cost_usd == pytest.approx(careful.steps * (1000 * 3.0 + 200 * 15.0) / 1e6)
    assert sloppy.cost_usd == pytest.approx(sloppy.steps * (500 * 1.0 + 100 * 5.0) / 1e6)


def test_the_sloppy_agent_fails_exactly_where_it_is_scripted_to(scenarios, shops, kb):
    out = run_comparison(list(scenarios.values()), models(), AgentConfig(model="x"), shops, FAST, knowledge=kb)
    failing = {r.id for r in out["sloppy"] if not r.passed}
    assert failing == {s.id for s in scenarios.values() if flaw_for(s)} and failing  # and nothing else
    assert all(r.passed for r in out["careful"])
    flawed = {r.id: [n for n, c in r.checks.items() if c["passed"] is False] for r in out["sloppy"] if not r.passed}
    assert {tuple(v) for v in flawed.values()} <= {("tools",), ("numbers",)}


def test_comparison_refuses_unfair_or_meaningless_setups(subset, shop):
    base = AgentConfig(model="x")
    with pytest.raises(ValueError, match="at least two distinct"):
        run_comparison(subset, models()[:1], base, shop, FAST)
    with pytest.raises(ValueError, match="at least two distinct"):
        run_comparison(subset, [models()[0], models()[0]], base, shop, FAST)
    with pytest.raises(ValueError, match="must not be one of the models"):
        run_comparison(subset, models(), base, shop, FAST, judge=(object(), "sloppy"))


def test_a_comparison_is_written_as_per_model_reports_plus_a_summary(subset, shop, tmp_path):
    out = tmp_path / "cmp"
    results = run_comparison(subset, models(), AgentConfig(model="x"), shop, FAST, out_dir=out)
    write_comparison(out, META, results)
    assert (out / "comparison.md").read_text().startswith("# Model comparison r1")
    data = json.loads((out / "comparison.json").read_text())
    assert set(data["metrics"]) == {"careful", "sloppy"}
    for name in ("careful", "sloppy"):
        assert (out / name / "report.md").exists() and (out / name / "results.json").exists()
        assert len(list((out / name / "traces").glob("*.jsonl"))) == len(subset)


# -- the command line ---------------------------------------------------------------------------------------


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for name in ("ANTHROPIC_MODEL", "ANTHROPIC_COMPARE_MODEL", "ANTHROPIC_JUDGE_MODEL", "JOBSHOP_PRICE_INPUT_PER_MTOK",
                 "JOBSHOP_PRICE_OUTPUT_PER_MTOK", "JOBSHOP_MAX_COST_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def compare(*args):
    return cli.main(["--evals-dir", str(EVALS), "compare", *args])


def test_the_demo_comparison_runs_end_to_end(quiet, tmp_path, capsys):
    assert compare("--demo", "--only", "simple_downtime", "--out", str(tmp_path), "--solve-seconds", "1") == 0
    out = capsys.readouterr().out
    assert "DEMO" in out and "demo-careful" in out and "demo-sloppy" in out
    assert "Sign test" in out or "exactly the same runs" in out
    [run_dir] = list(tmp_path.iterdir())
    assert (run_dir / "comparison.md").exists()


@pytest.mark.parametrize("args, message", [
    (["--model", "a"], "two different models"),
    ([], "two different models"),
    (["--model", "a", "--model", "a"], "two different models"),
    (["--model", "a", "--model", "b", "--judge-model", "a"], "must not be one of the models"),
    (["--model", "a", "--model", "b"], "no judge model"),
    (["--model", "a", "--model", "b", "--no-judge", "--price", "c=1,2"], "not one of the models"),
    (["--model", "a", "--model", "b", "--no-judge", "--price", "a=cheap"], "input,output"),
])
def test_comparison_configuration_mistakes_exit_2(quiet, capsys, args, message):
    assert compare(*args) == 2
    assert message in capsys.readouterr().err


def test_a_missing_api_key_is_reported_before_anything_runs(quiet, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert compare("--model", "a", "--model", "b", "--no-judge") == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_one_models_environment_prices_never_leak_into_a_comparison(quiet, monkeypatch):
    monkeypatch.setenv("JOBSHOP_PRICE_INPUT_PER_MTOK", "99")
    monkeypatch.setenv("JOBSHOP_PRICE_OUTPUT_PER_MTOK", "99")
    monkeypatch.setenv("JOBSHOP_MAX_COST_USD", "1")
    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda: object())
    seen = {}

    def capture(scenarios, specs, base, *a, **k):
        seen.update(base=base, specs=specs)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_comparison", capture)
    with pytest.raises(SystemExit):
        compare("--model", "a", "--model", "b", "--no-judge", "--price", "a=3,15")
    assert seen["base"].price_input_per_mtok is None and seen["base"].max_cost_usd is None
    assert {s.name: s.prices for s in seen["specs"]} == {"a": Prices(3.0, 15.0), "b": None}


def test_run_accepts_prices_on_the_command_line(quiet, monkeypatch, tmp_path):
    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda: object())
    seen = {}
    monkeypatch.setattr(cli, "run_suite", lambda scenarios, client_for, config, *a, **k: seen.update(config=config) or [])
    code = cli.main(["--evals-dir", str(EVALS), "run", "--model", "a", "--no-judge", "--price", "3,15", "--only", "q-01", "--out", str(tmp_path)])
    assert code == 0 and seen["config"].prices == Prices(3.0, 15.0)
