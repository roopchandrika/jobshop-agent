import json

import pytest

from jobshop.evals import cli
from jobshop.evals.oracle import OracleClient
from jobshop.evals.report import CHECKS, render_markdown, render_table, summarize, write_results
from jobshop.evals.runner import ScenarioResult, run_suite
from jobshop.evals.shop import fresh_context
from tests.evals.conftest import EVALS
from tests.helpers import FAST


def result(id_, category, passed, **checks):
    return ScenarioResult(id_, category, 1, "answered", passed,
                          checks={n: {"passed": p, "details": [f"{n} failed"] if p is False else []} for n, p in checks.items()},
                          input_tokens=100, output_tokens=50, cost_usd=0.01, wall_s=2.0, answer={"summary": "text"})


RESULTS = [
    result("a", "simple_downtime", True, outcome=True, tools=True, changes=True, validator=True, numbers=True),
    result("b", "simple_downtime", False, outcome=True, tools=True, changes=False, validator=True, numbers=True),
    result("c", "read_only", True, outcome=True, tools=True, changes=None, validator=None, numbers=True),
]


def test_summary_counts_passed_over_applicable_and_skips_checks_that_did_not_apply():
    s = summarize(RESULTS)
    assert s["overall"]["passed"] == 2 and s["overall"]["runs"] == 3
    assert s["overall"]["checks"]["changes"] == (1, 2)      # the read-only run had no changes check
    assert s["overall"]["checks"]["judge"] == (0, 0)        # judge off: nothing applicable
    assert s["by_category"]["read_only"]["checks"]["validator"] == (0, 0)
    assert s["agent_tokens"] == 450 and s["agent_cost_usd"] == 0.03


def test_the_table_shows_fractions_and_dashes_and_a_total_row():
    table = render_table(summarize(RESULTS))
    lines = table.splitlines()
    assert lines[0].split() == ["category", "runs", *CHECKS, "all"]
    total = lines[-1].split()
    assert total[0] == "TOTAL" and total[1] == "3" and "1/2" in total and total[-1] == "2/3"
    assert any(line.startswith("read_only") and " - " in line for line in lines)


def test_the_markdown_lists_each_failure_with_its_reason():
    meta = dict(run_id="r1", model="m", judge_model=None, scenarios=3, repeat=1, solve_seconds=5)
    text = render_markdown(meta, summarize(RESULTS), RESULTS)
    assert "### b (attempt 1)" in text and "**changes**: changes failed" in text
    assert "judge skipped" in text and "a (attempt" not in text


def test_judge_errors_and_crashes_are_called_out_not_hidden():
    crashed = ScenarioResult("x", "read_only", 1, "crashed", False, error="boom")
    errored = result("y", "read_only", True, outcome=True, judge=None)
    s = summarize([crashed, errored])
    assert s["crashed"] == ["x"] and s["judge_errors"] == 1
    meta = dict(run_id="r", model="m", judge_model="j", scenarios=2, repeat=1, solve_seconds=5)
    text = render_markdown(meta, s, [crashed, errored])
    assert "judge call(s) failed" in text and "Crashed runs" in text and "boom" in text


def test_results_are_written_as_json_and_markdown(tmp_path):
    meta = dict(run_id="r1", model="m", judge_model=None, scenarios=3, repeat=1, solve_seconds=5)
    out = write_results(tmp_path / "run", meta, RESULTS)
    data = json.loads((out / "results.json").read_text())
    assert data["meta"]["model"] == "m" and len(data["results"]) == 3 and data["summary"]["overall"]["runs"] == 3
    assert (out / "report.md").read_text().startswith("# Eval run r1")


def test_a_crash_in_one_scenario_costs_only_that_scenario(shop, scenarios):
    from jobshop.agent.loop import AgentConfig

    chosen = [scenarios["q-01"], scenarios["q-02"]]

    class Exploding:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            raise RuntimeError("harness bug")

    results = run_suite(chosen, lambda s: Exploding() if s.id == "q-01" else OracleClient(s),
                        AgentConfig(model="m"), shop, FAST)
    assert [r.status for r in results] == ["crashed", "answered"]
    assert results[0].error == "RuntimeError: harness bug" and results[1].passed


def test_each_scenario_runs_on_the_shop_it_names(shops, scenarios):
    from jobshop.agent.loop import AgentConfig

    # ts-05's reference answer quotes slack figures that only exist in the tight shop's plan.
    [tight] = run_suite([scenarios["ts-05"]], OracleClient, AgentConfig(model="m"), shops, FAST)
    assert tight.passed, tight.checks
    [wrong] = run_suite([scenarios["ts-05"]], OracleClient, AgentConfig(model="m"), {"default": shops["default"], "tight": shops["default"]}, FAST)
    assert not wrong.passed and wrong.checks["numbers"]["passed"] is False


def test_a_scenario_whose_shop_was_not_loaded_is_a_setup_error_not_a_crash(shop, scenarios):
    from jobshop.agent.loop import AgentConfig

    with pytest.raises(ValueError, match=r"not loaded: \['tight'\]"):
        run_suite([scenarios["ts-01"]], OracleClient, AgentConfig(model="m"), shop, FAST)   # a bare shop means "default"


def test_fresh_contexts_are_independent_and_plant_hostile_notes(shop):
    a = fresh_context(shop, FAST, poison={"O-108": "hostile"})
    b = fresh_context(shop, FAST)
    a.store.create_draft()
    assert a.store.drafts() and not b.store.drafts()
    assert a.store.committed.instance.order("O-108").notes == "hostile"
    assert b.store.committed.instance.order("O-108").notes == ""
    assert a.store.committed.instance.now == b.store.committed.instance.now > 0


def test_fresh_context_rejects_a_poison_for_an_order_that_does_not_exist(shop):
    with pytest.raises(KeyError):
        fresh_context(shop, FAST, poison={"O-999": "x"})


# -- the command line ---------------------------------------------------------------------------------


def run_cli(*args):
    return cli.main(["--evals-dir", str(EVALS), *args])


def test_the_oracle_run_writes_a_report_and_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    code = run_cli("run", "--oracle", "--only", "read_only", "--out", str(tmp_path), "--solve-seconds", "1")
    out = capsys.readouterr().out
    assert code == 0 and "TOTAL" in out and "5/5" in out   # read_only: q-01..q-04 on the default shop, ts-05 on the tight one
    [run_dir] = list(tmp_path.iterdir())
    assert (run_dir / "report.md").exists() and len(list((run_dir / "traces").glob("*.jsonl"))) == 5


def test_a_run_with_a_failure_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    broken = tmp_path / "evals"
    (broken / "scenarios").mkdir(parents=True)
    (broken / "shop.json").write_text((EVALS / "shop.json").read_text())
    # An unsatisfiable scenario: it expects a usable proposal, but a shaft cannot finish in the last hour.
    (broken / "scenarios" / "x.yaml").write_text(
        "scenarios:\n  - id: x\n    category: rush_order\n    description: d\n    now: '2026-01-05 21:00'\n"
        "    request: 'Add a rush shaft order due 21:30 today.'\n"
        "    expect:\n      outcome: proposal\n      changes: {rush_orders: [{family: shaft, due: '2026-01-05 21:30', priority: 5}]}\n"
    )
    assert cli.main(["--evals-dir", str(broken), "run", "--oracle", "--out", str(tmp_path / "o")]) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("args, message", [
    (["run", "--model", "m", "--judge-model", "m"], "different model"),
    (["run", "--model", "m"], "no judge model"),
    (["run", "--oracle", "--only", "no-such-scenario"], "matches no scenario"),
])
def test_configuration_mistakes_exit_2_with_a_clear_message(monkeypatch, capsys, args, message):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for name in ("ANTHROPIC_MODEL", "ANTHROPIC_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert run_cli(*args) == 2
    assert message in capsys.readouterr().err


def test_a_missing_api_key_is_reported_before_any_call(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run_cli("run", "--model", "m", "--judge-model", "j") == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_a_missing_shop_fixture_says_how_to_create_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "one.yaml").write_text((EVALS / "scenarios" / "01_simple_downtime.yaml").read_text())
    assert cli.main(["--evals-dir", str(tmp_path), "run", "--oracle", "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "build-shop" in err and "shop.json" in err


def test_a_scenario_on_the_tight_shop_needs_that_fixture_too(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "t.yaml").write_text((EVALS / "scenarios" / "06_goals_and_tight_shop.yaml").read_text())
    (tmp_path / "shop.json").write_text((EVALS / "shop.json").read_text())   # the default shop alone is not enough
    assert cli.main(["--evals-dir", str(tmp_path), "run", "--oracle", "--only", "ts-01", "--out", str(tmp_path / "o")]) == 2
    assert "build-shop --shop tight" in capsys.readouterr().err


def test_only_the_shops_a_selection_needs_are_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios" / "t.yaml").write_text((EVALS / "scenarios" / "06_goals_and_tight_shop.yaml").read_text())
    (tmp_path / "shop_tight.json").write_text((EVALS / "shop_tight.json").read_text())   # no default shop on disk
    assert cli.main(["--evals-dir", str(tmp_path), "run", "--oracle", "--only", "ts-01", "--out", str(tmp_path / "o")]) == 0
