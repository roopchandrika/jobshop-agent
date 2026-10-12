"""scripts/demo_all.py: the one-command demo runs each section's real command, in order, and reports failures."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo_all.py"


@pytest.fixture
def demo(monkeypatch):
    spec = importlib.util.spec_from_file_location("demo_all", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_all"] = module     # dataclass-free, but keeps the module importable by name
    spec.loader.exec_module(module)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(module, "run", lambda *args: calls.append(args) or 0)
    return module, calls


def test_the_sections_run_in_order_with_the_real_commands(demo, capsys):
    module, calls = demo
    assert module.main(["--quick"]) == 0
    flat = [" ".join(c) for c in calls]
    assert [c.split()[1] for c in flat] == ["jobshop.evals"] * 4
    assert [c.split()[2] for c in flat] == ["redteam", "run", "retrieval", "snapshot"]
    assert "--oracle" in flat[1] and "--only simple_downtime" in flat[1]                  # quick mode runs a subset
    out = capsys.readouterr().out
    assert "All sections passed" in out and out.index("1. ") < out.index("4. ")


def test_the_full_run_covers_every_scenario_and_ends_in_the_web_demo(demo):
    module, calls = demo
    assert module.main([]) == 0
    flat = [" ".join(c) for c in calls]
    assert "--only" not in flat[1] and flat[-1].startswith("scripts/demo_server.py") and "--port 8765" in flat[-1]


def test_no_web_skips_the_server(demo):
    module, calls = demo
    assert module.main(["--no-web"]) == 0 and not any("demo_server" in " ".join(c) for c in calls)


def test_a_failing_section_is_named_and_makes_the_demo_fail(demo, monkeypatch, capsys):
    module, _ = demo
    monkeypatch.setattr(module, "run", lambda *args: 1 if "snapshot" in args else 0)
    assert module.main(["--quick"]) == 1
    assert "Sections that failed: snapshot" in capsys.readouterr().out


def test_pause_waits_for_enter_before_each_section(demo, monkeypatch):
    module, _ = demo
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "")
    assert module.main(["--quick", "--pause"]) == 0 and len(asked) == 4
