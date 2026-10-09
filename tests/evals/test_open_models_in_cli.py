"""Open models on the command line: a run that uses only local models needs no Anthropic key; any Anthropic model still does."""

import pytest

from jobshop.agent.providers import MultiProviderClient
from jobshop.evals import cli
from tests.evals.conftest import EVALS


class Unreachable:
    """Stands in for a client whose server is not running, like an Ollama that was never started."""

    def __init__(self):
        import anthropic
        import httpx2

        self.error = anthropic.APIConnectionError(message="cannot reach Ollama at http://127.0.0.1:11434", request=httpx2.Request("POST", "http://x"))
        self.messages = self

    def create(self, **_):
        raise self.error


def run_cli(*args):
    return cli.main(["--evals-dir", str(EVALS), *args])


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_JUDGE_MODEL", "JOBSHOP_TRIAGE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_a_local_model_run_needs_no_key_and_a_dead_server_is_a_failed_scenario_not_a_crash(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "build_client", lambda env, *models: seen.append(models) or Unreachable())
    assert run_cli("run", "--model", "ollama:llama3.1", "--no-judge", "--only", "q-01", "--out", str(tmp_path)) == 1
    assert seen == [("ollama:llama3.1", None, None)] and "Configuration error" not in capsys.readouterr().err


def test_an_anthropic_model_still_needs_the_key_even_when_the_triage_model_is_local(tmp_path, capsys):
    assert run_cli("run", "--model", "claude-x", "--pattern", "route", "--triage-model", "ollama:llama3.1", "--no-judge", "--only", "q-01", "--out", str(tmp_path)) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_a_local_triage_model_beside_a_local_main_model_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_client", lambda env, *models: Unreachable())
    assert run_cli("run", "--model", "ollama:big", "--pattern", "route", "--triage-model", "ollama:small", "--no-judge", "--only", "q-01", "--out", str(tmp_path)) == 1


def test_a_local_model_with_an_anthropic_judge_needs_the_key_for_the_judge(tmp_path, capsys):
    assert run_cli("run", "--model", "ollama:llama3.1", "--judge-model", "claude-j", "--only", "q-01", "--out", str(tmp_path)) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_the_real_client_for_a_mixed_run_routes_by_name_and_makes_no_anthropic_client_up_front(monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda: pytest.fail("no Anthropic client should be built for a local-only run"))
    assert isinstance(cli.build_client({}, "ollama:a", "ollama:b", None), MultiProviderClient)
