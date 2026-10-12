"""scripts/load_test_container.py and scripts/demo_traces.py: what they ask Docker to do, and that they always clean up."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Docker:
    """A stand-in for the docker command line: records every call and answers from a table."""

    def __init__(self, fail=()):
        self.calls: list[tuple[str, ...]] = []
        self.fail = set(fail)

    def __call__(self, *args, **kw):
        self.calls.append(args)
        failed = any(f in args for f in self.fail)
        return subprocess.CompletedProcess(args, 1 if failed else 0, stdout="log line", stderr="boom" if failed else "")

    def ran(self, *words):
        return [c for c in self.calls if all(w in c for w in words)]


@pytest.fixture
def lt():
    return load("load_test_container")


@pytest.fixture
def dt():
    return load("demo_traces")


# -- the container load test ----------------------------------------------------------------------------------------------------------------------


def test_the_container_is_published_to_loopback_only_with_the_scripts_mounted_read_only(lt, monkeypatch):
    docker = Docker()
    monkeypatch.setattr(lt, "sh", docker)
    lt.start(8000)
    [run] = docker.ran("run", "-d")
    assert "127.0.0.1:8000:8000" in run and "-p" in run
    mount = run[run.index("-v") + 1]
    assert mount.endswith(":/app/scripts:ro") and "scripts" in mount
    assert "--entrypoint" in run and run[-2:] == ("-c", lt.SERVE)
    assert docker.ran("rm", "-f", lt.NAME)                       # a stale container of the same name is removed first


def test_docker_problems_are_named(lt, monkeypatch):
    monkeypatch.setattr(lt.shutil, "which", lambda name: None)
    assert "not installed" in lt.docker_ready()
    monkeypatch.setattr(lt.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(lt, "sh", Docker(fail=("info",)))
    assert "daemon is not running" in lt.docker_ready()
    monkeypatch.setattr(lt, "sh", Docker(fail=("inspect",)))
    assert "does not exist" in lt.docker_ready() and "docker build" in lt.docker_ready()
    monkeypatch.setattr(lt, "sh", Docker())
    assert lt.docker_ready() is None


def test_without_docker_the_command_exits_2_and_starts_nothing(lt, monkeypatch, capsys):
    monkeypatch.setattr(lt, "docker_ready", lambda: "the Docker daemon is not running (start Docker Desktop)")
    started = []
    monkeypatch.setattr(lt, "start", lambda port: started.append(port))
    assert lt.main([]) == 2 and not started
    assert "Cannot run" in capsys.readouterr().err


def test_the_container_is_stopped_even_when_the_load_test_fails(lt, monkeypatch):
    monkeypatch.setattr(lt, "docker_ready", lambda: None)
    monkeypatch.setattr(lt, "start", lambda port: None)
    monkeypatch.setattr(lt, "wait_healthy", lambda port, seconds=60: True)
    stopped = []
    monkeypatch.setattr(lt, "stop", lambda: stopped.append(1))
    monkeypatch.setattr(lt.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 3))
    assert lt.main(["--users", "5"]) == 3 and stopped == [1]


def test_a_container_that_never_gets_healthy_is_reported_with_its_log_and_stopped(lt, monkeypatch, capsys):
    monkeypatch.setattr(lt, "docker_ready", lambda: None)
    monkeypatch.setattr(lt, "start", lambda port: None)
    monkeypatch.setattr(lt, "wait_healthy", lambda port, seconds=60: False)
    stopped = []
    monkeypatch.setattr(lt, "stop", lambda: stopped.append(1))
    monkeypatch.setattr(lt, "sh", Docker())
    assert lt.main([]) == 1 and stopped == [1]
    assert "did not become healthy" in capsys.readouterr().err


def test_the_load_test_is_pointed_at_the_published_port_with_the_chosen_load(lt, monkeypatch):
    monkeypatch.setattr(lt, "docker_ready", lambda: None)
    monkeypatch.setattr(lt, "start", lambda port: None)
    monkeypatch.setattr(lt, "wait_healthy", lambda port, seconds=60: True)
    monkeypatch.setattr(lt, "stop", lambda: None)
    seen = []
    monkeypatch.setattr(lt.subprocess, "run", lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert lt.main(["--users", "7", "--requests", "9", "--port", "8123"]) == 0
    cmd = seen[0]
    assert cmd[cmd.index("--url") + 1] == "http://127.0.0.1:8123" and cmd[cmd.index("--users") + 1] == "7" and cmd[cmd.index("--requests") + 1] == "9"


# -- the Jaeger demo ----------------------------------------------------------------------------------------------------------------------


def test_jaeger_is_published_to_loopback_only(dt, monkeypatch):
    docker = Docker()
    monkeypatch.setattr(dt, "sh", docker)
    dt.start_jaeger()
    [run] = docker.ran("run", "-d")
    assert "127.0.0.1:16686:16686" in run and "127.0.0.1:4318:4318" in run and dt.IMAGE in run


def test_the_demo_sends_the_given_traces_to_the_collector_and_stops_jaeger(dt, monkeypatch, capsys, tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"event": "turn_start"}\n', encoding="utf-8")
    monkeypatch.setattr(dt, "docker_problem", lambda: None)
    monkeypatch.setattr(dt, "start_jaeger", lambda: None)
    monkeypatch.setattr(dt, "wait_ready", lambda seconds=90.0: True)
    stopped = []
    monkeypatch.setattr(dt, "sh", lambda *a: stopped.append(a) or subprocess.CompletedProcess(a, 0))
    import jobshop.agent.trace_export as te
    import jobshop.agent.trace_report as tr

    sent = []
    monkeypatch.setattr(tr, "read_trace", lambda path: [{"event": "x", "path": str(path)}])
    monkeypatch.setattr(te, "to_otlp", lambda records: {"resourceSpans": [{"scopeSpans": [{"spans": [1, 2, 3]}]}]})
    monkeypatch.setattr(te, "send", lambda doc, url: sent.append(url) or 200)
    assert dt.main(["--trace", str(trace), "--no-wait"]) == 0
    assert sent == ["http://127.0.0.1:4318/v1/traces"]
    assert any(c[:2] == ("docker", "stop") for c in stopped)
    out = capsys.readouterr().out
    assert "Sent 3 spans" in out and "http://127.0.0.1:16686/search?service=jobshop-agent" in out


def test_keep_leaves_jaeger_running_and_says_how_to_stop_it(dt, monkeypatch, capsys, tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(dt, "docker_problem", lambda: None)
    monkeypatch.setattr(dt, "start_jaeger", lambda: None)
    monkeypatch.setattr(dt, "wait_ready", lambda seconds=90.0: True)
    stopped = []
    monkeypatch.setattr(dt, "sh", lambda *a: stopped.append(a) or subprocess.CompletedProcess(a, 0))
    import jobshop.agent.trace_export as te
    import jobshop.agent.trace_report as tr

    monkeypatch.setattr(tr, "read_trace", lambda path: [])
    monkeypatch.setattr(te, "to_otlp", lambda records: {"resourceSpans": []})
    monkeypatch.setattr(te, "send", lambda doc, url: 200)
    assert dt.main(["--trace", str(trace), "--keep"]) == 0
    assert not stopped and f"docker stop {dt.NAME}" in capsys.readouterr().out


def test_without_docker_the_demo_exits_2(dt, monkeypatch, capsys):
    monkeypatch.setattr(dt, "docker_problem", lambda: "docker is not installed or not on PATH")
    assert dt.main([]) == 2 and "Cannot run" in capsys.readouterr().err


def test_jaeger_is_stopped_when_it_fails_to_come_up(dt, monkeypatch, capsys):
    monkeypatch.setattr(dt, "docker_problem", lambda: None)
    monkeypatch.setattr(dt, "start_jaeger", lambda: None)
    monkeypatch.setattr(dt, "wait_ready", lambda seconds=90.0: False)
    calls = []
    monkeypatch.setattr(dt, "sh", lambda *a: calls.append(a) or subprocess.CompletedProcess(a, 0, stdout="log"))
    assert dt.main([]) == 1
    assert any(c[:2] == ("docker", "stop") for c in calls) and "did not come up" in capsys.readouterr().err
