"""Health check, streamed progress (Server-Sent Events), container mode, and the Dockerfile's static properties."""

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from jobshop.agent.loop import AgentConfig
from jobshop.api import app as app_module
from jobshop.api import server
from jobshop.api.app import STATIC, create_app
from tests.api.conftest import Web, proposal_script
from tests.api.test_server import quiet, run_main_capturing_the_app  # noqa: F401  (a fixture and a helper)
from tests.evals.conftest import EVALS
from tests.fake_llm import FakeClient, message, submit, tool

ROOT = Path(__file__).resolve().parents[2]


def parse_events(text):
    """[(event, data-dict-or-None, id)] from a Server-Sent Events body; comments are returned as ('comment', text, None)."""
    out = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            out.append(("comment", block[1:].strip(), None))
            continue
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((fields["event"], json.loads(fields["data"]), fields.get("id")))
    return out


# -- health --------------------------------------------------------------------------------------------------------------------------


def test_the_health_endpoint_answers_without_a_token_and_shows_no_plan_data(web):
    r = web.http.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_configured": True}


def test_health_still_answers_while_the_assistant_is_working(web):
    web.app.state.web.lock.acquire()
    try:
        assert web.http.get("/healthz").json()["status"] == "ok"
    finally:
        web.app.state.web.lock.release()


def test_health_says_when_no_model_is_configured(ctx):
    from fastapi.testclient import TestClient
    app = create_app(ctx, None, AgentConfig(model="none"), allowed_hosts=("testserver",))
    assert TestClient(app).get("/healthz").json() == {"status": "ok", "model_configured": False}


def test_health_has_the_same_security_headers_and_host_check_as_everything_else(web):
    r = web.http.get("/healthz")
    assert "default-src 'none'" in r.headers["content-security-policy"] and r.headers["cache-control"] == "no-store"
    assert web.http.get("/healthz", headers={"Host": "attacker.example"}).status_code == 400


# -- the event stream (finished turns) -----------------------------------------------------------------------------------------------------------


def test_a_finished_turn_replays_its_progress_then_done(proposed):
    r = proposed.http.get("/api/chat/1/events")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no" and r.headers["cache-control"] == "no-store"
    events = parse_events(r.text)
    assert [e[0] for e in events] == ["progress"] * 4 + ["done"]
    assert [e[1]["tool"] for e in events[:4]] == ["create_draft", "change_priority", "reschedule", "compare_schedules"]
    assert [e[1]["index"] for e in events[:4]] == [1, 2, 3, 4] and [e[2] for e in events[:4]] == ["1", "2", "3", "4"]
    assert events[-1][1] == {"status": "done", "error": None} and events[-1][2] is None


def test_a_reconnecting_browser_gets_only_what_it_missed(proposed):
    everything = parse_events(proposed.http.get("/api/chat/1/events").text)
    missed = parse_events(proposed.http.get("/api/chat/1/events", headers={"Last-Event-ID": "2"}).text)
    assert [e[2] for e in missed[:-1]] == [e[2] for e in everything[2:-1]] and missed[-1][0] == "done"
    caught_up = parse_events(proposed.http.get("/api/chat/1/events", headers={"Last-Event-ID": str(len(everything) - 1)}).text)
    assert [e[0] for e in caught_up] == ["done"]


@pytest.mark.parametrize("header", ["abc", "-5", "", "1.5", "9" * 40])
def test_a_garbage_last_event_id_is_treated_as_the_start_not_an_error(proposed, header):
    assert parse_events(proposed.http.get("/api/chat/1/events", headers={"Last-Event-ID": header}).text)[0][0] in ("progress", "done")


def test_an_unknown_or_replaced_turn_is_a_404(proposed):
    assert proposed.http.get("/api/chat/99/events").status_code == 404
    proposed.fake.script += [submit(summary="Another.")]
    proposed.chat("again")
    assert proposed.http.get("/api/chat/1/events").status_code == 404 and proposed.http.get("/api/chat/2/events").status_code == 200


def test_a_turn_that_crashed_still_ends_its_stream_with_the_error(make_web):
    def explode(kwargs):
        raise RuntimeError("boom")

    w = make_web([explode])
    w.chat()
    [(name, data, _)] = parse_events(w.http.get("/api/chat/1/events").text)
    assert name == "done" and "Something went wrong" in data["error"]


def test_the_stream_needs_no_csrf_token_because_it_changes_nothing(proposed):
    assert proposed.http.get("/api/chat/1/events").status_code == 200
    assert proposed.state()["proposal"] is not None            # reading the stream did not touch the proposal


# -- the event stream (live) --------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def live(ctx):
    """A real server on a real socket, with a model that can be held before its first tool call and again before it answers."""
    go, release = threading.Event(), threading.Event()

    def first(kwargs):
        assert go.wait(20), "the test never released the model"
        return message(tool("get_schedule", "t1"))

    def last(kwargs):
        assert release.wait(20), "the test never released the model"
        return submit(summary="Done.")

    web = Web(ctx, [first, message(tool("list_orders", "t2")), last])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server_ = uvicorn.Server(uvicorn.Config(web.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server_.run, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).read()
            break
        except OSError:
            time.sleep(0.05)
    yield web, port, go, release
    go.set()
    release.set()
    server_.should_exit = True
    thread.join(timeout=10)


def open_stream(port, turn_id, timeout=5):
    return urllib.request.urlopen(f"http://127.0.0.1:{port}/api/chat/{turn_id}/events", timeout=timeout)


def wait_until_the_stream_is_waiting(turn, timeout=3.0):
    """Block until a stream's thread is actually asleep on the turn's condition. Without this the test is a race: if the turn
    moves on before the stream goes to sleep, no wake-up is needed and a missing wake-up goes unnoticed."""
    deadline = time.time() + timeout
    while not turn.changed._waiters and time.time() < deadline:   # CPython's list of threads asleep on the condition
        time.sleep(0.01)
    assert turn.changed._waiters, "the stream never went to sleep waiting for news"


def lines_until(response, want):
    """Read lines from an OPEN stream until ``want(lines)``; a socket timeout (nothing arrived) propagates as a failure."""
    lines = []
    for raw in response:
        lines.append(raw.decode().rstrip("\n"))
        if want(lines):
            break
    return lines


def test_a_stream_that_is_already_open_is_woken_by_each_event_not_by_the_idle_timeout(live, monkeypatch):
    """One connection, opened BEFORE anything happens, as a browser does. The idle timeout is set far beyond the test, so
    the only thing that can deliver an event is the wake-up when the turn changes."""
    web, port, go, release = live
    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 60)
    assert web.post("/api/chat", {"message": "hello"}).status_code == 202       # the model is held before its first tool call
    turn = web.app.state.web.turn
    with open_stream(port, 1) as stream:
        wait_until_the_stream_is_waiting(turn)                                    # asleep, with nothing to report yet
        go.set()                                                                  # now tools run and progress events are published
        lines = lines_until(stream, lambda ls: sum(l.startswith("data: ") for l in ls) >= 2)
        assert [json.loads(l[6:])["tool"] for l in lines if l.startswith("data: ")] == ["get_schedule", "list_orders"]
        assert turn.status == "running"                                           # delivered mid-turn, not at the end
        wait_until_the_stream_is_waiting(turn)                                    # asleep again before the turn finishes
        release.set()                                                             # the turn finishes while the stream is still open
        lines = lines_until(stream, lambda ls: any(l.startswith("event: done") for l in ls))
        assert "event: done" in lines


def test_an_idle_stream_sends_keep_alive_comments(live, monkeypatch):
    web, port, go, release = live
    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.2)
    web.post("/api/chat", {"message": "hello"})
    with open_stream(port, 1) as stream:
        assert ": keep-alive" in lines_until(stream, lambda ls: ": keep-alive" in ls)
        assert web.app.state.web.turn.status == "running"


# -- the page ----------------------------------------------------------------------------------------------------------------------------------------


def test_the_page_follows_a_turn_with_the_stream_and_falls_back_to_polling():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "new EventSource(`/api/chat/${turnId}/events`)" in script and 'addEventListener("done"' in script
    assert "source.onerror" in script and "pollTurn(turnId)" in script and "async function pollTurn" in script
    assert "!(\"EventSource\" in window)" in script.replace("'", '"')


# -- container mode --------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("args", [["--host", "0.0.0.0"], ["--host", "192.168.1.20"], ["--container", "--host", "192.168.1.20"], ["--container", "--host", "::"]])
def test_listening_on_other_addresses_is_refused_unless_it_is_container_mode_on_all_interfaces(quiet, capsys, args):
    assert server.main(["--fixture", str(EVALS / "shop.json"), *args]) == 2
    assert "Refusing to listen" in capsys.readouterr().err


def test_container_mode_allows_all_interfaces_and_says_to_publish_to_loopback(quiet, monkeypatch, capsys):
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json"), "--container", "--host", "0.0.0.0"])
    assert seen["app"] is not None and seen["host"] == "0.0.0.0"
    refused = server.main(["--fixture", str(EVALS / "shop.json"), "--host", "0.0.0.0"])
    assert refused == 2 and "--container" in capsys.readouterr().err


def test_the_trusted_hosts_are_still_loopback_names_in_container_mode(quiet, monkeypatch):
    """Inside a container the app listens on 0.0.0.0 but still only answers requests addressed to a loopback host name."""
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json"), "--container", "--host", "0.0.0.0"])
    from fastapi.testclient import TestClient
    client = TestClient(seen["app"], base_url="http://127.0.0.1:8000")
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz", headers={"Host": "some-public-name.example"}).status_code == 400


# -- the Dockerfile (not built here: the properties that matter are checked as text) ----------------------------------------------------


@pytest.fixture(scope="module")
def dockerfile():
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_the_image_runs_as_a_normal_user_and_listens_in_container_mode(dockerfile):
    assert "USER app" in dockerfile and dockerfile.index("USER app") < dockerfile.index("CMD")
    assert '"--host", "0.0.0.0", "--container"' in dockerfile and "EXPOSE 8000" in dockerfile


def test_the_image_installs_exactly_what_the_lock_file_says_without_dev_tools(dockerfile):
    assert dockerfile.count("uv sync --locked") == 2 and "--no-dev" in dockerfile and "uv.lock" in dockerfile
    assert "COPY src" in dockerfile and "COPY evals" in dockerfile and "COPY knowledge" in dockerfile


def test_the_image_has_a_health_check_on_the_health_endpoint(dockerfile):
    assert "HEALTHCHECK" in dockerfile and "/healthz" in dockerfile


def test_no_secret_is_baked_into_the_image_and_the_run_command_publishes_to_loopback_only(dockerfile):
    assert ".env" not in [line.split()[1] for line in dockerfile.splitlines() if line.startswith("COPY")]
    assert "ANTHROPIC_API_KEY=sk" not in dockerfile and not any(l.startswith("ENV ANTHROPIC_API_KEY") for l in dockerfile.splitlines())
    assert "-p 127.0.0.1:8000:8000" in dockerfile


def test_the_build_context_leaves_out_secrets_history_and_local_state():
    ignored = {l.strip() for l in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
    assert {".env", ".git", ".venv", ".cache", "logs", "evals/results", "INTERVIEW_NOTES.md", "docs/SPEC.md"} <= ignored
