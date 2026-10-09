import http.client
import json
import socket
import threading
import time
import urllib.request

import anthropic
import pytest
import uvicorn

from jobshop.agent.loop import AgentConfig
from jobshop.api import server
from jobshop.api.app import create_app
from tests.evals.conftest import EVALS


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(server, "load_dotenv", lambda: None)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "JOBSHOP_SOLVE_SECONDS"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "example.com", "::"])
def test_it_refuses_to_listen_anywhere_but_loopback(quiet, capsys, monkeypatch, host):
    def must_not_serve(*args, **kwargs):  # a regression here would otherwise hang the suite AND open a real port
        raise AssertionError(f"tried to start a server for host {host}")

    monkeypatch.setattr(server.uvicorn, "run", must_not_serve)
    assert server.main(["--host", host]) == 2
    assert "no login" in capsys.readouterr().err


def run_main_capturing_the_app(monkeypatch, argv):
    seen = {}
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    assert server.main(argv) == 0
    return seen


def test_without_a_key_the_server_starts_with_chat_disabled(quiet, monkeypatch, capsys):
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json")])
    assert seen["host"] == "127.0.0.1" and seen["app"].state.web.conversation is None
    assert "No model configured" in capsys.readouterr().err


def test_with_a_key_and_model_chat_is_enabled(quiet, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-model")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: object())
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json")])
    web = seen["app"].state.web
    assert web.conversation is not None and web.config.model == "some-model"


def test_the_fixture_shop_starts_instantly_at_the_plant_clock(quiet, monkeypatch):
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json"), "--now", "2026-01-05 12:30"])
    committed = seen["app"].state.web.ctx.store.committed
    assert len(committed.instance.orders) == 12 and committed.instance.to_datetime(committed.instance.now).strftime("%H:%M") == "12:30"


def test_a_missing_fixture_falls_back_to_generating_a_shop(quiet, monkeypatch, tmp_path):
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(tmp_path / "none.json"), "--orders", "4", "--machines", "3", "--solve-seconds", "1"])
    assert len(seen["app"].state.web.ctx.store.committed.instance.orders) == 4


@pytest.fixture
def live_server(ctx):
    port = free_port()
    app = create_app(ctx, None, AgentConfig(model="m"))
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started, "the server did not start"
    yield port
    srv.should_exit = True
    thread.join(timeout=10)


def test_the_real_server_serves_the_page_the_state_and_refuses_a_forged_host(live_server):
    base = f"http://127.0.0.1:{live_server}"
    with urllib.request.urlopen(base + "/") as r:
        assert r.status == 200 and "Shop Floor Assistant" in r.read().decode() and "default-src 'none'" in r.headers["content-security-policy"]
    with urllib.request.urlopen(base + "/api/state") as r:
        assert json.load(r)["model_configured"] is False
    with urllib.request.urlopen(base + "/static/app.js") as r:
        assert r.status == 200

    forged = http.client.HTTPConnection("127.0.0.1", live_server)
    forged.request("GET", "/api/state", headers={"Host": "attacker.example"})
    assert forged.getresponse().status == 400                       # DNS rebinding is refused over a real socket

    post = http.client.HTTPConnection("127.0.0.1", live_server)
    post.request("POST", "/api/approve", body=json.dumps({"draft_id": "D1", "digest": "0" * 64}), headers={"Content-Type": "application/json"})
    assert post.getresponse().status == 403                         # and so is a POST without the page's token


def test_the_plant_documents_are_found_in_the_default_folder_and_can_be_switched_off(quiet, monkeypatch):
    monkeypatch.chdir(EVALS.parent)                                   # the repo root, where ./knowledge lives
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json")])
    assert len(seen["app"].state.web.ctx.knowledge.sources) == 12

    monkeypatch.setenv("JOBSHOP_KNOWLEDGE_DIR", "off")
    seen = run_main_capturing_the_app(monkeypatch, ["--fixture", str(EVALS / "shop.json")])
    assert seen["app"].state.web.ctx.knowledge is None


def test_a_mistyped_documents_folder_is_a_clear_configuration_error_not_a_traceback(quiet, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("JOBSHOP_KNOWLEDGE_DIR", str(tmp_path / "typo"))
    assert server.main(["--fixture", str(EVALS / "shop.json")]) == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err and "JOBSHOP_KNOWLEDGE_DIR" in err and "is not a folder" in err
