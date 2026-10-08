"""The scripted demo must keep working as the app changes, or it quietly rots."""

import importlib.util
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo_server.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_server", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Page:
    def __init__(self, app):
        self.http = TestClient(app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False)
        self.token = re.search(r'name="csrf-token" content="([^"]+)"', self.http.get("/").text).group(1)

    def post(self, path, body):
        return self.http.post(path, json=body, headers={"X-CSRF-Token": self.token})

    def say(self, text="anything"):
        turn = self.post("/api/chat", {"message": text}).json()["turn_id"]
        for _ in range(500):
            if self.http.get(f"/api/chat/{turn}").json()["status"] == "done":
                return self.http.get("/api/state").json()
            time.sleep(0.02)
        raise AssertionError("the turn did not finish")


def test_the_whole_demo_plays_through_the_real_app(demo):
    page = Page(demo.build_app(solve_seconds=2))
    assert page.http.get("/api/state").json()["model"] == "scripted-demo"

    # 1. the M1 outage: a real proposal with moved operations and a late order
    s = page.say("what if M1 breaks?")
    p = s["proposal"]
    assert p["draft_id"] == "D1" and p["changes"] == ["M1 down 2026-01-05 10:00 to 2026-01-05 16:00"]
    assert p["comparison"]["diff"]["moved_operation_count"] > 0 and p["comparison"]["diff"]["newly_late_orders"]
    assert s["transcript"][-1]["text"].startswith(demo.TAG)

    # approving it really changes the live plan
    version = s["live_version"]
    assert page.post("/api/approve", {"draft_id": "D1", "digest": p["digest"]}).status_code == 200
    assert page.http.get("/api/state").json()["live_version"] == version + 1

    # 2. a second proposal
    s = page.say("make O-103 urgent")
    assert s["proposal"]["draft_id"] == "D2" and s["proposal"]["changes"][0].startswith("O-103 priority")

    # 3. a question instead of a proposal
    s = page.say("a machine is down")
    assert s["transcript"][-1]["kind"] == "clarify" and s["proposal"] is None

    # 4. and it never runs out
    for _ in range(3):
        assert page.say("hello")["transcript"][-1]["text"].endswith("Nothing further to change.")


def test_the_demo_is_clearly_labelled_and_never_reaches_the_network(demo):
    assert "scripted" in demo.TAG.lower() and "canned" in demo.TAG.lower()
    model = demo.ScriptedModel(demo.build_steps())
    assert model.create().content[0].name == "create_draft"   # no HTTP client involved: it just answers


def test_the_demo_only_listens_on_loopback(demo, monkeypatch):
    seen = {}
    monkeypatch.setattr(demo.uvicorn, "run", lambda app, **kw: seen.update(kw))
    assert demo.main(["--port", "9123"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 9123
