import re
import time

import pytest
from fastapi.testclient import TestClient

from jobshop.agent.loop import AgentConfig
from jobshop.api.app import create_app
from tests.fake_llm import FakeClient, message, submit, tool

HOSTS = ("testserver", "127.0.0.1", "localhost")


def proposal_script(summary="O-101 goes first; nothing else is affected.", priority=5):
    """A well-behaved agent: draft, change, solve once, compare, answer."""
    return [
        message(tool("create_draft", "a1")),
        message(tool("change_priority", "a2", draft_id="D1", order_id="O-101", priority=priority),
                tool("reschedule", "a3", draft_id="D1")),
        message(tool("compare_schedules", "a4", after="D1")),
        submit(summary=summary, draft_id="D1"),
    ]


class Web:
    """A TestClient plus the page's CSRF token, and helpers that behave like the page does."""

    def __init__(self, ctx, script, config=None, client_obj="fake", **kw):
        self.ctx = ctx
        self.fake = FakeClient(script) if client_obj == "fake" else client_obj
        self.app = create_app(ctx, self.fake, config or AgentConfig(model="fake-model"), allowed_hosts=HOSTS, **kw)
        self.http = TestClient(self.app, raise_server_exceptions=False)
        self.token = re.search(r'name="csrf-token" content="([^"]+)"', self.http.get("/").text).group(1)

    def post(self, path, json=None, token="default", headers=None):
        h = {"X-CSRF-Token": self.token if token == "default" else token, **(headers or {})}
        h = {k: v for k, v in h.items() if v is not None}
        return self.http.post(path, json=json, headers=h)

    def chat(self, text="Make O-101 urgent.", wait=True):
        r = self.post("/api/chat", {"message": text})
        if wait and r.status_code == 202:
            self.wait(r.json()["turn_id"])
        return r

    def wait(self, turn_id, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            turn = self.http.get(f"/api/chat/{turn_id}").json()
            if turn["status"] == "done":
                return turn
            time.sleep(0.02)
        raise AssertionError("the turn did not finish")

    def state(self):
        return self.http.get("/api/state").json()


@pytest.fixture
def make_web(ctx):
    def build(script=None, **kw):
        return Web(ctx, script if script is not None else proposal_script(), **kw)

    return build


@pytest.fixture
def web(make_web):
    return make_web()


@pytest.fixture
def proposed(make_web):
    """A web app after one finished turn whose answer proposes draft D1."""
    w = make_web()
    w.chat()
    return w
