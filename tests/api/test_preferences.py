"""Standing preferences in the web app: the planner adds and forgets them; the model only reads them."""

import re

import pytest
from fastapi.testclient import TestClient

from jobshop.agent.loop import AgentConfig
from jobshop.agent.memory import MAX_PREFERENCES, PreferenceStore
from jobshop.api.app import STATIC, create_app
from tests.api.conftest import Web, proposal_script
from tests.fake_llm import submit


def test_the_state_lists_the_preferences_even_while_the_assistant_is_busy(web):
    web.post("/api/preferences", {"text": "I always want the earliest finish."})
    assert web.state()["preferences"] == [{"id": 1, "text": "I always want the earliest finish."}]
    web.app.state.web.lock.acquire()                 # the agent owns the store mid-turn...
    try:
        busy = web.http.get("/api/state").json()
    finally:
        web.app.state.web.lock.release()
    assert busy["busy"] is True and busy["preferences"] == [{"id": 1, "text": "I always want the earliest finish."}]


def test_adding_and_forgetting_returns_the_current_list(web):
    r = web.post("/api/preferences", {"text": "Never suggest overtime."})
    assert r.status_code == 200 and r.json() == {"preferences": [{"id": 1, "text": "Never suggest overtime."}]}
    web.post("/api/preferences", {"text": "Mention late orders first."})
    r = web.post("/api/preferences/forget", {"id": 1})
    assert r.json() == {"preferences": [{"id": 2, "text": "Mention late orders first."}]}


@pytest.mark.parametrize("path, body", [
    ("/api/preferences", {"text": "x"}), ("/api/preferences/forget", {"id": 1}),
])
def test_changing_preferences_needs_the_pages_csrf_token(web, path, body):
    web.post("/api/preferences", {"text": "keep me"})
    for token in (None, "wrong-token"):
        assert web.post(path, body, token=token).status_code == 403
    assert len(web.state()["preferences"]) == 1


def test_a_request_from_another_origin_is_refused(web):
    r = web.post("/api/preferences", {"text": "hostile"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and web.state()["preferences"] == []


@pytest.mark.parametrize("body, shown", [
    ({"text": ""}, None), ({"text": "   "}, "cannot be empty"), ({"text": "x" * 300}, "too long"),
    ({"text": "x" * 1001}, None), ({}, None), ({"text": "ok", "extra": 1}, None), ({"text": 5}, None),
])
def test_bad_preferences_are_refused_with_a_reason_and_store_nothing(web, body, shown):
    r = web.post("/api/preferences", body)
    assert r.status_code == 422 and web.state()["preferences"] == []
    if shown:
        assert shown in r.json()["detail"]


def test_forgetting_what_does_not_exist_is_a_404_and_bad_ids_are_rejected(web):
    assert web.post("/api/preferences/forget", {"id": 99}).status_code == 404
    for bad in (0, -1, "1", 1.5, None, True):
        assert web.post("/api/preferences/forget", {"id": bad}).status_code == 422


def test_the_cap_and_duplicates_are_enforced_through_the_page_too(web):
    for i in range(MAX_PREFERENCES):
        assert web.post("/api/preferences", {"text": f"preference {i}"}).status_code == 200
    assert "forget one first" in web.post("/api/preferences", {"text": "one more"}).json()["detail"]
    web.post("/api/preferences/forget", {"id": 1})
    assert "already remembered" in web.post("/api/preferences", {"text": "PREFERENCE 2"}).json()["detail"]


def test_a_preference_added_in_the_page_reaches_the_models_next_prompt(make_web):
    w = make_web([submit(summary="First."), submit(summary="Second.")])
    w.chat("hello")
    w.post("/api/preferences", {"text": "I always want the earliest finish."})
    w.chat("hello again")
    first, second = (r["system"] for r in w.fake.requests)
    assert "I always want the earliest finish." not in first and "1. I always want the earliest finish." in second


def test_preferences_work_even_with_no_model_configured(ctx):
    app = create_app(ctx, None, AgentConfig(model="none"), allowed_hosts=("testserver",))
    http = TestClient(app)
    token = re.search(r'name="csrf-token" content="([^"]+)"', http.get("/").text).group(1)
    r = http.post("/api/preferences", json={"text": "kept without a model"}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200 and http.get("/api/state").json()["preferences"][0]["text"] == "kept without a model"


def test_a_store_given_to_the_app_is_the_one_it_uses(ctx):
    store = PreferenceStore()
    store.add("already there")
    w = Web(ctx, proposal_script(), preferences=store)
    assert w.state()["preferences"] == [{"id": 1, "text": "already there"}]
    w.post("/api/preferences", {"text": "added in the page"})
    assert store.texts() == ["already there", "added in the page"]


def test_the_page_has_the_panel_and_inserts_preference_text_only_as_text():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for element in ('id="prefs"', 'id="prefs-form"', 'id="prefs-list"', 'id="prefs-error"', 'maxlength="200"'):
        assert element in html
    assert "renderPreferences" in script and "/api/preferences/forget" in script
    section = script[script.index("function renderPreferences"):script.index("async function changePreferences")]
    assert "innerHTML" not in section and "insertAdjacentHTML" not in section
