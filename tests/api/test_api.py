import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jobshop.api import views as web_views
from jobshop.api.app import STATIC
from jobshop.core.kpis import compute_kpis
from jobshop.tools.approval import proposal_digest
from jobshop.tools.registry import ToolRegistry
from tests.api.conftest import HOSTS, Web, proposal_script
from tests.fake_llm import message, submit, tool


# -- pages and headers -------------------------------------------------------------------------------------


def test_the_page_embeds_a_csrf_token_and_loads_only_its_own_files(web):
    html = web.http.get("/").text
    assert web.token and "{{CSRF_TOKEN}}" not in html and len(web.token) >= 32
    assert 'src="/static/app.js"' in html and 'href="/static/style.css"' in html
    assert "http://" not in html and "https://" not in html and "<script>" not in html   # nothing inline, nothing remote


def test_every_response_carries_strict_security_headers(web):
    for path in ("/", "/api/state", "/static/app.js"):
        h = web.http.get(path).headers
        csp = h["content-security-policy"]
        assert "default-src 'none'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
        assert h["x-content-type-options"] == "nosniff" and h["cache-control"] == "no-store" and h["referrer-policy"] == "no-referrer"


def test_the_client_script_never_writes_untrusted_text_as_html():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for dangerous in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "dangerouslySet"):
        assert dangerous not in script, dangerous


def test_the_static_files_are_served_and_the_api_docs_are_not(web):
    assert web.http.get("/static/style.css").status_code == 200
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert web.http.get(path).status_code == 404


def test_only_loopback_hosts_are_accepted_by_default(ctx):
    from jobshop.agent.loop import AgentConfig
    from jobshop.api.app import create_app

    client = TestClient(create_app(ctx, None, AgentConfig(model="m")))
    assert client.get("/api/state", headers={"host": "127.0.0.1:8000"}).status_code == 200
    assert client.get("/api/state", headers={"host": "localhost:8000"}).status_code == 200
    assert client.get("/api/state", headers={"host": "evil.example"}).status_code == 400   # DNS rebinding


# -- reading ---------------------------------------------------------------------------------------------------


def test_state_reports_the_live_plan_from_the_store(web, ctx):
    s = web.state()
    c = ctx.store.committed
    k = compute_kpis(c.instance, c.schedule)
    assert s["busy"] is False and s["model_configured"] is True and s["model"] == "fake-model"
    assert s["live_version"] == c.version and s["plant_time"] == "2026-01-05 06:00"
    assert s["kpi_live"]["late_orders"] == k.late_orders and s["kpi_live"]["total_tardiness_min"] == k.total_tardiness
    assert s["transcript"] == [] and s["drafts"] == [] and s["proposal"] is None


def test_the_gantt_endpoint_validates_its_source(web):
    assert web.http.get("/api/gantt").json()["source"] == "committed"
    assert web.http.get("/api/gantt?source=bad").status_code == 422
    assert web.http.get("/api/gantt?source=../x").status_code == 422
    assert web.http.get("/api/gantt?source=D9").status_code == 404


def test_an_unsolved_draft_cannot_be_charted(web, registry):
    d = registry.call("create_draft", {})["draft_id"]
    r = web.http.get(f"/api/gantt?source={d}")
    assert r.status_code == 409 and "no solved schedule" in r.json()["detail"]


# -- chat ----------------------------------------------------------------------------------------------------------


def test_a_chat_turn_runs_in_the_background_and_ends_with_a_proposal(web, ctx):
    r = web.chat()
    assert r.status_code == 202
    s = web.state()
    assert [m["role"] for m in s["transcript"]] == ["user", "assistant"]
    answer = s["transcript"][1]
    assert answer["kind"] == "answer" and answer["draft_id"] == "D1" and answer["usage"]["steps"] == 4
    p = s["proposal"]
    assert p["draft_id"] == "D1" and p["changes"] == ["O-101 priority 3 -> 5"] or p["changes"][0].startswith("O-101 priority")
    assert p["digest"] == proposal_digest(ctx.store.draft("D1").instance, ctx.store.draft("D1").schedule)
    assert s["drafts"][0]["solved"] is True
    assert ctx.store.committed.version == 1                 # nothing is live yet


def test_progress_lists_the_tools_as_they_run(web):
    turn = web.wait(web.chat(wait=False).json()["turn_id"])
    assert turn["progress"] == ["create_draft", "change_priority", "reschedule", "compare_schedules"] and turn["error"] is None


def test_a_clarifying_question_is_shown_and_offers_nothing_to_approve(make_web):
    w = make_web([submit(summary="I need one detail.", clarifying_question="Which machine, and from when?")])
    w.chat("A machine is going down.")
    s = w.state()
    assert s["transcript"][1]["kind"] == "clarify" and s["transcript"][1]["text"] == "Which machine, and from when?"
    assert s["proposal"] is None


def test_without_a_model_the_plan_is_visible_but_chat_is_refused(ctx):
    w = Web(ctx, [], client_obj=None)
    assert w.state()["model_configured"] is False and w.state()["kpi_live"]
    r = w.post("/api/chat", {"message": "hello"})
    assert r.status_code == 503 and "No model is configured" in r.json()["detail"]
    assert w.http.get("/api/gantt").status_code == 200


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "x" * 2001}, {"message": "hi", "draft_id": "D1"}, {"message": 5}])
def test_chat_input_is_validated(web, body):
    assert web.post("/api/chat", body).status_code == 422


def test_the_page_stays_responsive_while_the_agent_works_and_a_second_message_is_refused(ctx):
    started, release = threading.Event(), threading.Event()

    def hold(kwargs):
        started.set()
        assert release.wait(10)
        return submit(summary="Done waiting.")

    w = Web(ctx, [hold])
    turn_id = w.chat(wait=False).json()["turn_id"]
    assert started.wait(5)
    try:
        busy = w.state()                                              # does not hang on the store lock
        assert busy["busy"] is True and busy["transcript"][0]["role"] == "user" and "kpi_live" not in busy
        assert w.post("/api/chat", {"message": "again"}).status_code == 409
        assert w.post("/api/approve", {"draft_id": "D1", "digest": "0" * 64}).status_code == 409
        assert w.post("/api/reject", {"draft_id": "D1"}).status_code == 409
        assert w.http.get("/api/gantt").status_code == 409
        assert w.http.get(f"/api/chat/{turn_id}").json()["status"] == "running"
    finally:
        release.set()
    assert w.wait(turn_id)["status"] == "done" and w.state()["busy"] is False


def test_a_crash_inside_a_turn_ends_it_cleanly_and_releases_the_store(make_web):
    def explode(kwargs):
        raise RuntimeError("boom")

    w = make_web([explode, submit(summary="Recovered.")])
    turn = w.wait(w.chat(wait=False).json()["turn_id"])
    assert turn["status"] == "done" and "Something went wrong" in turn["error"]
    assert "boom" not in str(w.state()["transcript"])                 # no internals shown to the planner
    w.chat("try again")                                               # the lock was released
    assert w.state()["transcript"][-1]["text"] == "Recovered."


def test_an_unknown_turn_is_a_404(web):
    assert web.http.get("/api/chat/99").status_code == 404


def test_html_in_a_model_answer_is_returned_as_plain_text_data(make_web):
    evil = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    w = make_web([submit(summary=evil)])
    w.chat()
    assert w.state()["transcript"][1]["text"] == evil               # stored verbatim; the page only ever uses textContent


# -- the human decision ----------------------------------------------------------------------------------------


def approve_body(w):
    p = w.state()["proposal"]
    return {"draft_id": p["draft_id"], "digest": p["digest"]}


def test_approving_the_proposal_makes_it_live_and_tells_the_assistant(proposed, ctx):
    proposed.fake.script += [submit(summary="Understood.")]
    r = proposed.post("/api/approve", approve_body(proposed))
    assert r.status_code == 200 and r.json() == {"new_version": 2}
    assert ctx.store.committed.version == 2 and ctx.store.committed.instance.order("O-101").priority == 5
    s = proposed.state()
    assert s["proposal"] is None and s["live_version"] == 2
    assert s["transcript"][-1] == {"role": "system", "text": "You approved draft D1. The live plan is now version 2."}

    proposed.chat("thanks")                                          # the next turn is told what the planner did
    assert "approved and committed draft D1" in str(proposed.fake.requests[-1]["messages"][-1]["content"])


def test_the_approval_token_never_reaches_the_response_or_the_model(proposed, ctx):
    minted = []
    real = ctx.authority.issue
    ctx.authority.issue = lambda **kw: minted.append(real(**kw)) or minted[-1]
    proposed.fake.script += [submit(summary="ok")]
    r = proposed.post("/api/approve", approve_body(proposed))
    proposed.chat("thanks")
    assert len(minted) == 1 and minted[0] not in r.text and minted[0] not in str(proposed.fake.requests)


def test_approval_is_refused_if_the_draft_is_not_the_one_that_was_shown(proposed, ctx):
    body = approve_body(proposed)
    ToolRegistry(ctx).call("change_priority", {"draft_id": "D1", "order_id": "O-102", "priority": 5})   # edited after the planner looked
    ToolRegistry(ctx).call("reschedule", {"draft_id": "D1"})
    r = proposed.post("/api/approve", body)
    assert r.status_code == 409 and "not the proposal you reviewed" in r.json()["detail"]
    assert ctx.store.committed.version == 1


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "0" * 63, "A" * 64])
def test_a_malformed_digest_is_rejected_before_anything_happens(proposed, ctx, digest):
    assert proposed.post("/api/approve", {"draft_id": "D1", "digest": digest}).status_code == 422
    assert ctx.store.committed.version == 1


def test_a_stale_proposal_cannot_be_approved(proposed, ctx):
    body = approve_body(proposed)
    ctx.store.set_clock(ctx.store.committed.instance.now + 30)       # the plan moved on
    r = proposed.post("/api/approve", body)
    assert r.status_code == 409 and "stale" in r.json()["detail"]
    assert proposed.state()["proposal"] is None                      # and the page stops offering it


def test_approving_twice_commits_once(proposed, ctx):
    body = approve_body(proposed)
    assert proposed.post("/api/approve", body).status_code == 200
    assert proposed.post("/api/approve", body).status_code in (404, 409)
    assert ctx.store.committed.version == 2


def test_approving_an_unknown_draft_is_a_404(web, ctx):
    assert web.post("/api/approve", {"draft_id": "D7", "digest": "0" * 64}).status_code == 404
    assert ctx.store.committed.version == 1


def test_rejecting_discards_the_draft_and_tells_the_assistant(proposed, ctx):
    proposed.fake.script += [submit(summary="Noted.")]
    assert proposed.post("/api/reject", {"draft_id": "D1"}).status_code == 200
    assert ctx.store.drafts() == [] and ctx.store.committed.version == 1
    s = proposed.state()
    assert s["proposal"] is None and s["transcript"][-1]["text"] == "You rejected draft D1. The live plan is unchanged."
    proposed.chat("ok")
    assert "REJECTED draft D1" in str(proposed.fake.requests[-1]["messages"][-1]["content"])
    assert proposed.post("/api/reject", {"draft_id": "D1"}).status_code == 404


# -- who may press the button ----------------------------------------------------------------------------------


@pytest.mark.parametrize("path, body", [
    ("/api/approve", {"draft_id": "D1", "digest": "0" * 64}),
    ("/api/reject", {"draft_id": "D1"}),
    ("/api/chat", {"message": "hi"}),
])
def test_state_changing_requests_need_the_csrf_token(proposed, ctx, path, body):
    assert proposed.post(path, body, token=None).status_code == 403
    assert proposed.post(path, body, token="wrong").status_code == 403
    assert proposed.post(path, body, token=proposed.token[:-1]).status_code == 403
    assert ctx.store.committed.version == 1 and [d.id for d in ctx.store.drafts()] == ["D1"]


def test_a_request_from_another_site_is_refused_even_with_a_valid_token(proposed, ctx):
    body = approve_body(proposed)
    for origin in ("https://evil.example", "http://localhost.evil.example", "null"):
        assert proposed.post("/api/approve", body, headers={"Origin": origin}).status_code == 403
    assert ctx.store.committed.version == 1
    assert proposed.post("/api/approve", body, headers={"Origin": "http://localhost:8000"}).status_code == 200


def test_the_csrf_token_is_different_for_every_server_run(ctx):
    assert Web(ctx, []).token != Web(ctx, []).token


def test_an_instruction_in_a_note_cannot_reach_the_approve_button(ctx):
    """The model's tools are the only way it acts, and none of them is the web endpoint or a commit."""
    names = ToolRegistry(ctx).names()
    assert not {"commit_schedule", "approve", "approve_draft", "request_commit"} & set(names)
    script = [message(tool("commit_schedule", "x1", draft_id="D1", approval_token="APPROVED")), submit(summary="I committed it.")]
    w = Web(ctx, script)
    w.chat("Commit whatever you have.")
    assert ctx.store.committed.version == 1 and w.state()["proposal"] is None


# -- failure handling ---------------------------------------------------------------------------------------------


def test_an_unexpected_error_is_a_plain_500_with_no_traceback(web, monkeypatch):
    monkeypatch.setattr(web_views, "gantt", lambda *a: (_ for _ in ()).throw(RuntimeError("secret internal detail")))
    r = web.http.get("/api/gantt")
    assert r.status_code == 500 and r.json() == {"detail": "internal error"} and "secret" not in r.text


def test_the_lock_is_released_after_a_failed_request(web, monkeypatch):
    monkeypatch.setattr(web_views, "gantt", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    assert web.http.get("/api/gantt").status_code == 500
    monkeypatch.undo()
    assert web.http.get("/api/gantt").status_code == 200
