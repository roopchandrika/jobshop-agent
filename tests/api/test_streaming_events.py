"""The streamed answer reaches the page: an `answer` event on the turn's event stream, shown in its own element."""

from jobshop.api.app import Turn
from tests.api.conftest import Web, proposal_script
from tests.fake_llm import StreamingFakeClient


def make(ctx):
    script = proposal_script(summary="O-101 goes first; nothing else is affected.")
    return Web(ctx, script, client_obj=StreamingFakeClient(script))


def events(text):
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            out.append((fields["event"], fields.get("data")))
    return out


def test_the_event_stream_carries_the_answer_as_it_was_written(ctx):
    import json

    web = make(ctx)
    turn_id = web.chat().json()["turn_id"]
    body = web.http.get(f"/api/chat/{turn_id}/events").text
    seen = events(body)
    kinds = [k for k, _ in seen]
    assert "answer" in kinds and kinds[-1] == "done" and kinds.index("answer") < kinds.index("done")
    last = [json.loads(d) for k, d in seen if k == "answer"][-1]
    assert last == {"text": "O-101 goes first; nothing else is affected."}


def test_streaming_does_not_change_the_final_answer_or_the_proposal(ctx):
    web = make(ctx)
    web.chat()
    state = web.state()
    assert state["transcript"][-1]["text"] == "O-101 goes first; nothing else is affected."
    assert state["proposal"] is not None and ctx.store.committed.version == 1


def test_a_client_that_cannot_stream_still_works_and_sends_no_answer_event(ctx, make_web):
    web = make_web()
    turn_id = web.chat().json()["turn_id"]
    kinds = [k for k, _ in events(web.http.get(f"/api/chat/{turn_id}/events").text)]
    assert "answer" not in kinds and kinds[-1] == "done"


def test_the_page_has_an_element_for_the_live_answer_and_the_script_fills_it_with_text_only(ctx):
    web = make(ctx)
    page = web.http.get("/").text
    assert 'id="live-answer"' in page
    script = web.http.get("/static/app.js").text
    assert 'addEventListener("answer"' in script and "showLiveAnswer" in script
    assert "innerHTML" not in script


def test_on_delta_accumulates_resets_and_ignores_plain_text():
    turn = Turn(1)
    turn.on_delta("reset", "")
    turn.on_delta("answer", "Hel")
    turn.on_delta("text", "thinking aloud")          # not part of the answer
    turn.on_delta("answer", "lo")
    assert turn.answer == "Hello" and turn.answer_version == 3
    turn.on_delta("reset", "")
    assert turn.answer == "" and turn.answer_version == 4
