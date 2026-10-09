"""Open models: the Ollama adapter, tested against a stub that answers the way Ollama's documentation says it does.

None of this has met a real Ollama. These tests pin the translation, the failure handling and the dispatch, not the model."""

import json

import anthropic
import httpx2
import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.agent.providers import (
    DEFAULT_NUM_CTX, MultiProviderClient, OllamaClient, build_client, is_open_model, needs_anthropic, ollama_from_env,
)
from jobshop.agent.patterns import TRIAGE_SPEC, TRIAGE_TOOL
from tests.agent.test_loop import assert_history_is_valid

SPEC = {"name": "get_schedule", "description": "Read the plan.", "input_schema": {"type": "object", "properties": {}}}


class Stub:
    """A fake Ollama: records the requests and plays back canned answers (a dict, or an httpx2.Response, or an exception)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"url": str(request.url), "body": json.loads(request.content)})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, httpx2.Response) else httpx2.Response(200, json=answer)

    def client(self, **kw) -> OllamaClient:
        return OllamaClient("http://ollama.test:11434/", http=httpx2.Client(transport=httpx2.MockTransport(self)), **kw)


def chat(content="", tool_calls=None, prompt=120, out=30, done_reason="stop"):
    reply = {"role": "assistant", "content": content}
    if tool_calls is not None:
        reply["tool_calls"] = tool_calls
    return {"model": "llama3.1", "message": reply, "done": True, "done_reason": done_reason, "prompt_eval_count": prompt, "eval_count": out}


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def create(client, **kw):
    args = dict(model="ollama:llama3.1", max_tokens=500, system="SYSTEM RULES", tools=[SPEC], messages=[{"role": "user", "content": "hi"}])
    return client.create(**(args | kw))


# -- the request ---------------------------------------------------------------------------------------------------------------------


def test_the_request_goes_to_api_chat_with_the_model_name_without_the_prefix_and_a_large_context_window():
    stub = Stub(chat("hello"))
    create(stub.client())
    sent = stub.requests[0]
    assert sent["url"] == "http://ollama.test:11434/api/chat"
    body = sent["body"]
    assert body["model"] == "llama3.1" and body["stream"] is False
    assert body["options"] == {"num_predict": 500, "num_ctx": DEFAULT_NUM_CTX}
    assert body["messages"] == [{"role": "system", "content": "SYSTEM RULES"}, {"role": "user", "content": "hi"}]
    assert body["tools"] == [{"type": "function", "function": {"name": "get_schedule", "description": "Read the plan.", "parameters": SPEC["input_schema"]}}]


def test_a_forced_tool_is_approximated_by_offering_only_that_tool_and_saying_it_must_be_called():
    stub = Stub(chat(tool_calls=[call(TRIAGE_TOOL, route="read")]))
    create(stub.client(), tools=[SPEC, TRIAGE_SPEC], tool_choice={"type": "tool", "name": TRIAGE_TOOL})
    body = stub.requests[0]["body"]
    assert [t["function"]["name"] for t in body["tools"]] == [TRIAGE_TOOL]
    assert f"calling the tool '{TRIAGE_TOOL}'" in body["messages"][0]["content"] and body["messages"][0]["content"].startswith("SYSTEM RULES")


def test_no_tools_means_no_tools_key_and_cache_markers_and_block_systems_are_handled():
    stub = Stub(chat("ok"))
    create(stub.client(), tools=[], system=[{"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "B"}])
    body = stub.requests[0]["body"]
    assert "tools" not in body and body["messages"][0]["content"] == "A\n\nB"


def test_history_is_translated_tool_calls_results_and_errors_included():
    history = [
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": [{"type": "text", "text": "Looking."}, {"type": "tool_use", "id": "t1", "name": "get_schedule", "input": {"a": 1}},
                                           {"type": "tool_use", "id": "t2", "name": "get_order", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"v":1}'},
                                      {"type": "tool_result", "tool_use_id": "t2", "content": "bad id", "is_error": True},
                                      {"type": "text", "text": "Now finish.", "cache_control": {"type": "ephemeral"}}]},
    ]
    stub = Stub(chat("done"))
    create(stub.client(), messages=history)
    sent = stub.requests[0]["body"]["messages"][1:]
    assert sent == [
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": "Looking.", "tool_calls": [call("get_schedule", a=1), call("get_order")]},
        {"role": "tool", "tool_name": "get_schedule", "content": '{"v":1}'},
        {"role": "tool", "tool_name": "get_order", "content": "ERROR: bad id"},
        {"role": "user", "content": "Now finish."},
    ]


# -- the reply -----------------------------------------------------------------------------------------------------------------------------


def test_a_tool_call_becomes_a_tool_use_block_with_usage_and_a_tool_use_stop():
    reply = create(Stub(chat("Checking.", tool_calls=[call("get_schedule"), call("get_order", order_id="O-1")], prompt=300, out=40)).client())
    assert [b.type for b in reply.content] == ["text", "tool_use", "tool_use"]
    assert reply.content[1].name == "get_schedule" and reply.content[1].input == {} and reply.content[2].input == {"order_id": "O-1"}
    assert len({b.id for b in reply.content[1:]}) == 2
    assert reply.stop_reason == "tool_use" and (reply.usage.input_tokens, reply.usage.output_tokens) == (300, 40)
    assert getattr(reply.usage, "cache_read_input_tokens", None) in (None, 0)


def test_prose_ends_the_turn_and_a_cut_off_reply_says_max_tokens():
    assert create(Stub(chat("All fine.")).client()).stop_reason == "end_turn"
    assert create(Stub(chat("All fi", done_reason="length")).client()).stop_reason == "max_tokens"
    assert create(Stub({"message": {"role": "assistant", "content": "x"}}).client()).usage.input_tokens == 0     # counts may be absent


def test_arguments_sent_as_a_json_string_are_parsed_and_garbage_is_passed_on_by_name_not_dropped():
    reply = create(Stub(chat(tool_calls=[
        {"function": {"name": "get_order", "arguments": '{"order_id": "O-1"}'}},
        {"function": {"name": "get_order", "arguments": "not json"}},
        {"function": {"name": "get_order", "arguments": "[1, 2]"}},
        {"function": {"name": "get_order"}},
    ])).client())
    assert [b.input for b in reply.content] == [{"order_id": "O-1"}, {"unparsed_arguments": "not json"}, {"unparsed_arguments": "[1, 2]"}, {}]


# -- failures: all of them look like the API errors the loop already handles ------------------------------------------------------------


def test_an_unreachable_server_is_an_api_connection_error():
    client = Stub(httpx2.ConnectError("refused")).client()
    with pytest.raises(anthropic.APIConnectionError, match="cannot reach Ollama at http://ollama.test:11434"):
        create(client)


def test_an_error_status_is_an_api_status_error_with_the_servers_words():
    with pytest.raises(anthropic.APIStatusError, match="404.*model 'llama3.1' not found"):
        create(Stub(httpx2.Response(404, json={"error": "model 'llama3.1' not found"})).client())


@pytest.mark.parametrize("body", [httpx2.Response(200, text="<html>nope</html>"), httpx2.Response(200, json={"unexpected": 1}), httpx2.Response(200, json=[1])])
def test_a_reply_that_is_not_a_chat_message_is_an_api_error_not_a_crash(body):
    with pytest.raises(anthropic.APIConnectionError, match="not a chat message"):
        create(Stub(body).client())


def test_a_prompt_that_filled_the_window_is_refused_because_the_rules_were_probably_cut_off():
    with pytest.raises(anthropic.APIError, match="filled Ollama's whole 4096-token window"):
        create(Stub(chat("answer", prompt=4096)).client(num_ctx=4096))
    create(Stub(chat("answer", prompt=4000)).client(num_ctx=4096))          # close to full is still fine


def test_a_window_too_small_for_the_prompt_is_a_configuration_error():
    with pytest.raises(ValueError, match="num_ctx"):
        OllamaClient(num_ctx=512)


# -- dispatch ------------------------------------------------------------------------------------------------------------------------------


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return "from anthropic"


def test_a_model_name_decides_the_provider_and_the_anthropic_client_is_only_made_when_needed(monkeypatch):
    made = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda: made.append(1) or FakeAnthropic())
    stub = Stub(chat("local"))
    multi = MultiProviderClient(stub.client())
    assert multi.create(model="ollama:llama3.1", max_tokens=5, system="s", tools=[], messages=[{"role": "user", "content": "x"}]).content[0].text == "local"
    assert made == []                                                                       # no key needed for local-only use
    assert multi.create(model="claude-x", max_tokens=5, system="s", tools=[], messages=[]) == "from anthropic" and made == [1]
    multi.create(model="claude-x", max_tokens=5, system="s", tools=[], messages=[])
    assert made == [1]


def test_helpers_say_which_models_are_open_and_whether_a_key_is_needed():
    assert is_open_model("ollama:qwen2.5") and not is_open_model("claude-x") and not is_open_model(None) and not is_open_model("")
    assert needs_anthropic("ollama:a", "claude-x") and not needs_anthropic("ollama:a", None, "") and not needs_anthropic()


def test_build_client_uses_plain_anthropic_unless_a_model_is_open(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", lambda: "plain")
    assert build_client({}, "claude-x", None) == "plain"
    assert isinstance(build_client({}, "claude-x", "ollama:llama3.1"), MultiProviderClient)


def test_ollama_settings_come_from_the_environment():
    c = ollama_from_env({"JOBSHOP_OLLAMA_URL": "http://gpu-box:11434/", "JOBSHOP_OLLAMA_NUM_CTX": "32768", "JOBSHOP_OLLAMA_TIMEOUT_S": "60"})
    assert c.base_url == "http://gpu-box:11434" and c.num_ctx == 32768
    assert ollama_from_env({}).base_url == "http://127.0.0.1:11434"
    with pytest.raises(ValueError, match="JOBSHOP_OLLAMA_NUM_CTX must be a number"):
        ollama_from_env({"JOBSHOP_OLLAMA_NUM_CTX": "lots"})


# -- the whole loop on a stub open model -------------------------------------------------------------------------------------------------------


def run_on_stub(ctx, registry, stub, config, user="How is the plan?", **kw):
    from jobshop.agent.loop import run_turn
    from jobshop.agent.prompts import build_system_prompt

    messages: list = []
    return run_turn(stub.client(), registry, build_system_prompt(ctx), messages, user, config, **kw), messages


def test_the_agent_loop_runs_a_whole_turn_on_an_open_model_and_costs_it_at_zero(ctx, registry):
    stub = Stub(chat(tool_calls=[call("get_schedule")], prompt=500, out=20),
                chat(tool_calls=[call("submit_response", summary="All twelve orders are on time.")], prompt=700, out=30))
    config = AgentConfig(model="ollama:llama3.1", price_input_per_mtok=0.0, price_output_per_mtok=0.0)
    result, messages = run_on_stub(ctx, registry, stub, config)
    assert result.status == "answered" and result.final.summary == "All twelve orders are on time." and result.steps == 2
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (1200, 50, 0.0)
    assert all(r["body"]["model"] == "llama3.1" for r in stub.requests)
    second = stub.requests[1]["body"]["messages"]
    assert [m["role"] for m in second[-3:]] == ["user", "assistant", "tool"] and second[-1]["tool_name"] == "get_schedule"
    assert_history_is_valid(messages)


def test_an_open_model_that_only_talks_is_nudged_and_then_stopped_by_the_same_rules_as_any_model(ctx, registry):
    stub = Stub(*[chat("I think everything is fine.") for _ in range(8)])
    result, _ = run_on_stub(ctx, registry, stub, AgentConfig(model="ollama:llama3.1", max_steps=4))
    assert result.status == "no_final_response" and result.final is None and result.steps <= 4


def test_a_dead_server_ends_the_turn_cleanly_and_leaves_the_history_alone(ctx, registry):
    result, messages = run_on_stub(ctx, registry, Stub(httpx2.ConnectError("refused")), AgentConfig(model="ollama:llama3.1"))
    assert result.status == "api_error" and "cannot reach Ollama" in result.text and messages == []


def test_the_route_pattern_works_on_an_open_model_including_the_forced_triage_call(ctx, registry):
    stub = Stub(chat(tool_calls=[call(TRIAGE_TOOL, route="decline_commit", reason="asks to commit")]))
    result, _ = run_on_stub(ctx, registry, stub, AgentConfig(model="ollama:llama3.1", pattern="route"), user="Just commit it.")
    assert result.route == "decline_commit" and len(stub.requests) == 1
    assert [t["function"]["name"] for t in stub.requests[0]["body"]["tools"]] == [TRIAGE_TOOL]


def test_a_triage_model_that_answers_in_prose_falls_back_to_the_full_agent(ctx, registry):
    stub = Stub(chat("This looks like a question."), chat(tool_calls=[call("submit_response", summary="Fine.")]))
    result, _ = run_on_stub(ctx, registry, stub, AgentConfig(model="ollama:llama3.1", pattern="route"))
    assert result.route == "plan" and result.final.summary == "Fine."


# -- the chat command line --------------------------------------------------------------------------------------------------------------------


def test_the_chat_cli_asks_for_a_key_only_when_some_model_is_an_anthropic_model(monkeypatch, capsys):
    from jobshop.agent import cli

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "load_knowledge", lambda env: (_ for _ in ()).throw(ValueError("stop here")))      # a stop just after the key check
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JOBSHOP_TRIAGE_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "ollama:llama3.1")
    assert cli.main(["--solve-seconds", "1"]) == 2 and "stop here" in capsys.readouterr().err                   # past the key check
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    assert cli.main(["--solve-seconds", "1"]) == 2 and "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_MODEL", "ollama:llama3.1")
    monkeypatch.setenv("JOBSHOP_PATTERN", "route")
    monkeypatch.setenv("JOBSHOP_TRIAGE_MODEL", "claude-small")
    assert cli.main(["--solve-seconds", "1"]) == 2 and "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
