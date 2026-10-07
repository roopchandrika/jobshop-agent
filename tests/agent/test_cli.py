import json
from datetime import datetime

import pytest

from jobshop.agent.cli import ChatSession, ConfigError, agent_config_from_env, build_context, main
from jobshop.agent.loop import AgentConfig
from jobshop.core.generator import GeneratorSettings
from tests.fake_llm import FakeClient, message, submit, tool
from tests.helpers import FAST, SMALL

CONFIG = AgentConfig(model="fake-model")


def proposal(summary="O-101 goes first.", priority=5):
    return [
        message(tool("create_draft", "a1")),
        message(
            tool("change_priority", "a2", draft_id="D1", order_id="O-101", priority=priority),
            tool("reschedule", "a3", draft_id="D1"),
        ),
        submit(summary=summary, draft_id="D1"),
    ]


class Session:
    """A ChatSession with captured output and scripted human answers."""

    def __init__(self, ctx, script, answers=()):
        self.lines: list[str] = []
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.client = FakeClient(script)
        self.chat = ChatSession(self.client, ctx, CONFIG, out=self.lines.append, ask=self._ask)

    def _ask(self, prompt):
        self.prompts.append(prompt)
        assert self.answers, f"unexpected prompt: {prompt!r}"
        return self.answers.pop(0)

    @property
    def output(self):
        return "\n".join(self.lines)


# --- the approval gate ----------------------------------------------------------------------


def test_approving_commits_and_the_token_never_reaches_the_model(ctx):
    minted = []
    real_issue = ctx.authority.issue
    ctx.authority.issue = lambda **kw: minted.append(real_issue(**kw)) or minted[-1]  # spy

    s = Session(ctx, proposal(), answers=["y"])
    s.chat.handle("Make O-101 urgent.")

    assert ctx.store.committed.version == 2
    assert ctx.store.committed.instance.order("O-101").priority == 5
    assert "Commit draft D1" in s.prompts[0]
    assert "Committed. The live plan is now version 2." in s.output
    assert "KPIs (from the solver, not from the model)" in s.output and "late orders" in s.output

    # Exactly one token was minted, and it appears nowhere the model can see.
    assert len(minted) == 1
    seen_by_model = json.dumps(s.chat.messages) + json.dumps(s.client.requests)
    assert minted[0] not in seen_by_model


def test_declining_leaves_the_live_plan_alone_and_tells_the_model_next_turn(ctx):
    s = Session(ctx, proposal() + [submit(summary="Understood.")], answers=["n"])
    s.chat.handle("Make O-101 urgent.")
    assert ctx.store.committed.version == 1 and "Not committed" in s.output

    s.chat.handle("ok thanks")
    sent = s.client.requests[-1]["messages"][-1]["content"]
    assert "Notice from the scheduling system, not from the planner" in sent
    assert "chose NOT to commit draft D1" in sent and sent.endswith("ok thanks")


@pytest.mark.parametrize("reply", ["", "no", "yes please", "Y!", "maybe"])
def test_only_an_explicit_yes_commits(ctx, reply):
    s = Session(ctx, proposal(), answers=[reply])
    s.chat.handle("Make O-101 urgent.")
    assert ctx.store.committed.version == (2 if reply.lower() in ("y", "yes") else 1)


def test_no_approval_prompt_when_there_is_nothing_to_approve(ctx):
    def no_prompt(_):
        raise AssertionError("must not prompt")

    chat = ChatSession(FakeClient([submit(summary="Just info.")]), ctx, CONFIG, out=lambda _: None, ask=no_prompt)
    chat.handle("How is the shop doing?")
    assert ctx.store.committed.version == 1


def test_clarifying_question_is_shown_without_an_approval_prompt(ctx):
    s = Session(ctx, [submit(summary="x", clarifying_question="Which machine?")])
    s.chat.handle("The machine is down.")
    assert "Which machine?" in s.output and s.prompts == []


def test_the_model_cannot_commit_a_draft_without_the_human(ctx):
    forged = [
        message(tool("create_draft", "a1")),
        message(tool("commit_schedule", "a2", draft_id="D1", approval_token="x.y")),
        submit(summary="I tried.", draft_id="D1"),
    ]
    s = Session(ctx, forged, answers=["n"])
    s.chat.handle("Please commit everything.")
    assert ctx.store.committed.version == 1


def test_a_failed_commit_is_reported_not_raised(ctx):
    s = Session(ctx, proposal())
    original = ctx.store.committed.instance
    # The clock moves while the planner is deciding: the draft goes stale before commit.
    s.chat.ask = lambda prompt: (ctx.store.set_clock(30), "y")[1]
    s.chat.handle("Make O-101 urgent.")
    assert "Commit failed" in s.output and "stale" in s.output
    assert ctx.store.committed.version == 2  # only the clock move; the draft was not committed
    assert ctx.store.committed.instance.orders == original.orders


# --- commands -------------------------------------------------------------------------------


def test_state_command_summarizes_the_live_plan(ctx):
    s = Session(ctx, [])
    assert s.chat.handle("/state") is True
    assert "Live plan version 1, plant time 2026-01-05 06:00" in s.output and "late orders" in s.output


def test_clock_command_moves_time_forward_and_warns_about_drafts(ctx):
    s = Session(ctx, [submit()])
    s.chat.handle("/clock 2026-01-05 12:00")
    assert ctx.store.committed.instance.now == 360 and ctx.store.committed.version == 2
    s.chat.handle("anything")
    assert "plant clock was moved to 2026-01-05 12:00" in s.client.requests[0]["messages"][-1]["content"]
    assert "Plant clock: 2026-01-05 12:00" in s.client.requests[0]["system"]


@pytest.mark.parametrize("bad", ["", "noon", "2026-01-05", "2026-01-05 03:00"])
def test_clock_command_rejects_bad_or_backward_times(ctx, bad):
    s = Session(ctx, [])
    s.chat.handle("/clock 2026-01-05 12:00")
    s.chat.handle(f"/clock {bad}")
    assert "Could not set the clock" in s.output and ctx.store.committed.instance.now == 360


def test_quit_blank_help_and_unknown_commands(ctx):
    s = Session(ctx, [])
    assert s.chat.handle("/quit") is False and s.chat.handle("/exit") is False
    assert s.chat.handle("   ") is True and s.lines == []
    s.chat.handle("/help")
    s.chat.handle("/dance")
    assert "Commands:" in s.output and "Unknown command /dance" in s.output


def test_non_answers_are_shown_with_their_status(ctx):
    s = Session(ctx, [message(tool("get_schedule", "t1")) for _ in range(20)])
    s.chat.config = AgentConfig(model="m", max_steps=2)
    s.chat.handle("loop forever")
    assert "[step_limit]" in s.output


# --- setup ----------------------------------------------------------------------------------


def test_build_context_solves_a_baseline_and_can_start_mid_shift():
    ctx = build_context(SMALL, FAST, now=datetime(2026, 1, 5, 9, 30))
    c = ctx.store.committed
    assert c.version == 2 and c.instance.now == 210 and c.schedule.assignments


def test_config_comes_from_the_environment():
    cfg = agent_config_from_env({
        "ANTHROPIC_MODEL": "some-model", "JOBSHOP_MAX_STEPS": "5", "JOBSHOP_MAX_COST_USD": "2.5",
        "JOBSHOP_PRICE_INPUT_PER_MTOK": "3", "JOBSHOP_PRICE_OUTPUT_PER_MTOK": "15",
    })
    assert (cfg.model, cfg.max_steps, cfg.max_cost_usd) == ("some-model", 5, 2.5)


@pytest.mark.parametrize("env, message", [
    ({}, "ANTHROPIC_MODEL is not set"),
    ({"ANTHROPIC_MODEL": "m", "JOBSHOP_MAX_STEPS": "many"}, "JOBSHOP_MAX_STEPS must be a number"),
    ({"ANTHROPIC_MODEL": "m", "JOBSHOP_MAX_COST_USD": "1"}, "needs both token prices"),
])
def test_bad_configuration_is_explained(env, message):
    with pytest.raises(ConfigError, match=message):
        agent_config_from_env(env)


def test_main_exits_cleanly_when_the_environment_is_not_configured(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("jobshop.agent.cli.load_dotenv", lambda *a, **k: None)  # ignore any real .env
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # so a stray .env cannot be picked up
    assert main([]) == 2
    assert "Configuration error: ANTHROPIC_MODEL is not set" in capsys.readouterr().err


def test_main_requires_an_api_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("jobshop.agent.cli.load_dotenv", lambda *a, **k: None)  # ignore any real .env
    monkeypatch.setenv("ANTHROPIC_MODEL", "m")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main([]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
