"""End to end over real stdio: the SDK's own MCP client launches the real server process.

No mocks and no LLM. This is the same transport Claude Desktop and Claude Code use, so it
checks the protocol surface: what is advertised, how errors are reported, that state is shared
with the human approval command, and that a model cannot commit.
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from jobshop.core.generator import generate_instance
from jobshop.core.solver import solve
from jobshop.tools import human
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.functions import ToolContext
from jobshop.tools.store import Store
from tests.helpers import FAST, SMALL


@pytest.fixture
def state(tmp_path):
    path = tmp_path / "state.json"
    inst = generate_instance(SMALL)
    Store.create(path, inst, solve(inst, config=FAST))
    return path


def server_params(state_path):
    env = {
        "JOBSHOP_STATE": str(state_path),
        "JOBSHOP_SOLVE_SECONDS": "5",
        "JOBSHOP_SOLVER_WORKERS": "1",
    }
    return StdioServerParameters(command=sys.executable, args=["-m", "jobshop.mcp_server"], env=env)


def with_server(state_path, scenario):
    """Start the server, connect a client, run ``await scenario(session, init_result)``."""

    async def run():
        async with stdio_client(server_params(state_path)) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                return await scenario(session, init)

    return asyncio.run(asyncio.wait_for(run(), timeout=120))


def payload(result):
    """The JSON the tool returned."""
    assert len(result.content) == 1 and result.content[0].type == "text"
    return json.loads(result.content[0].text)


def test_the_server_advertises_the_right_tools_and_instructions(state):
    async def scenario(session, init):
        tools = (await session.list_tools()).tools
        return init, tools

    init, tools = with_server(state, scenario)
    names = {t.name for t in tools}

    assert init.server_info.name == "jobshop-scheduler"
    assert "request_commit" in init.instructions and "commits NOTHING" in init.instructions
    assert names == {
        "get_schedule", "list_orders", "get_order", "get_machine_status", "create_draft",
        "discard_draft", "simulate_downtime", "change_priority", "add_rush_order",
        "reschedule", "compare_schedules", "request_commit", "get_approval_status",
    }
    assert "commit_schedule" not in names  # a model can ask for approval, never grant it
    for tool in tools:
        assert tool.description.strip()
        assert tool.input_schema["type"] == "object" and tool.input_schema["additionalProperties"] is False


def test_errors_come_back_as_error_results_and_the_server_keeps_working(state):
    async def scenario(session, init):
        unknown = await session.call_tool("no_such_tool", {})
        commit = await session.call_tool("commit_schedule", {"draft_id": "D1", "approval_token": "x"})
        bad_args = await session.call_tool("change_priority", {"draft_id": "D1", "order_id": "O-101", "priority": 99})
        missing = await session.call_tool("reschedule", {})
        no_draft = await session.call_tool("reschedule", {"draft_id": "D9"})
        still_ok = await session.call_tool("get_schedule", {})
        return unknown, commit, bad_args, missing, no_draft, still_ok

    unknown, commit, bad_args, missing, no_draft, still_ok = with_server(state, scenario)
    for result in (unknown, commit, bad_args, missing, no_draft):
        assert result.is_error is True  # an error RESULT the model can read, not a protocol failure
    assert "unknown tool 'commit_schedule'" in payload(commit)["error"]
    assert "invalid arguments for change_priority" in payload(bad_args)["error"]
    assert "unknown draft 'D9'" in payload(no_draft)["error"]
    assert still_ok.is_error is False and payload(still_ok)["source"] == "committed"


def test_read_tools_return_what_the_state_file_says(state):
    async def scenario(session, init):
        return payload(await session.call_tool("get_schedule", {})), payload(await session.call_tool("list_orders", {}))

    schedule, orders = with_server(state, scenario)
    store = Store.open(state)
    with store.transaction():
        committed = store.committed
    assert schedule["version"] == 1 and schedule["now"] == "2026-01-05 06:00"
    assert [o["order_id"] for o in orders["orders"]] == [o.id for o in committed.instance.orders]


def test_full_workflow_with_approval_by_a_person_in_another_process(state):
    async def scenario(session, init):
        call = lambda name, **args: session.call_tool(name, args)
        draft = payload(await call("create_draft"))["draft_id"]
        await call("change_priority", draft_id=draft, order_id="O-101", priority=5)
        solved = payload(await call("reschedule", draft_id=draft))
        compared = payload(await call("compare_schedules", after=draft))
        requested = payload(await call("request_commit", draft_id=draft))
        before = payload(await call("get_approval_status", request_id=requested["request_id"]))
        live_before = payload(await call("get_schedule"))["version"]

        # ---- meanwhile, a person runs the approval command (a different process, same state file)
        store = Store.open(state)
        ctx = ToolContext(store, ApprovalAuthority(), FAST)
        with store.transaction():
            human.approve(ctx, requested["request_id"])

        after = payload(await call("get_approval_status", request_id=requested["request_id"]))
        live_after = payload(await call("get_schedule"))
        return solved, compared, requested, before, after, live_before, live_after

    solved, compared, requested, before, after, live_before, live_after = with_server(state, scenario)

    assert solved["feasible"] is True and compared["diff"]["kpi_after"]["late_orders"] >= 0
    assert requested["status"] == "pending" and "NOTHING HAS BEEN COMMITTED" in requested["message"]
    assert before["status"] == "pending" and live_before == 1  # asking changed nothing
    assert after["status"] == "approved" and live_after["version"] == 2  # the server saw the human's commit
    assert any(o["order_id"] == "O-101" and o["priority"] == 5 for o in live_after["kpis"]["orders"])


def test_a_model_cannot_make_an_approval_happen_by_asking(state):
    async def scenario(session, init):
        call = lambda name, **args: session.call_tool(name, args)
        draft = payload(await call("create_draft"))["draft_id"]
        await call("change_priority", draft_id=draft, order_id="O-101", priority=5)
        await call("reschedule", draft_id=draft)
        for _ in range(3):
            await call("request_commit", draft_id=draft)  # asking repeatedly does not escalate
        forged = await call("commit_schedule", draft_id=draft, approval_token="a.b")
        return forged, payload(await call("get_schedule"))["version"]

    forged, version = with_server(state, scenario)
    assert forged.is_error is True and version == 1


def test_the_server_refuses_to_start_without_state(tmp_path):
    env = {**os.environ, "JOBSHOP_STATE": str(tmp_path / "missing.json")}
    done = subprocess.run(
        [sys.executable, "-m", "jobshop.mcp_server"], env=env, capture_output=True, text=True, timeout=60, input=""
    )
    assert done.returncode == 2
    assert "No scheduling state found" in done.stderr and "admin init" in done.stderr
    assert done.stdout == ""  # stdout is the protocol channel: nothing may be printed to it
