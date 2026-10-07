"""ToolService error mapping, tested directly (the failures can't be triggered over a real subprocess)."""

import asyncio
import json
import logging

import pytest

from jobshop.mcp_server.server import ToolService
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import CreateDraftInput
from jobshop.tools.registry import Tool, ToolRegistry
from tests.helpers import FAST, SMALL, build_ctx


def call(service, name, arguments=None):
    return asyncio.run(service.call_tool(name, arguments))


def body(result):
    return json.loads(result.content[0].text)


def raising(exc):
    def fn(ctx, args):
        raise exc

    return Tool("fails", "always raises", CreateDraftInput, fn)


@pytest.fixture
def service_with():
    def make(*tools):
        ctx = build_ctx(SMALL)
        return ToolService(ctx.store, ToolRegistry(ctx, tools=list(tools), surface="agent"))

    return make


def test_a_tool_error_becomes_an_error_result_with_its_message(service_with):
    result = call(service_with(raising(ToolError("that draft is stale"))), "fails")
    assert result.is_error is True and body(result) == {"error": "that draft is stale"}


def test_an_unexpected_crash_becomes_a_generic_error_result_and_is_logged(service_with, caplog):
    with caplog.at_level(logging.ERROR, logger="jobshop.mcp"):
        result = call(service_with(raising(ZeroDivisionError("secret internals"))), "fails")
    assert result.is_error is True
    assert body(result) == {"error": "internal error in fails (see the server log)"}
    assert "secret internals" not in result.content[0].text  # details go to the log, not to the model
    assert "ZeroDivisionError" in caplog.text


def test_a_lock_timeout_is_reported_as_busy_not_as_a_crash(service_with):
    result = call(service_with(raising(TimeoutError("could not lock"))), "fails")
    assert result.is_error is True and "busy" in body(result)["error"]


def test_a_successful_call_returns_compact_json_and_no_error(service_with):
    ok = Tool("ok", "works", CreateDraftInput, lambda ctx, args: ctx.store.create_draft() and _Out())
    result = call(service_with(ok), "ok")
    assert result.is_error is False and body(result) == {"draft": "made"}
    assert " " not in result.content[0].text  # compact separators


class _Out:
    def model_dump(self, mode="json"):
        return {"draft": "made"}


def test_listing_matches_the_registry():
    ctx = build_ctx(SMALL)
    registry = ToolRegistry(ctx, surface="mcp")
    service = ToolService(ctx.store, registry)
    listed = asyncio.run(service.list_tools()).tools
    assert [t.name for t in listed] == registry.names()
    assert all(t.input_schema == s["input_schema"] for t, s in zip(listed, registry.api_specs()))
