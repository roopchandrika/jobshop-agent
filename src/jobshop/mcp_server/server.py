"""MCP server (stdio): the same tools as the chat agent, for any MCP client.

``python -m jobshop.mcp_server`` is meant to be launched BY an MCP client (Claude Desktop,
Claude Code, ...), which talks to it over stdin/stdout. That is why this module must never
print to stdout: stdout carries the protocol. Logs go to stderr.

What is and is not exposed:

* The tools come from ``ToolRegistry(surface="mcp")``: the same functions and the same
  Pydantic schemas the chat agent uses, plus ``request_commit`` and ``get_approval_status``.
* ``commit_schedule`` is NOT exposed. A model can only *ask* for approval; a human approves
  from another process (``python -m jobshop.mcp_server.admin approve``). This process holds an
  ApprovalAuthority whose secret nobody else has, so even a bug here could not commit.

State is a JSON file shared with that approval command, so every tool call runs inside
``store.transaction()`` (lock, reload, run, write back if changed).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from jobshop.agent.prompts import server_instructions
from jobshop.core.solver import SolverConfig
from jobshop.tools.approval import ApprovalAuthority
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import ToolContext
from jobshop.tools.registry import ToolRegistry
from jobshop.tools.store import Store

log = logging.getLogger("jobshop.mcp")

DEFAULT_STATE = Path.home() / ".jobshop" / "state.json"


def state_path_from_env(env: Mapping[str, str]) -> Path:
    return Path(env.get("JOBSHOP_STATE") or DEFAULT_STATE)


def solver_config_from_env(env: Mapping[str, str]) -> SolverConfig:
    return SolverConfig(
        time_limit_s=float(env.get("JOBSHOP_SOLVE_SECONDS") or 30),
        num_workers=int(env.get("JOBSHOP_SOLVER_WORKERS") or 8),
    )


def _result(payload: Any, *, is_error: bool = False) -> types.CallToolResult:
    text = json.dumps(payload, separators=(",", ":"))
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)


class ToolService:
    """Maps MCP requests onto the tool registry. Separate from the SDK so it can be tested directly."""

    def __init__(self, store: Store, registry: ToolRegistry) -> None:
        self._store = store
        self._registry = registry

    async def list_tools(self) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=s["name"], description=s["description"], input_schema=s["input_schema"])
                for s in self._registry.api_specs()
            ]
        )

    def _run(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._store.transaction():
            return self._registry.call(name, arguments)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        # Every failure becomes an error *result*. An exception escaping this handler would
        # reach the client as a protocol error instead, which models cannot read or recover from.
        try:
            # The solver can run for many seconds: keep it off the event loop.
            payload = await anyio.to_thread.run_sync(self._run, name, arguments or {})
            return _result(payload)
        except ToolError as e:
            return _result({"error": str(e)}, is_error=True)
        except TimeoutError:
            return _result({"error": "the shared scheduling state is busy; try again in a moment"}, is_error=True)
        except Exception:
            log.exception("tool %s failed unexpectedly", name)
            return _result({"error": f"internal error in {name} (see the server log)"}, is_error=True)


def build_server(store: Store, solver_config: SolverConfig, registry: ToolRegistry | None = None) -> Server:
    if registry is None:
        # This process can never commit: it holds an authority whose secret nobody else has, and
        # commit_schedule is not offered on the "mcp" surface in the first place.
        ctx = ToolContext(store=store, authority=ApprovalAuthority(), solver_config=solver_config)
        registry = ToolRegistry(ctx, surface="mcp")
    service = ToolService(store, registry)

    async def on_list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return await service.list_tools()

    async def on_call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        return await service.call_tool(params.name, params.arguments)

    try:
        version = importlib.metadata.version("jobshop-agent")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0"
    return Server(
        "jobshop-scheduler",
        version=version,
        instructions=server_instructions(),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    path = state_path_from_env(os.environ)
    if not path.exists():
        print(
            f"No scheduling state found at {path}.\n"
            "Create it first with:  uv run python -m jobshop.mcp_server.admin init\n"
            "(or set JOBSHOP_STATE to an existing state file)",
            file=sys.stderr,
        )
        return 2

    server = build_server(Store.open(path), solver_config_from_env(os.environ))
    log.info("serving %s over stdio", path)

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(serve)
    return 0
