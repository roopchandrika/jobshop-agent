# Using the scheduler from an MCP client

The MCP server exposes the same scheduling tools as the chat CLI to any MCP client
(Claude Desktop, Claude Code, ...). A model can try changes on drafts, solve, and compare. It
**cannot commit**: it can only *request* approval, and a person approves in a terminal.

## 1. One-time setup

```bash
uv sync
uv run python -m jobshop.mcp_server.admin init --now "2026-01-05 12:00"
```

`init` generates the synthetic shop and solves its baseline plan (about 30 s), then writes the
shared state file `~/.jobshop/state.json`. Set `JOBSHOP_STATE` to use another location. The
server refuses to start without this file, because a baseline solve at startup would outlast the
client's connection timeout.

## 2. Connect a client

### Claude Desktop

Add this under `mcpServers` in `%APPDATA%\Claude\claude_desktop_config.json` (merge it with any
servers already there), then quit and reopen Claude Desktop completely:

```json
{
  "mcpServers": {
    "jobshop": {
      "command": "C:\\path\\to\\jobshop-agent\\.venv\\Scripts\\python.exe",
      "args": ["-m", "jobshop.mcp_server"],
      "env": { "JOBSHOP_SOLVE_SECONDS": "25" }
    }
  }
}
```

The virtualenv's `python.exe` is used instead of `uv` because a desktop app may not have `uv` on
its PATH. Replace `C:\path\to\jobshop-agent` with the folder you cloned the repo into.

### Claude Code

```bash
claude mcp add jobshop -e JOBSHOP_SOLVE_SECONDS=25 -- C:\path\to\jobshop-agent\.venv\Scripts\python.exe -m jobshop.mcp_server
```

Check `claude mcp add --help` if the flags differ in your version; the command is the same
`python.exe -m jobshop.mcp_server` either way.

## 3. Try it

Ask the model: *"What is the plant time and how is the live plan doing?"* then *"M4 is down from
14:00 to 17:00 today and O-112 is now urgent. What happens?"* It should create a draft, apply the
changes, reschedule once, compare, and explain. Then say *"Go ahead and request approval."*

## 4. Approve (you, in a terminal)

```bash
uv run python -m jobshop.mcp_server.admin status      # live plan, drafts, pending requests
uv run python -m jobshop.mcp_server.admin approve     # review the comparison, then y/N
uv run python -m jobshop.mcp_server.admin deny R1
uv run python -m jobshop.mcp_server.admin clock "2026-01-05 15:00"   # advance the shop clock
```

The review screen is built from the stored schedules, not from what the model said. Approval
fails (and commits nothing) if the plan moved on, the request expired (30 min), or the draft was
edited after the request. `clock` makes existing drafts and requests stale on purpose.

## Notes

- **Solve time.** `reschedule` runs the solver for up to `JOBSHOP_SOLVE_SECONDS` (default 30).
  Keep it below your client's tool-call timeout; 25 is a cautious choice. I have not measured
  Claude Desktop's timeout.
- **Workers.** `JOBSHOP_SOLVER_WORKERS` (default 8) sets the CP-SAT threads.
- **Logs.** The server logs to stderr only; stdout carries the protocol.
- **Reset.** `admin init --force` replaces the state, discarding drafts and requests.

## What is and is not verified

- Verified in tests: the real server process over real stdio with the MCP SDK's own client
  (tools listed, errors reported as error results, state shared with the approval command, a
  model cannot commit).
- Not verified: behaviour inside Claude Desktop or Claude Code themselves (tool-call timeouts,
  how they present instructions or long JSON results, and whether the model follows the
  workflow). MCP *elicitation* (the server asking you to confirm inside the client) exists in the
  SDK but is not used, because client support could not be checked.
