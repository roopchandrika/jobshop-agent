# Roadmap: what has been done, and what is next

A one-page map of the project: the phases that are finished, the evidence behind each, the problems
that testing caught, what has *not* been verified, and what to do next.

## In one paragraph

An LLM agent helps a production planner handle disruptions in a job shop (a machine breaks, an order
turns urgent, a rush order arrives). The agent calls a CP-SAT scheduler as tools, tries changes on a
scratch **draft**, compares it with the live plan, and explains the trade-offs. **It can never commit a
schedule**: a person approves, in a chat prompt, a terminal command, or a web button. The agent loop is
hand-written on the Anthropic SDK (no framework), the scheduling core never depends on the LLM, all data
is synthetic, and the safety properties are covered by tests that were themselves checked by breaking
the code on purpose.

## At a glance

| Phase | What it delivered | Status | Commit |
|---|---|---|---|
| 0 | Proposal, working rules, design decisions | done | `b87ede1`, `c30cc31` |
| 1 | Scheduling core: data model, generator, CP-SAT solver, independent validator, KPIs | done | `43f84fc` |
| 2 | Tools, hand-written agent loop, chat CLI, approval tokens (+ stability fix) | done | `e8b25da`, `fe8a743` |
| 3 | MCP server with out-of-band human approval | done | `78aa16c` |
| 4 | Safety: strict inputs, caps, prompt-injection tests | done | `9c7f51d` |
| 5 | Evals: 29 scenarios, 5 deterministic checks, LLM judge | done | `de890ac` |
| 6 | Observability (per-step tokens, latency, cost) and a two-model comparison | done | `6c4a4c2` |
| 7 | FastAPI backend and web UI with an Approve button, README | done | `42da3b3` |
| after | Rename rules file, scripted demo, fixes from the first real answer | done | `35d212c`, `043bc28`, `2add2e2` |
| next | Real-model evals and the two-model comparison | **not run yet** | see [What is next](#what-is-next) |

## Phase by phase

### Phase 0: Ground rules
- **Goal:** agree on the rules before writing code.
- **Done:** hard rules (synthetic data only, no agent framework, keys from the environment, core
  independent of the LLM, ask before adding dependencies, stop after every phase); approved design
  decisions; the original spec kept local and out of Git history.
- **File:** [`PROJECT_RULES.md`](../PROJECT_RULES.md)

### Phase 1: The scheduling engine (no AI)
- **Goal:** a solver that is correct on its own, so the agent has something trustworthy to call.
- **Done:** flexible job-shop model; synthetic shop generator; CP-SAT solver with a strict order of
  goals (weighted tardiness, then fewest operations moved, then finish time); a validator that imports
  neither the solver nor its helpers; KPIs and schedule diffs; rules for outages, priority changes, rush
  orders, and what happens to work already started.
- **Key decision:** never call a result optimal unless the solver proved it; report which parts were proven.
- **Evidence:** the solver's own reported numbers equal numbers recomputed independently, in tests.

### Phase 2: Tools, agent loop, chat CLI
- **Goal:** let a model use the solver safely.
- **Done:** tool functions with typed inputs; drafts (change tools only edit a draft, only `reschedule`
  solves); a hand-written loop with step, token and cost limits; a terminal chat; signed, single-use,
  expiring approval tokens; a final answer whose KPIs and change list are computed by the system, not
  typed by the model.
- **Problem found by measuring:** the first re-plan moved 74 to 79 of 81 operations after a small
  disruption. A "move as few operations as possible" goal was folded into the solve, which cut it to
  about 29. (Single runs on an unproven solve, so treat the size of the gain as indicative.)

### Phase 3: MCP server
- **Goal:** use the same tools from Claude Desktop or Claude Code.
- **Done:** a stdio MCP server on the same tool registry; a lock-protected shared state file; the model can
  only *request* approval; a person runs `admin approve`, which shows numbers computed from stored
  schedules and re-validates at the moment of approval. Setup guide: [`docs/MCP.md`](MCP.md).
- **Key decision:** approval happens out of band (not through the client), so it works with any client.

### Phase 4: Safety
- **Goal:** assume the model can be confused or manipulated.
- **Done:** every tool argument has exactly one accepted form; caps on drafts, edits and pending requests;
  order notes reach the model as one cleaned, size-capped line in a field named `notes_untrusted_text`;
  terminal text cannot redraw the approval screen; the "changes made" list comes from the draft, not from
  the model; prompt-injection tests with a fully obedient scripted model. Threat model:
  [`docs/SAFETY.md`](SAFETY.md).
- **Evidence:** 20 guards broken on purpose, a test failed for each.

### Phase 5: Evals
- **Goal:** measure whether the agent behaves, not just whether the code runs.
- **Done:** 29 YAML scenarios in 8 categories; five checks that read the store and the tool log rather
  than the model's words (right kind of answer, right tools, exactly the right edits, valid schedule,
  only real numbers quoted); an LLM judge that must be a different model and treats the answer as
  untrusted; a committed fixture shop so runs are comparable. Details: [`docs/EVALS.md`](EVALS.md).
- **Evidence:** a scripted reference agent passes all 29 (this tests the harness, not any model);
  deliberately bad agents each fail the check aimed at them; 38 mutations of the eval code caught.

### Phase 6: Observability and model comparison
- **Goal:** know what a run cost and where the time went; compare two models fairly.
- **Done:** versioned JSONL traces with per-step tokens, latency and cost (null, never 0, without prices);
  a trace viewer; a `compare` command that reports quality with confidence intervals and a paired test,
  cost per *passing* run, and model time separated from solver time. Details:
  [`docs/OBSERVABILITY.md`](OBSERVABILITY.md).

### Phase 7: Web app
- **Goal:** the interface from the spec.
- **Done:** FastAPI backend; a dependency-free page with chat, KPI tiles with deltas, a proposal card,
  before/after Gantt charts, a table view, dark mode and a phone layout; an Approve button guarded by a
  CSRF token, Origin and Host checks, a strict Content-Security-Policy, loopback-only serving, and a
  fingerprint of the proposal that was displayed; the README with an architecture diagram.
- **Evidence:** driven end to end in a real browser; 31 of 32 guard mutations caught (the other is
  behaviourally identical to the original code).

### After Phase 7
- Renamed the rules file to `PROJECT_RULES.md`; published the repo; removed a local path from the docs.
- Added `scripts/demo_server.py`: the real app and solver with a scripted model, to try the UI with no API key.
- **First real-model answer** (one scenario, one run): every number matched the system's own comparison.
  Three weaknesses were fixed (a misleading utilization drop, a suggestion no tool could try, and a long
  answer); the fixes are tested but not yet re-checked against the live model.

## How the test suite grew

| After | Tests passing |
|---|---|
| Phase 3 | 305 |
| Phase 4 | 398 |
| Phase 5 | 527 |
| Phase 6 | 592 |
| Phase 7 | 663 |
| now | **679** (2 more are opt-in and call the real API) |

About 6,300 lines of source, 5,800 lines of tests.

## Problems that testing caught (the useful stories)

| Phase | Problem | How it was found | Fix |
|---|---|---|---|
| 2 | Re-planning moved nearly every operation | measuring a real disruption | folded a "fewest moves" goal into the solve |
| 5 | The eval's "live plan unchanged" check read its baseline after the run, so it could never fire | mutation testing of the eval code | read it before the run, plus a regression test |
| 7 | Approval bound only to the schedule: an edit that moved nothing could ride on an earlier approval (web and MCP) | writing the web approval test | approval now binds to the schedule *and* the draft's edits |
| 7 | KPI values wrapped; an axis label collided with the next tick | looking at the page in a real browser | layout fixes |
| 7 | A mutated test started a real server on all network interfaces | running mutation tests | test now fails fast if a server start is attempted |
| real run | Utilization drop read as "idle machines"; an offer to try overtime; a six-section answer | the first real model answer | a data note, prompt rules, and a UI caption |

## Honest status

| Verified | Not verified |
|---|---|
| Solver against an independent validator | **Any systematic real-model evaluation.** One real answer was checked by hand; the 29 scenarios, the LLM judge and the two-model comparison have not been run on a real model |
| Tools, loop, approval, store, MCP server over real stdio | Claude Desktop/Code connecting to the MCP server |
| The web API over a real socket, and the UI in a real browser | Screen readers; browsers other than one |
| Safety, eval and API code by mutation testing | Whether the new prompt rules change a real model's answer |

## What is next

**Now (small, cheap)**
1. Re-send "M2 is down 11:00 to 14:00" to the live model and check the three fixes took effect (about 30k tokens).
2. Put real prices in `.env` so costs show as dollars.

**Next (real evidence; spends API credit, roughly 30k tokens per scenario)**
3. `uv run python -m jobshop.evals run --no-judge`, then with a judge. Fix scenario wording the first real run exposes.
4. `uv run python -m jobshop.evals compare ...` on two models with `--repeat 3`; paste the table into the README.
5. Connect Claude Desktop or Claude Code to the MCP server and note what actually happens.

**Open design decisions**
- The goal order is lateness, then fewest moves, then finish time. In the first real run that gave 8 moves
  and a finish 3 hours later (14:25 to 17:30). Is that the right order, or should a tolerance trade a few extra
  moves for an earlier finish?
- Whether to add prompt caching (about 5k tokens are re-sent on every model call).

**Later (deliberately out of scope so far)**
- Accounts and multi-user; a persistent store for the web app; setup times, labour and preemption in the model;
  a CI workflow that runs the tests on every push; more shops and scenarios.

## How to explain it in 30 seconds

> It is an LLM agent for production planners. The model calls a constraint solver as tools to re-plan around
> a disruption, but it works on drafts and can never commit; a human approves exactly what they were shown.
> I built the tool-use loop by hand, exposed the same tools over MCP, defended against prompt injection, and
> wrote an eval suite whose checks read the system's state rather than the model's words. I tested the tests
> by breaking the code on purpose, which found real bugs. What I have not yet done is run the full evals on
> real models.

Interview preparation questions for each phase are at the end of each phase's report; the topics are:
how approval is made unforgeable (2, 3), why free text cannot steer the agent (4), why the checks read state
and not prose (5), why compare cost per passing run (6), and why approval must bind to the edits (7).
