# Design decisions, terms and limits

The reasoning behind the project, moved here from the README to keep the front page short. The
[README](../README.md) has the overview; [SAFETY.md](SAFETY.md), [EVALS.md](EVALS.md) and
[OBSERVABILITY.md](OBSERVABILITY.md) go deeper on those three areas.

## Terms

| Term | Meaning here |
|---|---|
| **Job shop** | A factory where each order needs a sequence of operations, each on a machine that has the right capability |
| **Live plan / draft** | The current plan / a scratch copy where changes are tried. Only a person can make a draft the live plan |
| **Tardiness** | Minutes an order finishes after its due time. Weighted by priority (1 to 5 weigh 1, 2, 4, 8, 16) |
| **Makespan** | When the last operation of the whole plan finishes |
| **Utilization** | Each machine's busy share of its open time from now until the last finish. It falls when the last finish moves later, even if the same work gets done |
| **CP-SAT** | The constraint solver in Google OR-Tools |
| **Proven optimal / feasible** | The solver proved nothing better exists / it found a valid schedule but may not have had time to prove it |
| **MCP** | Model Context Protocol: the standard that lets apps like Claude Desktop or Claude Code use external tools |
| **Approval token** | A signed, one-use, expiring proof that a person approved a specific draft |

## Design decisions

**Scheduling**
- A flexible job shop in CP-SAT with integer minutes and non-preemptive operations: each operation fits
  inside one availability window and never overlaps an outage.
- A strict order of goals: weighted tardiness first, then (against the live plan) fewest operations moved,
  then finish time. The consequence is stated, not hidden: one minute of tardiness outweighs any number of
  moves, and the second goal comes before the third, which is why the worked example in the README finishes later.
  The planner can swap the last two (`goal: earliest_finish`), and the agent offers that when the default
  pushes the finish noticeably later.
- The result always carries the solver status and which goals were proven. At the default size
  (81 operations, 30 s) nothing was proven optimal in my runs, and the agent is told to say so.
- An independent validator (it imports neither the solver nor its helpers) checks every schedule before a
  model sees it and again before a commit.

**The agent**
- A hand-written tool-use loop with the client injected, so tests drive it with a scripted stand-in that
  returns real SDK message objects. It stops on step, token and cost limits and never leaves its history invalid.
- Edit tools only touch a draft, and `reschedule` is the only tool that solves. A draft remembers the plan
  version it came from and is refused everywhere once that moves on.
- The model supplies words, not facts. KPIs, the change list and "needs approval" come from the store. Tool
  results use plant-local time strings and whole minutes, and utilization is rounded once, to 0.1%.
- The model is told never to calculate a number, and the eval checks that numbers in an answer appeared in
  something it was shown (the first real run found it breaks this rule occasionally).

**Safety** ([docs/SAFETY.md](SAFETY.md))
- The model has no commit tool. A commit needs a signed, single-use, expiring token bound to the draft, the
  plan version, and a fingerprint of the draft's schedule **and its edits**; only code a person drives can
  mint one. (The fingerprint once covered only the schedule; building the web UI showed that an edit that moved
  nothing could ride on an earlier approval. It was fixed, with regression tests.)
- Free text is data. Order notes reach the model on one line, size-capped, in a field named
  `notes_untrusted_text`. Tests show what even a fully obedient scripted model can do: edit a draft that a
  person then sees as it really is.
- Inputs are strict (times are exactly `YYYY-MM-DD HH:MM`, ids must match the shape the system generates),
  drafts, edits and pending requests are capped, and tests (plus a check in a real browser) confirm that model
  text cannot redraw the terminal or run as script in the page.
- The Approve endpoint is guarded: a per-run CSRF token, Origin and Host checks, a strict
  Content-Security-Policy, loopback-only serving, and the page sends the fingerprint of what it displayed so a
  changed draft cannot be approved.

**Evals and observability** ([docs/EVALS.md](EVALS.md), [docs/OBSERVABILITY.md](OBSERVABILITY.md))
- 39 YAML scenarios on two fixture shops; five checks that read the store and tool log; an LLM judge that must be a different
  model and treats the answer it grades as untrusted text. A scripted reference agent passes all 39 (this tests
  the harness, not a model) and deliberately bad agents each fail the check aimed at them.
- Traces are versioned JSONL with per-step tokens, latency and cost (null, never 0, when no prices are given),
  split into model time and solver time. The model comparison reports confidence intervals and a paired test
  instead of declaring a winner on 29 scenarios.

## Known limits

- At the default size nothing is proven optimal in 30 s, and strict priority means a tiny tardiness gain can
  justify moving many operations or a much later finish. `earliest_finish` is the escape hatch; a graded
  tardiness-versus-disruption trade-off is not implemented.
- One planner, one shop. State is in memory (web app) or a JSON file (MCP); there are no accounts or history,
  and restarting the web app resets drafts.
- The real model sometimes works out a number or date itself despite the rule against it (4 of 29 runs above, all correct).
- Setup times, labour, preemption and buffers are out of scope.
