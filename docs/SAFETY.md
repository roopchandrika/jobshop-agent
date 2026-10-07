# Safety model

What can go wrong when an LLM drives a scheduler, which layer stops it, and what is **not**
covered. The rule behind all of it: the model is treated as an untrusted client that may be
confused or manipulated, and nothing it says can change the live plan.

## The asset

The committed schedule, and the planner's trust in what they are shown before they approve.

## Threats and layers

| Threat | What stops it | Where it is tested |
|---|---|---|
| The model commits a schedule | No commit tool is offered to it (`commit_schedule` is `model_visible=False`; the MCP server has only `request_commit`). A commit needs an HMAC-signed, single-use, expiring token bound to the draft, the plan version and the schedule digest. Only code a person drives (the chat prompt, `admin approve`) mints one. | `test_approval`, `test_tools`, `test_server`, `test_injection` |
| Text in the data (an order's `notes`) tells the model what to do | **Structure:** whatever the model is persuaded to do is limited to editing drafts. **Visibility:** the approval screen lists the draft's changes as the system recorded them and the KPIs computed by the harness; the model supplies neither. **Judgement:** notes arrive only in a field named `notes_untrusted_text`, on one line, size-capped, and the system prompt says only the planner instructs the model. Judgement is probabilistic. | `test_injection` (scripted), `test_live_injection` (real model, opt-in) |
| Malformed or hostile tool arguments | Every field accepts one spelling (`"5"`, `true` and `5.0` are not priorities; times are exactly `YYYY-MM-DD HH:MM`; ids match the shape the system generates, so a 10 kB string is rejected without being echoed back). Errors say what was expected so the model can retry. | `test_hostile_inputs` |
| A model piles up work | Open drafts (10), changes per draft (20) and pending approval requests (5) are capped; outages are clipped to the plan, due dates must be within a year; a turn has step, token and cost limits. | `test_limits`, `test_changes`, `test_loop` |
| Text redraws the terminal to fake the approval screen | Everything printed by the chat CLI and `admin` goes through `terminal_safe`, which removes escape sequences and control, bidi and zero-width characters. | `test_text`, `test_injection`, `test_admin` |
| The model misdescribes its own changes | `changes_made` is not a field it can fill; it is copied from the draft. | `test_injection`, `test_loop` |
| A human approves something stale | Tokens and requests are bound to the plan version; a clock move or a commit makes older drafts and requests stale; `admin approve` re-validates at the moment of approval and never holds the lock while waiting for the person. | `test_approval_requests`, `test_admin` |

## What the text cleaning is, and is not

`untrusted_text` and `terminal_safe` are hygiene. They stop invisible characters and escape
sequences and bound the size. They do **not** make a sentence like "ignore your instructions"
harmless, and no filter can: recognising instructions in natural language is the model's job. I
deliberately did not add a keyword detector, which would catch the obvious attack, miss the
rest, and give false confidence.

## Residual risks (not solved)

- **A manipulated model can still lie in its summary.** The harness prints the real changes and
  KPIs beside it, but a human who ignores them can be misled. The structure bounds the damage;
  it does not make the prose true.
- **Real-model resistance is measured, not guaranteed.** `test_live_injection.py` has not been
  run by the author yet (no API key in the build environment). One passing run would be
  evidence, not proof; Phase 5 repeats scenarios like it.
- **Social engineering of the human.** A well-formed, plausible, harmful draft that a person
  approves is outside what code can prevent.
- **No rate limit on `reschedule`.** Each call can use up to `JOBSHOP_SOLVE_SECONDS`. In chat a
  turn is capped at `JOBSHOP_MAX_STEPS` model calls; over MCP the client decides.
- **Local trust.** The shared state file is not authenticated and the token secret lives in a
  process. Anyone who can write the file or run `admin approve` is the operator by definition
  (auth is a stated non-goal).
