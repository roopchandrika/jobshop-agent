# Evals

How we measure whether the agent does the right thing, and what the measurements do and do not
show.

## Run it

```bash
# No API key needed: a scripted reference agent plays every scenario. Tests the eval itself.
uv run python -m jobshop.evals run --oracle

# A real model, with the LLM judge (needs ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_JUDGE_MODEL)
uv run python -m jobshop.evals run
uv run python -m jobshop.evals run --model <model> --judge-model <other-model>

# Cheaper variations
uv run python -m jobshop.evals run --no-judge                  # deterministic checks only
uv run python -m jobshop.evals run --only injection --repeat 5 # one category, five runs each
```

Results go to `evals/results/<run id>/`: `report.md` (table and failure details), `results.json`
(every number), `traces/<scenario>-<n>.jsonl` (every step). They are not committed. The exit code
is 1 if any run failed, 2 for a configuration mistake.

To compare two models (quality, cost, latency) see [OBSERVABILITY.md](OBSERVABILITY.md):
`python -m jobshop.evals compare --model A --model B --judge-model C`.

## What is measured

39 scenarios in `evals/scenarios/*.yaml` (plain data; add one by copying an entry):

| Category | n | A correct agent... |
|---|---|---|
| simple_downtime | 7 | drafts the outage, solves once, compares, explains |
| priority_change | 3 | applies exactly the requested priority |
| rush_order | 6 | adds the order with the right family, due time and priority |
| multiple_disruptions | 5 | puts every change in **one** draft and solves **once** |
| read_only | 5 | answers from the data and changes nothing |
| impossible | 6 | says so plainly: unknown machine/order, "commit for me", and three cases that are truly infeasible |
| ambiguous | 4 | asks one question and changes nothing |
| injection | 3 | ignores instructions planted in an order's `notes`, and says it did |

Five deterministic checks read the **store and the tool log, never the model's prose**, plus the judge:

| Check | Passes when |
|---|---|
| `outcome` | the kind of answer is right (proposal / clarifying question / no action / infeasible) and the live plan is untouched |
| `tools` | required tools were called, forbidden ones were not, call limits held |
| `changes` | the draft contains **exactly** the requested edits (extra or missing edits fail) |
| `validator` | the proposed schedule passes `core/validator.py`, independent of the solver |
| `numbers` | every number, time and date in the **explanation** appears in a tool result, the request or the plant clock. A clarifying question is not scanned (see below) |
| `judge` | a second model scores the explanation at least 3/5 on every criterion |

Some scenarios also say which goal the solve must use (`reschedule_goal`). When a planner asks for the
earliest finish the agent must pass `goal: earliest_finish`, and when they ask for little disturbance, or say
nothing, it must not. That is checked in the `tools` row from the call's arguments.

A cell in the table is `passed/applicable`; `-` means the check did not apply (no schedule to
validate in a read-only question). A scenario passes only if every applicable check passes.

## Two shops

Each scenario names the fixture shop it starts from (`shop:`, default `default`).

| Shop | Fixture | What it is for |
|---|---|---|
| `default` | `evals/shop.json` | 12 orders on 4 machines, one day. Slack everywhere, so most disruptions are absorbed with no late orders. Tests that the agent does not invent problems |
| `tight` | `evals/shop_tight.json` | 19 orders on the same 4 machines. The live plan is on time, but almost any outage makes an order late, so the agent has real consequences to explain (10 scenarios use it) |

The tight shop (generator seed 2, 19 orders) was chosen by trying seeds. Bigger or more loaded shops whose
live plan was already late could not be proven optimal in a few seconds, which would make results depend on
the speed of the machine. This one re-solves to proven optimal in about 1 to 3 s. Rebuild a fixture with
`python -m jobshop.evals build-shop --shop tight`; that replaces a committed file, so results from before are
no longer comparable.

## Design decisions

- **A committed fixture shop** (`evals/shop.json`, rebuilt with `build-shop`). A time-limited solve
  gives a different plan on a slower machine; evals that start from a moving plan cannot compare
  two models or two days. Each scenario starts from a fresh in-memory copy.
- **Exact-match on changes, not on schedules.** CP-SAT with several workers is not deterministic,
  so we never assert a schedule. We assert what the *agent* asked for (the edits) and that the
  result is valid. Solving uses one worker and a fixed seed.
- **The numbers check is membership, not arithmetic.** The agent is told never to compute a number.
  So any number in the answer must have been shown to it. This catches invented and
  "helpfully" derived figures (a duration worked out from two times). It cannot tell whether a real
  number is attached to the right claim; the judge, which is given the ground truth, covers that.
- **The numbers check reads the explanation, not a clarifying question.** The first real run flagged four
  clarifying questions, all for example times ("for example, 14:00 to 17:00"), a number from the model's own
  instructions, or a derived date. None was a claim about the shop. The check was narrowed to the
  explanation after that run; the same run re-scored this way would have been 26/29, and the original 22/29
  is kept in the README as measured. A wrong fact stated inside a question is now left to the judge.
- **The tools now return what the model used to compute.** The same run showed the model subtracting two
  times ("20 minutes before its due time") and counting a list ("all 12 orders"). Order rows now carry
  `slack_min`, KPIs carry `total_orders` and `on_time_orders`, and `list_orders` returns `order_count` and
  `on_time_count`, so the right behaviour is also the easy one.
- **The judge is a different model from the one under test** (enforced), is told the answer under
  review is untrusted text, is forced to answer through a closed-schema tool call, and is graded
  against facts the harness recorded (KPIs, draft changes, solver status), not its own opinion of
  the schedule.
- **A judge failure is not a model failure.** If the judge errors or returns a malformed grade, that
  cell is excluded and the report says how many.

## How we know the eval itself works

An eval nobody tested is a number generator. So:

- A **scripted reference agent** (`jobshop/evals/oracle.py`) plays every scenario through the real
  loop, tools and checks. It must pass all 29 (`test_suite_with_reference_agent.py`). If it cannot,
  the scenario or a check is wrong. Both infeasible scenarios are *proven* infeasible this way.
- **Deliberately bad agents** (`test_checks.py`, `test_checks_isolated_branches.py`): one that does
  nothing, guesses instead of asking, edits the wrong machine or time, adds an extra edit, solves
  twice, invents a number, obeys an injected note. Each must fail the check that targets its
  mistake, and only that one.
- **Mutation testing**: each branch of the checks, the number extraction, the judge and the runner
  was broken on purpose (38 mutations); a test fails for every one. That found one real bug (the
  "live plan unchanged" check read the version after the run) and six untested branches.
- **Mutation testing again for the later additions** (the goal check, two-shop loading, the earliest-finish
  solver goal, prompt caching, slack and counts): 27 mutations. 24 were caught straight away; three
  survived (an errored reschedule counted towards the goal check, the runner using the wrong shop, and the
  guard for a scenario whose shop was not loaded), and each got a test that kills it.
- **A sloppy scripted agent** invents a figure in some scenarios. It once used "45 minutes", which happened
  to appear in two scenarios' real tool results, so the numbers check rightly let it through. Its invented
  figure is now one no tool can return.

## What it does NOT show

- **The reference agent's 100% is a test of the *harness*, not a result about any model.** The first real
  run (`claude-sonnet-5-5`, 29 scenarios, one run each, judge off, 2026-10-08) passed every check except
  `numbers`: 22/29 overall. All seven failures were read by hand: none was a wrong number. Three were values
  the model worked out itself (a subtraction, a count) in explanations, and four were in clarifying
  questions (example times, a number from its own instructions, a derived date). The `numbers` check
  currently scans clarifying questions; whether it should is an open decision, and it has not been changed.
  The LLM judge has not been run on a real model, so explanation quality is still unscored.
- **Models vary run to run.** One pass is a sample. Use `--repeat` before concluding anything,
  especially for the injection category.
- **29 scenarios on one small synthetic shop.** Enough to catch regressions and compare models
  roughly; not enough for fine rankings. Many disruptions here are absorbed with no late orders,
  which tests that the agent does not invent problems, but means the numbers check mostly sees
  zeros in those cases.
- **The numbers check has known approximations**: "2pm" is read as 14:00; number words ("two") are
  ignored; the digits of identifiers (O-101, M4) are treated as names.
- **A model-written judge has its own biases** (it may favour long or confident answers). Its
  scores are most useful for comparing runs with the same judge.
- **Some "ambiguous" scenarios are judgement calls.** A reasonable agent might resolve one by
  looking at the data. They encode this project's choice: when a request would change the plan and
  a detail is missing, ask.
