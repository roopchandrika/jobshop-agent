# Observability and model comparison

Two questions: *what happened in that run, and what did it cost?* (traces) and *which model
should I use?* (comparison).

## Traces

Every agent turn writes JSONL (one JSON object per line) to `logs/traces/` for the chat CLI and
to `evals/results/<run>/traces/` for evals. Read one with:

```bash
uv run python -m jobshop.agent.trace_report logs/traces/20260105-120000.jsonl
```

```
turn  step  in tok  out tok  llm ms     cost  tool ms  tools
------------------------------------------------------------
   1     1    1800      250      11  $0.0092        0  create_draft
   1     3    1800      250      11  $0.0092      499  reschedule
   ...
5 model call(s): 9000 in + 1250 out tokens, cost $0.0457, model time 0.1 s, tool time 0.5 s
```

| Event | Fields that matter |
|---|---|
| `turn_start` | `trace_version`, `model`, `user_text` |
| `llm_call` | `step`, `model`, `response_id`, `stop_reason`, `input_tokens`, `output_tokens`, `latency_ms`, **`step_cost_usd`**, `total_cost_usd`, `cache_read_tokens`, `cache_write_tokens`, `text`, `tool_calls` |
| `tool_call` | `step`, `tool`, `arguments`, `is_error`, `result` (in full), `latency_ms` |
| `turn_end` | `status`, `steps`, token totals, `cost_usd`, `llm_ms`, `tool_ms`, `wall_ms` |
| `api_error`, `tool_exception`, `dropped_block` | the unhappy paths, with how long a failed call took |

Things worth knowing:

- **Cost is per step and also a total.** Version 1 of the trace logged only a running total under
  the name `cost_usd`; version 2 (`trace_version`) adds `step_cost_usd`. The step costs add up to
  the turn cost (tested).
- **Costs are `null`, not 0, when no prices were given.** Prices are never built in: they differ
  per model and change. Pass them with `--price IN,OUT` (USD per million tokens) or the
  `JOBSHOP_PRICE_*` variables.
- **Model time and tool time are separate.** `llm_ms` is waiting for the model; `tool_ms` is
  running tools, which is almost entirely the solver (`reschedule`). When a turn is slow, this says
  whether to blame the model or the solver, and only one of those is the model's fault.
- **Tokens grow with every step.** The whole conversation and the tool definitions are re-sent on
  each model call, so a 5-step turn pays for the system prompt five times. Prompt caching would cut
  that; it is not implemented, and the cost figures assume it is not in use (cache token counts are
  recorded so a surprise would be visible).
- **Traces contain full tool results and the planner's text.** They are local and gitignored. They
  never contain an approval token or API key (tested), but treat them as private anyway.

## Comparing models

```bash
# Needs ANTHROPIC_API_KEY. Two different models to compare, and a third, different one to judge.
uv run python -m jobshop.evals compare \
    --model <model-A> --price <model-A>=3,15 \
    --model <model-B> --price <model-B>=1,5 \
    --judge-model <model-C> --repeat 3

# See the report format without an API key (two scripted agents, made-up prices):
uv run python -m jobshop.evals compare --demo
```

Replace the prices with your models' real ones. The command refuses to run if the judge is one of
the models, if only one model is given, or if a price names a model that is not being compared.
Each model gets the same scenarios, the same fixture shop, the same solver limit and the same
judge. Output goes to `evals/results/<run>/`: `comparison.md` (the side-by-side), plus a full
`report.md`, `results.json` and per-step traces for each model.

### How to read `comparison.md`

- **Quality.** Pass rate with a 95% confidence interval, then each check. With 29 scenarios the
  interval is wide (a 90% pass rate is anywhere from about 74% to 97%). Overlapping intervals mean
  you cannot call one model better on this evidence.
- **The paired view** lists the runs where exactly one model passed, with the check the other
  failed, and a sign test over those. Four wins to none gives p = 0.125: not convincing. Ten to
  none gives p = 0.002. Runs where both pass or both fail say nothing about which is better.
- **Cost.** Compare *cost per passing run*, not cost per run: a model that is half the price and
  passes a quarter as often costs twice as much per solved scenario (tested).
- **Latency.** Mean, median and p95 per scenario run, split into waiting-for-the-model and
  running-the-solver. The solver part is the same work for both models; compare the other line.
  With ~29 runs the p95 is nearly the maximum.

## What this does not tell you

- **The two-model comparison has not been run on real models yet.** The demo uses scripted agents; its
  numbers describe the report, not any model. (A single-model real run exists; see [EVALS.md](EVALS.md).)
- **Latency is noisy**: it includes the network, API load at that minute and rate limiting. Repeat
  runs at different times before trusting a difference of less than about 30%.
- **Quality is only what the scenarios measure.** A model can pass all 29 and still be worse at a
  request the scenarios do not cover. The scenarios were written before any model was run on them
  and may need adjusting after the first real run.
- **One judge.** Judge scores compare models fairly only against each other.

## Deliberately not built

Dashboards, OpenTelemetry export, MCP-server tracing (the MCP path makes no model calls, so there
are no tokens to count), and prompt caching. Each is a reasonable next step; none is needed to
answer the two questions above.
