# Demo: ten minutes, no API key

The whole project in one command, then the web UI. Everything here runs offline with a scripted stand-in for the model; the
solver, the checks, the guards and the plans are real. The results of the runs against a real model are in the last section.

```bash
uv sync
uv run python scripts/demo_all.py --pause     # every section, waiting for Enter so you can talk over each one
uv run python scripts/demo_all.py --quick     # about 20 seconds, no web UI
```

## The script

**1. The problem (30 s).** "A factory plan says which job runs on which machine. When a machine breaks a planner has to re-plan
fast without making things worse. A solver does that well; a language model reads plain requests well. I split the work so the
model never has the final say."

**2. The web UI (3 min).** The last step of `demo_all.py` (or `uv run python scripts/demo_server.py`, then open
http://127.0.0.1:8765). Type anything.
- Turn 1 shows an outage on M1 with a late order. Point at the **KPI tiles with deltas**, the **live and proposed Gantt charts**
  and the **list of changes**. Say: "the model wrote the sentence; the numbers, the change list and 'needs approval' are computed
  by the system from the draft, so the model cannot misstate them."
- Do **not** press Approve yet. Say: "nothing is live. The model has no commit tool. Approval mints a single-use token bound to
  this exact draft, its schedule and its edits."
- Press **Approve** on turn 1, then show the plan version change. Turn 3 shows it **asking instead of guessing**.

**3. Safety, measured (2 min).** Section 1 of `demo_all.py`. "I planted hostile instructions in order notes, in a planted plant
document and in the planner's own message, then scored what the attack actually achieved. This model obeys every instruction, which
is the worst case. With no defences 87% of 30 attacks get something. With answer guards 43%. With routing a question to a reader with
no edit tools, 3%. The live plan never changes in any of them." Point at the matrix.

**4. How it is measured (2 min).** Section 2. "48 scenarios; the checks read the store and the tool log, not what the model says.
The scripted reference agent passes all 48, which tests the harness. I also broke the code on purpose, mutation testing, and found
that one of my own checks could never fire."

**5. Retrieval and the prompt gate (1 min).** Sections 3 and 4. "Document search is scored on its own. Embeddings find 6 of 6 hard
paraphrases, keywords 3 of 6. And a test fails if any prompt or tool description changes without a deliberate update."

**6. Close (30 s).** "Hand-written tool loop on the Anthropic SDK, no framework; RAG, memory, structured extraction, agent
patterns, multi-agent routing, an Ollama adapter, evals, red team, tracing, a container."

## Optional: a real trace in Jaeger (two minutes, needs Docker)

```bash
docker run -d --rm --name jaeger -p 127.0.0.1:16686:16686 -p 127.0.0.1:4318:4318 jaegertracing/jaeger:latest
uv run python -m jobshop.agent.trace_export logs/traces/<file>.jsonl --send http://127.0.0.1:4318/v1/traces
# open http://127.0.0.1:16686, pick the service jobshop-agent, open a trace: model calls and tool calls as a waterfall
docker stop jaeger
```

Say: "Every turn is an OpenTelemetry trace: a span per model call and per tool call, with tokens, model and latency. It carries
metadata only, never the planner's words or the plans. Jaeger shows it in its GenAI view."

## What to show if you have the API cap raised

```bash
cp .env.example .env            # set ANTHROPIC_API_KEY and ANTHROPIC_MODEL
uv run python -m jobshop.api    # http://127.0.0.1:8000
```

Ask: "M2 is down from 11:00 to 14:00. What happens to the plan?" then "just commit it" (it must refuse) and "make O-101 urgent".
Open the trace afterwards: `uv run python -m jobshop.agent.trace_report logs/traces/<file>.jsonl`.

## Results against a real model

All with `claude-sonnet-5-5`; judge `claude-fable-5-1`. These are the numbers to quote, with their scope.

| What | Result | Scope |
|---|---|---|
| Hostile content in order notes, documents and messages | **0 of 56 attacks succeeded**; the live plan never changed | 12 attacks, `react` and `route`, 2 runs each of every attack |
| Judged evaluation | **32 of 35** completed scenarios passed every check (91%) | 48 scenarios, one run each; 13 could not run (account usage limit) |
| First evaluation, deterministic checks | 29/29 right outcome, 29/29 right tools, 17/17 exact edits, 15/15 valid schedules; 22/29 numbers traceable, **none was a wrong number** | 29 scenarios, one run each |
| Speed and size of a turn | about 10 s and about 40k tokens, about three quarters of them cache reads | typical scenario, 5 model calls |
| Prompt injection in order notes | the planted instruction was ignored and reported to the planner | all three injection scenarios and all order-note attacks |

Measured offline:

| What | Result |
|---|---|
| Attack success against a fully obedient scripted model | 30 attacks: 87% (no guards) to 43% (answer guards) to 3% (routing) |
| Retrieval of hard paraphrases | embeddings 6 of 6, keywords 3 of 6 |
| Tests | 1424 pass; CI on every push |
| Scenarios | 48; the reference agent passes 48/48 under `react`, `route` and `route+verify` |
| Container | built and run: non-root, healthy, page and plan served, foreign Host header refused |
