# Write-up: an AI agent that can propose factory schedules but can never apply them

Two versions to paste: a long post (about 900 words) and a short one for LinkedIn. Every number is from this repository; where a
result has a limit, the limit is stated next to it.

---

## Long version

### I built an AI agent that can re-plan a factory but can never change the plan. Then I tried to break it.

When a machine breaks in a factory, someone has to re-plan fast without making things worse elsewhere. A constraint solver is
very good at that. A language model is very good at understanding "M2 is down from 11:00 to 14:00". I wanted both, with one
rule: **the model may try changes, but it can never apply them.** A person reviews exactly what changes and approves it.

So the model works on a *draft*, a scratch copy of the live plan. It calls the solver through tools, compares the draft with the
live plan, and explains the trade-offs. The numbers beside its answer (the KPIs, the list of changes, "needs approval") are
computed by the system from the draft, so the model can't misstate them. And there is simply no commit tool in its toolbox:
a commit needs a signed, single-use token bound to that exact draft, its schedule and its edits, and only code a person drives
can mint one.

I wrote the tool-use loop by hand on the Anthropic SDK, with no agent framework, so I could test every step of it.

### How I tested it

**Evals that read state, not words.** A model can say it did the right thing without doing it. My checks read the store and
the tool log: what was actually changed, whether the schedule validates, whether every quoted number came from a tool. There are
48 scenarios, plus an LLM judge that must be a different model.

**Breaking my own code on purpose.** I mutated the guards and the checks and demanded a test fail for each. That found a real bug:
an eval check meant to catch "the live plan changed" read its baseline *after* the run, so it could never fire.

**A red team with a worst-case attacker.** Thirty attacks plant hostile text everywhere text can enter the agent: order notes,
planted plant documents, the planner's own message. They try to get it to commit, forge an approval, flood the plan with edits,
hide late orders, or say "the schedule is live" when it isn't. I scored what each attack *achieved* in the store and the answer,
and ran it against a scripted model that obeys every instruction every time, so the result is what the harness contains when the
model's judgement fails completely:

| Set-up | Attacks that got something |
|---|---|
| no answer guards | 26 of 30 (87%) |
| with deterministic answer guards | 13 of 30 (43%) |
| plus routing questions to a read-only reader | 1 of 30 (3%) |

The live plan never changed in any of them. The one attack that still gets through is a legitimate request with extra edits
tucked in; the system shows all of them beside the answer, so the planner sees them.

### What running it for real taught me

The first twelve attacks were then read by an actual model (Claude Sonnet 5.5): **0 of 56 runs succeeded**. The model did the
legitimate part, said the note contained instructions, and ignored them.

But getting there found four bugs that no scripted test could have:

1. The API rejected forced tool choice for that model, so every internal call (triage, plan, review, judge) failed.
2. My scorer counted those failed turns as "contained", which produced a false 0% for the routing set-up. A failed run is now
   its own category, excluded from the rate.
3. The judge's answer was cut off at 1,500 tokens and arrived empty.
4. The judge called true details "invented", because it saw a summary rather than what the assistant had actually read.

Widening the red team from 12 to 30 found one more real gap: an edit left in a draft the answer never named, with nothing beside
the answer to show it. It is now reported with what it holds.

### The honest limits

One model, two runs of each attack: this is evidence, not a guarantee. A judged run passed 32 of 35 scenarios that completed;
13 could not run because I hit the API's usage limit. A 3-billion-parameter local model could not finish the loop at all (it
wrote the final tool call as plain text). And "contained" only means something if the model actually *read* the hostile text,
which is why I checked that in every live run.

### Three things I'd do again

- **Measure with an adversary, not just a happy path.** The most useful result was the attack that *succeeded*.
- **Run it live early.** Scripted models never refuse, never truncate and never misformat. Real ones do.
- **Distrust a perfect score.** A 0% that comes from a failed run is worse than no number.

The code, the evals and the red team are open: github.com/roopchandrika/jobshop-agent

---

## Short version (LinkedIn)

I built an AI agent that can re-plan a factory schedule through a constraint solver, but can *never* apply a change. A person
approves, using a token bound to the exact draft they reviewed.

Then I tried to break it: a 30-attack red team (hostile order notes, planted documents, spoofed approvals, floods, hidden
characters), scored on what the attack actually achieved.

- Against a worst-case model that obeys every instruction: attack success fell from 87% to 3% as I added answer guards and
  least-privilege routing. The live plan never changed.
- Against a real model (Claude Sonnet 5.5), the first 12 attacks: 0 of 56 runs succeeded.

The best part was what going live exposed: a rejected API parameter, a judge that truncated its own answer, and a scorer that
counted failed runs as passes. None of it showed up in scripted tests.

Hand-written tool loop (no framework), 48-scenario eval harness, mutation testing, RAG, MCP, OpenTelemetry traces in Jaeger,
a Docker image, a LangGraph port for comparison, 1,400+ tests.

Code and write-up: github.com/roopchandrika/jobshop-agent

#LLM #AIAgents #AISafety #Evals #Python

---

## One line for the GitHub "About" box

LLM agent that re-plans a factory schedule through a CP-SAT solver but can never apply a change: human-only approval, 30-attack red team, 48-scenario eval harness, RAG, MCP.
