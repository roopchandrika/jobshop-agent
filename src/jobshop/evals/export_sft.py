"""Turn an eval run into a supervised fine-tuning dataset, using the evals as the quality filter.

This is rejection sampling: run the agent on many scenarios, keep only the conversations that passed every deterministic check
(and, if graded, the judge), and write them as training examples. The scenarios' checks read the system's state, so a kept
conversation did the right thing, not merely said it did. Training on such data is how a smaller model can be taught the loop.

Nothing is trained here. The output is a JSONL file, one conversation per line:

    {"id", "scenario", "category", "model", "pattern", "system", "tools", "messages", "checks", "judge_scores"}

``messages`` is in the Anthropic Messages format (user, assistant tool_use, user tool_result, ..., a final ``submit_response``),
so the same file replays through the API or converts to another provider's chat format. The conversation is rebuilt from the run's
trace (tool calls, arguments, results) and ``results.json`` (the final answer); the system prompt and tool list are rebuilt from the
code and the fixture shop, so they are the ones the model saw only if the prompts have not changed since the run (check
``evals snapshot``).

Privacy: traces and results hold the synthetic shop's data and the model's words only; no key or user data is involved.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobshop.agent.loop import SUBMIT, SUBMIT_SPEC
from jobshop.agent.prompts import build_system_prompt
from jobshop.core.solver import SolverConfig
from jobshop.evals.scenario import Scenario, load_scenarios
from jobshop.evals.shop import fresh_context, load_shops
from jobshop.knowledge import KnowledgeBase
from jobshop.tools.registry import ToolRegistry


@dataclass
class ExportSummary:
    considered: int = 0
    kept: int = 0
    skipped: Counter = field(default_factory=Counter)   # reason -> count


def conversation_from_trace(trace: Path, answer: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages of one turn: the planner's request, each step's tool calls and results, then the submitted answer."""
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(e for e in events if e["event"] == "turn_start")
    steps: dict[int, dict[str, Any]] = {}
    last_draft: str | None = None
    for e in events:
        if e["event"] == "llm_call" and e.get("purpose", "agent") == "agent":
            steps.setdefault(int(e["step"]), {"text": [], "calls": []})["text"] = [t for t in (e.get("text") or []) if t]
        elif e["event"] == "tool_call":
            step = steps.setdefault(int(e["step"]), {"text": [], "calls": []})
            step["calls"].append(e)
            args = e.get("arguments") or {}
            for key in ("draft_id", "after"):
                if isinstance(args.get(key), str):
                    last_draft = args[key]
    messages: list[dict[str, Any]] = [{"role": "user", "content": start["user_text"]}]
    for number in sorted(steps):
        step = steps[number]
        if not step["calls"]:
            continue
        blocks: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(step["text"])}] if step["text"] else []
        results: list[dict[str, Any]] = []
        for k, call in enumerate(step["calls"]):
            block_id = f"toolu_{number}_{k}"
            blocks.append({"type": "tool_use", "id": block_id, "name": call["tool"], "input": call.get("arguments") or {}})
            result = call.get("result")
            entry: dict[str, Any] = {"type": "tool_result", "tool_use_id": block_id,
                                     "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)}
            if call.get("is_error"):
                entry["is_error"] = True
            results.append(entry)
        messages.append({"role": "assistant", "content": blocks})
        messages.append({"role": "user", "content": results})
    payload: dict[str, Any] = {"summary": answer.get("summary") or ""}
    if answer.get("clarifying_question"):
        payload["clarifying_question"] = answer["clarifying_question"]
    draft = answer.get("draft_id") or (last_draft if answer.get("needs_approval") or answer.get("changes_made") else None)
    if draft:
        payload["draft_id"] = draft
    messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_final", "name": SUBMIT, "input": payload}]})
    return messages


def export(results_dir: Path, evals_dir: Path, out: Path, *, min_judge: int | None = None, include_failed: bool = False,
           knowledge_dir: Path | None = None) -> ExportSummary:
    data = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    scenarios: dict[str, Scenario] = {s.id: s for s in load_scenarios(evals_dir / "scenarios")}
    shops = load_shops(evals_dir, {scenarios[r["id"]].shop for r in data["results"] if r["id"] in scenarios})
    folder = knowledge_dir if knowledge_dir is not None else evals_dir.parent / "knowledge"
    knowledge = KnowledgeBase.from_directory(folder) if folder.is_dir() else None
    summary = ExportSummary()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in data["results"]:
            summary.considered += 1
            scenario = scenarios.get(r["id"])
            if scenario is None:
                summary.skipped["unknown scenario"] += 1
                continue
            if r["status"] != "answered" or not r.get("answer"):
                summary.skipped[f"turn did not answer ({r['status']})"] += 1
                continue
            if not r["passed"] and not include_failed:
                summary.skipped["failed a check"] += 1
                continue
            scores = (r.get("judge") or {}).get("scores") or {}
            if min_judge is not None and scores and min(scores.values()) < min_judge:
                summary.skipped[f"judge score below {min_judge}"] += 1
                continue
            if min_judge is not None and not scores:
                summary.skipped["not graded by the judge"] += 1
                continue
            trace = Path(r["trace"]) if r.get("trace") else None
            if trace is None or not trace.exists():
                alt = results_dir / "traces" / f"{r['id']}-{r['attempt']}.jsonl"
                trace = alt if alt.exists() else None
            if trace is None:
                summary.skipped["no trace file"] += 1
                continue
            ctx = fresh_context(shops[scenario.shop], SolverConfig(time_limit_s=1, num_workers=1, seed=0), scenario.now, scenario.poison, knowledge)
            line = {
                "id": f"{r['id']}#{r['attempt']}", "scenario": r["id"], "category": r["category"], "model": r.get("model"),
                "pattern": r.get("pattern"), "system": build_system_prompt(ctx, scenario.preferences),
                "tools": [*ToolRegistry(ctx).api_specs(), SUBMIT_SPEC],
                "messages": conversation_from_trace(trace, r["answer"]),
                "checks": {name: c["passed"] for name, c in r["checks"].items()},
                "judge_scores": scores or None,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            summary.kept += 1
    return summary
