"""The prompt-change gate: everything the model is told, saved as text, so a change to it is a deliberate act.

A prompt is code that no compiler checks. A reworded sentence in the system prompt or a loosened tool description can change
behaviour in ways no unit test sees, and the only judge of that is an eval run against a real model. So the full text of what the
model receives (system prompt, every tool schema, the triage, plan and critic prompts, the fixed refusal) lives in
``evals/prompts.snapshot.json``. The test suite compares the code with it and fails on any difference, showing the diff. To
change a prompt on purpose: change it, run the evals, then ``python -m jobshop.evals snapshot --update`` and commit the result
together with the eval numbers that justify it.

Recorded model responses (``run --record``) go stale at the same moment, for the same reason: their request fingerprints cover
this text, so ``run --replay`` reports them as stale instead of replaying an answer to a question the model is no longer asked.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from jobshop.agent import claims, patterns
from jobshop.agent.loop import SUBMIT_SPEC
from jobshop.agent.prompts import build_system_prompt
from jobshop.core.solver import SolverConfig
from jobshop.evals.shop import fresh_context, load_shop, shop_path
from jobshop.knowledge import KnowledgeBase
from jobshop.tools.registry import ToolRegistry

SNAPSHOT_NAME = "prompts.snapshot.json"
PREFERENCES = ["I always want the earliest finish.", "Never schedule rush orders on M3 after 18:00."]
UPDATE_COMMAND = "python -m jobshop.evals snapshot --update"


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def current(evals_dir: Path, knowledge_dir: Path | None = None) -> dict[str, str]:
    """Every piece of text the model is given, built from the committed fixture shop and plant documents."""
    folder = knowledge_dir if knowledge_dir is not None else evals_dir.parent / "knowledge"
    knowledge = KnowledgeBase.from_directory(folder) if folder.is_dir() else None
    ctx = fresh_context(load_shop(shop_path(evals_dir, "default")), SolverConfig(time_limit_s=1, num_workers=1, seed=0), knowledge=knowledge)
    registry = ToolRegistry(ctx)
    reader = registry.scoped(patterns.READ_ONLY_TOOLS)
    return {
        "system_prompt": build_system_prompt(ctx),
        "system_prompt_with_preferences": build_system_prompt(ctx, PREFERENCES),
        "tools_agent": _json(registry.api_specs()),
        "tools_reader": _json(reader.api_specs()),
        "submit_response_tool": _json(SUBMIT_SPEC),
        "plan_rules": patterns.PLAN_RULES,
        "plan_tool": _json(patterns.plan_spec(sorted(registry.names()))),
        "critic_system": patterns.CRITIC_SYSTEM,
        "review_tool": _json(patterns.REVIEW_SPEC),
        "triage_system": patterns.TRIAGE_SYSTEM,
        "triage_tool": _json(patterns.TRIAGE_SPEC),
        "reader_role": patterns.READER_ROLE,
        "commit_refusal": patterns.COMMIT_REFUSAL,
        "live_claim_warning": claims.LIVE_CLAIM_WARNING,
    }


def load(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v.replace("\r\n", "\n") for k, v in data.items()}


def save(path: Path, snapshot: dict[str, str]) -> None:
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def differences(saved: dict[str, str], now: dict[str, str]) -> list[str]:
    """One readable block per piece of text that was added, removed or changed. Empty if nothing differs."""
    blocks = []
    for name in sorted(set(saved) | set(now)):
        if name not in saved:
            blocks.append(f"+ {name}: new, not in the snapshot")
        elif name not in now:
            blocks.append(f"- {name}: in the snapshot, no longer produced")
        elif saved[name] != now[name]:
            diff = difflib.unified_diff(saved[name].splitlines(), now[name].splitlines(), "snapshot", "code", lineterm="", n=1)
            lines = list(diff)
            shown = lines[:40] + ([f"... {len(lines) - 40} more diff lines"] if len(lines) > 40 else [])
            blocks.append(f"~ {name}\n" + "\n".join(shown))
    return blocks


def check(evals_dir: Path, knowledge_dir: Path | None = None) -> list[str]:
    path = evals_dir / SNAPSHOT_NAME
    if not path.exists():
        return [f"{path} does not exist. Create it with: {UPDATE_COMMAND}"]
    return differences(load(path), current(evals_dir, knowledge_dir))
