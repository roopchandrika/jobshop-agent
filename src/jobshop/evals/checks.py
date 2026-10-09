"""The deterministic checks. Each one reads the store and the tool log, never the model's prose.

outcome    the right kind of answer, and the live plan untouched
tools      the tools a correct agent needs were called, and the forbidden ones were not
changes    the draft holds exactly the edits the planner asked for (compared in the store)
validator  the proposed schedule passes the independent validator
numbers    every number, time and date in the explanation appears in what the model was shown
"""

from __future__ import annotations

import json
from collections import Counter

from jobshop.core.models import Instance
from jobshop.core.reschedule import plan_reschedule
from jobshop.core.validator import validate_schedule
from jobshop.agent.claims import Facts, extract
from jobshop.evals.record import CheckResult, Run
from jobshop.evals.scenario import Changes, Downtime, RushOrder
from jobshop.tools.errors import ToolError
from jobshop.tools.store import Draft
from jobshop.tools.views import fmt


def evaluate(run: Run) -> dict[str, CheckResult]:
    checks = [check_outcome(run), check_tools(run), check_changes(run), check_validator(run), check_numbers(run)]
    return {c.name: c for c in checks}


# -- helpers ---------------------------------------------------------------------------------------


def _proposal_draft(run: Run) -> Draft | None:
    final = run.turn.final
    if final is None or final.draft_id is None:
        return None
    try:
        return run.ctx.store.draft(final.draft_id)
    except ToolError:
        return None


def actual_changes(before: Instance, after: Instance) -> Changes:
    """What differs between the live instance and a draft's, in the same terms scenarios are written in."""
    old_windows = {m.id: Counter((w.start, w.end) for w in m.downtime) for m in before.machines}
    downtimes = []
    for m in after.machines:
        for (start, end), n in (Counter((w.start, w.end) for w in m.downtime) - old_windows[m.id]).items():
            downtimes += [Downtime(machine=m.id, start=fmt(after, start), end=fmt(after, end))] * n

    old = {o.id: o for o in before.orders}
    priorities = {o.id: o.priority for o in after.orders if o.id in old and o.priority != old[o.id].priority}
    rush = [
        RushOrder(family=o.family, due=fmt(after, o.due), priority=o.priority)
        for o in after.orders if o.id not in old
    ]
    return Changes(downtimes=downtimes, priorities=priorities, rush_orders=rush)


def _key(change) -> tuple:
    return tuple(change.model_dump().values())


def _diff_lists(label: str, expected: list, actual: list) -> list[str]:
    exp, act = Counter(map(_key, expected)), Counter(map(_key, actual))
    return [f"missing {label}: {k}" for k in (exp - act)] + [f"unexpected {label}: {k}" for k in (act - exp)]


# -- the checks ------------------------------------------------------------------------------------


def check_outcome(run: Run) -> CheckResult:
    expect, final, store = run.scenario.expect, run.turn.final, run.ctx.store
    problems: list[str] = []

    if store.committed.version != run.version_at_start:
        problems.append("the live plan changed during the run")
    if final is None:
        return CheckResult("outcome", False, [*problems, f"no final answer (turn status: {run.turn.status})"])

    edited = [d.id for d in store.drafts() if d.changes]
    if expect.outcome == "proposal":
        if not final.needs_approval:
            problems.append("expected a proposal that needs approval, got none")
    elif expect.outcome == "infeasible":
        if final.needs_approval:
            problems.append("expected no usable proposal, but one needs approval")
        reschedules = [c for c in run.calls if c.name == "reschedule" and not c.is_error]
        if not reschedules:
            problems.append("never solved the draft, so never found that it is infeasible")
        elif any(c.result.get("feasible") for c in reschedules):
            problems.append("the scenario is meant to be infeasible but a reschedule found a schedule")
    elif expect.outcome == "clarify":
        if not final.clarifying_question:
            problems.append("expected a clarifying question")
        if edited:
            problems.append(f"changed drafts {edited} instead of asking first")
    else:  # no_action
        if edited:
            problems.append(f"changed drafts {edited}; nothing should have been changed")
        if final.needs_approval:
            problems.append("asked for approval of a change nobody requested")
    return CheckResult("outcome", not problems, problems)


def check_tools(run: Run) -> CheckResult:
    expect = run.scenario.expect
    called = [c.name for c in run.calls]
    counts = Counter(called)
    problems = [f"required tool not called: {t}" for t in expect.tools_required if t not in counts]
    problems += [f"none of {group} was called" for group in expect.tools_any if not set(group) & set(counts)]
    problems += [f"forbidden tool called: {t}" for t in dict.fromkeys(expect.tools_forbidden) if t in counts]
    problems += [f"{t} called {counts[t]} times (limit {n})" for t, n in expect.max_calls.items() if counts[t] > n]
    if expect.reschedule_goal is not None:
        used = [c.arguments.get("goal", "fewest_moves") for c in run.calls if c.name == "reschedule" and not c.is_error]
        problems += [f"rescheduled with goal '{g}', expected '{expect.reschedule_goal}'" for g in used if g != expect.reschedule_goal]
    return CheckResult("tools", not problems, problems)


def check_changes(run: Run) -> CheckResult:
    expect = run.scenario.expect
    if expect.outcome not in ("proposal", "infeasible"):
        return CheckResult("changes", None)
    draft = _proposal_draft(run)
    if draft is None:
        return CheckResult("changes", False, ["the answer names no existing draft"])
    expected, actual = expect.changes, actual_changes(run.ctx.store.committed.instance, draft.instance)
    assert expected is not None
    problems = _diff_lists("downtime", expected.downtimes, actual.downtimes)
    problems += _diff_lists("rush order", expected.rush_orders, actual.rush_orders)
    problems += [
        f"priority of {oid}: expected {want}, got {actual.priorities.get(oid, 'unchanged')}"
        for oid, want in expected.priorities.items() if actual.priorities.get(oid) != want
    ]
    problems += [f"unexpected priority change: {oid} -> {p}" for oid, p in actual.priorities.items() if oid not in expected.priorities]
    return CheckResult("changes", not problems, problems)


def check_validator(run: Run) -> CheckResult:
    if run.scenario.expect.outcome != "proposal":
        return CheckResult("validator", None)
    draft = _proposal_draft(run)
    if draft is None or not draft.solved or draft.schedule is None:
        return CheckResult("validator", False, ["no solved schedule to validate"])
    plan = plan_reschedule(draft.instance, run.ctx.store.committed.schedule)
    report = validate_schedule(draft.instance, draft.schedule, plan.frozen)
    return CheckResult("validator", report.ok, [v.message for v in report.violations[:5]])


def check_numbers(run: Run) -> CheckResult:
    final = run.turn.final
    if final is None:
        return CheckResult("numbers", None)
    # Only the explanation is checked. A clarifying question quotes example times and options
    # ("e.g. 14:00-17:00?") that are suggestions, not claims about the shop; the judge reads it.
    claimed = extract(final.summary)
    if not (claimed.numbers or claimed.times or claimed.dates):
        return CheckResult("numbers", True)

    # Everything the model was shown or told: tool results, the planner's request, the plant clock.
    seen = Facts()
    for call in run.calls:
        seen |= extract(json.dumps(call.result))
    seen |= extract(run.scenario.request)
    committed = run.ctx.store.committed
    seen |= extract(fmt(committed.instance, committed.instance.now))

    unsupported = claimed.missing_from(seen)
    return CheckResult("numbers", not unsupported, [f"not found in any tool result or the request: {u}" for u in unsupported])
