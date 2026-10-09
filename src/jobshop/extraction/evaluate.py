"""Scoring an extractor on labelled emails.

Four questions, because accuracy alone hides the failures that matter:

* **Field accuracy**: of the values the emails state, how many did it get right?
* **Invented values**: how often did it fill a field the email does not state? (The dangerous mistake: it looks
  like data and it is not.) Counted after validation, which is what a person would actually be shown.
* **Abstention**: on emails that need a human (missing time, unknown product, not an order), how often did it say
  so instead of presenting a finished order? And how often did it raise a false alarm on a clean email?
* **Exact**: all fields right *and* the right verdict.

Scores are reported on what survives ``validate``; ``raw_ungrounded`` separately counts how often the extractor
quoted words that are not in the email, which is how a model's confabulation shows up before validation hides it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from jobshop.extraction.schema import FIELDS, Extraction, Verdict
from jobshop.extraction.validate import normalise, validate

Extractor = Callable[[str, datetime, list[str]], Extraction]


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    family: str | None
    due: str | None
    priority: int | None
    customer: str | None


class LabelledEmail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tags: list[str]
    text: str
    expect: Expect


@dataclass
class Dataset:
    now: datetime
    families: list[str]
    emails: list[LabelledEmail]


def load_dataset(path: Path) -> Dataset:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(data) != {"now", "families", "emails"}:
        raise ValueError(f"{path.name}: top level must be exactly now, families, emails")
    emails = [LabelledEmail.model_validate(e) for e in data["emails"]]
    ids = [e.id for e in emails]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate email ids in {path.name}")
    return Dataset(datetime.strptime(data["now"], "%Y-%m-%d %H:%M"), list(data["families"]), emails)


def _same(field_name: str, got, want) -> bool:
    if got is None or want is None:
        return got is want
    if field_name in ("family", "customer"):
        return str(got).strip().lower() == str(want).strip().lower()
    return got == want


@dataclass
class EmailResult:
    id: str
    tags: list[str]
    verdict: Verdict | None
    error: str | None
    right: dict[str, bool] = field(default_factory=dict)
    invented: list[str] = field(default_factory=list)
    raw_ungrounded: int = 0
    status_right: bool = False

    @property
    def exact(self) -> bool:
        return self.status_right and all(self.right.values()) and not self.invented


def run_extractor(extractor: Extractor, data: Dataset) -> list[EmailResult]:
    results = []
    for e in data.emails:
        try:
            raw = extractor(e.text, data.now, data.families)
        except Exception as exc:   # one bad email costs one result, not the whole (possibly paid) run
            results.append(EmailResult(e.id, e.tags, None, f"{type(exc).__name__}: {exc}"))
            continue
        verdict = validate(raw, e.text, data.families, data.now)
        text = normalise(e.text)
        ungrounded = sum(1 for f in FIELDS if getattr(raw, f) is not None and normalise(raw.evidence.get(f, "")) not in text)
        got = verdict.order.model_dump()
        want = e.expect.model_dump()
        right = {f: _same(f, got[f], want[f]) for f in FIELDS}
        results.append(EmailResult(
            e.id, e.tags, verdict, None, right, [f for f in FIELDS if want[f] is None and got[f] is not None],
            ungrounded, verdict.status == e.expect.status,
        ))
    return results


@dataclass(frozen=True)
class Scores:
    emails: int
    exact: float
    field_accuracy: float            # over fields the email states
    invented: int                    # values filled where the email states nothing
    abstained: int                   # needs_review emails correctly sent to a person
    needs_review: int
    false_alarms: int                # ok emails wrongly sent to a person
    ok_emails: int
    raw_ungrounded: int
    errors: int


def score(results: list[EmailResult], data: Dataset) -> Scores:
    gold = {e.id: e.expect for e in data.emails}
    stated = [(r, f) for r in results if r.verdict for f in FIELDS if getattr(gold[r.id], f) is not None]
    needs = [r for r in results if gold[r.id].status == "needs_review"]
    clean = [r for r in results if gold[r.id].status == "ok"]
    return Scores(
        emails=len(results),
        exact=sum(r.exact for r in results) / len(results) if results else 0.0,
        field_accuracy=sum(r.right[f] for r, f in stated) / len(stated) if stated else 0.0,
        invented=sum(len(r.invented) for r in results),
        abstained=sum(1 for r in needs if r.verdict and r.verdict.status == "needs_review"),
        needs_review=len(needs),
        false_alarms=sum(1 for r in clean if r.verdict and r.verdict.status == "needs_review"),
        ok_emails=len(clean),
        raw_ungrounded=sum(r.raw_ungrounded for r in results),
        errors=sum(1 for r in results if r.error),
    )


def render(name: str, results: list[EmailResult], data: Dataset) -> str:
    s = score(results, data)
    lines = [
        f"{name}: {s.emails} emails",
        f"  exact (all fields and the verdict right)      {s.exact:.2f}",
        f"  field accuracy (fields the email states)      {s.field_accuracy:.2f}",
        f"  invented values (email states nothing)        {s.invented}",
        f"  sent to a person when it should be            {s.abstained} of {s.needs_review}",
        f"  false alarms on clean emails                  {s.false_alarms} of {s.ok_emails}",
        f"  quotes not found in the email, before checks  {s.raw_ungrounded}",
    ]
    if s.errors:
        lines.append(f"  extractor errors                              {s.errors}")
    by_tag: dict[str, list[EmailResult]] = defaultdict(list)
    for r in results:
        for t in r.tags:
            by_tag[t].append(r)
    missed = [r for r in results if not r.exact]
    if missed:
        lines += ["", "  not exactly right:"]
        for r in missed:
            why = r.error or "; ".join(
                [f"{f} wrong" for f, ok in r.right.items() if not ok] + [f"invented {f}" for f in r.invented]
                + ([] if r.status_right else [f"verdict {r.verdict.status if r.verdict else '?'}"]))
            lines.append(f"    {r.id} [{', '.join(r.tags)}]: {why}")
    return "\n".join(lines)


def render_by_tag(results: list[EmailResult]) -> str:
    tags = sorted({t for r in results for t in r.tags})
    width = max(len(t) for t in tags)
    return "\n".join(
        f"  {t:<{width}}  {sum(r.exact for r in results if t in r.tags)} of {sum(1 for r in results if t in r.tags)} exact"
        for t in tags
    )
