"""Does retrieval find the right document? Measured separately from the agent, with no model and no cost.

An answer built from the wrong passage is wrong however well it is written, so retrieval is tested on its
own first. Each question names the document(s) that answer it. A search *hits* if one of them is among the
top ``k`` results; the reciprocal rank rewards finding it earlier (1 for first place, 1/2 for second...).

    hit@k  = share of questions with a correct document in the top k
    MRR    = mean of 1 / (rank of the first correct document), 0 if it is not returned

Some questions are marked ``hard``: they ask for the same thing in words the document does not use
("the motor is seizing" for a document about a "bearing failure"). A keyword index cannot be expected to
find those, and the point of keeping them in is to see exactly how much a smarter retriever would add.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jobshop.knowledge import KnowledgeBase


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    sources: list[str] = Field(min_length=1)   # any of these documents answers it
    hard: bool = False                         # worded unlike the document: a stress test, not a target


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(data) != {"questions"}:
        raise ValueError(f"{path.name}: top level must be exactly 'questions: [...]'")
    questions = [Question.model_validate(q) for q in data["questions"]]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids in {path.name}")
    return questions


@dataclass(frozen=True)
class Outcome:
    question: Question
    rank: int | None            # 1-based position of the first correct document among the results, if any
    returned: list[str]         # the documents returned, best first (one entry per passage)


@dataclass(frozen=True)
class Scores:
    questions: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    k: int


def evaluate(kb: KnowledgeBase, questions: list[Question], k: int = 3) -> list[Outcome]:
    unknown = sorted({s for q in questions for s in q.sources} - set(kb.sources))
    if unknown:
        raise ValueError(f"questions name documents that do not exist: {unknown}")
    outcomes = []
    for q in questions:
        returned = [h.chunk.source for h in kb.search(q.question, k)]
        rank = next((i for i, source in enumerate(returned, 1) if source in q.sources), None)
        outcomes.append(Outcome(q, rank, returned))
    return outcomes


def score(outcomes: list[Outcome], k: int = 3) -> Scores:
    n = len(outcomes)
    if n == 0:
        return Scores(0, 0.0, 0.0, 0.0, k)
    return Scores(
        n,
        sum(1 for o in outcomes if o.rank == 1) / n,
        sum(1 for o in outcomes if o.rank is not None) / n,
        sum(1 / o.rank for o in outcomes if o.rank is not None) / n,
        k,
    )


def render(outcomes: list[Outcome], k: int = 3) -> str:
    groups = [("all", outcomes), ("ordinary", [o for o in outcomes if not o.question.hard]),
              ("hard (worded unlike the documents)", [o for o in outcomes if o.question.hard])]
    lines = [f"{'questions':<38} {'n':>3}  {'hit@1':>6}  {'hit@' + str(k):>6}  {'MRR':>6}", "-" * 66]
    for name, group in groups:
        s = score(group, k)
        lines.append(f"{name:<38} {s.questions:>3}  {s.hit_at_1:>6.2f}  {s.hit_at_k:>6.2f}  {s.mrr:>6.2f}")
    missed = [o for o in outcomes if o.rank is None]
    if missed:
        lines += ["", f"Not found in the top {k}:"]
        for o in missed:
            got = ", ".join(dict.fromkeys(o.returned)) or "nothing"
            lines.append(f"  {o.question.id}{' (hard)' if o.question.hard else ''}: {o.question.question}\n"
                         f"      wanted {', '.join(o.question.sources)}; got {got}")
    late = [o for o in outcomes if o.rank and o.rank > 1]
    if late:
        lines += ["", "Found, but not first:"]
        lines += [f"  {o.question.id}: rank {o.rank}  {o.question.question}" for o in late]
    return "\n".join(lines)
