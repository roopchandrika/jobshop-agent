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

from jobshop.knowledge import Retriever


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    sources: list[str] = Field(min_length=1)   # any of these documents answers it
    hard: bool = False                         # worded unlike the document: a stress test, not a target


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "questions" not in data or set(data) - {"questions", "unanswerable"}:
        raise ValueError(f"{path.name}: top level must be 'questions: [...]' (and optionally 'unanswerable: [...]')")
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


def evaluate(kb: Retriever, questions: list[Question], k: int = 3) -> list[Outcome]:
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


def render_comparison(results: dict[str, list[Outcome]], k: int = 3) -> str:
    """Several retrieval methods on the same questions: one row each, then the questions they disagree on."""
    names = list(results)
    first = results[names[0]]
    if any([o.question.id for o in outcomes] != [o.question.id for o in first] for outcomes in results.values()):
        raise ValueError("every method must be scored on the same questions")

    def cells(outcomes: list[Outcome]) -> str:
        s = score(outcomes, k)
        return f"{s.hit_at_1:>6.2f}  {s.hit_at_k:>6.2f}  {s.mrr:>6.2f}"

    head = f"{'':<10} {'ordinary questions':^24}   {'hard questions':^24}"
    sub = f"{'method':<10} {'hit@1':>6}  {'hit@' + str(k):>6}  {'MRR':>6}   {'hit@1':>6}  {'hit@' + str(k):>6}  {'MRR':>6}"
    lines = [head, sub, "-" * len(sub)]
    for name, outcomes in results.items():
        ordinary = [o for o in outcomes if not o.question.hard]
        hard = [o for o in outcomes if o.question.hard]
        lines.append(f"{name:<10} {cells(ordinary)}   {cells(hard)}")
    n_ord = sum(1 for o in first if not o.question.hard)
    lines.append(f"({n_ord} ordinary and {len(first) - n_ord} hard questions)")

    differing = [i for i in range(len(first)) if len({results[m][i].rank is not None for m in names}) > 1]
    if differing:
        lines += ["", "Questions where the methods disagree about finding the right document:"]
        for i in differing:
            q = first[i].question
            found = [m for m in names if results[m][i].rank is not None]
            lines.append(f"  {q.id}{' (hard)' if q.hard else ''}: found by {', '.join(found) or 'nobody'}   {q.question}")
    return "\n".join(lines)


# -- questions the documents do not cover -----------------------------------------------------------------------------------------


def load_unanswerable(path: Path) -> list[str]:
    """Off-topic questions from the same file (optional ``unanswerable:`` list); nothing in the documents answers them."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [str(q) for q in data.get("unanswerable", [])]


@dataclass(frozen=True)
class Separation:
    """Can a score tell a real answer from the nearest wrong passage?

    Keyword search returns nothing when no word matches. Vector search returns the closest passages whatever the
    question, so the only protection against answering from an irrelevant passage is the score (and the agent
    reading the passage). ``margin`` > 0 means every off-topic question scored below every answerable one.
    """

    answerable_low: float | None     # lowest best-score among questions that have an answer
    off_topic_high: float | None     # highest best-score among off-topic questions (None if nothing was returned)
    off_topic_returned: int          # how many off-topic questions got any passage back
    off_topic: int

    @property
    def margin(self) -> float | None:
        if self.answerable_low is None or self.off_topic_high is None:
            return None
        return round(self.answerable_low - self.off_topic_high, 3)


def measure_separation(kb: Retriever, questions: list[Question], off_topic: list[str]) -> Separation:
    top = lambda text: (kb.search(text, 1) or [None])[0]  # noqa: E731
    answerable = [h.score for h in map(top, (q.question for q in questions)) if h]
    off = [h.score for h in map(top, off_topic) if h]
    return Separation(min(answerable) if answerable else None, max(off) if off else None, len(off), len(off_topic))


def render_separation(results: dict[str, Separation]) -> str:
    lines = ["", "Questions the documents do not cover (best score returned; scores differ in meaning between methods):",
             f"  {'method':<8} {'off-topic with a result':>24}  {'highest off-topic':>18}  {'lowest answerable':>18}  {'margin':>7}"]
    for name, s in results.items():
        high = "-" if s.off_topic_high is None else f"{s.off_topic_high:.3f}"
        low = "-" if s.answerable_low is None else f"{s.answerable_low:.3f}"
        margin = "-" if s.margin is None else f"{s.margin:+.3f}"
        lines.append(f"  {name:<8} {f'{s.off_topic_returned} of {s.off_topic}':>24}  {high:>18}  {low:>18}  {margin:>7}")
    return "\n".join(lines)
