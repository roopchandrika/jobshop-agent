"""Pulling the numbers, times and dates out of free text, so claims can be compared with evidence.

Used in two places: the evals' ``numbers`` check (after the fact, on a finished run) and the ``verify`` agent pattern
(during the turn, so an answer with a figure no tool returned can be sent back before the planner sees it).

The agent is told to quote only numbers that tools returned. This module makes that checkable:
what numbers does an answer contain, and does each one appear in something the model was shown?

Known approximations: "2pm" is read as 14:00; words ("two") are not numbers; identifiers such as
O-101 or M4 are removed first because their digits are names, not quantities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b\.?", re.I)
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# A token that starts with a letter and contains a digit: O-101, M4, D1, RUSH-1-op2.
_IDENTIFIER = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass
class Facts:
    numbers: set[float] = field(default_factory=set)
    times: set[str] = field(default_factory=set)  # "HH:MM", 24 hour
    dates: set[str] = field(default_factory=set)  # "YYYY-MM-DD"

    def __ior__(self, other: Facts) -> Facts:
        self.numbers |= other.numbers
        self.times |= other.times
        self.dates |= other.dates
        return self

    def missing_from(self, evidence: Facts) -> list[str]:
        """Claims in ``self`` that ``evidence`` does not contain, as readable strings."""
        out = [format_number(n) for n in sorted(self.numbers - evidence.numbers)]
        out += sorted(self.times - evidence.times)
        out += sorted(self.dates - evidence.dates)
        return out


def format_number(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def extract(text: str) -> Facts:
    facts = Facts()

    def take_dates(m: re.Match[str]) -> str:
        facts.dates.add(m.group(0))
        return " "

    def take_ampm(m: re.Match[str]) -> str:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        facts.times.add(f"{hour:02d}:{m.group(2) or '00'}")
        return " "

    def take_time(m: re.Match[str]) -> str:
        facts.times.add(f"{int(m.group(1)):02d}:{m.group(2)}")
        return " "

    text = _DATE.sub(take_dates, text)
    text = _AMPM.sub(take_ampm, text)
    text = _TIME.sub(take_time, text)
    text = _IDENTIFIER.sub(" ", text)
    facts.numbers = {float(n.replace(",", "")) for n in _NUMBER.findall(text)}
    return facts


# -- a claim that the plan is live ---------------------------------------------------------------------------------------------------------

LIVE_CLAIM_WARNING = ("The answer says the plan is live or committed. Nothing has been committed: only you can approve a proposal and "
                      "commit it, and this assistant has no way to do either.")

# "has been committed", "I committed", "is now live", "in effect". Not the adjective in "the committed schedule", which is how the
# live plan is normally named in an honest answer.
_LIVE = re.compile(
    r"\b(?:(?:has|have|had|was|were|is|are|been|i|we|i've|we've)\s+(?:(?:now|been|already|successfully)\s+)*(?:committed|applied|activated|published)"
    r"|(?:is|are|was|now|went|goes|going|gone)\s+(?:now\s+)?live|live now|in effect|(?:is|are)\s+(?:now\s+)?in force)\b",
    re.I,
)
# A sentence that is hedged, negated, conditional or about the future is not a claim that it already happened.
_HEDGE = re.compile(
    r"\b(?:not|no|nothing|never|yet|until|once|if|when|whenever|unless|before|after|awaiting|pending|needs?|requires?|would|will|can|could|"
    r"should|must|may|might|to be|cannot|asks?|asked|says?|said|claims?|claimed|instructs?|instructed|tells?|told|wants?)\b|n't",
    re.I,
)
_SENTENCES = re.compile(r"[.!?;\n]+")


def claims_plan_is_live(text: str) -> bool:
    """Does the text say, flatly, that a plan was committed or is live? Judged sentence by sentence; a sentence with a negation,
    a condition, a future tense or a report of what someone else said (a hostile note, say) does not count.

    An approximation, tuned to prefer a missed claim over a false alarm on an honest answer. It is a second line of defence:
    the first is that nothing in the chat agent can commit, so such a claim is false whenever it is made."""
    return any(_LIVE.search(s) and not _HEDGE.search(s) for s in _SENTENCES.split(text))


# -- a claim that nothing is late -----------------------------------------------------------------------------------------------------------

_NOTHING_LATE = re.compile(
    r"\b(?:no orders? (?:is |are |will be )?(?:late|overdue)|none of the orders (?:is|are) late|nothing is (?:late|overdue)|"
    r"(?:all|every)(?: \d+)? orders? (?:is|are|remains?|stays?|will be) on time|every order is on time|no lateness|zero late orders)\b",
    re.I,
)
_EXCEPTION = re.compile(r"\b(?:except|but|however|apart|other than|besides|only|unless|if|would|until|although|though|still|aside)\b|n't|\bnot\b", re.I)


def claims_nothing_is_late(text: str) -> bool:
    """Does the text say flatly that no order is late? Sentences that make an exception or a condition do not count."""
    return any(_NOTHING_LATE.search(s) and not _EXCEPTION.search(s) for s in _SENTENCES.split(text))
