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
