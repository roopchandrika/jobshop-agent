"""A rule-based extractor: patterns, no model. The floor an LLM extractor has to beat.

It exists for two reasons. It runs offline and for free, so the extraction pipeline and its evaluation are
testable without an API key. And it gives a real number to compare a language model against: if a model is only
as good as a handful of regular expressions, it is not earning its cost.

It is deliberately simple, and fails in the ways such extractors do (it takes the first date it sees, the first
family word, and any "priority 5" even one planted as an instruction); the evaluation labels those cases.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from jobshop.extraction.schema import Extraction

_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})\b")
_DAY_TIME = re.compile(r"\b(today|tomorrow)\b[^.\n]{0,12}?\b(\d{1,2}):(\d{2})\b", re.I)
_DAY_AMPM = re.compile(r"\b(today|tomorrow)\b[^.\n]{0,12}?\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b", re.I)
_BY_CLOCK = re.compile(r"\b(?:by|before|due)\s+(\d{1,2}):(\d{2})\b(?:\s+(today))?", re.I)
_PRIORITY = re.compile(r"\bpriority\s*(?:level\s*)?(?:is\s*)?([1-5])\b", re.I)
_CUSTOMER = re.compile(r"\bcustomer:\s*([^\n.]+)", re.I)


def extract_baseline(email: str, now: datetime, families: list[str]) -> Extraction:
    ev: dict[str, str] = {}
    family = None
    positions = [(m.start(), f, m.group(0)) for f in families if (m := re.search(rf"\b{re.escape(f)}s?\b", email, re.I))]
    if positions:
        _, family, quote = min(positions)           # the family word that appears first
        ev["family"] = quote

    due = None
    if m := _ISO.search(email):
        due, ev["due"] = f"{m.group(1)} {int(m.group(2)):02d}:{m.group(3)}", m.group(0)
    elif m := _DAY_TIME.search(email):
        due, ev["due"] = _day(now, m.group(1), int(m.group(2)), m.group(3)), m.group(0)
    elif m := _DAY_AMPM.search(email):
        hour = int(m.group(2)) % 12 + (12 if m.group(4).lower() == "p" else 0)
        due, ev["due"] = _day(now, m.group(1), hour, m.group(3) or "00"), m.group(0)
    elif m := _BY_CLOCK.search(email):
        due, ev["due"] = _day(now, "today", int(m.group(1)), m.group(2)), m.group(0)

    priority = None
    if m := _PRIORITY.search(email):
        priority, ev["priority"] = int(m.group(1)), m.group(0)

    customer = None
    if m := _CUSTOMER.search(email):
        customer, ev["customer"] = m.group(1).strip(), m.group(0)
        ev["customer"] = ev["customer"].strip()

    return Extraction(family=family, due=due, priority=priority, customer=customer, evidence=ev)


def _day(now: datetime, word: str, hour: int, minute: str) -> str:
    day = now.date() + timedelta(days=1 if word.lower() == "tomorrow" else 0)
    return f"{day:%Y-%m-%d} {hour:02d}:{minute}"
