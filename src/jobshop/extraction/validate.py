"""Checking a proposed extraction against the email it came from. Pure code, no model.

The point is to catch what a language model gets wrong without noticing: a value it made up, a family it
rounded to the nearest known one, a date it resolved wrongly, a priority it took from an instruction hidden in
the email. Each check is simple and explainable:

* **grounded**   the quoted evidence really occurs in the email (so a value with no source is dropped);
* **supporting** the quote actually says the value: it names the family or customer, contains the priority digit,
                 and a due time contains a clock time (a model can quote real words that prove nothing);
* **known**      the family is one the plant makes;
* **well formed** the due time is 'YYYY-MM-DD HH:MM' and a real date;
* **consistent** if the quote itself contains a date or a clock time, the resolved value agrees with it;
* **sane**       a due time in the past, or more than a year away, is flagged (kept, but a person must look).

Rejected values are removed from the order, so a bad value can never be mistaken for a fact downstream.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from jobshop.extraction.schema import FIELDS, Extraction, ValidatedOrder, Verdict

REQUIRED = ("family", "due")
MAX_DAYS_AHEAD = 366

_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b", re.I)
_PLANT_TIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


def normalise(text: str) -> str:
    """Lowercase with all runs of whitespace (including line breaks) collapsed, for comparing quotes with the email."""
    return " ".join(text.lower().split())


def explicit_parts(quote: str) -> tuple[str | None, str | None]:
    """A date ('YYYY-MM-DD') and/or a 24-hour clock time ('HH:MM') written out in a quote, if any."""
    date = _DATE.search(quote)
    time = None
    if m := _AMPM.search(quote):                      # first: "7:30 p.m." must not be read as 07:30
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        time = f"{hour:02d}:{m.group(2) or '00'}"
    elif clock := _CLOCK.search(quote):
        time = f"{int(clock.group(1)):02d}:{clock.group(2)}"
    return (date.group(0) if date else None), time


def validate(extraction: Extraction, email: str, known_families: list[str], now: datetime) -> Verdict:
    text = normalise(email)
    values = {f: getattr(extraction, f) for f in FIELDS}
    order = ValidatedOrder()
    problems: list[str] = []

    for field, value in values.items():
        if value is None:
            continue
        quote = extraction.evidence.get(field, "")
        if not quote.strip() or normalise(quote) not in text:
            problems.append(f"{field}: no matching words in the email, so the value was dropped")
            continue
        reason = _unsupported(field, value, quote)
        if reason:
            problems.append(f"{field}: {reason}")
            continue
        if field == "family":
            match = next((f for f in known_families if f.lower() == str(value).strip().lower()), None)
            if match is None:
                problems.append(f"family: '{value}' is not one of {', '.join(known_families)}")
                continue
            value = match
        elif field == "due":
            error = _check_due(str(value), quote, now)
            if error:
                problems.append(f"due: {error}")
                if error.startswith(("in the past", "more than")):   # flagged for a person, but the value is kept
                    order.due = str(value)
                continue
            value = str(value)
        elif field == "customer":
            value = str(value).strip()
        setattr(order, field, value)

    for field in REQUIRED:
        if getattr(order, field) is None and not any(p.startswith(f"{field}:") for p in problems):
            problems.append(f"{field}: the email does not give it")
    ok = order.family is not None and order.due is not None and not problems
    return Verdict(status="ok" if ok else "needs_review", order=order, problems=problems)


def _unsupported(field: str, value: object, quote: str) -> str | None:
    """Why the quote does not back the value, or None. Cheap on purpose: it catches quotes that are real but irrelevant."""
    q = normalise(quote)
    if field == "family" and normalise(str(value)) not in q:
        return f"the quote does not mention {value}, so the value was dropped"
    if field == "customer" and normalise(str(value)) not in q:
        return f"the quote does not mention {value}, so the value was dropped"
    if field == "priority" and not re.search(rf"(?<!\d){value}(?!\d)", q):
        return f"the quote does not contain {value}, so the value was dropped"
    if field == "due" and explicit_parts(quote)[1] is None:
        return "the quote contains no time of day, so the value was dropped"
    return None


def _check_due(value: str, quote: str, now: datetime) -> str | None:
    """A reason the due value is unusable or suspect, or None. Messages starting 'in the past' / 'more than' keep the value."""
    if not _PLANT_TIME.match(value):
        return "not in the form YYYY-MM-DD HH:MM (a day and a time are both needed)"
    try:
        moment = datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return "not a real date"
    date, clock = explicit_parts(quote)
    if date is not None and date != value[:10]:
        return f"the quote says {date} but the value is {value[:10]}"
    if clock is not None and clock != value[11:]:
        return f"the quote says {clock} but the value is {value[11:]}"
    if moment < now:
        return "in the past, so check the email"
    if moment > now + timedelta(days=MAX_DAYS_AHEAD):
        return "more than a year ahead, so check the email"
    return None
