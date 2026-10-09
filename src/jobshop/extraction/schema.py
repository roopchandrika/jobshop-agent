"""What an extractor produces, and what the system accepts from it.

An email is free text from outside the plant: untrusted, messy, sometimes hostile. An extractor (a rule-based one
or a language model) reads it and proposes a structured order. The proposal is *not* trusted either: it goes
through ``validate`` before a person sees it, and a person decides what to do with it. Nothing here touches the
schedule.

Every value the extractor gives must come with ``evidence``: the words in the email it was read from. That is
what lets the system check a model's claim against the source instead of taking its word.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FIELDS = ("family", "due", "priority", "customer")


class Extraction(BaseModel):
    """The raw proposal. Values are strings except priority; ``None`` means the email does not say."""

    model_config = ConfigDict(extra="forbid")

    family: str | None = Field(None, description="Product family named in the email, exactly one of the known families.")
    due: str | None = Field(None, description="Due time as 'YYYY-MM-DD HH:MM' (plant-local), only if the email gives both a day and a time.")
    priority: int | None = Field(None, ge=1, le=5, description="Only if the email states a priority number; 'urgent' or 'rush' alone is not a number.")
    customer: str | None = Field(None, description="Customer name, only if the email states it.")
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description="For every field that is not null: the exact words from the email it was read from (copied, not paraphrased).",
    )


class ValidatedOrder(BaseModel):
    """What survived validation. A field the checks rejected is ``None`` here, never a bad value."""

    model_config = ConfigDict(extra="forbid")

    family: str | None = None
    due: str | None = None
    priority: int | None = None
    customer: str | None = None


Status = Literal["ok", "needs_review"]


class Verdict(BaseModel):
    """The validated result. ``ok`` means every required field is present and checked; it does NOT mean a person
    can skip reading the email. ``needs_review`` always says why."""

    model_config = ConfigDict(extra="forbid")

    status: Status
    order: ValidatedOrder
    problems: list[str] = Field(default_factory=list)

    @property
    def suggested_request(self) -> str | None:
        """A sentence the planner could paste into the assistant, if the order is complete."""
        if self.status != "ok" or self.order.family is None or self.order.due is None:
            return None
        priority = f", priority {self.order.priority}" if self.order.priority is not None else ""
        return f"Add a rush {self.order.family} order due {self.order.due}{priority}."
