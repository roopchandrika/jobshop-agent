"""Reading orders out of emails (Phase 10): structured extraction with checks and an evaluation.

Nothing in here touches the schedule. An extraction is a proposal for a person; see ``schema`` for why it is
checked and ``validate`` for how.
"""

from jobshop.extraction.schema import Extraction, ValidatedOrder, Verdict
from jobshop.extraction.validate import validate

__all__ = ["Extraction", "ValidatedOrder", "Verdict", "validate"]
