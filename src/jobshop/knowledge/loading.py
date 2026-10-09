"""Finding the plant documents. One rule for every front end."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from jobshop.knowledge.base import KnowledgeBase

ENV_VAR = "JOBSHOP_KNOWLEDGE_DIR"


def load_knowledge(env: Mapping[str, str], default: str | None = "knowledge") -> KnowledgeBase | None:
    """The documents in ``$JOBSHOP_KNOWLEDGE_DIR`` (or ``default`` if the variable is unset), or None.

    ``off`` (or an empty value) switches them off. A folder named explicitly that does not exist is an
    error, so a typo is not mistaken for "no documents"; an unset variable with no default folder is not.
    """
    value = env.get(ENV_VAR)
    if value is not None and value.strip().lower() in ("", "off", "none"):
        return None
    folder = value if value is not None else default
    if folder is None:
        return None
    path = Path(folder)
    if not path.is_dir():
        if value is not None:
            raise ValueError(f"{ENV_VAR}={folder!r} is not a folder")
        return None
    return KnowledgeBase.from_directory(path)
