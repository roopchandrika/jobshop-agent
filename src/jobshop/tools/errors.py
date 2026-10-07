class ToolError(Exception):
    """An expected failure (bad id, stale draft, rejected approval, ...).

    The message is written for the model to read and act on, so it must say what was wrong
    and, where possible, what to do instead. Unexpected exceptions are NOT ToolErrors.
    """
