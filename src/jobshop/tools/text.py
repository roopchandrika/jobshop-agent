"""Cleaning free text before it reaches a model or a human's terminal.

This is hygiene, not an injection defense. Removing control characters stops text from redrawing
a terminal (an ANSI "clear screen" could fake the KPI block above an approval prompt) and stops
invisible or direction-flipping characters from hiding what a note says. A sentence like
"ignore your instructions" passes through untouched, because that is a question about what the
model obeys, not about characters. The guarantees against that live in the architecture: the
model has no commit tool, and the approval screen is built from stored data.
"""

from __future__ import annotations

import re
import unicodedata

# OSC (title/hyperlink) and CSI (cursor, colour, clear) sequences, plus two-character escapes.
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
_LINE_BREAKS = re.compile(r"\r\n|[\r  \x85\x0b\x0c]")
# Control (Cc), format (Cf: zero-width, bidi overrides, BOM), private-use, surrogate, unassigned.
_DROPPED_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn"}

NOTE_MAX_CHARS = 500


def _clean(text: str, *, keep_newlines: bool) -> str:
    text = _LINE_BREAKS.sub("\n", _ANSI.sub("", text))
    kept = []
    for ch in text:
        if ch == "\n":
            kept.append("\n" if keep_newlines else " ")
        elif ch == "\t":
            kept.append(" ")
        elif unicodedata.category(ch) not in _DROPPED_CATEGORIES:
            kept.append(ch)
    return "".join(kept)


def untrusted_text(text: str, max_chars: int = NOTE_MAX_CHARS) -> str:
    """Free text for a model: one line (so it cannot fake a new section), bounded, no control characters."""
    single_line = " ".join(_clean(text, keep_newlines=False).split())
    if len(single_line) > max_chars:
        return single_line[:max_chars].rstrip() + " [truncated]"
    return single_line


def terminal_safe(text: str) -> str:
    """Text for a human's terminal: line breaks stay, anything that could move the cursor or restyle goes."""
    return _clean(text, keep_newlines=True)
