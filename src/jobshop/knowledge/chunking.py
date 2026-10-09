"""Splitting a markdown document into passages a retriever can return.

Why chunk at all: a whole document is too long to hand to a model for every question, and one
sentence is too short to mean anything. The natural unit in these documents is a section under a
heading, so that is the chunk; a section that is still long is split at paragraph breaks so no single
passage dominates the prompt. Each chunk remembers its document and heading, which is what lets the
agent say where a statement came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_WORDS = 110

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    source: str   # the document's file name
    heading: str  # "Document title > Section title"
    text: str


def _split_long(paragraphs: list[str], max_words: int) -> list[str]:
    """Group paragraphs into passages of at most ``max_words`` (a single longer paragraph stays whole)."""
    passages: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and size + words > max_words:
            passages.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += words
    if current:
        passages.append("\n\n".join(current))
    return passages


def chunk_markdown(source: str, text: str, max_words: int = MAX_WORDS) -> list[Chunk]:
    """One chunk per heading section (long sections split by paragraph); empty sections are dropped."""
    title = ""
    heading = ""
    body: list[str] = []
    sections: list[tuple[str, list[str]]] = []

    def close() -> None:
        paragraphs = [p.strip() for p in "\n".join(body).split("\n\n") if p.strip()]
        if paragraphs:
            sections.append((heading or title or source, paragraphs))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            body.append(line)
            continue
        close()
        level, name = len(match.group(1)), match.group(2)
        if level == 1:
            title, heading = name, name
        else:
            heading = f"{title} > {name}" if title else name
    close()

    return [Chunk(source, h, passage) for h, paragraphs in sections for passage in _split_long(paragraphs, max_words)]
