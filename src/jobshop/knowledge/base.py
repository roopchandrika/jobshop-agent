"""A small keyword-search index over the plant documents (BM25), written out so it can be read.

BM25 scores a passage against a question by how many of the question's words it contains, counting
rare words (``spindle``) for more than common ones (``machine``) and not letting a long passage win
just by being long. It is lexical: it matches words, not meaning, so "what if the motor seizes"
will not find a passage that only says "bearing failure". That limit is real and is measured by the
retrieval evaluation (``python -m jobshop.evals retrieval``); semantic search with embeddings is the
next step and is deliberately not here yet.

    score(q, d) = sum over query words w of  idf(w) * tf*(k1+1) / (tf + k1*(1 - b + b*len(d)/avg_len))
    idf(w)      = ln(1 + (N - n_w + 0.5) / (n_w + 0.5))
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jobshop.knowledge.chunking import Chunk, chunk_markdown

K1 = 1.5  # how quickly repeating a word stops adding to the score
B = 0.75  # how strongly a long passage is discounted

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by can do does for from had has have how if in into is it its me my of on or our "
    "so than that the their then there these this to us was we what when where which who why will with would you "
    "your".split()
)


def tokens(text: str) -> list[str]:
    """Lowercase words with common words dropped and a crude plural/-ing/-ed fold (``bearings`` -> ``bearing``)."""
    out = []
    for word in _WORD.findall(text.lower()):
        if word in _STOP:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        out.append(word)
    return out


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("a knowledge base needs at least one passage")
        self.chunks = chunks
        # The heading is part of what a passage is about, so it is indexed with the text.
        self._terms = [Counter(tokens(f"{c.heading} {c.text}")) for c in chunks]
        self._lengths = [sum(t.values()) for t in self._terms]
        self._avg_len = sum(self._lengths) / len(chunks)
        document_frequency: Counter[str] = Counter()
        for terms in self._terms:
            document_frequency.update(terms.keys())
        n = len(chunks)
        self._idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in document_frequency.items()}

    @classmethod
    def from_directory(cls, directory: Path) -> KnowledgeBase:
        """Every ``*.md`` file under ``directory``, in a stable order."""
        files = sorted(Path(directory).rglob("*.md"))
        if not files:
            raise ValueError(f"no .md documents found under {directory}")
        chunks: list[Chunk] = []
        for path in files:
            chunks += chunk_markdown(path.relative_to(directory).as_posix(), path.read_text(encoding="utf-8"))
        return cls(chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """The ``k`` best passages for ``query``, best first. Passages sharing no word with it are never returned."""
        words = tokens(query)
        scored = []
        for i, terms in enumerate(self._terms):
            score = 0.0
            for w in words:
                tf = terms.get(w, 0)
                if tf:
                    norm = tf + K1 * (1 - B + B * self._lengths[i] / self._avg_len)
                    score += self._idf[w] * tf * (K1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))   # ties: document order, so results are repeatable
        return [Hit(self.chunks[i], round(score, 3)) for score, i in scored[:k]]
