"""A deterministic stand-in for an embedding model, so no test needs a download or a network.

Words are mapped to a few "concepts"; words in the same concept land on the same axis, which is the property a
real embedding model has and a keyword index lacks ("motor" and "bearing" mean the same thing here). Everything
else is spread over a few filler axes so unrelated texts are not identical.
"""

from __future__ import annotations

import numpy as np

from jobshop.knowledge.base import tokens
from jobshop.knowledge.semantic import normalise

_GROUPS = [
    ["bearing", "motor", "seize", "seizing", "fail", "failure", "broken"],   # machine trouble
    ["overtime", "extra", "hours", "night"],                                   # working beyond the shift
    ["rush", "faster", "quick", "urgent"],                                     # speed
]
CONCEPTS = {t: i for i, group in enumerate(_GROUPS) for word in group for t in tokens(word)}
DIMS = len(_GROUPS) + 3


class FakeEmbedder:
    def __init__(self, name: str = "fake-concepts") -> None:
        self.name = name
        self.passage_calls = 0   # how many times passages were embedded (what the cache is meant to save)
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        v = np.zeros(DIMS)
        for t in tokens(text):
            if t in CONCEPTS:
                v[CONCEPTS[t]] += 1.0
            else:
                v[len(_GROUPS) + sum(map(ord, t)) % 3] += 0.2
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_calls += 1
        return normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return normalise(self._vector(text))
