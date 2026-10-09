"""Search by meaning: embeddings, a vector index, and a hybrid of keyword and vector search.

The keyword search (``base.py``) finds passages that share *words* with a question. An embedding model
turns text into a list of numbers (a vector) so that texts with similar *meaning* get similar vectors,
even with no words in common: "the motor is seizing" lands near "bearing failure". Searching is then
just "which passage vectors point in nearly the same direction as the question's vector?", measured by
cosine similarity. With vectors scaled to length 1 that is a plain dot product.

Three pieces:

* ``Embedder``      anything with ``embed_passages`` / ``embed_query``. ``FastEmbedder`` runs a small model
                    locally (optional dependency); tests use a fake one, so nothing here needs a network.
* ``DenseKnowledgeBase``   passage vectors (cached on disk) + cosine search.
* ``HybridKnowledgeBase``  keyword and vector rankings merged with reciprocal rank fusion, because each
                    finds things the other misses: exact identifiers ("4100", "M3") favour keywords, paraphrases
                    favour vectors.

Scores differ in meaning between methods (BM25 score, cosine, fused rank score) and must only be compared
within one method.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from jobshop.knowledge.base import Chunk, Hit, KnowledgeBase

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# The model's documented prefix for search questions (passages get none). Using it is the model authors'
# recommendation, not something tuned on our questions.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60  # the usual damping constant of reciprocal rank fusion


class Embedder(Protocol):
    name: str  # identifies the model; part of the cache key, so a different model never reuses old vectors

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Shape (n, d), rows scaled to length 1."""

    def embed_query(self, text: str) -> np.ndarray:
        """Shape (d,), scaled to length 1."""


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedder:
    """A small embedding model run locally through the optional ``fastembed`` package (ONNX, no GPU, no API).

    The model file (about 70 to 130 MB) is downloaded from Hugging Face the first time it is used and kept in
    ``model_dir``, so later runs are offline.
    """

    def __init__(self, model: str = DEFAULT_MODEL, model_dir: Path | None = None) -> None:
        self.name = model
        self._model_dir = model_dir
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "semantic search needs the optional 'embeddings' extra: run  uv sync --extra embeddings"
                ) from None
            self._model = TextEmbedding(model_name=self.name, cache_dir=str(self._model_dir) if self._model_dir else None)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return normalise(np.array(list(self._load().embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = BGE_QUERY_PREFIX if "bge" in self.name.lower() else ""
        return normalise(np.array(list(self._load().embed([prefix + text]))[0]))


def passage_text(chunk: Chunk) -> str:
    """What is embedded for a passage: its heading and its text (the same view the keyword index has)."""
    return f"{chunk.heading}\n{chunk.text}"


class DenseKnowledgeBase:
    """Vector search over the same passages as ``KnowledgeBase``."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache_dir: Path | None = None) -> None:
        if not chunks:
            raise ValueError("a knowledge base needs at least one passage")
        self.chunks = chunks
        self._embedder = embedder
        self._vectors = self._passage_vectors(cache_dir)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def _passage_vectors(self, cache_dir: Path | None) -> np.ndarray:
        texts = [passage_text(c) for c in self.chunks]
        if cache_dir is None:
            return self._embedder.embed_passages(texts)
        # Keyed by the model and every passage, so editing a document or changing the model re-embeds.
        key = hashlib.sha256("\x00".join([self._embedder.name, *texts]).encode()).hexdigest()[:24]
        path = Path(cache_dir) / f"{key}.npy"
        if path.exists():
            cached = np.load(path)
            if cached.shape[0] == len(texts):
                return cached
        vectors = self._embedder.embed_passages(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """The ``k`` passages whose meaning is closest to ``query``. Cosine similarity, best first.

        Unlike keyword search, something always comes back (every passage has *some* similarity), so a
        question the documents do not cover still gets its nearest neighbours. Callers must not read
        "returned" as "relevant"; that is why the agent is told to check the passage actually answers.
        """
        if not query.strip():
            return []
        scores = self._vectors @ self._embedder.embed_query(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]   # ties: document order
        return [Hit(self.chunks[i], round(float(scores[i]), 3)) for i in order]


class HybridKnowledgeBase:
    """Keyword and vector rankings merged by reciprocal rank fusion.

    A passage's fused score is the sum over the two rankings of ``1 / (RRF_K + rank)``. Only ranks are used,
    so the two incomparable score scales never have to be reconciled; a passage both methods like beats one
    that only one method likes.
    """

    def __init__(self, keyword: KnowledgeBase, dense: DenseKnowledgeBase, depth: int = 10) -> None:
        if keyword.chunks != dense.chunks:
            raise ValueError("both indexes must be built over the same passages")
        self.chunks = keyword.chunks
        self._keyword, self._dense, self._depth = keyword, dense, depth

    @property
    def sources(self) -> list[str]:
        return self._keyword.sources

    def search(self, query: str, k: int = 3) -> list[Hit]:
        depth = max(self._depth, k)
        fused: dict[int, float] = {}
        index = {c: i for i, c in enumerate(self.chunks)}   # passages are frozen, so they compare and hash by value
        for ranking in (self._keyword.search(query, depth), self._dense.search(query, depth)):
            for rank, hit in enumerate(ranking, 1):
                i = index[hit.chunk]
                fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [Hit(self.chunks[i], round(fused[i], 4)) for i in order]
