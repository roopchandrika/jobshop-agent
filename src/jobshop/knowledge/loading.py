"""Finding the plant documents and choosing how to search them. One rule for every front end."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from jobshop.knowledge.base import KnowledgeBase, Retriever
from jobshop.knowledge.semantic import DEFAULT_MODEL, DenseKnowledgeBase, Embedder, FastEmbedder, HybridKnowledgeBase

ENV_VAR = "JOBSHOP_KNOWLEDGE_DIR"      # folder of .md documents
RETRIEVER_VAR = "JOBSHOP_RETRIEVER"    # bm25 (default, no extra install) | dense | hybrid
MODEL_VAR = "JOBSHOP_EMBED_MODEL"      # embedding model for dense and hybrid
CACHE_VAR = "JOBSHOP_CACHE_DIR"        # where downloaded models and computed vectors are kept
RETRIEVERS = ("bm25", "dense", "hybrid")


def build_retriever(
    kind: str, directory: Path, *, embedder: Embedder | None = None, cache_dir: Path | None = None, model: str = DEFAULT_MODEL,
) -> Retriever:
    """The documents in ``directory`` searched by keyword, by meaning, or both.

    ``dense`` and ``hybrid`` need an embedding model: pass an ``embedder`` (tests do) or install the optional
    ``embeddings`` extra, in which case the model is downloaded on first use into ``cache_dir/models``.
    """
    if kind not in RETRIEVERS:
        raise ValueError(f"unknown retriever {kind!r}; choose one of {', '.join(RETRIEVERS)}")
    keyword = KnowledgeBase.from_directory(directory)
    if kind == "bm25":
        return keyword
    try:
        embedder = embedder or FastEmbedder(model, cache_dir / "models" if cache_dir else None)
        dense = DenseKnowledgeBase(keyword.chunks, embedder, cache_dir / "embeddings" if cache_dir else None)
        # Cached passage vectors mean the model may not have been needed yet; every search needs it. Load it now so a
        # missing package or a failed download is reported when the retriever is built, not on someone's first question.
        dense.search("ready", 1)
    except ImportError as e:   # the optional extra is not installed: a configuration problem, said plainly
        raise ValueError(str(e)) from None
    return dense if kind == "dense" else HybridKnowledgeBase(keyword, dense)


def load_knowledge(env: Mapping[str, str], default: str | None = "knowledge") -> Retriever | None:
    """The documents in ``$JOBSHOP_KNOWLEDGE_DIR`` (or ``default`` if the variable is unset), or None.

    ``off`` (or an empty value) switches them off. A folder named explicitly that does not exist is an
    error, so a typo is not mistaken for "no documents"; an unset variable with no default folder is not.
    ``$JOBSHOP_RETRIEVER`` picks the search method (keyword unless told otherwise).
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
    return build_retriever(
        env.get(RETRIEVER_VAR, "bm25").strip().lower(), path,
        cache_dir=Path(env.get(CACHE_VAR, ".cache")), model=env.get(MODEL_VAR, DEFAULT_MODEL),
    )
