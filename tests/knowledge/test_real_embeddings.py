"""The real embedding model. Opt-in: it needs the optional extra and downloads a model (about 65 MB) the first time.

    uv sync --extra embeddings
    JOBSHOP_RUN_EMBEDDINGS=1 uv run pytest tests/knowledge/test_real_embeddings.py
"""

import os
from pathlib import Path

import pytest

from jobshop.evals import retrieval
from jobshop.knowledge import build_retriever

pytestmark = pytest.mark.skipif(not os.environ.get("JOBSHOP_RUN_EMBEDDINGS"), reason="set JOBSHOP_RUN_EMBEDDINGS=1 to run (downloads a model)")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def results():
    pytest.importorskip("fastembed")
    questions = retrieval.load_questions(ROOT / "evals" / "retrieval.yaml")
    off_topic = retrieval.load_unanswerable(ROOT / "evals" / "retrieval.yaml")
    out = {}
    for method in ("bm25", "dense", "hybrid"):
        kb = build_retriever(method, ROOT / "knowledge", cache_dir=ROOT / ".cache")
        out[method] = (retrieval.evaluate(kb, questions, 3), retrieval.measure_separation(kb, questions, off_topic))
    return out


def scores(results, method, hard):
    return retrieval.score([o for o in results[method][0] if o.question.hard == hard], 3)


def test_meaning_search_finds_the_paraphrases_that_keyword_search_misses(results):
    assert scores(results, "bm25", hard=True).hit_at_k <= 0.5
    assert scores(results, "dense", hard=True).hit_at_k >= 5 / 6


def test_meaning_search_does_not_give_up_ordinary_questions_or_exact_tokens(results):
    assert scores(results, "dense", hard=False).hit_at_k >= 0.95


def test_vector_search_returns_passages_even_for_questions_the_documents_do_not_cover(results):
    assert results["dense"][1].off_topic_returned == results["dense"][1].off_topic
    assert results["bm25"][1].off_topic_returned < results["bm25"][1].off_topic
