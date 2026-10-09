"""Vector search, its disk cache, the hybrid, the optional embedding model, and choosing a retriever."""

import sys

import numpy as np
import pytest

from jobshop.knowledge import Chunk, DenseKnowledgeBase, FastEmbedder, HybridKnowledgeBase, KnowledgeBase, build_retriever, load_knowledge
from jobshop.knowledge.semantic import BGE_QUERY_PREFIX, RRF_K, passage_text
from jobshop.tools.registry import ToolRegistry
from tests.helpers import build_ctx
from tests.knowledge.fakes import DIMS, FakeEmbedder

CHUNKS = [
    Chunk("machine.md", "Machine > Bearings", "Bearing failure: stop the machine and call maintenance."),
    Chunk("hours.md", "Hours > Overtime", "Overtime needs approval from the production manager."),
    Chunk("rush.md", "Rush > Policy", "A rush order needs the production manager to approve it."),
    Chunk("misc.md", "Misc > Parking", "The car park is behind the loading dock."),
]


# -- dense search ----------------------------------------------------------------------------------------------------


def test_a_question_with_no_words_in_common_still_finds_the_passage_that_means_the_same():
    question = "the motor keeps seizing up"                                  # shares no word with the bearing passage
    assert KnowledgeBase(CHUNKS).search(question) == []                       # a keyword index cannot find it
    [first, *_] = DenseKnowledgeBase(CHUNKS, FakeEmbedder()).search(question)
    assert first.chunk.source == "machine.md"


def test_scores_are_cosine_similarities_best_first_and_k_limits_the_results():
    hits = DenseKnowledgeBase(CHUNKS, FakeEmbedder()).search("bearing failure", 3)
    assert len(hits) == 3 and [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1.0 <= h.score <= 1.0 for h in hits) and hits[0].score > 0.9


def test_the_stored_vectors_are_unit_length_so_a_dot_product_is_a_cosine():
    dense = DenseKnowledgeBase(CHUNKS, FakeEmbedder())
    assert dense._vectors.shape == (len(CHUNKS), DIMS)
    assert np.allclose(np.linalg.norm(dense._vectors, axis=1), 1.0, atol=1e-5)


def test_identical_scores_keep_document_order():
    twins = [Chunk("1.md", "x", "same text"), Chunk("2.md", "x", "same text"), Chunk("3.md", "x", "same text")]
    assert [h.chunk.source for h in DenseKnowledgeBase(twins, FakeEmbedder()).search("same text", 3)] == ["1.md", "2.md", "3.md"]


def test_a_blank_question_returns_nothing_but_an_unrelated_one_still_returns_the_nearest_passages():
    dense = DenseKnowledgeBase(CHUNKS, FakeEmbedder())
    assert dense.search("") == [] and dense.search("   ") == []
    unrelated = dense.search("what is the capital of france", 2)
    assert len(unrelated) == 2          # vector search always has a nearest neighbour; "returned" does not mean "relevant"


def test_what_is_embedded_is_the_heading_and_the_text():
    assert passage_text(CHUNKS[0]) == "Machine > Bearings\nBearing failure: stop the machine and call maintenance."


def test_an_empty_dense_index_is_refused():
    with pytest.raises(ValueError, match="at least one passage"):
        DenseKnowledgeBase([], FakeEmbedder())


# -- the cache of passage vectors --------------------------------------------------------------------------------------------


def test_passage_vectors_are_computed_once_and_then_read_from_disk(tmp_path):
    first = FakeEmbedder()
    a = DenseKnowledgeBase(CHUNKS, first, tmp_path / "vectors")
    second = FakeEmbedder()
    b = DenseKnowledgeBase(CHUNKS, second, tmp_path / "vectors")
    assert (first.passage_calls, second.passage_calls) == (1, 0)
    assert np.array_equal(a._vectors, b._vectors) and len(list((tmp_path / "vectors").glob("*.npy"))) == 1


def test_without_a_cache_folder_nothing_is_written_and_every_build_embeds():
    e = FakeEmbedder()
    DenseKnowledgeBase(CHUNKS, e)
    DenseKnowledgeBase(CHUNKS, e)
    assert e.passage_calls == 2


def test_editing_a_document_or_changing_the_model_makes_the_cache_miss(tmp_path):
    DenseKnowledgeBase(CHUNKS, FakeEmbedder(), tmp_path)
    edited = [*CHUNKS[:-1], Chunk("misc.md", "Misc > Parking", "The car park is by the front gate.")]
    for chunks, embedder in ((edited, FakeEmbedder()), (CHUNKS, FakeEmbedder(name="another-model"))):
        DenseKnowledgeBase(chunks, embedder, tmp_path)
        assert embedder.passage_calls == 1
    assert len(list(tmp_path.glob("*.npy"))) == 3


def test_a_damaged_cache_file_is_replaced_not_trusted(tmp_path):
    DenseKnowledgeBase(CHUNKS, FakeEmbedder(), tmp_path)
    [cache] = tmp_path.glob("*.npy")
    np.save(cache, np.zeros((1, DIMS)))                                      # wrong number of passages
    rebuilt = FakeEmbedder()
    dense = DenseKnowledgeBase(CHUNKS, rebuilt, tmp_path)
    assert rebuilt.passage_calls == 1 and dense._vectors.shape[0] == len(CHUNKS)
    assert np.load(cache).shape[0] == len(CHUNKS)                            # and the file is healed


# -- the hybrid ------------------------------------------------------------------------------------------------------------


class Stub:
    """A retriever with a fixed ranking, to test the fusion arithmetic without any search behind it."""

    def __init__(self, chunks, order):
        self.chunks, self._order = chunks, order

    sources = property(lambda self: sorted({c.source for c in self.chunks}))

    def search(self, query, k=3):
        from jobshop.knowledge.base import Hit
        return [Hit(self.chunks[i], 1.0) for i in self._order[:k]]


def test_fusion_adds_one_over_sixty_plus_rank_for_each_ranking_a_passage_appears_in():
    hybrid = HybridKnowledgeBase(Stub(CHUNKS, [0, 1, 2]), Stub(CHUNKS, [1, 0, 3]))
    scores = {h.chunk.source: h.score for h in hybrid.search("q", 4)}
    assert scores["machine.md"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2), abs=1e-4)   # rank 1 and 2
    assert scores["hours.md"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1), abs=1e-4)     # rank 2 and 1
    assert scores["rush.md"] == pytest.approx(1 / (RRF_K + 3), abs=1e-4)                        # only the keyword list
    assert scores["misc.md"] == pytest.approx(1 / (RRF_K + 3), abs=1e-4)                        # only the vector list


def test_a_passage_both_methods_like_beats_one_only_a_single_method_likes():
    hybrid = HybridKnowledgeBase(Stub(CHUNKS, [2, 0]), Stub(CHUNKS, [3, 0]))
    assert [h.chunk.source for h in hybrid.search("q", 3)][0] == "machine.md"          # 2nd in both beats 1st in one


def test_a_passage_only_one_method_finds_still_makes_the_results():
    hybrid = HybridKnowledgeBase(Stub(CHUNKS, [2]), Stub(CHUNKS, [0, 1, 3]))
    assert "rush.md" in {h.chunk.source for h in hybrid.search("q", 4)}


def test_ties_keep_document_order_and_k_limits_the_results():
    hybrid = HybridKnowledgeBase(Stub(CHUNKS, [3, 2]), Stub(CHUNKS, [2, 3]))
    assert [h.chunk.source for h in hybrid.search("q", 2)] == ["rush.md", "misc.md"]


def test_the_two_indexes_must_cover_the_same_passages():
    with pytest.raises(ValueError, match="same passages"):
        HybridKnowledgeBase(Stub(CHUNKS, [0]), Stub(CHUNKS[:2], [0]))


def test_the_real_pair_combines_a_keyword_hit_and_a_meaning_hit():
    keyword = KnowledgeBase(CHUNKS)
    hybrid = HybridKnowledgeBase(keyword, DenseKnowledgeBase(CHUNKS, FakeEmbedder()))
    assert hybrid.search("the motor keeps seizing up", 1)[0].chunk.source == "machine.md"     # keywords alone find nothing
    assert hybrid.search("car park", 1)[0].chunk.source == "misc.md"                          # an exact phrase still works
    assert hybrid.sources == keyword.sources


# -- the optional embedding model ---------------------------------------------------------------------------------------------


def test_creating_the_embedder_needs_no_package_and_no_download():
    FastEmbedder()          # nothing is imported or fetched until the first embed


def test_a_missing_package_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)      # makes `import fastembed` fail
    with pytest.raises(ImportError, match="uv sync --extra embeddings"):
        FastEmbedder().embed_query("anything")


class FakeModel:
    def __init__(self):
        self.seen = []

    def embed(self, texts):
        self.seen.append(list(texts))
        return iter([np.array([3.0, 4.0]) for _ in texts])


def test_questions_get_the_models_search_prefix_and_passages_do_not():
    embedder = FastEmbedder("BAAI/bge-small-en-v1.5")
    embedder._model = FakeModel()
    embedder.embed_query("who approves overtime")
    embedder.embed_passages(["a passage"])
    assert embedder._model.seen == [[BGE_QUERY_PREFIX + "who approves overtime"], ["a passage"]]


def test_a_model_that_is_not_bge_gets_no_prefix_and_vectors_are_scaled_to_length_one():
    embedder = FastEmbedder("some/other-model")
    embedder._model = FakeModel()
    vector = embedder.embed_query("question")
    assert embedder._model.seen == [["question"]]
    assert np.allclose(vector, [0.6, 0.8]) and np.isclose(np.linalg.norm(embedder.embed_passages(["p"])[0]), 1.0)


# -- choosing a retriever ------------------------------------------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "machine.md").write_text("# Machine\n\n## Bearings\n\nBearing failure: stop the machine.\n")
    (docs / "hours.md").write_text("# Hours\n\n## Overtime\n\nOvertime needs approval.\n")
    return docs


def test_each_kind_builds_the_right_retriever(folder, tmp_path):
    fake = FakeEmbedder()
    assert type(build_retriever("bm25", folder)) is KnowledgeBase
    assert type(build_retriever("dense", folder, embedder=fake, cache_dir=tmp_path / "c")) is DenseKnowledgeBase
    assert type(build_retriever("hybrid", folder, embedder=fake, cache_dir=tmp_path / "c")) is HybridKnowledgeBase
    assert (tmp_path / "c" / "embeddings").is_dir()                       # vectors cached under <cache>/embeddings


def test_an_unknown_kind_is_a_configuration_error(folder):
    with pytest.raises(ValueError, match="unknown retriever 'fuzzy'"):
        build_retriever("fuzzy", folder)


def test_asking_for_vector_search_without_the_package_is_a_clear_configuration_error(folder, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    for kind in ("dense", "hybrid"):
        with pytest.raises(ValueError, match="uv sync --extra embeddings"):
            build_retriever(kind, folder)


def test_the_environment_chooses_the_retriever_and_keyword_is_the_default(folder, monkeypatch):
    assert type(load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": str(folder)})) is KnowledgeBase
    assert type(load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": str(folder), "JOBSHOP_RETRIEVER": " BM25 "})) is KnowledgeBase
    with pytest.raises(ValueError, match="unknown retriever"):
        load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": str(folder), "JOBSHOP_RETRIEVER": "magic"})
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ValueError, match="uv sync --extra embeddings"):
        load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": str(folder), "JOBSHOP_RETRIEVER": "dense"})


def test_the_search_tool_works_the_same_over_a_vector_retriever(folder, tmp_path):
    ctx = build_ctx()
    ctx.knowledge = build_retriever("dense", folder, embedder=FakeEmbedder())
    out = ToolRegistry(ctx).call("search_knowledge", {"query": "the motor keeps seizing up", "k": 1})
    [p] = out["passages"]
    assert p["source"] == "machine.md" and p["score"] > 0.5


def test_a_missing_package_is_reported_when_the_retriever_is_built_even_if_the_vectors_are_cached(folder, tmp_path, monkeypatch):
    """With cached passage vectors the model is not needed to build the index, only to answer a question."""
    build_retriever("dense", folder, embedder=FakeEmbedder("bge-small"), cache_dir=tmp_path)       # fills the cache
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ValueError, match="uv sync --extra embeddings"):
        build_retriever("dense", folder, cache_dir=tmp_path, model="bge-small")                   # real embedder, cache hit


def test_a_text_with_no_signal_gives_a_zero_vector_not_a_division_by_zero():
    from jobshop.knowledge.semantic import normalise

    out = normalise(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    assert not np.isnan(out).any() and np.allclose(out[0], 0) and np.allclose(out[1], [0.6, 0.8, 0.0])
    assert FakeEmbedder().embed_query("the of and").tolist() == [0.0] * DIMS            # only common words: no vector, no crash


def test_fusion_looks_deeper_than_k_so_a_passage_ranked_second_in_both_lists_can_win():
    # Passage 0 is second in both rankings; 2 and 1 are first in one each. Looking only k=1 deep would see a tie
    # between 1 and 2 and miss that 0 is the one both methods like.
    hybrid = HybridKnowledgeBase(Stub(CHUNKS, [2, 0]), Stub(CHUNKS, [1, 0]))
    assert hybrid.search("q", 1)[0].chunk is CHUNKS[0]
