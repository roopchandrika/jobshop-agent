"""Chunking splits documents where they have structure; the index ranks passages by BM25."""

import math

import pytest

from jobshop.knowledge import KnowledgeBase, chunk_markdown, load_knowledge
from jobshop.knowledge.base import tokens
from jobshop.knowledge.chunking import Chunk

DOC = """# Pumps

Intro text about pumps.

## Seals

Seals leak when worn. Replace the seal.

## Motors

Motors overheat when the vent is blocked.
"""


# -- chunking ---------------------------------------------------------------------------------------------------


def test_a_document_is_split_at_its_headings_and_each_chunk_knows_where_it_came_from():
    chunks = chunk_markdown("pumps.md", DOC)
    assert [(c.source, c.heading) for c in chunks] == [("pumps.md", "Pumps"), ("pumps.md", "Pumps > Seals"), ("pumps.md", "Pumps > Motors")]
    assert chunks[1].text == "Seals leak when worn. Replace the seal."


def test_a_long_section_is_split_between_paragraphs_never_inside_one():
    paragraphs = [" ".join(f"w{i}x{j}" for j in range(40)) for i in range(5)]       # five 40-word paragraphs
    chunks = chunk_markdown("long.md", "# Title\n\n## Big\n\n" + "\n\n".join(paragraphs), max_words=100)
    assert len(chunks) == 3 and all(c.heading == "Title > Big" for c in chunks)       # 2 + 2 + 1 paragraphs
    assert "\n\n".join(c.text for c in chunks).split("\n\n") == paragraphs            # nothing lost, nothing cut


def test_one_paragraph_longer_than_the_limit_is_kept_whole():
    chunks = chunk_markdown("x.md", "# T\n\n" + " ".join(["word"] * 300), max_words=50)
    assert len(chunks) == 1 and len(chunks[0].text.split()) == 300


def test_empty_sections_and_empty_documents_produce_no_chunks():
    assert chunk_markdown("e.md", "") == []
    assert [c.heading for c in chunk_markdown("e.md", "# A\n\n## Empty\n\n## Full\n\nbody")] == ["A > Full"]


def test_text_before_any_heading_still_belongs_to_the_document():
    [chunk] = chunk_markdown("plain.md", "just some text\n")
    assert chunk == Chunk("plain.md", "plain.md", "just some text")


# -- tokens ------------------------------------------------------------------------------------------------------


def test_tokens_lowercase_drop_common_words_and_fold_plurals_and_endings():
    assert tokens("The Bearings are failing, and it failed") == ["bearing", "fail", "fail"]
    assert tokens("M2 and M3") == ["m2", "m3"]            # short identifiers are left alone
    assert tokens("is it the of") == []


# -- ranking -------------------------------------------------------------------------------------------------------


@pytest.fixture
def kb():
    return KnowledgeBase([
        Chunk("a.md", "A > Seals", "Seals leak when worn. Replace the seal on the pump."),
        Chunk("b.md", "B > Motors", "Motors overheat when the vent is blocked. Clean the vent."),
        Chunk("c.md", "C > Misc", "The pump room is cold in winter."),
    ])


def test_the_passage_with_the_query_words_ranks_first_and_unrelated_ones_are_not_returned(kb):
    hits = kb.search("seal leak", 3)
    assert [h.chunk.source for h in hits] == ["a.md"]


def test_a_word_in_few_passages_counts_for_more_than_a_word_in_many():
    kb = KnowledgeBase([Chunk("1.md", "x", "pump pump rare"), Chunk("2.md", "x", "pump pump"), Chunk("3.md", "x", "pump other")])
    assert [h.chunk.source for h in kb.search("pump rare")][0] == "1.md"
    assert kb._idf["rare"] > kb._idf["pump"]


def test_repeating_a_word_helps_but_with_diminishing_returns():
    kb = KnowledgeBase([Chunk("1.md", "x", " ".join(["leak"] * 1 + ["pad"] * 20)),
                        Chunk("2.md", "x", " ".join(["leak"] * 2 + ["pad"] * 19)),
                        Chunk("3.md", "x", " ".join(["leak"] * 4 + ["pad"] * 17)),
                        Chunk("4.md", "x", "unrelated filler text " * 7)])
    s = {h.chunk.source: h.score for h in kb.search("leak", 4)}
    assert s["3.md"] > s["2.md"] > s["1.md"] and (s["3.md"] - s["2.md"]) < 2 * (s["2.md"] - s["1.md"])


def test_a_long_passage_does_not_win_just_by_being_long():
    short = Chunk("short.md", "x", "bearing failure")
    long = Chunk("long.md", "x", "bearing failure " + "filler words about nothing " * 40)
    assert KnowledgeBase([short, long, Chunk("o.md", "x", "other text")]).search("bearing failure")[0].chunk.source == "short.md"


def test_the_score_follows_the_bm25_formula():
    kb = KnowledgeBase([Chunk("a.md", "h", "leak"), Chunk("b.md", "h", "other thing"), Chunk("c.md", "h", "more stuff")])
    avg = sum(kb._lengths) / 3
    tf, length = 1, kb._lengths[0]
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * length / avg))
    assert kb.search("leak")[0].score == pytest.approx(expected, abs=1e-3)


def test_the_heading_counts_as_part_of_what_a_passage_is_about():
    kb = KnowledgeBase([Chunk("a.md", "Coolant pump > What happened", "It stopped."), Chunk("b.md", "Other", "It stopped on Monday.")])
    assert kb.search("coolant pump")[0].chunk.source == "a.md"


def test_k_limits_the_results_and_ties_keep_document_order(kb):
    twin = KnowledgeBase([Chunk("1.md", "x", "same words"), Chunk("2.md", "x", "same words"), Chunk("3.md", "x", "same words")])
    assert [h.chunk.source for h in twin.search("same words", 2)] == ["1.md", "2.md"]


def test_a_question_of_only_common_or_unknown_words_returns_nothing(kb):
    assert kb.search("what is the") == [] and kb.search("canteen menu") == [] and kb.search("") == []


def test_an_empty_knowledge_base_is_refused():
    with pytest.raises(ValueError, match="at least one passage"):
        KnowledgeBase([])


# -- loading ---------------------------------------------------------------------------------------------------------


def test_a_folder_of_markdown_files_becomes_one_knowledge_base_in_a_stable_order(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\nbeta text")
    (tmp_path / "a.md").write_text("# A\n\nalpha text")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("# C\n\ngamma text")
    (tmp_path / "notes.txt").write_text("ignored")
    kb = KnowledgeBase.from_directory(tmp_path)
    assert kb.sources == ["a.md", "b.md", "sub/c.md"] and [c.source for c in kb.chunks] == ["a.md", "b.md", "sub/c.md"]


def test_a_folder_with_no_documents_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no .md documents"):
        KnowledgeBase.from_directory(tmp_path)


def test_the_documents_folder_comes_from_the_environment_or_the_default(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha")
    assert load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": str(tmp_path)}).sources == ["a.md"]
    assert load_knowledge({}, default=str(tmp_path)).sources == ["a.md"]
    assert load_knowledge({}, default=None) is None
    assert load_knowledge({}, default=str(tmp_path / "missing")) is None            # no default folder: simply none


@pytest.mark.parametrize("value", ["", "off", "OFF", "none"])
def test_the_documents_can_be_switched_off(tmp_path, value):
    (tmp_path / "a.md").write_text("# A\n\nalpha")
    assert load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": value}, default=str(tmp_path)) is None


def test_a_folder_named_explicitly_that_does_not_exist_is_an_error_not_silence(tmp_path):
    with pytest.raises(ValueError, match="is not a folder"):
        load_knowledge({"JOBSHOP_KNOWLEDGE_DIR": str(tmp_path / "typo")})
