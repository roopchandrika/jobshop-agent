"""Evaluating retrieval on its own, and the pieces that let the agent evals use the plant documents."""

import pytest

from jobshop.agent.loop import AgentConfig
from jobshop.evals import cli, retrieval
from jobshop.evals.judge import GROUNDING_CRITERION, criteria_for, ground_truth
from jobshop.evals.oracle import OracleClient
from jobshop.evals.runner import execute, run_suite
from jobshop.evals.scenario import Scenario
from jobshop.knowledge import Chunk, KnowledgeBase
from tests.evals.conftest import EVALS
from tests.helpers import FAST

KNOWLEDGE_IDS = ("kn-01", "kn-02", "kn-03", "kn-04", "kn-05")


# -- retrieval metrics ----------------------------------------------------------------------------------------------


@pytest.fixture
def tiny():
    return KnowledgeBase([
        Chunk("a.md", "A", "seal leak seal"), Chunk("b.md", "B", "motor heat"), Chunk("c.md", "C", "seal motor"),
    ])


def q(id_, text, *sources, hard=False):
    return retrieval.Question(id=id_, question=text, sources=list(sources), hard=hard)


def test_a_question_is_a_hit_when_a_right_document_is_in_the_top_k_and_its_rank_is_recorded(tiny):
    first, second, missed = retrieval.evaluate(
        tiny, [q("1", "seal leak", "a.md"), q("2", "motor heat", "b.md"), q("3", "seal leak", "b.md")], 2)
    assert first.rank == 1 and second.rank == 1
    assert missed.rank is None and missed.returned[0] == "a.md"


def test_the_rank_counts_from_one_so_a_second_place_answer_scores_a_half():
    outcomes = retrieval.evaluate(
        KnowledgeBase([Chunk("x.md", "X", "seal seal seal"), Chunk("y.md", "Y", "seal"), Chunk("z.md", "Z", "other")]),
        [q("1", "seal", "y.md")], 3)
    assert outcomes[0].rank == 2 and retrieval.score(outcomes, 3).mrr == 0.5


def test_scores_are_hit_at_1_hit_at_k_and_mean_reciprocal_rank():
    def outcome(rank):
        return retrieval.Outcome(q("x", "t", "a.md"), rank, [])

    s = retrieval.score([outcome(1), outcome(2), outcome(None), outcome(4)], k=5)
    assert (s.questions, s.hit_at_1, s.hit_at_k) == (4, 0.25, 0.75)
    assert s.mrr == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert retrieval.score([], 3) == retrieval.Scores(0, 0.0, 0.0, 0.0, 3)


def test_a_question_naming_a_document_that_does_not_exist_is_an_error(tiny):
    with pytest.raises(ValueError, match="do not exist"):
        retrieval.evaluate(tiny, [q("1", "seal", "ghost.md")])


def test_the_report_separates_ordinary_from_hard_questions_and_lists_the_misses(tiny):
    outcomes = retrieval.evaluate(tiny, [q("o1", "seal leak", "a.md"), q("h1", "unrelated gibberish", "b.md", hard=True)], 3)
    text = retrieval.render(outcomes, 3)
    assert "ordinary" in text and "hard (worded unlike the documents)" in text
    assert "Not found in the top 3" in text and "h1 (hard)" in text and "got nothing" in text


def test_question_files_are_validated(tmp_path):
    (tmp_path / "bad.yaml").write_text("qs: []")
    with pytest.raises(ValueError, match="top level"):
        retrieval.load_questions(tmp_path / "bad.yaml")
    (tmp_path / "dup.yaml").write_text("questions:\n - {id: a, question: x, sources: [d.md]}\n - {id: a, question: y, sources: [d.md]}")
    with pytest.raises(ValueError, match="duplicate"):
        retrieval.load_questions(tmp_path / "dup.yaml")
    (tmp_path / "typo.yaml").write_text("questions:\n - {id: a, question: x, sources: [d.md], hrad: true}")
    with pytest.raises(Exception, match="hrad"):
        retrieval.load_questions(tmp_path / "typo.yaml")


# -- the shipped documents and questions: a floor under retrieval quality ------------------------------------------------


@pytest.fixture(scope="module")
def shipped():
    kb = KnowledgeBase.from_directory(EVALS.parent / "knowledge")
    return retrieval.evaluate(kb, retrieval.load_questions(EVALS / "retrieval.yaml"), 3)


def test_every_ordinary_question_finds_its_document_in_the_top_three(shipped):
    ordinary = [o for o in shipped if not o.question.hard]
    missed = [o.question.id for o in ordinary if o.rank is None]
    assert not missed and len(ordinary) >= 20


def test_the_hard_paraphrases_are_kept_in_and_expose_the_keyword_limit(shipped):
    hard = [o for o in shipped if o.question.hard]
    assert len(hard) >= 5
    assert any(o.rank is None for o in hard), "if a keyword index now finds every paraphrase, the hard set no longer tests anything"


def test_the_retrieval_command_prints_the_table_and_can_gate_on_a_minimum(capsys):
    assert cli.main(["--evals-dir", str(EVALS), "retrieval", "--min-hit", "0.9"]) == 0
    out = capsys.readouterr().out
    assert "hit@3" in out and "ordinary" in out and "12 documents" in out
    assert cli.main(["--evals-dir", str(EVALS), "retrieval", "--k", "1", "--min-hit", "1.01"]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_the_retrieval_command_reports_a_missing_folder_as_a_configuration_error(tmp_path, capsys):
    assert cli.main(["--evals-dir", str(EVALS), "retrieval", "--knowledge-dir", str(tmp_path / "nope")]) == 2
    assert "Configuration error" in capsys.readouterr().err


# -- scenarios that need the documents ------------------------------------------------------------------------------------------


def test_the_knowledge_scenarios_declare_that_they_need_the_documents(scenarios):
    assert {s.id for s in scenarios.values() if s.needs_knowledge} == set(KNOWLEDGE_IDS)


def test_a_scenario_that_lists_the_tool_among_alternatives_also_needs_it():
    base = {"id": "x", "category": "read_only", "description": "d", "request": "r"}
    with_tool = {"outcome": "no_action", "tools_any": [["list_orders", "search_knowledge"]]}
    assert Scenario.model_validate({**base, "expect": with_tool}).needs_knowledge
    assert not Scenario.model_validate({**base, "expect": {"outcome": "no_action"}}).needs_knowledge


def test_running_them_without_documents_is_a_setup_error_found_before_anything_runs(scenarios, shops):
    with pytest.raises(ValueError, match="none were loaded.*kn-01.*kn-02"):
        run_suite([scenarios["kn-02"], scenarios["kn-01"]], OracleClient, AgentConfig(model="m"), shops, FAST)


def test_the_documents_reach_the_agent_only_when_they_are_given(scenarios, shops):
    scenario = scenarios["q-01"]
    without = execute(scenario, OracleClient(scenario), AgentConfig(model="oracle"), shops["default"], FAST)
    assert without.ctx.knowledge is None
    given = KnowledgeBase([Chunk("a.md", "A", "text")])
    with_docs = execute(scenario, OracleClient(scenario), AgentConfig(model="oracle"), shops["default"], FAST, knowledge=given)
    assert with_docs.ctx.knowledge is given


def test_the_cli_says_which_scenarios_need_a_missing_documents_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    code = cli.main(["--evals-dir", str(EVALS), "run", "--oracle", "--only", "kn-01", "--knowledge-dir", str(tmp_path / "nope"),
                     "--out", str(tmp_path / "o")])
    assert code == 2 and "kn-01 need the plant documents" in capsys.readouterr().err


def test_scenarios_that_do_not_need_documents_run_without_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert cli.main(["--evals-dir", str(EVALS), "run", "--oracle", "--only", "q-01", "--knowledge-dir", str(tmp_path / "nope"),
                     "--out", str(tmp_path / "o")]) == 0


# -- the judge sees what the assistant was shown ----------------------------------------------------------------------------------------


def test_the_judge_is_given_the_retrieved_passages_and_a_grounding_criterion(scenarios, shops, kb):
    run = execute(scenarios["kn-02"], OracleClient(scenarios["kn-02"]), AgentConfig(model="oracle"), shops["default"], FAST, knowledge=kb)
    truth = ground_truth(run)
    assert "incident-2025-11-m3-coolant-pump.md" in {p["source"] for p in truth["plant_document_passages"]}
    assert all(set(p) == {"source", "section", "text"} for p in truth["plant_document_passages"])
    assert GROUNDING_CRITERION[0] in criteria_for(run)


def test_a_question_that_never_used_the_documents_is_not_graded_for_grounding(scenarios, shops):
    run = execute(scenarios["q-01"], OracleClient(scenarios["q-01"]), AgentConfig(model="oracle"), shops["default"], FAST)
    assert GROUNDING_CRITERION[0] not in criteria_for(run) and "plant_document_passages" not in ground_truth(run)


def test_the_same_passage_found_twice_is_listed_once(scenarios, shops, kb):
    run = execute(scenarios["kn-02"], OracleClient(scenarios["kn-02"]), AgentConfig(model="oracle"), shops["default"], FAST, knowledge=kb)
    run.calls.append(run.calls[0])
    passages = ground_truth(run)["plant_document_passages"]
    assert len(passages) == len({(p["source"], p["section"], p["text"]) for p in passages})


def test_an_agent_that_never_searched_is_still_graded_for_grounding_when_the_question_needed_the_documents(scenarios, shops, kb):
    from tests.fake_llm import FakeClient, submit

    guess = [submit(summary="The coolant pump was out for about two hours, I believe.")]
    run = execute(scenarios["kn-02"], FakeClient(guess), AgentConfig(model="fake"), shops["default"], FAST, knowledge=kb)
    assert not any(c.name == "search_knowledge" for c in run.calls)
    assert GROUNDING_CRITERION[0] in criteria_for(run)
    assert "plant_document_passages" not in ground_truth(run)       # it was shown nothing, which is what the judge should see


# -- comparing retrieval methods, and what happens on questions the documents do not cover ----------------------------------------


def outcome(id_, rank, hard=False):
    return retrieval.Outcome(q(id_, "text", "a.md", hard=hard), rank, [])


def test_the_comparison_has_a_row_per_method_and_names_the_questions_they_disagree_on():
    results = {
        "bm25": [outcome("o1", 1), outcome("h1", None, hard=True), outcome("h2", 1, hard=True)],
        "dense": [outcome("o1", 1), outcome("h1", 1, hard=True), outcome("h2", 3, hard=True)],
    }
    text = retrieval.render_comparison(results, 3)
    assert "bm25" in text and "dense" in text and "1 ordinary and 2 hard" in text
    assert "h1 (hard): found by dense" in text and "h2" not in text.split("disagree")[1]       # h2 is found by both


def test_methods_that_agree_everywhere_list_no_disagreements():
    same = [outcome("o1", 1), outcome("h1", 2, hard=True)]
    assert "disagree" not in retrieval.render_comparison({"a": same, "b": list(same)}, 3)


def test_methods_scored_on_different_questions_cannot_be_compared():
    with pytest.raises(ValueError, match="same questions"):
        retrieval.render_comparison({"a": [outcome("o1", 1)], "b": [outcome("o2", 1)]}, 3)


def test_separation_is_the_lowest_answerable_score_against_the_highest_off_topic_one():
    kb = KnowledgeBase([Chunk("a.md", "A", "seal leak seal"), Chunk("b.md", "B", "motor heat")])
    s = retrieval.measure_separation(kb, [q("1", "seal leak", "a.md"), q("2", "motor heat", "b.md")], ["seal", "canteen menu"])
    assert s.off_topic == 2 and s.off_topic_returned == 1            # "canteen menu" shares no word with anything
    assert s.answerable_low is not None and s.off_topic_high is not None
    assert s.margin == pytest.approx(s.answerable_low - s.off_topic_high, abs=1e-3)


def test_the_lowest_answerable_score_is_taken_over_every_question_not_just_the_first():
    kb = KnowledgeBase([Chunk("a.md", "A", "seal seal seal leak"), Chunk("b.md", "B", "motor heat pump"), Chunk("c.md", "C", "filler")])
    strong, weak = q("1", "seal", "a.md"), q("2", "motor pump heat", "b.md")
    best = {x.id: kb.search(x.question, 1)[0].score for x in (strong, weak)}
    assert best["1"] != best["2"]
    for order in ([strong, weak], [weak, strong]):
        assert retrieval.measure_separation(kb, order, []).answerable_low == min(best.values())


def test_with_nothing_returned_for_off_topic_questions_there_is_no_off_topic_score_or_margin():
    kb = KnowledgeBase([Chunk("a.md", "A", "seal leak")])
    s = retrieval.measure_separation(kb, [q("1", "seal leak", "a.md")], ["canteen menu"])
    assert (s.off_topic_returned, s.off_topic_high, s.margin) == (0, None, None)
    assert "0 of 1" in retrieval.render_separation({"bm25": s}) and "-" in retrieval.render_separation({"bm25": s})


def test_a_negative_margin_means_a_score_cannot_tell_answers_from_non_answers():
    s = retrieval.Separation(answerable_low=0.53, off_topic_high=0.57, off_topic_returned=6, off_topic=6)
    assert s.margin == pytest.approx(-0.04) and "-0.040" in retrieval.render_separation({"dense": s})


def test_the_shipped_questions_include_off_topic_ones_and_the_file_still_rejects_unknown_keys(tmp_path):
    assert len(retrieval.load_unanswerable(EVALS / "retrieval.yaml")) >= 5
    (tmp_path / "x.yaml").write_text("questions: []\nsurprise: 1")
    with pytest.raises(ValueError, match="top level"):
        retrieval.load_questions(tmp_path / "x.yaml")
    (tmp_path / "y.yaml").write_text("questions: []")
    assert retrieval.load_unanswerable(tmp_path / "y.yaml") == []


def test_keyword_search_is_the_default_method_and_vector_search_without_the_package_is_a_clear_error(capsys, monkeypatch):
    import sys
    assert cli.main(["--evals-dir", str(EVALS), "retrieval"]) == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.strip().startswith("bm25") and " of " in line)
    assert "Questions the documents do not cover" in out and "1 of 6" in row          # keyword search: one stray match
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert cli.main(["--evals-dir", str(EVALS), "retrieval", "--method", "dense"]) == 2
    assert "uv sync --extra embeddings" in capsys.readouterr().err


def test_asking_for_all_methods_builds_every_one_and_reports_the_first_that_cannot_be_built(capsys, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "fastembed", None)
    assert cli.main(["--evals-dir", str(EVALS), "retrieval", "--method", "all"]) == 2
    err = capsys.readouterr().err
    assert "uv sync --extra embeddings" in err and "unknown retriever" not in err
