"""The query log: one row per turn, and never at the turn's expense."""

from __future__ import annotations

import sqlite3

import pytest

from llmwiki.project import open_project
from llmwiki.query import ask
from llmwiki.retrieval import log as query_log
from llmwiki.retrieval import search


def _page(project, path, title, body, page_type="concept"):
    project.write(
        path,
        f"---\ntype: {page_type}\ntitle: {title}\nsources: []\n---\n\n# {title}\n\n{body}\n",
    )


def _rows(project):
    with query_log.QueryLog(query_log.path_for(project)) as log:
        return log.recent(50)


def test_ask_writes_one_row_carrying_what_the_answer_cited(wiki, settings, stub_llm):
    _page(wiki, "wiki/concepts/round-trip-efficiency.md", "Round-trip Efficiency",
          "Energy out divided by energy in. Flow batteries reach 75%.")
    _page(wiki, "wiki/concepts/solar-pv.md", "Solar PV",
          "Flow batteries are not photovoltaic; their efficiency is measured differently.")
    stub_llm({"answer": "Flow batteries reach about 75% [1]."})

    answer = ask(wiki, "What round-trip efficiency do flow batteries reach?", settings)

    rows = _rows(wiki)
    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "What round-trip efficiency do flow batteries reach?"
    assert row["profile"] == "balanced"
    assert row["retrieved"], "the ranking is logged"
    assert row["cited"] == [c.path for c in answer.citations if c.cited]
    # The judgement the loop learns from is narrower than the ranking, which is
    # the whole reason the write happens after the citation parse.
    assert len(row["cited"]) < len(row["retrieved"])
    assert row["lanes"]["lexical"] is True
    assert row["stage_ms"], "the telemetry seam lands on the row"
    assert row["raw_query"] is None


def test_search_logs_the_turn_with_no_answer_to_cite(wiki, settings):
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Batteries and thermal stores.")
    response = search(wiki, "storage", top_k=5)
    assert query_log.record(wiki, "storage", response, profile="voice") is None

    row = _rows(wiki)[0]
    # NULL, not []: no answer was generated, which is a different observation
    # from an answer that cited nothing.
    assert row["cited"] is None
    assert row["profile"] == "voice"


def test_a_custom_configuration_is_not_logged_under_a_profile_name(wiki, settings, stub_llm):
    from llmwiki.retrieval import RetrievalOptions

    _page(wiki, "wiki/concepts/storage.md", "Storage", "Batteries and thermal stores.")
    stub_llm({"answer": "ok"})
    ask(wiki, "storage", settings, options=RetrievalOptions(graph_ppr=False))

    assert _rows(wiki)[0]["profile"] == "custom"


def test_the_gate_verdict_is_stored_beside_the_score_it_compared(wiki, settings):
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Batteries and thermal stores.")
    response = search(wiki, "storage", top_k=5)
    query_log.record(wiki, "storage", response, profile="balanced")

    row = _rows(wiki)[0]
    assert row["gate_fired"] is False
    # A lexical-only turn has a top score and no fence to compare it to; both
    # facts are on the row, which is what separates "found nothing" from
    # "found something the gate did not like".
    assert row["lexical_top"] is not None
    assert row["vector_top"] is None


def test_the_query_vector_round_trips_as_float32(wiki):
    vector = [0.5, -0.25, 0.125]
    blob = query_log.pack_vector(vector)
    assert query_log.unpack_vector(blob) == pytest.approx(vector)
    assert query_log.unpack_vector(None) is None
    assert query_log.pack_vector(None) is None


def test_a_log_that_cannot_be_written_costs_a_note_and_not_the_turn(
    wiki, settings, stub_llm, monkeypatch
):
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Batteries and thermal stores.")
    stub_llm({"answer": "Storage covers batteries [1]."})

    def explode(_project):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(query_log, "open_log", explode)
    answer = ask(wiki, "storage", settings)

    assert "Storage covers batteries" in answer.text
    assert any("query log could not be written" in note for note in answer.notes)
    assert not query_log.path_for(wiki).exists()


def test_status_reports_the_log_on_both_of_its_surfaces(tmp_path, capsys):
    """`status` is where a user finds out a log exists at all."""
    import json

    from llmwiki.cli import main

    root = tmp_path / "w"
    main(["init", str(root), "--quiet"])
    project = open_project(root)
    _page(project, "wiki/concepts/storage.md", "Storage", "Grid storage economics.")

    assert main(["status", "-p", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["query_log"] is None

    assert main(["search", "storage", "-p", str(root)]) == 0
    capsys.readouterr()

    assert main(["status", "-p", str(root), "--json"]) == 0
    logged = json.loads(capsys.readouterr().out)["query_log"]
    assert logged["turns"] == 1 and logged["first"] and logged["bytes"] > 0

    assert main(["status", "-p", str(root)]) == 0
    assert "queries   1 turn ·" in capsys.readouterr().err  # printer writes to stderr


def test_the_stats_and_the_ignore_line_arrive_with_the_first_turn(wiki, settings):
    assert query_log.stats(wiki) is None, "no log until a turn happens"

    response = search(wiki, "anything", top_k=5)
    query_log.record(wiki, "anything", response, profile="balanced")

    stats = query_log.stats(wiki)
    assert stats.rows == 1
    assert stats.first == stats.last
    assert stats.bytes > 0

    ignored = (wiki.state_dir / ".gitignore").read_text().split()
    assert "query-log.db" in ignored, "the user's questions are not a shared artifact"


def test_an_older_project_gets_the_ignore_line_when_its_log_is_created(wiki):
    (wiki.state_dir / ".gitignore").write_text("vectors.db\n")
    response = search(wiki, "anything", top_k=5)
    query_log.record(wiki, "anything", response, profile="balanced")

    ignored = (wiki.state_dir / ".gitignore").read_text().split()
    assert "vectors.db" in ignored and "query-log.db" in ignored


def test_the_rewritten_query_is_stored_beside_the_utterance_only_when_it_differs(wiki):
    response = search(wiki, "anything", top_k=5)
    query_log.record(wiki, "anything", response, profile="balanced", raw_query="anything")
    query_log.record(wiki, "the second one", response, profile="balanced", raw_query="and it?")

    rows = _rows(wiki)
    assert rows[0]["raw_query"] == "and it?"
    assert rows[1]["raw_query"] is None


def test_the_vector_lane_lands_its_own_stages_and_its_query_vector(wiki, settings, monkeypatch):
    """The embedding round trip is somebody else's network; it is timed apart.

    And the vector it paid for is kept, because clustering the log later would
    otherwise re-pay one provider call per query for a vector this turn already
    had in hand.
    """
    from llmwiki.embeddings import index_documents
    from llmwiki.retrieval import load_documents

    _page(wiki, "wiki/concepts/rtt.md", "Latency", "The delay before a transfer begins.")

    def fake_embed(texts, _config):
        return [
            [1.0, 0.0] if any(w in t.lower() for w in ("latency", "delay", "slow")) else [0.0, 1.0]
            for t in texts
        ]

    monkeypatch.setattr("llmwiki.embeddings.embed_texts", fake_embed)
    monkeypatch.setattr("llmwiki.embeddings.embed_query", lambda text, cfg: fake_embed([text], cfg)[0])
    settings.embedding.enabled = True
    settings.embedding.model = "toy"
    index_documents(wiki, load_documents(wiki, include_sources=False), settings.embedding)

    response = search(
        wiki, "why is it slow to start", include_sources=False,
        embedding_config=settings.embedding,
    )
    assert {"embed", "vector"} <= set(response.stage_ms)
    assert response.vector_top is not None
    query_log.record(wiki, "why is it slow to start", response, profile="research")

    row = _rows(wiki)[0]
    assert row["query_vector"] == pytest.approx([1.0, 0.0])
    assert row["vector_top"] == pytest.approx(response.vector_top)
