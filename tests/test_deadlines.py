"""Deadlines: a degraded backend costs a worse answer, not a lost turn."""

from __future__ import annotations

import time

import pytest

from llmwiki.config import StageBudgets
from llmwiki.errors import ProviderError
from llmwiki.retrieval import RetrievalOptions, search
from llmwiki.retrieval.profiles import resolve as resolve_profile
from llmwiki.retrieval.telemetry import EXPIRED, OK, Deadline, RecordingSink


def _page(project, path, title, body):
    project.write(
        path,
        f"---\ntype: concept\ntitle: {title}\nsources: []\n---\n\n# {title}\n\n{body}\n",
    )


def _corpus(project):
    _page(project, "wiki/concepts/storage.md", "Storage", "Grid storage economics and batteries.")
    _page(project, "wiki/concepts/latency.md", "Latency", "The delay before a transfer begins.")


def _vector_ready(settings):
    settings.embedding.enabled = True
    settings.embedding.model = "toy"
    return settings


def _with_vectors(project, settings, monkeypatch):
    """A real vector index over the corpus, embedded by a toy model."""
    from llmwiki.embeddings import index_documents
    from llmwiki.retrieval import load_documents

    def fake_embed(texts, _config):
        return [
            [1.0, 0.0] if any(w in t.lower() for w in ("latency", "delay", "slow")) else [0.0, 1.0]
            for t in texts
        ]

    monkeypatch.setattr("llmwiki.embeddings.embed_texts", fake_embed)
    _vector_ready(settings)
    index_documents(project, load_documents(project, include_sources=False), settings.embedding)


# ── the clock itself ──────────────────────────────────────────────────────

def test_a_stage_budget_is_capped_by_what_is_left_of_the_turn():
    deadline = Deadline(StageBudgets(turn=100.0, embedding=60.0), started=time.perf_counter())
    assert deadline.for_stage("embedding") == pytest.approx(60.0, abs=5)

    spent = Deadline(StageBudgets(turn=100.0, embedding=60.0))
    spent.started -= 0.08  # 80 ms of a 100 ms turn already gone
    # The stage would like 60 ms and the user is owed an answer in 20.
    assert spent.for_stage("embedding") == pytest.approx(20.0, abs=5)
    assert spent.affords("embedding")

    over = Deadline(StageBudgets(turn=100.0))
    over.started -= 0.2
    assert over.expired() and not over.affords("embedding")


def test_no_turn_budget_means_nothing_expires_and_stage_budgets_still_apply():
    deadline = Deadline(StageBudgets())
    assert deadline.remaining_ms() is None
    assert not deadline.expired()
    # The point of the merge delta: a 60-second embedding timeout was wrong on
    # the text path too, so the text path has a budget without having a turn.
    assert deadline.for_stage("embedding") == 3_000.0


def test_only_the_spoken_profile_carries_a_turn_deadline():
    assert resolve_profile("voice").budgets.turn == 400.0
    assert resolve_profile("voice").budgets.embedding == 60.0
    for name in ("balanced", "deep", "research"):
        assert resolve_profile(name).budgets.turn is None


# ── the fallbacks ─────────────────────────────────────────────────────────

def test_a_spent_turn_skips_the_embedding_and_answers_from_the_lexical_lane(
    wiki, settings, monkeypatch
):
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)

    def never_called(*_args, **_kwargs):
        raise AssertionError("the embedding was called after the turn's budget was spent")

    monkeypatch.setattr("llmwiki.embeddings.embed_query", never_called)

    deadline = Deadline(StageBudgets(turn=100.0, embedding=60.0))
    deadline.started -= 0.2
    response = search(
        wiki, "storage", include_sources=False,
        embedding_config=settings.embedding, deadline=deadline,
    )

    assert response.results, "a spent budget costs a lane, not the turn"
    # Both optional rungs go, in the order the ladder drops them, and the
    # ranking that remains is the lexical lane's.
    assert response.lanes.expired == ("vector", "graph")
    assert response.lanes.vector is False
    assert any("budget was spent" in note for note in response.notes)


def test_a_blackholed_endpoint_falls_back_inside_the_budget_rather_than_its_timeout(
    wiki, settings, monkeypatch
):
    """The `Done when` for this step, as a test.

    60 seconds was the configured embedding timeout and the reason a dead
    endpoint cost a whole turn. The fallback it needed already existed; all a
    deadline does is reach it sooner.
    """
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)

    def blackhole(_text, _config, timeout=None):
        time.sleep(timeout if timeout else 60.0)
        raise ProviderError(f"could not reach the embedding endpoint: timed out after {timeout}s")

    monkeypatch.setattr("llmwiki.embeddings.embed_query", blackhole)

    started = time.perf_counter()
    response = search(
        wiki, "storage", include_sources=False,
        embedding_config=settings.embedding,
        deadline=Deadline(StageBudgets.voice()),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert response.results, "lexical results, not an exception"
    assert elapsed_ms < 1_000, f"the turn took {elapsed_ms:.0f} ms"
    # Expiry, not failure: the endpoint is slow, which is a different page of
    # the runbook from the endpoint being broken.
    assert response.lanes.expired == ("vector",)
    assert any("within its budget" in note for note in response.notes)


def test_an_endpoint_that_refuses_is_a_failure_and_not_an_expiry(wiki, settings, monkeypatch):
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)

    def refused(_text, _config, timeout=None):
        raise ProviderError("could not reach http://localhost:1/v1/embeddings: Connection refused")

    monkeypatch.setattr("llmwiki.embeddings.embed_query", refused)

    response = search(
        wiki, "storage", include_sources=False,
        embedding_config=settings.embedding,
        deadline=Deadline(StageBudgets.voice()),
    )
    assert response.lanes.expired == ()
    assert any("vector search unavailable" in note for note in response.notes)


def test_diffusion_is_the_first_thing_dropped_when_the_clock_runs_out(wiki, settings):
    _corpus(wiki)
    deadline = Deadline(StageBudgets(turn=50.0, neighbourhood=15.0))
    deadline.started -= 0.1

    response = search(
        wiki, "storage", include_sources=False,
        options=RetrievalOptions(vector=False), deadline=deadline,
    )
    assert response.results
    assert "graph" in response.lanes.expired
    assert response.lanes.graph is False
    assert any("graph diffusion was skipped" in note for note in response.notes)


def test_a_turn_over_its_search_budget_says_so_rather_than_losing_the_ranking(wiki, settings):
    """Where the implementation departs from the plan, and why.

    The plan says a hybrid search over budget fails the turn. By the time the
    overrun is knowable the ranking exists, and discarding an answer already
    paid for to honour a budget helps nobody — so it is recorded instead.
    """
    _corpus(wiki)
    sink = RecordingSink()
    deadline = Deadline(StageBudgets(search=0.0001))

    response = search(
        wiki, "storage", include_sources=False,
        options=RetrievalOptions(vector=False), deadline=deadline, sink=sink,
    )
    assert response.results
    assert sink.outcomes("search") == [EXPIRED]
    assert any("against a 0 ms budget" in note for note in response.notes)


# ── the telemetry ─────────────────────────────────────────────────────────

def test_the_sink_sees_every_stage_and_a_replay_produces_percentiles(wiki, settings):
    _corpus(wiki)
    sink = RecordingSink()
    for query in ("storage", "latency", "grid", "transfer", "batteries"):
        search(wiki, query, include_sources=False, sink=sink, deadline=Deadline(StageBudgets()))

    table = sink.percentiles()
    assert {"lexical", "fuse", "materialize"} <= set(table)
    assert table["lexical"]["n"] == 5
    assert table["lexical"]["p95"] >= table["lexical"]["p50"]
    assert set(sink.outcomes("lexical")) == {OK}


def test_a_failing_stage_is_recorded_as_failed_and_not_as_a_duration(wiki, settings, monkeypatch):
    _corpus(wiki)
    sink = RecordingSink()

    def explode(*_args, **_kwargs):
        raise RuntimeError("the fuser broke")

    monkeypatch.setattr("llmwiki.retrieval.pipeline._fuse", explode)
    with pytest.raises(RuntimeError):
        search(wiki, "storage", include_sources=False, sink=sink)

    assert sink.outcomes("fuse") == ["failed"]
