"""Deadlines: a degraded backend costs a worse answer, not a lost turn."""

from __future__ import annotations

import time

import pytest

from llmwiki.config import StageBudgets
from llmwiki.errors import ProviderError
from llmwiki.retrieval import RetrievalOptions, search
from llmwiki.retrieval.profiles import Budget, resolve as resolve_profile
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
    voice = resolve_profile("voice").budget
    assert voice.total_ms == 400 and voice.path == "fast"
    assert voice.stages().embedding == 60.0
    for name in ("balanced", "deep", "research"):
        assert resolve_profile(name).budget.total_ms is None
        assert resolve_profile(name).budget.path == "slow"


# ── the fallbacks ─────────────────────────────────────────────────────────

def test_a_spent_turn_skips_the_embedding_and_answers_from_the_lexical_lane(
    wiki, settings, monkeypatch
):
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)

    def never_called(*_args, **_kwargs):
        raise AssertionError("the embedding was called after the turn's budget was spent")

    monkeypatch.setattr("llmwiki.embeddings.embed_query", never_called)

    deadline = Deadline(StageBudgets(turn=100.0, embedding=60.0), path="fast")
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
    assert any("did not cover a round trip" in note for note in response.notes)


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
        deadline=Deadline(StageBudgets.voice(), path="fast"),
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
        deadline=Deadline(StageBudgets.voice(), path="fast"),
    )
    assert response.lanes.expired == ()
    assert any("vector search unavailable" in note for note in response.notes)


def test_diffusion_goes_last_and_only_when_there_is_no_room_at_all(wiki, settings):
    _corpus(wiki)
    deadline = Deadline(StageBudgets(turn=50.0, neighbourhood=15.0), path="fast")
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


# ── the budget as the knob (A3) ───────────────────────────────────────────

def test_a_number_chooses_the_path_and_the_call_ceiling():
    fast = Budget.for_ms(40)
    assert (fast.total_ms, fast.path, fast.max_llm_calls) == (40, "fast", 2)
    slow = Budget.for_ms(4_000)
    assert (slow.total_ms, slow.path, slow.max_llm_calls) == (4_000, "slow", 4)
    assert Budget.for_ms(None).total_ms is None

    # A fast path that adds a round trip is a construction error rather than a
    # tuning opinion — which is the whole reason the path is a field.
    with pytest.raises(ValueError):
        Budget(total_ms=40, path="fast", max_llm_calls=3)
    with pytest.raises(ValueError):
        Budget(total_ms=None, path="fast")


def test_a_budget_keeps_the_profile_s_lanes_and_changes_only_the_wall():
    voice = resolve_profile("voice")
    at_40 = voice.with_budget(40)
    assert at_40.options == voice.options
    assert at_40.budget.total_ms == 40 and at_40.budget.path == "fast"
    assert voice.budget.total_ms == 400, "profiles are frozen; the copy carries the change"


def test_forty_milliseconds_drops_the_round_trip_and_keeps_the_local_rung(
    wiki, settings, monkeypatch
):
    """The ordering A3 corrects, as a test.

    The graph lane is ~8 ms of local CPU and carries multi-hop; the vector lane
    is somebody else's network. So the round trip is the first thing a budget
    makes conditional and diffusion is the last thing it drops — the opposite of
    what the shipped `voice` profile did.
    """
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)

    calls: list[float | None] = []

    def recording_embed(_text, _config, timeout=None):
        calls.append(timeout)
        return [1.0, 0.0]

    monkeypatch.setattr("llmwiki.embeddings.embed_query", recording_embed)

    at_40 = search(
        wiki, "storage", include_sources=False,
        embedding_config=settings.embedding,
        budget=resolve_profile("voice").with_budget(40).budget,
    )
    assert calls == [], "a 40 ms turn makes no round trip at all"
    assert at_40.lanes.expired == ("vector",)
    assert at_40.lanes.graph is True, "the cheap local rung survives the tight budget"
    assert at_40.results

    at_400 = search(
        wiki, "storage", include_sources=False,
        embedding_config=settings.embedding,
        budget=resolve_profile("voice").budget,
    )
    assert len(calls) == 1, "the same profile at 400 ms affords the round trip"
    assert calls[0] == pytest.approx(0.06, abs=0.01), "and hands it the stage's share, in seconds"
    assert at_400.lanes.vector is True
    assert at_400.lanes.expired == ()


def test_the_slow_path_drops_nothing_and_records_the_overrun_instead(
    wiki, settings, monkeypatch
):
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)
    monkeypatch.setattr(
        "llmwiki.embeddings.embed_query", lambda _t, _c, timeout=None: [1.0, 0.0]
    )
    sink = RecordingSink()

    response = search(
        wiki, "storage", include_sources=False,
        embedding_config=settings.embedding,
        # A slow-path turn with a wall it cannot possibly meet.
        budget=Budget(total_ms=2_000, path="slow", max_llm_calls=4),
        deadline=Deadline(StageBudgets.for_turn(2_000), path="slow"),
        sink=sink,
    )
    assert response.lanes.expired == (), "the slow path may not drop a lane"
    assert response.lanes.vector is True


def test_the_vector_lane_survives_a_budget_when_it_is_the_only_lane_left(
    wiki, settings, monkeypatch
):
    """A budget that drops the last lane has not degraded the answer, it removed it."""
    _corpus(wiki)
    _with_vectors(wiki, settings, monkeypatch)
    monkeypatch.setattr(
        "llmwiki.embeddings.embed_query", lambda _t, _c, timeout=None: [1.0, 0.0]
    )

    # A query with no handhold in this corpus's vocabulary: the lexical lane
    # scores under the fence, so it is the vector lane or nothing.
    response = search(
        wiki, "why does the thing take so long to get going", include_sources=False,
        embedding_config=settings.embedding,
        budget=Budget.for_ms(40),
    )
    assert response.lanes.vector is True
    assert "vector" not in response.lanes.expired
