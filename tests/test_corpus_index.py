"""The `CorpusIndex` seam: the protocol, the kit that checks it, and a second implementation.

A conformance kit with one implementation is a description of that
implementation, so everything here runs against both `InMemoryIndex` and a
`StoredCorpus` written deliberately unlike it: it holds its adjacency as data,
has no `WikiGraph`, no `EntityIndex`, and keys its documents on an opaque id
rather than a path. If the ladder ranks that, the protocol is the interface and
not a re-description of the default.
"""

from __future__ import annotations

import pytest

from llmwiki.retrieval import (
    CorpusIndex,
    DocumentNaming,
    InMemoryIndex,
    LexicalSearcher,
    SearchIndex,
    VectorSearcher,
    build_index,
    search_index,
)
from llmwiki.retrieval.calibration import LexicalCalibration
from llmwiki.retrieval.conformance import assert_corpus_index, check_corpus_index
from llmwiki.retrieval.keyword import Document
from llmwiki.retrieval.lexical import LexicalHit
from llmwiki.retrieval.naming import DEFAULT_NAMING
from llmwiki.retrieval.pipeline import RetrievalOptions
from llmwiki.retrieval.ppr import transitions


# ── a second implementation, deliberately unlike the first ────────────────

class ListLexical:
    """Ranks by how many query tokens a document contains. Not FTS5, on purpose."""

    def __init__(self, documents: dict[str, Document]) -> None:
        self._documents = documents
        self.closed = 0

    def search(self, tokens: list[str], limit: int) -> list[LexicalHit]:
        scored = []
        for key, document in self._documents.items():
            body = document.content.lower()
            score = float(sum(body.count(token.lower()) for token in tokens))
            if score:
                scored.append(LexicalHit(path=key, score=score))
        scored.sort(key=lambda hit: (-hit.score, hit.path))
        return scored[:limit]

    def close(self) -> None:
        self.closed += 1


class StoredCorpus:
    """A `CorpusIndex` over data it did not derive.

    The shape a persisted store has: edges are rows, not a graph object; keys
    are ids, not paths; nothing is built at construction.
    """

    def __init__(self, documents: dict[str, Document], edges: dict[str, dict[str, float]]):
        self.documents = list(documents.values())
        self.graph = None
        self.entities = None
        self.build_seconds = 0.0
        self._documents = documents
        self._edges = edges
        self.lexical: LexicalSearcher | None = ListLexical(documents)
        self.closed = 0

    @property
    def by_path(self) -> dict[str, Document]:
        return dict(self._documents)

    def adjacency(self, entity_edges=True, curated_links=True,
                  mentions_per_document=8, mention_scale=0.05):
        if not curated_links:
            return {}
        return {node: dict(edges) for node, edges in self._edges.items()}

    def transitions(self, entity_edges=True, curated_links=True,
                    mentions_per_document=8, self_weight=0.15, mention_scale=0.05):
        return transitions(
            self.adjacency(entity_edges, curated_links, mentions_per_document, mention_scale),
            self_weight,
        )

    def calibration(self) -> LexicalCalibration:
        return LexicalCalibration(scores=())

    def close(self) -> None:
        self.closed += 1


class StubVectors:
    """A `VectorSearcher` that returns a fixed ranking. No file, no provider."""

    def __init__(self, hits, covered: int) -> None:
        self._hits = hits
        self._covered = covered
        self.calls = 0

    def search(self, vector, n):
        self.calls += 1
        return list(self._hits)[:n]

    def count(self):
        return self._covered, len(self._hits)


DOCS = {
    "a1b2c3d4_grid-storage": Document(
        path="a1b2c3d4_grid-storage", title="Grid Storage",
        content="Grid storage smooths demand. See flow batteries.", kind="wiki"),
    "e5f6a7b8_flow-battery": Document(
        path="e5f6a7b8_flow-battery", title="Flow Battery",
        content="A flow battery stores energy in liquid electrolyte tanks.", kind="wiki"),
    "c9d0e1f2_electrolyte": Document(
        path="c9d0e1f2_electrolyte", title="Electrolyte",
        content="Vanadium electrolyte is the costly part of a flow battery.", kind="wiki"),
}
EDGES = {
    "a1b2c3d4_grid-storage": {"e5f6a7b8_flow-battery": 3.0},
    "e5f6a7b8_flow-battery": {"a1b2c3d4_grid-storage": 3.0, "c9d0e1f2_electrolyte": 4.0},
    "c9d0e1f2_electrolyte": {"e5f6a7b8_flow-battery": 4.0},
}


def wiki_documents() -> list[Document]:
    return [
        Document(path="wiki/station-keeping.md", title="Station Keeping",
                 content="Station keeping holds a slot. See [[Ion Thruster]].", kind="wiki"),
        Document(path="wiki/ion-thruster.md", title="Ion Thruster",
                 content="An ion thruster gives low thrust for station keeping.", kind="wiki"),
        Document(path="wiki/orbital-decay.md", title="Orbital Decay",
                 content="Orbital decay is altitude lost to atmospheric drag.", kind="wiki"),
    ]


# ── the protocol ───────────────────────────────────────────────────────────

def test_the_default_index_is_a_corpus_index():
    assert isinstance(build_index(wiki_documents()), CorpusIndex)


def test_a_store_that_derives_nothing_is_also_a_corpus_index():
    assert isinstance(StoredCorpus(DOCS, EDGES), CorpusIndex)


def test_search_index_is_the_old_name_pointing_at_the_new_class():
    """⛔ `ragharness` annotates on `SearchIndex` from another repository."""
    assert SearchIndex is InMemoryIndex


# ── the kit passes on both implementations ─────────────────────────────────

def test_the_default_index_conforms():
    assert_corpus_index(build_index(wiki_documents()), name="InMemoryIndex",
                        symmetric_adjacency=True)


def test_an_empty_default_index_conforms():
    """The abstention fence is asked for on a cold corpus, so it must answer."""
    assert_corpus_index(build_index([]), name="InMemoryIndex([])")


def test_the_minimal_second_implementation_conforms():
    """G3: a kit with one implementation describes that implementation."""
    assert_corpus_index(StoredCorpus(DOCS, EDGES), name="StoredCorpus",
                        symmetric_adjacency=True)


# ── the kit is not vacuous: each rule catches its own violation ────────────

def test_the_kit_catches_a_document_missing_from_by_path():
    corpus = StoredCorpus(DOCS, EDGES)
    corpus.documents = corpus.documents + [
        Document(path="zz_unlisted", title="Unlisted", content="nothing points here")
    ]
    assert any("by-path-covers-documents" in f for f in check_corpus_index(corpus))


def test_the_kit_catches_a_transitions_row_that_is_not_a_distribution():
    corpus = StoredCorpus(DOCS, EDGES)
    corpus.transitions = lambda **_: {"a1b2c3d4_grid-storage": [("e5f6a7b8_flow-battery", 0.5)]}
    assert any("transitions-rows-sum-to-one" in f for f in check_corpus_index(corpus))


def test_the_kit_catches_a_calibration_that_raises():
    corpus = StoredCorpus(DOCS, EDGES)

    def boom():
        raise RuntimeError("no fence on a cold store")

    corpus.calibration = boom
    assert any("calibration-is-usable" in f for f in check_corpus_index(corpus))


def test_the_kit_catches_a_non_positive_edge():
    corpus = StoredCorpus(DOCS, {"a1b2c3d4_grid-storage": {"e5f6a7b8_flow-battery": 0.0}})
    assert any("adjacency-is-finite-and-positive" in f for f in check_corpus_index(corpus))


def test_the_kit_catches_an_object_that_is_not_an_index_at_all():
    failures = check_corpus_index(object())
    assert failures and "satisfies-protocol" in failures[0]


def test_the_kit_catches_a_close_that_is_not_idempotent():
    corpus = StoredCorpus(DOCS, EDGES)
    state = {"open": True}

    def close_once():
        if not state["open"]:
            raise ValueError("already closed")
        state["open"] = False

    corpus.close = close_once
    assert any("close-is-idempotent" in f for f in check_corpus_index(corpus, close=True))


def test_assert_corpus_index_names_every_failure():
    corpus = StoredCorpus(DOCS, {"a1b2c3d4_grid-storage": {"e5f6a7b8_flow-battery": -1.0}})
    with pytest.raises(AssertionError, match="adjacency-is-finite-and-positive"):
        assert_corpus_index(corpus, name="StoredCorpus")


# ── the ladder actually ranks the second implementation ────────────────────

def test_search_index_ranks_a_corpus_it_did_not_build():
    """The point of the whole seam: no project, no directory, no file."""
    response = search_index(StoredCorpus(DOCS, EDGES), "vanadium", top_k=5)
    assert [result.path for result in response.results][:1] == ["c9d0e1f2_electrolyte"]
    assert response.lanes.lexical and response.lanes.graph
    assert response.mode == "keyword"
    # A store that derives no WikiGraph and no EntityIndex is still a corpus:
    # `adjacency()` is what diffusion reads, and those two are only reported.
    assert response.graph is None and response.entities is None


def test_the_graph_lane_reaches_a_document_the_lexical_lane_did_not():
    """Diffusion over edges the corpus supplied as data rather than as a graph."""
    response = search_index(StoredCorpus(DOCS, EDGES), "vanadium", top_k=5)
    reached = {result.path for result in response.results}
    assert "e5f6a7b8_flow-battery" in reached
    assert response.graph_hits >= 1


def test_search_index_and_search_agree_on_an_in_memory_corpus():
    """`search` is a delegator, so the two must produce the same ranking."""
    index = build_index(wiki_documents())
    through_project = __import__("llmwiki.retrieval", fromlist=["search"]).search(
        None, "station keeping", index=index, embedding_config=None
    )
    directly = search_index(index, "station keeping")
    assert [(r.path, r.score) for r in through_project.results] == [
        (r.path, r.score) for r in directly.results
    ]


def test_search_index_refuses_nothing_on_an_empty_query():
    assert search_index(build_index(wiki_documents()), "   ").results == []


def test_a_caller_may_pass_its_own_stage_map():
    stages = {"rewrite": 4.0}
    response = search_index(build_index(wiki_documents()), "station keeping", stages=stages)
    assert response.stage_ms["rewrite"] == 4.0
    assert "lexical" in response.stage_ms


# ── U3: the vector lane takes a searcher, not a path ──────────────────────

def test_the_vector_lane_runs_against_an_injected_searcher(monkeypatch):
    from llmwiki import embeddings
    from llmwiki.config import EmbeddingConfig
    from llmwiki.embeddings import ChunkHit

    monkeypatch.setattr(embeddings, "embed_query", lambda *a, **k: [0.1, 0.2, 0.3])
    vectors = StubVectors(
        [ChunkHit(page_id="c9d0e1f2_electrolyte", chunk_index=0, heading_path="",
                  text="vanadium", score=0.9)],
        covered=3,
    )
    response = search_index(
        StoredCorpus(DOCS, EDGES), "what is costly", top_k=5, vectors=vectors,
        embedding_config=EmbeddingConfig(enabled=True, model="stub"),
    )
    assert vectors.calls == 1
    assert response.lanes.vector and response.mode in {"hybrid", "vector"}
    assert response.vector_hits == 1


def test_a_configured_embedder_with_no_searcher_says_so():
    """The note that used to fire on a missing `vectors.db` now fires on `None`."""
    from llmwiki.config import EmbeddingConfig

    response = search_index(
        StoredCorpus(DOCS, EDGES), "what is costly",
        embedding_config=EmbeddingConfig(enabled=True, model="stub"), vectors=None,
    )
    assert any("no index exists yet" in note for note in response.notes)
    assert not response.lanes.vector


def test_an_unconfigured_embedder_says_something_different():
    from llmwiki.config import EmbeddingConfig

    response = search_index(
        StoredCorpus(DOCS, EDGES), "what is costly",
        embedding_config=EmbeddingConfig(enabled=False, model=""), vectors=None,
    )
    assert any("no embedding model" in note for note in response.notes)


def test_search_index_never_closes_a_searcher_it_was_handed(monkeypatch):
    """Lifecycle belongs to the caller; a closed store breaks the next turn."""
    from llmwiki import embeddings
    from llmwiki.config import EmbeddingConfig
    from llmwiki.embeddings import ChunkHit

    monkeypatch.setattr(embeddings, "embed_query", lambda *a, **k: [0.1])
    vectors = StubVectors([ChunkHit("e5f6a7b8_flow-battery", 0, "", "t", 0.5)], covered=3)
    for _ in range(3):
        search_index(StoredCorpus(DOCS, EDGES), "energy", vectors=vectors,
                     embedding_config=EmbeddingConfig(enabled=True, model="stub"))
    assert vectors.calls == 3


# ── U4: naming ─────────────────────────────────────────────────────────────

def test_the_default_naming_is_todays_behaviour():
    document = Document(path="wiki/station-keeping.md", title="Station Keeping", content="x")
    assert DEFAULT_NAMING.key(document) == "wiki/station-keeping.md"
    assert DEFAULT_NAMING.title_field(document) == "Station Keeping station keeping"
    assert DEFAULT_NAMING.is_page(document) is True
    assert DEFAULT_NAMING.surface_forms(document) == {
        "Station Keeping", "station-keeping", "station keeping"
    }
    assert "wiki/station-keeping.md" in DEFAULT_NAMING.aliases(document)


def test_surface_forms_are_not_aliases():
    """⛔ A link target is addressed on purpose; a surface form is typed by accident."""
    document = Document(path="wiki/station-keeping.md", title="Station Keeping", content="x")
    assert "wiki/station-keeping.md" not in DEFAULT_NAMING.surface_forms(document)


def test_a_title_only_naming_keeps_an_id_out_of_the_lexical_title_column():
    """The reason Envolved needs the hook: `{uuid}_{name}` in a 10x-weighted field."""
    documents = [
        Document(path="3f9a1c04_quarterly-report", title="Quarterly Report",
                 content="revenue rose"),
        Document(path="8b2e77d1_annual-review", title="Annual Review",
                 content="3f9a1c04 revenue rose sharply and materially"),
    ]
    naming = DocumentNaming(title_field=lambda d: d.title)
    ranked = search_index(build_index(documents, naming=naming), "3f9a1c04", top_k=5)
    default = search_index(build_index(documents), "3f9a1c04", top_k=5)

    # Under the default the uuid is in the title column of the document it names,
    # so a query that is only a uuid retrieves it; under the Envolved naming the
    # uuid is not searchable text at all and only the prose mention matches.
    assert [r.path for r in default.results][:1] == ["3f9a1c04_quarterly-report"]
    assert [r.path for r in ranked.results] == ["8b2e77d1_annual-review"]


def test_a_corpus_with_no_pages_still_builds_and_ranks():
    """`is_page` defaults to `kind == "wiki"`; a corpus of raw sources overrides it."""
    documents = [
        Document(path="raw/a.txt", title="A", content="alpha beta", kind="source"),
        Document(path="raw/b.txt", title="B", content="beta gamma", kind="source"),
    ]
    naming = DocumentNaming(is_page=lambda d: True)
    index = build_index(documents, naming=naming)
    assert index.entities is not None and index.entities.mentions
    assert search_index(index, "beta").results


def test_naming_is_comparable_so_a_caller_can_assert_on_it():
    assert DocumentNaming() == DEFAULT_NAMING
    assert DocumentNaming(title_field=lambda d: d.title) != DEFAULT_NAMING


# ── U5: the index-time primitives are public API ──────────────────────────

def test_the_index_time_primitives_are_importable_from_the_package_root():
    import llmwiki

    for name in ("chunk_markdown", "ChunkingOptions", "split_source_into_semantic_chunks",
                 "CorpusIndex", "DocumentNaming", "search_index"):
        assert name in llmwiki.__all__, name
        assert hasattr(llmwiki, name), name


def test_hub_pruning_is_reachable_without_an_entity_index():
    """Envolved holds its postings in SQLite and has no `EntityIndex` to pass."""
    from llmwiki.retrieval import hub_entities

    postings = {"everywhere": {f"d{i}": 1 for i in range(9)},
                "rare": {"d0": 1}}
    assert hub_entities(postings, document_count=10) == {"everywhere"}


def test_prune_hubs_removes_exactly_what_hub_entities_names():
    """The published decision and the published wrapper cannot drift apart."""
    from llmwiki.retrieval import hub_entities, prune_hubs
    from llmwiki.retrieval.entities import EntityIndex

    mentions = {
        "everywhere": {f"d{i}": 1 for i in range(9)},
        "common": {"d0": 1, "d1": 1},
        "rare": {"d0": 1},
    }
    index = EntityIndex(mentions=mentions, documents=10)
    hubs = hub_entities(mentions, 10)
    pruned = prune_hubs(index)

    assert hubs == {"everywhere"}
    assert set(pruned.pruned) == hubs
    assert set(pruned.mentions) == set(mentions) - hubs


def test_pruning_an_index_twice_is_a_no_op():
    """`build_entity_index` prunes on the way out, so its result is already clean."""
    from llmwiki.retrieval import build_entity_index, hub_entities, prune_hubs

    documents = [
        Document(path=f"wiki/d{i}.md", title=f"Doc {i}",
                 content=f"ubiquitous term appears here unique{i}")
        for i in range(10)
    ]
    documents.append(Document(path="wiki/ubiquitous.md", title="Ubiquitous", content="self"))
    index = build_entity_index(documents)
    assert hub_entities(index.mentions, index.documents) == set()
    assert prune_hubs(index).mentions.keys() == index.mentions.keys()
