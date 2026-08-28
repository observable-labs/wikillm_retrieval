"""Retrieval: lexical ranking, the graph model, entities, and diffusion."""

from __future__ import annotations

from llmwiki.retrieval import (
    build_entity_index,
    build_graph,
    build_index,
    load_documents,
    open_index,
    related_pages,
    search,
    tokenize_query,
)
from llmwiki.retrieval.entities import ENTITY_PREFIX, surface_forms
from llmwiki.retrieval.graph import normalize_alias
from llmwiki.retrieval.keyword import Document, build_snippet, score_document
from llmwiki.retrieval.lexical import LexicalIndex, extract_headings, match_expression
from llmwiki.retrieval.pipeline import RetrievalOptions
from llmwiki.retrieval.ppr import personalized_pagerank, rank_by_ppr, seed_masses


def _page(project, path, title, body, sources=(), page_type="concept"):
    source_field = ", ".join(f'"{s}"' for s in sources)
    project.write(
        path,
        f"---\ntype: {page_type}\ntitle: {title}\nsources: [{source_field}]\n---\n\n# {title}\n\n{body}\n",
    )


def _documents(*pairs):
    return [
        Document(path=path, title=title, content=content)
        for path, title, content in pairs
    ]


# ── tokenizer and snippets ────────────────────────────────────────────────

def test_tokenizer_drops_stopwords_and_expands_cjk():
    assert tokenize_query("What is the Transformer?") == ["transformer"]
    tokens = tokenize_query("上下文窗口")
    assert "上下" in tokens and "上下文窗口" in tokens


def test_snippet_centres_on_the_match():
    content = "x" * 300 + "NEEDLE" + "y" * 300
    snippet = build_snippet(content, "needle")
    assert "NEEDLE" in snippet
    assert snippet.startswith("...") and snippet.endswith("...")


# ── the substring scorer, still the CJK lane ──────────────────────────────

def test_substring_scorer_weights_title_over_body():
    body_hit = Document(path="wiki/a.md", title="Unrelated", content="mentions attention once")
    title_hit = Document(path="wiki/b.md", title="Attention", content="unrelated body text")
    tokens = ["attention"]
    body_score = score_document(body_hit, tokens, "attention", "attention").score
    title_score = score_document(title_hit, tokens, "attention", "attention").score
    assert title_score > body_score


def test_non_matching_document_is_not_a_result():
    document = Document(path="wiki/a.md", title="Bananas", content="nothing relevant here")
    assert score_document(document, ["quantum"], "quantum", "quantum") is None


# ── L2: BM25 ──────────────────────────────────────────────────────────────

def test_bm25_weights_a_rare_term_over_a_ubiquitous_one():
    """The defect the lexical rewrite exists to remove.

    `spacecraft` is in every document and says nothing about which one answers
    the query; `xenon` is in one. The old scorer counted both as one token
    present and ranked them identically.
    """
    documents = _documents(
        ("wiki/a.md", "Alpha", "spacecraft " * 20 + "nothing else here"),
        ("wiki/b.md", "Beta", "spacecraft propulsion uses xenon as propellant"),
        ("wiki/c.md", "Gamma", "spacecraft " * 20 + "orbital mechanics"),
    )
    with LexicalIndex(documents) as index:
        hits = index.search(["spacecraft", "xenon"], 10)
    assert hits[0].path == "wiki/b.md"


def test_bm25_does_not_match_a_substring():
    documents = _documents(("wiki/a.md", "Cartesian", "cartesian coordinates"))
    with LexicalIndex(documents) as index:
        assert index.search(["art"], 10) == []


def test_a_page_outranks_the_source_it_was_written_from():
    """What `SOURCE_SCORE_FACTOR` used to force, BM25 produces unaided.

    The factor existed because the substring scorer had no length
    normalization. Removing it was worth 0.43 MRR on questions whose gold *is* a
    raw source, and this is the case it was protecting.
    """
    documents = [
        Document(path="wiki/a.md", title="Grid Storage", content="grid storage costs", kind="wiki"),
        Document(path="raw/sources/n.md", title="n.md", content="grid storage costs", kind="source"),
    ]
    with LexicalIndex(documents) as index:
        hits = index.search(["grid", "storage"], 10)
    assert hits[0].path == "wiki/a.md"


def test_fts5_query_is_quoted_not_interpolated():
    assert match_expression(["a", 'b"c']) == '"a" OR "b""c"'
    assert match_expression([]) == "", "an empty query must not become a MATCH-all"
    assert match_expression(["and", "or"]) == '"and" OR "or"', "operators are neutralised"


def test_headings_are_extracted_as_their_own_column():
    assert extract_headings("# Title\n\ntext\n\n## Section two\n") == "Title\nSection two"


# ── L4: entities ──────────────────────────────────────────────────────────

def test_entity_index_links_documents_that_share_no_wikilink():
    """Two hops through an entity node, which is the bridge a link graph misses."""
    documents = _documents(
        ("wiki/orgs/helios.md", "Helios Propulsion", "A supplier of thrusters."),
        ("wiki/missions/borealis.md", "Borealis-2", "Flies units from Helios Propulsion."),
        ("wiki/concepts/orbit.md", "Orbit", "Unrelated content entirely."),
    )
    edges = build_entity_index(documents).edges()
    helios = ENTITY_PREFIX + "wiki/orgs/helios.md"
    assert helios in edges["wiki/missions/borealis.md"]
    assert "wiki/orgs/helios.md" in edges[helios], "the entity is anchored to the page defining it"
    assert helios not in edges["wiki/concepts/orbit.md"]


def test_entity_nodes_never_occupy_a_result_slot():
    """The bipartite graph carries mass through entity nodes; the window holds
    documents. Collapsing the two made a page compete with its own mentions."""
    fused = [("wiki/a.md", 0.02)]
    adjacency = {
        "wiki/a.md": {ENTITY_PREFIX + "e": 1.0},
        ENTITY_PREFIX + "e": {"wiki/a.md": 1.0, "wiki/b.md": 1.0},
        "wiki/b.md": {ENTITY_PREFIX + "e": 1.0},
    }
    ranked = rank_by_ppr(fused, adjacency, limit=5, keep={"wiki/a.md", "wiki/b.md"})
    assert [path for path, _ in ranked] == ["wiki/a.md", "wiki/b.md"]


def test_entity_scan_respects_word_boundaries():
    documents = _documents(
        ("wiki/art.md", "Art", "A page about art."),
        ("wiki/other.md", "Other", "Cartesian coordinates and cartography."),
    )
    index = build_entity_index(documents)
    assert "wiki/other.md" not in index.mentions.get("wiki/art.md", {})


def test_hub_entities_are_pruned():
    """A title mentioned by most of the corpus carries no signal about which
    document is relevant, and it is exactly the node that misdirects PageRank."""
    documents = _documents(
        ("wiki/hub.md", "Overview", "The index of everything."),
        *[(f"wiki/p{i}.md", f"Page {i}", "See Overview for context.") for i in range(6)],
    )
    index = build_entity_index(documents)
    assert "wiki/hub.md" in index.pruned
    assert "wiki/hub.md" not in index.mentions


def test_surface_forms_read_a_hyphenated_stem_as_words():
    document = Document(path="wiki/station-keeping.md", title="Station Keeping", content="")
    assert "station keeping" in {form.lower() for form in surface_forms(document)}


# ── S2: seeded PPR ────────────────────────────────────────────────────────

def test_ppr_with_no_edges_preserves_the_fused_order():
    """The property that makes the lane safe to leave on: an empty graph is a
    no-op, so the graph can reorder but never destroy."""
    fused = [("a", 0.03), ("b", 0.02), ("c", 0.01)]
    assert [path for path, _ in rank_by_ppr(fused, {}, limit=3)] == ["a", "b", "c"]


def test_ppr_never_drops_a_retrieved_document():
    fused = [("a", 0.03), ("b", 0.02), ("c", 0.01)]
    adjacency = {"a": {"z": 1.0}, "z": {"a": 1.0}}
    ranked = [path for path, _ in rank_by_ppr(fused, adjacency, limit=10, seed_count=1)]
    assert set(fused[0][0]) <= set(ranked)
    assert {"a", "b", "c"} <= set(ranked), "retrieval's own results must survive diffusion"


def test_ppr_reaches_a_document_the_query_never_matched():
    fused = [("a", 0.03)]
    adjacency = {"a": {"z": 1.0}, "z": {"a": 1.0}}
    ranked = [path for path, _ in rank_by_ppr(fused, adjacency, limit=5)]
    assert "z" in ranked


def test_ppr_mass_is_conserved():
    scores = personalized_pagerank(
        {"a": 1.0}, {"a": {"b": 1.0}, "b": {"a": 1.0, "c": 1.0}, "c": {"b": 1.0}}
    )
    assert abs(sum(scores.values()) - 1.0) < 1e-9


def test_seed_masses_radiate_only_from_the_head():
    masses = seed_masses([("a", 0.5), ("b", 0.4), ("c", 0.3)], seed_count=2, tail_weight=0.0)
    assert set(masses) == {"a", "b"}


# ── the graph model ───────────────────────────────────────────────────────

def test_graph_relevance_signals(wiki):
    _page(wiki, "wiki/concepts/attention.md", "Attention", "See [[transformer]].", ["paper.pdf"])
    _page(wiki, "wiki/entities/transformer.md", "Transformer", "Built on attention.", ["paper.pdf"], "entity")
    _page(wiki, "wiki/concepts/rag.md", "RAG", "Unrelated.", ["other.pdf"])

    graph = build_graph(load_documents(wiki, include_sources=False))
    related = related_pages(graph, "wiki/concepts/attention.md")
    assert related[0][0] == "wiki/entities/transformer.md"
    # direct link (3.0) + shared source (4.0), different types so no affinity
    assert related[0][1] == 3.0 + 4.0
    assert "direct link" in related[0][2]
    assert any("shares source" in reason for reason in related[0][2])


def test_alias_normalization_matches_link_forms():
    assert normalize_alias("Chain of Thought") == "chain-of-thought"
    assert normalize_alias("wiki/concepts/chain-of-thought.md") == "wiki/concepts/chain-of-thought"
    assert normalize_alias("foo#section") == "foo"


def test_relevance_is_the_edge_weight_the_adjacency_uses(wiki):
    _page(wiki, "wiki/a.md", "Alpha", "Links to [[Beta]].", ["shared.pdf"])
    _page(wiki, "wiki/b.md", "Beta", "Nothing in common textually.", ["shared.pdf"])
    index = build_index(load_documents(wiki, include_sources=False))
    weight = index.adjacency()["wiki/a.md"]["wiki/b.md"]
    assert weight >= 3.0 + 4.0, "a curated link with a shared source is the strongest edge"


# ── the pipeline end to end ───────────────────────────────────────────────

def test_graph_expansion_surfaces_a_page_with_no_keyword_match(wiki):
    _page(wiki, "wiki/concepts/photosynthesis.md", "Photosynthesis", "Plants use [[chlorophyll]].")
    # Deliberately contains none of the query's terms.
    _page(wiki, "wiki/entities/chlorophyll.md", "Chlorophyll", "A green pigment.", (), "entity")

    response = search(wiki, "photosynthesis", top_k=5, include_sources=False)
    paths = [result.path for result in response.results]
    assert "wiki/entities/chlorophyll.md" in paths
    assert response.graph_hits >= 1
    reached = next(r for r in response.results if r.path == "wiki/entities/chlorophyll.md")
    assert reached.graph_related_to == ["Photosynthesis"]


def test_graph_expansion_does_not_evict_a_keyword_hit(wiki):
    """The displacement defect, pinned.

    The retired `blend_graph_results` reserved 15-30% of the window
    unconditionally and took those slots from the tail of the ranked list, so a
    graph neighbour evicted a document the query actually matched *even when
    there was room for both*. Diffusion reorders one set and appends; with a
    window at least as wide as the candidates, every lexical match survives.
    """
    for i in range(5):
        _page(wiki, f"wiki/m{i}.md", f"Match {i}", "xenon propellant discussion")
    _page(wiki, "wiki/hub.md", "Hub", "Links to [[Match 0]].")

    response = search(wiki, "xenon propellant", top_k=20, include_sources=False)
    paths = {result.path for result in response.results}
    assert {f"wiki/m{i}.md" for i in range(5)} <= paths


def test_wiki_pages_outrank_raw_sources_on_similar_matches(wiki):
    _page(wiki, "wiki/concepts/storage.md", "Grid Storage", "Grid storage costs are falling.")
    wiki.sources_dir.mkdir(parents=True, exist_ok=True)
    (wiki.sources_dir / "notes.txt").write_text("Grid storage costs are falling.")

    response = search(wiki, "grid storage", top_k=5)
    assert response.results[0].kind == "wiki"


def test_mode_reports_what_ran_not_what_was_hit(wiki):
    """`_mode` used to return "hybrid" whenever a graph neighbour appeared, so a
    keyword-only run reported itself as hybrid."""
    _page(wiki, "wiki/a.md", "Alpha", "Links to [[Beta]].")
    _page(wiki, "wiki/b.md", "Beta", "Something else.")
    response = search(wiki, "alpha", top_k=5, include_sources=False)
    assert response.mode == "keyword"
    assert response.lanes.as_dict() == {"lexical": True, "vector": False, "graph": True}


def test_the_index_cache_never_serves_a_stale_corpus(wiki):
    """The cache is keyed on a fingerprint of the corpus, so an edit invalidates
    it on the next query with no explicit invalidation call anywhere.

    Worth a test out of proportion to its size: a stale index does not fail, it
    quietly answers from a corpus that no longer exists, and every measurement
    taken through it is wrong in a way nothing downstream can detect.
    """
    _page(wiki, "wiki/a.md", "Alpha", "original body about turbines")
    first = search(wiki, "turbines", top_k=5, include_sources=False)
    assert [r.path for r in first.results] == ["wiki/a.md"]

    _page(wiki, "wiki/b.md", "Beta", "a second page also about turbines")
    second = search(wiki, "turbines", top_k=5, include_sources=False)
    assert {r.path for r in second.results} == {"wiki/a.md", "wiki/b.md"}

    _page(wiki, "wiki/a.md", "Alpha", "rewritten, now about hydrofoils")
    third = search(wiki, "turbines", top_k=5, include_sources=False)
    assert [r.path for r in third.results] == ["wiki/b.md"]

    (wiki.root / "wiki" / "b.md").unlink()
    fourth = search(wiki, "turbines", top_k=5, include_sources=False)
    assert fourth.results == []


def test_the_index_cache_is_actually_reused(wiki):
    """...and when nothing changed, it is not rebuilt. Otherwise the test above
    passes for the trivial reason that there is no cache."""
    _page(wiki, "wiki/a.md", "Alpha", "body about turbines")
    first = open_index(wiki, include_sources=False)
    assert open_index(wiki, include_sources=False) is first

    _page(wiki, "wiki/a.md", "Alpha", "a different body about turbines")
    assert open_index(wiki, include_sources=False) is not first


def test_lanes_can_be_switched_off_for_an_ablation(wiki):
    _page(wiki, "wiki/a.md", "Alpha", "Links to [[Beta]].")
    _page(wiki, "wiki/b.md", "Beta", "Something else.")
    options = RetrievalOptions(graph_ppr=False)
    response = search(wiki, "alpha", top_k=5, include_sources=False, options=options)
    assert response.lanes.graph is False
    assert response.graph_hits == 0
