"""Retrieval: keyword scoring, the graph model, and expansion."""

from __future__ import annotations

from llmwiki.retrieval import build_graph, load_documents, related_pages, search, tokenize_query
from llmwiki.retrieval.graph import graph_result_quota, normalize_alias
from llmwiki.retrieval.keyword import Document, build_snippet, score_document


def _page(project, path, title, body, sources=(), page_type="concept"):
    source_field = ", ".join(f'"{s}"' for s in sources)
    project.write(
        path,
        f"---\ntype: {page_type}\ntitle: {title}\nsources: [{source_field}]\n---\n\n# {title}\n\n{body}\n",
    )


def test_tokenizer_drops_stopwords_and_expands_cjk():
    assert tokenize_query("What is the Transformer?") == ["transformer"]
    tokens = tokenize_query("上下文窗口")
    assert "上下" in tokens and "上下文窗口" in tokens


def test_scoring_weights_title_over_body():
    body_hit = Document(path="wiki/a.md", title="Unrelated", content="mentions attention once")
    title_hit = Document(path="wiki/b.md", title="Attention", content="unrelated body text")
    tokens = ["attention"]
    body_score = score_document(body_hit, tokens, "attention", "attention").score
    title_score = score_document(title_hit, tokens, "attention", "attention").score
    assert title_score > body_score


def test_non_matching_document_is_not_a_result():
    document = Document(path="wiki/a.md", title="Bananas", content="nothing relevant here")
    assert score_document(document, ["quantum"], "quantum", "quantum") is None


def test_snippet_centres_on_the_match():
    content = "x" * 300 + "NEEDLE" + "y" * 300
    snippet = build_snippet(content, "needle")
    assert "NEEDLE" in snippet
    assert snippet.startswith("...") and snippet.endswith("...")


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


def test_graph_expansion_surfaces_a_page_with_no_keyword_match(wiki):
    _page(wiki, "wiki/concepts/photosynthesis.md", "Photosynthesis", "Plants use [[chlorophyll]].")
    # Deliberately contains none of the query's terms.
    _page(wiki, "wiki/entities/chlorophyll.md", "Chlorophyll", "A green pigment.", (), "entity")

    response = search(wiki, "photosynthesis", top_k=5, include_sources=False)
    paths = [result.path for result in response.results]
    assert "wiki/entities/chlorophyll.md" in paths
    assert response.graph_hits == 1
    reached = next(r for r in response.results if r.path == "wiki/entities/chlorophyll.md")
    assert reached.graph_related_to == ["Photosynthesis"]


def test_wiki_pages_outrank_raw_sources_on_similar_matches(wiki):
    _page(wiki, "wiki/concepts/storage.md", "Grid Storage", "Grid storage costs are falling.")
    wiki.sources_dir.mkdir(parents=True, exist_ok=True)
    (wiki.sources_dir / "notes.txt").write_text("Grid storage costs are falling.")

    response = search(wiki, "grid storage", top_k=5)
    assert response.results[0].kind == "wiki"


def test_alias_normalization_matches_link_forms():
    assert normalize_alias("Chain of Thought") == "chain-of-thought"
    assert normalize_alias("wiki/concepts/chain-of-thought.md") == "wiki/concepts/chain-of-thought"
    assert normalize_alias("foo#section") == "foo"


def test_graph_quota_shrinks_as_vector_coverage_grows():
    full = graph_result_quota(20, vector_hits=0)
    partial = graph_result_quota(20, vector_hits=20)
    assert full > partial >= 1
    assert graph_result_quota(1, 0) == 0
