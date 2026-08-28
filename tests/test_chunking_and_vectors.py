"""Chunking invariants and the optional vector lane."""

from __future__ import annotations

import pytest

from llmwiki.chunking import ChunkingOptions, chunk_markdown, split_source_into_semantic_chunks
from llmwiki.embeddings import VectorStore, group_by_page
from llmwiki.retrieval import search


def test_frontmatter_is_stripped_from_embedding_chunks():
    doc = "---\ntitle: Secret Metadata\ntags: [a]\n---\n\n# Body\n\nThe actual content.\n"
    chunks = chunk_markdown(doc)
    assert chunks
    assert all("Secret Metadata" not in chunk.text for chunk in chunks)


def test_code_fences_and_tables_are_never_split():
    fence = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n```"
    table = "\n".join(["| a | b |", "|---|---|"] + [f"| {i} | {i * 2} |" for i in range(200)])
    doc = f"# Doc\n\n{fence}\n\n{table}\n"
    chunks = chunk_markdown(doc, ChunkingOptions(target_chars=200, max_chars=300, min_chars=50))

    fence_chunks = [c for c in chunks if "```python" in c.text]
    assert len(fence_chunks) == 1, "a code block must not be torn across chunks"
    assert fence_chunks[0].text.count("```") == 2
    table_chunks = [c for c in chunks if "| a | b |" in c.text]
    assert len(table_chunks) == 1, "a table must not be torn across chunks"


def test_chunks_carry_heading_breadcrumbs():
    doc = "# Top\n\nIntro.\n\n## Middle\n\n### Deep\n\nLeaf content here.\n"
    chunks = chunk_markdown(doc, ChunkingOptions(target_chars=40, max_chars=60, min_chars=10))
    leaf = next(c for c in chunks if "Leaf content" in c.text)
    assert leaf.heading_path == "# Top > ## Middle > ### Deep"


def test_chunking_is_deterministic():
    doc = "# A\n\n" + ("sentence one. sentence two. " * 80)
    assert [c.text for c in chunk_markdown(doc)] == [c.text for c in chunk_markdown(doc)]


def test_source_chunks_overlap_and_are_ordered():
    doc = "\n\n".join(f"## Section {i}\n\n" + ("word " * 500) for i in range(5))
    chunks = split_source_into_semantic_chunks(doc, 2000, 200)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.total == len(chunks) for c in chunks)
    assert chunks[0].overlap_before == ""
    assert chunks[1].overlap_before, "later chunks carry a tail of the previous one"
    assert "continues from previous section" in chunks[1].text


def test_vector_store_roundtrip_and_pruning(tmp_path):
    with VectorStore(tmp_path / "v.db") as store:
        store.upsert_page("wiki/a.md", "A", "hash-a", "m", [(0, "# H", "alpha", [1.0, 0.0])])
        store.upsert_page("wiki/b.md", "B", "hash-b", "m", [(0, "", "beta", [0.0, 1.0])])
        assert store.count() == (2, 2)
        assert store.page_hash("wiki/a.md") == "hash-a"

        hits = store.search([1.0, 0.0], top_k=5)
        assert hits[0].page_id == "wiki/a.md"
        assert hits[0].score == pytest.approx(1.0)

        # A re-embed replaces a page's chunks rather than accumulating them.
        store.upsert_page("wiki/a.md", "A", "hash-a2", "m", [(0, "", "gamma", [1.0, 0.0])])
        assert store.count() == (2, 2)

        assert store.prune({"wiki/a.md"}) == 1
        assert store.count() == (1, 1)


def test_mismatched_dimensions_are_ignored_not_crashed(tmp_path):
    with VectorStore(tmp_path / "v.db") as store:
        store.upsert_page("wiki/old.md", "Old", "h", "old-model", [(0, "", "x", [1.0, 0.0, 0.0])])
        store.upsert_page("wiki/new.md", "New", "h", "new-model", [(0, "", "y", [1.0, 0.0])])
        hits = store.search([1.0, 0.0], top_k=5)
    assert [hit.page_id for hit in hits] == ["wiki/new.md"]


def test_a_page_scores_its_best_chunk_and_chunk_count_buys_nothing():
    """Coverage must not outrank quality, however much of it there is.

    The rule this replaces added a capped share of the remaining chunk scores.
    Cosine similarities sit in a narrow band, so the cap was reached at three
    chunks and every multi-chunk page saturated at 1.00 — which on the atlas
    corpus put the fourteen raw sources at the head of the vector ranking for
    every query and cost recall@1 0.705 -> 0.023.
    """
    from llmwiki.embeddings import ChunkHit

    def score(chunk_scores):
        return group_by_page(
            [ChunkHit("wiki/a.md", i, "", "t", s) for i, s in enumerate(chunk_scores)]
        )[0].score

    assert score([0.9]) > score([0.3])
    assert score([0.3] * 500) == score([0.3])
    # One excellent chunk beats any amount of mediocre coverage.
    assert score([0.85]) > score([0.6] * 50)


def test_the_vector_ranking_does_not_depend_on_how_deep_the_scan_went():
    """Scan depth is a latency constant; it must not be a quality constant.

    `_vector_lane` scans `max(3 * max(2k, 20), 30)` chunks and the dense
    baseline scans `max(30, 3k)`. Under the old tail term those two produced
    different page orders from the same store, which is how the assembled
    pipeline came to score below a baseline built from one of its own lanes.
    """
    from llmwiki.embeddings import ChunkHit

    chunks = [
        ChunkHit("wiki/deep.md", 0, "", "t", 0.61),
        ChunkHit("wiki/sharp.md", 0, "", "t", 0.60),
    ] + [ChunkHit("wiki/deep.md", i, "", "t", 0.59) for i in range(1, 8)]

    shallow = [page.page_id for page in group_by_page(chunks[:2], top_k=2)]
    deep = [page.page_id for page in group_by_page(chunks, top_k=2)]
    assert shallow == deep == ["wiki/deep.md", "wiki/sharp.md"]


def test_search_falls_back_to_keyword_when_the_embedder_is_down(wiki, settings, monkeypatch):
    from llmwiki.errors import ProviderError

    wiki.write("wiki/concepts/storage.md", "---\ntype: concept\ntitle: Storage\n---\n\n# Storage\n\nGrid storage.\n")
    (wiki.state_dir / "vectors.db").write_bytes(b"")  # store exists but is unusable
    settings.embedding.enabled = True
    settings.embedding.model = "text-embedding-3-small"

    def boom(*_args, **_kwargs):
        raise ProviderError("connection refused")

    monkeypatch.setattr("llmwiki.embeddings.embed_query", boom)
    response = search(wiki, "grid storage", embedding_config=settings.embedding)

    assert response.results, "a dead embedder must degrade to keyword search, not fail the query"
    assert response.mode == "keyword"
    assert any("vector search unavailable" in note for note in response.notes)


def test_vector_lane_end_to_end_finds_a_page_with_no_keyword_overlap(
    wiki, settings, monkeypatch, tmp_path
):
    """The point of the vector lane: match meaning, not words.

    The query shares no term with the target page, so keyword scoring alone
    returns nothing. A stub embedder places them close in vector space, and
    RRF fusion has to carry that into the final ranking.
    """
    from llmwiki.embeddings import index_documents
    from llmwiki.retrieval import load_documents

    wiki.write(
        "wiki/concepts/rtt.md",
        "---\ntype: concept\ntitle: Latency\n---\n\n# Latency\n\n"
        "The delay before a transfer begins following an instruction.\n",
    )
    wiki.write(
        "wiki/concepts/other.md",
        "---\ntype: concept\ntitle: Bananas\n---\n\n# Bananas\n\nA yellow fruit.\n",
    )

    # Toy 2-d "embeddings": anything about delay points one way, fruit the other.
    def fake_embed(texts, _config):
        vectors = []
        for text in texts:
            lowered = text.lower()
            is_delay = any(word in lowered for word in ("latency", "delay", "slow", "wait"))
            vectors.append([1.0, 0.0] if is_delay else [0.0, 1.0])
        return vectors

    monkeypatch.setattr("llmwiki.embeddings.embed_texts", fake_embed)
    monkeypatch.setattr("llmwiki.embeddings.embed_query", lambda text, cfg: fake_embed([text], cfg)[0])

    settings.embedding.enabled = True
    settings.embedding.model = "toy"
    # The two pages above plus the project's own overview.md stub.
    summary = index_documents(wiki, load_documents(wiki, include_sources=False), settings.embedding)
    assert summary["indexed"] == 3
    assert summary["failed"] == 0

    question = "why is it slow to start"
    keyword_only = search(wiki, question, include_sources=False)
    assert keyword_only.results == [], "no keyword overlap, by construction"

    hybrid = search(wiki, question, include_sources=False, embedding_config=settings.embedding)
    assert hybrid.mode in {"vector", "hybrid"}
    assert hybrid.results[0].path == "wiki/concepts/rtt.md"
    assert hybrid.results[0].vector_score is not None


def test_reindex_skips_unchanged_pages(wiki, settings, monkeypatch):
    from llmwiki.embeddings import index_documents
    from llmwiki.retrieval import load_documents

    wiki.write("wiki/concepts/a.md", "---\ntype: concept\ntitle: A\n---\n\n# A\n\nContent.\n")
    calls = []

    def fake_embed(texts, _config):
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("llmwiki.embeddings.embed_texts", fake_embed)
    settings.embedding.enabled = True
    settings.embedding.model = "toy"

    documents = load_documents(wiki, include_sources=False)
    first = index_documents(wiki, documents, settings.embedding)
    second = index_documents(wiki, documents, settings.embedding)

    assert first["indexed"] == len(documents)
    assert second["indexed"] == 0 and second["skipped"] == len(documents)
    # One request for the whole pass, not one per page. Batching within a
    # document never fires on a wiki of short pages — every page is one chunk —
    # so the batch is filled from as many documents as it holds.
    assert len(calls) == 1, "chunks must be batched across documents"
    assert sum(calls) == len(documents), "and each page embedded exactly once"

    forced = index_documents(wiki, documents, settings.embedding, force=True)
    assert forced["indexed"] == len(documents)
    assert sum(calls) == 2 * len(documents), "and re-embedded exactly once on --force"
