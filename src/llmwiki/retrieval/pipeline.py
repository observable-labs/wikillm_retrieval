"""The retrieval pipeline: keyword -> vector -> RRF -> graph expansion.

    Phase 1   tokenized keyword scoring over wiki/ and raw/sources/
    Phase 1.5 vector search over chunk embeddings          (optional)
    Phase 2   reciprocal-rank fusion of the two rankings   (when both ran)
    Phase 3   graph expansion into the reserved slice of the window

Fusion is RRF rather than score normalization on purpose: keyword scores
(hundreds) and cosine similarities (0-1) are not comparable magnitudes, but
their *ranks* are, and RRF needs no per-corpus tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import EmbeddingConfig
from ..errors import ProviderError
from .graph import RRF_K, WikiGraph, blend_graph_results, build_graph
from .keyword import Document, SearchResult, build_snippet, load_documents, score_document
from .tokenize import tokenize_query, trim_query_punctuation

DEFAULT_TOP_K = 20
MAX_TOP_K = 50


@dataclass
class SearchResponse:
    results: list[SearchResult]
    mode: str  # "keyword" | "vector" | "hybrid"
    token_hits: int = 0
    vector_hits: int = 0
    graph_hits: int = 0
    documents: list[Document] = field(default_factory=list)
    graph: WikiGraph | None = None
    notes: list[str] = field(default_factory=list)


def search(
    project,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    include_sources: bool = True,
    embedding_config: EmbeddingConfig | None = None,
    documents: list[Document] | None = None,
) -> SearchResponse:
    if not query.strip():
        return SearchResponse(results=[], mode="keyword")

    limit = max(1, min(top_k, MAX_TOP_K))
    documents = documents if documents is not None else load_documents(project, include_sources)
    graph = build_graph(documents)
    notes: list[str] = []

    # ── Phase 1: keyword ──────────────────────────────────────────────
    tokens = tokenize_query(query)
    effective_tokens = tokens or [query.strip().lower()]
    query_phrase = trim_query_punctuation(query.lower())

    results: list[SearchResult] = []
    for document in documents:
        hit = score_document(document, effective_tokens, query_phrase, query)
        if hit is not None:
            results.append(hit)

    results.sort(key=lambda result: (-result.score, result.path))
    token_rank = {result.path: index + 1 for index, result in enumerate(results)}

    # ── Phase 1.5: vector ─────────────────────────────────────────────
    vector_rank: dict[str, int] = {}
    vector_score: dict[str, float] = {}
    vector_hits = 0
    if embedding_config is not None and embedding_config.enabled and embedding_config.model:
        store_path = Path(project.state_dir) / "vectors.db"
        if not store_path.exists():
            notes.append(
                "vector search is configured but no index exists yet — run 'llmwiki embed'"
            )
        try:
            vector_hits = _apply_vector_lane(
                project,
                query,
                embedding_config,
                limit,
                documents,
                results,
                vector_rank,
                vector_score,
            )
        except ProviderError as exc:
            # A dead embedding endpoint degrades to keyword+graph rather than
            # failing the query — the same fallback the desktop app takes.
            notes.append(f"vector search unavailable, using keyword search only ({exc})")

    # ── Phase 2: fusion ───────────────────────────────────────────────
    if vector_hits:
        for result in results:
            rrf = 0.0
            if result.path in token_rank:
                rrf += 1.0 / (RRF_K + token_rank[result.path])
            if result.path in vector_rank:
                rrf += 1.0 / (RRF_K + vector_rank[result.path])
            result.score = rrf
            if result.path in vector_score:
                result.vector_score = vector_score[result.path]
        results.sort(key=lambda result: (-result.score, result.path))

    # ── Phase 3: graph expansion ──────────────────────────────────────
    blended, graph_hits = blend_graph_results(results, graph, limit, vector_hits)

    return SearchResponse(
        results=blended,
        mode=_mode(bool(token_rank), vector_hits, graph_hits),
        token_hits=len(token_rank),
        vector_hits=vector_hits,
        graph_hits=graph_hits,
        documents=documents,
        graph=graph,
        notes=notes,
    )


def _apply_vector_lane(
    project,
    query: str,
    embedding_config: EmbeddingConfig,
    limit: int,
    documents: list[Document],
    results: list[SearchResult],
    vector_rank: dict[str, int],
    vector_score: dict[str, float],
) -> int:
    from ..embeddings import VectorStore, embed_query, group_by_page

    store_path = Path(project.state_dir) / "vectors.db"
    if not store_path.exists():
        return 0

    query_vector = embed_query(query, embedding_config)
    with VectorStore(store_path) as store:
        chunk_hits = store.search(query_vector, max(limit * 3, 30))
    if not chunk_hits:
        return 0

    page_hits = group_by_page(chunk_hits, max(limit, 10))
    by_path = {document.path: document for document in documents}
    known = {result.path for result in results}

    for rank, page in enumerate(page_hits, start=1):
        vector_rank[page.page_id] = rank
        vector_score[page.page_id] = page.score
        if page.page_id in known:
            continue
        # A page only vector search found still has to enter the ranking,
        # otherwise semantic-only matches are computed and then discarded.
        document = by_path.get(page.page_id)
        if document is None:
            continue
        best = page.matched_chunks[0] if page.matched_chunks else None
        results.append(
            SearchResult(
                path=document.path,
                title=document.title,
                snippet=(
                    build_snippet(best.text, best.text[:40]) if best else document.content[:160]
                ),
                score=0.0,  # RRF assigns the real score below
                kind=document.kind,
                vector_score=page.score,
                document=document,
            )
        )
    return len(page_hits)


def _mode(has_tokens: bool, vector_hits: int, graph_hits: int) -> str:
    if graph_hits > 0:
        return "hybrid"
    if vector_hits == 0:
        return "keyword"
    if not has_tokens:
        return "vector"
    return "hybrid"
