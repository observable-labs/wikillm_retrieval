"""The retrieval pipeline: BM25 -> vector -> RRF -> seeded PPR.

    S1a  lexical ranking over wiki/ and raw/sources/, by FTS5 bm25()
    S1b  vector search over chunk embeddings                    (optional)
    S1c  reciprocal-rank fusion of the two rankings
    S2   personalized PageRank seeded from the fused list        (optional)

Fusion is RRF rather than score normalization on purpose: bm25 scores and cosine
similarities are not comparable magnitudes, but their *ranks* are, and RRF needs
no per-corpus tuning.

The graph lane is applied as a re-ranking of the fused list rather than as a
reserved slice of the window. That is SPRIG's measured distinction — seeding PPR
from the fused list gains recall, blending graph scores into the fused list loses
more than having no graph at all — and it is why `blend_graph_results` was
replaced rather than tuned. See `ppr.py`.

Every lane is separately switchable through `RetrievalOptions`, and the response
reports which of them actually ran. A configuration that could not be delivered
is a fact about the run, not something to infer from hit counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import EmbeddingConfig
from ..errors import ProviderError
from .entities import MAX_MENTIONS_PER_DOCUMENT, EntityIndex
from .graph import RRF_K, WikiGraph
from .index import MENTION_SCALE, SearchIndex, open_index
from .keyword import Document, SearchResult, build_snippet, score_document
from .lexical import usable_for
from .ppr import (
    DEFAULT_ALPHA,
    DEFAULT_ITERATIONS,
    DEFAULT_SEEDS,
    DEFAULT_SELF_WEIGHT,
    DEFAULT_TAIL_WEIGHT,
    rank_by_ppr,
)
from .tokenize import tokenize_query, trim_query_punctuation

DEFAULT_TOP_K = 20
MAX_TOP_K = 50

# How deep each lane ranks before fusion. Fusing only the final window would mean
# a document that is 15th lexically and 3rd by vector never meets its own second
# opinion, and it is the graph's only chance to see a bridge candidate.
CANDIDATE_MULTIPLIER = 5
MIN_CANDIDATES = 50

# How deep the vector lane votes. The cap was introduced because letting the
# lane rank as deep as the lexical one cost 0.10 MRR on atlas — but that
# measurement was taken while `group_by_page` scored a page by chunk count as
# much as by chunk quality, so scanning deeper pulled in more multi-chunk pages
# and every one of them saturated at the top. With a page scored by its best
# chunk the ranking is prefix-stable in the scan depth, and re-sweeping both
# corpora at 2k, 5k, 10k and 20k moves no recall at any k at all.
#
# The cap therefore stays for latency and nothing else: a deeper scan buys
# nothing and costs a longer dot-product pass. It is no longer load-bearing for
# quality, and that is the point — a depth constant that changes results is a
# tuning parameter nobody swept.
VECTOR_DEPTH_MULTIPLIER = 2
MIN_VECTOR_DEPTH = 20


@dataclass(frozen=True)
class RetrievalOptions:
    """One switch per lane, and the S2 parameters.

    Defaults are the measured configuration rather than the cautious one: every
    flag here was carried through an ablation on the eval suite before it was
    turned on, and the flag stays so the ablation can be re-run rather than
    trusted.
    """

    lexical_bm25: bool = True
    vector: bool = True
    # How sharply fusion prefers rank 1 over rank n. RRF's published constant is
    # 60, tuned on TREC-scale result lists of 1,000; these lanes rank 20 to 50,
    # where 60 sits above the whole list and a lane that is merely adequate
    # outvotes one that is good. Rescaled to the depth actually fused. Measured,
    # not inherited — see `graph.RRF_K`.
    rrf_k: float = RRF_K
    vector_depth: int = 0  # 0 = max(2k, 20); see VECTOR_DEPTH_MULTIPLIER
    graph_ppr: bool = True
    entity_edges: bool = True
    curated_links: bool = True
    mentions_per_document: int = MAX_MENTIONS_PER_DOCUMENT
    mention_scale: float = MENTION_SCALE
    seed_count: int = DEFAULT_SEEDS  # 0 = as many as the caller asked for
    tail_weight: float = DEFAULT_TAIL_WEIGHT
    alpha: float = DEFAULT_ALPHA
    iterations: int = DEFAULT_ITERATIONS
    self_weight: float = DEFAULT_SELF_WEIGHT


@dataclass
class LanesRun:
    """Which lanes actually executed, as opposed to which were requested.

    `mode` used to be derived from hit counts, so a keyword-only query that
    happened to pick up one graph neighbour reported itself as "hybrid". A
    caller comparing two configurations cannot notice that from the outside, so
    the pipeline states what ran instead of implying it.
    """

    lexical: bool = False
    vector: bool = False
    graph: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {"lexical": self.lexical, "vector": self.vector, "graph": self.graph}


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
    lanes: LanesRun = field(default_factory=LanesRun)
    entities: EntityIndex | None = None
    index_seconds: float = 0.0


def search(
    project,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    include_sources: bool = True,
    embedding_config: EmbeddingConfig | None = None,
    documents: list[Document] | None = None,
    options: RetrievalOptions | None = None,
    index: SearchIndex | None = None,
) -> SearchResponse:
    if not query.strip():
        return SearchResponse(results=[], mode="keyword")

    options = options or RetrievalOptions()
    limit = max(1, min(top_k, MAX_TOP_K))
    depth = max(limit * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    if index is None:
        index = open_index(
            project,
            include_sources=include_sources,
            documents=documents,
            lexical=options.lexical_bm25,
        )
    notes: list[str] = []
    lanes = LanesRun()

    tokens = tokenize_query(query)
    effective_tokens = tokens or [query.strip().lower()]
    query_phrase = trim_query_punctuation(query.lower())

    # ── S1a: lexical ──────────────────────────────────────────────────
    lexical_ranked = _lexical_lane(index, query, effective_tokens, query_phrase, depth, options)
    lanes.lexical = True
    lexical_rank = {path: rank for rank, (path, _) in enumerate(lexical_ranked, start=1)}

    # ── S1b: vector ───────────────────────────────────────────────────
    vector_rank: dict[str, int] = {}
    vector_score: dict[str, float] = {}
    vector_hits = 0
    if options.vector and embedding_config is not None:
        if not (embedding_config.enabled and embedding_config.model):
            notes.append("vector search is not configured — no embedding model")
        elif not (Path(project.state_dir) / "vectors.db").exists():
            notes.append(
                "vector search is configured but no index exists yet — run 'llmwiki embed'"
            )
        else:
            try:
                vector_hits, covered = _vector_lane(
                    project,
                    query,
                    embedding_config,
                    options.vector_depth
                    or max(limit * VECTOR_DEPTH_MULTIPLIER, MIN_VECTOR_DEPTH),
                    vector_rank,
                    vector_score,
                )
                lanes.vector = True
                if covered < len(index.documents):
                    # Partial coverage is not a neutral degradation: fusion
                    # demotes every document the lane could not rank.
                    notes.append(
                        f"vector index covers {covered} of {len(index.documents)} "
                        "documents; the rest are ranked by the lexical lane alone "
                        "and fusion will place them below covered documents"
                    )
            except ProviderError as exc:
                # A dead embedding endpoint degrades to lexical+graph rather
                # than failing the query — the same fallback the desktop app
                # takes. The note is what stops that being invisible.
                notes.append(f"vector search unavailable, using keyword search only ({exc})")

    # ── S1c: fusion ───────────────────────────────────────────────────
    fused = _fuse(lexical_rank, vector_rank, index, options.rrf_k)

    # ── S2: seeded PPR ────────────────────────────────────────────────
    ranked = fused[:limit]
    graph_hits = 0
    if options.graph_ppr and fused:
        adjacency = index.adjacency(
            entity_edges=options.entity_edges,
            curated_links=options.curated_links,
            mentions_per_document=options.mentions_per_document,
            mention_scale=options.mention_scale,
        )
        if adjacency:
            already_fused = {path for path, _ in fused}
            ranked = rank_by_ppr(
                fused,
                None,
                limit=limit,
                seed_count=options.seed_count,
                tail_weight=options.tail_weight,
                alpha=options.alpha,
                iterations=options.iterations,
                self_weight=options.self_weight,
                keep=set(index.by_path),
                outgoing=index.transitions(
                    entity_edges=options.entity_edges,
                    curated_links=options.curated_links,
                    mentions_per_document=options.mentions_per_document,
                    self_weight=options.self_weight,
                    mention_scale=options.mention_scale,
                ),
            )
            lanes.graph = True
            graph_hits = sum(1 for path, _ in ranked if path not in already_fused)

    results = _materialize(
        ranked,
        index=index,
        seeds={path for path, _ in fused[: max(1, options.seed_count or limit)]},
        fused={path for path, _ in fused},
        vector_score=vector_score,
        tokens=effective_tokens,
        query_phrase=query_phrase,
        query=query,
        options=options,
    )

    return SearchResponse(
        results=results,
        mode=_mode(lanes),
        token_hits=len(lexical_rank),
        vector_hits=vector_hits,
        graph_hits=graph_hits,
        documents=index.documents,
        graph=index.graph,
        notes=notes,
        lanes=lanes,
        entities=index.entities,
        index_seconds=index.build_seconds,
    )


# ── lanes ─────────────────────────────────────────────────────────────────

def _lexical_lane(
    index: SearchIndex,
    query: str,
    tokens: list[str],
    query_phrase: str,
    depth: int,
    options: RetrievalOptions,
) -> list[tuple[str, float]]:
    """BM25 where it applies, the substring scorer where it does not.

    The fallback is not a legacy path kept out of caution: `unicode61` cannot
    segment CJK, so for those queries the bigram expansion in `tokenize_query`
    plus substring matching is genuinely the better retriever.
    """
    if options.lexical_bm25 and index.lexical is not None and usable_for(query):
        return [(hit.path, hit.score) for hit in index.lexical.search(tokens, depth)]

    scored = []
    for document in index.documents:
        hit = score_document(document, tokens, query_phrase, query)
        if hit is not None:
            scored.append((hit.path, hit.score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:depth]


def _vector_lane(
    project,
    query: str,
    embedding_config: EmbeddingConfig,
    depth: int,
    vector_rank: dict[str, int],
    vector_score: dict[str, float],
) -> tuple[int, int]:
    """Returns (pages ranked, documents the index covers)."""
    from ..embeddings import VectorStore, embed_query, group_by_page

    store_path = Path(project.state_dir) / "vectors.db"
    query_vector = embed_query(query, embedding_config)
    with VectorStore(store_path) as store:
        chunk_hits = store.search(query_vector, max(depth * 3, 30))
        covered, _chunks = store.count()
    if not chunk_hits:
        return 0, covered

    page_hits = group_by_page(chunk_hits, depth)
    for rank, page in enumerate(page_hits, start=1):
        vector_rank[page.page_id] = rank
        vector_score[page.page_id] = page.score
    return len(page_hits), covered


def _fuse(
    lexical_rank: dict[str, int],
    vector_rank: dict[str, int],
    index: SearchIndex,
    rrf_k: float = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion over whichever lanes produced a ranking.

    Computed even for a single lane. RRF over one list is a monotone transform of
    its rank, so the ordering is untouched — and having one scale for the fused
    score is what lets PPR's restart masses mean the same thing whether or not
    the vector lane ran.
    """
    known = index.by_path
    scores: dict[str, float] = {}
    for ranks in (lexical_rank, vector_rank):
        for path, rank in ranks.items():
            if path in known:
                scores[path] = scores.get(path, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _materialize(
    ranked: list[tuple[str, float]],
    index: SearchIndex,
    seeds: set[str],
    fused: set[str],
    vector_score: dict[str, float],
    tokens: list[str],
    query_phrase: str,
    query: str,
    options: RetrievalOptions,
) -> list[SearchResult]:
    """Turn (path, score) pairs into results with snippets and provenance."""
    by_path = index.by_path
    adjacency = index.adjacency(
        entity_edges=options.entity_edges,
        curated_links=options.curated_links,
        mentions_per_document=options.mentions_per_document,
        mention_scale=options.mention_scale,
    )
    results: list[SearchResult] = []

    for path, score in ranked:
        document = by_path.get(path)
        if document is None:
            continue
        related: list[str] = []
        if path not in fused:
            # Reached only through diffusion: name the seeds it hangs off, which
            # is the whole explanation of why it is here.
            related = sorted(_seeds_reached(path, adjacency, seeds, by_path))
        results.append(
            SearchResult(
                path=document.path,
                title=document.title,
                snippet=_snippet(document, tokens, query_phrase, query, related),
                score=score,
                kind=document.kind,
                title_match=_title_match(document, tokens, query_phrase),
                vector_score=vector_score.get(path),
                graph_related_to=related,
                document=document,
            )
        )
    return results


def _seeds_reached(
    path: str,
    adjacency: dict[str, dict[str, float]],
    seeds: set[str],
    by_path: dict[str, Document],
) -> set[str]:
    """Titles of the seeds this document hangs off, one entity hop included.

    The mention graph is bipartite, so a document reached through a shared
    entity is two hops from its seed rather than one, and a one-hop explanation
    would silently report nothing for exactly the bridges the lane exists to
    find.
    """
    titles: set[str] = set()
    for neighbor in adjacency.get(path, {}):
        if neighbor in seeds and neighbor in by_path:
            titles.add(by_path[neighbor].title)
        elif neighbor not in by_path:
            for second in adjacency.get(neighbor, {}):
                if second in seeds and second in by_path:
                    titles.add(by_path[second].title)
    return titles


def _snippet(
    document: Document,
    tokens: list[str],
    query_phrase: str,
    query: str,
    related: list[str],
) -> str:
    content_lower = document.content.lower()
    if query_phrase and query_phrase in content_lower:
        anchor = query_phrase
    else:
        anchor = next((token for token in tokens if token in content_lower), "")
    if not anchor:
        if related:
            return f"Graph neighbour of {', '.join(related)}"
        anchor = query
    return build_snippet(document.content, anchor)


def _title_match(document: Document, tokens: list[str], query_phrase: str) -> bool:
    title_text = f"{document.title} {document.path.rsplit('/', 1)[-1]}".lower()
    if query_phrase and query_phrase in title_text:
        return True
    return any(token in title_text for token in tokens)


def _mode(lanes: LanesRun) -> str:
    """The label, derived from what ran and nothing else."""
    if lanes.vector and lanes.lexical:
        return "hybrid"
    if lanes.vector:
        return "vector"
    return "keyword"


__all__ = [
    "CANDIDATE_MULTIPLIER",
    "MIN_VECTOR_DEPTH",
    "VECTOR_DEPTH_MULTIPLIER",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "LanesRun",
    "RetrievalOptions",
    "SearchResponse",
    "search",
]
