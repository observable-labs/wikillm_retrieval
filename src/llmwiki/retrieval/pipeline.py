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

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from ..config import EmbeddingConfig
from ..errors import ProviderError
from .calibration import ABSTAIN_QUANTILE
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
    # Whether a lane that found nothing may abstain instead of voting. Equal
    # weights are right when both lanes are competent and wrong when one has no
    # handhold at all: on questions phrased without any of the corpus's own
    # vocabulary the lexical lane still returns fifty documents and RRF still
    # counts that ranking as evidence, which cost 0.21 recall at k=10 against
    # the vector lane alone. The fence is the corpus's own score distribution,
    # not a constant — see `calibration.py`.
    lexical_gate: bool = True
    # Where in the corpus's own score distribution the abstention fence sits.
    # A profile's instrument: `research` is more sceptical of the lexical lane
    # than `balanced` because the questions it is for are phrased without the
    # corpus's vocabulary, and the fence is the one dial that says so.
    abstain_quantile: float = ABSTAIN_QUANTILE
    # How much each lane's vote is worth in fusion. Equal by default, which is
    # what RRF assumes and what every measurement before profiles existed was
    # taken under. A weight is the continuous form of the gate above: the gate
    # is the right instrument when a lane has *no* handhold, and a weight is the
    # right one when it has a weak one, which is most of a real question set.
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    # Whether diffusion is skipped when the lexical lane abstained. Seeding PPR
    # from a fused list the lexical lane could not rank is diffusing from a
    # ranking with no lexical evidence in it, and it is measurably worse than
    # not diffusing at all on exactly those queries.
    graph_gate: bool = True
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
    # Lanes that were configured, available, and chose not to contribute to this
    # query. Distinct from a lane that is off (it would not be configured) and
    # from one that failed (it would leave a note): an abstention is the
    # configuration working, and reporting it as either of the other two would
    # make a working gate look like a broken lane.
    abstained: tuple[str, ...] = ()

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
    # ── the telemetry seam ────────────────────────────────────────────
    # Per-stage milliseconds. Named stages rather than one wall clock because
    # the four costs here have four different owners: opening the index is
    # amortised and occasionally the whole turn, the embedding is somebody
    # else's network, and ranking and diffusion are the only parts this
    # repository can move. A budget cannot be set against a number that mixes
    # them, and every later percentile is measured from this map.
    stage_ms: dict[str, float] = field(default_factory=dict)
    # What the abstention fence was compared to, and what it decided. The score
    # is kept alongside the verdict because a lane that scored just under the
    # fence and one that found nothing at all are the same `gate_fired` and
    # different queries — which is the distinction a gap queue is built on.
    lexical_top: float | None = None
    vector_top: float | None = None
    gate_fired: bool = False
    # The query embedding, when the vector lane ran. Kept so that clustering the
    # log later does not re-pay a provider call this turn already made.
    query_vector: list[float] | None = None


@contextmanager
def _stage(stages: dict[str, float], name: str):
    """Time one stage into `stages`, accumulating across repeat entries."""
    started = perf_counter()
    try:
        yield
    finally:
        elapsed = (perf_counter() - started) * 1000.0
        stages[name] = round(stages.get(name, 0.0) + elapsed, 3)


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
    stages: dict[str, float] = {}
    if index is None:
        with _stage(stages, "open"):
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
    with _stage(stages, "lexical"):
        lexical_ranked = _lexical_lane(index, query, effective_tokens, query_phrase, depth, options)
    lanes.lexical = True
    lexical_rank = {path: rank for rank, (path, _) in enumerate(lexical_ranked, start=1)}

    # ── S1b: vector ───────────────────────────────────────────────────
    vector_rank: dict[str, int] = {}
    vector_score: dict[str, float] = {}
    vector_hits = 0
    query_vector: list[float] | None = None
    if options.vector and embedding_config is not None:
        if not (embedding_config.enabled and embedding_config.model):
            notes.append("vector search is not configured — no embedding model")
        elif not (Path(project.state_dir) / "vectors.db").exists():
            notes.append(
                "vector search is configured but no index exists yet — run 'llmwiki embed'"
            )
        else:
            try:
                vector_hits, covered, query_vector = _vector_lane(
                    project,
                    query,
                    embedding_config,
                    options.vector_depth
                    or max(limit * VECTOR_DEPTH_MULTIPLIER, MIN_VECTOR_DEPTH),
                    vector_rank,
                    vector_score,
                    stages,
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

    # ── S1b'/S1c: abstention, then fusion ─────────────────────────────
    # A lane with nothing to say should not get an equal vote. The lexical lane
    # abstains when its own top score falls below what a well-aimed query scores
    # on this corpus — and only when there is another lane to fall back on,
    # because a ranking from a lane that found little is still better than no
    # ranking at all.
    abstain = False
    lexical_top = lexical_ranked[0][1] if lexical_ranked else None
    if options.lexical_gate and vector_rank and lexical_ranked:
        with _stage(stages, "calibrate"):
            calibration = index.calibration()
        top_score = lexical_ranked[0][1]
        if calibration.abstains(top_score, options.abstain_quantile):
            abstain = True
            lanes.abstained += ("lexical",)
            notes.append(
                f"the lexical lane scored {top_score:.2f}, below the "
                f"{options.abstain_quantile:.0%} mark of what a query naming "
                f"something in this corpus scores "
                f"({calibration.fence_at(options.abstain_quantile):.2f} over "
                f"{calibration.sampled} sampled titles); it abstained rather "
                "than diluting the ranking"
            )
            lexical_rank = {}

    with _stage(stages, "fuse"):
        fused = _fuse(
            lexical_rank,
            vector_rank,
            index,
            options.rrf_k,
            options.lexical_weight,
            options.vector_weight,
        )

    # ── S2: seeded PPR ────────────────────────────────────────────────
    ranked = fused[:limit]
    graph_hits = 0
    if options.graph_ppr and abstain and options.graph_gate:
        # Nothing to diffuse *from*. The fused list is the vector lane's ranking
        # alone, so seeding PPR from it spreads mass through a graph built out
        # of entity mentions on the strength of no lexical evidence whatever —
        # which on the keyword-hostile suite costs 0.03 recall at k=3 and 64 ms.
        lanes.abstained += ("graph",)
    elif options.graph_ppr and fused:
        with _stage(stages, "diffuse"):
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

    with _stage(stages, "materialize"):
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
        stage_ms=stages,
        lexical_top=lexical_top,
        vector_top=max(vector_score.values()) if vector_score else None,
        gate_fired=abstain,
        query_vector=query_vector,
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
    stages: dict[str, float] | None = None,
) -> tuple[int, int, list[float] | None]:
    """Returns (pages ranked, documents the index covers, the query vector).

    The embedding round trip is timed apart from the scan it feeds: one is a
    provider's network and the other is a dot-product pass over local rows, and
    a turn that is late is late in one of them.
    """
    from ..embeddings import VectorStore, embed_query, group_by_page

    stages = {} if stages is None else stages
    store_path = Path(project.state_dir) / "vectors.db"
    with _stage(stages, "embed"):
        query_vector = embed_query(query, embedding_config)
    with _stage(stages, "vector"):
        with VectorStore(store_path) as store:
            chunk_hits = store.search(query_vector, max(depth * 3, 30))
            covered, _chunks = store.count()
        if not chunk_hits:
            return 0, covered, query_vector

        page_hits = group_by_page(chunk_hits, depth)
        for rank, page in enumerate(page_hits, start=1):
            vector_rank[page.page_id] = rank
            vector_score[page.page_id] = page.score
    return len(page_hits), covered, query_vector


def _fuse(
    lexical_rank: dict[str, int],
    vector_rank: dict[str, int],
    index: SearchIndex,
    rrf_k: float = RRF_K,
    lexical_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion over whichever lanes produced a ranking.

    Computed even for a single lane. RRF over one list is a monotone transform of
    its rank, so the ordering is untouched — and having one scale for the fused
    score is what lets PPR's restart masses mean the same thing whether or not
    the vector lane ran.

    Weights default to 1.0 and are a profile's instrument, not a per-query one.
    Nothing here calibrates them: a weight derived from a lane's own score margin
    was tried and was worse on both corpora, because BM25 margins (0.5-0.7) and
    cosine margins (0.1-0.25) are not comparable quantities. What is comparable
    is a lane against *itself* on the same corpus, which is what the abstention
    gate uses; a weight is the caller saying which lane this kind of question
    should trust, which is a decision the caller has and the ranker does not.
    """
    known = index.by_path
    scores: dict[str, float] = {}
    for ranks, weight in ((lexical_rank, lexical_weight), (vector_rank, vector_weight)):
        if weight <= 0.0:
            continue
        for path, rank in ranks.items():
            if path in known:
                scores[path] = scores.get(path, 0.0) + weight / (rrf_k + rank)
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
    """The label, derived from what ran and nothing else.

    A lane that abstained did not contribute a ranking, so a query where the
    lexical lane stood down is a `vector` query however it was configured. The
    label describes the ranking that was produced, which is the only thing a
    caller comparing two of them can act on.
    """
    if lanes.vector and lanes.lexical and "lexical" not in lanes.abstained:
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
