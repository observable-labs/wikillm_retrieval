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
from typing import Protocol, runtime_checkable

from ..config import EmbeddingConfig
from ..errors import ProviderError
from .calibration import ABSTAIN_QUANTILE
from .entities import MAX_MENTIONS_PER_DOCUMENT, EntityIndex
from .graph import RRF_K, WikiGraph
from .index import MENTION_SCALE, CorpusIndex, InMemoryIndex, open_index
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
from .telemetry import EXPIRED, FAILED, NULL_SINK, OK, SKIPPED, Deadline, Sink
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

# What a stage usually costs, for deciding whether to start it at all. Local
# figures are measured — a replay of the atlas suite's 44 questions puts
# diffusion at a 9.4 ms p95, everything else under 1 — and the round trip is an
# estimate, because it is somebody else's network and this corpus's runs serve
# it from a cache. The query log now records `stage_ms` per turn, so a later
# step can replace the estimate with the deployment's own p50 instead of a
# constant chosen here.
TYPICAL_EMBED_MS = 120.0
TYPICAL_DIFFUSE_MS = 10.0


@runtime_checkable
class VectorSearcher(Protocol):
    """Nearest chunks to a query vector, and how much of the corpus is covered.

    The vector lane used to derive a path — `project.state_dir / "vectors.db"` —
    which made a library that can rank an in-memory corpus insist its vectors be
    a file on disk. That was the last filesystem opinion in the ranking path.

    Lifecycle belongs to whoever constructed it: `search_index` never opens or
    closes a searcher it was handed. `count()` returns (documents covered,
    chunks), and coverage below the corpus size is reported as a note because
    fusion demotes every document the lane could not rank.
    """

    def search(self, vector: list[float], n: int) -> list["ChunkHit"]: ...

    def count(self) -> tuple[int, int]: ...


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
    # Lanes that would have run and were not given the time. The fourth state,
    # and it has to stay distinct from the three above: a lane that is off was
    # a configuration choice, a lane that failed is a broken backend, a lane
    # that abstained is the gate working — and a lane that expired is a slow
    # backend, which is a different page of the runbook from a broken one.
    expired: tuple[str, ...] = ()

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
def _stage(stages: dict[str, float], name: str, sink: Sink = NULL_SINK, outcome: str = OK):
    """Time one stage into `stages`, accumulating across repeat entries.

    The outcome travels with the duration because a stage that took 4 ms because
    it was skipped and one that took 4 ms because it succeeded are the same
    number and different facts. A raising body is recorded as `failed` and the
    exception is left to the caller.
    """
    started = perf_counter()
    result = outcome
    try:
        yield
    except BaseException:
        result = FAILED
        raise
    finally:
        elapsed = (perf_counter() - started) * 1000.0
        stages[name] = round(stages.get(name, 0.0) + elapsed, 3)
        sink.stage(name, round(elapsed, 3), result)


def search(
    project,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    include_sources: bool = True,
    embedding_config: EmbeddingConfig | None = None,
    documents: list[Document] | None = None,
    options: RetrievalOptions | None = None,
    index: CorpusIndex | None = None,
    budget: "Budget | None" = None,
    deadline: Deadline | None = None,
    sink: Sink = NULL_SINK,
) -> SearchResponse:
    """Rank `query` over the project.

    `budget` is the knob: a caller states what the turn may cost and the pipeline
    decides what fits, rather than choosing a profile name and hoping it means
    the right thing on this corpus. `deadline` is the same thing with the clock
    already running, for a caller that spent part of the turn before retrieval
    started — a query rewrite, say — because a deadline each stage restarts is
    not a deadline.

    Neither ever fails the turn. On the fast path a stage that cannot be
    afforded falls back to the one below it and says so, because a spoken answer
    built from the lexical lane alone is worth more than a better answer nobody
    waited for; on the slow path nothing is dropped and the overrun is recorded.

    This is the *project* entry point and its signature is fixed: it opens the
    index, opens the project's vector store, and hands both to `search_index`,
    which is the ranking ladder with no filesystem opinion at all. A caller who
    already has a corpus should call that one directly.
    """
    if not query.strip():
        return SearchResponse(results=[], mode="keyword")

    options = options or RetrievalOptions()
    # ⛔ The deadline is built HERE, before the index is opened, and not left to
    # `search_index`. `Deadline.started` is set at construction, and opening the
    # index is the stage this module's own `SearchResponse` docstring calls
    # "amortised and occasionally the whole turn" — 33 ms on a 78-document
    # corpus, cold. Building it afterwards hands the turn a clock that does not
    # include the largest thing it may have just paid for, so a budget that
    # should have dropped a lane silently affords it.
    if deadline is None and budget is not None:
        deadline = Deadline(budget.stages(), path=budget.path)
    stages: dict[str, float] = {}
    if index is None:
        with _stage(stages, "open", sink):
            index = open_index(
                project,
                include_sources=include_sources,
                documents=documents,
                lexical=options.lexical_bm25,
            )

    # Opened here rather than inside the lane so that the lane takes an object
    # and not a path. The conditions are exactly the ones that used to guard the
    # `store_path` derivation, so a project with no `vectors.db` still arrives at
    # `search_index` with `vectors=None` and still gets the same note.
    vectors = _open_project_vectors(project, options, embedding_config)
    try:
        return search_index(
            index,
            query,
            top_k=top_k,
            options=options,
            vectors=vectors,
            embedding_config=embedding_config,
            budget=budget,
            deadline=deadline,
            sink=sink,
            stages=stages,
        )
    finally:
        if vectors is not None:
            vectors.close()


def _open_project_vectors(project, options: RetrievalOptions, embedding_config):
    """The project's own `VectorStore`, when the vector lane could use one.

    Deliberately the same three conditions the pipeline used to test inline: the
    lane is on, an embedder is configured *and* enabled, and the file exists.
    The third is what `search_index` now reads as `vectors is None`, so the
    "run 'llmwiki embed'" note fires from the same fact it always did.

    Imported lazily. `llmwiki.embeddings` optionally reaches for numpy, and the
    keyword-only install must not pay for it — nor must anything that has
    monkey-patched the module from outside see a different import order.
    """
    if not (options.vector and embedding_config is not None):
        return None
    if not (embedding_config.enabled and embedding_config.model):
        return None
    if project is None:
        return None
    store_path = Path(project.state_dir) / "vectors.db"
    if not store_path.exists():
        return None
    from ..embeddings import VectorStore

    return VectorStore(store_path)


def search_index(
    index: CorpusIndex,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    options: RetrievalOptions | None = None,
    vectors: VectorSearcher | None = None,
    embedding_config: EmbeddingConfig | None = None,
    budget: "Budget | None" = None,
    deadline: Deadline | None = None,
    sink: Sink = NULL_SINK,
    stages: dict[str, float] | None = None,
) -> SearchResponse:
    """The ranking ladder. Knows nothing about where the corpus lives.

        S1a  lexical ranking over the corpus index
        S1b  vector search through an injected searcher      (optional)
        S1c  reciprocal-rank fusion of the two rankings
        S2   personalized PageRank seeded from the fused list (optional)

    Every stage reads `index`, which is a `CorpusIndex` — a directory of
    markdown built in memory, a persisted per-tenant store, anything answering
    the protocol. Nothing here opens a file, derives a path, or knows what a
    project is.

    `vectors` is the vector lane's searcher, already open; this function never
    opens or closes one. `stages` lets a caller that timed work *before*
    retrieval — opening the index, rewriting the query — pass its own map in, so
    the response reports one timeline instead of two.

    Keyword-only after `query` on purpose: this is new surface, and positional
    parameters are the half of a signature that cannot later be reordered.
    """
    if not query.strip():
        return SearchResponse(results=[], mode="keyword")

    options = options or RetrievalOptions()
    # For a caller that arrives with a budget and no clock. `search` builds its
    # own first, so that the index open it pays for is inside the turn.
    if deadline is None and budget is not None:
        deadline = Deadline(budget.stages(), path=budget.path)
    limit = max(1, min(top_k, MAX_TOP_K))
    depth = max(limit * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    stages = {} if stages is None else stages
    notes: list[str] = []
    lanes = LanesRun()

    tokens = tokenize_query(query)
    effective_tokens = tokens or [query.strip().lower()]
    query_phrase = trim_query_punctuation(query.lower())

    # ── S1a: lexical ──────────────────────────────────────────────────
    with _stage(stages, "lexical", sink):
        lexical_ranked = _lexical_lane(index, query, effective_tokens, query_phrase, depth, options)
    lanes.lexical = True
    lexical_rank = {path: rank for rank, (path, _) in enumerate(lexical_ranked, start=1)}

    # ── S1b: vector ───────────────────────────────────────────────────
    vector_rank: dict[str, int] = {}
    vector_score: dict[str, float] = {}
    vector_hits = 0
    query_vector: list[float] | None = None
    if options.vector and embedding_config is not None:
        if (
            deadline is not None
            and deadline.may_degrade
            and not deadline.affords("embedding", TYPICAL_EMBED_MS)
            and not _lexical_found_nothing(index, lexical_ranked, options)
        ):
            # The `ProviderError` branch below is the right fallback and this is
            # a deadline in front of it: the same degradation to lexical+graph,
            # reached before the round trip rather than after waiting out its
            # timeout.
            # … except when the lexical lane found nothing, which the guard
            # above checks: then the vector lane is the only lane there is, and
            # a budget that drops it has not degraded the answer, it has
            # removed it.
            lanes.expired += ("vector",)
            sink.stage("embed", 0.0, EXPIRED)
            notes.append(
                "the query embedding was skipped: the turn's budget did not "
                "cover a round trip, so this ranking is the lexical lane's"
            )
        elif not (embedding_config.enabled and embedding_config.model):
            notes.append("vector search is not configured — no embedding model")
        elif vectors is None:
            notes.append(
                "vector search is configured but no index exists yet — run 'llmwiki embed'"
            )
        else:
            try:
                vector_hits, covered, query_vector = _vector_lane(
                    vectors,
                    query,
                    embedding_config,
                    options.vector_depth
                    or max(limit * VECTOR_DEPTH_MULTIPLIER, MIN_VECTOR_DEPTH),
                    vector_rank,
                    vector_score,
                    stages,
                    sink,
                    None if deadline is None else deadline.for_stage("embedding"),
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
                #
                # A timeout arrives here too, and it is recorded as an expiry
                # rather than a failure: the endpoint answered nothing in the
                # time it was given, which is a slow backend and not a broken
                # one, and the two have different fixes.
                if _looks_like_a_timeout(exc):
                    lanes.expired += ("vector",)
                    notes.append(
                        f"the query embedding did not answer within its budget, "
                        f"using keyword search only ({exc})"
                    )
                else:
                    notes.append(
                        f"vector search unavailable, using keyword search only ({exc})"
                    )

    # ── S1b'/S1c: abstention, then fusion ─────────────────────────────
    # A lane with nothing to say should not get an equal vote. The lexical lane
    # abstains when its own top score falls below what a well-aimed query scores
    # on this corpus — and only when there is another lane to fall back on,
    # because a ranking from a lane that found little is still better than no
    # ranking at all.
    abstain = False
    lexical_top = lexical_ranked[0][1] if lexical_ranked else None
    if options.lexical_gate and vector_rank and lexical_ranked:
        with _stage(stages, "calibrate", sink):
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

    with _stage(stages, "fuse", sink):
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
    elif (
        options.graph_ppr
        and fused
        and deadline is not None
        and deadline.may_degrade
        and not deadline.affords("neighbourhood", TYPICAL_DIFFUSE_MS)
    ):
        # `graph_gate` already knows how to run without diffusion; this is the
        # same path, reached for a different reason. It is also the *last* rung a
        # budget drops, not the first: ~8 ms of local CPU against a round trip
        # for the vector lane, and it is what carries multi-hop. The shipped
        # `voice` profile had this backwards.
        lanes.expired += ("graph",)
        sink.stage("diffuse", 0.0, EXPIRED)
        notes.append(
            "graph diffusion was skipped: the turn's budget did not cover it, "
            "so this ranking is the fused list without it"
        )
    elif options.graph_ppr and fused:
        with _stage(stages, "diffuse", sink):
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

    with _stage(stages, "materialize", sink):
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

    if deadline is not None:
        search_budget = deadline.budgets.search
        if search_budget is not None and deadline.elapsed_ms > search_budget:
            # The plan says a hybrid search over its budget fails the turn.
            # It does not, and the reason is that by the time this is knowable
            # the ranking exists: aborting here would discard an answer that has
            # already been paid for. The overrun is recorded instead, which is
            # what a budget is for — `sink` sees `expired` on the turn and the
            # caller sees the note.
            sink.stage("search", round(deadline.elapsed_ms, 3), EXPIRED)
            notes.append(
                f"retrieval took {deadline.elapsed_ms:.0f} ms against a "
                f"{search_budget:.0f} ms budget"
            )
        else:
            sink.stage("search", round(deadline.elapsed_ms, 3), OK)

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
    index: CorpusIndex,
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
    vectors: VectorSearcher,
    query: str,
    embedding_config: EmbeddingConfig,
    depth: int,
    vector_rank: dict[str, int],
    vector_score: dict[str, float],
    stages: dict[str, float] | None = None,
    sink: Sink = NULL_SINK,
    budget_ms: float | None = None,
) -> tuple[int, int, list[float] | None]:
    """Returns (pages ranked, documents the index covers, the query vector).

    The embedding round trip is timed apart from the scan it feeds: one is a
    provider's network and the other is a dot-product pass over local rows, and
    a turn that is late is late in one of them.

    `vectors` is owned by the caller and is neither opened nor closed here.
    `embed_query` stays an attribute lookup on the module at call time, because
    it is patched from outside the package by at least one consumer.
    """
    from ..embeddings import embed_query, group_by_page

    stages = {} if stages is None else stages
    with _stage(stages, "embed", sink):
        # The timeout is passed only when there is one. A turn with no deadline
        # calls the embedder exactly as it did before deadlines existed, which
        # keeps the shipped path — and anything that has stubbed this function —
        # off a code path it never asked for.
        query_vector = (
            embed_query(query, embedding_config)
            if budget_ms is None
            else embed_query(query, embedding_config, budget_ms / 1000.0)
        )
    with _stage(stages, "vector", sink):
        chunk_hits = vectors.search(query_vector, max(depth * 3, 30))
        covered, _chunks = vectors.count()
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
    index: CorpusIndex,
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
    index: CorpusIndex,
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
    "VectorSearcher",
    "search_index",
]


def _looks_like_a_timeout(exc: Exception) -> bool:
    """Whether a `ProviderError` is a clock running out rather than a refusal.

    The HTTP helper flattens `urllib` errors into one exception type, so the
    distinction has to be read back off the message. It is a heuristic and it
    fails safe: a misread timeout is recorded as a plain failure, which is the
    state the code had before deadlines existed.
    """
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text




def _lexical_found_nothing(
    index: CorpusIndex, lexical_ranked: list[tuple[str, float]], options: RetrievalOptions
) -> bool:
    """Whether the lexical lane's best score is under this corpus's own fence.

    Asked *before* the vector lane runs, which is the only place it can be
    asked: the gate itself needs a second lane to fall back on, so by the time
    it fires the round trip has already been paid for. A budget deciding whether
    to skip that round trip needs the same evidence one stage earlier.
    """
    if not options.lexical_gate:
        return not lexical_ranked
    if not lexical_ranked:
        return True
    return index.calibration().abstains(lexical_ranked[0][1], options.abstain_quantile)


