# Can retrieval strategies be paired? Evidence review

Compiled 2026-08-27. Web research plus analysis against llmwiki at `216d96f`.

**Question:** can multiple RAG strategies be combined to get the best of both
worlds — at indexing time, at retrieval time, or both?

**Short answer:** yes at retrieval time, reliably and cheaply; yes at indexing
time, but selectively and at 50–100× the cost. The field converged on this
during 2025–2026. The useful finding is not *that* pairing works but *where*
to pay for it.

Related: [`../future_work/retrieval-vs-sota/`](../future_work/retrieval-vs-sota/README.md)
— the assessment of llmwiki against these systems. §-references below point into
that document.

---

## 1. Evidence

Evidence quality is labelled because much of this is recent and some is
vendor-published. Weight accordingly.

| Finding | Source | Quality |
|---|---|---|
| Late fusion of two retrievers beats routing between them, 6.4% vs 1.1% | RAG vs. GraphRAG systematic eval | preprint |
| Graph structure helps multi-hop, not fact retrieval | GraphRAG-Bench | ICLR'26 accepted |
| Interleaving vector + graph search beats both, at 0.1% index cost | LazyGraphRAG | vendor blog benchmark |
| Hierarchy + graph in one index beats either alone | HiRAG, E²GraphRAG | preprints |
| BM25 + dense + RRF + rerank is production standard | multiple practitioner sources | practitioner consensus |
| Graph indexes assume static corpora; incremental updates are lossy | multiple | mixed, incl. vendor |

### 1.1 Retrieval-time pairing — settled

The standard production pipeline is BM25 (sparse) + dense vector, fused with
Reciprocal Rank Fusion, then reranked with a cross-encoder. RRF is used
specifically because it needs no score-scale calibration between heterogeneous
retrievers — the same reasoning `retrieval/pipeline.py` already documents.

Reported gains: on the WANDS e-commerce benchmark, tuned hybrid reaches 0.7497
NDCG against 0.6983 for BM25 alone and 0.6953 for vector alone — the fusion
beats *both* constituents, not merely the weaker one. On mixed text-and-table
financial documents, two-stage hybrid + neural rerank reaches Recall@5 0.816.
(Practitioner sources; directionally consistent across many, but not
peer-reviewed.)

### 1.2 How to pair: Integration beats Selection

The most directly applicable result. [RAG vs. GraphRAG: A Systematic
Evaluation](https://arxiv.org/html/2502.11371v3) tested two pairing strategies
explicitly:

| Strategy | Mechanism | Gain over best single baseline |
|---|---|---|
| **Selection** | classify the query, route to RAG *or* GraphRAG | **+1.1%** |
| **Integration** | run both, concatenate retrieved contexts | **+6.4%** |

On MultiHop-RAG with Llama 3.1-70B, Integration reached 75.77% overall accuracy
against a 71.17% best baseline. Running both and fusing beat routing by roughly
6×: the router's own error rate consumes most of the available gain.

Same paper, on where each wins individually:
- RAG on single-hop factual detail (NQ: 64.78% F1)
- GraphRAG on multi-hop (HotpotQA: Community-GraphRAG Local 61.66% F1 vs RAG
  60.04%)

Costs measured in the same work: KG construction 7,702s vs vanilla RAG 135s
(~57×); KG-GraphRAG retrieval latency reaching 14,434s. Performance was also
sensitive to the extraction model — GPT-4o beat GPT-4o-mini on reasoning tasks,
meaning index quality is bounded by what you were willing to spend at ingest.

### 1.3 Pairing at the algorithm level

[LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
frames the two families as complementary search strategies:

- vector RAG = **best-first** search (rank by query similarity)
- GraphRAG global = **breadth-first** search (traverse community structure for
  coverage)

LazyGraphRAG interleaves them with iterative deepening, and defers *all* LLM
summarization to query time — indexing is lightweight graph construction only.
Reported: won all 96 comparisons against GraphRAG local/global/drift, vector RAG
at 8k and 120k windows, LightRAG, RAPTOR, and TREX; indexing cost equal to plain
vector RAG and 0.1% of full GraphRAG; >700× lower query cost than GraphRAG
global search at comparable quality. (Microsoft's own benchmark — the
architecture insight is more durable than the win rate.)

### 1.4 Pairing at index time

Two recent systems build hierarchy *and* graph into one index:

- **[HiRAG](https://arxiv.org/pdf/2503.10150)** — entity/relation graph, plus
  progressive "bridge-level" summaries, plus Louvain communities; indexed at dual
  granularity and retrieved coarse-to-fine (match community summaries, drill to
  entities).
- **[E²GraphRAG](https://arxiv.org/pdf/2505.24226)** — summary tree *and* entity
  graph maintained in parallel, with adaptive routing: graph for entity-local
  queries, tree for global/thematic ones.

Both report gains over GraphRAG, LightRAG, and RAPTOR individually. Both name
the same limitation: index-time construction and community detection are the
cost, and community-detection parameters need per-corpus tuning.

### 1.5 The counter-evidence

[GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
(ICLR'26) exists because "GraphRAG frequently underperforms vanilla RAG on many
real-world tasks." Across four task levels — fact retrieval, complex reasoning,
contextual summarization, creative generation — graph structure pays on
multi-hop and dense entity relationships and **does not pay on fact retrieval**,
where the indexing overhead buys nothing.

Reported retrieval-speed ordering, useful when picking a structure: RAPTOR
fastest (tree localization), then GFM-RAG and HippoRAG (GNN / PageRank).
HippoRAG has the longest indexing time from building entity↔relationship and
relationship↔chunk mappings; GraphRAG adds community-report generation on top.

---

## 2. The principle

> **Pair liberally at retrieval time.** It is nearly free, reliably additive,
> and fusion beats routing.
>
> **Pair selectively at index time.** It is 50–100× more expensive, pays only
> for specific query classes, and every eager structure is another thing that
> goes stale.

LazyGraphRAG is the strongest form of this claim: defer LLM work to query time
wherever the query can afford it.

---

## 3. Implications for llmwiki

### 3.1 The architecture is already right

llmwiki has three lanes with late fusion:

```
Phase 1    lexical           retrieval/keyword.py
Phase 1.5  dense             embeddings.py
Phase 2    RRF fusion        retrieval/pipeline.py    ← recommended fusion method
Phase 3    graph expansion   retrieval/graph.py       ← "Integration", not "Selection"
```

That is hybrid + RRF + graph integration — structurally what §1.1 and §1.2
endorse. The reserved 15–30% graph quota in `blend_graph_results` is the
*Integration* strategy that beat routing 6.4% to 1.1%. That call was right,
likely by inheritance from the Rust original.

**The problem is not the architecture; each lane is individually
underpowered.** That is a much cheaper problem than a redesign, and it reframes
the existing work items:

- **P1 (BM25)** is not "catching up to SOTA." It is making the lexical leg of a
  correct hybrid function at all. RRF over a broken lexical lane forfeits most
  of the gain hybrid retrieval exists to capture (§1.1).
- **P2 (query-conditioned graph)** is the difference between Integration and
  noise injection. Integration wins in §1.2 because *both* retrievers are
  query-relevant. llmwiki's graph leg is query-blind
  (`retrieval/graph.py:182`), so the reserved slice is not a second opinion —
  it is unfiltered link structure.

### 3.2 Do not add a second eager index

llmwiki sits at the opposite pole from LazyGraphRAG: it has already paid maximum
index-time LLM cost by compiling prose pages at ingest. That is defensible only
because **the index is also the product** — a community report has no standalone
value; a wiki page does.

The corollary is firm: **no RAPTOR tree, no Louvain communities in ingest.**
Porting HiRAG or E²GraphRAG-style structures would double index cost and double
the drift surface (`future_work` §4) to serve query classes that can be served
more cheaply at query time.

### 3.3 The missing lane is nearly free

llmwiki has no multi-scale abstraction; `overview.md` is written at init and
never updated (`future_work` §3.5). Two cheap routes, neither requiring a new
eager index:

**Navigation lane, PageIndex-style.**
[PageIndex](https://github.com/VectifyAI/PageIndex) does vectorless retrieval by
having an LLM reason down a hierarchical tree — read top-level summaries, choose
a branch, drill down — on the premise that *similarity ≠ relevance*. llmwiki
already maintains exactly that tree for free: `schema.md` → `index.md` →
page-type directories → pages, with `[[wikilinks]]` as cross-edges. A fourth
lane navigating it costs one cheap call, or zero if the answering model does it
from the index already in its system prompt. It also produces an auditable
retrieval path, which suits a system whose pitch is auditability.

> **Latency caveat.** This is an *agentic* lane — its cost is LLM calls
> proportional to traversal depth, giving it a high latency floor. See
> [latency-knobs.md](latency-knobs.md) §6: it belongs behind a `thorough`
> profile, never in a latency-sensitive path.

**Deferred community summarization.** For corpus-level questions, do not
pre-build community reports. `build_graph` already materializes the whole graph
in memory on every query — cluster it *at query time* and summarize only the
matched cluster. This is LazyGraphRAG's trick (§1.3) and fits llmwiki better
than porting Louvain into ingest, because it adds zero index cost and zero drift
surface.

> **Latency caveat.** Same as above — deferring work to query time is exactly
> what a latency-sensitive profile cannot afford. Correct for `thorough`, wrong
> as a default.

### 3.4 Drift: corroborated, and more urgent than filed

The drift concern in `future_work` §4 is a known failure of the whole family,
not a quirk of this design. Standard GraphRAG assumes a static corpus;
incremental updates "tend to be lossy and still require periodic full
re-indexing," and the index grows super-linearly with corpus size.

One widely-cited figure — **42.3% of retrieved document sets containing
temporally conflicting information, correlating with a 31.2% contradiction rate
in generated answers** — should be treated as *unverified*: it reached this
review through secondary sources and the primary study was not read.

The recommended mitigation is provenance versioning: every statement traceable
to a chunk, to a span, to a document, *at the version it was derived from*.
llmwiki has the substrate for this and does not use it — frontmatter `sources:`,
immutable `raw/`, append-only `log.md`, git history.

This argues for promoting drift detection out of the backlog. It is the
differentiator (nothing in the graph-RAG family models contradiction at all) and
it is the maintenance problem the field is currently walking into.

---

## 4. Suggested reprioritization

Not yet applied to
[`../future_work/retrieval-vs-sota/work-items.md`](../future_work/retrieval-vs-sota/work-items.md).

| | Item | Change | Why |
|---|---|---|---|
| P1 | BM25 keyword lane | unchanged | §3.1 — precondition for hybrid working at all |
| P2 | Query-conditioned graph | unchanged | §3.1 — precondition for Integration |
| P3 | Eval harness | unchanged | still gates honest assessment of everything else |
| **new** | Navigation lane over `index.md` | **add** | §3.3 — cheapest fix for the widest gap |
| **new** | Query-time cluster summarization | **add** | §3.3 — sensemaking without an eager index |
| — | Drift detection | **promote from backlog** | §3.4 — corroborated; the differentiator |
| — | Reranking | **keep in backlog, lower** | §3.1 — less headroom once P1/P2 land, and it breaks the one-call-per-query cost profile |
| — | Hierarchical index-time structures | **rule out** | §3.2 — wrong pole of the lazy/eager tradeoff for this system |

---

## 5. Sources

Peer-reviewed / accepted:
- [GraphRAG-Bench — *When to use Graphs in RAG*](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) (ICLR'26) · [paper](https://arxiv.org/pdf/2506.02404)

Preprints:
- [RAG vs. GraphRAG: A Systematic Evaluation and Key Insights](https://arxiv.org/html/2502.11371v3)
- [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/pdf/2503.10150)
- [E²GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness](https://arxiv.org/pdf/2505.24226)

Vendor / practitioner — architecture insights durable, benchmark numbers self-reported:
- [LazyGraphRAG — Microsoft Research](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [PageIndex — vectorless, reasoning-based RAG](https://github.com/VectifyAI/PageIndex)
- [Hybrid Search: BM25, Vector & Reranking Reference 2026](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
- [Designing a Persistent Knowledge Layer That Refuses to Guess](https://towardsdatascience.com/designing-a-persistent-knowledge-layer-that-refuses-to-guess/)

Not consulted, flagged for follow-up: the primary source for the 42.3% / 31.2%
drift figures (§3.4); BEAM (ICLR'26) and LongMemEval, which benchmark exactly
the fact-tracking / contradiction-resolution / knowledge-update abilities
llmwiki's ingest claims to handle, and would be the closest thing to an
off-the-shelf eval for §3.4.
