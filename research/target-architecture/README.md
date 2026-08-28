# Recommended architecture: a layered index with a stop-early retrieval ladder

Compiled 2026-08-27 against llmwiki `216d96f`. Supersedes the ordering in
[`../../future_work/retrieval-vs-sota/work-items.md`](../../future_work/retrieval-vs-sota/work-items.md);
see §9.

**Question:** given four requirements at once — a voice-grade latency floor, deep
multi-hop reasoning when latency is affordable, corpus-level thematic
comprehension, cross-document *and* intra-document depth — all under efficient
incremental updates, what should be built?

**The executable version of this document is [build-plan.md](build-plan.md)** — §9
here is the summary table; that is the step-by-step with schemas, touch points,
and acceptance criteria.

Companions: [combining-rag-strategies.md](../combining-rag-strategies.md),
[latency-knobs.md](../latency-knobs.md),
[incremental-updates.md](../incremental-updates.md). §8 corrects three claims made
in those documents.

---

## 1. The recommendation in one page

**Keep the wiki. Demote it from *the system* to *one layer of it*.**

Build a **layered index** in which every layer is constructed per-document and
therefore incrementally updatable, and a **retrieval ladder** whose rungs can be
stopped at, so latency is chosen per query rather than baked into the design.

```
INDEX (all per-document local, all incremental)

  L0  raw/            immutable sources                       exists
  L1  text cache      extracted text, SHA256-keyed            new, cheap
  L2  lexical         SQLite FTS5 + bm25()                    new, replaces keyword.py
  L3  dense           chunk vectors, pages AND sources        exists, extend
  L4  structural      entity↔passage bipartite graph          new, ZERO LLM cost
  L5  curated         wikilinks + aliases                     exists, persist it
  L6  abstraction     compiled wiki pages + topic pages       exists, extend

RETRIEVE (each rung strictly adds latency; stop where the budget ends)

  S1  BM25 ∥ dense  → RRF                     ~10-40 ms    voice
  S2  + PPR over L4/L5 seeded from S1          +40-90 ms   balanced
  S3  + local cross-encoder rerank             +20-50 ms   deep
  S4  + agentic loop over S1-S3 as tools       seconds     research
```

The single most important claim: **L4 costs zero LLM tokens, updates in O(1
document), and is what buys multi-hop.** The expensive LLM compilation you
already pay for is not what buys multi-hop — it buys the abstraction layer (L6)
and the readable artifact. Those are different jobs, and conflating them is why
the current system pays a large index cost and still answers multi-hop questions
with a query-blind link walk.

Four requirements, four owners:

| Requirement | Owned by | Rung |
|---|---|---|
| Voice latency | L1–L3, all local, no network hop | S1 |
| Multi-hop across documents | L4 + L5 via seeded PPR | S2 |
| Corpus themes and topics | L6 topic pages, retrieved as documents | S1 (they are just pages) |
| Depth inside one document | L3 source chunks + `chunk_read` | S3/S4 |
| Incremental updates | every layer is per-document | all |

---

## 2. What changes, and what survives

| | Today | Recommended | Why |
|---|---|---|---|
| Lexical lane | substring presence count, no IDF (`keyword.py:155`) | SQLite FTS5 `bm25()` | verified available on this machine (SQLite 3.51.1); no new dependency |
| Graph lane | query-blind `1/(rank+1)` score blend (`graph.py:182`) | PPR seeded from the fused S1 list | §4.2 — score-blending is measurably the *wrong* shape |
| Entity structure | none | dictionary-matched entity↔passage matrix | §4.1 — zero LLM cost, linear, incremental |
| Sources | re-parsed per query, uncached | L1 cache; also embedded into L3 | biggest current latency term |
| Rerank | absent, filed in backlog | small local cross-encoder at S3 | §8.2 — it is not an LLM call |
| Themes | `overview.md` written once at init, never updated | maintained topic pages, clustered incrementally | §5 |
| Deep reasoning | one shot, fixed `top_k` | S4 agentic loop over the same lanes | §6 |
| The knob | five uncoordinated flags | one profile selecting a rung | §7 |

**Survives unchanged and is genuinely differentiated:** immutable `raw/`,
compiled pages with `sources:` provenance, `[[wikilinks]]` as curated edges,
append-only `log.md`, RRF as the fusion method, per-page delete-then-insert
embedding updates.

You are not married to llmwiki, and you do not need to be. What is worth keeping
is a *subset* of it, and the subset is defensible on evidence rather than
sentiment.

---

## 3. The index layers

Every layer below satisfies the same construction rule, which is what makes the
whole thing incremental:

> **Locality rule.** A layer may only be built from one document plus a
> dictionary. No layer may require a global fit (GMM, UMAP, Louvain over the
> whole corpus) to be correct.

### L1 — extracted text cache
SHA256-keyed, written at ingest, read at query. Removes the single largest
current query cost ([latency-knobs](../latency-knobs.md) §4). Append-only.

### L2 — lexical, via FTS5
SQLite FTS5 with the built-in `bm25()` ranking function. Verified present:
SQLite 3.51.1, `CREATE VIRTUAL TABLE … USING fts5` and `bm25()` both work in the
stdlib `sqlite3` on this machine. This replaces the hand-rolled BM25 planned as
P3 in work-items — FTS5 maintains N, avgdl, and df internally, handles
delete-and-reinsert per document, and is 25 years of Lucene-lineage tuning you
do not have to write or test.

Keep the structural bonuses (title match, heading match) as a *separate* score
combined by RRF, not folded into the BM25 value.

### L3 — dense
Already correct in shape: per-page chunks, delete-then-insert on change,
heading-breadcrumb prefixing (`_embedding_text`, `embeddings.py:353`) — which is
a zero-cost approximation of Anthropic's contextual retrieval, the technique that
cut top-20 retrieval failures 49% (67% with reranking).

Two extensions: (a) embed **source** chunks, not only wiki pages, so a document
is answerable before it has been compiled; (b) when the corpus outgrows the
brute-force scan, move to an ANN index that supports incremental insert — HNSW
inserts incrementally and handles deletes by tombstoning, so it does not violate
the locality rule.

### L4 — structural graph, at zero LLM cost
This is the new load-bearing piece.

Build a bipartite entity↔passage graph. Two published systems establish that it
works and costs nothing:

- **LinearRAG** (ICLR 2026) builds a "Tri-Graph" of passage, sentence and entity
  nodes using spaCy NER and sparse contain/mention matrices. The paper is
  explicit about incrementality: *"When new passages arrive, only those passages
  undergo sentence segmentation, NER, and edge construction, yielding overall
  linear complexity."*
- **SPRIG** builds an entity↔document bipartite graph with spaCy (~9–10 ms/doc)
  or a regex heuristic (~0.02 ms/doc), TF-IDF edge weights, and CPU-only sparse
  Personalized PageRank at query time, inside a 4 GB budget.

**llmwiki can do this without spaCy.** `build_graph` already constructs an alias
table — path, wiki-relative path, stem, and title, normalized
(`graph.py:39`, `:68–73`). That table *is* an entity dictionary, and it is
curated by the ingest LLM rather than guessed by a statistical tagger. Entity
extraction becomes dictionary matching over passages: higher precision than NER,
no new dependency, and it improves as the wiki grows.

Store the contain matrix sparsely. Adding a document touches only its own rows.

### L5 — curated links
Persist as **raw links + alias table**, never as resolved adjacency, so a
previously dangling `[[Foo]]` resolves the moment Foo is created
([incremental-updates](../incremental-updates.md) §3). At query time L4 and L5 are
one graph with two edge weights: mechanical edges are dense and high-recall,
curated wikilinks are sparse and high-precision.

### L6 — abstraction
The compiled wiki pages, plus topic pages (§5). This is the layer the LLM ingest
cost buys. It is retrieved like any other document — no special path, no
query-time summarization.

---

## 4. The retrieval ladder

### 4.1 S1 — candidates (voice rung)

FTS5 BM25 and a dense scan, fused by RRF. All local. The only network hop is the
query embedding, which a local embedding model or a cached/precomputed query
vector removes entirely.

Reference latency, retrieval only, CPU, from SPRIG's tables (QTime is total over
the query set; per-query shown):

| | HotpotQA R@10 | per query | 2Wiki R@10 | per query |
|---|---|---|---|---|
| BM25 | 0.742 | 60 ms | 0.643 | 30 ms |
| Dense | 0.811 | 9.6 ms | 0.609 | 8.6 ms |
| **RRF** | **0.851** | **70 ms** | **0.697** | **41 ms** |

LinearRAG independently reports 93 ms average retrieval for a comparable
structural pipeline. A voice rung in the 10–100 ms band is not aspirational; it
is what these systems measure.

### 4.2 S2 — graph expansion, and the shape that matters

SPRIG ablates the two ways to combine a graph signal with a fused lexical+dense
list. The difference is not small:

| Method | HotpotQA R@10 | 2Wiki R@10 |
|---|---|---|
| RRF alone | 0.851 | 0.697 |
| **GraphRRF** — PPR *seeded from* the RRF list | **0.867** | **0.794** |
| RRF+PPR — PPR scores *blended into* RRF | 0.782 | 0.602 |

Seeding gains +1.6 points on HotpotQA and **+9.7 points on 2WikiMultiHopQA**
(+14% relative). Blending *loses* 6.9 and 9.5 points — worse than not having a
graph at all.

**llmwiki currently implements the losing shape.** `blend_graph_results`
(`graph.py:163–182`) reserves a quota of slots and scores neighbours with a
query-blind `1.0 / (rank + 1)`, which is score-level blending of an unfiltered
signal. This single finding is the strongest, most specific, and cheapest
correction available: change *where* the graph enters, not how much of it enters.

The gain concentrates on 2Wiki, the more purely multi-hop benchmark — consistent
with GraphRAG-Bench's conclusion that graph structure pays on multi-hop and not
on fact retrieval. That is exactly why S2 is a rung and not a default.

Cost: GraphRRF runs 142 ms/query on HotpotQA, 81 ms on 2Wiki. Hub pruning
(dropping the top 1% highest-degree entities) cut SPRIG's query time 485 s → 350 s
with negligible recall change — a free 28% if S2 is ever tight.

### 4.3 S3 — rerank

A small local cross-encoder over the top ~50. Reported cost: ~20–50 ms for
BGE-class rerankers on a small candidate set; MiniLM-L-6 and mxbai-base are the
sub-200 ms tier. SPRIG's RRF+CE produced the best MRR on HotpotQA (0.887 vs
0.865 for RRF) at 86 ms/query.

This is where Anthropic's 49% → 67% failure-rate reduction lives. It is a local
forward pass, not an API call.

### 4.4 S4 — agentic loop (research rung)

**A-RAG** is the clearest result here, and it is a direct fit. It exposes three
tools to the model — `keyword_search`, `semantic_search`, `chunk_read` — over a
plain chunk/sentence hierarchy with **no graph index at all**, and runs a ReAct
loop:

| | A-RAG (GPT-5-mini) | best baseline |
|---|---|---|
| HotpotQA | **94.5%** | 86.2% (LinearRAG) |
| 2WikiMultiHopQA | **89.7%** | 87.2% (LinearRAG) |
| MuSiQue | **74.1%** | 62.4% (LinearRAG) |
| GraphRAG-Bench | **93.1%** | 90.5% |

Baselines included GraphRAG, HippoRAG2, LinearRAG, FaithfulRAG, MA-RAG, RAGentA.
Retrieved-token cost was comparable to naive RAG (5,663 vs 5,387 on MuSiQue;
2,737 on 2Wiki). Raising the step cap from 5 to 20 gained ~8% on MuSiQue-300 —
the latency/quality dial, made explicit.

Two implications:

1. **The deep end needs good interfaces, not a fancier index.** A tool-using
   model over simple lanes beat every graph system it was compared against.
2. **llmwiki already has all three tools.** `search` is keyword+semantic;
   `chunk_read` is reading a page or a source span. S4 is mostly plumbing:
   expose the lanes, add a context tracker so re-reads cost zero tokens, cap the
   steps.

`chunk_read` is also the answer to "go deep within a single document" — the
ablation shows removing it costs accuracy, and it is the only mechanism in the
stack that reads a document *in sequence* rather than as ranked fragments.

---

## 5. Corpus themes, without a global fit

Requirement: comprehend high-level topics across the corpus. Three ways to get
it, ranked by fit to the constraints:

**1. Maintained topic pages (recommended).** llmwiki's compiled pages already
*are* what GraphRAG spends its index budget producing — abstracted, sourced prose
about an entity or concept — except written incrementally, one document at a
time, and readable. The gap is that there is no *level*: no page summarizes the
corpus. Add a small set of topic pages plus a maintained `overview.md`, updated
only when their constituent pages change. They then retrieve through S1 like any
other document, at zero extra query cost. This is the one place the LLM ingest
budget is uniquely justified.

**2. Incremental clustering to decide what the topics are.** BERTopic supports
online modelling via `partial_fit` with IncrementalPCA + MiniBatchKMeans +
OnlineCountVectorizer; its maintainer notes `merge_models` is the more stable
route. Either way the cluster assignment is non-LLM and incremental; only the
*labels* of changed clusters need regeneration.

**3. Incremental Leiden, if you ever want real communities.** HIT-Leiden bounds
update work to the 2-hop neighbourhood of affected supernodes and reports up to
five orders of magnitude speedup over recomputation. This is why §8.1 retracts
the claim that community detection is categorically un-incremental — but the
clustering was never the expensive part. The LLM community *reports* are, and
option 1 gets the same artifact for less.

---

## 6. The knob

One profile flag selecting a rung, not five independent dials.

| | `voice` | `balanced` | `deep` | `research` |
|---|---|---|---|---|
| Rungs | S1 | S1+S2 | S1+S2+S3 | S1–S4 loop |
| Retrieval budget | ~10–40 ms | ~70–150 ms | ~150–250 ms | seconds |
| LLM calls | 1, streaming | 1 | 1 | N turns (capped) |
| Sources | L1 cache | L1 cache | L1 cache | + `chunk_read` |
| top_k | ~8 | 20 | 40 | agent-chosen |
| Effort | `off`/`low` | default | `high` | `high`/`max` |
| Query embedding | local model or cached | local or API | API fine | API fine |

Three refinements worth having:

**Automatic escalation.** AB-RAG estimates confidence from three signals — model
certainty, answer/evidence agreement, and *variance of retrieval scores* — and
escalates until confident or out of budget, separating correct from incorrect
answers sharply (57.6% EM high-confidence vs 0% low-confidence on a factoid set).
The third signal is free: you have the scores. A flat top-1/top-5 score margin
at S1 is a cheap trigger to climb a rung.

**Per-query routing is a measured win, not a guess.** CA-RAG selects retrieval
depth per query from a discrete bundle catalog, reporting 26% lower billed token
cost than always-heavy and 34% lower mean latency than always-direct, at quality
parity. The rung ladder *is* a bundle catalog.

> Note this does not contradict [combining-rag-strategies](../combining-rag-strategies.md)
> §1.2, where Selection (routing between strategies) lost to Integration
> (running both and fusing), 1.1% vs 6.4%. Routing *between retrievers* is bad;
> routing *how deep to go with the same retrievers* is good. Every rung here
> keeps all lanes fused and only adds stages.

**Prefetch during dead time.** For voice specifically, VoiceAgentRAG's dual-agent
prefetching and the Predictive Prefetching line of work both hide retrieval
inside conversational think-time. Reported 110 ms → 0.35 ms on cache hits at a
75% hit rate. Additive to everything above; add it last, and only if S1 measures
too slow in practice.

---

## 7. Live updates

Unchanged from [incremental-updates](../incremental-updates.md), and every new layer
obeys it:

| Layer | Insert cost | Delete |
|---|---|---|
| L1 text cache | one hash + write | by key |
| L2 FTS5 | one row per doc | FTS5 delete |
| L3 vectors | chunks of that doc | by page_id (already correct) |
| L4 entity matrix | that doc's rows | that doc's rows |
| L5 links+aliases | that doc's links; re-resolve matching danglers | that doc's rows |
| L6 pages | that doc's pages + touched topic labels | page delete |

Still required first, all verified against `216d96f`: `PRAGMA journal_mode=WAL`,
`IngestCache` moved off whole-file JSON, atomic page writes via tmp + `os.replace`
(`project.py:88`), and a single-writer ingest queue if documents arrive
concurrently — related documents collide on shared entity pages by design.

Two-tier freshness stays the definition of "live": the raw source is searchable
in ~1 s via L1+L2+L3, and the answer gets better when the compiled page lands.

**Industry corroboration.** Zep/Graphiti is the closest production system to
these constraints — real-time incremental knowledge-graph updates with no batch
recomputation, bi-temporal versioning, hybrid semantic+BM25+graph-traversal
retrieval, reporting 94.8% vs 93.4% on DMR, up to +18.5% accuracy on LongMemEval
with 90% lower latency, and sub-200 ms p95 retrieval in their hosted engine.
Their stated design principle is the same one recommended here: *remove LLM
summarization from the query path.* (Vendor-published; the architecture is the
transferable part.)

---

## 8. Corrections to earlier documents in this directory

Stated plainly because they change the advice.

**8.1 "Hierarchical and community structures cannot be updated incrementally"
was too strong** ([incremental-updates](../incremental-updates.md) §2).

Both have published incremental algorithms. **adRAP** maintains a RAPTOR tree by
holding fitted UMAP/GMM models and regenerating only affected clusters and their
ancestors. **HIT-Leiden** maintains Leiden communities with work bounded to the
2-hop neighbourhood of affected supernodes.

The conclusion survives on cost and quality instead of impossibility, which is a
better reason. adRAP's own numbers: 638 s vs 1,093 s on QASPER and 342 s vs
524 s on QuALITY — roughly 1.7×, not the order of magnitude the phrase
"incremental" suggests — with ~3% lower context relevance and *worse* performance
on the multi-hop split specifically. Paying for a structure that is 1.7× cheaper
to maintain, lossy, and weakest exactly where you need it is a bad trade. The ❌
marks in that table should read "possible but not worth it," not "impossible."

**8.2 Reranking was wrongly deprioritized** ([combining-rag-strategies](../combining-rag-strategies.md)
§4, "breaks the one-call-per-query cost profile").

A cross-encoder is a small local forward pass, not an LLM call. At 20–50 ms over
a short candidate list it fits inside `deep` comfortably, and it is the step that
took Anthropic's contextual retrieval from 49% to 67% failure reduction. It moves
from backlog to rung S3.

**8.3 "The graph lane is query-blind" understated the problem.** It is not merely
that the signal is unfiltered — the *combination shape* is the losing one. §4.2:
seeded PPR gains up to +9.7 points; score-blended PPR loses up to 9.5. The fix is
structural, not a matter of tuning the quota.

---

## 9. Build order

Supersedes the P0–P8 ordering in
[`../../future_work/retrieval-vs-sota/work-items.md`](../../future_work/retrieval-vs-sota/work-items.md).
Each step is independently shippable and independently measurable.

| | Step | Buys | Size |
|---|---|---|---|
| 1 | WAL, atomic writes, `IngestCache` → SQLite | safety under any concurrency | hours |
| 2 | Eval set (~30 questions, tagged single-hop / multi-hop / thematic / intra-doc) | the ability to tell whether 3–8 helped | half a day |
| 3 | L1 text cache | removes the largest query-time cost | small |
| 4 | L2 FTS5 + `bm25()`, RRF against the existing dense lane | a working hybrid; the precondition for everything | small |
| 5 | L5 persisted as raw links + aliases | incremental graph, no dangling-link bug | small |
| 6 | L4 entity matrix from the alias dictionary | multi-hop, at zero LLM cost | medium |
| 7 | **S2: replace `blend_graph_results` with PPR seeded from the fused list** | the +9.7-point change | medium |
| 8 | Profile flag: `voice` / `balanced` / `deep` | the knob | small |
| 9 | Embed source chunks; two-tier freshness | "live" in the useful sense | small |
| 10 | S3 local cross-encoder | precision at the top of the list | medium |
| 11 | Topic pages + maintained `overview.md` | thematic comprehension | medium |
| 12 | S4 agentic loop over the lanes as tools | the deep end | medium |
| 13 | `rebuild` | drift becomes maintenance, not decay | medium |
| 14 | Score-variance escalation; prefetch for voice | automatic knob; voice headroom | later |

**Eval moves to step 2, ahead of all optimization.** Steps 4, 7 and 10 each claim
a specific measurable gain; without the eval set they are faith.

---

## 10. What would falsify this

- **If the corpus stays small** (hundreds of pages), the dense brute-force scan
  is already fast and L4/S2 buys less than the eval will show noise. Check corpus
  size before building step 6.
- **If multi-hop questions are rare in practice**, S2's +9.7 points is a benchmark
  artifact for this use case. The eval set's multi-hop tag answers this directly.
- **If topic pages drift faster than they are useful**, §5 option 1 fails and the
  honest fallback is query-time clustering behind `research` only, accepting the
  latency.
- **If S1 measures above ~150 ms after step 3**, the bottleneck is elsewhere
  (probably the embedding round-trip) and prefetching or a local embedding model
  becomes step 8, not step 14.

---

## 11. Sources

Peer-reviewed / accepted:
- [LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora](https://arxiv.org/abs/2510.10114) (ICLR 2026) · [code](https://github.com/DEEP-PolyU/LinearRAG) — Table 2 numbers read from the PDF
- [GraphRAG-Bench — *When to use Graphs in RAG*](https://arxiv.org/pdf/2506.02404) (ICLR'26)

Preprints (numbers read from the papers where quoted):
- [Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA (SPRIG)](https://arxiv.org/abs/2602.23372) — the GraphRRF vs RRF+PPR ablation
- [A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces](https://arxiv.org/html/2602.03442v1) · [code](https://github.com/Ayanami0730/arag)
- [Recursive Abstractive Processing for Retrieval in Dynamic Datasets (adRAP)](https://arxiv.org/html/2410.01736v1)
- [Maintaining Leiden Communities in Large Dynamic Graphs (HIT-Leiden)](https://arxiv.org/abs/2601.08554)
- [Cost-Aware Query Routing in RAG: Empirical Analysis of Retrieval Depth Tradeoffs (CA-RAG)](https://arxiv.org/html/2606.02581v1)
- [AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation](https://arxiv.org/html/2606.29090)
- [VoiceAgentRAG](https://arxiv.org/html/2603.02206v1) · [Predictive Prefetching for RAG](https://arxiv.org/pdf/2605.17989)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)

Vendor / practitioner — architecture durable, benchmarks self-reported:
- [Contextual Retrieval — Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)
- [Graphiti / Zep documentation](https://help.getzep.com/graphiti/getting-started/welcome) · [Neo4j on Graphiti](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [BERTopic — online topic modelling](https://maartengr.github.io/BERTopic/getting_started/online/online.html)
- Reranker latency surveys (BGE-2 ~20–50 ms on top candidates; mxbai-rerank-v2) — [mixedbread](https://www.mixedbread.com/blog/mxbai-rerank-v2)

Not read, flagged as thin ice: vstash (local-first hybrid retrieval with adaptive
fusion) — retrieved but the PDF summary was too vague to cite; the primary source
for the 42.3%/31.2% drift figures still unverified
([combining-rag-strategies](../combining-rag-strategies.md) §3.4); LongMemEval and
BEAM as off-the-shelf evals for drift.
