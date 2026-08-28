# llmwiki retrieval vs. RAPTOR, GraphRAG, LightRAG

Assessed 2026-08-27 against `216d96f` (branch `dotenv-and-parser-fixes`).
Line references are from that commit and will drift.

Companion document: [work-items.md](work-items.md) — the prioritized fixes,
revised 2026-08-27 against the latency and incremental-update constraints.

External research this assessment feeds into:
[`../../research/`](../../research/README.md).

---

## 1. Positioning: this is a different class of system

RAPTOR, GraphRAG, and LightRAG all build a **machine-readable index structure**
at ingest time — a summary tree, a community hierarchy, an entity/relation
graph. The structure is internal scaffolding. Nobody reads it.

llmwiki does **write-time knowledge compilation**: the LLM produces *prose
pages a human reads*, and retrieval runs over that artifact. The index and the
deliverable are the same object.

This matters for how the system should be evaluated. Retrieval accuracy is only
half the metric; "is the wiki any good" is the other half. A straight benchmark
comparison against the systems below misleads in both directions — it
undersells the artifact and oversells the retrieval.

| | Index built at ingest | Update cost | Query cost | Global sensemaking | Artifact is |
|---|---|---|---|---|---|
| **RAPTOR** | GMM-clustered recursive summary tree over chunks | re-cluster | 1 call | via upper tree levels | internal |
| **GraphRAG** | KG + Leiden communities + LLM community reports | full re-index | map-reduce over community reports (expensive) | its entire design goal | internal |
| **LightRAG** | entity/relation graph + dual-level keyword index | incremental (its headline claim) | 1 call | high-level keyword lane | internal |
| **HippoRAG** | open KG + Personalized PageRank | incremental | ~1 call | weak | internal |
| **llmwiki** | LLM-written wiki pages + `[[wikilinks]]` | 2 LLM calls, no global recompute | 1 call | **absent** | **the product** |

---

## 2. Where llmwiki is genuinely ahead

**Incremental update is more native than LightRAG's.** LightRAG's pitch is
avoiding GraphRAG's full re-index. llmwiki goes further: adding a document is
two LLM calls, deterministic file writes, and embedding only the touched pages
(`ingest/pipeline.py:432`, `_index_without_prune` — the comment at :425 about
deliberately skipping prune is correct and load-bearing). There is no global
structure to recompute, ever.

**Contradiction handling at write time.** The ingest prompt asks the model for
conflicts with existing pages, and `reviews.md` escalates them to a human.
RAPTOR, GraphRAG, and LightRAG have no notion of conflict — they index both
claims and leave it to the reader. This is a real capability none of them have.

**Graceful degradation.** The keyword + graph path needs no embedding model and
no dependencies at all; a dead embedding endpoint degrades rather than fails
(`retrieval/pipeline.py`, the `ProviderError` branch). All three comparators
hard-require embeddings.

**Chunk contextualization, cheaply.** `_embedding_text` (`embeddings.py:353`)
prefixes each chunk with title + heading breadcrumb. That is a poor-man's
version of contextual retrieval, capturing much of the benefit without a
per-chunk LLM call.

**Query cost.** One LLM call. GraphRAG global search is map-reduce over
community reports and is dramatically more expensive per query.

---

## 3. Where llmwiki is behind SOTA

### 3.1 The keyword lane is not BM25, and is not close

`_token_match_score` (`retrieval/keyword.py:155`, used at :202–203):

```python
return sum(1 for token in tokens if token in lowered)
```

Three problems:

- **Presence, not frequency.** A page mentioning the term once scores the same
  as one built around it.
- **No IDF.** A rare, discriminating term counts exactly as much as a common one.
- **No length normalization.** Long pages win by surface area.
- **Substring matching.** `"art"` matches `"cartesian"`; `"ion"` matches almost
  everything.

Every SOTA hybrid baseline uses BM25 for this lane. Highest value-per-line fix
in the codebase.

### 3.2 Graph expansion ignores the query

`blend_graph_results` scores expansion candidates by seed rank alone
(`retrieval/graph.py:182`):

```python
candidate_scores[neighbor] += 1.0 / (rank + 1)
```

Nothing checks whether the neighbour has anything to do with the question.
Meanwhile the well-designed 4-signal `relevance()` function (`graph.py:92`) is
**only called by `related_pages` (`graph.py:129`) and never by the search
pipeline**.

Compounding it, `graph_result_quota` floors at `max(1, ...)` (`graph.py:160`),
so the reserved 15–30% of the result window is spent unconditionally — even
when every candidate is irrelevant.

HippoRAG's Personalized PageRank and LightRAG's dual-level keyword retrieval
both keep graph traversal query-conditioned. This does not.

### 3.3 Chunk-level precision is bought, then discarded

The vector lane retrieves *chunks*, `group_by_page` (`embeddings.py:248`)
collapses them to *pages*, and then `_pack_context` packs the **entire page**
(`query.py:213`):

```python
content = document.content if document else result.snippet
```

You pay for chunk embeddings and throw the localization away. With `top_k=20`
against a ~102KB page budget this is really "retrieve broadly and let the
long-context model sort it out" — a defensible 2026 bet, but a different one
than the pipeline's architecture implies.

### 3.4 No reranking

RRF output goes straight into the context packer. A cross-encoder or LLM
reranker after fusion is standard in modern hybrid stacks and is typically the
single largest quality jump available.

### 3.5 No global sensemaking, and `overview.md` is a stub

The README notes Louvain community detection was not ported — fine. But the
fallback is not working either: **`overview.md` is written at init and never
updated.** `ingest/writer.py:77` explicitly rejects model-authored rewrites of
it, and there is no `update_overview()` counterpart to `update_index()`.

So corpus-level questions ("what are the main themes here?", "what changed in
my thinking across these papers?") — precisely the query class GraphRAG exists
to serve — are answered from `index.md` (a link catalog) plus whatever pages
happen to keyword-match. This is the widest capability gap in the system.

### 3.6 Single-shot retrieval

No query decomposition, no HyDE, no iterative retrieve-and-reason (IRCoT,
Self-RAG, CRAG). One hop, one round. Multi-hop questions fail unless the wiki
happens to have compiled the hop into a single page — which, to be fair,
write-time compilation sometimes does. That is a real if unquantified hedge.

### 3.7 No evaluation harness

The tuning constants are all inherited from the Rust port and unvalidated
against any corpus:

| Constant | Location |
|---|---|
| `200 / 50 / 20 / 5 / 1` scoring weights | `retrieval/keyword.py` |
| `SOURCE_SCORE_FACTOR = 0.6` | `retrieval/keyword.py` |
| `RRF_K = 60` | `retrieval/graph.py` |
| 15–30% graph quota | `retrieval/graph.py:149` |
| `tail * 0.3` page blending | `embeddings.py:271` |
| 50% / 5% / 15% budget split | `budget.py` |

RAPTOR, LightRAG, and GraphRAG all report on public benchmarks (QuALITY,
NarrativeQA, QASPER, UltraDomain). There is currently no way to know whether
any of the above is right for this corpus, which makes tuning any of §3.1–3.6
guesswork.

---

## 4. The failure mode unique to this design

**Compounding drift.**

Pages are rewritten by a model that reads previously written pages as context.
An early misreading becomes evidence for later ingests. RAPTOR summarizes
bottom-up from immutable leaves, so an error stays contained to one node. Here
it propagates — and the SHA256 ingest cache means re-running `add` will not
re-derive the page.

Mitigations already present:

- raw sources are immutable and remain retrievable at 0.6× damping
- `log.md` is append-only and greppable
- everything is plain markdown in git, so drift is diffable in principle

Mitigations absent:

- no consistency check between a page and the source it claims to summarize
- no periodic re-derivation or "rebuild from sources" path
- no detection that page X now contradicts source Y

At tens of documents this is theoretical. At 500 it is the thing to worry
about, more than any retrieval score in §3.

---

## 5. Summary

The **retrieval** layer is roughly 2019-era hybrid search plus one-hop link
expansion. It is behind SOTA and the gaps are concrete and fixable.

The **index** layer is doing something none of these systems attempt — compiling
a durable, auditable, human-editable artifact — and it is the part worth
defending and building on.

See [work-items.md](work-items.md) for what to do about §3.
