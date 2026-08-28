# Latency knobs: which architectures let you trade speed for quality

Compiled 2026-08-27 against llmwiki `216d96f`.

**Question:** which retrieval approaches expose a usable latency/quality dial —
fast enough for voice at one end, "take your time and think" at the other?

Companion: [combining-rag-strategies.md](combining-rag-strategies.md).

---

## 1. There are three knobs, not one

They sit at different pipeline stages and their costs differ by orders of
magnitude.

| Knob | Stage | Cost per unit | Range |
|---|---|---|---|
| **Index eagerness** | ingest | none at query time — *sets the floor* | huge |
| **Retrieval breadth** | query, local | ~ms if in-memory | wide, cheap |
| **Query-time LLM calls** | query, network | 100s of ms – seconds each | dominates |

The test that follows:

> An approach has a usable latency knob if it can vary quality **without
> varying the number of query-time LLM calls.** If the knob *is* LLM calls, it
> is a quality dial with a bad floor.

## 2. Ranking

| Approach | Knob | Floor | Mechanism |
|---|---|---|---|
| RAPTOR | narrow, clean | **very low** | top-k + tree levels; pure ANN, no query-time LLM. Fastest retriever in GraphRAG-Bench |
| Hybrid + RRF (llmwiki) | good, cheap | **very low** | lanes on/off, top-k, optional rerank — all local |
| HippoRAG | weak | low | PPR is fixed-cost, single-step |
| E²GraphRAG | automatic, not user-facing | low | routes graph-local vs tree-global *per query* |
| LazyGraphRAG | **best in the field** | medium–high | explicit relevance-test budget, smooth quality curve |
| PageIndex / agentic | wide range | **very high** | knob is traversal depth; every step is an LLM call |
| GraphRAG global | poor | very high | map-reduce over communities; subsampling breaks the coverage guarantee that is its purpose |

### The inverse relationship

**Knob width and latency floor are inversely related.** LazyGraphRAG and
agentic retrieval have the widest quality range *because* their knob is
query-time LLM work — which is exactly why their floor is high. Eagerly-indexed
systems have low floors and narrower ranges.

So "has the best dial" and "is right for voice" have nearly opposite answers.
LazyGraphRAG is the wrong choice for voice despite winning on knob quality.

---

## 3. Voice budgets

- ~200ms total response latency to feel conversationally natural
- ~100ms for retrieval to land under an 800ms end-to-end budget
- a single vector DB query is **50–300ms of network round-trip** — enough to
  exhaust the budget before generation starts

Implications for any architecture: no extra LLM calls in the retrieval path (no
reranker, no query rewriting, no agentic traversal), retrieval in-memory, and
stream the answer.

[VoiceAgentRAG](https://arxiv.org/html/2603.02206v1) reports a dual-agent
approach — a background process predicting follow-ups and pre-warming an
in-memory semantic cache during the 3–7s between conversational turns, with the
foreground checking cache first. Reported 110ms → 0.35ms on hits (316×) at a 75%
hit rate. The transferable idea is **prefetching during dead time**; it does not
require adopting their architecture.

---

## 4. llmwiki's position

Because it pays everything at ingest, the query path can be zero network calls
before the answer call:

| Stage | Cost | Network |
|---|---|---|
| keyword lane | in-memory scan | no |
| graph lane (`build_graph`) | in-memory | no |
| **vector lane (`embed_query`)** | **50–300ms** | **yes** |
| rerank | does not exist | — |
| answer | streaming | yes, unavoidable |

The vector lane is the only retrieval-path network hop, and it is already
optional. **Embeddings disabled is effectively a voice profile**: keyword +
graph, one LLM call total.

### The knobs already exist, uncoordinated

| Knob | Where |
|---|---|
| `--top-k` | `cli.py:510` (ask), `:529` (search) |
| `LLMWIKI_EFFORT` `off`…`max` | `reasoning.py`, already per-lane via `for_ingest()` |
| `--no-sources` | `cli.py:519` |
| embeddings on/off | `LLMWIKI_EMBEDDING_MODEL` |
| page budget → TTFT | `budget.py` `PAGE_BUDGET_FRAC`, `max_page_size` |

### The largest hidden cost

`--no-sources` is not only a quality knob. `load_documents`
(`retrieval/keyword.py`) calls `extract_text(path)` on every file in
`raw/sources/` on every query, and `extract_text` (`parsers.py:53`) has **no
cache of any kind**. Every `ask` re-parses every PDF in the corpus from scratch.

On a 50-PDF wiki this is seconds — plausibly dominating the LLM call at small
corpus sizes — and it scales linearly with corpus size while producing identical
output every time. Sources are on by default.

Fix: cache extracted text on disk keyed by SHA256 (already computed in
`IngestCache`) or alongside the vector store. Turns the worst latency term into
a file read, and helps every profile.

---

## 5. Proposed shape

One profile flag setting the knobs coherently rather than expecting per-knob
tuning:

| | `fast` (voice) | `balanced` (today's default) | `thorough` |
|---|---|---|---|
| sources | off | cached | cached |
| embeddings | off | on | on |
| `top_k` | ~8 | 20 | 40 |
| effort | `off` / `low` | provider default | `high` |
| page budget | reduced (TTFT) | 50% | 50% |
| rerank, if added | never | no | yes |
| navigation lane, if added | never | no | yes |

`fast` is local retrieval, no network hop before generation, one streaming call
— a genuine sub-second path that the current architecture already supports.
Only the source cache and the flag itself need building.

---

## 6. Reconciliation with the pairing research

This revises [combining-rag-strategies.md](combining-rag-strategies.md) §3.3.

The PageIndex-style navigation lane recommended there is an **agentic** lane —
its cost is LLM calls proportional to traversal depth. By §2 above that gives it
a high latency floor. It remains the cheapest fix for the sensemaking gap, but
it belongs in `thorough` only, never in a voice path.

Same for query-time cluster summarization (§3.3, second proposal): it is
LazyGraphRAG's deferred-work trick, and deferred work is precisely what a
latency-sensitive profile cannot afford. Correct for `thorough`; wrong as a
default.

**General principle:** every recommendation in the pairing document that moves
work *to* query time improves quality and worsens the floor. In a system with a
latency profile, those belong behind the profile, not in the default path.
