# Work items

Derived from [README.md](README.md) and the research in
[`../../research/`](../../research/README.md).

**Revised 2026-08-27** against three stated constraints:

1. **Index-time cost is acceptable** — pay at ingest.
2. **Retrieval must be fast** — query-time latency is the priority.
3. **Updates must be live and incremental** — no rebuild to add a document.

These three together decide most of the list. They rule *in* eager
precomputation, rule *out* anything that defers work to query time, and rule out
any structure that cannot be updated in place. See
[`../../research/latency-knobs.md`](../../research/latency-knobs.md) §2 and
[`../../research/incremental-updates.md`](../../research/incremental-updates.md) §2.

> **Superseded 2026-08-27 (later same day) by
> [`../../research/target-architecture.md`](../../research/target-architecture/README.md).**
> That document keeps the substance of everything below but changes the ordering
> and three specifics: BM25 should be SQLite FTS5 rather than hand-rolled (§3, L2);
> the graph fix is *seeded PPR*, not a re-weighted score blend (§4.2); and
> reranking moves out of the backlog into a first-class rung (§8.2). It also adds
> the entity layer and the agentic rung, which are not in this list at all. Use
> §9 there as the build order; use the P-items here for the implementation detail
> they carry.

Nothing here is started. Sizes are rough.

---

## The shape of the problem

llmwiki pays the largest index cost of any system reviewed — full LLM
compilation of prose pages — and then still rebuilds cheap structures on every
query: source text, the link graph, and (once BM25 lands) corpus statistics.

**It is philosophically eager and operationally lazy.** Most of the latency is
there, not in the architecture. P1–P3 move that work to ingest, where constraint
1 says it belongs.

---

## P0 — Durability and concurrency

**Addresses:** [incremental-updates](../../research/incremental-updates.md) §7
**Touches:** `embeddings.py`, `ingest/cache.py`, `project.py`
**Size:** small — hours, not days

Preconditions for everything else. Cheap, and currently unsafe under any
concurrent read/write.

| Gap | Where | Fix |
|---|---|---|
| No WAL on SQLite | `embeddings.py`, `VectorStore.__init__` | `PRAGMA journal_mode=WAL` — writers currently block readers |
| Whole-file JSON read-modify-write | `ingest/cache.py` | concurrent ingests clobber entries; move to SQLite |
| Non-atomic page writes | `project.py:88` (bare `write_text`) | tmp file + `os.replace`; a query reading mid-write currently sees a torn page |

---

## P1 — Cache extracted source text

**Addresses:** [latency-knobs](../../research/latency-knobs.md) §4
**Touches:** `parsers.py`, `retrieval/keyword.py`
**Size:** small

The single largest latency cost in the system, and it is on by default.

`load_documents` calls `extract_text(path)` on every file in `raw/sources/` for
every query, and `extract_text` (`parsers.py:53`) has **no cache of any kind**.
Every `ask` re-parses every PDF in the corpus from scratch — seconds on a
50-PDF wiki, plausibly dominating the LLM call at small corpus sizes, scaling
linearly while producing identical output each time.

**Design.** Key on SHA256, which `IngestCache` already computes. Store beside
the vector store. Incremental by construction: one row per source, written at
ingest, read at query. Never invalidated except by content change.

**Done when** `ask` on a PDF-heavy project does no parsing.

---

## P2 — Persist the graph

**Addresses:** [incremental-updates](../../research/incremental-updates.md) §3
**Touches:** `retrieval/graph.py`
**Size:** medium

`build_graph` reconstructs the entire link graph on every query. Precompute it —
but store it in a form that survives incremental insertion.

**Design — persist raw links and the alias table, not resolved adjacency.**

The non-local hazard is *incoming* edges: a previously dangling `[[Foo]]` in an
existing page resolves the moment Foo is created. A materialized adjacency list
misses this silently.

- store `(source_page, raw_link_text)` rows and `(normalized_alias → path)` rows
- index dangling links by normalized alias
- on insert: add the new page's aliases, re-resolve only the dangling entries
  matching them — O(matching dangling links), not O(corpus)
- on edit or delete: delete-then-reinsert that page's rows, the shape
  `VectorStore.upsert_page` already uses

---

## P3 — BM25, with incrementally maintained statistics

**Addresses:** README §3.1; [incremental-updates](../../research/incremental-updates.md) §4
**Touches:** `retrieval/keyword.py`
**Size:** ~50 lines plus tests. No new dependencies.

`_token_match_score` (`retrieval/keyword.py:155`, used at :202–203) is term
*presence*, not frequency: no IDF, no length normalization, and substring
matching (`"art"` matches `"cartesian"`). RRF over a broken lexical lane
forfeits most of what hybrid retrieval exists to capture.

```
score(page, query) = Σ_token  IDF(token) · (tf · (k1+1)) / (tf + k1 · (1 - b + b · len/avglen))
```

Start at `k1 = 1.2`, `b = 0.75`.

**Statistics are incremental** — N as a counter, avgdl as a running sum, df(term)
incremented per unique term on insert and decremented on delete. Scores are
computed at query time from the statistics, so nothing goes stale as N grows.

**Decisions to make:**

- **Word-boundary matching.** The substring `in` test must go. Tokenize
  documents as `tokenize.py::tokenize_query` tokenizes queries, minus stop-word
  removal — the existing asymmetry (stop words stripped from queries only) is
  correct and should be preserved.
- **Keep the structural bonuses.** Filename-exact (+200), phrase-in-title (+50),
  and phrase-in-body (+20/occurrence) are useful and are not what BM25 replaces.
  BM25 replaces the `+5`/`+1` token terms only. Because BM25 is on a different
  scale, the bonuses need re-derivation — do not assume current magnitudes still
  dominate correctly.
- **CJK.** `tokenize_query` expands CJK into bigrams; document-side tokenization
  must expand identically or the two sides will not meet.

---

## P4 — Query-conditioned graph expansion

**Addresses:** README §3.2
**Touches:** `retrieval/graph.py`
**Size:** ~30 lines. Land with P2 — same file.

`blend_graph_results` scores expansion candidates by seed rank alone
(`graph.py:182`); nothing checks relevance to the query. The reserved 15–30% of
the window is therefore not a second opinion but unfiltered link structure.

This matters because of the Integration-vs-Selection result
([combining-rag-strategies](../../research/combining-rag-strategies.md) §1.2):
running two retrievers and fusing beat routing 6.4% to 1.1% — but only because
*both* retrievers were query-relevant.

1. **Use the relevance model that already exists.** `relevance()` (`graph.py:92`)
   implements direct link ×3.0, shared source ×4.0, Adamic-Adar ×1.5, shared type
   ×1.0, and is currently reachable only from `related_pages` (`graph.py:129`).
   Blend it with seed rank rather than replacing it: seed rank encodes how good
   the anchor was, relevance encodes how connected the candidate is.
2. **Add a query term.** Neither signal looks at the query. Cheapest version:
   reuse the candidate's keyword score, already computed in Phase 1 for every
   document that matched at all.
3. **Let the quota reach zero.** `graph_result_quota` floors at `max(1, ...)`
   (`graph.py:160`). Gate on an absolute score floor — an empty graph slice is a
   valid outcome, not a bug.

**Watch out:** `related_pages` powers relatedness explanations. Prefer adding the
query term inside `blend_graph_results` over mutating `relevance()`.

---

## P5 — Evaluation harness

**Addresses:** README §3.7 — and gates honest assessment of P1–P4
**Touches:** new `tests/eval/` or a standalone script; no `src/` changes
**Size:** half a day, mostly writing questions

Every tuning constant in the retrieval stack (README §3.7) is an unvalidated
inheritance from the Rust port. Without measurement, P3 and P4 are changes of
unknown sign.

It also answers the question that decides the architecture: **do compiled wiki
pages beat a plain chunk index on this corpus?** Right now that is an article of
faith.

- **One real wiki**, dozens of pages, not three.
- **~30 questions**, each labelled with the page(s) that should be retrieved.
  Write them from the sources, not the wiki, so labels do not inherit the
  system's blind spots.
- **Cover query classes separately** — they fail differently:
  direct lookup · synthesis (two+ pages) · multi-hop (README §3.6) ·
  corpus-level (README §3.5, expected to fail — record the baseline) ·
  negative controls (the wiki genuinely does not know; correct behaviour is
  saying so, which `query.py`'s citation rules already request)
- **Metrics:** recall@k and MRR on retrieval; separately, citation rate from
  `Answer.pages_cited / pages_used` — plumbing already exists and is a decent
  cheap proxy for whether retrieved pages were useful.
- **No LLM judge to start.** It adds cost and a second source of variance before
  the first is understood.

---

## P6 — Latency profile flag

**Addresses:** [latency-knobs](../../research/latency-knobs.md) §5
**Touches:** `cli.py`, `config.py`
**Size:** small once P1–P3 land

Every knob already exists — `--top-k`, `LLMWIKI_EFFORT`, `--no-sources`,
embeddings on/off, page budget. None of them are coordinated.

| | `fast` (voice) | `balanced` (today) | `thorough` |
|---|---|---|---|
| sources | off | cached (P1) | cached |
| embeddings | off | on | on |
| `top_k` | ~8 | 20 | 40 |
| effort | `off` / `low` | provider default | `high` |
| page budget | reduced (TTFT) | 50% | 50% |
| rerank / navigation lane | never | no | yes |

`fast` is local retrieval, no network hop before generation, one streaming call.
The only retrieval-path network hop today is `embed_query` (50–300ms), and it is
already optional.

---

## P7 — `rebuild`

**Addresses:** README §4 (drift)
**Size:** medium — mostly prompt and ordering design

Regenerate every page from `raw/` + `schema.md`, ignoring current wiki state.

This is **the change that de-risks the architecture.** It converts drift from
existential to periodic maintenance, and demotes the wiki from *the only copy of
the knowledge* to *a cache over immutable sources*. Once pages are re-derivable,
eager LLM compilation is strictly better than a RAPTOR-style tree: same
precomputed abstraction layer, plus legibility, plus error containment.

Not in tension with incremental updates — it is the **merge step**, in Lucene's
sense: immutable segments, search the union, compact in the background. `add`
stays incremental; `rebuild` stands to it as `VACUUM` does to `INSERT`. See
[incremental-updates](../../research/incremental-updates.md) §5.

---

## P8 — Single-writer ingest queue

**Addresses:** [incremental-updates](../../research/incremental-updates.md) §7
**Size:** medium
**Only if** genuinely concurrent live ingest is required. One-at-a-time appends
need only P0.

Two concurrent ingests touching the same entity page lose an update — and
related documents collide on shared entity pages *by design*, which is the whole
point of the architecture. Locks are the wrong tool; serialize writes instead.

Composes with **two-tier freshness**
([incremental-updates](../../research/incremental-updates.md) §6): make the raw
source searchable in ~1s (extract, embed, insert — it already participates in
retrieval at 0.6× damping), and queue the two-call compile behind it. A document
becomes findable immediately, then gets better when its page lands.

---

## Behind `thorough` only

Both improve quality by moving work *to* query time, which is correct for a
patient profile and wrong as a default (constraint 2).

**Navigation lane, PageIndex-style.** llmwiki already maintains the tree for
free: `schema.md` → `index.md` → page-type directories → pages, with
`[[wikilinks]]` as cross-edges. An LLM reasoning down it is the cheapest fix for
the sensemaking gap (README §3.5) and yields an auditable retrieval path. But it
is agentic — cost is LLM calls proportional to depth.

**Query-time cluster summarization.** For corpus-level questions, cluster the
in-memory graph at query time and summarize only the matched cluster, rather
than pre-building community reports. LazyGraphRAG's deferred-work trick; zero
index cost, zero drift surface, high latency floor.

---

## Backlog

**Maintain `overview.md`.** Written at init and never updated;
`ingest/writer.py:77` rejects model rewrites and there is no `update_overview()`.
Largely superseded by the two items above, but a cheap deterministic
regeneration from `index.md` plus page titles and types remains the low-effort
option.

**Pack chunks, not whole pages** (README §3.3). `_pack_context` (`query.py:213`)
packs full page content, discarding the localization the vector lane paid for.
Test rather than assume — whole-page packing may win with current long-context
models, in which case the finding is that the chunk-level vector lane is
over-built for its role.

**Reranking** (README §3.4). Standard and effective, but it is a query-time LLM
or cross-encoder call against a system whose current cost is exactly one call —
directly opposed to constraint 2. Revisit only if P5 shows headroom after
P3 and P4.

**Drift detection.** Largely absorbed by P7. If `rebuild` proves too expensive to
run often, the cheaper version is a `verify` that re-reads a page against the
sources in its frontmatter and files disagreements into `reviews.md`, reusing
machinery that already exists.

---

## Ruled out

| | Why |
|---|---|
| RAPTOR-style summary trees in ingest | GMM re-clustering is global — violates constraint 3 |
| GraphRAG / HiRAG / E²GraphRAG community detection | Louvain is global — violates constraint 3; and README §3.2 says do not add a second eager index |
| LazyGraphRAG-style deferred indexing | optimized for the opposite of constraint 1, and its query cost violates constraint 2 |
| Query classification / routing between retrievers | Integration beat Selection 6.4% to 1.1% ([combining-rag-strategies](../../research/combining-rag-strategies.md) §1.2) |
