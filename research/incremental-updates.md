# Live incremental indexing: what updates cheaply, what forces a rebuild

Compiled 2026-08-27 against llmwiki `216d96f`.

**Requirement:** add documents live and efficiently, without rebuilding the
index.

Companions: [combining-rag-strategies.md](combining-rag-strategies.md),
[latency-knobs.md](latency-knobs.md).

---

## 1. Current state

### Ingest is already properly incremental

Adding a document touches only the new source, its own pages, appends to
`index.md` and `log.md`, and embeddings for the touched pages
(`_index_without_prune`, with the `page_hash` check skipping unchanged ones).
No global recompute anywhere.

This is LightRAG's headline advantage over GraphRAG, implemented more cleanly:
llmwiki has no global structure that *could* require recomputation.

### But there is no retrieval index to update

The constraint is currently satisfied **vacuously** — nothing persists between
queries, so nothing can go stale. The price is a full rebuild on every query:

| Rebuilt per query | Where |
|---|---|
| every wiki page read from disk | `load_documents` |
| every source file re-parsed (PDFs included) | `extract_text`, uncached |
| the entire link graph | `build_graph` |

That is the worst form of incrementality: you obtain it by paying rebuild
continuously.

### The consequence

The moment the caching work in [latency-knobs.md](latency-knobs.md) §4 lands, a
persistent index exists — and this requirement becomes a **design constraint on
that work**, not a separate project. Build the caches incrementally-updatable
from the start; build them as snapshots and they get rewritten.

---

## 2. What is incrementally updatable

| Structure | Incremental? | Note |
|---|---|---|
| Vector store (per-page chunks) | ✅ trivial | delete+insert by page — already correct |
| Source text cache | ✅ trivial | SHA256-keyed, append-only |
| Wiki pages, `index.md`, `log.md` | ✅ | writes and appends |
| Graph adjacency | ⚠️ with care | §3 |
| BM25 statistics | ⚠️ with care | §4 |
| **RAPTOR tree (GMM clustering)** | ❌ | re-clustering is global |
| **GraphRAG / HiRAG communities (Louvain)** | ❌ | community detection is global |

Everything in llmwiki's design is incrementally maintainable. Nothing in
RAPTOR's or GraphRAG's is.

This is the third independent argument against switching architectures, after
quadrant fit ([latency-knobs.md](latency-knobs.md) §2) and the artifact
question ([combining-rag-strategies.md](combining-rag-strategies.md) §3.2). It
is also precisely the argument LightRAG made against GraphRAG.

---

## 3. The graph: the dangling-link hazard

A new page's *outgoing* edges are local and trivial to add. The non-local effect
is **incoming**: a previously dangling `[[Foo]]` in an existing page now
resolves, because Foo exists.

`build_graph` currently resolves links against an alias table built fresh each
time (path, wiki-relative path, stem, title → path), so this is invisible today.
A materialized adjacency list would silently miss it.

**Design: persist raw links plus the alias table, not resolved adjacency.**

- store `(source_page, raw_link_text)` rows and `(normalized_alias → path)` rows
- index the *dangling* links by normalized alias
- on insert: add the new page's aliases, then re-resolve only the dangling
  entries matching them
- on page edit or delete: delete-then-reinsert that page's rows — the same shape
  `VectorStore.upsert_page` already uses

Cost is O(dangling links matching the new aliases), not O(corpus).

---

## 4. BM25 statistics

BM25 needs three corpus statistics: N (document count), avgdl (average document
length), and df(term) per query term. All three are incrementally maintainable:

- **N** — counter
- **avgdl** — running sum of lengths ÷ N
- **df(term)** — increment per unique term on insert, decrement on delete

IDF shifts slightly for every term as N grows, but scores are *computed at query
time from the statistics* rather than stored, so nothing goes stale. This is why
Lucene has supported incremental BM25 for twenty-five years.

---

## 5. `rebuild` is the merge step, not a contradiction

Lucene's answer to this exact problem: **immutable segments, search the union,
merge in the background.** Writes land in a small new segment; queries search
all segments; compaction runs periodically, off the critical path.

The `rebuild` proposal — regenerate wiki pages from `raw/` + `schema.md`,
ignoring current wiki state — is the merge. It is periodic repair, not the
update path, standing in the same relation to `add` that `VACUUM` does to
`INSERT`.

Incremental writes stay incremental. `rebuild` is what stops years of them from
compounding into drift
([`../future_work/retrieval-vs-sota/README.md`](../future_work/retrieval-vs-sota/README.md) §4).

---

## 6. Two-tier freshness

Ingest is two LLM calls — tens of seconds. That is the floor for a *compiled
page*. The **raw source** can be queryable in about a second: extract, embed,
insert.

Because raw sources already participate in retrieval at 0.6× damping
(`SOURCE_SCORE_FACTOR`, `retrieval/keyword.py:38`), this falls out of the
existing design. A document becomes findable immediately via its raw text, then
gets *better* when its wiki page lands.

Worth making deliberate rather than accidental — it is what "live" should mean
here, and it costs nothing to adopt.

---

## 7. What blocks "live" today

All four verified against `216d96f`.

| Gap | Where | Consequence | Fix |
|---|---|---|---|
| No WAL on SQLite | `embeddings.py`, `VectorStore.__init__` | a writer blocks readers | `PRAGMA journal_mode=WAL` |
| Whole-file JSON read-modify-write | `ingest/cache.py`, `IngestCache.load`/`save` | concurrent ingests clobber each other's entries | move to SQLite |
| Non-atomic page writes | `project.py:88`, bare `write_text` | a query reading mid-write sees a torn page | tmp file + `os.replace` |
| No page-level serialization | — | two concurrent ingests touching one entity page = lost update | single-writer queue |

The last one is what actually bites when documents stream in, because **related
documents collide on shared entity pages by design** — that is the whole point
of the architecture.

The simplest correct answer is a **single-writer ingest queue**, not locks. It
also composes with §6: queue the compile, make the raw source searchable
immediately.

If "live" means one-at-a-time appends, only WAL and atomic writes are required.

---

## 8. Ordering

1. Atomic writes + WAL — small, unblocks read-during-write
2. Source text cache — largest latency win; design as an incremental store
3. Persisted graph as raw-links + alias table — incremental by construction
4. BM25 with maintained N/avgdl/df — incremental by construction
5. Eval set — what tells you whether any of this worked
6. `rebuild` — the merge/repair pass
7. Single-writer ingest queue — only if concurrent live ingest is needed

Items 2–4 are the caching work from [latency-knobs.md](latency-knobs.md); this
requirement specifies *how* to build them rather than adding new work.

Carried into
[`../future_work/retrieval-vs-sota/work-items.md`](../future_work/retrieval-vs-sota/work-items.md).
