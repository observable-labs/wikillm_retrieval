# Build plan

The executable form of [README.md](README.md). That document argues for the
architecture; this one says what to write, in what order, and how you will know
each step worked.

Written 2026-08-27 against `216d96f`. Nothing here is started.

**Supersedes** the P0–P8 ordering in
[`../../future_work/retrieval-vs-sota/work-items.md`](../../future_work/retrieval-vs-sota/work-items.md),
which remains useful for the implementation detail it carries (BM25 tokenizer
decisions, eval query classes, `relevance()` wiring notes).

---

## How to use this

Every step below has the same four parts:

- **Goal** — one sentence, in terms of a capability, not a file.
- **Touches** — the modules that change.
- **Design** — schemas, signatures, and the decisions that are easy to get wrong.
- **Done when** — a criterion you can check, ideally a number.

Rules that apply to all of them:

1. **Nothing lands without a number.** From step 2 onward, every step reports
   its effect on the eval set. A step that cannot show a change is a step whose
   value is unverified — that is allowed, but say so.
2. **One feature flag per step**, defaulting to *off*, until its number exists.
   The retrieval path is the hot path; do not make it un-revertible.
3. **Each step is shippable alone.** No step depends on a later one.

---

## Step 0 — Baseline

Before changing anything, record what today costs. You cannot claim an
improvement against a number you never took.

Measure on a real corpus, not a toy one:

| Measurement | How |
|---|---|
| `ask` wall clock, p50 / p95 | 20 real questions, `--no-sources` on and off |
| Time inside `load_documents` | crude timer, printed under a debug flag |
| Time inside `extract_text`, summed | same |
| Time inside `build_graph` | same |
| Query-embedding round-trip | same |
| Corpus size | pages, sources, total chars, chunk count |

**Done when:** you have a table showing where the current query second goes, and
you know your corpus size. Corpus size decides whether step 7 is worth doing at
all (see README §10).

---

## Step 1 — Durability

**Goal:** two processes touching the project cannot corrupt it.
**Touches:** `embeddings.py`, `ingest/cache.py`, `project.py`
**Size:** hours

### Design

Three independent, small fixes — all verified as gaps against `216d96f`:

**1a. WAL.** `VectorStore.__init__` opens a plain `sqlite3.connect` with no
pragmas, so a writer blocks readers. Add, immediately after connecting:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA busy_timeout = 5000;
```

`journal_mode` is persistent (stored in the file header); the other two are
per-connection and must be set every time.

**1b. Atomic page writes.** `project.py:88` is a bare `write_text`, so a query
reading during ingest can see a torn page:

```python
def write(self, relative: str, content: str) -> Path:
    path = self.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)          # atomic within a filesystem
    return path
```

**1c. `IngestCache` off whole-file JSON.** `load`/`save` are a read-modify-write
of one JSON document; two ingests racing lose entries. Move it into the same
SQLite database as everything else (step 2).

**Done when:** a script running `add` and `ask` concurrently in a loop for a
minute produces no torn reads, no lost cache entries, and no `database is
locked`.

---

## Step 2 — One database

**Goal:** every derived structure lives in one SQLite file, so adding a document
is one transaction.
**Touches:** new `src/llmwiki/index/store.py`; `embeddings.py` migrates into it
**Size:** small

### Design

This is the step that makes the rest cheap, and it is easy to skip by accident.

Today `vectors.db` holds vectors and nothing else, `IngestCache` is JSON, and the
graph and lexical statistics do not persist at all. If those land as separate
files, "add a document" becomes four un-coordinated writes and you are back to
needing a lock.

Put L1–L5 in **`.llm-wiki/index.db`**, one connection, one schema version, and
make document insertion a single transaction:

```python
with index.transaction():          # BEGIN IMMEDIATE
    index.text_cache.put(sha, text)
    index.lexical.replace(doc_id, text)
    index.vectors.replace(doc_id, chunks)
    index.entities.replace(doc_id, mentions)
    index.links.replace(doc_id, raw_links, aliases)
```

Either the document is fully indexed or it is not indexed. No queue, no lock
file, no partial state — this is the elegant version of the "single-writer
queue" filed in [`../incremental-updates.md`](../incremental-updates.md) §7, and
it removes the need for one in the single-writer case.

Keep the existing `vectors.db` readable and migrate on first open; do not force a
re-embed.

Schema, all of it, in one place:

```sql
PRAGMA user_version = 1;

-- L1: extracted source text
CREATE TABLE IF NOT EXISTS text_cache (
    sha256      TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    text        TEXT NOT NULL,
    extracted_at INTEGER NOT NULL
);

-- documents: the spine every other table hangs off
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,      -- the path, as today
    kind     TEXT NOT NULL,         -- 'wiki' | 'source'
    title    TEXT NOT NULL DEFAULT '',
    sha256   TEXT NOT NULL,
    indexed_at INTEGER NOT NULL
);

-- L2: lexical
CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(
    doc_id UNINDEXED,
    title,
    headings,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- L4: entity <-> document, weights computed at query time from counts
CREATE TABLE IF NOT EXISTS entities (
    entity_id INTEGER PRIMARY KEY,
    alias     TEXT NOT NULL UNIQUE   -- normalize_alias() output
);
CREATE TABLE IF NOT EXISTS mentions (
    entity_id INTEGER NOT NULL,
    doc_id    TEXT NOT NULL,
    count     INTEGER NOT NULL,
    PRIMARY KEY (entity_id, doc_id)
);
CREATE INDEX IF NOT EXISTS mentions_doc ON mentions(doc_id);

-- L5: curated links, stored RAW and resolved at query time
CREATE TABLE IF NOT EXISTS links (
    doc_id    TEXT NOT NULL,
    raw       TEXT NOT NULL,
    norm      TEXT NOT NULL          -- normalize_alias(raw)
);
CREATE INDEX IF NOT EXISTS links_norm ON links(norm);
CREATE TABLE IF NOT EXISTS aliases (
    norm      TEXT PRIMARY KEY,
    doc_id    TEXT NOT NULL
);
```

**Why links are stored raw:** a dangling `[[Foo]]` must start resolving the
moment Foo is created. Resolving at write time silently misses that; resolving
at read time against `aliases` is a join and costs nothing.
([`../incremental-updates.md`](../incremental-updates.md) §3.)

**Done when:** `add` writes one transaction; killing the process mid-`add` leaves
the index consistent; `vectors.db` migrates without re-embedding.

---

## Step 3 — Eval set

**Goal:** the ability to tell whether steps 4–14 helped.
**Touches:** new `src/llmwiki/eval/`, `cli.py`
**Size:** half a day, mostly writing questions

This is deliberately ahead of every optimization. Steps 4, 7, 8 and 11 each claim
a specific measurable gain; without this they are faith.

### Design

**~30 questions**, hand-written against your real corpus, tagged by the class
they exercise — the four capabilities in the brief, so a regression tells you
*which* one broke:

| Tag | What it tests | Target count |
|---|---|---|
| `single-hop` | one page has the answer | 10 |
| `multi-hop` | requires joining two or more documents | 8 |
| `thematic` | "what does this corpus say about X overall" | 6 |
| `intra-doc` | detail buried deep in one long source | 6 |

```yaml
- id: mh-003
  question: "Which of the vendors we evaluated also supplied the 2024 pilot?"
  tags: [multi-hop]
  expect_pages: ["wiki/orgs/acme.md", "wiki/projects/2024-pilot.md"]
  expect_strings: ["Acme"]
  forbid_strings: []
```

**Metrics, none of them requiring an LLM judge:**

| Metric | Definition |
|---|---|
| `recall@k` | fraction of `expect_pages` in the top k results |
| `MRR` | reciprocal rank of the first `expect_page` |
| `answer_hit` | all `expect_strings` present in the answer text |
| `citation_rate` | `pages_cited / pages_used` (both already on `Answer`) |
| `p50` / `p95` retrieval | retrieval time only, excluding the answer call |

Deliberately no LLM judge to start: it adds cost, variance, and a second thing to
debug. Add one later only if `answer_hit` proves too blunt.

**Interface:**

```
llmwiki eval --profile balanced --k 20 [--tags multi-hop]
```

writes `eval/runs/<iso-date>-<profile>.json` and prints a table against the
previous run, so a regression is visible at the moment it is introduced.

### Public suites: use them, but not first

Full survey in [`../evaluation/`](../evaluation/README.md). The short version:

| When | Suite | Answers |
|---|---|---|
| now | your ~30 hand-written questions | does it work on *your* corpus — nothing public can |
| before step 8 | HotpotQA + 2WikiMultiHopQA, SPRIG protocol | is the PPR implementation correct (gold passages labelled, **no LLM judge needed**) |
| before step 8 | GraphRAG-Bench, complex-reasoning level | does the gain land where theory says it should |
| after step 9 | EraRAG's insertion protocol (§ below) | the incremental claim |
| once the format proves out | BenchmarkQED `AutoQ` | scales your corpus-specific set to a few hundred |

The hand-written set comes first because it is the only one that measures *your*
material, and because `recall@k` against known-correct pages is unambiguous and
free. Skip RAGAS / ARES / RAGChecker for now — an LLM judge per question per
metric is cost and variance added to a system whose point is a fast local path.

### The growth test

The plan claims updates are O(document). Make that falsifiable, using EraRAG's
protocol: **insert 5% of the corpus, ten times, measuring at each step.**

| Curve | Falsifies |
|---|---|
| accuracy vs. number of insertions | "quality is stable under growth" |
| update cost per insertion | "updates are O(document), not O(corpus)" |
| incremental vs. from-scratch answer agreement | drift — the reason step 14 exists |

The third is the one nothing published measures well and the one your
architecture most needs: build the index incrementally, build it again from
scratch, ask the same questions, count where the answers differ.

### Design

Full design in [`../evaluation/harness-v1.md`](../evaluation/harness-v1.md). Two
things there that this step's earlier framing missed and that change the work:

- **The harness talks to a `System` protocol, not to `search()` directly.** A
  harness welded to llmwiki can only measure llmwiki against llmwiki. The
  protocol costs ~120 lines and is what lets you evaluate a baseline, a
  competitor, or an external system on the same questions.
- **Ship two baseline adapters with it** — FTS5-only and dense-only, ~30 lines
  each over machinery that already exists. Without them, "recall went up 0.03"
  never becomes "the hybrid beats both of its parts", which is the claim hybrid
  retrieval exists to make and the one steps 5 and 8 both rest on.

Also report **`recall@context`**, not only `recall@k`: `_pack_context`
(`query.py:196`) drops pages that no longer fit the budget, so a page ranked 4th
can be absent from the evidence the model saw. When the two diverge, the problem
is packing, not retrieval.

**Done when:** `llmwiki eval` runs end-to-end, produces a baseline row for every
tag across all three lane configurations, and the null-change test passes — two
consecutive runs on an unchanged corpus must differ by exactly zero. Record the
baseline; this is the number every later step is measured against.

---

## Step 4 — L1 text cache

**Goal:** stop re-parsing every PDF on every query.
**Touches:** `parsers.py`, `retrieval/keyword.py`, `ingest/pipeline.py`
**Size:** small

### Design

`extract_text` (`parsers.py:53`) has no cache of any kind, and `load_documents`
calls it for every file in `raw/sources/` on every query. Identical output, every
time, linear in corpus size.

Write through the cache at ingest, read from it at query:

```python
def extract_text_cached(path: Path, index: Index) -> str:
    sha = sha256_file(path)                 # IngestCache already computes this
    hit = index.text_cache.get(sha)
    if hit is not None:
        return hit
    text = extract_text(path)
    index.text_cache.put(sha, path, text)
    return text
```

Key on **content hash, not path**, so a renamed file is free and an edited one
correctly misses.

**Done when:** step 0's `extract_text` line is ~0 on a warm cache, and `ask` p50
drops by the amount step 0 predicted. This is the largest single latency win
available and it changes no rankings.

---

## Step 5 — L2 lexical, via FTS5

**Goal:** a lexical lane that ranks by term *importance*, not term presence.
**Touches:** new `index/lexical.py`; `retrieval/keyword.py` becomes a thin caller
**Size:** small

### Design

Today `_token_match_score` (`retrieval/keyword.py:155`) counts how many query
tokens appear *anywhere* in the text, by substring. No frequency, no IDF, no
length normalization, and `"art"` matches `"cartesian"`.

Use SQLite FTS5 with the built-in `bm25()`. **Verified available in this
project's interpreter: SQLite 3.51.1, `CREATE VIRTUAL TABLE … USING fts5` and
`bm25()` both work in the stdlib `sqlite3`.** No new dependency, and no
hand-rolled N/avgdl/df maintenance to write and test.

```sql
SELECT doc_id, bm25(lexical, 10.0, 5.0, 1.0) AS score
FROM lexical
WHERE lexical MATCH ?
ORDER BY score          -- bm25() returns negative; smaller is better
LIMIT ?;
```

The three weights are the title / headings / body columns — this is how the
existing structural bonuses (200/50/20/5/1 in `keyword.py`) survive, as column
weights rather than as ad-hoc additions to a score.

Two things to get right:

- **Query construction.** Do not pass raw user text to `MATCH`; FTS5 has an
  operator syntax and unescaped input is both a crash and an injection. Quote
  each token, join with `OR`, keep the existing `tokenize_query`.
- **Sources damping.** `SOURCE_SCORE_FACTOR = 0.6` (`keyword.py:38`) must survive
  — apply it after ranking, or keep sources in a second query.

Fuse with the dense lane by RRF exactly as now (`RRF_K = 60.0`).

**Done when:** `recall@20` on the `single-hop` tag improves, and no tag
regresses. If nothing moves, your corpus is small enough that presence-matching
was sufficient — record that and move on.

---

## Step 6 — L5 persisted links

**Goal:** stop rebuilding the whole link graph on every query.
**Touches:** new `index/links.py`; `retrieval/graph.py`
**Size:** small

### Design

`build_graph(documents)` materializes the entire graph in memory per query. Move
it into `links` + `aliases` (step 2 schema) and resolve at read time:

```sql
SELECT l.doc_id AS src, a.doc_id AS dst
FROM links l JOIN aliases a ON a.norm = l.norm;
```

Insert is that document's rows only. The dangling-link case resolves for free
because resolution happens at read.

Keep `normalize_alias` (`graph.py:39`) exactly as it is — it is the contract
between the two tables, and changing it later means a reindex.

**Done when:** step 0's `build_graph` line is ~0, and graph-expanded results are
identical to before for the same query. This step must not change rankings; it is
pure relocation.

---

## Step 7 — L4 entity layer

**Goal:** a high-recall structural graph, at zero LLM cost.
**Touches:** new `index/entities.py`
**Size:** medium

### Design

The insight that makes this cheap: **you already have an entity dictionary.**
`build_graph` constructs an alias table from path, wiki-relative path, stem, and
title (`graph.py:68–73`), normalized. LinearRAG and SPRIG both use spaCy NER for
this; your dictionary is curated by the ingest LLM and is strictly more precise.

At ingest, for each document, count occurrences of each known alias in its text
and write `mentions` rows. Adding a document is O(aliases × document length) with
an Aho–Corasick automaton, or a plain scan if the alias set is small.

**The rebuild subtlety.** A *new* alias (a new page title) means older documents
may mention it and have no row. Options, in order of preference:

1. On new alias, scan only documents whose text is in L1 — a background pass,
   bounded, and off the query path.
2. Accept staleness until the next `rebuild` (step 13).

Do not resolve this by rebuilding the whole matrix per insert.

Edge weight, computed at query time from counts, per SPRIG:

```
w(e, d) = tf(e, d) · log((N + 1) / (df(e) + 1)) + 1
```

Store `count` only; N and df are `SELECT COUNT(*)` — cheap, always current.

**Hub pruning:** drop the top ~1% highest-degree entities at query time. SPRIG
measured 485 s → 350 s total query time with negligible recall change.

**Done when:** the matrix builds, insert of one document touches only its rows,
and entity-count statistics look sane (no alias matching 90% of documents — if
one does, it is a stopword-like title and belongs in the pruned set).

---

## Step 8 — S2: seeded PPR

**Goal:** the +9.7-point multi-hop change.
**Touches:** new `retrieval/ppr.py`; `retrieval/graph.py` (`blend_graph_results` retires)
**Size:** medium

### Design

**This is the step that matters most, and it is a change of shape, not of
weights.** SPRIG's ablation, from README §4.2:

| | HotpotQA R@10 | 2Wiki R@10 |
|---|---|---|
| RRF alone | 0.851 | 0.697 |
| PPR *seeded from* the RRF list | **0.867** | **0.794** |
| PPR scores *blended into* RRF | 0.782 | 0.602 |

`blend_graph_results` (`graph.py:163–182`) implements the third row: it reserves
a quota of slots and scores neighbours with a query-blind `1.0 / (rank + 1)`.
Replace it, do not tune it.

```python
def ppr(seeds: dict[str, float], adjacency, alpha=0.15, iterations=5) -> dict[str, float]:
    """Sparse personalized PageRank by power iteration.

    seeds: doc_id -> restart mass, L1-normalized. Take these from the fused
    RRF list — the top ~5, weighted by fused rank.
    """
```

Parameters from SPRIG's tuning: `alpha = 0.15`, 5 iterations, top-5 seeds.
Run over the union of L4 (mechanical, dense, high-recall) and L5 (curated,
sparse, high-precision) with different edge weights — one graph, two edge types.

Then **rank by the PPR score**, rather than mixing PPR into the RRF score. The
quota mechanism (`graph_result_quota`, `MIN_GRAPH_RESULT_RATIO`) disappears with
it; there is no reserved slice, because the graph is no longer a separate
opinion to make room for.

`relevance()` (`graph.py:92`) — direct link ×3.0, shared source ×4.0,
Adamic-Adar ×1.5, type affinity ×1.0 — is currently unreachable from search. It
becomes the **edge weight** in the L5 half of the adjacency, which is where a
signal like that belongs.

**Done when:** `recall@20` on the `multi-hop` tag improves and `single-hop` does
not regress. If multi-hop does not move on your corpus, your questions are not
actually multi-hop — check the tag before blaming the implementation.

**Validate the implementation separately from its fitness.** Run HotpotQA and
2WikiMultiHopQA and compare directly against SPRIG's published table (README §4.2):
you should see roughly RRF 0.851 → GraphRRF 0.867 and 0.697 → 0.794. If you do
not reproduce the direction of that gap, the bug is in the PPR, not in your
corpus. This costs no API calls — gold supporting passages are labelled.

---

## Step 9 — The profile flag

**Goal:** one knob, not five.
**Touches:** `cli.py`, `config.py`, new `profiles.py`, `retrieval/pipeline.py`
**Size:** small

### Design

```python
@dataclass(frozen=True)
class Profile:
    name: str            # voice | balanced | deep | research
    top_k: int
    use_graph: bool      # S2
    use_rerank: bool     # S3
    agentic_steps: int   # S4, 0 = off
    effort: str          # feeds reasoning.resolve()
    include_sources: bool
    local_embeddings: bool
```

`--profile` on `ask` and `search`, `LLMWIKI_PROFILE` in the environment, default
`balanced`. Existing flags (`--top-k`, `--no-sources`, `LLMWIKI_EFFORT`) stay and
override the profile, so nothing breaks.

The rung table is README §6. The one non-obvious entry: **`voice` must have no
network hop before generation.** That means a local embedding model or a
precomputed/cached query vector — otherwise the 50–300 ms embedding round-trip
alone eats the budget.

**Done when:** `llmwiki eval --profile voice` reports p95 retrieval under 100 ms
on your corpus, and `deep` beats `balanced` on `multi-hop`.

---

## Step 10 — Source chunks and two-tier freshness

**Goal:** a document is findable seconds after `add`, not after compilation.
**Touches:** `ingest/pipeline.py`, `embeddings.py`
**Size:** small

### Design

Ingest is two LLM calls — tens of seconds. That is the floor for a *compiled
page*, not for searchability. Split `add` into two phases:

1. **Immediate (~1 s):** extract → L1, insert → L2, chunk and embed → L3, entity
   scan → L4. The document is now retrievable as a raw source, already damped by
   `SOURCE_SCORE_FACTOR`.
2. **Deferred:** the LLM compilation, its pages, its links.

`index_documents` currently filters `document.kind == "wiki"`
(`embeddings.py:296`); source chunks need the same treatment.

**Done when:** `add` returns in about a second with the document answerable, and
a question answered before compilation cites the raw source, then cites the page
after.

---

## Step 11 — S3 rerank

**Goal:** precision at the top of the list.
**Touches:** new `retrieval/rerank.py`, `profiles.py`
**Size:** medium

### Design

A small local cross-encoder over the top ~50, behind `deep`. This is a local
forward pass, not an LLM call — ~20–50 ms for BGE-class models on a short list.
It is the step that took Anthropic's contextual retrieval from 49% to 67%
failure reduction.

Keep it optional and out-of-process-able: an ONNX or `sentence-transformers`
dependency should not become mandatory for a tool that currently has none.

**Done when:** MRR improves on `single-hop` and `intra-doc`, added latency is
measured (not assumed), and the tool still runs with the dependency absent.

---

## Step 12 — Topic pages

**Goal:** answer "what does this corpus say about X overall".
**Touches:** `ingest/pipeline.py`, `ingest/writer.py`, `templates.py`
**Size:** medium

### Design

`overview.md` is written at init and never updated — `update_index`
(`ingest/writer.py:169`) is the only `update_*` function in `src/`, and
`writer.py:77` explicitly rejects model rewrites of it.

Two pieces:

1. **Cluster** page embeddings incrementally to decide what the topics *are*.
   MiniBatchKMeans, or BERTopic's `partial_fit` / `merge_models` route. The
   assignment is non-LLM and incremental.
2. **Summarize only changed clusters** into topic pages, at ingest. They are
   ordinary wiki pages: they retrieve through S1 at zero extra query cost.

This is the one place the LLM ingest budget is uniquely justified — it is what
GraphRAG spends its budget producing, except written incrementally and readable.

**Done when:** the `thematic` tag improves, and adding one document re-summarizes
at most a couple of topic pages rather than all of them.

---

## Step 13 — S4 agentic rung

**Goal:** the deep end.
**Touches:** `query.py`, new `agent.py`
**Size:** medium

### Design

A-RAG's result is that a tool-using model over *simple* lanes beat every graph
system it was compared against. The three tools map onto what exists:

| A-RAG tool | llmwiki equivalent |
|---|---|
| `keyword_search` | L2 query |
| `semantic_search` | L3 query |
| `chunk_read` | read a page, or a chunk range of a source |

Add a **context tracker** so re-reading an already-read chunk costs zero tokens —
A-RAG is explicit that this is what keeps token cost at naive-RAG levels. Cap the
steps; A-RAG gained ~8% going from 5 to 20 on MuSiQue, which is the dial.

`chunk_read` is also the answer to intra-document depth: it is the only mechanism
in the stack that reads a document *in sequence* rather than as ranked fragments.

**Done when:** `research` beats `deep` on `multi-hop` and `intra-doc`, and its
token cost per question is recorded.

---

## Step 14 — `rebuild`

**Goal:** drift becomes periodic maintenance instead of slow decay.
**Touches:** `cli.py`, `ingest/pipeline.py`
**Size:** medium

Regenerate wiki pages from `raw/` + `schema.md`, ignoring current wiki state.
This is Lucene's merge, or `VACUUM` — periodic repair, not the update path
([`../incremental-updates.md`](../incremental-updates.md) §5). It also fixes the
step 7 staleness case for free.

**Done when:** the incremental-vs-from-scratch agreement curve from
[`../evaluation/harness-v1.md`](../evaluation/harness-v1.md) §9 is flat — build
the index incrementally, rebuild it from scratch, run the same questions, and
compare the `returned` sets per question. "Semantically equivalent" was not a
criterion; this is. A gap between the two curves *is* the accumulated drift, and
its size tells you how often `rebuild` needs to run.

---

## Step 15 — Later

Only after the above have numbers:

- **Score-variance escalation.** AB-RAG's cheapest signal is the variance of
  retrieval scores, which you already compute. A flat top-1/top-5 margin at S1
  is a free trigger to climb a rung automatically.
- **Prefetch for voice.** Predict the follow-up during conversational dead time
  and pre-warm the cache (VoiceAgentRAG: 110 ms → 0.35 ms on hits, 75% hit rate).
  Additive; add only if S1 measures too slow after step 4.
- **ANN index.** When the brute-force dense scan stops being milliseconds. HNSW
  inserts incrementally and tombstones deletes, so it does not violate the
  locality rule.

---

## Dependency graph

```
1 durability ─┬─> 2 one DB ─┬─> 4 L1 cache ──> 5 L2 FTS5 ─┐
              │             ├─> 6 L5 links ──> 7 L4 entities ─> 8 seeded PPR
              │             └─> 10 source chunks                      │
              └─> 3 eval  (gates everything from here on)             │
                                                                      v
                          9 profiles <──────────────────────── 11 rerank
                                 │
                                 ├─> 12 topic pages
                                 ├─> 13 agentic rung
                                 └─> 14 rebuild
```

Steps 4, 6 and 10 are independent of each other and can go in any order.
Step 8 is the only one with a large expected effect; steps 4 and 6 are pure
latency and must not change rankings at all.

---

## The three ways this plan goes wrong

**Building the index layers as separate files.** Step 2 exists to prevent this.
Four independent stores means four writes per document and a lock to coordinate
them; one database means one transaction and no lock.

**Skipping step 3.** Steps 5, 8, 11 and 12 all claim gains. Without the eval set
you will ship them believing the claims, and you will not notice when one of them
regresses a different query class — which is precisely what graph expansion is
known to do to single-hop questions.

**Tuning `blend_graph_results` instead of replacing it.** The measured failure is
the combination shape, not the quota. Score-blending a graph signal loses 6.9–9.5
points against not having a graph at all; the quota parameter cannot fix that.
