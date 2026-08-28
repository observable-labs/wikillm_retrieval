# Evaluation harness v1 — design

Compiled 2026-08-27 against `216d96f`. API references checked against source.
Supersedes and absorbs the earlier `starter-kit.md`.

**Scope:** a harness that evaluates *any* RAG system with an
ingest → index → retrieve shape, of which llmwiki is one adapter. Retrieval-only
by default, zero new dependencies, ~450 lines.

---

## 1. Gaps found in the previous draft

Recorded because they change the design, not just the wording.

| # | Gap | Fix |
|---|---|---|
| 1 | Harness bound directly to `llmwiki.retrieval.search` — **could only measure llmwiki against llmwiki** | `System` protocol + adapters (§3) |
| 2 | **No baselines.** A change moving recall says nothing about whether the system is good | `bm25` and `dense` adapters, ~30 lines each (§3.4) |
| 3 | Nothing measured **ingest or index cost**, despite the plan requiring it in every report | `IngestReport` / `IndexReport` (§3.2) |
| 4 | Questions filed in `.llm-wiki/` — a system-specific directory for corpus-specific data | suite directory, system-agnostic (§4) |
| 5 | `expect_pages` had one semantic: **all** gold pages required | `expect_mode: all \| any` (§5) |
| 6 | Measured `recall@k`, but the packer drops pages that do not fit — **retrieval recall ≠ evidence recall** | `recall@context` (§6) |
| 7 | Growth protocol (Tier 2) had **no hooks** in the design | optional `reset()` + ordered ingest (§9) |
| 8 | No way to evaluate a system **not written in Python** | subprocess adapter, JSON over stdio (§3.5) |
| 9 | Drift-probe questions scored on pages, but their whole point is that **the right page changes** | drift probes score on strings (§5.3) |
| 10 | YAML question file, `Profile` class that does not exist yet | JSONL; `Lanes` (§4, §3.4) |

Items 1–3 are the substantive ones. A harness that cannot hold two systems side
by side cannot answer the question the rest of this research exists to ask.

---

## 2. What v1 is, and is not

**In:** the `System` protocol and four adapters; a suite format; retrieval
metrics including `recall@context`; latency and index cost; run artifacts; a
paired diff; optional generation.

**Out — decisions, not omissions:**

| Excluded | Why | Where it goes |
|---|---|---|
| LLM judge | cost, variance, a second thing to debug | maybe never; RAGChecker if ever |
| BenchmarkQED integration | needs Python ≥3.11, own venv | out-of-process, quarterly |
| Plots, HTML reports | a terminal table suffices at n=30 | probably never |
| `--fail-under` for CI | thresholds on noisy numbers create false alarms | v2, once stable |
| Query-vector caching | makes eval behave differently from production | v2 (§12) |

---

## 3. The general interface

The core of the design. Everything else hangs off it.

### 3.1 The `System` protocol

```python
from typing import Protocol, Sequence, runtime_checkable

@runtime_checkable
class System(Protocol):
    """A RAG system under evaluation. Only `name` and `retrieve` are required."""

    name: str

    def retrieve(self, query: str, k: int) -> Sequence["Retrieved"]:
        """Rank corpus documents for a query. The hot path; timed by the harness."""

    # ── optional; presence declares the capability ──────────────────────
    def ingest(self, paths: Sequence[Path]) -> "IngestReport": ...
    def build_index(self) -> "IndexReport": ...
    def answer(self, query: str, k: int) -> "Generated": ...
    def snapshot(self) -> dict: ...          # corpus stats for the run artifact
    def reset(self) -> None: ...             # drop all derived state
```

**Capability detection is by `hasattr`, not by subclassing.** A read-only adapter
over an already-built index implements one method and is complete; the growth
protocol (§9) runs only against systems that also expose `ingest` and `reset`;
`answer_hit` requires `answer`. The harness reports which stages it could
exercise rather than failing on a system that does less.

This is what makes the harness general across the ingest/index/retrieve shape
without forcing every system to pretend it has all three stages.

### 3.2 The data the protocol moves

```python
@dataclass(frozen=True)
class Retrieved:
    doc_id: str                    # the join key with gold — see §3.3
    score: float
    rank: int                      # 1-based, as the system ranked it
    chunk_id: str | None = None    # for chunk-level systems
    text: str | None = None        # needed only for recall@context
    kind: str = ""                 # free-form: "wiki" | "source" | "chunk"
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IngestReport:
    documents: int
    seconds: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    bytes_written: int = 0


@dataclass(frozen=True)
class IndexReport:
    seconds: float
    entries: int                   # chunks, postings, nodes — system's own unit
    bytes_on_disk: int = 0


@dataclass(frozen=True)
class Generated:
    text: str
    cited_ids: tuple[str, ...] = ()
    packed_ids: tuple[str, ...] = ()   # what actually reached the model — §6
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

`IngestReport` and `IndexReport` close gap 3: cost stops being something the plan
asks for and nothing collects. `packed_ids` closes gap 6.

### 3.3 The `doc_id` contract — the part that decides generality

The hard question in any cross-system harness: **what is the unit of ground
truth?** llmwiki ranks wiki pages; a chunk-based RAG ranks chunks; HotpotQA gold
is paragraph ids. These are not comparable by default.

Resolve it explicitly at the suite level. A suite declares its id space:

| Suite kind | `doc_id` means | Comparable across systems? |
|---|---|---|
| `native` | whatever the system emits (llmwiki page paths) | **no** — only to itself over time |
| `canonical` | source documents in the corpus | **yes** |

**Adapters are responsible for mapping into the suite's id space.** For llmwiki
in a `canonical` suite, a wiki page maps to the source(s) it was compiled from —
which the frontmatter `sources:` field already records. A chunk-level system maps
chunk → parent document.

```python
def canonical(self, retrieved: Retrieved) -> tuple[str, ...]:
    """Map a native result to canonical corpus ids. Default: (doc_id,)."""
```

Fan-out to a tuple is deliberate: one llmwiki page can be compiled from three
sources, and a hit on that page is evidence for all three.

Getting this wrong is the classic way cross-system RAG comparisons produce
nonsense, so v1 makes it a declared property of the suite rather than an
assumption in the runner.

### 3.4 The adapters that ship with v1

```
adapters/llmwiki.py     wraps search(); Lanes selects the lane configuration
adapters/bm25.py        SQLite FTS5 over the same corpus                ~30 lines
adapters/dense.py       the existing VectorStore, no fusion, no graph   ~30 lines
adapters/subprocess.py  any external system, JSON over stdio            ~40 lines
```

The llmwiki adapter needs no `Profile` class — `search()` gates the vector lane
on `embedding_config is not None and embedding_config.enabled and
embedding_config.model` (`retrieval/pipeline.py:73`), so three fields describe a
configuration:

```python
@dataclass(frozen=True)
class Lanes:
    name: str                # lexical | hybrid | full
    sources: bool
    embeddings: bool
    top_k: int = 20
```

Replaced by `Profile` at build-plan step 9; same shape, one-line change.

**The two baseline adapters close gap 2** and cost almost nothing, because FTS5
is stdlib and `VectorStore` already exists. They are what turn "recall went up
0.03" into "the hybrid beats both of its parts" — the claim hybrid retrieval
exists to make, and the one the current plan could not check. They also make the
SPRIG-style table reproducible on your own corpus.

### 3.5 External systems

```
→ {"op": "retrieve", "query": "...", "k": 20}
← {"results": [{"doc_id": "...", "score": 1.24}, ...]}

→ {"op": "ingest", "paths": ["..."]}
← {"documents": 12, "seconds": 41.2, "llm_calls": 24}
```

Newline-delimited JSON on stdin/stdout. ~40 lines, and it means the harness can
evaluate a system in any language — including the research repos in
[`../tooling.md`](../tooling.md) §4 that cannot be imported and, in two cases,
cannot legally be vendored.

---

## 4. Suite layout

System-agnostic, and not inside `.llm-wiki/` (gap 4):

```
<suite>/
  suite.json            {"name": ..., "id_space": "native"|"canonical", "corpus": "..."}
  questions.jsonl       one question per line
  known-gaps.jsonl      questions the corpus cannot yet answer — see §5.3
  runs/
    2026-08-27T14-02-llmwiki-full.json
```

`--suite PATH`, defaulting to `<project>/eval/` when run inside a llmwiki
project. The corpus is referenced, never copied.

JSONL rather than YAML: `json` is stdlib and the core package is
`dependencies = []` ([`../tooling.md`](../tooling.md) §0). It also diffs cleanly
in git and appends without reformatting.

---

## 5. Questions

### 5.1 Format

```json
{"id": "mh-003",
 "question": "Which vendor we evaluated also supplied the 2024 pilot?",
 "tags": ["multi-hop"],
 "expect_ids": ["wiki/orgs/acme.md", "wiki/projects/2024-pilot.md"],
 "expect_mode": "all",
 "expect_strings": ["Acme"],
 "forbid_strings": [],
 "note": "answer exists on neither page alone"}
```

`expect_pages` is renamed `expect_ids` — the harness is no longer page-specific.

**`expect_mode` closes gap 5.** `"all"` (default) requires every gold id, which is
right for multi-hop. `"any"` requires one, which is right when two pages
independently answer a single-hop question — and scoring those as 0.5 recall
understates the system for no reason.

### 5.2 Validation on load

| Condition | Response |
|---|---|
| unknown tag | **error** — a typo silently creates an empty tag group |
| `expect_ids` empty and no `expect_strings` | **error** — no ground truth |
| duplicate `id` | **error** — ids are the join key for paired diffs |
| gold id absent from the corpus | **warn**, record in `stale_targets` |

The last one earns its place: a gold id that no longer resolves means the ground
truth is stale — renamed, merged, or deleted. Do not fail the run; a growing
`stale_targets` list is the signal that the suite needs maintenance. **An eval
set silently scoring against dead ids reports a regression that is really a
rename.**

### 5.3 Two probe types that need different scoring

**Drift probes** (gap 9). An early document says one thing, a later one
supersedes it. The *point* is that the correct page changes over time, so gold
ids are the wrong ground truth. Score these on `expect_strings` (the current
fact) and `forbid_strings` (the superseded one), with `expect_ids` omitted. Tag
them `drift`.

**Refusal probes.** Questions the corpus genuinely cannot answer, kept in
`known-gaps.jsonl`. The correct behaviour is the refusal text; the failure is a
confident answer. Scored only with `--generate`, and no recall metric applies.
A system that hallucinates here is worse than one that retrieves nothing, and no
ranking metric will ever tell you that.

Both are cheap to write and neither is covered by any public benchmark on your
corpus.

---

## 6. Metrics

```python
def recall_at_k(returned, expected, k, mode="all") -> float
def hit_at_k(returned, expected, k) -> int
def mrr(returned, expected) -> float
def recall_at_context(packed, expected, mode="all") -> float     # gap 6
def percentile(values, p) -> float
def paired_bootstrap(before, after, iterations=10_000, seed=0)
```

All pure, all stdlib, all in a module that imports nothing else in the package —
these are the functions everything rests on and they should be testable with
hand-computed values and no fixtures.

**`recall@context` is the metric the previous draft was missing.** `_pack_context`
(`query.py:196`) fills a character budget in rank order and *skips* pages that no
longer fit. So a page ranked 4th can be absent from the evidence the model saw.
`recall@20` measures the retriever; `recall@context` measures what the answer
could possibly have been based on. When they diverge, the problem is packing, not
retrieval — and every RAG system has a context budget, so this generalizes.

Reported per tag and in total; latency per adapter configuration.

---

## 7. The runner

```python
def run(system, suite, k, generate=False, repeat=1) -> Run:
    for question in suite.questions:
        best = min(
            (timed_retrieve(system, question.question, k) for _ in range(repeat)),
            key=lambda pair: pair[1],
        )
        retrieved, elapsed_ms = best
        ...
```

Four details that decide whether the numbers mean anything:

**Time `retrieve` only** — never generation. Including the answer call means
measuring the provider, whose variance drowns any retrieval effect.

**Split out load time.** Until build-plan step 4 lands, llmwiki's
`load_documents` re-parses every PDF on every query, and that term dominates and
masks everything else. The llmwiki adapter reports it in `Retrieved.meta` /
`Row.load_ms`. After step 4 it should collapse toward zero — which is that step's
acceptance test.

**`repeat` defaults to 1; use 3 for latency runs and take the minimum.** The
minimum is the standard estimator under jitter; the mean of three network calls
measures the network.

**Keep `returned`** — the top-k ids per question. It is what makes a delta
explainable rather than merely visible, and it costs bytes.

Also record whatever `snapshot()` returns and any adapter notes. llmwiki's
`SearchResponse.notes` carries "vector search unavailable, using keyword search
only", which would otherwise present as a mysterious quality drop.

---

## 8. Run artifact and diff

```json
{"timestamp": "2026-08-27T14:02:11Z",
 "commit": "216d96f",
 "system": "llmwiki", "config": "full", "k": 20,
 "suite": {"name": "personal", "id_space": "native", "questions": 30},
 "snapshot": {"pages": 412, "sources": 87, "chunks": 5310, "embedding_model": "..."},
 "ingest": {"documents": 87, "seconds": 2140.0, "llm_calls": 174},
 "stale_targets": ["wiki/orgs/acme-corp.md"],
 "rows": ["..."]}
```

`snapshot` is not decoration. A recall change with a changed chunk count is not a
retrieval result, and in six months that field is the only thing that will tell
you which it was.

```
  eval · llmwiki/full · k=20 · 30 questions · 216d96f

  tag            recall@20   Δ       flipped     recall@ctx
  single-hop        0.79   +0.02     +1 / -0        0.74
  multi-hop         0.68   +0.11     +4 / -1        0.51   ← packing, not retrieval
  thematic          0.61    0.00      0 / -0        0.58
  intra-doc         0.52   -0.03     +0 / -1        0.49
  ───────────────────────────────────────────────────────
  total             0.67   +0.03     +5 / -2        0.60   CI [+0.01, +0.06]

  retrieval    p50 88ms  p95 140ms   (load 61 / 94)
  baselines    bm25 0.51 · dense 0.58 · llmwiki/full 0.67
  vs 2026-08-26T09-14-llmwiki-full

  ! 1 stale target: wiki/orgs/acme-corp.md (mh-003)
```

`flipped` is the column to read — `+4 / -1` at n=8 is a result, `+0.11` alone is
not. The **baselines line** is the one that says whether the system is any good
rather than merely improving.

---

## 9. Growth protocol hooks (gap 7)

Tier 2 needs three things from the protocol, all optional and all now present:
`reset()`, `ingest(paths)` accepting an ordered subset, and `snapshot()`.

```python
def growth(system, suite, corpus, slices=10):
    """Insert 1/slices of the corpus at a time; evaluate after each."""
    system.reset()
    for step, batch in enumerate(chunked(corpus.documents, slices), start=1):
        report = system.ingest(batch)
        yield step, report, run(system, suite, k=20)
```

Three curves, each falsifying a claim the architecture makes:

| Curve | Falsifies |
|---|---|
| recall vs. number of insertions | "quality is stable under growth" |
| `IngestReport.seconds` per batch | "updates are O(document), not O(corpus)" |
| incremental vs. from-scratch answer agreement | drift — the reason `rebuild` exists |

The third is a second full run against a freshly built index, comparing
`returned` sets per question. Nothing published measures it well, and it is the
direct test of the drift concern in
[`../combining-rag-strategies.md`](../combining-rag-strategies.md) §3.4 — and the
acceptance test for build-plan step 14, which currently says "semantically
equivalent" with no metric behind it.

---

## 10. Module layout

```
src/llmwiki/eval/__init__.py       public API                          ~20
src/llmwiki/eval/types.py          System, Retrieved, reports          ~60
src/llmwiki/eval/suite.py          suite + JSONL parse + validation    ~80
src/llmwiki/eval/metrics.py        pure functions, no I/O              ~70
src/llmwiki/eval/runner.py         run(), growth()                     ~90
src/llmwiki/eval/report.py         aggregate, render, diff            ~110
src/llmwiki/eval/adapters/*.py     llmwiki, bm25, dense, subprocess   ~130
tests/test_eval.py
```

~560 lines, zero new dependencies. The generality costs roughly 120 lines over a
version welded to `search()` — cheap for the ability to hold two systems side by
side, which is the entire point.

---

## 11. Testing the harness

An eval harness with a bug is worse than none: it produces confident wrong
numbers. Four tests, and the second is the one that matters.

1. **Metric unit tests.** Hand-computed: `expected={a,b}`, `returned=[c,a,d,b]`,
   `k=3` → `recall(all)=0.5`, `recall(any)=1.0`, `hit=1`, `MRR=0.5`. Plus the
   degenerate cases: no hits, all hits, `k` beyond the list length.
2. **Null-change test.** Run twice against an unchanged project: every delta
   exactly zero, `flipped` `0 / 0`. Ties are already broken by path in
   `results.sort(key=lambda r: (-r.score, r.path))`, so it should hold — the test
   confirms it holds through the whole harness. **Run this before trusting the
   first baseline.**
3. **Inversion test.** Reverse the ranking before scoring; recall and MRR must
   collapse. Catches metrics that accidentally ignore rank.
4. **Adapter conformance.** One shared test parameterized over every adapter:
   returns `Retrieved` with 1-based contiguous ranks, respects `k`, and
   `canonical()` produces ids in the suite's id space. This is what stops a new
   adapter from silently scoring differently for structural reasons.

---

## 12. Build order

| | Step | Output |
|---|---|---|
| 1 | `types.py` + `metrics.py` + unit tests | the foundation, verified first |
| 2 | `suite.py` + validation | loads three questions; rejects a malformed one |
| 3 | `adapters/llmwiki.py` | one system, retrieving |
| 4 | **Write the 30 questions** | the only part that cannot be mechanized |
| 5 | `runner.py` → raw rows | a run |
| 6 | `report.py` → table, artifacts, `--compare` | the diff |
| 7 | Null-change test on the real corpus | trust |
| 8 | `adapters/bm25.py`, `adapters/dense.py` | baselines; the "is it good" answer |
| 9 | Baseline recorded for all three lane configs | what everything is measured against |

Steps 1–3 and 5–6 are mechanical. Step 4 is the real work. Steps 8–9 are what
make the first number meaningful rather than merely recorded.

**v2:** `Profile` replaces `Lanes`; query-vector cache keyed by
`sha256(model|dims|query)`; a `canonical` suite loading HotpotQA/2Wiki gold
passages, which with the adapter boundary is now a *suite*, not a separate
program; `--fail-under` for CI.

---

## Appendix A — seed questions

Templates by tag. Adapt the wording; the shape is the point.

**`single-hop`** — one document has it.
1. "What is the definition of ⟨term with its own page⟩?"
2. "What date did ⟨event⟩ happen?"
3. "Which ⟨category⟩ does ⟨entity⟩ belong to?"

**`multi-hop`** — no single document states the answer.
4. "Which ⟨entity type⟩ appears in both ⟨document 1⟩ and ⟨document 2⟩?"
5. "⟨Entity X⟩ connects to ⟨entity Z⟩ through what intermediate?"
6. "Did ⟨person/org⟩ work on ⟨thing⟩ before or after ⟨other thing⟩?"

**`thematic`** — the corpus as a whole is the answer.
7. "What are the recurring themes across the ⟨category⟩ documents?"
8. "What does this corpus say about ⟨broad topic⟩ overall?"
9. "Which topics are well covered here, and which are thin?"

**`intra-doc`** — buried past where a ranked snippet reaches.
10. "In ⟨long document⟩, what does section ⟨N⟩ conclude about ⟨sub-topic⟩?"
11. "What caveat does ⟨long document⟩ attach to its ⟨headline claim⟩?"
12. "What are the exact figures in ⟨table/appendix⟩ of ⟨document⟩?"

Plus, per §5.3: at least two `drift` probes scored on strings, and at least two
refusal probes in `known-gaps.jsonl`.

Writing `expect_ids` is the part that takes the time and the part that makes the
suite worth having. If you cannot name the documents that answer a question, it
belongs in `known-gaps.jsonl` — which is its own useful artifact.

---

## Appendix B — paired bootstrap

At n=30 the *independent* interval is ±7 points, but the comparison is **paired**:
the same questions before and after, so per-question difficulty cancels. Resample
questions, not outcomes.

```python
import random

def paired_bootstrap(before, after, iterations=10_000, seed=0):
    """95% CI on the mean per-question delta. `before`/`after`: {id: score}."""
    rng = random.Random(seed)
    ids = sorted(set(before) & set(after))
    deltas = [after[i] - before[i] for i in ids]
    means = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (sum(deltas) / len(deltas),
            means[int(0.025 * iterations)],
            means[int(0.975 * iterations)])
```

An interval spanning zero means no effect shown — which for the steps whose
expected effect *is* zero (the L1 cache, the L5 relocation) is the pass
condition, not a failure. For binary `hit@k`, McNemar on the discordant pairs is
sharper: only the flipped questions carry information.

---

## Appendix C — pin these, or the numbers lie

| Pin | Why |
|---|---|
| corpus snapshot (git SHA or a copy) | a document added between runs invalidates the comparison |
| embedding model *and* dimensions | re-embedding shifts every dense rank |
| `k` | `recall@k` is not comparable across different k |
| chunking parameters | `ChunkingOptions` moves every boundary |
| temperature 0, fixed seed | only with `--generate`, but then it matters a lot |

All of them belong in the run artifact's `snapshot` block.
