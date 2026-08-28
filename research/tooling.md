# What is a package, what is a paper

Compiled 2026-08-27. Integration cost of every system named across
[`target-architecture/`](target-architecture/README.md) and
[`evaluation/`](evaluation/README.md). PyPI and GitHub metadata checked on that
date.

**Question:** are these systems easy to integrate with — installable packages, or
implement-from-scratch?

**Short answer:** three categories, and the split is not what the star counts
suggest. Two things are worth installing, one is already in your stdlib, and most
of the rest are *papers whose idea is 20–100 lines* — which is fine, because the
recommendation was built that way on purpose.

---

## 0. The constraint you already set

`pyproject.toml` says it plainly:

```toml
# The core package is dependency-free: everything below the LLM boundary is
# stdlib. Providers and document parsers are opt-in extras so a keyword-only
# install stays small.
dependencies = []
```

That is a real architectural commitment and it should drive this decision rather
than be discovered by it. Every row below is judged against it: **does this
survive as `dependencies = []` plus an optional extra?**

The good news is that the recommendation in
[`target-architecture/`](target-architecture/README.md) was assembled to route
around dependencies — FTS5 is stdlib, the entity dictionary is your own alias
table, and PPR is arithmetic. That was not an accident, but it also was not
stated. It is stated now.

---

## 1. Three categories

| Category | What you do | Examples |
|---|---|---|
| **Installable library** | `pip install`, call it | `sqlite-vec`, `benchmark-qed`, rerankers |
| **Dataset + metric definition** | download the data, write ~50 lines | HotpotQA, 2Wiki, LongMemEval, GraphRAG-Bench |
| **Research reproduction code** | read it, reimplement the idea | LinearRAG, SPRIG, A-RAG, EraRAG, adRAP |

The third category is the largest and it is routinely mistaken for the first.
Academic repos are written to reproduce a paper's table, not to be called by
someone else's program: hard-wired dataset paths, config files as the only API,
no packaging, and a `requirements.txt` pinning a whole ML stack. Cloning one to
"just use it" usually costs more than reimplementing the mechanism.

---

## 2. Evaluation tooling

Checked on PyPI 2026-08-27.

| Package | Version | Deps | Licence | Verdict |
|---|---|---|---|---|
| `ranx` | 0.3.21 | 13 — numba, pandas, seaborn, fastparquet | — | **skip** — for five lines of arithmetic |
| `ir-measures` | 0.4.3 | 5, pulls `ranx` transitively | — | skip, same reason |
| `pytrec-eval` | 0.5 | 0 | — | last release 2020; needs a C build |
| `beir` | 2.2.0 | 13 | Apache-2.0 | skip the package, take the datasets |
| `mteb` | 2.20.2 | **159** | Apache-2.0 | never in-process |
| `datasets` (HF) | 5.0.1 | **143** | Apache-2.0 | never in-process; download JSON directly |
| `benchmark-qed` | 0.4.0 | 25, incl. the `graphrag-*` stack + Azure SDK | MIT | **use — but as a separate tool** (§2.2) |
| `ragas` | 0.4.3 | 50 | Apache-2.0 | defer; LLM judge |
| `deepeval` | 4.2.0 | 31 | Apache-2.0 | defer |
| `ragchecker` | 0.1.9 | 3 | Apache-2.0 | light, but last release 2024-09 |
| `ares-ai` | 0.6.6 | 31 | Apache-2.0 | last release 2024-07; needs ~150 annotations |

### 2.1 Tier 0 metrics: write them

`recall@k`, `hit@k`, `MRR` are three or four lines each
([`evaluation/harness-v1.md`](evaluation/harness-v1.md) §6). `ranx` would give
you those and bring numba, pandas, seaborn and fastparquet along. That trade is
not close.

The rule of thumb worth applying generally: **if the formula fits in a tweet,
importing it costs more than writing it** — in install size, in a version to
track, and in the debugging you do when its definition of `recall@k` differs
subtly from yours.

### 2.2 BenchmarkQED: use it, but not as a dependency

It is a genuine library — MIT, actively maintained (last push 2026-08-14),
`pip install benchmark-qed`, real CLI, real docs. It is the only tool named
anywhere in these documents that solves a problem you cannot cheaply solve
yourself: generating a corpus-specific query set across local/global scope.

But it wants `graphrag-common`, `graphrag-llm`, `graphrag-storage`,
`azure-ai-inference`, `scikit-learn`, `matplotlib`, `pandas`, `pyarrow` — and
**`requires-python >= 3.11`, while llmwiki declares `>= 3.10`.** That is a
concrete conflict, not a stylistic objection.

**Use it out-of-process:** its own virtualenv, run occasionally, consume the
generated questions as YAML. Nothing enters `pyproject.toml`. This is the right
shape for any generator: you need its output once a quarter, not its code on
every install.

### 2.3 Public benchmarks: take the data, not the harness

| Repo | Stars | Licence | Packaged? |
|---|---|---|---|
| GraphRAG-Bench | 485 | MIT | **no** — `Datasets/`, `Evaluation/`, `requirements.txt` |
| LongMemEval | 1,039 | MIT | **no** — `src/`, two requirements files |

Both are scripts around a dataset. For Tier 1 you need HotpotQA and 2Wiki
questions plus their gold supporting passages — a JSON download and a loop.
Downloading the data and writing your own loop is less work than making someone
else's evaluation script accept your retriever, and it keeps the metric
definitions yours.

---

## 3. Architecture components

The plan's steps, by what they actually cost to obtain.

| Step | Component | Cost | Notes |
|---|---|---|---|
| 5 | BM25 lexical | **stdlib — verified** | SQLite 3.51.1 here; FTS5 + `bm25()` both work in `sqlite3` |
| 7 | Entity extraction | **zero** | your `normalize_alias` table *is* the dictionary; spaCy would be 47 deps |
| 8 | Personalized PageRank | **~20 lines** | power iteration over dict-of-lists; `networkx` is 37 deps for one function |
| 12 | Topic clustering | ~40 lines | k-means over normalized vectors; `bertopic` is 30 deps (umap, hdbscan, numba) |
| 11 | Cross-encoder rerank | **a real dependency** | §3.1 |
| 15 | ANN index | `sqlite-vec`, **0 deps** | MIT/Apache, 2026-03-31; `usearch` (3 deps) or `hnswlib` (1 dep, 2023) as alternatives |
| — | Temporal KG | not a fit | `graphiti-core` is 47 deps *and* needs Neo4j or FalkorDB |

Two of these are worth dwelling on.

### 3.1 The reranker is the one genuine new dependency

There is no way to run a cross-encoder without a model runtime. The options, by
weight:

| Route | Weight | Note |
|---|---|---|
| `rerankers` | torch + transformers + flash-attn + rank-llm | unified API, very heavy |
| `FlagEmbedding` | 11 deps | BGE family, the most-deployed open cross-encoder |
| `sentence-transformers` | 27 deps | familiar, still pulls torch |
| ONNX Runtime + exported model | ~1 dep | smallest; you export once, ship the `.onnx` |
| HTTP to a local service | 0 | but adds a process to run |

**Recommendation: an optional `[rerank]` extra, ONNX-based.** It keeps
`dependencies = []` intact, it is the only route that does not put torch in the
install path, and step 11 already says the tool must still run with the
dependency absent. If that proves fiddly, the local-HTTP route preserves the
zero-dependency core at the cost of a process.

### 3.2 PPR is arithmetic, not a library

The step with the largest expected effect has the smallest integration cost.
Power iteration on a sparse graph is a loop over an adjacency dict, ~20 lines,
with the same optional-`numpy` pattern `embeddings.py` already uses for cosine
scoring. Pulling `networkx` (37 deps) or `scipy` (40) for `pagerank()` inverts
the cost-benefit completely.

---

## 4. The licensing trap

Worth checking before copying anything, and easy to miss.

| Project | Licence | What you may do |
|---|---|---|
| LinearRAG | **GPL-3.0** | read it; **do not vendor code** into an MIT project |
| A-RAG (`Ayanami0730/arag`) | **none** | no licence = all rights reserved; read, do not copy |
| EraRAG (`EverM0re/EraRAG-Official`) | **none** | same |
| GraphRAG-Bench | MIT | fine |
| LongMemEval | MIT | fine |
| BenchmarkQED | MIT | fine |
| PageIndex | MIT | fine |
| Graphiti | Apache-2.0 | fine |

The two most influential results in these documents — LinearRAG's token-free
graph and A-RAG's tool interface — come from the top two rows. Ideas are not
copyrightable and both are described precisely enough in their papers to
reimplement, which is what the build plan specifies anyway. But if the intent had
been "clone it and wire it in," this would have been a problem discovered late.

---

## 5. What to actually install

The whole plan, in dependency terms:

```toml
dependencies = []                      # unchanged

[project.optional-dependencies]
rerank = ["onnxruntime>=1.17"]         # step 11, and only step 11
vectors = ["sqlite-vec>=0.1.9"]        # step 15, when brute force stops scaling
eval   = []                            # stdlib; the harness is yours
```

Separate virtualenv, run occasionally, never imported:

```
benchmark-qed        # generate corpus-specific query sets  (needs Python >= 3.11)
```

Datasets downloaded as files, not packages: HotpotQA, 2WikiMultiHopQA, optionally
GraphRAG-Bench's novel/medical corpora and LongMemEval.

**Everything else in these documents is a paper.** That is a feature of the
recommendation rather than a limitation of the ecosystem: the components chosen
were chosen partly because they are simple enough to own. A token-free entity
graph, seeded PPR, and a tool-loop are each small enough to read in one sitting,
which means they are also small enough to debug at 2am — which the alternative,
a research repo pinned to someone's 2025 CUDA stack, is not.

---

## 6. Sources

PyPI JSON API and GitHub REST API, both queried 2026-08-27. Package versions,
dependency counts, licences, and last-push dates as reported on that date;
re-check before relying on any of them, particularly the stale entries
(`ragchecker` 2024-09, `ares-ai` 2024-07, `hnswlib` 2023-12).
