# Benchmarking: what exists, and what actually validates this system

Compiled 2026-08-27. Companion to
[target-architecture/](../target-architecture/README.md) and its
[build plan](../target-architecture/build-plan.md) step 3.

**Question:** is there an existing benchmark suite to validate against, and how do
SOTA methods benchmark?

**Short answer:** yes for the machinery, no for the corpus, and *barely* for the
thing that matters most here. Three different questions get benchmarked and
conflating them is the usual mistake:

| Question | Answerable by a public suite? |
|---|---|
| A. Does my retrieval machinery work? | **Yes** — and you get numbers comparable to the literature |
| B. Does it work on *my* corpus? | **No** — nothing public can answer this |
| C. Does it stay correct as the corpus grows? | **Weakly** — this is the field's blind spot, and it is your constraint |

---

## 1. How SOTA methods actually benchmark

Every system in [target-architecture/](../target-architecture/README.md) reports on
some subset of the same small set. This is worth knowing because it means you can
produce directly comparable numbers.

| System | Benchmarks used | Metrics |
|---|---|---|
| A-RAG | HotpotQA, 2WikiMultiHopQA, MuSiQue, GraphRAG-Bench | LLM-Acc, retrieved tokens |
| LinearRAG | HotpotQA, 2WikiMQA, MuSiQue, + large-scale ATLAS-Wiki | Contain-Acc / GPT-Acc, index seconds, tokens |
| SPRIG | HotpotQA, 2WikiMultiHopQA | R@5, R@10, Hit@10, MRR, QTime, peak RSS |
| GraphRAG-Bench | its own novel + medical corpora | Accuracy, ROUGE-L, Coverage, Factual Score |
| RAPTOR / adRAP | QASPER, QuALITY, NarrativeQA | accuracy; adRAP adds rebuild seconds and summary calls |
| Zep / Graphiti | DMR, LongMemEval | accuracy, latency |
| LightRAG | UltraDomain | win rates via LLM judge |

Three observations:

1. **The multi-hop trio is the de-facto standard.** HotpotQA, 2WikiMultiHopQA and
   MuSiQue appear in nearly everything. Publishing a number on them makes your
   system legible to anyone who reads this literature.
2. **The best retrieval-side papers avoid LLM judges.** SPRIG reports pure IR
   metrics — recall, MRR, latency, memory — against gold supporting passages. No
   judge, no variance, no API cost. This is the protocol to copy first.
3. **Cost is reported as a first-class metric now.** LinearRAG's headline table is
   indexing seconds *and* token consumption *and* accuracy together; SPRIG reports
   peak RSS under a 4 GB cap. A retrieval result without a cost column no longer
   reads as complete.

---

## 2. (A) Validating the machinery — use these

### GraphRAG-Bench — the best single fit

[GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) (ICLR'26)
has four task levels that map almost exactly onto the four capabilities in the
brief:

| GraphRAG-Bench level | Your requirement |
|---|---|
| Fact retrieval | single-hop |
| Complex reasoning | multi-hop across documents |
| Contextual summarization | corpus themes |
| Creative generation | — (ignore) |

Public dataset on Hugging Face, evaluation code in the repo, two domains (novel,
medical) with different information density. Crucially its whole thesis is *when
graph structure pays and when it does not* — which is precisely the question the
seeded-PPR step asks. If your PPR implementation is correct, its gains should
concentrate in the complex-reasoning level and vanish in fact retrieval. If they
do not, something is wrong with the implementation, not with the idea.

### HotpotQA / 2WikiMultiHopQA / MuSiQue — for comparability

Gold supporting passages are labelled, so **Recall@k is computable without any
LLM**. Run the SPRIG protocol: R@5, R@10, MRR, and per-query time, and compare
against its published table directly. That table is reproduced in
[target-architecture/README.md](../target-architecture/README.md) §4.

This is the cheapest sanity check in the whole plan and it costs no API calls.

### BEIR / MTEB — for the lanes in isolation

[BEIR](https://github.com/beir-cellar/beir) (18 datasets, NDCG@10) validates the
retrieval lane alone; MTEB is the broader embedding-model board. Use these only
to answer "is my BM25 configured sanely, is my embedding model a reasonable
choice" — they say nothing about fusion, graphs, or generation.

### The caveat that matters

**Public multi-hop benchmarks are Wikipedia-derived, and their "multi-hop" is
usually two-entity bridging.** A personal or organizational corpus has different
structure: fewer named entities, more implicit reference, more temporal
sequencing, more documents that disagree. Strong HotpotQA numbers prove the
machinery works. They do not prove the system is good on your material. That is
question B.

---

## 3. (B) Validating on your corpus — nothing public can

### BenchmarkQED — the automated way to build one

[microsoft/benchmark-qed](https://github.com/microsoft/benchmark-qed), from the
GraphRAG team, is the closest thing to an off-the-shelf answer. Three parts:

- **AutoQ** — generates synthetic query sets from *your* corpus across two axes:
  source (data-driven vs activity-driven) × scope (local vs global), giving four
  classes. The local/global axis is the same distinction as the plan's
  `single-hop` / `thematic` tags.
- **AutoE** — scores comprehensiveness, diversity, empowerment, relevance, by
  LLM judge and pairwise comparison.
- **AutoD** — samples a corpus by topic-cluster breadth so the query set is
  representative rather than accidental.

Ships with the AP News health corpus (1,397 openly-licensed articles) and podcast
transcripts, so you can validate the harness before pointing it at your data.

**What it does not give you:** `expect_pages`. AutoQ produces questions, not gold
retrieval targets, so it cannot compute recall — only judged answer quality. That
is why the hand-written set in build-plan step 3 is still required, and why the
sequencing is: hand-write ~30 with gold pages first, then use AutoQ to scale to a
few hundred once the format has proved itself.

### Why the hand-written set comes first

Half a day of writing 30 questions with known correct pages buys you the only
metric that is both cheap and unambiguous — did the right page come back — and it
is computable on every commit with no API cost. Every LLM-judge metric is more
expensive, noisier, and harder to debug. Start with the cheap unambiguous one.

---

## 4. (C) Validating incremental updates — the field's blind spot

This is your third constraint and the literature is thin. Most RAG evaluation
assumes one-time indexing, which is why "incremental update cost" almost never
appears in a results table.

### The protocol worth copying: EraRAG

[EraRAG](https://arxiv.org/abs/2506.20963)
([code](https://github.com/EverM0re/EraRAG-Official)) evaluates growth directly:
**insert 5% of the corpus, ten times, measuring at each step.** Reported against
RAPTOR: up to 57.6% less token usage (PopQA) and 77.5% less graph rebuilding time
(QuALITY), with larger reductions against GraphRAG and HippoRAG.

Adopt the protocol regardless of the system. Three curves, each one a claim in the
architecture doc made falsifiable:

| Curve | What it falsifies |
|---|---|
| accuracy vs. number of insertions | "quality is stable under growth" |
| update cost per insertion | "updates are O(document), not O(corpus)" |
| answer agreement between incremental and from-scratch index | drift — the thing `rebuild` exists to fix |

The third is the one nothing published measures well, and it is the direct test
of the drift concern in
[combining-rag-strategies.md](../combining-rag-strategies.md) §3.4: build the index
incrementally, build it again from scratch, ask the same questions, and count
where the answers differ.

### Public options, in order of usefulness here

- **[LongMemEval](https://arxiv.org/pdf/2410.10813)** — 500 questions over
  long multi-session histories, with categories for *knowledge updates*,
  *temporal reasoning*, and *abstention*. The knowledge-updates category is the
  closest public proxy for drift: does the system return the current fact after
  it has been superseded? A [v2](https://arxiv.org/html/2605.12493v1) exists.
- **[LoCoMo](https://www.emergentmind.com/topics/locomo-and-longmemeval-_s-benchmarks)**
  — complementary; multi-session dialogues, ~200 questions each, single-hop /
  multi-hop / open-domain / temporal.
- **StreamingQA** — the only benchmark carrying *both* question-asked time and
  document publication date, which is what temporal correctness actually needs.
- **[FreshStack](https://www.emergentmind.com/topics/freshstack)** and
  **[DRAGOn](https://arxiv.org/html/2507.05713)** — frameworks for *building*
  freshness benchmarks from evolving corpora rather than fixed datasets.

LongMemEval and LoCoMo are conversational-memory benchmarks, not document-corpus
ones, so they are a proxy rather than a fit. But they test the right abilities,
and Zep benchmarks on LongMemEval — so a number there is comparable to a
production system with the same constraints.

---

## 5. Evaluation frameworks (harnesses, not benchmarks)

Worth distinguishing: a *benchmark* is data plus gold labels; a *framework* is a
way to score arbitrary outputs, usually with an LLM judge.

| Framework | Shape | Cost to adopt |
|---|---|---|
| [RAGAS](https://github.com/explodinggradients/ragas) | faithfulness, answer relevance, context precision/recall | low; LLM judge per metric |
| [ARES](https://github.com/stanford-futuredata/ARES) | fine-tuned compact judges; synthetic query generation | ~150 human-annotated samples to calibrate |
| RAGChecker | decomposes answers into atomic claims, checks entailment | highest; best diagnostics |
| [TREC RAG track](https://arxiv.org/pdf/2603.09891) | MS MARCO V2.1, nugget-recall protocol | large corpus; academic scale |

**Recommendation: skip all of them initially.** They add an LLM call per question
per metric, plus judge variance, to a system whose defining property is a fast
local path. `recall@k` against gold pages plus string containment answers most
questions at zero cost. Add RAGChecker later, when you have a quality regression
you cannot explain with retrieval metrics — its claim-level decomposition is the
right tool for *that* problem specifically.

---

## 6. What to actually do

In build order, mapped onto
[build-plan.md](../target-architecture/build-plan.md):

| When | Do | Answers |
|---|---|---|
| Step 3 (now) | ~30 hand-written questions with gold pages, four tags | B — the only thing that measures your corpus |
| Before step 8 | HotpotQA + 2Wiki, SPRIG protocol, no judge | A — is the PPR implementation right |
| Before step 8 | GraphRAG-Bench, complex-reasoning level | A — does the gain land where theory says |
| After step 9 | EraRAG's 5%-×10 insertion protocol on your corpus | C — the incremental claim |
| Once step 3 proves the format | BenchmarkQED AutoQ to scale to a few hundred queries | B at volume |
| Only if quality regresses inexplicably | RAGChecker | diagnosis |

Two numbers to report alongside every accuracy figure, because the current
literature does and because they are your constraints: **retrieval latency p50/p95**
and **index cost per document (seconds and tokens)**.

---

## 7. One system worth reading before step 12

**EraRAG** is not just a protocol — it is a hierarchical multi-layer Graph-RAG
that buckets chunks by hyperplane LSH and re-partitions *locally* on insert, which
is a direct alternative to the topic-pages approach in
[target-architecture/README.md](../target-architecture/README.md) §5. It postdates
the earlier documents in this directory, and it is the strongest published answer
to "hierarchy that survives a growing corpus."

It does not change the recommendation — topic pages are cheaper, readable, and
reuse machinery you already have — but if step 12 disappoints, EraRAG is the next
thing to try rather than RAPTOR.

---

## 8. Sources

- [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) · [paper](https://arxiv.org/pdf/2506.02404) · [dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)
- [BenchmarkQED — Microsoft Research](https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/) · [code](https://github.com/microsoft/benchmark-qed)
- [EraRAG: Efficient and Incremental RAG for Growing Corpora](https://arxiv.org/abs/2506.20963) · [code](https://github.com/EverM0re/EraRAG-Official)
- [LongMemEval](https://arxiv.org/pdf/2410.10813) · [LongMemEval-V2](https://arxiv.org/html/2605.12493v1)
- [DRAGOn: Designing RAG On Periodically Updated Corpus](https://arxiv.org/html/2507.05713) · [FreshStack](https://www.emergentmind.com/topics/freshstack)
- [Overview of the TREC 2025 RAG Track](https://arxiv.org/pdf/2603.09891)
- [MTRAG / MT-RAG multi-turn benchmark](https://github.com/IBM/mt-rag-benchmark)
- SPRIG, A-RAG, LinearRAG protocols as cited in [target-architecture/README.md](../target-architecture/README.md) §11

Not read: RAGBench, CRAG (KDD Cup 2024), MIRAGE-Bench, mmRAG — all surfaced in
the search and all plausible additions, none read closely enough to recommend.
