# Where to start evaluating

Compiled 2026-08-27 against llmwiki `216d96f`.

**Question:** given everything in [`../`](../README.md), what is the best starting
place for evaluating this system — SOTA, affordable, and robust for these use
cases?

**Answer in one line:** a 30-question golden set on your own corpus, scored with
retrieval-only metrics and no LLM judge, reported per capability *and per latency
profile*. Zero marginal cost, deterministic, runs in seconds, and it is the same
protocol the strongest 2026 retrieval papers use.

Contents:
- **This document** — the decision and the first week.
- [harness-v1.md](harness-v1.md) — the buildable design: the general `System`
  interface, adapters, suite format, metrics, and the gap audit.
- [benchmarking.md](benchmarking.md) — the survey of what already exists and how
  SOTA methods benchmark.

---

## 1. The one thing most RAG evals get wrong for this system

Standard RAG benchmarks report **one number per system**. Your system does not
have one operating point — it has four
([`../target-architecture/README.md`](../target-architecture/README.md) §6:
`voice`, `balanced`, `deep`, `research`). A scalar score cannot express "fast
enough for voice" and "good enough for multi-hop" at the same time, and an eval
that reports one will happily accept a change that doubles latency to gain two
points of recall.

> **The primary artifact is not a score. It is a quality-versus-latency curve
> across profiles.**

Concretely, every run produces a matrix, not a number:

```
                 single-hop   multi-hop   thematic   intra-doc   p50    p95
  voice             0.71        0.42       0.55       0.38      22ms   41ms
  balanced          0.79        0.68       0.61       0.52      88ms  140ms
  deep              0.82        0.71       0.63       0.66     190ms  260ms
  research          0.84        0.83       0.74       0.79      3.1s   6.8s
```

A change is an improvement only if it is **Pareto-improving** — or if it is an
explicit trade you name. That framing is what makes the eval fit *your*
constraints rather than the literature's.

Two columns must appear beside quality in every report, because they are
constraints and not free variables: **retrieval latency p50/p95** and **index
cost per document** (seconds and tokens). LinearRAG's headline table does exactly
this, and it is why that table is legible.

---

## 2. Start here — Tier 0

**~30 questions you write yourself against your real corpus, tagged by
capability, scored on whether the right page came back.**

| Property | Value |
|---|---|
| Cost per run | **$0** — no LLM call in the retrieval-only path |
| Runtime | seconds |
| Determinism | total — same input, same number |
| Effort to build | half a day |
| Catches | every regression that matters, on the material that matters |

Four tags, one per capability in the brief, so a gain in one cannot mask a
regression in another:

| Tag | Tests | Count |
|---|---|---|
| `single-hop` | one page has the answer | 10 |
| `multi-hop` | requires joining two or more documents | 8 |
| `thematic` | "what does the corpus say about X overall" | 6 |
| `intra-doc` | detail buried deep in one long source | 6 |

Metrics, all computable without a model: `recall@k`, `MRR`, `answer_hit` (string
containment), `citation_rate` (already on `Answer`), and latency percentiles.
Schema and formulas in [harness-v1.md](harness-v1.md) §5–6.

### Why this beats every alternative as a starting point

**It is the only thing that measures your corpus.** Public multi-hop benchmarks
are Wikipedia-derived and their "multi-hop" is two-entity bridging. A personal or
organizational corpus has fewer named entities, more implicit reference, more
temporal sequencing, and more documents that disagree. Public numbers validate
machinery; they do not validate fitness.

**It is SOTA practice, not a compromise.** SPRIG — the paper supplying the
strongest single finding in the architecture doc — reports exactly this: recall,
MRR, query time, peak memory, against gold supporting passages, no judge. The
best retrieval-side work in 2026 avoids LLM judges because they add cost,
variance, and a second thing to debug.

**It is robust in the specific sense that matters:** ground truth that does not
depend on a model, and cheap enough to run on every commit. An eval you run
monthly does not catch regressions; it documents them afterwards.

---

## 3. The sensitivity question, answered honestly

Thirty questions sounds too small, and for *independent* comparison it is: the
standard error on a proportion near 0.8 at n=30 is about 7 points, so only large
effects clear the noise. Per tag (n=6–10) the interval is wider still.

**But you are not doing independent comparison.** You run the same fixed
questions before and after a change, so the comparison is **paired**, and paired
tests are far more sensitive than the n suggests — they cancel per-question
difficulty entirely. What matters is not "did the mean move" but "how many
questions changed outcome, and in which direction."

Practical consequences:

- Report **per-question deltas**, not just aggregate means. Five questions
  flipping to correct and one flipping to wrong is a clear result at n=30.
- Use a **paired bootstrap** over question-level outcomes for the interval
  (snippet in [harness-v1.md](harness-v1.md) appendix B); McNemar's test if you want a
  p-value on binary hit/miss.
- Treat per-tag numbers as **directional**, and use the total when you need
  confidence. A tag with n=6 tells you where to look, not whether you won.
- **Pin everything else.** Same corpus snapshot, same embedding model, same seed,
  temperature 0 for any generation. Unpinned variance will swamp a real effect at
  this scale.

Grow the set to ~150 once the format proves out — `BenchmarkQED`'s `AutoQ`
generates corpus-specific queries automatically, though not the gold pages, so
the hand-written core stays.

---

## 4. The ladder above Tier 0

Add tiers only when the tier below stops answering your question. Each is a
different question, not a better version of the same one.

| Tier | What | Cost | Run when | Answers |
|---|---|---|---|---|
| **0** | golden set, retrieval-only | **$0** | every commit | did I break my corpus |
| **0b** | golden set + generated answers | ~30 answer calls | every PR | did the answer degrade |
| **1** | HotpotQA + 2Wiki, recall@k | **$0 in LLM cost** | before shipping the graph work | is the implementation correct |
| **2** | growth protocol (EraRAG's) | 10 × ingest of a subset | per release | do updates stay cheap and stable |
| **3** | GraphRAG-Bench / judged quality | generation + judge per question | rarely | comparability to the literature |

**Tier 1 deserves emphasis.** HotpotQA and 2WikiMultiHopQA label gold supporting
passages, so recall@k costs no API calls at all — only the one-time embedding of
a subset. It is the cheapest correctness check available for the highest-risk
step in the plan: you should reproduce roughly RRF 0.851 → seeded-PPR 0.867 on
HotpotQA and 0.697 → 0.794 on 2Wiki. **If you do not reproduce the direction of
that gap, the bug is in your PPR, not in your corpus** — and you would never learn
that from a corpus-specific eval alone.

**Tier 2 is the one nobody else runs**, and it is your third constraint. Insert
5% of the corpus, ten times, measuring at each step — accuracy, cost per
insertion, and *agreement between the incrementally-built index and a
from-scratch rebuild*. That last curve is the direct test of drift, and it is
what tells you whether `rebuild` is a nicety or overdue.

---

## 5. What not to do first

| Don't | Why |
|---|---|
| Start with RAGAS / ARES / RAGChecker | an LLM judge per question per metric, added to a system whose defining property is a fast local path. RAGChecker later, when a regression resists retrieval-metric explanation |
| Start with a public benchmark | measures machinery, not fitness; and you cannot debug a corpus you did not write questions for |
| Score a single aggregate number | hides exactly the per-capability trade this architecture exists to manage (§1) |
| Generate the whole question set with an LLM | AutoQ gives questions, not gold pages — so no recall, and the questions inherit the generator's idea of what is answerable |
| Defer the eval until after the optimizations | steps 5, 8, 11 and 12 of the build plan each claim a measurable gain; without a baseline they are faith |

---

## 6. The first week

| Day | Do | Output |
|---|---|---|
| 1 | Step 0 baseline: where does the current query second go, how big is the corpus | a table |
| 1–2 | Write 30 questions with gold pages, four tags | `eval/questions.yaml` |
| 2 | Harness: run `search()` per question, compute recall@k / MRR / latency | `llmwiki eval` |
| 3 | Record the baseline matrix across all four profiles | `eval/runs/<date>.json` |
| 3 | Wire paired-delta reporting against the previous run | regression visible on introduction |
| 4–5 | Ship the L1 text cache; confirm latency drops and **rankings do not move** | the first Pareto improvement |

By the end of the week you have a baseline, a regression detector, and one
verified win — and every later step in
[`../target-architecture/build-plan.md`](../target-architecture/build-plan.md)
has something to be measured against.

The L1 cache is deliberately the first change measured: it should improve latency
and change *no* rankings. If rankings move, the harness is wrong — which is
exactly what you want to discover on a change with a known-zero quality effect
rather than on the PPR swap.

---

## 7. Sources

Evidence and the full survey are in [benchmarking.md](benchmarking.md). The
protocols named above:

- SPRIG's retrieval protocol — [Democratizing GraphRAG](https://arxiv.org/abs/2602.23372)
- EraRAG's growth protocol — [EraRAG](https://arxiv.org/abs/2506.20963)
- [BenchmarkQED](https://github.com/microsoft/benchmark-qed) for scaling the corpus-specific set
- [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) for the capability-level breakdown
