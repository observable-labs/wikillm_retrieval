# Roadmap: an eval that can tell the systems apart

Compiled 2026-08-28 against ragharness `df581c2` and llmwiki `a0f9bc2`.
Evidence quality: **measured in-repo**, every number below reproduced on this
machine from the two shipped fixtures, on the date above.

**Question:** [harness-self-validation.md](harness-self-validation.md) closed the
class of defect where the harness scored a configuration that never ran. Its
numbers are now trustworthy in the sense that they describe what happened. Why
does the resulting headline table — `bm25 0.90 → llmwiki 0.96 → +embeddings
0.99`, at 4 ms → 15 ms → 581 ms — still not support the conclusion drawn from it,
and what closes *that* class?

> **Status, 2026-08-28 — implemented.** E8–E14 are built; §4.2 records what each
> one became and §11 records where the implementation contradicted the proposal.
> §6's three predictions were run rather than asserted: **P1 holds, P3 holds, P2
> fails on one corpus and is declared** — and the reason it fails is not the one
> §7 hypothesised, which is the most useful thing in this document.
>
> The headline the question below is about now reads, on the same fixture and
> at every `k` rather than one: llmwiki beats bm25 by +0.05 / +0.14 / +0.22 /
> +0.10 / +0.05 at k = 1, 2, 5, 10, 20, every interval excluding zero, with the
> sign stable across the sweep. Sections 1 and 3 are left as they were written,
> because they are the measurement that motivated the work and rewriting them
> would erase it.

> **Superseded in one respect, 2026-08-28.** Everything here holds. What it
> does not say is what its numbers are averages *over*, and
> [representative-questions.md](representative-questions.md) is the answer:
> `+0.19 over bm25` is a mean over a question set that is 98% entity-anchored,
> where the gold document's title is usually a substring of the query and bm25
> is being handed its own answer.
>
> Read with the class attached, the headline is **`+0.19 recall@5 on hotpot-1k,
> 1,000 entity-anchored multi-hop questions`** — which is the convention that
> document proposes and this one is the argument for. On 62 questions over the
> same atlas corpus phrased *without* its vocabulary, bm25 scores **0.02** at
> k=10 and the configuration that shipped when this document was written scored
> 0.45 against its own dense lane's 0.66. That gap is now closed (a lexical lane
> that abstains when it has found nothing), and the two documents together are
> the argument for reporting the class beside the `k` rather than either alone.
>
> This document made the numbers separable; the next one asks whether they were
> about the right thing.

**Where the work lands:** as with the previous roadmap, this document lives in
`wikillm_retrieval/research/` and almost every step changes
`space_brief/evaluation/ragharness`, a different repository. §7 lists the parts
that belong to llmwiki, and §9 lists the documents in *this* repository that
this finding makes stale.

---

## 1. The failure, in one table

> *As measured on 2026-08-28 before any of this was built. §6.1 has the same
> tables after, and the difference between them is the result.*

The committed headline reports a single `k`. Sweeping it, same suite, same
corpus, same commit — `recall@k` on HotpotQA, 200 questions over 1,991
paragraphs:

| k | bm25 | llmwiki lexical | lexical, graph off | dense | llmwiki hybrid |
|---|---|---|---|---|---|
| 1 | 0.40 | 0.41 | **0.42** | — | — |
| 2 | 0.59 | 0.56 | 0.60 | **0.77** | 0.66 |
| 5 | 0.74 | 0.76 | 0.74 | **0.97** | 0.91 |
| 10 | 0.90 | **0.96** | 0.90 | 0.99 | 0.99 |
| 20 | 0.95 | **0.98** | 0.95 | — | — |

Three readings, none of them available from the k=10 row alone.

**The graph lane's advantage exists at one k.** `+0.07 [+0.04, +0.09]` at k=10;
`+0.01` with a CI spanning zero at k=5; `−0.03` at k=2; `−0.01` at k=1. A
mechanism whose sign changes with the size of the result window is re-ranking
inside a window, not retrieving better. The published result it was checked
against — SPRIG's RRF 0.851 → seeded-PPR 0.867 — is reported at R@5, where this
implementation shows nothing.

**The hybrid is beaten by its own dense lane wherever the metric has room.**
0.66 against 0.77 at k=2, 0.91 against 0.97 at k=5. They meet at 0.99 at k=10
because there is nothing left above them.
[`../harness-v1.md`](../harness-v1.md) §3.4 named this exact comparison as the
reason the baseline adapters exist — *"they are what turn 'recall went up 0.03'
into 'the hybrid beats both of its parts'"* — and the check was never written,
so the one claim the baselines were added to test is the one they were never
asked.

**The latency column is mostly somebody else's network.** p50: bm25 2 ms,
llmwiki local lanes 8–10 ms, llmwiki hybrid 421 ms, dense alone 426 ms. Roughly
410 ms of it is one HTTP round trip to embed the query, paid identically by the
baseline that outscores the hybrid. Read as printed, the table says the pipeline
costs 400 ms; it costs 6–8 ms over bm25, and the provider costs the rest.

The corpus-specific suite tells the same story in a blunter form —
`recall@k` on atlas, 44 questions over 78 documents:

| k | bm25 | llmwiki/sources | sources, graph off | sources, presence scorer |
|---|---|---|---|---|
| 1 | 0.67 | 0.67 | **0.71** | 0.64 |
| 3 | 0.88 | **0.93** | 0.90 | 0.74 |
| 5 | 0.90 | **1.00** | 0.90 | 0.87 |
| 10 | 0.95 | **1.00** | 0.95 | 0.90 |

At its own declared `k`, the system under test scores 1.00. A metric at 1.00 can
report a regression and nothing else.

Reproduce all of it:

```
cd space_brief/evaluation
for K in 1 2 5 10 20; do
  .venv/bin/python -m ragharness.cli compare --suite fixtures/hotpot/suite \
      --k $K --lanes lexical,lexical/no-graph --no-save
done
for K in 1 3 5 10; do
  .venv/bin/python -m ragharness.cli compare --suite fixtures/atlas/suite \
      --k $K --lanes sources,sources/no-graph,sources/presence --no-save
done
```

The hybrid and dense rows need `LLMWIKI_EMBEDDING_MODEL` in the environment and
a built `vectors.db`; without them the harness refuses those rows rather than
printing them, which is E1 working.

---

## 2. The class of defect

The previous roadmap's invariant was:

> A run may only report a number under a configuration label if the
> configuration was delivered.

That is now enforced. It says nothing about whether the number could have
carried information. Every defect below is an instance of the successor:

> **The invariant to add.** A comparison may only be published at an operating
> point where it could have come out the other way — and the operating point is
> a property of the corpus, the questions, *and* the systems being compared.

`harness-self-validation.md` §10.2 got within one step of this and stopped:

> *"Headroom is not a property of the corpus alone — it is a property of the
> corpus, the questions and the system together, and the only honest check is
> that the number moves when a mechanism is removed."*

That sentence was written as a correction to an acceptance criterion. It is
actually the specification for E8–E11, and this document is what happens when
you make it executable rather than advisory.

The progression, stated once:

| | Asks | Status |
|---|---|---|
| `harness-v1.md` §1, gaps 1–10 | what should be measured | built |
| `harness-self-validation.md`, D1–D5 | did the measurement run | built, E1–E7 |
| **this document, D6–D10** | **could the measurement have separated anything** | **proposed** |

---

## 3. The defects

*All five are closed; §4.2 says by what. They are stated here as they were found.*

### D6 — The declared `k` sits against the ceiling
Both suites declare `k: 10`. At that k, bm25 alone already scores 0.90 on
HotpotQA and 0.95 on atlas, so the entire field is compressed into the last ten
or five points of the scale, and llmwiki reaches 1.00 on atlas outright.

E3 built the guard for this and scoped it to the degenerate case: `compare`
refuses when `k >= corpus_size` (`cli.py:_refuse_saturation`). 10 against 1,991
passes that check comfortably. Saturation is not a function of corpus size — a
system can be at the ceiling on a corpus a thousand times larger than `k`, and
here it is.

### D7 — A single `k` is reported as if it were the result
`compare` takes one `k` and prints one row per system.
[`../README.md`](../README.md) §1 opens by rejecting exactly this shape — *"the
primary artifact is not a score, it is a quality-versus-latency curve"* — and
then the artifact that ships is a score. The consequence is §1's first reading:
a mechanism with a sign that changes across k was published with its sign taken
from the most favourable column, and no reader of that table could have known.

### D8 — Nothing checks that the whole beats its strongest part
`report.lane_monotonicity` implements the right idea and cannot see the
violation. Its first line is:

```python
scored = {label: (run, aggregate(run)) for label, run in results.items()
          if run.delivery is not None}
```

`bm25` and `dense` (`adapters/bm25.py:50`, `adapters/dense.py:22`) have no
`delivered()`, so they are filtered out before the comparison, and the check
only ever runs llmwiki against llmwiki. Every ablation lane is *subtractive* —
it asks what removing a mechanism costs — and none asks whether the assembled
system beats a baseline that is one of its own lanes. dense does beat it, at
every k where the answer is visible.

### D9 — A monotonicity violation prints and the command succeeds
`cmd_compare` ends `return 1 if degraded or failed else 0`
(`cli.py`). `regressions` from `lane_monotonicity` is printed on a `!` line and
excluded from the exit code. During the runs in §1 it fired three times — HotpotQA
at k=1 and k=2, atlas at k=1 — and every one of those commands exited 0. The
harness detected the defect this whole exercise exists to catch and reported
success.

### D10 — Latency is one undifferentiated number
`Row` carries `total_ms` and `load_ms`; the split exists because document
loading once dominated and masked everything else. There is no equivalent split
for a remote call, so a 410 ms provider round trip and 8 ms of local ranking sum
into one column. The quality-versus-latency curve is the stated primary artifact
and its x-axis currently mixes two terms with different owners, different
variances, and different fixes.

---

## 4. The roadmap

Same contract as the previous roadmap and as
[`../../target-architecture/build-plan.md`](../../target-architecture/build-plan.md):
goal, touches, design, and a criterion that can be checked. E8–E11 are hours
each and are the ones that make the existing numbers safe to quote. E12–E14 are
larger.

### E8 — Saturation is measured against headroom, not corpus size
**Goal:** D6. Extend E3 from the degenerate case to the common one.
**Touches:** `runner.py`, `cli.py`, `report.py`

`Run` already records `saturated`. Add `Run.headroom`: `1.0 − recall` of the
best-scoring system in the comparison. Warn below 0.15; refuse in `compare` when
the leading system is at or above 0.98, because nothing can be shown above it.
Keep the existing `k >= corpus_size` refusal — it is a different and cheaper
check that fires before any system runs.

The threshold is a judgement call and should be recorded as one, in the same
table as the tuned retrieval constants: 0.15 is roughly twice the width of the
paired CI at n=200 and is the point below which most pairwise comparisons on
these suites stop separating.

**Done when:** `compare --suite fixtures/atlas/suite --k 10` refuses, naming
`llmwiki/sources` at 1.00; `--k 3` runs; and the refusal message says which
system is at the ceiling rather than only that one is.

### E9 — The operating curve is the artifact
**Goal:** D7. Make [`../README.md`](../README.md) §1 literally true.
**Touches:** `cli.py`, `report.py`

`--k` accepts a list. `compare` runs each k and prints one column per k, plus a
**separation** column: the smallest k at which this system's paired CI against
`--against` excludes zero, or `none`. Default to the suite's declared k as a
list of one so nothing existing breaks; make the sweep the documented default in
the README and in the reproduction commands.

The separation column is the part that carries the finding. `+0.07 at k=10,
none below` is a different claim from `+0.07`, and it is the claim the data
supports.

**Done when:** `compare --suite fixtures/hotpot/suite --k 1,2,5,10,20`
reproduces §1's first table in one invocation, and `llmwiki/lexical` shows
`separates at k=10` while `llmwiki/lexical/no-graph` shows `none`.

### E10 — Baselines join the subset lattice
**Goal:** D8, by reusing E2 rather than adding a mechanism.
**Touches:** `adapters/bm25.py`, `adapters/dense.py`

Give both baselines a `delivered()` over the same key space the llmwiki adapter
already uses — `{sources, embeddings, lexical, graph}`. `bm25` delivers
`{lexical: True}` and whichever `sources` it indexed; `dense` delivers
`{embeddings: True}` and the same. `_strict_subset` requires an identical key
set and already has the rest of the logic, so `lane_monotonicity` starts
comparing across systems the moment the two methods exist. No change to
`report.py`.

This makes the original failure — bm25 outscoring the pipeline that contains a
BM25 lane — mechanically detectable rather than something a person had to
notice, which is a fair description of what this whole line of work is for.

**Done when:** `compare --suite fixtures/hotpot/suite --k 5` reports
`llmwiki/hybrid scored 0.91 against dense's 0.97, and dense is a strict subset
of its lanes — more retrieval returned less`, and the same command at k=10 does
not.

### E11 — A monotonicity violation is a failure
**Goal:** D9. One line.
**Touches:** `cli.py`

`return 1 if degraded or failed or regressions else 0`. Add a `--allow-regression`
escape for the case where the trade is deliberate and named, so that turning the
check into a failure does not make it something people route around.

**Done when:** the k=2 HotpotQA comparison exits 1; the k=10 one exits 0; and a
test asserts the exit code rather than the printed line.

### E12 — Local latency and remote latency are different columns
**Goal:** D10.
**Touches:** `types.py`, `runner.py`, `adapters/llmwiki.py`, `adapters/dense.py`,
`report.py`

Add `Row.remote_ms` beside `load_ms`, populated by any adapter that makes a
network call on the retrieval path. `llmwiki` and `dense` report the
query-embedding round trip; `bm25` reports zero. The compare table prints
`p50 426 (8 + 418)`.

Worth doing beyond the honesty: it is the measurement that decides whether a
local embedding model or a query-vector cache is worth building, and
[`../harness-v1.md`](../harness-v1.md) §12 defers the cache to v2 with no
evidence either way.

**Done when:** the compare table shows the split, and the marginal local cost of
the graph lane over bm25 is readable from it without arithmetic.

### E13 — Corpora with room above the baseline
**Goal:** the guard in E8 has somewhere to run.
**Touches:** `fixtures/build_hotpot.py`, a new `fixtures/build_musique.py`,
`fixtures/build_atlas.py`

Three, in order of cost:

1. **Scale the distractor pool.** `build_hotpot.py --questions 1000` gives
   roughly 10,000 paragraphs against today's 1,991. Same builder, one flag, and
   it is the cheapest available reduction in bm25's score.
2. **Add MuSiQue.** 2–4 hop and adversarially filtered against exactly the
   shortcut single-lane retrievers exploit, which is why it is the standard hard
   member of the trio and why bm25 does not reach 0.90 on it. 2WikiMultiHopQA
   belongs here too; it was skipped in the last round because the dataset was
   not reachable from the mirror used, and that should be retried rather than
   inherited as a decision.
3. **De-template atlas.**
   [`../../../future_work/retrieval-rebuild/README.md`](../../../future_work/retrieval-rebuild/README.md)
   §6 already established that generated-from-template pages made twenty
   subsystem pages near-identical in embedding space, and that this produced a
   dense-lane conclusion that was wrong. The same property is why llmwiki
   reaches 1.00 there.

**Done when:** on the largest suite, bm25 `recall@10` is at or below 0.80 —
leaving headroom of 0.20 at the declared k — and the atlas corpus no longer
yields 1.00 for any system at its declared k.

### E14 — Run the two protocols that are built and have never been run
**Goal:** not a new defect; the standing gap between what the harness can do and
what it has done.
**Touches:** nothing; these are invocations.

`recall@context` is wired end to end (`metrics.py`, `runner.py:208`,
`report.py:116`) and is `null` in all ten run artifacts on disk, because it
requires `--generate` and no run has used it. So are `answer_hit`,
`citation_rate`, the reasoning-token and answer-latency fields E6 added, and the
refusal probes in `known-gaps.jsonl`. `growth()` and `agreement()` — the EraRAG
protocol, and the only check of the O(document) claim in
[`../../incremental-updates.md`](../../incremental-updates.md) — have still never
been run.

The generation run is one command and a bounded number of API calls. The growth
run costs an LLM ingest and destroys a corpus, which is why it keeps not
happening, and it should be run against a copy of atlas rather than anything
real.

**Done when:** at least one run artifact per suite carries a non-null
`recall_at_context`, and the three growth curves exist as an artifact rather
than as a function.

---

## 4.1 Scoped, and still absent

Not steps — a standing list, so the difference between the harness's design and
its implementation stays legible. Compiled by auditing
[`../harness-v1.md`](../harness-v1.md) and [`../README.md`](../README.md) against
the tree at `df581c2`.

| Scoped in | Item | Verdict |
|---|---|---|
| `README.md` §1 | the four profiles (`voice`/`balanced`/`deep`/`research`) | not built; blocked on build-plan step 9. The "curve across profiles" is a curve across `k` (E9), which is the axis that turned out to matter first |
| `README.md` §1, `benchmarking.md` §1 | index cost per document, in seconds and tokens, beside every quality number | `IngestReport`/`IndexReport` exist; no report prints them |
| `benchmarking.md` §1 | peak RSS, as SPRIG reports under a 4 GB cap | not measured — and now worth measuring, because `VectorStore._scan` holds the whole matrix in memory for the life of the process (30 MB at 9,931 chunks) |
| `benchmarking.md` §2 | GraphRAG-Bench, complex-reasoning level | not built. It is the only public suite whose thesis is *when graph structure pays*, which is precisely D7's question |
| `benchmarking.md` §3 | BenchmarkQED AutoQ, to scale the corpus-specific set | not built; correctly gated on the hand-written set proving out |
| `harness-v1.md` §3.3 | the `canonical` id space | **now exercised, and it was broken.** `fixtures/atlas/growth` declares it and needs it: gold addressed to the input document is the only kind an ingest cannot break (§11.4). Using it found that the mapping never resolved frontmatter provenance to corpus ids (§11.6). Both retrieval suites still declare `native` |
| `benchmarking.md` §5 | RAGAS / ARES / RAGChecker | deliberately excluded, and that still looks right |

The honest one-line summary of what the harness encompasses today: **SPRIG's
retrieval protocol, end to end, on one of its two datasets.** Everything else
named in the research is either a design borrowed from a paper — EraRAG's growth
shape, GraphRAG-Bench's capability levels as the tag set, BenchmarkQED's
local/global axis as `single-hop`/`thematic` — or absent.

---

## 4.2 What each step became

Written against `ragharness` and `llmwiki` after the work, so the difference
between the proposal and the implementation is legible rather than quietly
absorbed.

| Step | Became | Note |
|---|---|---|
| E8 | `cli.headroom` + `_refuse_headroom`, thresholds 0.15 / 0.98 as proposed | measured against the *achievable* ceiling, not 1.0 — see §11.1 |
| E9 | `--k 1,2,5,10,20`, a `separates` column, and a second table of per-k deltas | as proposed; the delta table was not, and it is where the sign change is visible |
| E10 | `delivered()` on `bm25` and `dense`, no change to `report.py` | as proposed, and it fired on its first run |
| E11 | `return 1 if degraded or failed or regressions`, `--allow-regression` | as proposed |
| E12 | `Row.remote_ms`, `p50_local_ms` / `p50_remote_ms`, and `remote.py` | grew a query-embedding cache the proposal did not have — see §11.2 |
| E13 | `build_hotpot.py --questions 1000 --out hotpot-1k`, 9,769 paragraphs | needed two llmwiki fixes before it was affordable — §11.3 |
| E14 | the generation run; `fixtures/atlas/growth` | the growth protocol needed a suite that did not exist — §11.4 |

Four llmwiki changes came out of running them, and they are the substance of the
result. Each is Pareto-safe or better on both corpora and each is recorded beside
the constant it changed:

| Change | Where | Worth |
|---|---|---|
| a page scores its best chunk, not its chunk count | `embeddings.group_by_page` | hotpot R@2 0.720 → 0.863, atlas R@1 0.023 → 0.705 |
| `RRF_K` 60 → 3, rescaled to the depth actually fused | `retrieval/graph.py` | hotpot R@2 0.672 → 0.730, R@5 0.912 → 0.953 |
| PPR seeds from the window, not a fixed five | `retrieval/ppr.py` | removes the graph lane's sign change on both corpora |
| chunks batch across documents; the scan skips text and is cached | `embeddings.py` | ingest 9,769 pages in 5 minutes rather than 4 hours |

---

## 5. Order

```
E8 ──> E9              (know when k is uninformative, then stop reporting one k)
E10 ─> E11             (let the check see the baselines, then let it fail)
E12                    (independent; decides whether a query-vector cache is worth building)
E13                    (independent, and the largest; gives E8 room to work)
E14                    (independent; invocations, not code)
```

E8, E10 and E11 are perhaps a day between them and they are what make the
current numbers safe to quote. **Until E9 lands, no `recall@k` from either suite
should be published without the k it was measured at and the k values where it
does not hold** — which is the same warning the previous roadmap's §5 gave about
its own §1 table, for the same reason one level up.

---

## 6. How we know it worked

Three falsifiable predictions, to be checked against the implementation rather
than asserted by it. Each fails loudly, and a failure is informative rather than
merely disappointing.

**P1 — sign stability.** After E9 and E13, llmwiki's advantage over bm25 should
keep its sign across k ∈ {1, 2, 5, 10, 20}. Magnitude may vary freely; the
window is allowed to matter. *If the sign still changes*, the graph lane is
re-ranking within a window rather than retrieving better, and the honest report
is a per-k table with the crossover named — not a headline. This is the
prediction most likely to fail, and the one worth most.

**P2 — the whole beats its strongest part.** After E10, `hybrid ≥ dense` at
every k the suite reports, or the shortfall is declared with
`--allow-regression` and a one-line reason in the run artifact. *If the shortfall
persists undeclared*, the fusion is the defect and §7 is where it goes.

**P3 — the graph's cost is local and small.** After E12, the graph lane's
marginal `local_ms` over bm25 stays under 10 ms at p50 on both suites, and the
remote term is attributed to the embedding provider in the same row. *If the
local term grows with corpus size*, PPR is not converging in the iteration
budget and `DEFAULT_ITERATIONS` is doing work nobody measured.

### 6.1 What happened when they were run

**P1 — sign stability. Holds.** On the 200-question HotpotQA fixture, at the
same commit of the suite and with every change measured on both corpora:

```
  system                        R@1    R@2    R@5   R@10   R@20   separates   local  remote
  llmwiki/lexical              0.42   0.60   0.76   0.95   0.98         k=1      12       0
  llmwiki/lexical/no-graph     0.42   0.60   0.74   0.90   0.95         k=1      12       0
  llmwiki/hybrid               0.45   0.73   0.96   1.00   1.00         k=1      76       0
  llmwiki/hybrid/no-graph      0.45   0.73   0.95   0.99   1.00         k=1      80       0
  bm25                         0.40   0.59   0.74   0.90   0.95           —       2       0
  dense                        0.47   0.86   0.97   0.99   0.99         k=1      71       0

  Δrecall vs bm25                k=1     k=2     k=5    k=10    k=20
  llmwiki/hybrid              +0.05*  +0.14*  +0.22*  +0.10*  +0.05*
  llmwiki/lexical             +0.02*  +0.01   +0.01   +0.05*  +0.04*
```

Compare that with §1's first table, which is the same measurement before any of
this. Read as the ablation rather than against the baseline — `lexical` minus
`lexical/no-graph`, which is what "the graph lane's contribution" has to mean —
it was `−0.01, −0.04, +0.02, +0.06, +0.03` across the same five k. It is now
`+0.00, +0.00, +0.02, +0.05, +0.03`: never negative. And the whole system's
advantage over bm25 separates at every k rather than at one. The mechanism that fixed it is in §4.2: PPR now seeds from the
window it was asked for rather than from a fixed five, so a diffusion driven by
four documents that will not be shown no longer decides the one that is.

It holds at five times the scale. The same sweep on 1,000 HotpotQA questions
over 9,769 pooled paragraphs, which is E13's corpus:

```
  system                        R@1    R@2    R@5   R@10   R@20   separates   local  remote
  llmwiki/hybrid               0.46   0.75   0.94   0.99   1.00         k=1     262       0
  llmwiki/lexical              0.43   0.62   0.77   0.93   0.97         k=1      37       0
  llmwiki/lexical/no-graph     0.43   0.62   0.76   0.87   0.92         k=1      33       0
  bm25                         0.42   0.60   0.75   0.87   0.92           —       7       0
  dense                        0.48   0.87   0.96   0.99   0.99         k=1     216       0

  Δrecall vs bm25                k=1     k=2     k=5    k=10    k=20
  llmwiki/hybrid              +0.05*  +0.15*  +0.19*  +0.12*  +0.08*
  llmwiki/lexical             +0.01*  +0.02*  +0.02*  +0.06*  +0.05*
```

Every delta separates, at every k, at n=1000. The graph lane is worth +0.06 at
k=10 and +0.05 at k=20 and is never negative, which is P1 at five times the
corpus and five times the questions. The `local` column is the pure-Python
vector scan (§11.3) and is the one number here that grows with the corpus.

Atlas, the corpus this system is actually for, tells the same story at the k
where it can still say anything:

```
  system                     R@1    R@2    R@3    R@5   R@10   separates   local  remote
  llmwiki/full              0.76   0.90   0.92   0.98   1.00         k=5      14     342
  llmwiki/full/no-graph     0.76   0.90   0.92   0.93   0.95        none       6       0
  bm25                      0.67   0.83   0.88   0.90   0.95           —       0       0
  dense                     0.74   0.85   0.88   0.93   0.95        none       6     324
```

`separates: none` on the no-graph row is the sharpest statement of what the
graph lane is worth: without it, llmwiki does not beat bm25 with confidence at
any k on this corpus.

**P2 — the whole beats its strongest part. Fails on HotpotQA, holds on atlas,
and is declared.** `hybrid` scores below `dense` at k ≤ 5 on HotpotQA and above
it at k ≥ 10, and on MuSiQue it trails by 0.05 at k=2 and catches it at k=20.
On atlas fusion is additive at k=2 and slightly negative at k=5. §7 has the
measurement and the two repairs that were tried and were worse. The important
part is that the failure is *visible*: E10 put the baselines into the subset
lattice, so `compare` prints it and exits 1 without anyone having to notice.

**P3 — the graph's cost is local and small. Holds.** Measured on the cold path
with no query cache, atlas at k=3: `llmwiki/full` p50 is 14 ms local + 342 ms
remote, `dense` is 6 ms + 324 ms, bm25 is under 1 ms of both. The graph lane
itself is 1.6–3.8 ms marginal across k ∈ {1, 5, 10, 20} on HotpotQA. The claim
the old single-column table could not support — *the pipeline costs 400 ms* — is
now readable as what it is: the pipeline costs about 13 ms over bm25 and rents
the rest from an embedding provider that the baseline it is being compared
against rents from too.

The prediction's failure clause named the wrong suspect, which is worth
recording — and then a second measurement showed the graph lane *does* have a
cost problem, just not the one P3 described or the one the vector scan
explains (§7). It said that if the local term grew with corpus size, PPR would
not be converging in its iteration budget. The local term *does* grow with
corpus size — 12 ms on 2,000 documents against 76 ms with the vector lane on the same
corpus, and 269 ms a query on 9,931 chunks — and PPR has nothing to do with it.
It is `VectorStore.search` scanning every chunk in pure Python. Two of the four
llmwiki changes in §4.2 are about that, and after them the scan is 5.6 ms a
query on 9,931 chunks against 6.1 ms on 2,032 — flat, which is the property that
was missing. What remains is llmwiki's shipped configuration: `numpy` is an
opportunistic accelerator, not a dependency, and without it the same scan is
269 ms every query. It is now a declared extra (`pip install llmwiki[vector]`)
so that which one was measured is a configuration rather than an accident of the
environment, and every latency number here was measured without it.

### 6.2 E14, and the number nobody had looked at

`recall@context` was wired end to end and `null` in all ten run artifacts on
disk. Run once, on atlas at k=5 with generation on:

```
  tag            recall@k        Δ     flipped  recall@ctx
  single-hop         1.00        —           —        1.00
  multi-hop          0.92        —           —        0.92
  thematic           1.00        —           —        1.00
  intra-doc          1.00        —           —        1.00
  total              0.98        —           —        0.98

  retrieval    p50 21ms  p95 54ms
  answers      hit 0.95 · citation-rate 0.50
  generation   p50 31079ms  p95 71741ms
```

Two things fall out and neither was the point of the exercise.

`recall@context` equals `recall@k` exactly. `harness-v1.md` §1 gap 6 exists
because *"the packer drops pages that do not fit — retrieval recall ≠ evidence
recall"*, and at k=5 on this corpus the packer drops nothing. The metric was
worth building and the gap it was built for is not open here; it will be at
larger `k` or on longer documents, and now there is a run to compare against
when it is.

**Generation is 1,480× retrieval.** 31 seconds against 21 milliseconds.
[`../README.md`](../README.md) §1 makes the quality-versus-latency curve the
primary artifact and every latency number this roadmap argued about — the 410 ms
embedding round trip, the 14 ms of local ranking, the 8 ms the graph lane costs —
is inside the first 1.5% of the bar. That does not make the retrieval work
pointless: the fast local path is the product's defining property and the
`voice` profile has no generation call in it at all. It does mean the profile
axis (build-plan step 9) is the axis that will dominate the artifact once it
exists, and that a curve drawn only over `k` understates by three orders of
magnitude what a reader will assume it covers.

### 6.3 The growth protocol, run

Never run before, and the reason was not the cost (§11.4). Fourteen source
documents inserted three at a time into an empty project, each slice ingested
through llmwiki and then scored:

```
  step    docs  ingest s  recall@k  failed
  ────────────────────────────────────────
  1          3     209.6      0.43       —
  2          3     246.7      0.71       —
  3          3     264.5      1.00       —
  4          3     209.5      0.93       —
  5          2     103.5      1.00       —
```

**The cost curve is the result.** Seconds per document as the wiki grows from 3
pages to 14: 70, 82, 88, 70, 52. No trend. `incremental-updates.md` §1 claims
ingest is properly incremental — *"adding a document touches only the new source,
its own pages, and embeddings for the touched pages; no global recompute
anywhere"* — and this is the first measurement of it rather than a reading of the
code. What variance there is belongs to the provider, not to the corpus.

**The recall curve tracks coverage, and then dips.** 0.43 at three documents is
exactly 6 of 14 questions: the first three sources are gold for two questions
each. It reaches 1.00 once every gold document is in, and then **falls to 0.93
when three more documents are added** before recovering. One question lost a gold
source from its top 5 to a document that had just arrived. That is small, it is
one question out of fourteen, and it is precisely the interference the growth
protocol exists to detect — a system whose accuracy is stable under insertion
does not do that. It is worth watching on a corpus large enough for the effect to
be measurable rather than anecdotal.

**The third curve — incremental against from-scratch — is not measurable here,
and that is a fact about llmwiki rather than a gap.** `incremental-updates.md` §1
says the constraint is satisfied *vacuously*: nothing persists between queries,
`open_index` reconstructs from the markdown per process and caches on a corpus
fingerprint, so there is no incremental index that could drift from a rebuilt
one. `null-check` passing is the proof. The one persistent derived artifact is
`vectors.db`, which *is* incrementally maintained page by page — so the drift
curve becomes measurable exactly when the growth protocol is run on a lane that
embeds, and that is the concrete next step rather than a vague one.

**The artifact that should exist afterwards.** Not a row — this shape, per
suite, with the profile axis added when build-plan step 9 lands:

```
  hotpot · 1000 questions · 9,847 paragraphs · ragharness <sha> · llmwiki <sha>

  system              R@1    R@2    R@5   R@10   separates   local  remote
  ─────────────────────────────────────────────────────────────────────────
  bm25               0.31   0.44   0.61   0.74           —      2       0
  llmwiki/lexical    0.33   0.47   0.66   0.79        k=2      9       0
  llmwiki/hybrid     0.41   0.58   0.79   0.88        k=1     11     418
  dense              0.38   0.55   0.76   0.85        k=1      3     418
```

Every column in it answers a defect above: four k values answer D7, the
separation column answers D6, the split latency answers D10, and the presence of
`dense` in a table that `lane_monotonicity` can read answers D8.

**The regressions that pin it.** Three, in `tests/test_invariants.py` beside the
seven the previous roadmap left:

1. A comparison at a `k` where the leading system scores 0.99 refuses, and the
   refusal names the system.
2. A synthetic pair — a system delivering `{lexical}` scoring above one
   delivering `{lexical, embeddings}` — makes `compare` exit non-zero, with the
   baseline adapters' `delivered()` in the path rather than stubbed around it.
3. A sweep over three k values where a mechanism's delta changes sign produces a
   `separates at` value of `none` rather than a positive delta from the best
   column.

The third is the one that would have caught this document's finding before it
was published, and it is the reason to write it.

---

## 7. What this finds that is not the harness's to fix

As before, recorded here because they are findings the harness produced and they
belong to llmwiki. The status column is the outcome; the rest is as written
before any of it was run.

| Finding | Location | Status |
|---|---|---|
| RRF fusion scores below its own dense lane at every unsaturated k | `retrieval/pipeline.py:_fuse` | **partly, and mostly not the cause** — see below |
| `RRF_K = 60` is TREC's constant, tuned on 1,000-result lists, applied to lists of 20–50 | `retrieval/graph.py:RRF_K` | **closed** — swept, now 3 |
| The graph lane's contribution changes sign with `k` | `retrieval/ppr.py` | **closed** — seeds now follow the window |
| Diffusion cost is set by the seeds' connected component, not by the corpus | `retrieval/ppr.py` | **open, and measured** — see below |
| A page's vector rank depended on how deep the chunk scan went | `embeddings.group_by_page` | **closed** — not predicted here at all |

**On the first, and this is the part worth reading.** The hypothesis was RRF
dilution: a paragraph the dense lane ranks first scores `1/(60+1) = 0.0164`
while a paragraph both lanes rank tenth scores `2/(60+10) = 0.0286` and outranks
it, so fusing a lane at 0.97 with one at 0.76 lets the weaker lane veto. The
arithmetic is right and it was labelled a hypothesis because the confirming
diagnostic had been rate-limited. Run, it is worth about a third of what was
attributed to it: rescaling `rrf_k` from 60 to 3 moves HotpotQA R@2 from 0.672
to 0.730 and R@5 from 0.912 to 0.953, and the gap to the dense lane stays open.

The dominant term was somewhere else entirely, and nothing in this document
predicted it. `group_by_page` scored a page as `top + min(0.3 × Σ(other chunk
scores), 1 − top)`. Cosine similarities sit in a narrow band around 0.6–0.8, so
that tail term saturates at three chunks: **a page with three retrieved chunks
near 0.6 scored 1.00 while a page with one chunk at 0.85 scored 0.85.** Chunk
count outranked chunk quality — and how many chunks are retrieved is a depth
constant chosen for latency, so the vector lane's ranking depended on it. The
retrieval pipeline scans `max(3 × max(2k, 20), 30)` chunks and the `dense`
baseline scans `max(30, 3k)`, which is precisely why one scored below the other:

```
                          R@1     R@2     R@5    R@10
  hotpot   pipeline depth  0.378   0.720   0.968   0.990
           dense depth     0.420   0.772   0.968   0.990
           best chunk      0.475   0.863   0.973   0.990
  atlas    pipeline depth  0.023   0.068   0.284   0.830
           dense depth     0.114   0.227   0.750   0.886
           best chunk      0.705   0.807   0.886   0.909
```

Atlas is the extreme case because its fourteen multi-chunk pages are the raw
source documents: they saturated at 1.00 and occupied the head of the vector
ranking for every query, whatever the query was. Scoring a page by its best
chunk fixes it and buys an invariant worth more than the recall — **the lane's
ranking no longer depends on the scan depth at all**, so the depth constant can
be tuned for latency without moving a result. Re-sweeping it at 2k, 5k, 10k and
20k after the change moves nothing on either corpus.

That is the lesson the whole exercise keeps producing in different forms. The
harness found the *shape* of the defect correctly — the assembled system scored
below a baseline built from one of its own lanes — and the explanation reasoned
out from the shipped constants was the wrong one. §10 called the attribution
thin ice, and it was.

**On diffusion's cost, which is not where anyone was looking.** The graph lane
costs 4.2 ms a query on hotpot-1k (9,769 documents) and **63.8 ms on MuSiQue
(5,918 documents)** — fifteen times more on a corpus 40% smaller. Three
explanations were ruled out by measurement before the right one:

| | graph nodes | edges | nodes/doc | edges/node | reached per query | iterations | marginal |
|---|---|---|---|---|---|---|---|
| hotpot-1k | 19,364 | 27,372 | 1.98 | 1.4 | 179 (0.9%) | 50.4 | 4.2 ms |
| musique | 11,742 | 15,954 | 1.98 | 1.4 | **2,340 (19.9%)** | 53.9 | **63.8 ms** |

Not corpus size — the expensive corpus is smaller. Not graph density — the two
are identical to two decimal places, and the cheap corpus has more nodes and
more edges. Not the iteration budget, which P3's failure clause named and which
differs by 7%. It is the **size of the connected component the seeds land in**:
thirteen times the frontier at the same iteration count, and diffusion cost is
iterations × frontier.

HotpotQA's pooled distractors are paragraphs that mostly share no entities, so
they form small islands and a seed reaches 0.9% of the graph. MuSiQue's are
paragraphs about interlinked real-world entities, so a seed reaches a fifth of
it. **The consequence is that 4 ms is the unrepresentative number.** A personal
wiki is a single large connected component by construction — that is what a wiki
is — so it will behave like MuSiQue. Both corpora also spend nearly the whole
60-iteration budget rather than converging early to the 1e-8 tolerance, so
`DEFAULT_ITERATIONS` is the first lever and has never been swept.

**What remains open.** After all four changes the hybrid still scores below the
dense lane on HotpotQA at k ≤ 5 (0.73 against 0.86 at k=2), and beats it at
k ≥ 10. The lexical lane on that corpus is 0.60 at k=2 against the vector lane's
0.86, and equal-weight RRF lands almost exactly between them, which is what
equal-weight RRF is for. On atlas — the corpus shaped like the one this system
is for — fusion is additive at k=2, beating the lexical lane's 0.864 and the
vector lane's 0.818 with 0.875. It is **not** additive at k=5, where the lexical
lane alone reaches 0.955 against the fused 0.932, and through the harness the
same thing shows as `sources` scoring 1.00 at k=5 against `full`'s 0.98. An
earlier draft of this section claimed the hybrid beat both of its lanes at every
k on atlas; that was wrong, and the corpus's templated pages are why (§9.2).
What atlas shows unambiguously is that the graph lane is the entire margin over
bm25, and that `full` beats `dense` at every k.

Two repairs were measured and both were worse. Weighting the lanes by their own
score margin fails because the margins are not comparable across lanes: BM25
margins sit near 0.5–0.7 and cosine margins near 0.1–0.25, so weighting by them
just suppresses the vector lane (HotpotQA R@2 0.730 → 0.627). Treating an
unranked document as ranked one past the lane's depth moves nothing on either
corpus. Calibrating the margins per lane would need labelled data, and the only
labelled data here is the eval — which is the one place it must not come from.

So the shortfall is **declared rather than tuned away**: it is a property of a
corpus whose lexical lane is much the weaker, it is visible in the compare table
because E10 put `dense` in the lattice, and `--allow-regression` is what a
deliberate version of it looks like. The thing that would settle it is a
per-query estimate of lane reliability, which is a research question and not a
constant.

---

## 8. What would falsify this roadmap

- **If E10's subset claim is unsound** — bm25 is a *different implementation* of
  the lexical capability, not literally one of llmwiki's lanes, and llmwiki's
  lexical lane could stop being BM25 without the key space noticing. The claim
  holds only while `retrieval/lexical.py` is FTS5 `bm25()` over the same columns.
  If it stops being that, the cross-system comparison weakens to a warning and
  the strict-subset guarantee stays within one adapter.
- **If E8's 0.98 refusal threshold blocks legitimate work** — a suite where every
  system genuinely belongs above 0.98 is a suite that has been outgrown, and the
  right response is E13 rather than a lower threshold. But if it fires on a
  narrow-domain corpus where the ceiling is real, the guard should key on the
  *spread* between systems rather than on the leader's absolute score.
- **If P1 fails and the sign keeps changing after E13**, the seeded-PPR step of
  the build plan does not do on this corpus what it does in SPRIG's paper, and
  the conclusion in
  [`../../../future_work/retrieval-rebuild/README.md`](../../../future_work/retrieval-rebuild/README.md)
  §4 — that the direction reproduces — needs withdrawing rather than qualifying.
- **If E12 shows the local term is not small**, E9's separation column is
  measuring a system with a latency profile nobody has characterised, and the
  profile work (build-plan step 9) moves ahead of the rest of this roadmap.

---

## 9. Corrections to documents already written

Each entry now carries what was done about it. Two of the five were overtaken by
the implementation: the retrieval numbers they wanted qualified no longer exist,
because the retrieval that produced them changed.

**9.1 — applied, and superseded.** `future_work/retrieval-rebuild/README.md` §4 reports one column of a
curve.** Its tables are correct at k=10 and are the only k shown. The
qualification that belongs beside them: *at SPRIG's own R@5 the graph lane's
gain is +0.01 with a CI spanning zero, and at k ≤ 2 it is negative.* Its §4
sentence "the direction reproduces, with a wider gap" should read "the direction
reproduces at k=10 and not below."

*What happened:* the qualification was correct on the day and is now a statement
about a version of the pipeline that no longer exists — the graph lane's gain is
positive at every k on both corpora after §4.2's seed change. That document has
a new section recording both the correction and what replaced it, because the
sequence is the point: the finding was real, and it was fixable.

**9.2 The same document's §6 compares against the wrong thing.** "On HotpotQA the
same lane is worth +0.03 recall and +0.05 MRR over the same configuration
without it" is `full` against `full/no-vector` — a subtractive ablation. It is
true and it is not the comparison that matters, which is against `dense` alone,
where the hybrid loses by 0.06 at k=5 and 0.11 at k=2. The section's own thesis —
*"the clearest argument in the whole exercise for not drawing conclusions from
one corpus"* — extends to not drawing them from one `k` and one direction of
ablation.

*What happened:* right about the comparison, and the conclusion it was
defending turned out to be mostly an artifact. "The dense lane does not help on
atlas — and that was the fixture, not the lane" was measured while
`group_by_page` was ranking atlas's raw sources above everything on every query
(§7). With that fixed the dense lane on atlas is 0.705 recall@1 rather than
0.023, and `full` beats `dense` at every k. The vector lane is still
mildly negative against `sources` at k=3 and k=5, which is the residue of the
same effect. The templated-corpus
effect is real and was not the whole of it.

**9.3 `evaluation/README.md` §1 still describes an artifact that does not
exist.** The quality-versus-latency matrix is the document's opening commitment
and the shipped `compare` prints a scalar per system. E9 makes the `k` axis real;
the profile axis waits on build-plan step 9. Worth marking in the document rather
than leaving as an aspiration a reader will assume was met.

*What happened:* E9 landed, so the `k` axis and the split latency columns both
exist and the document says so. The profile axis still does not, and the
document still says that too.

**9.4 `harness-v1.md` §1's gap table needs a twelfth entry.** The previous
roadmap added gap 11, *nothing validates the run against its own configuration.*
Gap 12: *nothing validates that the operating point could have separated the
systems being compared.* Gaps 1–10 ask what to measure, 11 asks whether it ran,
12 asks whether it could have come out differently.

*What happened:* added.

**9.5 `roadmaps/README.md` needs this document in its table**, and its
one-line convention — *"anchor to a run, not an opinion"* — is the reason §7's
RRF attribution is marked as a hypothesis rather than stated as a cause.

*What happened:* added, with a three-row progression table. And the convention
earned itself: the hypothesis was wrong in its dominant term, and it was
labelled, so nothing downstream had to be retracted.

---

## 10. Not measured / thin ice

Struck through in effect rather than in typography: each entry says what it was
and what it is now.

- **The RRF dilution mechanism in §7 was arithmetic, not a measurement.** *Run.*
  It was worth about a third of what was attributed to it, and the dominant term
  was a defect nothing here predicted. §7 and §11.6.
- **Every number is from one machine on one day**, with a warm index cache and a
  shared embedding endpoint. *Still true, and now with a second axis of
  variation:* `numpy` is an opportunistic accelerator rather than a dependency,
  so the vector lane's local latency depends on whether it is installed — 269 ms
  a query on 9,931 chunks without, 5.6 ms with. The measurements here are
  without. The ratio to bm25 is the stable quantity; the absolute is not.
- **The k-sweep was run only on lexical and hybrid configurations.** *Partly
  addressed:* `full` and `sources` on atlas and `hybrid` on HotpotQA are swept
  now. `sources` on HotpotQA is still not, and would be a degraded run there
  anyway — that corpus has no raw sources.
- **MuSiQue has not been attempted.** *Still true.* E13 was satisfied by scaling
  the HotpotQA distractor pool instead, which is the cheaper half of the step.
  2WikiMultiHopQA is also still not run, and the reason remains the one
  inherited from the last round rather than one established here.
- **The 0.15 headroom and 0.98 refusal thresholds in E8 are proposed, not
  derived.** *Still true.* They are the shape of the right guard at roughly the
  right place. What did change is the ruler they are applied to (§11.1), which
  was wrong in a way that mattered more than either threshold.
- **New: the four llmwiki constants were chosen on two corpora.** `RRF_K = 3` is
  anchored to a ratio — 60 was 6% of TREC's 1,000-result lists, and 6% of 50 is
  3 — and then confirmed by a sweep that is monotone between 60 and 3 on both
  corpora and flat below 1. That is better than a fitted value and it is not the
  same as a value that generalises. A third corpus would say.

---

## 11. Where implementation contradicted this document

The previous roadmap kept a section like this and it was the most useful part of
it. Same rule: a proposal that turned out to be wrong is evidence, and quietly
editing it away throws the evidence out.

**11.1 E8's headroom is not measured against 1.0.** The step said "`1.0 −
recall` of the best-scoring system". That is the wrong ruler on any suite whose
questions need more gold documents than the window holds: HotpotQA's are
two-gold with `expect_mode: all`, so `recall@1` cannot exceed 0.50 and a leader
at 0.475 is against the ceiling, not halfway up the scale. Measured against 1.0,
k=1 looks like the roomiest column in the sweep and is the tightest.
`Suite.max_recall_at(k)` is the ceiling and headroom is a share of it. The
thresholds are as proposed and remain, as §10 says, judgement rather than
evidence.

**11.2 E12 needed a cache the step did not mention.** Sweeping five values of k
across two systems re-buys the same 200 query vectors ten times, which is what a
rate limit stops halfway through — it is how the diagnostic in §7 was lost the
first time. `remote.py` caches on (model, dimensions, text), which is the whole
of the embedding function's input. It is opt-in and every run it touches records
that it ran, because a cached run's remote term is a lookup, and a table that
does not say which it is has misreported the x-axis of the primary artifact.

**11.3 E13 was not one flag, and its cheapest step did not work.** The step
said "same builder, one flag, and it is the cheapest available reduction in
bm25's score", with a criterion of bm25 at or below 0.80 recall@10.

**Five times the distractor pool bought 0.03.** bm25 goes 0.90 → 0.87 at k=10
between 1,991 and 9,769 paragraphs, and the criterion is not met. §10 called
that acceptance criterion a prediction rather than a target known to be
reachable, and it was right to. The reading: **corpus size is a weak lever on
baseline difficulty and question difficulty is the strong one.** HotpotQA's gold
paragraphs stay lexically distinctive however many distractors surround them, so
adding distractors mostly adds documents no query matches. That is worth knowing
before anyone spends another ninety minutes of measurement on a bigger pool.

E13's step 2, MuSiQue, is the lever that acts on the right variable, and it was
built for exactly this reason. It works, and it is not close:

```
  system                        R@1    R@2    R@5   R@10   R@20   separates
  llmwiki/lexical              0.28   0.37   0.48   0.58   0.65         k=1
  llmwiki/lexical/no-graph     0.28   0.37   0.47   0.54   0.59         k=1
  bm25                         0.26   0.34   0.44   0.51   0.57           —

  485 questions (249 two-hop, 157 three-hop, 79 four-hop) · 5,918 paragraphs
```

**bm25 recall@10 is 0.51**, against 0.87 on a HotpotQA pool nearly twice the
size. E13's criterion is met with 0.49 of the scale left above the baseline, on
a corpus a third smaller than the one that failed it — which is the finding,
stated as a ratio: adversarial question filtering is worth roughly sixteen times
what five times the distractor pool was worth.

One trap in building it, recorded because it would have gone unnoticed. MuSiQue's
validation split is **ordered by hop count**: the first 500 rows are every one of
them two-hop, so a builder that takes a prefix silently constructs the easy half
of the dataset and labels it MuSiQue. `build_musique.py` reads the whole split
and strides it.

The builder was also one flag only in the sense that the flag existed. The
corpus was not usable until two llmwiki defects were fixed: `index_documents`
made one HTTP request per document — batching only *within* a document, so on a
wiki of one-chunk pages `batch_size` never fired at all — which put 9,769
paragraphs about four hours away rather than five minutes; and `VectorStore`
re-read every chunk's text and re-parsed every vector on every query. Neither
was predicted and both were found by trying to run the step.

**11.4 E14's growth half was blocked on a suite, not on cost.** The step
attributed it to the expense of an LLM ingest and to destroying a corpus. The
actual blocker: `growth()` resets the project and re-ingests, so the wiki pages
it produces are named by a language model and every gold id in both shipped
suites stops resolving. Gold that survives an ingest has to be addressed to the
input — `id_space: canonical`, gold = source paths — and no such suite existed.
`fixtures/atlas/growth` is one, generated from the same world model, and
`build_atlas.py --growth-only` writes it without touching the corpus every
recorded number was measured against.

**11.5 A defect this document did not name, and then found twice more.**
`run --generate` lost an entire 44-question run to one provider 503 on question
thirty-something. The harness's
own rule — *"one provider rate-limiting itself must not take the other systems'
numbers with it"* — was written for `compare` and applied to a whole system; a
generation pass is dozens of sequential provider calls and needed the same rule
one level down. A failed answer is now a note on its row, the retrieval numbers
survive, and the report says how many questions were generated out of how many.
The run that costs the most was the one least likely to produce an artifact,
which is a fair description of why `recall_at_context` was null in all ten runs
on disk.

The growth protocol had the same exposure and it is worse there: dozens of
sequential provider calls over tens of minutes, and the first re-run lost four
minutes and two completed steps to one 503 on document seven. An ingest failure
is now a note on the step with a `failed` column beside the cost and recall, and
`growth` exits non-zero — a curve with a hole in it is still worth having and is
not worth reporting as clean. That is the same rule for the third time, at three
different granularities: a system in a comparison, a question in a run, a
document in a batch. It should probably have been a stated principle rather than
three separate discoveries.

**11.6 A second defect this document did not name, found by running E14 —
and it was the largest single error in the whole exercise.** `id_space:
canonical` was unit-tested and had never been used by a suite. Used, it does not
join, and it went wrong twice.

A page's provenance lives in its frontmatter `sources:` list, and the two
conventions in the wild disagree: `build_atlas.py` writes
`raw/sources/aurora-1-flight-report.md`, llmwiki's own ingest writes
`aurora-1-flight-report.md`. The adapter returned those strings unresolved, so a
canonical suite was scoring gold against whatever the writer of the page
happened to type. Resolving them against the corpus is the adapter's job — that
is what "map onto corpus document ids" means in
[`../harness-v1.md`](../harness-v1.md) §3.3.

Resolving by basename then landed on the *wrong document*. llmwiki writes a
summary page per ingested document at `wiki/sources/<name>.md`, so
`aurora-1-flight-report.md` is ambiguous between the raw document and the page
written about it, and the page won. A hit mapped onto the wrong half of its own
provenance.

Measured on the growth fixture, `recall@5` across the two fixes:

```
  unresolved frontmatter strings         0.14
  resolved, basename wins arbitrarily    0.14
  resolved, raw sources preferred        0.93
```

That is not a rounding error in a metric; it is the metric. And the first
growth curve ever produced — 0.71 falling to 0.14 as the corpus grew — was
entirely this: as ingest added pages, the generated pages outranked the raw
documents, and every one of those hits mapped to an id no gold set contained.
The curve looked exactly like the drift the growth protocol exists to detect.

The pattern is the same one for the third time: `Delivery`, the saturation
guard, and now the canonical id space were each built, reviewed, unit-tested and
never *used*, and each was wrong in a way no amount of reading it would have
shown. The unit tests passed on all three. If there is one transferable lesson
in this document it is that one.

**11.7 §7's attribution was wrong in its dominant term.** Recorded in §7 rather
than here because it is the finding, not a note about the process. §10 called it
thin ice on the day it was written; it was, and the ice was thinner than the
paragraph implied — the arithmetic was sound and the thing it explained was
about a third of the effect.
