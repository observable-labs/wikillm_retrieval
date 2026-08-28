# Roadmap: an eval whose questions resemble the work

Compiled 2026-08-28 against ragharness `2a8c987` and llmwiki `8dbebf0`.
Evidence quality: **measured in-repo**, every number below reproduced on this
machine from the shipped fixtures, on the date above.

**Question:** [discriminating-power.md](discriminating-power.md) closed the class
of defect where a comparison was published at an operating point that could not
have separated the systems. The numbers now mean what they say *and* could have
come out the other way. Why does the resulting headline —
`llmwiki +0.19 over bm25` — still not describe the product, and what closes
*that* class?

> **Status, 2026-08-28 — partly implemented.** D11–D15 are reproduced below.
> The fixture that exposes them is built and committed
> (`fixtures/atlas/thematic`, 20 questions, hostility verified at build time);
> E15–E21 and R1–R3 are proposed and none is built. §6 is the acceptance
> protocol to run when they are.

**Where the work lands:** as with the previous two roadmaps, this document lives
in `wikillm_retrieval/research/` while most of the harness steps change
`space_brief/evaluation/ragharness`, a different repository. The R-series belongs
to llmwiki and is the larger half of the value this time, which is itself the
finding.

---

## 1. The failure, in one table

Same 78 documents. Same facts. The questions asked without the corpus's own
vocabulary — `recall@k` on `fixtures/atlas/thematic`, 20 questions:

| system | R@1 | R@3 | R@5 | R@10 | R@20 | separates |
|---|---|---|---|---|---|---|
| dense | **0.26** | **0.65** | **0.74** | **0.81** | **0.88** | k=1 |
| llmwiki/full/no-graph | 0.06 | 0.31 | 0.47 | 0.72 | 0.82 | k=3 |
| llmwiki/full | 0.06 | 0.27 | 0.40 | 0.68 | 0.82 | k=3 |
| llmwiki/sources | 0.04 | 0.09 | 0.12 | 0.16 | 0.30 | k=10 |
| bm25 | 0.04 | 0.10 | 0.10 | 0.10 | 0.12 | — |

Beside the same systems on the same corpus with the regular question set, where
bm25 scores **0.95** at k=10 and every configuration is within 0.05 of it.

**bm25 goes from 0.95 to 0.10.** Not because the corpus changed and not because
the facts changed, but because the questions stopped naming the thing they were
asking about. That is the entire finding, and everything below is a consequence
of it.

Three readings.

**The public benchmarks were measuring string matching.** In HotpotQA and
MuSiQue the gold document's *title* is usually a literal substring of the query:
*"Were **Scott Derrickson** and **Ed Wood** of the same nationality?"* has gold
documents titled `scott-derrickson` and `ed-wood`. Split MuSiQue's 485 questions
into quartiles by how much of the query's IDF-weighted vocabulary the gold page
actually contains, and bm25's `recall@10` climbs monotonically with it:

| quartile | overlap | bm25 | llmwiki/lexical | dense | llmwiki/hybrid |
|---|---|---|---|---|---|
| Q1 lowest | 0.21 | 0.32 | 0.40 | **0.64** | 0.60 |
| Q2 | 0.31 | 0.49 | 0.54 | 0.73 | 0.71 |
| Q3 | 0.41 | 0.59 | 0.66 | 0.81 | 0.82 |
| Q4 highest | 0.57 | 0.62 | 0.72 | 0.84 | 0.83 |

bm25 nearly doubles from Q1 to Q4. The vector lane is far flatter — 0.64 to 0.84
— and its margin over bm25 is *largest where the keywords run out*: +0.32 at Q1
against +0.22 at Q4.

**The two mechanisms are for different failures, and each is worst where the
other is needed.** On atlas at k=3, by capability tag:

| tag | n | overlap | bm25 | dense | llmwiki/sources | llmwiki/full |
|---|---|---|---|---|---|---|
| single-hop | 14 | 0.32 | 1.00 | 1.00 | 1.00 | 1.00 |
| multi-hop | 12 | 0.37 | 0.83 | **0.67** | 0.83 | 0.79 |
| thematic | 8 | 0.28 | **0.62** | 0.88 | 0.88 | 0.88 |
| intra-doc | 8 | 0.58 | 1.00 | 1.00 | 1.00 | 1.00 |

`dense` is the *worst* system on entity bridges, because bridging needs the rare
token embeddings blur. It is the *best* on thematic, which is also the tag with
the lowest lexical overlap. Neither corpus alone shows this; neither mechanism is
redundant.

**And the shipped configuration is the wrong one for the harder class.** On the
keyword-hostile suite `llmwiki/full` scores 0.68 at k=10 where `dense` alone
scores 0.81, and 0.27 against 0.65 at k=3. Equal-weight reciprocal-rank fusion is
mixing a lane scoring 0.10 with a lane scoring 0.81 and landing between them.
The graph lane makes it slightly worse again — `full` 0.27 against
`full/no-graph` 0.31 at k=3, −0.07 at k=5.

Reproduce all of it:

```
cd space_brief/evaluation
python3 fixtures/build_atlas.py --thematic-only
.venv/bin/python -m ragharness.cli compare --suite fixtures/atlas/thematic/suite \
    --k 1,3,5,10,20 --lanes sources,sources/no-graph,full,full/no-graph \
    --cache-queries --allow-saturation --allow-regression
```

---

## 2. The class of defect

The three invariants so far, each the successor of the one above it:

> A run may only report a number under a configuration label if the
> configuration was delivered.

> A comparison may only be published at an operating point where it could have
> come out the other way.

Both hold. Neither says anything about *what was asked*. The successor:

> **The invariant to add.** A comparison may only be published as a claim about
> a system if its question set is representative of the work that system is for
> — and representativeness is a property of the questions, not of the corpus,
> the metric or the operating point.

Every defect below is an instance. The three previous roadmaps each found a way
for a number to be true and uninformative; this is the fourth and the largest,
because the previous three could be caught by a check inside the harness and
this one cannot. Nothing in a suite file says what the product is for.

The progression, stated once:

| | Asks | Defects | Steps |
|---|---|---|---|
| `harness-v1.md` | what should be measured | gaps 1–12 | built |
| `harness-self-validation.md` | did the measurement run | D1–D5 | E1–E7, built |
| `discriminating-power.md` | could it have separated anything | D6–D10 | E8–E14, built |
| **this document** | **were the questions the work** | **D11–D15** | **E15–E21, R1–R3** |

---

## 3. The defects

### D11 — The question set is not the traffic
Across all four suites, 1,729 questions:

| tag | count | share |
|---|---|---|
| multi-hop | 1,697 | 98.1% |
| single-hop | 14 | 0.8% |
| thematic | 8 | 0.5% |
| intra-doc | 8 | 0.5% |
| drift | 2 | 0.1% |

The product is described throughout
[`../../target-architecture/README.md`](../../target-architecture/README.md) as a
research assistant over a personal or organisational corpus. A research
assistant's traffic is dominated by questions asked *before* the user knows what
the corpus calls things — what is in here about X, what relates to what, what
changed. The eval weights that class at one two-hundredth of the entity-lookup
class, and then reports the mean.

### D12 — The benchmarks hand the retriever its answer
HotpotQA and MuSiQue label gold supporting paragraphs, which is why they cost no
API call and why they were chosen ([`../README.md`](../README.md) §4). The cost
of that labelling is that the questions are *constructed from* the gold
paragraphs' entities, so the entity is in the query by construction. bm25 is not
being tested on retrieval; it is being tested on string matching, at which it is
excellent. MuSiQue's adversarial filtering reduces this and does not remove it —
its Q1 quartile still gives bm25 0.32.

### D13 — Nothing measured the class the product exists for
Before this document there were eight thematic questions, all on the
44-question corpus, all scored `expect_mode: any` — so finding one of four gold
pages counted the same as finding four — and all at a `k` where every system
scores 1.00. Three independent reasons why the one capability that matters most
could not have produced a number, in a harness whose previous two roadmaps were
entirely about numbers that could not produce a result.

### D14 — The default configuration is tuned against the unrepresentative class
Every constant in
[`../../../future_work/retrieval-rebuild/README.md`](../../../future_work/retrieval-rebuild/README.md)
§5 was tuned on atlas and checked on HotpotQA — both entity-anchored. On the
keyword-hostile suite the shipped `full` configuration is 0.13 *below* its own
dense lane at k=10 and 0.38 below at k=3. The system ships a default that is
wrong for the query class it exists to serve, and no measurement in the repo
could have said so.

### D15 — Nothing detects that a suite is unrepresentative
The harness refuses a configuration that did not run, a `k` at or past the
corpus size, an operating point with no headroom, and a configuration that
scores below a subset of itself. It has no opinion whatever about the questions.
A suite of a thousand questions that all name their own answer passes every
check in the harness and produces a confident, reproducible, well-separated,
correctly-labelled number about nothing in particular.

---

## 4. The roadmap — harness

Same contract as the previous roadmaps: goal, touches, design, and a criterion
that can be checked.

### E15 — Report by question class and by lexical overlap
**Goal:** D11 and D15, and the cheapest step here by a wide margin.
**Touches:** `report.py`, `cli.py`

`compare` already prints one column per `k`. Add a second decomposition: one
block per capability tag, and one block splitting the question set into
quartiles by **IDF-weighted query/gold overlap** — the share of a query's
discriminating vocabulary that its gold documents actually contain, with terms
appearing on more than half the corpus zeroed because those are grammar rather
than a handhold.

Both are arithmetic over data the runner already has. The overlap measure needs
document frequency over the corpus, which the bm25 adapter builds anyway.

The point is not the extra numbers. It is that a single mean over a question set
of unknown composition is the same category of statement as a single `recall@k`
at an unknown `k`, and the previous roadmap spent itself establishing that the
second one is not publishable.

**Done when:** `compare --suite fixtures/musique/suite --k 10` reproduces §1's
quartile table in one invocation, and the per-tag block on atlas shows `dense`
below bm25 on `multi-hop` and above it on `thematic`.

### E16 — A suite declares what it is representative of
**Goal:** D15. Make the harness able to hold an opinion.
**Touches:** `suite.py`, `cli.py`, both suite files

`suite.json` gains an optional `represents` field — free text naming the traffic
the suite stands for — and an optional `mix`, a tag→share map. `compare` prints
the declared mix beside the measured one and warns when they diverge by more
than a stated tolerance.

This is deliberately weak. The harness cannot know what a product is for, and a
check that pretends to would be worse than none. What it can do is make the
claim explicit and refuse to let it go unstated: a suite whose `represents` is
empty prints `representativeness: undeclared` under every table it produces, in
the same place `SATURATED` and `DEGRADED` appear.

**Done when:** `fixtures/hotpot/suite` declares
`"represents": "entity-anchored multi-hop lookup where the query names its own
entities"`, and a comparison on it prints that line above the paired deltas.

### E17 — Grow the keyword-hostile suite, and remove its author's bias
**Goal:** D13. Twenty questions is a probe, not a measurement.
**Touches:** `fixtures/build_atlas.py`, and a person who is not the author

Three parts, in increasing order of what they cost and what they are worth.

1. **More families from the world model.** The generator currently asks four
   shapes. The model also knows recency, supplier overlap, shared flight
   heritage and per-mission configuration, which support *contrast* ("what do
   these two have in common"), *recency* ("what changed most recently") and
   *negation* ("which of these does not carry X") — all natural research
   phrasings and all expressible with gold sets correct by construction.
2. **A counter-check for the bias this suite has.** `verify_hostility` proves no
   lexical handhold. Nothing proves the paraphrases are not unusually *easy for
   an embedding model* — and "avoids the corpus's words" correlates with "sits
   near the corpus in embedding space". Add a symmetric check: flag a question
   whose gold is the nearest neighbour by a margin far above the suite median,
   which is the signature of a paraphrase that is really a synonym.
3. **Questions from someone who has not read the corpus.** The only genuine fix
   for authorship bias. Thirty questions written by a person given the corpus's
   *subject* and not its text.

**Done when:** the suite is at least 60 questions, the embedding-margin check
runs in the build beside the hostility check, and at least one family was
written by someone other than the person who wrote the corpus generator.

### E18 — A global tier, and a judge confined to it
**Goal:** the part of D11 that no gold set can reach.
**Touches:** a new suite kind; `runner.py`; not the fast path

*"What are the themes of this corpus"* has no answer key — that is what makes it
the real research question and why it has been absent. Two ways to score it and
only one of them generalises:

- **Synthesised gold**, which works for atlas because the world model defines
  the true answer set, and works for no real corpus.
- **A judge**, pairwise, on comprehensiveness and grounding, as BenchmarkQED's
  AutoE does.

[`../README.md`](../README.md) §5 excluded LLM judges from the starting point and
was right to: they add cost, variance and a second thing to debug to a system
whose defining property is a fast local path. That reasoning covers the
regression suite. It does not cover a question class that cannot be scored any
other way, and inheriting it here is inheriting a decision made about a
different problem.

**Done when:** a `global` tier exists with at least 20 AutoQ-style questions, is
scored by pairwise win rate against a named comparator, runs on demand rather
than per commit, and is excluded from `compare` by default.

### E19 — Publish the class, not the mean
**Goal:** D11, at the level of what leaves the repository.
**Touches:** `README.md` in both repositories, and every table already written

Once E15 exists, a headline number without its question class is the same defect
as a `recall@k` without its `k`. The convention to adopt, and the one this
document is an argument for: **name the suite, the `k`, and the question class,
or do not publish the number.**

**Done when:** no table in either repository reports a bare aggregate, and the
`README` states which suite is the headline for which product claim.

---

## 4.1 The roadmap — llmwiki

The R-series is not the harness's to fix, and it is where most of the value is.
Given the same structure as the E-series because it is equally actionable.

### R1 — Fuse by lane confidence, not equally
**Goal:** the largest measured loss in the system.
**Touches:** `retrieval/pipeline.py:_fuse`, `retrieval/index.py`

Equal-weight RRF fusing a lane at 0.10 with a lane at 0.81 lands at 0.68. The
measured cost of not gating: **−0.13 recall at k=10 on the keyword-hostile
suite, −0.38 at k=3**, −0.12 on MuSiQue at k=2, −0.02 on atlas.

An earlier attempt weighted each lane by its own score margin and was worse on
both corpora, because BM25 margins (0.5–0.7) and cosine margins (0.1–0.25) are
not comparable quantities. The repair is to calibrate **within** a lane rather
than across lanes. The signal exists and is already computed — the lexical
lane's own top score, on the same corpus and index:

| question set | n | p25 | median | p75 | max |
|---|---|---|---|---|---|
| atlas, regular questions | 42 | 5.98 | 7.68 | 10.06 | 14.50 |
| atlas, keyword-hostile | 20 | 2.20 | 3.62 | 3.87 | 5.84 |

Three quarters of the hostile queries fall below 3.87; three quarters of the
regular ones sit above 5.98. A lane whose top score falls in the bottom decile
of *its own* distribution over the corpus has nothing to contribute and should
abstain rather than vote.

The distribution has to be computed per corpus — raw BM25 scores do not transfer
between corpora, and a fixed threshold is a constant tuned on one fixture, which
is the mistake this whole line of work keeps finding. Sample a few hundred
queries at index time, or derive it from the score distribution over the corpus's
own page titles, and store it beside the index.

**Done when:** `llmwiki/full` is at or above `dense` at every `k` on the
keyword-hostile suite, at or above `llmwiki/lexical` at every `k` on atlas, and
`compare` reports no monotonicity violation on either.

### R2 — Gate the graph lane on the same signal
**Goal:** two mechanisms that fight become two that switch.
**Touches:** `retrieval/pipeline.py`

Diffusion is worth +0.05 to +0.08 on entity-bridging questions and **−0.04 to
−0.07 on paraphrased ones**, and it costs 64 ms a query on a wiki-shaped graph
(`discriminating-power.md` §7). Seeding PPR from a fused list that the lexical
lane could not rank is diffusing from noise.

The gate is the one R1 already computes: if the lexical lane abstained, the
fused list is the vector lane's ranking and there is no lexical evidence for the
graph to spread. Skip it, and save the 64 ms on exactly the queries where it was
hurting.

**Done when:** on the keyword-hostile suite `full` is at or above
`full/no-graph` at every `k`, and its p50 local latency on that suite falls by
at least 40 ms.

### R3 — Profiles select lanes, not just latency
**Goal:** make the finding configurable rather than global.
**Touches:** the `Profile` work in
[`../../target-architecture/build-plan.md`](../../target-architecture/build-plan.md) step 9

Step 9 specifies `voice` / `balanced` / `deep` / `research` on a latency axis.
The measurement says the right *lane mix* depends on the query class, which
makes profiles the natural home for it — and `research`, the profile named for
this exact use, should probably be vector-first with the lexical lane as a
tie-break rather than a peer.

This reframes step 9 from a performance nicety into the mechanism the rest of
this document needs, and it should move up the build plan accordingly.

**Done when:** a profile selects lane weights as well as depths, and the
`research` profile is at or above `dense` on the keyword-hostile suite without
regressing `balanced` on atlas.

---

## 5. Order

```
E15 ──> E19            (decompose the number, then stop publishing the mean)
E16                    (independent; makes the claim explicit)
E17 ──> E18            (grow the probe into a measurement, then the class no gold set reaches)
R1  ──> R2 ──> R3      (the system half; R1 is the prerequisite for both)
```

**E15 and R1 are the two that matter.** E15 is hours and stops the next headline
from being unreadable. R1 recovers a third of the recall on the query class the
product exists for, and the signal it needs is already computed and thrown away.

Everything else is worth doing and none of it changes a number.

---

## 6. How we know it worked

Three predictions, to be checked against the implementation rather than asserted
by it.

**P4 — the mean stops being the message.** After E15, every table in both
repositories carries its question class, and the `+0.19` headline resolves into
its parts: roughly +0.22 on the high-overlap quartile and +0.28 on the low. *If
the decomposition shows no variation across classes*, then the question set was
more homogeneous than this document claims and D11 is overstated — which would
be a genuine and welcome refutation.

**P5 — the shipped default wins on its own use case.** After R1 and R2,
`llmwiki/full` is at or above `dense` at every `k` on the keyword-hostile suite.
*If it is not*, then equal-weight fusion is not the defect and the vector lane
should be the default outright for this class — which is a smaller and blunter
change than R1, and should then be made.

**P6 — the confidence gate does not cost the entity case.** After R1, atlas and
MuSiQue results are unchanged within their confidence intervals. *If gating
costs the entity-anchored case*, the threshold is too aggressive and the right
form is a weight rather than an abstention.

**The artifact that should exist afterwards.** Per suite, per class:

```
  atlas-thematic · 60 questions · 78 documents · represents: research phrasing

  system              R@1    R@3    R@5   R@10   separates   local  remote
  ────────────────────────────────────────────────────────────────────────
  bm25               0.04   0.10   0.10   0.10           —       1       0
  dense              0.26   0.65   0.74   0.81         k=1      14     324
  llmwiki/full       0.27   0.66   0.76   0.84         k=1      19     324

  by overlap quartile          Q1     Q2     Q3     Q4
  ────────────────────────────────────────────────────
  bm25                       0.02   0.08   0.11   0.19
  llmwiki/full               0.26   0.31   0.29   0.28
```

The second block is the one that carries the claim. A system whose advantage is
flat across overlap quartiles is a system that does not depend on the user
knowing the vocabulary, which is the property a research assistant is supposed
to have and the one nothing in this repository had ever measured.

**The regressions that pin it.** In `tests/test_invariants.py`, beside the ten
the previous roadmaps left:

1. A suite with no `represents` field produces a comparison that says so.
2. A question set where every question names its gold document's title produces
   an overlap decomposition concentrated in the top quartile — the shape that
   should make a reader distrust the mean.
3. A synthetic pair of lanes, one scoring near zero, fused: the result is at or
   above the stronger lane alone.

The third is the one that would have caught D14 before it shipped.

---

## 7. What would falsify this roadmap

- **If the keyword-hostile suite is easy for embeddings by construction**, then
  §1's 0.81 is an artifact of the author's paraphrases rather than a property of
  the lane, and E17's second and third parts are not refinements but
  prerequisites. This is the most likely way this document is wrong, and it is
  why the magnitudes here should be read as upper bounds.
- **If real traffic is entity-anchored after all** — if people using a personal
  wiki mostly do look things up by name — then D11 is backwards, the existing
  suites are representative, and the correct response is to keep the default and
  discard this document. Nobody has looked at real traffic. That is the single
  cheapest piece of evidence available and it has not been gathered.
- **If R1's gate cannot be calibrated per corpus without labels**, the fallback
  is a profile switch (R3) chosen by the caller rather than inferred from the
  query, which is worse but is not nothing.
- **If the 0.10 bm25 floor is an artifact of a 78-document corpus**, the whole
  §1 table needs rebuilding at a scale where bm25 has room to be mediocre rather
  than absent. atlas is small enough that its behaviour at the floor may not
  resemble a real corpus's.

---

## 8. Corrections to documents already written

**8.1 `discriminating-power.md`'s headline understates its own qualification.**
Its §6.1 reports `+0.19` at k=5 on hotpot-1k as the result. That number is an
average over a question set that is 100% entity-anchored, and the document does
not say so. It should carry the class beside the `k`, which is the convention
E19 proposes and which this document is the argument for.

**8.2 `../README.md` §2's tag table describes a suite that does not exist.** It
specifies 10 single-hop, 8 multi-hop, 6 thematic and 6 intra-doc — 20% thematic.
The suites that were built are 98% multi-hop. The design was right and the
implementation drifted, for the ordinary reason that public benchmarks are
available and hand-written thematic questions are not.

**8.3 `future_work/retrieval-rebuild/README.md` §5's constants were all tuned on
one question class.** Every row in that table was fitted on atlas or HotpotQA,
both entity-anchored. The table should say so, because a constant tuned on one
question class and applied to another is the same error as a constant tuned on
one corpus — which is the caveat that table already carries for `MENTION_SCALE`.

**8.4 `../README.md` §5's exclusion of LLM judges needs scoping.** It is right
for the retrieval regression suite and it has been inherited by a question class
it was never argued about. E18 is the scoping.

---

## 9. Not measured / thin ice

- **The keyword-hostile suite is 20 questions and its author wrote the corpus
  generator.** Both halves of that are serious. The effect sizes are large enough
  that the direction is not in doubt; the magnitudes are not trustworthy.
- **Nobody has looked at what users actually ask.** Every claim here about "the
  traffic" is an inference from what the product is described as being for. One
  week of real query logs would settle D11 in either direction and would be
  worth more than all of §4.
- **The overlap measure is one definition among several.** IDF-weighted share of
  query vocabulary present in the gold document, stopwords zeroed above 50%
  document frequency. It correlates with what bm25 can do, which is the point,
  but it is not the only reasonable measure and the quartile boundaries move if
  it changes. It was computed both ways — the stopword-zeroed measure and the
  naive one — and the MuSiQue quartile table is identical to two decimal places
  under both (Q1 bm25 0.31 against 0.32, Q4 0.62 against 0.62), so the
  conclusion is at least not an artifact of that one choice.
- **The confidence distributions in R1 are from one corpus of 78 documents.**
  Whether the separation between 3.87 and 5.98 survives on a corpus where the
  score distribution is wider is exactly the question calibration has to answer,
  and it has not been asked.
- **R1's predicted gain is arithmetic, not a measurement.** "At or above `dense`"
  assumes a perfect gate. A real gate will sit somewhere between 0.68 and 0.81
  and the interesting number is where.
