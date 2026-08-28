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

> **Status, 2026-08-28 — implemented.** E15–E19 and R1–R3 are built; §4 and
> §4.1 record what each became and §10 records where the implementation
> contradicted the proposal. §6's three predictions were run rather than
> asserted: **P4 holds, P5 holds, P6 holds** — and the most useful result in
> this document is one none of them predicted, which is that the retrieval gain
> R1 buys does not show up in the answers a judge prefers on the global tier
> (§6.4).
>
> The headline the question below is about now reads, on the same 78 documents
> and the same facts: on questions phrased without the corpus's vocabulary,
> `llmwiki/full` went from **0.45 to 0.66** `recall@10` and from 0.13 below its
> own dense lane to exactly level with it, while bm25 scores **0.02**. The
> `k=10` figure `discriminating-power.md` published is now reported with its
> question class beside it, everywhere it appears.
>
> §1 and §3 are left as they were written. They are the measurement that
> motivated the work, and the fixture they were taken on has since been rebuilt
> three times larger and with a stricter hostility measure — §6.1 has the
> current numbers and §10.1 has why they moved.

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

> **Built.** `ragharness/overlap.py` and `report.decompose`, rendered by
> `compare` as two blocks. The MuSiQue quartile table reproduces in one
> invocation (§6.1). The atlas per-tag block shows `dense` at 0.62 on `thematic`
> against bm25's 0.25, and level with it on `multi-hop` rather than below —
> which is a change in the measurement, not in the system: it is now drawn at
> the widest `k` in the sweep that still has room above the leader, and on
> atlas that is `k=1`, not the `k=3` §1 used.
>
> The measure itself moved twice, and both times because it was wrong. Plain
> IDF gives a term the corpus does not contain the *largest* weight in the
> query, so "What does the Corvid unit do?" scored 0.17 against its own gold
> page because *what* and *do* outweighed *corvid*; and document frequency is
> only a proxy for "this is grammar", which fails on a small corpus — *once*
> appears on two of atlas's 82 pages. Terms are now zeroed at both ends of the
> frequency scale and for closed-class English. §10.1 has what that did to the
> published numbers.

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

> **Built.** All six suites declare it, and an undeclared one prints
> `representativeness: undeclared — this comparison is a claim about these
> questions, not about the product` in the same place `SATURATED` and
> `DEGRADED` appear. `mix` is checked against the questions and the divergence
> printed; on the two generated suites the generator writes the mix from the
> questions it just built, so it cannot drift the way the design in
> [`../README.md`](../README.md) §2 did.

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

> **Two of three.** The suite is **62 questions in fifteen families** — the
> four that existed, plus recency, absence, contrast, supplier and employer
> hops, orbit and tenure. A question whose gold set is more than 60% of the
> pool it draws from is dropped rather than shipped, which removed six.
>
> The **embedding-margin check** runs as `build_atlas.py --verify-embeddings`
> and passes: over the 62 questions the median margin between the best cosine
> on gold and the best off it is **−0.008**, the interquartile range is 0.031,
> and no question exceeds the outlier fence. The suite's gold does *not* stand
> out from the corpus in embedding space, which is the counter-check §7's first
> falsification asked for and the strongest available evidence that these are
> paraphrases rather than synonyms.
>
> **The third part is not done**, and it is the one that matters most. Every
> question here still shares an author with the corpus generator. The build
> prints that on every run and names the file — `fixtures/atlas-authored.jsonl`,
> outside the tree `--force` deletes — where hand-written questions go; they
> pass through the same hostility check. The file is empty.

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

> **Built.** `fixtures/atlas/global`, 24 questions in four groups — what is
> here, how it is structured, where it concentrates, how it changed — tagged
> `global`, carrying no gold set. `compare` excludes them and says how many;
> `ragharness judge` scores them.
>
> Three design decisions carry the number. Every pair is judged **twice with
> the presentation order swapped** and a system wins only if it wins both;
> the judge sees two answers labelled A and B and **nothing else**; and the
> **criteria are judged separately**, because an answer can be more
> comprehensive and less grounded. The swap turned out to be load-bearing: the
> two orders agreed on only 50–79% of pairs, so a single-order run would have
> been reporting the layout as much as the answers.
>
> Both baselines gained an `answer()` — the same generator, the same prompt,
> the same budget, a different ranking — which is what makes "which of these
> two answers would you rather read" a question that can be put about bm25 at
> all. §6.4 has the results, and they are not what R1's retrieval numbers
> predict.

### E19 — Publish the class, not the mean
**Goal:** D11, at the level of what leaves the repository.
**Touches:** `README.md` in both repositories, and every table already written

Once E15 exists, a headline number without its question class is the same defect
as a `recall@k` without its `k`. The convention to adopt, and the one this
document is an argument for: **name the suite, the `k`, and the question class,
or do not publish the number.**

**Done when:** no table in either repository reports a bare aggregate, and the
`README` states which suite is the headline for which product claim.

> **Partly, and the criterion as written is not checkable.** "No table reports a
> bare aggregate" is a claim about every table ever written in two repositories,
> and nobody has read them all. What was done: `compare` now prints the question
> class beside every mean it produces, so no *new* table can be bare; the four
> corrections in §8 are applied; and the headline tables in
> [`discriminating-power.md`](discriminating-power.md),
> [`../README.md`](../README.md) and
> [`../../../future_work/retrieval-rebuild/README.md`](../../../future_work/retrieval-rebuild/README.md)
> carry their class.
>
> The convention this proposes — **name the suite, the `k`, and the question
> class, or do not publish the number** — is now enforced by the tool for
> anything it emits, and by nothing at all for prose. That is the honest state
> of it.

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

> **Built, and it holds.** `retrieval/calibration.py` builds the reference
> distribution from the corpus's own document titles — the cheapest available
> example of a query that genuinely names something in the corpus, needing no
> labels and no provider call, rebuilt whenever the index is. A query whose
> lexical top score falls below the 5th percentile of that distribution
> abstains rather than voting.
>
> The fence separates the two question classes almost perfectly on atlas: the
> 5th percentile of the title distribution is 6.17, **all 62** keyword-hostile
> questions score below it, and 12 of the 44 regular ones do — and gating those
> 12 costs nothing at all. On MuSiQue it fires on 28 of 485; on HotpotQA, on 1
> of 200.
>
> `llmwiki/full` on the keyword-hostile suite went from
> `0.03 / 0.20 / 0.33 / 0.45 / 0.64` to `0.15 / 0.39 / 0.52 / 0.66 / 0.80`,
> which is `dense` to two decimal places at every `k`, and `compare` reports no
> monotonicity violation on that suite at all — where before it reported twelve.
>
> The proposal said the distribution has to be computed per corpus and it does.
> What it got wrong is the phrase "a few hundred queries at index time": no
> queries are needed. §10.2.

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

> **Built, and both halves hold — the second only when measured on a corpus
> with a graph worth skipping.** `full` and `full/no-graph` are now identical at
> every `k` on the hostile suite, which satisfies "at or above". Against the
> lexical gate alone the graph gate is worth
> `+0.00 / +0.04 / +0.04 / +0.04 / +0.04`.
>
> The latency criterion as written — p50 over a suite — cannot show what it is
> about: a suite where the gate fires on 6% of queries has a p50 drawn from the
> 94% where nothing changed. Measured on exactly the queries it fires on, which
> is the number the criterion meant:
>
> | corpus | documents | gated | graph skipped | graph run | saved (p50) |
> |---|---|---|---|---|---|
> | MuSiQue | 5,918 | 28/485 | 7 ms | 144 ms | **137 ms** |
> | atlas | 78 | 62/62 | 1 ms | 10 ms | 8 ms |
>
> Three times the criterion on a corpus with a real graph, and a twentieth of it
> on one with 78 documents. Stating "40 ms" without naming a corpus was the same
> defect this roadmap is about, one level down.
>
> On atlas's *regular* questions the graph gate costs 0.02 at k=5. That column
> is one the harness itself flags as having 2% of the usable scale left above
> the leader, so it is a loss inside the band where it also says differences
> cannot be shown — declared here rather than argued away.

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

> **Built, and it is the least useful of the three.** `retrieval/profiles.py`
> defines `voice` / `balanced` / `deep` / `research` over both axes — depth, and
> two instruments for how much to trust the lexical lane: `abstain_quantile`
> (where the fence sits in the corpus's own distribution) and `lexical_weight`
> (what its vote is worth when it clears the fence). `--profile` is on `ask` and
> `search`; the harness exposes every profile as a lane.
>
> `research` meets its criterion, the criterion turned out to be weak, and the
> profile turned out to be worth more than the criterion could show. On the
> keyword-hostile suite it is *identical* to `balanced`, because R1's gate
> already fires on all 62 questions and there is nothing left for a weight to
> do. The place it matters is the middle — a query with a weak handhold rather
> than none — and on MuSiQue it closes the entire remaining gap to the vector
> lane. §6.5.
>
> One thing the profile work fixed that was not in the proposal: `ask()` never
> took the caller's retrieval options at all, so **every ablation and every
> profile generated its answers from the shipped ranking**. A generation
> comparison between two lanes was comparing one retriever with itself. §10.3.

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

---

### 6.1 P4 — the mean stops being the message. Holds.

MuSiQue, 485 questions, `recall@10`, split into equal-count quartiles by how
much of each query's discriminating vocabulary its own gold documents contain:

```
  by query/gold lexical overlap · recall@10 · 485 of 485 questions
  system                         Q1       Q2       Q3       Q4
                         n      121      121      121      122
                   overlap     0.29     0.47     0.60     0.78
  ────────────────────────────────────────────────────────────
  bm25                         0.32     0.47     0.55     0.68
  llmwiki/lexical              0.40     0.52     0.64     0.75
  dense                        0.63     0.69     0.80     0.89
  llmwiki/hybrid               0.60     0.68     0.81     0.88
```

bm25 more than doubles from Q1 to Q4. The vector lane climbs by 41%, and its
margin over bm25 is **+0.31 at Q1 against +0.21 at Q4** — largest exactly where
the keywords run out. One invocation of `compare` produces that table, which was
E15's criterion.

On atlas the per-tag block separates the two mechanisms, at `k=1` because that
is the only column in the sweep with room above the leader — and the block says
so rather than leaving a reader to notice that every cell at `k=10` is 1.00:

```
  by capability tag · recall@1 · no k in this sweep has room above the leader
  system                   single-hop   multi-hop    thematic   intra-doc
                        n          14          12           8           8
                  overlap        0.79        0.92        0.70        1.00
  ───────────────────────────────────────────────────────────────────────
  bm25                           0.86        0.50        0.25        1.00
  dense                          0.86        0.50        0.62        1.00
  llmwiki/sources                1.00        0.50        0.25        1.00
  llmwiki/full                   0.93        0.50        0.62        1.00
```

`thematic` is the tag with the lowest query/gold overlap and the one where the
vector lane is worth 0.37 over bm25; `intra-doc`, at overlap 1.00, is where
every system is level. That is the variation P4 predicted, and its absence would
have refuted D11.

### 6.2 P5 — the shipped default wins on its own use case. Holds.

The keyword-hostile suite, now 62 questions over the same 78 documents:

```
  atlas-thematic · 62 questions · 78 documents
  represents: research phrasing, every corpus term withheld by construction

  system                          R@1    R@3    R@5   R@10   R@20   separates   local
  ─────────────────────────────────────────────────────────────────────────────────
  llmwiki/full                   0.15   0.39   0.52   0.66   0.80         k=1       2
  llmwiki/full/no-graph-gate     0.15   0.35   0.48   0.62   0.76         k=1      16
  llmwiki/full/no-gate           0.03   0.20   0.33   0.45   0.64         k=1      16
  dense                          0.15   0.39   0.52   0.66   0.80         k=1       1
  bm25                           0.01   0.02   0.02   0.02   0.09           —       0

  · llmwiki/full: delivered and abstained — graph on 62/62, lexical on 62/62
```

`full` is at `dense` at every `k`, which is P5. `no-gate` is the configuration
that shipped before this work and is the row §1 measured. The abstention line is
the whole explanation and is printed under every table: on this suite the
lexical lane stands down on every question, so `full` *is* `dense`, and saying
so is more useful than a reader inferring it from two identical rows.

bm25 at 0.02 rather than §1's 0.10 is the grown suite and the stricter hostility
measure, not a change in bm25 (§10.1).

### 6.3 P6 — the confidence gate does not cost the entity case. Holds.

Every entity-anchored suite, `full`/`hybrid` against the same configuration with
the gate removed:

| suite | n | gated | R@1 | R@3 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|---|---|
| atlas, gate | 44 | 12/44 | 0.76 | 0.92 | 0.96 | 1.00 | — |
| atlas, no gate | 44 | — | 0.76 | 0.92 | **0.98** | 1.00 | — |
| atlas, lexical gate only | 44 | 12/44 | 0.76 | 0.92 | 0.98 | 1.00 | — |
| MuSiQue, gate | 485 | 28/485 | **0.33** | **0.57** | 0.66 | 0.74 | 0.82 |
| MuSiQue, no gate | 485 | — | 0.32 | 0.56 | 0.66 | 0.74 | 0.82 |
| HotpotQA, gate | 200 | 1/200 | 0.45 | **0.86** | 0.96 | 1.00 | 1.00 |
| HotpotQA, no gate | 200 | — | 0.45 | 0.85 | 0.96 | 1.00 | 1.00 |

The lexical gate on its own costs **nothing anywhere** — the atlas rows for
"gate" and "lexical gate only" differ by exactly the graph gate — and is worth
+0.01 at the narrow end on both public benchmarks. The only cost in the table is
the 0.02 the *graph* gate takes at atlas k=5, in a column where the harness
reports 2% of the usable scale left above the leader.

P6 asked whether the threshold was too aggressive and whether a weight would be
the better form. On this evidence it is not too aggressive, and the weight is
worth having for a different reason (§6.5).

### 6.4 The result none of the three predicted

E18's global tier scores what no gold set can: 24 corpus-level questions, two
systems, a judge that sees two answers and nothing else, each pair judged in
both presentation orders. Wins are the left-hand system's.

| comparison | criterion | W–L–T | win rate | 95% CI | orders agreed |
|---|---|---|---|---|---|
| `full` vs **bm25** | comprehensiveness | 11–3–10 | **0.79** | [0.52, 0.92] | 67% |
| `full` vs **bm25** | grounding | 6–4–14 | 0.60 | [0.31, 0.83] | 50% |
| `full` vs **dense** | comprehensiveness | 9–8–7 | 0.53 | [0.31, 0.74] | 79% |
| `full` vs **dense** | grounding | 6–5–10 | 0.55 | [0.28, 0.79] | 76% |
| `full` vs **`full/no-gate`** | comprehensiveness | 6–8–9 | 0.43 | [0.21, 0.67] | 70% |
| `full` vs **`full/no-gate`** | grounding | 5–5–6 | 0.50 | [0.24, 0.76] | 69% |

Three readings, in increasing order of how much they should change what anyone
does.

**Against bm25, on the question class the product exists for, the assembled
system wins on comprehensiveness and ties on grounding.** The comprehensiveness
interval excludes 0.5; the grounding interval does not. That is the first
head-to-head anyone has run on a corpus-level question, and it says the system is
more complete and no better attributed than a keyword search — which is a
narrower claim than the retrieval numbers alone would suggest and is the one the
evidence supports.

**Against `dense`, both criteria tie — which is what the retrieval numbers
predict and is therefore a check on the judge.** The gate fires on these
questions, so `full` and `dense` return the same ranking and the two answers
differ only by sampling. A judge that reported a winner there would be measuring
itself. It did not.

**Against its own pre-gate configuration, the judge shows no difference — and
R1 is worth 0.21 recall at k=10 on the keyword-hostile suite.** Both intervals
span 0.5 and the point estimate on comprehensiveness is *below* it. At n=24 with
this many ties the interval cannot rule out a real gain either, so the honest
statement is that no difference was shown, not that none exists.

The most likely explanation is that a global question is not a retrieval-bound
task on a 78-document corpus: almost any ten pages support a plausible answer to
*"what are the main themes here"*, so a large improvement in *which* ten pages
does not move what a reader prefers. If that is right, then retrieval recall and
answer preference are measuring different things and the roadmap's own argument
applies to itself — a `recall@k` gain on a question class is not a claim about
the answers a person reads until someone has read them.

The order-agreement column is the reason the swap is in the design. At 50% on
one comparison, a single-presentation-order run would have been reporting the
layout as much as the answers.

### 6.5 Where a weight rather than a gate matters

R1's gate is the right instrument when the lexical lane has *no* handhold. It
does nothing for a query with a weak one, and after R1 the shipped configuration
was still below its own vector lane at every `k` on MuSiQue — by 0.003 to 0.03,
small, consistent, and the harness's monotonicity check reported it every time.

`research` — the same fence moved from the 5th percentile to the 25th, plus half
a vote for the lexical lane when it clears it — closes it:

```
  system                       R@1    R@3    R@5   R@10   R@20   separates   local
  ────────────────────────────────────────────────────────────────────────────────
  llmwiki/hybrid              0.33   0.57   0.66   0.74   0.82         k=1     288
  llmwiki/hybrid/research     0.34   0.60   0.68   0.76   0.82         k=1     227
  dense                       0.33   0.60   0.68   0.75   0.82         k=1       3
  bm25                        0.26   0.39   0.44   0.51   0.57           —      15

  by query/gold lexical overlap · recall@10
                                  Q1       Q2       Q3       Q4
  ─────────────────────────────────────────────────────────────
  llmwiki/hybrid                0.60     0.68     0.81     0.88
  llmwiki/hybrid/research       0.63     0.70     0.81     0.90
  dense                         0.63     0.69     0.80     0.89
  bm25                          0.32     0.47     0.55     0.68

  · llmwiki/hybrid:          abstained on  28/485
  · llmwiki/hybrid/research: abstained on 154/485
```

`research` is at or above `dense` at every `k` — the two remaining shortfalls are
0.003 and 0.0002, which is the check reporting rounding — and it is above it in
the lowest and highest overlap quartiles both. It reaches 0.76 at k=10 against
`balanced`'s 0.74 and bm25's 0.51.

This is the shape the whole document has been arguing for. A system whose
advantage over bm25 is +0.31 in the bottom overlap quartile and +0.22 in the top
does not depend on the user already knowing what the corpus calls things, and it
took a profile — a caller saying which kind of question this is — rather than a
better ranker to get there.

It is not the default, and should not be. On atlas — where the questions do name
their own answers — the same profile scores `0.74 / 0.88 / 0.94 / 1.00` against
`balanced`'s `0.76 / 0.92 / 0.96 / 1.00`, and its paired advantage over bm25 at
k=3 falls to zero. Which lane to trust is a property of the question class, and
the caller knows the class in a way the ranker does not.

### 6.6 The regressions that pin it

In `tests/test_invariants.py`, beside the ten the previous roadmaps left, and in
llmwiki's `tests/test_retrieval.py`:

1. A suite with no `represents` field produces a comparison that says so, and a
   declared `mix` that does not match the questions is reported.
2. A query made of its gold document's own rare terms scores above 0.9 on the
   overlap measure and a query of words the corpus does not contain scores 0.0 —
   the two shapes the decomposition has to be able to tell apart, and if it
   cannot, every quartile table it produces is noise.
3. A synthetic pair of lanes, one ranking backwards, fused: equal weights let the
   noise lane take the top slot, an abstaining lane leaves the competent ranking
   untouched, and a 0.1 weight recovers it without an abstention. That is the
   defect D14 named, at the level the ranker can be tested at.
4. A lane that abstains is still a lane that was delivered. Without this the gate
   would mark every gated run degraded and fail the command — the fix would have
   looked exactly like the defect the previous roadmap built that check for.
5. A calibration that cannot be built never gates, so a fence that fails to
   compute cannot silently remove a lane.

---

## 7. What would falsify this roadmap

- **If the keyword-hostile suite is easy for embeddings by construction**, then
  §1's 0.81 is an artifact of the author's paraphrases rather than a property of
  the lane, and E17's second and third parts are not refinements but
  prerequisites. This is the most likely way this document is wrong, and it is
  why the magnitudes here should be read as upper bounds.

  > **Tested, and it survives the test that can be automated.** Over the 62
  > questions the median margin between the best cosine similarity on gold and
  > the best off it is **−0.008**, with an interquartile range of 0.031 and no
  > question above the outlier fence. For a typical question the nearest
  > non-gold chunk is as close as the nearest gold one, which is the opposite
  > of the synonym signature. The paraphrases are not secretly the corpus's own
  > words in embedding space.
  >
  > That does not clear the deeper form of the objection. The check asks whether
  > the *gold stands out*; it cannot ask whether the author, knowing the corpus,
  > chose concepts the embedding model happens to place well. Only E17's third
  > part answers that, and it is not done.
- **If real traffic is entity-anchored after all** — if people using a personal
  wiki mostly do look things up by name — then D11 is backwards, the existing
  suites are representative, and the correct response is to keep the default and
  discard this document. Nobody has looked at real traffic. That is the single
  cheapest piece of evidence available and it has not been gathered, and nothing
  in this implementation changes that.

- **If retrieval recall is not what decides whether an answer is good**, then
  the entire R-series bought a number and not a product. §6.4 is the first
  evidence either way and it points at the uncomfortable side: R1 is worth 0.21
  `recall@10` on the keyword-hostile suite and a judge shows **no difference at
  all** between it and the pre-gate configuration on the global tier. At n=24
  the interval cannot rule out a real gain, so this is not yet a refutation —
  but it is the measurement that would produce one, and it is now cheap to
  repeat at a larger n on a larger corpus. Doing that is the most valuable thing
  left in this document.
- **If R1's gate cannot be calibrated per corpus without labels**, the fallback
  is a profile switch (R3) chosen by the caller rather than inferred from the
  query, which is worse but is not nothing.
- **If the 0.02 bm25 floor is an artifact of a 78-document corpus**, the whole
  §1 table needs rebuilding at a scale where bm25 has room to be mediocre rather
  than absent. atlas is small enough that its behaviour at the floor may not
  resemble a real corpus's. Still open, and the MuSiQue quartile table is the
  partial answer: on 5,918 documents bm25's bottom overlap quartile is 0.32
  rather than 0.02, so the floor is indeed a small-corpus effect and the
  *direction* — bm25 collapsing as the handhold disappears — survives at scale.

---

## 8. Corrections to documents already written

> **All four applied, 2026-08-28.** Each entry now records what the correction
> became as well as what it was.

**8.1 `discriminating-power.md`'s headline understates its own qualification.**
Its §6.1 reports `+0.19` at k=5 on hotpot-1k as the result. That number is an
average over a question set that is 100% entity-anchored, and the document does
not say so. It should carry the class beside the `k`, which is the convention
E19 proposes and which this document is the argument for.
*Applied:* its status banner now carries the class, and its "superseded in one
respect" note carries the current numbers rather than the ones measured on the
20-question probe.

**8.2 `../README.md` §2's tag table describes a suite that does not exist.** It
specifies 10 single-hop, 8 multi-hop, 6 thematic and 6 intra-doc — 20% thematic.
The suites that were built are 98% multi-hop. The design was right and the
implementation drifted, for the ordinary reason that public benchmarks are
available and hand-written thematic questions are not.
*Applied:* the design table now carries the drift and the fact that the gap is
closed — `fixtures/atlas/thematic` (62 questions, vocabulary withheld and
checked) and `fixtures/atlas/global` (24 questions, no answer key) both exist.

**8.3 `future_work/retrieval-rebuild/README.md` §5's constants were all tuned on
one question class.** Every row in that table was fitted on atlas or HotpotQA,
both entity-anchored. The table should say so, because a constant tuned on one
question class and applied to another is the same error as a constant tuned on
one corpus — which is the caveat that table already carries for `MENTION_SCALE`.
*Applied:* §5 carries the caveat above the table and a rule separating the rows
fitted before the second question class existed from the four fitted after it.

**8.4 `../README.md` §5's exclusion of LLM judges needs scoping.** It is right
for the retrieval regression suite and it has been inherited by a question class
it was never argued about. E18 is the scoping.
*Applied:* §5's first row is scoped to the regression suite in the table itself,
with the global tier named as the one place the argument does not reach.

---

## 9. Not measured / thin ice

- **The keyword-hostile suite is 62 questions and its author wrote the corpus
  generator.** The first half is fixed; the second is not, and it is the serious
  one. The embedding-margin check (§7) rules out the mechanical version of the
  bias and cannot rule out the conceptual version. The magnitudes here are still
  upper bounds.

- **Sixty-two questions are not sixty-two independent observations.** The world
  model has four specialities, eight unit templates, ten suppliers and sixteen
  flights, so the fifteen families are combinations of the same few axes and are
  correlated by construction. The paired bootstrap the harness runs does not know
  that, so its intervals on this suite are narrower than they should be. They are
  wide enough to separate a lane with a handhold from one without — a 0.64 gap —
  and they are not wide enough to rank two systems that differ by a few points.

- **The suite is more hostile than real research phrasing.** Fifty-nine of the
  62 questions share *no* term with the corpus at any frequency, so the lexical
  lane has literally nothing to match. Real questions asked before the user knows
  the vocabulary would still land on some of it. This is the extreme of the
  scale, deliberately, and the MuSiQue quartile decomposition (§6.1) is what
  covers the middle of it.
- **Nobody has looked at what users actually ask.** Every claim here about "the
  traffic" is an inference from what the product is described as being for. One
  week of real query logs would settle D11 in either direction and would be
  worth more than all of §4.
- **The overlap measure is one definition among several, and it moved twice
  during implementation.** IDF-weighted share of query vocabulary present in the
  gold document, with terms zeroed at both ends of the frequency scale and for
  closed-class English. Both changes were repairs to real errors (§10.1) and
  both moved the published quartile *boundaries*; neither moved the shape of the
  finding. The recall in each MuSiQue quartile changed by at most 0.06 and bm25
  still roughly doubles across them. The measure is still not the only reasonable
  one, and a different one would move the boundaries again.
- **The confidence distributions in R1 are from one corpus of 78 documents.**
  Whether the separation between 3.87 and 5.98 survives on a corpus where the
  score distribution is wider is exactly the question calibration has to answer,
  and it has not been asked.
- **R1's predicted gain is arithmetic, not a measurement.** "At or above `dense`"
  assumes a perfect gate. A real gate will sit somewhere between 0.68 and 0.81
  and the interesting number is where.

---

## 10. Where the implementation contradicted the proposal

Four places. Each is recorded because a roadmap that only records the parts that
worked is a roadmap nobody can use to estimate the next one.

### 10.1 The measure that defines the fixture was wrong, twice

`build_atlas.py` carried its own copy of the overlap measure and the harness had
another, and they drifted in the way two copies of anything drift. Both had the
same bug and the local copy had it worse: **plain IDF hands a term the corpus
does not contain the largest weight in the query**. So *"What does the Corvid
unit do?"* scored 0.17 against its own gold page, because *what* and *do*
outweighed *corvid*; and in the hostility check, which divides by the query's
total IDF mass, every unseen word inflated the denominator and made every
question look more hostile than it was.

Recomputed correctly, **five of the original twenty questions leaked 0.18 to
0.43** of their IDF mass to their own gold, against a stated ceiling of 0.12.
The fixture's defining property had been verified by a measure that could not
see the violations.

Fixing that exposed the second error. With unseen terms zeroed, *once* — which
appears on two of atlas's 82 pages — became one of the highest-weighted terms in
the corpus, and one question scored 0.43 for containing the word *once*.
Document frequency is only a proxy for "this is grammar", and on a small
templated corpus the proxy fails. Closed-class English is now zeroed by word
class regardless of frequency, which is what the threshold was approximating.

There is now one definition, in `ragharness/overlap.py`, imported by the
generator. The published MuSiQue quartile *recalls* moved by at most 0.06 and
the shape is unchanged; the quartile *boundaries* moved more, and §1's
`bm25 0.10` on the hostile suite is `0.02` under the corrected measure on the
grown suite.

**What this says about the method.** The previous three roadmaps each built a
check that fires when a number is not evidence. This one built a fixture whose
defining property is a *measurement*, and then verified that property with a
second implementation of the same measurement. That is the same defect one level
up, and the fix is the same one the harness applies everywhere else: one
definition, imported, not two that agree by inspection.

### 10.2 R1 said "sample a few hundred queries"; no queries are needed

The proposal offered two ways to build the reference distribution — sample
queries at index time, or derive it from the corpus's page titles — and treated
the first as the obvious one. The second is strictly better and there is no
trade: a title is already an example of a query that names something in the
corpus, it costs one FTS5 lookup, it needs no labels, no provider call and no
traffic, and it is regenerated by the same corpus fingerprint that invalidates
the index, so the fence can never be stale with respect to the documents it was
measured on. 14 ms over 78 titles; 152 ms over 400 on 1,991 documents.

The proposal also said the gate should fire "in the bottom decile". The decile
gates 41% of atlas's regular questions for no additional gain; the 5th
percentile gates all 62 hostile ones and costs nothing. Both were measured
before either was chosen, which is the only part of this that went to plan.

### 10.3 `ask()` never took the caller's retrieval options

Found while wiring profiles: `llmwiki.query.ask` did its own retrieval with
llmwiki's defaults and had no parameter for anything else. So **every ablation
lane and every profile generated its answers from the shipped ranking**, and any
generation comparison between two configurations was comparing one retriever
with itself — a null result guaranteed by the plumbing, which nobody would have
suspected from the output.

Nothing published was wrong, because no generation comparison between lanes had
ever been run. That is the uncomfortable part: the defect was invisible because
the measurement it breaks had never been taken, which is the same shape as D13.

The fix split `ask` into retrieval plus `answer_from(project, question,
settings, response)`. That split is also what let both baselines gain an
`answer()` — the same generator, the same prompt, the same budget, a different
ranking — which is what makes E18's comparison against bm25 possible at all.

### 10.4 The best number in the R-series does not survive contact with a judge

R1 is worth 0.21 `recall@10` on the question class the product exists for. On the
global tier, a judge comparing the same two configurations shows no difference
on either criterion (§6.4).

Neither number is wrong and they are not in conflict: they measure different
things, and the roadmap's own argument — that a metric is only a claim about the
thing it was measured on — applies to `recall@k` exactly as it applied to the
question set. The honest summary of the whole R-series is: it substantially
improves which documents come back on paraphrased questions, and on a
78-document corpus that has not yet been shown to change what a reader prefers.

The right response is neither to discard R1 nor to keep quoting its recall alone.
It is to run §6.4 again at a larger n on a corpus where the answer *could*
depend on retrieval — which is now a single command, and is the most valuable
thing left undone here.
