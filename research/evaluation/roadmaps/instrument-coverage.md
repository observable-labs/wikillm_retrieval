# Instrument coverage: can this harness see the system that is about to be built?

Written 2026-08-29 against ragharness `9e6aec5` and llmwiki `ef796c7`.
**Implemented 2026-08-29.** §9 records what each step became, §10 what the
instruments reported the first time they were run, and §11 the five places the
proposal below was wrong.

**Question:** four roadmaps made the existing numbers mean what their labels say.
A build plan now proposes capabilities this harness has never been asked to
measure — latency as an input, retrieval that improves with use, depth inside a
single document, a spoken follow-up turn. **Which of them can it already see,
which can it not, and what is the smallest set of changes that closes the gap?**

The plan under test is `research/voice-and-text/build-plan.md`, steps A1–F4.

> ⚠️ **Two documents this one references are not in the tracked tree.** The build
> plan above and the D16–D34 / E20–E45 work items
> (`future_work/ragharness-envolved-improvements/`) currently exist only under
> the gitignored `development/` directory. This roadmap continues their numbering
> at **D35 / E46** because the defects are one progression; the references do not
> resolve until those two are committed. That is a repository-hygiene item, not a
> content one, and it is listed in §7.

---

## 0. How this one differs from the four before it, and what that costs

The roadmaps README states the convention this document is most at risk of
breaking:

> Corrective roadmaps for the evaluation stack — written *after* a harness
> exists and has been run adversarially, so each one is anchored to an observed
> failure rather than to a design review.

The four before it are **retrospective**. Each asks a question about a harness
that has already run, and each could be settled by looking at what it produced:
did the measurement run; could it have separated anything; were the questions the
work; is any of it read against a scale that exists.

This one is **prospective**, and that is a weaker evidential position. "Can it
see a capability nobody has built" cannot be settled by looking at a run, because
the run does not exist. Stated plainly so a later reader can discount it
appropriately:

| | Basis | Strength |
|---|---|---|
| **D35** | Reproduced from 375 banked runs across five swept `k` | **observed** — the anchor |
| **D36–D39** | Reproduced as *absences* from a census of all eight shipped suites and from the type signatures | **verified, but negative** |
| Every step E46–E55 | Design argument against a build plan that has not shipped | **hypothesis** |

The distinction that matters: D35 is a defect in what is measured *today*, and it
would be worth fixing if the build plan were cancelled tomorrow. D36–D39 are
absences that only become defects when the corresponding capability is built. The
sequencing in §6 puts D35's fix first for exactly that reason.

**One roadmap-level claim carries no evidence at all and is labelled here rather
than buried:** that the capabilities in the build plan are worth measuring. This
document assumes the plan; it does not argue for it.

---

## 1. The observed defect

`intra-doc` does not measure intra-document retrieval. It measures whether raw
sources are in corpus scope.

```
ragharness compare --suite fixtures/atlas/suite --k 1,2,3,5,10
```

Per-tag `recall@k` spread across the seven systems in the banked runs — the gap
between the best and worst system on each tag, at each swept `k`:

| tag | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---:|---:|---:|---:|---:|
| single-hop | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 |
| multi-hop | 0.00 | 0.21 | 0.17 | 0.25 | 0.17 |
| thematic | 0.38 | 0.62 | 0.38 | 0.25 | 0.00 |
| **intra-doc** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

(`drift` is omitted: it scores on strings by construction — `Question.scores_on_ids`
excludes it — so its `recall@k` is 0.00 for every system and carries no
information either way.)

Every other tag's spread moves with `k`, which is what a retrieval measurement
does. `intra-doc`'s is **exactly 1.00 at every `k`** — every sources-scoped
system scores 1.00 and every non-sources system scores 0.00, on all eight
questions, at every depth. That is not a hard tag or an easy one. It is a
**binary switch on corpus scope wearing a retrieval tag's name.**

There are two causes and the second is worse than the first.

**The gold is document-level.** `Retrieved` carries `chunk_id` and nothing scores
against it: gold is `Question.expect_ids` matched against canonical *document*
ids. So "did you retrieve the right passage" is never asked — only "was that
document reachable at all".

**And the questions name their own gold documents.** All eight are one template
with the entity substituted:

```
id-001  gold=1  "What wet mass at launch does the Aurora-1 flight report record?"
id-002  gold=1  "What wet mass at launch does the Aurora-2 flight report record?"
…
id-008  gold=1  "What wet mass at launch does the Fulmar-1 flight report record?"
```

Each has exactly one gold document and names it in the query, which is why
`recall@1` is already 1.00. This is the **title-echo** case that D16's proposed
control adapter (E24) exists to detect, arriving in a tag rather than in a
baseline — so the tag is not merely undiscriminating, it is scoring a shortcut
that the sibling roadmap is separately building an instrument to expose.

This is [`discriminating-power`](discriminating-power.md)'s D17 with its cause
identified, and [`representative-questions`](representative-questions.md)'s
finding one level down: the questions were not the work here either.

It is also the sharpest available demonstration that the `k`-sweep
`discriminating-power` built is still earning its cost. A tag whose spread is
**invariant in `k`** is measuring something other than ranking, and no single-`k`
table could have shown it: at k=10 `intra-doc` reads `1.00` beside `single-hop`
and `thematic` at `1.00` and looks like one more spent tag.

---

## 2. Defects

Census reproducing D36–D39, over all eight shipped suites — 1,841 questions:

```
{'multi-hop': 1730, 'paraphrased': 62, 'single-hop': 36, 'thematic': 30,
 'global': 24, 'intra-doc': 18, 'drift': 3, 'refusal': 0}
questions carrying conversation history: 0
```

| # | Defect | Where | Basis |
|---|---|---|---|
| **D35** | ⛔ `intra-doc` scores corpus scope, not passage localisation: spread is exactly 1.00 at every swept `k`. `Retrieved.chunk_id` exists and is never scored against. | `suite.Question.expect_ids`, `metrics.recall_at_k` | **observed** (§1) |
| **D36** | No suite carries a conversational turn, and `System.retrieve(query, k)` has no parameter for one. Every question in every suite is self-contained — the single property real spoken traffic does not have. | `types.System`, `suite.Question` | absence, census |
| **D37** | The `refusal` tag is declared in `TAGS` and used by **zero** questions. No suite contains a question that should return nothing, so the abstention gate — the system's principal differentiator — has no adversarial case and can be tuned away undetected. | `suite.TAGS` | absence, census |
| **D38** | Retrieval configuration is swept by *name*, never by *cost*. `--lanes` selects a named profile; nothing lets the harness ask what 40 ms buys against 400 ms. `Row` carries `total_ms` / `load_ms` / `remote_ms` but no per-stage decomposition, so a budget cannot be attributed to the stage that spent it. | `cli.py --lanes`, `runner.Row` | absence, code |
| **D39** | No protocol varies anything *over use*. `growth_protocol` varies the corpus; nothing varies the query history. A system claimed to improve with use has no instrument, and the naive measurement — scoring the questions just written into the index — is a tautology. | `runner.growth_protocol` | absence, code |

Two defects are **deliberately not renumbered here** because they already exist
and this document depends on them:

- **D33** (the answered surface has only a pairwise judge, one bit per question)
  gates everything the build plan's E3 claims. See §5.
- **D34** (`recall_at_context` records presence, not position) is half of what
  the build plan's B1 needs. Its fix, **E45**, is a prerequisite below rather
  than a duplicate.

---

## 3. Steps

Cost as in the sibling document: **XS** ≤ an hour, **S** ≤ a day, **M** a few
days. Everything is stdlib and `$0` except where **provider** or **person**
appears.

### Wave 1 — the observed defect

| # | Step | Closes | Cost | Depends | Falsifier |
|---|---|---|---|---|---|
| **E46** | **Passage-level gold.** Add `expect_spans` to `Question` — a gold set in the suite's chunk id space — and `recall_at_span` / `span_mrr` in `metrics`. Score it only where declared, so every existing suite is unaffected. Re-gold `atlas`'s 18 `intra-doc` questions at passage level. | D35 | S | E21 (unknown-key rejection, or the field is dropped silently) | Passage recall ranks the systems in the same order document recall does — then localisation carries no information on this corpus, which is a real finding and the cheapest possible refutation of build-plan B1 and F2. |
| **E47** | **Declare the corpus-scope pedestal.** `intra-doc` results print beside the scope the lane ran with, and a tag whose spread is invariant across the swept `k` is flagged `SUSPECT_SWITCH`. Fails the *suite*, never the system — same rule as E26. | D35, D17 | XS | — | No other tag is invariant in `k` — then `intra-doc` was the only switch, and one line keeps it from recurring. |

### Wave 2 — the surfaces the new capabilities land on

| # | Step | Closes | Cost | Depends | Falsifier |
|---|---|---|---|---|---|
| **E48** | **Conversational suites.** `Question.history: tuple[tuple[str, str], ...]` (prior user turn, prior answer), and an optional `retrieve_with_history` on `System` declared by presence like every other capability. A system without it is `Unavailable` on a conversational suite rather than silently scored on the bare utterance. Fixture: ~30 follow-up turns over `atlas`, each with an anaphoric surface form and a self-contained gold form. | D36 | M | — | The bare-utterance and resolved-query scores are equal — then coreference is not costing this corpus anything and build-plan D1 is unjustified *here*, whatever the literature reports. This is the single most valuable falsifier in the document. |
| **E49** | **Refusal questions and abstention metrics.** ~20 questions whose answer is absent from the corpus, tagged `refusal`, scored on `forbid_strings` and on whether the system abstained. Report `abstention_rate` on answerable and unanswerable questions separately — the two together are the gate's operating point, and one without the other is unreadable. | D37 | S | — | The system abstains at the same rate on both — then the gate is not discriminating and its measured 27%-on-clean-questions firing rate is the whole story, not half of it. |

### Wave 3 — the budget axis

| # | Step | Closes | Cost | Depends | Falsifier |
|---|---|---|---|---|---|
| **E50** | **Budget as a swept dimension.** `--budget 40,100,400` alongside `--lanes`, passed to adapters that declare it. ⭐ The honesty rule is **already built**: a system that cannot serve a budget raises `Unavailable`, and one that overruns records it through `Delivery(requested, delivered)`, which makes the run degraded. This is a CLI and adapter change, not a new concept. | D38 | S | build-plan A3 | Quality is flat across the swept budgets — then the ladder's rungs are not buying anything and the knob is decoration, which is the cheapest possible refutation of build-plan A3/A4. |
| **E51** | **Per-stage latency.** Promote a `stage_ms` map onto `Row` beside `total_ms`/`load_ms`/`remote_ms`, populated from whatever the adapter reports. Without it a blown budget cannot be attributed and `remote_ms` remains the only decomposition. | D38 | XS | — | The stages sum to `total_ms` with no residual — then the existing three-way split was sufficient and this is provenance rather than diagnosis. |
| **E52** | **Promote cost to a reported column.** `prompt_tokens` and `completion_tokens` are already recorded into `Row.meta` (`runner.py:282`) and never reported; only `reasoning_tokens` is a field. Promote both and print cost per question beside latency. Build-plan A4's entire acceptance criterion is a cost-versus-quality comparison. | D38 | XS | — | The token counts track `reasoning_tokens` exactly — then one column was enough and this is two lines to prove it. |

### Wave 4 — measurement over use

| # | Step | Closes | Cost | Depends | Falsifier |
|---|---|---|---|---|---|
| **E53** | **`depth_protocol`.** Hold the corpus fixed, replay a query stream in slices, and after each slice score a **held-out** question set in the same topic areas plus control areas the stream never touches. Mirrors `growth_protocol`'s shape on the other variable. `ragharness depth --suite … --stream … --slices N`. | D39 | M | E54 | The held-out curve is flat in streamed areas — then build-plan E1 is a logging feature, which is a publishable outcome and the reason this step precedes building it. |
| **E54** | **The query-stream fixture.** Several distinct phrasings per topic area, a held-out set per area, and untouched control areas. `atlas/thematic`'s 62 `paraphrased` questions are the right source material — vocabulary mismatch is exactly what build-plan E1 claims to close — but 62 is thin once split three ways; expect to write questions, not only partition them. | D39 | M + **person** | — | The stream and held-out splits score identically before any learning — the required precondition. If they differ, the split is confounded and the curve would have been unreadable. |
| **E55** | **Ranking agreement between two systems.** ⭐ `runner.agreement(left, right)` already compares two arbitrary runs' `returned` sets per question; `null-check` calls it on one system twice. Expose it as `ragharness agree --a <lane> --b <lane>` for the storage-port and shadow-read comparisons in build-plan C1 and C5. | — | XS | — | The two implementations already agree on every question — the expected and desired result, and the point is that nothing currently *asks*. |
| **E56** | **Post-ingest query latency.** ⚠️ *Found during §4's coverage check, not before it — it closes no enumerated defect because the absence it addresses was not enumerated.* `growth_protocol` ingests in slices and measures; what build-plan C2 and C3 accept on is narrower — **touch one document, then time the next query**, against the warm p50. A variant of an existing protocol: `ragharness invalidation --suite … --touch 1`. | — | XS | — | First-query and warm-query latency are already within tolerance — then the 866× cliff does not reproduce through the harness's own ingest path, which would mean the measurement scripts and the harness disagree and that is worth knowing before either is trusted. |

---

## 4. Coverage check: does this achieve the goal?

The goal is that every step in the build plan has an instrument. Checked step by
step against its committed acceptance criterion, rather than asserted.

| Build step | Acceptance needs | Instrument | Status |
|---|---|---|---|
| A1 query log | rows written; degrades when unwritable | llmwiki unit tests | ✅ not a harness concern |
| A2 deadlines | per-stage duration and budget; bounded p99 | **E51 + E57** | ✅ closed — nine named stages, each with an outcome (§12) |
| A3 budget input | sweep budget, not profile name | **E50** | ✅ closed here |
| A4 escalation | 3-way cost/quality comparison | **E50 + E52** | ✅ closed here |
| B1 chunk packing | swept `k`, judged, position-aware | **E46** + existing **E45** | ✅ closed, with a dependency |
| B2 eval classes | — | *this roadmap* | ✅ **rewritten** — B2 now points here (§5) |
| C1 storage ports | identical rankings through the port | **E55** | ✅ closed here |
| C2 invalidation | next query within 2× warm p50 after one ingest | **E56** | ✅ closed — *by this check* |
| C3 ingest tables | first-query ≈ warm-query | **E56** | ✅ closed — *by this check* |
| C4 recalibrate | gate firing rate on clean vs hostile sets | **E49** | ✅ closed here |
| C5 AlloyDB | shadow read agrees within tolerance | **E55** | ✅ closed here |
| D0 speech baseline | real transcripts, partials | — | ❌ **out of scope, deliberately** (§5) |
| D1 query rewriting | follow-up retrieves on the resolved query | **E48** | ✅ closed here |
| D2 speculation | first-token latency on a partial stream | — | ❌ **out of scope** (§5) |
| D3 voice profile | graph lane kept; multi-hop restored | banked per-tag runs | ✅ exists today |
| D4 ASR regression | gate fires on corrupted entities | llmwiki unit test | ✅ not a harness concern |
| E1 learned queries | rising held-out curve, flat controls | **E53 + E54** | ✅ closed here |
| E2 gap queue | an unanswerable question surfaces | **E49** | ✅ closed here |
| E3 crystallisation | `global` improves; others don't regress | existing **E43** | ⚠️ **blocked on E43** (§5) |
| F1 rerank | MRR up at fixed recall | exists | ✅ exists today |
| F2 `chunk_read` | intra-doc improves; step cap honoured | **E46** | ✅ closed here |
| F3 navigation lane | thematic improves, single-hop doesn't regress | exists **at k ≤ 5 only** | ⚠️ **partial** (§5) |
| F4 rebuild | reviewable diff | git | ✅ not a harness concern |

**Eighteen of twenty-three closed or already present.** The remaining five are
two deliberate exclusions (D0, D2), one blocked step (E3), one partial (F3), and
one document to rewrite (B2). All five are in §5 rather than papered over.

> **After building it, seventeen.** One row moved the wrong way, and it is worth
> more than the sixteen that held. **A2** is marked ✅ against E51, and E51 is
> built and works — but a per-stage decomposition is only as good as what the
> adapter names, and llmwiki names two stages: `load` and `remote`. Everything
> else is `residual`, which on the invalidation run was 29 ms of a 106 ms turn
> with no further attribution available. A2's criterion is "per-stage duration
> and budget", and what exists is per-*reported*-stage duration plus a bucket.
> The instrument is real and the criterion is not fully served, which is the
> distinction §7's "an acceptance criterion is not an instrument" was about,
> arriving in the one row that looked safest.
>
> **And after closing it, eighteen — with the judgement above unrevised.** E57
> did not add instrumentation to llmwiki. llmwiki had been keeping a per-stage
> map and reporting an outcome for each stage through a `Sink` since A2 landed;
> the adapter was discarding both and substituting the two names it timed from
> outside. The criterion was served by the system under test and thrown away at
> the boundary, which is a worse failure than the one recorded above and was
> invisible for the same reason: a residual is a number, and a number gets read
> as a measurement. §12.

**Two steps appear in no row, correctly.** E47 (`SUSPECT_SWITCH`) and E55's
falsifier are suite-health guards rather than instruments for a build step — they
exist so D35 cannot recur and so C1/C5 have something to *ask*, not because a
criterion demands them. Every other step in §3 is cited above.

**The check changed the plan, which is the point of running it.** C2 and C3 had
no instrument when §3 was written; **E56 exists because this table was filled in
row by row rather than asserted.** A coverage matrix that finds nothing has
usually been read backwards from the steps rather than forwards from the
acceptance criteria.

---

## 5. What this roadmap does not fix

**C2 and C3 had no instrument until this check ran.** Both accept on post-ingest
query latency — the 866× cliff. `growth_protocol` ingests in slices and measures,
which is close but not it: what is needed is *touch one document, then time the
next query*. **Now E56**, added during §4 rather than left as prose, which is
where it sat in the first draft of this document.

**D0 and D2 are deliberately out of scope.** Audio fixtures, streaming partials
and first-token latency are a different kind of artifact from a document
retrieval suite, and llmwiki already keeps `measurements/asr.py`, `spoken.py` and
`fence.py` as standalone scripts. Forcing speech into ragharness would widen its
contract substantially to serve two steps. **This is a decision, not an
oversight, and it should be made explicitly rather than by drift** — the argument
against it is that the speech scripts then never get the controls, multiplicity
handling and noise floor that D16–D34 are building for everything else.

**E3 is blocked on E43, and this is the most important line in the document.**
Crystallisation's acceptance lands on the `global` suite, whose only instrument
is the pairwise judge — D33's one bit per question, an `n` the same plan's own
power arithmetic cannot support. E43 (eRAG per-document downstream utility) is
the fix, and its own sequencing note already says *"if only one item from this
document is built, it is not one of the `$0` ones."*

It is sharper than a dependency. The roadmaps README closes on exactly the
question E1 and E3 need answered first: R1 moved recall 0.21 on the question
class the product exists for, and the judge saw **no difference at all** on
corpus-level questions. Until E43 arbitrates whether that was the judge or the
corpus, every write-back-loop step is building toward a surface with no working
instrument — and would report a null it could not interpret.

**F3 is measurable only at k ≤ 5.** `thematic`'s spread is 0.38 / 0.62 / 0.38 /
0.25 / 0.00 across k = 1, 2, 3, 5, 10. The tag still discriminates at low `k` —
so this is *not* a second D35 — but it is spent at the depth the headline uses,
and E26's band would freeze it. A successor thematic set is needed before F3
ships, not before it is planned.

**Build-plan B2 is wrong and must be rewritten.** It specifies two new question
classes on the claim that "neither has any questions today". False: `intra-doc`
ships 18 questions, `global` ships 24 and is already the cross-document synthesis
instrument, and `paraphrased` ships 62 and is already E1's target class. The step
should be replaced by a reference to this roadmap. The underlying concern
survives in a worse form — the questions exist and the *instrument* does not —
which is D35.

---

## 6. Sequencing

- **E46 and E47 come first and do not depend on the build plan.** They fix a
  defect in what is measured today, and they would be worth doing if every other
  step here were cancelled. Everything else is conditional on a plan shipping.
- **E48 is the long pole and the most informative.** A conversational fixture is
  the largest single piece of work in this document and its falsifier — bare
  utterance scores equal to resolved query — would retire the build plan's most
  confidently-argued voice step. Start it early precisely because it might come
  back negative.
- **E50 cannot start before build-plan A3.** It is the only step here gated on
  the implementation rather than the reverse. Everything else can precede the
  code it will eventually measure, which is the ordering the previous four
  roadmaps established and the reason they worked.
- **E53 and E54 are one build.** A protocol with no fixture measures nothing and
  a fixture with no protocol is a text file. Sequencing them apart is how this
  becomes two half-finished steps.
- **E55 and E52 are nearly free** — one exposes a function that exists, the other
  promotes two values already recorded. Do them beside Wave 1.
- **E43 should be pulled forward out of the sibling plan** if the write-back loop
  is going to be built. It is listed there as the expensive one; it is here as a
  blocker.

---

## 7. What would falsify this roadmap

- **The build plan does not ship.** Every step but E46, E47, E52 and E55 exists
  to measure something that does not exist. If the plan is cancelled or
  substantially reordered, this document is four steps long, and the four are the
  ones anchored to D35 and to code already written.
- **E48 comes back flat.** If resolved and unresolved queries score the same on
  this corpus, the most-cited voice gap in the build plan is not a gap *here*,
  and D36 is a defect about a literature result rather than about this system.
- **Passage-level gold turns out not to change the ranking.** E46's falsifier is
  a real possibility: if localisation carries no information at this corpus size,
  D35 is a labelling defect rather than a measurement one — `intra-doc` should
  simply be renamed `sources-in-scope` and the two build-plan steps that depend
  on it lose their justification.
- **The residual gaps are larger than §5 admits.** The coverage check in §4 was
  run once, by the same author as the plan it checks, against acceptance criteria
  that author also wrote — which is the weakest possible review arrangement. It
  found one missing step (E56) and one document to rewrite (build-plan B2) on its
  first pass, and a check that finds two defects on its first pass has not
  usually found the last two. A second reader filling in the same matrix from the
  acceptance criteria forward is the cheapest way to find out whether eighteen of
  twenty-three was the real number.
- **An acceptance criterion is not an instrument.** §4 marks a row ✅ when a
  criterion has something that could measure it. It does not check that the
  measurement would be *powered* — D24's arithmetic applies to every new question
  class here, and 18 `intra-doc`, 24 `global` and ~20 proposed `refusal`
  questions are all well inside the range where E34's `UNDERPOWERED_EXACT` should
  fire. **The honest reading of §4 is "an instrument exists", not "the question
  can be answered".** Running E32's MDE header against each new class, before
  writing its fixture, is what would turn the second claim into the first.
- **The untracked-tree problem is not hygiene.** §0 assumes committing
  `development/research/voice-and-text/` and
  `development/future_work/ragharness-envolved-improvements/` makes the
  references resolve. If those trees have diverged from their tracked
  counterparts in content rather than only in extent — `research/README.md`,
  `latency-knobs.md` and both `target-architecture` files already differ — then
  the numbering continuity D35/E46 claims may be continuing a sequence that two
  documents disagree about.

---

## 8. Conventions

Inherited from [`README.md`](README.md), including its one addition — anchor to a
run, not an opinion. This document adds a second, because it is the first
prospective roadmap in the series:

- **Label the tense.** A defect about a running system and a defect about a
  system in a plan are different claims with different evidence, and a table that
  mixes them without saying so is the failure mode this whole line of work
  exists to catch. §0 is that label; §2's `Basis` column carries it per row.

---

## 9. What each step became

Eleven steps proposed, eleven built, plus one prerequisite the proposal named
and one llmwiki change it did not know it needed.

### E46 — Passage-level gold. Built, differently.
`ragharness/spans.py`, `Question.expect_spans`, `metrics.recall_at_span` /
`span_mrr`, `Row.recall_at_span` / `span_mrr` / `returned_spans` /
`scored_spans` / `span_source`.

> **The id space is not `chunk_id`.** The proposal said "a gold set in the
> suite's chunk id space", and that cannot work: every system chunks
> differently — llmwiki embeds ~1000-character windows, a lexical baseline has
> no chunks at all — so a gold set written in one system's chunk ids can only
> ever score that system, and the comparison the metric exists for is between
> systems. The span space is a property of the **corpus**: markdown sections,
> `raw/sources/aurora-1-flight-report.md#appendix-a-mass-properties`. It is what
> the author wrote, it survives re-chunking, and it is writable in a JSONL
> fixture. `Retrieved.chunk_id` is still honoured — first, in fact — when it
> names a span the corpus has.
>
> A hit is otherwise resolved from its **snippet**, by word n-gram overlap
> rather than substring containment, because a snippet routinely straddles a
> section boundary and carries ellipses. That makes a document-level retriever
> measurable here, which is the whole point: a snippet *is* a retriever's claim
> about where in the document the evidence is. The bm25 baseline gained one line
> — FTS5's own `snippet()` — so that the system the comparison exists to beat
> appears as the floor rather than as "not measured".
>
> Gold is **derived, never typed**: `SpanIndex.covering` finds the section that
> literally contains the fact each question already asks for. A gold span
> written by hand beside a gold id is two copies of one claim and they drift.

### E47 — Declare the corpus-scope pedestal. Built.
`report.suspect_switches` / `tag_spread`, `cli._switches`, `--allow-switch`.

> Flags a tag whose spread across the compared systems is identical at every
> swept `k` (within 0.02) and larger than 0.5. Prints which systems sit on each
> side, and says to look at whether the questions name their own gold documents.
> Fails the suite, not the system, and exits non-zero unless declared — the same
> shape as `--allow-regression` and `--allow-saturation`, which the proposal did
> not think to give it.

### E48 — Conversational suites. Built, three arms rather than two.
`Question.history` / `resolved`, `System.retrieve_with_history`,
`ragharness conversation`, `fixtures/atlas/conversation` (35 turns).

> The proposal described a fixture and a capability. What it needed was a
> **command**, because two arms cannot be read against each other without a
> third: `bare` (the utterance as spoken), `history` (the turns handed over) and
> `resolved` (the oracle rewrite — not a system anyone ships, the ceiling the
> other two are read against). `resolved − bare` is the measurement.
>
> `compare` and `run` now **refuse** a conversational suite outright and name
> this command. Putting "and the second one?" to a retriever with no prior turn
> produces a table of near-zeros that looks like a retrieval finding and is an
> artifact of asking the wrong thing.

### E49 — Refusal questions and abstention metrics. Built.
`Row.abstained` / `unanswerable`, `Aggregate.abstention_answerable` /
`abstention_unanswerable`, `cli._gate_operating_point`,
`fixtures/atlas/refusal` (20 unanswerable + 20 matched answerable).

> Each unanswerable question is a **matched pair** with an answerable one: same
> sentence, same shape, one entity that exists and one that does not, so
> anything the two rates differ on is the entity. `forbid_strings` is exact
> rather than hopeful — asked the launch mass of a flight that never flew, any
> answer containing `kg` has invented a number, and hedging cannot satisfy it.
>
> One case the proposal did not anticipate: a configuration where **no gate can
> run**. llmwiki's lexical gate needs a second lane to fall back on, so
> `lexical` and `sources` never abstain at all. Reporting that as "not
> discriminating" would be a claim about a mechanism's quality when the finding
> is that the mechanism did not run, so it is reported as its own state.

### E50 — Budget as a swept dimension. Built twice, in one day.
`--budget`, `System.retrieve_within`, `cli._budget_sweep`,
`adapters.llmwiki.BUDGET_LADDER`.

> **This step was built against an assumption that stopped being true while it
> was being built**, and the sequence is worth keeping because it is the most
> instructive thing that happened here.
>
> The proposal gated E50 on build-plan A3 (§6: "the only step here gated on the
> implementation rather than the reverse"). It was built anyway, because the
> *axis* did not need A3 — a `--budget 40,100,400` sweep, a table of systems
> against budgets, `Unavailable` for a system with no `retrieve_within`, and
> `Delivery` marking an overrun degraded. What did need A3 was the cost model,
> and the adapter stood in for it with a declared ladder: 40 ms → `voice`,
> 100 ms → `balanced`, 400 ms → `deep`, stated as a **declaration, not a
> measurement**, in `params()`.
>
> Then A3 landed (llmwiki `c2ef88b`), and `c25385b` replaced the interim form:
> `retrieve_within` now passes `Budget.for_ms(budget_ms)` into `search`, and the
> pipeline drops the rungs it cannot afford — the network round trip first,
> local diffusion last — and names them on the response. **The ladder keeps the
> one job it is still the right place for**: naming which lane configuration
> each rung is asking about. What was declared out here and is now decided
> inside is whether the turn fit.
>
> No reported number moved. The difference is that an overrun is a decision the
> system made and reported, rather than one the harness inferred from outside
> afterwards — which is the same distinction `Delivery` draws everywhere else,
> arriving on the newest axis.
>
> Opening the index is still not charged to the budget: llmwiki amortises it
> behind a fingerprint cache, and the case where it *is* on the query path has
> its own command (E56).

### E51 — Per-stage latency. Built.
`Row.stage_ms`, `Aggregate.stage_ms`, `runner._stages`.

> Promoted from the adapter's `meta`, **completed with a residual**. The residual
> is the part of the wall clock the adapter did not account for, and it is what
> makes two named stages a decomposition rather than two numbers. E56 is where
> it earned itself: it attributed a 7.19× first-query cost to `load 77.1 ms`
> against a warm 1.1, plus `residual 29.2` against a warm 12.8 — the index
> rebuild and the structures derived from it, which are different stages with
> different fixes.

### E52 — Promote cost to a reported column. Built, one level deeper.
`Row.prompt_tokens` / `completion_tokens`, `Aggregate`, the render, **and
`llmwiki.query.Answer`**.

> The proposal's premise was wrong; see §11. `llmwiki.llm.Completion` has carried
> `input_tokens` / `output_tokens` since the client was written and `Answer`
> dropped them, so every caller downstream could report how long an answer took
> and not what it cost. Two lines in llmwiki, one in each of three adapters.

### E53 — `depth_protocol`. Built.
`runner.depth`, `ragharness depth`, `Question.area`, `Row.area`.

> Yields its own origin (step 0, before a single query has been observed),
> splits the curve by streamed against control areas, and reports the verdict as
> a **difference of differences** so a warming cache cannot read as learning.
>
> Two guards the proposal did not name, both found by running it. The held-out
> set is checked for **headroom in the streamed group specifically** — a control
> group at its own ceiling is the protocol working, a streamed group at its own
> ceiling means no rise could have been shown and the run measures nothing. And
> when the two groups start more than 0.05 apart, the report says their levels
> are not comparable and only the change in each is.

### E54 — The query-stream fixture. Built.
`fixtures/atlas/depth/` — 25 stream queries over 5 areas, 24 held-out questions
over 8, three areas the stream never touches.

> Areas are the eight subsystem templates rather than the four specialities:
> four is too few to hold three back, and a control set of one can be moved by a
> single question. The held-out set asks about the same *need* in three
> directions the stream never takes — the units, their suppliers, the flights
> that carried them — because the claim a write-back loop makes is not "the
> queries I was shown now work", which is a cache, but "the area those queries
> were about is better served".

### E55 — Ranking agreement between two systems. Built.
`ragharness agree --a --b [--expect-identical]`.

> Exposes a function that has existed since the harness was written and that
> only `null-check` ever called, on one system against itself. `--expect-identical`
> is what a port or a shadow read is accepted on; without it, divergence is
> reported and not failed, because for two genuinely different retrievers
> divergence is the point.

### E56 — Post-ingest query latency. Built.
`runner.invalidation`, `ragharness invalidation --touch --tolerance --document`.

> A **touch is an mtime bump and nothing else**. llmwiki fingerprints its corpus
> by path, size and mtime, so it is exactly the invalidation an edit causes, with
> none of an edit's provider cost and no change to the bytes every other
> measurement is taken over.
>
> It defaults to a document a query could actually **return**. The first path
> alphabetically is `purpose.md`, which llmwiki excludes from retrieval — a valid
> measurement of the cache and a poor stand-in for the case a product is accepted
> on. Choosing it measured 2.18×; choosing a source document measured 7.19×.

### E21 — Unknown-key rejection. Pulled in, as the proposal said it must.
`suite.FIELDS`, `Question.from_dict`.

> E46 named this as its dependency and was right to. `expect_spans` misspelled
> once loaded as a question with no passage gold, which is indistinguishable
> from a question that declares none — so the column would report document
> recall under a passage-recall heading and nothing could tell.

---

## 10. Acceptance: what the instruments reported on first run

Every number below is from one pass over `fixtures/atlas/*`, ragharness at this
commit, llmwiki `ef796c7`, query embeddings served from the corpus cache. **The
44-question atlas suite reproduces its banked `recall@k` exactly at every swept
`k` for all six systems** — every addition is additive or it would be a
re-baselining of four roadmaps' worth of series, and that is the first thing
checked.

### 10.1 E46 — passage recall disagrees with document recall completely

`fixtures/atlas/passage/suite`, 14 questions, five swept `k`:

| system | `recall@10` | `recall@span` | span MRR |
|---|---:|---:|---:|
| llmwiki/lexical | 0.00 | 0.00 | 0.00 |
| llmwiki/sources | **1.00** | **0.00** | 0.00 |
| llmwiki/hybrid | 0.00 | 0.00 | 0.00 |
| llmwiki/full | **1.00** | **0.00** | 0.00 |
| bm25 | 1.00 | 0.43 | 0.43 |
| dense | 1.00 | **1.00** | **1.00** |

**E46's falsifier did not fire, and the result is stronger than the defect that
motivated it.** The proposal's cheapest refutation was "passage recall ranks the
systems in the same order document recall does". It ranks them in a *different*
order: the shipped configuration returns the right document every time and
points at section 1 — the summary — never at the appendix that holds the answer,
while the chunk-ranking dense baseline points at the right section every time.

Every span on that table is `snippet`-sourced, not `declared`, and the report
says so under it. That is the honest reading and it is the actionable one:
llmwiki emits no passage identity at all today, and its snippet falls back to the
head of the document when the query phrase does not match. Build-plan F2
(`chunk_read`) now has a number to beat and a baseline that already beats it.

### 10.2 E47 — one tag is flagged, and only one

`SUSPECT_SWITCH intra-doc — spread 1.00 identical at k=1,2,3,5,10`, on all three
suites that carry the tag. `1.00: llmwiki/sources, llmwiki/full, bm25, dense` /
`0.00: llmwiki/lexical, llmwiki/hybrid`. No other tag on any shipped suite is
flagged, which is E47's own falsifier answered: `intra-doc` was the only switch.

### 10.3 E48 — the most informative result in the document

`fixtures/atlas/conversation/suite`, 35 follow-up turns, `llmwiki/full`, k=10:

| arm | `recall@k` | MRR | hit | vs bare | 95% CI |
|---|---:|---:|---:|---:|---|
| bare | 0.37 | 0.10 | 0.37 | — | |
| history | — | | | | *not measured* |
| resolved | **1.00** | 0.88 | 1.00 | **+0.63** | [+0.46, +0.77] |

`+22 / −0` flipped. **COSTLY**: the bare utterance loses 0.63 recall against the
oracle rewrite, and that gap is the ceiling on what a rewriting step could be
worth. §7 called this "the most valuable falsifier in the document" and predicted
it might retire the build plan's most confidently-argued voice step. It did the
opposite, decisively — which is the same thing a falsifier is for.

The `history` arm is **not measured**, not zero: llmwiki declares no
`retrieve_with_history`. Its `ask()` already accepts `history` and passes it to
the *generation* prompt while `search()` ignores it entirely — so llmwiki has
conversational generation and non-conversational retrieval, which is build-plan
D1 stated as a fact about the code rather than as a literature result.

### 10.4 E49 — the gate is discriminating, and two lanes have no gate at all

`fixtures/atlas/refusal/suite`, 20 unanswerable and 20 matched answerable:

| system | abstained, answerable | abstained, unanswerable | verdict |
|---|---:|---:|---|
| llmwiki/full | 15% | 50% | discriminating |
| llmwiki/hybrid | 50% | 95% | discriminating |
| llmwiki/lexical | 0% | 0% | *no gate ran* |
| llmwiki/sources | 0% | 0% | *no gate ran* |
| bm25, dense | 0% | 0% | *no gate ran* |

The gate the system's differentiator rests on now has an operating point rather
than a single firing rate. `full` is the shipped configuration and the shape is
the right one — it fires three times more often on questions the corpus cannot
answer than on questions it can — but it still stands down on 15% of answerable
questions, and `hybrid` on half of them.

### 10.5 E50, E51, E52 — the ladder, the stages, the cost

Budget sweep at k=3 (`—` is `Unavailable`, `!` is an overrun):

| system | 40 ms | 100 ms | 400 ms |
|---|---:|---:|---:|
| llmwiki/lexical | 0.76 | 0.76 | 0.76 |
| llmwiki/sources | 0.90 | 0.93 | 0.93 |
| llmwiki/hybrid | 0.71 | 0.71! | 0.71 |
| llmwiki/full | 0.92 | 0.92 | 0.92 |
| bm25, dense | — | — | — |

**The top rung buys nothing.** Nothing gains between 100 ms and 400 ms, and only
`sources` gains anything at all (+0.03). E50's falsifier — "quality is flat
across the swept budgets, so the knob is decoration" — is *half* fired: the
cheapest rung is nearly free of quality cost, which is the finding a voice path
wants, and the expensive rung is unjustified on this corpus.

> **Re-run after `c25385b` moved the budget inside llmwiki**, the table reads the
> same: `lexical` 0.76 at every rung with a worst turn of 16 ms, `sources` 0.93,
> `full` 0.92, and no rung overrunning. That the numbers did not move is the
> result — the interim ladder and the real cost model agree about this corpus,
> which is the only evidence available that the interim one was not flattering
> anything while it stood in.

Cost, `llmwiki/full` generating on atlas `intra-doc`: **3,893 prompt + 40
completion tokens per question**, against a retrieval p50 of 16 ms and a
generation p50 of 1,920 ms. E52's falsifier assumed these would track
`reasoning_tokens`; `reasoning_tokens` is 0 on this provider and these are not.

### 10.6 E53 and E54 — NULL, which is not FLAT

`fixtures/atlas/depth`, 25 queries in 5 slices, held-out at k=5:

```
step    queries  streamed  control     gap
0             0      0.61     0.78   -0.17
5            25      0.61     0.78   -0.17
```

> **NULL** — the system observed the whole stream and reported indexing none of
> it, so the curve above is flat by construction. This is today's baseline, not
> evidence that query traffic cannot help.

`LlmWikiMutable.observe()` writes the query log — build-plan A1's first tier and
the only tier that exists — and returns `indexed: 0`. That distinction is the
whole value of the run: a loop that did not help and a loop that does not exist
are different findings, and the protocol is now unable to report the second as
the first. Build-plan E1 has a baseline to beat, and it is this table.

### 10.7 E55 and E56 — agreement, and the query after an edit

`agree --a full --b sources`: **0/44 identical rankings**, recall 1.000 both
sides. Two different retrievers, so divergence is expected and is not failed;
what matters is that a storage port or a shadow read can now be asked the
question at all, with `--expect-identical`.

`invalidation --touch 1`, touching `raw/sources/aurora-1-flight-report.md`. Five
consecutive runs, because **the first query after an invalidation is a
single-shot measurement by construction** — a corpus can only be invalidated
once, so `--repeat` cannot apply to it and the spread has to come from repeating
the whole protocol:

| | ms |
|---|---|
| warm p50 | 9.9 – 10.5 |
| **first query after the touch** | **54.0 – 61.8** |
| second query after | ~19 |
| ratio | **5.27× – 6.00×** |

**Against a 2× tolerance: FAIL, on every run.** Attributed by E51's stage map to
`load 38 – 43` against a warm 0.7, plus `residual 14 – 23` against a warm 10 —
the index rebuild, and the lexical calibration and entity graph derived from it,
which are rebuilt lazily on the query path rather than at open. Build-plan C2 and
C3 now have the measurement they accept on, and it currently fails.

Two things this is *not*. It is not the 866× cliff the measurement scripts
report — that is a different corpus scale, and this is 60 ms in absolute terms.
And it is paid once: the second query is back to ~19 ms and the p50 after is back
to the warm figure.

### 10.8 The defect no unit test could have found

Running the whole eval against llmwiki HEAD after A2/A3 landed produced two
failed cells in the budget table:

```
llmwiki/full at 100ms: TypeError: install.<locals>.timed_embed_query()
                       takes 2 positional arguments but 3 were given
```

`remote.install` monkey-patches `llmwiki.embeddings.embed_query` to time and
optionally cache the one network call on the retrieval path. It was declared
`(text, config)`. A2 added a `timeout` to that function, and every budgeted turn
— the only turns that set a deadline — began raising.

**Nothing in either repository's unit tests could see it.** llmwiki's 198 tests
pass, because llmwiki calls its own function. ragharness's 142 pass, because the
patch is only exercised through an adapter and no unit test sets a deadline. The
only thing that fails is *the two systems run together, on the axis that was
added last* — which is what the eval is, and which is why running it end to end
after every change to either side is not optional.

The wrapper is signature-transparent now: `(text, config, *args, **kwargs)`,
forwarding everything and keying the cache on `text` and `config` alone, because
a `timeout` changes whether the call returns rather than what it returns. The
regression test asserts the forwarding through the installed wrapper rather than
a copy of it, since a test that rebuilds the shape it is checking cannot fail.

This is the *class* of defect a harness spanning two repositories has and a
single-repository one does not, and it has an obvious general form: **every
monkey-patch across a repository boundary is a signature that will drift.** There
is exactly one other in this harness — none — but the rule is cheap to keep.

---

## 11. Where the implementation contradicted the proposal

Five places, in descending order of how much they mattered.

**1. The passage id space cannot be `chunk_id`.** §2's D35 says
"`Retrieved.chunk_id` exists and is never scored against", which invited the
reading that scoring against it was the fix. It is not: a gold set in one
system's chunk ids can only score that system, and the metric exists to compare
systems. The span space had to be corpus-defined. `chunk_id` is honoured when it
names a span the corpus has, and a hit's *provenance* — `declared` against
`snippet` — is now reported beside the number, because they are different
strengths of claim.

**2. E52's premise was wrong.** The step says `prompt_tokens` and
`completion_tokens` "are already recorded into `Row.meta` (`runner.py:282`) and
never reported". They were recorded — **as zeros**. `llmwiki.query.Answer` never
carried the provider's usage, so the adapter had nothing to pass and the harness
was faithfully recording nothing. The step was one level deeper than described
and needed a change in llmwiki, not only in the harness. An "XS, two lines to
prove it" estimate that turns out to require a change in the system under test is
exactly the kind of error a coverage matrix filled in from the steps rather than
the code produces, and §7 predicted it in general terms.

**3. E46 says "atlas's 18 `intra-doc` questions"; `atlas/suite` has 8.** The
other 10 are in `atlas/growth`, which uses `canonical` ids over a corpus llmwiki
re-ingests — the section slugs do not survive that, so passage gold there would
be scoring against headings a language model has not written yet. The 8 were
re-golded in place; a 14-question `atlas/passage` suite was built for the
population, adding the six vendor reviews the shipped questions never touch. §7's
power caveat still applies and 14 is not much better than 8 — it is honest about
which it is.

**4. E53's rule would have made the command unrunnable.** "A system that cannot
take a query stream is `Unavailable`" is right in the abstract and produces
nothing: no shipped system declares `observe()`, so the command would refuse and
report a blank. The resolution is the distinction the step's own argument
implied without naming — a system can declare `observe()` and honestly report
that it **indexed nothing**, which is today's llmwiki, and the protocol then
records a baseline rather than either refusing or reporting a fake flat curve.
`NULL` and `FLAT` are now different verdicts.

**5. Adding a metric means teaching the existing guards which metric they
guard.** The proposal treated the new instruments as additive and they are, for
the numbers. They are not for the *refusals*: the headroom check refuses a
comparison whose leader is at the ceiling on `recall@k`, and on both new suites
that saturation is the design — a localisation suite gives every question its own
gold document by name, and a refusal suite's answerable half is a control chosen
to be easy. Both were refused on first run. `_recall_is_not_the_headline` states
the two cases; nothing in §3 anticipated needing it.

**6. "E50 cannot start before build-plan A3" was wrong, and usefully so.** §6
called it the one step gated on the implementation rather than the reverse. It
was not: the *axis* — a sweep, a table, `Unavailable`, `Delivery` — needed
nothing from A3, and only the cost model did. Building it against a declared
ladder meant that when A3 landed the same afternoon, replacing the ladder was a
twenty-line change to one method and **no reported number moved**. Had the step
waited, the harness would have had no way to check A3 on the day it arrived.
The general form: a measurement gated on an implementation is usually gated on
one *part* of it, and naming which part is what lets the rest be built first.

**And one thing §7 predicted correctly.** "A check that finds two defects on its
first pass has not usually found the last two." Building the eleven steps found
six more, five of which are above and none of which the coverage matrix could
have caught, because every one of them is a fact about code the matrix was not
reading. The sixth is not a design error at all — it is a `TypeError`, and §10.8
is about why no unit test could see it.

## 12. Closing the gaps the acceptance run left

§10 reported what the instruments found. This section reports what closing their
own gaps found, which turned out to be more interesting, because three of the
four gaps were hiding a conclusion rather than merely limiting one.

The gaps were named in this order of severity: **A2's residual** (a blown budget
could be detected and not attributed), **statistical power** (per-class blocks
with n between 8 and 14 and no statement of what that n supports), **the
cross-repo seam** (§10.8's `TypeError`, which no unit test in either repository
could see), and **generated passage gold** (correct by construction, which is not
the same as checked). Five steps, numbered on from E56.

### E57 — the adapter forwards the stages llmwiki already names

**The finding is the shape of the defect, not the fix.** `SearchResponse.stage_ms`
has carried `open`, `lexical`, `embed`, `vector`, `calibrate`, `fuse`, `diffuse`
and `materialize` since A2; `search()` takes a `sink` and reports an outcome —
`ok`, `expired`, `skipped`, `failed` — for every one of them. The adapter read
neither. It timed the index open and the provider round trip from the outside
and shipped `{"load", "remote"}`, and the runner dutifully computed a residual
against a wall clock. Nothing was missing. It was being dropped at the boundary
and replaced with something that looked like an answer.

Two things had to be got right to forward it.

**`remote` is not a stage.** The provider round trip happens *inside* llmwiki's
`embed` stage, so a map holding both counts it twice, and on a turn that is
almost entirely round trip the residual goes negative — where a `max(0.0, …)`
clamp had been quietly turning it into `0.00`, which reads as "fully attributed".
`_stages` now returns a note when the named stages exceed the turn, and the note
reaches the row. A decomposition that cannot be trusted has to say so; clamping
it to zero is the class of quiet wrong number this harness exists to refuse.

**A duration and an outcome are different facts.** A stage that took 0 ms because
it was fast and one that took 0 ms because the deadline dropped it are the same
number. The dropped one has no duration at all — llmwiki reports it straight to
the sink — so it appears *only* in the outcome map, and on a degraded turn that
is the whole finding.

**D40 — the budget ladder was never flat.** Four roadmaps have recorded
`llmwiki/full` scoring 0.92 at 40 ms, 0.92 at 100 ms and 0.92 at 400 ms, and read
the flat row the way §3's E50 predicted a flat row should be read: the rungs buy
nothing. The stage outcomes say otherwise.

```
  what each budget stopped the system doing — a rung's cost, in stages
  llmwiki/full at 40ms:  embed expired on 32/44, search expired on 32/44
  llmwiki/full at 100ms: embed expired on 32/44
```

At the cheapest rung the vector lane never ran on three quarters of the
questions, and recall was 0.92 anyway. That is not a ladder whose rungs are
decoration. It is a corpus on which the vector lane is not carrying the result,
and it points somewhere completely different: at what the lexical and graph lanes
are already doing, and at whether the round trip is worth its 547 ms at all.

**D41 — the overrun is somebody else's network.** Cold, `embed` is 547 ms of a
557 ms turn, p95 687. C2 and C3 were reported in §10 as failing at 5.3×–6.0× the
warm p50 with no attribution available. The attribution is now one line, and it
is not in the ranking: the ranking is 10 ms of it.

### E58 — what a question class can resolve, printed above the numbers it cannot

The harness has had a paired bootstrap since its first roadmap and prints it on
the total. The per-class blocks — by capability tag, by lexical overlap — never
carried one, and they are where every finding about *which questions* a system
wins has come from.

The bound is exact and derived from the test the harness already runs, rather
than from a normal approximation that would itself be unreliable at these n.
`sign_test(5, 0) = 0.062` and `sign_test(6, 0) = 0.031`: a two-sided sign test at
0.05 needs six discordant pairs before it can reject **whatever the effect size**.
So the smallest difference `n` paired questions can resolve is `6/n`, and below
six questions the answer is `None` — not a large number, because a large number
invites the reading this exists to prevent. The harness's own rule that a thing
not measured is never a number, applied to its own statistics.

**D42 — every per-class block ever printed here is underpowered.**

```
  system              single-hop   multi-hop    thematic   intra-doc
                   n          14          12           8           8
            resolves        0.43        0.50        0.75        0.75
  ! UNDERPOWERED_EXACT — every class in this block (n=8–14) resolves 0.43–0.75
    at best, and the widest spread among them is 0.25. The block describes how
    these questions fell; it does not rank the systems.
```

An eight-question tag cannot show a 0.25 difference. Not *did not* — cannot, at
any effect size, because the smallest significant split on eight paired questions
is six of them flipping the same way. §7's power caveat said this in prose and
§11.3 repeated it about the passage suite; a caveat in prose does not stop anyone
reading the column, and it does not update when the suite grows. The dashboard's
hand-written version of the same claim ("these n are 8 to 14") is now computed,
and its advice — "read a class where the system loses as a finding" — was the
opposite of what the bound says and has been replaced.

**This does not touch §10's two headline findings.** E48's coreference cost is
0.63 on 35 questions and E46's passage gap is 0.00 against 1.00; both are far
outside any interval a wider set would draw. What it touches is every *next*
finding, which will be smaller.

### E59 — the contract between the repositories, where a unit test can see it

§10.8 recorded the `TypeError` and why neither suite saw it. `tests/test_contract.py`
is the general form: the assumptions ragharness makes about llmwiki, executable,
against the **installed** llmwiki rather than a stub of it. A stub is a copy of
the assumption, and a copy cannot disagree with the original.

The existing regression test proved the wrapper forwards `*args, **kwargs` to a
recording double — and would still pass if llmwiki renamed `text` or added a
required parameter, because the double is written in the test file and agrees
with the wrapper by construction. The contract test reads llmwiki's own signature
and asks whether the patch can be called every way llmwiki can be. Reverting
`remote.py` to its pre-fix signature fails it with both signatures in the
message. Nine further tests pin `search()`'s parameters, `SearchResponse`'s
fields, `SearchResult`'s, the sink's triples, `Budget.for_ms`, the outcome
vocabulary, and `Answer`'s token counts.

Every one skips when llmwiki is not importable, because the harness is not bound
to llmwiki and a missing optional dependency must not fail its suite — which is
also why `scripts/check.sh` asserts importability before running them. A skip
here would make the whole check pass on the one thing it exists to check.

### E60 — one definition of "checked"

`scripts/check.sh`, run by `.github/workflows/check.yml`: both test suites, the
contract tests with skips treated as failure, a fixture rebuild that must
reproduce the committed suites, and an end-to-end `compare` on the two lanes that
need no provider. No network and no API key — a gate whose result depends on
someone else's endpoint fails for reasons the change did not cause.

**What it cannot do, stated rather than papered over.** The change that breaks
the contract is usually an *llmwiki* change, and a workflow in the harness's
repository is not triggered by one. Hence a daily schedule, which bounds the
delay rather than removing it. The real fix is a mirrored workflow in llmwiki
calling the same script.

### E61 — passage gold that is checked, not merely constructed

`SpanIndex.covering` returns *a* section containing the fact; gold needs *the*
section. A fact appearing in two sections has two right answers, the fixture
names whichever came first, and the passage column starts measuring document
order. `covering_all` plus `_unambiguous_span` makes the generator refuse at
build time. Nothing in the corpus triggers it today — all 22 spans are unique,
and the suites rebuild byte-identically — which is the point: the thing that
would trigger it is an ordinary edit to the filler prose that nobody would think
to re-check by hand.

This bounds the failure mode; it does not make the gold independent. The
generator still writes both the question and the fact, and §11.3's author-bias
argument applies here with no `verify_hostility` equivalent to bound it.

### What was not closed

**E3 stays blocked on E43.** No shipped system builds the write-back loop, so
`depth` reports NULL — the honest baseline, and the reason the protocol's ability
to detect a rise is demonstrated by a test double rather than by any system.

**The two research trees were not merged.** The gap list proposed a symlink. That
was wrong: `development/research` holds two subtrees `research/` does not, and
four shared files differ deliberately. They are not copies of one document, they
are two documents that overlap, and a symlink would have deleted the difference.
Left alone.
