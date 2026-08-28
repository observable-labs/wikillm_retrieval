# Roadmap: a harness that validates its own runs

Compiled 2026-08-27 against ragharness `ad83ae3` and llmwiki `93966fd`.
Evidence quality: **measured in-repo**, every number below reproduced from the
shipped fixture on this machine.

**Question:** [`../harness-v1.md`](../harness-v1.md) closed ten design gaps and
its implementation is faithful. Why did the first adversarial run of that harness
report numbers that did not mean what their labels said, and what closes the
class?

**Where the work lands:** this document lives in `wikillm_retrieval/research/`,
but every step below changes `space_brief/evaluation/ragharness`, a different
repository. §7 lists the one change that belongs to llmwiki instead.

> **Status, 2026-08-27 — implemented.** E1–E7 are built and covered by
> `tests/test_invariants.py`, where every test corresponds to a defect recorded
> below. What the closed harness then measured, and the retrieval rebuild it
> made possible, is in
> [`../../../future_work/retrieval-rebuild/`](../../../future_work/retrieval-rebuild/README.md).
> §10 records where implementation contradicted this document.

---

## 1. The failure, in one table

`ragharness compare --suite fixtures/suite --k 3`, verbatim:

```
  dense            unavailable: no vector index at .../vectors.db

  system             recall@k     MRR    hit    p50ms    p95ms
  ──────────────────────────────────────────────────────────
  llmwiki/lexical        0.59    0.59   0.64        3        5
  llmwiki/hybrid         0.59    0.59   0.64        4       47
  llmwiki/full           0.77    0.71   0.82        4        6
  bm25                   0.82    0.61   0.82        0        0
```

`lexical` and `hybrid` differ by exactly one variable
(`adapters/llmwiki.py:31–35`):

```python
"lexical": Lanes("lexical", sources=False, embeddings=False),
"hybrid":  Lanes("hybrid",  sources=False, embeddings=True),
```

They report identical recall, identical MRR, identical hit-rate. A reader draws
the obvious conclusion — *embeddings buy nothing* — and the obvious conclusion is
wrong. The corpus has no embedding model configured, so the dense lane never ran
and `hybrid` **is** `lexical`. The harness printed the two as distinct rows and
said nothing.

Note the row directly above them. The `dense` adapter, asked for the same
missing index, refused: `Unavailable: no vector index`. The harness already
contains the correct behaviour. It simply does not apply it where the
configuration degrades silently rather than failing outright.

---

## 2. The class of defect

`harness-v1.md` §1 lists ten gaps found in its predecessor. Every one of them
asks *what should be measured*. None asks whether the measurement ran.

That is visible in what the harness validates today:

| Checked | Where | Not checked |
|---|---|---|
| gold ids still resolve in the corpus | `suite.stale_targets` (`suite.py:170`) | did the requested lanes actually execute |
| the harness is deterministic | `null-check` | is `k` capable of separating anything |
| the suite's tags and counts | `validate` | does the run's label describe the run |

> **The invariant to add.** A run may only report a number under a configuration
> label if the configuration was delivered. Anything else is recorded as
> unavailable, not scored.

Everything in §4 is that sentence, made executable.

---

## 3. The defects

### D1 — A requested lane degrades to silence
`adapters/llmwiki.py:72` passes `settings.embedding` through whenever
`lanes.embeddings` is set. `retrieval/pipeline.py:74` then requires
`enabled and model`; the fixture's model is `""`, so the vector lane is skipped
with no exception, no note on the row, and no mark on the run. The snapshot
faithfully records `"embedding_model": ""` — the fact is captured and nothing
acts on it.

### D2 — The default `k` makes the metric a constant
The fixture holds 9 pages and 3 sources. The default is `k=20`. Every document
is therefore always inside the window:

| system | recall@k, k=20 | recall@k, k=3 |
|---|---|---|
| bm25 | 1.00 | 0.82 |
| llmwiki/full | 1.00 | 0.77 |

At its own default, on its own fixture, the harness cannot separate any two
systems on any tag. `recall@k` is not measuring; it is returning 1.0. The only
`k` that discriminates on this corpus is small enough to sit inside llmwiki's
graph-quota pathology (§7), so the one usable operating point is also the least
representative one.

### D3 — `validate` reports an unusable system and exits 0
`cli.py:134` wraps corpus checking in `except Exception` and prints
`(corpus not checked: …)`. Confirmed: `validate --system dense` against a corpus
with no vector index prints the message and returns **exit 0**. The command whose
job is to establish that a run can be trusted passes when the system under test
cannot run at all.

### D4 — Runs carry no delivered-configuration record
`Run` (`runner.py:42–69`) stores `config` — the *requested* lane name — and
a `snapshot`. There is no field recording what the adapter actually did, so
neither `compare` nor a later diff of two run artifacts can detect D1. The run
JSON is self-consistent and misleading.

### D5 — The generation side is unmeasured
`Generated` (`harness-v1.md` §3.2) carries `prompt_tokens` and
`completion_tokens` and no reasoning-token count or answer wall clock.
`evaluation/README.md` §1 states the primary artifact is "a quality-versus-latency
curve", while `latency-knobs.md` §1 states query-time LLM calls dominate that
latency by orders of magnitude. The curve currently excludes its dominant term,
which is why the `voice`/`balanced`/`deep`/`research` ladder in
[`../../target-architecture/README.md`](../../target-architecture/README.md) §6
cannot be validated by the harness that is supposed to gate it.

---

## 4. The roadmap

Same contract as [`../../target-architecture/build-plan.md`](../../target-architecture/build-plan.md):
goal, touches, design, and a criterion you can check. Every step is hours to
half a day; none depends on a later one.

### E1 — Fail loudly when a requested lane cannot run
**Goal:** D1 becomes impossible for the llmwiki adapter.
**Touches:** `adapters/llmwiki.py`

Copy the pattern the `dense` adapter already uses (`adapters/dense.py:33`): raise
`Unavailable` in `__init__` when `lanes.embeddings` is requested and
`settings.embedding.model` is empty, or when the store path is absent.

**Done when:** `compare` against a corpus with no vector index prints
`llmwiki/hybrid  unavailable: …` instead of a second copy of the `lexical` row.

### E2 — The delivered-configuration invariant
**Goal:** generalize E1 so the next adapter cannot reintroduce it.
**Touches:** `runner.py`, `types.py`, `report.py`

Add an optional `delivered() -> dict` to the `System` protocol, detected by
presence like every other optional method. Record it on `Run` beside
`config`. The runner compares requested against delivered and marks the run
**degraded** on any mismatch; `report` prints degraded runs with a `!` and
excludes them from a `compare` table rather than ranking them.

**Done when:** a deliberately mis-declared adapter — `embeddings=True`, dense
lane stubbed out — is reported degraded rather than scored, and the assertion is
covered by a test.

### E3 — `k` must be able to discriminate
**Goal:** D2 becomes visible at the moment it occurs.
**Touches:** `runner.py`, `cli.py`, `report.py`

The snapshot already carries the corpus document count. When `k` is greater than
or equal to it, `recall@k` is degenerate: warn on every run, print a
`saturated` marker on the results table, and refuse in `compare`, whose whole
purpose is ranking systems against each other.

Have the suite declare the `k` it was designed for, and make that the default in
preference to the constant 20.

**Done when:** `run --k 20` against the 12-document fixture emits a saturation
warning, `compare --k 20` refuses, and `compare` with no `--k` uses the suite's
declared value.

### E4 — Non-zero exits for unusable configurations
**Goal:** D3 — `validate` stops passing runs that cannot happen.
**Touches:** `cli.py:113–136`

Catch `Unavailable` specifically rather than bare `Exception`, report it as a
failure, and return 1. Keep the permissive path for genuinely optional
capabilities, which is what the broad catch was reaching for.

**Done when:** `validate --system dense` with no vector index exits 1; `validate
--system llmwiki` on a healthy corpus still exits 0.

### E5 — A fixture with headroom
**Goal:** a corpus on which the default `k` measures something.
**Touches:** `fixtures/build_fixture.py`

Today: 9 pages, 3 markdown sources of under 1 KB each, 12 questions
(single-hop 4, multi-hop 3, thematic 2, intra-doc 2, drift 1). It cannot exercise
`k` separation, and — because every source is markdown — it cannot exercise
`extract_text` cost at all, which is the evidence base for build-plan step 4's
claim to be "the largest single latency win available."

Grow it until `recall@k` at the declared `k` is strictly between 0 and 1 for at
least one system on at least one tag, and include at least one PDF source.

**Done when:** no tag reads 1.00 for every system at the default `k`, and step 4
of the build plan is measurable rather than assumed.

### E6 — Generation-side cost
**Goal:** D5 — the latency curve includes its dominant term.
**Touches:** `types.py`, `adapters/llmwiki.py`, `report.py`

Add `reasoning_tokens` and `answer_seconds` to `Generated`. Have the llmwiki
adapter report the effort level `reasoning.resolve()` actually produced, not the
one requested — `reasoning.py:_CLAMP` silently promotes `off` to `low` on the
gpt-5, o-series and gemini-3 families, so the `voice` profile's declared effort
is not the effort that runs.

**Done when:** `run --generate` reports p50/p95 answer latency and reasoning
tokens per effort level, and the profile table in target-architecture §6 can be
checked against measurements instead of assumed.

### E7 — Pin the found defects as suite regressions
**Goal:** the bugs this exercise surfaced cannot return unnoticed.
**Touches:** `fixtures/suite/`, a new invariant in `runner.py`

Two additions:

1. Keep `mh-003` ("Which mission flies a Helios thruster?") permanently. At k=3
   llmwiki's keyword lane ranks `wiki/missions/aurora-1.md` third — inside the
   window — and graph expansion evicts it in favour of
   `wiki/concepts/station-keeping.md`, a neighbour of the answer rather than the
   answer. It is the cheapest known reproduction of §7's displacement bug.
2. Add a suite-level invariant: **a system may not return fewer gold documents at
   k than a strict subset of its own lanes returns.** `full` scoring below
   `lexical` on any question is a displacement bug by definition, and it
   generalizes past the specific implementation being replaced.

**Done when:** both are in the suite, both fail against `93966fd`, and both pass
once build-plan step 8 lands.

---

## 4.1 What each step became

| Step | Built as | Covered by |
|---|---|---|
| E1 | `LlmWiki._require_requested_lanes` raises the shared `types.Unavailable` | `test_a_lane_that_cannot_run_refuses_instead_of_degrading` |
| E2 | `types.Delivery`, optional `System.delivered()`, `Run.degraded`, non-zero exit | `test_a_degraded_run_is_marked_not_silently_scored` |
| E3 | `Run.saturated`, `suite.json` declares `k`, `compare` refuses | `test_saturation_is_detected_at_the_default_k`, `test_the_suite_declares_its_own_k` |
| E4 | `cmd_validate` catches `Unavailable` and returns 1 | exercised by `validate --system dense` |
| E5 | `fixtures/build_atlas.py` (78 documents, 44 questions) and `fixtures/build_hotpot.py` (1,991 paragraphs, 200 questions) | both suites ship |
| E6 | `Generated.reasoning_tokens/seconds/effort`, `Row.answer_ms`, a generation line in the report | `run --generate` |
| E7 | `report.lane_monotonicity`, called from `compare` | `test_invariants.py` in full |

E7's second invariant was generalised while being built. The roadmap proposed
"a system may not return fewer gold documents at k than a strict subset of its
own lanes returns", scoped to llmwiki's lane names. It ships comparing
**delivered configurations** instead, so it works for any adapter that reports
one and cannot be fooled by a lane's name — and it immediately earned its place
by catching a defect the roadmap had not predicted: a vector index covering only
part of the corpus, which made `full` score below `lexical`.

---

## 5. Order

```
E1 ──> E2 ──> E7        (trust the labels, then pin what they revealed)
E3 ──> E5               (make k honest, then give it room to work)
E4                      (independent, one function)
E6                      (independent; gates target-architecture §6)
```

E1, E3 and E4 are the ones that make the existing numbers trustworthy, and
between them they are perhaps a day. **Nothing measured before E1 and E3 should
be quoted**, including the four rows in §1.

---

## 6. What would falsify this

- **If `delivered()` proves unimplementable for the external subprocess adapter**,
  E2's invariant weakens to a warning for out-of-process systems and the
  in-process guarantee is the only one. Check `adapters/external.py` before
  committing to the protocol change.
- **If the suite-declared `k` in E3 turns out to be corpus-dependent rather than
  suite-dependent**, it belongs on the corpus snapshot, not the suite file.
- **If growing the fixture (E5) makes the offline demo slow enough that people
  stop running it**, ship two fixtures — a fast smoke corpus and a discriminating
  one — and let `compare` require the latter.
- **If E7's second invariant fires on legitimate behaviour** — a lane that
  correctly outranks a superset because sources are damped by
  `SOURCE_SCORE_FACTOR` — it is too strong and should be scoped to wiki-only
  comparisons.

---

## 7. What the harness found that is not the harness's to fix

Recorded here because these are the first real findings the harness produced,
and they belong to llmwiki:

| Finding | Location | Owner |
|---|---|---|
| Graph expansion **displaces** rather than augments: `base` is truncated to `limit - len(candidates)`, so a reserved graph slot always evicts the tail of the ranked list | `retrieval/graph.py:163` | build-plan step 8 |
| Graph neighbours are chosen query-blind by `1.0 / (rank + 1)` | `retrieval/graph.py:182` | build-plan step 8 |
| The graph lane is inert once the window exceeds the corpus — `seeds` becomes the whole result list, so every neighbour is already a seed and is skipped | `retrieval/graph.py:173` | build-plan step 8 |
| Keyword scoring has no IDF and no stopword list: `tokenize_query` returns `which` as a scored token, and title bonuses dominate | `retrieval/keyword.py:155` | build-plan step 5 |
| `_mode` returns `"hybrid"` whenever `graph_hits > 0`, regardless of whether the vector lane ran — which is how a keyword-only run came to be labelled hybrid | `retrieval/pipeline.py:177` | small fix, no step owns it |

The last one is a two-line change and is the llmwiki-side twin of D1: both are a
label asserting more than the run delivered.

---

## 8. Corrections to `harness-v1.md`

**8.1 §1's gap table is missing an eleventh entry.** All ten concern what to
measure. The failure that actually occurred was a measurement that did not run
and was scored anyway. Gap 11: *nothing validates the run against its own
configuration.*

**8.2 §3.2's `Generated` is incomplete for the stated purpose.** It cannot
support the quality-versus-latency curve `evaluation/README.md` §1 calls the
primary artifact, because it omits reasoning tokens and answer latency. E6.

**8.3 The claim that the harness is "retrieval-only by default, zero new
dependencies, ~450 lines" understates it.** The shipped implementation is ~1,834
lines across twelve modules including `growth`, `null-check` and `compare`
subcommands that the design did not specify. This is a correction in the harness's
favour, and worth recording so a later reader does not mistake the design for the
artifact.

**8.4 `build-plan.md` step 3 is stale.** It specifies `src/llmwiki/eval/` and
says "Nothing here is started." The harness exists, is committed as `ad83ae3`,
and was deliberately built as a standalone package in a sibling repository —
which is what `harness-v1.md` §3.1 argued for. Step 3 should read *done,
relocated*, and the build plan's dependency graph should point at ragharness.

---

## 9. Not read / thin ice

- `adapters/external.py` (146 lines) was listed but not read; E2's protocol
  change assumes a subprocess adapter can report delivered configuration over
  JSON, and that assumption is unverified.
- `growth` was confirmed to exist and has still not been exercised. It is the
  only check of the O(document) claim in
  [`../../incremental-updates.md`](../../incremental-updates.md), and nobody has
  run it. `null-check` has since been run against the atlas suite and passes.
- Every number in §1 and §3 comes from a 12-question fixture over 12 documents
  with no dense lane. The *mechanisms* are established; no effect size here
  survives contact with a real corpus, and none should be quoted as one.


## 10. Where implementation contradicted this document

**10.1 §6's first falsification did not fire.** `adapters/external.py` declares
capabilities at handshake, so `delivered` is one more declared capability and
needs no weakening of the invariant for out-of-process systems.

**10.2 E5's acceptance criterion was too weak.** "No tag reads 1.00 for every
system at the default k" was met by the atlas fixture at k=10 and was still not
enough: llmwiki reached 1.00 on every tag, and only `k=5` separated the
configurations. Headroom is not a property of the corpus alone — it is a property
of the corpus, the questions and the system together, and the only honest check
is that the number moves when a mechanism is removed. That is what the ablation
lanes are for.

**10.3 §7's fifth finding was one line, not two.** `_mode` now derives from a
`LanesRun` record of what executed, rather than from hit counts. The record is
what the adapter's `delivered()` reads, so the llmwiki-side and harness-side
halves of D1 are closed by the same mechanism rather than separately.

**10.4 §7 understated the llmwiki findings.** Five were listed; the rebuilt
harness found three more, of which two were invisible until E1 and E3 landed: a
vector index covering only part of the corpus (which fusion converts into a
penalty on everything it cannot rank), single-character query tokens discarded
including the digits that distinguish `Aurora-1` from `Aurora-2`, and
`relevance()` weights being used as diffusion weights without ever having been
tuned as such. All three are in
[`../../../future_work/retrieval-rebuild/`](../../../future_work/retrieval-rebuild/README.md).

---
