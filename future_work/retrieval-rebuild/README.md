# Rebuilding retrieval, and the harness that could not see it

Assessed and rebuilt 2026-08-27, from llmwiki `93966fd` and ragharness `ad83ae3`.
Evidence quality: **measured in-repo**, every number below reproduced on this
machine with the commands given. Effect sizes come from 44 questions over 78
documents and 200 questions over 1,991 — small enough that direction is the
claim and magnitude is not.

**The question:** a plain BM25 baseline was outscoring the retrieval pipeline
that was supposed to contain it. Was that a real result about the pipeline, an
artifact of the harness, or both?

**The answer:** both, and they were hiding each other. The harness was scoring
configurations that never ran, at a `k` where the metric was a constant. Behind
that, the pipeline had five defects, and one of them — graph expansion — was
implementing the exact shape the literature measures as *worse than having no
graph at all*.

Companion documents:
[`../../research/target-architecture/build-plan.md`](../../research/target-architecture/build-plan.md)
(what to build), and
[`../../research/evaluation/roadmaps/harness-self-validation.md`](../../research/evaluation/roadmaps/harness-self-validation.md)
(the harness defects, D1–D5, and the roadmap E1–E7 that closed them).

---

## 1. Where it started

```
$ ragharness compare --suite fixtures/suite --k 3      # ragharness ad83ae3

  system             recall@k     MRR    hit
  llmwiki/lexical        0.59    0.59   0.64
  llmwiki/hybrid         0.59    0.59   0.64
  llmwiki/full           0.77    0.71   0.82
  bm25                   0.82    0.61   0.82
```

Three things are wrong with this table and only one of them is about retrieval.

`lexical` and `hybrid` differ by exactly one variable — embeddings off, on — and
report identical numbers, because the fixture had no embedding model and the
vector lane was skipped in silence. At the harness's own default of `k=20` every
row reads 1.00, because the fixture is 12 documents and every document is always
in the window. And `bm25` beating `llmwiki/full` was real.

---

## 2. The five defects in retrieval

Each was confirmed by reading the code and reproducing the behaviour, not
inferred from the scores.

**Line references in this section are against `93966fd`, before the rebuild.**
Two of the functions named no longer exist and one has moved; chasing the
numbers against the current tree will not find them. That is the point of
recording the commit.

### 2.1 The lexical lane counted presence, not importance

`_token_match_score` (`retrieval/keyword.py:155`) counted how many query tokens
appeared *anywhere* in a document, by substring:

```python
return sum(1 for token in tokens if token in lowered)
```

No term frequency, no inverse document frequency, no length normalization, and
`art` matched `cartesian`. A term in every document and a term in two documents
weighted the same. That is the whole reason a plain BM25 baseline could beat the
system: the baseline had a real ranking function and the system did not.

### 2.2 Graph expansion displaced results instead of adding them

`blend_graph_results` (`retrieval/graph.py:163`) reserved 15–30% of the result
window for graph neighbours and took those slots from the tail:

```python
base = [r for r in ranked if r.path not in selected][: max(0, limit - len(candidates))]
```

The reservation was unconditional — it fired whether or not the neighbours were
any good — and neighbours were scored `1.0 / (rank + 1)`, which does not mention
the query. SPRIG measures this exact shape (graph scores blended into the fused
list) at 0.782 on HotpotQA against 0.851 for having *no graph at all*. The quota
parameter cannot fix that, because the quota is not what is wrong.

Worse, the lane went inert exactly when the window was widest: `seeds` is the
whole result list, so once the window covered the corpus every neighbour was
already a seed and was skipped.

### 2.3 The vector lane covered part of the corpus, and fusion assumed it covered all

`index_documents` (`embeddings.py:296`) filtered `document.kind == "wiki"`, so
raw sources were never embedded. This is not a neutral gap. RRF adds `1/(k+rank)`
for every document a lane ranked, so a lane covering half the corpus pushes the
other half down by exactly the amount it pushes its own half up. Switching the
vector lane on dropped recall@5 from **1.00 to 0.79** — the raw sources, which no
vector could rank, fell out of the window entirely.

### 2.4 Single-character query tokens were discarded, including digits

`tokenize_query` dropped every token of length 1. Separators split `Aurora-1`
into `aurora` and `1`, so the digit — the only thing distinguishing `Aurora-1`
from `Aurora-2` — never reached the index. On a corpus of numbered missions that
is not an edge case. Worth 0.12 MRR.

### 2.5 `_mode` asserted more than the run delivered

`_mode` (`retrieval/pipeline.py:177`) returned `"hybrid"` whenever
`graph_hits > 0`, regardless of whether the vector lane ran. A keyword-only query
that picked up one graph neighbour reported itself as hybrid — the llmwiki-side
twin of the harness's own D1.

---

## 3. What was rebuilt

| Build-plan step | Status | Where |
|---|---|---|
| 4 — L1 text cache | **done, in process** | `retrieval/index.py` |
| 5 — L2 lexical via FTS5 | **done** | `retrieval/lexical.py` |
| 6 — L5 persisted links | **done, in process** | `retrieval/index.py` |
| 7 — L4 entity layer | **done** | `retrieval/entities.py` |
| 8 — S2 seeded PPR | **done**; `blend_graph_results` retired | `retrieval/ppr.py` |
| 10 — source chunks | **half done** — sources are embedded; two-tier freshness is not | `embeddings.py`, `cli.py` |
| 2 — one database | **open** — the caches are in process, not on disk | |

**L2 (`lexical.py`).** SQLite FTS5 with the stdlib `bm25()`, columns
title/headings/body weighted 10/5/1. The structural bonuses the old scorer added
ad hoc — filename +200, phrase-in-title +50 — survive as column weights, which is
where a signal like that belongs: a title match is worth more *per occurrence*,
not worth a flat constant regardless of how well the rest of the document
matches. CJK queries keep the substring lane, because `unicode61` cannot segment
Chinese and the bigram expansion is genuinely better there.

`SOURCE_SCORE_FACTOR = 0.6` did **not** survive, and that was measured rather
than assumed. It existed because the substring scorer had no length
normalization, so a short raw source repeating the query terms beat the compiled
page. BM25 produces the same ordering unaided — the wiki-outranks-source case
still holds without it — and keeping it cost 0.43 MRR on `intra-doc` questions,
whose gold *is* a raw source.

**L4 (`entities.py`).** The entity dictionary already existed: `build_graph`
assembles aliases from every page's path, stem and title, curated by the ingest
model and strictly more precise than the spaCy NER that LinearRAG and SPRIG use
for the same purpose. Counting alias occurrences per document gives a bipartite
entity↔document graph at zero LLM cost, weighted by SPRIG's
`tf · log((N+1)/(df+1)) + 1`, with hub pruning and per-document sparsification.

The graph is **bipartite, and that is not a detail**. Collapsing entity nodes
onto the pages that name them was tried first and is wrong: it makes a page
compete with the documents that mention it for the same diffused mass, so a
source quoting a page outranked the page. Routing mass doc → entity → doc keeps
the entity a conduit rather than a candidate.

**S2 (`ppr.py`).** Personalized PageRank seeded from the fused list, diffusing
over the union of the curated link graph (weighted by `relevance()`, reachable
from `search()` for the first time) and the mention graph. Ranking is by the PPR
score. Two properties make it safe to leave on: an empty graph returns exactly
the fused ranking, and a document retrieval already found is never dropped to
make room for a neighbour.

Three departures from a literal reading of SPRIG, each forced by a measured
failure:

- **Seed mass is `1/rank`, not the fused score.** RRF scores are `1/(60+rank)`,
  which differ by 2% between rank 1 and rank 2. Used as a restart distribution
  they give PPR essentially no prior, and a marginally better-connected document
  overturned a lexical margin of five orders of magnitude.
- **Every node carries a self-loop.** Without one, how much of its own mass a
  document keeps is decided by its degree: a page with one strong link handed 85%
  of its evidence to its neighbour each iteration and sank below documents that
  happened to be unconnected.
- **Iterate to a tolerance, not for five steps.** Power iteration converges at
  rate `1 - alpha`, so five iterations leave ~44% of the initial error and a
  two-node component still oscillates between odd and even steps. On graphs this
  size convergence costs microseconds and removes a parameter that was silently
  deciding results.

---

## 4. What it measures

### 4.1 A controlled corpus with known bridges

`fixtures/atlas` — 78 documents, 44 questions, generated from one world model so
gold ids are correct by construction. Multi-hop questions name a mission; the
answer is on the subsystem page, which the question never names.

```
$ ragharness compare --suite fixtures/atlas/suite --k 5 \
    --lanes sources,sources/no-graph,sources/no-entities,sources/no-links,sources/presence

  system                        recall@k     MRR    hit
  llmwiki/sources                   1.00    0.89   1.00   the shipped configuration
  llmwiki/sources/no-graph          0.90    0.90   0.95   S1 only — BM25 + RRF
  llmwiki/sources/no-entities       1.00    0.86   1.00   curated links only
  llmwiki/sources/no-links          0.90    0.90   0.95   mention edges only
  llmwiki/sources/presence          0.87    0.85   1.00   the old substring scorer, no graph
  bm25                              0.90    0.87   0.95   baseline
```

A lane name states its corpus scope *and* the mechanism removed, because they
are independent. That was learned late: the ablations were originally named
`no-graph` and so on, all quietly requesting raw sources, and on a corpus with
none — HotpotQA — every one of them was a degraded run wearing a clean label.
The delivery invariant caught it, which is the first time one of these checks
found something the person who wrote it had not already suspected.

Reading it: the lexical rewrite is worth **+0.05 MRR** (`presence` 0.85 →
`no-graph` 0.90) and the graph lane is worth **+0.10 recall** (`no-graph` 0.90 →
`sources` 1.00). Per tag, recall/MRR:

| | single-hop | multi-hop | thematic | intra-doc |
|---|---|---|---|---|
| `presence` (old scorer) | 1.00 / 0.73 | **0.54** / 1.00 | 1.00 / 0.69 | 1.00 / 1.00 |
| `bm25` | 1.00 / 0.93 | 0.83 / 1.00 | 0.75 / 0.43 | 1.00 / 1.00 |
| `no-graph` | 1.00 / 1.00 | 0.83 / 1.00 | 0.75 / 0.48 | 1.00 / 1.00 |
| `sources` | 1.00 / 0.96 | **1.00** / 1.00 | **1.00** / 0.57 | 1.00 / 0.94 |

Three things fall out, and one of them is a cost.

The old scorer's `multi-hop` recall of 0.54 against the baseline's 0.83 is the
clearest single statement of the original problem: on the query class the graph
lane is supposed to serve, the pipeline was retrieving barely two thirds of what
plain BM25 retrieved, and the graph lane it had was not making that up.

The graph lane's recall gain lands on `multi-hop` (0.83 → 1.00) and `thematic`
(0.75 → 1.00). The first is where it was expected. The second is not something
the design predicted, and the honest reading is that a "what does the corpus say
about X" question is a bridge question wearing different clothes — the pages that
answer it are linked to each other and to the same entities.

**And it costs rank position on the two tags it does not help:** `single-hop` MRR
1.00 → 0.96, `intra-doc` 1.00 → 0.94. Diffusion moves a correct top-1 to second
place on two of the 22 questions in those tags. That is the trade graph expansion
is known for, it is small, and it is the reason `no-graph` still ships as an
addressable lane rather than being deleted.

Whether the gap against the baseline is a result rather than noise, paired over
the same 44 questions:

```
  paired vs bm25                Δrecall            95% CI     flipped
  llmwiki/sources                 +0.10    [+0.02, +0.18]     +2 / -0
  llmwiki/sources/no-graph        +0.00    [+0.00, +0.00]     +0 / -0   (spans zero)
  llmwiki/sources/presence        -0.04    [-0.12, +0.06]     +2 / -0   (spans zero)
```

The middle row is the one worth dwelling on, because it is the honest form of
the headline. **llmwiki's lexical lane retrieves exactly what the BM25 baseline
retrieves** — the same documents on every question, differing only in their order
(MRR 0.90 against 0.87). That is what it should do: both are FTS5 `bm25()` over
the same corpus, and a lexical lane that beat a lexical baseline on recall would
mean the baseline was built wrong. The +0.10 comes from the graph lane and from
nothing else, which is a much narrower and more checkable claim than "the system
beats BM25".

### 4.2 A public benchmark with labelled gold

`fixtures/hotpot` — 200 HotpotQA questions over the pooled distractor corpus,
1,991 paragraphs. **No wikilinks at all**, so the curated graph is empty by
construction and the entity layer is the entire graph. On the hand-written
fixture the two are confounded; this is what separates them.

```
$ ragharness compare --suite fixtures/hotpot/suite \
    --lanes lexical,lexical/no-graph,lexical/no-entities,lexical/no-links,lexical/presence,hybrid

  system                        recall@10    MRR    hit   p50ms
  llmwiki/lexical                    0.96    0.88   1.00      15
  llmwiki/lexical/no-graph           0.90    0.89   1.00      15
  llmwiki/lexical/no-entities        0.90    0.89   1.00      16   !
  llmwiki/lexical/no-links           0.96    0.88   1.00      15
  llmwiki/lexical/presence           0.66    0.74   0.91      68
  llmwiki/hybrid                     0.99    0.93   1.00     581
  bm25                               0.90    0.87   0.99       4
  dense                              0.99    0.91   1.00     549

  paired vs bm25                  Δrecall            95% CI     flipped
  llmwiki/lexical                   +0.07    [+0.04, +0.09]     +1 / -0
  llmwiki/lexical/no-graph          +0.00    [-0.01, +0.01]     +1 / -0   (spans zero)
  llmwiki/lexical/no-links          +0.07    [+0.04, +0.09]     +1 / -0
  llmwiki/lexical/presence          -0.24    [-0.28, -0.19]    +1 / -19
  llmwiki/hybrid                    +0.09    [+0.06, +0.12]     +1 / -0
  dense                             +0.09    [+0.06, +0.12]     +1 / -0

  ! llmwiki/lexical/no-entities: did not deliver graph
```

Five readings, and the last two are the ones that complicate the story.

**The original defect, stated as starkly as it can be.** The old substring scorer
reaches 0.66 against the baseline's 0.90 — a paired −0.24, CI [−0.28, −0.19],
nineteen questions losing a gold document and one gaining. A BM25 baseline was
not beating this pipeline by a nose.

**The entity layer is the whole graph here.** `no-links` equals the full lane
(0.96) and `no-graph` equals the baseline (0.90). On a corpus with no wikilinks,
every bridge the graph finds is one the mention layer built.

**The delivery invariant earned its place, unprompted.** `no-entities` is flagged
because on this corpus, removing mention edges leaves the graph lane with no
edges at all — it was *requested* and did not run. Its 0.90 is therefore not
evidence about curated links; it is the no-graph number under a different name,
and without the flag it would have been read as the former.

**The dense lane is very strong on Wikipedia prose, and says so.** `dense` alone
reaches 0.99 recall — as high as the full hybrid. Everything the graph lane adds
over BM25 on this corpus, embeddings also find. That is the opposite of the atlas
result and it is the honest one: atlas is templated text where twenty pages are
near-identical in embedding space, and its dense numbers should be discounted
accordingly.

**Which leaves the trade the architecture is actually for.** Hybrid is the best
configuration on both metrics (0.99 / 0.93, against dense's 0.99 / 0.91), and it
costs 581 ms per query against 15 ms for BM25 + graph — because every query is a
network round trip to an embedding endpoint. The lexical + graph lane gets 0.96
recall in 15 ms with no API call at all. That is a quality-versus-latency curve
rather than a ranking, which is what
[`../../research/evaluation/README.md`](../../research/evaluation/README.md) §1
says the primary artifact has to be, and it is the first time this system has had
one.

The paired rows have the same shape as on atlas and the same reading: llmwiki's
lexical lane ties the BM25 baseline exactly, and the graph lane is the whole of
the difference.

**RRF alone 0.90 → seeded PPR 0.96**, with no embeddings in either. SPRIG's
published direction on this dataset is 0.851 → 0.867, and build-plan step 8 sets
exactly this as its acceptance test — *if you do not reproduce the direction of
that gap, the bug is in your PPR*. The direction reproduces, with a wider gap.
The absolute numbers are not comparable: their corpus is the full HotpotQA
collection, this one is a pooled distractor set of 1,991 paragraphs.

### 4.3 Latency

Retrieval on 1,991 documents went from **136 ms to 16 ms** per query, and none of
it came from the ranking:

| | ms/query |
|---|---|
| before | 136 |
| `corpus_fingerprint` off `pathlib` onto `os.scandir` | 40 |
| row-normalizing the graph once per corpus instead of per query | 24 |
| caching the path→document map | 16 |

The first was 70% of the query: `Project.wiki_pages()` returns `Path` objects and
filters with `relative_to`, which resolves symlinks per file. More than
retrieval, fusion and diffusion combined.

Index construction — read, parse, FTS5 insert, entity scan — is 20 s for 1,991
documents and is cached against a fingerprint of the corpus (path, size, mtime
per file), so an edit invalidates it on the next query with no explicit
invalidation call anywhere.

---

## 5. Constants that were tuned, and on what

Recorded because a tuned constant with no record of its evidence is a guess with
a decimal point.

| Constant | Value | Tuned on | Shape of the optimum |
|---|---|---|---|
| `MENTION_SCALE` | 0.05 | atlas, 44 questions | flat 0.02–0.05; past 0.25 recall is traded for rank |
| `MAX_MENTIONS_PER_DOCUMENT` | 4 | atlas, checked on HotpotQA | flat from 4 upward on both; falls off below — 0.900 against 0.965 at 1 on HotpotQA |
| `VECTOR_DEPTH_MULTIPLIER` | 2 | atlas | capping the vector lane recovered 0.10 MRR against letting it rank as deep as the lexical lane |
| `DEFAULT_SELF_WEIGHT` | 1.0 | atlas + HotpotQA jointly | recall peaks at 1.0, MRR at 2–3; recall is the primary metric so 1.0 wins by 0.02 recall@5 against 0.02 MRR |
| `DEFAULT_TAIL_WEIGHT` | 0.0 | atlas + HotpotQA jointly | 0 is best on recall on both (0.965 vs 0.958 on HotpotQA); MRR flat across 0–1 |
| `DEFAULT_ALPHA` | 0.15 | SPRIG, unchanged | |

Two of these were checked against HotpotQA, which no tuning touched, to make
sure an atlas-tuned constant was not quietly deciding the external result.
`MAX_MENTIONS_PER_DOCUMENT` sits on a plateau there — 0.965 recall@10 at 4, 8 and
32, falling to 0.900 only at 1 — and `DEFAULT_SELF_WEIGHT` moves HotpotQA recall
not at all between 0 and 8. `MENTION_SCALE` is a no-op on that corpus by
construction, since it has no curated links to scale mentions against.

`MENTION_SCALE` is therefore the one to be suspicious of. A curated wikilink is
an author's assertion that two pages are related; a text mention is an inference
from prose,
and given equal weight the mentions win by sheer number. The *direction* is
principled; the value is one constant tuned on one corpus, and it should be
re-derived on a second linked corpus before it is trusted as more than a sensible
default. It is free on a corpus with no links at all, where diffusion
row-normalizes any global scale away — which is why the HotpotQA numbers are
unaffected by it.

---

## 6. What did not work

**The dense lane does not help on atlas — and that was the fixture, not the
lane.** With the whole corpus embedded and fusion depth capped, `full` reaches
0.96 recall@5 against `sources` at 1.00, and no setting of `rrf_k` or vector
depth closes it. On HotpotQA the same lane is worth +0.03 recall and +0.05 MRR
over the same configuration without it. The difference is the corpus: atlas pages
are generated from templates, so twenty subsystem pages are near-identical in
embedding space and differ only in the marker token BM25 matches exactly.

This is the clearest argument in the whole exercise for not drawing conclusions
from one corpus, and it is worth stating plainly because the conclusion was drawn
and was wrong: with only atlas in hand, the reasonable-looking inference was "the
dense lane is not additive here", and a reader would have carried that away.

**Entity edges cost recall on a densely linked wiki.** On atlas, `no-entities`
(links only) reaches 1.00 recall@5; the full graph reached 0.94 with mentions
unscaled and recovers to 1.00 once they are scaled to 0.05. Every bridge atlas needs already exists as a link, so the
mention graph adds candidates and no information. On HotpotQA the same layer is
worth +0.06 recall. Both are true; `MENTION_SCALE` is the dial between them, and
a corpus at either extreme wants a different value.

**`relevance()` weights are not diffusion weights.** Direct link ×3.0 plus shared
source ×4.0 gives a sibling pair an edge stronger than any mention edge, and two
pages compiled from the same source then trade rank positions under diffusion.
The weights were designed for a "related pages" panel. They are used as edge
weights here because the build plan says to, and they work once mentions are
scaled — but they have never been tuned *as* diffusion weights and should be.

---

## 7. What is still open

- **Build-plan step 2, one database.** The L1/L2/L4/L5 caches are in process and
  rebuilt per process. That is enough to remove the latency and leave rankings
  identical, which was steps 4 and 6's acceptance criterion, but a 20-second
  index build on 2,000 documents is paid by every new process.
- **Step 10's second half.** Sources are embedded now; two-tier freshness — a
  document searchable a second after `add`, compiled later — is not built.
- **Steps 9, 11, 12, 13, 14.** Profiles, rerank, topic pages, the agentic rung,
  `rebuild`. Untouched. The `thematic` tag is the weakest column in every table
  above (MRR ~0.5), which is exactly what step 12 exists to fix.
- **The growth protocol has still never been run.** It is the only check of the
  O(document) claim, and `ragharness growth` exists and works.
- **2WikiMultiHopQA.** Build-plan step 8 names it alongside HotpotQA, and SPRIG's
  gap there is much larger (0.697 → 0.794). It was not run; the dataset was not
  reachable from the mirror used here.

---

## 8. Reproducing this

```bash
# harness
cd space_brief/evaluation
python3 fixtures/build_atlas.py --force
python3 fixtures/build_hotpot.py --questions 200 --force     # downloads

PYTHONPATH=. python3 -m ragharness.cli null-check --suite fixtures/atlas/suite
PYTHONPATH=. python3 -m ragharness.cli compare --suite fixtures/atlas/suite --k 5 \
    --lanes sources,sources/no-graph,sources/no-entities,sources/no-links,sources/presence
PYTHONPATH=. python3 -m ragharness.cli compare --suite fixtures/hotpot/suite \
    --lanes lexical,lexical/no-graph,lexical/no-links,lexical/presence

# tests
cd space_brief/wikillm_retrieval && python3 -m pytest -q
cd space_brief/evaluation        && PYTHONPATH=. python3 -m pytest -q
```

A lane name is `configuration[/ablation]`. The configuration is which corpus and
which lanes; the ablation removes exactly one mechanism, so the table says what
that mechanism is worth rather than only that the whole beats a baseline. Both
halves are named because they are independent — an ablation that also silently
requested raw sources was a degraded run on any corpus without them.

The `hybrid` and `full` lanes and the `dense` baseline need an embedding model
and a built index (`llmwiki embed --project <corpus>`), and every query is a
network round trip. Everything else runs offline.
