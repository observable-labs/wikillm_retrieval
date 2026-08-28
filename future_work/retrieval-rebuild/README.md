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

> **Read §9 first if you are here for the numbers.** Sections 1–7 are the
> 2026-08-27 rebuild and are left as they were measured. On 2026-08-28 the
> harness learned to sweep `k`, and four more defects in retrieval fell out of
> that — larger than several of the five in §2, and all of them invisible at the
> single `k` these sections report. §9 has them, and says which conclusions
> above they overturn.

Companion documents:
[`../../research/target-architecture/build-plan.md`](../../research/target-architecture/build-plan.md)
(what to build),
[`../../research/evaluation/roadmaps/harness-self-validation.md`](../../research/evaluation/roadmaps/harness-self-validation.md)
(the harness defects D1–D5 and the roadmap E1–E7 that closed them), and
[`../../research/evaluation/roadmaps/discriminating-power.md`](../../research/evaluation/roadmaps/discriminating-power.md)
(D6–D10, E8–E14, and the acceptance protocol §9 below is the retrieval half of).

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

> **Every row above the rule was fitted on one *class* of question.** atlas and
> HotpotQA are both entity-anchored: the query names the thing it is asking
> about, and the gold document's title is usually a substring of it. A constant
> tuned on one question class and applied to another is the same error as a
> constant tuned on one corpus, which is the caveat `MENTION_SCALE` already
> carries below — and it is a larger error, because there was no second question
> class to notice it against until `fixtures/atlas/thematic` existed. The rows
> below the rule are the ones fitted with that class in the room.

| Constant | Value | Tuned on | Shape of the optimum |
|---|---|---|---|
| `MENTION_SCALE` | 0.05 | atlas, 44 questions | flat 0.02–0.05; past 0.25 recall is traded for rank |
| `MAX_MENTIONS_PER_DOCUMENT` | 4 | atlas, checked on HotpotQA | flat from 4 upward on both; falls off below — 0.900 against 0.965 at 1 on HotpotQA |
| `VECTOR_DEPTH_MULTIPLIER` | 2 | atlas | **superseded — see §9.** The 0.10 MRR was an artifact of the page-scoring defect; re-swept, depth moves no recall at any k on either corpus, and the cap now buys latency only |
| `RRF_K` | 3 | atlas + HotpotQA jointly | **new — see §9.** 60 was 6% of TREC's 1,000-result lists; 6% of the 50 actually fused is 3. Monotone between 60 and 3 on both corpora, flat below 1 |
| `DEFAULT_SEEDS` | 0 = the window | atlas + HotpotQA jointly | **new — see §9.** A fixed 5 made the graph lane's sign depend on `k`; tying it to the window is at or above no-graph at every k on both |
| `group_by_page` tail term | removed | atlas + HotpotQA jointly | **new — see §9.** Chunk count outranked chunk quality; removing it is the largest single gain in this table |
| `DEFAULT_SELF_WEIGHT` | 1.0 | atlas + HotpotQA jointly | recall peaks at 1.0, MRR at 2–3; recall is the primary metric so 1.0 wins by 0.02 recall@5 against 0.02 MRR |
| `DEFAULT_TAIL_WEIGHT` | 0.0 | atlas + HotpotQA jointly | 0 is best on recall on both (0.965 vs 0.958 on HotpotQA); MRR flat across 0–1 |
| `DEFAULT_ALPHA` | 0.15 | SPRIG, unchanged | |
| — | — | — | — |
| `ABSTAIN_QUANTILE` | 0.05 | atlas + atlas-thematic jointly | **new — see §10.** p05 gates every one of the 62 keyword-hostile questions and costs nothing at all on the 44 regular ones; p10 gates 41% of the regular ones for the same gain; p25 costs 0.02 recall at k=1 and k=3. Set where it is free |
| `CALIBRATION_SAMPLE` | 400 titles | latency only | 14 ms over 78 titles, 152 ms over 400 on 1,991 documents, once per index build. Not a quality parameter — the fence is a percentile and moves by under 3% between 100 and 400 samples |
| `graph_gate` | on | atlas + atlas-thematic jointly | **new — see §10.** +0.04 recall at k=3, 5, 10 and 20 on the hostile suite; −0.02 at k=5 on atlas, in a column where the harness itself reports 2% of the usable scale left above the leader |
| `lexical_weight` / `vector_weight` | 1.0 | not tuned; a profile's instrument | on atlas, monotone downward — 0.75 costs 0.01 at k=5, 0.5 costs 0.02 at k=1, 0.1 costs 0.03. Which is what an entity-anchored suite *should* say, and is why this is a profile rather than a default |

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

> **Superseded in part, 2026-08-28.** This section's first finding was measured
> while `group_by_page` was ranking atlas's raw sources above everything on every
> query. With that fixed the dense lane on atlas scores 0.705 recall@1 rather
> than 0.023, and `full` beats `dense` at every k — though the vector lane is
> still mildly negative against `sources` at k=3 and k=5. The templated-corpus
> effect below is real and was not the whole of it — §9 has the correction, and
> the moral the section draws is if anything stronger than it was.

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
- **The growth protocol — run, 2026-08-28.** The blocker was never the cost:
  `growth()` re-ingests, so the wiki pages it produces are named by a language
  model and every gold id in both shipped suites stops resolving.
  `fixtures/atlas/growth` is a source-addressed suite (`id_space: canonical`)
  built for it. Two of the three curves exist:
  [`../../research/evaluation/roadmaps/discriminating-power.md`](../../research/evaluation/roadmaps/discriminating-power.md)
  §6.3. Ingest cost per document does not grow with the corpus; recall tracks
  coverage and dips once, by one question out of fourteen, when a newly inserted
  document displaces a gold source. The third curve has no subject until the
  protocol is run on a lane that embeds.
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

# §4, as it was measured: one k.
PYTHONPATH=. python3 -m ragharness.cli compare --suite fixtures/atlas/suite --k 5 \
    --lanes sources,sources/no-graph,sources/no-entities,sources/no-links,sources/presence

# §9, and the form every comparison should now take: the curve, not a column.
PYTHONPATH=. python3 -m ragharness.cli compare --suite fixtures/atlas/suite \
    --k 1,2,3,5,10 --lanes full,full/no-graph,sources --cache-queries
PYTHONPATH=. python3 -m ragharness.cli compare --suite fixtures/hotpot/suite \
    --k 1,2,5,10,20 --lanes lexical,lexical/no-graph,hybrid,hybrid/no-graph \
    --cache-queries

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

---

## 9. What the eval found next, 2026-08-28

Everything above was measured at one `k`. Making the harness sweep it — the work
in
[`../../research/evaluation/roadmaps/discriminating-power.md`](../../research/evaluation/roadmaps/discriminating-power.md)
— found four more defects in retrieval, and they are larger than several of the
five in §2. They are recorded here rather than folded into the sections above
because the sequence is the finding: every one of them was invisible at k=10 on
a corpus where the baseline already scored 0.90.

### 9.1 A page's vector rank depended on how deep the chunk scan went

`group_by_page` scored a page as `top + min(0.3 × Σ(other chunk scores), 1 −
top)`. Cosine similarities sit in a narrow band around 0.6–0.8, so the tail term
saturated at three chunks: a page with three retrieved chunks near 0.6 scored
1.00 while a page with one chunk at 0.85 scored 0.85. **Chunk count outranked
chunk quality.** How many chunks are retrieved is a depth constant chosen for
latency, so the vector lane's ranking depended on it — and the pipeline scans
`max(3 × max(2k, 20), 30)` chunks where the `dense` baseline scans
`max(30, 3k)`, which is why one scored below the other.

Vector-lane `recall@k`, by the depth each caller scans:

| | R@1 | R@2 | R@5 | R@10 |
|---|---|---|---|---|
| hotpot, pipeline depth | 0.378 | 0.720 | 0.968 | 0.990 |
| hotpot, dense depth | 0.420 | 0.772 | 0.968 | 0.990 |
| hotpot, **best chunk** | **0.475** | **0.863** | 0.973 | 0.990 |
| atlas, pipeline depth | 0.023 | 0.068 | 0.284 | 0.830 |
| atlas, dense depth | 0.114 | 0.227 | 0.750 | 0.886 |
| atlas, **best chunk** | **0.705** | **0.807** | **0.886** | **0.909** |

Atlas is the extreme because its fourteen multi-chunk pages are the raw source
documents: they saturated at 1.00 and sat at the head of the vector ranking for
every query, whatever the query was. **This is what §6's "the dense lane does not
help on atlas" was actually measuring.** The templated-corpus explanation given
there is real — twenty subsystem pages are near-identical in embedding space —
and it was not the dominant term, and the section drew a conclusion from it.

Scoring a page by its best chunk buys an invariant worth more than the recall:
the lane's ranking no longer depends on the scan depth, so a constant chosen for
latency stops deciding results. Weaker variants of the coverage bonus — a mean
instead of a sum, a capped per-chunk count — were measured too and cost 0.08
recall@1 on atlas. A bonus small enough to be safe is a bonus too small to do
anything.

### 9.2 `RRF_K = 60` is 6% of a list this pipeline never has

RRF's published constant was tuned against TREC runs of 1,000 results. These
lanes rank 20 to 50, where 60 sits above the whole list and flattens it: a
document one lane ranks first scores `1/61 = 0.0164` while a document both lanes
rank tenth scores `2/70 = 0.0286` and outranks it, so a lane that is merely
adequate outvotes one that is good. Rescaled to the same fraction of the depth
actually fused, 6% of 50 is 3.

|  | R@1 | R@2 | R@5 | R@10 |
|---|---|---|---|---|
| hotpot, `rrf_k = 60` | 0.435 | 0.672 | 0.912 | 0.988 |
| hotpot, `rrf_k = 3` | 0.453 | 0.730 | 0.953 | 0.990 |
| atlas, `rrf_k = 60` | 0.705 | 0.864 | 0.920 | 0.955 |
| atlas, `rrf_k = 3` | 0.727 | 0.864 | 0.932 | 0.955 |

Monotone between 60 and 3 on both and flat below 1, so the value is not on a
cliff.

### 9.3 The graph lane's sign changed with `k`, and the seed count is why

§4.2 reported the graph lane as worth +0.07 recall@10 on HotpotQA. Swept, its
contribution was `−0.01, −0.03, +0.01, +0.07, +0.04` at k = 1, 2, 5, 10, 20. A
mechanism whose sign changes with the size of the result window is re-ranking
inside a window, not retrieving better — and the published result it was checked
against, SPRIG's RRF 0.851 → seeded-PPR 0.867, is reported at R@5, where this
implementation showed nothing.

The cause was `DEFAULT_SEEDS = 5`, fixed. Returning one result while five
documents radiate mass means a diffusion driven by four documents that will not
be shown decides the one that is. `seed_count = 0` now means "as many as the
caller asked for":

| | R@1 | R@2 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| hotpot, no graph | 0.453 | 0.730 | 0.953 | 0.990 | 0.998 |
| hotpot, seeds = 5 | 0.450 | 0.698 | 0.960 | 0.993 | 0.998 |
| hotpot, **seeds = k** | 0.453 | 0.730 | 0.960 | **0.998** | **1.000** |
| atlas, no graph | 0.727 | 0.864 | 0.886 | 0.909 | 0.920 |
| atlas, seeds = 5 | 0.727 | **0.875** | 0.932 | 0.955 | 0.955 |
| atlas, **seeds = k** | 0.727 | 0.864 | 0.932 | 0.955 | 0.955 |

`seeds = k` is at or above no-graph at every k on both corpora, which `seeds = 5`
is not. It costs one question of atlas R@2 against the fixed value and buys the
property that makes the lane safe to leave on at any `k`.

### 9.4 The vector lane's local cost grew with the corpus, not the work

Two defects, both in `embeddings.py`, both found by trying to build a corpus
large enough for the baseline to have room to lose on.

`index_documents` batched chunks *within* a document. A wiki of short pages is
one chunk per page, so `batch_size` never fired at all and embedding was one
HTTP round trip per page: 9,769 paragraphs at about 1.4 s each is roughly four
hours. Filling the batch from as many documents as it holds made the same work
306 requests and about five minutes.

`VectorStore.search` selected every chunk's `text` alongside its vector and
re-parsed every vector on every query — 45 MB read to compute a dot product that
never looks at the text. The scan now reads vectors only, holds the parsed
matrix for as long as the store is unchanged, and fetches text for the surviving
`top_k` in a second statement.

Together they take the lane from a cost that grows with the corpus to one that
grows with the work: on the 9,931-chunk store, 145 ms for the first query and
**5.6 ms** for every one after, against **6.1 ms** on the five-times-smaller
store. What remains is llmwiki's shipped configuration rather than a defect —
`numpy` is an opportunistic accelerator and not a dependency, and without it the
same scan is **269 ms every query**. It is now a declared extra,
`pip install llmwiki[vector]`, so that which one was measured is a configuration
rather than an accident of the environment. Every latency number in this
document was measured without it.

### 9.5 What it adds up to

The same fixture as §4.2, swept rather than reported at one `k`:

```
  system                        R@1    R@2    R@5   R@10   R@20   separates   local  remote
  llmwiki/hybrid               0.45   0.73   0.96   1.00   1.00         k=1      76       0
  llmwiki/lexical              0.42   0.60   0.76   0.95   0.98         k=1      12       0
  bm25                         0.40   0.59   0.74   0.90   0.95           —       2       0
  dense                        0.47   0.86   0.97   0.99   0.99         k=1      71       0

  Δrecall vs bm25                k=1     k=2     k=5    k=10    k=20
  llmwiki/hybrid              +0.05*  +0.14*  +0.22*  +0.10*  +0.05*
                              * paired bootstrap 95% CI excludes zero
```

And on atlas, at the k where that corpus can still separate anything:

```
  system                     R@1    R@2    R@3    R@5   R@10   separates   local  remote
  llmwiki/full              0.76   0.90   0.92   0.98   1.00         k=5      14     342
  llmwiki/full/no-graph     0.76   0.90   0.92   0.93   0.95        none       6       0
  bm25                      0.67   0.83   0.88   0.90   0.95           —       0       0
  dense                     0.74   0.85   0.88   0.93   0.95        none       6     324
```

Three readings. The advantage over bm25 keeps its sign at every `k` and its
interval excludes zero at every `k`, which the k=10 headline in §4.2 could not
have shown in either direction. `separates: none` on the no-graph row is the
sharpest available statement of what the graph lane is worth — without it,
llmwiki does not beat bm25 with confidence at any `k` on atlas. And the latency
column finally says what it means: 14 ms of local work against bm25's under 1,
plus 342 ms rented from an embedding provider that the `dense` baseline rents
from too.

**One thing did not get fixed.** On HotpotQA the hybrid still scores below the
`dense` baseline at k ≤ 5 — 0.73 against 0.86 at k=2 — and above it at k ≥ 10.
That corpus's lexical lane is 0.60 at k=2 against the vector lane's 0.86, and
equal-weight RRF lands almost exactly between them, which is what equal-weight
RRF is for. On atlas fusion is additive at k=2 and slightly negative at k=5,
where the lexical lane alone reaches 0.955 against the fused 0.932. Two repairs
were measured and both were worse: weighting the lanes by their own score margin
fails because BM25 margins (0.5–0.7) and cosine margins (0.1–0.25) are not
comparable, and treating an unranked document as ranked one past the lane's
depth moves nothing. Calibrating per lane would need labelled data, and the only
labelled data here is the eval. The shortfall is declared rather than tuned away,
and `compare` now prints it and exits non-zero without anyone having to notice.

> **This is where §10 starts, and the last sentence of that paragraph is the
> part that was wrong.** Calibrating *across* lanes needs labels. Calibrating a
> lane against *itself* on the same corpus needs none, and the reference data
> was sitting in the index the whole time.

---

---

## 10. Fusing a lane that found nothing

Compiled 2026-08-28, after
[`research/evaluation/roadmaps/representative-questions.md`](../../research/evaluation/roadmaps/representative-questions.md).
Everything above this section was measured on questions that name the thing they
are asking about — atlas and HotpotQA both — and §9 closed the last defect
visible from inside that class. This one was not.

### 10.1 The measurement that made it visible

Sixty-two questions over the same 78 atlas documents and the same facts, phrased
without any of the corpus's own vocabulary and checked at build time to be sure:

```
  atlas-thematic · 62 questions · 78 documents · recall@k

  system                          R@1    R@3    R@5   R@10   R@20   local
  ──────────────────────────────────────────────────────────────────────
  dense                          0.15   0.39   0.52   0.66   0.80       1
  llmwiki/full  (before)         0.03   0.20   0.33   0.45   0.64      16
  bm25                           0.01   0.02   0.02   0.02   0.09       0
```

bm25 goes from 0.95 on the regular atlas questions to 0.02 on these. That is the
finding the roadmap is about and it is not this document's subject. The row that
is: **the assembled system scored 0.21 below its own vector lane at k=10.**
Equal-weight RRF was fusing a ranking worth 0.02 with a ranking worth 0.66 and
landing between them, which is exactly what equal-weight RRF is for and exactly
wrong here.

### 10.2 The signal was already computed

The lexical lane's own top BM25 score. On atlas, questions that use the corpus's
words score a median 7.99; questions that avoid them score a median 2.20. The
distributions barely touch.

A raw BM25 score does not transfer between corpora — it depends on term
distribution, document lengths and column weights — so the threshold has to be a
percentile of *this corpus's* distribution. The reference data needs no labels
and no traffic: **run each of a sample of document titles as a query and record
what the lexical lane scores.** A title is the cheapest available example of a
query that genuinely names something in the corpus. 14 ms over 78 titles, 152 ms
over 400 on 1,991 documents, once per index build, invalidated by the same
corpus fingerprint that invalidates the index.

`retrieval/calibration.py`. The fence is the 5th percentile; a query below it
abstains rather than voting, and only when there is another lane to fall back on
— a ranking from a weak lane still beats no ranking at all.

| corpus | fence | questions gated |
|---|---|---|
| atlas, keyword-hostile | 6.17 | 62 of 62 |
| atlas, regular | 6.17 | 12 of 44 |
| MuSiQue | 16.74 | 28 of 485 |
| HotpotQA | 14.87 | 1 of 200 |

Four question sets, ordered exactly as their phrasing predicts, from one rule
with no per-corpus constant in it.

### 10.3 What it bought, and what it cost

```
  atlas-thematic · recall@k

  system                          R@1    R@3    R@5   R@10   R@20   local
  ──────────────────────────────────────────────────────────────────────
  llmwiki/full  (gated)          0.15   0.39   0.52   0.66   0.80       2
  llmwiki/full  (lexical only)   0.15   0.35   0.48   0.62   0.76      16
  llmwiki/full  (before)         0.03   0.20   0.33   0.45   0.64      16
```

The lexical gate is worth +0.12 to +0.17; gating the graph lane on the same
signal — diffusing from a fused list with no lexical evidence in it is diffusing
from noise — adds +0.04 at every `k` above 1. `full` now equals `dense` to two
decimal places at every `k`, and `compare` reports no monotonicity violation on
that suite where it previously reported twelve.

**It costs nothing on the entity-anchored corpora.** atlas is unchanged at every
`k` except a 0.02 at k=5 that belongs to the *graph* gate, in a column the
harness itself flags as having 2% of the usable scale left. MuSiQue gains 0.01 at
k=1 and k=3. HotpotQA gains 0.01 at k=3. That is the whole of P6 and it is why
the fence sits at the 5th percentile rather than the 10th: p10 gates 41% of
atlas's regular questions for no additional gain.

**Latency, measured on the queries the gate fires on** — a p50 over a suite
where 6% of queries gate is drawn from the 94% where nothing changed, so the
per-suite number cannot show this:

| corpus | documents | graph skipped | graph run | saved |
|---|---|---|---|---|
| MuSiQue | 5,918 | 7 ms | 144 ms | **137 ms** |
| atlas | 78 | 1 ms | 10 ms | 8 ms |

### 10.4 Profiles, and the part a gate cannot do

A gate is the right instrument when a lane has *no* handhold. It does nothing for
a query with a weak one, and after the gate the hybrid was still below `dense` at
every `k` on MuSiQue — by 0.003 to 0.03, small, consistent, and reported every
time by the monotonicity check.

`retrieval/profiles.py` adds the continuous form: `abstain_quantile`, where the
fence sits, and `lexical_weight`, what the lane's vote is worth when it clears
it. Four profiles — `voice`, `balanced`, `deep`, `research` — over both the depth
axis build-plan step 9 specified and the lane-trust axis it did not.

```
  musique · 485 questions · recall@k

  system                       R@1    R@3    R@5   R@10   R@20   local
  ────────────────────────────────────────────────────────────────────
  llmwiki/hybrid/research     0.34   0.60   0.68   0.76   0.82     227
  llmwiki/hybrid  (balanced)  0.33   0.57   0.66   0.74   0.82     288
  dense                       0.33   0.60   0.68   0.75   0.82       3
  bm25                        0.26   0.39   0.44   0.51   0.57      15
```

`research` is at or above `dense` at every `k` — the two remaining shortfalls are
0.003 and 0.0002 — where `balanced` was below it at every `k`. It is not the
default and should not be: on atlas it scores `0.74 / 0.88 / 0.94 / 1.00` against
`balanced`'s `0.76 / 0.92 / 0.96 / 1.00`. Which lane to trust is a property of the
question class, and the caller knows the class in a way the ranker does not.

### 10.5 What this does not show

A judge comparing the gated configuration against the pre-gate one on 24
corpus-level questions — *"what are the main themes here"*, no gold set, scored
by pairwise win rate with the presentation order swapped — finds **no difference
on either comprehensiveness or grounding**. Both intervals span 0.5.

Against bm25 on the same questions the assembled system wins on comprehensiveness
(0.79, interval [0.52, 0.92]) and ties on grounding. Against `dense`, where the
gate makes the two retrievals identical, the judge correctly returns a tie — which
is the only evidence available that it is reading the answers rather than the
layout.

At n=24 the interval cannot rule out a real gain from the gate. What it does rule
out is quoting 0.21 recall as though it were a claim about the answers a person
reads. On a 78-document corpus almost any ten pages support a plausible answer to
a thematic question, and a large improvement in *which* ten may simply not reach
the reader. Running that comparison at a larger n on a larger corpus is one
command and is the most valuable measurement not yet taken.
