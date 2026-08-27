---
name: llmwiki-ask
description: Ask a question of an llmwiki corpus — retrieval runs over every wiki page and every raw source, then answers with numbered citations. Use when the user wants to query, ask, look something up in, or "find out what the docs say" about a knowledge base built with the llmwiki CLI, and when diagnosing why an answer came back thin or uncited.
---

# Ask a question of an llmwiki corpus

`ask` retrieves across **the whole project** — every compiled wiki page and every
raw source — then answers with numbered citations. Each call is independent:
there is no conversation state, so a follow-up must restate its own context.

## Ask

```bash
llmwiki ask "How do the two chemistries compare?" -p /path/to/wiki
```

`-p` is required. `ask` will not infer a project from `$PWD` or
`$LLMWIKI_PROJECT`; without it you get `error: --project is required`. Supply the
project rather than setting `LLMWIKI_STRICT_PROJECT=0` — querying the wrong
corpus produces an answer that is confidently thin rather than obviously wrong.
If which wiki to query is unclear, `llmwiki status` still walks up from `$PWD`
and will say where you are.

Output is split deliberately:

- **stdout** — the answer text only, streamed as it generates
- **stderr** — progress, citations, retrieval stats

So `llmwiki ask "..." -p ... > answer.md` captures a clean answer with the
citation list still visible in the terminal. Don't redirect both unless the user
wants the diagnostics in the file.

Useful flags:

```bash
llmwiki ask "..." -p ... --top-k 40      # widen retrieval (default 20)
llmwiki ask "..." -p ... --no-sources    # compiled wiki pages only
llmwiki ask "..." -p ... --json          # answer + citations + retrieval stats
```

## Write the question for retrieval, not for chat

Phase 1 is tokenized keyword scoring. A question carrying the corpus's own
vocabulary retrieves far better than a conversational one.

- "What did the paper conclude?" — nothing to match on. Which paper?
- "What did the 2024 vanadium flow battery paper conclude about round-trip
  efficiency?" — proper nouns and domain terms to score against.

If the user's phrasing is vague and you know the corpus, ask the sharper version.
Don't ask five narrow questions where one will do; each is a full LLM call over a
packed context.

## Read the citation block

```
Sources:
  [1] wiki/concepts/round-trip-efficiency.md
  [2] wiki/sources/flow-batteries.md
  [3] wiki/entities/vanadium-flow-battery.md  (via round-trip-efficiency)
  [4] raw/sources/flow-batteries.md

  keyword retrieval · 4 of 4 matches used · 2,152 chars of context
```

- `wiki/…` are compiled pages; `raw/sources/…` is original document text, scored
  lower on purpose so the compiled layer stays the answer layer.
- `(via …)` marks a page pulled in by **graph expansion** — it didn't match the
  query itself, a page that did links to it. This is the mechanism that makes the
  wiki more than keyword search, and it's a real signal when it fires.
- **`N of M matches used`** — if used is well below found, pages were dropped at
  the budget boundary. Raise `--max-context` or narrow the question.
- **mode** is `keyword` unless a vector index exists, then `hybrid`.

Answers are grounded in retrieved text. When you relay one to the user, keep the
citations attached — an assertion from this system is only as good as the page
behind it, and stripping the markers makes it unverifiable.

## When the answer is thin

Diagnose retrieval before blaming the model. `search` runs the identical
retrieval pipeline with **no LLM call** — free and instant:

```bash
llmwiki search "vanadium efficiency" -p /path/to/wiki
llmwiki search "..." -p ... --top-k 30 --json
```

It prints scores, which lane each hit came from, snippets, and graph provenance —
exactly the material `ask` would have worked from.

| What `search` shows | What it means | Fix |
|---|---|---|
| No results at all | Nothing on this topic is in the corpus | The document was never ingested — check `llmwiki status`, then ingest it |
| Only `raw/sources/…` hits | Ingested, but no wiki page was built for the topic | Re-ingest with `--force` after tightening `schema.md` |
| Right pages, low scores | Query vocabulary doesn't match the corpus | Rephrase using the terms the pages actually use |
| Good hits, thin answer | Retrieval fine, context budget bit | Raise `--top-k`, or `--max-context` if the model window allows |
| Relevant page never surfaces | Isolated in the graph, weak keyword overlap | Enable vector search (below) |

If the answer says the corpus doesn't cover something, that is a real finding —
the prompt instructs the model to say what's missing rather than fill the gap.
Report it as a gap, don't paper over it with your own knowledge.

## Vector search, when keyword isn't enough

Optional second retrieval lane, fused with keyword results by reciprocal-rank
fusion. Worth enabling when questions are paraphrases rather than term matches:

```bash
export LLMWIKI_EMBEDDING_MODEL=text-embedding-3-small
llmwiki embed -p /path/to/wiki
```

Needs an OpenAI-compatible `/v1/embeddings` endpoint. If the index is configured
but not built, `ask` says so in a note and degrades to keyword + graph rather
than failing. It also degrades gracefully — with a note — if the embedding
endpoint errors mid-query.

## Contradictions and multiple sources

Where sources disagree, the answer attributes each position to its citation
rather than picking a winner. That is intended behaviour — surface both sides to
the user rather than collapsing them. Recurring contradictions usually also show
up in `wiki/reviews.md`, flagged at ingest time.
