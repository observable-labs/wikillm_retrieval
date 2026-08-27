---
name: llmwiki-add-document
description: Ingest one document into an llmwiki project — running the two-step analysis-then-generation build that turns a PDF, markdown file, transcript, spreadsheet, or code file into linked wiki pages. Use when the user wants to add, ingest, index, file, or "put a document into" a wiki/knowledge base built with the llmwiki CLI. For a whole folder at once, use llmwiki-add-directory instead.
---

# Add a document to an llmwiki project

Ingesting a document is **two LLM calls plus deterministic file writes**. It is not
free and it is not instantaneous — a long PDF can take a minute or more and is
analyzed in sections. Get the preflight right before spending the call.

## 1. Find the project

```bash
llmwiki status                 # from inside the project (walks up from $PWD)
llmwiki status -p /path/to/wiki
```

`status` prints the project root, page and source counts, provider, model, and
context budget. If it errors with `no wiki project found here`, either the user is in the
wrong directory or none exists yet:

```bash
llmwiki init /path/to/wiki --template research
```

Templates are `research`, `reading`, `personal`, `business`, `general`. The
template writes `schema.md`, which is the **authoritative routing rule** for what
page types the model may create — picking the wrong one produces plausible pages
filed in the wrong shape. Ask which fits if it isn't obvious from the corpus.

`add` and `ask` **require** `-p/--project`. They refuse to infer a project — not
from `$PWD`, not from `$LLMWIKI_PROJECT` — and stop with:

```
error: --project is required. Pass -p <dir>, or set LLMWIKI_STRICT_PROJECT=0
to allow discovery from $PWD and $LLMWIKI_PROJECT.
```

When you see that, supply the project. **Never set `LLMWIKI_STRICT_PROJECT=0` to
make a command run.** The check exists because ingesting into the wrong wiki is
not cleanly reversible: the file lands in `raw/sources/` and the model rewrites
`index.md`, `log.md`, and whatever pages it decided to merge into.

`status` and `search` are free and read-only, so they still walk up from `$PWD`.
That makes `llmwiki status` the way to confirm which wiki you are about to write
to when the user's phrasing — "my wiki", "the research one" — is not a path.

## 2. Check credentials before spending a call

`status` shows the resolved provider and model. Provider is inferred from
whichever key is present: `ANTHROPIC_API_KEY` wins; `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, or `LLMWIKI_BASE_URL` select the OpenAI-compatible lane.

Failure modes to recognize before running:

- **`no model configured. Set LLMWIKI_MODEL or pass --model`** — the
  OpenAI-compatible lane has no default model. Set `LLMWIKI_MODEL` or pass
  `--model`. Anthropic defaults to `claude-opus-5` and needs no model flag.
- Missing key entirely — tell the user which env var to export. Never invent a
  key, and never pass a key on the command line if an env var will do (it lands
  in shell history).

## 3. Check the format is readable

Handled by the standard library: `.md .markdown .txt .rst .org .text .log`,
source files (`.py .js .ts .tsx .jsx .rs .go .java .rb .sh .sql`), config
(`.yaml .yml .toml .ini .cfg .xml`), `.json .jsonl .ndjson`, `.csv .tsv`,
`.html .htm`.

`.pdf .docx .xlsx .xlsm .pptx` need optional parsers:

```bash
pip install 'llmwiki[docs]'
```

Anything else (`.heic`, `.zip`, images) fails with `looks like binary data`.
That is the parser refusing to ingest garbage, not a bug — the fix is to convert
the file first, not to retry.

## 4. Run it

```bash
llmwiki add paper.pdf -p /path/to/wiki
llmwiki add notes.md -p /path/to/wiki --folder research   # folder hint for the model
llmwiki add paper.pdf -p /path/to/wiki --force            # re-ingest an unchanged file
```

What happens, in order:

1. The source is copied into `raw/sources/` (immutable — the original is never
   modified, and the copy is what the wiki cites).
2. **Step 1, analysis.** The model reads the document against the current wiki
   index and writes structured findings: entities, concepts, arguments and their
   evidence strength, connections to existing pages, contradictions. Nothing
   reaches disk. Long documents are split into overlapping sections here and the
   per-section analyses are consolidated.
3. **Step 2, generation.** The model reads *its own analysis* and emits the pages.
4. Deterministic writes: pages land under `wiki/`, `sources` frontmatter is
   canonicalized, `index.md` / `log.md` / `reviews.md` are updated by code — the
   model never writes those.

Sources are SHA256-hashed. Re-adding an unchanged file is a no-op that costs
nothing and prints `unchanged`; `--force` overrides.

## 5. Read the output, don't just report success

```
✓ flow-batteries.md → 3 page(s)
    wiki/sources/flow-batteries.md
    wiki/entities/vanadium-flow-battery.md
    wiki/concepts/round-trip-efficiency.md
    updated: wiki/index.md, wiki/log.md, wiki/reviews.md
    1 review item(s) → wiki/reviews.md
      [missing-page] Compare flow batteries to lithium-ion
```

- **Pages** are what was created. One page for a dense document usually means the
  model didn't find much to route — worth checking `schema.md`.
- **updated:** are the maintained aggregates. They are not new knowledge; don't
  count them when reporting how many pages were built.
- **Review items** are the model flagging gaps it noticed but was not asked to
  fill — a missing page, an unresolved contradiction. They accumulate in
  `wiki/reviews.md`. Surface them to the user; they're the highest-signal
  follow-up work in the whole system.
- **Warnings** (`!`) mean something was salvaged: a truncated block repaired, a
  page filed somewhere unexpected, a path rejected. Report them verbatim rather
  than paraphrasing.

Use `--json` when you need to act on the result programmatically; it carries the
same fields plus `stored_at` and per-review `search` queries.

## 6. When the pages come out wrong

The lever is **`purpose.md` and `schema.md`, not the prompt**. `purpose.md`
states why the wiki exists and steers every ingest; `schema.md`'s Page Types
table decides what may be created and where it goes. Edit those, then:

```bash
llmwiki add paper.pdf -p /path/to/wiki --force
```

Re-ingesting the same document after fixing the schema is the intended workflow.
There is no prompt flag to override it.

## Verifying

```bash
llmwiki search "term from the document" -p /path/to/wiki
```

`search` is retrieval with no LLM call — the fastest way to confirm a document
actually landed and is reachable. To read a page, just open it; the wiki is plain
markdown with `[[wikilinks]]`, readable by hand or in Obsidian.
