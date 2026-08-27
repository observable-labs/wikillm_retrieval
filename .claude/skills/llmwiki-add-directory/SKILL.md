---
name: llmwiki-add-directory
description: Bulk-ingest a folder of documents into an llmwiki project — scoping the run, piloting on a few files, then walking the tree. Use when the user wants to add a directory, a corpus, a document library, "all my notes/papers/PDFs", or otherwise ingest many files at once into a wiki built with the llmwiki CLI. For one file, use llmwiki-add-document.
---

# Ingest a directory into an llmwiki project

A bulk run is **two LLM calls per document**, sequential. A hundred files is a
hundred pairs of calls. The whole skill is about not discovering a problem on
document ninety.

Read `llmwiki-add-document` first if you haven't — project discovery, credential
checks, supported formats, and how to read per-file output all apply unchanged
here. In particular: `add` requires `-p/--project` and will not infer one. Do not
work around that with `LLMWIKI_STRICT_PROJECT=0` — a hundred documents written
into the wrong wiki is the most expensive mistake available here.

## 1. Scope the run before starting it

Count what will actually be ingested. `add` silently skips unsupported
extensions and anything under a dot-directory, so a raw `ls | wc -l` overstates:

```bash
find ~/papers -type f \
  \( -iname '*.md' -o -iname '*.pdf' -o -iname '*.txt' -o -iname '*.docx' \) \
  -not -path '*/.*' | wc -l
```

Adjust the extension list to what's actually in the tree. Then report the count
and get confirmation before a long run — this spends the user's API budget, and
scaling it down is their call, not yours.

If the tree contains `.pdf`/`.docx`/`.xlsx`/`.pptx`, confirm the parsers are
installed *now* rather than failing per-file later:

```bash
pip install 'llmwiki[docs]'
```

## 2. Pilot on two or three files first

Non-negotiable for a corpus you haven't ingested before. Page quality is
governed by `schema.md` and `purpose.md`, and you cannot tell whether they fit
this corpus until you've seen real output:

```bash
llmwiki add ~/papers/one.pdf ~/papers/two.pdf -p /path/to/wiki
```

Look at what got created. Wrong page types, everything collapsing into one
summary page, entities that should be concepts — all of these are `schema.md`
problems, and all of them are cheap to fix now and expensive to fix after a
hundred documents. Edit `schema.md` / `purpose.md`, then re-run the pilot with
`--force`.

## 3. Run the tree

```bash
llmwiki add ~/papers -r -p /path/to/wiki
```

- **`-r` is required to descend.** Without it, only the top level of the
  directory is ingested — no error, no warning, just fewer files than expected.
  This is the single most common mistake.
- Under `-r`, each file's subdirectory path is passed to the model as a folder
  hint automatically. Only pass `--folder` if you want to override that for the
  entire run.
- Multiple roots work: `llmwiki add ~/papers ~/notes -r -p ...`.

**Ingest order matters.** Each document is analyzed against the wiki index *as it
stands at that moment*, so later documents can link into pages that earlier ones
created. Foundational or overview material first produces a better-connected
graph than alphabetical order. If the user has a natural reading order, follow
it; if you're reordering deliberately, pass the files explicitly rather than
relying on the walk.

## 4. Failures don't stop the run

A file that fails — unreadable format, empty text, provider error — is reported
and skipped; the remaining files still ingest. The command exits `1` if anything
failed, `0` only on a clean run.

This makes bulk runs **resumable**. Because sources are SHA256-cached, re-running
the exact same command after a partial failure skips everything already done at
no cost and retries only what didn't land:

```bash
llmwiki add ~/papers -r -p /path/to/wiki   # safe to re-run verbatim
```

Never reach for `--force` to resume — it discards the cache and re-ingests the
entire tree from scratch.

For a long run, capture the outcome instead of scrolling:

```bash
llmwiki add ~/papers -r -p /path/to/wiki --json > ingest-log.json
```

Then report: how many ingested, how many were already cached, which files failed
and why, and the accumulated review items.

## 5. Report the aggregate, not a wall of per-file output

After the run:

```bash
llmwiki status -p /path/to/wiki
```

Worth surfacing to the user:

- pages built vs. sources ingested (a low ratio means the schema is routing
  everything into one shape)
- files that failed, grouped by cause — usually one missing parser explains a
  whole class of them
- `wiki/reviews.md` — bulk ingest accumulates review items fast, and they are the
  best available map of what the corpus is missing or contradicts itself about
- unreachable material: if a document ingested but nothing links to its pages,
  it's isolated in the graph

## 6. Optional: build the vector index afterwards

Retrieval works on keyword + graph expansion with no extra setup. Vector search
is an opt-in second lane, and it's worth building once after a bulk ingest
rather than per-file:

```bash
export LLMWIKI_EMBEDDING_MODEL=text-embedding-3-small
llmwiki embed -p /path/to/wiki
```

Needs an OpenAI-compatible `/v1/embeddings` endpoint (`LLMWIKI_EMBEDDING_BASE_URL`
/ `LLMWIKI_EMBEDDING_API_KEY` if it isn't OpenAI itself). Re-running is
incremental — unchanged pages are skipped by hash.
