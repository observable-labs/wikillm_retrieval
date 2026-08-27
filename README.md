# llmwiki

Build and query a self-maintaining wiki from your documents.

Instead of retrieving from raw files at query time, an LLM incrementally
builds a persistent, interlinked wiki that sits between you and your sources.
Adding a document doesn't just index it — the model reads it, extracts what
matters, and integrates it into the pages that already exist. Knowledge is
compiled once and kept current rather than re-derived on every question.

This is a pure-Python extraction of the wiki-building pipeline from
[llm_wiki](https://github.com/nashsu/llm_wiki) (a Tauri/React/Rust desktop
app), which in turn implements
[Karpathy's llm-wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
The ingest strategy, retrieval pipeline, prompts, scoring weights, and budget
math are ported rather than reinvented.

```
pip install llmwiki                # core: keyword + graph retrieval, no deps
pip install 'llmwiki[anthropic]'   # + the Anthropic SDK
pip install 'llmwiki[docs]'        # + PDF / DOCX / XLSX / PPTX parsing
pip install 'llmwiki[all]'         # everything
```

## Quick start

```bash
export ANTHROPIC_API_KEY=...

llmwiki init ~/wikis/energy --template research
llmwiki add ~/papers/flow-batteries.pdf --project ~/wikis/energy
llmwiki ask "What limits round-trip efficiency?" --project ~/wikis/energy
```

### Choosing the project

`add` and `ask` require `--project`. They rewrite the wiki and spend model
calls, and aiming either by ambient default is how a document ends up in the
wrong project — so neither will infer one, even from inside the project
directory:

```
$ llmwiki add paper.pdf
error: --project is required. Pass -p <dir>, or set LLMWIKI_STRICT_PROJECT=0
       to allow discovery from $PWD and $LLMWIKI_PROJECT.
```

`search` and `status` are free and read-only, so they still find a project by
walking up from `$PWD` — `status` is how you work out where you are.

For a wiki you use constantly, a shell function keeps `-p` explicit:

```bash
energy() { llmwiki "$@" -p ~/wikis/energy; }
energy ask "What limits round-trip efficiency?"
```

To turn the requirement off instead and get discovery back everywhere:

```bash
export LLMWIKI_STRICT_PROJECT=0        # or "strict_project": false in the user config
export LLMWIKI_PROJECT=~/wikis/energy  # optional default project
```

Resolution is then `--project` → `$LLMWIKI_PROJECT` → the walk-up. The env var
beats the walk-up, so an exported default stays in effect even while you are
standing inside a *different* wiki.

## What a project looks like

```
energy/
├── schema.md          page types and routing rules — the authoritative config
├── purpose.md         why this wiki exists; read on every ingest and query
├── raw/sources/       your documents, immutable
├── wiki/              LLM-generated pages
│   ├── index.md       catalog of every page (app-maintained)
│   ├── log.md         append-only, greppable: ## [2026-08-25] ingest | paper.pdf
│   ├── overview.md    high-level summary
│   ├── sources/       one page per ingested document
│   ├── entities/      named things
│   ├── concepts/      ideas, methods, phenomena
│   └── reviews.md     items the ingest flagged for a human
└── .llm-wiki/         ingest cache, vector store, warning log
```

Everything is plain markdown, so the directory doubles as an Obsidian vault
and as a git repo. `wiki/` is the model's; `raw/` and `purpose.md` are yours.

## Commands

### `llmwiki add <file>...`

Adds a document. Two LLM calls, in sequence:

**Step 1 — analysis.** The model reads the source against the current wiki
index and writes a structured analysis: key entities, key concepts, main
arguments and their evidence, connections to existing pages, contradictions,
and recommendations. Nothing is written to disk.

**Step 2 — generation.** The model reads its own analysis and emits the wiki
pages as `---FILE:---` blocks — a source summary, entity and concept pages
with YAML frontmatter and `[[wikilinks]]`, and a log entry.

Splitting read-then-write into two calls is the single biggest quality lever
in the original project; a model asked to do both at once does both worse.

Afterwards, deterministically (no model involved): pages are written, their
`sources` frontmatter is canonicalized to the document they came from,
`index.md` and `log.md` are updated, and a source summary is guaranteed to
exist even if the model omitted it.

```bash
llmwiki add paper.pdf -p ~/wikis/energy
llmwiki add ~/notes -r --folder research -p ~/wikis/energy   # walk a directory
llmwiki add paper.pdf --force -p ~/wikis/energy              # re-ingest an unchanged file
```

Sources are SHA256-hashed, so re-adding an unchanged document costs nothing.

### `llmwiki ask "<question>"`

Retrieves across **everything in the project** — every wiki page and every raw
source — then answers with numbered citations.

```
Phase 1    tokenized keyword scoring (CJK-aware) over wiki/ and raw/sources/
Phase 1.5  vector search over chunk embeddings                    [optional]
Phase 2    reciprocal-rank fusion of the keyword and vector rankings
Phase 3    graph expansion — the top hits' [[wikilink]] neighbours take a
           reserved 15-30% of the result window
Phase 4    budget-controlled packing: 50% of the context window for pages,
           5% for the index, 15% held back for the response
```

Graph expansion is what makes this more than keyword search: a page that
never mentions your query still surfaces when a page that does links to it.

```bash
llmwiki ask "How do the two chemistries compare?" -p ~/wikis/energy
llmwiki ask "..." -p ... --no-sources     # wiki pages only
llmwiki ask "..." -p ... --top-k 40 --json
```

### `llmwiki search "<query>"`, `embed`, `status`

`search` runs retrieval with no LLM call — useful for seeing what `ask` will
be working from. `embed` builds the optional vector index. `status` reports
what's in the project and how it's configured.

## Configuration

Precedence: defaults → `~/.config/llmwiki/config.json` →
`<project>/.llm-wiki/config.json` → `.env` → environment → CLI flags.

| Variable | Meaning |
|---|---|
| `LLMWIKI_DOTENV` | path to a specific `.env`, or `0` to disable `.env` loading |
| `LLMWIKI_STRICT_PROJECT` | `0` lets `add` and `ask` discover a project instead of requiring `--project` (default: on) |
| `LLMWIKI_PROJECT` | default project directory, once strict mode is off — read *before* the config files, since it decides which project's config gets loaded |
| `LLMWIKI_PROVIDER` | `anthropic` (default) or `openai` |
| `LLMWIKI_MODEL` | model id; defaults to `claude-opus-5` on Anthropic |
| `LLMWIKI_API_KEY` | falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` |
| `LLMWIKI_BASE_URL` | for OpenAI-compatible endpoints |
| `LLMWIKI_MAX_CONTEXT` | context budget in **characters** (default 204,800) |
| `LLMWIKI_EFFORT` | how hard the model may think when answering — see below |
| `LLMWIKI_INGEST_EFFORT` | the same for `add`; defaults to the least a thinking model allows |
| `LLMWIKI_EMBEDDING_MODEL` | enables the vector lane |
| `LLMWIKI_EMBEDDING_BASE_URL` | OpenAI-compatible `/v1/embeddings` |

### Reasoning effort

`off`, `low`, `medium`, `high`, `max`, clamped to what the model actually
accepts. Gemini 3 and OpenAI's reasoning models have no "off" — the request
is real but the API won't honour it — so `off` becomes the lowest level they
do take rather than being dropped. On Anthropic the level maps to
`output_config.effort`; on OpenAI-compatible endpoints to `reasoning_effort`.

Left unset, answering runs at the provider's default. Ingest doesn't: its two
calls are structured extraction, where thinking buys little and costs a lot.
On a thinking model the `max_tokens` ceiling covers reasoning *and* content
together, so a model that thinks too long returns an empty response and the
pages it was writing are lost. `add` therefore asks for the least thinking
the model permits, which `LLMWIKI_INGEST_EFFORT` overrides for an endpoint
that refuses a wound-down level.

The default only applies to model families known to reason unprompted, since
elsewhere `reasoning_effort` may be rejected outright. `llmwiki status` prints
what each lane resolved to:

```
provider  openai · gemini-3.7-flash
thinking  ask provider default · ingest low
```

### `.env` files

Those variables are read from a `.env` file if one is found, so credentials
don't have to be exported by hand in every shell. The search runs in this
order, and the first file to define a name wins:

```
$LLMWIKI_DOTENV                  explicit path; skips the rest
<project>/.env                   travels with the wiki
$PWD/.env, then each parent      stops below $HOME
~/.config/llmwiki/.env           user-global
```

A variable already exported in the shell is never overwritten, so `.env` is a
floor rather than a ceiling — `LLMWIKI_MODEL=x llmwiki ask …` still wins.
`LLMWIKI_DOTENV=0` turns loading off entirely.

The syntax is the intersection of what shells and dotenv libraries agree on:
`KEY=value`, an optional `export` prefix, `#` comments, single quotes
(literal), double quotes (with `\n`-style escapes), and `$VAR` / `${VAR}`
expansion against earlier lines and the real environment. A line that doesn't
parse is reported on stderr rather than skipped in silence. Multi-line values
and command substitution are not supported. `chmod 600` the file if it holds a
key; `.env` should be in your `.gitignore`.

`LLMWIKI_STRICT_PROJECT` is deliberately excluded: it decides *which* project
to open, so it can't be read from a file found by way of the project.

The Anthropic lane uses the official SDK with adaptive thinking and
streaming. The OpenAI-compatible lane is stdlib-only and speaks
`/v1/chat/completions`, so it works against OpenAI, OpenRouter, Ollama,
LM Studio, and vLLM:

```bash
export LLMWIKI_PROVIDER=openai
export LLMWIKI_BASE_URL=http://localhost:11434/v1
export LLMWIKI_MODEL=qwen3:8b
```

## Vector search (optional)

Off by default; keyword + graph retrieval needs no API calls and no
dependencies. To enable it:

```bash
export LLMWIKI_EMBEDDING_MODEL=text-embedding-3-small
llmwiki embed
```

Chunk vectors live in a single SQLite file (`.llm-wiki/vectors.db`), scored by
brute-force cosine similarity — fast enough at personal scale, and portable.
New pages are embedded automatically as they're ingested. If the embedding
endpoint is unreachable at query time, retrieval degrades to keyword + graph
rather than failing.

## Templates

`llmwiki init --template` picks the page types written into `schema.md`:
`research`, `reading`, `personal`, `business`, `general`. The schema is the
authoritative routing rule — add a row and a directory, and ingest starts
filing pages there. Nothing else hardcodes the type list.

## Claude Code skills

`.claude/skills/` ships three skills so an agent can drive the CLI without
being told how each time:

| Skill | Covers |
|---|---|
| `llmwiki-add-document` | one document — preflight, formats, reading the output, what to fix when pages come out wrong |
| `llmwiki-add-directory` | a corpus — scoping, piloting before a long run, `-r`, resumability, ingest order |
| `llmwiki-ask` | querying — phrasing for retrieval, reading citations, diagnosing thin answers with `search` |

They load automatically when Claude Code runs inside this repository. To use
them against wikis anywhere on the machine:

```bash
ln -s "$PWD/.claude/skills/llmwiki-"* ~/.claude/skills/
```

## Library use

```python
from llmwiki import config, ingest_document, ask, open_project

project = open_project("~/wikis/energy")
settings = config.load(project.root)

result = ingest_document(project, "paper.pdf", settings)
print(result.pages, result.reviews)

answer = ask(project, "What limits round-trip efficiency?", settings)
print(answer.text)
for citation in answer.citations:
    print(citation.number, citation.path)
```

## Differences from llm_wiki

Kept: the three-layer architecture, two-step chain-of-thought ingest, the
`---FILE:---` protocol and its parser hazards, path sandboxing, SHA256
caching, deterministic index/log maintenance, keyword scoring weights, RRF
fusion, graph expansion and the 4-signal relevance model, and the character
budget allocator.

Changed:

- **LanceDB → SQLite.** No native dependency; brute-force cosine instead of
  ANN, which is the right trade at personal scale.
- **Raw sources are damped, not separated.** llm_wiki puts sources behind a
  separate tool; here they share one ranking at 0.6× so the compiled wiki
  stays the answer layer while sources remain reachable.
- **No `temperature`.** The current Anthropic models reject sampling
  parameters, so determinism comes from the prompts.
- **Not ported:** the desktop UI, multimodal image captioning, deep research
  and web search, the Chrome clipper, MinerU, the HTTP API and MCP server,
  Louvain community detection, and the tool-using chat agent.

## License

MIT.
