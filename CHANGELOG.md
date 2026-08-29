# Changelog

Two consumers in two other repositories read this package's public surface, so a
version bump with no changelog is a contract change with no notice.

The rule this project holds itself to: **changes to the public API are additive.**
Nothing in `llmwiki.retrieval.__all__` is renamed, removed or reordered, and no
existing function's parameters are renamed, removed or reordered. New names, and
new keyword parameters with defaults, are free.

## 0.2.0

The retrieval half becomes usable on a corpus this package did not build.
**No existing call site changes, and no ranking changes** — the eval suite
produces byte-identical rankings across all eight profiles.

### Added

- **`CorpusIndex`** (`llmwiki.retrieval`) — a `Protocol` for the surface the
  ranking ladder already used. Implement it over a database, an object store or
  a per-tenant index and the ladder ranks it unchanged.
- **`search_index(index, query, *, ...)`** — the ladder, with no filesystem
  opinion. `search()` is now a wrapper that opens a project's corpus and vector
  store and calls it.
- **`InMemoryIndex`** — the concrete class extracted from `SearchIndex`, which
  survives as an alias.
- **`VectorSearcher`** — the vector lane takes an injected searcher instead of
  deriving `project.state_dir / "vectors.db"`. `search()` still builds the
  default, under exactly the conditions the lane used to test inline.
- **`LexicalSearcher`** — the lexical lane's surface, so a corpus with a
  persisted FTS5 table need not rebuild an in-memory copy of it.
- **`DocumentNaming`** / `DEFAULT_NAMING` — the five naming conventions
  retrieval used to hold unconditionally (`key`, `aliases`, `surface_forms`,
  `title_field`, `is_page`), as one strategy object. `build_index`,
  `open_index`, `build_graph`, `build_entity_index`, `surface_forms` and
  `LexicalIndex` take a `naming=` keyword; the default reproduces today
  verbatim.
- **`llmwiki.retrieval.conformance`** — `check_corpus_index` /
  `assert_corpus_index`, so a second implementation learns the rules from
  assertions rather than from breaking.
- **Index-time primitives named as API**: `chunk_markdown`, `ChunkingOptions`,
  `SourceChunk`, `split_source_into_semantic_chunks` from the package root;
  `extract_headings`, `surface_forms`, `normalize_alias`, `prune_hubs`,
  `hub_entities`, `transitions` from `llmwiki.retrieval`.
- **`hub_entities(postings, document_count)`** — hub pruning's decision without
  an `EntityIndex`, for a consumer whose mention table lives elsewhere.
- **`MAX_QUERY_TOKENS`** and `tokenize_query(query, max_tokens=...)`.
- **`.github/workflows/check.yml`** — the mirror of the harness's cross-repo
  gate, triggered from the side that usually breaks it.
- `docs/corpus-index.md`, and a README section on using the engine standalone.

### Fixed

- **Query token count was unbounded.** Every token becomes a term in an FTS5
  `OR` chain matched against every document: a 20,000-word paste measured 200 ms
  of FTS5 work per query, linear in word count, against a measured maximum of
  **9** tokens across 44 real questions. Now capped at 64 (`max_tokens=0`
  lifts it), which is inert on every real query — rankings are unchanged.

  ⚠️ Recorded because it looks true and is not: the CJK bigram expansion is
  **not** an amplifier. It multiplies characters, but the token list is
  deduplicated, so a 4,000-character repetition collapses to 9 tokens.

  There is no injection surface, and this release does not claim to have closed
  one. Tokenisation already stripped every ASCII punctuation character, quoting
  already made bare `AND`/`OR`/`NEAR` literals, and the expression already
  reached SQLite as a bound parameter. `tests/test_query_hardening.py` is the
  executable form of that read.

### Unchanged, deliberately

`search()`'s signature and parameter order · `embed_query`'s module, name and
signature (it is patched from outside the package) · `Document`'s fields ·
`open_index`'s corpus-fingerprint cache · `dependencies = []`.

## 0.1.0

Initial extraction of the wiki-building and retrieval pipeline from `llm_wiki`.
