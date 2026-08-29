# `CorpusIndex` — ranking a corpus this package did not build

> Added in **0.2.0**. Everything here is additive: `search()`, `open_index()`,
> `SearchIndex` and every existing signature are unchanged.

## Why this exists

The ranking ladder — BM25, optional vector fusion by reciprocal rank, seeded
personalized PageRank over a link/mention graph — has never had a filesystem
opinion. `pipeline.search()` did: it took a `Project`, walked a directory, and
built every derived structure in memory per query.

That is the right shape for a personal wiki of a few hundred markdown files. It
is the wrong shape for a corpus that is already indexed somewhere — a database,
an object store, a per-tenant SQLite file — because the only way to use the
ladder was to project that corpus back into a directory, or to reimplement the
ladder. Both have been done, and the second one is what this document exists to
stop.

⇒ `search_index(index, query)` is the ladder over a `CorpusIndex`. `search()` is
now a thin wrapper that opens the project's corpus and its vector store and
calls it.

## The protocol

```python
from llmwiki.retrieval import CorpusIndex
```

| member | type | contract |
|---|---|---|
| `documents` | `list[Document]` | every retrievable unit |
| `graph` | `WikiGraph \| None` | reported on the response; **not** what diffusion reads |
| `entities` | `EntityIndex \| None` | reported on the response; **not** what diffusion reads |
| `lexical` | `LexicalSearcher \| None` | `None` for a corpus with no term index |
| `build_seconds` | `float` | for the caller's stage map; `0.0` for a store |
| `by_path` | `property -> dict[str, Document]` | covers `documents` exactly; stable across calls |
| `adjacency(...)` | `dict[str, dict[str, float]]` | weighted edges; may be empty; weights strictly positive |
| `transitions(...)` | `dict[str, list[tuple[str, float]]]` | `adjacency` row-normalised — **every row sums to 1** |
| `calibration()` | `LexicalCalibration` | the abstention fence; must answer on an empty corpus |
| `close()` | `None` | idempotent |

⛔ **A `CorpusIndex` is one corpus and cannot widen its own scope.** There is no
filter, no tenant and no query-time selector anywhere in the protocol, and that
is deliberate. An implementation over a multi-tenant store is constructed
*already scoped* to one tenant by the layer that resolved access; a protocol
able to ask for a different corpus is an access-control bug no conformance test
can catch.

⚠️ **`graph` and `entities` are `| None` because they describe how the default
implementation derives its edges.** A store that holds its adjacency as rows has
neither, and is still a legal corpus. Diffusion reads `adjacency()` and
`transitions()`; those two are the contract, and the other two are reporting.

⚠️ **`transitions` is not optional to get right.** PPR conserves mass only if
every row is a probability distribution. If your edges are already stored, the
easiest correct implementation is to delegate:

```python
from llmwiki.retrieval import transitions

def transitions(self, *, self_weight=0.15, **kwargs):
    return transitions(self.adjacency(**kwargs), self_weight)
```

## The conformance kit

A published protocol that ships no conformance kit is a protocol whose second
implementation discovers its rules by breaking.

```python
from llmwiki.retrieval.conformance import assert_corpus_index, check_corpus_index

def test_my_store_is_a_corpus_index():
    assert_corpus_index(MyCorpus(store), name="MyCorpus")
```

`check_corpus_index` returns the failures as a list instead of raising, for a
caller reporting several at once. Both read the index and never write to it, so
they are safe — and cheap enough — to run against a real store rather than a
two-document fixture.

`symmetric_adjacency=True` adds the reverse-edge check. It is **opt-in**: the
default implementation's edges are symmetric, but a corpus whose edges come from
directed references is still legal and PPR does not require symmetry. A kit that
asserted it unconditionally would be describing the default implementation
rather than the protocol.

## `DocumentNaming` — the wiki assumptions, made passable

Retrieval derives four things from a document's *name* rather than its text.
Each was a statement about markdown files in a directory, held in eleven places,
none of which said so:

| hook | default | what it decides |
|---|---|---|
| `key` | `document.path` | the identity every lane joins on |
| `aliases` | path, wiki-relative path, stem, title | what a `[[wikilink]]` may resolve through |
| `surface_forms` | title, stem, stem with `-` as spaces | what, appearing in prose, counts as a mention |
| `title_field` | `f"{title} {stem with - as spaces}"` | the 10×-weighted FTS5 title column |
| `is_page` | `document.kind == "wiki"` | which documents are pages rather than raw evidence |

⚠️ **`aliases` and `surface_forms` are separate on purpose.** An alias is what a
*link* may be written as, and a link is addressed deliberately;
`wiki/station-keeping.md` is a legitimate link target. A surface form is what an
author types in a sentence without meaning to link at all, and scanning prose
for a file path finds nothing while costing a regex pass over the corpus.

### The worked non-wiki example

A corpus whose documents are keyed `{8-char id}_{name}`:

```python
naming = DocumentNaming(
    key=lambda d: d.path,                  # already an opaque id; nothing to strip
    title_field=lambda d: d.title,         # ⛔ NOT f"{title} {stem}"
    surface_forms=lambda d: {d.title},     # an id is not a phrase anyone writes
    is_page=lambda d: True,                # every document is first-class
    aliases=lambda d: {d.path, d.title},
)
index = build_index(documents, naming=naming)
```

⭐ **`title_field` is the one worth explaining, because the default actively
harms this corpus rather than merely not helping it.** Upstream appends the path
stem to a column weighted ten times the body. For `station-keeping.md` that is
right — the filename is a second spelling of the title. For
`3f9a1c04_quarterly-report` it injects `3f9a1c04` into the field that dominates
BM25, so a query containing a hex fragment retrieves an unrelated document by
its id. `tests/test_corpus_index.py::test_a_title_only_naming_keeps_an_id_out_of_the_lexical_title_column`
pins both directions.

## The vector lane

`search_index` takes a `VectorSearcher`, not a path:

```python
class VectorSearcher(Protocol):
    def search(self, vector: list[float], n: int) -> list[ChunkHit]: ...
    def count(self) -> tuple[int, int]: ...   # (documents covered, chunks)
```

Lifecycle belongs to whoever constructed it — `search_index` never opens or
closes a searcher it was handed. `search()` builds the project's own
`VectorStore` under exactly the conditions the lane used to test inline (the
lane is on, an embedder is configured *and* enabled, the file exists) and closes
it afterwards, so a project with no `vectors.db` still reaches `search_index`
with `vectors=None` and still gets the same note.

⚠️ `embed_query` is deliberately **not** injectable and is looked up on the
module at call time, because at least one consumer patches it from outside the
package to route embeddings remotely.

## Index-time primitives

The functions a second consumer would otherwise copy are public API as of 0.2.0:

| import from | names |
|---|---|
| `llmwiki` | `chunk_markdown`, `ChunkingOptions`, `split_source_into_semantic_chunks` |
| `llmwiki.retrieval` | `extract_headings`, `surface_forms`, `normalize_alias`, `build_entity_index`, `prune_hubs`, `hub_entities`, `transitions` |

`hub_entities(postings, document_count)` is `prune_hubs` with the bookkeeping
removed: it answers *"which entities appear nearly everywhere"* from posting
lists alone, so a corpus holding its mention table in a database can ask without
first materialising an `EntityIndex` it does not otherwise need.

⛔ **Chunk boundaries and alias sets are persisted by anything that stores an
index.** Adopting these functions in place of a copy changes what is *written*,
which no retrieval-side comparison can observe. Bump your index format version
in the same change.

## Queries are bounded

`tokenize_query` caps a query at `MAX_QUERY_TOKENS` (64) distinct terms. Query
text is the one unbounded input on the ranking path — every token becomes a term
in an FTS5 `OR` chain matched against every document — and a 20,000-word paste
measured 200 ms of FTS5 work against a real-question maximum of 9 tokens. Pass
`max_tokens=0` to lift it.

There is no injection surface: tokenisation strips every ASCII punctuation
character, so no token can contain an FTS5 metacharacter; `match_expression`
quotes each term, which makes bare `AND`/`OR`/`NEAR` literals; and the
expression reaches SQLite as a bound parameter. `tests/test_query_hardening.py`
holds the adversarial corpus.

## Compatibility

| name | status |
|---|---|
| `search(project, query, ...)` | unchanged, including parameter order |
| `open_index`, `build_index`, `clear_cache` | unchanged; both gained a `naming=` keyword with today's default |
| `SearchIndex` | **alias** of `InMemoryIndex`; kept because consumers annotate on it |
| `embed_query` | untouched — module, name, signature |
| `Document` | no new required fields |
| `dependencies` | still `[]` |
