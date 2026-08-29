"""Retrieval: lexical ranking, a link/mention graph, and the fused pipeline.

Two entry points, and the difference between them is the whole shape of this
package's public surface:

    search(project, query, ...)   the project entry point — opens the corpus
                                  from disk, opens its vector store, ranks
    search_index(index, query)    the ranking ladder, over any `CorpusIndex`

`search` is what the CLI and the eval harness call and its signature is fixed.
`search_index` is what a consumer with its own corpus calls: implement
`CorpusIndex` over whatever holds the documents, hand it over, and the BM25 /
vector / RRF / PPR ladder runs unchanged. `conformance.py` is how an
implementation checks that it is a legal one.
"""

from .calibration import ABSTAIN_QUANTILE, LexicalCalibration, calibrate
from .conformance import assert_corpus_index, check_corpus_index
from .entities import (
    EntityIndex,
    build_entity_index,
    hub_entities,
    prune_hubs,
    surface_forms,
)
from .graph import WikiGraph, build_graph, normalize_alias, related_pages, relevance
from .index import (
    CorpusIndex,
    InMemoryIndex,
    SearchIndex,
    build_index,
    clear_cache,
    open_index,
)
from .keyword import Document, SearchResult, load_documents
from .lexical import LexicalIndex, LexicalSearcher, extract_headings
from .log import QueryLog, QueryRecord, open_log
from .naming import DEFAULT_NAMING, DocumentNaming
from .pipeline import (
    LanesRun,
    QueryEmbedder,
    RetrievalOptions,
    SearchResponse,
    VectorSearcher,
    search,
    search_index,
)
from .ppr import personalized_pagerank, rank_by_ppr, transitions
from .profiles import DEFAULT_PROFILE, PROFILES, Profile, resolve as resolve_profile
from .telemetry import NULL_SINK, Deadline, RecordingSink, Sink
from .tokenize import tokenize_query

__all__ = [
    "ABSTAIN_QUANTILE",
    "Document",
    "EntityIndex",
    "LanesRun",
    "LexicalCalibration",
    "DEFAULT_PROFILE",
    "LexicalIndex",
    "PROFILES",
    "Profile",
    "QueryLog",
    "QueryRecord",
    "RetrievalOptions",
    "SearchIndex",
    "SearchResponse",
    "SearchResult",
    "Deadline",
    "NULL_SINK",
    "RecordingSink",
    "Sink",
    "WikiGraph",
    "build_entity_index",
    "build_graph",
    "build_index",
    "calibrate",
    "clear_cache",
    "load_documents",
    "open_index",
    "open_log",
    "personalized_pagerank",
    "rank_by_ppr",
    "related_pages",
    "resolve_profile",
    "relevance",
    "search",
    "tokenize_query",
    # ── added in 0.2.0 ───────────────────────────────────────────────────
    # Appended rather than merged into the sorted body above: two consumers in
    # two other repositories read this surface, and a reordered `__all__` is a
    # noisy diff on the one file where a reviewer most needs a quiet one.
    "CorpusIndex",
    "DEFAULT_NAMING",
    "DocumentNaming",
    "InMemoryIndex",
    "LexicalSearcher",
    "QueryEmbedder",
    "VectorSearcher",
    "assert_corpus_index",
    "check_corpus_index",
    "extract_headings",
    "hub_entities",
    "normalize_alias",
    "prune_hubs",
    "search_index",
    "surface_forms",
    "transitions",
]
