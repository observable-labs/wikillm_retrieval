"""The derived structures one query needs, built once and reused.

Before this, every query re-read every file, re-parsed every PDF, and rebuilt the
whole link graph. That is linear in corpus size per query and it dominated the
latency measurement badly enough to mask every other change — build-plan step 0
exists precisely to find that term, and step 4 to remove it.

The cache is keyed on a **fingerprint of the corpus** (path, mtime, size for
every file), not on time or on a flag. So an edit invalidates it on the next
query with no explicit invalidation call anywhere, a rename invalidates it, and
two processes cannot disagree about whether it is fresh. Recomputing the
fingerprint is a directory walk and a `stat` per file; rebuilding is a read and a
parse per file. The ratio is the whole win.

This is the in-process form of build-plan steps 4 and 6. The on-disk form —
`.llm-wiki/index.db`, step 2 — is a larger change to ingest and is still open;
what is here gets the latency and leaves the ranking identical, which was step 4
and 6's stated acceptance criterion.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .calibration import LexicalCalibration, calibrate
from .entities import MAX_MENTIONS_PER_DOCUMENT, EntityIndex, build_entity_index
from .graph import WikiGraph, build_graph, relevance
from .keyword import Document, load_documents
from .lexical import LexicalIndex
from .ppr import DEFAULT_SELF_WEIGHT, transitions

# How much a text mention is worth against a curated wikilink, before row
# normalization. A link is an author's assertion that two pages are related; a
# mention is an inference from prose. Given equal weight the mentions win by
# sheer number — a page mentions far more titles than it links — and on a
# well-linked wiki that measurably *cost* recall. Scaling makes L4 a fallback
# rather than a competitor, and it is free on a corpus with no links at all,
# where diffusion row-normalizes any global scale away.
#
# Measured on the atlas suite, recall@5 / MRR: 0.0 -> 1.00/0.86, 0.05 ->
# 1.00/0.89, 0.25 -> 0.98/0.91, 1.0 -> 0.94/0.91. The optimum is flat between
# 0.02 and 0.05; past that recall is traded for rank position. One corpus tuned
# one constant, and the value should be re-derived on a second linked corpus
# before it is trusted as more than a sensible default.
MENTION_SCALE = 0.05

# Two entries would do — a project is usually queried with sources on or off,
# and both are hot in an eval run — but an ablation sweep opens a few more.
# Beyond this, cached indexes hold whole corpora alive for no gain.
_CACHE_LIMIT = 4
_cache: "dict[tuple[str, bool], tuple[str, SearchIndex]]" = {}


@dataclass
class SearchIndex:
    """Documents plus every structure derived from them."""

    documents: list[Document]
    graph: WikiGraph
    entities: EntityIndex
    lexical: LexicalIndex | None = None
    build_seconds: float = 0.0
    fingerprint: str = ""
    _adjacency: dict = field(default_factory=dict, repr=False)
    _transitions: dict = field(default_factory=dict, repr=False)
    _by_path: dict = field(default_factory=dict, repr=False)
    _calibration: LexicalCalibration | None = field(default=None, repr=False)

    @property
    def by_path(self) -> dict[str, Document]:
        # Rebuilt per call this was O(corpus) twice per query, which on a corpus
        # of any size is most of what the query costs.
        if not self._by_path:
            self._by_path = {document.path: document for document in self.documents}
        return self._by_path

    def adjacency(
        self,
        entity_edges: bool = True,
        curated_links: bool = True,
        mentions_per_document: int = MAX_MENTIONS_PER_DOCUMENT,
        mention_scale: float = MENTION_SCALE,
    ) -> dict[str, dict[str, float]]:
        """The graph PPR diffuses over: curated links plus text mentions.

        One graph, two edge types, per build-plan step 8. L5 edges are weighted
        by `relevance()` — direct link, shared source, Adamic-Adar, type
        affinity — which is where a four-signal relatedness model belongs and is
        the first time it has been reachable from `search()` at all. L4 edges
        carry SPRIG's tf-idf mention weight.

        Cached per edge-type selection because it is the same for every query.
        """
        key = (entity_edges, curated_links, mentions_per_document, mention_scale)
        if key in self._adjacency:
            return self._adjacency[key]

        merged: dict[str, dict[str, float]] = defaultdict(dict)
        if curated_links:
            for path, neighbors in self.graph.adjacency.items():
                for neighbor in neighbors:
                    weight, _ = relevance(self.graph, path, neighbor)
                    if weight > 0:
                        merged[path][neighbor] = weight

        if entity_edges:
            for path, neighbors in self.entities.edges(mentions_per_document).items():
                for neighbor, weight in neighbors.items():
                    current = merged[path].get(neighbor, 0.0)
                    merged[path][neighbor] = max(current, weight * mention_scale)

        resolved = {node: dict(edges) for node, edges in merged.items() if edges}
        self._adjacency[key] = resolved
        return resolved

    def transitions(
        self,
        entity_edges: bool = True,
        curated_links: bool = True,
        mentions_per_document: int = MAX_MENTIONS_PER_DOCUMENT,
        self_weight: float = DEFAULT_SELF_WEIGHT,
        mention_scale: float = MENTION_SCALE,
    ) -> dict[str, list[tuple[str, float]]]:
        """The row-normalized adjacency, cached. See `ppr.transitions`."""
        key = (entity_edges, curated_links, mentions_per_document, self_weight, mention_scale)
        if key not in self._transitions:
            self._transitions[key] = transitions(
                self.adjacency(
                    entity_edges, curated_links, mentions_per_document, mention_scale
                ),
                self_weight,
            )
        return self._transitions[key]

    def calibration(self) -> LexicalCalibration:
        """What a well-aimed query scores here — the abstention fence. Cached.

        Beside the index rather than in the pipeline because it is a property of
        the corpus, it costs a few hundred FTS5 queries to build, and it has
        exactly the same lifetime as the index it is derived from: the corpus
        fingerprint that invalidates one invalidates the other, so the fence can
        never be stale with respect to the documents it was measured on.

        Lazy rather than built eagerly, because a caller that never fuses two
        lanes never needs it and should not pay 150 ms for it.
        """
        if self._calibration is None:
            self._calibration = calibrate(self.documents, self.lexical)
        return self._calibration

    def close(self) -> None:
        if self.lexical is not None:
            self.lexical.close()


def build_index(documents: list[Document], lexical: bool = True) -> SearchIndex:
    """Build every derived structure from documents already in memory."""
    started = time.perf_counter()
    index = SearchIndex(
        documents=documents,
        graph=build_graph(documents),
        entities=build_entity_index(documents),
        lexical=LexicalIndex(documents) if lexical else None,
    )
    index.build_seconds = time.perf_counter() - started
    return index


def open_index(
    project,
    include_sources: bool = True,
    documents: list[Document] | None = None,
    lexical: bool = True,
    use_cache: bool = True,
) -> SearchIndex:
    """The cached index for a project, rebuilt when the corpus has changed.

    Passing `documents` explicitly bypasses the cache: the caller has stated
    what the corpus is, and second-guessing that with a fingerprint of the disk
    would let the two disagree.
    """
    if documents is not None:
        return build_index(documents, lexical=lexical)

    key = (str(project.root), include_sources)
    if use_cache:
        current = corpus_fingerprint(project, include_sources)
        cached = _cache.get(key)
        if cached and cached[0] == current:
            return cached[1]

    index = build_index(load_documents(project, include_sources), lexical=lexical)
    if use_cache:
        index.fingerprint = current
        previous = _cache.pop(key, None)
        if previous:
            previous[1].close()
        _cache[key] = (current, index)
        while len(_cache) > _CACHE_LIMIT:
            _, evicted = _cache.pop(next(iter(_cache)))
            evicted.close()
    return index


def corpus_fingerprint(project, include_sources: bool = True) -> str:
    """Identity of the corpus on disk, cheaply.

    Size and mtime rather than content hash: hashing means reading, and reading
    is the cost being avoided. The failure mode — an edit preserving both size
    and mtime to the nanosecond — takes deliberate effort to produce.

    Deliberately not built on `Project.wiki_pages()`. That returns `Path`
    objects and filters with `relative_to`, and on a 2,000-document corpus the
    pathlib allocations alone were 70% of the query — more than retrieval,
    fusion and diffusion combined. This walk is `os.scandir`, whose `DirEntry`
    already carries the stat data the directory read returned.
    """
    parts: list[str] = []
    roots = [(project.wiki_dir, ".md")]
    if include_sources:
        roots.append((project.sources_dir, ""))
    for root, suffix in roots:
        _scan(str(root), suffix, parts)
    parts.sort()
    # sha256 rather than hash(): the built-in is salted per process, so two
    # processes would compute different fingerprints for the same corpus and a
    # persisted form of this cache could never be shared.
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"{len(parts)}|{digest[:16]}"


def _scan(directory: str, suffix: str, parts: list[str]) -> None:
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                _scan(entry.path, suffix, parts)
                continue
            if suffix and not entry.name.endswith(suffix):
                continue
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        parts.append(f"{entry.path}:{stat.st_size}:{stat.st_mtime_ns}")


def clear_cache() -> None:
    """Drop every cached index. For tests and for `reset()`."""
    while _cache:
        _, index = _cache.popitem()[1]
        index.close()


__all__ = [
    "MENTION_SCALE",
    "SearchIndex",
    "build_index",
    "clear_cache",
    "corpus_fingerprint",
    "open_index",
]
