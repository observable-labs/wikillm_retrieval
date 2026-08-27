"""Phase 2: graph expansion over the wiki's link structure.

Two mechanisms, both from llm_wiki:

`expand_from_seeds` is what the search pipeline calls — top-ranked results
become seeds, their one-hop neighbours become candidates scored by
`1/(seed_rank+1)`, and the best of them take a reserved 15-30% slice of the
result window. This is what lets a query match a page that never mentions the
query terms, because a page that *does* mention them links to it.

`relevance` implements the 4-signal relevance model used for deeper
traversal and for explaining *why* two pages are related:

    direct wikilink       x3.0
    shared raw source     x4.0   (the strongest signal — same evidence)
    Adamic-Adar           x1.5   (shared neighbours, weighted by rarity)
    same page type        x1.0
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .keyword import Document, SearchResult

DIRECT_LINK_WEIGHT = 3.0
SHARED_SOURCE_WEIGHT = 4.0
ADAMIC_ADAR_WEIGHT = 1.5
TYPE_AFFINITY_WEIGHT = 1.0

MIN_GRAPH_RESULT_RATIO = 0.15
MAX_GRAPH_RESULT_RATIO = 0.30
MAX_GRAPH_SEEDS = 20
RRF_K = 60.0


def normalize_alias(value: str) -> str:
    """How a wikilink target is matched against a page path.

    `[[Chain of Thought]]`, `concepts/chain-of-thought`, and
    `wiki/concepts/chain-of-thought.md` all normalize to the same key.
    """
    head = value.split("#")[0].strip().replace("\\", "/")
    if head.lower().endswith(".md"):
        head = head[:-3]
    return head.replace(" ", "-").lower()


@dataclass
class WikiGraph:
    documents: dict[str, Document]           # path -> document
    adjacency: dict[str, set[str]]           # path -> linked paths (undirected)
    by_source: dict[str, set[str]]           # source identity -> paths

    def neighbors(self, path: str) -> set[str]:
        return self.adjacency.get(path, set())

    def degree(self, path: str) -> int:
        return len(self.adjacency.get(path, ()))


def build_graph(documents: list[Document]) -> WikiGraph:
    pages = {doc.path: doc for doc in documents if doc.kind == "wiki"}

    # A page is reachable by full path, wiki-relative path, stem, or title.
    aliases: dict[str, str] = {}
    for path, document in pages.items():
        wiki_relative = path[5:] if path.startswith("wiki/") else path
        for alias in (path, wiki_relative, document.stem, document.title):
            if alias:
                aliases[normalize_alias(alias)] = path

    adjacency: dict[str, set[str]] = defaultdict(set)
    for path, document in pages.items():
        for link in document.links:
            target = aliases.get(normalize_alias(link))
            if not target or target == path:
                continue
            adjacency[path].add(target)
            adjacency[target].add(path)

    by_source: dict[str, set[str]] = defaultdict(set)
    for path, document in pages.items():
        for source in document.sources:
            by_source[normalize_alias(source)].add(path)

    return WikiGraph(documents=pages, adjacency=dict(adjacency), by_source=dict(by_source))


def relevance(graph: WikiGraph, left: str, right: str) -> tuple[float, list[str]]:
    """4-signal relatedness between two pages, with the reasons that fired."""
    if left == right:
        return 0.0, []
    score = 0.0
    reasons: list[str] = []

    if right in graph.neighbors(left):
        score += DIRECT_LINK_WEIGHT
        reasons.append("direct link")

    left_doc = graph.documents.get(left)
    right_doc = graph.documents.get(right)
    if left_doc and right_doc:
        shared = {normalize_alias(s) for s in left_doc.sources} & {
            normalize_alias(s) for s in right_doc.sources
        }
        if shared:
            score += SHARED_SOURCE_WEIGHT
            reasons.append(f"shares source {sorted(shared)[0]}")
        if left_doc.page_type and left_doc.page_type == right_doc.page_type:
            score += TYPE_AFFINITY_WEIGHT
            reasons.append(f"both {left_doc.page_type}")

    common = graph.neighbors(left) & graph.neighbors(right)
    if common:
        # Adamic-Adar: a shared neighbour that links to everything says less
        # than a shared neighbour that links to almost nothing.
        adamic_adar = sum(
            1.0 / math.log(max(2, graph.degree(neighbor))) for neighbor in common
        )
        score += ADAMIC_ADAR_WEIGHT * adamic_adar
        reasons.append(f"{len(common)} shared neighbour(s)")

    return score, reasons


def related_pages(graph: WikiGraph, path: str, limit: int = 10) -> list[tuple[str, float, list[str]]]:
    """Rank every page against `path` by the 4-signal model."""
    candidates: set[str] = set(graph.neighbors(path))
    for neighbor in list(candidates):
        candidates |= graph.neighbors(neighbor)  # 2-hop
    document = graph.documents.get(path)
    if document:
        for source in document.sources:
            candidates |= graph.by_source.get(normalize_alias(source), set())
    candidates.discard(path)

    scored = []
    for candidate in candidates:
        score, reasons = relevance(graph, path, candidate)
        if score > 0:
            scored.append((candidate, score, reasons))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def graph_result_quota(limit: int, vector_hits: int) -> int:
    """Reserve 15-30% of the window for graph neighbours.

    A full vector window leaves the minimum share; sparse vector retrieval
    slides toward the maximum, because that's when link structure is doing
    the most work.
    """
    if limit < 2:
        return 0
    coverage = min(vector_hits, limit) / limit
    ratio = MAX_GRAPH_RESULT_RATIO - (MAX_GRAPH_RESULT_RATIO - MIN_GRAPH_RESULT_RATIO) * coverage
    return max(1, min(limit - 1, math.ceil(limit * ratio)))


def blend_graph_results(
    ranked: list[SearchResult],
    graph: WikiGraph,
    limit: int,
    vector_hits: int,
) -> tuple[list[SearchResult], int]:
    """Fold one-hop neighbours of the top results into the result window."""
    if not ranked or not graph.documents:
        return ranked[:limit], 0

    seeds = [result.path for result in ranked[: min(limit, MAX_GRAPH_SEEDS)]]
    seed_set = set(seeds)

    candidate_scores: dict[str, float] = defaultdict(float)
    candidate_seeds: dict[str, set[str]] = defaultdict(set)
    for rank, seed in enumerate(seeds):
        for neighbor in graph.neighbors(seed):
            if neighbor in seed_set:
                continue
            candidate_scores[neighbor] += 1.0 / (rank + 1)
            seed_document = graph.documents.get(seed)
            if seed_document:
                candidate_seeds[neighbor].add(seed_document.title)

    candidates = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    candidates = candidates[: graph_result_quota(limit, vector_hits)]
    if not candidates:
        return ranked[:limit], 0

    selected = {path for path, _ in candidates}
    existing = {result.path: result for result in ranked}

    # Graph picks take their slots from the tail of the keyword/vector list.
    base = [result for result in ranked if result.path not in selected][
        : max(0, limit - len(candidates))
    ]

    for path, graph_score in candidates:
        related = sorted(candidate_seeds.get(path, set()))
        if path in existing:
            promoted = existing[path]
            promoted.graph_related_to = related
            base.append(promoted)
            continue
        document = graph.documents.get(path)
        if not document:
            continue
        base.append(
            SearchResult(
                path=path,
                title=document.title,
                snippet=f"Graph neighbour of {', '.join(related)}" if related else "Graph neighbour",
                score=graph_score / (RRF_K + 1.0),
                kind="wiki",
                graph_related_to=related,
                document=document,
            )
        )
    return base, len(candidates)
