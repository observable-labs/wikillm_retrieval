"""L5: the curated link graph, and the model of how two pages relate.

`build_graph` resolves every `[[wikilink]]` against an alias table built from
page paths, stems and titles, giving an undirected page-to-page graph. It is
high precision and low recall by construction — it contains exactly the edges a
writer chose to draw, and nothing a document merely mentions. `entities.py`
supplies the recall half.

`relevance` is the 4-signal model of how strongly two pages relate:

    direct wikilink       x3.0
    shared raw source     x4.0   (the strongest signal — same evidence)
    Adamic-Adar           x1.5   (shared neighbours, weighted by rarity)
    same page type        x1.0

It is the **edge weight** for the L5 half of the adjacency PPR diffuses over
(`index.SearchIndex.adjacency`). Until that wiring it was unreachable from
`search()` at all: expansion used a query-blind `1/(rank+1)` instead, and this
model was only ever reached by `related_pages` from the CLI.

What used to live here — `blend_graph_results` and `graph_result_quota`, which
reserved 15-30% of the result window for graph neighbours and let them displace
the tail of the ranked list — is gone. SPRIG measured that shape as *worse than
having no graph at all* (README §4.2), and the quota was never the parameter at
fault. `ppr.py` replaces it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .keyword import Document
from .naming import DEFAULT_NAMING, DocumentNaming

DIRECT_LINK_WEIGHT = 3.0
SHARED_SOURCE_WEIGHT = 4.0
ADAMIC_ADAR_WEIGHT = 1.5
TYPE_AFFINITY_WEIGHT = 1.0

# RRF's published constant is 60, chosen against TREC runs of 1,000 results —
# 6% of the list depth. These lanes rank 20 to 50, so 60 sits above the whole
# list and flattens it: rank 1 and rank 50 differ by less than a factor of two,
# and a lane that is merely adequate can then outvote one that is good. Rescaled
# to the same fraction of the depth actually fused, 6% of 50 is 3.
#
# Measured across both eval corpora, recall@k at 60 against 3:
#
#     hotpot   R@2  0.672 -> 0.730    R@5  0.912 -> 0.953
#     atlas    R@1  0.705 -> 0.727    R@5  0.920 -> 0.932
#
# The sweep is monotone between 60 and 3 on both and flat below 1, so the value
# is not on a cliff. `RetrievalOptions.rrf_k` keeps it swappable.
RRF_K = 3.0


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


def build_graph(
    documents: list[Document], naming: DocumentNaming = DEFAULT_NAMING
) -> WikiGraph:
    """The curated-link graph, over whichever documents `naming` calls pages.

    `naming` carries the three assumptions this function used to hold silently:
    which documents are pages at all, what key they are known by, and which
    strings a link may reach them through. Its default is the wiki convention,
    verbatim.
    """
    pages = {naming.key(doc): doc for doc in documents if naming.is_page(doc)}

    # A page is reachable by whichever aliases the corpus names it with; on a
    # wiki that is full path, wiki-relative path, stem, or title.
    aliases: dict[str, str] = {}
    for path, document in pages.items():
        for alias in naming.aliases(document):
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


__all__ = [
    "ADAMIC_ADAR_WEIGHT",
    "DIRECT_LINK_WEIGHT",
    "RRF_K",
    "SHARED_SOURCE_WEIGHT",
    "TYPE_AFFINITY_WEIGHT",
    "WikiGraph",
    "build_graph",
    "normalize_alias",
    "related_pages",
    "relevance",
]
