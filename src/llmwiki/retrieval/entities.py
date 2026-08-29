"""L4: a high-recall structural graph, at zero LLM cost.

The insight that makes this cheap is that **the entity dictionary already
exists.** `build_graph` assembles an alias table from every page's path,
wiki-relative path, stem and title; those aliases are curated by the ingest
model, so they are strictly more precise than the spaCy NER that LinearRAG and
SPRIG use for the same purpose — and they cost nothing at query time.

What the layer adds over the curated wikilink graph (L5) is *recall*. A page
that mentions "Helios Propulsion" in prose without writing `[[Helios
Propulsion]]` has no L5 edge at all, and on a corpus that was not authored as a
wiki — a directory of PDFs, a public QA benchmark — L5 is empty entirely. L4 is
built from text, so it is always there.

Edge weight is SPRIG's, computed at query time from stored counts:

    w(e, d) = tf(e, d) · log((N + 1) / (df(e) + 1)) + 1

Only `count` is stored. N and df are aggregates over the mention table, so they
are always current and never need invalidating.

**The graph is bipartite, and that is not a detail.** Entity nodes are distinct
from document nodes even though every entity here happens to be named by a page.
Collapsing the two was tried and is wrong: it makes a page compete with the
documents that mention it for the same diffused mass, so a source that quotes a
page outranks the page, and a page one link from the top hit outranks the second
lexical match. Routing mass doc -> entity -> doc keeps the entity a conduit
rather than a candidate, which is the shape both LinearRAG and SPRIG use and the
reason a two-hop bridge works at all.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sized
from dataclasses import dataclass, field

from .keyword import Document
from .naming import DEFAULT_NAMING, DocumentNaming

# A surface form shorter than this matches too much to be an entity: two-letter
# page stems ("AI", "ML") appear inside ordinary prose constantly, and the hub
# pruning below would only remove them after they had already cost a scan.
MIN_SURFACE_LENGTH = 3

# SPRIG prunes the highest-degree entities before diffusion: an alias appearing
# in nearly every document carries no information about which document is
# relevant, and it is exactly the node that makes PageRank concentrate mass in
# the wrong place. Their measurement was 485s -> 350s with negligible recall
# change; ours is about correctness more than speed.
HUB_PRUNE_FRACTION = 0.01
HUB_DOCUMENT_SHARE = 0.5

# The strongest mention edges a single document keeps. A page in a well-linked
# wiki mentions dozens of other page titles in passing, and every one of them
# takes a share of that page's diffused mass — so an unpruned mention graph
# washes out the sparse, high-precision curated links it is supposed to
# supplement. Measured on the atlas suite: unpruned, adding mention edges to the
# link graph *cost* 0.06 recall@5; pruned to the strongest few, it gains.
# Sparsification is what LinearRAG and SPRIG do for the same reason.
MAX_MENTIONS_PER_DOCUMENT = 4

# Entity node ids share a namespace with document ids inside the adjacency, so
# they carry a prefix no corpus-relative path can produce.
ENTITY_PREFIX = "@entity/"

# Credited to a page for its own title when its text never spells it out. Every
# entity needs an edge to the document that defines it, or a query that reaches
# the entity has nowhere to send the mass.
DEFINITION_TF = 1

# Compiling one alternation of every surface form is far faster than looping
# over aliases, but a single pattern with thousands of branches hits the regex
# engine's limits, so the scan runs in batches.
_BATCH = 400


@dataclass
class EntityIndex:
    """Which documents mention which pages, and how often.

    `mentions[entity][doc_id] = count`, where the entity key is the path of the
    page that names it. A page mentioning its own title is kept rather than
    skipped: that self-mention *is* the definition edge, and it arrives with a
    measured frequency instead of an invented constant.
    """

    mentions: dict[str, dict[str, int]] = field(default_factory=dict)
    documents: int = 0
    pruned: tuple[str, ...] = ()

    def weight(self, entity: str, doc_id: str) -> float:
        """SPRIG's edge weight, from counts held and aggregates computed."""
        postings = self.mentions.get(entity)
        if not postings:
            return 0.0
        count = postings.get(doc_id, 0)
        if not count:
            return 0.0
        idf = math.log((self.documents + 1) / (len(postings) + 1))
        return count * idf + 1.0

    def edges(self, per_document: int = MAX_MENTIONS_PER_DOCUMENT) -> dict[str, dict[str, float]]:
        """The mention graph as an undirected weighted bipartite adjacency.

        Undirected on purpose. The forward direction ("this document mentions
        that entity") finds context for a hit; the reverse ("which documents
        mention this entity") is the one that answers a bridge question, and it
        is the direction a link graph almost never records.

        Kept sparse: each document keeps only its `per_document` strongest
        mentions, because diffusion divides a node's mass among its edges and a
        long tail of incidental mentions is how a graph lane stops working.
        """
        strongest: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for entity, postings in self.mentions.items():
            for doc_id in postings:
                weight = self.weight(entity, doc_id)
                if weight > 0:
                    strongest[doc_id].append((weight, ENTITY_PREFIX + entity))

        adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        for doc_id, candidates in strongest.items():
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for weight, node in candidates[: max(1, per_document)]:
                adjacency[doc_id][node] = max(adjacency[doc_id].get(node, 0.0), weight)
                adjacency[node][doc_id] = max(adjacency[node].get(doc_id, 0.0), weight)
        return dict(adjacency)


def surface_forms(
    document: Document, naming: DocumentNaming = DEFAULT_NAMING
) -> set[str]:
    """The strings that, appearing in someone else's prose, mean this page.

    On a wiki: title and stem, plus the stem with hyphens read as spaces —
    `[[station keeping]]` and `station-keeping.md` are the same entity and a
    corpus will use both spellings. Which strings those are is the corpus's
    naming convention, so it comes from `naming.surface_forms`; a corpus keyed
    on opaque ids passes one yielding the title alone, or the scan would hunt
    for uuids in prose.

    ⛔ Not `naming.aliases`. An alias is what a *link* may be written as and is
    addressed deliberately; a surface form is what an author types in a
    sentence without meaning to link. `wiki/station-keeping.md` is a legitimate
    link target and would be absurd to scan prose for.

    Forms shorter than `MIN_SURFACE_LENGTH` are dropped whatever the naming
    says: that is a property of scanning free text, not of the corpus.
    """
    forms = naming.surface_forms(document)
    return {
        form.strip()
        for form in forms
        if form and len(form.strip()) >= MIN_SURFACE_LENGTH
    }


def build_entity_index(
    documents: list[Document], naming: DocumentNaming = DEFAULT_NAMING
) -> EntityIndex:
    """Scan every document for every known page name."""
    pages = [document for document in documents if naming.is_page(document)]
    if not pages:
        return EntityIndex(documents=len(documents))

    by_form: dict[str, str] = {}
    for page in pages:
        for form in surface_forms(page, naming):
            # First writer wins, and pages are in load order, so the mapping is
            # deterministic across runs even when two pages share a stem.
            by_form.setdefault(form.lower(), naming.key(page))

    mentions: dict[str, dict[str, int]] = defaultdict(dict)
    forms = sorted(by_form, key=len, reverse=True)
    patterns = [
        re.compile(
            "|".join(_bounded(re.escape(form)) for form in forms[start : start + _BATCH]),
            re.IGNORECASE,
        )
        for start in range(0, len(forms), _BATCH)
    ]

    for document in documents:
        counts: dict[str, int] = defaultdict(int)
        for pattern in patterns:
            for match in pattern.finditer(document.content):
                entity = by_form.get(match.group(0).lower())
                if entity:
                    counts[entity] += 1
        key = naming.key(document)
        for entity, count in counts.items():
            mentions[entity][key] = count

    # An entity whose page never writes its own name still has to be anchored to
    # it, or mass that reaches the entity has nowhere to go.
    for page in pages:
        page_key = naming.key(page)
        mentions[page_key].setdefault(page_key, DEFINITION_TF)

    index = EntityIndex(mentions=dict(mentions), documents=len(documents))
    return prune_hubs(index)


def _bounded(escaped: str) -> str:
    """Word boundaries only where the form actually begins or ends in a word.

    Without this, `art` matches inside `cartesian` — which is the substring bug
    the lexical lane was rewritten to remove, and it would come straight back in
    through the entity scan.
    """
    prefix = r"\b" if escaped[:1].isalnum() or escaped.startswith("\\") else ""
    suffix = r"\b" if escaped[-1:].isalnum() else ""
    return f"{prefix}(?:{escaped}){suffix}"


def hub_entities(
    postings: Mapping[str, Sized], document_count: int
) -> set[str]:
    """Which entities appear nearly everywhere — the decision, on its own.

    Two rules, because a small corpus and a large one fail differently: the top
    1% by degree (SPRIG's rule, which needs a population to be meaningful), and
    anything mentioned by more than half the corpus regardless of rank — which
    is the case that actually bites on a hundred documents.

    Takes the posting lists rather than an `EntityIndex` so that a corpus which
    stores its mention table somewhere else can ask the same question without
    first materialising an index it does not otherwise need. `prune_hubs` is
    this plus the bookkeeping.
    """
    if not postings:
        return set()

    ranked = sorted(postings.items(), key=lambda item: (-len(item[1]), item[0]))
    cut = int(len(ranked) * HUB_PRUNE_FRACTION)
    ceiling = max(2, int(document_count * HUB_DOCUMENT_SHARE))

    pruned = {entity for entity, _ in ranked[:cut]}
    pruned |= {entity for entity, posting in ranked if len(posting) > ceiling}
    return pruned


def prune_hubs(index: EntityIndex) -> EntityIndex:
    """Drop the entities `hub_entities` names, recording which."""
    if not index.mentions:
        return index

    pruned = hub_entities(index.mentions, index.documents)
    if not pruned:
        return index
    return EntityIndex(
        mentions={e: p for e, p in index.mentions.items() if e not in pruned},
        documents=index.documents,
        pruned=tuple(sorted(pruned)),
    )


# The name before it was published. Kept so an internal caller does not have to
# change in the same commit that widens the surface.
_prune_hubs = prune_hubs


__all__ = [
    "ENTITY_PREFIX",
    "MAX_MENTIONS_PER_DOCUMENT",
    "EntityIndex",
    "build_entity_index",
    "surface_forms",
    "HUB_DOCUMENT_SHARE",
    "HUB_PRUNE_FRACTION",
    "hub_entities",
    "prune_hubs",
]
