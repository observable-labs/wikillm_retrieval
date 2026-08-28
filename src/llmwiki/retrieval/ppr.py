"""S2: personalized PageRank, *seeded from* the fused list.

This is a change of shape, not of weights, and the distinction is the whole
point. SPRIG's ablation:

    RRF alone                       HotpotQA R@10 0.851   2Wiki R@10 0.697
    PPR seeded from the RRF list                  0.867                0.794
    PPR scores blended into RRF                   0.782                0.602

The third row is *worse than having no graph at all*, and it is what
`blend_graph_results` implemented: reserve a quota of slots, score neighbours
with a query-blind `1/(rank+1)`, and let them displace the tail of the ranked
list. No setting of the quota fixes that, because the quota is not the problem —
treating the graph as a second opinion to make room for is.

Here the graph gets no slots. The fused list becomes a restart distribution,
mass diffuses over the union of the mention graph (L4) and the curated link
graph (L5), and the result *is* the ranking.

Two properties keep the lane safe to leave on:

* **No edges, no change.** With an empty adjacency every seed is dangling, its
  mass returns to the restart vector, and the ranking is the fused ranking.
* **Nothing is dropped.** Fused documents the graph never reaches are appended
  in fused order rather than falling out of the window, so the lane can only add
  candidates, never lose the ones retrieval already found.
"""

from __future__ import annotations

from collections import defaultdict

# SPRIG's tuning. `alpha` is the restart probability: each iteration returns 15%
# of the mass to the seeds, so a seed cannot be diffused into irrelevance and a
# neighbour two hops out still gets a share.
DEFAULT_ALPHA = 0.15

# SPRIG reports five iterations. Power iteration converges at rate (1 - alpha),
# so five leaves ~44% of the initial error — enough that a two-node component
# still oscillates between odd and even steps, and the ranking depends on which
# one you stopped at. Iterating to a tolerance costs microseconds on graphs this
# size and removes a parameter that was silently deciding results.
DEFAULT_ITERATIONS = 60
TOLERANCE = 1e-8

# How many of the fused results radiate. Seeding from the whole window instead
# concentrates mass on hubs and dilutes exactly the bridge signal the lane
# exists to find, which is why SPRIG restricts it to the head of the list.
DEFAULT_SEEDS = 5

# What the rest of the fused list gets, as a fraction of its own rank weight.
# Zero is SPRIG's literal reading: the tail does not radiate, and `rank_by_ppr`
# restores its order afterwards by appending it. A positive value carries that
# order through the diffusion instead. Measured on both eval corpora, the two
# do not separate, so the simpler reading wins.
DEFAULT_TAIL_WEIGHT = 0.0

# A self-loop on every node, as a multiple of that node's total edge weight.
# Without one, how much of its own mass a document keeps is decided entirely by
# its degree: a page with one strong link hands 85% of its evidence to its
# neighbour every iteration and sinks below documents that happen to be
# unconnected. With `self_weight = 1` a node keeps half of what it radiates,
# uniformly, so diffusion reorders on evidence rather than on connectivity.
DEFAULT_SELF_WEIGHT = 1.0


def transitions(
    adjacency: dict[str, dict[str, float]],
    self_weight: float = DEFAULT_SELF_WEIGHT,
) -> dict[str, list[tuple[str, float]]]:
    """Row-normalize once, so the query does not pay for it.

    Out-degree does not depend on the query, so this is the same map for every
    question against an unchanged corpus. Rebuilding it per query was most of
    what diffusion cost.
    """
    normalized: dict[str, list[tuple[str, float]]] = {}
    for node, neighbors in adjacency.items():
        weight_sum = sum(weight for weight in neighbors.values() if weight > 0)
        if weight_sum <= 0:
            continue
        total_out = weight_sum * (1.0 + max(0.0, self_weight))
        edges = [
            (neighbor, weight / total_out)
            for neighbor, weight in neighbors.items()
            if weight > 0
        ]
        if self_weight > 0:
            edges.append((node, weight_sum * self_weight / total_out))
        normalized[node] = edges
    return normalized


def personalized_pagerank(
    seeds: dict[str, float],
    adjacency: dict[str, dict[str, float]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    self_weight: float = DEFAULT_SELF_WEIGHT,
    tolerance: float = TOLERANCE,
    outgoing: dict[str, list[tuple[str, float]]] | None = None,
) -> dict[str, float]:
    """Sparse PPR by power iteration, run to convergence. No matrix, no numpy.

    `seeds` is a doc_id -> restart mass map, L1-normalized here so callers can
    pass raw scores. `adjacency` is weighted and taken as given — the caller
    decides whether an edge is a curated link or a text mention, and what either
    is worth.
    """
    total = sum(mass for mass in seeds.values() if mass > 0)
    if total <= 0:
        return {}
    restart = {node: mass / total for node, mass in seeds.items() if mass > 0}
    if outgoing is None:
        outgoing = transitions(adjacency or {}, self_weight)

    scores = dict(restart)
    for _ in range(max(1, iterations)):
        previous = scores
        spread: dict[str, float] = defaultdict(float)
        dangling = 0.0
        for node, mass in scores.items():
            edges = outgoing.get(node)
            if not edges:
                # A node with no out-edges would leak its mass out of the
                # system. Returning it to the restart vector is what keeps the
                # total at 1 and keeps an unlinked document from decaying away.
                dangling += mass
                continue
            for neighbor, share in edges:
                spread[neighbor] += mass * share

        updated: dict[str, float] = defaultdict(float)
        for node, mass in restart.items():
            updated[node] += alpha * mass + (1.0 - alpha) * dangling * mass
        for node, mass in spread.items():
            updated[node] += (1.0 - alpha) * mass
        scores = dict(updated)
        # The node set only grows, so iterating the new scores covers the union.
        drift = sum(
            abs(mass - previous.get(node, 0.0)) for node, mass in scores.items()
        )
        if drift < tolerance:
            break
    return scores


def seed_masses(
    fused: list[tuple[str, float]],
    seed_count: int = DEFAULT_SEEDS,
    tail_weight: float = DEFAULT_TAIL_WEIGHT,
) -> dict[str, float]:
    """Restart mass for the fused list: the head radiates, the tail may not.

    Mass is `1/rank`, not the fused score. RRF scores are `1/(60 + rank)` by
    construction, which is nearly flat — rank 1 and rank 2 differ by 2% — so
    using them as a restart distribution hands PPR almost no prior and lets a
    marginally better-connected document overturn a lexical margin of five
    orders of magnitude. Rank is the information RRF actually carries, and
    `1/rank` is what "weighted by fused rank" has to mean for the weighting to
    do anything.
    """
    masses: dict[str, float] = {}
    for rank, (doc_id, _score) in enumerate(fused, start=1):
        weight = 1.0 if rank <= max(1, seed_count) else tail_weight
        mass = weight / rank
        if mass > 0:
            masses[doc_id] = mass
    return masses


def rank_by_ppr(
    fused: list[tuple[str, float]],
    adjacency: dict[str, dict[str, float]] | None,
    limit: int,
    seed_count: int = DEFAULT_SEEDS,
    tail_weight: float = DEFAULT_TAIL_WEIGHT,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_ITERATIONS,
    self_weight: float = DEFAULT_SELF_WEIGHT,
    keep: set[str] | None = None,
    outgoing: dict[str, list[tuple[str, float]]] | None = None,
) -> list[tuple[str, float]]:
    """The fused list, re-ranked by diffusion. Returns (doc_id, score).

    Documents reached only through the graph enter here — that is what the lane
    is for — but they enter by earning mass, not by being handed a reserved slot
    at the tail's expense.

    `keep` is the set of nodes that may appear in the output. The adjacency is
    bipartite, so entity nodes carry mass through the diffusion and must not
    then occupy a slot in the result window.
    """
    if not fused:
        return []
    scores = personalized_pagerank(
        seed_masses(fused, seed_count=seed_count, tail_weight=tail_weight),
        adjacency,
        alpha=alpha,
        iterations=iterations,
        self_weight=self_weight,
        outgoing=outgoing,
    )
    if not scores:
        return fused[:limit]

    fused_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(fused, start=1)}
    ordered = sorted(
        (item for item in scores.items() if keep is None or item[0] in keep),
        key=lambda item: (-item[1], fused_rank.get(item[0], len(fused) + 1), item[0]),
    )
    # Anything retrieval found and diffusion did not reach keeps its place at the
    # back. The graph adds candidates; it never removes them.
    seen = {doc_id for doc_id, _ in ordered}
    ordered.extend((doc_id, 0.0) for doc_id, _ in fused if doc_id not in seen)
    return ordered[:limit]


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_ITERATIONS",
    "DEFAULT_SEEDS",
    "DEFAULT_SELF_WEIGHT",
    "DEFAULT_TAIL_WEIGHT",
    "TOLERANCE",
    "personalized_pagerank",
    "rank_by_ppr",
    "seed_masses",
    "transitions",
]
