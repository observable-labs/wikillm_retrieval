"""Is this a legal `CorpusIndex`? — the rules, as assertions a second implementer runs.

A published protocol that ships no conformance kit is a protocol whose second
implementation discovers its rules by breaking. `CorpusIndex` has five members
whose contracts are entirely implicit in how `pipeline.search_index` uses them:
`by_path` must cover `documents` or fusion silently drops candidates, every
`transitions` row must sum to 1 or PPR leaks mass, `calibration()` must return
something on an empty corpus or the abstention gate raises on a cold store.
None of that is checkable by reading the protocol.

Usage, from the implementer's own test suite::

    from llmwiki.retrieval.conformance import assert_corpus_index

    def test_my_store_is_a_corpus_index():
        assert_corpus_index(MyCorpus(store), name="MyCorpus")

`check_corpus_index` returns the failures instead of raising, for a caller that
wants to report several at once or to accept a known one.

⛔ **The kit reads the index; it never writes one.** Every check here is a query
against whatever the caller handed over, so it is safe to run against a live
store — and it is deliberately cheap enough to run against a real one, because a
conformance suite that only ever sees a two-document fixture checks the fixture.

⚠️ **Adjacency symmetry is opt-in.** The default implementation's edges are
symmetric — a curated link and a mention both go both ways — but a corpus whose
edges come from directed references is still a legal corpus, and PPR does not
require symmetry. Pass `symmetric_adjacency=True` only if your implementation
claims it; a kit that asserted it unconditionally would be describing the
default implementation rather than the protocol.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONFORMANCE_CHECKS",
    "assert_corpus_index",
    "check_corpus_index",
]

#: What a conforming index must do, in the order the kit checks it. Named so a
#: report can say which rule failed rather than only that something did.
CONFORMANCE_CHECKS = (
    "satisfies-protocol",
    "by-path-covers-documents",
    "by-path-is-stable",
    "keys-are-unique-strings",
    "adjacency-is-finite-and-positive",
    "adjacency-nodes-are-known-or-declared",
    "transitions-rows-sum-to-one",
    "transitions-respects-self-weight",
    "calibration-is-usable",
    "close-is-idempotent",
)

# Row normalisation is float arithmetic over a variable number of edges, so an
# exact 1.0 is the wrong assertion; this is loose enough for a few thousand
# terms and far tighter than any error that would change a ranking.
_ROW_TOLERANCE = 1e-6


def check_corpus_index(
    index: Any,
    *,
    symmetric_adjacency: bool = False,
    sample_nodes: int = 200,
    close: bool = False,
) -> list[str]:
    """Every rule this index breaks, as sentences. Empty means conforming.

    `close` is off by default: most callers are checking an index they intend to
    keep using, and a kit that closed it out from under them would be a rule
    nobody could satisfy twice. Turn it on for the last check in a suite.
    """
    failures: list[str] = []

    def fail(check: str, detail: str) -> None:
        failures.append(f"{check}: {detail}")

    # ── the protocol itself ────────────────────────────────────────────────
    from .index import CorpusIndex

    if not isinstance(index, CorpusIndex):
        missing = [
            name
            for name in (
                "documents", "graph", "entities", "lexical", "build_seconds",
                "by_path", "adjacency", "transitions", "calibration", "close",
            )
            if not hasattr(index, name)
        ]
        fail("satisfies-protocol", f"missing {missing or 'nothing — check member types'}")
        return failures

    documents = list(index.documents)

    # ── by_path ────────────────────────────────────────────────────────────
    try:
        by_path = index.by_path
    except Exception as exc:  # noqa: BLE001 - the failure is the finding
        fail("by-path-covers-documents", f"by_path raised {exc!r}")
        return failures

    if len(by_path) != len(documents):
        fail(
            "by-path-covers-documents",
            f"{len(documents)} documents but {len(by_path)} keys — a document "
            "absent from by_path is retrievable by the lexical lane and then "
            "dropped by fusion, which reads as a ranking bug",
        )

    for key in by_path:
        if not isinstance(key, str) or not key:
            fail("keys-are-unique-strings", f"key {key!r} is not a non-empty str")
            break

    missing = [d for d in documents if by_path.get(_key_of(by_path, d)) is not d]
    if missing and len(by_path) == len(documents):
        fail(
            "by-path-covers-documents",
            f"{len(missing)} document(s) in `documents` are not the object "
            "`by_path` returns for their key",
        )

    if index.by_path is not by_path and index.by_path != by_path:
        fail(
            "by-path-is-stable",
            "two calls returned different mappings; every lane calls it and a "
            "key that moves between them fuses two rankings of different corpora",
        )

    # ── adjacency ──────────────────────────────────────────────────────────
    try:
        adjacency = index.adjacency()
    except Exception as exc:  # noqa: BLE001
        fail("adjacency-is-finite-and-positive", f"adjacency() raised {exc!r}")
        adjacency = {}

    checked = 0
    for node, edges in adjacency.items():
        for neighbour, weight in edges.items():
            if not isinstance(weight, (int, float)) or weight != weight or weight in (
                float("inf"), float("-inf")
            ):
                fail("adjacency-is-finite-and-positive", f"{node} -> {neighbour} is {weight!r}")
                break
            if weight <= 0:
                fail(
                    "adjacency-is-finite-and-positive",
                    f"{node} -> {neighbour} is {weight}; a non-positive edge is "
                    "dropped by row normalisation, so it is an edge that is not one",
                )
                break
            if symmetric_adjacency and adjacency.get(neighbour, {}).get(node) is None:
                fail(
                    "adjacency-nodes-are-known-or-declared",
                    f"{node} -> {neighbour} has no reverse edge, but this index "
                    "declared symmetric adjacency",
                )
                break
        checked += 1
        if checked >= sample_nodes:
            break

    # ── transitions ────────────────────────────────────────────────────────
    for self_weight in (0.0, 0.15):
        try:
            rows = index.transitions(self_weight=self_weight)
        except Exception as exc:  # noqa: BLE001
            fail("transitions-rows-sum-to-one", f"transitions() raised {exc!r}")
            break
        for count, (node, edges) in enumerate(rows.items()):
            total = sum(weight for _, weight in edges)
            if abs(total - 1.0) > _ROW_TOLERANCE:
                fail(
                    "transitions-rows-sum-to-one",
                    f"row {node!r} sums to {total!r} at self_weight={self_weight}; "
                    "PPR conserves mass only if every row is a distribution",
                )
                break
            if self_weight > 0 and edges and not any(n == node for n, _ in edges):
                fail(
                    "transitions-respects-self-weight",
                    f"row {node!r} has no self edge at self_weight={self_weight}",
                )
                break
            if count >= sample_nodes:
                break

    # ── calibration ────────────────────────────────────────────────────────
    try:
        calibration = index.calibration()
    except Exception as exc:  # noqa: BLE001
        fail(
            "calibration-is-usable",
            f"calibration() raised {exc!r}; it is called on the first query that "
            "fuses two lanes, including against an empty corpus",
        )
    else:
        for member in ("abstains", "fence_at", "sampled"):
            if not hasattr(calibration, member):
                fail("calibration-is-usable", f"calibration() has no {member!r}")
                break
        else:
            try:
                calibration.abstains(0.0, 0.25)
                calibration.fence_at(0.25)
            except Exception as exc:  # noqa: BLE001
                fail("calibration-is-usable", f"the fence raised {exc!r} on a zero score")

    # ── close ──────────────────────────────────────────────────────────────
    if close:
        try:
            index.close()
            index.close()
        except Exception as exc:  # noqa: BLE001
            fail(
                "close-is-idempotent",
                f"a second close() raised {exc!r}; a cache eviction can close an "
                "index a caller still holds",
            )

    return failures


def assert_corpus_index(index: Any, *, name: str = "index", **kwargs: Any) -> None:
    """`check_corpus_index`, raising `AssertionError` with every failure named."""
    failures = check_corpus_index(index, **kwargs)
    if failures:
        listed = "\n  - ".join(failures)
        raise AssertionError(f"{name} is not a conforming CorpusIndex:\n  - {listed}")


def _key_of(by_path: dict, document: Any) -> str:
    """The key `by_path` filed this document under, without assuming it is `.path`.

    A corpus keyed on something other than the document's own path is exactly
    what `DocumentNaming.key` exists for, so the kit must not re-derive the key
    it is checking.
    """
    for key, value in by_path.items():
        if value is document:
            return key
    return getattr(document, "path", "")
