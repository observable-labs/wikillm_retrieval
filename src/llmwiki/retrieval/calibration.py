"""What a well-aimed query scores on *this* corpus, so a bad one can be recognised.

Reciprocal-rank fusion gives every lane an equal vote. That is the right default
when both lanes are competent and it is the wrong one when a lane has nothing to
say: on a question phrased without any of the corpus's own vocabulary the lexical
lane still returns fifty documents, still ranks one of them first, and RRF still
counts that ranking as evidence. Fusing a lane scoring 0.02 with a lane scoring
0.66 lands between them — measured at 0.45 on the keyword-hostile suite, against
0.66 for the vector lane alone.

The signal that separates the two cases is already computed and thrown away: the
lexical lane's own top BM25 score. On the atlas fixture, questions that use the
corpus's words score a median 7.99 and questions that deliberately avoid them
score a median 2.20, and the two distributions barely touch.

**Why it has to be calibrated per corpus.** A raw BM25 score is not comparable
across corpora — it depends on the term distribution, the document lengths and
the column weights — so a fixed threshold is a constant tuned on one fixture,
which is the mistake this line of work keeps finding. The reference distribution
here is built from the corpus itself: run each of a sample of document *titles*
as a query and record what the lexical lane scores. A title is the cheapest
available example of a query that genuinely names something in the corpus, it
needs no labels and no provider call, and it is regenerated whenever the index is.

A query whose top score falls below the `ABSTAIN_QUANTILE` of that distribution
is scoring worse than a well-aimed query would on almost any document here. It
has not found anything; it should abstain rather than vote.

Measured cost of building the reference: 14 ms over 78 titles, 152 ms over 400
on a 1,991-document corpus, once per index build.
"""

from __future__ import annotations

from dataclasses import dataclass

# The 5th percentile, not the 10th. Both were measured: on atlas p05 gates every
# one of the 62 keyword-hostile questions and costs nothing at all on the 44
# regular ones, while p10 gates 41% of the regular questions for the same gain
# and p25 costs 0.02 recall at k=1 and k=3. The fence is set where it is free,
# because a gate that trades the entity-anchored case for the paraphrased one has
# moved the problem rather than fixed it.
ABSTAIN_QUANTILE = 0.05

# Titles sampled to build the reference distribution. Evenly strided rather than
# random: a corpus is often ordered by directory and a prefix is one subject.
CALIBRATION_SAMPLE = 400


@dataclass(frozen=True)
class LexicalCalibration:
    """The corpus's own distribution of well-aimed query scores.

    The whole distribution rather than one threshold, because how sceptical to
    be of the lexical lane is a caller's decision and not the corpus's. A
    latency profile and a research profile want different fences over the same
    reference scores, and recomputing the reference for each would cost a few
    hundred FTS5 queries to answer a question about a percentile.
    """

    scores: tuple[float, ...]
    quantile: float = ABSTAIN_QUANTILE

    @property
    def sampled(self) -> int:
        return len(self.scores)

    @property
    def fence(self) -> float:
        return self.fence_at(self.quantile)

    def fence_at(self, quantile: float) -> float:
        """The score at `quantile` of the reference distribution."""
        if not self.scores:
            return 0.0
        index = min(
            len(self.scores) - 1,
            max(0, round(max(0.0, min(1.0, quantile)) * (len(self.scores) - 1))),
        )
        return self.scores[index]

    def abstains(self, top_score: float, quantile: float | None = None) -> bool:
        """Whether a lexical ranking topping out at `top_score` has found nothing.

        A corpus that produced no reference distribution — no lexical index, or
        every title scoring zero — has a fence of zero and never gates, which is
        the shipped behaviour. Failing open is deliberate: a calibration that
        cannot be computed must not silently remove a lane.
        """
        fence = self.fence if quantile is None else self.fence_at(quantile)
        return fence > 0.0 and top_score < fence


def calibrate(
    documents,
    lexical,
    sample: int = CALIBRATION_SAMPLE,
    quantile: float = ABSTAIN_QUANTILE,
) -> LexicalCalibration:
    """Build the reference distribution from the corpus's own titles."""
    from .tokenize import tokenize_query

    if lexical is None or not documents:
        return LexicalCalibration(scores=(), quantile=quantile)

    step = max(1, len(documents) // max(1, sample))
    scores: list[float] = []
    for document in documents[::step][:sample]:
        tokens = tokenize_query(document.title)
        if not tokens:
            continue
        hits = lexical.search(tokens, 1)
        scores.append(hits[0].score if hits else 0.0)

    scores.sort()
    return LexicalCalibration(scores=tuple(scores), quantile=quantile)


__all__ = [
    "ABSTAIN_QUANTILE",
    "CALIBRATION_SAMPLE",
    "LexicalCalibration",
    "calibrate",
]
