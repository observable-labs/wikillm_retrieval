"""Named retrieval configurations: how deep to look, and which lane to trust.

Build-plan step 9 specified `voice` / `balanced` / `deep` / `research` on a
**latency** axis — shallower for a spoken answer, wider when the user is willing
to wait. The measurement that produced this module says the axis is incomplete:
the right *lane mix* depends on the class of question, not on the latency budget.

On a question set phrased without any of the corpus's own vocabulary the vector
lane scores 0.66 at k=10 and the lexical lane 0.02, and equal-weight fusion of
the two landed at 0.45 — below either sensible choice. On an entity-anchored set
the ordering reverses on the questions that matter, and the lexical lane is
carrying the bridge cases the vector lane blurs. One configuration cannot be
right for both, and nothing in a query says which one it is.

Two instruments, in increasing order of bluntness:

* **`abstain_quantile`** — where in the corpus's own score distribution the
  lexical lane has to land before its ranking is worth fusing. Calibrated per
  corpus (`calibration.py`), so this is a percentile rather than a score and it
  transfers between corpora.
* **`lexical_weight`** — how much its vote is worth when it does clear the fence.
  The continuous form, for the weak-but-not-empty case a gate is the wrong tool
  for.

`research` is the profile named for the use this whole line of work is about, and
it is the one that leans on both. It is not the default: the measured cost of
leaning vector-first on entity-anchored questions is real, and a caller who knows
which kind of question they are asking is a better judge than a ranker guessing
from the query string.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .calibration import ABSTAIN_QUANTILE
from .pipeline import DEFAULT_TOP_K, RetrievalOptions


@dataclass(frozen=True)
class Profile:
    """A retrieval configuration with a name and a reason."""

    name: str
    top_k: int
    options: RetrievalOptions
    description: str

    def with_top_k(self, top_k: int | None) -> "Profile":
        return self if top_k is None else replace(self, top_k=top_k)


_BALANCED = RetrievalOptions()

PROFILES: dict[str, Profile] = {
    "voice": Profile(
        name="voice",
        top_k=5,
        # No diffusion and a narrow window: a spoken answer cites two pages and
        # the graph lane's 64 ms on a wiki-shaped corpus buys nothing a listener
        # will hear.
        options=replace(_BALANCED, graph_ppr=False, vector_depth=20),
        description="shallow and local; for an answer that has to start speaking",
    ),
    "balanced": Profile(
        name="balanced",
        top_k=DEFAULT_TOP_K,
        options=_BALANCED,
        description="the shipped defaults; every published number is this one",
    ),
    "deep": Profile(
        name="deep",
        top_k=30,
        # More diffusion and a deeper vector scan. Buys recall at the tail on
        # multi-hop questions and costs latency; changes no lane weight, because
        # depth and trust are different axes and conflating them is what made
        # step 9 look like a performance nicety.
        options=replace(_BALANCED, iterations=100, vector_depth=60),
        description="wider window and more diffusion; for a question worth waiting on",
    ),
    "research": Profile(
        name="research",
        top_k=DEFAULT_TOP_K,
        # Vector-first, with the lexical lane as a tie-break rather than a peer:
        # a fence five times higher than `balanced`, and half a vote when it
        # clears it. For questions asked before the user knows what the corpus
        # calls things, which is the traffic a research assistant is for and the
        # class every default in this repository was tuned against.
        options=replace(_BALANCED, abstain_quantile=0.25, lexical_weight=0.5),
        description="vector-first; for questions that do not name their own answer",
    ),
}

DEFAULT_PROFILE = "balanced"


def resolve(name: str | Profile | None = None) -> Profile:
    """A profile by name, defaulting to the shipped configuration."""
    if isinstance(name, Profile):
        return name
    key = (name or DEFAULT_PROFILE).strip().lower()
    if key not in PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; expected one of {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[key]


__all__ = ["DEFAULT_PROFILE", "PROFILES", "Profile", "resolve"]
