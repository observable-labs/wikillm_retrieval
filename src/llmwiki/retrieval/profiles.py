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

from dataclasses import dataclass, field, replace

from ..config import StageBudgets
from .calibration import ABSTAIN_QUANTILE
from .pipeline import DEFAULT_TOP_K, RetrievalOptions


FAST = "fast"
SLOW = "slow"

# Above this, a stated budget is a slow-path turn: the caller is willing to wait
# and would rather have every lane than an early answer. Below it, the budget is
# a promise to somebody listening.
FAST_PATH_CEILING_MS = 1_000


@dataclass(frozen=True)
class Budget:
    """What a caller is willing to spend on one turn, and on which path.

    The knob, and the thing that replaces choosing a profile by name. A profile
    is now a named budget plus lane defaults; nothing outside the pipeline picks
    lane flags directly.

    The two paths have different contracts because they have different failure
    modes. **Fast** may drop a lane, skip diffusion, or retrieve on the raw
    utterance, and must answer: a spoken turn that arrives late has already
    failed, whatever it says. **Slow** may drop nothing; when it needs more it
    spends another call. Encoding that as a field rather than as convention is
    what stops the next voice profile from being inverted — a fast path that
    adds a round trip is now a construction error and not a tuning opinion.
    """

    total_ms: int | None = None
    path: str = SLOW
    max_llm_calls: int = 4

    def __post_init__(self) -> None:
        if self.path not in (FAST, SLOW):
            raise ValueError(f"unknown path {self.path!r}; expected {FAST!r} or {SLOW!r}")
        if self.path == FAST:
            if self.total_ms is None:
                raise ValueError("a fast-path budget is a wall clock; it needs a total_ms")
            # Rewrite plus answer. A third call on this path is a third round
            # trip, which is the thing the path exists to avoid.
            if self.max_llm_calls > 2:
                raise ValueError(
                    f"a fast-path turn may make at most 2 model calls, not {self.max_llm_calls}"
                )
        elif self.max_llm_calls < 2:
            raise ValueError("a slow-path turn may make more calls than a fast one, not fewer")

    @classmethod
    def for_ms(cls, total_ms: int | None) -> "Budget":
        """The budget a bare number means: a wall, and the path it implies."""
        if total_ms is None:
            return cls()
        fast = total_ms <= FAST_PATH_CEILING_MS
        return cls(
            total_ms=int(total_ms),
            path=FAST if fast else SLOW,
            max_llm_calls=2 if fast else 4,
        )

    def stages(self) -> StageBudgets:
        """The per-stage shares of this budget."""
        if self.total_ms is None:
            return StageBudgets()
        return StageBudgets.for_turn(float(self.total_ms))

    @property
    def may_degrade(self) -> bool:
        return self.path == FAST


@dataclass(frozen=True)
class Profile:
    """A retrieval configuration with a name, a reason, and a clock.

    `budgets` is the third axis, added by build-plan A2. Depth is how much to
    look at, lane weights are who to trust, and budgets are how long the turn
    may take — and only `voice` sets a turn deadline, because a spoken answer
    that arrives late has already failed while a research answer has not.
    """

    name: str
    top_k: int
    options: RetrievalOptions
    description: str
    budget: Budget = field(default_factory=Budget)

    def with_top_k(self, top_k: int | None) -> "Profile":
        return self if top_k is None else replace(self, top_k=top_k)

    def with_budget(self, total_ms: int | None) -> "Profile":
        """This profile's lanes, spent against a stated wall clock.

        What makes a budget sweep possible without a profile per rung: the
        caller states milliseconds, the lanes stay the ones this profile means,
        and the path follows the number.
        """
        return self if total_ms is None else replace(self, budget=Budget.for_ms(total_ms))

    def deadline(self) -> "Deadline":
        """A fresh clock for one turn under this profile."""
        from .telemetry import Deadline

        return Deadline(self.budget.stages(), path=self.budget.path)


_BALANCED = RetrievalOptions()

PROFILES: dict[str, Profile] = {
    "voice": Profile(
        name="voice",
        top_k=5,
        # No diffusion and a narrow window: a spoken answer cites two pages and
        # the graph lane's 64 ms on a wiki-shaped corpus buys nothing a listener
        # will hear.
        # Not `graph_ppr=False`, which is what this profile shipped with and
        # what A3 calls the inverted ladder. Diffusion is ~8 ms of local CPU and
        # is what carries multi-hop; the vector lane is a network round trip.
        # Under a deadline the round trip is the first thing made conditional
        # and the local rung is the last thing dropped — so what `voice` states
        # is a narrower window, and the clock decides the rest.
        options=replace(_BALANCED, vector_depth=20),
        # The only profile with a turn deadline. Every stage below it falls back
        # rather than failing, so the budget buys a worse answer on time instead
        # of a better one late — which on this path is the same as no answer.
        budget=Budget(total_ms=400, path=FAST, max_llm_calls=2),
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


__all__ = ["Budget", "DEFAULT_PROFILE", "FAST", "PROFILES", "Profile", "SLOW", "resolve"]
