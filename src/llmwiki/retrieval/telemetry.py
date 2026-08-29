"""A2: one deadline per turn, and a sink that can see where it went.

Two halves of the same feature. A budget nobody measures is a constant, and a
measurement with nothing to compare against is a number.

**The deadline is carried, not repeated.** Per-call timeouts each hold their own
opinion about how long a turn may take, so five of them sum to five times the
budget and a slow first stage is invisible until the total blows. `Deadline`
starts once per turn and every stage asks it what is left; a rewrite that spends
150 ms of a 400 ms turn leaves the embedding 60 ms or whatever is left of it,
whichever is smaller.

**Expiry is a fourth state.** `LanesRun` already separates a lane that is off
from one that failed from one that chose to stand down, and a deadline adds a
fourth: a lane that would have run and was not given time. Collapsing that into
`failed` would make a slow backend look like a broken one, which is the wrong
page of the runbook.

**The sink is a Protocol with a no-op default.** Nothing in `src/` imports
`logging`, and adding OpenTelemetry to a package whose only extra is numpy is
the wrong trade for a library that is also a CLI. A deployment wires its own
sink at the edge; the metric names below are fixed here because later steps are
written against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

# Metric names. Fixed here rather than at each call site: a name that drifts is
# a dashboard that silently stops filling in.
STAGE_MS = "retrieval.stage.ms"
ROUND_TRIPS = "retrieval.round_trips"
GATE_FIRED = "retrieval.gate.fired"
LANE_SELECTED = "retrieval.lane.selected"

# What a stage can be, and the reason `expired` is not `failed`.
OK = "ok"
EXPIRED = "expired"
FAILED = "failed"
SKIPPED = "skipped"


class Sink(Protocol):
    """Where stage timings go. The default goes nowhere."""

    def stage(self, name: str, ms: float, outcome: str) -> None: ...

    def counter(self, name: str, value: int = 1, **labels: str) -> None: ...


class NullSink:
    """The default. A library that logs by default is a library that surprises."""

    def stage(self, name: str, ms: float, outcome: str) -> None:
        return None

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        return None


NULL_SINK: Sink = NullSink()


@dataclass
class RecordingSink:
    """Keeps everything, for a test or a replay over an eval suite."""

    stages: list[tuple[str, float, str]] = field(default_factory=list)
    counters: list[tuple[str, int, dict]] = field(default_factory=list)

    def stage(self, name: str, ms: float, outcome: str) -> None:
        self.stages.append((name, ms, outcome))

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        self.counters.append((name, value, labels))

    def outcomes(self, name: str) -> list[str]:
        return [outcome for stage, _ms, outcome in self.stages if stage == name]

    def percentiles(self, quantiles: tuple[float, ...] = (0.5, 0.95)) -> dict[str, dict[str, float]]:
        """Per stage, the requested quantiles over every turn recorded.

        A p50 alone reads a stage as a constant. The gap between p50 and p95 is
        the whole reason a budget exists: a stage that is 8 ms every time and one
        that is 8 ms with a 900 ms tail are the same median and different
        products.
        """
        by_stage: dict[str, list[float]] = {}
        for name, ms, _outcome in self.stages:
            by_stage.setdefault(name, []).append(ms)
        table: dict[str, dict[str, float]] = {}
        for name, values in by_stage.items():
            values.sort()
            row = {f"p{int(q * 100)}": _quantile(values, q) for q in quantiles}
            row["n"] = float(len(values))
            table[name] = row
        return table


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return round(sorted_values[index], 3)


@dataclass
class Deadline:
    """The turn's remaining time, and each stage's share of it.

    `budgets.turn is None` means no deadline — the shipped text path, where a
    research question is allowed to take as long as it takes. Every method below
    returns `None` in that case rather than a large number, so a caller cannot
    accidentally impose a timeout nobody asked for.
    """

    budgets: object  # config.StageBudgets; typed loosely to keep the import one-way
    started: float = field(default_factory=perf_counter)

    @property
    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started) * 1000.0

    def remaining_ms(self) -> float | None:
        turn = getattr(self.budgets, "turn", None)
        return None if turn is None else turn - self.elapsed_ms

    def expired(self) -> bool:
        remaining = self.remaining_ms()
        return remaining is not None and remaining <= 0.0

    def for_stage(self, name: str) -> float | None:
        """This stage's budget, capped by what is left of the turn.

        The cap is the point. A stage budget is a promise about that stage; the
        turn deadline is a promise to the user, and when they disagree the user
        wins.
        """
        stage_budget = getattr(self.budgets, name, None)
        remaining = self.remaining_ms()
        if stage_budget is None:
            return remaining
        if remaining is None:
            return float(stage_budget)
        return min(float(stage_budget), remaining)

    def affords(self, name: str) -> bool:
        """Whether the stage has any time at all. An expired turn affords nothing."""
        budget = self.for_stage(name)
        return budget is None or budget > 0.0


__all__ = [
    "EXPIRED",
    "FAILED",
    "GATE_FIRED",
    "LANE_SELECTED",
    "NULL_SINK",
    "OK",
    "ROUND_TRIPS",
    "SKIPPED",
    "STAGE_MS",
    "Deadline",
    "NullSink",
    "RecordingSink",
    "Sink",
]
