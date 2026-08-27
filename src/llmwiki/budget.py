"""Character budgets for prompt assembly.

Ported from llm_wiki's `src/lib/context-budget.ts`. The unit is *characters*,
not tokens — a quirk of the original, kept deliberately so a project's
configured context size means the same thing in both implementations.

    +-----------------------------------------------------+
    |              max_ctx (100%)                         |
    +------+---------------+------------------+-----------+
    | idx  |   pages       |  history + sys   |  resp     |
    |  5%  |    50%        |    ~30%          |   15%     |
    +------+---------------+------------------+-----------+

`history + system` is not returned: the system prompt is roughly fixed-size
and history is bounded by message count, so the remainder is just headroom.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_CTX = 204_800
RESPONSE_RESERVE_FRAC = 0.15
INDEX_BUDGET_FRAC = 0.05
PAGE_BUDGET_FRAC = 0.5
PER_PAGE_FRAC = 0.3
PER_PAGE_FLOOR = 5_000

LONG_SOURCE_MIN_BUDGET = 20_000
LONG_SOURCE_MAX_SINGLE_PASS_BUDGET = 600_000


@dataclass(frozen=True)
class ContextBudget:
    """All values are character counts."""

    max_ctx: int
    response_reserve: int
    index_budget: int
    page_budget: int
    max_page_size: int


def compute_context_budget(max_context_size: int | None) -> ContextBudget:
    max_ctx = max_context_size if max_context_size and max_context_size > 0 else DEFAULT_MAX_CTX
    response_reserve = int(max_ctx * RESPONSE_RESERVE_FRAC)
    index_budget = int(max_ctx * INDEX_BUDGET_FRAC)
    page_budget = int(max_ctx * PAGE_BUDGET_FRAC)
    # Floor so a small config still fits one short page; capped at page_budget
    # so a tiny config can't admit a single page larger than the whole budget
    # (which the packer would then reject outright, yielding zero pages).
    max_page_size = min(page_budget, max(PER_PAGE_FLOOR, int(page_budget * PER_PAGE_FRAC)))
    return ContextBudget(
        max_ctx=max_ctx,
        response_reserve=response_reserve,
        index_budget=index_budget,
        page_budget=page_budget,
        max_page_size=max_page_size,
    )


def compute_ingest_source_budget(max_context_size: int | None, stable_context_length: int) -> int:
    """How much of a source document fits in one analysis pass.

    Anything longer is chunked and analyzed in pieces (see
    `ingest.pipeline.analyze_long_source`).
    """
    budget = compute_context_budget(max_context_size)
    stable_reserve = min(int(budget.max_ctx * 0.25), max(12_000, stable_context_length))
    instruction_reserve = max(12_000, int(budget.max_ctx * 0.08))
    available = budget.max_ctx - budget.response_reserve - stable_reserve - instruction_reserve
    upper = min(
        LONG_SOURCE_MAX_SINGLE_PASS_BUDGET,
        max(LONG_SOURCE_MIN_BUDGET, int(budget.max_ctx * 0.6)),
    )
    return max(LONG_SOURCE_MIN_BUDGET, min(int(available), upper))


def compute_generation_max_tokens(max_context_size: int | None) -> int:
    """Output ceiling for step 2, which emits every wiki page in one response."""
    max_ctx = compute_context_budget(max_context_size).max_ctx
    if max_ctx >= 512_000:
        return 32_000
    if max_ctx >= 256_000:
        return 24_000
    if max_ctx >= 128_000:
        return 16_000
    return 8_192


def trim_long_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[...trimmed for prompt budget...]"
