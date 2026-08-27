"""How much a model is allowed to think, and what each family accepts.

Ported from llm_wiki's `src/lib/reasoning-capabilities.ts`. Three facts drive
this module, none of them guessable from the OpenAI wire format:

**Thinking is not always optional.** Gemini 3 accepts `low`/`medium`/`high`
and has no "off" — upstream models this as THINKING_REQUIRED_LEVELS, and the
same holds for OpenAI's reasoning models. Asking those families not to think
is not a request the API honours, so "off" clamps up to the lowest level they
do accept rather than being silently dropped.

**Thinking shares the output budget.** On Gemini the `max_tokens` ceiling
covers reasoning *and* content, so a model that thinks past it returns an
empty `content` and the caller loses the whole response. Upstream disables
reasoning on every structured ingest call for exactly this reason: thinking
"buys little on structured extraction and a model that spends its budget on
chain-of-thought can return empty content, losing the page".

**Silence is the portable default.** `reasoning_effort` is a real OpenAI
parameter, but one adapter here serves OpenAI, OpenRouter, Ollama, LM Studio
and vLLM, and a model that doesn't reason may answer 400 rather than ignore
it. So the *implicit* ingest default only applies to families named below; an
effort the user set by hand is always sent, since they know their endpoint
and we don't.
"""

from __future__ import annotations

import re

# Weakest to strongest. "off" is a request, not a guarantee — see `resolve`.
LEVELS = ("off", "low", "medium", "high", "max")

# Neither family accepts "off", so both ends of the range clamp inward.
_CLAMP = {"off": "low", "low": "low", "medium": "medium", "high": "high", "max": "high"}

_GEMINI_3 = re.compile(r"gemini[-_.]?3(?:[-_.]|$)", re.IGNORECASE)
_OPENAI_REASONING = re.compile(r"^(?:gpt-5|o\d+)(?:[.\-_]|$)", re.IGNORECASE)


def thinks_by_default(model: str) -> bool:
    """Whether this family reasons unless told otherwise.

    Deliberately narrow: it gates the ingest default, and a wrong `True` puts
    an unsupported parameter on every ingest call.
    """
    name = (model or "").strip()
    return bool(_GEMINI_3.search(name) or _OPENAI_REASONING.match(name))


def resolve(requested: str | None, model: str) -> str | None:
    """The `reasoning_effort` to send, or None to omit the field."""
    if not requested:
        return None
    level = requested.strip().lower()
    if not thinks_by_default(model):
        # An endpoint we can't characterise, and an effort the user typed on
        # purpose. Pass it through unclamped and let the API judge it; the
        # ingest default never reaches here.
        return level
    return _CLAMP.get(level, level)
