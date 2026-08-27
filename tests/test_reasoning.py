"""Effort levels: what each model family accepts, and what ingest asks for.

The bug this covers is a silent one. `effort` was configurable long before
anything outside the Anthropic client read it, so `LLMWIKI_EFFORT` in a
`.env` looked applied while every OpenAI-compatible call ignored it — and
ingest, which upstream runs with reasoning disabled, thought at full tilt
against a `max_tokens` ceiling it shares with the pages it is writing.
"""

from __future__ import annotations

import pytest

from llmwiki.config import LLMConfig
from llmwiki.ingest import ingest_document
from llmwiki.llm.base import Completion
from llmwiki.reasoning import resolve, thinks_by_default

from test_ingest import ANALYSIS, GENERATION, document  # noqa: F401


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.7-flash", True),
        ("gemini-3-pro", True),
        ("models/gemini_3.1", True),
        ("gpt-5", True),
        ("gpt-5.1-mini", True),
        ("o3", True),
        ("gemini-2.5-flash", False),
        ("gpt-4o", False),
        ("qwen3:8b", False),  # unrecognised, not disproven — see `resolve`
        ("", False),
    ],
)
def test_which_families_reason_unprompted(model, expected):
    assert thinks_by_default(model) is expected


@pytest.mark.parametrize(
    "requested,expected",
    [
        # Gemini 3 has no "off" — upstream's THINKING_REQUIRED_LEVELS. Clamp
        # up to the floor it does accept rather than dropping the request.
        ("off", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("max", "high"),
    ],
)
def test_a_thinking_family_clamps_into_its_own_vocabulary(requested, expected):
    assert resolve(requested, "gemini-3.7-flash") == expected


def test_an_unrecognised_endpoint_is_passed_through_unclamped():
    """We don't know what a gateway accepts; the user who set it does."""
    assert resolve("off", "qwen3:8b") == "off"
    assert resolve("max", "some-local-model") == "max"


def test_no_effort_means_no_field():
    assert resolve(None, "gemini-3.7-flash") is None
    assert resolve("", "gpt-5") is None


def test_ingest_winds_thinking_down_only_where_the_model_reasons_anyway():
    gemini = LLMConfig(provider="openai", model="gemini-3.7-flash")
    assert gemini.for_ingest().effort == "off"
    assert gemini.effort is None, "asking is left at the provider default"

    # A model that doesn't reason may reject the parameter outright, so the
    # implicit default stays off the wire entirely.
    assert LLMConfig(provider="openai", model="gpt-4o").for_ingest().effort is None


def test_an_explicit_ingest_effort_wins_everywhere():
    """The escape hatch: some endpoints refuse a wound-down reasoning level."""
    config = LLMConfig(provider="openai", model="gpt-4o", ingest_effort="medium")
    assert config.for_ingest().effort == "medium"

    config = LLMConfig(provider="openai", model="gemini-3.7-flash", effort="high", ingest_effort="high")
    assert config.for_ingest().effort == "high"
    assert config.effort == "high"


def test_ingest_builds_its_client_from_the_ingest_profile(wiki, monkeypatch, document):  # noqa: F811
    """Regression: ingest used the ask profile, so it thought on every call."""
    from llmwiki import config as config_module

    seen: list[LLMConfig] = []

    class _Stub:
        def __init__(self, cfg):
            seen.append(cfg)

        def complete(self, messages, *, max_tokens, on_token=None):
            system = next(m.content for m in messages if m.role == "system")
            text = GENERATION if "wiki maintainer" in system else ANALYSIS
            return Completion(text=text, model="stub")

    monkeypatch.setattr("llmwiki.ingest.pipeline.build_client", _Stub)
    settings = config_module.Settings(
        llm=LLMConfig(provider="openai", model="gemini-3.7-flash")
    )
    ingest_document(wiki, document, settings)

    assert seen and all(cfg.effort == "off" for cfg in seen)
    assert settings.llm.effort is None, "the caller's settings are not mutated"
