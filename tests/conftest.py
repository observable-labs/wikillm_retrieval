from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmwiki import config, project as project_module  # noqa: E402
from llmwiki.llm.base import Completion  # noqa: E402

# Each pipeline stage is identified by a phrase unique to its system prompt,
# so a test can script "what the model says during generation" without having
# to know how many calls precede it.
PHASE_MARKERS = (
    ("chunk", "ONE SECTION"),
    ("consolidate", "consolidating per-section"),
    ("analysis", "expert research analyst"),
    ("repair", "repairing truncated"),
    ("generation", "You are a wiki maintainer"),
    ("answer", "maintainer and reader of a personal wiki"),
)


def classify(system_prompt: str) -> str:
    for phase, marker in PHASE_MARKERS:
        if marker in system_prompt:
            return phase
    return "unknown"


class StubClient:
    """A model stubbed per pipeline phase.

    `responses` maps a phase name to either a string or a list of strings
    consumed in order. Recording every call lets tests assert on what the
    prompts actually contained.
    """

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = {
            key: list(value) if isinstance(value, list) else [value]
            for key, value in responses.items()
        }
        self.calls: list[tuple[str, list]] = []

    def complete(self, messages, *, max_tokens: int, on_token=None):
        messages = list(messages)
        system = next((m.content for m in messages if m.role == "system"), "")
        phase = classify(system)
        self.calls.append((phase, messages))

        bucket = self.responses.get(phase)
        if not bucket:
            raise AssertionError(f"stub has no scripted response for phase {phase!r}")
        text = bucket.pop(0) if len(bucket) > 1 else bucket[0]
        if on_token:
            on_token(text)
        return Completion(text=text, model="stub")

    def phases(self) -> list[str]:
        return [phase for phase, _ in self.calls]

    def prompt(self, phase: str, role: str = "system") -> str:
        for called_phase, messages in self.calls:
            if called_phase == phase:
                return next(m.content for m in messages if m.role == role)
        raise AssertionError(f"phase {phase!r} was never called; saw {self.phases()}")


@pytest.fixture(autouse=True)
def _no_ambient_dotenv(monkeypatch):
    """Keep the developer's own `.env` out of the suite.

    `config.load` walks up from `$PWD` looking for `.env`, and pytest runs
    from the repo root — where a working `.env` usually sits. Without this,
    tests that assert on provider detection pass or fail depending on whether
    the machine happens to be configured, and a test run quietly loads real
    credentials.

    Stubbing the function rather than setting `LLMWIKI_DOTENV=0` is
    deliberate: several modules have their own autouse fixture that deletes
    every `LLMWIKI_*` variable, which would erase an env-var guard before the
    test body ran. `tests/test_dotenv.py` puts the real loader back.
    """
    monkeypatch.setattr(config, "load_dotenv", lambda project_dir=None: None)


@pytest.fixture
def stub_llm(monkeypatch):
    def install(responses: dict[str, object]) -> StubClient:
        client = StubClient(responses)
        monkeypatch.setattr("llmwiki.ingest.pipeline.build_client", lambda _cfg: client)
        monkeypatch.setattr("llmwiki.query.build_client", lambda _cfg: client)
        return client

    return install


@pytest.fixture
def settings():
    return config.Settings(llm=config.LLMConfig(provider="anthropic", model="stub-model"))


@pytest.fixture
def wiki(tmp_path):
    return project_module.create(tmp_path / "wiki", "research", "Test Wiki")
