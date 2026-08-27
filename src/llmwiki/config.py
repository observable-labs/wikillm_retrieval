"""Configuration resolution.

Precedence, lowest to highest:

    built-in defaults
    ~/.config/llmwiki/config.json          (user)
    <project>/.llm-wiki/config.json        (project)
    .env files                             (see dotenv.py for the search order)
    LLMWIKI_* environment variables
    CLI flags

`.env` sits directly below the real environment because that is where it
would land if the user had run `set -a; . ./.env` themselves — the file
supplies variables, it does not outrank ones already exported.

Nothing here talks to a network; `llm.build_client` turns an `LLMConfig`
into a live client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .dotenv import load_dotenv
from .errors import ConfigError

ANTHROPIC = "anthropic"
OPENAI = "openai"
PROVIDERS = (ANTHROPIC, OPENAI)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_MAX_CONTEXT_CHARS = 204_800


def user_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "llmwiki" / "config.json"


@dataclass
class LLMConfig:
    """Everything needed to make a chat completion."""

    provider: str = ANTHROPIC
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    # Character budget for prompt assembly (see budget.py for the unit quirk).
    max_context_size: int = DEFAULT_MAX_CONTEXT_CHARS
    # Anthropic-only knobs. `effort` maps to output_config.effort; `thinking`
    # toggles adaptive thinking. Both are ignored by the OpenAI adapter.
    effort: str | None = None
    thinking: bool = True
    temperature: float | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 600.0

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == ANTHROPIC:
            return DEFAULT_ANTHROPIC_MODEL
        raise ConfigError(
            "no model configured. Set LLMWIKI_MODEL or pass --model "
            "(OpenAI-compatible endpoints have no safe default)."
        )


@dataclass
class EmbeddingConfig:
    """Optional vector lane. Disabled unless a model is configured.

    Only an OpenAI-compatible `/v1/embeddings` endpoint is supported, which
    covers OpenAI, Ollama, LM Studio, vLLM, and most gateways.
    """

    enabled: bool = False
    model: str = ""
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    dimensions: int | None = None
    batch_size: int = 32
    timeout: float = 60.0

    def require_enabled(self) -> None:
        if not self.enabled or not self.model:
            raise ConfigError(
                "vector search is not configured. Set LLMWIKI_EMBEDDING_MODEL "
                "(and LLMWIKI_EMBEDDING_BASE_URL / _API_KEY as needed)."
            )


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # Fall back to the source text when the wiki has no answer.
    search_sources: bool = True
    output_language: str = "auto"


def _load_json(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _int_env(name: str) -> int | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _detect_provider() -> str:
    """Pick a provider from whichever credential is actually present."""
    if _env("LLMWIKI_PROVIDER"):
        provider = _env("LLMWIKI_PROVIDER").strip().lower()  # type: ignore[union-attr]
        if provider not in PROVIDERS:
            raise ConfigError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")
        return provider
    if _env("LLMWIKI_BASE_URL") or _env("OPENAI_API_KEY") or _env("OPENAI_BASE_URL"):
        # An explicit OpenAI-compatible endpoint or key means that's the intent,
        # unless an Anthropic key is also present — then Anthropic wins as the
        # package default.
        if not _env("ANTHROPIC_API_KEY"):
            return OPENAI
    return ANTHROPIC


def strict_project_enabled() -> bool:
    """Whether a command must be given `--project` explicitly.

    On by default. Discovering a project from `$PWD` or `$LLMWIKI_PROJECT` is a
    convenience for a human at a prompt and a hazard for anything scripted,
    which inherits an ambient default it never chose. `LLMWIKI_STRICT_PROJECT=0`
    opts back into discovery.

    Only the environment and the *user* config are consulted: the project-local
    config lives inside the project, so it cannot help decide which project to
    open.
    """
    return _bool(
        _load_json(user_config_path()).get("strict_project"),
        default=True,
        env="LLMWIKI_STRICT_PROJECT",
    )


def load(
    project_dir: Path | None = None,
    overrides: dict | None = None,
) -> Settings:
    """Assemble settings from files, environment, and explicit overrides."""
    # Before the first `_env()` read: everything below resolves against
    # os.environ, so the `.env` values have to be in place by now.
    load_dotenv(project_dir)

    merged: dict = {}
    merged.update(_load_json(user_config_path()))
    if project_dir is not None:
        merged.update(_load_json(Path(project_dir) / ".llm-wiki" / "config.json"))

    llm_file = merged.get("llm", {}) if isinstance(merged.get("llm"), dict) else {}
    emb_file = merged.get("embedding", {}) if isinstance(merged.get("embedding"), dict) else {}

    provider = _env("LLMWIKI_PROVIDER") or llm_file.get("provider") or _detect_provider()
    provider = str(provider).strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")

    if provider == ANTHROPIC:
        api_key = _env("LLMWIKI_API_KEY") or _env("ANTHROPIC_API_KEY") or llm_file.get("api_key")
        base_url = _env("LLMWIKI_BASE_URL") or _env("ANTHROPIC_BASE_URL") or llm_file.get("base_url")
    else:
        api_key = _env("LLMWIKI_API_KEY") or _env("OPENAI_API_KEY") or llm_file.get("api_key")
        base_url = (
            _env("LLMWIKI_BASE_URL")
            or _env("OPENAI_BASE_URL")
            or llm_file.get("base_url")
            or "https://api.openai.com/v1"
        )

    llm = LLMConfig(
        provider=provider,
        model=_env("LLMWIKI_MODEL") or llm_file.get("model", ""),
        api_key=api_key,
        base_url=base_url,
        max_context_size=(
            _int_env("LLMWIKI_MAX_CONTEXT")
            or llm_file.get("max_context_size")
            or DEFAULT_MAX_CONTEXT_CHARS
        ),
        effort=_env("LLMWIKI_EFFORT") or llm_file.get("effort"),
        thinking=_bool(llm_file.get("thinking"), default=True, env="LLMWIKI_THINKING"),
        temperature=_float(llm_file.get("temperature"), env="LLMWIKI_TEMPERATURE"),
        extra_headers=llm_file.get("extra_headers", {}) or {},
        timeout=float(llm_file.get("timeout", 600.0)),
    )

    embedding_model = _env("LLMWIKI_EMBEDDING_MODEL") or emb_file.get("model", "")
    embedding = EmbeddingConfig(
        enabled=bool(embedding_model) and _bool(emb_file.get("enabled"), default=True, env="LLMWIKI_EMBEDDING_ENABLED"),
        model=embedding_model,
        api_key=(
            _env("LLMWIKI_EMBEDDING_API_KEY")
            or emb_file.get("api_key")
            or _env("OPENAI_API_KEY")
            or (api_key if provider == OPENAI else None)
        ),
        base_url=(
            _env("LLMWIKI_EMBEDDING_BASE_URL")
            or emb_file.get("base_url")
            or (base_url if provider == OPENAI else None)
            or "https://api.openai.com/v1"
        ),
        dimensions=_int_env("LLMWIKI_EMBEDDING_DIMENSIONS") or emb_file.get("dimensions"),
        batch_size=int(emb_file.get("batch_size", 32)),
    )

    settings = Settings(
        llm=llm,
        embedding=embedding,
        search_sources=_bool(merged.get("search_sources"), default=True, env="LLMWIKI_SEARCH_SOURCES"),
        output_language=_env("LLMWIKI_LANGUAGE") or merged.get("output_language", "auto"),
    )

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in {"provider", "model", "api_key", "base_url", "max_context_size", "effort", "temperature"}:
            settings.llm = replace(settings.llm, **{key: value})
        elif key == "embedding_model":
            settings.embedding = replace(settings.embedding, model=value, enabled=bool(value))
        elif hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def _bool(file_value, default: bool, env: str) -> bool:
    raw = _env(env)
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    if isinstance(file_value, bool):
        return file_value
    return default


def _float(file_value, env: str) -> float | None:
    raw = _env(env)
    if raw is not None:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"{env} must be a number, got {raw!r}") from exc
    if isinstance(file_value, (int, float)):
        return float(file_value)
    return None
