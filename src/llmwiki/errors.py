"""Exception hierarchy for llmwiki.

Every failure the CLI can surface to a user is one of these, so `cli.main`
can render a single-line error instead of a traceback.
"""

from __future__ import annotations


class LlmWikiError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(LlmWikiError):
    """Configuration is missing or contradictory (no API key, no model, ...)."""


class ProjectError(LlmWikiError):
    """The target directory is not a wiki project, or already is one."""


class ParseError(LlmWikiError):
    """A source document could not be turned into text."""


class ProviderError(LlmWikiError):
    """The LLM or embedding provider rejected the request or was unreachable."""


class IngestError(LlmWikiError):
    """The two-step ingest could not complete for a source."""
