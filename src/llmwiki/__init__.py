"""llmwiki — build and query a self-maintaining wiki from your documents.

The pattern (Karpathy's llm-wiki, as implemented by llm_wiki): instead of
retrieving from raw documents at query time, the LLM incrementally builds a
persistent, interlinked wiki that sits between you and your sources. Adding a
document doesn't just index it — the model reads it, extracts what matters,
and integrates it into the existing pages. Knowledge is compiled once and
kept current rather than re-derived on every question.

Typical use::

    from llmwiki import config, ingest_document, ask, open_project

    project = open_project("~/wikis/energy")
    settings = config.load(project.root)
    ingest_document(project, "paper.pdf", settings)
    print(ask(project, "What limits grid-scale storage?", settings).text)
"""

from .config import EmbeddingConfig, LLMConfig, Settings
from .errors import (
    ConfigError,
    IngestError,
    LlmWikiError,
    ParseError,
    ProjectError,
    ProviderError,
)
from .ingest import IngestResult, ingest_document
from .project import Project, create, open_project
from .query import Answer, ask
from .retrieval import search

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "ConfigError",
    "EmbeddingConfig",
    "IngestError",
    "IngestResult",
    "LLMConfig",
    "LlmWikiError",
    "ParseError",
    "Project",
    "ProjectError",
    "ProviderError",
    "Settings",
    "__version__",
    "ask",
    "create",
    "ingest_document",
    "open_project",
    "search",
]
