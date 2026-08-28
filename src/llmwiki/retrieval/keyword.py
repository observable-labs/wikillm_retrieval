"""Documents, results, and the substring scorer that BM25 replaced.

`load_documents`, `Document` and `SearchResult` are the corpus-facing half and
are used by everything. `score_document` is the other half: the scorer ported
verbatim from llm_wiki's `search.rs`, which ranked by term *presence*.

It is no longer the lexical lane — `lexical.py` is, and ranks by BM25. This
survives as the lane for CJK queries, where FTS5's `unicode61` tokenizer cannot
segment an unbroken run of characters and `tokenize_query`'s bigram expansion
plus substring matching is genuinely the better retriever. `SOURCE_SCORE_FACTOR`
still applies here, because this scorer has no length normalization and without
it a short raw source outranks the page compiled from it.

The absolute numbers only matter relative to each other:

    filename exactly matches the query      +200
    query phrase appears in the title        +50
    query phrase appears in the body    +20 each (max 10 counted)
    a token appears in the title          +5 each
    a token appears in the body           +1 each

A page scoring zero on every signal is not a result at all — it never enters
the ranking, so diffusion has a clean set of seeds to work from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..frontmatter import extract_title
from ..paths import normalize_path

FILENAME_EXACT_BONUS = 200.0
PHRASE_IN_TITLE_BONUS = 50.0
PHRASE_IN_CONTENT_PER_OCC = 20.0
MAX_PHRASE_OCC_COUNTED = 10
TITLE_TOKEN_WEIGHT = 5.0
CONTENT_TOKEN_WEIGHT = 1.0
SNIPPET_CONTEXT = 80

# Raw sources participate in retrieval but are damped relative to wiki pages.
# llm_wiki keeps them in a separate lane entirely (`source.search`, and a
# "Read Sources Only" mode) because the compiled wiki is meant to be the
# answer layer and the raw text the supporting evidence. Damping preserves
# that ordering inside one ranking: a source still surfaces when it is the
# only thing that matches, but it does not displace the page written from it.
SOURCE_SCORE_FACTOR = 0.6

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Navigation and bookkeeping files are not retrievable content. index.md is a
# catalog of links, so it matches almost any query on title overlap alone, and
# it is already injected into the query prompt separately — retrieving it too
# would spend a citation slot and part of the page budget on a duplicate.
NON_CONTENT_PAGES = frozenset(
    {"wiki/index.md", "wiki/log.md", "wiki/reviews.md"}
)


@dataclass
class Document:
    """One retrievable unit: a wiki page or a raw source file."""

    path: str  # project-relative, forward slashes
    title: str
    content: str
    kind: str = "wiki"  # "wiki" | "source"
    links: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    page_type: str = ""

    @property
    def stem(self) -> str:
        name = self.path.rsplit("/", 1)[-1]
        return name[:-3] if name.lower().endswith(".md") else name


@dataclass
class SearchResult:
    path: str
    title: str
    snippet: str
    score: float
    kind: str = "wiki"
    title_match: bool = False
    vector_score: float | None = None
    graph_related_to: list[str] = field(default_factory=list)
    document: Document | None = None


def extract_wikilinks(content: str) -> list[str]:
    """`[[target|alias]]` and `[[target#anchor]]` both yield `target`."""
    links: list[str] = []
    for match in WIKILINK_RE.finditer(content):
        target = match.group(1).split("|")[0].split("#")[0].strip()
        if target:
            links.append(target)
    return links


def load_documents(project, include_sources: bool = True, max_files: int = 10_000) -> list[Document]:
    """Read every wiki page (and optionally every raw source) into memory.

    Personal-scale wikis are hundreds of small markdown files, so a full scan
    per query costs milliseconds and avoids an index that can go stale. The
    `max_files` guard is the same backstop the Rust implementation uses.
    """
    from ..frontmatter import parse as parse_frontmatter

    documents: list[Document] = []
    for path in project.wiki_pages(limit=max_files):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = project.relative(path)
        if relative.lower() in NON_CONTENT_PAGES:
            continue
        parsed = parse_frontmatter(content)
        documents.append(
            Document(
                path=relative,
                title=extract_title(content, path.name),
                content=content,
                kind="wiki",
                links=extract_wikilinks(content),
                sources=parsed.get_list("sources"),
                page_type=parsed.get_str("type"),
            )
        )

    if include_sources:
        from ..parsers import extract_text
        from ..errors import ParseError

        remaining = max(0, max_files - len(documents))
        for path in project.source_files(limit=remaining):
            try:
                content = extract_text(path)
            except ParseError:
                continue  # unreadable formats simply don't participate
            if not content.strip():
                continue
            relative = project.relative(path)
            documents.append(
                Document(
                    path=relative,
                    title=path.name,
                    content=content,
                    kind="source",
                    links=[],
                    sources=[],
                )
            )
    return documents


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return haystack.count(needle)


def _token_match_score(text: str, tokens: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for token in tokens if token in lowered)


def build_snippet(content: str, anchor: str) -> str:
    """~160 characters of body text centred on the first match of `anchor`.

    Frontmatter is skipped: a snippet of `type:`/`tags:`/`created:` tells the
    reader nothing about whether the page answers their question.
    """
    from ..chunking import strip_frontmatter

    content, _ = strip_frontmatter(content)
    content = content.strip()
    lowered = content.lower()
    index = lowered.find(anchor.lower())
    if index < 0:
        index = 0
    start = max(0, index - SNIPPET_CONTEXT)
    end = min(len(content), index + len(anchor) + SNIPPET_CONTEXT)
    snippet = content[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(content):
        snippet = f"{snippet}..."
    return snippet


def score_document(
    document: Document,
    tokens: list[str],
    query_phrase: str,
    query: str,
) -> SearchResult | None:
    """Score one document, or return None when nothing matched at all."""
    file_name = document.path.rsplit("/", 1)[-1]
    title_text = f"{document.title} {file_name}"
    title_lower = title_text.lower()
    content_lower = document.content.lower()
    stem = document.stem.lower()

    filename_exact = bool(query_phrase) and stem == query_phrase
    title_has_phrase = bool(query_phrase) and query_phrase in title_lower
    content_phrase_occurrences = min(
        _count_occurrences(content_lower, query_phrase), MAX_PHRASE_OCC_COUNTED
    )
    title_token_score = _token_match_score(title_text, tokens)
    content_token_score = _token_match_score(document.content, tokens)

    if not (
        filename_exact
        or title_has_phrase
        or content_phrase_occurrences
        or title_token_score
        or content_token_score
    ):
        return None

    score = (
        (FILENAME_EXACT_BONUS if filename_exact else 0.0)
        + (PHRASE_IN_TITLE_BONUS if title_has_phrase else 0.0)
        + content_phrase_occurrences * PHRASE_IN_CONTENT_PER_OCC
        + title_token_score * TITLE_TOKEN_WEIGHT
        + content_token_score * CONTENT_TOKEN_WEIGHT
    )

    if document.kind == "source":
        score *= SOURCE_SCORE_FACTOR

    if content_phrase_occurrences:
        anchor = query_phrase
    else:
        anchor = next((token for token in tokens if token in content_lower), query)

    return SearchResult(
        path=normalize_path(document.path),
        title=document.title,
        snippet=build_snippet(document.content, anchor),
        score=score,
        kind=document.kind,
        title_match=bool(title_token_score) or title_has_phrase,
        document=document,
    )
