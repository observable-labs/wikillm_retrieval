"""The two-step ingest: analysis, then generation.

    step 0   read + store the document, check the SHA256 cache
    step 1   LLM reads the source -> structured analysis
             (over-budget sources are analyzed chunk-by-chunk, then merged)
    step 2   LLM reads the analysis -> FILE blocks (the wiki pages)
    step 2.5 re-request any block truncated mid-stream
    step 3   write pages, canonicalize sources, update index.md and log.md
    step 4   guarantee a source summary page exists, file review items,
             record the cache entry, optionally embed the new pages

Steps 3 and 4 are deterministic on purpose. Aggregates rewritten by a model
lose entries when a response truncates, and a source page that silently
fails to appear breaks the audit trail that makes the wiki trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..budget import (
    compute_context_budget,
    compute_generation_max_tokens,
    compute_ingest_source_budget,
)
from ..chunking import split_source_into_semantic_chunks
from ..config import Settings
from ..errors import IngestError, ParseError
from ..llm import Message, build_client
from ..parsers import extract_text
from ..paths import source_identity as compute_identity
from ..paths import source_summary_slug
from ..project import Project, today
from .blocks import ParseResult, parse_blocks
from .cache import IngestCache
from .prompts import (
    build_analysis_prompt,
    build_analysis_user_message,
    build_chunk_analysis_prompt,
    build_consolidation_prompt,
    build_fallback_source_summary,
    build_generation_prompt,
    build_generation_user_message,
    build_repair_prompt,
)
from .writer import (
    deterministic_log_entry,
    record_reviews,
    store_source,
    update_index,
    write_blocks,
)

ANALYSIS_MAX_TOKENS = 8_192
CHUNK_OVERLAP_CHARS = 800


@dataclass
class IngestResult:
    source_identity: str
    source_path: str
    files_written: list[str] = field(default_factory=list)
    reviews: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    analysis: str = ""
    cached: bool = False
    chunks_analyzed: int = 0

    @property
    def pages(self) -> list[str]:
        """Wiki pages generated from this source, excluding maintained aggregates."""
        maintained = {"wiki/index.md", "wiki/log.md", "wiki/overview.md", "wiki/reviews.md"}
        return [
            path
            for path in self.files_written
            if path.endswith(".md") and path not in maintained
        ]


def ingest_document(
    project: Project,
    document_path: Path | str,
    settings: Settings,
    folder: str = "",
    force: bool = False,
    on_status=None,
    on_token=None,
) -> IngestResult:
    """Add one document to `project`, building wiki pages from it."""
    path = Path(document_path).expanduser()
    if not path.is_file():
        raise IngestError(f"'{path}' is not a file")

    def status(message: str) -> None:
        if on_status:
            on_status(message)

    # ── step 0: text, storage, cache ──────────────────────────────────
    status(f"Reading {path.name}")
    try:
        source_content = extract_text(path)
    except ParseError as exc:
        raise IngestError(str(exc)) from exc
    if not source_content.strip():
        raise IngestError(f"{path.name} contains no extractable text")

    stored = store_source(project, path, folder)
    identity = compute_identity(str(project.root), str(stored))
    summary_slug = source_summary_slug(identity)
    summary_path = f"wiki/sources/{summary_slug}.md"

    cache = IngestCache.load(project)
    if not force:
        cached_files = cache.check(project, identity, source_content)
        if cached_files is not None:
            status("Unchanged since the last ingest — skipping (use --force to re-ingest)")
            return IngestResult(
                source_identity=identity,
                source_path=project.relative(stored),
                files_written=cached_files,
                cached=True,
            )

    schema = project.schema()
    purpose = project.purpose()
    index = project.index()
    overview = project.overview()
    language = settings.output_language
    client = build_client(settings.llm)
    date = today()

    # ── step 1: analysis ──────────────────────────────────────────────
    stable_length = len(schema) + len(purpose) + len(index) + len(overview)
    source_budget = compute_ingest_source_budget(settings.llm.max_context_size, stable_length)

    chunks_analyzed = 0
    if len(source_content) > source_budget:
        status(
            f"Source is {len(source_content):,} chars (budget {source_budget:,}) — analyzing in sections"
        )
        analysis, chunks_analyzed = _analyze_long_source(
            client,
            source_content,
            source_budget,
            purpose=purpose,
            schema=schema,
            index=index,
            language=language,
            status=status,
        )
        # Step 2 still needs to see the document; it gets the budgeted head,
        # with the consolidated analysis carrying what the tail contained.
        source_context = source_content[:source_budget]
    else:
        status("Step 1/2: analyzing source")
        analysis = _complete(
            client,
            [
                Message("system", build_analysis_prompt(purpose, index, source_content, schema, language)),
                Message("user", build_analysis_user_message(identity, source_content, folder)),
            ],
            ANALYSIS_MAX_TOKENS,
        )
        source_context = source_content

    if not analysis.strip():
        raise IngestError("step 1 (analysis) returned nothing")

    # ── step 2: generation ────────────────────────────────────────────
    status("Step 2/2: generating wiki pages")
    generation_system = build_generation_prompt(
        schema=schema,
        purpose=purpose,
        index=index,
        source_identity=identity,
        source_summary_path=summary_path,
        today=date,
        overview=overview,
        source_content=source_context,
        output_language=language,
    )
    generation = _complete(
        client,
        [
            Message("system", generation_system),
            Message("user", build_generation_user_message(identity, analysis, source_context)),
        ],
        compute_generation_max_tokens(settings.llm.max_context_size),
        on_token=on_token,
    )

    parsed = parse_blocks(generation)
    warnings = list(parsed.warnings)

    # ── step 2.5: repair truncated blocks ─────────────────────────────
    if parsed.truncated_paths:
        status(f"Repairing {len(parsed.truncated_paths)} truncated page(s)")
        repaired = _repair_truncated(
            client,
            parsed.truncated_paths,
            identity=identity,
            schema=schema,
            purpose=purpose,
            analysis=analysis,
            source_context=source_context,
            settings=settings,
            language=language,
        )
        recovered = {block.path for block in repaired.files}
        parsed.files.extend(repaired.files)
        warnings.extend(repaired.warnings)
        warnings = [
            warning
            for warning in warnings
            if not any(f'FILE block "{path}" was not closed' in warning for path in recovered)
        ]

    if not parsed.files:
        raise IngestError(
            "step 2 produced no usable FILE blocks"
            + (f" ({warnings[0]})" if warnings else "")
        )

    # ── step 3: write ─────────────────────────────────────────────────
    status(f"Writing {len(parsed.files)} page(s)")
    write_result = write_blocks(project, parsed.files, identity, date)
    warnings.extend(write_result.warnings)
    written = list(write_result.written)

    if update_index(project, written):
        written.append("wiki/index.md")
    if "wiki/log.md" not in written:
        deterministic_log_entry(project, identity, date)
        written.append("wiki/log.md")

    # ── step 4: safety nets ───────────────────────────────────────────
    # Every source needs exactly one summary page — it is the audit trail
    # linking the wiki back to the document. Two failure modes are handled
    # differently: a model that filed the summary under a different name has
    # still done the work (accept it, note the deviation), while a model that
    # skipped it entirely needs a recovery page written from the analysis.
    if summary_path not in written:
        misfiled = [path for path in written if path.startswith("wiki/sources/")]
        if len(misfiled) == 1:
            warnings.append(
                f"the model filed the source summary at {misfiled[0]} instead of "
                f"{summary_path}; accepted as-is rather than writing a duplicate"
            )
        else:
            project.write(summary_path, build_fallback_source_summary(identity, analysis, date))
            written.append(summary_path)
            warnings.append(
                f"the model did not produce {summary_path}; wrote a summary from the "
                "step 1 analysis instead"
            )

    if parsed.reviews:
        record_reviews(project, identity, parsed.reviews)
        written.append("wiki/reviews.md")

    cache.record(identity, source_content, written)
    cache.save()

    if warnings:
        _append_warning_log(project, identity, warnings)

    if settings.embedding.enabled and settings.embedding.model:
        status("Embedding new pages")
        _embed_written_pages(project, written, settings, warnings)

    return IngestResult(
        source_identity=identity,
        source_path=project.relative(stored),
        files_written=written,
        reviews=parsed.reviews,
        warnings=warnings,
        analysis=analysis,
        chunks_analyzed=chunks_analyzed,
    )


# ── helpers ───────────────────────────────────────────────────────────────

def _complete(client, messages: list[Message], max_tokens: int, on_token=None) -> str:
    return client.complete(messages, max_tokens=max_tokens, on_token=on_token).text


def _analyze_long_source(
    client,
    source_content: str,
    source_budget: int,
    purpose: str,
    schema: str,
    index: str,
    language: str,
    status,
) -> tuple[str, int]:
    """Analyze an over-budget source section by section, then consolidate."""
    target = max(4_000, int(source_budget * 0.8))
    chunks = split_source_into_semantic_chunks(source_content, target, CHUNK_OVERLAP_CHARS)
    if not chunks:
        return "", 0

    chunk_system = build_chunk_analysis_prompt(purpose, schema, language, source_content[:2000])
    analyses: list[str] = []
    for chunk in chunks:
        status(f"  analyzing section {chunk.index}/{chunk.total}")
        heading = f" ({chunk.heading_path})" if chunk.heading_path else ""
        analyses.append(
            f"### Section {chunk.index}/{chunk.total}{heading}\n\n"
            + _complete(
                client,
                [
                    Message("system", chunk_system),
                    Message(
                        "user",
                        f"Section {chunk.index} of {chunk.total}{heading}:\n\n{chunk.text}",
                    ),
                ],
                ANALYSIS_MAX_TOKENS,
            )
        )

    if len(analyses) == 1:
        return analyses[0], 1

    status(f"  consolidating {len(analyses)} section analyses")
    consolidated = _complete(
        client,
        [
            Message("system", build_consolidation_prompt(purpose, index, schema, language, source_content[:2000])),
            Message("user", "\n\n".join(analyses)),
        ],
        ANALYSIS_MAX_TOKENS * 2,
    )
    return consolidated, len(chunks)


def _repair_truncated(
    client,
    paths: list[str],
    identity: str,
    schema: str,
    purpose: str,
    analysis: str,
    source_context: str,
    settings: Settings,
    language: str,
) -> ParseResult:
    """Ask only for the blocks that were cut off, and keep only those."""
    max_ctx = compute_context_budget(settings.llm.max_context_size).max_ctx
    prompt = build_repair_prompt(
        paths, identity, schema, purpose, analysis, source_context, max_ctx, language
    )
    try:
        text = _complete(
            client,
            [
                Message("system", prompt),
                Message(
                    "user",
                    "Emit one complete FILE block for each requested path. Begin with `---FILE:`.",
                ),
            ],
            compute_generation_max_tokens(settings.llm.max_context_size),
        )
    except Exception as exc:  # provider failure here is not fatal to the ingest
        return ParseResult(warnings=[f"truncated-page repair failed: {exc}"])

    repaired = parse_blocks(text)
    allowed = {path for path in paths}
    kept = [block for block in repaired.files if block.path in allowed]
    dropped = len(repaired.files) - len(kept)
    warnings = list(repaired.warnings)
    if dropped:
        warnings.append(f"repair pass returned {dropped} unrequested file(s); ignored")
    return ParseResult(files=kept, reviews=[], warnings=warnings)


def _append_warning_log(project, identity: str, warnings: list[str]) -> None:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = "\n".join(
        [f"## {stamp} | {identity}", "", *[f"{i + 1}. {w}" for i, w in enumerate(warnings)], ""]
    )
    existing = project.read(".llm-wiki/ingest-warnings.log")
    project.write(
        ".llm-wiki/ingest-warnings.log",
        (existing.rstrip() + "\n\n" if existing.strip() else "") + entry,
    )


def _embed_written_pages(project, written: list[str], settings: Settings, warnings: list[str]) -> None:
    """Incrementally embed just the pages this ingest touched."""
    from ..errors import LlmWikiError
    from ..retrieval.keyword import Document, extract_wikilinks
    from ..frontmatter import extract_title, parse as parse_frontmatter

    documents: list[Document] = []
    for relative in dict.fromkeys(written):
        if not relative.endswith(".md"):
            continue
        content = project.read(relative)
        if not content.strip():
            continue
        parsed = parse_frontmatter(content)
        documents.append(
            Document(
                path=relative,
                title=extract_title(content, relative.rsplit("/", 1)[-1]),
                content=content,
                kind="wiki",
                links=extract_wikilinks(content),
                sources=parsed.get_list("sources"),
                page_type=parsed.get_str("type"),
            )
        )
    if not documents:
        return
    try:
        # Pruning is skipped here: this call only sees the pages this ingest
        # wrote, so a prune would delete every other page's vectors.
        _index_without_prune(project, documents, settings)
    except LlmWikiError as exc:
        warnings.append(f"embedding failed: {exc}")


def _index_without_prune(project, documents, settings) -> None:
    import hashlib

    from ..chunking import ChunkingOptions, chunk_markdown
    from ..embeddings import VectorStore, _embedding_text, embed_texts

    store = VectorStore(project.state_dir / "vectors.db")
    try:
        for document in documents:
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            if store.page_hash(document.path) == content_hash:
                continue
            chunks = chunk_markdown(document.content, ChunkingOptions())
            if not chunks:
                continue
            vectors = embed_texts(
                [_embedding_text(document.title, chunk) for chunk in chunks], settings.embedding
            )
            store.upsert_page(
                document.path,
                document.title,
                content_hash,
                settings.embedding.model,
                [(c.index, c.heading_path, c.text, v) for c, v in zip(chunks, vectors)],
            )
    finally:
        store.close()
