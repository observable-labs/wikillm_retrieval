"""The `llmwiki` command line.

    llmwiki init <path>          create a project
    llmwiki add <file>...        add a document (the two-step wiki build)
    llmwiki ask "<question>"     ask the project, retrieving over everything
    llmwiki search "<query>"     retrieval only, no LLM call — for debugging
    llmwiki embed                build/refresh the optional vector index
    llmwiki status               what's in the project and how it's configured
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, config, project as project_module
from .errors import LlmWikiError, ProjectError
from .ingest import ingest_document
from .query import ask
from .retrieval import search
from .templates import TEMPLATES

# ── output helpers ────────────────────────────────────────────────────────

def _supports_color(stream) -> bool:
    return (
        hasattr(stream, "isatty")
        and stream.isatty()
        and os.environ.get("TERM") != "dumb"
        and "NO_COLOR" not in os.environ
    )


class Printer:
    def __init__(self, quiet: bool = False, stream=None) -> None:
        self.quiet = quiet
        self.stream = stream or sys.stderr
        self.color = _supports_color(self.stream)

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def status(self, message: str) -> None:
        if not self.quiet:
            print(self._paint(f"  {message}", "2"), file=self.stream, flush=True)

    def info(self, message: str) -> None:
        if not self.quiet:
            print(message, file=self.stream, flush=True)

    def ok(self, message: str) -> None:
        if not self.quiet:
            print(self._paint(f"✓ {message}", "32"), file=self.stream, flush=True)

    def warn(self, message: str) -> None:
        print(self._paint(f"! {message}", "33"), file=self.stream, flush=True)

    def error(self, message: str) -> None:
        print(self._paint(f"error: {message}", "31"), file=sys.stderr, flush=True)


# ── commands ──────────────────────────────────────────────────────────────

def _resolve_project(args) -> project_module.Project:
    """Open the project for a command that must not guess which one it is.

    `add` rewrites the wiki and `ask` spends a model call — aiming either by
    ambient default is how a document lands in the wrong project. Discovery is
    still available, but only behind an explicit opt-out.
    """
    if not getattr(args, "project", None) and config.strict_project_enabled():
        raise ProjectError(
            "--project is required. Pass -p <dir>, or set LLMWIKI_STRICT_PROJECT=0 "
            "to allow discovery from $PWD and $LLMWIKI_PROJECT."
        )
    return project_module.open_project(args.project)


def cmd_init(args, printer: Printer) -> int:
    destination = Path(args.path).expanduser()
    created = project_module.create(destination, args.template, args.name)
    printer.ok(f"created {created.root}")
    printer.info("")
    printer.info(f"  schema.md    page types and routing rules ({args.template} template)")
    printer.info("  purpose.md   why this wiki exists — edit this first, it steers every ingest")
    printer.info("  raw/sources/ your documents (immutable)")
    printer.info("  wiki/        LLM-generated pages")
    printer.info("")
    printer.info(f"  next: llmwiki add <file> --project {created.root}")
    return 0


def cmd_add(args, printer: Printer) -> int:
    project = _resolve_project(args)
    settings = config.load(project.root, _overrides(args))

    results = []
    failures = 0
    for raw_path in args.paths:
        path = Path(raw_path).expanduser()
        targets = _expand(path, args.recursive)
        if not targets:
            printer.warn(f"{path}: nothing to add")
            failures += 1
            continue

        for target in targets:
            printer.info(f"→ {target.name}")
            folder = args.folder
            if not folder and args.recursive and path.is_dir():
                relative = target.parent.relative_to(path)
                folder = str(relative) if str(relative) != "." else ""
            try:
                result = ingest_document(
                    project,
                    target,
                    settings,
                    folder=folder,
                    force=args.force,
                    on_status=printer.status,
                )
            except LlmWikiError as exc:
                printer.error(f"{target.name}: {exc}")
                failures += 1
                continue

            results.append(result)
            if result.cached:
                printer.ok(f"{result.source_identity} unchanged ({len(result.pages)} page(s) already built)")
                continue

            printer.ok(f"{result.source_identity} → {len(result.pages)} page(s)")
            for written in result.pages:
                printer.info(f"    {written}")
            maintained = [p for p in result.files_written if p not in result.pages]
            if maintained:
                printer.info(f"    updated: {', '.join(maintained)}")
            if result.reviews:
                printer.info(f"    {len(result.reviews)} review item(s) → wiki/reviews.md")
                for review in result.reviews:
                    printer.info(f"      [{review.type}] {review.title}")
            for warning in result.warnings:
                printer.warn(f"    {warning}")

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "source": r.source_identity,
                        "stored_at": r.source_path,
                        "cached": r.cached,
                        "files": r.files_written,
                        "reviews": [
                            {"type": rev.type, "title": rev.title, "search": rev.search}
                            for rev in r.reviews
                        ],
                        "warnings": r.warnings,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    return 1 if failures else 0


def cmd_ask(args, printer: Printer) -> int:
    project = _resolve_project(args)
    settings = config.load(project.root, _overrides(args))
    question = " ".join(args.question).strip()
    if not question:
        printer.error("ask what?")
        return 2

    streaming = not args.json and not args.quiet
    if streaming:
        printer.status("retrieving…")

    printed_any = False

    def on_token(chunk: str) -> None:
        nonlocal printed_any
        if not printed_any:
            printed_any = True
        sys.stdout.write(chunk)
        sys.stdout.flush()

    answer = ask(
        project,
        question,
        settings,
        top_k=args.top_k,
        include_sources=None if args.sources is None else args.sources,
        on_token=on_token if streaming else None,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.text,
                    "citations": [
                        {
                            "n": c.number,
                            "path": c.path,
                            "title": c.title,
                            "kind": c.kind,
                            "cited": c.cited,
                            "related_via": c.graph_related_to,
                        }
                        for c in answer.citations
                    ],
                    "retrieval": {
                        "mode": answer.retrieval.mode if answer.retrieval else "none",
                        "pages_found": answer.pages_found,
                        "pages_used": answer.pages_used,
                        "pages_cited": answer.pages_cited,
                    },
                    "notes": answer.notes,
                },
                indent=2,
            )
        )
        return 0

    if not printed_any:
        print(answer.text)
    else:
        sys.stdout.write("\n")

    for note in answer.notes:
        printer.warn(note)
    if answer.citations:
        cited = [c for c in answer.citations if c.cited]
        printer.info("")
        printer.info("Sources:")
        # A model that cites nothing still gets a source list — better an
        # over-long one than an answer with no way back to the pages.
        for citation in cited or answer.citations:
            printer.info(f"  [{citation.number}] {citation.path}{_via(citation.graph_related_to)}")
        uncited = len(answer.citations) - len(cited)
        if cited and uncited:
            noun = "page" if uncited == 1 else "pages"
            printer.info(f"  (+{uncited} retrieved {noun} the answer didn't cite)")
        mode = answer.retrieval.mode if answer.retrieval else "keyword"
        printer.info("")
        printer.status(
            f"{mode} retrieval · {answer.pages_found} matched · {answer.pages_used} packed "
            f"· {answer.pages_cited} cited · {answer.context_chars:,} chars of context"
        )
    return 0


def _via(names: list[str], limit: int = 3) -> str:
    """The graph neighbours a page was reached through, capped.

    A hub page is linked from nearly every other result, so the full list runs
    to a dozen titles on one line and buries the citation it annotates. Three
    names carry the signal — that graph expansion fired, and roughly from
    where; `search` prints the complete list for when that isn't enough.
    """
    if not names:
        return ""
    rest = len(names) - limit
    return f"  (via {', '.join(names[:limit])}{f' +{rest} more' if rest > 0 else ''})"


def cmd_search(args, printer: Printer) -> int:
    project = project_module.open_project(args.project)
    settings = config.load(project.root, _overrides(args))
    response = search(
        project,
        " ".join(args.query),
        top_k=args.top_k,
        include_sources=settings.search_sources if args.sources is None else args.sources,
        embedding_config=settings.embedding,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "mode": response.mode,
                    "token_hits": response.token_hits,
                    "vector_hits": response.vector_hits,
                    "graph_hits": response.graph_hits,
                    "results": [
                        {
                            "path": r.path,
                            "title": r.title,
                            "kind": r.kind,
                            "score": round(r.score, 4),
                            "vector_score": r.vector_score,
                            "graph_related_to": r.graph_related_to,
                            "snippet": r.snippet,
                        }
                        for r in response.results
                    ],
                },
                indent=2,
            )
        )
        return 0

    for note in response.notes:
        printer.warn(note)
    if not response.results:
        printer.info("no matches")
        return 0
    print(
        f"{response.mode} · {response.token_hits} keyword · "
        f"{response.vector_hits} vector · {response.graph_hits} graph\n"
    )
    for result in response.results:
        marker = "src" if result.kind == "source" else "   "
        print(f"{result.score:8.3f} {marker} {result.path}")
        print(f"         {result.title}")
        if result.snippet:
            print(f"         {result.snippet[:160]}")
        if result.graph_related_to:
            print(f"         via graph: {', '.join(result.graph_related_to)}")
        print()
    return 0


def cmd_embed(args, printer: Printer) -> int:
    from .embeddings import index_documents
    from .retrieval import load_documents

    project = project_module.open_project(args.project)
    settings = config.load(project.root, _overrides(args))
    settings.embedding.require_enabled()

    documents = load_documents(project, include_sources=False)
    printer.info(f"embedding {len(documents)} wiki page(s) with {settings.embedding.model}")

    def on_progress(done: int, total: int, path: str, state: str) -> None:
        if state != "cached":
            printer.status(f"[{done}/{total}] {state}: {path}")

    summary = index_documents(
        project, documents, settings.embedding, force=args.force, on_progress=on_progress
    )
    printer.ok(
        f"{summary['indexed']} indexed, {summary['skipped']} unchanged, "
        f"{summary['failed']} failed, {summary['pruned']} pruned"
    )
    printer.info(
        f"  store: {summary['pages_stored']} pages / {summary['chunks_stored']} chunks "
        f"→ {project.state_dir / 'vectors.db'}"
    )
    return 1 if summary["failed"] else 0


def cmd_status(args, printer: Printer) -> int:
    from .ingest import IngestCache

    project = project_module.open_project(args.project)
    settings = config.load(project.root, _overrides(args))
    pages = project.wiki_pages()
    sources = project.source_files()
    cache = IngestCache.load(project)

    vectors = project.state_dir / "vectors.db"
    vector_summary = "not built"
    if vectors.exists():
        from .embeddings import VectorStore

        with VectorStore(vectors) as store:
            stored_pages, stored_chunks = store.count()
        vector_summary = f"{stored_pages} pages / {stored_chunks} chunks"

    if args.json:
        print(
            json.dumps(
                {
                    "project": str(project.root),
                    "pages": len(pages),
                    "sources": len(sources),
                    "ingested": cache.identities(),
                    "provider": settings.llm.provider,
                    "model": settings.llm.model or "(default)",
                    "effort": _effort_summary(settings.llm),
                    "embedding_model": settings.embedding.model or None,
                    "vectors": vector_summary,
                },
                indent=2,
            )
        )
        return 0

    printer.info(f"project   {project.root}")
    printer.info(f"pages     {len(pages)} under wiki/")
    printer.info(f"sources   {len(sources)} under raw/sources/ ({len(cache.identities())} ingested)")
    printer.info(f"provider  {settings.llm.provider} · {settings.llm.model or '(default)'}")
    printer.info(f"context   {settings.llm.max_context_size:,} chars")
    effort = _effort_summary(settings.llm)
    printer.info(f"thinking  ask {effort['ask']} · ingest {effort['ingest']}")
    printer.info(
        f"vectors   {vector_summary}"
        + (f" · {settings.embedding.model}" if settings.embedding.model else " (disabled)")
    )
    return 0


# ── plumbing ──────────────────────────────────────────────────────────────

def _effort_summary(llm) -> dict[str, str]:
    """What each lane will actually ask the model for.

    Worth printing: `effort` was configurable long before anything outside
    the Anthropic client read it, so a value set in `.env` looked applied
    while every OpenAI-compatible call ignored it.
    """
    from .reasoning import resolve

    model = llm.model
    return {
        "ask": resolve(llm.effort, model) or "provider default",
        "ingest": resolve(llm.for_ingest().effort, model) or "provider default",
    }


def _overrides(args) -> dict:
    return {
        key: getattr(args, key, None)
        for key in ("provider", "model", "base_url", "api_key", "max_context_size", "embedding_model")
        if getattr(args, key, None) is not None
    }


def _expand(path: Path, recursive: bool) -> list[Path]:
    from .parsers import supported_extensions

    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    allowed = supported_extensions()
    candidates = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        child
        for child in candidates
        if child.is_file()
        and child.suffix.lower() in allowed
        and not any(part.startswith(".") for part in child.parts)
    )


def _add_common(parser: argparse.ArgumentParser, *, project_required: bool = False) -> None:
    parser.add_argument(
        "--project",
        "-p",
        help=(
            "project directory (required unless LLMWIKI_STRICT_PROJECT=0)"
            if project_required
            else "project directory (default: found from $PWD)"
        ),
    )
    parser.add_argument("--provider", choices=("anthropic", "openai"), help="LLM provider")
    parser.add_argument("--model", help="model id")
    parser.add_argument("--base-url", dest="base_url", help="API base URL")
    parser.add_argument("--api-key", dest="api_key", help="API key (prefer an env var)")
    parser.add_argument(
        "--max-context",
        dest="max_context_size",
        type=int,
        help="context budget in characters",
    )
    parser.add_argument("--embedding-model", dest="embedding_model", help="enable vector search with this model")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmwiki",
        description="Build and query a self-maintaining wiki from your documents.",
    )
    parser.add_argument("--version", action="version", version=f"llmwiki {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a new wiki project")
    init.add_argument("path", help="directory to create the project in")
    init.add_argument("--name", help="project name (default: directory name)")
    init.add_argument(
        "--template",
        default="general",
        choices=sorted(TEMPLATES),
        help="scenario template: sets the page types in schema.md",
    )
    init.add_argument("--quiet", "-q", action="store_true")
    init.set_defaults(func=cmd_init)

    add = subparsers.add_parser("add", help="add a document to the project")
    add.add_argument("paths", nargs="+", help="files or directories to ingest")
    add.add_argument("--folder", default="", help="folder context hint passed to the LLM")
    add.add_argument("-r", "--recursive", action="store_true", help="walk directories")
    add.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    _add_common(add, project_required=True)
    add.set_defaults(func=cmd_add)

    ask_parser = subparsers.add_parser("ask", help="ask a question, retrieving over the whole project")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--top-k", type=int, default=20, help="pages to retrieve (default 20)")
    ask_parser.add_argument(
        "--sources",
        dest="sources",
        action="store_true",
        default=None,
        help="include raw source documents in retrieval",
    )
    ask_parser.add_argument(
        "--no-sources",
        dest="sources",
        action="store_false",
        help="retrieve over wiki pages only",
    )
    _add_common(ask_parser, project_required=True)
    ask_parser.set_defaults(func=cmd_ask)

    search_parser = subparsers.add_parser("search", help="retrieval only, no LLM call")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--sources", dest="sources", action="store_true", default=None)
    search_parser.add_argument("--no-sources", dest="sources", action="store_false")
    _add_common(search_parser)
    search_parser.set_defaults(func=cmd_search)

    embed = subparsers.add_parser("embed", help="build or refresh the vector index")
    embed.add_argument("--force", action="store_true", help="re-embed unchanged pages")
    _add_common(embed)
    embed.set_defaults(func=cmd_embed)

    status = subparsers.add_parser("status", help="show project contents and configuration")
    _add_common(status)
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    printer = Printer(quiet=getattr(args, "quiet", False))
    try:
        return args.func(args, printer)
    except LlmWikiError as exc:
        printer.error(str(exc))
        return 1
    except KeyboardInterrupt:
        printer.error("interrupted")
        return 130
