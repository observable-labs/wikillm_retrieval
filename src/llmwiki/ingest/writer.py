"""Turn parsed FILE blocks into files on disk, then fix up the aggregates.

Three things happen here that the model is not trusted to do itself:

* `sources` is canonicalized — the page's frontmatter must reference the
  document it actually came from, because that field is both the audit trail
  and a retrieval signal (two pages sharing a source are related).
* `wiki/index.md` is updated deterministically instead of being regenerated
  by the model, so a large wiki is never rewritten through model output and
  can't lose entries to a truncated response.
* `wiki/log.md` gets a deterministic entry if the model didn't emit one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import frontmatter as fm
from ..paths import file_name, normalize_path
from ..project import AGGREGATE_PAGES
from .blocks import FileBlock

LOG_PATH = "wiki/log.md"
INDEX_PATH = "wiki/index.md"
RECENT_SECTION = "## Recently Updated"
MAX_RECENT_ENTRIES = 200
INDEX_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


@dataclass
class WriteResult:
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_index_target(target: str) -> str:
    value = normalize_path(target)
    if value.lower().startswith("wiki/"):
        value = value[5:]
    if value.lower().endswith(".md"):
        value = value[:-3]
    return value.lower()


def is_aggregate(relative_path: str) -> bool:
    return normalize_path(relative_path).lower() in AGGREGATE_PAGES


def _is_log(relative_path: str) -> bool:
    return normalize_path(relative_path).lower() == LOG_PATH


def canonicalize_sources(content: str, source_identity: str) -> str:
    """Guarantee the page's `sources` includes — and is anchored to — its origin.

    Also drops references the model sometimes invents: aggregate pages, the
    state directory, absolute paths, and anything with a `..` segment.
    """
    parsed = fm.parse(content)
    if not parsed.present:
        return content

    identity_key = normalize_path(source_identity).lower()
    identity_base = file_name(source_identity).lower()

    canonical: list[str] = []
    for value in parsed.get_list("sources"):
        normalized = normalize_path(value).lstrip("./")
        if not normalized or normalized.startswith("/") or re.match(r"^[a-z]:/", normalized, re.I):
            continue
        if any(part == ".." for part in normalized.split("/")):
            continue
        lowered = normalized.lower()
        if lowered in {"wiki/index.md", "wiki/overview.md", "wiki/log.md"}:
            continue
        if lowered == ".llm-wiki" or lowered.startswith(".llm-wiki/"):
            continue
        if lowered == identity_key or ("/" not in normalized and lowered == identity_base):
            canonical.append(source_identity)
        else:
            canonical.append(normalized)

    if not any(normalize_path(value).lower() == identity_key for value in canonical):
        canonical.append(source_identity)

    seen: set[str] = set()
    deduped: list[str] = []
    for value in canonical:
        key = normalize_path(value).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)

    return fm.write_sources(content, deduped)


def stamp_dates(content: str, date: str) -> str:
    """Replace `YYYY-MM-DD` placeholders and fill missing created/updated."""
    parsed = fm.parse(content)
    if not parsed.present:
        return content
    data = dict(parsed.data)
    for key in ("created", "updated"):
        value = data.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            data[key] = date
    return fm.render(data, parsed.body)


def write_blocks(
    project,
    blocks: list[FileBlock],
    source_identity: str,
    date: str,
) -> WriteResult:
    """Write every FILE block, routing log entries to an append."""
    result = WriteResult()
    for block in blocks:
        relative = normalize_path(block.path)
        content = block.content.strip("\n") + "\n"

        if is_aggregate(relative):
            # index.md/overview.md are app-managed; a model-authored rewrite
            # of a large aggregate silently drops entries.
            result.skipped.append(relative)
            continue

        if _is_log(relative):
            append_log_entry(project, content)
            if LOG_PATH not in result.written:
                result.written.append(LOG_PATH)
            continue

        content = stamp_dates(content, date)
        content = canonicalize_sources(content, source_identity)
        try:
            project.write(relative, content)
        except OSError as exc:
            result.warnings.append(f"could not write {relative}: {exc}")
            continue
        result.written.append(relative)
    return result


def append_log_entry(project, entry: str) -> None:
    existing = project.read(LOG_PATH).rstrip()
    body = entry.strip()
    if not body:
        return
    if existing:
        project.write(LOG_PATH, f"{existing}\n\n{body}\n")
    else:
        project.write(LOG_PATH, f"# Wiki Log\n\n{body}\n")


def deterministic_log_entry(project, source_identity: str, date: str) -> None:
    """The fallback log line when the model didn't emit one.

    The `## [date] ingest | name` prefix is what makes the log greppable:
    `grep "^## \\[" wiki/log.md | tail -5`.
    """
    append_log_entry(project, f"## [{date}] ingest | {source_identity}")


def update_index(project, written_paths: list[str]) -> bool:
    """Add newly created pages to index.md's `## Recently Updated` section."""
    candidates = [
        normalize_path(path)
        for path in dict.fromkeys(written_paths)
        if path.startswith("wiki/")
        and path.endswith(".md")
        and not is_aggregate(path)
        and not _is_log(path)
    ]
    if not candidates:
        return False

    index = project.read(INDEX_PATH) or "# Wiki Index\n"
    known = {_normalize_index_target(match.group(1)) for match in INDEX_LINK_RE.finditer(index)}

    additions: list[str] = []
    for path in candidates:
        target = path[5:]
        if target.lower().endswith(".md"):
            target = target[:-3]
        if _normalize_index_target(target) in known:
            continue
        content = project.read(path)
        title = fm.parse(content).get_str("title").strip() or file_name(path)[:-3]
        additions.append(f"- [[{target}]] — {title}")

    if not additions:
        return False
    project.write(INDEX_PATH, _update_recent_section(index, additions))
    return True


def _update_recent_section(index: str, additions: list[str]) -> str:
    """Prepend to `## Recently Updated`, bounded so the index can't grow forever."""
    lines = index.rstrip().split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == RECENT_SECTION)
    except StopIteration:
        start = -1

    if start < 0:
        recent = additions[:MAX_RECENT_ENTRIES]
        return "\n".join([*lines, "", RECENT_SECTION, *recent, ""])

    section_end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^##\s+", lines[i])),
        -1,
    )
    prefix = lines[:start]
    body = lines[start + 1 : section_end if section_end >= 0 else len(lines)]
    suffix = lines[section_end:] if section_end >= 0 else []
    existing = [line for line in body if re.match(r"^-\s+", line)]

    merged: list[str] = []
    seen: set[str] = set()
    for line in [*additions, *existing]:
        if line in seen:
            continue
        seen.add(line)
        merged.append(line)

    out = [*prefix, "", RECENT_SECTION, *merged[:MAX_RECENT_ENTRIES]]
    if suffix:
        out += ["", *suffix]
    return "\n".join(out) + "\n"


def store_source(project, source_path: Path, folder: str = "") -> Path:
    """Copy a document into `raw/sources/`, never overwriting a different file."""
    import shutil

    destination_dir = project.sources_dir / folder if folder else project.sources_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name

    if destination.resolve() == source_path.resolve():
        return destination
    if destination.exists() and destination.read_bytes() != source_path.read_bytes():
        stem, suffix = destination.stem, destination.suffix
        counter = 2
        while destination.exists():
            destination = destination_dir / f"{stem}-{counter}{suffix}"
            counter += 1
    shutil.copy2(source_path, destination)
    return destination


def record_reviews(project, source_identity: str, reviews) -> Path | None:
    """Append review items to `wiki/reviews.md` for a human to work through."""
    if not reviews:
        return None
    lines = [f"\n## {source_identity}\n"]
    for review in reviews:
        lines.append(f"### [{review.type}] {review.title}")
        if review.description:
            lines.append("")
            lines.append(review.description)
        lines.append("")
        lines.append(f"- Options: {' | '.join(review.options)}")
        if review.pages:
            lines.append(f"- Pages: {', '.join(review.pages)}")
        if review.search:
            lines.append(f"- Suggested searches: {' | '.join(review.search)}")
        lines.append("")

    existing = project.read("wiki/reviews.md") or "# Review Queue\n\nItems the ingest flagged for human judgment.\n"
    return project.write("wiki/reviews.md", existing.rstrip() + "\n" + "\n".join(lines))
