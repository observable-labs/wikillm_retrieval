"""Parse the FILE and REVIEW blocks step 2 emits.

The naive regex `---FILE:(.*?)---END FILE---` walks into five distinct
failure modes, every one of which was hit in production by llm_wiki. This
parser is a line state machine that handles all of them:

  H1  Windows CRLF line endings — a `\\n`-anchored regex matched nothing.
  H2  Stream truncation — the final block's closer never arrived and the
      block vanished silently. Can't be fixed at parse time, but it is
      reported so the caller can request a repair pass.
  H3  Marker variants — `--- END FILE ---`, `---end file---`, trailing
      spaces. All accepted, case-insensitively.
  H5  A literal `---END FILE---` inside a fenced code block (which happens
      the moment the wiki documents its own ingest format) truncated the
      page and spilled the rest into no-man's-land. Fence state is tracked,
      so a closer inside a fence is body text.
  H6  Empty path — matched, then silently dropped downstream.

Unsafe paths are rejected here, at the parse boundary, because this is the
only chokepoint every generated path passes through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..paths import is_safe_ingest_path

OPENER_LINE = re.compile(r"^---\s*FILE:\s*(.+?)\s*---\s*$", re.IGNORECASE)
CLOSER_LINE = re.compile(r"^---\s*END\s+FILE\s*---\s*$", re.IGNORECASE)
REVIEW_OPENER = re.compile(r"^---\s*REVIEW:\s*(.+?)\s*---\s*$", re.IGNORECASE)
REVIEW_CLOSER = re.compile(r"^---\s*END\s+REVIEW\s*---\s*$", re.IGNORECASE)
# CommonMark: 3+ backticks or tildes, indented at most 3 spaces.
FENCE_LINE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

REVIEW_TYPES = ("contradiction", "duplicate", "missing-page", "suggestion")
REVIEW_OPTIONS = ("Create Page", "Skip")


@dataclass
class FileBlock:
    path: str
    content: str


@dataclass
class ReviewBlock:
    type: str
    title: str
    description: str
    options: list[str] = field(default_factory=lambda: list(REVIEW_OPTIONS))
    pages: list[str] = field(default_factory=list)
    search: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    files: list[FileBlock] = field(default_factory=list)
    reviews: list[ReviewBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated_paths: list[str] = field(default_factory=list)


def parse_blocks(text: str) -> ParseResult:
    result = ParseResult()
    lines = text.replace("\r\n", "\n").split("\n")
    index = 0

    while index < len(lines):
        opener = OPENER_LINE.match(lines[index])
        if opener:
            index = _consume_file_block(lines, index, opener.group(1).strip(), result)
            continue
        review = REVIEW_OPENER.match(lines[index])
        if review:
            index = _consume_review_block(lines, index, review.group(1).strip(), result)
            continue
        index += 1

    return result


def _consume_file_block(lines: list[str], index: int, path: str, result: ParseResult) -> int:
    index += 1  # consume the opener
    content: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    closed = False

    while index < len(lines):
        line = lines[index]
        fence = FENCE_LINE.match(line)
        if fence:
            run = fence.group(1)
            if fence_char is None:
                fence_char, fence_len = run[0], len(run)
            elif run[0] == fence_char and len(run) >= fence_len:
                fence_char, fence_len = None, 0
            content.append(line)
            index += 1
            continue

        if fence_char is None and CLOSER_LINE.match(line):
            closed = True
            index += 1
            break

        content.append(line)
        index += 1

    if not closed:
        label = path or "(unnamed)"
        result.warnings.append(
            f'FILE block "{label}" was not closed before end of stream — likely truncation '
            "(the model hit max_tokens, or the connection dropped). Block dropped."
        )
        if is_safe_ingest_path(path):
            result.truncated_paths.append(path)
        return index

    if not path:
        result.warnings.append("FILE block with an empty path skipped (no path after `---FILE:`).")
        return index

    if not is_safe_ingest_path(path):
        result.warnings.append(
            f'FILE block with unsafe path "{path}" rejected (must live under wiki/, '
            "no `..`, no absolute paths, portable file names only)."
        )
        return index

    result.files.append(FileBlock(path=path, content="\n".join(content)))
    return index


def _consume_review_block(lines: list[str], index: int, header: str, result: ParseResult) -> int:
    index += 1
    body: list[str] = []
    closed = False
    while index < len(lines):
        if REVIEW_CLOSER.match(lines[index]):
            closed = True
            index += 1
            break
        body.append(lines[index])
        index += 1

    if not closed:
        result.warnings.append(f'REVIEW block "{header}" was not closed; dropped.')
        return index

    review_type, _, title = header.partition("|")
    review_type = review_type.strip().lower()
    if review_type not in REVIEW_TYPES:
        result.warnings.append(
            f'REVIEW block with unknown type "{review_type}" dropped '
            f"(expected one of {', '.join(REVIEW_TYPES)})."
        )
        return index

    description: list[str] = []
    options: list[str] = []
    pages: list[str] = []
    search: list[str] = []
    for line in body:
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("options:"):
            # Only the predefined labels are accepted; the model inventing an
            # action would produce a review nothing knows how to execute.
            options = [
                option.strip()
                for option in stripped.split(":", 1)[1].split("|")
                if option.strip() in REVIEW_OPTIONS
            ]
        elif lowered.startswith("pages:"):
            pages = [page.strip() for page in stripped.split(":", 1)[1].split(",") if page.strip()]
        elif lowered.startswith("search:"):
            search = [q.strip() for q in stripped.split(":", 1)[1].split("|") if q.strip()]
        else:
            description.append(line)

    result.reviews.append(
        ReviewBlock(
            type=review_type,
            title=title.strip() or "(untitled)",
            description="\n".join(description).strip(),
            options=options or list(REVIEW_OPTIONS),
            pages=pages,
            search=search,
        )
    )
    return index
