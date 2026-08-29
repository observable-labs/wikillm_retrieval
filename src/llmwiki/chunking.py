"""Two chunkers, for two different jobs.

`split_source_into_semantic_chunks` cuts a document that is too large for one
analysis pass into overlapping pieces, each carrying its heading breadcrumb,
so step 1 of ingest can read a long PDF in sequence and still know where it
is in the document.

`chunk_markdown` cuts a *wiki page* into embedding-sized pieces. Its contract
(ported from llm_wiki's `text-chunker.ts`):

  1. every chunk carries a `heading_path` breadcrumb, so a short chunk is
     never semantically orphaned;
  2. split priority is headings > paragraphs > lines > sentences > words >
     hard slice, each level engaging only when the level above still
     overflows;
  3. fenced code blocks and markdown tables are never split — a torn table
     embeds as garbage;
  4. YAML frontmatter is stripped so metadata doesn't pollute vectors;
  5. adjacent chunks overlap, so an idea severed at a boundary survives;
  6. chunks below `min_chars` are merged into a neighbour;
  7. pure and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
TABLE_ROW_RE = re.compile(r"^\s*\|")
FRONTMATTER_RE = re.compile(r"^---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)
SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？;；])\s+")


@dataclass(frozen=True)
class ChunkingOptions:
    target_chars: int = 1000
    max_chars: int = 1500
    min_chars: int = 200
    overlap_chars: int = 200


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    heading_path: str
    char_start: int
    char_end: int
    oversized: bool = False


@dataclass(frozen=True)
class SourceChunk:
    index: int
    total: int
    heading_path: str
    overlap_before: str
    main: str

    @property
    def text(self) -> str:
        if not self.overlap_before:
            return self.main
        return f"[...continues from previous section...]\n{self.overlap_before}\n\n{self.main}"


def strip_frontmatter(content: str) -> tuple[str, int]:
    """Return (body, offset-of-body-in-original)."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return content, 0
    return content[match.end():], match.end()


# ── atomic blocks ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Block:
    text: str
    heading_path: str
    start: int
    end: int
    atomic: bool  # code fence or table: never split


def _iter_blocks(body: str, offset: int) -> list[_Block]:
    """Split a body into paragraph-ish blocks, tracking the heading path.

    Fenced code and contiguous table rows come back as single atomic blocks.
    """
    lines = body.split("\n")
    blocks: list[_Block] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_start = 0
    position = 0
    fence: str | None = None
    fence_len = 0
    in_table = False

    def heading_path() -> str:
        return " > ".join(f"{'#' * level} {title}" for level, title in heading_stack)

    def flush(end: int, atomic: bool = False) -> None:
        nonlocal buffer, buffer_start
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(
                _Block(
                    text=text,
                    heading_path=heading_path(),
                    start=offset + buffer_start,
                    end=offset + end,
                    atomic=atomic,
                )
            )
        buffer = []

    for line in lines:
        line_start = position
        position += len(line) + 1

        fence_match = FENCE_RE.match(line)
        if fence_match:
            run = fence_match.group(1)
            if fence is None:
                # A fence opens: everything buffered so far is a normal block.
                flush(line_start)
                buffer_start = line_start
                fence, fence_len = run[0], len(run)
                buffer.append(line)
            elif run[0] == fence and len(run) >= fence_len:
                buffer.append(line)
                fence, fence_len = None, 0
                flush(position, atomic=True)
                buffer_start = position
            else:
                buffer.append(line)
            continue
        if fence is not None:
            buffer.append(line)
            continue

        is_table_row = bool(TABLE_ROW_RE.match(line))
        if is_table_row and not in_table:
            flush(line_start)
            buffer_start = line_start
            in_table = True
            buffer.append(line)
            continue
        if in_table:
            if is_table_row:
                buffer.append(line)
                continue
            flush(line_start, atomic=True)
            buffer_start = line_start
            in_table = False

        heading = HEADING_RE.match(line)
        if heading:
            flush(line_start)
            buffer_start = line_start
            level = len(heading.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading.group(2).strip()))
            continue

        if not line.strip():
            flush(line_start)
            buffer_start = position
            continue

        if not buffer:
            buffer_start = line_start
        buffer.append(line)

    flush(position, atomic=in_table or fence is not None)
    return blocks


# ── embedding chunks ──────────────────────────────────────────────────────

def _split_text(text: str, max_chars: int) -> list[str]:
    """Recursive splitter: paragraphs -> lines -> sentences -> words -> slice."""
    if len(text) <= max_chars:
        return [text]
    for separator in ("\n\n", "\n"):
        if separator in text:
            pieces = _regroup(text.split(separator), separator, max_chars)
            if len(pieces) > 1:
                return [p for piece in pieces for p in _split_text(piece, max_chars)]
    sentences = SENTENCE_END_RE.split(text)
    if len(sentences) > 1:
        pieces = _regroup(sentences, " ", max_chars)
        return [p for piece in pieces for p in _split_text(piece, max_chars)]
    words = text.split(" ")
    if len(words) > 1:
        pieces = _regroup(words, " ", max_chars)
        return [p for piece in pieces for p in _split_text(piece, max_chars)]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _regroup(pieces: list[str], separator: str, max_chars: int) -> list[str]:
    """Greedily reassemble split pieces up to `max_chars`."""
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if current and len(candidate) > max_chars:
            out.append(current)
            current = piece
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def chunk_markdown(content: str, options: ChunkingOptions | None = None) -> list[Chunk]:
    opts = options or ChunkingOptions()
    max_chars = max(opts.max_chars, opts.target_chars)
    overlap = opts.overlap_chars if opts.overlap_chars < opts.target_chars else opts.target_chars // 2

    body, offset = strip_frontmatter(content)
    if not body.strip():
        return []

    blocks = _iter_blocks(body, offset)
    if not blocks:
        return []

    # Group blocks into target-sized runs that share a heading path.
    grouped: list[tuple[str, list[_Block]]] = []
    current: list[_Block] = []
    current_len = 0
    current_heading = blocks[0].heading_path
    for block in blocks:
        if current and (
            block.heading_path != current_heading
            or current_len + len(block.text) > opts.target_chars
        ):
            grouped.append((current_heading, current))
            current, current_len = [], 0
            current_heading = block.heading_path
        current.append(block)
        current_len += len(block.text) + 2
    if current:
        grouped.append((current_heading, current))

    raw: list[tuple[str, str, int, int, bool]] = []
    for heading, group in grouped:
        text = "\n\n".join(b.text for b in group)
        start, end = group[0].start, group[-1].end
        if len(text) <= max_chars or all(b.atomic for b in group):
            raw.append((text, heading, start, end, len(text) > max_chars))
            continue
        cursor = start
        for piece in _split_text(text, max_chars):
            raw.append((piece, heading, cursor, cursor + len(piece), len(piece) > max_chars))
            cursor += len(piece)

    # Merge sub-minimum chunks into a neighbour sharing the same heading.
    merged: list[list] = []
    for text, heading, start, end, oversized in raw:
        if (
            merged
            and len(text) < opts.min_chars
            and merged[-1][1] == heading
            and len(merged[-1][0]) + len(text) <= max_chars
        ):
            merged[-1][0] = f"{merged[-1][0]}\n\n{text}"
            merged[-1][3] = end
            continue
        merged.append([text, heading, start, end, oversized])

    chunks: list[Chunk] = []
    for index, (text, heading, start, end, oversized) in enumerate(merged):
        prefixed = text
        if overlap and index > 0 and merged[index - 1][1] == heading:
            tail = merged[index - 1][0][-overlap:].lstrip()
            if tail:
                prefixed = f"{tail}\n\n{text}"
        chunks.append(
            Chunk(
                index=index,
                text=prefixed,
                heading_path=heading,
                char_start=start,
                char_end=end,
                oversized=oversized,
            )
        )
    return chunks


# ── long-source chunks ────────────────────────────────────────────────────

def _overlap_suffix(text: str, max_chars: int) -> str:
    """The tail of a chunk, cut at a paragraph or sentence boundary."""
    if len(text) <= max_chars:
        return text
    raw = text[-max_chars:]
    paragraph = re.search(r"\n\s*\n", raw)
    if paragraph and len(raw) - paragraph.start() > max_chars * 0.4:
        return raw[paragraph.start():].strip()
    sentence = re.search(r"[.!?。！？]\s+", raw)
    if sentence and len(raw) - sentence.start() > max_chars * 0.4:
        return raw[sentence.start() + 1:].strip()
    return raw.strip()


def split_source_into_semantic_chunks(
    content: str,
    target_chars: int,
    overlap_chars: int,
) -> list[SourceChunk]:
    """Cut an over-budget source into ordered, overlapping analysis chunks."""
    target = max(1_000, target_chars)
    blocks = _iter_blocks(content, 0)
    if not blocks:
        return []

    groups: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0
    current_heading = blocks[0].heading_path

    def flush() -> None:
        nonlocal current, current_len
        text = "\n\n".join(current).strip()
        if text:
            groups.append((text, current_heading))
        current, current_len = [], 0

    for block in blocks:
        # An oversized atomic block gets a chunk of its own rather than
        # being torn apart.
        next_len = current_len + len(block.text) + (2 if current else 0)
        if current and next_len > target:
            flush()
        if not current:
            current_heading = block.heading_path
        current.append(block.text)
        current_len += len(block.text) + (2 if len(current) > 1 else 0)
    flush()

    total = len(groups)
    return [
        SourceChunk(
            index=index + 1,
            total=total,
            heading_path=heading,
            overlap_before=_overlap_suffix(groups[index - 1][0], overlap_chars) if index > 0 else "",
            main=text,
        )
        for index, (text, heading) in enumerate(groups)
    ]


# The index-time half of this package's public surface. `chunk_markdown` and
# `split_source_into_semantic_chunks` are what turn a document into the units a
# lexical index and a vector store hold, and a second consumer that reimplements
# them has an index whose boundaries silently disagree with this one's. Named
# here so they are supported rather than merely importable.
__all__ = [
    "Chunk",
    "ChunkingOptions",
    "SourceChunk",
    "chunk_markdown",
    "split_source_into_semantic_chunks",
    "strip_frontmatter",
]
