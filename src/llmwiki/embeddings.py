"""Optional vector lane: an OpenAI-compatible embedder plus a SQLite store.

llm_wiki keeps chunk vectors in LanceDB behind a Rust backend. A pure-Python
port needs neither: personal-scale wikis are thousands of chunks, not
millions, and a brute-force cosine scan over a SQLite table answers in
milliseconds while keeping the store a single portable file with no native
dependency.

Vectors are stored as float32 blobs and L2-normalized on write, so scoring is
a plain dot product. NumPy is used when present and a pure-Python fallback
runs when it isn't.
"""

from __future__ import annotations

import array
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .chunking import ChunkingOptions, chunk_markdown
from .config import EmbeddingConfig
from .errors import ProviderError
from .llm._http import post_json

try:  # optional accelerator
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by environment, not tests
    _np = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    page_id      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    vector       BLOB NOT NULL,
    dims         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_page_id ON chunks(page_id);
CREATE TABLE IF NOT EXISTS pages (
    page_id   TEXT PRIMARY KEY,
    title     TEXT NOT NULL DEFAULT '',
    hash      TEXT NOT NULL,
    model     TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class ChunkHit:
    page_id: str
    chunk_index: int
    heading_path: str
    text: str
    score: float


@dataclass
class PageHit:
    page_id: str
    score: float
    matched_chunks: list[ChunkHit]


# ── embedding provider ────────────────────────────────────────────────────

def embed_texts(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """POST /v1/embeddings. Batched, with a halving retry for oversize input."""
    config.require_enabled()
    if not texts:
        return []

    url = _embeddings_url(config.base_url)
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    out: list[list[float]] = []

    for start in range(0, len(texts), max(1, config.batch_size)):
        batch = texts[start : start + max(1, config.batch_size)]
        out.extend(_embed_batch(url, batch, headers, config))
    return out


def _embed_batch(url: str, batch: list[str], headers: dict, config: EmbeddingConfig) -> list[list[float]]:
    payload: dict = {"model": config.model, "input": batch}
    if config.dimensions:
        payload["dimensions"] = config.dimensions
    try:
        response = post_json(url, payload, headers, config.timeout)
    except ProviderError:
        # Some servers cap request size rather than input count; halving the
        # batch is the cheapest way to find the ceiling without configuration.
        if len(batch) > 1:
            middle = len(batch) // 2
            return _embed_batch(url, batch[:middle], headers, config) + _embed_batch(
                url, batch[middle:], headers, config
            )
        raise

    data = response.get("data")
    if not isinstance(data, list) or len(data) != len(batch):
        raise ProviderError(
            f"embedding endpoint returned {len(data) if isinstance(data, list) else 'no'} "
            f"vectors for {len(batch)} inputs"
        )
    vectors: list[list[float]] = []
    for item in sorted(data, key=lambda d: d.get("index", 0)):
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ProviderError("embedding endpoint returned an empty vector")
        vectors.append([float(value) for value in vector])
    return vectors


def embed_query(text: str, config: EmbeddingConfig) -> list[float]:
    return embed_texts([text], config)[0]


def _embeddings_url(base_url: str) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


# ── store ─────────────────────────────────────────────────────────────────

def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class VectorStore:
    """Chunk vectors for one project, in `.llm-wiki/vectors.db`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def page_hash(self, page_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT hash FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        return row[0] if row else None

    def count(self) -> tuple[int, int]:
        chunks = self._connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        pages = self._connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        return pages, chunks

    def delete_page(self, page_id: str) -> None:
        self._connection.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        self._connection.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))
        self._connection.commit()

    def upsert_page(
        self,
        page_id: str,
        title: str,
        content_hash: str,
        model: str,
        chunks: list[tuple[int, str, str, list[float]]],
    ) -> None:
        """Replace a page's chunks atomically: delete-then-insert."""
        cursor = self._connection
        cursor.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        for index, heading_path, text, vector in chunks:
            normalized = _normalize(vector)
            blob = array.array("f", normalized).tobytes()
            cursor.execute(
                "INSERT INTO chunks (page_id, chunk_index, heading_path, text, vector, dims) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (page_id, index, heading_path, text, blob, len(normalized)),
            )
        cursor.execute(
            "INSERT INTO pages (page_id, title, hash, model) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(page_id) DO UPDATE SET title=excluded.title, hash=excluded.hash, model=excluded.model",
            (page_id, title, content_hash, model),
        )
        cursor.commit()

    def prune(self, keep_page_ids: set[str]) -> int:
        """Drop embeddings for pages that no longer exist on disk."""
        rows = self._connection.execute("SELECT page_id FROM pages").fetchall()
        stale = [row[0] for row in rows if row[0] not in keep_page_ids]
        for page_id in stale:
            self.delete_page(page_id)
        return len(stale)

    def search(self, query_vector: list[float], top_k: int = 30) -> list[ChunkHit]:
        rows = self._connection.execute(
            "SELECT page_id, chunk_index, heading_path, text, vector, dims FROM chunks"
        ).fetchall()
        if not rows:
            return []

        query = _normalize(query_vector)
        dims = len(query)
        hits: list[ChunkHit] = []

        if _np is not None:
            query_array = _np.asarray(query, dtype=_np.float32)
            for page_id, index, heading, text, blob, row_dims in rows:
                if row_dims != dims:
                    continue  # a re-embedding with a different model is in flight
                vector = _np.frombuffer(blob, dtype=_np.float32)
                hits.append(
                    ChunkHit(page_id, index, heading, text, float(vector @ query_array))
                )
        else:
            for page_id, index, heading, text, blob, row_dims in rows:
                if row_dims != dims:
                    continue
                vector = array.array("f")
                vector.frombytes(blob)
                hits.append(
                    ChunkHit(
                        page_id,
                        index,
                        heading,
                        text,
                        sum(a * b for a, b in zip(vector, query)),
                    )
                )

        hits.sort(key=lambda hit: -hit.score)
        return hits[:top_k]


def group_by_page(hits: list[ChunkHit], top_k: int = 10) -> list[PageHit]:
    """Blend chunk scores into page scores.

    A page scores its best chunk plus a capped share of the rest, so broad
    coverage counts for something without letting chunk count run away: the
    tail contribution is capped at `1 - top`, which bounds any page at 1.0.

    Note this is a genuine trade-off, not a strict ordering — a page with
    many moderate chunks can outrank one with a single excellent chunk. That
    is the behaviour llm_wiki settled on empirically, and it is usually the
    right call for a wiki page, where sustained relevance beats one lucky
    paragraph. RRF fusion downstream reduces how much the exact value
    matters, since only the rank survives.
    """
    buckets: dict[str, list[ChunkHit]] = {}
    for hit in hits:
        buckets.setdefault(hit.page_id, []).append(hit)

    pages: list[PageHit] = []
    for page_id, chunks in buckets.items():
        chunks.sort(key=lambda hit: -hit.score)
        top = chunks[0].score
        tail = sum(hit.score for hit in chunks[1:])
        blended = top + min(tail * 0.3, max(0.0, 1.0 - top))
        pages.append(PageHit(page_id=page_id, score=blended, matched_chunks=chunks[:3]))

    pages.sort(key=lambda page: -page.score)
    return pages[:top_k]


def index_documents(
    project,
    documents,
    config: EmbeddingConfig,
    force: bool = False,
    on_progress=None,
) -> dict:
    """Embed every document whose content hash changed. Returns a summary.

    Everything it is given, not only `kind == "wiki"`. Filtering sources out
    here made the vector lane cover a strict subset of the corpus, and fusing a
    partial-coverage ranking with a full-coverage one is not a neutral act: RRF
    adds a second reciprocal for every document the covered lane ranked, so
    every uncovered document is pushed down by exactly the amount the covered
    ones are pushed up. Measured on the atlas suite, turning the vector lane on
    dropped recall@5 from 1.00 to 0.79 — the raw sources, which no vector could
    rank, fell out of the window. Whether to embed sources is the caller's
    decision and is expressed by what it passes in.
    """
    import hashlib

    config.require_enabled()
    store = VectorStore(project.state_dir / "vectors.db")
    options = ChunkingOptions()
    indexed = skipped = failed = 0
    total_chunks = 0

    try:
        page_ids: set[str] = set()
        pages = list(documents)
        for position, document in enumerate(pages, start=1):
            page_id = document.path
            page_ids.add(page_id)
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            if not force and store.page_hash(page_id) == content_hash:
                skipped += 1
                if on_progress:
                    on_progress(position, len(pages), document.path, "cached")
                continue

            chunks = chunk_markdown(document.content, options)
            if not chunks:
                store.delete_page(page_id)
                skipped += 1
                continue
            try:
                vectors = embed_texts(
                    [_embedding_text(document.title, chunk) for chunk in chunks], config
                )
            except ProviderError:
                failed += 1
                if on_progress:
                    on_progress(position, len(pages), document.path, "failed")
                continue

            store.upsert_page(
                page_id,
                document.title,
                content_hash,
                config.model,
                [
                    (chunk.index, chunk.heading_path, chunk.text, vector)
                    for chunk, vector in zip(chunks, vectors)
                ],
            )
            indexed += 1
            total_chunks += len(chunks)
            if on_progress:
                on_progress(position, len(pages), document.path, "indexed")

        pruned = store.prune(page_ids)
        pages_stored, chunks_stored = store.count()
    finally:
        store.close()

    return {
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "pruned": pruned,
        "chunks_written": total_chunks,
        "pages_stored": pages_stored,
        "chunks_stored": chunks_stored,
    }


def _embedding_text(title: str, chunk) -> str:
    """Prefix each chunk with its title and heading breadcrumb.

    A bare chunk often lacks the noun it is about ("It was released in
    March"); the breadcrumb restores it without a second LLM call.
    """
    header = " > ".join(part for part in (title, chunk.heading_path) if part)
    return f"{header}\n\n{chunk.text}" if header else chunk.text
