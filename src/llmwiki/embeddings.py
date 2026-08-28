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


# The scanned matrix, per store file, for the life of the process.
#
# Every query used to read every vector back out of SQLite: 9,931 chunks is
# 30 MB of blobs, and paying that per query made the vector lane's local cost
# grow with the corpus rather than with the work. The rows do not change between
# queries — only an `embed` changes them — so the parse belongs once per corpus
# state, not once per question.
#
# Keyed on the file's identity *and* a counter this process bumps on every
# write, because a same-process write can land inside one mtime tick and a cache
# that serves a corpus that no longer exists is worse than no cache.
_SCAN_CACHE: dict[str, tuple] = {}
_WRITES = 0


def clear_vector_cache() -> None:
    """Drop the scanned matrices. For tests and for `reset()`."""
    _SCAN_CACHE.clear()


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
        global _WRITES
        self._connection.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        self._connection.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))
        self._connection.commit()
        _WRITES += 1

    def upsert_page(
        self,
        page_id: str,
        title: str,
        content_hash: str,
        model: str,
        chunks: list[tuple[int, str, str, list[float]]],
    ) -> None:
        """Replace a page's chunks atomically: delete-then-insert."""
        global _WRITES
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
        _WRITES += 1

    def prune(self, keep_page_ids: set[str]) -> int:
        """Drop embeddings for pages that no longer exist on disk."""
        rows = self._connection.execute("SELECT page_id FROM pages").fetchall()
        stale = [row[0] for row in rows if row[0] not in keep_page_ids]
        for page_id in stale:
            self.delete_page(page_id)
        return len(stale)

    def _scan(self) -> tuple:
        """(row ids, page ids, chunk indices, vectors, dims), parsed once.

        Invalidated by the store file changing on disk or by a write from this
        process. Both are needed: mtime has a granularity, and an `embed` that
        finishes inside one tick would otherwise leave the next query ranking
        against a corpus that is gone.
        """
        try:
            stat = self.path.stat()
            identity = (stat.st_mtime_ns, stat.st_size, _WRITES)
        except OSError:
            identity = (0, 0, _WRITES)
        key = str(self.path)
        cached = _SCAN_CACHE.get(key)
        if cached is not None and cached[0] == identity:
            return cached[1]

        rows = self._connection.execute(
            "SELECT id, page_id, chunk_index, vector, dims FROM chunks"
        ).fetchall()
        ids = [row[0] for row in rows]
        page_ids = [row[1] for row in rows]
        indices = [row[2] for row in rows]
        dims = [row[4] for row in rows]
        if _np is not None and rows:
            width = max(set(dims), key=dims.count)
            keep = [i for i, d in enumerate(dims) if d == width]
            matrix = _np.frombuffer(
                b"".join(rows[i][3] for i in keep), dtype=_np.float32
            ).reshape(len(keep), width) if keep else _np.zeros((0, 0), dtype=_np.float32)
            scan = (
                [ids[i] for i in keep],
                [page_ids[i] for i in keep],
                [indices[i] for i in keep],
                matrix,
                width if keep else 0,
            )
        else:
            vectors = []
            for row in rows:
                buffer = array.array("f")
                buffer.frombytes(row[3])
                vectors.append(buffer)
            scan = (ids, page_ids, indices, vectors, dims)

        _SCAN_CACHE[key] = (identity, scan)
        return scan

    def search(self, query_vector: list[float], top_k: int = 30) -> list[ChunkHit]:
        """Brute-force cosine over every chunk, then read the text of the winners.

        Two things this deliberately does not do. It does not read `text` during
        the scan — the dot product never looks at it, and selecting it meant a
        query read the corpus prose back out of SQLite to rank vectors. And it
        does not re-parse the vectors per query; `_scan` holds them for as long
        as the store is unchanged.

        Measured on a 9,931-chunk store, per query after the first: **5.6 ms
        with `numpy` installed against 145 ms for the first call**, and flat
        against the 2,032-chunk store's 6.1 ms — the cost stops growing with the
        corpus and starts growing with the work. Without `numpy` the two
        savings still apply but the pure-Python dot products dominate at 269 ms
        a query, which is why `numpy` is a declared extra rather than a
        coincidence: `pip install llmwiki[vector]`.
        """
        ids, page_ids, indices, vectors, width = self._scan()
        if not ids:
            return []

        query = _normalize(query_vector)
        dims = len(query)
        scored: list[tuple[float, int]] = []

        if _np is not None:
            if width != dims:
                return []  # a re-embedding with a different model is in flight
            similarities = vectors @ _np.asarray(query, dtype=_np.float32)
            order = _np.argsort(-similarities)[: max(0, top_k)]
            scored = [(float(similarities[i]), int(i)) for i in order]
        else:
            for position, vector in enumerate(vectors):
                if width[position] != dims:
                    continue
                scored.append(
                    (sum(a * b for a, b in zip(vector, query)), position)
                )
            scored.sort(key=lambda item: -item[0])
            scored = scored[: max(0, top_k)]

        if not scored:
            return []
        placeholders = ",".join("?" * len(scored))
        text_by_id = {
            row[0]: (row[1], row[2])
            for row in self._connection.execute(
                f"SELECT id, heading_path, text FROM chunks WHERE id IN ({placeholders})",
                [ids[position] for _, position in scored],
            )
        }
        return [
            ChunkHit(
                page_ids[position],
                indices[position],
                *text_by_id.get(ids[position], ("", "")),
                score,
            )
            for score, position in scored
        ]


def group_by_page(hits: list[ChunkHit], top_k: int = 10) -> list[PageHit]:
    """A page scores its best chunk, and nothing else scores it.

    The rule this replaces was `top + min(0.3 * sum(other scores), 1 - top)`,
    meant to let broad coverage count for something. Cosine similarities on a
    real corpus sit in a narrow band around 0.6-0.8, so that tail term saturates
    almost immediately: a page with three retrieved chunks near 0.6 scores 1.00
    while a page with one chunk at 0.85 scores 0.85. **Chunk count outranked
    chunk quality**, and how many chunks are retrieved is a depth constant
    chosen for latency.

    Measured on both eval corpora, vector-lane `recall@k` with the tail term,
    at the depth the retrieval pipeline actually scans, against this rule:

        hotpot   R@1  0.378 -> 0.475    R@2  0.720 -> 0.863
        atlas    R@1  0.023 -> 0.705    R@2  0.068 -> 0.807

    Atlas is the extreme because its fourteen multi-chunk pages are the raw
    source documents: they saturated at 1.00 and occupied the head of the
    ranking for every query, whatever the query was.

    Scoring by the best chunk also makes the lane's ranking **independent of
    the scan depth**. That is worth having for its own sake — the depth
    constant can then be tuned for latency without moving any result — and it
    is what makes the vector lane's contribution to fusion mean the same thing
    at every `k`. Weaker variants of the coverage bonus (a mean rather than a
    sum, a capped per-chunk count) were measured too and cost 0.08 recall@1 on
    atlas; a bonus small enough to be safe is a bonus too small to do anything.

    `matched_chunks` still carries the page's three best chunks, because a
    snippet wants the evidence even though the ranking does not.
    """
    buckets: dict[str, list[ChunkHit]] = {}
    for hit in hits:
        buckets.setdefault(hit.page_id, []).append(hit)

    pages: list[PageHit] = []
    for page_id, chunks in buckets.items():
        chunks.sort(key=lambda hit: -hit.score)
        pages.append(
            PageHit(page_id=page_id, score=chunks[0].score, matched_chunks=chunks[:3])
        )

    # Ties broken by page id so the ranking is a function of the scores alone
    # and not of the order rows came back from SQLite.
    pages.sort(key=lambda page: (-page.score, page.page_id))
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
    batch_size = max(1, config.batch_size)

    try:
        page_ids: set[str] = set()
        pages = list(documents)
        # Chunks are batched across documents, not within one. A wiki of short
        # pages is one chunk per page, so batching inside a document meant one
        # HTTP round trip per page and `batch_size` never fired at all: 9,769
        # paragraphs took one request each. Filling the batch from as many
        # documents as it holds turns that into 306 requests for the same work.
        batch: list[tuple[int, object, str, list, list[str]]] = []
        pending_texts = 0

        def flush() -> None:
            nonlocal batch, pending_texts, indexed, failed, total_chunks
            if not batch:
                return
            texts = [text for _, _, _, _, page_texts in batch for text in page_texts]
            try:
                vectors = embed_texts(texts, config)
            except ProviderError:
                # One document in the batch may be the problem, and the rest are
                # not. Fall back to a request per document so a single bad page
                # costs one page rather than thirty-two.
                vectors = None
            offset = 0
            for position, document, content_hash, chunks, page_texts in batch:
                if vectors is None:
                    try:
                        page_vectors = embed_texts(page_texts, config)
                    except ProviderError:
                        failed += 1
                        if on_progress:
                            on_progress(position, len(pages), document.path, "failed")
                        continue
                else:
                    page_vectors = vectors[offset : offset + len(page_texts)]
                offset += len(page_texts)
                store.upsert_page(
                    document.path,
                    document.title,
                    content_hash,
                    config.model,
                    [
                        (chunk.index, chunk.heading_path, chunk.text, vector)
                        for chunk, vector in zip(chunks, page_vectors)
                    ],
                )
                indexed += 1
                total_chunks += len(chunks)
                if on_progress:
                    on_progress(position, len(pages), document.path, "indexed")
            batch = []
            pending_texts = 0

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

            batch.append(
                (
                    position,
                    document,
                    content_hash,
                    chunks,
                    [_embedding_text(document.title, chunk) for chunk in chunks],
                )
            )
            pending_texts += len(chunks)
            if pending_texts >= batch_size:
                flush()
        flush()

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
