"""A1: one append-only row per turn, on both the answer path and the search one.

Written *after* the answer streams, never before, so a logging failure cannot
cost a turn — every entry point here swallows its own errors and returns a note
instead of raising. A query log that can break `ask` is worse than no query log.

Three fields carry more than they look:

* **`cited`** is the relevance judgement the write-back loop learns from, which
  is why the citation parse has to run before the write. `NULL` means no answer
  was generated (the `search` path); `[]` means an answer was generated and
  cited nothing, and those are different observations.
* **`lexical_top` with `gate_fired`** is the gap queue. A lane that scored just
  under the fence and one that found nothing at all record the same verdict and
  different scores, and the difference is which of them is a missing document.
* **`stage_ms`** is the telemetry seam (`pipeline._stage`). Budgets are set
  against it and every later percentile is measured from it, so it is stored
  per stage rather than as a wall clock that mixes a provider's network with
  local ranking.

`query_vector` is kept because clustering the log later would otherwise re-pay,
per query, an embedding call this turn already made.

**Privacy.** The log holds the user's questions verbatim and lives inside their
own git repo. `ensure_ignored` adds it to `.llm-wiki/.gitignore` the first time
the database is created, including for projects made before this module existed:
a wiki that is shared is not a query log that is shared. Committing it is a
deliberate act — delete the line.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from array import array
from dataclasses import dataclass, field
from pathlib import Path

LOG_FILE = "query-log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL,
  query         TEXT NOT NULL,
  raw_query     TEXT,
  profile       TEXT NOT NULL,
  lanes         TEXT NOT NULL,
  gate_fired    INTEGER NOT NULL,
  lexical_top   REAL,
  vector_top    REAL,
  retrieved     TEXT NOT NULL,
  cited         TEXT,
  stage_ms      TEXT NOT NULL,
  query_vector  BLOB
);
CREATE INDEX IF NOT EXISTS query_log_ts ON query_log(ts);
"""


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class QueryRecord:
    """One turn, as the log stores it."""

    query: str
    profile: str
    retrieved: list[str] = field(default_factory=list)
    lanes: dict = field(default_factory=dict)
    gate_fired: bool = False
    lexical_top: float | None = None
    vector_top: float | None = None
    # The utterance before a rewrite, when one happened. Nothing rewrites yet
    # (D1); the column exists so that when something does, the pair is on one
    # row rather than inferred by joining timestamps.
    raw_query: str | None = None
    cited: list[str] | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)
    query_vector: list[float] | None = None
    ts: str = ""


@dataclass(frozen=True)
class LogStats:
    rows: int
    first: str | None
    last: str | None
    bytes: int


class QueryLog:
    """The log as a database. Callers who want the safe form want `record`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        # A log is appended to on the query path while `status` and, later, the
        # write-back loop read it. WAL is what lets those overlap without a
        # reader blocking the turn that is writing.
        try:
            self._connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def append(self, record: QueryRecord) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO query_log
              (ts, query, raw_query, profile, lanes, gate_fired, lexical_top,
               vector_top, retrieved, cited, stage_ms, query_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.ts or now(),
                record.query,
                record.raw_query,
                record.profile,
                json.dumps(record.lanes, sort_keys=True),
                int(record.gate_fired),
                record.lexical_top,
                record.vector_top,
                json.dumps(record.retrieved),
                None if record.cited is None else json.dumps(record.cited),
                json.dumps(record.stage_ms, sort_keys=True),
                pack_vector(record.query_vector),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid or 0)

    def stats(self) -> LogStats:
        rows, first, last = self._connection.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM query_log"
        ).fetchone()
        size = sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                self.path.with_name(self.path.name + "-wal"),
            )
            if candidate.exists()
        )
        return LogStats(rows=int(rows or 0), first=first, last=last, bytes=size)

    def recent(self, limit: int = 20) -> list[dict]:
        """The last `limit` turns, newest first, decoded."""
        self._connection.row_factory = sqlite3.Row
        try:
            rows = self._connection.execute(
                "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
        finally:
            self._connection.row_factory = None
        return [_decode(row) for row in rows]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "QueryLog":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def path_for(project) -> Path:
    return Path(project.state_dir) / LOG_FILE


def open_log(project) -> QueryLog:
    """Open (creating if needed) the project's log, ignoring it in git first."""
    path = path_for(project)
    existed = path.exists()
    log = QueryLog(path)
    if not existed:
        ensure_ignored(project)
    return log


def record(
    project,
    query: str,
    response,
    *,
    profile: str,
    cited: list[str] | None = None,
    raw_query: str | None = None,
) -> str | None:
    """Log one turn from a `SearchResponse`. Returns a note on failure, else None.

    Never raises. The call site is the end of a turn the user has already been
    served, so every failure mode here — a read-only project, a corrupt file, a
    locked database — is worth a line in `notes` and nothing more.
    """
    try:
        with open_log(project) as log:
            log.append(
                QueryRecord(
                    query=query,
                    raw_query=raw_query if raw_query and raw_query != query else None,
                    profile=profile,
                    retrieved=[result.path for result in response.results],
                    lanes=_lanes(response),
                    gate_fired=bool(getattr(response, "gate_fired", False)),
                    lexical_top=getattr(response, "lexical_top", None),
                    vector_top=getattr(response, "vector_top", None),
                    cited=cited,
                    stage_ms=dict(getattr(response, "stage_ms", {}) or {}),
                    query_vector=getattr(response, "query_vector", None),
                )
            )
    except (sqlite3.Error, OSError, ValueError) as exc:
        return f"the query log could not be written ({type(exc).__name__}: {exc})"
    return None


def stats(project) -> LogStats | None:
    """The log's size and date range, or None when there is no log to read."""
    path = path_for(project)
    if not path.exists():
        return None
    try:
        with QueryLog(path) as log:
            return log.stats()
    except (sqlite3.Error, OSError):
        return None


def ensure_ignored(project) -> None:
    """Add the log to `.llm-wiki/.gitignore`, for projects of any vintage."""
    ignore = Path(project.state_dir) / ".gitignore"
    try:
        existing = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if LOG_FILE in existing.split():
            return
        ignore.parent.mkdir(parents=True, exist_ok=True)
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        ignore.write_text(f"{existing}{prefix}{LOG_FILE}\n{LOG_FILE}-*\n", encoding="utf-8")
    except OSError:
        # Best effort by design: failing to write the ignore file is not a
        # reason to fail the turn, and it is not a reason to skip the log
        # either — the user's own `.gitignore` may already cover it.
        pass


def pack_vector(vector: list[float] | None) -> bytes | None:
    return None if not vector else array("f", vector).tobytes()


def unpack_vector(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _lanes(response) -> dict:
    lanes = getattr(response, "lanes", None)
    if lanes is None:
        return {}
    ran = lanes.as_dict()
    ran["abstained"] = list(lanes.abstained)
    # A lane that ran out of time is why a turn's ranking looks unlike the same
    # turn's ranking yesterday, and the log is where that is answered later.
    ran["expired"] = list(lanes.expired)
    ran["mode"] = getattr(response, "mode", "")
    return ran


def _decode(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "ts": row["ts"],
        "query": row["query"],
        "raw_query": row["raw_query"],
        "profile": row["profile"],
        "lanes": json.loads(row["lanes"]),
        "gate_fired": bool(row["gate_fired"]),
        "lexical_top": row["lexical_top"],
        "vector_top": row["vector_top"],
        "retrieved": json.loads(row["retrieved"]),
        "cited": None if row["cited"] is None else json.loads(row["cited"]),
        "stage_ms": json.loads(row["stage_ms"]),
        "query_vector": unpack_vector(row["query_vector"]),
    }


__all__ = [
    "LOG_FILE",
    "LogStats",
    "QueryLog",
    "QueryRecord",
    "ensure_ignored",
    "open_log",
    "pack_vector",
    "path_for",
    "record",
    "stats",
    "unpack_vector",
]
