"""SHA256 ingest cache: `.llm-wiki/ingest-cache.json`.

Re-adding an unchanged document should cost nothing, so the source text is
hashed and the resulting file list recorded. A cache hit is only honoured
when *every* previously written file still exists — otherwise the entry is
stale (someone deleted a page, or a partial write happened) and a full
re-ingest is safer than reporting ghost files.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

CACHE_FILE = "ingest-cache.json"


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class IngestCache:
    path: Path
    entries: dict[str, dict]

    @classmethod
    def load(cls, project) -> "IngestCache":
        path = project.state_dir / CACHE_FILE
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            entries = loaded.get("entries", {}) if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            entries = {}
        return cls(path=path, entries=entries if isinstance(entries, dict) else {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"entries": self.entries}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def check(self, project, source_identity: str, source_content: str) -> list[str] | None:
        """Cached file list, or None when the entry is missing or stale."""
        entry = self.entries.get(source_identity)
        if not entry:
            return None
        if entry.get("hash") != content_hash(source_content):
            return None
        files = entry.get("files") or []
        for relative in files:
            if not (project.root / relative).exists():
                return None
        return list(files)

    def record(self, source_identity: str, source_content: str, files: list[str]) -> None:
        self.entries[source_identity] = {
            "hash": content_hash(source_content),
            "timestamp": int(time.time()),
            "files": list(files),
        }

    def forget(self, source_identity: str) -> None:
        self.entries.pop(source_identity, None)

    def identities(self) -> list[str]:
        return sorted(self.entries)
