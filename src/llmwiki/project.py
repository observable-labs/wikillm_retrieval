"""Project layout: create, open, and read/write the three layers.

The three-layer architecture is the whole point of the pattern:

    raw/sources/   immutable source documents — the LLM reads, never writes
    wiki/          LLM-generated markdown — the LLM owns this entirely
    schema.md      structural rules; purpose.md holds directional intent

Everything the tool generates is plain markdown on disk, so the project
directory doubles as an Obsidian vault and as a git repo.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ProjectError
from .paths import normalize_path
from .templates import DEFAULT_TEMPLATE, get_template

STATE_DIR = ".llm-wiki"
AGGREGATE_PAGES = ("wiki/index.md", "wiki/overview.md")


def today() -> str:
    return _dt.date.today().isoformat()


@dataclass
class Project:
    """An open wiki project rooted at `root`."""

    root: Path

    # ── locations ────────────────────────────────────────────────────────
    @property
    def wiki_dir(self) -> Path:
        return self.root / "wiki"

    @property
    def sources_dir(self) -> Path:
        return self.root / "raw" / "sources"

    @property
    def assets_dir(self) -> Path:
        return self.root / "raw" / "assets"

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIR

    @property
    def schema_path(self) -> Path:
        return self.root / "schema.md"

    @property
    def purpose_path(self) -> Path:
        return self.root / "purpose.md"

    @property
    def index_path(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def log_path(self) -> Path:
        return self.wiki_dir / "log.md"

    @property
    def overview_path(self) -> Path:
        return self.wiki_dir / "overview.md"

    @property
    def name(self) -> str:
        return self.root.name

    # ── io ───────────────────────────────────────────────────────────────
    def read(self, relative: str, default: str = "") -> str:
        path = self.root / relative
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return default

    def write(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def schema(self) -> str:
        # schema.md moved to the project root in llm_wiki; older projects and
        # hand-made vaults keep it under wiki/. Accept both.
        return self.read("schema.md") or self.read("wiki/schema.md")

    def purpose(self) -> str:
        return self.read("purpose.md") or self.read("wiki/purpose.md")

    def index(self) -> str:
        return self.read("wiki/index.md")

    def overview(self) -> str:
        return self.read("wiki/overview.md")

    def wiki_pages(self, limit: int = 10_000) -> list[Path]:
        """Every markdown page under wiki/, sorted for deterministic output."""
        if limit <= 0 or not self.wiki_dir.is_dir():
            return []
        pages: list[Path] = []
        for path in sorted(self.wiki_dir.rglob("*.md")):
            if any(part.startswith(".") for part in path.relative_to(self.root).parts):
                continue
            pages.append(path)
            if len(pages) >= limit:
                break
        return pages

    def source_files(self, limit: int = 10_000) -> list[Path]:
        if limit <= 0 or not self.sources_dir.is_dir():
            return []
        out: list[Path] = []
        for path in sorted(self.sources_dir.rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(self.root).parts):
                continue
            out.append(path)
            if len(out) >= limit:
                break
        return out

    def relative(self, path: Path | str) -> str:
        try:
            return normalize_path(str(Path(path).resolve().relative_to(self.root.resolve())))
        except ValueError:
            return normalize_path(str(path))

    # ── project-local settings ───────────────────────────────────────────
    def settings(self) -> dict:
        raw = self.read(f"{STATE_DIR}/config.json")
        if not raw.strip():
            return {}
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def save_settings(self, settings: dict) -> None:
        self.write(f"{STATE_DIR}/config.json", json.dumps(settings, indent=2) + "\n")


def create(path: Path | str, template_id: str = DEFAULT_TEMPLATE, name: str | None = None) -> Project:
    """Scaffold a new project. Fails if the directory already has one."""
    root = Path(path).expanduser().resolve()
    template = get_template(template_id)
    if (root / "schema.md").exists() or (root / "wiki").exists():
        raise ProjectError(f"'{root}' already contains a wiki project")

    directories = ["raw/sources", "raw/assets", STATE_DIR, *template.wiki_directories()]
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)

    project = Project(root=root)
    project.write("schema.md", template.schema_markdown())
    project.write("purpose.md", template.purpose)

    index_sections = "\n".join(
        f"## {pt.name.title()}\n" for pt in template.page_types
    )
    project.write("wiki/index.md", f"# Wiki Index\n\n{index_sections}\n## Recently Updated\n")
    project.write("wiki/log.md", f"# Wiki Log\n\n## [{today()}] created | {name or root.name}\n")
    project.write(
        "wiki/overview.md",
        "---\n"
        "type: overview\n"
        "title: Project Overview\n"
        f"created: {today()}\n"
        f"updated: {today()}\n"
        "tags: []\n"
        "related: []\n"
        "---\n\n"
        "# Overview\n\n"
        "<!-- A high-level summary of what this wiki covers. Regenerated as it grows. -->\n",
    )
    project.write(f"{STATE_DIR}/.gitignore", "vectors.db\n*.log\n")
    project.save_settings({"template": template_id, "name": name or root.name})

    # Obsidian reads the same directory as a vault; point attachments at
    # raw/assets and hide the tool's own state folder.
    project.write(
        ".obsidian/app.json",
        json.dumps(
            {
                "attachmentFolderPath": "raw/assets",
                "userIgnoreFilters": [STATE_DIR],
                "useMarkdownLinks": False,
                "newLinkFormat": "shortest",
            },
            indent=2,
        )
        + "\n",
    )
    return project


def open_project(path: Path | str | None = None) -> Project:
    """Open the project at `path`, or find one from the current directory.

    With no argument, walks up from `$PWD` (and honours `$LLMWIKI_PROJECT`)
    so `llmwiki ask` works from anywhere inside a project.
    """
    if path is None:
        env = os.environ.get("LLMWIKI_PROJECT")
        if env:
            path = env
        else:
            found = _find_upwards(Path.cwd())
            if found is None:
                raise ProjectError(
                    "no wiki project found here. Run 'llmwiki init <path>' to create one, "
                    "or pass --project."
                )
            return Project(root=found)

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ProjectError(f"'{root}' is not a directory")
    if not (root / "schema.md").exists() and not (root / "wiki" / "schema.md").exists():
        raise ProjectError(f"'{root}' is not a wiki project (no schema.md)")
    return Project(root=root)


def _find_upwards(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "schema.md").exists() and (candidate / "wiki").is_dir():
            return candidate
    return None
