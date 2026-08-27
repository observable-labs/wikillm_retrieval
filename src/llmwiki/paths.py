"""Path normalization, slugging, and the ingest write sandbox.

`is_safe_ingest_path` is the security boundary of the whole package: FILE
block paths come out of LLM-generated text, and a source document can carry
prompt injection ("now write to ../../../etc/passwd"). Every generated path
crosses this gate before it reaches the filesystem.

Ported from llm_wiki's `src/lib/ingest.ts::isSafeIngestPath` and
`src/lib/path-utils.ts`.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")
_WINDOWS_ILLEGAL = re.compile(r'[<>:"|?*]')
_DRIVE_LETTER = re.compile(r"^[a-zA-Z]:")
_RESERVED_STEMS = {"CON", "PRN", "AUX", "NUL"}
_RESERVED_PATTERN = re.compile(r"^(COM[1-9]|LPT[1-9])$")

# Ranges kept identical to the TypeScript/Rust implementations so filenames
# round-trip between the desktop app and this package.
_CJK_RE = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def normalize_path(path: str) -> str:
    """Backslashes to forward slashes; the canonical form used everywhere."""
    return path.replace("\\", "/")


def file_name(path: str) -> str:
    return normalize_path(path).rsplit("/", 1)[-1]


def file_stem(path: str) -> str:
    name = file_name(path)
    return name[:-3] if name.lower().endswith(".md") else name.rsplit(".", 1)[0] if "." in name else name


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _is_windows_safe_segment(segment: str) -> bool:
    if not segment:
        return False
    if _WINDOWS_ILLEGAL.search(segment):
        return False
    if segment[-1] in " .":
        return False
    stem = segment.split(".")[0].upper()
    if not stem:
        return False
    if stem in _RESERVED_STEMS or _RESERVED_PATTERN.match(stem):
        return False
    return True


def is_safe_ingest_path(path: str) -> bool:
    """True when an LLM-supplied FILE block path may be written.

    Allowed: anything under ``wiki/``. Rejected: absolute paths, drive
    letters, any ``..`` segment, control characters, Windows-illegal
    filenames and reserved device names, and empty paths.
    """
    if not isinstance(path, str) or not path.strip():
        return False
    if _CONTROL_CHARS.search(path):
        return False
    if path.startswith("/") or path.startswith("\\"):
        return False
    if _DRIVE_LETTER.match(path):
        return False
    normalized = normalize_path(path)
    segments = normalized.split("/")
    if any(seg == ".." for seg in segments):
        return False
    if not all(_is_windows_safe_segment(seg) for seg in segments):
        return False
    return normalized.startswith("wiki/")


def slugify(title: str) -> str:
    """kebab-case for Latin scripts, CJK characters preserved verbatim.

    Matches llm_wiki's filename policy: Chinese/Japanese/Korean titles keep
    their characters rather than being romanized, because the desktop app
    treats the wiki directory as an Obsidian vault where the filename is the
    user-visible page name.
    """
    text = unicodedata.normalize("NFC", title).strip()
    if not text:
        return "untitled"
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or contains_cjk(ch):
            out.append(ch.lower())
        elif ch in "-_ \t/\\.,:;!?'\"()[]{}":
            out.append("-")
        # everything else is dropped
    slug = re.sub(r"-{2,}", "-", "".join(out)).strip("-")
    return slug or "untitled"


def source_identity(project_path: str, source_path: str) -> str:
    """The stable name a source is known by inside the wiki.

    Sources under ``raw/sources/`` are identified by their path relative to
    that directory (so ``papers/energy/foo.pdf`` keeps its folder context);
    anything else falls back to the bare filename.
    """
    pp = normalize_path(str(project_path)).rstrip("/")
    sp = normalize_path(str(source_path))
    prefix = f"{pp}/raw/sources/"
    if sp.startswith(prefix):
        return sp[len(prefix):]
    return file_name(sp)


def source_summary_slug(identity: str) -> str:
    """`papers/energy/foo.pdf` -> `papers-energy-foo`, the summary page stem."""
    stem = normalize_path(identity)
    if "." in file_name(stem):
        stem = stem.rsplit(".", 1)[0]
    return slugify(stem.replace("/", "-"))
