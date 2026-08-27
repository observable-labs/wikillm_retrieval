"""A deliberately small YAML-frontmatter reader/writer.

llm_wiki hand-rolls this rather than depending on a YAML parser, and this
port keeps that choice: the frontmatter contract is fixed (strings and flat
arrays only) and LLM output is frequently *almost* valid YAML. A strict
parser turns "almost" into a dropped page; a forgiving line reader keeps it.

Supported shapes::

    ---
    type: entity
    title: "Foo: Bar"
    tags: [a, b]
    sources: ["paper.pdf"]
    related:
      - foo
      - bar
    ---
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FrontmatterValue = str | list[str]

_FM_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s*-\s+(.*)$")


@dataclass
class Frontmatter:
    data: dict[str, FrontmatterValue] = field(default_factory=dict)
    body: str = ""
    raw: str = ""
    present: bool = False

    def get_str(self, key: str, default: str = "") -> str:
        value = self.data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return value[0]
        return default

    def get_list(self, key: str) -> list[str]:
        value = self.data.get(key)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_inline_list(value: str) -> list[str]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    depth = 0
    for ch in inner:
        if quote:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
            continue
        if ch in "\"'":
            quote = ch
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth = max(0, depth - 1)
            continue
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    items.append("".join(current).strip())
    return [item for item in items if item]


def parse(content: str) -> Frontmatter:
    """Split `content` into frontmatter mapping + body. Never raises."""
    match = _FM_RE.match(content)
    if not match:
        return Frontmatter(data={}, body=content, raw="", present=False)

    raw = match.group(1)
    body = content[match.end():]
    data: dict[str, FrontmatterValue] = {}
    pending_key: str | None = None

    for line in raw.split("\n"):
        if not line.strip():
            continue
        item = _LIST_ITEM_RE.match(line)
        if item and pending_key:
            bucket = data.get(pending_key)
            if not isinstance(bucket, list):
                bucket = []
                data[pending_key] = bucket
            bucket.append(_strip_quotes(item.group(1)))
            continue

        key_match = _KEY_RE.match(line)
        if not key_match:
            continue
        key, value = key_match.group(1), key_match.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            data[key] = _parse_inline_list(value)
            pending_key = None
        elif value == "":
            # Either an empty scalar or the head of a block list; the next
            # line decides. Seed with "" so a truly empty value is preserved.
            data[key] = ""
            pending_key = key
        else:
            data[key] = _strip_quotes(value)
            pending_key = None

    # A key seeded as "" that never received list items stays "".
    return Frontmatter(data=data, body=body, raw=raw, present=True)


def _format_value(value: FrontmatterValue) -> str:
    if isinstance(value, list):
        rendered = ", ".join(_quote_if_needed(item) for item in value)
        return f"[{rendered}]"
    return _quote_if_needed(value, scalar=True)


def _quote_if_needed(value: str, scalar: bool = False) -> str:
    text = str(value)
    needs_quote = any(ch in text for ch in ':#"') or (scalar and text.strip() != text)
    if needs_quote:
        escaped = text.replace('"', '\\"')
        return f'"{escaped}"'
    return text


def render(data: dict[str, FrontmatterValue], body: str) -> str:
    """Serialize a page. Key order is the dict's insertion order."""
    lines = ["---"]
    for key, value in data.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append("---")
    # A blank line after the closing fence: the convention every wiki page in
    # the prompt's own example follows, and what Obsidian renders cleanly.
    return "\n".join(lines) + "\n\n" + body.lstrip("\n")


def set_field(content: str, key: str, value: FrontmatterValue) -> str:
    """Set one frontmatter key, preserving everything else.

    Returns `content` unchanged when it has no frontmatter block — callers
    treat a missing block as "not a wiki page" rather than fabricating one.
    """
    parsed = parse(content)
    if not parsed.present:
        return content
    data = dict(parsed.data)
    data[key] = value
    return render(data, parsed.body)


def parse_sources(content: str) -> list[str]:
    return parse(content).get_list("sources")


def write_sources(content: str, sources: list[str]) -> str:
    return set_field(content, "sources", sources)


def extract_title(content: str, fallback_file_name: str) -> str:
    """frontmatter `title:` > first `# heading` > de-kebabed filename.

    Mirrors `search.rs::extract_title` so keyword hits and index entries
    agree on what a page is called.
    """
    parsed = parse(content)
    title = parsed.get_str("title").strip()
    if title:
        return title
    body = parsed.body if parsed.present else content
    heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if heading:
        return heading.group(1).strip()
    stem = fallback_file_name
    if stem.lower().endswith(".md"):
        stem = stem[:-3]
    return stem.replace("-", " ")
