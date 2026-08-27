from __future__ import annotations

import pytest

from llmwiki import frontmatter as fm
from llmwiki.paths import is_safe_ingest_path, slugify, source_identity, source_summary_slug


@pytest.mark.parametrize(
    "path",
    [
        "wiki/concepts/foo.md",
        "wiki/a/b/c.md",
        "wiki/概念/注意力.md",
    ],
)
def test_safe_paths_allowed(path):
    assert is_safe_ingest_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "   ",
        "notes/foo.md",              # outside wiki/
        "/etc/passwd",               # absolute
        "C:/Windows/system32",       # drive letter
        "wiki/../../etc/passwd",     # traversal
        "wiki/..\\..\\etc",          # traversal, windows separators
        "wiki/con.md",               # reserved device name
        "wiki/trailing /x.md",       # directory segment ending in a space
        "wiki/dot./x.md",            # directory segment ending in a dot
        "wiki/a\x00b.md",            # NUL
        "wiki/pi|pe.md",             # illegal on Windows
    ],
)
def test_unsafe_paths_rejected(path):
    assert not is_safe_ingest_path(path)


def test_slugify_preserves_cjk_and_kebabs_latin():
    assert slugify("Chain of Thought") == "chain-of-thought"
    assert slugify("Foo: Bar!") == "foo-bar"
    assert slugify("注意力机制") == "注意力机制"


def test_source_identity_keeps_folder_context():
    assert source_identity("/p", "/p/raw/sources/papers/energy/foo.pdf") == "papers/energy/foo.pdf"
    assert source_identity("/p", "/elsewhere/foo.pdf") == "foo.pdf"
    assert source_summary_slug("papers/energy/foo.pdf") == "papers-energy-foo"


def test_frontmatter_roundtrip_and_shapes():
    doc = (
        "---\n"
        "type: entity\n"
        'title: "Foo: Bar"\n'
        "tags: [a, b]\n"
        "related:\n"
        "  - x\n"
        "  - y\n"
        'sources: ["p.pdf"]\n'
        "---\n\n# Foo\n\nbody\n"
    )
    parsed = fm.parse(doc)
    assert parsed.present
    assert parsed.get_str("title") == "Foo: Bar"
    assert parsed.get_list("tags") == ["a", "b"]
    assert parsed.get_list("related") == ["x", "y"]
    assert parsed.body.strip().startswith("# Foo")

    updated = fm.write_sources(doc, ["p.pdf", "q.pdf"])
    assert fm.parse_sources(updated) == ["p.pdf", "q.pdf"]
    assert fm.parse(updated).get_str("title") == "Foo: Bar"


def test_frontmatter_absent_is_not_fabricated():
    assert fm.parse("# Just a heading").present is False
    assert fm.set_field("# Just a heading", "type", "entity") == "# Just a heading"


def test_extract_title_precedence():
    assert fm.extract_title("---\ntitle: From FM\n---\n\n# Heading\n", "file.md") == "From FM"
    assert fm.extract_title("# Heading\n\nbody", "file.md") == "Heading"
    assert fm.extract_title("no title at all", "chain-of-thought.md") == "chain of thought"
