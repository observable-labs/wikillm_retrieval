"""The FILE/REVIEW parser must survive real LLM output, not ideal output."""

from __future__ import annotations

from llmwiki.ingest.blocks import parse_blocks


def test_fenced_end_marker_does_not_truncate_the_page():
    # A wiki that documents its own ingest format will quote the closer.
    text = (
        "---FILE: wiki/concepts/format.md---\n"
        "# Format\n\n"
        "```\n---END FILE---\n```\n\n"
        "Still part of the page.\n"
        "---END FILE---\n"
    )
    result = parse_blocks(text)
    assert len(result.files) == 1
    assert "Still part of the page." in result.files[0].content


def test_crlf_and_marker_variants():
    text = (
        "--- FILE: wiki/a.md ---\r\nalpha\r\n--- END FILE ---\r\n"
        "---file: wiki/b.md---\r\nbeta\r\n---end file---\r\n"
    )
    result = parse_blocks(text)
    assert [block.path for block in result.files] == ["wiki/a.md", "wiki/b.md"]
    assert result.files[0].content == "alpha"


def test_truncated_block_is_reported_not_silently_dropped():
    result = parse_blocks("---FILE: wiki/a.md---\nbody with no closer\n")
    assert result.files == []
    assert result.truncated_paths == ["wiki/a.md"]
    assert "not closed" in result.warnings[0]


def test_unsafe_and_empty_paths_are_rejected_with_a_warning():
    text = (
        "---FILE: ../../etc/passwd---\nx\n---END FILE---\n"
        "---FILE: ---\ny\n---END FILE---\n"
        "---FILE: wiki/ok.md---\nz\n---END FILE---\n"
    )
    result = parse_blocks(text)
    assert [block.path for block in result.files] == ["wiki/ok.md"]
    assert len(result.warnings) == 2


def test_review_blocks_parse_and_constrain_options():
    text = (
        "---FILE: wiki/a.md---\nbody\n---END FILE---\n"
        "---REVIEW: suggestion | Investigate storage costs---\n"
        "Worth a dedicated page.\n"
        "OPTIONS: Create Page | Delete Everything | Skip\n"
        "PAGES: wiki/a.md, wiki/b.md\n"
        "SEARCH: grid storage cost | battery lcoe 2026\n"
        "---END REVIEW---\n"
    )
    result = parse_blocks(text)
    review = result.reviews[0]
    assert review.type == "suggestion"
    assert review.title == "Investigate storage costs"
    # An invented action label is dropped: nothing downstream could execute it.
    assert review.options == ["Create Page", "Skip"]
    assert review.pages == ["wiki/a.md", "wiki/b.md"]
    assert review.search == ["grid storage cost", "battery lcoe 2026"]
    assert review.description == "Worth a dedicated page."


def test_unknown_review_type_dropped():
    text = "---REVIEW: rewrite-everything | X---\nbody\n---END REVIEW---\n"
    result = parse_blocks(text)
    assert result.reviews == []
    assert "unknown type" in result.warnings[0]


def test_preamble_prose_is_ignored():
    text = (
        "Here are the files you asked for:\n\n"
        "---FILE: wiki/a.md---\nbody\n---END FILE---\n\n"
        "Let me know if you need changes!\n"
    )
    result = parse_blocks(text)
    assert [block.path for block in result.files] == ["wiki/a.md"]
    assert result.files[0].content == "body"


def test_opener_without_trailing_marker_is_accepted():
    """H7: models reached through the OpenAI-compatible lane (Gemini) routinely
    omit the trailing `---` the prompt asks for. Before this was tolerated, a
    complete, well-formed generation parsed to zero files AND zero warnings."""
    text = (
        "---FILE: wiki/a.md\nalpha\n---END FILE---\n"
        "---FILE: wiki/b.md---\nbeta\n---END FILE---\n"
        "---FILE: wiki/c.md   \ngamma\n---END FILE---\n"
    )
    result = parse_blocks(text)
    assert [block.path for block in result.files] == ["wiki/a.md", "wiki/b.md", "wiki/c.md"]
    assert [block.content for block in result.files] == ["alpha", "beta", "gamma"]
    assert result.warnings == []
