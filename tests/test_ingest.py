"""End-to-end ingest against a scripted model."""

from __future__ import annotations

import pytest

from llmwiki import frontmatter as fm
from llmwiki.errors import IngestError
from llmwiki.ingest import ingest_document
from llmwiki.ingest.cache import IngestCache

ANALYSIS = """## Key Entities
- Vanadium flow battery (central)

## Key Concepts
- Round-trip efficiency

## Main Arguments & Findings
- Flow batteries favour long-duration storage.

## Recommendations
- Create pages for the battery and the efficiency concept.
"""

GENERATION = """---FILE: wiki/sources/grid-storage.md---
---
type: source
title: "Source: grid-storage.md"
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [storage]
related: [vanadium-flow-battery]
sources: ["grid-storage.md"]
---

# Source: grid-storage.md

Covers [[vanadium-flow-battery]] and [[round-trip-efficiency]].
---END FILE---

---FILE: wiki/entities/vanadium-flow-battery.md---
---
type: entity
title: Vanadium Flow Battery
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [storage]
related: [round-trip-efficiency]
sources: ["grid-storage.md", "wiki/index.md"]
---

# Vanadium Flow Battery

Suited to long-duration storage. Related: [[round-trip-efficiency]].
---END FILE---

---FILE: wiki/concepts/round-trip-efficiency.md---
---
type: concept
title: Round-trip Efficiency
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [storage]
related: []
sources: ["grid-storage.md"]
---

# Round-trip Efficiency

Energy out divided by energy in.
---END FILE---

---FILE: wiki/log.md---
## [YYYY-MM-DD] ingest | grid-storage.md
---END FILE---

---REVIEW: missing-page | Compare flow batteries to lithium-ion---
No page compares the two chemistries.
OPTIONS: Create Page | Skip
SEARCH: vanadium flow vs lithium ion grid storage | flow battery lcoe
---END REVIEW---
"""


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "grid-storage.md"
    path.write_text("# Grid storage\n\nVanadium flow batteries suit long-duration storage.\n")
    return path


def test_two_step_ingest_writes_pages_index_and_log(wiki, settings, stub_llm, document):
    client = stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    result = ingest_document(wiki, document, settings)

    assert client.phases() == ["analysis", "generation"], "ingest is two calls: analysis then generation"
    assert "wiki/entities/vanadium-flow-battery.md" in result.files_written
    assert "wiki/concepts/round-trip-efficiency.md" in result.files_written
    assert "wiki/index.md" in result.files_written

    # The source document is copied into the immutable raw layer.
    assert (wiki.sources_dir / "grid-storage.md").exists()

    # index.md gained entries; log.md gained a dated line.
    index = wiki.read("wiki/index.md")
    assert "[[entities/vanadium-flow-battery]]" in index
    assert "ingest | grid-storage.md" in wiki.read("wiki/log.md")

    # Review items are filed for a human rather than acted on.
    assert len(result.reviews) == 1
    assert "Compare flow batteries" in wiki.read("wiki/reviews.md")


def test_generation_prompt_carries_schema_purpose_and_index(wiki, settings, stub_llm, document):
    wiki.write("purpose.md", "# Purpose\n\nUnderstand grid-scale storage economics.\n")
    client = stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    ingest_document(wiki, document, settings)

    generation_system = client.prompt("generation")
    assert "Understand grid-scale storage economics" in generation_system
    assert "## Project Schema and Routing (AUTHORITATIVE)" in generation_system
    assert "wiki/sources/grid-storage.md" in generation_system
    # The step 1 analysis is handed to step 2 as context.
    assert "Round-trip efficiency" in client.prompt("generation", role="user")


def test_sources_are_canonicalized_and_dates_stamped(wiki, settings, stub_llm, document):
    stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    ingest_document(wiki, document, settings)

    page = wiki.read("wiki/entities/vanadium-flow-battery.md")
    parsed = fm.parse(page)
    # The bogus aggregate reference the model emitted is dropped; the real
    # origin survives.
    assert parsed.get_list("sources") == ["grid-storage.md"]
    assert parsed.get_str("created") != "YYYY-MM-DD"
    assert parsed.get_str("created") == parsed.get_str("updated")


def test_aggregate_pages_are_never_written_by_the_model(wiki, settings, stub_llm, document):
    hostile = GENERATION + (
        "---FILE: wiki/index.md---\n# Wiki Index\n\n(everything else deleted)\n---END FILE---\n"
        "---FILE: wiki/overview.md---\nwiped\n---END FILE---\n"
    )
    stub_llm({"analysis": ANALYSIS, "generation": hostile})
    ingest_document(wiki, document, settings)

    assert "(everything else deleted)" not in wiki.read("wiki/index.md")
    assert "wiped" not in wiki.read("wiki/overview.md")


def test_path_traversal_in_generation_is_refused(wiki, settings, stub_llm, document, tmp_path):
    escape = tmp_path / "pwned.md"
    hostile = GENERATION + f"---FILE: ../../{escape.name}---\nowned\n---END FILE---\n"
    stub_llm({"analysis": ANALYSIS, "generation": hostile})
    result = ingest_document(wiki, document, settings)

    assert not escape.exists()
    assert any("unsafe path" in warning for warning in result.warnings)


def test_missing_source_summary_falls_back_to_the_analysis(wiki, settings, stub_llm, document):
    without_summary = "\n".join(
        block
        for block in GENERATION.split("---END FILE---\n")
        if "wiki/sources/grid-storage.md" not in block
    )
    stub_llm({"analysis": ANALYSIS, "generation": without_summary + "---END FILE---\n"})
    result = ingest_document(wiki, document, settings)

    summary = wiki.read("wiki/sources/grid-storage.md")
    assert "Round-trip efficiency" in summary, "the analysis is preserved in the recovery page"
    assert any("did not produce" in warning for warning in result.warnings)


def test_unchanged_document_hits_the_cache(wiki, settings, stub_llm, document):
    client = stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    first = ingest_document(wiki, document, settings)
    assert not first.cached

    second = ingest_document(wiki, document, settings)
    assert second.cached
    assert client.phases() == ["analysis", "generation"], "a cache hit must not call the model again"
    assert second.files_written == first.files_written


def test_cache_is_invalidated_when_a_written_page_disappears(wiki, settings, stub_llm, document):
    stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    ingest_document(wiki, document, settings)
    (wiki.root / "wiki/concepts/round-trip-efficiency.md").unlink()

    result = ingest_document(wiki, document, settings)
    assert not result.cached
    assert (wiki.root / "wiki/concepts/round-trip-efficiency.md").exists()


def test_force_re_ingests(wiki, settings, stub_llm, document):
    client = stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    ingest_document(wiki, document, settings)
    ingest_document(wiki, document, settings, force=True)
    assert client.phases() == ["analysis", "generation"] * 2


def test_truncated_page_triggers_a_repair_pass(wiki, settings, stub_llm, document):
    # Real truncation is end-of-stream: the closer never arrives, and
    # everything the model had left to say is gone with it.
    cut = GENERATION.index("Energy out divided by") + len("Energy out divided by")
    truncated = GENERATION[:cut]
    repair = (
        "---FILE: wiki/concepts/round-trip-efficiency.md---\n"
        "---\ntype: concept\ntitle: Round-trip Efficiency\nsources: [\"grid-storage.md\"]\n---\n\n"
        "# Round-trip Efficiency\n\nEnergy out divided by energy in.\n"
        "---END FILE---\n"
    )
    client = stub_llm({"analysis": ANALYSIS, "generation": truncated, "repair": repair})
    result = ingest_document(wiki, document, settings)

    assert client.phases() == ["analysis", "generation", "repair"]
    assert "wiki/concepts/round-trip-efficiency.md" in result.files_written
    assert "Energy out divided by energy in." in wiki.read("wiki/concepts/round-trip-efficiency.md")


def test_empty_generation_is_an_error_not_a_silent_success(wiki, settings, stub_llm, document):
    stub_llm({"analysis": ANALYSIS, "generation": "I'm sorry, I can't do that."})
    with pytest.raises(IngestError, match="no usable FILE blocks"):
        ingest_document(wiki, document, settings)


def test_long_source_is_analyzed_in_sections(wiki, settings, stub_llm, tmp_path):
    settings.llm.max_context_size = 40_000  # forces the chunked path
    long_document = tmp_path / "long.md"
    long_document.write_text(
        "\n\n".join(f"## Section {i}\n\n" + ("content " * 900) for i in range(6))
    )
    client = stub_llm({"chunk": "section analysis", "consolidate": ANALYSIS, "generation": GENERATION})
    result = ingest_document(wiki, long_document, settings)

    assert result.chunks_analyzed > 1
    # one call per section, one consolidation, one generation
    assert client.phases() == ["chunk"] * result.chunks_analyzed + ["consolidate", "generation"]


def test_unreadable_document_reports_clearly(wiki, settings, stub_llm, tmp_path):
    binary = tmp_path / "image.heic"
    binary.write_bytes(b"\x00\x01\x02\x03")
    stub_llm({})
    with pytest.raises(IngestError):
        ingest_document(wiki, binary, settings)


def test_cache_file_records_the_ingest(wiki, settings, stub_llm, document):
    stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    ingest_document(wiki, document, settings)
    cache = IngestCache.load(wiki)
    assert cache.identities() == ["grid-storage.md"]


def test_misfiled_source_summary_is_accepted_not_duplicated(wiki, settings, stub_llm, document):
    """One source, one summary page — even when the model picks its own name."""
    renamed = GENERATION.replace(
        "---FILE: wiki/sources/grid-storage.md---", "---FILE: wiki/sources/grid-storage-report.md---"
    )
    stub_llm({"analysis": ANALYSIS, "generation": renamed})
    result = ingest_document(wiki, document, settings)

    source_pages = [path for path in result.files_written if path.startswith("wiki/sources/")]
    assert source_pages == ["wiki/sources/grid-storage-report.md"]
    assert any("filed the source summary at" in warning for warning in result.warnings)


def test_folder_context_is_preserved_in_the_source_identity(wiki, settings, stub_llm, tmp_path):
    nested = tmp_path / "paper.md"
    nested.write_text("# Paper\n\nContent about storage.\n")
    client = stub_llm({"analysis": ANALYSIS, "generation": GENERATION})
    result = ingest_document(wiki, nested, settings, folder="papers/energy")

    assert result.source_identity == "papers/energy/paper.md"
    assert (wiki.sources_dir / "papers" / "energy" / "paper.md").exists()
    # The folder is passed to step 1 as a categorization hint.
    assert "papers/energy" in client.prompt("analysis", role="user")
