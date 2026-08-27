"""Asking a project: retrieval feeds a citation-bearing answer."""

from __future__ import annotations

from llmwiki.query import _pack_context, ask
from llmwiki.retrieval.keyword import Document, SearchResult


def _page(project, path, title, body, sources=(), page_type="concept"):
    source_field = ", ".join(f'"{s}"' for s in sources)
    project.write(
        path,
        f"---\ntype: {page_type}\ntitle: {title}\nsources: [{source_field}]\n---\n\n# {title}\n\n{body}\n",
    )


def test_ask_retrieves_over_the_whole_project_and_cites(wiki, settings, stub_llm):
    _page(wiki, "wiki/concepts/round-trip-efficiency.md", "Round-trip Efficiency",
          "Energy out divided by energy in. Flow batteries reach 75%.", ["grid.md"])
    _page(wiki, "wiki/entities/vanadium-flow-battery.md", "Vanadium Flow Battery",
          "Long-duration storage. See [[round-trip-efficiency]].", ["grid.md"], "entity")
    _page(wiki, "wiki/concepts/solar-pv.md", "Solar PV", "Unrelated to storage chemistry.")

    client = stub_llm({"answer": "Flow batteries reach about 75% round-trip efficiency [1]."})
    answer = ask(wiki, "What round-trip efficiency do flow batteries reach?", settings)

    assert "[1]" in answer.text
    assert answer.citations[0].path == "wiki/concepts/round-trip-efficiency.md"
    assert answer.pages_used >= 1

    context = client.prompt("answer", role="user")
    assert "### [1] Round-trip Efficiency" in context
    assert "Flow batteries reach 75%." in context, "full page content, not just a snippet"


def test_answer_context_includes_purpose_and_index(wiki, settings, stub_llm):
    wiki.write("purpose.md", "# Purpose\n\nTrack grid storage economics.\n")
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Storage content about batteries.")
    client = stub_llm({"answer": "ok"})
    ask(wiki, "storage", settings)

    system = client.prompt("answer")
    assert "Track grid storage economics" in system
    assert "## Wiki Index" in system
    assert "[1], [2]" in system, "the citation contract is stated to the model"


def test_graph_reached_pages_are_labelled_in_the_context(wiki, settings, stub_llm):
    _page(wiki, "wiki/concepts/photosynthesis.md", "Photosynthesis", "Plants use [[chlorophyll]].")
    _page(wiki, "wiki/entities/chlorophyll.md", "Chlorophyll", "A green pigment.", (), "entity")
    client = stub_llm({"answer": "ok"})
    answer = ask(wiki, "photosynthesis", settings)

    assert "Related via: Photosynthesis" in client.prompt("answer", role="user")
    reached = next(c for c in answer.citations if c.path == "wiki/entities/chlorophyll.md")
    assert reached.graph_related_to == ["Photosynthesis"]


def test_empty_project_answers_without_calling_the_model(wiki, settings, stub_llm):
    client = stub_llm({})
    answer = ask(wiki, "anything at all", settings)
    assert client.calls == []
    assert "llmwiki add" in answer.text


def test_no_sources_flag_restricts_retrieval_to_wiki_pages(wiki, settings, stub_llm):
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Wiki page about storage.")
    wiki.sources_dir.mkdir(parents=True, exist_ok=True)
    (wiki.sources_dir / "raw.txt").write_text("raw notes about storage")

    stub_llm({"answer": "ok"})
    with_sources = ask(wiki, "storage", settings, include_sources=True)
    without_sources = ask(wiki, "storage", settings, include_sources=False)

    assert any(c.kind == "source" for c in with_sources.citations)
    assert all(c.kind == "wiki" for c in without_sources.citations)


def test_packing_respects_the_page_budget_and_renumbers():
    results = [
        SearchResult(path="wiki/a.md", title="A", snippet="", score=5,
                     document=Document("wiki/a.md", "A", "a" * 100)),
        SearchResult(path="wiki/big.md", title="Big", snippet="", score=4,
                     document=Document("wiki/big.md", "Big", "b" * 5000)),
        SearchResult(path="wiki/c.md", title="C", snippet="", score=3,
                     document=Document("wiki/c.md", "C", "c" * 50)),
    ]
    packed, citations = _pack_context(results, page_budget=400, max_page_size=300)

    # The oversized page is skipped, but the smaller one after it still fits,
    # and citation numbers stay contiguous with what was actually packed.
    assert [c.path for c in citations] == ["wiki/a.md", "wiki/c.md"]
    assert [c.number for c in citations] == [1, 2]
    assert len(packed) <= 400


def test_only_the_pages_the_answer_cited_are_marked(wiki, settings, stub_llm):
    """The context is numbered for the model's benefit, not as a bibliography.

    Twenty packed pages routinely yield a four-page answer; treating the whole
    packed list as sources overstates what the answer rests on.
    """
    for name in ("alpha", "beta", "gamma"):
        _page(wiki, f"wiki/concepts/{name}.md", name.title(), f"Storage notes about {name}.")

    stub_llm({"answer": "Storage is covered [2], and see also [2][3] together."})
    answer = ask(wiki, "storage", settings)

    assert answer.pages_used == 3, "all three were packed into the context"
    assert answer.pages_cited == 2
    assert [c.number for c in answer.citations if c.cited] == [2, 3]


def test_uncited_answers_leave_every_page_unmarked(wiki, settings, stub_llm):
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Storage notes.")
    stub_llm({"answer": "No citations here at all."})
    answer = ask(wiki, "storage", settings)

    assert answer.pages_cited == 0
    assert not any(c.cited for c in answer.citations)


def test_the_model_is_told_the_output_is_read_in_a_terminal(wiki, settings, stub_llm):
    """Without this the model emits LaTeX and headings into a plain terminal."""
    _page(wiki, "wiki/concepts/storage.md", "Storage", "Storage notes.")
    client = stub_llm({"answer": "ok [1]"})
    ask(wiki, "storage", settings)

    system = client.prompt("answer")
    assert "No LaTeX" in system
    assert "Sources or References section" in system
