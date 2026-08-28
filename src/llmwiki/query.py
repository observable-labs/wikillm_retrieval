"""Phase 3 and 4 of the query pipeline: budget control, then the answer.

Retrieval (`retrieval.pipeline.search`) hands back ranked pages. This module
packs as many of them as the budget allows into a numbered context block and
asks the model to answer citing `[1]`, `[2]`, so every claim in the answer
traces to a page the user can open.

Budget allocation is proportional, from `budget.py`: 50% of the context
window for retrieved pages, 5% for the index, 15% held back so the model has
room to write. Pages are added in rank order and truncated individually at
`max_page_size`, so one enormous page can't consume the whole window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .budget import compute_context_budget, trim_long_text
from .config import Settings
from .llm import Message, build_client
from .retrieval import SearchResponse, search
from .retrieval.keyword import SearchResult

SYSTEM_HEADER = (
    "You are the maintainer and reader of a personal wiki built from the user's own documents.\n"
    "Answer from the retrieved pages below. They are the user's compiled knowledge; treat them as\n"
    "the primary evidence."
)

CITATION_RULES = """
Rules:
- Cite the pages you use by their number: [1], [2]. Cite the specific page a claim came from.
- If the pages disagree, say so and attribute each position to its page rather than picking silently.
- If the pages do not contain the answer, say what is missing and what document would settle it.
  Do not fill the gap from general knowledge without labelling it as outside the wiki.
- Prefer wiki pages over raw source excerpts when both cover a point; the wiki page is the
  maintained version. Use a raw source when it carries detail the page dropped.
- Be concrete. Quote exact figures, names, and definitions from the pages instead of paraphrasing
  them into vagueness.
"""

ANSWER_SHAPE = r"""
Shape:
- Lead with a direct answer in one or two sentences. Elaborate after it, not before it.
- Then the supporting detail, as short paragraphs or a single-level bullet list — whichever the
  material fits. Don't nest bullets more than one level, and don't add section headings.
- Plain Markdown, read in a terminal that does not render it. No LaTeX or math delimiters: write
  PrP^C, not $\text{PrP}^\text{C}$. No tables unless the question asks you to compare things.
- Don't write a Sources or References section. The tool prints one from the numbers you cite.
"""

# `[1]`, the `[1][2]` / `[1], [3]` runs models write when one claim rests on
# several pages, and `[1, 3]` — one bracket, several numbers, which is what a
# model reaches for when it isn't thinking hard about format. Missing that last
# form undercounts: an answer resting on four pages reported one. A markdown
# link's `[1](path)` matches too, which is correct — the page was still cited.
CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    number: int
    path: str
    title: str
    kind: str
    score: float
    graph_related_to: list[str] = field(default_factory=list)
    cited: bool = False


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieval: SearchResponse | None = None
    pages_used: int = 0
    pages_cited: int = 0
    pages_found: int = 0
    context_chars: int = 0
    notes: list[str] = field(default_factory=list)


def ask(
    project,
    question: str,
    settings: Settings,
    top_k: int = 20,
    include_sources: bool | None = None,
    history: list[tuple[str, str]] | None = None,
    on_token=None,
    profile: str | None = None,
    options=None,
) -> Answer:
    """Retrieve over the whole project, then answer with citations.

    `profile` selects a named retrieval configuration — how deep to look and
    which lane to trust. The default is the shipped one, so this argument
    changes nothing unless a caller asks it to; `research` is the one that leans
    on the vector lane, for questions asked before the user knows what the
    corpus calls things. `options` overrides it outright, for a caller — an
    ablation, a harness — holding a configuration rather than a name.
    """
    from .retrieval.profiles import resolve as resolve_profile

    include = settings.search_sources if include_sources is None else include_sources
    selected = options if options is not None else resolve_profile(profile).options
    response = search(
        project,
        question,
        top_k=top_k,
        include_sources=include,
        embedding_config=settings.embedding,
        options=selected,
    )
    return answer_from(
        project, question, settings, response,
        history=history, on_token=on_token,
    )


def answer_from(
    project,
    question: str,
    settings: Settings,
    response: SearchResponse,
    history: list[tuple[str, str]] | None = None,
    on_token=None,
) -> Answer:
    """Phases 3 and 4 alone: pack a ranking that already exists, then answer it.

    Split out of `ask` so that "generate an answer from *these* results" is a
    first-class operation. An evaluation harness comparing two retrievers on the
    text a user actually reads has to hold the generator fixed and vary only the
    ranking, and it cannot do that through a function that insists on doing the
    retrieval itself. The baseline that this makes possible — a lexical retriever
    answered by the same model with the same prompt and the same budget — is the
    only way to attribute a difference in answers to retrieval rather than to
    everything else.
    """
    budget = compute_context_budget(settings.llm.max_context_size)
    packed, citations = _pack_context(response.results, budget.page_budget, budget.max_page_size)

    if not packed:
        return Answer(
            question=question,
            text=(
                "Nothing in this project matches that question yet. "
                "Add a document with `llmwiki add <file>` and ask again."
            ),
            retrieval=response,
            pages_found=len(response.results),
            notes=response.notes,
        )

    index = trim_long_text(project.index(), budget.index_budget)
    purpose = project.purpose()
    system = "\n\n".join(
        part
        for part in (
            SYSTEM_HEADER,
            CITATION_RULES.strip(),
            ANSWER_SHAPE.strip(),
            f"## Wiki Purpose\n{purpose}" if purpose.strip() else "",
            f"## Wiki Index\n{index}" if index.strip() else "",
        )
        if part
    )

    conversation: list[Message] = [Message("system", system)]
    for role, content in history or []:
        conversation.append(Message(role, content))
    conversation.append(
        Message(
            "user",
            "\n".join(
                [
                    "## Retrieved pages",
                    "",
                    packed,
                    "",
                    "---",
                    "",
                    f"Question: {question}",
                    "",
                    "Answer using the pages above, citing them by number.",
                ]
            ),
        )
    )

    client = build_client(settings.llm)
    completion = client.complete(
        conversation,
        max_tokens=max(1024, budget.response_reserve // 3),
        on_token=on_token,
    )

    text = completion.text.strip()
    cited = _mark_cited(text, citations)

    return Answer(
        question=question,
        text=text,
        citations=citations,
        retrieval=response,
        pages_used=len(citations),
        pages_cited=cited,
        pages_found=len(response.results),
        context_chars=len(packed),
        notes=response.notes,
    )


def _mark_cited(text: str, citations: list[Citation]) -> int:
    """Flag the packed pages the answer actually referenced; return how many.

    `_pack_context` numbers every page it packs, which is what the model needs
    to cite. It is not the answer's bibliography: a twenty-page context
    routinely yields a four-page answer, and presenting all twenty as sources
    overstates what the answer rests on and buries the four that carry it.
    """
    referenced = {
        int(number)
        for group in CITATION_MARKER.findall(text)
        for number in group.split(",")
    }
    for citation in citations:
        citation.cited = citation.number in referenced
    return sum(1 for citation in citations if citation.cited)


def _pack_context(
    results: list[SearchResult],
    page_budget: int,
    max_page_size: int,
) -> tuple[str, list[Citation]]:
    """Fill the page budget in rank order; skip anything that no longer fits.

    A page too large for the remaining budget is skipped rather than
    truncated to nothing, and the loop continues — a later, smaller page may
    still fit, and dropping it would waste budget the ranking earned.
    """
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0

    for result in results:
        document = result.document
        content = document.content if document else result.snippet
        body = trim_long_text(content.strip(), max_page_size)
        header_kind = "source" if result.kind == "source" else "wiki page"
        related = (
            f"\nRelated via: {', '.join(result.graph_related_to)}" if result.graph_related_to else ""
        )
        number = len(citations) + 1
        block = (
            f"### [{number}] {result.title}\n"
            f"({header_kind}: {result.path}){related}\n\n"
            f"{body}\n"
        )
        if used + len(block) > page_budget:
            if used == 0:
                # Nothing packed yet and the first page overflows: hard-trim
                # it so the query still has evidence to work with.
                block = block[:page_budget]
            else:
                continue
        blocks.append(block)
        citations.append(
            Citation(
                number=number,
                path=result.path,
                title=result.title,
                kind=result.kind,
                score=result.score,
                graph_related_to=list(result.graph_related_to),
            )
        )
        used += len(block)

    return "\n".join(blocks), citations
