"""Scenario templates: the schema.md / purpose.md a new project starts from.

`schema.md` is the authoritative routing rule for the ingest step — it tells
the LLM which page types exist and which directory each one lives in. Adding
a row here (and a matching directory) is how you teach a project a new page
type; nothing else in the pipeline hardcodes the type list.

`purpose.md` is the wiki's *why*: goals, key questions, scope. It is read on
every ingest and every query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_NAMING = """## Naming Conventions

- Files: `kebab-case.md`
- Entities: match the official name where possible (e.g. `gpt-4.md`, `openai.md`)
- Concepts: descriptive noun phrases (e.g. `chain-of-thought.md`)
- For Chinese/Japanese/Korean titles keep the CJK characters — do not romanize
"""

BASE_FRONTMATTER = """## Frontmatter

All pages must include YAML frontmatter:

```yaml
---
type: entity
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag1, tag2]
related: [other-page-slug]
sources: ["original-file.pdf"]
---
```

`sources` is what makes a page traceable back to the raw document it came
from, and it is also a retrieval signal — two pages sharing a source are
treated as related even without an explicit link.
"""

BASE_INDEX_FORMAT = """## Index Format

`wiki/index.md` lists pages grouped by type. Each entry:

`- [[page-slug]] — one-line summary`
"""

BASE_LOG_FORMAT = """## Log Format

`wiki/log.md` records activity in reverse chronological order:

`## [YYYY-MM-DD] ingest | Source Title`

The consistent prefix keeps the log greppable: `grep "^## \\[" wiki/log.md | tail -5`.
"""

BASE_CROSSREF = """## Cross-references

- Use `[[page-slug]]` syntax in page bodies to link between wiki pages
- Every entity and concept should appear in `wiki/index.md`
- Prefer linking to creating a near-duplicate page
"""

BASE_CONTRADICTION = """## Contradictions

When sources disagree, do not silently pick a winner. State both positions on
the page, attribute each to its source, and raise a `contradiction` review
item so a human can adjudicate.
"""


@dataclass(frozen=True)
class PageType:
    name: str
    directory: str
    purpose: str


@dataclass(frozen=True)
class Template:
    id: str
    label: str
    description: str
    page_types: list[PageType]
    purpose: str
    extra_rules: str = ""
    directories: list[str] = field(default_factory=list)

    def schema_markdown(self) -> str:
        rows = "\n".join(
            f"| {pt.name} | {pt.directory} | {pt.purpose} |" for pt in self.page_types
        )
        sections = [
            "# Wiki Schema",
            "",
            "## Page Types",
            "",
            "| Type | Directory | Purpose |",
            "|------|-----------|---------|",
            rows,
            "",
            BASE_NAMING,
            BASE_FRONTMATTER,
            BASE_INDEX_FORMAT,
            BASE_LOG_FORMAT,
            BASE_CROSSREF,
            BASE_CONTRADICTION,
        ]
        if self.extra_rules:
            sections.append(self.extra_rules)
        return "\n".join(sections).rstrip() + "\n"

    def wiki_directories(self) -> list[str]:
        dirs = {pt.directory.rstrip("/") for pt in self.page_types}
        dirs.update(d.rstrip("/") for d in self.directories)
        return sorted(dirs)


_COMMON = [
    PageType("source", "wiki/sources/", "One page per ingested document"),
    PageType("query", "wiki/queries/", "Answers worth keeping, filed back into the wiki"),
    PageType("synthesis", "wiki/synthesis/", "Cross-cutting summaries and conclusions"),
]

RESEARCH = Template(
    id="research",
    label="Research",
    description="Papers, articles, and reports on a topic, with an evolving thesis.",
    page_types=[
        PageType("entity", "wiki/entities/", "Named things (models, companies, people, datasets)"),
        PageType("concept", "wiki/concepts/", "Ideas, techniques, phenomena"),
        PageType("comparison", "wiki/comparisons/", "Side-by-side analysis of related entities"),
        *_COMMON,
    ],
    purpose="""# Purpose

## Goal

<!-- What are you trying to understand? Be specific — this steers every ingest. -->

## Key Questions

<!-- The questions this wiki exists to answer. Update them as they sharpen. -->

- 
- 

## Scope

**In scope:** 

**Out of scope:** 

## Evolving Thesis

<!-- Your current best answer. The LLM revises this as sources accumulate. -->
""",
)

READING = Template(
    id="reading",
    label="Reading a book",
    description="File chapters as you read; build a companion wiki of characters and themes.",
    page_types=[
        PageType("character", "wiki/characters/", "People in the text"),
        PageType("theme", "wiki/themes/", "Recurring ideas and motifs"),
        PageType("place", "wiki/places/", "Locations and settings"),
        PageType("event", "wiki/events/", "Plot events and turning points"),
        PageType("concept", "wiki/concepts/", "Ideas the text argues for"),
        *_COMMON,
    ],
    purpose="""# Purpose

## What I'm Reading

<!-- Title, author, edition. -->

## Why

<!-- What you want out of this book. -->

## Track

- Characters and how they change
- Themes and where they recur
- Arguments I agree or disagree with
""",
)

PERSONAL = Template(
    id="personal",
    label="Personal",
    description="Journal entries, health, goals — a structured picture of yourself over time.",
    page_types=[
        PageType("goal", "wiki/goals/", "Things you're working toward"),
        PageType("habit", "wiki/habits/", "Recurring practices and their outcomes"),
        PageType("reflection", "wiki/reflections/", "Journal-derived observations"),
        PageType("concept", "wiki/concepts/", "Frameworks and ideas you're applying"),
        PageType("entity", "wiki/entities/", "People, places, tools that recur"),
        *_COMMON,
    ],
    extra_rules="""## Personal-wiki rules

Only record goals, habits, and reflections the source actually contains.
Never invent a goal or a journal entry to fill out a page type.
""",
    purpose="""# Purpose

## What this wiki tracks

<!-- Health, career, relationships, learning — name the areas. -->

## Questions I'm asking myself

- 
- 
""",
)

BUSINESS = Template(
    id="business",
    label="Business / team",
    description="Meeting notes, customer calls, project docs — a wiki that maintains itself.",
    page_types=[
        PageType("person", "wiki/people/", "Colleagues, customers, contacts"),
        PageType("organization", "wiki/organizations/", "Companies, teams, vendors"),
        PageType("project", "wiki/projects/", "Ongoing workstreams"),
        PageType("decision", "wiki/decisions/", "Decisions made, with rationale"),
        PageType("meeting", "wiki/meetings/", "Meeting and call records"),
        PageType("concept", "wiki/concepts/", "Domain ideas and methods"),
        *_COMMON,
    ],
    purpose="""# Purpose

## What this wiki is for

<!-- The team, the domain, what people should be able to look up here. -->

## Key Questions

- 
- 
""",
)

GENERAL = Template(
    id="general",
    label="General",
    description="A neutral starting point: entities, concepts, sources.",
    page_types=[
        PageType("entity", "wiki/entities/", "Named things (people, tools, organizations, datasets)"),
        PageType("concept", "wiki/concepts/", "Ideas, techniques, phenomena"),
        PageType("comparison", "wiki/comparisons/", "Side-by-side analysis"),
        *_COMMON,
    ],
    purpose="""# Purpose

## Goal

<!-- What is this wiki accumulating knowledge about? -->

## Key Questions

- 
- 

## Scope

<!-- What belongs here and what doesn't. -->
""",
)

TEMPLATES: dict[str, Template] = {
    t.id: t for t in (RESEARCH, READING, PERSONAL, BUSINESS, GENERAL)
}

DEFAULT_TEMPLATE = "general"


def get_template(template_id: str) -> Template:
    try:
        return TEMPLATES[template_id]
    except KeyError:
        known = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"unknown template {template_id!r}; known templates: {known}") from None
