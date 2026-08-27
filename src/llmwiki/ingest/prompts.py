"""The ingest prompts — the actual content of the two-step chain of thought.

Step 1 reads the source and writes a structured *analysis*: entities,
concepts, arguments, connections to what the wiki already knows,
contradictions, and recommendations. Nothing is written to disk.

Step 2 reads that analysis (plus the source, schema, purpose, and index) and
emits FILE blocks — the actual wiki pages. Splitting the work in two is the
single biggest quality lever in llm_wiki: a model asked to read and file
simultaneously does both worse.

Ordering inside step 2's prompt is deliberate. The output format is the
*last* section because models weight recent instructions most heavily, and
the language directive is repeated at the very end so it wins the
most-recent-instruction tie-break — small models otherwise drift back to
their training language partway through a long generation.
"""

from __future__ import annotations

from ..budget import trim_long_text

WIKI_TYPES = (
    "entity",
    "concept",
    "source",
    "query",
    "comparison",
    "synthesis",
    "overview",
)


def language_rule(output_language: str, source_content: str = "") -> str:
    """Pin the output language, auto-detecting from the source when asked."""
    if output_language and output_language != "auto":
        language = output_language
    else:
        language = _detect_language(source_content)
    return (
        f"MANDATORY OUTPUT LANGUAGE: {language}. Write all prose, page titles, and section "
        "headings in this language. Preserve proper nouns, acronyms, model and dataset names, "
        "tool and library names, code identifiers, URLs, file names, and technical terms with "
        "no widely-used localized equivalent in their original form."
    )


def _detect_language(text: str) -> str:
    """Cheap script detection; ties go to English."""
    sample = text[:2000]
    if not sample.strip():
        return "English"
    counts = {"Japanese": 0, "Korean": 0, "Chinese": 0}
    for ch in sample:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            counts["Japanese"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            counts["Korean"] += 1
        elif 0x4E00 <= code <= 0x9FFF:
            counts["Chinese"] += 1
    best = max(counts, key=lambda key: counts[key])
    # Kana settles Japanese-vs-Chinese: both use Han characters, only one
    # uses kana, so any kana at all outweighs a larger Han count.
    if counts["Japanese"] > 0:
        return "Japanese"
    if counts[best] > len(sample) * 0.05:
        return best
    return "English"


def build_analysis_prompt(
    purpose: str,
    index: str,
    source_content: str = "",
    schema: str = "",
    output_language: str = "auto",
) -> str:
    """Step 1: read the source, produce a structured analysis."""
    sections = [
        "You are an expert research analyst. Read the source document and produce a structured analysis.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript. Reason internally and write only the concise final analysis.",
        "",
        language_rule(output_language, source_content),
        "",
        "Your analysis should cover:",
        "",
        "## Key Entities",
        "List people, organizations, products, datasets, tools mentioned. For each:",
        "- Name and type",
        "- Role in the source (central vs. peripheral)",
        "- Whether it likely already exists in the wiki (check the index)",
        "",
        "## Key Concepts",
        "List theories, methods, techniques, phenomena. For each:",
        "- Name and brief definition",
        "- Why it matters in this source",
        "- Whether it likely already exists in the wiki",
        "",
        "## Main Arguments & Findings",
        "- What are the core claims or results?",
        "- What evidence supports them?",
        "- How strong is the evidence?",
        "- Which named subject is each claim about? Do not transfer claims, limits, or evaluations from one entity/model/product/method to another just because they share keywords.",
        "- Preserve structured source data verbatim in the analysis when present: include SQL DDL / CREATE TABLE statements, schema definitions, API signatures, configuration, and tables in fenced code blocks or Markdown tables. Do not reduce exact field names, types, constraints, keys, or indexes to prose.",
        "",
        "## Connections to Existing Wiki",
        "- What existing pages does this source relate to?",
        "- Does it strengthen, challenge, or extend existing knowledge?",
        "",
        "## Contradictions & Tensions",
        "- Does anything in this source conflict with existing wiki content?",
        "- Are there internal tensions or caveats?",
        "",
        "## Recommendations",
        "- What wiki pages should be created or updated?",
        "- If the project schema (below) defines page types beyond entity/concept (e.g. goal, habit, reflection, finding, decision, meeting), and the source genuinely contains matching content, recommend pages of those types — name the type explicitly. Only when the source actually supports it; never invent goals, habits, or journal entries that aren't in the source.",
        "- What should be emphasized vs. de-emphasized?",
        "- Any open questions worth flagging for the user?",
        "",
        "Be thorough but concise. Focus on what's genuinely important.",
        "",
        "If a folder context is provided, use it as a hint for categorization — the folder structure often reflects the user's organizational intent (e.g. 'papers/energy' suggests an energy-related paper).",
        "",
    ]
    if schema:
        sections.append(
            "## Project Schema (page types available — map source content to schema-defined types when it fits)\n"
            + schema
        )
    if purpose:
        sections.append(f"## Wiki Purpose (for context)\n{purpose}")
    if index:
        sections.append(f"## Current Wiki Index (for checking existing content)\n{index}")
    return "\n".join(section for section in sections if section != "" or True).strip()


def build_analysis_user_message(source_identity: str, source_context: str, folder_context: str = "") -> str:
    folder = f"\n**Folder context:** {folder_context}" if folder_context else ""
    return (
        f"Analyze this source document:\n\n**File:** {source_identity}{folder}\n\n---\n\n{source_context}"
    )


def build_generation_prompt(
    schema: str,
    purpose: str,
    index: str,
    source_identity: str,
    source_summary_path: str,
    today: str,
    overview: str = "",
    source_content: str = "",
    output_language: str = "auto",
) -> str:
    """Step 2: turn the analysis into FILE blocks."""
    sections: list[str] = [
        "You are a wiki maintainer. Based on the analysis provided, generate wiki files.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble. Reason internally and output only the requested FILE/REVIEW blocks.",
        "",
        language_rule(output_language, source_content),
        "",
        "## IMPORTANT: Source File",
        f"The original source file is: **{source_identity}**",
        "All wiki pages generated from this source MUST include this filename in their frontmatter `sources` field.",
        f"Today's date is **{today}**. Use this exact date for all new `created` and `updated` values and for the log entry.",
        "",
    ]

    if schema:
        sections += [
            "## Project Schema and Routing (AUTHORITATIVE)",
            schema,
            "",
            "Use this schema as the primary routing rule for page types and directories.",
            "If it defines custom folders (for example people, technologies, organizations, methods, decisions, or cases), write pages into those schema-defined folders instead of forcing them into wiki/entities/ or wiki/concepts/.",
            "Use wiki/entities/ and wiki/concepts/ only when the schema does not provide a more specific destination.",
            "Every generated page's frontmatter `type` must match the schema directory used in its FILE path.",
            "",
        ]

    sections += [
        "## What to generate",
        "",
        f"1. A source summary page at **{source_summary_path}** (MUST use this exact path)",
        "2. Entity or schema-defined typed pages for the key named things identified in the analysis. Prefer schema-defined directories when present; otherwise use wiki/entities/.",
        "3. Concept or schema-defined typed pages for key ideas, methods, techniques, and abstractions. Prefer schema-defined directories when present; otherwise use wiki/concepts/.",
        "4. A log entry for wiki/log.md — just the new entry to append, in the format `## [YYYY-MM-DD] ingest | Title`.",
        "Do not generate wiki/index.md or wiki/overview.md. The application maintains those aggregate files separately so a large wiki is never rewritten through model output.",
        "",
        "## Frontmatter Rules (CRITICAL — the parser is strict)",
        "",
        "Every page begins with a YAML frontmatter block. Format rules, in order of importance:",
        "",
        "1. The VERY FIRST line of the file MUST be exactly `---` (three hyphens, nothing else).",
        "   Do NOT wrap the file in a ```yaml ... ``` code fence.",
        "   Do NOT prefix it with a `frontmatter:` key or any other line.",
        "2. Each frontmatter line is a `key: value` pair on its own line.",
        "3. The frontmatter ends with another `---` line on its own.",
        "4. The next line after the closing `---` starts the page body.",
        "5. Arrays use the inline form `[a, b, c]`.",
        "   Wikilinks belong in the BODY only — never write `related: [[a]], [[b]]` (invalid YAML);",
        "   write `related: [a, b]` with bare slugs.",
        "",
        "Required fields and types:",
        f"  - type     — one of the known types ({' | '.join(WIKI_TYPES)}), or a custom type defined by the project schema",
        '  - title    — string (quote it if it contains a colon, e.g. `title: "Foo: Bar"`)',
        f"  - created  — {today} for new pages (YYYY-MM-DD, no quotes)",
        f"  - updated  — {today} for new pages (same as created)",
        "  - tags     — array of bare strings: `tags: [microbiology, ai]`",
        "  - related  — array of bare wiki page slugs: `related: [foo, bar-baz]`. Do NOT include",
        "               `wiki/`, `.md`, or `[[...]]` here — slugs only.",
        f'  - sources  — array of source filenames; MUST include "{source_identity}".',
        "",
        "A complete, parseable page looks like this:",
        "",
        "    ---",
        "    type: entity",
        "    title: Example Entity",
        f"    created: {today}",
        f"    updated: {today}",
        "    tags: [example, demo]",
        "    related: [related-slug-1, related-slug-2]",
        f'    sources: ["{source_identity}"]',
        "    ---",
        "",
        "    # Example Entity",
        "",
        "    Body content goes here. Use [[wikilink]] syntax in the body for cross-references.",
        "",
        "Other rules:",
        "- Use [[wikilink]] syntax in the BODY for cross-references between pages.",
        "- Preserve subject boundaries: when a source discusses multiple entities, models, products, or methods, keep claims, evaluations, limitations, and benchmark results attached to the exact subject they describe.",
        "- Do not merge or generalize a claim about one subject into another subject's page just because they share terms (context window size, benchmark name, dataset, architecture, feature name).",
        "- If a page mentions another subject for comparison, write it explicitly as a comparison and say which source supports the statement.",
        "- Use kebab-case for Latin-script filenames; for Chinese/Japanese/Korean titles keep the CJK characters (do NOT romanize to pinyin/romaji or translate to English).",
        "- Derive filenames from the page title, but short proper nouns and technical identifiers take precedence: preserve names such as OpenAI, GPT-5, Transformer, CLIP, ImageNet, PyTorch, CUDA, GitHub, arXiv, and model, dataset, and tool names in their standard form. Do not put raw URLs, citation strings, or full paper titles into file paths.",
        "- Preserve structured source data verbatim: copy SQL DDL / CREATE TABLE statements, schema definitions, API signatures, configuration, and tabular data into fenced code blocks or Markdown tables in the source summary page instead of paraphrasing them. Exact column names, types, constraints, keys, and indexes must survive ingest — a prose-only summary loses the structure the document was imported to keep.",
        "- Follow the analysis's recommendations on what to emphasize.",
        "- If the analysis found connections to existing pages, add the cross-references.",
        "",
        "## Review block types",
        "",
        "After all FILE blocks, optionally emit REVIEW blocks for anything that needs human judgment:",
        "",
        "- contradiction: the analysis found a conflict with existing wiki content",
        "- duplicate: an entity or concept may already exist under a different name",
        "- missing-page: an important concept is referenced but has no page of its own",
        "- suggestion: further research, related sources to look for, or connections worth exploring",
        "",
        "Only create reviews for things that genuinely need human input. Do not create trivial reviews.",
        "",
        "## OPTIONS allowed values (only these predefined labels):",
        "",
        "- contradiction: OPTIONS: Create Page | Skip",
        "- duplicate: OPTIONS: Create Page | Skip",
        "- missing-page: OPTIONS: Create Page | Skip",
        "- suggestion: OPTIONS: Create Page | Skip",
        "",
        "Do NOT invent custom option labels.",
        "",
        "For suggestion and missing-page reviews, the SEARCH field must contain 2-3 web search queries",
        "(keyword-rich, specific, suitable for a search engine — NOT titles or sentences). Example:",
        "  SEARCH: automated technical debt detection AI generated code | software quality metrics LLM code generation | static analysis agentic development",
        "",
    ]

    if purpose:
        sections.append(f"## Wiki Purpose\n{purpose}")
    if index:
        sections.append(f"## Current Wiki Index (existing pages — link to them rather than duplicating)\n{index}")
    if overview:
        sections.append(f"## Current Overview\n{overview}")

    # Output format goes last: models weight the most recent instructions
    # most heavily, and a mis-formatted response is unrecoverable.
    sections += [
        "",
        "## Output Format (MUST FOLLOW EXACTLY — this is how the parser reads your response)",
        "",
        "Your ENTIRE response consists of FILE blocks followed by optional REVIEW blocks. Nothing else.",
        "",
        "FILE block template:",
        "```",
        "---FILE: wiki/path/to/page.md---",
        "(complete file content with YAML frontmatter)",
        "---END FILE---",
        "```",
        "",
        "REVIEW block template (optional, after all FILE blocks):",
        "```",
        "---REVIEW: type | Title---",
        "Description of what needs the user's attention.",
        "OPTIONS: Create Page | Skip",
        "PAGES: wiki/page1.md, wiki/page2.md",
        "SEARCH: query 1 | query 2 | query 3",
        "---END REVIEW---",
        "```",
        "",
        "## Output Requirements (STRICT — deviations cause parse failure)",
        "",
        "1. The FIRST character of your response MUST be `-` (the opening of `---FILE:`).",
        '2. DO NOT output a preamble such as "Here are the files:" or "Based on the analysis...".',
        "3. DO NOT echo or restate the analysis — that was step 1's job. Your job is to emit FILE blocks.",
        "4. DO NOT output markdown tables, bullet lists, or headings outside of FILE/REVIEW blocks.",
        "5. DO NOT output trailing commentary after the last `---END FILE---` or `---END REVIEW---`.",
        "6. Between blocks, use only blank lines — no prose.",
        "",
        "If you start with anything other than `---FILE:`, the entire response will be discarded.",
        "",
        "---",
        "",
        language_rule(output_language, source_content),
    ]
    return "\n".join(sections)


def build_generation_user_message(source_identity: str, analysis: str, source_context: str) -> str:
    return "\n".join(
        [
            f"Source document to process: **{source_identity}**",
            "",
            "The step 1 analysis below is CONTEXT to inform your output. Do NOT echo its tables,",
            "bullet points, or prose. Your output must be FILE/REVIEW blocks as specified in the",
            "system prompt — nothing else.",
            "",
            "## Step 1 Analysis (context only — do not repeat)",
            "",
            analysis,
            "",
            "## Source Context",
            "",
            source_context,
            "",
            "---",
            "",
            f"Now emit the FILE blocks for the wiki pages derived from **{source_identity}**.",
            "Your response MUST begin with `---FILE:` as the very first characters.",
            "No preamble. No analysis prose. Start immediately.",
        ]
    )


def build_chunk_analysis_prompt(
    purpose: str,
    schema: str,
    output_language: str,
    source_content: str,
) -> str:
    """Step 1 for one chunk of an over-budget source."""
    sections = [
        "You are analyzing ONE SECTION of a longer source document.",
        "Produce a focused analysis of this section only. Do not speculate about sections you have not seen.",
        "Do not output chain-of-thought or a thinking transcript.",
        "",
        language_rule(output_language, source_content),
        "",
        "Cover, briefly: key entities, key concepts, main claims and their evidence, and anything",
        "that contradicts or qualifies what earlier sections established.",
        "Preserve structured data (SQL DDL, schemas, API signatures, tables) verbatim in fenced blocks.",
        "",
    ]
    if schema:
        sections.append(f"## Project Schema\n{trim_long_text(schema, 4000)}")
    if purpose:
        sections.append(f"## Wiki Purpose\n{trim_long_text(purpose, 4000)}")
    return "\n".join(sections)


def build_consolidation_prompt(
    purpose: str,
    index: str,
    schema: str,
    output_language: str,
    source_content: str,
) -> str:
    """Merge per-chunk analyses into the single analysis step 2 consumes."""
    sections = [
        "You are consolidating per-section analyses of one long source document into a single",
        "structured analysis, in the exact shape a wiki maintainer expects.",
        "Do not output chain-of-thought or a thinking transcript.",
        "",
        language_rule(output_language, source_content),
        "",
        "Merge duplicates, reconcile section-level disagreements (and note them explicitly under",
        "Contradictions & Tensions), and keep every structured data block that appeared in a section",
        "analysis. Use these headings:",
        "",
        "## Key Entities",
        "## Key Concepts",
        "## Main Arguments & Findings",
        "## Connections to Existing Wiki",
        "## Contradictions & Tensions",
        "## Recommendations",
        "",
    ]
    if schema:
        sections.append(f"## Project Schema\n{trim_long_text(schema, 4000)}")
    if purpose:
        sections.append(f"## Wiki Purpose\n{trim_long_text(purpose, 4000)}")
    if index:
        sections.append(f"## Current Wiki Index\n{trim_long_text(index, 4000)}")
    return "\n".join(sections)


def build_repair_prompt(
    paths: list[str],
    source_identity: str,
    schema: str,
    purpose: str,
    analysis: str,
    source_context: str,
    max_ctx: int,
    output_language: str = "auto",
) -> str:
    """Re-request only the FILE blocks that were truncated mid-stream."""
    cap = max(4_000, int(max_ctx * 0.12))
    sections = [
        "You are repairing truncated wiki FILE blocks from an earlier generation.",
        "Return exactly one complete FILE block for each requested path and no other files.",
        "Every block must end with `---END FILE---`. Do not output a preamble, REVIEW blocks, or trailing commentary.",
        "Preserve the requested paths exactly and include the source identity in each page's frontmatter `sources` field.",
        "",
        language_rule(output_language, source_context),
        "",
        "## Requested paths",
        *[f"- {path}" for path in paths],
        "",
        f"## Source identity\n{source_identity}",
    ]
    if schema:
        sections.append(f"## Project schema\n{trim_long_text(schema, cap)}")
    if purpose:
        sections.append(f"## Wiki purpose\n{trim_long_text(purpose, cap)}")
    sections.append(f"## Step 1 analysis\n{trim_long_text(analysis, cap)}")
    sections.append(f"## Source context\n{trim_long_text(source_context, cap)}")
    return "\n".join(sections)


def build_fallback_source_summary(source_identity: str, analysis: str, date: str) -> str:
    """The safety net: a source page always exists, even if step 2 omitted it.

    The full analysis is kept rather than truncated — this is a recovery
    page, and a syntactically valid but silently incomplete summary is worse
    than a long one.
    """
    return "\n".join(
        [
            "---",
            "type: source",
            f'title: "Source: {source_identity}"',
            f"created: {date}",
            f"updated: {date}",
            f'sources: ["{source_identity}"]',
            "tags: []",
            "related: []",
            "---",
            "",
            f"# Source: {source_identity}",
            "",
            analysis or "(Analysis not available)",
            "",
        ]
    )
