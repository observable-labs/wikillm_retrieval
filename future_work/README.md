# future_work/

Assessments of *this* implementation, and the work derived from them.

Distinct from [`../research/`](../research/README.md), which summarizes outside
work and says what it implies. Documents here say what this codebase does, what
is wrong with it, and what was done about it — against a named commit, with the
commands that reproduce the numbers.

| Document | Question it answers |
|---|---|
| [retrieval-vs-sota/](retrieval-vs-sota/README.md) | How does llmwiki's retrieval compare to RAPTOR, GraphRAG, LightRAG and HippoRAG, and what does that imply? |
| [retrieval-vs-sota/work-items.md](retrieval-vs-sota/work-items.md) | The prioritized fixes derived from that assessment. |
| **[retrieval-rebuild/](retrieval-rebuild/README.md)** | **A BM25 baseline was beating the pipeline. What was actually wrong, what was rebuilt, and what does it measure now?** |

## Conventions

- Date each document and record the commit it was written against.
- **Quote no number without the command that produces it.** A measurement whose
  method is not written down is an opinion with a decimal point.
- Record what did *not* work, and what a result would look like if it were wrong.
