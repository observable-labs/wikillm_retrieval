# research/

External research consulted while working on llmwiki's retrieval stack, with
findings mapped onto this codebase.

Distinct from [`../future_work/`](../future_work/), which holds assessments of
*this* implementation and the work queue derived from them. Documents here
summarize outside work and say what it implies; documents there say what to
build.

[target-architecture.md](target-architecture/README.md) is the exception and the entry
point: it is the synthesis of the other three into a recommendation, and its §9
supersedes the work-items ordering. Read it first; read the others for the
evidence behind it.

| Document | Question it answers |
|---|---|
| [combining-rag-strategies.md](combining-rag-strategies.md) | Can retrieval strategies be paired for the best of both worlds, at indexing time and at retrieval time? |
| [latency-knobs.md](latency-knobs.md) | Which approaches expose a usable latency/quality dial, and what does a voice-grade path require? |
| [incremental-updates.md](incremental-updates.md) | What can be updated live without a rebuild, and what blocks concurrent ingest today? |
| **[target-architecture/](target-architecture/README.md)** | **Synthesis: given all of the above, what should actually be built?** |
| [target-architecture/build-plan.md](target-architecture/build-plan.md) | The same recommendation as an ordered, checkable build plan. |
| **[evaluation/](evaluation/README.md)** | **Where should evaluation start, what should it measure, and what suites already exist?** |
| [evaluation/harness-v1.md](evaluation/harness-v1.md) | The buildable design for the first evaluation harness. |
| [evaluation/benchmarking.md](evaluation/benchmarking.md) | Survey: how SOTA methods benchmark, and which public suites are worth running. |
| [tooling.md](tooling.md) | Which of these systems are installable packages, and which are papers to reimplement? |

## Conventions

- Date each document and record the commit the analysis was made against.
- Label evidence quality — peer-reviewed, preprint, vendor benchmark,
  practitioner consensus. Much of the 2025–2026 RAG literature is
  self-published and self-benchmarked.
- Record what was *not* read, so a later reader knows where the thin ice is.
