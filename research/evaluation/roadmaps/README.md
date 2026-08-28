# roadmaps/

Corrective roadmaps for the evaluation stack — written *after* a harness exists
and has been run adversarially, so each one is anchored to an observed failure
rather than to a design review.

Distinct from [`../harness-v1.md`](../harness-v1.md), which is the original
design. Documents here say what to change and why, against a named commit of the
implementation.

| Document | Question it answers | Status |
|---|---|---|
| [harness-self-validation.md](harness-self-validation.md) | The harness measures the right things. Why did its first adversarial run report numbers that did not mean what their labels said? | implemented; §4b and §10 record what each step became |

What the closed harness then measured — and the retrieval rebuild it made
possible — is in
[`../../../future_work/retrieval-rebuild/`](../../../future_work/retrieval-rebuild/README.md),
which is an assessment of this implementation rather than a roadmap, and so
belongs there rather than here.

## Conventions

Inherited from [`../../README.md`](../../README.md), plus one:

- **Anchor to a run, not an opinion.** Every defect names the command that
  reproduces it and the commit it reproduces against. A finding that cannot be
  reproduced from a shipped fixture is a hypothesis and is labelled as one.
