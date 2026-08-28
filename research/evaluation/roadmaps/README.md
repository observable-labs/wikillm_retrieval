# roadmaps/

Corrective roadmaps for the evaluation stack — written *after* a harness exists
and has been run adversarially, so each one is anchored to an observed failure
rather than to a design review.

Distinct from [`../harness-v1.md`](../harness-v1.md), which is the original
design. Documents here say what to change and why, against a named commit of the
implementation.

| Document | Question it answers | Status |
|---|---|---|
| [harness-self-validation.md](harness-self-validation.md) | The harness measures the right things. Why did its first adversarial run report numbers that did not mean what their labels said? | implemented; §4.1 and §10 record what each step became |
| [discriminating-power.md](discriminating-power.md) | The labels are now honest. Why does the resulting headline table still not support the conclusion drawn from it? | implemented; §4.2 records what each step became, §6.1 the acceptance results, §11 where the proposal was wrong |

They are one progression, and the numbering is continuous because the defects
are:

| | Asks | Defects | Steps |
|---|---|---|---|
| [`../harness-v1.md`](../harness-v1.md) | what should be measured | gaps 1–10 | built |
| harness-self-validation.md | did the measurement run | D1–D5 | E1–E7, built |
| discriminating-power.md | could the measurement have separated anything | D6–D10 | E8–E14, built |

Each one is only visible once the one above it is closed. The second could not
be seen until runs stopped being mislabelled; the third could not be seen until
the numbers meant what they said, at which point the remaining question was
whether they meant anything.

The third one's own findings say the same thing a level down. Sweeping `k`
turned up four defects in retrieval larger than several of the five the previous
round fixed, and every one of them was invisible at the single `k` the previous
round reported. Whatever the fourth document is about, it is probably already
sitting inside a number that currently looks fine.

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
