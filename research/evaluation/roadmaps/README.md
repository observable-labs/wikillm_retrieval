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
| [representative-questions.md](representative-questions.md) | The numbers now separate the systems. Why do they separate them on a question class the product barely serves? | implemented; §4 and §4.1 record what each step became, §6 the acceptance results, §10 where the proposal was wrong |
| [instrument-coverage.md](instrument-coverage.md) | The questions are now the work. Can this harness see the system that is about to be built? | implemented; §9 records what each step became, §10 the acceptance results, §11 where the proposal was wrong |

They are one progression, and the numbering is continuous because the defects
are:

| | Asks | Defects | Steps |
|---|---|---|---|
| [`../harness-v1.md`](../harness-v1.md) | what should be measured | gaps 1–10 | built |
| harness-self-validation.md | did the measurement run | D1–D5 | E1–E7, built |
| discriminating-power.md | could the measurement have separated anything | D6–D10 | E8–E14, built |
| representative-questions.md | were the questions the work | D11–D15 | E15–E19 + R1–R3, built |
| *(ragharness work items)* | is any of it read against a scale that exists | D16–D34 | E20–E45, proposed |
| instrument-coverage.md | can it see what is about to be built | D35–D39 | E46–E56 + E21, built |

Each one is only visible once the one above it is closed. The second could not
be seen until runs stopped being mislabelled; the third could not be seen until
the numbers meant what they said, at which point the remaining question was
whether they meant anything.

**The last row breaks the pattern deliberately.** The first five are
retrospective — each asks something about a harness that has already run, and
each is settled by looking at what it produced. `instrument-coverage` is
prospective: it asks whether an instrument exists for a capability nobody has
built, which cannot be settled by looking at a run. Its one *observed* defect
(D35) is the anchor; the rest are absences, and its §0 says what that costs.

Building it converted four of those absences into measurements on the first run,
and the two that matter most point in opposite directions. A spoken follow-up
costs **0.63 recall** against the oracle rewrite on this corpus — the largest
single gap the series has measured, and it justifies a step the roadmap was
prepared to retire. And the shipped configuration returns the right document on
every intra-document question while pointing at the wrong *section* of it every
time, scoring **0.00 passage recall against the dense baseline's 1.00** — the
same defect D35 named, now with a number and a system that already beats it.

Which is the pattern holding rather than breaking: a prospective document's
value is not that its predictions were right. Two of them were wrong in the
first hour of building — the passage id space cannot be a chunk id, and the
token counts it called "already recorded" were recorded as zeros — and its §11
says so.

The third one's own findings said the same thing a level down. Sweeping `k`
turned up four defects in retrieval larger than several of the five the previous
round fixed, and every one of them was invisible at the single `k` the previous
round reported. The line closed with *"whatever the fourth document is about, it
is probably already sitting inside a number that currently looks fine."*

It was. The fourth is the first one the harness could not have caught: the
previous three are all defects a check inside the tool can detect, and *"these
questions do not resemble the work"* is not. It took a person looking at a
headline and finding it implausible — the observation that a keyword retriever
should not score 0.95 on questions a research assistant would be asked. Asked
without the corpus's vocabulary, on the same 78 documents, bm25 scores **0.02**.

Closing it moved the largest number in the series so far. The shipped
configuration went from 0.45 to 0.66 `recall@10` on that question class — level
with its own vector lane, where it had been 0.21 below — because a lexical lane
that has found nothing now abstains instead of voting. And the fourth document
found the same thing a level down twice over: the fixture's defining property
had been verified by a second copy of the measure that defines it, and
`ask()` had never taken the caller's retrieval configuration at all, so every
generation comparison between two lanes had been comparing one retriever with
itself.

Which leaves the fifth question already visible in the fourth's own results. R1
is worth 0.21 recall on the question class the product exists for, and a judge
comparing the same two configurations on corpus-level questions shows **no
difference at all**. Either the judge or the corpus is too small to see it, or
retrieval recall is not what decides whether an answer is good — and nothing in
four roadmaps of retrieval measurement can tell those apart.

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
