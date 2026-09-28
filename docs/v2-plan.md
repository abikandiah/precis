# precis v2 — remaining work

Decided 2026-09-26, rescoped 2026-09-27 to whole-book notes with no
chapters. What's built is described in [blueprint.md](blueprint.md); this
file tracks what isn't. Tick the boxes as phases land and fold each into
the blueprint.

## Targets

- **Cost:** under **$0.50 per book** on average, measured on the eval set.
  Expected well below: 2–3 LLM calls and 3 searches per book.
- **Quality:** a reader who finished the book can recall what it said from
  the notes alone — judged against the eval set's references.

## Decisions still in force

- **Paid model.** Haiku 4.5 vs Sonnet 5 is chosen by evals. Forced
  `tool_choice` rules out Opus 5.5 / Fable 5.1 (they 400 on it) unless we
  move to `tool_choice: auto` + strict schemas.
- **Evals before exemplars.** Exemplars only if evals show a gap a worked
  example would close, and exemplar books must never be eval books.
- **The first full eval run is the baseline**; later runs are judged
  pairwise against it, with the references as the coverage guide.
- **Paid runs are confirmed first** — an eval run spends real money.

## Status

- [x] **Plan, evals, research, write, switch.** Eval harness and set,
  `research.py`, `write.py`, schema v2, and v1 deleted (LangGraph,
  checkpoints, chapters, table-of-contents parsing). `generate` runs
  research → write.
- [x] **Checks + review** — `checks.py`, `review.py`, wired into
  `generate`.
- [x] **Baseline** — `evals/runs/haiku-4-5`: $0.087 a book, OpenRouter
  reports `usage.cost`, and the review call reads the research from the
  prompt cache.
- [ ] **Tune** — done so far, compared by reading the runs: review
  faithfulness and spoiler rules (`haiku-4-5-v2`), bigger research pages,
  follow-up searches for thin research, and two lesser-known eval books
  (`haiku-4-5-research`, $0.09 a book). Left: Haiku vs Sonnet, above all
  on the lesser-known books, where the model's own knowledge is the
  ceiling.
- [ ] **book-keeper on schema v2** — book-keeper's `schema.ts` is still v1
  (`chapters`, `parts`, no `ideas`), so it can't show v2's notes. Needed
  before generating the library.
- [ ] **Exemplars** — only if evals show a gap.
