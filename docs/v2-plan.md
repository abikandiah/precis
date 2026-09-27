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
- [ ] **Baseline** — the first full eval run becomes the baseline; confirm
  < $0.50 a book, that OpenRouter reports `usage.cost`, and that
  `cache_control` reaches Anthropic (`cached_tokens` on the review call).
- [ ] **Tune** — Haiku vs Sonnet and prompt variants, judged pairwise
  against the baseline.
- [ ] **Exemplars** — only if evals show a gap.
