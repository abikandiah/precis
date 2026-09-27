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
- [ ] **Checks + review** — below. Then the first eval run becomes the
  baseline; confirm < $0.50, that OpenRouter reports `usage.cost`, and that
  `cache_control` reaches Anthropic (`cached_tokens` on the review call).
- [ ] **Tune** — Haiku vs Sonnet and prompt variants, judged pairwise
  against the baseline.
- [ ] **Exemplars** — only if evals show a gap.

## Checks + review

```
research ─► write ─► checks ─► review ─► validate
                                 (revise only if review finds issues)
```

- **Code checks** on the written notes: meta-descriptions ("the book
  examines…", "the author discusses…"), near-duplicate ideas (word overlap),
  key claims that restate an idea verbatim, and **evidence specifics** —
  numbers and proper nouns in an idea's `evidence` (study names, figures,
  people, dates: where invention hides) must appear in the sources it cites.
- **One review call** against the research, sharing the write call's cached
  prefix (`write.context_messages`), given the code-check findings as
  specific issues: unsupported claims, vague or generic ideas, major ideas
  the research covers that the notes miss, and — for fiction — a spoiler
  audit. Returns revised fields only; no revision means no second write.
- **An idea the research can't validate:**
  - *Supported* — a cited source backs it: kept.
  - *Unsupported, not contradicted* — the research is silent: kept with
    empty `sources` and a per-idea warning. ~30k tokens of research can't
    cover everything, and the model knows well-known books; the reader, who
    has read the book, is the final check via book-keeper's `verified`
    step.
  - *Contradicted, or not placeable in this book* (generic enough to be from
    any book on the topic): dropped, and replaced with a supported idea if
    the research has one. A wrong idea is worse than a missing one.
  - Key claims follow the same rules; a claim resting on a dropped idea is
    dropped with it.
- Fewer ideas left than the minimum (5 non-fiction, 3 fiction) fails the run
  with a clear message. More than half the ideas uncited adds a book-level
  warning that the research was thin.
