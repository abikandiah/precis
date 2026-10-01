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

- **Paid model: Haiku 4.5** (decided 2026-09-29 from the eval runs below;
  Sonnet 5 for lesser-known books shelved 2026-10-01 — at scale it costs
  2-3x Haiku, and Claude's review covers the gap for now). Forced
  `tool_choice` rules out Opus 5.5 / Fable 5.1 (they 400 on it) unless we
  move to `tool_choice: auto` + strict schemas.
- **Library books get a Claude review before they're committed**
  (2026-10-01): precis + Haiku generates, Claude checks each book against
  its research (full text where the research has it) and fixes it, the
  reader signs off with `verified`. The goal is a pipeline good enough to
  drop that step: recurring fixes go back into precis.
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
- [x] **Tune** — compared by reading the runs, not the judge: review
  faithfulness and spoiler rules (`haiku-4-5-v2`), bigger research pages,
  follow-up searches for thin research, and two lesser-known eval books
  (`haiku-4-5-research`, $0.09 a book). Model comparison, all from the same
  cached research:
  - **Haiku 4.5** (`haiku-4-5-research`, all 10 books): good on
    well-known books and on *A Month in the Country*; vague and
    text-describing on *The Integrity of the Personality*, where research
    is thin (~11k tokens) and its own knowledge runs out. ~56s a book.
  - **Sonnet 5** (`sonnet-5`, the two lesser-known books): the best of the
    three — the fullest, most specific *Month in the Country*, and on Storr
    it states the book's arguments from its own knowledge where the others
    couldn't. Still adds a solitude idea that likely belongs to Storr's
    later *Solitude*. $0.21 a book measured.
  - **Free: `nvidia/nemotron-3-ultra-550b-a55b:free`**
    (`nemotron-ultra-free`, four books): content close to Haiku, sometimes
    more specific (good-to-great, *A Month in the Country*), but synopses
    far short of what's asked (500-1,000 characters against Haiku's
    2,000-2,900), fewer ideas, ~6x slower (336s a book), and flaky — 504s
    and a reply written as text instead of a tool call, which retries
    recovered. On Storr it grounded ideas in the sources but described the
    sources more than the book, and it made the same *Solitude* confusion.
    A workable zero-cost fallback, not a default: it saves ~$9 per 100
    books.
- [x] **Read long pages whole** (2026-10-01) — `digest.py`, after
  reviewing the first five library books by hand. Every page was cut to
  ~6k tokens, so books whose full text was in the research (three of five)
  were noted from their first chapters plus summary sites: chapters
  missing, evidence vague, a quote from another of Jung's essays. Long
  non-fiction pages are now digested chunk by chunk; a code check flags
  quotes not in the research (or only in pages about a book whose own text
  is there); evidence may be empty and the write prompt says never pad.
  Rerun of the three full-text books from cached research ($0.66 in all,
  $0.13-0.37 a book): far more specific and complete than before, close to
  the hand-reviewed versions, with one made-up example left for the
  Claude review to catch. Kept simple on purpose: the notes are a book's
  key ideas, not a chapter-by-chapter account, so no coverage check and no
  per-chapter digests (a coverage check tried here led the review to pad).
- [ ] **Sonnet's empty first attempt** (low priority while Sonnet is shelved) — on 3 of its 4 calls, Sonnet 5's
  first tool call through OpenRouter came back with no arguments (every
  field "Field required") and the retry succeeded, roughly doubling those
  calls' cost. Haiku and the free model never did it. Worth finding the
  cause before Sonnet is used for more than a few books.
- [x] **book-keeper on schema v2** — `schema.ts` takes only v2
  (`schema_version: "2"`), the book page shows ideas in place of chapters,
  `fix-chapter` is gone, and the 24 v1 books became v2 known-files to
  regenerate from.
- [ ] **Generate the library** — the 24 known-files in book-keeper's
  `known/`, with Haiku 4.5; Sonnet 5 for any that warn of thin research.
- [ ] **Exemplars** — only if evals show a gap.
