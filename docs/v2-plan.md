# precis v2 — remaining work

Decided 2026-09-26, rescoped 2026-09-27 to whole-book notes with no
chapters, split into two modes on 2026-10-02. What's built is described in
[blueprint.md](blueprint.md); this file holds the decisions steering what's
left, and what's left. Git history has the phases that got here.

## Targets

- **Cost:** under **$0.50 per book** on average, measured on library runs.
  Expected well below: an overview is 2–3 LLM calls and 3 searches; full
  notes add the digest of the whole book (~$0.15–0.35 in all, by length)
  and drop the search.
- **Quality:** a reader who finished the book can recall what it said from
  full notes alone, and knows what it's about from an overview — judged by
  the Claude review of each library book.

## Decisions still in force

- **Paid model: Haiku 4.5** (2026-09-29). From the eval runs, all from the
  same research: Haiku was good on well-known books and vague where
  research was thin, at ~$0.09 a book. Sonnet 5 wrote the fullest notes on
  lesser-known books at ~2-3x the cost, and was shelved (2026-10-01) since
  Claude's review covers the gap. The best free model
  (`nvidia/nemotron-3-ultra-550b-a55b:free`) came close on content but
  wrote short synopses, ran ~6x slower and failed intermittently, and
  `data_collection: "deny"` now rules most free models out. Forced
  `tool_choice` rules out Opus 5.5 / Fable 5.1 (they 400 on it) unless we
  move to `tool_choice: auto` + strict schemas.
- **Library books get a Claude review before they're committed**
  (2026-10-01): precis + Haiku generates, Claude checks each book against
  its source — the reader's copy for full notes, the research for an
  overview, with a lighter sanity check — and fixes it; the reader signs
  off with `verified`. The goal is a pipeline good enough to drop that
  step: recurring fixes go back into precis.
- **Live generations, no evals** (2026-10-02): new work is judged on the
  library books it generates, through the Claude review. The eval harness
  is gone (git history has it and its runs): runs cost generation too, it
  couldn't exercise full mode (no book files), and it judged overviews by
  rules they no longer follow.
- **Paid runs are confirmed first** — every generation spends real money.

## Remaining

- [ ] **Generate the library** — the 28 known-files in book-keeper's
  `known/`, with Haiku 4.5: full notes where there's a `book_file`,
  overviews elsewhere; each gets the Claude review before it's committed.
  None has a `book_file` yet: the reader adds them for the books they own
  first, since upgrading an overview later pays for the book twice. The
  site is empty until then — the five books generated from search came
  off it with the switch to schema v3 — so book-keeper's `main` isn't
  pushed before the first reviewed books land.
