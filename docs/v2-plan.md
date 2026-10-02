# precis v2 — remaining work

Decided 2026-09-26, rescoped 2026-09-27 to whole-book notes with no
chapters. What's built is described in [blueprint.md](blueprint.md); this
file tracks what isn't. Tick the boxes as phases land and fold each into
the blueprint.

## Targets

- **Cost:** under **$0.50 per book** on average, measured on library runs.
  Expected well below: an overview is 2–3 LLM calls and 3 searches; full
  notes add the digest of the whole book (~$0.15–0.35 in all, by length)
  and drop the search.
- **Quality:** a reader who finished the book can recall what it said from
  full notes alone, and knows what it's about from an overview — judged by
  the Claude review of each library book.

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
- **Live generations, no evals** (2026-10-02): new work is judged on the
  library books it generates, through the Claude review. The eval harness
  (`precis eval`, `evals/`) is removed: runs cost generation too, it can't
  exercise full mode (no book files), and it judged overviews by rules they
  no longer follow. Its findings are recorded below; git history has the
  harness and its runs.
- **Exemplars** only if the reviews show a gap a worked example would
  close, and an exemplar book is never one the notes are judged on.
- **Paid runs are confirmed first** — every generation spends real money.

## Two modes (decided 2026-10-02)

Search can't carry thorough notes: they're as good as what search turns up,
which is choppy — fine for famous books, where the model's own knowledge
fills in, thin and error-prone for the rest. The reviews found its errors
come from pages *about* the book: a quote from another of Jung's essays
that a summary site credited to *The Undiscovered Self*, ideas from
Storr's later *Solitude*. So precis splits by source, decided by the
known-file alone — a `book_file` means full notes, none means an overview.

**Full mode — the reader's own copy (`book_file`).**
- No search: the book is the research. Title, author, year and ISBN
  already come from Open Library through make-known.
- The digest reads the whole book; the write states its key ideas, each
  with the book's own example; quotes are checked against the text.
- Each idea names where in the book it comes from (a chapter or part) —
  the digest reads in page order, so it's nearly free — for the reader to
  go back to, and for the review to check the right passage. A pointer,
  not a chapter-by-chapter account.
- Key claims for review (flashcards) as today.
- The Claude review checks it against the text.

**Overview mode — no `book_file`.**
- Search as today, minus full texts from anywhere but legitimately free
  sources (below).
- Takeaway, synopsis, and a few headline ideas (title and summary, no
  evidence) where the research supports them; never padded to a count.
- No key claims: flashcards on claims the research can't back teach the
  wrong thing.
- A light Claude review: a sanity check, not a rebuild.
- `--overview` forces it on a known-file with a `book_file`, for a quick one.

**Both modes.**
- **No provider keeps or trains on the prompts**: OpenRouter's
  `provider: {"data_collection": "deny"}` on every call. Free on paid
  Haiku; it rules out most free models, which are shelved anyway.
- **Full texts only from legitimately free sources**: a search page the
  digest finds to be the book itself is kept only from Project Gutenberg,
  Standard Ebooks, Wikisource and open-access repositories (DOAB, OAPEN) —
  a publisher's own free edition can join the list when one turns up;
  anything else is dropped, not cut to an excerpt, and logged in the
  research warnings. archive.org stays off the
  list: its scans of in-copyright books are what *Hachette v. Internet
  Archive* ruled against, and its public-domain books are on Gutenberg.
  In full mode this never comes up — there's no search.
- **Schema:** a `depth: "full" | "overview"` field.

**book-keeper.** An overview page says so — a short note that it's written
from published sources online, not the book itself; full notes carry no
label. Overviews have no review banner, and with few ideas, no idea index.
Adding a `book_file` and regenerating turns an overview into full notes.

**Fiction in full mode** reads the whole book, not only the opening as
today: the target is a reader who finished it recalling it. The ending's
resolution goes behind a spoiler toggle on the page, which keeps the
library spoiler-safe for other people viewing it — the reason fiction is
spoiler-safe at all.

**The five books generated so far** (from search, three with a full text it
found online) are the reader's to deal with later, as are the library's
`book_file`s.

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
- [x] **Your own copy of the book** (2026-10-01) — the known-file's
  `book_file` (`.epub`, `.pdf`, `.txt`) becomes research source S1 and is
  read whole by the digest, so a book doesn't depend on search turning up
  a copy. book-keeper's `generate.sh` mounts it into the container.
- [ ] **Sonnet's empty first attempt** (low priority while Sonnet is shelved) — on 3 of its 4 calls, Sonnet 5's
  first tool call through OpenRouter came back with no arguments (every
  field "Field required") and the retry succeeded, roughly doubling those
  calls' cost. Haiku and the free model never did it. Worth finding the
  cause before Sonnet is used for more than a few books.
- [x] **book-keeper on schema v2** — `schema.ts` takes only v2
  (`schema_version: "2"`), the book page shows ideas in place of chapters,
  `fix-chapter` is gone, and the 24 v1 books became v2 known-files to
  regenerate from.
- [x] **Two modes: both-mode guards** (2026-10-02) — `data_collection:
  "deny"` on every call, and `digest.py`'s `FREE_HOSTS`: a long page from
  any other site is screened by its first two chunks and dropped if it's
  the book's own text, fiction included.
- [x] **Two modes: full mode** (2026-10-02) — schema v3: `depth`, ideas'
  `where`, fiction's `resolution`. With a `book_file`, research is the book
  alone (no search), the digest reads it whole — fiction too, with its own
  notes rule and up to 40 chunks — and the write and review prompts write
  from it, point full non-fiction ideas into the book, and keep a novel's
  ending in `resolution`. book-keeper still takes only v2 until its phase
  lands.
- [x] **Two modes: overview mode** (2026-10-02) — headline ideas
  (`OVERVIEW_COUNT_RULE`) with no evidence or where — the overview's write
  and review tools don't offer them — no key claims for any overview
  (`has_deck`: full non-fiction only), a lighter review count, and
  `--overview` to force one over a known-file's `book_file`. The eval
  harness went with it.
- [ ] **Two modes: book-keeper** — schema v3 (`depth`, `resolution`, and
  ideas' `where`, which precis writes as `""` when empty, like `evidence`),
  the overview note, no banner or index on overviews, the spoiler toggle.
- [ ] **Generate the library** — the known-files in book-keeper's `known/`,
  with Haiku 4.5: full notes where there's a `book_file`, overviews
  elsewhere; each gets the Claude review before it's committed.
- [ ] **Exemplars** — only if the reviews show a gap.
