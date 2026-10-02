# precis — design

precis turns a book into study notes a reader can use to recall it after
reading: what it's about, its main ideas (or, for fiction, its themes), and
what it teaches. It's a standalone module with a JSON-file boundary: a
known-file in, a book JSON out. Work still in progress is tracked in
[v2-plan.md](v2-plan.md).

Every book gets the same treatment — no chapter walkthroughs. Chapter-level
material was the weak spot in grounding (the web describes books as a
whole: reviews, publisher copy, interviews, summaries) and the source of
most of the old pipeline's complexity.

## How the module is used

1. **Known-file creation** (`create-known-file`, ISBN → known-file). Looks
   the ISBN up on Open Library and fills in `title`, `author`, `year` and
   `page_count` where found — a placeholder where the title or author
   wasn't. No model, no search.
2. **The reader checks it**: corrects the title or author if needed, sets
   `kind`, and optionally adds `notes`.
3. **Generation** (`generate`, in Docker): research, write the notes, check
   them in code, and review them against the research.

## Module boundary

The module reads a known-file and writes a book JSON; nothing in its design
references a particular consumer. book-keeper is today's consumer and syncs
its own schema to precis's (`precis tags` exports the closed tag
vocabulary for that), but a different project could use precis unmodified.

## Docker boundary

Docker contains AI: anything that calls a model or feeds web content to
one — `generate`, `research`, `eval`. `create-known-file` (a lookup against
a well-defined public API) and the known-file preflight (local field
checks) run fine on a host.

## Input: known-file

```
isbn        required
title       required before generation (not a placeholder)
author      required before generation (not a placeholder)
year        optional
page_count  optional
kind        "fiction" | "non-fiction"
notes       optional — what matters to the reader about this book
book_file   optional — the reader's own DRM-free copy of the book: .epub,
            .pdf (with a text layer) or .txt, relative to the known-file
```

`notes` is a weighting signal for the notes, never quoted into them, and is
carried verbatim into the book's `reader_notes`.

`book_file` makes the book itself part of the research (`book_file.py`):
EPUB chapters in spine order, a PDF's text layer, or a text file, read
every run (not cached) and checked before any paid work — a missing file,
unknown type, a scanned PDF with no text or a DRM-locked copy fails the run.
A locked EPUB is one with Adobe's `rights.xml`, Apple's `sinf.xml`, or an
`encryption.xml` encrypting more than fonts (font obfuscation is common in
DRM-free EPUBs); its chapters are ciphertext that would otherwise pass the
length check as garbage. A locked PDF is one pypdf can't open without a key
or whose encryption handler is a DRM scheme's own; one encrypted only
against editing opens (pypdf's `crypto` extra reads its AES). precis never
removes DRM. It becomes source
`S1`, ahead of the web pages, shown as its opening and read whole by the
digest for non-fiction (fiction keeps the spoiler-safe opening); a copy of
the same book in the search results is dropped as a mirror of it, and the
research is never thin with it. Without it, a book is read whole only when
search happens to turn up a copy. `--book-file` overrides the path, for a
container where the known-file's host path doesn't exist. A 400-page book
adds ~$0.20-0.25 of digest, once: the digest is cached. The known-file's filename
stem is the book's slug: it names the research cache, and the consumer
names the output after it too.

## Output: book JSON

```
schema_version         "2"
title, author, year, isbn, page_count, kind    from the known-file
one_line_takeaway      one sentence
synopsis               2-5 paragraphs, fewer when there's less to say
ideas                  [{ title, summary, evidence, sources }]
                         non-fiction: key ideas; fiction: themes
                         as many as the book makes, each as long as it needs
key_claims_for_review  [{ prompt, answer }], non-fiction only, one per idea
                         a reader needs to remember
tags                   2-4 from the closed vocabulary for the kind
reader_notes           the known-file's notes, when given
warnings               anything the reader should check
```

- **Fiction is spoiler-safe** — premise and setup only, nothing past roughly
  the first act — because other people browse the library before reading
  the book. It has no review deck: reading fiction isn't about retaining
  claims.
- **`evidence`** is the study, story, example or figure the author uses
  (non-fiction), or the characters and situations that carry a theme
  (fiction). It makes the notes memorable and the grounding checkable.
  It's empty when there's no specific example to give: a required field
  got filled with general statements ("Storr examines…") whenever research
  was thin.
- **Never pad**: thin research means fewer ideas and shorter fields, not
  vaguer ones, and an idea that can only be stated in general terms is
  left out (write.py's rules).
- **No count to meet** (`IDEA_COUNT_RULE`, shared with the review): one
  idea per distinct point the book makes, whether three or thirty, and a
  book that numbers its own ideas (48 laws, 7 habits) gets its list, one
  idea each. Each idea is as long as it needs to be. The first library's
  5-12 range made the model pad to 11-12, restating a few points several
  times. Only notes with no ideas, or non-fiction without a review deck,
  are rejected.
- **`sources`** lists the research sources (`S1`, `S2`, …) behind an idea;
  empty means it rests on the model's own knowledge of the book.
- Fields with no value are absent, not null.

## Pipeline

```
known-file ─► research ─► digest ─► write ─► checks ─► review ─► validate ─► book JSON
               (cached)   (cached,             (code)   (one call)
                          non-fiction)
```

A plain async function (`generate.py`). The research and digest caches are
the only persisted state, so an interrupted run just starts again without
searching, or digesting, again.

### Research (`research.py`)

Three advanced searches run in parallel, with full page text: for
non-fiction, summary/key ideas, book review, author interview; for fiction,
themes/analysis, novel review, synopsis (not "plot summary", which gives
away the ending).

**Follow-ups for thin research:** when those come back with fewer than 5
pages naming both the book and its author, or under ~20k tokens of text in
all (The Integrity of the Personality's first searches found 18 pages but
~11k tokens of snippets), three wider searches run — the
book alone, a chapter summary and a publisher description (fiction: book
review, publisher description and literary criticism — never the book alone
or chapter summaries, which bring back the ending). Famous books
never need them; lesser-known ones, which lean hardest on the research, get
the extra credits. Pages naming the book but never its author don't count
as thin: that's a wrong author, which more searching won't fix.

Of the results:

- Only pages naming both the book and its author are kept, judged on the
  provider's matched excerpt rather than the whole page (a "best books"
  list names dozens).
- Duplicates go: the same page under another URL, and copies of one article
  (80%+ shared vocabulary).
- Each page is cleaned of navigation-sized lines and excerpted to ~6k
  tokens, cut from just before it first names the book; the total is capped
  at ~60k tokens. A lesser-known book's few pages are often its only
  detailed ones, so each gets room. Sources are numbered `S1`, `S2`, … in
  rank order, interleaved across the searches. A page whose excerpt lacks
  any of it keeps its whole raw text too (`full_text`, every line kept:
  cleaning drops short lines, which in an OCR'd book are the ends of
  sentences — so even an uncut page can need it), for the quote check to
  search and, when the excerpt cut it (`cut`), the digest to read.

The raw results, follow-ups included, are cached per slug
(`PRECIS_CACHE_DIR/research/`) and reused until `--fresh` or the
known-file's title, author or kind changes. A cache hit never searches:
it's the research exactly as first fetched, so runs comparing models write
from identical research. Follow-ups run only as part of a fresh search, and
the cache is written once every search, follow-ups included, has run.
Everything after the fetch is a pure function over the cache, so changing
it never needs a refetch. A search that fails keeps the others' results but
isn't cached; one that ran and found nothing is.

**Identity checks**, in code, before any model call: the run fails when no
page names the book (a mistyped or made-up title), or pages name it but none
names its author. `--trust-known` turns both into warnings. Research that ends thin — fewer
than two sources or under ~20k tokens — warns that the notes lean on the
model's own knowledge and will be general: the fallback when the web has
little on a book is general but true notes, flagged as such.

### Digest (`digest.py`)

The excerpt cut every long page to its opening: *A Way of Being*'s research
held the whole book (670k characters) and the notes saw 4% of it, filling
the rest from summary sites — misattributions and missing chapters
included. Long pages are common and the most detailed ones a book has: a
copy of the book, chapter-by-chapter summaries, interview transcripts.

A non-fiction page the excerpt would cut by more than half (over ~12k
tokens) is read whole instead: split into ~15k-token chunks at line
breaks, and one structured call per chunk notes its kind (the book's own
text, writing about the book, or other — front and back matter,
navigation), the section it sits under, what it says in its own terms with
its examples, names and figures, and up to five of the author's sentences
copied exactly. Those notes, in page order, replace the excerpt as the
source's text; the source keeps them as `parts`. A page
that's mostly book text is the book's own text (`is_book_text`). Shorter
cut pages keep their excerpt, which holds most of them verbatim.

Fiction is never digested: a novel's full text holds the ending, and its
opening excerpt is the spoiler-safe part.

Each chunk's notes are cached per slug (`PRECIS_CACHE_DIR/digests/`),
keyed by everything its call depends on — the chunk and its place in the
page, the book's title and author, the model and `DIGEST_VERSION` — so a
rerun from the same research pays nothing; the cache keeps only the
research's current chunks. Every chunk settles before anything is decided:
a page with a failed chunk keeps its excerpt with a warning (it costs
detail, not the run), its other chunks stay cached so a retry pays only for
the failed one, and an unexpected error fails the run only after what
succeeded is saved.

Cost and prompt size are capped: at most 20 chunks of one page (~1.2M
characters, a long book) and 30 in all, in rank order — a page past either
cap is noted from its start, and its notes say so. A page's notes are cut
at 90k characters, room for all 20 parts' notes: a safety net for runaway
replies. A book's full text costs ~$0.10 per 300k characters
with Haiku 4.5; the caps hold a book to ~$0.80 of digests at worst.

### Write (`write.py`)

One structured call writes every field. The system message holds a
task-independent preamble, the book and its research, marked for prompt
caching; the task's instructions go in the user message. Both calls send
the same tool list too (the write's and the review's tools, each call
forced to its own), since a provider's cache prefix runs tools → system →
messages: that's what should let the review read the research from the
cache. Not yet confirmed through OpenRouter — the baseline eval run checks
the review call's `cached_tokens` (docs/v2-plan.md).

- Ideas cover the whole book, as many as it makes. They name the book's
  own terms and never describe the text ("the author discusses…").
- Key claims ask what a reader needs to recall about an idea, never just
  "What is <the idea's title>?"; the review fixes any that do.
- Accuracy comes first: never invent a study, figure, quote, name or event;
  describe a detail more generally when unsure of it.
- **The notes report the book, not its critics or the author's other
  books** (`FAITHFUL_RULE`, shared with the review): claims critics dispute
  stay as the author argues them, and critics' views stay out. The research
  holds reviews and critiques, and pages about the author's other books (a
  publisher's page for another title, an author profile): only what a page
  says about this book counts. *The Integrity of the Personality*'s first
  notes took *Solitude*'s thesis from *Solitude*'s publisher page.
- **Fiction's spoiler line** (`SPOILER_RULE`, shared with the review):
  setup — the world, the main characters' situations, work and
  relationships, the conflicts the opening establishes — is safe and should
  be specific; spoilers are what a reader only learns later. Unsure about a
  later detail means leaving it out, never blurring the setup. A theme is
  stated as the setup raises it, never as the story resolves it: stating how
  themes play out gave away *The Island of Dr. Moreau*'s second half.

The response is validated as it arrives — at least one idea, a review deck
for non-fiction and none for fiction, 2-4 tags from the closed vocabulary —
and a retry after a failure shows the model its rejected call and the
error. Retries are for output that can't be used, and what can be left
out is left out instead, since a failed retry loses the whole run: tags
outside the vocabulary, repeated, or past the fourth (only fewer than two
usable ones are sent back), and citations of sources the research doesn't
have.

**Identity backstop:** a real but wrong author named next to the book on
some page (a comparison, a reading list) passes research's code checks, so
the model reports when the research credits the book to someone else. That
fails the run unless `--trust-known`. Another form of the same name, a pen
name and the real name, or an added co-author isn't a mismatch.

### Checks (`checks.py`)

Code checks on the written notes, each a specific lead for the review:
ideas, evidence or answers that describe the text ("the book examines…",
"Storr traces…") instead of stating the idea or giving the book's example,
and near-duplicate ideas (half their vocabulary shared — the same point in
other words is the review's `same_point`). A general
statement of what the book argues is a fine fallback when research is
thin; a description of the text isn't. The author's surname counts as a
subject in summaries and answers, not in evidence, where "Krakauer
describes his 1977 climb of Devils Thumb" is the example itself. Evidence
is flagged only when it opens by describing the text ("The book examines
ten firms."): partway through, "the book includes the letters Krakauer
received" is the example too.

**Quotes** are checked against the research, in code: every quoted span
of four or more words in the notes must appear in some source's whole page,
compared as bare letters (so punctuation, quote style and line-break
hyphens don't matter) and in 20-letter pieces of which 70% must be found
(so OCR damage, a page header mid-sentence or one changed word don't
either). Single quotes count — Haiku quotes in them inside its JSON — and
quote marks only open or close at a word's edge, so an apostrophe
("man's") or an inch mark never pairs with a real quote. Titles in quote marks (every word but the small ones capitalized)
aren't quotes. When the research
holds the whole book's own text — a book-text page of at least 100k
characters, since a shorter one may be a preview of a few chapters — a
quote found only in pages about the book is flagged too: *The Undiscovered Self*'s notes
quoted Jung from another essay that a summary site credited to this book.
The review keeps a flagged quote only when it's sure of the exact words,
and paraphrases it otherwise — a paraphrase of a real quote loses nothing.

**Long quotes** are flagged with or without research: one over 300
characters, or over 2,000 quoted in all. The notes are published, so they
state a book's ideas in their own words and quote a line only where the
exact words matter — never enough of the book to stand in for it. The write
prompt says so; the review cuts a flagged quote to its line or paraphrases
it. Every library and eval book so far sits well under both (longest quote
~200 characters, most in one book ~1,200).

No coverage check: a test that flagged parts of a digested book no idea drew
on led the review to add ideas repeating ones the notes had, and the notes
aim at a book's key ideas, not a chapter-by-chapter account.

No other check matches facts against the research. The first baseline run had
one — numbers and proper nouns in an idea's evidence that weren't in its
cited sources — and it caught no inventions across 8 books, while its
flags (Tolstoy in *Into the Wild*, Wickham in *Pride and Prejudice*, all
correct) led the review to strip correct details. Tens of thousands of tokens of excerpts
can't hold everything, so "not in the research" isn't "invented".
Accuracy is the review's job, and the evals' judge measures it.

### Review (`review.py`)

One structured call over the same cached prefix, given the notes, the check
findings and the count rule, returns only what needs changing: a verdict
per idea (titled, so a skipped one can't shift the rest), new ideas, and the
whole corrected takeaway, synopsis or claims list where one needs fixing.
Empty or placeholder fields mean "nothing to fix". For fiction it also
audits every field for spoilers, with the write prompt's guards.

The review judges whether the notes are faithful to the book, not whether
the book is right: a critic disputing the author is never a reason to
change an idea. And the research's silence isn't contradiction — the
excerpts can't hold everything, so a specific the model is
confident is from the book stays; the reader, who has read the book, is
the final check. The first baseline's review broke both: it dropped a
Sapiens idea because critics dispute Harari, and made correct details
vaguer because the research didn't mention them.

- **keep** — accurate to the book, citations included: its cited sources
  support it, or it cites none and the model is confident it's accurate.
- **revise** — the right idea with something wrong: a detail that misstates
  the book (corrected, or that detail alone removed), vagueness, a
  description of the text, or a citation that doesn't support it
  (corrected, or emptied when the research is silent but the idea is
  right). Every specific that's right is kept: fix what's wrong, never make
  the idea vaguer.
- **drop** — the book doesn't make this argument, it isn't specific to this
  book, or it repeats another idea. A wrong idea is worse than a missing
  one; claims resting on it go too.
- **same point** — first, before any verdict: pairs of ideas a reader
  would recall as one — the same claim from different angles, or a
  framework and one of its own parts given separately — each naming the
  idea to keep (revised to take in what the other adds) and the one to
  drop. The pair decides, whatever the dropped idea's verdict says. This is
  the model's job because the word-overlap check can't do it: the pairs in
  the first library books shared 3-17% of their words, while distinct ideas
  in the same books shared up to 17.5%.
- **new ideas** — major ideas the book makes that the notes miss, or
  replacements for dropped ones; they must cite the research. An idea is
  something the book argues (or a theme a novel develops), not an
  observation about the book or its reception.

The review is validated by building the book it would produce, by the
book's own rules: no ideas left, or a verdict naming no idea or a second
verdict for one, is sent back with the reason. What can be left out
instead is, since a retry that repeats it would lose the whole review, its
drops of wrong ideas included: a new idea repeating one the notes keep (the
near-duplicate check) or citing nothing, a `same_point` pair that isn't two
of the ideas (or would drop both), a citation of a source the
research doesn't have, a half-returned synopsis (the original stays), and
fiction's key claims. A drop with the key claims left as they were keeps
them, with a warning to check none rests on the dropped idea; an added
idea with the claims left as they were is warned about too, since it has
no claim. After the review, a deck covering under half the ideas is a
warning: there's no count to meet, but most of the notes couldn't be
reviewed. An idea with
no verdict is kept as written. If it still can't fix it, or the
call fails outright, the run keeps the written notes (already paid for),
with a warning that they're unreviewed and the check findings as warnings.
A revised idea that leaves out its sources keeps the
original's. Afterwards the checks run again, and whatever they still find
is a warning for the reader, beside one naming the uncited ideas (and a
book-level one when most are uncited).

## CLI contract

```
precis create-known-file <isbn>... [--kind fiction|non-fiction] [--output <path> | --output-dir <dir>] [--force]
    → looks each ISBN up on Open Library and writes a known-file. More than
      one ISBN needs --output-dir (files named after the title slug) and no
      --kind (it defaults to non-fiction, to correct by hand). An existing
      known-file (likely hand-edited) is only replaced with --force; a batch
      skips it and writes the rest.

precis generate <known-file.json> [--output <path>] [--trust-known] [--fresh] [--book-file <path>]
    → researches the book and writes its notes. Refuses a known-file that
      isn't ready, an --output it can't write, or a book file it can't
      read, before any paid work. If
      the final write fails anyway, the book goes to stdout rather than
      being lost.

precis research <known-file.json> [--trust-known] [--fresh] [--book-file <path>]
    → the research step alone: prints the rendered research to stdout,
      sources and warnings to stderr. Free: long pages show as their
      excerpts, since digesting them is paid work.

precis tags [--output <path>]
    → the closed tag vocabulary as JSON, for a consumer to sync against.

precis eval run <label> [--book <slug>]... [--trust-known] [--fresh]
    → generates every eval book (evals/) into evals/runs/<label>/: each
      book's JSON plus <slug>.metrics.json — measured cost (the gateway's
      usage.cost), searches and their credits, duration, idea and claim counts,
      warnings, duplicate-idea rate, citation coverage. A book already
      there is skipped, so a rerun finishes a partial run without paying
      twice. Research is cached per book, not per run, so runs comparing
      models write from the same research; each book's metrics record the
      research's fingerprint.

precis eval judge <candidate> <baseline> [--judge-model <id>] [--book <slug>]...
    → pairwise judgement of two runs by PRECIS_JUDGE_MODEL against a rubric
      (accuracy, specificity, distinctness, coverage vs the reference
      summary, plus review-deck quality for non-fiction or spoiler safety
      for fiction). Each book is judged twice with the order swapped; a pick
      counts only when both orders agree. Writes
      evals/runs/<candidate>/judge-vs-<baseline>.json; the score is 1 per
      win, 0.5 per tie, so above 0.5 beats the baseline. Books whose two
      runs wrote from different research (a failed search isn't cached, and
      --fresh refetches) are listed with a warning. A book that fails to
      judge is recorded under `failed` and left out of the score; the rest
      are still written.
```

Errors go to stderr with a non-zero exit, never a traceback. Progress —
including every retry of the write or review call and why (a rejected
attempt's validation error, a provider error, or the HTTP client's own
retry of a timeout, rate limit or 5xx) — and a closing `usage:` line (LLM calls, tokens, cost, searches — printed on
failure too, since the money is spent either way) go to stderr, so stdout
carries only the command's output.

## Infrastructure decisions

- **LLM access:** an OpenAI-compatible gateway (OpenRouter) — base URL, key
  and model from env vars, with the `openai` package used only as an HTTP
  client. Evals choose the model (Haiku 4.5 vs Sonnet 5). Structured output
  uses a forced `tool_choice`, which rules out models that reject it.
  `llm.py` retries HTTP-level failures through the SDK, provider errors
  OpenRouter returns inside a 200 itself, and a structured call that
  doesn't call the tool or doesn't validate. Any other HTTP error (a 400
  for a context that's too long, a 402 for credits) is a provider error,
  so a review that hits one still leaves the written notes.
- **Search:** Tavily, behind a `SearchClient` protocol. Results are
  untrusted content: the rendered research frames them as reference data,
  not instructions, and a page can't close or open a source tag.
- **Timeouts:** each LLM call has a ~2 minute timeout
  (`PRECIS_LLM_CALL_TIMEOUT_SECONDS`), a circuit breaker for a hung call;
  the write and review calls set 10 minutes and fewer HTTP retries, since
  a book with dozens of ideas is ~20k tokens of output. There's no
  whole-run limit. A stalled provider costs about 30 minutes a call (three
  10-minute tries); through every retry layer (2 attempts × 4
  requests for provider errors × 3 HTTP tries) the bound is hours.
- **Cost:** every run reports the gateway-reported cost of its calls.
  Target: under $0.50 a book on average, measured on the eval set.
  Searches aren't priced: they run on Tavily's free tier (1,000 credits a
  month; a book's 3 advanced searches use 6), and runs report the credits
  used against it.
- **Docker:** see the Docker boundary above. The image runs as a non-root
  user and writes only to `/output`, `/data` (the research cache, on a
  named volume) and `/tmp`; README.md has the `docker run` hardening flags.

## Out of scope

- A storage layer — generation's contract (JSON in, JSON out) leaves storage
  to the consumer.
- Consumer integration (book-keeper's schema and publishing tooling) —
  follow-up work on the consumer's side.
