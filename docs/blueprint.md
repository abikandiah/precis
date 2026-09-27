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
3. **Generation** (`generate`, in Docker): research, then write the notes.

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
```

`notes` is a weighting signal for the notes, never quoted into them, and is
carried verbatim into the book's `reader_notes`. The known-file's filename
stem is the book's slug: it names the research cache, and the consumer
names the output after it too.

## Output: book JSON

```
schema_version         "2"
title, author, year, isbn, page_count, kind    from the known-file
one_line_takeaway      one sentence
synopsis               3-5 paragraphs
ideas                  [{ title, summary, evidence, sources }]
                         non-fiction: 5-12 key ideas
                         fiction: 3-6 themes
key_claims_for_review  [{ prompt, answer }], non-fiction only, 5-15
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
- **`sources`** lists the research sources (`S1`, `S2`, …) behind an idea;
  empty means it rests on the model's own knowledge of the book.
- Fields with no value are absent, not null.

## Pipeline

```
known-file ─► research ─► write ─► validate ─► book JSON
               (cached)
```

A plain async function (`generate.py`). The research cache is the only
persisted state, so an interrupted run just starts again without searching
again.

### Research (`research.py`)

Three advanced searches run in parallel, with full page text: for
non-fiction, summary/key ideas, book review, author interview; for fiction,
themes/analysis, novel review, synopsis (not "plot summary", which gives
away the ending). Of the results:

- Only pages naming both the book and its author are kept, judged on the
  provider's matched excerpt rather than the whole page (a "best books"
  list names dozens).
- Duplicates go: the same page under another URL, and copies of one article
  (80%+ shared vocabulary).
- Each page is cleaned of navigation-sized lines and excerpted to ~3k
  tokens, cut from just before it first names the book; the total is capped
  at ~30k tokens. Sources are numbered `S1`, `S2`, … in rank order,
  interleaved across the three searches.

The raw results are cached per slug (`PRECIS_CACHE_DIR/research/`) and
reused until `--fresh` or the known-file's title, author or kind changes.
Everything after the fetch is a pure function over the cache, so changing
it never needs a refetch. A search that fails keeps the others' results but
isn't cached; one that ran and found nothing is.

**Identity checks**, in code, before any model call: the run fails when no
page names the book (a mistyped or made-up title), or pages name it but none
names its author. `--trust-known` turns both into warnings. Fewer than two
sources warns that the notes lean on the model's own knowledge.

### Write (`write.py`)

One structured call writes every field. The system message holds a
task-independent preamble, the book and its research, marked for prompt
caching, so a later call about the same book (the review, Phase 4) reuses
the cached research; the task's instructions go in the user message.

- Ideas cover the whole book, and their count scales with how much it
  argues. They name the book's own terms and never describe the text ("the
  author discusses…").
- Accuracy comes first: never invent a study, figure, quote, name or event;
  describe a detail more generally when unsure of it.
- Fiction's instructions repeat the spoiler rule and warn that the research
  contains spoilers.

The response is validated as it arrives — counts per kind, tags from the
closed vocabulary, citation IDs that exist in the research — and a retry
after a failure shows the model its rejected call and the error.

**Identity backstop:** a real but wrong author named next to the book on
some page (a comparison, a reading list) passes research's code checks, so
the model reports when the research credits the book to someone else. That
fails the run unless `--trust-known`. Another form of the same name, a pen
name and the real name, or an added co-author isn't a mismatch.

## CLI contract

```
precis create-known-file <isbn>... [--kind fiction|non-fiction] [--output <path> | --output-dir <dir>]
    → looks each ISBN up on Open Library and writes a known-file. More than
      one ISBN needs --output-dir (files named after the title slug) and no
      --kind (it defaults to non-fiction, to correct by hand).

precis generate <known-file.json> [--output <path>] [--trust-known] [--fresh]
    → researches the book and writes its notes. Refuses a known-file that
      isn't ready, or an --output it can't write, before any paid work. If
      the final write fails anyway, the book goes to stdout rather than
      being lost.

precis research <known-file.json> [--trust-known] [--fresh]
    → the research step alone: prints the rendered research to stdout,
      sources and warnings to stderr.

precis tags [--output <path>]
    → the closed tag vocabulary as JSON, for a consumer to sync against.

precis eval run <label> [--book <slug>]... [--trust-known] [--fresh]
    → generates every eval book (evals/) into evals/runs/<label>/: each
      book's JSON plus <slug>.metrics.json — measured cost (the gateway's
      usage.cost plus search credits), duration, idea and claim counts,
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
      --fresh refetches) are listed with a warning.
```

Errors go to stderr with a non-zero exit, never a traceback. Progress and a
closing `usage:` line (LLM calls, tokens, cost, searches — printed on
failure too, since the money is spent either way) go to stderr, so stdout
carries only the command's output.

## Infrastructure decisions

- **LLM access:** an OpenAI-compatible gateway (OpenRouter) — base URL, key
  and model from env vars, with the `openai` package used only as an HTTP
  client. Evals choose the model (Haiku 4.5 vs Sonnet 5). Structured output
  uses a forced `tool_choice`, which rules out models that reject it.
  `llm.py` retries HTTP-level failures through the SDK, provider errors
  OpenRouter returns inside a 200 itself, and a structured call that
  doesn't call the tool or doesn't validate.
- **Search:** Tavily, behind a `SearchClient` protocol. Results are
  untrusted content: the rendered research frames them as reference data,
  not instructions, and a page can't close or open a source tag.
- **Timeouts:** each LLM call has a ~2 minute timeout
  (`PRECIS_LLM_CALL_TIMEOUT_SECONDS`), a circuit breaker for a hung call;
  the write call sets 5 minutes and fewer HTTP retries, since it's long.
  There's no whole-run limit: the per-call timeouts bound a run, at about
  half an hour in the worst case (a stalled provider on the write call
  through every retry).
- **Cost:** every run reports the gateway-reported cost of its calls plus
  its search credits. Target: under $0.50 a book on average, measured on
  the eval set.
- **Docker:** see the Docker boundary above. The image runs as a non-root
  user and writes only to `/output`, `/data` (the research cache, on a
  named volume) and `/tmp`; README.md has the `docker run` hardening flags.

## Out of scope

- A storage layer — generation's contract (JSON in, JSON out) leaves storage
  to the consumer.
- Consumer integration (book-keeper's schema and publishing tooling) —
  follow-up work on the consumer's side.
