# Part 6 — Generation Module Rewrite (Python)

**Context:** A ground-up rewrite of the AI generation pipeline as a
standalone, generic Python module — not a port of, and not designed around,
the existing TypeScript/LangGraph.js pipeline (`scripts/generate-book.ts` and
its `lib/` files) or this project's own git-based storage tooling. The old
TS pipeline is useful reference for what worked and didn't; it is not a
constraint. Supersedes `02-generation-pipeline.md`, which stays only as
historical record of the outgoing version.

Two concrete problems drove this: title-only generation (no known chapter
list) performed too poorly to keep as a supported path, and measurement
against the 18 published books showed 42% of chapters (102/243) hitting the
old `key_points` schema's max of 10, with visible paraphrased-restatement
padding rather than genuinely distinct content.

**Repo split (decided, not yet executed as of this note):** this module is
being developed as its own git repository — working name **`precis`**,
tentative — not as a `generation/` subdirectory of book-keeper. Two reasons,
both concrete rather than speculative: a second real consumer (a SQLite
storage module) is being built immediately after this one, so "multiple
independent consumers" stopped being hypothetical; and book-keeper's own
devcontainer has no Python tooling and no Docker access at all (verified
directly — no `docker`, no `pip`, and this project's `.devcontainer/`  was
built specifically around Node/pnpm), so the module needs its own
purpose-built devcontainer (Python + `docker-outside-of-docker`) regardless.
This doc's content is the founding design, carried over into the new repo
where active development continues — book-keeper's own copy becomes a
historical snapshot once that copy happens, same treatment as
`02-generation-pipeline.md` below.

## How the module is used — three phases

1. **Known-file creation** (ISBN → known-file). Deterministic, no AI: looks
   up the ISBN via Open Library and writes a known-file with `isbn`,
   `title`/`author`/`year`/`page_count` filled in where found (placeholder
   text where not — the reader fills these in by hand when the lookup is
   wrong, empty, or the wrong edition). `chapters` and `parts` are
   pre-filled from Open Library's `table_of_contents` when it has a clean
   one — for this edition, or failing that another edition of the same
   work (the reader is told which) — and left empty otherwise. Coverage is
   patchy (roughly a third of popular non-fiction in spot checks) and the
   data is hand-entered, so parsing is conservative: entries that lump
   several chapters together are rejected outright rather than guessed at,
   front/back matter (acknowledgements, notes, index…) is dropped, parts
   come from "Part …"/"Book …" labels or titles, and entry `level`s are
   read with the most common one as the chapter level — deeper entries
   are subsections, and a shallower one that isn't a part or an
   intro/conclusion rejects the whole list (an unlabelled grouping can't
   be told apart from a chapter). Contents are only borrowed from another
   edition in the same language, or with no language recorded when this
   one has none — most works' other editions are mostly translations. Either way it's a starting
   point for the next phase, not a substitute for it. Runs on host, not in Docker — see "Docker boundary" below for
   why that's fine here.
2. **The reader fills in `chapters`** (and corrects `title`/`author`/etc. if
   needed, adds `notes`) by hand. A lightweight structural pre-flight check
   — also host-run, also non-AI — confirms the known-file is complete
   (`isbn` present, `chapters` non-empty for the non-fiction full path)
   before it's worth submitting for generation, catching an incomplete file
   in milliseconds rather than after a full Docker run.
3. **Generation** — the completed known-file is submitted to the module,
   which always runs in Docker, and produces a validated book JSON (or a
   single-chapter JSON — see "Output" below).

Because `title` is already resolved and present on the known-file by the
time generation is ever invoked, generation itself never needs to resolve
identity or compute a slug — that already happened in phase 1, on the host,
outside Docker entirely.

## Module boundary

The module talks to the rest of the world only through a JSON-file
boundary: it reads a known-file, it writes a book JSON (or chapter JSON).
Nothing about its own design references this project's git-based storage
workflow (`publish-book.ts`/`accept-book.ts`/`reject-book.ts`) or any other
specific consumer — that tooling is this project's *current* storage/
delivery mechanism, exactly as replaceable as it would be by a future
SQLite-backed version, and has no bearing on how the generation module
itself is built. Now that it's a separate repository (see the repo-split
note above), this isn't just a design discipline — book-keeper's source
literally isn't reachable from it. The module must stay generic enough that
a different project, or book-keeper's own future storage swap, could reuse
it unmodified.

Getting this project's existing TS tooling to actually consume the new
module's output (updated schema, the phase-1/2 known-file tools, whatever
CLI contract phase 3 exposes) is real, separate follow-up work — likely
touching more than just `publish-book.ts` — but it is *integration* work on
the TS side, in this repo, not something the Python module's design
accommodates or is scoped by. Tracked as out of scope for this part; see
below.

## Docker boundary

Docker exists specifically to contain AI — anything that calls an LLM or
otherwise produces non-deterministic, unverifiable output. It is not a
blanket rule over every piece of tooling in this module's orbit. Concretely:

- **Always in Docker:** phase 3 (generation itself) — every LLM call, every
  search call that feeds a model prompt.
- **Fine on host:** phase 1 (Open Library lookup — a deterministic call to a
  verifiable, well-defined public API) and the phase-2 structural pre-flight
  check (pure local JSON shape-checking, no network, no model). Both are
  deterministic, safe, and guarded against a known, verifiable source — the
  thing Docker-always is protecting against doesn't apply to them.

## Input: known-file

- `isbn` — required.
- `title` / `author` / `year` / `page_count` — optional. Auto-filled from
  Open Library by phase 1 when found; left as placeholder text for the
  reader to fill in by hand when the lookup is wrong, missing, or the wrong
  edition. These are a real override mechanism, not vestigial — Open
  Library coverage/accuracy is inconsistent, and the reader is the actual
  source of truth when it's wrong.
- `chapters: list[str]` — required for the full non-fiction study-guide path
  (`kind: non-fiction`, `narrative: false`) before generation will accept
  the file. Optional/absent for fiction and narrative non-fiction, which
  don't decompose into a per-chapter breakdown at all.
- `parts: list[{title, chapters?}]` — optional, non-fiction only
  (both branches: full and narrative). Same reasoning as the top-level
  `chapters` field: when the reader already knows the book's real part
  structure, that's a fact to supply, not something worth an LLM guessing
  from chapter titles alone. A part's own `chapters` (a list of chapter
  *numbers* — named to match the published Part shape below, distinct from
  the top-level `chapters` list of titles) only means something against a
  known chapter list, so it's only meaningful (and only checked by the
  phase-2 preflight) on the full non-fiction path; narrative non-fiction
  known-parts entries are title-only. Rejected outright by preflight for
  `kind: fiction` — fiction's parts are spoiler-safe invented beats (see
  Output below), not a fact the reader could supply even if they wanted to.
- `kind: "fiction" | "non-fiction"`
- `narrative: bool` — non-fiction only; routes to the lighter parts-based
  treatment instead of the full chapter/claims study guide. The phase-2
  structural preflight check rejects `narrative: true` set alongside
  `kind: fiction` outright (not silently ignored) — the combination only
  means something for non-fiction, and a reader who set it on a fiction
  known-file almost certainly misunderstood the field rather than meant
  something by it.
- `notes` — optional reader notes, a weighting signal only, persisted
  verbatim into the output's `reader_notes` field, never quoted into
  generated prose.
- No title-only path. `chapters` must be known and filled in by hand before
  generation runs — the direct response to the quality problem above, not
  an incidental restriction. This makes the known-file a hard precondition
  for every future book, not an optional convenience the way it is today.

## Output: two modes

- **Whole-book** (the normal case): the full pipeline below, output is one
  validated book JSON.
- **Single-chapter** (targeted regeneration of one chapter on an
  already-generated book): runs only the per-chapter drafting stage for the
  requested chapter, output is just that one chapter object, validated
  against the chapter schema alone — not the whole book schema, and not
  merged into anything.

Merging a single-chapter output back into wherever the book currently lives
(a git-committed JSON file today, a SQLite row later) is a storage-specific
integration concern, deliberately outside this module's scope — same
reasoning as the module boundary above.

### Book JSON shape (whole-book mode)

Redesigned in Pydantic as this module's own source of truth. Whatever a
consumer uses to represent this shape on its own side (book-keeper's zod
schema or otherwise) is that consumer's concern, not this module's — the
module's job ends at emitting valid JSON per `schema_version` (below).

- Common: `schema_version`, `title`, `author`, `year`, `isbn`, `page_count`,
  `kind` (`"fiction" | "non-fiction"`), `narrative`, `one_line_takeaway`,
  `synopsis`, `tags`, `parts?`, `parts_source?` (`"known" | "generated"`),
  `reader_notes?`, `warnings[]`.
  `kind`/`narrative` are passed through verbatim from the known-file, same
  as `title`/`author`/`page_count` — they're facts about the book a consumer
  needs (book-keeper's own storage schema discriminates on `kind`), not
  merely an internal routing input, even though generation itself only ever
  reads them to decide which pipeline path to run.
  `date_added`/`verified` are not generation output at all —
  set by whatever consumes the output (this project's publish tooling sets
  `date_added` itself and always defaults `verified: false`; a different
  consumer would do the same in its own way).
  - `schema_version`: a plain string tag identifying the shape of this
    output. Not a defensive-validation mechanism — the module's own
    pre-emission validation (Stage 4) is what guarantees a given output is
    internally valid; this field exists so a consumer built against an
    older or newer schema shape has something concrete to check against
    instead of guessing why fields don't match. Whether any given consumer
    bothers to check it is entirely up to that consumer.
  - `tags`: 2-4 unique entries drawn from one of two closed vocabularies
    (`NONFICTION_TAGS` / `FICTION_TAGS`, schema.py), chosen by `kind` alone
    (narrative non-fiction uses the non-fiction list too — the split isn't
    fiction-vs-narrative, it's fiction-vs-not). Curated from BISAC Subject
    Headings, mirroring book-keeper's own schema.ts exactly — a closed
    taxonomy the model picks from, not free-form labels of its own invention.
    Enforced twice: Stage 3's `Synthesis` model_validator checks it against
    `known_file.kind` via `validation_context`, so a bad tag is a schema
    failure `complete_structured`'s own retry already handles (cheap, before
    the run continues); `Book`'s own model_validator re-checks it at Stage 4
    as the final safety net, using `self.kind` directly since `Book` always
    carries it by then.
  - `warnings[]`: run-level notices worth a human's attention — e.g. one
    entry per chapter that fell back to a critique-failed candidate (see
    Stage 2 below). Empty on a clean run. Distinct from `quality_flag`
    (below) in that it's the single place to scan for "did anything need a
    second look," without walking every chapter.
- Non-fiction, full path: `chapters[] { number, title, key_points[],
  core_claim, quality_flag? }`, `key_claims_for_review[] { prompt, answer }`.
  - `quality_flag`: set on a chapter only when Stage 2 fell back to the
    last schema-valid candidate after critique kept failing past the retry
    cap. Absent (not `false`) on a clean pass. Carries the fallback signal
    through to the output instead of losing it the moment the chapter
    object is written — this is what "flagged, not discarded" (below)
    actually means in the schema, not just in the pipeline's behavior.
  - `key_points`: capped at **6**, down from the old max of 10 — a ceiling
    against padding, not a target to hit. Prompt explicitly instructs
    against restating the same idea in different words. No artificial
    floor forcing content that isn't there — the fix is anti-redundancy,
    never dropping real distinct ideas to hit a lower number.
- Fiction / narrative non-fiction: no `chapters`, no `key_claims_for_review`
  — not something to drill recall-style, per the original schema's own
  rationale (reading fiction isn't about retaining facts the way non-fiction
  is).
- `parts`: one shared shape (`title`, `summary`, optional `chapters` —
  chapter-number references, named to match book-keeper's own `partSchema`)
  used by both branches — the difference is entirely in the prompt behind
  it, not the schema:
  - Non-fiction: groups already-known chapters into named structural
    sections.
  - Fiction / narrative non-fiction: spoiler-safe, structural beats — what
    changes and what's at stake at each transition, not what happens — for
    sketching the story's shape in memory without giving away plot turns.
- `parts` are AI-generated only when the known-file didn't already supply
  them. When it did (non-fiction, either branch), Stage 3 keeps the
  known-file's title/chapters verbatim and asks the model only for
  `summary` — never for fiction, whose known-file `parts` is always empty
  (rejected by preflight). `parts_source: "known" | "generated"` on the
  output records which happened, so a consumer doesn't present an
  AI-invented grouping as the book's real published structure.
- The module validates its own output against this schema before ever
  emitting it (see Stage 4 below) — including cross-field checks like
  `parts` referencing real chapter numbers, and stripping a part's
  `chapters` to null whenever the book itself has no `chapters` array
  (fiction/narrative non-fiction) for such a reference to mean anything
  against, regardless of whether the model invented one. That guarantee
  lives entirely inside the module; what a consumer chooses to do on ingest
  (re-validate defensively or not) is the consumer's own business.

## Pipeline stages (whole-book mode)

1. **Verify** — one search + one model critique against the known-file's
   `isbn`/`chapters`/`parts` (when supplied), scoped narrowly to confirming
   edition/chapter-list/part-structure correctness (not general thematic
   research — see Stage 3). `parts` gets checked here for the same reason
   `chapters` is: once Stage 3 passes it through as fact instead of
   inventing it, this is the one point that confirms it's actually the
   book's real structure, not a reader's typo or misremembering. Fail fast
   on mismatch (wrong edition, wrong book, bad chapter list, bad part
   claims) before any expensive per-chapter work runs. Skippable via a
   `--trust-known`-style flag for a known-file the reader is already
   confident about.
   The model reports a list of issues (kind, what the known-file claims,
   what the sources say, source URL), each either **contradicted** (a
   result says otherwise) or **unconfirmed** (the results don't mention
   it); the pass/fail decision is made in code, not by the model.
   Unconfirmed is the normal case for a table of contents — search
   snippets rarely carry a full one — and an earlier single-bool verdict
   failed correct known-files by treating "not in the results" or unusual
   chapter titles as fabrication. Every issue is surfaced either way: on a
   pass as this stage's progress line, on a fail in the error, with a
   pointer to fix the known-file or rerun with `--trust-known`.
   - **Search:** "deep" (fuller page excerpts), on the short title +
     author with a title-only fallback query — a quoted full title/ISBN
     filters out most pages that list a book's contents. Results are kept
     if they name the short title, deliberately not filtered on the
     claimed author: that's one of the claims being checked, and filtering
     on it would hide every page crediting the real author. If no result
     names the book at all, verify **fails** before the model call ("couldn't
     find this book online") — a mistyped or made-up title would otherwise
     pass as "unconfirmed" and pay for every chapter draft; `--trust-known`
     is the way past it for a book the web really doesn't know. (A search
     that returns nothing at all is reported as a search-service problem
     instead.) The short title drops a parenthetical ("(Revised
     Edition)") as well as the subtitle.
   - **What can fail the run:** only a contradiction with an actual
     contrary value, for three kinds, each held to a source that pins the
     claim to this book (the model cites results by their `[n]` number,
     not by copying a URL that can drift):
     - *author* — a source that identifies this exact book (full title
       including subtitle, or the ISBN), so a different "Range" or "Grit"
       can't overrule it;
     - *chapter* / *part* — a source that names the claimed author too
       (`is_book_relevant`) *and* reproduces the book's real table of
       contents (the model flags this per issue). Summary and "key
       takeaways" sites reword books into their own section headings, and
       one of them "contradicted" The Diet Myth's correct chapter list.
       Chapters are shown to the model unnumbered (unless parts refer to
       them by number) and compared by title and order only — contents
       listings number their own way and include front/back matter.
     `title` (only the subtitle can differ, since every result names the
     short title), `isbn`, `year` and `page_count` vary by edition, and
     results often describe a different one, so they only ever warn.
2. **Draft chapters** (non-fiction full path only) — parallel, bounded
   concurrency (configurable, default **3**). Per chapter: search-ground
   (`"<short title>" <author> "<chapter>" summary`, with an unquoted
   fallback), draft `key_points` + `core_claim`,
   critique the draft against the search results, repair-and-retry on
   failure up to a small local cap, falling back to the last schema-valid
   candidate if critique still fails after retries are exhausted — a
   critique failure is a quality signal, not proof the content is unusable.
   This retry cap is for **content-quality failures only** (critique
   rejects the draft). Transient/technical failures (rate limits, 5xx,
   dropped connections) are a different problem and are handled separately,
   with exponential backoff at the HTTP-client level — they never consume
   the quality-repair budget and never set `quality_flag`, since nothing
   about the content itself was actually in question.
   A fallback candidate is never silently accepted as equivalent to a clean
   pass: it sets `quality_flag` on that chapter (see Book JSON shape below)
   so it surfaces to whoever reviews the output, rather than being
   indistinguishable from a chapter that passed critique cleanly.
   Only search results about *this book* count: a result must name the
   author's surname together with the short title, or the full title when
   it has a subtitle (`search.is_book_relevant`). Topic-only pages (a
   generic "diet myths" article for a nutrition book) otherwise let
   critique pass content the author never wrote. A chapter with no
   book-specific results is drafted from the model's own knowledge,
   critiqued for scope and distinctness only, and always flagged.
   Grounding alone doesn't keep a draft on its chapter: results about the
   book in general support a whole-book summary, or a neighbouring
   chapter's material, as well as this chapter's (The Diet Myth's
   "Contains Alcohol" passed as a book summary with no mention of
   alcohol). So both the draft and critique prompts list the book's other
   chapters, and critique checks **scope** — fail a draft that summarizes
   the book (unless the chapter is itself an introduction, overview,
   conclusion or epilogue), has a key point mainly about another listed
   chapter's topic (book-wide themes can still come up), or gives generic
   advice on the title's topic rather than the author's argument. Key
   points that only describe the chapter ("the chapter examines ...")
   fail the distinctness check. There's deliberately no minimum key-point
   count: a thin chapter with two real points is fine, and a floor would
   invite the padding the distinctness check exists to stop.
   Critique's distinctness check (does any `key_point` restate another) is
   what actually enforces the anti-padding rule from the draft prompt — the
   42% max-out measurement that motivated this rewrite was against the old
   pipeline, which had no critique loop at all judging this; there's no
   separate automated (non-LLM) similarity/overlap check today. Worth
   revisiting only if this critique-based check turns out insufficient on
   its own, the same way the plain prompt instruction did under the old
   pipeline.
3. **Synthesize** — `synopsis`, `one_line_takeaway`, `tags`,
   `key_claims_for_review` (non-fiction full path), `parts`. Fed by the
   finished chapters (where they exist) plus its **own dedicated
   themes-oriented search** — deliberately not reusing Stage 1's
   edition/chapter-list-oriented search results as its primary source,
   since the two stages are answering different questions and a single
   search pass can't serve both well. Runs *after* chapter drafting so
   claims/parts are grounded in real, already-critiqued content rather than
   a pre-chapter guess — this ordering is deliberate (an earlier draft of
   this plan had synthesis first; the catch: `key_claims_for_review` would
   read as generic back-cover material, and `parts` can't meaningfully group
   chapters that don't exist yet).
   On the known-parts path, the exact expected part count is enforced as a
   real schema constraint (a `model_validator` reading a `validation_context`
   passed into `llm.complete_structured`, not a check bolted on after the
   fact) — so a model that returns the wrong count is a validation failure
   `complete_structured`'s own retry already handles, giving it a second,
   cheap attempt at Stage 3 before the whole run fails. A same-count title
   mismatch isn't fatal (the known title always wins), but it's recorded as
   a `warnings` entry on the output rather than silently overwritten, since
   the model disagreeing about a title is itself a signal worth a second
   look, whichever side turns out to be wrong.
4. **Assemble + validate the whole book** — full schema validation
   including cross-field checks (`parts` referencing real chapter numbers).
   One repair-and-retry pass if invalid; if that also fails, generation
   fails outright with no output file — a whole-book structural failure is
   not something to silently paper over.
5. **Output** the single validated book JSON.

Fiction / narrative non-fiction follow the same shape, minus stage 2.
Single-chapter mode runs stage 2's per-chapter loop alone, for one chapter,
skipping the rest.

## CLI contract

```
precis generate <known-file.json> [--output <path>] [--trust-known] [--fresh]
    → whole-book mode, writes a validated book JSON. The known-file's
      filename stem is the book's slug and names its checkpoint thread,
      which is deleted once the book is written; a rerun of an interrupted
      run resumes it (see Run identity below). --fresh discards an
      interrupted run's checkpoint and starts clean. An unwritable
      --output is refused up front, before any paid work — a finished book
      that can't be written isn't recoverable from its checkpoint.

precis generate-chapter <known-file.json> --chapter <n> [--output <path>]
    → single-chapter mode, writes just that chapter object, validated
      against the chapter schema alone. Takes the known-file (not the
      already-generated book) as input — the known-file is already the
      source of truth for the chapter list, so nothing else is needed to
      ground the regeneration.

precis tags [--output <path>]
    → prints the closed tag vocabulary (`schema_version`, `non_fiction_tags`,
      `fiction_tags` — see TagVocabulary in schema.py) as JSON. Exists so a
      consumer repo (book-keeper's `schema.ts`, or any future one) can pull
      this instead of hand-copying NONFICTION_TAGS/FICTION_TAGS verbatim —
      see the "Mirrors book-keeper's own schema.ts" comment on those two
      tuples in schema.py.

precis checkpoints [--prune] [--older-than-days <n>] [--include-incomplete]
    → lists every thread in the checkpoint store (thread id, last-updated
      timestamp, done/in-progress) with no args. `--prune` deletes matching
      threads instead of listing them. Nothing is deleted automatically,
      ever — a checkpoint accumulates forever otherwise (every book, and
      every edited draft of every known-file, gets its own thread; see
      Orchestration above). Only "done" threads (reached assemble, nothing
      left to resume) are eligible by default; `--include-incomplete`
      widens that to threads still mid-run, which forfeits resuming them,
      not just reclaiming disk space, so it's opt-in. `--older-than-days`
      narrows either set by the last checkpoint's age.
```

Errors (verify-stage mismatch, exhausted retries on the whole-book assemble
step, run-budget timeout) are a non-zero exit code plus a message on
stderr. On a whole-book structural failure, no output file is written at
all — no partial or corrupt file left behind for a caller to trip over. A
run-budget timeout is not a structural failure in this sense — the
checkpoint from whatever completed survives it, ready for a resumed run.

Merging a single-chapter output back into wherever the book currently lives
is still the consumer's job, per the module boundary above — this CLI only
ever writes a standalone JSON file, never merges into an existing one.

## Infrastructure decisions

- **LLM access:** provider-agnostic gateway (OpenRouter-style) — base URL +
  API key + model name via env vars, not a vendor-specific SDK. Matches the
  longer-term goal of the eventual app letting a user supply their own
  `.env` settings.
- **Search:** kept, not dropped, and used for two different jobs by two
  different stages (see Stages 1 and 3 above) — it grounds per-chapter/
  per-book content and backs the critique/repair loop, which is
  structurally load-bearing for output quality, not just a discovery aid.
  Only the old outline-discovery/chapter-guessing search is gone, since
  chapters are always supplied now. Search results are untrusted content,
  not instructions: they're fed to the model as data to ground against, and
  nothing in a search result should be treated as overriding the prompt
  it's grounding — a basic prompt-injection guard given the content is
  fetched from the open web.
- **Orchestration:** LangGraph (Python), reinstated — the "revisit if a
  concrete need shows up" condition from the earlier no-LangGraph decision
  was met almost immediately: whole-book runs take 10+ minutes, run inside
  a container with no host-process durability guarantee, and a crash with
  no checkpoint means redoing the entire run, including already-finished
  chapters. That's a real cost, not a hypothetical one, and it's exactly
  what LangGraph's checkpointing exists to solve — plain `asyncio` was the
  right call when the only question was fan-out, but resumability wasn't in
  scope at that point.
  - **Checkpointer:** `langgraph-checkpoint-sqlite`, writing to a SQLite
    file on a Docker volume mounted into the generation container (separate
    from and in addition to the devcontainer's own volumes) — so the
    checkpoint survives the *container* dying, not just the process inside
    it. This is standard LangGraph usage, not a custom persistence layer:
    the checkpointer captures state after each node completes, including
    per-branch state in Stage 2's parallel fan-out, so a resumed run only
    redoes chapters that hadn't finished, not the whole batch.
  - **Run identity:** the known-file's slug (its filename stem — the same
    name the consumer gives the output), so a book has exactly one thread
    and editing its known-file doesn't silently start a new one.
  - **Lifetime:** a thread exists only to resume an interrupted run. The
    CLI deletes it once the finished book is written (only after — a
    failed write keeps it), so a finished book is never handed back from
    a checkpoint: re-publishing and history are the consumer's job, and a
    leftover finished book would only resurface a rejected or stale one.
  - **On a rerun:** no checkpoint, one that never got past verify (only
    verify's search + model call spent), or a finished one that outlived
    a failed write, starts clean — which is also what lets `--trust-known`
    or a known-file edit take effect after a failed verify. Past verify,
    the saved known-file is compared with the current one: unchanged
    resumes where it stopped (passing `None` as the graph input — passing
    the input state again would restart from START and re-append every
    chapter); changed aborts with the list of changed fields unless
    `--fresh` is given, since resuming would mix chapters drafted from two
    different known-files. `notes` is exempt: no stage reads it (it's
    copied verbatim into `reader_notes`), so a notes edit resumes and the
    current notes go into the book. It also aborts when the saved
    known-file has a different `isbn` (a slug collision, not an edit), or
    when the saved run skipped verify via `--trust-known` and this one
    doesn't pass it — resuming would otherwise finish a book that was
    never verified.
  - **CLI:** resuming is "run the same command again," not a separate
    resume-specific invocation; `--fresh` discards the interrupted run
    instead.
- **Run budget:** unchanged at a wall-clock ceiling of **1 hour** for a
  whole-book run, plus a **~2 minute** timeout on each individual LLM call
  — still a circuit breaker for a genuinely hung run, not a constraint
  meant to bind on a normal one (a legitimately-behaving run taking a full
  hour would itself be surprising). Enforced as an in-process
  `asyncio.wait_for` around the whole graph invocation, not an external
  process kill — hitting it cancels the run and raises cleanly within the
  same process, it doesn't terminate the container. Checkpointing changes
  what happens when the ceiling *is* hit: progress isn't lost, since
  LangGraph has already persisted completed nodes independently of the
  run being cancelled — hitting the budget means "resume later," not
  "start over." No token/cost budget for now — token spend is worth
  logging for visibility, but a hard spend cap is a separate concern from
  run safety and isn't needed to ship this.
- **Docker:** contains AI generation only (phase 3) — see "Docker boundary"
  above. The new repo's devcontainer provides real Docker access via the
  `docker-outside-of-docker` feature (mounts the host's socket) specifically
  so this is possible in day-to-day development, not just in CI — book-
  keeper's own devcontainer was never built for this. The generation
  container needs one persistent volume beyond its own ephemeral
  filesystem: the checkpoint SQLite file (see Orchestration above), so a
  checkpoint written by one container invocation is still there for the
  next one. Concrete Dockerfile/compose shape for the generation image
  itself (distinct from the devcontainer) to be finalized when the
  foundation is actually built.

## Explicitly out of scope for this part

- Regenerating the 18 already-published books under the new pipeline/prompt
  — planned, but later, not blocking this rewrite.
- A SQLite (or any) storage module — generation's contract (JSON in, JSON
  out) already supports swapping storage later without touching this
  module at all.
- Desktop/tablet/mobile app shell, user-facing settings UI — later, separate
  concern.
- Updating this project's existing TS tooling (`publish-book.ts` and
  whatever else needs it) to actually integrate with the new module's
  output/CLI contract — real follow-up work, likely touching more than one
  file, but it's integration work on the storage side, not part of this
  module's own design.

## Status of the other blueprint docs

- `02-generation-pipeline.md` (TS/LangGraph.js) is superseded by this part
  once the Python module ships — kept as historical record of why the old
  version was shaped the way it was, not current guidance.
- `00-INDEX.md`, `00-setup.md`, `03-astro-frontend.md`,
  `04-spaced-repetition-review.md` remain current — they document the Astro
  frontend/site and review feature, none of which this rewrite touches.
- `01-schema-and-content.md` and `05-operations-and-future.md` will need
  in-place updates once this lands (new schema shape, new `generate`
  command, the TS-side integration work noted above) — same as they've
  always been kept in sync with what actually shipped, not being retired.
- **This doc itself**, once copied into the new `precis` (working name)
  repo: that copy becomes the living design doc, kept in sync with what
  actually ships there. This copy, left behind in book-keeper, becomes a
  historical snapshot of the founding design as of the repo split — treat
  divergence between the two as the new repo having moved on, not as an
  inconsistency to reconcile here.

