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
   wrong, empty, or the wrong edition), and `chapters` left empty for the
   next phase. Runs on host, not in Docker — see "Docker boundary" below for
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
  `one_line_takeaway`, `synopsis`, `tags`, `parts?`, `reader_notes?`,
  `warnings[]`. `date_added`/`verified` are not generation output at all —
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
- `parts`: one shared shape (`title`, `summary`, optional chapter-number
  references) used by both branches — the difference is entirely in the
  prompt behind it, not the schema:
  - Non-fiction: groups already-known chapters into named structural
    sections.
  - Fiction / narrative non-fiction: spoiler-safe, structural beats — what
    changes and what's at stake at each transition, not what happens — for
    sketching the story's shape in memory without giving away plot turns.
- `parts` are AI-generated in every case, never hard-determined from input
  structure — the known-file supplies at most a flat chapter list (or
  nothing, for fiction).
- The module validates its own output against this schema before ever
  emitting it (see Stage 4 below) — including cross-field checks like
  `parts` referencing real chapter numbers. That guarantee lives entirely
  inside the module; what a consumer chooses to do on ingest (re-validate
  defensively or not) is the consumer's own business.

## Pipeline stages (whole-book mode)

1. **Verify** — one search + one model critique against the known-file's
   `isbn`/`chapters`, scoped narrowly to confirming edition/chapter-list
   correctness (not general thematic research — see Stage 3). Fail fast on
   mismatch (wrong edition, wrong book, bad chapter list) before any
   expensive per-chapter work runs. Skippable via a `--trust-known`-style
   flag for a known-file the reader is already confident about.
2. **Draft chapters** (non-fiction full path only) — parallel, bounded
   concurrency (configurable, default **3**). Per chapter: search-ground
   (`"<title>" "<chapter>" summary`), draft `key_points` + `core_claim`,
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
   Critique also runs an automated redundancy check (similarity/overlap
   between `key_points` entries) as a second line of defense against the
   restatement-padding problem this rewrite exists to fix — the prompt
   instruction against restating ideas is necessary but was already shown
   (by the 42% max-out measurement) to be insufficient on its own.
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
    → whole-book mode, writes a validated book JSON. Auto-resumes from an
      existing checkpoint for this known-file's thread_id if one exists;
      --fresh discards it and starts clean instead.

precis generate-chapter <known-file.json> --chapter <n> [--output <path>]
    → single-chapter mode, writes just that chapter object, validated
      against the chapter schema alone. Takes the known-file (not the
      already-generated book) as input — the known-file is already the
      source of truth for the chapter list, so nothing else is needed to
      ground the regeneration.
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
  - **Run identity:** a `thread_id` derived deterministically from the
    known-file's content (e.g. a hash of its canonical JSON) — the same
    known-file resolves to the same thread, so resuming is "run the same
    command again," not a separate resume-specific invocation the caller
    has to construct.
  - **CLI:** `precis generate` auto-detects an existing checkpoint for that
    known-file's `thread_id` and resumes from it; `--fresh` forces a clean
    run and discards any existing checkpoint for that thread instead.
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

