# precis v2 — redesign plan

Decided 2026-09-26; rescoped 2026-09-27 to drop chapters entirely. Work
through the phases below across sessions; tick the status boxes and fold each
landed phase into docs/blueprint.md (which keeps describing the current
pipeline until a phase replaces part of it).

## Why

v1 was shaped around free models: retries for provider errors returned in a
200, `tool_choice`-capable provider routing, a per-chapter draft/critique
loop to catch weak drafts, LangGraph checkpoints for long rate-limited runs,
exemplars. Its grounding is also thin: per-chapter snippet searches
(~10 × ~500 chars) rarely find chapter-specific material, and drafting
chapters in isolation is what forced the scope machinery (other-chapter
lists, a critique scope check).

v2 assumes a **paid model** and drops chapters. The goal is study notes for
refreshing a book after reading it: what it's about, its main ideas or
themes, and what it teaches. That's whole-book material, which is also what
the web actually has (reviews, publisher copy, interviews, summaries).
Chapter-by-chapter coverage was the weak spot in grounding and the source of
most of v1's complexity.

Every book goes through the same pipeline; fiction differs only in its
prompts (spoiler-safe, themes instead of ideas) and in having no review
deck.

## Targets

- **Cost:** under **$0.50 per book** on average, measured on the eval set
  (not estimated). Expected well below that: 2–3 LLM calls per book.
- **Quality:** a reader who finished the book can recall what it said from
  the notes alone — judged against the eval set's references (Phase 1).
- **Calls:** 2–3 LLM calls and 3–5 searches per book (v1: ~35–50 calls,
  ~20–35 searches).

## Decisions

- **No chapters, no parts.** Parts are structure, not content; the `ideas`
  list carries what a part summary would have. This removes chapter lists and
  parts from the known-file, the Open Library table-of-contents parser, the
  chapter preflight, part→chapter validation, batching, `generate-chapter`,
  and the planned `suggest-chapters`.
- **`narrative` goes.** Its only job was routing who gets chapters. Spoiler
  rules follow `kind` alone: fiction is spoiler-safe (the library is viewed by
  others), non-fiction is unrestricted.
- **Paid model by default.** Haiku 4.5 vs Sonnet 5 is chosen by evals.
  Stay on the OpenAI-compatible gateway (OpenRouter) so evals can compare
  models. Structured output keeps forced `tool_choice` — which rules out
  Claude Opus 5.5 / Fable 5.1 (they 400 on it) unless we move to
  `tool_choice: auto` + strict schemas.
- **Docker stays** for everything that calls a model (including evals) —
  cross-platform, and the boundary is cheap.
- **Evals before exemplars.** Exemplars only if evals show a gap a worked
  example would close, and exemplar books must never be eval books.
- **No v1 baseline.** A chapter guide and a whole-book summary aren't
  comparable pairwise. The first full v2 run is the baseline; later runs are
  judged pairwise against it, with the references as the coverage guide.
- **Replace v1, don't run beside it.** With no v1 comparison, there's no
  reason to keep it runnable: build the v2 pipeline, and delete v1 once v2
  runs end to end.
- **Schema v2** (`schema_version` bump); book-keeper syncs its own schema
  afterwards.

## Target pipeline

```
known-file ─► research ─► write ─► checks + review ─► validate ─► book JSON
               (cached)                (revise only if review finds issues)
```

1. **Research** (replaces verify). 3–5 searches with full page text (Tavily
   advanced search with raw content, or search + extract): publisher
   description/excerpt, substantial reviews, author interviews, summaries.
   Keep only pages about this book (`search.is_book_relevant`), trim each,
   dedupe, cap the whole file at ~30k tokens. Sources are numbered (`S1`…)
   for citation.
   - Fails the run if nothing about the book is found (v1 verify's "couldn't
     find this book online"). Author check in code: sources must name the
     claimed author's surname, or it warns/fails as v1 does.
     `--trust-known` skips the checks, not the research.
   - Cached on disk per slug (`PRECIS_CACHE_DIR`, on the `/data` volume in
     Docker) so reruns don't search again; `--fresh` refetches.
2. **Write the book in one call**: research + known-file → every output
   field. Research marked for prompt caching (verify OpenRouter passes
   `cache_control` through for Anthropic models) so the review call reuses
   it.
   - Non-fiction: ideas cover the book's whole range, not just its opening;
     the count scales with how much the book argues.
   - Fiction: premise and setup only — no twists, reveals, deaths or ending,
     nothing past roughly the first act. The research will be full of
     spoilers; the prompt says so and says to leave them out.
3. **Checks in code, then one review call.**
   - Code checks: meta-descriptions ("the book examines…", "the author
     discusses…"), near-duplicate ideas (word overlap), limits, citation IDs
     that exist, key claims that restate an idea verbatim, and **evidence
     specifics**: numbers and proper nouns in an idea's `evidence` (study
     names, figures, people, dates — where invention hides) must appear in
     the sources it cites.
   - One review call against the research, given the code-check findings as
     specific issues: unsupported claims, vague or generic ideas, missing
     major ideas the research covers, and — for fiction — a spoiler audit.
     Returns revised fields only; no revision means no second write.
   - **What happens to an idea the research can't validate:**
     - *Supported* — a cited source backs it: kept.
     - *Unsupported, not contradicted* — the research is silent: kept with
       empty `sources` and a per-idea warning. A ~30k-token research file
       can't cover everything, and the model knows well-known books; the
       reader (who has read the book) is the final check via book-keeper's
       `verified` step.
     - *Contradicted, or not placeable in this book* (generic enough to be
       from any book on the topic): dropped, and replaced with a supported
       idea if the research has one. A wrong idea is worse than a missing
       one.
     - Key claims follow the same rules; a claim resting on a dropped idea
       is dropped with it.
   - Fewer ideas left than the minimum (5 non-fiction, 3 fiction) fails the
     run with a clear message, like "couldn't find this book online". More
     than half the ideas uncited adds a book-level warning that research
     was thin.
4. **Validate** the `Book`. No model call.

Orchestration is a plain async function — LangGraph, the checkpoint DB,
`CheckpointMismatch`, and the `checkpoints` command go. The research cache
is the only persisted state.

### Known-file

Bibliographic facts only — nothing for the reader to hand-edit before
generating beyond checking them:

```
isbn, title, author, year, page_count, kind, notes
```

### Output schema

```
one_line_takeaway
synopsis               3–5 paragraphs
                         non-fiction: premise, how the argument builds, where it lands
                         fiction: premise, setting, main characters, what the story explores (spoiler-safe)
ideas: [{ title, summary, evidence, sources: ["S2", "S5"] }]
                         non-fiction: key ideas, 5–12
                         fiction: themes, 3–6 — the theme and how the book develops it
                         (through which characters or situations), spoiler-safe
key_claims_for_review: [{ prompt, answer }]    non-fiction only, 5–15
tags                   closed vocabulary per kind, as now
warnings
```

`evidence` is the study, story or example the author uses (non-fiction) or
the characters and situations that carry the theme (fiction) — makes the
notes memorable and grounding checkable. Empty `sources` = drawn from model
knowledge, flagged.

### Commands

- `known` — as now, minus table-of-contents parsing and `--narrative`.
- `generate` — as now, minus checkpoint semantics (`--fresh` now means
  "refetch research").
- `eval` (Docker) — see Phase 1.
- `generate-chapter` and `checkpoints` go.

## Eval set (Phase 1)

- **Books:** the 8 in `evals/`, disjoint from any exemplar: 4 argument-driven
  non-fiction, 2 narrative non-fiction (now ordinary non-fiction — checks the
  uniform treatment works on a story-shaped book), 2 fiction. Thinking, Fast
  and Slow stays as the dense-book coverage test.
- **Per book:** a known-file (bibliographic only) plus a reference summary
  written by Claude in-session, in the v2 output shape — used for coverage
  (did it miss a main idea?), not as the only truth.
- **Judging:** pairwise — a strong judge model picks the better of two
  outputs (candidate vs baseline) against a rubric: accuracy to the book,
  specificity, distinct ideas, coverage vs the reference, review-deck quality
  (non-fiction), spoiler safety (fiction). Both orders, to cancel position
  bias.
- **Metrics in code:** cost (gateway-reported usage), duration, flags,
  citation coverage, duplicate-idea rate.
- Runs cost real money — confirm before each full run.

## Phases

Each phase ends with tests, `ruff check`, `mypy` green, and for 4+ an eval
run against the stored baseline.

- [x] **0. Park + plan** — exemplar work parked on the `exemplars` branch; this plan
  committed.
- [ ] **1. Evals** — usage logging, `eval` command + judge, eval set.
  - [x] usage logging (`usage.py`), `precis eval run|judge`, eval set in
    `evals/` (chapter-shaped — reworked below).
  - [x] Rework for v2: chapters, parts and `narrative` stripped from the
    known-files; references rewritten in the v2 shape; judge rubric and
    rendering on ideas/claims, with per-kind criteria (`review_deck`,
    `spoiler_safety`); metrics on ideas. v1 removed from `eval run` — no
    pipeline is registered until Phase 4.
- [x] **2. Research** — `research.py` + `precis research`: three
  parallel advanced searches with raw page text, raw results cached per
  slug (filtering is a pure function over the cache, so changing it never
  refetches), identity checks in code. A real but wrong author named
  alongside the book on some page passes the code check — Phase 3's write
  call is the backstop.
- [ ] **3. Write** — schema v2 (`Book`, slimmed `KnownFile`), the write call
  and its prompts (non-fiction and fiction). The write call reports when
  the research shows the book is by someone other than the known-file's
  author, and that fails the run (the identity backstop from Phase 2).
- [ ] **4. Checks + review + validate** — code checks, the review call,
  plain async orchestration. v2 now runs end to end: first eval run becomes
  the baseline; confirm < $0.50; OpenRouter reports `usage.cost`.
- [ ] **5. Switch** — CLI on v2, delete v1 (pipeline/, checkpoints,
  LangGraph deps, TOC parsing, chapter preflight, `generate-chapter`,
  free-model workarounds that no longer earn their place), fold into
  blueprint.md.
- [ ] **6. Tune** — Haiku vs Sonnet and prompt variants, judged pairwise
  against the baseline.
- [ ] **7. Exemplars** — only if evals show a gap.
