# precis v2 — redesign plan

Decided 2026-09-26. Work through the phases below across sessions; tick the
status boxes and fold each landed phase into docs/blueprint.md (which keeps
describing the current pipeline until a phase replaces part of it).

## Why

v1 was shaped around free models: retries for provider errors returned in a
200, `tool_choice`-capable provider routing, a per-chapter draft/critique
loop to catch weak drafts, LangGraph checkpoints for long rate-limited runs,
exemplars. Its grounding is also thin: per-chapter snippet searches
(~10 × ~500 chars) rarely find chapter-specific material, and drafting
chapters in isolation is what forced the scope machinery (other-chapter
lists, a critique scope check).

v2 assumes a **paid model** and fixes grounding at the root: research the
book once, write every chapter from that research.

## Targets

- **Cost:** under **$0.50 per book** on average, measured on the eval set
  (not estimated). Estimate: ~$0.15–0.45 on Sonnet 5, less on Haiku 4.5.
- **Quality:** beats v1 on the eval set's pairwise judgement (Phase 1).
- **Calls:** ~3–4 LLM calls and 3–5 searches per book (v1: ~35–50 calls,
  ~20–35 searches).

## Decisions

- **Paid model by default.** Haiku 4.5 vs Sonnet 5 is chosen by evals.
  Stay on the OpenAI-compatible gateway (OpenRouter) so evals can compare
  models. Structured output keeps forced `tool_choice` — which rules out
  Claude Opus 5.5 / Fable 5.1 (they 400 on it) unless we move to
  `tool_choice: auto` + strict schemas.
- **Docker stays** for everything that calls a model (including evals and
  the chapter suggester) — cross-platform, and the boundary is cheap.
- **Evals before exemplars.** Exemplars only if evals show a gap a worked
  example would close, and exemplar books must never be eval books.
- **Build v2 beside v1** (new package path), switch once it wins on evals,
  then delete v1.
- **Schema v2** (`schema_version` bump) for the key-point evidence change;
  book-keeper syncs its own schema afterwards.

## Target pipeline

```
known-file ─► research ─► draft (whole book) ─► checks + review ─► synthesize ─► validate ─► book JSON
               (cached)                                              (fiction/narrative skip draft/review)
```

1. **Research** (replaces verify). 3–5 searches with full page text (Tavily
   advanced search with raw content, or search + extract): chapter-by-chapter
   summaries, publisher description/excerpt, substantial reviews, author
   interviews. Keep only pages about this book (`search.is_book_relevant`),
   trim each, dedupe, cap the whole file at ~30k tokens. Sources are
   numbered (`S1`…) for citation.
   - Fails the run if nothing about the book is found (v1 verify's "couldn't
     find this book online"). Author check in code: sources must name the
     claimed author's surname, or it warns/fails as v1 does.
     `--trust-known` skips the checks, not the research.
   - Cached on disk per slug (`PRECIS_CACHE_DIR`, on the `/data` volume in
     Docker) so reruns don't search again; `--fresh` refetches.
2. **Draft the whole book in one call**: research + full table of contents
   (+ parts) → every chapter's `key_points` and `core_claim`. Books over
   ~25 chapters are drafted in batches, each batch seeing earlier chapters.
   Research file marked for prompt caching (verify OpenRouter passes
   `cache_control` through for Anthropic models).
3. **Checks in code, then one review pass.**
   - Code checks: meta-descriptions ("the chapter examines…"), near-duplicate
     points (word overlap), limits, citation IDs that exist.
   - One review call over the whole book against the research, given the
     code-check findings as specific issues: unsupported claims, off-scope
     points, vague points. Returns only the chapters it revises.
   - Points left uncited are flagged per point; chapters still failing
     review keep `quality_flag`.
4. **Synthesize**: synopsis, takeaway, tags, key claims, parts — from the
   final chapters + research. Part→chapter references validated in its own
   schema check (like the part count), so the structured-call retry fixes
   them and Stage 4's LLM repair goes away.
5. **Validate** the `Book`. No model call.

Orchestration is a plain async function — LangGraph, the checkpoint DB,
`CheckpointMismatch`, and the `checkpoints` command go. The research cache
is the only persisted state.

### Schema change

Each key point gains its evidence and sources:

```
key_points: [{ point, evidence, sources: ["S2", "S5"] }]   # max 6, as now
```

`evidence` is the study, story or example the author uses — makes the guide
memorable and grounding checkable. Empty `sources` = drawn from model
knowledge, flagged.

### Commands

- `generate` — as now, minus checkpoint semantics (`--fresh` now means
  "refetch research").
- `generate-chapter` — known-file + optional `--book <existing.json>` for
  neighbouring chapters; reuses cached research.
- `suggest-chapters` (new, Docker) — proposes a table of contents from search
  + model knowledge into the known-file, marked unconfirmed for the reader to
  correct. Doesn't reopen title-only generation: the reader still confirms.
- `eval` (new, Docker) — see Phase 1.

## Eval set (Phase 1)

- **Books:** 8 very well-known titles, disjoint from any exemplar: 4 full
  non-fiction (popular science, business, self-help, history with parts),
  2 narrative non-fiction, 2 fiction. Well-known so Claude-written references
  are reliable.
- **Per book:** a checked known-file plus a reference summary written by
  Claude in-session (same shape as the output) — used for coverage (did it
  miss a main idea?), not as the only truth.
- **Judging:** pairwise — a strong judge model picks the better of two
  outputs (candidate vs baseline) against a rubric: accuracy to the book,
  specificity, distinct points, chapter scope, coverage vs the reference.
  Both orders, to cancel position bias.
- **Metrics in code:** cost (gateway-reported usage), duration, flags,
  citation coverage, duplicate-point rate.
- **Baseline:** v1 on the chosen paid model, run once and stored.
- Runs cost real money — confirm before each full run.

## Phases

Each phase ends with tests, `ruff check`, `mypy` green, and for 2+ an eval
run against the stored baseline.

- [x] **0. Park + plan** — exemplar work parked on the `exemplars` branch; this plan
  committed.
- [ ] **1. Evals** — per-run cost/usage logging; `eval` command + judge;
  write the 8 known-files and reference summaries; run the v1 baseline.
- [ ] **2. Research** — research step + cache + code-based book/author
  checks, in the v2 package.
- [ ] **3. Draft** — whole-book draft call, schema v2 key points, batching.
- [ ] **4. Checks + review** — code checks and the single review pass.
- [ ] **5. Synthesize + validate** — v2 synthesize, part-ref validation,
  no repair call. v2 now runs end to end: eval vs baseline, pick
  Haiku vs Sonnet, confirm < $0.50.
- [ ] **6. Switch** — CLI on v2, `generate-chapter` rework, delete v1
  (pipeline/, checkpoints, LangGraph deps, free-model workarounds that no
  longer earn their place), fold into blueprint.md.
- [ ] **7. suggest-chapters.**
- [ ] **8. Exemplars** — only if evals show a gap.
