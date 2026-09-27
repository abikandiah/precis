# Eval set

The fixed set of books every pipeline change is measured on (docs/v2-plan.md's
Eval set section). Commands and output are described in docs/blueprint.md's
CLI contract; the Docker invocation is in the top-level README.

| Slug | Type | Why it's here |
|------|------|---------------|
| `thinking-fast-and-slow` | full non-fiction, popular science | 38 chapters in 5 known parts — exercises batching past ~25 chapters |
| `good-to-great` | full non-fiction, business | 9 chapters, no parts |
| `atomic-habits` | full non-fiction, self-help | 20 chapters in 6 known parts |
| `sapiens` | full non-fiction, history | 20 chapters in 4 known parts |
| `into-the-wild` | narrative non-fiction | |
| `the-immortal-life-of-henrietta-lacks` | narrative non-fiction | |
| `nineteen-eighty-four` | fiction | |
| `pride-and-prejudice` | fiction | |

- `books/<slug>.json` — the known-file, checked by hand. Chapters are the
  numbered chapters only (no introduction, conclusion or afterword).
- `references/<slug>.json` — a reference summary written by Claude
  in-session, in the output's field names. The judge uses it as a guide to
  what a good summary covers (coverage), not as the only truth. Fiction and
  narrative references are spoiler-safe, like the pipeline's own output.
- `runs/<label>/` — one directory per run: each book's JSON, its
  `.metrics.json`, and any `judge-vs-<baseline>.json`. Label runs
  `<pipeline>-<model>`, e.g. `v1-sonnet-5`. Baseline runs are committed.

Rules:

- **Never use these books as exemplars** — an exemplar from the eval set
  would teach to the test.
- Runs cost real money: confirm before a full run.
- Don't edit a book or its reference once runs exist against it — earlier
  runs stop being comparable. Add a new book instead.
