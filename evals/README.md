# Eval set

The fixed set of books every pipeline change is measured on. The first eight
are famous, so the model knows them well; the last two are lesser-known,
like much of a real library, so the notes lean on the research. Commands and
output are described in docs/blueprint.md's CLI contract; the Docker
invocation is in the top-level README.

| Slug | Kind | Why it's here |
|------|------|---------------|
| `thinking-fast-and-slow` | non-fiction, popular science | dense — dozens of distinct ideas; tests coverage |
| `good-to-great` | non-fiction, business | one tight framework |
| `atomic-habits` | non-fiction, self-help | practical techniques alongside the ideas |
| `sapiens` | non-fiction, history | a sweeping argument over a long timeline |
| `into-the-wild` | non-fiction, story-shaped | narrative non-fiction under the same treatment |
| `the-immortal-life-of-henrietta-lacks` | non-fiction, story-shaped | braided narrative with science and ethics |
| `nineteen-eighty-four` | fiction | themes; spoiler safety |
| `pride-and-prejudice` | fiction | themes; spoiler safety |
| `the-integrity-of-the-personality` | non-fiction, psychology | lesser-known (1960): research is thin, so it tests the follow-up searches and notes that lean on few sources |
| `a-month-in-the-country` | fiction | lesser-known short novel: thin research, themes and spoiler safety |

- `books/<slug>.json` — the known-file, checked by hand.
- `references/<slug>.json` — a reference summary written by Claude
  in-session, in the output's field names: takeaway, synopsis, `ideas`
  (5–12 key ideas for non-fiction, 3–6 themes for fiction) and, for
  non-fiction only, `key_claims_for_review`. The judge uses it as a guide to
  what good notes cover, not as the only truth. Fiction references are
  spoiler-safe — premise and setup only — like the pipeline's own output.
- `runs/<label>/` — one directory per run: each book's JSON, its
  `.metrics.json`, and any `judge-vs-<baseline>.json`. Label runs after
  what they vary, e.g. `sonnet-5` or `sonnet-5-review`. Baseline runs are
  committed.

Rules:

- **Never use these books as exemplars** — an exemplar from the eval set
  would teach to the test.
- Runs cost real money: confirm before a full run.
- Don't edit a book or its reference once runs exist against it — earlier
  runs stop being comparable. Add a new book instead.
