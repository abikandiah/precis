# precis

Standalone Python module: reads a known-file (ISBN, title, author, kind),
researches the book on the web, and writes validated whole-book notes —
takeaway, synopsis, key ideas (themes for fiction, spoiler-safe) and, for
non-fiction, key claims for review. Generation always runs in Docker.

See [docs/blueprint.md](docs/blueprint.md) for the design and
[docs/v2-plan.md](docs/v2-plan.md) for what's still being built.

## Making a known-file

`create-known-file` looks the ISBN up on Open Library. It calls no model, so
it runs fine on a host:

```
uv run precis create-known-file 9780374533557 --kind non-fiction --output known-files/thinking-fast-and-slow.json
```

Check the title and author it found (a miss leaves `TODO: fill in by hand`)
and set `kind`. `notes` is optional: what matters to you about the book,
used to weight the notes and carried into the output, never quoted.

## Generating

```
cp .env.example .env   # fill in PRECIS_LLM_API_KEY, PRECIS_LLM_MODEL, PRECIS_SEARCH_API_KEY
docker build -t precis .

docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit=256 \
  --memory=512m --memory-swap=512m \
  --env-file .env \
  -v "$(pwd)/known-files:/input:ro" \
  -v "$(pwd)/output:/output" \
  -v precis-data:/data \
  precis generate /input/thinking-fast-and-slow.json --output /output/thinking-fast-and-slow.json
```

A run researches the book (three web searches), then writes the notes in
one model call. It fails — before the model call where it can — when no
page online names the book, or pages name it but not its author, or the
research credits the book to someone else. `--trust-known` turns those into
warnings in the output, for a known-file you've checked against the book.

The research is cached on the `precis-data` volume, named after the
known-file's filename, so a rerun doesn't search again; `--fresh` does.
Progress and a closing `usage:` line (LLM calls, tokens, gateway-reported
cost, searches) go to stderr, so stdout stays clean JSON when `--output` is
left off.

`precis research` runs the research step alone and prints what the notes
would be written from — the pages found, excerpted and numbered S1…:

```
docker run --rm ... --env-file .env \
  -v "$(pwd)/known-files:/input:ro" -v precis-data:/data \
  precis research /input/thinking-fast-and-slow.json > research.txt
```

### Recommended hardening

Generation feeds untrusted web content into an LLM, so run it locked down.
The image runs as the non-root `precis` user and only writes to `/output`,
`/data` and `/tmp`; the rest is up to whoever runs the container:

| Flag | Why |
|------|-----|
| `--read-only` | Root filesystem read-only; the mounts are the only writable paths. |
| `--tmpfs /tmp:rw,noexec,nosuid,size=64m` | Scratch space, in memory, not executable, capped. |
| `--cap-drop=ALL` | No Linux capabilities — precis needs none. |
| `--security-opt no-new-privileges` | Nothing inside can gain privileges. |
| `--pids-limit=256` | Caps processes. |
| `--memory=512m --memory-swap=512m` | Caps memory, swap included. |

Mount the known-files read-only (`:ro`). Pass in only the variables precis
reads — `--env-file .env` is fine with precis's own `.env`, but if yours
holds anything else, name them instead (`-e PRECIS_LLM_API_KEY -e
PRECIS_LLM_MODEL -e PRECIS_SEARCH_API_KEY`). This doesn't cut network
access: the LLM and search APIs need it.

`precis-data` must be a **named volume**, not a host bind mount: the image's
non-root user only gets write access to `/data` on a named volume's first
creation. A bind-mounted directory must be owned by uid 1000.

## Evals

`evals/` holds the eval set — 8 well-known books (`books/`, known-files)
with a reference summary each (`references/`) — and the runs generated from
it (`runs/<label>/`). See [evals/README.md](evals/README.md). Both commands
call paid models, so they run in Docker, with `evals/` bind-mounted at
`/evals` (writable by uid 1000):

```
docker run --rm ... --env-file .env \
  -v "$(pwd)/evals:/evals" -v precis-data:/data \
  precis eval run sonnet-5
# generates every eval book into evals/runs/sonnet-5/ with PRECIS_LLM_MODEL,
# skipping books already there; --book <slug> for just one

docker run --rm ... --env-file .env -v "$(pwd)/evals:/evals" \
  precis eval judge haiku-4-5 sonnet-5
# judges the first run against the second with PRECIS_JUDGE_MODEL
```

Research is cached per book, not per run, so runs comparing models write
from identical research; `eval run --fresh` searches again.
