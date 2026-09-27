# precis

Standalone Python module: reads a known-file (ISBN + a hand-filled chapter
list), always runs generation in Docker, writes a validated book JSON (or a
single chapter JSON for targeted regeneration of one chapter).

See [docs/blueprint.md](docs/blueprint.md) for the full design.

## Using the generation image

```
cp .env.example .env   # fill in PRECIS_LLM_API_KEY, PRECIS_LLM_MODEL, PRECIS_SEARCH_API_KEY
docker build -t precis .
```

Phase 3 (generation) is the only part that runs in Docker — it's the only
part that calls an LLM. Phases 1–2 (`create-known-file`, and the structural
preflight check that runs automatically before generation) are deterministic
and fine to run on a host directly (`uv run precis create-known-file ...`),
no container needed.

A generation run needs three things mounted/passed in: the known-file as
input, an output path to write the result to, and a persistent volume for
the checkpoint database (so an interrupted run can resume instead of
starting over — see docs/blueprint.md's Orchestration section):

```
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
  -v precis-checkpoints:/data \
  precis generate /input/mybook.json --output /output/mybook.json
```

### Recommended hardening

Generation feeds untrusted third-party content (search results) into an
LLM, so run it locked down. The image does what an image can on its own —
it runs as the non-root `precis` user and only ever writes to `/output`,
`/data` and `/tmp` — but Docker only lets whoever runs the container
apply the rest, so consumers should pass these flags on every `generate`
and `generate-chapter` run (as above):

| Flag | Why |
|------|-----|
| `--read-only` | Root filesystem read-only; the mounts below are the only writable paths. |
| `--tmpfs /tmp:rw,noexec,nosuid,size=64m` | Scratch space precis needs, in memory, not executable, capped. |
| `--cap-drop=ALL` | No Linux capabilities — precis needs none. |
| `--security-opt no-new-privileges` | Nothing inside can gain privileges (setuid binaries etc.). |
| `--pids-limit=256` | Caps processes, so a runaway can't fork-bomb the host. |
| `--memory=512m --memory-swap=512m` | Caps memory, swap included; a generation run fits well within it. |

Mount the known-file (or its directory) read-only (`:ro`); only `/output`
and `/data` need to be writable. Pass in only the variables precis reads —
`--env-file .env` is fine with precis's own `.env` (see `.env.example`), but
if yours holds anything else, name the variables instead
(`-e PRECIS_LLM_API_KEY -e PRECIS_LLM_MODEL -e PRECIS_SEARCH_API_KEY`) so
nothing unrelated is forwarded into the container.

This doesn't cut network access: the LLM and search APIs need it, and the
container holds those API keys. The point is protecting the host and its
files, not the keys.

`precis-checkpoints` must be a **named volume** (as above), not a host bind
mount (`-v ./somedir:/data`) — the image runs as a non-root user and only
gets write access to `/data` on a named volume's first creation. A bind
mount keeps the host directory's own ownership, which will fail to write
the checkpoint file unless that directory is already owned by uid 1000.

Stage 1 (verify) checks the known-file against search results and lists
every issue it finds. It fails the run only when a source about this exact
book contradicts the author or the chapter/part list (showing what the
source says, and its URL), or when no search result mentions the book at
all. Anything it merely couldn't confirm — and any isbn/year/subtitle
difference, which vary by edition — is printed as a warning. `--trust-known`
skips Stage 1 for a known-file you've already checked against the book.
`generate-chapter` (targeted regeneration of one chapter) takes the same
mounts.

The known-file's filename is the book's slug, and a run's checkpoint is
named after it. The checkpoint only exists to resume an interrupted run
(`PRECIS_RUN_BUDGET_SECONDS`, a killed container, a failure partway) and is
deleted once the finished book is written, so rerunning after a success
generates the book again from scratch. Rerunning an interrupted run on the
same `precis-checkpoints` volume resumes where it stopped. If you edit the
known-file (anything but `notes`) after verify has passed, the rerun stops
and lists what changed; pass `--fresh` to discard the interrupted run and
start over. A run started with `--trust-known` only resumes with it.

Both `generate` and `generate-chapter` print progress to stderr as they
run (verify/chapter/synthesize/assemble completions) — stdout stays clean
JSON, so piping `--output`-less output elsewhere still works.

At the end of every `generate`/`generate-chapter` run (successful or not),
a `usage:` line on stderr reports LLM calls, tokens, the gateway-reported
cost and searches used.

## Research

`precis research` runs v2's research step alone and prints what the notes
will be written from — the pages found, excerpted and numbered S1…. Its
search results are cached under `/data/cache`, so run it with the same
volume as `generate` to reuse them; `--fresh` searches again.

```
docker run --rm ... --env-file .env \
  -v "$(pwd)/known-files:/input:ro" \
  -v precis-checkpoints:/data \
  precis research /input/mybook.json > research.txt
```

## Evals

`evals/` holds the eval set — 8 well-known books (`books/`, checked
known-files) with a reference summary each (`references/`) — and the runs
generated from it (`runs/<label>/`). See [evals/README.md](evals/README.md).
Both commands call paid models, so they run in Docker like `generate`,
with the repo's `evals/` bind-mounted at `/evals`:

```
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop=ALL --security-opt no-new-privileges --pids-limit=256 \
  --memory=512m --memory-swap=512m \
  --env-file .env \
  -v "$(pwd)/evals:/evals" \
  precis eval run v2-sonnet-5
# generates every eval book into evals/runs/v2-sonnet-5/ with PRECIS_LLM_MODEL,
# skipping books already there; --book <slug> for just one

docker run --rm ... --env-file .env -v "$(pwd)/evals:/evals" \
  precis eval judge v2-haiku-4-5 v2-sonnet-5
# judges the first run against the second with PRECIS_JUDGE_MODEL
```

No pipeline is registered for `eval run` yet: the eval set is in the v2
shape (no chapters), and v2 registers once it runs end to end
(docs/v2-plan.md, Phase 4).

`evals/` must be writable by uid 1000 (the image's user), since it's a
bind mount.

## Cleaning up checkpoints

`generate` deletes a book's checkpoint once the book is written, so what
accumulates here is interrupted runs that were never rerun. Same named
volume as `generate` above — no other mount needed, and no `--env-file`
either, since this never calls an LLM:

```
docker run --rm -v precis-checkpoints:/data precis checkpoints
# list every thread: id, status, last updated

docker run --rm -v precis-checkpoints:/data precis checkpoints --prune
# delete finished threads (nothing left to resume)

docker run --rm -v precis-checkpoints:/data precis checkpoints --prune --older-than-days 30

docker run --rm -v precis-checkpoints:/data precis checkpoints --prune --include-incomplete
# also delete in-progress threads — forfeits resuming them, not just disk space
```

Only threads that finished (reached the assemble stage) are pruned by
default; an in-progress thread is exactly what the resume behavior above
depends on, so deleting one needs the explicit `--include-incomplete` flag.
