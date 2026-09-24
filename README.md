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
  --env-file .env \
  -v "$(pwd)/known-files:/input:ro" \
  -v "$(pwd)/output:/output" \
  -v precis-checkpoints:/data \
  precis generate /input/mybook.json --output /output/mybook.json
```

`precis-checkpoints` must be a **named volume** (as above), not a host bind
mount (`-v ./somedir:/data`) — the image runs as a non-root user and only
gets write access to `/data` on a named volume's first creation. A bind
mount keeps the host directory's own ownership, which will fail to write
the checkpoint file unless that directory is already owned by uid 1000.

`--trust-known` skips Stage 1 (verify) for a known-file you're already
confident about; `--fresh` discards any existing checkpoint for that
known-file instead of resuming it. `generate-chapter` (targeted regeneration
of one chapter) takes the same mounts.

If a run is interrupted (hits `PRECIS_RUN_BUDGET_SECONDS`, the container is
killed, etc.), rerunning the identical command against the same
`precis-checkpoints` volume resumes from the last completed pipeline stage
rather than starting over.

Both `generate` and `generate-chapter` print progress to stderr as they
run (verify/chapter/synthesize/assemble completions) — stdout stays clean
JSON, so piping `--output`-less output elsewhere still works.

## Cleaning up checkpoints

Nothing deletes a run's checkpoint automatically — that's what makes resume
work, but it also means `precis-checkpoints` grows forever otherwise, since
every generated book (and every edited draft of every known-file along the
way) gets its own entry:

```
precis checkpoints                    # list every thread: id, status, last updated
precis checkpoints --prune            # delete finished threads (nothing left to resume)
precis checkpoints --prune --older-than-days 30
precis checkpoints --prune --include-incomplete   # also delete in-progress threads —
                                                   # forfeits resuming them, not just disk space
```

Only threads that finished (reached the assemble stage) are pruned by
default; an in-progress thread is exactly what the resume behavior above
depends on, so deleting one needs the explicit `--include-incomplete` flag.
