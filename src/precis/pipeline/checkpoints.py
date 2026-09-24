"""Checkpoint-store maintenance: listing and pruning thread history.

Every `precis generate` run gets its own thread in the checkpoint DB (see
graph.py's `thread_id_for`), and nothing deletes one automatically once the
run finishes — that's what makes resuming an interrupted run possible
without any extra bookkeeping, but it also means the DB grows without bound
across every book (and every edited draft of every known-file) ever run.
This module is the maintenance side of that tradeoff.

"Safe to prune" here means "has already reached assemble" — i.e. the run
produced a `book`, so there is nothing left to resume. A thread that hasn't
reached assemble is exactly what `precis generate`'s resume feature depends
on; deleting one doesn't just reclaim disk space, it forfeits resuming that
interrupted run, so it's excluded unless the caller explicitly opts in via
`include_incomplete`.

One residual risk `include_incomplete` doesn't cover: `thread_id_for()` is
content-deterministic, so a *completed* thread's id is reused verbatim if
`generate` is re-run on that known-file without `--fresh` — "completed" is
not the same as "dead." `prune_checkpoint_threads` re-fetches each thread's
latest checkpoint immediately before deleting it and skips any thread that
changed since it was listed, which narrows that race to "a concurrent run
progressed in between," but there's no cross-process lock here — running
`--prune` concurrently with a `generate` re-run of the exact same known-file
in a tight enough window is still possible to lose to. Safe for the
interactive, single-operator CLI use this was built for; not a guarantee
under concurrent automation against the same DB.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from precis.config import settings
from precis.pipeline.state import COMPLETED_STATE_KEY


@dataclass(frozen=True)
class CheckpointThreadSummary:
    thread_id: str
    last_updated: str  # ISO 8601, as written by langgraph's Checkpoint.ts
    completed: bool  # True once the thread's latest checkpoint has a `book`


def _ensure_checkpoint_dir() -> None:
    checkpoint_dir = os.path.dirname(settings.checkpoint_db_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)


def _summarize(thread_id: str, tup: CheckpointTuple) -> CheckpointThreadSummary:
    return CheckpointThreadSummary(
        thread_id=thread_id,
        last_updated=tup.checkpoint["ts"],
        completed=COMPLETED_STATE_KEY in tup.checkpoint["channel_values"],
    )


async def _distinct_thread_ids(checkpointer: AsyncSqliteSaver) -> list[str]:
    """`AsyncSqliteSaver` has no "distinct thread ids" or "latest per
    thread" query of its own — `alist(None)` returns every checkpoint ever
    written (every superstep of every thread), which would mean
    deserializing the whole history just to dedupe down to this same list.
    This reaches into the saver's own connection instead, for the one thing
    its public API can't do cheaply. `setup()` is idempotent (every public
    method already calls it first); called directly here since this
    bypasses those methods and needs the schema to exist itself.
    """
    await checkpointer.setup()
    async with checkpointer.conn.execute("SELECT DISTINCT thread_id FROM checkpoints") as cur:
        return [row[0] async for row in cur]


async def _latest_checkpoint(checkpointer: AsyncSqliteSaver, thread_id: str) -> CheckpointTuple | None:
    return await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})


async def list_checkpoint_threads() -> list[CheckpointThreadSummary]:
    """One summary per thread, from its latest checkpoint only — O(threads),
    not O(every checkpoint ever written).
    """
    _ensure_checkpoint_dir()
    summaries: list[CheckpointThreadSummary] = []
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        for thread_id in await _distinct_thread_ids(checkpointer):
            if tup := await _latest_checkpoint(checkpointer, thread_id):
                summaries.append(_summarize(thread_id, tup))
    return summaries


async def prune_checkpoint_threads(
    *, older_than_days: float | None = None, include_incomplete: bool = False
) -> list[CheckpointThreadSummary]:
    """Deletes every thread matching the filters, returning what was
    deleted. Completed-only and no age filter by default — the narrowest,
    always-safe-barring-the-race-described-above cleanup. Widening either
    filter is an explicit, separate choice by the caller.
    """
    _ensure_checkpoint_dir()
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days) if older_than_days is not None else None

    deleted: list[CheckpointThreadSummary] = []
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        for thread_id in await _distinct_thread_ids(checkpointer):
            tup = await _latest_checkpoint(checkpointer, thread_id)
            if tup is None:
                continue
            summary = _summarize(thread_id, tup)
            if not (summary.completed or include_incomplete):
                continue
            if cutoff is not None and datetime.fromisoformat(summary.last_updated) >= cutoff:
                continue

            # Re-fetch right before deleting: see the module docstring's
            # race note. If a newer checkpoint than the one just summarized
            # exists now, this thread is live again — skip it rather than
            # delete state a concurrent run may depend on.
            recheck = await _latest_checkpoint(checkpointer, thread_id)
            if recheck is None or recheck.checkpoint["ts"] != summary.last_updated:
                continue

            await checkpointer.adelete_thread(thread_id)
            deleted.append(summary)

    return deleted
