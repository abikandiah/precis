"""`precis eval run`: generate every eval book with one pipeline and store
the outputs plus per-book metrics under `runs/<label>/`.

Books run one after another (a pipeline already parallelizes within a
book), each in its own usage scope. A book whose output already exists is
skipped, so an interrupted or partly failed run is finished by rerunning it
without paying for the books already done; delete a book's files to redo it.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from precis import usage
from precis.config import settings
from precis.eval.data import EvalBook, read_json, run_dir, write_json
from precis.eval.metrics import book_metrics
from precis.schema import KnownFile

ProgressCallback = Callable[[str], None]
Pipeline = Callable[[KnownFile, str, bool, ProgressCallback], Awaitable[dict[str, Any]]]


# The v2 pipeline registers here once it runs end to end (docs/v2-plan.md,
# Phase 4). v1 can't run on the chapter-less eval set, and there's no v1
# baseline to compare against.
PIPELINES: dict[str, Pipeline] = {}


def book_path(evals_dir: str | Path, label: str, slug: str) -> Path:
    return run_dir(evals_dir, label) / f"{slug}.json"


def metrics_path(evals_dir: str | Path, label: str, slug: str) -> Path:
    return run_dir(evals_dir, label) / f"{slug}.metrics.json"


async def run_book(
    evals_dir: str | Path,
    label: str,
    book: EvalBook,
    *,
    pipeline: str,
    trust_known: bool,
    on_progress: ProgressCallback,
) -> dict[str, Any]:
    def progress(message: str) -> None:
        on_progress(f"[{book.slug}] {message}")

    metrics: dict[str, Any] = {
        "slug": book.slug,
        "label": label,
        "pipeline": pipeline,
        "model": settings.llm_model,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "error": None,
    }
    started = time.monotonic()
    with usage.track() as book_usage:
        try:
            output = await PIPELINES[pipeline](book.known_file, f"eval-{label}-{book.slug}", trust_known, progress)
        except Exception as exc:  # noqa: BLE001 — one book failing is a result to record, not a reason to stop
            output = None
            metrics["error"] = f"{type(exc).__name__}: {exc}"
            progress(f"failed: {metrics['error']}")
    metrics["duration_seconds"] = round(time.monotonic() - started, 1)
    metrics["usage"] = book_usage.to_dict()
    progress(book_usage.summary())
    if output is not None:
        metrics.update(book_metrics(output))
        write_json(book_path(evals_dir, label, book.slug), output)
    write_json(metrics_path(evals_dir, label, book.slug), metrics)
    return metrics


async def run_eval(
    evals_dir: str | Path,
    label: str,
    books: list[EvalBook],
    *,
    pipeline: str,
    trust_known: bool,
    on_progress: ProgressCallback,
) -> list[dict[str, Any]]:
    """Returns every book's metrics, including ones skipped as already done."""
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r} (have: {', '.join(PIPELINES) or 'none yet'})")
    results = []
    for book in books:
        if book_path(evals_dir, label, book.slug).exists():
            on_progress(f"[{book.slug}] already generated in run {label!r}, skipping")
            results.append(read_json(metrics_path(evals_dir, label, book.slug)) or {"slug": book.slug})
            continue
        results.append(
            await run_book(evals_dir, label, book, pipeline=pipeline, trust_known=trust_known, on_progress=on_progress)
        )
    return results
