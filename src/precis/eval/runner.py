"""`precis eval run`: generate every eval book and store the outputs plus
per-book metrics under `runs/<label>/`.

Books run one after another, each in its own usage scope. A book whose
output already exists is skipped, so an interrupted or partly failed run is
finished by rerunning it without paying for the books already done; delete
a book's files to redo it.

Research is cached per book, not per run, so runs comparing models or
prompts write from identical research; `--fresh` searches again.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from precis import usage
from precis.config import settings
from precis.eval.data import EvalBook, book_path, metrics_path, read_metrics, write_json
from precis.eval.metrics import book_metrics
from precis.generate import generate

ProgressCallback = Callable[[str], None]


async def run_book(
    evals_dir: str | Path,
    label: str,
    book: EvalBook,
    *,
    trust_known: bool,
    fresh: bool,
    on_progress: ProgressCallback,
) -> dict[str, Any]:
    def progress(message: str) -> None:
        on_progress(f"[{book.slug}] {message}")

    metrics: dict[str, Any] = {
        "slug": book.slug,
        "label": label,
        "model": settings.llm_model,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "error": None,
    }
    started = time.monotonic()
    with usage.track() as book_usage:
        try:
            generated = await generate(
                book.known_file, slug=book.slug, trust_known=trust_known, fresh=fresh, on_progress=progress
            )
            output = generated.book.model_dump(exclude_none=True)
            metrics["research"] = {
                "fingerprint": generated.research.fingerprint,
                "sources": len(generated.research.sources),
            }
        except Exception as exc:  # noqa: BLE001 — one book failing is a result to record, not a reason to stop
            output = None
            metrics["error"] = f"{type(exc).__name__}: {exc}"
            progress(f"failed: {metrics['error']}")
    metrics["duration_seconds"] = round(time.monotonic() - started, 1)
    metrics["usage"] = book_usage.to_dict()
    progress(book_usage.summary())
    if output is not None:
        metrics.update(book_metrics(output))
        try:
            write_json(book_path(evals_dir, label, book.slug), output)
        except OSError as exc:
            metrics["error"] = f"couldn't save the book: {exc}"
            progress(f"failed: {metrics['error']}")
    try:
        write_json(metrics_path(evals_dir, label, book.slug), metrics)
    except OSError as exc:
        progress(f"couldn't save the metrics: {exc}")
    return metrics


async def run_eval(
    evals_dir: str | Path,
    label: str,
    books: list[EvalBook],
    *,
    trust_known: bool,
    fresh: bool = False,
    on_progress: ProgressCallback,
) -> list[dict[str, Any]]:
    """Returns every book's metrics, including ones skipped as already done."""
    results = []
    for book in books:
        if book_path(evals_dir, label, book.slug).exists():
            on_progress(f"[{book.slug}] already generated in run {label!r}, skipping")
            results.append(read_metrics(metrics_path(evals_dir, label, book.slug)) or {"slug": book.slug})
            continue
        results.append(
            await run_book(evals_dir, label, book, trust_known=trust_known, fresh=fresh, on_progress=on_progress)
        )
    return results
