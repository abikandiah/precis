"""Quality metrics computed in code from a generated book — no model call.

Works on the book as a plain dict so the same code measures v1 output
(`key_points` are strings) and v2 output (`key_points` are
`{point, evidence, sources}` objects).
"""

from __future__ import annotations

import re
from itertools import combinations
from statistics import mean
from typing import Any

# Two points count as near-duplicates at this Jaccard overlap of their
# content words — enough shared vocabulary that they say the same thing.
DUPLICATE_OVERLAP = 0.5

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does", "for", "from", "has",
        "have", "how", "in", "into", "is", "it", "its", "more", "most", "not", "of", "on", "or", "our", "so",
        "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "was",
        "we", "were", "what", "when", "which", "who", "why", "will", "with", "without", "you", "your",
    }
)


def point_text(point: str | dict[str, Any]) -> str:
    return point if isinstance(point, str) else str(point.get("point", ""))


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2 and w not in _STOPWORDS)


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _all_points(book: dict[str, Any]) -> list[str | dict[str, Any]]:
    return [p for chapter in book.get("chapters") or [] for p in chapter.get("key_points") or []]


def duplicate_point_rate(book: dict[str, Any]) -> float | None:
    """Share of the book's key points that near-duplicate another key point
    anywhere in the book. None for a book without chapters.
    """
    words = [content_words(point_text(p)) for p in _all_points(book)]
    if not words:
        return None
    duplicated: set[int] = set()
    for i, j in combinations(range(len(words)), 2):
        if overlap(words[i], words[j]) >= DUPLICATE_OVERLAP:
            duplicated.update((i, j))
    return len(duplicated) / len(words)


def citation_coverage(book: dict[str, Any]) -> float | None:
    """Share of key points citing at least one research source. None when
    there are no points, or they carry no `sources` field (v1 output).
    """
    points = _all_points(book)
    if not points or not all(isinstance(p, dict) and "sources" in p for p in points):
        return None
    return sum(1 for p in points if isinstance(p, dict) and p["sources"]) / len(points)


def book_metrics(book: dict[str, Any]) -> dict[str, Any]:
    chapters = book.get("chapters") or []
    return {
        "chapters": len(chapters),
        "key_points": len(_all_points(book)),
        "flagged_chapters": sum(1 for c in chapters if c.get("quality_flag")),
        "warnings": len(book.get("warnings") or []),
        "duplicate_point_rate": duplicate_point_rate(book),
        "citation_coverage": citation_coverage(book),
    }


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """One run's per-book metrics rolled up. Failed books count toward
    `failed` and toward cost (the money was spent), not the quality means.
    """
    ok = [m for m in metrics if not m.get("error")]
    costs = [m["usage"]["total_cost_usd"] for m in metrics if m.get("usage")]
    return {
        "books": len(metrics),
        "failed": len(metrics) - len(ok),
        "total_cost_usd": sum(costs),
        "mean_cost_usd": _mean(costs),
        "max_cost_usd": max(costs, default=None),
        "mean_llm_calls": _mean([m["usage"]["llm_calls"] for m in metrics if m.get("usage")]),
        "mean_searches": _mean([m["usage"]["searches"] for m in metrics if m.get("usage")]),
        "mean_duration_seconds": _mean([m.get("duration_seconds") for m in ok]),
        "flagged_chapters": sum(m.get("flagged_chapters", 0) for m in ok),
        "warnings": sum(m.get("warnings", 0) for m in ok),
        "mean_duplicate_point_rate": _mean([m.get("duplicate_point_rate") for m in ok]),
        "mean_citation_coverage": _mean([m.get("citation_coverage") for m in ok]),
    }


def format_summary(label: str, summary: dict[str, Any]) -> str:
    def fmt(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    return (
        f"{label}: {summary['books']} book(s), {summary['failed']} failed; "
        f"cost ${summary['total_cost_usd']:.4f} total, ${fmt(summary['mean_cost_usd'], '.4f')} mean, "
        f"${fmt(summary['max_cost_usd'], '.4f')} max; "
        f"{fmt(summary['mean_llm_calls'], '.1f')} LLM calls and {fmt(summary['mean_searches'], '.1f')} searches "
        f"per book; {fmt(summary['mean_duration_seconds'], '.0f')} seconds mean; "
        f"{summary['flagged_chapters']} flagged chapter(s), {summary['warnings']} warning(s); "
        f"duplicate points {fmt(summary['mean_duplicate_point_rate'], '.1%')}; "
        f"cited points {fmt(summary['mean_citation_coverage'], '.1%')}"
    )
