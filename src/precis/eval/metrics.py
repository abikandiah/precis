"""Quality metrics computed in code from a generated book — no model call.

Works on the book as a plain dict (its `ideas` are `{title, summary,
evidence, sources}` objects), so a run's stored JSON is measured as written.
"""

from __future__ import annotations

from itertools import combinations
from statistics import mean
from typing import Any

# One definition of "near-duplicate" for the pipeline's check and the eval
# metric, so the two agree.
from precis.checks import DUPLICATE_OVERLAP, content_words, overlap


def idea_text(idea: dict[str, Any]) -> str:
    return f"{idea.get('title', '')} {idea.get('summary', '')}"


def duplicate_idea_rate(book: dict[str, Any]) -> float | None:
    """Share of the book's ideas that near-duplicate another of its ideas.
    None for a book without ideas.
    """
    words = [content_words(idea_text(i)) for i in book.get("ideas") or []]
    if not words:
        return None
    duplicated: set[int] = set()
    for i, j in combinations(range(len(words)), 2):
        if overlap(words[i], words[j]) >= DUPLICATE_OVERLAP:
            duplicated.update((i, j))
    return len(duplicated) / len(words)


def citation_coverage(book: dict[str, Any]) -> float | None:
    """Share of ideas citing at least one research source. None for a book
    without ideas.
    """
    ideas = book.get("ideas") or []
    if not ideas:
        return None
    return sum(1 for i in ideas if i.get("sources")) / len(ideas)


def book_metrics(book: dict[str, Any]) -> dict[str, Any]:
    return {
        "ideas": len(book.get("ideas") or []),
        "key_claims": len(book.get("key_claims_for_review") or []),
        "warnings": len(book.get("warnings") or []),
        "duplicate_idea_rate": duplicate_idea_rate(book),
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
        "mean_ideas": _mean([m.get("ideas") for m in ok]),
        "warnings": sum(m.get("warnings", 0) for m in ok),
        "mean_duplicate_idea_rate": _mean([m.get("duplicate_idea_rate") for m in ok]),
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
        f"{fmt(summary['mean_ideas'], '.1f')} ideas per book; {summary['warnings']} warning(s); "
        f"duplicate ideas {fmt(summary['mean_duplicate_idea_rate'], '.1%')}; "
        f"cited ideas {fmt(summary['mean_citation_coverage'], '.1%')}"
    )
