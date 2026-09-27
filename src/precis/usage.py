"""Per-run cost and usage accounting: LLM calls, tokens, the gateway's own
reported cost, and search credits.

`track()` opens a scope; every LLM response (llm._create) and search
(search.TavilySearchClient) inside it is recorded onto that scope's `Usage`.
Scoped through a ContextVar rather than a module global so an eval can keep a
book's generation apart from the judge calls that follow it. asyncio tasks
copy the context they're created in, so the parallel chapter drafts all
record onto the one `Usage` their run opened.

Cost is whatever the gateway reports per response (OpenRouter's
`usage.cost`, in USD) — measured, not estimated from a price table. A
response with no reported cost is counted in `cost_missing` so a total that
undercounts says so.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any

# Tavily pay-as-you-go price per credit; basic search costs 1 credit,
# advanced 2. Only used to put searches and LLM calls on one total.
SEARCH_USD_PER_CREDIT = 0.008


@dataclass
class Usage:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    llm_cost_usd: float = 0.0
    cost_missing: int = 0
    searches: int = 0
    search_credits: int = 0

    @property
    def search_cost_usd(self) -> float:
        return self.search_credits * SEARCH_USD_PER_CREDIT

    @property
    def total_cost_usd(self) -> float:
        return self.llm_cost_usd + self.search_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "search_cost_usd": round(self.search_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    def summary(self) -> str:
        tokens = f"{self.prompt_tokens:,} in / {self.completion_tokens:,} out tokens"
        if self.cached_tokens:
            tokens += f" ({self.cached_tokens:,} cached)"
        missing = f", {self.cost_missing} call(s) reported no cost" if self.cost_missing else ""
        return (
            f"usage: {self.llm_calls} LLM call(s), {tokens}, ${self.llm_cost_usd:.4f}{missing}; "
            f"{self.searches} search(es), {self.search_credits} credit(s) ~${self.search_cost_usd:.4f}; "
            f"total ~${self.total_cost_usd:.4f}"
        )


_current: ContextVar[Usage | None] = ContextVar("precis_usage", default=None)


@contextmanager
def track() -> Iterator[Usage]:
    usage = Usage()
    token = _current.set(usage)
    try:
        yield usage
    finally:
        _current.reset(token)


def _number(value: object) -> float | None:
    # bool is an int subclass; a mocked response's attributes are neither.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def record_llm_response(response: object) -> None:
    usage = _current.get()
    if usage is None:
        return
    usage.llm_calls += 1
    reported = getattr(response, "usage", None)
    usage.prompt_tokens += int(_number(getattr(reported, "prompt_tokens", None)) or 0)
    usage.completion_tokens += int(_number(getattr(reported, "completion_tokens", None)) or 0)
    details = getattr(reported, "prompt_tokens_details", None)
    usage.cached_tokens += int(_number(getattr(details, "cached_tokens", None)) or 0)
    extra = getattr(reported, "model_extra", None)
    cost = _number(extra.get("cost")) if isinstance(extra, dict) else None
    if cost is None:
        usage.cost_missing += 1
    else:
        usage.llm_cost_usd += cost


def record_search(*, deep: bool) -> None:
    usage = _current.get()
    if usage is None:
        return
    usage.searches += 1
    usage.search_credits += 2 if deep else 1
