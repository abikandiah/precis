"""Per-run cost and usage accounting: LLM calls, tokens, the gateway's own
reported cost, and searches with the credits they use.

`track()` opens a scope; every LLM response (llm._create) and search
(search.TavilySearchClient) inside it is recorded onto that scope's `Usage`.
Scoped through a ContextVar rather than a module global so an eval can keep a
book's generation apart from the judge calls that follow it. asyncio tasks
copy the context they're created in, so research's parallel searches all
record onto the one `Usage` their run opened.

Cost is whatever the gateway reports per response (OpenRouter's
`usage.cost`, in USD) — measured, not estimated from a price table. A
response with no reported cost is counted in `cost_missing` so a total that
undercounts says so. Searches aren't priced: they run on Tavily's free
monthly credits, so they're counted in credits against that allowance.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any

# Research's advanced searches cost 2 Tavily credits each.
CREDITS_PER_SEARCH = 2


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

    def to_dict(self) -> dict[str, Any]:
        # A sum of per-call floats; rounded so metrics.json doesn't carry
        # float noise like 0.30000000000000004.
        return {**asdict(self), "llm_cost_usd": round(self.llm_cost_usd, 6)}

    def summary(self) -> str:
        tokens = f"{self.prompt_tokens:,} in / {self.completion_tokens:,} out tokens"
        if self.cached_tokens:
            tokens += f" ({self.cached_tokens:,} cached)"
        missing = f", {self.cost_missing} call(s) reported no cost" if self.cost_missing else ""
        return (
            f"usage: {self.llm_calls} LLM call(s), {tokens}, ${self.llm_cost_usd:.4f}{missing}; "
            f"{self.searches} search(es), {self.search_credits} credit(s)"
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


def record_search() -> None:
    usage = _current.get()
    if usage is None:
        return
    usage.searches += 1
    usage.search_credits += CREDITS_PER_SEARCH
