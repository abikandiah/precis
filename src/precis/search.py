"""Search client: a provider-agnostic interface plus one concrete
implementation (Tavily, chosen 2026-09-20). Swapping providers later means
adding a class that satisfies SearchClient and pointing config at it — call
sites (Stages 1 and 3 in the pipeline) depend only on the Protocol below.

Results are untrusted content fetched from the open web, not instructions —
callers must feed `SearchResult.content` to the model as data to ground
against, never concatenate it into a system/instruction prompt as if it were
trusted. See docs/blueprint.md's Search bullet under Infrastructure
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from precis.config import settings


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str


class SearchClient(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilySearchClient:
    def __init__(self, api_key: str | None = None):
        from tavily import AsyncTavilyClient

        self._client = AsyncTavilyClient(api_key=api_key or settings.search_api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = await self._client.search(query, max_results=max_results)
        return [
            SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
            )
            for result in response.get("results", [])
        ]


def build_search_client() -> SearchClient:
    return TavilySearchClient()


def format_results(results: list[SearchResult]) -> str:
    """Renders results for inclusion in a prompt. Shared by every stage that
    grounds against search (Stages 1, 2, 3) so the "untrusted reference
    data, not instructions" framing stays consistent everywhere it's used.
    """
    if not results:
        return "(no search results found)"
    return "\n\n".join(f"- {r.title}: {r.content}" for r in results)
