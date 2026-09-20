"""Shared helpers used by every pipeline node (verify/draft/synthesize/
assemble). Extracted once the pattern was proven identical across all four
stages, rather than kept as copy-pasted boilerplate in each one.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI

from precis.schema import KnownFile
from precis.search import SearchClient


def resolve_search_client(config: RunnableConfig | None, override: SearchClient | None) -> SearchClient | None:
    """`override` wins if given (tests / a direct non-graph caller);
    otherwise pulled from `config["configurable"]` — see
    `pipeline/graph.py`'s `run_whole_book`, which builds one client per run
    and threads it through config so every node (including every parallel
    per-chapter fan-out branch) shares it instead of building its own.
    `None` if neither is set, meaning the caller should build its own
    default.
    """
    if override is not None:
        return override
    return ((config or {}).get("configurable") or {}).get("search_client")


def resolve_llm_client(config: RunnableConfig | None, override: AsyncOpenAI | None) -> AsyncOpenAI | None:
    """Same resolution order as `resolve_search_client`, for the LLM client."""
    if override is not None:
        return override
    return ((config or {}).get("configurable") or {}).get("llm_client")


def book_header(known_file: KnownFile) -> str:
    """The 'Book: "title" by author' line every stage's prompt opens with —
    no trailing separator, since stages differ slightly in what follows it
    (a blank line, a Chapter: line, etc.) — callers add their own.
    """
    return f'Book: "{known_file.title}" by {known_file.author}'
