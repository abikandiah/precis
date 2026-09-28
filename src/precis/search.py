"""Search client: a provider-agnostic interface plus one concrete
implementation (Tavily). Swapping providers means adding a class that
satisfies SearchClient — research.py depends only on the Protocol — plus
the text-matching helpers research uses to tell pages about a book from
pages about its topic.

Results are untrusted content fetched from the open web, not instructions:
research.Research.render frames them as reference data for the model.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from precis import usage
from precis.config import settings
from precis.schema import PLACEHOLDER


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    # The provider's query-relevant excerpt of the page — what relevance is
    # judged on.
    content: str
    # The page's full text — what research reads.
    raw_content: str = ""


class SearchClient(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilySearchClient:
    """Advanced depth (better-matched excerpts, 2 credits) with each page's
    full text.
    """

    def __init__(self, api_key: str | None = None):
        from tavily import AsyncTavilyClient

        self._client = AsyncTavilyClient(api_key=api_key or settings.search_api_key)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = await self._client.search(
            query, max_results=max_results, search_depth="advanced", include_raw_content="text"
        )
        usage.record_search()
        return [
            SearchResult(
                title=result.get("title") or "",
                url=result.get("url") or "",
                content=result.get("content") or "",
                raw_content=result.get("raw_content") or "",
            )
            for result in response.get("results", [])
        ]


def build_search_client() -> SearchClient:
    return TavilySearchClient()


# Name suffixes and contributor-role words that aren't a surname — "Martin
# Luther King Jr." should match on "king", "Jane Doe (Translator)" or
# "edited by Jane Doe" on "doe".
_NOT_A_SURNAME = {
    "jr", "sr", "ii", "iii", "iv", "phd", "md",
    "ed", "eds", "editor", "editors", "edited", "trans", "translator", "translators", "translated",
    "foreword", "introduction", "illustrator", "illustrated", "by",
}  # fmt: skip

# Titles before a name — "Dr. Tim Spector" is Tim Spector. Dropped only
# from the front, and never the last word, so they can't cost a surname.
_HONORIFICS = {"dr", "prof", "professor", "sir", "dame", "lord", "lady", "mr", "mrs", "ms", "miss", "rev", "reverend"}

# Leading articles dropped before matching a title — a page saying "Diet Myth"
# is about "The Diet Myth" just as much.
_ARTICLES = {"the", "a", "an"}


def short_title(title: str | None) -> str:
    """The title without its subtitle or a trailing parenthetical — "The
    Diet Myth: The Real Science Behind What We Eat" -> "The Diet Myth",
    "Thinking, Fast and Slow (Revised Edition)" -> "Thinking, Fast and
    Slow". Search engines match the short form far more often than a long
    quoted subtitle, and pages rarely repeat an edition note. Only a
    trailing one: "The (Mis)Behavior of Markets" keeps its own. Empty for a
    missing or placeholder title.
    """
    if not title or title == PLACEHOLDER:
        return ""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title.split(":", 1)[0]).strip()


def normalize_text(text: str) -> str:
    """Lowercase, accent-free words separated by single spaces, so matching
    ignores punctuation, apostrophe style, "&" vs "and", stray whitespace and
    accents ("García Márquez" vs the "Garcia Marquez" of English pages/URLs).
    """
    folded = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[\W_]+", " ", folded.lower().replace("&", " and ")).split())


def contains(haystack: str, needle: str) -> bool:
    """Whole-word containment on normalize_text()d text."""
    return bool(needle) and f" {needle} " in f" {haystack} "


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """How much two word sets share: Jaccard similarity, 0 when either is empty."""
    return len(a & b) / len(a | b) if a and b else 0.0


def title_key(title: str) -> str:
    words = normalize_text(title).split()
    if len(words) > 1 and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def author_names(author: str | None) -> list[list[str]]:
    """Each credited person's name as normalize_text() words, given names
    first, without role words ("editor", "Jr.") or titles ("Dr.") — "Spector, Tim" is
    ["tim", "spector"]. Empty for a missing or placeholder author.
    """
    if not author or author == PLACEHOLDER:
        return []
    author = re.sub(r"\([^)]*\)", " ", author)
    pieces = [p.strip() for p in re.split(r",|;|&|\band\b|\bwith\b", author) if p.strip()]
    # "Spector, Tim" is one person written last-name-first, not two authors
    # "Spector" and "Tim" — the only comma form with a one-word first piece.
    # (Open Library's multi-author join is "Full Name, Full Name".)
    if "," in author and len(pieces) == 2 and len(pieces[0].split()) == 1 and not re.search(r";|&|\band\b|\bwith\b", author):
        pieces = [f"{pieces[1]} {pieces[0]}"]
    names = []
    for piece in pieces:
        words = [w for w in normalize_text(piece).split() if w not in _NOT_A_SURNAME]
        while len(words) > 1 and words[0] in _HONORIFICS:
            words.pop(0)
        if words:
            names.append(words)
    return names


def author_surnames(author: str | None) -> list[str]:
    return [words[-1] for words in author_names(author) if len(words[-1]) >= 2]


def _result_text(result: SearchResult) -> str:
    # Not raw_content: a whole page (a "best books" list, say) names many
    # books, so relevance is judged on the excerpt the provider matched to
    # the query.
    return _normalized_result_text(result.title, result.url, result.content)


@functools.lru_cache(maxsize=256)
def _normalized_result_text(title: str, url: str, content: str) -> str:
    # Cached: one result goes through several of these checks in turn
    # (mentions_title, then is_book_relevant), and normalizing is the
    # costly part. Keyed on the fields used, not the whole result, so
    # the cache never holds a result's full page text.
    return normalize_text(f"{title} {url} {content}")


def mentions_title(result: SearchResult, title: str | None) -> bool:
    """Whether a result names the book's short title, whoever it credits as
    author — looser than is_book_relevant, for checking the author claim
    itself (see research.py).
    """
    return contains(_result_text(result), title_key(short_title(title)))


def _names_full_title(text: str, title: str | None) -> bool:
    """A normalize_text()d result names the full title — only meaningful
    (distinctive) when the title has a subtitle.
    """
    return title is not None and ":" in title and contains(text, title_key(title))


def is_book_relevant(result: SearchResult, *, title: str | None, author: str | None) -> bool:
    """Whether a result is actually about this book, not just its topic.

    Without this, a nutrition book's research fills up with generic health
    pages that say nothing about the book.

    A result counts if it names the author's surname *and* the short title —
    neither alone is enough: "Diet Myth" matches every "diet myths" listicle,
    and a surname like Brown matches every "brown rice" page. The full title
    alone also counts when it has a subtitle, which is distinctive by itself.
    """
    text = _result_text(result)
    surnames = author_surnames(author)
    short = title_key(short_title(title))
    if short and _names_full_title(text, title):
        return True
    if not surnames:
        return False
    names_author = any(contains(text, s) for s in surnames)
    return names_author and (not short or contains(text, short))
