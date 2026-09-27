"""Known-file creation and preflight — both on the host, no model and no
search (see docs/blueprint.md's Docker boundary). Creation is a lookup
against Open Library, a well-defined public API; preflight is local
field-checking.
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from typing import Literal

from precis.schema import PLACEHOLDER, KnownFile

OPEN_LIBRARY_BASE_URL = "https://openlibrary.org"
OPEN_LIBRARY_SEARCH_URL = f"{OPEN_LIBRARY_BASE_URL}/search.json"
OPEN_LIBRARY_EDITION_URL = f"{OPEN_LIBRARY_BASE_URL}/isbn"
_LOOKUP_TIMEOUT_SECONDS = 10


def create_known_file(isbn: str, kind: Literal["fiction", "non-fiction"]) -> tuple[KnownFile, list[str]]:
    """Looks up `isbn` via Open Library and returns a KnownFile with
    whatever it found, plus notes for the reader about what still needs
    checking. A missing title or author gets a placeholder; a missing year
    or page count is left None.
    """
    data = _lookup_open_library(isbn)
    notes = [
        f"Open Library has no {field} for this isbn — fill it in by hand."
        for field in ("title", "author")
        if field not in data
    ]
    known_file = KnownFile(
        isbn=isbn,
        title=data.get("title", PLACEHOLDER),
        author=data.get("author", PLACEHOLDER),
        year=data.get("year"),
        page_count=data.get("page_count"),
        kind=kind,
    )
    return known_file, notes


def _fetch_json(url: str) -> dict | None:
    """GETs `url` and returns the decoded body, or None on any failure —
    network error, non-JSON body, or a JSON body that isn't an object (a
    malformed or unexpected API response shouldn't crash the lookup, just
    degrade it like any other failure mode).
    """
    # One retry, only for a reset connection — Open Library drops them
    # often enough under load to make a single attempt flaky. Not for a
    # timeout, DNS/SSL failure or HTTP error, which a retry won't fix and
    # would only double the wait.
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=_LOOKUP_TIMEOUT_SECONDS) as resp:
                body = json.loads(resp.read())
            return body if isinstance(body, dict) else None
        except urllib.error.HTTPError:
            return None
        except urllib.error.URLError as exc:
            if attempt or not isinstance(exc.reason, ConnectionResetError):
                return None
        except ConnectionResetError:
            if attempt:
                return None
        # Anything else — a truncated body (IncompleteRead), an SSL error
        # mid-read, bytes that aren't UTF-8/JSON — degrades the lookup the
        # same way rather than crashing a batch partway.
        except (OSError, http.client.HTTPException, ValueError):
            return None
    return None


def _lookup_open_library(isbn: str) -> dict:
    # Open Library's old bibkeys/jscmd=data "Books API" (api/books) has been
    # retired — it now 404s on every request. Two live replacements, used
    # together: search.json for title/author (stable across editions of the
    # same work), and the isbn/{isbn}.json edition record for year/page
    # count — search.json's first_publish_year/number_of_pages_median are
    # aggregated across every edition of the work, not the specific
    # printing the caller's ISBN identifies.
    result: dict = {}

    search_url = f"{OPEN_LIBRARY_SEARCH_URL}?isbn={isbn}&fields=title,author_name"
    search_body = _fetch_json(search_url)
    docs = search_body.get("docs") if search_body else None
    if docs:
        record = docs[0]
        if title := record.get("title"):
            result["title"] = title
        if authors := record.get("author_name"):
            result["author"] = ", ".join(authors)

    edition_body = _fetch_json(f"{OPEN_LIBRARY_EDITION_URL}/{isbn}.json")
    if edition_body:
        publish_date = edition_body.get("publish_date")
        if publish_date and (match := re.search(r"\d{4}", publish_date)):
            result["year"] = int(match.group())
        if (page_count := edition_body.get("number_of_pages")) is not None:
            result["page_count"] = page_count

    return result


def slugify_title(title: str | None) -> str:
    """Book title -> filename slug: lowercase, non-alphanumeric runs
    collapsed to single hyphens, e.g. "Guns, Germs, and Steel" ->
    "guns-germs-and-steel". Empty string for a missing title, so callers
    can use the result directly without a separate None-check.
    """
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def preflight_check(known_file: KnownFile) -> list[str]:
    """Problems blocking generation — empty means ready. Research searches
    for the book by its title and author and only keeps pages naming both,
    so neither can be missing or a placeholder.
    """
    problems: list[str] = []
    if not known_file.isbn:
        problems.append("isbn is required")
    if not known_file.has_title:
        problems.append("title is required (still missing or a placeholder)")
    if not known_file.has_author:
        problems.append("author is required (still missing or a placeholder)")
    return problems
