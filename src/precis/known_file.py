"""Phases 1 and 2: known-file creation and structural preflight.

Both run on the host, not in Docker — see the "Docker boundary" section of
docs/blueprint.md. Neither calls an LLM; phase 1 is one deterministic HTTP
call to a well-defined public API, phase 2 is pure local JSON shape-checking.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Literal

from precis.schema import PLACEHOLDER, KnownFile, invalid_chapter_numbers

OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPEN_LIBRARY_EDITION_URL = "https://openlibrary.org/isbn"
_LOOKUP_TIMEOUT_SECONDS = 10


def create_known_file(
    isbn: str,
    kind: Literal["fiction", "non-fiction"],
    narrative: bool = False,
) -> KnownFile:
    """Phase 1. Looks up `isbn` via Open Library and returns a KnownFile
    with whatever fields it could find; anything it couldn't gets a
    placeholder (str fields) or is left None (int fields — None already
    means "not filled in", no fake placeholder needed). `chapters` is
    always empty here; the reader fills it in before phase 2 runs.
    """
    data = _lookup_open_library(isbn)
    return KnownFile(
        isbn=isbn,
        title=data.get("title", PLACEHOLDER),
        author=data.get("author", PLACEHOLDER),
        year=data.get("year"),
        page_count=data.get("page_count"),
        chapters=[],
        kind=kind,
        narrative=narrative,
    )


def _fetch_json(url: str) -> dict | None:
    """GETs `url` and returns the decoded body, or None on any failure —
    network error, non-JSON body, or a JSON body that isn't an object (a
    malformed or unexpected API response shouldn't crash the lookup, just
    degrade it like any other failure mode).
    """
    try:
        with urllib.request.urlopen(url, timeout=_LOOKUP_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    return body if isinstance(body, dict) else None


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
    """Phase 2. Returns a list of problems blocking generation — empty
    means the known-file is ready to submit. Never raises; the caller (CLI)
    decides how to present the list.
    """
    problems: list[str] = []

    if not known_file.isbn:
        problems.append("isbn is required")

    if known_file.kind == "fiction" and known_file.narrative:
        problems.append(
            "narrative: true only means something for kind: non-fiction — "
            "it's set here alongside kind: fiction, which is almost "
            "certainly a misunderstanding of the field, not intentional"
        )

    if known_file.is_full_nonfiction_path and not known_file.chapters:
        problems.append(
            "chapters is required and must be non-empty for the "
            "non-fiction full study-guide path (kind: non-fiction, "
            "narrative: false)"
        )

    if known_file.kind == "fiction" and known_file.parts:
        problems.append(
            "parts is not meaningful for kind: fiction — fiction's parts "
            "are spoiler-safe structural beats the model invents, not a "
            "fact to supply by hand"
        )

    if known_file.kind == "non-fiction" and known_file.narrative and any(
        part.chapter_numbers is not None for part in known_file.parts
    ):
        problems.append(
            "chapter_numbers on parts requires a known chapter list, which "
            "narrative non-fiction doesn't have — leave chapter_numbers "
            "unset on parts for narrative books"
        )

    if known_file.is_full_nonfiction_path and known_file.parts:
        valid_numbers = set(range(1, len(known_file.chapters) + 1))
        for part in known_file.parts:
            for n in invalid_chapter_numbers(part.chapter_numbers, valid_numbers):
                problems.append(
                    f"part {part.title!r} references chapter number "
                    f"{n}, which doesn't exist in chapters (1-{len(known_file.chapters)})"
                )

    # Not a correctness requirement for Stage 3 (_finalize_parts matches by
    # position, not title), but two parts sharing a title is almost always
    # an authoring mistake worth catching here rather than shipping a
    # confusing output with two identically-named parts.
    titles = [part.title for part in known_file.parts]
    if len(titles) != len(set(titles)):
        dupes = {t for t in titles if titles.count(t) > 1}
        problems.append(f"parts has duplicate titles: {sorted(dupes)!r} — each part title must be unique")

    return problems


def ensure_ready(known_file: KnownFile) -> None:
    """Convenience wrapper: raises ValueError with all problems joined if
    the known-file isn't ready for generation.
    """
    problems = preflight_check(known_file)
    if problems:
        raise ValueError("known-file is not ready for generation:\n- " + "\n- ".join(problems))
