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

from precis.schema import KnownFile

OPEN_LIBRARY_URL = "https://openlibrary.org/api/books"
PLACEHOLDER = "TODO: fill in by hand"
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


def _lookup_open_library(isbn: str) -> dict:
    url = f"{OPEN_LIBRARY_URL}?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
    try:
        with urllib.request.urlopen(url, timeout=_LOOKUP_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        # Deterministic call to a known API, but the network itself isn't
        # guaranteed — a lookup failure just means an all-placeholder
        # known-file, not a hard error. The reader fills it in by hand
        # either way.
        return {}

    record = body.get(f"ISBN:{isbn}")
    if not record:
        return {}

    result: dict = {}
    if title := record.get("title"):
        result["title"] = title
    if authors := record.get("authors"):
        result["author"] = ", ".join(a["name"] for a in authors if a.get("name"))
    publish_date = record.get("publish_date")
    if publish_date and (match := re.search(r"\d{4}", publish_date)):
        result["year"] = int(match.group())
    if (page_count := record.get("number_of_pages")) is not None:
        result["page_count"] = page_count
    return result


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

    return problems


def ensure_ready(known_file: KnownFile) -> None:
    """Convenience wrapper: raises ValueError with all problems joined if
    the known-file isn't ready for generation.
    """
    problems = preflight_check(known_file)
    if problems:
        raise ValueError("known-file is not ready for generation:\n- " + "\n- ".join(problems))
