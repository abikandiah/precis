"""Phases 1 and 2: known-file creation and structural preflight.

Both run on the host, not in Docker — see the "Docker boundary" section of
docs/blueprint.md. Neither calls an LLM; phase 1 is deterministic HTTP calls
to a well-defined public API, phase 2 is pure local JSON shape-checking.
"""

from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from precis.schema import PLACEHOLDER, KnownFile, KnownPart, invalid_chapter_numbers
from precis.search import normalize_text

OPEN_LIBRARY_BASE_URL = "https://openlibrary.org"
OPEN_LIBRARY_SEARCH_URL = f"{OPEN_LIBRARY_BASE_URL}/search.json"
OPEN_LIBRARY_EDITION_URL = f"{OPEN_LIBRARY_BASE_URL}/isbn"
_LOOKUP_TIMEOUT_SECONDS = 10


def create_known_file(
    isbn: str,
    kind: Literal["fiction", "non-fiction"],
    narrative: bool = False,
) -> tuple[KnownFile, list[str]]:
    """Phase 1. Looks up `isbn` via Open Library and returns a KnownFile
    with whatever fields it could find, plus notes for the reader about
    what still needs checking. Anything it couldn't find gets a placeholder
    (str fields) or is left None (int fields — None already means "not
    filled in", no fake placeholder needed).

    `chapters`/`parts` are pre-filled from Open Library's table of contents
    when it has a clean one (see parse_table_of_contents) — for this
    edition, or failing that another edition of the same work, which the
    notes call out. Otherwise they're left empty for the reader to fill in.
    Either way the reader still reviews them before phase 2: Stage 1 checks
    them against search results, but they're treated as known fact from
    then on.
    """
    data = _lookup_open_library(isbn, want_toc=kind == "non-fiction")
    notes: list[str] = []
    chapters: list[str] = []
    parts: list[KnownPart] = []

    toc = data.get("toc")
    if toc:
        chapters, parts = toc.chapters, toc.parts
        if narrative:
            # Narrative non-fiction has no chapter list to bind parts to —
            # keep part titles only (see preflight_check).
            chapters = []
            parts = [KnownPart(title=p.title) for p in parts]
        found = [_count(len(chapters), "chapter")] if chapters else []
        if parts:
            found.append(_count(len(parts), "part"))
        if found:
            edition = "this edition" if toc.source_isbn == isbn else f"another edition ({toc.source_edition})"
            notes.append(
                f"{' and '.join(found)} pre-filled from Open Library's table of contents for {edition} "
                "— check them against your copy."
            )
    elif kind == "non-fiction" and not narrative:
        notes.append("Open Library has no usable table of contents for this book — fill in chapters by hand.")

    known_file = KnownFile(
        isbn=isbn,
        title=data.get("title", PLACEHOLDER),
        author=data.get("author", PLACEHOLDER),
        year=data.get("year"),
        page_count=data.get("page_count"),
        chapters=chapters,
        parts=parts,
        kind=kind,
        narrative=narrative,
    )
    return known_file, notes


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


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


def _lookup_open_library(isbn: str, *, want_toc: bool = False) -> dict:
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
        if want_toc and (toc := _find_table_of_contents(isbn, edition_body)):
            result["toc"] = toc

    return result


@dataclass(frozen=True)
class TableOfContents:
    chapters: list[str]
    parts: list[KnownPart]
    source_isbn: str | None  # the edition it came from — may not be the one asked for
    source_edition: str  # human-readable, for the reader-facing note


def _languages(edition: dict) -> set[str]:
    return {lang["key"] for lang in edition.get("languages") or [] if isinstance(lang, dict) and "key" in lang}


def _first_str(value: object) -> str | None:
    """The first string in an Open Library list field — hand-entered data
    sometimes has a bare string or other junk where a list belongs.
    """
    items = value if isinstance(value, list) else []
    return next((item for item in items if isinstance(item, str) and item), None)


def _find_table_of_contents(isbn: str, edition_body: dict) -> TableOfContents | None:
    """This edition's contents list, or failing that the first other
    edition of the same work that has a clean one. Chapter lists rarely
    change between printings of a work, but the reader is told when the
    list came from a different edition so they can check.
    """
    if (parsed := parse_table_of_contents(edition_body.get("table_of_contents"))) is not None:
        return TableOfContents(*parsed, source_isbn=isbn, source_edition=isbn)

    # A translation's chapter titles are no use — only borrow from an
    # edition in the same language(s). When this edition doesn't say, only
    # from editions that don't either: most works' other editions are
    # mostly translations, and guessing a language is worse than borrowing
    # nothing.
    languages = _languages(edition_body)

    for work in edition_body.get("works") or []:
        key = work.get("key") if isinstance(work, dict) else None
        if not key:
            continue
        editions = _fetch_json(f"{OPEN_LIBRARY_BASE_URL}{key}/editions.json?limit=100")
        entries = (editions or {}).get("entries")
        for edition in entries if isinstance(entries, list) else []:
            if not isinstance(edition, dict) or _languages(edition) != languages:
                continue
            if (parsed := parse_table_of_contents(edition.get("table_of_contents"))) is None:
                continue
            other_isbn = _first_str(edition.get("isbn_13")) or _first_str(edition.get("isbn_10"))
            published = edition.get("publish_date") if isinstance(edition.get("publish_date"), str) else None
            label = " ".join(x for x in (other_isbn and f"ISBN {other_isbn}", published) if x)
            return TableOfContents(*parsed, source_isbn=other_isbn, source_edition=label or str(edition.get("key", "?")))
    return None


# Front/back matter that isn't a chapter. Intros, prologues, epilogues and
# conclusions stay — they're usually substantive and the reader's own
# known-files list them as chapters.
# Single words that are also plausible chapter openings ("Copyright Wars",
# "Dedication and Grit") only count as the whole entry.
_NOT_A_CHAPTER_EXACT = {
    "index", "notes", "references", "sources", "contents", "glossary", "credits", "endnotes",
    "cover", "title page", "epigraph", "map", "maps", "timeline", "chronology",
    "copyright", "dedication", "permissions",
}  # fmt: skip
_NOT_A_CHAPTER_PREFIX = re.compile(
    r"^(acknowledg\w*|(selected )?bibliograph\w*|further reading|suggested reading|about the author|"
    r"preface|foreword|author s note|a note on|appendi\w*|notes (and|on sources)|sources and|picture credits|"
    r"illustration credits|list of (illustrations|figures|tables|maps)|also by|praise for)\b"
)
# Chapters that sit outside any part — a conclusion after the last part
# isn't part of it.
_STANDS_ALONE = re.compile(r"^(introduction|prologue|conclusions?|epilogue|afterword|coda)\b", re.IGNORECASE)
# A number as books print it: digits, a Roman numeral (spelled properly,
# non-empty — the lookbehind — and the whole word, so "Civil" or "Ill"
# isn't one) or a number word, hyphenated tens included ("Twenty-One").
_UNITS = "one|two|three|four|five|six|seven|eight|nine"
_NUMBER = (
    r"(?:\d+|(?=[ivxl]+\b)(?:xl|l?x{0,3})(?:ix|iv|v?i{0,3})(?<=[ivxl])"
    rf"|(?:twenty|thirty|forty|fifty)(?:-(?:{_UNITS}))?|{_UNITS}|ten|eleven|twelve|thirteen|fourteen|fifteen"
    r"|sixteen|seventeen|eighteen|nineteen)"
)
_DIVISION = r"(part|book)"  # "Part Two", "Book One"
_PART_LABEL = re.compile(rf"^{_DIVISION}\b", re.IGNORECASE)
# A single letter counts for parts ("Part A") — so "Part of the Problem"
# needs the word boundary after it to stay a chapter.
_PART_TITLE = re.compile(rf"^{_DIVISION}\s+({_NUMBER}|[a-z])\b\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
# "Chapter 3: …", "Chapter Three …" or "3. …" — at most three digits, so a
# title that starts with a year ("1914: The Guns of August") keeps it, and
# only real numbers, so "Chapter and Verse" keeps its first word.
_CHAPTER_PREFIX = re.compile(rf"^(chapter\s+{_NUMBER}\b\s*[.:\-–—]?\s*|\d{{1,3}}\s*[.:]\s+)", re.IGNORECASE)


def _clean_toc_text(text: str) -> str:
    # Library cataloguing writes subtitles as "Title : subtitle" and often
    # ends an entry with a full stop — but not an abbreviation's own
    # ("The U.S.", "Washington, D.C."), or an ellipsis.
    text = re.sub(r"\s+:\s+", ": ", text.strip())
    if text.endswith(".") and not text.endswith("..") and not re.search(r"(?:^|[\s.])[A-Z]\.$", text):
        return text[:-1].rstrip()
    return text


def _is_front_or_back_matter(title: str) -> bool:
    key = normalize_text(title)
    return key in _NOT_A_CHAPTER_EXACT or bool(_NOT_A_CHAPTER_PREFIX.match(key))


def parse_table_of_contents(entries: object) -> tuple[list[str], list[KnownPart]] | None:
    """Open Library `table_of_contents` -> (chapters, parts), or None when
    the list isn't clean enough to trust as a starting point.

    Conservative on purpose: Open Library's contents data is hand-entered
    and inconsistent. Some editions lump several chapters into one entry
    ("A ; B ; C" or "A -- B"); rather than guess where one title ends,
    those are rejected outright and the reader types the list instead.
    Parts are recognised by a "Part …"/"Book …" label or title; every
    chapter after one belongs to it until the next.
    """
    if not isinstance(entries, list):
        return None

    # First pass: classify entries as part headings or chapter candidates.
    items: list[KnownPart | tuple[str, int | None]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = _clean_toc_text(str(entry.get("title") or ""))
        label = _clean_toc_text(str(entry.get("label") or ""))
        level = entry.get("level") if isinstance(entry.get("level"), int) else None
        if " ; " in title or " -- " in title:
            return None

        # Checked before the empty-title skip: a "Part Two" label with no
        # title is still a part boundary.
        if _PART_LABEL.match(label):
            part_label = f"{label[0].upper()}{label[1:]}"
            items.append(KnownPart(title=f"{part_label}: {title}" if title else part_label, chapters=[]))
            continue
        if part_match := _PART_TITLE.match(title):
            division, number, rest = part_match.groups()
            heading = f"{division.capitalize()} {number}"
            items.append(KnownPart(title=f"{heading}: {rest}" if rest else heading, chapters=[]))
            continue

        if not title:
            continue
        # Stripped before the checks below, so "14. Notes" is still back
        # matter and "12. Epilogue" still stands alone.
        title = _CHAPTER_PREFIX.sub("", title)
        if not title:
            # A bare "Chapter 2" with no title: dropping it would shift
            # every later chapter's number (and part grouping) by one.
            return None
        if not _is_front_or_back_matter(title):
            items.append((title, level))

    # Open Library's `level` nests entries. Chapters are the most common
    # level; anything deeper is a subsection. Anything shallower that isn't
    # a recognised part or an intro/conclusion is an unlabelled grouping
    # ("The Rise" over its chapters) that can't be told apart from a
    # chapter reliably — give up rather than guess.
    levels = [lvl for item in items if isinstance(item, tuple) and (lvl := item[1]) is not None]
    chapter_level = Counter(levels).most_common(1)[0][0] if levels else None

    chapters: list[str] = []
    parts: list[KnownPart] = []
    for item in items:
        if isinstance(item, KnownPart):
            parts.append(item)
            continue
        title, level = item
        if chapter_level is not None and level is not None:
            if level > chapter_level:
                continue
            if level < chapter_level and not _STANDS_ALONE.match(title):
                return None
        chapters.append(title)
        if parts and not _STANDS_ALONE.match(title):
            parts[-1].chapters.append(len(chapters))  # type: ignore[union-attr]

    parts = [p for p in parts if p.chapters]
    # Two parts with one title is a data-entry slip preflight_check would
    # reject anyway — better no pre-fill than a file that can't run.
    if len(chapters) < 2 or len({p.title for p in parts}) != len(parts):
        return None
    return chapters, parts


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

    # Every search the pipeline runs is built from these, and results are
    # only trusted if they name the book's author and title (see
    # search.is_book_relevant) — a missing or placeholder value would search
    # for "None"/"TODO: fill in by hand" and discard every real result.
    if not known_file.has_title:
        problems.append("title is required (still missing or a placeholder)")
    if not known_file.has_author:
        problems.append("author is required (still missing or a placeholder)")

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

    if (
        known_file.kind == "non-fiction"
        and known_file.narrative
        and any(part.chapters is not None for part in known_file.parts)
    ):
        problems.append(
            "chapters on parts requires a known chapter list, which "
            "narrative non-fiction doesn't have — leave chapters "
            "unset on parts for narrative books"
        )

    if known_file.is_full_nonfiction_path and known_file.parts:
        valid_numbers = set(range(1, len(known_file.chapters) + 1))
        for part in known_file.parts:
            for n in invalid_chapter_numbers(part.chapters, valid_numbers):
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
        problems.append(
            f"parts has duplicate titles: {sorted(dupes)!r} — each part title must be unique"
        )

    return problems


def ensure_ready(known_file: KnownFile) -> None:
    """Convenience wrapper: raises ValueError with all problems joined if
    the known-file isn't ready for generation.
    """
    problems = preflight_check(known_file)
    if problems:
        raise ValueError(
            "known-file is not ready for generation:\n- " + "\n- ".join(problems)
        )
