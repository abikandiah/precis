"""A whole run: research the book, then write its notes. A plain async
function — the research cache (research.py) is the only persisted state, so
an interrupted run just starts again without searching again.

There's no whole-run time limit: every call has its own timeout, which
bounds a run (the write call's worst case, a stalled provider through every
retry, is about half an hour).
"""

from __future__ import annotations

from dataclasses import dataclass

from precis.research import ProgressCallback, Research, research
from precis.schema import Book, KnownFile
from precis.write import write_notes


@dataclass(frozen=True)
class Generated:
    book: Book
    # What the notes were written from — the eval runner records its
    # fingerprint so a comparison between runs can tell whether both saw
    # the same research.
    research: Research


async def generate(
    known_file: KnownFile,
    *,
    slug: str,
    trust_known: bool = False,
    fresh: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Generated:
    """`slug` names the research cache; `fresh` searches again instead of
    using it. `trust_known` turns the book and author checks into warnings.
    Raises research.ResearchError for a known-file that isn't ready.
    """
    progress = on_progress or (lambda _: None)
    found = await research(known_file, slug=slug, trust_known=trust_known, fresh=fresh, on_progress=progress)
    progress("write: writing the notes")
    book = await write_notes(known_file, found, trust_known=trust_known)
    progress(f"write: {len(book.ideas)} ideas, {len(book.key_claims_for_review or [])} key claims")
    # Research already printed its own warnings.
    for warning in book.warnings:
        if warning not in found.warnings:
            progress(f"write warning: {warning}")
    return Generated(book=book, research=found)
