"""A whole run: research the book, write its notes, check them in code,
then review them against the research in one more call. A review that fails
leaves the written notes, which are already paid for, with a warning that
they're unreviewed. A plain async
function — the research cache (research.py) is the only persisted state, so
an interrupted run just starts again without searching again.

There's no whole-run time limit: every request has its own timeout. A
stalled provider costs about 15 minutes a call (three 5-minute tries), but
the bound through every retry layer is hours: 2 attempts × 4 requests (for
provider errors) × 3 HTTP tries × 5 minutes, for the write and the review.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from precis import llm
from precis.checks import check_notes
from precis.research import ProgressCallback, Research, research
from precis.review import review_notes, uncited_warnings
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
    async with llm.build_client() as client:
        return await _write_and_review(known_file, found, client, trust_known=trust_known, progress=progress)


async def _write_and_review(
    known_file: KnownFile,
    found: Research,
    client: AsyncOpenAI,
    *,
    trust_known: bool,
    progress: ProgressCallback,
) -> Generated:
    progress("write: writing the notes")
    written = await write_notes(known_file, found, trust_known=trust_known, client=client, on_progress=progress)
    progress(f"write: {len(written.ideas)} ideas, {len(written.key_claims_for_review or [])} key claims")

    issues = check_notes(written)
    for issue in issues:
        progress(f"check: {issue}")
    progress("review: checking the notes against the research")
    try:
        book, changes = await review_notes(known_file, found, written, issues, client=client, on_progress=progress)
    except (llm.StructuredOutputError, llm.TransientLLMError, llm.ProviderError) as exc:
        progress(f"review: failed, keeping the unreviewed notes ({exc})")
        failed = f"the review call failed, so these notes are unreviewed ({exc})"
        # The check findings the review would have dealt with go to the reader
        # instead, with the uncited ideas the review would have named.
        warnings = [*written.warnings, failed, *issues, *uncited_warnings(written.ideas)]
        book = written.model_copy(update={"warnings": warnings})
    else:
        for change in changes:
            progress(f"review: {change}")
        if not changes:
            progress("review: no changes")
    # Research already printed its own warnings.
    for warning in book.warnings:
        if warning not in found.warnings:
            progress(f"warning: {warning}")
    return Generated(book=book, research=found)
