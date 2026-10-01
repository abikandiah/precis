from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from precis import generate as generate_module
from precis.llm import StructuredOutputError
from precis.research import Research
from precis.schema import Book, KnownFile

BOOK = KnownFile(isbn="1", title="T", author="A", kind="fiction")


async def test_generate_researches_digests_writes_checks_then_reviews():
    searched = Research(sources=[], warnings=["thin research"])
    found = Research(sources=[], warnings=["thin research"])
    written = SimpleNamespace(ideas=[], key_claims_for_review=None)
    reviewed = SimpleNamespace(kind="fiction", ideas=[], key_claims_for_review=None, warnings=["thin research", "author mismatch"])
    messages: list[str] = []
    with (
        patch.object(generate_module.llm, "build_client") as build_client,
        patch.object(generate_module, "research", new=AsyncMock(return_value=searched)) as research,
        patch.object(generate_module, "digest", new=AsyncMock(return_value=found)) as digest,
        patch.object(generate_module, "write_notes", new=AsyncMock(return_value=written)) as write,
        patch.object(generate_module, "check_notes", new=MagicMock(return_value=["idea 1: a finding"])) as check,
        patch.object(
            generate_module, "review_notes", new=AsyncMock(return_value=(reviewed, ['dropped "X": generic']))
        ) as review,
    ):
        result = await generate_module.generate(
            BOOK, slug="t", trust_known=True, fresh=True, on_progress=messages.append
        )
    assert (result.book, result.research) == (reviewed, found)
    kwargs = research.await_args.kwargs
    assert (kwargs["slug"], kwargs["trust_known"], kwargs["fresh"]) == ("t", True, True)
    # Everything after research works from the digested research.
    assert digest.await_args.args == (BOOK, searched) and digest.await_args.kwargs["slug"] == "t"
    assert write.await_args.args == (BOOK, found)
    assert write.await_args.kwargs["trust_known"] is True
    check.assert_called_once_with(written, found)
    assert review.await_args.args == (BOOK, found, written, ["idea 1: a finding"])
    # One client for every call, closed at the end of the run.
    client = build_client.return_value.__aenter__.return_value
    assert all(call.await_args.kwargs["client"] is client for call in (digest, write, review))
    build_client.return_value.__aexit__.assert_awaited_once()
    assert "check: idea 1: a finding" in messages
    assert 'review: dropped "X": generic' in messages
    # Warnings the write and review steps added are shown; research printed its own.
    assert "warning: author mismatch" in messages
    assert "warning: thin research" not in messages


async def test_a_failed_review_keeps_the_written_notes_with_a_warning():
    found = Research(sources=[], warnings=[])
    written = MagicMock(ideas=[], key_claims_for_review=None, warnings=["from write"])
    messages: list[str] = []
    with (
        patch.object(generate_module.llm, "build_client"),
        patch.object(generate_module, "research", new=AsyncMock(return_value=found)),
        patch.object(generate_module, "write_notes", new=AsyncMock(return_value=written)),
        patch.object(generate_module, "check_notes", new=MagicMock(return_value=["idea 1: a finding"])),
        patch.object(generate_module, "review_notes", new=AsyncMock(side_effect=StructuredOutputError("bad"))),
    ):
        result = await generate_module.generate(BOOK, slug="t", on_progress=messages.append)
    assert result.book is written.model_copy.return_value
    warnings = written.model_copy.call_args.kwargs["update"]["warnings"]
    assert warnings[0] == "from write" and "unreviewed" in warnings[1] and warnings[2] == "idea 1: a finding"
    assert "review: no changes" not in messages


def _nonfiction(ideas: int) -> Book:
    return Book.model_validate({
        "title": "T", "author": "A", "isbn": "1", "kind": "non-fiction", "one_line_takeaway": "t", "synopsis": "s",
        "ideas": [{"title": f"Idea {n}", "summary": "s", "evidence": "e"} for n in range(ideas)],
        "key_claims_for_review": [{"prompt": "Q?", "answer": "A."}] * 5, "tags": ["business", "economics"],
    })  # fmt: skip


async def test_a_thin_review_deck_is_a_warning_whether_or_not_the_review_ran():
    found = Research(sources=[], warnings=[])
    written = _nonfiction(13)  # 5 claims
    for review in (AsyncMock(return_value=(written, [])), AsyncMock(side_effect=StructuredOutputError("bad"))):
        with (
            patch.object(generate_module.llm, "build_client"),
            patch.object(generate_module, "research", new=AsyncMock(return_value=found)),
            patch.object(generate_module, "write_notes", new=AsyncMock(return_value=written)),
            patch.object(generate_module, "review_notes", new=review),
        ):
            result = await generate_module.generate(BOOK, slug="t")
        assert result.book.warnings[-1] == "5 key claims for 13 ideas — the review deck covers under half the notes"


async def test_a_failed_review_still_names_the_uncited_ideas():
    written = _nonfiction(6)
    with (
        patch.object(generate_module.llm, "build_client"),
        patch.object(generate_module, "research", new=AsyncMock(return_value=Research(sources=[], warnings=[]))),
        patch.object(generate_module, "write_notes", new=AsyncMock(return_value=written)),
        patch.object(generate_module, "review_notes", new=AsyncMock(side_effect=StructuredOutputError("bad"))),
    ):
        result = await generate_module.generate(BOOK, slug="t")
    assert any(w.startswith("6 idea(s) rest on the model's knowledge") for w in result.book.warnings)
