from unittest.mock import AsyncMock

import pytest

from precis.llm import StructuredOutputError
from precis.pipeline.nodes import synthesize
from precis.pipeline.nodes.synthesize import (
    EXPECTED_PART_COUNT_KEY,
    KIND_KEY,
    Synthesis,
    SynthesisWithClaims,
)
from precis.schema import KeyClaim, KnownFile, KnownPart, Part
from precis.search import SearchResult


def _nonfiction_known_file() -> KnownFile:
    return KnownFile(
        isbn="123", title="A Book", author="An Author", kind="non-fiction", narrative=False, chapters=["Ch 1", "Ch 2"]
    )


def _fiction_known_file() -> KnownFile:
    return KnownFile(isbn="456", title="A Novel", author="A Novelist", kind="fiction")


def _search_client_with_results() -> AsyncMock:
    client = AsyncMock()
    # Names every fixture author and title, so the book-relevance filter
    # keeps it whichever known-file a test uses.
    client.search.return_value = [
        SearchResult(
            title="Author, Novelist, Memoirist reviewed", url="u", content="A Book, A Novel, A Memoir: themes of X and Y"
        )
    ]
    return client


def _synthesis_kwargs(count: int) -> dict:
    return {
        "synopsis": "s",
        "one_line_takeaway": "t",
        "tags": ["tag1", "tag2"],
        "parts": [{"title": f"p{i}", "summary": f"s{i}"} for i in range(count)],
    }


def test_synthesis_model_validator_allows_any_count_without_context():
    Synthesis.model_validate(_synthesis_kwargs(5))
    Synthesis.model_validate(_synthesis_kwargs(1))


def test_synthesis_model_validator_enforces_expected_count_from_context():
    Synthesis.model_validate(_synthesis_kwargs(4), context={EXPECTED_PART_COUNT_KEY: 4})

    with pytest.raises(Exception, match="expected exactly 4 parts"):
        Synthesis.model_validate(_synthesis_kwargs(5), context={EXPECTED_PART_COUNT_KEY: 4})


@pytest.mark.asyncio
async def test_nonfiction_path_produces_claims_and_chapter_referencing_parts(monkeypatch):
    chapters = [
        {"number": 1, "title": "Ch 1", "key_points": ["a"], "core_claim": "claim 1", "quality_flag": None},
        {"number": 2, "title": "Ch 2", "key_points": ["b"], "core_claim": "claim 2", "quality_flag": None},
    ]

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        assert response_model is SynthesisWithClaims
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[
                KeyClaim(prompt="q1", answer="a1"),
                KeyClaim(prompt="q2", answer="a2"),
                KeyClaim(prompt="q3", answer="a3"),
            ],
            parts=[Part(title="Part One", summary="covers ch 1-2", chapters=[1, 2])],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _nonfiction_known_file().model_dump(), "chapters": chapters},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["synopsis"] == "a synopsis"
    assert result["one_line_takeaway"] == "the takeaway"
    assert result["tags"] == ["tag1", "tag2"]
    assert result["key_claims_for_review"] == [
        {"prompt": "q1", "answer": "a1"},
        {"prompt": "q2", "answer": "a2"},
        {"prompt": "q3", "answer": "a3"},
    ]
    assert result["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapters": [1, 2]}]
    assert result["parts_source"] == "generated"
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_fiction_path_produces_parts_but_no_key_claims_key(monkeypatch):
    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        assert response_model is Synthesis
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[Part(title="Beginning", summary="stakes are introduced")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["synopsis"] == "a synopsis"
    assert result["tags"] == ["tag1", "tag2"]
    assert result["parts"] == [{"title": "Beginning", "summary": "stakes are introduced", "chapters": None}]
    assert result["parts_source"] == "generated"
    assert result["warnings"] == []
    assert "key_claims_for_review" not in result


@pytest.mark.asyncio
async def test_fiction_generated_parts_strip_model_invented_chapter_numbers(monkeypatch):
    """Regression test: a fiction (or narrative non-fiction) book has no
    `chapters` array on the Book output for a part's `chapters` (chapter-
    number references) to mean anything against — nothing in the fiction
    prompt asks for them, but nothing stops the model from inventing some
    anyway. They must be stripped to None rather than shipped as dangling,
    unverified references.
    """

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[Part(title="Beginning", summary="stakes are introduced", chapters=[1, 2, 3])],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["parts"] == [{"title": "Beginning", "summary": "stakes are introduced", "chapters": None}]


@pytest.mark.asyncio
async def test_known_parts_are_passed_through_verbatim_for_full_nonfiction(monkeypatch):
    known_file = _nonfiction_known_file()
    known_file.parts = [KnownPart(title="Part One", chapters=[1, 2])]
    chapters = [
        {"number": 1, "title": "Ch 1", "key_points": ["a"], "core_claim": "claim 1", "quality_flag": None},
        {"number": 2, "title": "Ch 2", "key_points": ["b"], "core_claim": "claim 2", "quality_flag": None},
    ]

    captured_context = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        assert response_model is SynthesisWithClaims
        captured_context["value"] = validation_context
        # The model echoes back a different title/grouping than the known
        # one — this should be discarded, keeping only the summary.
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[
                KeyClaim(prompt="q1", answer="a1"),
                KeyClaim(prompt="q2", answer="a2"),
                KeyClaim(prompt="q3", answer="a3"),
            ],
            parts=[Part(title="Part One", summary="covers ch 1-2", chapters=[1])],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": known_file.model_dump(), "chapters": chapters},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapters": [1, 2]}]
    assert result["parts_source"] == "known"
    assert result["warnings"] == []
    assert captured_context["value"] == {KIND_KEY: "non-fiction", EXPECTED_PART_COUNT_KEY: 1}


@pytest.mark.asyncio
async def test_known_parts_are_title_only_for_narrative_nonfiction(monkeypatch):
    known_file = KnownFile(
        isbn="789", title="A Memoir", author="A Memoirist", kind="non-fiction", narrative=True,
        parts=[KnownPart(title="Early Years")],
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        assert response_model is Synthesis
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[Part(title="Early Years", summary="spoiler-safe summary")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": known_file.model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["parts"] == [{"title": "Early Years", "summary": "spoiler-safe summary", "chapters": None}]
    assert result["parts_source"] == "known"


@pytest.mark.asyncio
async def test_known_parts_raises_when_model_returns_wrong_count(monkeypatch):
    known_file = _fiction_known_file()
    known_file.kind = "non-fiction"
    known_file.narrative = True
    known_file.parts = [KnownPart(title="Part One"), KnownPart(title="Part Two")]

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[Part(title="Part One", summary="only one part came back")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    with pytest.raises(StructuredOutputError):
        await synthesize.run(
            {"known_file": known_file.model_dump()},
            search_client=_search_client_with_results(),
            llm_client=AsyncMock(),
        )


@pytest.mark.asyncio
async def test_known_parts_match_by_position_not_by_echoed_title(monkeypatch):
    """The model rephrasing a title (or two known parts sharing a title)
    must not break the summary correlation — position, not title text, is
    the correlation key.
    """
    known_file = _fiction_known_file()
    known_file.kind = "non-fiction"
    known_file.narrative = True
    known_file.parts = [KnownPart(title="Part One"), KnownPart(title="Part One")]

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[
                Part(title="Part One (rephrased)", summary="first summary"),
                Part(title="Part One, again", summary="second summary"),
            ],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": known_file.model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["parts"] == [
        {"title": "Part One", "summary": "first summary", "chapters": None},
        {"title": "Part One", "summary": "second summary", "chapters": None},
    ]
    # Both positions got a title different from the known one — each is a
    # discrepancy worth surfacing, even though the known title always wins.
    assert len(result["warnings"]) == 2
    assert "Part One (rephrased)" in result["warnings"][0]
    assert "Part One, again" in result["warnings"][1]


@pytest.mark.asyncio
async def test_known_parts_with_matching_titles_produce_no_warnings(monkeypatch):
    known_file = _fiction_known_file()
    known_file.kind = "non-fiction"
    known_file.narrative = True
    known_file.parts = [KnownPart(title="Part One")]

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[Part(title="Part One", summary="matches exactly")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": known_file.model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_no_expected_part_count_on_generated_parts_path(monkeypatch):
    """Only the known-parts path constrains the model's part count —
    EXPECTED_PART_COUNT_KEY absent means Synthesis's model_validator skips
    that check entirely, matching the generated path's "any count is fine"
    behavior. KIND_KEY is always present regardless, since every call site
    knows known_file.kind and there's always a tags vocabulary to check.
    """
    captured_context = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        captured_context["value"] = validation_context
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            parts=[Part(title="p1", summary="s1"), Part(title="p2", summary="s2")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert captured_context["value"] == {KIND_KEY: "fiction"}
    assert result["parts_source"] == "generated"


@pytest.mark.asyncio
async def test_clients_from_config_are_used_when_not_passed_explicitly(monkeypatch):
    search_client = _search_client_with_results()
    llm_client = AsyncMock()

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        assert client is llm_client
        return Synthesis(
            synopsis="s",
            one_line_takeaway="t",
            tags=["tag1", "tag2"],
            parts=[Part(title="p", summary="s")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()},
        config={"configurable": {"search_client": search_client, "llm_client": llm_client}},
    )
    assert result["synopsis"] == "s"
    search_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_no_book_specific_results_adds_a_warning(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="generic themes")]

    async def fake_complete_structured(client, *, messages, response_model, model=None, validation_context=None):
        return Synthesis(
            synopsis="s",
            one_line_takeaway="t",
            tags=["tag1", "tag2"],
            parts=[Part(title="p", summary="s")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()}, search_client=search_client, llm_client=AsyncMock()
    )
    assert result["warnings"] == [synthesize._UNGROUNDED_WARNING]
