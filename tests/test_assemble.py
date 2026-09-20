from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import assemble
from precis.pipeline.nodes.synthesize import (
    KeyClaimDraft,
    SynthesisPart,
    SynthesisWithClaims,
)
from precis.schema import KnownFile


def _nonfiction_known_file() -> KnownFile:
    return KnownFile(
        isbn="123", title="A Book", author="An Author", kind="non-fiction", narrative=False, chapters=["Ch 1", "Ch 2"]
    )


def _fiction_known_file() -> KnownFile:
    return KnownFile(isbn="456", title="A Novel", author="A Novelist", kind="fiction")


def _chapters() -> list[dict]:
    return [
        {"number": 1, "title": "Ch 1", "key_points": ["a"], "core_claim": "claim 1", "quality_flag": None},
        {"number": 2, "title": "Ch 2", "key_points": ["b"], "core_claim": "claim 2", "quality_flag": None},
    ]


def _nonfiction_state(**overrides) -> dict:
    state = {
        "known_file": _nonfiction_known_file().model_dump(),
        "chapters": _chapters(),
        "one_line_takeaway": "the takeaway",
        "synopsis": "a synopsis",
        "tags": ["tag1", "tag2"],
        "parts": [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}],
        "key_claims_for_review": [{"prompt": "q1", "answer": "a1"}],
        "warnings": [],
    }
    state.update(overrides)
    return state


def _fiction_state(**overrides) -> dict:
    state = {
        "known_file": _fiction_known_file().model_dump(),
        "chapters": [],
        "one_line_takeaway": "the takeaway",
        "synopsis": "a synopsis",
        "tags": ["tag1"],
        "parts": [{"title": "Beginning", "summary": "stakes are introduced", "chapter_numbers": None}],
        "warnings": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_nonfiction_path_valid_on_first_attempt_no_repair_called(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    result = await assemble.run(_nonfiction_state(), llm_client=AsyncMock())

    fake.assert_not_called()
    book = result["book"]
    assert book["title"] == "A Book"
    assert book["author"] == "An Author"
    assert book["isbn"] == "123"
    assert book["one_line_takeaway"] == "the takeaway"
    assert book["synopsis"] == "a synopsis"
    assert book["tags"] == ["tag1", "tag2"]
    assert book["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}]
    assert book["key_claims_for_review"] == [{"prompt": "q1", "answer": "a1"}]
    assert len(book["chapters"]) == 2
    assert book["reader_notes"] is None
    assert book["warnings"] == []


@pytest.mark.asyncio
async def test_fiction_path_with_empty_chapters_list_becomes_none(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    result = await assemble.run(_fiction_state(), llm_client=AsyncMock())

    fake.assert_not_called()
    book = result["book"]
    assert book["chapters"] is None
    assert book["key_claims_for_review"] is None


@pytest.mark.asyncio
async def test_reader_notes_passed_through_verbatim(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    known_file = _fiction_known_file()
    known_file = KnownFile.model_validate({**known_file.model_dump(), "notes": "read this slowly"})
    state = _fiction_state(known_file=known_file.model_dump())

    result = await assemble.run(state, llm_client=AsyncMock())

    assert result["book"]["reader_notes"] == "read this slowly"


@pytest.mark.asyncio
async def test_invalid_parts_chapter_reference_triggers_one_repair_call(monkeypatch):
    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
    )

    calls = []

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        calls.append(response_model)
        assert response_model is SynthesisWithClaims
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaimDraft(prompt="q1", answer="a1")],
            parts=[SynthesisPart(title="Part One", summary="covers ch 1-2", chapter_numbers=[1, 2])],
        )

    monkeypatch.setattr(assemble.llm, "complete_structured", fake_complete_structured)

    result = await assemble.run(state, llm_client=AsyncMock())

    assert len(calls) == 1
    book = result["book"]
    assert book["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}]


@pytest.mark.asyncio
async def test_repair_still_invalid_raises_informative_error(monkeypatch):
    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        # Repaired result still references a nonexistent chapter.
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaimDraft(prompt="q1", answer="a1")],
            parts=[SynthesisPart(title="Part One", summary="still bad", chapter_numbers=[1, 99])],
        )

    monkeypatch.setattr(assemble.llm, "complete_structured", fake_complete_structured)

    with pytest.raises(ValueError, match="Stage 4 assemble: book still invalid after one repair attempt"):
        await assemble.run(state, llm_client=AsyncMock())


@pytest.mark.asyncio
async def test_clients_from_config_are_used_when_not_passed_explicitly(monkeypatch):
    llm_client = AsyncMock()
    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert client is llm_client
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaimDraft(prompt="q1", answer="a1")],
            parts=[SynthesisPart(title="Part One", summary="covers ch 1-2", chapter_numbers=[1, 2])],
        )

    monkeypatch.setattr(assemble.llm, "complete_structured", fake_complete_structured)

    result = await assemble.run(state, config={"configurable": {"llm_client": llm_client}})
    assert result["book"]["parts"][0]["chapter_numbers"] == [1, 2]
