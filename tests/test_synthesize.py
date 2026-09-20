from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import synthesize
from precis.pipeline.nodes.synthesize import Synthesis, SynthesisWithClaims
from precis.schema import KeyClaim, KnownFile, Part
from precis.search import SearchResult


def _nonfiction_known_file() -> KnownFile:
    return KnownFile(
        isbn="123", title="A Book", author="An Author", kind="non-fiction", narrative=False, chapters=["Ch 1", "Ch 2"]
    )


def _fiction_known_file() -> KnownFile:
    return KnownFile(isbn="456", title="A Novel", author="A Novelist", kind="fiction")


def _search_client_with_results() -> AsyncMock:
    client = AsyncMock()
    client.search.return_value = [SearchResult(title="t", url="u", content="themes of X and Y")]
    return client


@pytest.mark.asyncio
async def test_nonfiction_path_produces_claims_and_chapter_referencing_parts(monkeypatch):
    chapters = [
        {"number": 1, "title": "Ch 1", "key_points": ["a"], "core_claim": "claim 1", "quality_flag": None},
        {"number": 2, "title": "Ch 2", "key_points": ["b"], "core_claim": "claim 2", "quality_flag": None},
    ]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert response_model is SynthesisWithClaims
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaim(prompt="q1", answer="a1")],
            parts=[Part(title="Part One", summary="covers ch 1-2", chapter_numbers=[1, 2])],
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
    assert result["key_claims_for_review"] == [{"prompt": "q1", "answer": "a1"}]
    assert result["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}]


@pytest.mark.asyncio
async def test_fiction_path_produces_parts_but_no_key_claims_key(monkeypatch):
    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert response_model is Synthesis
        return Synthesis(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1"],
            parts=[Part(title="Beginning", summary="stakes are introduced")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()},
        search_client=_search_client_with_results(),
        llm_client=AsyncMock(),
    )

    assert result["synopsis"] == "a synopsis"
    assert result["tags"] == ["tag1"]
    assert result["parts"] == [{"title": "Beginning", "summary": "stakes are introduced", "chapter_numbers": None}]
    assert "key_claims_for_review" not in result


@pytest.mark.asyncio
async def test_clients_from_config_are_used_when_not_passed_explicitly(monkeypatch):
    search_client = _search_client_with_results()
    llm_client = AsyncMock()

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert client is llm_client
        return Synthesis(
            synopsis="s",
            one_line_takeaway="t",
            tags=["tag"],
            parts=[Part(title="p", summary="s")],
        )

    monkeypatch.setattr(synthesize.llm, "complete_structured", fake_complete_structured)

    result = await synthesize.run(
        {"known_file": _fiction_known_file().model_dump()},
        config={"configurable": {"search_client": search_client, "llm_client": llm_client}},
    )
    assert result["synopsis"] == "s"
    search_client.search.assert_called_once()
