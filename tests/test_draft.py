from unittest.mock import AsyncMock

import pytest

from precis import llm
from precis.pipeline.nodes import draft
from precis.pipeline.nodes.draft import ChapterDraft, Critique
from precis.schema import KnownFile
from precis.search import SearchResult


def _known_file() -> KnownFile:
    return KnownFile(isbn="123", title="A Book", author="An Author", kind="non-fiction", chapters=["Ch 1", "Ch 2"])


def _state() -> dict:
    return {"known_file": _known_file().model_dump(), "chapter_number": 1, "chapter_title": "Ch 1"}


def _search_client_with_results() -> AsyncMock:
    client = AsyncMock()
    client.search.return_value = [SearchResult(title="t", url="u", content="chapter 1 covers X and Y")]
    return client


@pytest.mark.asyncio
async def test_first_draft_passing_critique_returns_chapter_with_no_flag(monkeypatch):
    responses = iter(
        [
            ChapterDraft(key_points=["point a", "point b"], core_claim="claim"),
            Critique(passed=True, feedback="looks good"),
        ]
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        return next(responses)

    monkeypatch.setattr(draft.llm, "complete_structured", fake_complete_structured)

    result = await draft.run_one(_state(), search_client=_search_client_with_results(), llm_client=AsyncMock())

    assert result["chapters"] == [
        {
            "number": 1,
            "title": "Ch 1",
            "key_points": ["point a", "point b"],
            "core_claim": "claim",
            "quality_flag": None,
        }
    ]
    assert "warnings" not in result


@pytest.mark.asyncio
async def test_clients_from_config_are_used_when_not_passed_explicitly(monkeypatch):
    search_client = _search_client_with_results()
    llm_client = AsyncMock()

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert client is llm_client
        if response_model is ChapterDraft:
            return ChapterDraft(key_points=["point a"], core_claim="claim")
        return Critique(passed=True, feedback="fine")

    monkeypatch.setattr(draft.llm, "complete_structured", fake_complete_structured)

    result = await draft.run_one(
        _state(),
        config={"configurable": {"search_client": search_client, "llm_client": llm_client}},
    )
    assert result["chapters"][0]["key_points"] == ["point a"]
    search_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_repair_after_one_failed_critique_then_passes(monkeypatch):
    responses = iter(
        [
            ChapterDraft(key_points=["weak point"], core_claim="weak claim"),
            Critique(passed=False, feedback="core_claim isn't supported by the search results"),
            ChapterDraft(key_points=["revised point"], core_claim="revised claim"),
            Critique(passed=True, feedback="fixed"),
        ]
    )
    call_log: list[str] = []

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        call_log.append(response_model.__name__)
        return next(responses)

    monkeypatch.setattr(draft.llm, "complete_structured", fake_complete_structured)

    result = await draft.run_one(_state(), search_client=_search_client_with_results(), llm_client=AsyncMock())

    assert call_log == ["ChapterDraft", "Critique", "ChapterDraft", "Critique"]
    assert result["chapters"][0]["key_points"] == ["revised point"]
    assert result["chapters"][0]["quality_flag"] is None


@pytest.mark.asyncio
async def test_exhausted_retries_falls_back_with_quality_flag_and_warning(monkeypatch):
    async def fake_complete_structured(client, *, messages, response_model, model=None):
        if response_model is ChapterDraft:
            return ChapterDraft(key_points=["same point restated"], core_claim="claim")
        return Critique(passed=False, feedback="point restates itself")

    monkeypatch.setattr(draft.llm, "complete_structured", fake_complete_structured)

    result = await draft.run_one(_state(), search_client=_search_client_with_results(), llm_client=AsyncMock())

    chapter = result["chapters"][0]
    assert chapter["quality_flag"] is not None
    assert "point restates itself" in chapter["quality_flag"]
    assert len(result["warnings"]) == 1
    assert "chapter 1" in result["warnings"][0]


@pytest.mark.asyncio
async def test_structured_output_error_from_draft_propagates_uncaught(monkeypatch):
    """Resilience against a malformed/missing tool call now lives entirely
    in llm.complete_structured (see test_llm.py) — run_one doesn't catch
    StructuredOutputError itself. If it's raised (meaning complete_structured
    already exhausted its own retry budget), that's a real, escalated
    failure for this chapter, and run_one should let it propagate rather
    than reinterpret it as something else.
    """

    async def always_broken(client, *, messages, response_model, model=None):
        raise llm.StructuredOutputError("never usable")

    monkeypatch.setattr(draft.llm, "complete_structured", always_broken)

    with pytest.raises(llm.StructuredOutputError, match="never usable"):
        await draft.run_one(_state(), search_client=_search_client_with_results(), llm_client=AsyncMock())


@pytest.mark.asyncio
async def test_structured_output_error_from_critique_propagates_uncaught(monkeypatch):
    """Same as above, but the failure comes from the critique call after a
    successful draft -- also propagates, not silently absorbed into a
    quality_flag fallback with stale draft/feedback state.
    """
    responses = iter(
        [
            ChapterDraft(key_points=["draft one"], core_claim="claim one"),
            llm.StructuredOutputError("model returned malformed critique tool call"),
        ]
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(draft.llm, "complete_structured", fake_complete_structured)

    with pytest.raises(llm.StructuredOutputError, match="malformed critique tool call"):
        await draft.run_one(_state(), search_client=_search_client_with_results(), llm_client=AsyncMock())
