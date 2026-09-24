from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import verify
from precis.schema import KnownFile, KnownPart
from precis.search import SearchResult


def _known_file() -> KnownFile:
    return KnownFile(isbn="123", title="A Book", author="An Author", kind="non-fiction", chapters=["Ch 1"])


@pytest.mark.asyncio
async def test_trust_known_skips_search_and_llm_entirely():
    search_client = AsyncMock()
    result = await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": True},
        search_client=search_client,
        llm_client=AsyncMock(),
    )
    assert result == {"verified": True, "verify_reason": "skipped (--trust-known)"}
    search_client.search.assert_not_called()


@pytest.mark.asyncio
async def test_verified_verdict_returns_true(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="matches the book")]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        return response_model(verified=True, reason="looks right")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    result = await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )
    assert result == {"verified": True, "verify_reason": "looks right"}
    search_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_clients_from_config_are_used_when_not_passed_explicitly(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="matches the book")]
    llm_client = AsyncMock()

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert client is llm_client
        return response_model(verified=True, reason="looks right")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    result = await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": False},
        config={"configurable": {"search_client": search_client, "llm_client": llm_client}},
    )
    assert result == {"verified": True, "verify_reason": "looks right"}
    search_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_unverified_verdict_raises_with_reason(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = []

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        return response_model(verified=False, reason="wrong edition")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    with pytest.raises(ValueError, match="wrong edition"):
        await verify.run(
            {"known_file": _known_file().model_dump(), "trust_known": False},
            search_client=search_client,
            llm_client=AsyncMock(),
        )


@pytest.mark.asyncio
async def test_known_parts_are_included_in_search_query_and_prompt(monkeypatch):
    known_file = KnownFile(
        isbn="123",
        title="A Book",
        author="An Author",
        kind="non-fiction",
        chapters=["Ch 1", "Ch 2"],
        parts=[KnownPart(title="Part One", chapter_numbers=[1, 2])],
    )
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="matches the book")]

    captured_prompt = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        captured_prompt["content"] = messages[1]["content"]
        return response_model(verified=True, reason="looks right")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    result = await verify.run(
        {"known_file": known_file.model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )

    assert result == {"verified": True, "verify_reason": "looks right"}
    search_query = search_client.search.call_args[0][0]
    assert "parts" in search_query
    assert "Part One" in captured_prompt["content"]
    assert "parts and their chapter groupings" in captured_prompt["content"]


@pytest.mark.asyncio
async def test_no_parts_question_when_known_file_has_no_parts(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="matches the book")]

    captured_prompt = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        captured_prompt["content"] = messages[1]["content"]
        return response_model(verified=True, reason="looks right")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )

    assert "parts and their chapter groupings" not in captured_prompt["content"]
