from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import verify
from precis.schema import KnownFile
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
    assert result == {"verified": True}
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
    assert result == {"verified": True}
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
    assert result == {"verified": True}
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
