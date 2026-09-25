from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import verify
from precis.schema import KnownFile, KnownPart
from precis.search import SearchResult

# Names the title, the author and the ISBN, so it can back any fatal
# contradiction (see verify._Sources).
_BOOK_RESULT = SearchResult(
    title="A Book by An Author — contents", url="https://example.com/toc", content="A Book, ISBN 978-0-00-000000-2"
)


def _known_file() -> KnownFile:
    return KnownFile(isbn="9780000000002", title="A Book", author="An Author", kind="non-fiction", chapters=["Ch 1"])


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
        return response_model(summary="looks right")

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
        return response_model(summary="looks right")

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
    search_client.search.return_value = [_BOOK_RESULT]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        return response_model(
            summary="wrong edition",
            issues=[
                verify.VerifyIssue(
                    kind="chapter",
                    field="chapter 1",
                    claimed="Ch 1",
                    status="contradicted",
                    expected="Chapter One Real Title",
                    source=1,
                    source_is_table_of_contents=True,
                )
            ],
        )

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    with pytest.raises(ValueError) as exc_info:
        await verify.run(
            {"known_file": _known_file().model_dump(), "trust_known": False},
            search_client=search_client,
            llm_client=AsyncMock(),
        )
    message = str(exc_info.value)
    assert "wrong edition" in message
    assert (
        "chapter 1: known-file says 'Ch 1'; sources say 'Chapter One Real Title' (https://example.com/toc)" in message
    )
    assert "--trust-known" in message


async def _run_with_issues(
    monkeypatch, issues: list[verify.VerifyIssue], results: list[SearchResult] | None = None
) -> dict:
    search_client = AsyncMock()
    search_client.search.return_value = results or [_BOOK_RESULT]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        return response_model(summary="right book", issues=issues)

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)
    return await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_unconfirmed_issues_pass_and_are_reported(monkeypatch):
    # The Diet Myth case: chapters that just aren't in the search snippets
    # must not fail the run.
    result = await _run_with_issues(
        monkeypatch,
        [verify.VerifyIssue(kind="chapter", field="chapter 1", claimed="Ch 1", status="unconfirmed")],
    )
    assert result["verified"] is True
    assert "Unconfirmed (not a failure by itself):" in result["verify_reason"]
    assert "chapter 1: known-file says 'Ch 1' — not found in search results" in result["verify_reason"]


@pytest.mark.asyncio
async def test_contradiction_without_evidence_is_downgraded_to_unconfirmed(monkeypatch):
    result = await _run_with_issues(
        monkeypatch,
        [verify.VerifyIssue(kind="chapter", field="chapter 1", claimed="Ch 1", status="contradicted")],
    )
    assert result["verified"] is True
    assert "Unconfirmed" in result["verify_reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["isbn", "year", "page_count", "other"])
async def test_edition_level_contradictions_never_fail(monkeypatch, kind):
    # Keyed on the structured kind, not the free-text label — "publication
    # year" or "number of pages" must not slip through as fatal.
    result = await _run_with_issues(
        monkeypatch,
        [verify.VerifyIssue(kind=kind, field="publication year", claimed="2020", status="contradicted", expected="2015")],
    )
    assert result["verified"] is True
    assert "publication year: known-file says '2020'; sources say '2015'" in result["verify_reason"]


@pytest.mark.asyncio
async def test_results_that_dont_name_the_book_are_not_shown_to_the_model(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [
        _BOOK_RESULT,
        SearchResult(title="Diet tips", url="https://other", content="unrelated listicle"),
    ]
    captured = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        captured["prompt"] = messages[1]["content"]
        return response_model(summary="ok")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)
    await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )
    assert "https://example.com/toc" in captured["prompt"]
    assert "https://other" not in captured["prompt"]


@pytest.mark.asyncio
async def test_wrong_author_is_caught_from_results_crediting_someone_else(monkeypatch):
    # Results aren't filtered on the claimed author, so a page crediting the
    # real author still reaches the model.
    real = SearchResult(title="A Book by Real Writer", url="https://real", content="A Book, ISBN 978-0-00-000000-2")
    with pytest.raises(ValueError, match="Stage 1 verify failed"):
        await _run_with_issues(
            monkeypatch,
            [
                verify.VerifyIssue(
                    kind="author",
                    field="author",
                    claimed="An Author",
                    status="contradicted",
                    expected="Real Writer",
                    source=1,
                )
            ],
            results=[real],
        )


@pytest.mark.asyncio
async def test_same_title_book_by_another_author_cant_contradict_chapters(monkeypatch):
    other = SearchResult(title="A Book by Someone Else", url="https://same-title", content="A Book contents")
    result = await _run_with_issues(
        monkeypatch,
        [
            verify.VerifyIssue(
                kind="chapter",
                field="chapter 1",
                claimed="Ch 1",
                status="contradicted",
                expected="Something else",
                source=2,
                source_is_table_of_contents=True,
            )
        ],
        results=[_BOOK_RESULT, other],
    )
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_no_results_about_the_book_fails_before_the_model_call(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="Diet tips", url="https://other", content="listicle")]
    llm_called = False

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        nonlocal llm_called
        llm_called = True

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)
    with pytest.raises(ValueError, match="couldn't find this book online.*--trust-known"):
        await verify.run(
            {"known_file": _known_file().model_dump(), "trust_known": False},
            search_client=search_client,
            llm_client=AsyncMock(),
        )
    assert not llm_called
    assert search_client.search.call_count == 2  # the fallback query ran too


@pytest.mark.asyncio
async def test_same_short_title_by_another_author_cant_contradict_the_author(monkeypatch):
    # A different "A Book" with no matching subtitle or ISBN.
    other = SearchResult(title="A Book by Someone Else", url="https://same-title", content="A Book")
    result = await _run_with_issues(
        monkeypatch,
        [
            verify.VerifyIssue(
                kind="author",
                field="author",
                claimed="An Author",
                status="contradicted",
                expected="Someone Else",
                source=1,
            )
        ],
        results=[other],
    )
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_title_contradiction_never_fails(monkeypatch):
    result = await _run_with_issues(
        monkeypatch,
        [
            verify.VerifyIssue(
                kind="title",
                field="subtitle",
                claimed="A Book: US Subtitle",
                status="contradicted",
                expected="A Book: UK Subtitle",
                source=1,
            )
        ],
    )
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_search_is_deep_and_falls_back_to_title_without_author():
    search_client = AsyncMock()
    search_client.search.return_value = []
    known_file = KnownFile(
        isbn="9781474619301",
        title="The Diet Myth: The Real Science Behind What We Eat",
        author="Tim Spector",
        kind="non-fiction",
        chapters=["Introduction: A Bad Taste"],
    )
    with pytest.raises(ValueError):
        await verify.run(
            {"known_file": known_file.model_dump(), "trust_known": False},
            search_client=search_client,
            llm_client=AsyncMock(),
        )
    first, fallback = search_client.search.call_args_list
    assert first.args[0].startswith("The Diet Myth Tim Spector")
    assert fallback.args[0] == '"The Diet Myth" book'
    assert first.kwargs["deep"] is True
    assert fallback.kwargs["deep"] is False


@pytest.mark.asyncio
async def test_known_parts_are_included_in_search_query_and_prompt(monkeypatch):
    known_file = KnownFile(
        isbn="123",
        title="A Book",
        author="An Author",
        kind="non-fiction",
        chapters=["Ch 1", "Ch 2"],
        parts=[KnownPart(title="Part One", chapters=[1, 2])],
    )
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="matches the book")]

    captured_prompt = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        captured_prompt["content"] = messages[1]["content"]
        return response_model(summary="looks right")

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
    assert "claimed parts and their chapter groupings" in captured_prompt["content"]


@pytest.mark.asyncio
async def test_no_parts_question_when_known_file_has_no_parts(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [SearchResult(title="t", url="u", content="matches the book")]

    captured_prompt = {}

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        captured_prompt["content"] = messages[1]["content"]
        return response_model(summary="looks right")

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)

    await verify.run(
        {"known_file": _known_file().model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )

    assert "claimed parts and their chapter groupings" not in captured_prompt["content"]


@pytest.mark.asyncio
async def test_chapter_contradiction_from_a_summary_site_does_not_fail(monkeypatch):
    # The Diet Myth case: a summary site's own section headings "contradict"
    # every real chapter title.
    result = await _run_with_issues(
        monkeypatch,
        [
            verify.VerifyIssue(
                kind="chapter",
                field="chapter 1",
                claimed="Introduction: A Bad Taste",
                status="contradicted",
                expected="Microbes: The Hidden Influence on Our Health and Diet",
                source=1,
            )
        ],
    )
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_author_contradiction_fails_without_a_contents_listing(monkeypatch):
    with pytest.raises(ValueError, match="Stage 1 verify failed"):
        await _run_with_issues(
            monkeypatch,
            [verify.VerifyIssue(kind="author", field="author", claimed="An Author", status="contradicted",
                                expected="Someone Else", source=1)],
        )


@pytest.mark.asyncio
async def test_many_unconfirmed_chapters_collapse_to_one_line(monkeypatch):
    issues = [
        verify.VerifyIssue(kind="chapter", field=f"chapter {n}", claimed=f"Ch {n}", status="unconfirmed")
        for n in range(1, 6)
    ]
    issues.append(verify.VerifyIssue(kind="year", field="year", claimed="2020", status="unconfirmed"))
    reason = (await _run_with_issues(monkeypatch, issues))["verify_reason"]
    assert "  - 5 chapters: not found in search results" in reason
    assert "year: known-file says '2020'" in reason
    assert "Ch 1" not in reason


@pytest.mark.asyncio
async def test_search_service_returning_nothing_is_not_reported_as_book_not_found():
    search_client = AsyncMock()
    search_client.search.return_value = []
    with pytest.raises(ValueError, match="search service returned no results at all"):
        await verify.run(
            {"known_file": _known_file().model_dump(), "trust_known": False},
            search_client=search_client,
            llm_client=AsyncMock(),
        )


@pytest.mark.asyncio
async def test_out_of_range_source_number_cant_back_a_contradiction(monkeypatch):
    result = await _run_with_issues(
        monkeypatch,
        [verify.VerifyIssue(kind="author", field="author", claimed="An Author", status="contradicted",
                            expected="Someone Else", source=7)],
    )
    assert result["verified"] is True
    assert "sources say 'Someone Else'" in result["verify_reason"]


@pytest.mark.asyncio
async def test_non_fatal_contradiction_is_shown_as_disputed_not_unconfirmed(monkeypatch):
    reason = (
        await _run_with_issues(
            monkeypatch,
            [
                verify.VerifyIssue(kind="year", field="year", claimed="2020", status="contradicted", expected="2015"),
                verify.VerifyIssue(kind="chapter", field="chapter 1", claimed="Ch 1", status="unconfirmed"),
            ],
        )
    )["verify_reason"]
    disputed, unconfirmed = reason.split("Unconfirmed (not a failure by itself):")
    assert "Contradicted, but not by a source that settles it" in disputed
    assert "year: known-file says '2020'; sources say '2015'" in disputed
    assert "chapter 1" in unconfirmed
