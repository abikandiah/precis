from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import verify
from precis.schema import KnownFile, KnownPart
from precis.search import SearchResult

# Names the title, the author and the ISBN, so it can back an author
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
async def test_author_contradiction_raises_with_the_difference_and_a_way_out(monkeypatch):
    search_client = AsyncMock()
    search_client.search.return_value = [_BOOK_RESULT]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        return response_model(
            summary="wrong book",
            issues=[
                verify.VerifyIssue(
                    kind="author",
                    field="author",
                    claimed="An Author",
                    status="contradicted",
                    expected="Real Writer",
                    source=1,
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
    assert message.startswith("verify failed: the known-file's author doesn't match the book found online:\n")
    assert (
        "author: known-file says 'An Author'; search results say 'Real Writer' (https://example.com/toc)" in message
    )
    assert "wrong book" in message
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
async def test_unconfirmed_issues_are_not_listed(monkeypatch):
    # "Not in the search results" is the normal case for most fields —
    # listing it buried the differences that matter.
    result = await _run_with_issues(
        monkeypatch,
        [
            verify.VerifyIssue(kind="chapter", field="chapter 1", claimed="Ch 1", status="unconfirmed"),
            verify.VerifyIssue(kind="year", field="year", claimed="2020", status="unconfirmed"),
            # A "contradiction" with nothing to contradict it with.
            verify.VerifyIssue(kind="author", field="author", claimed="An Author", status="contradicted"),
        ],
    )
    assert result == {"verified": True, "verify_reason": "right book"}


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
    assert "publication year: known-file says '2020'; search results say '2015'" in result["verify_reason"]


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
    with pytest.raises(ValueError, match="verify failed"):
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
@pytest.mark.parametrize("kind", ["chapter", "part"])
async def test_chapter_and_part_contradictions_only_warn(monkeypatch, kind):
    # The known-file owns the chapter list and parts. Even a source that
    # looks like a real contents listing for this book only warns — summary
    # sites label their own headings "Chapter 1", "Chapter 2" too.
    result = await _run_with_issues(
        monkeypatch,
        [
            verify.VerifyIssue(
                kind=kind, field=f"{kind} 1", claimed="Ch 1", status="contradicted", expected="Chapter One", source=1
            )
        ],
    )
    assert result["verified"] is True
    assert result["verify_reason"] == (
        "right book\n"
        "Differences from search results (warnings only — the known-file is used as written):\n"
        f"  - {kind} 1: known-file says 'Ch 1'; search results say 'Chapter One' (https://example.com/toc)"
    )


@pytest.mark.asyncio
async def test_many_chapter_differences_from_one_source_collapse_to_one_line(monkeypatch):
    # The Diet Myth case: a summary site's own headings "differ" from most
    # real chapter titles — one line, not a wall of them.
    summary_site = SearchResult(title="A Book summary", url="https://summaries.example/a-book", content="A Book")
    issues = [
        verify.VerifyIssue(
            kind="chapter", field=f"chapter {n}", claimed=f"Ch {n}", status="contradicted",
            expected=f"Heading {n}", source=2,
        )
        for n in range(1, 5)
    ]
    issues.append(
        verify.VerifyIssue(
            kind="chapter", field="chapter 9", claimed="Ch 9", status="contradicted", expected="Chapter Nine", source=1
        )
    )
    reason = (await _run_with_issues(monkeypatch, issues, results=[_BOOK_RESULT, summary_site]))["verify_reason"]
    assert "chapter 9: known-file says 'Ch 9'; search results say 'Chapter Nine'" in reason
    assert "  - 4 chapter/part titles differ from https://summaries.example/a-book" in reason
    assert "Heading" not in reason


@pytest.mark.asyncio
async def test_author_contradiction_from_a_source_naming_this_book_fails(monkeypatch):
    with pytest.raises(ValueError, match="verify failed"):
        await _run_with_issues(
            monkeypatch,
            [verify.VerifyIssue(kind="author", field="author", claimed="An Author", status="contradicted",
                                expected="Someone Else", source=1)],
        )


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
    assert "search results say 'Someone Else'" in result["verify_reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", ["Timothy Spector", "Tim Spector with Jane Doe", "T. D. Spector"])
async def test_another_form_of_the_same_author_never_fails(monkeypatch, expected):
    known_file = KnownFile(isbn="9780000000002", title="A Book", author="Tim Spector", kind="non-fiction", chapters=["Ch 1"])
    search_client = AsyncMock()
    search_client.search.return_value = [_BOOK_RESULT]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        issue = verify.VerifyIssue(
            kind="author", field="author", claimed="Tim Spector", status="contradicted", expected=expected, source=1
        )
        return response_model(summary="right book", issues=[issue])

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)
    result = await verify.run(
        {"known_file": known_file.model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )
    assert result["verified"] is True
    assert f"search results say {expected!r}" in result["verify_reason"]


@pytest.mark.asyncio
async def test_chapter_differences_without_a_citable_source_are_listed_not_collapsed(monkeypatch):
    # With no source number they may come from several pages — naming one
    # "summary site" for all of them would be a guess.
    issues = [
        verify.VerifyIssue(
            kind="chapter", field=f"chapter {n}", claimed=f"Ch {n}", status="contradicted", expected=f"Heading {n}"
        )
        for n in range(1, 5)
    ]
    reason = (await _run_with_issues(monkeypatch, issues))["verify_reason"]
    assert all(f"chapter {n}: known-file says 'Ch {n}'" in reason for n in range(1, 5))
    assert "chapter/part titles differ from" not in reason


@pytest.mark.asyncio
async def test_author_with_no_comparable_surname_never_fails(monkeypatch):
    known_file = KnownFile(isbn="9780000000002", title="A Book", author="X", kind="non-fiction", chapters=["Ch 1"])
    search_client = AsyncMock()
    search_client.search.return_value = [_BOOK_RESULT]

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        issue = verify.VerifyIssue(
            kind="author", field="author", claimed="X", status="contradicted", expected="Someone Else", source=1
        )
        return response_model(summary="right book", issues=[issue])

    monkeypatch.setattr(verify.llm, "complete_structured", fake_complete_structured)
    result = await verify.run(
        {"known_file": known_file.model_dump(), "trust_known": False},
        search_client=search_client,
        llm_client=AsyncMock(),
    )
    assert result["verified"] is True
