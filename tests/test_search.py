from unittest.mock import AsyncMock

import pytest

from precis.schema import PLACEHOLDER
from precis.search import (
    SearchResult,
    author_surnames,
    identifies_book,
    is_book_relevant,
    search_book,
    short_title,
)

DIET_MYTH = "The Diet Myth: The Real Science Behind What We Eat"


def _result(content: str, title: str = "t", url: str = "https://example.org") -> SearchResult:
    return SearchResult(title=title, url=url, content=content)


def test_short_title_drops_subtitle_and_placeholder():
    assert short_title(DIET_MYTH) == "The Diet Myth"
    assert short_title("Brain") == "Brain"
    assert short_title("Thinking, Fast and Slow (Revised Edition)") == "Thinking, Fast and Slow"
    assert short_title("The (Mis)Behavior of Markets: A Fractal View") == "The (Mis)Behavior of Markets"
    assert short_title(None) == ""
    assert short_title(PLACEHOLDER) == ""


@pytest.mark.parametrize(
    ("author", "expected"),
    [
        ("Tim Spector", ["spector"]),
        ("Martin Luther King Jr.", ["king"]),
        ("Carl Sagan and Ann Druyan", ["sagan", "druyan"]),
        ("Carl Sagan & Ann Druyan", ["sagan", "druyan"]),
        ("Tim Spector with Jane Doe", ["spector", "doe"]),
        ("Richard P. Feynman; Ralph Leighton", ["feynman", "leighton"]),
        ("Richard P. Feynman, Ralph Leighton", ["feynman", "leighton"]),
        ("Tim Spector (Author), Jane Doe (Translator)", ["spector", "doe"]),
        ("edited by Jane Doe", ["doe"]),
        ("Spector, Tim", ["spector"]),
        ("Amy Wu", ["wu"]),
        (None, []),
        (PLACEHOLDER, []),
    ],
)
def test_author_surnames(author, expected):
    assert author_surnames(author) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Tim Spector's The Diet Myth argues fibre feeds gut microbes", True),
        ("SPECTOR on fibre in Diet Myth", True),
        ("The Diet Myth — The Real Science Behind What We Eat, reviewed", True),
        # The short title alone matches generic listicles.
        ("Ten common diet myths debunked", False),
        ("The diet myth about breakfast", False),
        # The surname alone matches anyone else called Spector.
        ("Phil Spector's wall of sound", False),
        ("Inspectors found diet myth claims misleading", False),
    ],
)
def test_is_book_relevant(content, expected):
    assert is_book_relevant(_result(content), title=DIET_MYTH, author="Tim Spector") is expected


def test_common_word_surname_needs_the_title_too():
    title, author = "Daring Greatly", "Brené Brown"
    assert not is_book_relevant(_result("Brown rice is a whole grain"), title=title, author=author)
    assert is_book_relevant(_result("In Daring Greatly, Brown argues..."), title=title, author=author)


def test_title_and_url_count_as_well_as_content():
    assert is_book_relevant(_result("fibre", title="Spector's Diet Myth"), title="Diet Myth", author="Tim Spector")
    assert is_book_relevant(
        _result("fibre", url="https://example.org/tim-spector-diet-myth"), title="Diet Myth", author="Tim Spector"
    )


def test_punctuation_and_apostrophe_style_are_ignored():
    title, author = "Man's Search for Meaning", "Viktor E. Frankl"
    assert is_book_relevant(_result("Frankl’s  Man’s Search for Meaning"), title=title, author=author)
    assert is_book_relevant(
        _result("Kahneman: Thinking Fast & Slow"), title="Thinking, Fast and Slow", author="Daniel Kahneman"
    )


def test_accents_are_folded():
    title, author = "One Hundred Years of Solitude", "Gabriel García Márquez"
    assert is_book_relevant(_result("Garcia Marquez's One Hundred Years of Solitude"), title=title, author=author)
    assert is_book_relevant(_result("García Márquez: One Hundred Years of Solitude"), title=title, author=author)


def test_short_surname_is_kept():
    assert is_book_relevant(_result("Wu's Brain explained"), title="Brain", author="Amy Wu")
    assert not is_book_relevant(_result("All about the brain"), title="Brain", author="Amy Wu")


def test_missing_author_falls_back_to_full_title_only():
    assert is_book_relevant(_result(f"{DIET_MYTH} reviewed"), title=DIET_MYTH, author=None)
    assert not is_book_relevant(_result("The Diet Myth reviewed"), title=DIET_MYTH, author=None)
    assert not is_book_relevant(_result("Thinking, Fast and Slow"), title="Thinking, Fast and Slow", author=None)


def test_missing_title_matches_on_author_alone():
    assert is_book_relevant(_result("Spector on fibre"), title=None, author="Tim Spector")


def _client(*batches: list[SearchResult]) -> AsyncMock:
    client = AsyncMock()
    client.search.side_effect = list(batches)
    return client


_ON_TOPIC = _result("Tim Spector's The Diet Myth on fibre")
_OFF_TOPIC = _result("generic diet myths")


@pytest.mark.asyncio
async def test_search_book_returns_first_query_with_relevant_results():
    client = _client([_OFF_TOPIC, _ON_TOPIC], [_ON_TOPIC])
    results = await search_book(["q1", "q2"], title=DIET_MYTH, author="Tim Spector", client=client)
    assert results == [_ON_TOPIC]
    client.search.assert_called_once_with("q1", max_results=10, deep=False)


@pytest.mark.asyncio
async def test_search_book_falls_through_empty_and_off_topic_batches():
    client = _client([], [_OFF_TOPIC], [_ON_TOPIC])
    results = await search_book(["q1", "q2", "q3"], title=DIET_MYTH, author="Tim Spector", client=client)
    assert results == [_ON_TOPIC]
    assert [c.args[0] for c in client.search.call_args_list] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_search_book_returns_empty_when_nothing_is_about_the_book():
    client = _client([_OFF_TOPIC], [_OFF_TOPIC])
    assert await search_book(["q1", "q2"], title=DIET_MYTH, author="Tim Spector", client=client) == []


@pytest.mark.parametrize(
    "printed", ["978-0-393-31755-8", "9780393317558", "ISBN-13 9780393317558", "ISBN-13: 978-0393317558", "0393317552"]
)
def test_identifies_book_by_isbn_however_its_printed(printed):
    result = SearchResult(title="Guns", url="https://x", content=f"ISBN {printed}")
    assert identifies_book(result, title="Guns", isbn="9780393317558")


def test_identifies_book_rejects_an_isbn_embedded_in_a_longer_number():
    result = SearchResult(title="Guns", url="https://x", content="order 197803933175581")
    assert not identifies_book(result, title="Guns", isbn="9780393317558")


@pytest.mark.parametrize("printed", ["https://example.com/range-9781594484964", "ISBN9781594484964"])
def test_identifies_book_by_isbn_glued_to_a_slug_or_label(printed):
    result = SearchResult(title="Range", url="https://x", content=printed)
    assert identifies_book(result, title="Range", isbn="9781594484964")
