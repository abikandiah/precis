import pytest

from precis.schema import PLACEHOLDER
from precis.search import (
    SearchResult,
    author_names,
    author_surnames,
    is_book_relevant,
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


@pytest.mark.parametrize(
    ("author", "expected"),
    [
        ("Daniel Kahneman", [["daniel", "kahneman"]]),
        ("Spector, Tim", [["tim", "spector"]]),
        ("J.K. Rowling", [["j", "k", "rowling"]]),
        ("Carl Sagan and Ann Druyan (editor)", [["carl", "sagan"], ["ann", "druyan"]]),
        ("Dr. Tim Spector", [["tim", "spector"]]),
        ("Prof Sir Tim Spector", [["tim", "spector"]]),
        ("Miss", [["miss"]]),  # a title is never the whole name
        (None, []),
    ],
)
def test_author_names_gives_each_person_given_names_first(author, expected):
    assert author_names(author) == expected
