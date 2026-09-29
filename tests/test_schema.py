import pytest
from pydantic import ValidationError

from precis.schema import Book, Idea, KeyClaim, KnownFile, deck_coverage_warnings

PLACEHOLDER = "TODO: fill in by hand"


def _book(kind: str = "non-fiction", ideas: int = 5, claims: int | None = 5, **overrides) -> dict:
    return {
        "title": "T",
        "author": "A",
        "isbn": "1",
        "kind": kind,
        "one_line_takeaway": "take",
        "synopsis": "syn",
        "ideas": [{"title": f"Idea {n}", "summary": "s", "evidence": "e"} for n in range(ideas)],
        "key_claims_for_review": None if claims is None else [{"prompt": "Q?", "answer": "A."}] * claims,
        "tags": ["psychology", "science"] if kind == "non-fiction" else ["dystopian", "fiction-literary"],
        **overrides,
    }


def test_a_valid_book_of_each_kind():
    assert Book.model_validate(_book()).schema_version == "2"
    assert Book.model_validate(_book("fiction", ideas=3, claims=None)).key_claims_for_review is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_book(ideas=0), "at least one idea"),
        (_book(claims=None), "non-fiction needs key_claims_for_review"),
        (_book("fiction", ideas=3), "fiction has no key_claims_for_review"),
        (_book(tags=["psychology"]), "at least 2 items"),
        (_book(tags=["psychology", "psychology"]), "must not repeat"),
        (_book("fiction", ideas=3, claims=None, tags=["psychology", "science"]), "closed fiction vocabulary"),
    ],
)
def test_book_shape_is_validated_per_kind(data, message):
    with pytest.raises(ValidationError, match=message):
        Book.model_validate(data)


def test_any_number_of_ideas_and_claims_is_valid():
    # A book has as many ideas as it makes: 48 Laws of Power has 48.
    for ideas, claims in ((1, 1), (48, 48)):
        assert len(Book.model_validate(_book(ideas=ideas, claims=claims)).ideas) == ideas


def test_a_deck_covering_under_half_the_ideas_is_a_warning():
    ideas = [Idea(title=f"Idea {n}", summary="s", evidence="e") for n in range(25)]
    claims = [KeyClaim(prompt="Q?", answer="A.")]
    assert deck_coverage_warnings("non-fiction", ideas, claims * 2) == [
        "2 key claims for 25 ideas — the review deck covers under half the notes"
    ]
    assert deck_coverage_warnings("non-fiction", ideas[:4], claims * 2) == []
    assert deck_coverage_warnings("fiction", ideas, None) == []


def test_known_file_title_and_author_placeholders_dont_count():
    assert KnownFile(isbn="1", kind="fiction", title="T", author="A").has_title
    for value in (None, "", PLACEHOLDER):
        known_file = KnownFile(isbn="1", kind="fiction", title=value, author=value)
        assert not known_file.has_title and not known_file.has_author


def test_an_unknown_known_file_field_is_an_error():
    with pytest.raises(ValidationError, match="note"):
        KnownFile(isbn="1", kind="fiction", note="typo for notes")  # type: ignore[call-arg]
