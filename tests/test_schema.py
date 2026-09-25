import pytest
from pydantic import ValidationError

from precis.schema import Book, Chapter, KeyClaim, Part


def make_chapter(number: int = 1, key_points: list[str] | None = None) -> Chapter:
    return Chapter(
        number=number,
        title=f"Chapter {number}",
        key_points=["point one"] if key_points is None else key_points,
        core_claim="the core claim",
    )


def test_chapter_key_points_capped_at_six():
    with pytest.raises(ValidationError):
        make_chapter(key_points=[f"point {i}" for i in range(7)])


def test_chapter_key_points_requires_at_least_one():
    with pytest.raises(ValidationError):
        make_chapter(key_points=[])


def test_chapter_key_points_six_is_allowed():
    make_chapter(key_points=[f"point {i}" for i in range(6)])


def make_key_claims(count: int = 3) -> list[KeyClaim]:
    return [KeyClaim(prompt=f"q{i}", answer=f"a{i}") for i in range(count)]


def _base_book_kwargs() -> dict:
    return {
        "title": "Some Book",
        "author": "Some Author",
        "isbn": "123",
        "kind": "non-fiction",
        "one_line_takeaway": "takeaway",
        "synopsis": "synopsis",
        "tags": ["history", "philosophy"],
    }


def test_fiction_book_with_no_chapters_and_no_claims_is_valid():
    Book(**_base_book_kwargs())


def test_nonfiction_book_with_chapters_and_claims_is_valid():
    Book(
        **_base_book_kwargs(),
        chapters=[make_chapter(1)],
        key_claims_for_review=make_key_claims(),
    )


def test_chapters_without_key_claims_is_rejected():
    with pytest.raises(ValidationError):
        Book(**_base_book_kwargs(), chapters=[make_chapter(1)])


def test_key_claims_without_chapters_is_rejected():
    with pytest.raises(ValidationError):
        Book(
            **_base_book_kwargs(),
            key_claims_for_review=make_key_claims(),
        )


def test_part_referencing_unknown_chapter_number_is_rejected():
    with pytest.raises(ValidationError):
        Book(
            **_base_book_kwargs(),
            chapters=[make_chapter(1)],
            key_claims_for_review=make_key_claims(),
            parts=[Part(title="Part 1", summary="s", chapters=[1, 2])],
        )


def test_part_referencing_known_chapter_number_is_valid():
    Book(
        **_base_book_kwargs(),
        chapters=[make_chapter(1)],
        key_claims_for_review=make_key_claims(),
        parts=[Part(title="Part 1", summary="s", chapters=[1])],
    )


def test_tags_outside_vocabulary_error_has_tags_loc():
    """Regression test: a tags-vocabulary error used to raise from a
    whole-model validator (empty `loc`), indistinguishable from the parts/
    chapter-number check that shares the same validator — which made
    assemble.py's _is_repairable treat a tags failure as an unrepairable
    parts problem whenever parts_source == "known". Tags validation is now
    a field_validator, so the error must be attributed to `tags` specifically.
    """
    kwargs = {**_base_book_kwargs(), "tags": ["not-a-real-tag", "also-fake"]}
    with pytest.raises(ValidationError) as exc_info:
        Book(**kwargs)

    (error,) = exc_info.value.errors()
    assert error["loc"] == ("tags",)


def test_duplicate_tags_rejected():
    kwargs = {**_base_book_kwargs(), "tags": ["history", "history"]}
    with pytest.raises(ValidationError, match="must not repeat"):
        Book(**kwargs)
