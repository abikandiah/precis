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


def _base_book_kwargs() -> dict:
    return {
        "title": "Some Book",
        "author": "Some Author",
        "isbn": "123",
        "one_line_takeaway": "takeaway",
        "synopsis": "synopsis",
        "tags": ["tag"],
    }


def test_fiction_book_with_no_chapters_and_no_claims_is_valid():
    Book(**_base_book_kwargs())


def test_nonfiction_book_with_chapters_and_claims_is_valid():
    Book(
        **_base_book_kwargs(),
        chapters=[make_chapter(1)],
        key_claims_for_review=[KeyClaim(prompt="q", answer="a")],
    )


def test_chapters_without_key_claims_is_rejected():
    with pytest.raises(ValidationError):
        Book(**_base_book_kwargs(), chapters=[make_chapter(1)])


def test_key_claims_without_chapters_is_rejected():
    with pytest.raises(ValidationError):
        Book(
            **_base_book_kwargs(),
            key_claims_for_review=[KeyClaim(prompt="q", answer="a")],
        )


def test_part_referencing_unknown_chapter_number_is_rejected():
    with pytest.raises(ValidationError):
        Book(
            **_base_book_kwargs(),
            chapters=[make_chapter(1)],
            key_claims_for_review=[KeyClaim(prompt="q", answer="a")],
            parts=[Part(title="Part 1", summary="s", chapter_numbers=[1, 2])],
        )


def test_part_referencing_known_chapter_number_is_valid():
    Book(
        **_base_book_kwargs(),
        chapters=[make_chapter(1)],
        key_claims_for_review=[KeyClaim(prompt="q", answer="a")],
        parts=[Part(title="Part 1", summary="s", chapter_numbers=[1])],
    )
