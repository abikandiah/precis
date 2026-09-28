from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from precis import review, write
from precis.research import Research, Source
from precis.schema import Book, KnownFile

KNOWN = KnownFile(isbn="1", title="Good to Great", author="Jim Collins", kind="non-fiction")
RESEARCH = Research(sources=[Source(id="S1", title="t", url="u", text="text")], warnings=["research warning"])
SYNOPSIS = "Paragraph one of the synopsis.\n\nParagraph two of the synopsis.\n\nParagraph three."


def _idea(n: int, sources: list[str] | None = None) -> dict:
    return {"title": f"Idea {n}", "summary": f"topic{n}word", "evidence": "e", "sources": sources if sources is not None else ["S1"]}


def _book(kind: str = "non-fiction", ideas: int = 6) -> Book:
    return Book.model_validate({
        "title": "T", "author": "A", "isbn": "1", "kind": kind, "one_line_takeaway": "take", "synopsis": SYNOPSIS,
        "ideas": [_idea(n) for n in range(ideas)],
        "key_claims_for_review": [{"prompt": f"Q{n}?", "answer": "A."} for n in range(5)] if kind == "non-fiction" else None,
        "tags": ["business", "economics"] if kind == "non-fiction" else ["dystopian", "drama"],
        "warnings": ["research warning"],
    })  # fmt: skip


def _verdicts(n: int = 6, **overrides: dict) -> list[dict]:
    """Keep verdicts for ideas 0..n-1, with some replaced by index."""
    verdicts = [{"title": f"Idea {i}", "verdict": "keep"} for i in range(n)]
    for index, verdict in overrides.items():
        i = int(index.removeprefix("i"))
        verdicts[i] = {"title": f"Idea {i}", **verdict}
    return verdicts


_CLAIMS = [{"prompt": f"Q{n}?", "answer": "A."} for n in range(5)]


def _validate(data: dict, book: Book | None = None) -> review.Review:
    return review.Review.model_validate(data, context={review.BOOK_KEY: book or _book(), write.SOURCE_IDS_KEY: {"S1"}})


def _apply(data: dict, book: Book | None = None) -> tuple[Book, list[str]]:
    book = book or _book()
    return review.apply_review(book, _validate(data, book), RESEARCH)


# --- validation: the book the review would produce --------------------------------------


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"ideas": _verdicts(5)}, "one verdict per idea, in order"),
        # A skipped verdict can't shift the rest onto the wrong ideas.
        ({"ideas": _verdicts(6)[1:] + [{"title": "Idea 0", "verdict": "keep"}]}, "one verdict per idea, in order"),
        ({"ideas": _verdicts(i1={"verdict": "revise", "reason": "r"})}, "revise but has no revised idea"),
        ({"ideas": _verdicts(), "new_ideas": [_idea(n) for n in range(10, 17)]}, "5-12 ideas .*got 13"),
        ({"ideas": _verdicts(), "new_ideas": [_idea(9, ["S4"])]}, r"cites \['S4'\]"),
        ({"ideas": _verdicts(), "new_ideas": [_idea(9, [])]}, "new idea 'Idea 9' must cite"),
        ({"ideas": _verdicts(), "key_claims_for_review": _CLAIMS[:3]}, "5-15"),
        (
            {"ideas": _verdicts(i0={"verdict": "drop"}, i1={"verdict": "drop"})},
            "keep or revise an idea you dropped",
        ),
        # Claims resting on a dropped idea must be dealt with.
        ({"ideas": _verdicts(i0={"verdict": "drop", "reason": "contradicted"})}, "return key_claims_for_review"),
        ({"ideas": _verdicts(), "synopsis": "Paragraph two, fixed."}, "the whole corrected synopsis"),
    ],
)
def test_the_review_is_validated_against_the_book_it_would_produce(data, message):
    with pytest.raises(ValidationError, match=message):
        _validate(data)


def test_fiction_rejects_a_review_deck():
    fiction = _book("fiction", ideas=4)
    with pytest.raises(ValidationError, match="fiction has no"):
        _validate({"ideas": _verdicts(4), "key_claims_for_review": _CLAIMS}, fiction)


def test_empty_or_placeholder_fields_mean_nothing_to_fix():
    data = {
        "ideas": [{"title": f"Idea {i}", "verdict": "keep", "reason": None} for i in range(6)],
        "new_ideas": None,
        "one_line_takeaway": "",
        "synopsis": "N/A",
        "key_claims_for_review": [],
    }
    reviewed, changes = _apply(data)
    assert (reviewed.one_line_takeaway, reviewed.synopsis) == ("take", SYNOPSIS)
    assert reviewed.key_claims_for_review == _book().key_claims_for_review
    assert changes == []


def test_a_revised_idea_on_a_keep_or_drop_is_ignored_even_if_malformed():
    verdicts = _verdicts(
        i0={"verdict": "keep", "revised": {"title": "x", "sources": ["S0"]}},
        i1={"verdict": "drop", "reason": "repeat", "revised": _idea(9, ["S9"])},
    )
    _validate({"ideas": verdicts, "key_claims_for_review": _CLAIMS})


# --- applying it ----------------------------------------------------------------------


def test_keep_revise_drop_and_add():
    verdicts = _verdicts(
        i1={"verdict": "revise", "reason": "vague", "revised": _idea(10)},
        i2={"verdict": "drop", "reason": "generic"},
    )
    new_synopsis = SYNOPSIS.replace("three", "three, corrected")
    reviewed, changes = _apply(
        {"ideas": verdicts, "new_ideas": [_idea(11)], "synopsis": new_synopsis, "key_claims_for_review": _CLAIMS[:5]}
    )
    assert [i.title for i in reviewed.ideas] == ["Idea 0", "Idea 10", "Idea 3", "Idea 4", "Idea 5", "Idea 11"]
    assert reviewed.synopsis == new_synopsis and reviewed.one_line_takeaway == "take"
    assert changes == [
        'revised "Idea 1": vague', 'dropped "Idea 2": generic', 'added "Idea 11"',
        "revised key_claims_for_review", "revised synopsis",
    ]  # fmt: skip


def test_no_changes_leaves_the_book_as_it_was():
    book = _book()
    reviewed, changes = _apply({"ideas": _verdicts()}, book)
    assert (reviewed, changes) == (book, [])


def test_a_revised_idea_that_leaves_out_its_sources_keeps_the_originals():
    revised = {"title": "Idea 1, sharper", "summary": "s", "evidence": "e"}
    reviewed, _ = _apply({"ideas": _verdicts(i1={"verdict": "revise", "reason": "vague", "revised": revised})})
    assert reviewed.ideas[1].sources == ["S1"]


def test_revising_a_citation_away_names_the_idea_in_the_uncited_warning():
    verdicts = _verdicts(i0={"verdict": "revise", "reason": "S1 doesn't say this", "revised": _idea(0, sources=[])})
    reviewed, _ = _apply({"ideas": verdicts})
    assert reviewed.warnings[1] == "1 idea(s) rest on the model's knowledge of the book, not the research — check them: Idea 0"


def test_mostly_uncited_ideas_warn_of_thin_research():
    verdicts = _verdicts(**{f"i{n}": {"verdict": "revise", "reason": "r", "revised": _idea(n, sources=[])} for n in range(4)})
    reviewed, _ = _apply({"ideas": verdicts})
    assert "most ideas aren't backed by the research" in reviewed.warnings[2]


def test_what_the_checks_still_find_after_the_review_becomes_a_warning():
    invented = {"title": "Idea 1", "summary": "s", "evidence": "Smith's 1999 study", "sources": ["S1"]}
    reviewed, _ = _apply({"ideas": _verdicts(i1={"verdict": "revise", "reason": "r", "revised": invented})})
    assert any(w.startswith("after review, idea 2 \"Idea 1\": its evidence names ['1999', 'Smith']") for w in reviewed.warnings)


# --- the call -------------------------------------------------------------------------------


async def test_review_notes_sends_the_write_calls_exact_prefix():
    book = _book()
    call = AsyncMock(return_value=_validate({"ideas": _verdicts()}))
    with patch.object(review.llm, "complete_structured", new=call):
        await review.review_notes(KNOWN, RESEARCH, book, ["idea 2: a finding"], client=MagicMock())
    kwargs = call.await_args.kwargs
    system, user = kwargs["messages"]
    # Same tools and system message as the write call, so the cached research is reused.
    assert system == write.context_messages(KNOWN, RESEARCH)[0]
    assert kwargs["tool_models"] == write.shared_tools("non-fiction")
    assert kwargs["response_model"] is review.Review
    assert "- idea 2: a finding" in user["content"] and '"title": "Idea 5"' in user["content"]
    assert "5-12 ideas" in user["content"] and "5-15 key claims" in user["content"]
    assert kwargs["validation_context"] == {review.BOOK_KEY: book, write.SOURCE_IDS_KEY: {"S1"}}


def test_fiction_review_audits_for_spoilers_with_the_write_prompts_guards():
    text = review._instructions(_book("fiction", ideas=4), [])
    assert "The research contains spoilers" in text and "When unsure whether something is a spoiler" in text
    assert "3-6 ideas" in text and "key claims" not in text and "key_claims_for_review" not in text
