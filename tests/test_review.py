from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from precis import review, write
from precis.research import Research, Source
from precis.schema import Book, KnownFile

KNOWN = KnownFile(isbn="1", title="Good to Great", author="Jim Collins", kind="non-fiction")
FICTION = KnownFile(isbn="2", title="1984", author="George Orwell", kind="fiction")
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
    return review.apply_review(book, _validate(data, book))


# --- validation: the book the review would produce --------------------------------------


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"ideas": [*_verdicts(), {"title": "Idea 9", "verdict": "keep"}]}, "'Idea 9' isn't one of the ideas"),
        ({"ideas": [*_verdicts(), {"title": "Idea 2", "verdict": "drop"}]}, "'Idea 2' has more than one verdict"),
        ({"ideas": _verdicts(i1={"verdict": "revise", "reason": "r"})}, "revise but has no revised idea"),
        ({"ideas": _verdicts(**{f"i{n}": {"verdict": "drop", "reason": "r"} for n in range(6)})}, "at least one idea"),
    ],
)
def test_the_review_is_validated_against_the_book_it_would_produce(data, message):
    with pytest.raises(ValidationError, match=message):
        _validate(data)


def test_counts_outside_the_limits_dont_fail_the_review():
    reviewed, _ = _apply({"ideas": _verdicts(), "new_ideas": [_idea(n) for n in range(10, 17)]})
    assert len(reviewed.ideas) == 13


def test_a_review_can_drop_below_the_minimum():
    verdicts = _verdicts(**{f"i{n}": {"verdict": "drop", "reason": "r"} for n in range(5)})
    reviewed, _ = _apply({"ideas": verdicts, "key_claims_for_review": _CLAIMS})
    assert [i.title for i in reviewed.ideas] == ["Idea 5"]


def test_fiction_ignores_a_review_deck():
    reviewed, _ = _apply({"ideas": _verdicts(4), "key_claims_for_review": _CLAIMS}, _book("fiction", ideas=4))
    assert reviewed.key_claims_for_review is None


def test_uncited_new_ideas_and_unknown_sources_are_left_out_not_sent_back():
    verdicts = _verdicts(i0={"verdict": "revise", "reason": "r", "revised": _idea(10, ["S1", "S7"])})
    new_ideas = [_idea(11, []), _idea(12, ["S9"]), _idea(13, ["S1", "S4"])]
    reviewed, changes = _apply({"ideas": verdicts, "new_ideas": new_ideas})
    assert [(i.title, i.sources) for i in reviewed.ideas if i.title in {"Idea 10", "Idea 13"}] == [
        ("Idea 10", ["S1"]),
        ("Idea 13", ["S1"]),
    ]
    assert not {"Idea 11", "Idea 12"} & {i.title for i in reviewed.ideas}
    assert 'left out new idea "Idea 11": it cites no research source' in changes
    assert "removed unknown sources ['S9'] from \"Idea 12\"" in changes


def test_a_drop_without_new_claims_keeps_them_with_a_warning():
    reviewed, _ = _apply({"ideas": _verdicts(i0={"verdict": "drop", "reason": "contradicted"})})
    assert reviewed.key_claims_for_review == _book().key_claims_for_review
    assert any("dropped Idea 0" in w and "key claims" in w for w in reviewed.warnings)


def test_a_partial_synopsis_is_left_out():
    reviewed, changes = _apply({"ideas": _verdicts(), "synopsis": "Paragraph two, fixed."})
    assert reviewed.synopsis == SYNOPSIS
    assert any("synopsis as written" in c for c in changes)


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


def test_verdicts_are_matched_by_title_and_an_idea_without_one_is_kept():
    # Out of order, idea 5 skipped: the drop still lands on idea 1.
    verdicts = _verdicts(i1={"verdict": "drop", "reason": "generic"})[:5][::-1]
    reviewed, changes = _apply({"ideas": verdicts, "key_claims_for_review": _CLAIMS})
    assert [i.title for i in reviewed.ideas] == ["Idea 0", "Idea 2", "Idea 3", "Idea 4", "Idea 5"]
    assert changes == [
        'dropped "Idea 1": generic', 'no verdict for "Idea 5", kept as written', "revised key_claims_for_review",
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
    meta = {"title": "Idea 1", "summary": "The author discusses it.", "evidence": "e", "sources": ["S1"]}
    reviewed, _ = _apply({"ideas": _verdicts(i1={"verdict": "revise", "reason": "r", "revised": meta})})
    assert 'after review, idea 2 "Idea 1": describes the text instead of stating the idea' in reviewed.warnings


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
    assert write.SPOILER_RULE in text and write.SPOILER_RULE in write._fiction_instructions(FICTION)
    assert "3-6 ideas" in text and "key claims" not in text and "key_claims_for_review" not in text


def test_the_review_and_the_write_call_share_the_rule_to_report_the_book_not_its_critics():
    assert write.FAITHFUL_RULE in review._instructions(_book(), [])
    assert write.FAITHFUL_RULE in write._nonfiction_instructions(KNOWN)
    assert write.FAITHFUL_RULE in write._fiction_instructions(FICTION)


def test_titles_match_whatever_quotes_dashes_and_case_the_model_retypes():
    book = _book().model_copy(update={"ideas": [
        review.Idea(title="Kahneman’s “System 1” — fast", summary="s", evidence="e", sources=["S1"]),
        *_book().ideas[1:],
    ]})  # fmt: skip
    verdicts = [{"title": 'kahneman\'s "system 1" - fast', "verdict": "drop", "reason": "r"}, *_verdicts()[1:]]
    _, changes = _apply({"ideas": verdicts, "key_claims_for_review": _CLAIMS}, book)
    assert changes[0] == 'dropped "Kahneman’s “System 1” — fast": r'


def test_ideas_sharing_a_title_take_their_verdicts_in_order():
    ideas = _book().ideas
    book = _book().model_copy(update={"ideas": [ideas[0], ideas[1].model_copy(update={"title": "Idea 0"}), *ideas[2:]]})
    verdicts = [{"title": "Idea 0", "verdict": "keep"}, {"title": "Idea 0", "verdict": "drop", "reason": "r"}, *_verdicts()[2:]]
    reviewed, _ = _apply({"ideas": verdicts, "key_claims_for_review": _CLAIMS}, book)
    assert [i.summary for i in reviewed.ideas] == ["topic0word", "topic2word", "topic3word", "topic4word", "topic5word"]
    with pytest.raises(ValidationError, match="'Idea 0' has more than one verdict"):
        _validate({"ideas": [*verdicts, {"title": "Idea 0", "verdict": "keep"}]}, book)


def test_a_new_idea_repeating_one_the_notes_keep_is_left_out_not_sent_back():
    repeat = {"title": "Idea 2 again", "summary": "topic2word", "evidence": "e", "sources": ["S1"]}
    fresh = {"title": "Idea 9", "summary": "topic9word", "evidence": "e", "sources": ["S1"]}
    reviewed, changes = _apply({"ideas": _verdicts(), "new_ideas": [repeat, fresh, {**fresh, "title": "Idea 9 again"}]})
    assert [i.title for i in reviewed.ideas][-1] == "Idea 9"
    assert changes == [
        'added "Idea 9"',
        'left out new idea "Idea 2 again": it repeats an idea the notes have',
        'left out new idea "Idea 9 again": it repeats an idea the notes have',
    ]
    # A dropped idea's replacement may cover the same ground.
    verdicts = _verdicts(i2={"verdict": "drop", "reason": "r"})
    reviewed, _ = _apply({"ideas": verdicts, "new_ideas": [repeat], "key_claims_for_review": _CLAIMS})
    assert reviewed.ideas[-1].title == "Idea 2 again"
