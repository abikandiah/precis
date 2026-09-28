from precis.checks import check_notes
from precis.schema import Book


def _idea(title: str, summary: str = "s", evidence: str = "e", sources: list[str] | None = None) -> dict:
    return {"title": title, "summary": summary, "evidence": evidence, "sources": sources or []}


def _book(ideas: list[dict], claims: list[dict] | None = None) -> Book:
    fillers = ["Flywheel momentum", "Stockdale paradox", "Technology accelerators", "Culture discipline", "Council"]
    ideas = ideas + [_idea(title, title.lower()) for title in fillers[: 5 - len(ideas)]]
    return Book(
        title="Good to Great", author="Jim Collins", isbn="1", kind="non-fiction", one_line_takeaway="t",
        synopsis="s", tags=["business", "economics"], ideas=ideas,
        key_claims_for_review=claims or [{"prompt": f"Q{n}?", "answer": "A."} for n in range(5)],
    )  # fmt: skip


def test_a_clean_book_has_no_findings():
    assert check_notes(_book([_idea("Delayed gratification", "waiting for a larger reward")])) == []


def test_meta_descriptions_are_flagged():
    book = _book(
        [_idea("Hedgehog", summary="The author discusses focus.")],
        claims=[{"prompt": "Q?", "answer": "This book explores focus."}] * 5,
    )
    issues = check_notes(book)
    assert 'idea 1 "Hedgehog": describes the text instead of stating the idea' in issues
    assert "key claim 1: the answer describes the text instead of stating the idea" in issues


def test_near_duplicate_ideas_are_flagged():
    book = _book(
        [
            _idea("Loss aversion", "Losses loom larger than equivalent gains"),
            _idea("Loss aversion again", "Equivalent gains loom smaller than losses"),
        ]
    )
    assert any(i.startswith('ideas 1 and 2 ("Loss aversion"') for i in check_notes(book))


def test_evidence_and_the_authors_surname_as_subject_are_checked_too():
    book = _book(
        [
            _idea("Stages", summary="Collins traces growth from good to great.", evidence="The book examines ten firms."),
            _idea(
                "Hedgehog",
                summary="Collins argues that great firms focus on one thing.",
                evidence="Collins describes how Walgreens bet everything on convenient drugstores.",
            ),
        ]
    )
    issues = check_notes(book)
    assert 'idea 1 "Stages": describes the text instead of stating the idea' in issues
    assert 'idea 1 "Stages": its evidence describes the text instead of giving the book\'s example' in issues
    assert not any('"Hedgehog"' in i for i in issues)
