import pytest

from precis.checks import check_notes, specifics
from precis.research import Research, Source
from precis.schema import Book

_RESEARCH = Research(
    sources=[
        Source(id="S1", title="t", url="u", text="Walter Mischel's marshmallow study at Stanford in 1972."),
        Source(id="S2", title="t", url="u", text="Collins screened 1,435 companies."),
    ],
    warnings=[],
)


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("In the study, Walter Mischel tested 600 children. The results held.", ["600", "Walter", "Mischel"]),
        # A name opening the evidence, or after "(", "Dr." or an initial.
        ("Henderson's 1994 study of 300 nurses", ["1994", "300", "Henderson"]),
        ("(Henderson, 1994) and Dr. Henderson and J. K. Rowling", ["1994", "Henderson", "Rowling"]),
        ("narrated by Gabriel García Márquez", ["Gabriel", "García", "Márquez"]),
        ("a Harvard-trained psychologist, a Nobel-winning idea", ["Harvard", "Nobel"]),
        ("a $2.5m return in the 1970s: the 20th trial, a 10x gain, 1,435 firms", ["2.5m", "1970s", "20th", "10x", "1,435"]),
    ],
)
def test_specifics_are_numbers_with_their_units_and_capitalized_names(text, expected):
    assert specifics(text) == expected


def test_commas_are_handled_the_same_way_in_evidence_sources_and_the_title():
    research = Research(
        sources=[Source(id="S1", title="t", url="u", text="The work of Kahneman,Tversky and Thaler.")], warnings=[]
    )
    book = _book([_idea("Heuristics", evidence="as shown by Kahneman and Tversky", sources=["S1"])])
    assert check_notes(book, research) == []
    leagues = _book([_idea("Voyage", evidence="travels 20,000 leagues", sources=["S1"])]).model_copy(
        update={"title": "20,000 Leagues Under the Sea"}
    )
    assert check_notes(leagues, research) == []


def test_names_are_matched_word_by_word_so_phrasing_and_possessives_dont_matter():
    research = Research(
        sources=[Source(id="S1", title="t", url="u", text="Philip Zimbardo ran the Stanford prison experiment.")],
        warnings=[],
    )
    book = _book([_idea("Situations", evidence="as in Zimbardo's Stanford Prison Experiment", sources=["S1"])])
    assert check_notes(book, research) == []


def test_a_clean_book_has_no_findings():
    book = _book([_idea("Delayed gratification", evidence="Walter Mischel's study in 1972", sources=["S1"])])
    assert check_notes(book, _RESEARCH) == []


def test_meta_descriptions_are_flagged():
    book = _book(
        [_idea("Hedgehog", summary="The author discusses focus.")],
        claims=[{"prompt": "Q?", "answer": "This book explores focus."}] * 5,
    )
    issues = check_notes(book, _RESEARCH)
    assert 'idea 1 "Hedgehog": describes the text instead of stating the idea' in issues
    assert "key claim 1: the answer describes the text instead of stating the idea" in issues


def test_near_duplicate_ideas_are_flagged():
    book = _book(
        [
            _idea("Loss aversion", "Losses loom larger than equivalent gains"),
            _idea("Loss aversion again", "Equivalent gains loom smaller than losses"),
        ]
    )
    assert any(i.startswith('ideas 1 and 2 ("Loss aversion"') for i in check_notes(book, _RESEARCH))


def test_evidence_specifics_missing_from_the_cited_sources_are_flagged():
    book = _book(
        [
            # 1,435 is in S2 but this idea cites S1; Harvard is nowhere.
            _idea("Screening", evidence="Collins screened 1,435 companies at Harvard", sources=["S1"]),
            # Uncited: checked against all the research, where 1,435 is.
            _idea("Method", evidence="The team screened 1,435 companies"),
            # The book's own author isn't a specific to check.
            _idea("Level 5", evidence="as Jim Collins puts it", sources=["S2"]),
        ]
    )
    issues = check_notes(book, _RESEARCH)
    assert issues == [
        "idea 1 \"Screening\": its evidence names ['1,435', 'Harvard'], which aren't in its cited sources (S1)"
    ]


def test_without_research_there_is_nothing_to_check_specifics_against():
    book = _book([_idea("Screening", evidence="Collins screened 9,999 companies at Harvard")])
    assert check_notes(book, Research(sources=[], warnings=[])) == []
