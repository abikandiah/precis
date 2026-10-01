from precis import checks
from precis.checks import check_notes, quotes
from precis.research import Part, Research, Source
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
                evidence="The author describes how Walgreens bet everything on convenient drugstores.",
            ),
        ]
    )
    issues = check_notes(book)
    assert 'idea 1 "Stages": describes the text instead of stating the idea' in issues
    assert 'idea 1 "Stages": its evidence describes the text instead of giving the book\'s example' in issues
    assert not any('"Hedgehog"' in i for i in issues)


def test_evidence_describing_the_text_after_its_example_is_fine():
    book = _book(
        [
            _idea(
                "Critics",
                evidence="McCandless died in the bus on the Stampede Trail. The book includes critical letters "
                "Krakauer received from Alaskans who called him reckless.",
            ),
        ]
    )
    assert not any("evidence" in i for i in check_notes(book))


# --- quotes and coverage, against the research ---------------------------------------

BOOK_TEXT = (
    "In such moments I have had the fantasy of a prisoner in a dun-\ngeon, tapping out day after day a Morse "
    "code message, ''Does anybody hear me? Is anybody there?\" Where love stops, power begins."
)


def _research(*, book_text: bool = True, whole: bool = True) -> Research:
    parts = tuple(Part("book" if book_text else "about", "", "n", ()) for _ in range(2))
    # A whole book is long; a preview of a few chapters isn't.
    full_text = BOOK_TEXT + "\nfiller" * (checks.WHOLE_BOOK_CHARS // 7 if whole else 0)
    return Research(
        sources=[
            Source(id="S1", title="t", url="u1", text="excerpt", full_text=full_text, parts=parts, cut=True),
            Source(id="S2", title="t", url="u2", text='A summary quoting "What the nation does is done also by each individual."'),
        ],
        warnings=[],
    )


def test_quotes_found_in_the_book_text_pass_despite_ocr_and_quote_style():
    book = _book([_idea("Being heard", evidence="He pictures a prisoner in a dungeon tapping out “Does anybody hear me? Is anybody there?”")])
    assert check_notes(book, _research()) == []


def test_a_quote_in_no_source_is_flagged():
    book = _book([_idea("Being heard", summary='Rogers says "the curious paradox is that when I accept myself"')])
    issues = check_notes(book, _research())
    assert any(i.startswith('idea 1 "Being heard": the quote "the curious paradox') and "isn't in any source" in i for i in issues)


def test_a_quote_only_in_pages_about_the_book_is_flagged_when_the_book_text_is_there():
    book = _book([_idea("The nation", summary='Jung: "What the nation does is done also by each individual."')])
    issues = check_notes(book, _research())
    assert any("is in S2 but not in the book's own text (S1)" in i for i in issues)
    # Without the book's own text in the research, a summary's quote is all there is.
    assert check_notes(book, _research(book_text=False)) == []
    # Nor when the book's text is only a preview: the quote may be from a later chapter.
    assert check_notes(book, _research(whole=False)) == []


def test_short_quoted_terms_are_not_quotes():
    book = _book([_idea("Realities", summary='He doubts the "real world" exists.')])
    assert check_notes(book, _research()) == []


def test_quoted_titles_are_not_quotes():
    book = _book([_idea("Willpower", evidence='The study "Ego Depletion: Is the Active Self a Limited Resource?" found it.')])
    assert check_notes(book, _research()) == []


def test_a_stray_straight_quote_does_not_pair_with_a_real_one():
    assert quotes('He wrote a 5" scroll; the idea is that "we must act now and here"') == ["we must act now and here"]
    # A summary's stray mark doesn't pair with a quote in the evidence.
    book = _book([_idea("Being heard", summary='A 5" prisoner taps a code', evidence='"Does anybody hear me? Is anybody there?"')])
    assert check_notes(book, _research()) == []


def test_single_quotes_are_quotes_too_but_apostrophes_are_not():
    from precis.checks import quotes

    text = "Jung: 'what the nation does is done also by each individual.' and ‘a million zeros joined together’"
    assert quotes(text) == ["what the nation does is done also by each individual.", "a million zeros joined together"]
    # Apostrophes inside words neither open nor close a quote.
    assert quotes("man's evil isn't the others' fault, says the author's book") == []
    assert quotes("'evil, without man's ever having chosen it, is lodged'") == [
        "evil, without man's ever having chosen it, is lodged"
    ]


def test_a_single_quoted_quote_in_no_source_is_flagged():
    book = _book([_idea("Change", evidence="Jung: 'change occurs in individuals over centuries through spirit'")])
    assert any("isn't in any source" in i for i in check_notes(book, _research()))
