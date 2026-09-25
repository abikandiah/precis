import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from precis.known_file import (
    PLACEHOLDER,
    create_known_file,
    ensure_ready,
    parse_table_of_contents,
    preflight_check,
)
from precis.schema import KnownFile, KnownPart


def _known_file(**overrides) -> KnownFile:
    defaults = {
        "isbn": "123",
        "title": "A Book",
        "author": "An Author",
        "kind": "non-fiction",
        "narrative": False,
        "chapters": ["Ch 1"],
    }
    defaults.update(overrides)
    return KnownFile(**defaults)


def test_preflight_requires_isbn():
    problems = preflight_check(_known_file(isbn=""))
    assert any("isbn" in p for p in problems)


def test_preflight_requires_chapters_for_full_nonfiction_path():
    problems = preflight_check(_known_file(chapters=[]))
    assert any("chapters" in p for p in problems)


def test_preflight_allows_empty_chapters_for_narrative_nonfiction():
    problems = preflight_check(_known_file(narrative=True, chapters=[]))
    assert problems == []


def test_preflight_allows_empty_chapters_for_fiction():
    problems = preflight_check(_known_file(kind="fiction", narrative=False, chapters=[]))
    assert problems == []


def test_preflight_rejects_narrative_true_with_fiction():
    problems = preflight_check(_known_file(kind="fiction", narrative=True, chapters=[]))
    assert any("narrative" in p for p in problems)


def test_preflight_passes_for_ready_nonfiction_known_file():
    assert preflight_check(_known_file()) == []


def test_preflight_rejects_parts_for_fiction():
    problems = preflight_check(
        _known_file(kind="fiction", narrative=False, chapters=[], parts=[KnownPart(title="Part One")])
    )
    assert any("parts" in p for p in problems)


def test_preflight_rejects_chapters_on_narrative_nonfiction_parts():
    problems = preflight_check(
        _known_file(narrative=True, chapters=[], parts=[KnownPart(title="Part One", chapters=[1])])
    )
    assert any("chapters" in p for p in problems)


def test_preflight_allows_title_only_parts_on_narrative_nonfiction():
    problems = preflight_check(_known_file(narrative=True, chapters=[], parts=[KnownPart(title="Part One")]))
    assert problems == []


def test_preflight_rejects_empty_chapters_list_on_narrative_nonfiction_parts():
    """chapters=[] is falsy but not None — the check must catch it
    the same as a non-empty list would, since narrative books have no known
    chapter list to bind against either way.
    """
    problems = preflight_check(
        _known_file(narrative=True, chapters=[], parts=[KnownPart(title="Part One", chapters=[])])
    )
    assert any("chapters" in p for p in problems)


def test_preflight_rejects_duplicate_part_titles():
    problems = preflight_check(
        _known_file(
            chapters=["Ch 1", "Ch 2", "Ch 3", "Ch 4"],
            parts=[
                KnownPart(title="Part One", chapters=[1, 2]),
                KnownPart(title="Part One", chapters=[3, 4]),
            ],
        )
    )
    assert any("duplicate" in p for p in problems)


def test_preflight_rejects_out_of_range_chapters_on_full_nonfiction_parts():
    problems = preflight_check(
        _known_file(chapters=["Ch 1"], parts=[KnownPart(title="Part One", chapters=[1, 2])])
    )
    assert any("chapter number 2" in p for p in problems)


def test_preflight_allows_valid_chapters_on_full_nonfiction_parts():
    problems = preflight_check(
        _known_file(chapters=["Ch 1", "Ch 2"], parts=[KnownPart(title="Part One", chapters=[1, 2])])
    )
    assert problems == []


def test_ensure_ready_raises_when_not_ready():
    try:
        ensure_ready(_known_file(chapters=[]))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "chapters" in str(exc)


def test_ensure_ready_does_not_raise_when_ready():
    ensure_ready(_known_file())


def _mock_open_library_response(body: dict):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_create_known_file_fills_fields_from_lookup():
    search_body = {"docs": [{"title": "The Book", "author_name": ["Jane Author"]}]}
    edition_body = {"publish_date": "March 2003", "number_of_pages": 320}
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[_mock_open_library_response(search_body), _mock_open_library_response(edition_body)],
    ):
        known_file, _ = create_known_file("9780000000000", kind="non-fiction")

    assert known_file.title == "The Book"
    assert known_file.author == "Jane Author"
    assert known_file.year == 2003
    assert known_file.page_count == 320
    assert known_file.chapters == []


def test_create_known_file_uses_edition_specific_year_and_page_count_not_work_aggregate():
    # search.json's first_publish_year/number_of_pages_median describe the
    # work across all editions, not the specific ISBN queried — year/page
    # count must come from the isbn/{isbn}.json edition record instead.
    search_body = {
        "docs": [
            {
                "title": "The Book",
                "author_name": ["Jane Author"],
                "first_publish_year": 1997,
                "number_of_pages_median": 528,
            }
        ]
    }
    edition_body = {"publish_date": "1999", "number_of_pages": 494}
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[_mock_open_library_response(search_body), _mock_open_library_response(edition_body)],
    ):
        known_file, _ = create_known_file("9780393317558", kind="non-fiction")

    assert known_file.year == 1999
    assert known_file.page_count == 494


def test_create_known_file_uses_placeholders_when_lookup_misses():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[_mock_open_library_response({"docs": []}), _mock_open_library_response({})],
    ):
        known_file, _ = create_known_file("0000000000000", kind="fiction")

    assert known_file.title == PLACEHOLDER
    assert known_file.author == PLACEHOLDER
    assert known_file.year is None
    assert known_file.page_count is None


def test_create_known_file_uses_placeholders_when_lookup_returns_non_dict_body():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[_mock_open_library_response(None), _mock_open_library_response([1, 2, 3])],
    ):
        known_file, _ = create_known_file("000", kind="fiction")

    assert known_file.title == PLACEHOLDER
    assert known_file.year is None


def test_create_known_file_uses_placeholders_on_network_error():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no network"),
    ):
        known_file, _ = create_known_file("111", kind="fiction")

    assert known_file.title == PLACEHOLDER


@pytest.mark.parametrize("field", ["title", "author"])
@pytest.mark.parametrize("value", [None, "", PLACEHOLDER])
def test_preflight_requires_title_and_author(field, value):
    problems = preflight_check(_known_file(**{field: value}))
    assert any(p.startswith(f"{field} is required") for p in problems)


def _toc(*entries: tuple[str | None, str]) -> list[dict]:
    return [{"level": 0, "label": label, "title": title} for label, title in entries]


def test_parse_toc_groups_chapters_under_parts_and_drops_back_matter():
    # Shape of Quiet's real Open Library record: parts as labels, "Title :
    # subtitle" cataloguing style, trailing full stops, front/back matter.
    entries = _toc(
        (None, "Author's note"),
        (None, "Introduction : The north and south of temperament"),
        ("Part one", "The extrovert ideal."),
        (None, "The rise of the mighty likeable fellow"),
        (None, "The myth of charismatic leadership"),
        ("Part two", "Your biology, your self?"),
        (None, "Is temperament destiny?"),
        (None, "Conclusion : Wonderland"),
        (None, "A note on the dedication"),
        (None, "Acknowledgments"),
        (None, "Notes"),
        (None, "Index"),
    )
    chapters, parts = parse_table_of_contents(entries)
    assert chapters == [
        "Introduction: The north and south of temperament",
        "The rise of the mighty likeable fellow",
        "The myth of charismatic leadership",
        "Is temperament destiny?",
        "Conclusion: Wonderland",
    ]
    assert [(p.title, p.chapters) for p in parts] == [
        ("Part one: The extrovert ideal", [2, 3]),
        ("Part two: Your biology, your self?", [4]),
    ]


def test_parse_toc_reads_parts_from_titles_and_strips_chapter_numbers():
    entries = _toc(
        (None, "PART I. Two Systems"),
        ("1", "The Characters of the Story"),
        (None, "Chapter 2: Attention and Effort"),
        (None, "Part of the Problem"),
    )
    chapters, parts = parse_table_of_contents(entries)
    assert chapters == [
        "The Characters of the Story",
        "Attention and Effort",
        "Part of the Problem",
    ]
    assert [(p.title, p.chapters) for p in parts] == [("Part I: Two Systems", [1, 2, 3])]


@pytest.mark.parametrize("lumped", ["A ; B ; C", "A -- B -- C"])
def test_parse_toc_rejects_lumped_entries(lumped):
    assert parse_table_of_contents(_toc((None, "Prologue"), (None, lumped))) is None


@pytest.mark.parametrize("entries", [None, [], _toc((None, "Only one chapter"))])
def test_parse_toc_rejects_missing_or_trivial_lists(entries):
    assert parse_table_of_contents(entries) is None


def test_create_known_file_prefills_chapters_and_parts_from_this_edition():
    search_body = {"docs": [{"title": "The Book", "author_name": ["Jane Author"]}]}
    edition_body = {"table_of_contents": _toc(("Part one", "Start"), (None, "Ch A"), (None, "Ch B"))}
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response(search_body),
            _mock_open_library_response(edition_body),
        ],
    ):
        known_file, notes = create_known_file("9780000000000", kind="non-fiction")

    assert known_file.chapters == ["Ch A", "Ch B"]
    assert [(p.title, p.chapters) for p in known_file.parts] == [("Part one: Start", [1, 2])]
    assert notes == [
        (
            "2 chapters and 1 part pre-filled from Open Library's table of contents for this edition "
            "— check them against your copy."
        )
    ]


def test_create_known_file_falls_back_to_another_edition_of_the_work():
    search_body = {"docs": [{"title": "The Book", "author_name": ["Jane Author"]}]}
    edition_body = {"works": [{"key": "/works/OL1W"}]}
    editions_body = {
        "entries": [
            {"key": "/books/OL2M", "table_of_contents": _toc((None, "A ; B"))},
            {
                "key": "/books/OL3M",
                "isbn_13": ["9781111111111"],
                "publish_date": "2011",
                "table_of_contents": _toc((None, "Ch A"), (None, "Ch B")),
            },
        ]
    }
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response(search_body),
            _mock_open_library_response(edition_body),
            _mock_open_library_response(editions_body),
        ],
    ):
        known_file, notes = create_known_file("9780000000000", kind="non-fiction")

    assert known_file.chapters == ["Ch A", "Ch B"]
    assert "another edition (ISBN 9781111111111 2011)" in notes[0]


def test_create_known_file_keeps_only_part_titles_for_narrative_nonfiction():
    edition_body = {"table_of_contents": _toc(("Part one", "Start"), (None, "Ch A"), (None, "Ch B"))}
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response({"docs": []}),
            _mock_open_library_response(edition_body),
        ],
    ):
        known_file, _ = create_known_file("9780000000000", kind="non-fiction", narrative=True)

    assert known_file.chapters == []
    assert [(p.title, p.chapters) for p in known_file.parts] == [("Part one: Start", None)]


def test_create_known_file_notes_when_no_toc_is_found():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response({"docs": []}),
            _mock_open_library_response({}),
        ],
    ):
        _, notes = create_known_file("9780000000000", kind="non-fiction")

    assert notes == ["Open Library has no usable table of contents for this book — fill in chapters by hand."]


def test_fetch_retries_once_on_a_dropped_connection():
    body = {"docs": [{"title": "The Book", "author_name": ["Jane Author"]}]}
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            urllib.error.URLError(ConnectionResetError()),
            _mock_open_library_response(body),
            _mock_open_library_response({}),
        ],
    ):
        known_file, _ = create_known_file("9780000000000", kind="fiction")

    assert known_file.title == "The Book"


def test_parse_toc_keeps_a_year_that_starts_a_title():
    chapters, _ = parse_table_of_contents(_toc((None, "1914: The Guns of August"), (None, "12. Aftermath")))
    assert chapters == ["1914: The Guns of August", "Aftermath"]


def test_parse_toc_treats_a_titleless_part_label_as_a_part_boundary():
    entries = _toc(("Part One", "Start"), (None, "A"), (None, "B"), ("Part Two", ""), (None, "C"))
    _, parts = parse_table_of_contents(entries)
    assert [(p.title, p.chapters) for p in parts] == [("Part One: Start", [1, 2]), ("Part Two", [3])]


def test_parse_toc_numbered_conclusion_stands_outside_the_last_part():
    entries = _toc(("Part one", "Start"), (None, "11. A"), (None, "Chapter 12: Conclusion"))
    chapters, parts = parse_table_of_contents(entries)
    assert chapters == ["A", "Conclusion"]
    assert [(p.title, p.chapters) for p in parts] == [("Part one: Start", [1])]


def test_create_known_file_skips_other_editions_in_another_language():
    edition_body = {"works": [{"key": "/works/OL1W"}], "languages": [{"key": "/languages/eng"}]}
    editions_body = {
        "entries": [
            {"key": "/books/ES", "languages": [{"key": "/languages/spa"}], "table_of_contents": _toc(
                (None, "Capítulo uno"), (None, "Capítulo dos"))},
            {"key": "/books/EN", "languages": [{"key": "/languages/eng"}], "table_of_contents": _toc(
                (None, "The first"), (None, "The second"))},
        ]
    }
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response({"docs": []}),
            _mock_open_library_response(edition_body),
            _mock_open_library_response(editions_body),
        ],
    ):
        known_file, _ = create_known_file("9780000000000", kind="non-fiction")

    assert known_file.chapters == ["The first", "The second"]


@pytest.mark.parametrize("error", [urllib.error.URLError("dns failure"), TimeoutError()])
def test_fetch_does_not_retry_errors_a_retry_wont_fix(error):
    with patch("precis.known_file.urllib.request.urlopen", side_effect=error) as urlopen:
        create_known_file("9780000000000", kind="fiction")
    assert urlopen.call_count == 2  # search + edition, one attempt each


def test_parse_toc_rejects_a_bare_chapter_number_rather_than_renumber():
    entries = _toc(
        ("Part One", "Start"),
        (None, "Chapter 1: A"),
        (None, "Chapter 2"),
        (None, "Chapter 3: C"),
    )
    assert parse_table_of_contents(entries) is None


def test_parse_toc_rejects_duplicate_part_titles():
    entries = _toc((None, "Part One"), (None, "A"), (None, "Part One"), (None, "B"))
    assert parse_table_of_contents(entries) is None


def test_parse_toc_keeps_an_abbreviations_final_period():
    chapters, _ = parse_table_of_contents(_toc((None, "The U.S."), (None, "Washington, D.C."), (None, "The end.")))
    assert chapters == ["The U.S.", "Washington, D.C.", "The end"]


def test_parse_toc_drops_ebook_front_matter():
    entries = _toc(
        (None, "Cover"),
        (None, "Title Page"),
        (None, "Praise for A Book"),
        (None, "Epigraph"),
        (None, "Maps"),
        (None, "Real One"),
        (None, "Real Two"),
        (None, "Notes & Sources"),
    )
    chapters, _ = parse_table_of_contents(entries)
    assert chapters == ["Real One", "Real Two"]


def test_parse_toc_word_made_of_numeral_letters_is_not_a_part_number():
    entries = _toc((None, "Part Civil War, Part Revolution"), (None, "Part XIV: Late"), (None, "After"))
    chapters, parts = parse_table_of_contents(entries)
    assert chapters == ["Part Civil War, Part Revolution", "After"]
    assert [(p.title, p.chapters) for p in parts] == [("Part XIV: Late", [2])]


def test_fetch_survives_a_truncated_response():
    import http.client

    truncated = MagicMock()
    truncated.read.side_effect = http.client.IncompleteRead(b"{")
    truncated.__enter__.return_value = truncated
    with patch("precis.known_file.urllib.request.urlopen", return_value=truncated):
        known_file, _ = create_known_file("9780000000000", kind="fiction")
    assert known_file.title == PLACEHOLDER


@pytest.mark.parametrize("title", ["Part Invasion of the Body", "Part Victory", "Part Illusion"])
def test_parse_toc_word_starting_with_numeral_letters_is_not_a_part(title):
    chapters, parts = parse_table_of_contents(_toc((None, title), (None, "A")))
    assert chapters == [title, "A"]
    assert parts == []


def test_parse_toc_strips_numbers_before_filtering_and_keeps_non_numeric_chapter_words():
    entries = _toc(
        (None, "Chapter and Verse: The Law"),
        (None, "Chapter Two: Real"),
        (None, "Chapter 12: Acknowledgments"),
        (None, "14. Notes"),
    )
    chapters, _ = parse_table_of_contents(entries)
    assert chapters == ["Chapter and Verse: The Law", "Real"]


def test_parse_toc_uses_levels_to_drop_subsections_and_reads_book_divisions():
    entries = [
        {"level": 0, "title": "Introduction"},
        {"level": 0, "title": "Book One: Beginnings"},
        {"level": 1, "title": "The First"},
        {"level": 2, "title": "A subsection"},
        {"level": 1, "title": "The Second"},
        {"level": 0, "label": "Part two", "title": "Middles"},
        {"level": 1, "title": "The Third"},
    ]
    chapters, parts = parse_table_of_contents(entries)
    assert chapters == ["Introduction", "The First", "The Second", "The Third"]
    assert [(p.title, p.chapters) for p in parts] == [("Book One: Beginnings", [2, 3]), ("Part two: Middles", [4])]


def test_create_known_file_without_a_language_only_borrows_from_editions_without_one():
    edition_body = {"works": [{"key": "/works/OL1W"}]}
    editions_body = {
        "entries": [
            {"key": "/books/DE", "languages": [{"key": "/languages/ger"}], "table_of_contents": _toc(
                (None, "Erstes"), (None, "Zweites"))},
            {"key": "/books/XX", "table_of_contents": _toc((None, "The first"), (None, "The second"))},
        ]
    }
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response({"docs": []}),
            _mock_open_library_response(edition_body),
            _mock_open_library_response(editions_body),
        ],
    ):
        known_file, _ = create_known_file("9780000000000", kind="non-fiction")

    assert known_file.chapters == ["The first", "The second"]


def test_parse_toc_keeps_chapters_that_start_with_a_front_matter_word():
    entries = _toc(
        (None, "Copyright Wars"),
        (None, "Dedication and Grit"),
        (None, "Copyright"),
        (None, "Acknowledgements"),
        (None, "Preface to the Second Edition"),
    )
    chapters, _ = parse_table_of_contents(entries)
    assert chapters == ["Copyright Wars", "Dedication and Grit"]


def test_parse_toc_strips_hyphenated_chapter_number_words():
    chapters, _ = parse_table_of_contents(_toc((None, "Chapter Twenty-One: The End"), (None, "Chapter Five Alive")))
    assert chapters == ["The End", "Alive"]


def test_create_known_file_tolerates_junk_types_in_the_fallback_edition():
    edition_body = {"works": [{"key": "/works/OL1W"}]}
    editions_body = {
        "entries": [
            None,
            {"key": "/books/XX", "isbn_13": "9781111111111", "publish_date": 2011,
             "table_of_contents": _toc((None, "The first"), (None, "The second"))},
        ]
    }  # fmt: skip
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[
            _mock_open_library_response({"docs": []}),
            _mock_open_library_response(edition_body),
            _mock_open_library_response(editions_body),
        ],
    ):
        known_file, notes = create_known_file("9780000000000", kind="non-fiction")

    assert known_file.chapters == ["The first", "The second"]
    assert "another edition (/books/XX)" in notes[0]
