import json
import urllib.error
from unittest.mock import MagicMock, patch

from precis.known_file import (
    PLACEHOLDER,
    create_known_file,
    ensure_ready,
    preflight_check,
)
from precis.schema import KnownFile, KnownPart


def _known_file(**overrides) -> KnownFile:
    defaults = {"isbn": "123", "kind": "non-fiction", "narrative": False, "chapters": ["Ch 1"]}
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


def test_preflight_rejects_chapter_numbers_on_narrative_nonfiction_parts():
    problems = preflight_check(
        _known_file(narrative=True, chapters=[], parts=[KnownPart(title="Part One", chapter_numbers=[1])])
    )
    assert any("chapter_numbers" in p for p in problems)


def test_preflight_allows_title_only_parts_on_narrative_nonfiction():
    problems = preflight_check(_known_file(narrative=True, chapters=[], parts=[KnownPart(title="Part One")]))
    assert problems == []


def test_preflight_rejects_empty_chapter_numbers_list_on_narrative_nonfiction_parts():
    """chapter_numbers=[] is falsy but not None — the check must catch it
    the same as a non-empty list would, since narrative books have no known
    chapter list to bind against either way.
    """
    problems = preflight_check(
        _known_file(narrative=True, chapters=[], parts=[KnownPart(title="Part One", chapter_numbers=[])])
    )
    assert any("chapter_numbers" in p for p in problems)


def test_preflight_rejects_duplicate_part_titles():
    problems = preflight_check(
        _known_file(
            chapters=["Ch 1", "Ch 2", "Ch 3", "Ch 4"],
            parts=[
                KnownPart(title="Part One", chapter_numbers=[1, 2]),
                KnownPart(title="Part One", chapter_numbers=[3, 4]),
            ],
        )
    )
    assert any("duplicate" in p for p in problems)


def test_preflight_rejects_out_of_range_chapter_numbers_on_full_nonfiction_parts():
    problems = preflight_check(
        _known_file(chapters=["Ch 1"], parts=[KnownPart(title="Part One", chapter_numbers=[1, 2])])
    )
    assert any("chapter number 2" in p for p in problems)


def test_preflight_allows_valid_chapter_numbers_on_full_nonfiction_parts():
    problems = preflight_check(
        _known_file(chapters=["Ch 1", "Ch 2"], parts=[KnownPart(title="Part One", chapter_numbers=[1, 2])])
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
        known_file = create_known_file("9780000000000", kind="non-fiction")

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
        known_file = create_known_file("9780393317558", kind="non-fiction")

    assert known_file.year == 1999
    assert known_file.page_count == 494


def test_create_known_file_uses_placeholders_when_lookup_misses():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[_mock_open_library_response({"docs": []}), _mock_open_library_response({})],
    ):
        known_file = create_known_file("0000000000000", kind="fiction")

    assert known_file.title == PLACEHOLDER
    assert known_file.author == PLACEHOLDER
    assert known_file.year is None
    assert known_file.page_count is None


def test_create_known_file_uses_placeholders_when_lookup_returns_non_dict_body():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=[_mock_open_library_response(None), _mock_open_library_response([1, 2, 3])],
    ):
        known_file = create_known_file("000", kind="fiction")

    assert known_file.title == PLACEHOLDER
    assert known_file.year is None


def test_create_known_file_uses_placeholders_on_network_error():
    with patch(
        "precis.known_file.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no network"),
    ):
        known_file = create_known_file("111", kind="fiction")

    assert known_file.title == PLACEHOLDER
