import http.client
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from precis.known_file import (
    PLACEHOLDER,
    create_known_file,
    preflight_check,
    slugify_title,
)
from precis.schema import KnownFile

_URLOPEN = "precis.known_file.urllib.request.urlopen"


def _known_file(**overrides) -> KnownFile:
    return KnownFile(**{"isbn": "123", "title": "A Book", "author": "An Author", "kind": "non-fiction", **overrides})


def _response(body: object) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(body).encode("utf-8")
    response.__enter__.return_value = response
    return response


# --- preflight -----------------------------------------------------------------


def test_preflight_passes_a_ready_known_file():
    assert preflight_check(_known_file()) == []


def test_preflight_requires_isbn():
    assert preflight_check(_known_file(isbn="")) == ["isbn is required"]


@pytest.mark.parametrize("field", ["title", "author"])
@pytest.mark.parametrize("value", [None, "", PLACEHOLDER])
def test_preflight_requires_title_and_author(field, value):
    assert preflight_check(_known_file(**{field: value})) == [
        f"{field} is required (still missing or a placeholder)"
    ]


def test_slugify_title():
    assert slugify_title("Guns, Germs, and Steel") == "guns-germs-and-steel"
    assert slugify_title(None) == ""


# --- create_known_file ------------------------------------------------------------


def test_create_known_file_fills_fields_from_the_lookup():
    search_body = {"docs": [{"title": "The Book", "author_name": ["Jane Author", "Joe Author"]}]}
    edition_body = {"publish_date": "March 2003", "number_of_pages": 320}
    with patch(_URLOPEN, side_effect=[_response(search_body), _response(edition_body)]):
        known_file, notes = create_known_file("9780000000000", kind="non-fiction")
    assert (known_file.title, known_file.author, known_file.year, known_file.page_count) == (
        "The Book", "Jane Author, Joe Author", 2003, 320
    )
    assert known_file.kind == "non-fiction"
    assert notes == []


def test_year_and_page_count_come_from_the_edition_not_the_work():
    # search.json's first_publish_year/number_of_pages_median describe every
    # edition of the work, not the one the ISBN names.
    search_body = {
        "docs": [{"title": "T", "author_name": ["A"], "first_publish_year": 1997, "number_of_pages_median": 528}]
    }
    with patch(_URLOPEN, side_effect=[_response(search_body), _response({"publish_date": "1999", "number_of_pages": 494})]):
        known_file, _ = create_known_file("9780393317558", kind="non-fiction")
    assert (known_file.year, known_file.page_count) == (1999, 494)


@pytest.mark.parametrize(
    "side_effect",
    [
        [_response({"docs": []}), _response({})],  # a miss
        [_response(None), _response([1, 2, 3])],  # bodies that aren't objects
        urllib.error.URLError("no network"),
    ],
)
def test_a_failed_lookup_leaves_placeholders_and_says_so(side_effect):
    with patch(_URLOPEN, side_effect=side_effect):
        known_file, notes = create_known_file("111", kind="fiction")
    assert (known_file.title, known_file.author, known_file.year) == (PLACEHOLDER, PLACEHOLDER, None)
    assert notes == [
        "Open Library has no title for this isbn — fill it in by hand.",
        "Open Library has no author for this isbn — fill it in by hand.",
    ]


def test_the_lookup_retries_once_on_a_dropped_connection():
    body = {"docs": [{"title": "The Book", "author_name": ["Jane Author"]}]}
    with patch(_URLOPEN, side_effect=[urllib.error.URLError(ConnectionResetError()), _response(body), _response({})]):
        known_file, _ = create_known_file("9780000000000", kind="fiction")
    assert known_file.title == "The Book"


@pytest.mark.parametrize("error", [urllib.error.URLError("dns failure"), TimeoutError()])
def test_the_lookup_does_not_retry_errors_a_retry_wont_fix(error):
    with patch(_URLOPEN, side_effect=error) as urlopen:
        create_known_file("9780000000000", kind="fiction")
    assert urlopen.call_count == 2  # search + edition, one attempt each


def test_the_lookup_survives_a_truncated_response():
    truncated = MagicMock()
    truncated.read.side_effect = http.client.IncompleteRead(b"{")
    truncated.__enter__.return_value = truncated
    with patch(_URLOPEN, return_value=truncated):
        known_file, _ = create_known_file("9780000000000", kind="fiction")
    assert known_file.title == PLACEHOLDER
