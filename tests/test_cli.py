from precis.cli import _known_file_filename, build_parser
from precis.schema import KnownFile


def _run(args: list[str]) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)
    return parsed.func(parsed)


def test_generate_with_missing_known_file_reports_clean_error_not_traceback(capsys):
    exit_code = _run(["generate", "/nonexistent/path/does-not-exist.json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "could not load known-file" in captured.err
    assert "Traceback" not in captured.err


def test_generate_with_malformed_json_reports_clean_error_not_traceback(tmp_path, capsys):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json at all")

    exit_code = _run(["generate", str(bad_file)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "could not load known-file" in captured.err
    assert "Traceback" not in captured.err


def test_generate_chapter_with_missing_known_file_reports_clean_error(capsys):
    exit_code = _run(["generate-chapter", "/nonexistent/path.json", "--chapter", "1"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "could not load known-file" in captured.err
    assert "Traceback" not in captured.err


def _known_file(**overrides) -> KnownFile:
    defaults = {"isbn": "123", "kind": "non-fiction", "title": "Same Book"}
    defaults.update(overrides)
    return KnownFile(**defaults)


def test_known_file_filename_uses_title_slug():
    assert _known_file_filename("123", _known_file(), set()) == "same-book.json"


def test_known_file_filename_falls_back_to_isbn_without_a_real_title():
    kf = _known_file(title=None)
    assert _known_file_filename("123", kf, set()) == "123.json"


def test_known_file_filename_resolves_repeated_collisions_for_the_same_isbn():
    # A repeated isbn in one batch (e.g. `create-known-file 123 123 123`)
    # used to make the 2nd and 3rd calls collide on the same fallback name,
    # silently overwriting the 2nd's file.
    used: set[str] = set()
    kf = _known_file()
    names = []
    for _ in range(3):
        name = _known_file_filename("123", kf, used)
        used.add(name)
        names.append(name)

    assert names == ["same-book.json", "same-book-123.json", "same-book-123-2.json"]
    assert len(set(names)) == 3
