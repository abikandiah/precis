from precis.cli import build_parser


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
