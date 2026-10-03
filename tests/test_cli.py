import json
from unittest.mock import AsyncMock

import pytest

from precis import cli as cli_module
from precis.cli import _known_file_filename, build_parser
from precis.research import Research, ResearchError, Source
from precis.schema import FICTION_TAGS, NONFICTION_TAGS, SCHEMA_VERSION, Book, KnownFile

_READY = KnownFile(isbn="1", title="A Novel", author="A Novelist", kind="fiction")
_BOOK = Book(
    title="A Novel", author="A Novelist", isbn="1", kind="fiction", depth="overview", one_line_takeaway="takeaway",
    synopsis="synopsis", tags=["fantasy", "adventure"],
    ideas=[{"title": f"Theme {n}", "summary": "s", "evidence": "e"} for n in range(3)],
)  # fmt: skip
_GENERATED = _BOOK


def _run(args: list[str]) -> int:
    parsed = build_parser().parse_args(args)
    return parsed.func(parsed)


@pytest.fixture
def known_file_path(tmp_path):
    path = tmp_path / "the-book.json"
    path.write_text(_READY.model_dump_json())
    return path


# --- loading ------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["generate", "research"])
def test_a_missing_or_malformed_known_file_is_a_clean_error(tmp_path, capsys, command):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    for path in ("/nonexistent/book.json", str(bad)):
        assert _run([command, path]) == 1
        err = capsys.readouterr().err
        assert "could not load known-file" in err and "Traceback" not in err


def test_a_known_file_that_isnt_ready_is_refused_before_any_work(tmp_path, capsys, monkeypatch):
    path = tmp_path / "book.json"
    path.write_text(KnownFile(isbn="1", title="A Novel", kind="fiction").model_dump_json())
    monkeypatch.setattr(cli_module, "generate", AsyncMock(side_effect=AssertionError("started")))
    assert _run(["generate", str(path)]) == 1
    assert "author is required" in capsys.readouterr().err


# --- generate -----------------------------------------------------------------------


def test_generate_writes_the_book_and_names_the_research_cache_after_the_file(known_file_path, tmp_path, monkeypatch):
    fake = AsyncMock(return_value=_GENERATED)
    monkeypatch.setattr(cli_module, "generate", fake)
    out = tmp_path / "out.json"
    assert _run(["generate", str(known_file_path), "--output", str(out), "--fresh"]) == 0
    written = json.loads(out.read_text())
    assert written["schema_version"] == SCHEMA_VERSION and written["title"] == "A Novel"
    assert "key_claims_for_review" not in written  # absent, not null
    assert fake.await_args.kwargs["slug"] == "the-book"
    assert fake.await_args.kwargs["fresh"] is True


def test_generate_reads_the_book_file_relative_to_the_known_file(tmp_path, monkeypatch):
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "a-novel.txt").write_text("A long line of the novel itself, in the reader's copy.\n" * 500)
    path = tmp_path / "the-book.json"
    path.write_text(_READY.model_copy(update={"book_file": "books/a-novel.txt"}).model_dump_json())
    fake = AsyncMock(return_value=_GENERATED)
    monkeypatch.setattr(cli_module, "generate", fake)
    assert _run(["generate", str(path), "--output", str(tmp_path / "out.json")]) == 0
    assert fake.await_args.kwargs["book_text"].startswith("A long line of the novel itself")
    # --overview writes from search even so — and can't be asked for alongside a book file.
    assert _run(["generate", str(path), "--overview", "--output", str(tmp_path / "out.json")]) == 0
    assert fake.await_args.kwargs["book_text"] is None
    with pytest.raises(SystemExit):
        _run(["generate", str(path), "--overview", "--book-file", "books/a-novel.txt"])


@pytest.mark.parametrize("command", ["generate", "research"])
def test_an_unreadable_book_file_is_refused_before_any_work(known_file_path, monkeypatch, capsys, command):
    monkeypatch.setattr(cli_module, "generate", AsyncMock(side_effect=AssertionError("started")))
    monkeypatch.setattr(cli_module.book_research, "research", AsyncMock(side_effect=AssertionError("started")))
    assert _run([command, str(known_file_path), "--book-file", "/nonexistent/book.epub"]) == 1
    assert "book_file: no such file" in capsys.readouterr().err


@pytest.mark.parametrize(("output", "message"), [("missing/out.json", "doesn't exist"), ("", "directory")])
def test_generate_refuses_an_unusable_output_before_any_work(known_file_path, tmp_path, monkeypatch, capsys, output, message):
    monkeypatch.setattr(cli_module, "generate", AsyncMock(side_effect=AssertionError("started")))
    assert _run(["generate", str(known_file_path), "--output", str(tmp_path / output)]) == 1
    assert message in capsys.readouterr().err


def test_generate_wont_write_the_book_over_its_known_file(known_file_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "generate", AsyncMock(side_effect=AssertionError("started")))
    assert _run(["generate", str(known_file_path), "--output", str(known_file_path)]) == 1
    assert "it's the known-file" in capsys.readouterr().err


def test_ctrl_c_is_a_clean_exit_not_a_traceback(capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    args = build_parser().parse_args(["tags"])
    args.func = interrupted
    assert cli_module._dispatch(args) == 130
    assert "interrupted" in capsys.readouterr().err


def test_generate_prints_the_book_when_the_final_write_fails(known_file_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "generate", AsyncMock(return_value=_GENERATED))
    real_write = cli_module._write_output

    def failing_write(model, output_path, *, exclude_none=False):
        if output_path:
            raise OSError(28, "No space left on device")
        real_write(model, output_path, exclude_none=exclude_none)

    monkeypatch.setattr(cli_module, "_write_output", failing_write)
    assert _run(["generate", str(known_file_path), "--output", str(tmp_path / "out.json")]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["title"] == "A Novel"
    assert "printing the book to stdout instead" in captured.err


def test_a_failed_run_reports_cleanly_and_still_prints_its_cost(known_file_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "generate", AsyncMock(side_effect=ResearchError("couldn't find this book online")))
    assert _run(["generate", str(known_file_path)]) == 1
    err = capsys.readouterr().err
    assert "generation failed: couldn't find this book online" in err
    assert "usage:" in err


# --- research -----------------------------------------------------------------------


def test_research_prints_the_rendered_research(known_file_path, monkeypatch, capsys):
    found = Research(sources=[Source(id="S1", title="t", url="https://a.org", text="text")], warnings=[])
    fake = AsyncMock(return_value=found)
    monkeypatch.setattr(cli_module.book_research, "research", fake)
    assert _run(["research", str(known_file_path)]) == 0
    assert fake.await_args.kwargs["slug"] == "the-book"
    out, err = capsys.readouterr()
    assert '<source id="S1"' in out
    assert "S1: https://a.org" in err


# --- create-known-file and tags ---------------------------------------------------------


def test_write_output_keeps_none_explicit_unless_asked(tmp_path):
    out = tmp_path / "out.json"
    cli_module._write_output(KnownFile(isbn="123", kind="fiction"), str(out))
    assert json.loads(out.read_text())["year"] is None
    cli_module._write_output(KnownFile(isbn="123", kind="fiction"), str(out), exclude_none=True)
    assert "year" not in json.loads(out.read_text())


def test_known_file_filenames_fall_back_to_the_isbn_and_never_collide():
    same = KnownFile(isbn="123", kind="non-fiction", title="Same Book")
    assert _known_file_filename("123", KnownFile(isbn="123", kind="fiction"), set()) == "123.json"
    used: set[str] = set()
    for _ in range(3):
        used.add(_known_file_filename("123", same, used))
    assert used == {"same-book.json", "same-book-123.json", "same-book-123-2.json"}


def test_tags_prints_both_closed_vocabularies(capsys):
    assert _run(["tags"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == SCHEMA_VERSION
    assert (output["non_fiction_tags"], output["fiction_tags"]) == (list(NONFICTION_TAGS), list(FICTION_TAGS))


def test_tags_to_an_unwritable_path_is_an_error_not_a_traceback(tmp_path, capsys):
    assert _run(["tags", "--output", str(tmp_path / "missing" / "tags.json")]) == 1
    assert "couldn't write" in capsys.readouterr().err


def _fake_lookup(monkeypatch, titles: dict[str, str]) -> None:
    monkeypatch.setattr(
        cli_module,
        "create_known_file",
        lambda isbn, kind: (KnownFile(isbn=isbn, kind=kind, title=titles[isbn], author="A"), []),
    )


def test_create_known_file_wont_replace_an_existing_file_without_force(tmp_path, capsys, monkeypatch):
    _fake_lookup(monkeypatch, {"1": "One"})
    out = tmp_path / "one.json"
    out.write_text("hand-edited")
    assert _run(["create-known-file", "1", "--kind", "fiction", "--output", str(out)]) == 1
    assert "--force" in capsys.readouterr().err and out.read_text() == "hand-edited"
    assert _run(["create-known-file", "1", "--kind", "fiction", "--output", str(out), "--force"]) == 0
    assert json.loads(out.read_text())["title"] == "One"


def test_a_batch_skips_existing_files_and_still_writes_the_rest(tmp_path, capsys, monkeypatch):
    _fake_lookup(monkeypatch, {"1": "One", "2": "Two"})
    (tmp_path / "one.json").write_text("hand-edited")
    assert _run(["create-known-file", "1", "2", "--output-dir", str(tmp_path)]) == 1
    assert "1: not writing" in capsys.readouterr().err
    assert (tmp_path / "one.json").read_text() == "hand-edited"
    assert json.loads((tmp_path / "two.json").read_text())["title"] == "Two"
