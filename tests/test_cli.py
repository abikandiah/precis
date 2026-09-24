import asyncio
import dataclasses
import json

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from precis import cli as cli_module
from precis.cli import _known_file_filename, build_parser
from precis.pipeline.state import COMPLETED_STATE_KEY
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


def test_generate_chapter_prints_progress_in_the_shared_whole_book_format(tmp_path, capsys, monkeypatch):
    """generate-chapter's progress line must use the same wording
    (position/total, title, flagged-suffix) as the whole-book `generate`
    path's per-chapter progress messages — the two used to drift
    independently before format_chapter_progress was extracted.
    """
    known_file_path = tmp_path / "book.json"
    known_file = KnownFile(isbn="123", kind="non-fiction", chapters=["Ch 1", "Ch 2", "Ch 3"])
    known_file_path.write_text(known_file.model_dump_json())

    async def fake_run_one(state):
        return {
            "chapters": [
                {
                    "number": state["chapter_number"],
                    "title": state["chapter_title"],
                    "key_points": ["a point"],
                    "core_claim": "a claim",
                    "quality_flag": None,
                }
            ]
        }

    monkeypatch.setattr(cli_module.draft, "run_one", fake_run_one)

    exit_code = _run(["generate-chapter", str(known_file_path), "--chapter", "2"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "chapter 2/3 drafted: 'Ch 2'" in captured.err
    output = json.loads(captured.out)
    assert output["number"] == 2


async def _seed_checkpoint_thread(db_path: str, thread_id: str, *, ts: str, book: bool) -> None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "id": "1",
        "ts": ts,
        "channel_values": {COMPLETED_STATE_KEY: {"sentinel": True}} if book else {"chapters": []},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        await saver.aput(config, checkpoint, {"source": "update", "step": 1, "writes": {}, "parents": {}}, {})


def test_checkpoints_with_empty_db_reports_none_found(tmp_path, capsys, monkeypatch):
    db_path = str(tmp_path / "checkpoints.sqlite")
    fast_settings = dataclasses.replace(cli_module.pipeline_checkpoints.settings, checkpoint_db_path=db_path)
    monkeypatch.setattr(cli_module.pipeline_checkpoints, "settings", fast_settings)

    exit_code = _run(["checkpoints"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no checkpoint threads found" in captured.err


def test_checkpoints_lists_threads_with_status(tmp_path, capsys, monkeypatch):
    db_path = str(tmp_path / "checkpoints.sqlite")
    fast_settings = dataclasses.replace(cli_module.pipeline_checkpoints.settings, checkpoint_db_path=db_path)
    monkeypatch.setattr(cli_module.pipeline_checkpoints, "settings", fast_settings)
    asyncio.run(_seed_checkpoint_thread(db_path, "done-thread", ts="2020-01-01T00:00:00+00:00", book=True))
    asyncio.run(_seed_checkpoint_thread(db_path, "wip-thread", ts="2020-01-01T00:00:00+00:00", book=False))

    exit_code = _run(["checkpoints"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "done-thread  done" in captured.out
    assert "wip-thread  in-progress" in captured.out


def test_checkpoints_prune_deletes_only_completed_by_default(tmp_path, capsys, monkeypatch):
    db_path = str(tmp_path / "checkpoints.sqlite")
    fast_settings = dataclasses.replace(cli_module.pipeline_checkpoints.settings, checkpoint_db_path=db_path)
    monkeypatch.setattr(cli_module.pipeline_checkpoints, "settings", fast_settings)
    asyncio.run(_seed_checkpoint_thread(db_path, "done-thread", ts="2020-01-01T00:00:00+00:00", book=True))
    asyncio.run(_seed_checkpoint_thread(db_path, "wip-thread", ts="2020-01-01T00:00:00+00:00", book=False))

    exit_code = _run(["checkpoints", "--prune"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "deleted done-thread" in captured.out
    assert "wip-thread" not in captured.out
    assert "pruned 1 checkpoint thread(s)" in captured.err


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
