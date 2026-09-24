import dataclasses

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from precis.pipeline import checkpoints as checkpoints_module
from precis.pipeline.state import COMPLETED_STATE_KEY


async def _seed_thread(db_path: str, thread_id: str, *, ts: str, book: bool, checkpoint_id: str = "1") -> None:
    """Writes one checkpoint directly via AsyncSqliteSaver, bypassing the
    real graph — cheaper than running the whole pipeline just to get a
    thread with known completed/timestamp properties into the DB.
    """
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "ts": ts,
        "channel_values": {COMPLETED_STATE_KEY: {"sentinel": True}} if book else {"chapters": []},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        await saver.aput(config, checkpoint, {"source": "update", "step": 1, "writes": {}, "parents": {}}, {})


@pytest.fixture(autouse=True)
def _use_temp_checkpoint_db(monkeypatch, tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    fast_settings = dataclasses.replace(checkpoints_module.settings, checkpoint_db_path=db_path)
    monkeypatch.setattr(checkpoints_module, "settings", fast_settings)
    return db_path


@pytest.mark.asyncio
async def test_list_checkpoint_threads_reports_completed_and_in_progress(_use_temp_checkpoint_db):
    db_path = _use_temp_checkpoint_db
    await _seed_thread(db_path, "done-thread", ts="2020-01-01T00:00:00+00:00", book=True)
    await _seed_thread(db_path, "wip-thread", ts="2099-01-01T00:00:00+00:00", book=False)

    threads = await checkpoints_module.list_checkpoint_threads()

    by_id = {t.thread_id: t for t in threads}
    assert by_id["done-thread"].completed is True
    assert by_id["wip-thread"].completed is False
    assert len(threads) == 2


@pytest.mark.asyncio
async def test_list_checkpoint_threads_empty_db_returns_empty_list(_use_temp_checkpoint_db):
    assert await checkpoints_module.list_checkpoint_threads() == []


@pytest.mark.asyncio
async def test_prune_default_only_deletes_completed_threads(_use_temp_checkpoint_db):
    db_path = _use_temp_checkpoint_db
    await _seed_thread(db_path, "done-thread", ts="2020-01-01T00:00:00+00:00", book=True)
    await _seed_thread(db_path, "wip-thread", ts="2020-01-01T00:00:00+00:00", book=False)

    deleted = await checkpoints_module.prune_checkpoint_threads()

    assert [t.thread_id for t in deleted] == ["done-thread"]
    remaining = await checkpoints_module.list_checkpoint_threads()
    assert [t.thread_id for t in remaining] == ["wip-thread"]


@pytest.mark.asyncio
async def test_prune_include_incomplete_deletes_everything_eligible(_use_temp_checkpoint_db):
    db_path = _use_temp_checkpoint_db
    await _seed_thread(db_path, "done-thread", ts="2020-01-01T00:00:00+00:00", book=True)
    await _seed_thread(db_path, "wip-thread", ts="2020-01-01T00:00:00+00:00", book=False)

    deleted = await checkpoints_module.prune_checkpoint_threads(include_incomplete=True)

    assert {t.thread_id for t in deleted} == {"done-thread", "wip-thread"}
    assert await checkpoints_module.list_checkpoint_threads() == []


@pytest.mark.asyncio
async def test_prune_older_than_days_excludes_recent_completed_threads(_use_temp_checkpoint_db):
    db_path = _use_temp_checkpoint_db
    await _seed_thread(db_path, "old-done", ts="2020-01-01T00:00:00+00:00", book=True)
    await _seed_thread(db_path, "recent-done", ts="2099-01-01T00:00:00+00:00", book=True)

    deleted = await checkpoints_module.prune_checkpoint_threads(older_than_days=30)

    assert [t.thread_id for t in deleted] == ["old-done"]
    remaining = await checkpoints_module.list_checkpoint_threads()
    assert [t.thread_id for t in remaining] == ["recent-done"]


@pytest.mark.asyncio
async def test_prune_with_no_matches_deletes_nothing(_use_temp_checkpoint_db):
    db_path = _use_temp_checkpoint_db
    await _seed_thread(db_path, "wip-thread", ts="2020-01-01T00:00:00+00:00", book=False)

    deleted = await checkpoints_module.prune_checkpoint_threads()

    assert deleted == []
    assert len(await checkpoints_module.list_checkpoint_threads()) == 1


@pytest.mark.asyncio
async def test_prune_skips_thread_that_changed_between_list_and_delete(_use_temp_checkpoint_db, monkeypatch):
    """Simulates a concurrent `generate` run extending a thread's
    checkpoint history in the window between prune's initial scan and its
    delete step (thread_id_for is content-deterministic, so a "completed"
    thread's id is reused verbatim by a re-run without --fresh) — the
    recheck-before-delete guard must skip it rather than delete state the
    concurrent run now depends on.
    """
    db_path = _use_temp_checkpoint_db
    await _seed_thread(db_path, "done-thread", ts="2020-01-01T00:00:00+00:00", book=True, checkpoint_id="1")

    real_latest_checkpoint = checkpoints_module._latest_checkpoint
    calls = {"count": 0}

    async def fake_latest_checkpoint(checkpointer, thread_id):
        calls["count"] += 1
        if calls["count"] == 2:
            # A concurrent write lands right before the recheck.
            await _seed_thread(db_path, thread_id, ts="2030-01-01T00:00:00+00:00", book=False, checkpoint_id="2")
        return await real_latest_checkpoint(checkpointer, thread_id)

    monkeypatch.setattr(checkpoints_module, "_latest_checkpoint", fake_latest_checkpoint)

    deleted = await checkpoints_module.prune_checkpoint_threads()

    assert deleted == []
    remaining = await checkpoints_module.list_checkpoint_threads()
    assert [t.thread_id for t in remaining] == ["done-thread"]
