import asyncio
import dataclasses
import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from precis.pipeline import graph as graph_module
from precis.pipeline.graph import build_graph
from precis.schema import KnownFile


def test_building_the_graph_raises_no_config_typing_warning():
    """Regression test: each pipeline node file used to have
    `from __future__ import annotations`, which makes the `config`
    parameter's annotation a string at runtime instead of a real type
    object -- LangGraph's node-registration introspection compares that
    annotation against real type objects and silently mismatched, emitting
    a UserWarning on every node registration in every real run (see
    verify.py/draft.py/synthesize.py/assemble.py, all of which had the
    future-import removed specifically to fix this). Nothing else enforces
    it doesn't come back, so this test builds the real graph and fails if
    the warning reappears.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_graph(InMemorySaver())

    config_typing_warnings = [w for w in caught if "should be typed as" in str(w.message)]
    assert not config_typing_warnings, [str(w.message) for w in config_typing_warnings]


@pytest.mark.asyncio
async def test_run_budget_timeout_raises_clear_error(monkeypatch, tmp_path):
    """run_budget_seconds is a circuit breaker for a hung run (see
    docs/blueprint.md's Run budget section) -- this confirms it's actually
    enforced, not just a config field nothing reads.
    """
    fast_settings = dataclasses.replace(
        graph_module.settings,
        run_budget_seconds=0.01,
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
    )
    monkeypatch.setattr(graph_module, "settings", fast_settings)
    monkeypatch.setattr(graph_module, "build_search_client", lambda: object())
    monkeypatch.setattr(graph_module.llm, "build_client", lambda: object())

    class _NeverFinishesGraph:
        async def ainvoke(self, *args, **kwargs):
            await asyncio.sleep(10)
            return {"book": {}}

    monkeypatch.setattr(graph_module, "build_graph", lambda checkpointer: _NeverFinishesGraph())

    known_file = KnownFile(isbn="1", kind="fiction")
    with pytest.raises(TimeoutError, match="run budget"):
        await graph_module.run_whole_book(known_file, trust_known=True)
