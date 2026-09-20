import asyncio
import dataclasses
import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from precis.pipeline import graph as graph_module
from precis.pipeline.graph import RunBudgetExceeded, build_graph
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
async def test_invoke_with_budget_passes_through_the_exact_call_it_was_given(monkeypatch):
    """Unlike a permissive `*args, **kwargs` fake, this asserts the wrapped
    call actually receives the same input_state/config _invoke_with_budget
    was given -- so a future regression that wraps the wrong awaitable, or
    silently drops/mutates the config passed to graph.ainvoke, fails this
    test instead of only failing to be caught by it.
    """
    fast_settings = dataclasses.replace(graph_module.settings, run_budget_seconds=10)
    monkeypatch.setattr(graph_module, "settings", fast_settings)

    received: dict = {}

    class _FakeGraph:
        async def ainvoke(self, input_state, config):
            received["input_state"] = input_state
            received["config"] = config
            return {"book": {"sentinel": True}}

    expected_input = {"known_file": {"isbn": "1"}, "trust_known": True}
    expected_config = {"configurable": {"thread_id": "abc"}, "max_concurrency": 3}

    result = await graph_module._invoke_with_budget(_FakeGraph(), expected_input, expected_config)

    assert received["input_state"] is expected_input
    assert received["config"] is expected_config
    assert result == {"book": {"sentinel": True}}


@pytest.mark.asyncio
async def test_invoke_with_budget_raises_run_budget_exceeded_on_timeout(monkeypatch):
    """run_budget_seconds is a circuit breaker for a hung run (see
    docs/blueprint.md's Run budget section) -- this confirms it's actually
    enforced, not just a config field nothing reads, and that the raised
    exception is the dedicated RunBudgetExceeded type, not a bare
    TimeoutError indistinguishable from any other timeout in the stack.
    """
    fast_settings = dataclasses.replace(graph_module.settings, run_budget_seconds=0.01)
    monkeypatch.setattr(graph_module, "settings", fast_settings)

    class _NeverFinishesGraph:
        async def ainvoke(self, input_state, config):
            await asyncio.sleep(10)
            return {"book": {}}

    with pytest.raises(RunBudgetExceeded, match="run budget"):
        await graph_module._invoke_with_budget(_NeverFinishesGraph(), {}, {})


@pytest.mark.asyncio
async def test_run_whole_book_propagates_run_budget_exceeded(monkeypatch, tmp_path):
    """End-to-end version of the above, through the real run_whole_book
    (client construction, checkpointer setup, thread_id derivation) to
    confirm nothing in that path swallows or reinterprets the timeout.
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
        async def ainvoke(self, input_state, config):
            await asyncio.sleep(10)
            return {"book": {}}

    monkeypatch.setattr(graph_module, "build_graph", lambda checkpointer: _NeverFinishesGraph())

    known_file = KnownFile(isbn="1", kind="fiction")
    with pytest.raises(RunBudgetExceeded, match="run budget"):
        await graph_module.run_whole_book(known_file, trust_known=True)
