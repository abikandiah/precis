import warnings

from langgraph.checkpoint.memory import InMemorySaver

from precis.pipeline.graph import build_graph


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
