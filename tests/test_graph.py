import asyncio
import dataclasses
import warnings

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from precis.pipeline import graph as graph_module
from precis.pipeline.graph import RunBudgetExceeded, build_graph
from precis.schema import Book, KnownFile


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
async def test_stream_with_budget_passes_through_the_exact_call_it_was_given(monkeypatch):
    """Unlike a permissive `*args, **kwargs` fake, this asserts the wrapped
    call actually receives the same input_state/config _stream_with_budget
    was given -- so a future regression that wraps the wrong awaitable, or
    silently drops/mutates the config passed to graph.astream, fails this
    test instead of only failing to be caught by it.
    """
    fast_settings = dataclasses.replace(graph_module.settings, run_budget_seconds=10)
    monkeypatch.setattr(graph_module, "settings", fast_settings)

    received: dict = {}

    class _FakeGraph:
        async def astream(self, input_state, config, stream_mode=None):
            received["input_state"] = input_state
            received["config"] = config
            received["stream_mode"] = stream_mode
            yield {"assemble": {"book": {"sentinel": True}}}

    expected_input = {"known_file": {"isbn": "1"}, "trust_known": True}
    expected_config = {"configurable": {"thread_id": "abc"}, "max_concurrency": 3}

    result = await graph_module._stream_with_budget(
        _FakeGraph(), expected_input, expected_config, total_chapters=0, on_progress=None
    )

    assert received["input_state"] is expected_input
    assert received["config"] is expected_config
    assert received["stream_mode"] == "updates"
    assert result == {"book": {"sentinel": True}}


@pytest.mark.asyncio
async def test_stream_with_budget_raises_run_budget_exceeded_on_timeout(monkeypatch):
    """run_budget_seconds is a circuit breaker for a hung run (see
    docs/blueprint.md's Run budget section) -- this confirms it's actually
    enforced, not just a config field nothing reads, and that the raised
    exception is the dedicated RunBudgetExceeded type, not a bare
    TimeoutError indistinguishable from any other timeout in the stack.
    """
    fast_settings = dataclasses.replace(graph_module.settings, run_budget_seconds=0.01)
    monkeypatch.setattr(graph_module, "settings", fast_settings)

    class _NeverFinishesGraph:
        async def astream(self, input_state, config, stream_mode=None):
            await asyncio.sleep(10)
            yield {"assemble": {"book": {}}}

    with pytest.raises(RunBudgetExceeded, match="run budget"):
        await graph_module._stream_with_budget(_NeverFinishesGraph(), {}, {}, total_chapters=0, on_progress=None)


@pytest.mark.asyncio
async def test_stream_with_budget_raises_if_assemble_never_runs():
    """A graph that finishes streaming without ever producing an `assemble`
    update means the routing is broken (assemble is the only node that
    produces `book`, and it's always the last one before END) -- this must
    surface as a loud bug report, not a silent `None`/KeyError downstream.
    """

    class _NeverAssemblesGraph:
        async def astream(self, input_state, config, stream_mode=None):
            yield {"verify": {"verified": True, "verify_reason": "known-file confirmed against search results"}}

    with pytest.raises(RuntimeError, match="assemble"):
        await graph_module._stream_with_budget(_NeverAssemblesGraph(), {}, {}, total_chapters=0, on_progress=None)


@pytest.mark.asyncio
async def test_stream_with_budget_calls_on_progress_for_each_update():
    class _FakeGraph:
        async def astream(self, input_state, config, stream_mode=None):
            yield {"verify": {"verified": True, "verify_reason": "known-file confirmed against search results"}}
            yield {"draft_one_chapter": {"chapters": [{"number": 1, "title": "Ch 1", "quality_flag": None}]}}
            yield {"assemble": {"book": {"sentinel": True}}}

    messages: list[str] = []
    result = await graph_module._stream_with_budget(
        _FakeGraph(), {}, {}, total_chapters=1, on_progress=messages.append
    )

    assert result == {"book": {"sentinel": True}}
    assert messages == [
        "verify: known-file confirmed against search results",
        f"drafting 1 chapter, up to {graph_module.settings.concurrency} at a time...",
        "chapter 1/1 drafted: 'Ch 1'",
        "assemble: book finalized",
    ]


def test_format_chapter_progress_with_and_without_total_and_flag():
    assert graph_module.format_chapter_progress(2, "Ch 2", None, total_chapters=5) == "chapter 2/5 drafted: 'Ch 2'"
    assert (
        graph_module.format_chapter_progress(2, "Ch 2", "fallback used", total_chapters=5)
        == "chapter 2/5 drafted: 'Ch 2' (flagged: fallback used)"
    )
    # No total_chapters given (generate-chapter without a known count) —
    # bare chapter number, same title/flag wording otherwise.
    assert graph_module.format_chapter_progress(2, "Ch 2", None) == "chapter 2 drafted: 'Ch 2'"
    # A critique's whole feedback is cut short — it's in the book's warnings.
    long_flag = "critique failed after 3 attempts: " + "x" * 200
    line = graph_module.format_chapter_progress(2, "Ch 2", long_flag)
    assert line.startswith("chapter 2 drafted: 'Ch 2' (flagged: critique failed after 3 attempts: xxx")
    assert line.endswith("…)")
    assert "x" * 200 not in line
    # Kept to one line even when the feedback has its own line breaks.
    assert graph_module.format_chapter_progress(2, "Ch 2", "Issues:\n- claim X\n- claim Y") == (
        "chapter 2 drafted: 'Ch 2' (flagged: Issues: - claim X - claim Y)"
    )


def test_progress_messages_for_each_node_shape():
    assert graph_module._progress_messages("verify", {"verified": True, "verify_reason": "looks right"}, 0) == [
        "verify: looks right"
    ]
    # Differences are indented under the verdict, then drafting is announced.
    assert graph_module._progress_messages(
        "verify", {"verified": True, "verify_reason": "right book\nDifferences:\n  - year: 2020 vs 2015"}, 3
    ) == [
        "verify: right book",
        "  Differences:",
        "    - year: 2020 vs 2015",
        f"drafting 3 chapters, up to {graph_module.settings.concurrency} at a time...",
    ]
    assert graph_module._progress_messages("verify", {"verified": True}, 0) == ["verify: known-file confirmed"]
    assert graph_module._progress_messages(
        "draft_one_chapter", {"chapters": [{"number": 2, "title": "Ch 2", "quality_flag": None}]}, 5
    ) == ["chapter 2/5 drafted: 'Ch 2'"]
    assert graph_module._progress_messages(
        "draft_one_chapter",
        {"chapters": [{"number": 2, "title": "Ch 2", "quality_flag": "fallback used"}]},
        5,
    ) == ["chapter 2/5 drafted: 'Ch 2' (flagged: fallback used)"]
    assert graph_module._progress_messages("synthesize", {"parts_source": "known"}, 0) == [
        "synthesize: synopsis/tags/parts complete (parts: known)"
    ]
    assert graph_module._progress_messages(
        "synthesize", {"parts_source": "known", "warnings": ["part 'X': model suggested a different title"]}, 0
    ) == ["synthesize: synopsis/tags/parts complete (parts: known) — 1 warning(s) noted"]
    assert graph_module._progress_messages("assemble", {"book": {}}, 0) == ["assemble: book finalized"]
    book = {
        "chapters": [{"quality_flag": None}, {"quality_flag": "critique failed"}],
        "warnings": ["chapter 2: critique failed"],
    }
    assert graph_module._progress_messages("assemble", {"book": book}, 2) == [
        "assemble: book finalized (2 chapters, 1 warning(s) — see the book's warnings)"
    ]
    assert graph_module._progress_messages("some_other_node", {}, 0) == []


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
        async def astream(self, input_state, config, stream_mode=None):
            await asyncio.sleep(10)
            yield {"assemble": {"book": {}}}

    monkeypatch.setattr(graph_module, "build_graph", lambda checkpointer: _NeverFinishesGraph())

    known_file = KnownFile(isbn="1", title="A Novel", author="A Novelist", kind="fiction")
    with pytest.raises(RunBudgetExceeded, match="run budget"):
        await graph_module.run_whole_book(known_file, slug="a-novel", trust_known=True)


@pytest.mark.asyncio
async def test_run_whole_book_threads_on_progress_through_to_the_real_stream(monkeypatch, tmp_path):
    """Confirms on_progress isn't just accepted and dropped -- it has to
    reach _stream_with_budget's actual astream loop through run_whole_book's
    full setup (client construction, checkpointer, thread_id).
    """
    fast_settings = dataclasses.replace(
        graph_module.settings,
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
    )
    monkeypatch.setattr(graph_module, "settings", fast_settings)
    monkeypatch.setattr(graph_module, "build_search_client", lambda: object())
    monkeypatch.setattr(graph_module.llm, "build_client", lambda: object())

    fake_book = {
        "schema_version": "1",
        "title": "T",
        "author": "A",
        "isbn": "1",
        "kind": "fiction",
        "one_line_takeaway": "takeaway",
        "synopsis": "synopsis",
        "tags": ["fantasy", "adventure"],
    }

    class _FakeGraph:
        async def astream(self, input_state, config, stream_mode=None):
            yield {"verify": {"verified": True, "verify_reason": "known-file confirmed against search results"}}
            yield {"assemble": {"book": fake_book}}

    monkeypatch.setattr(graph_module, "build_graph", lambda checkpointer: _FakeGraph())

    messages: list[str] = []
    known_file = KnownFile(isbn="1", title="A Novel", author="A Novelist", kind="fiction")
    book = await graph_module.run_whole_book(known_file, slug="a-novel", trust_known=True, on_progress=messages.append)

    assert book.title == "T"
    assert messages == [
        "verify: known-file confirmed against search results",
        "assemble: book finalized",
    ]


_FAKE_BOOK = {
    "schema_version": "1",
    "title": "T",
    "author": "A",
    "isbn": "1",
    "kind": "fiction",
    "one_line_takeaway": "takeaway",
    "synopsis": "synopsis",
    "tags": ["fantasy", "adventure"],
}


class _StubbedPipeline:
    """Real graph + real SQLite checkpointer, with every node replaced by a
    stub that records its calls — exercises run_whole_book's resume rules
    end to end without any LLM/search.
    """

    def __init__(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            graph_module,
            "settings",
            dataclasses.replace(
                graph_module.settings,
                checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
            ),
        )
        monkeypatch.setattr(graph_module, "build_search_client", lambda: object())
        monkeypatch.setattr(graph_module.llm, "build_client", lambda: object())
        self.calls: list[str] = []
        self.fail_verify = False
        self.fail_chapter: int | None = None

        async def verify_run(state, config=None):
            self.calls.append(f"verify(trust_known={state['trust_known']})")
            if self.fail_verify:
                raise ValueError("Stage 1 verify failed")
            return {"verified": True, "verify_reason": "ok"}

        async def draft_run_one(state, config=None):
            n = state["chapter_number"]
            self.calls.append(f"draft {n}")
            if n == self.fail_chapter:
                raise RuntimeError("chapter failed")
            return {"chapters": [{"number": n}]}

        async def synthesize_run(state, config=None):
            self.calls.append("synthesize")
            return {}

        async def assemble_run(state, config=None):
            self.calls.append(f"assemble ({len(state['chapters'])} chapters)")
            return {"book": _FAKE_BOOK}

        monkeypatch.setattr(graph_module.verify, "run", verify_run)
        monkeypatch.setattr(graph_module.draft, "run_one", draft_run_one)
        monkeypatch.setattr(graph_module.synthesize, "run", synthesize_run)
        monkeypatch.setattr(graph_module.assemble, "run", assemble_run)

    async def run(self, known_file: KnownFile, **kwargs) -> Book:
        self.calls.clear()
        return await graph_module.run_whole_book(known_file, slug="the-book", **kwargs)


def _nonfiction(**overrides) -> KnownFile:
    return KnownFile(
        **{
            "isbn": "1",
            "title": "B",
            "author": "A",
            "kind": "non-fiction",
            "chapters": ["One", "Two"],
            **overrides,
        }
    )


@pytest.mark.asyncio
async def test_rerun_resumes_only_the_unfinished_work(monkeypatch, tmp_path):
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    pipeline.fail_chapter = 2
    with pytest.raises(RuntimeError):
        await pipeline.run(_nonfiction())

    pipeline.fail_chapter = None
    await pipeline.run(_nonfiction())
    assert pipeline.calls == ["draft 2", "synthesize", "assemble (2 chapters)"]


@pytest.mark.asyncio
async def test_leftover_finished_thread_is_never_handed_back(monkeypatch, tmp_path):
    # The CLI deletes a finished thread once the book is written; one that
    # outlived a failed write just starts over.
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    await pipeline.run(_nonfiction())

    await pipeline.run(_nonfiction(chapters=["One", "Two", "Three"]))
    assert pipeline.calls[0] == "verify(trust_known=False)"
    assert pipeline.calls[-1] == "assemble (3 chapters)"


@pytest.mark.asyncio
async def test_changed_known_file_after_verify_aborts_unless_fresh(monkeypatch, tmp_path):
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    pipeline.fail_chapter = 2
    with pytest.raises(RuntimeError):
        await pipeline.run(_nonfiction())
    pipeline.fail_chapter = None

    with pytest.raises(graph_module.CheckpointMismatch, match=r"\(chapters\).*--fresh"):
        await pipeline.run(_nonfiction(chapters=["One", "Two", "Three"], notes="new"))

    await pipeline.run(_nonfiction(chapters=["One", "Two", "Three"], notes="new"), fresh=True)
    assert pipeline.calls[0] == "verify(trust_known=False)"
    assert pipeline.calls[-1] == "assemble (3 chapters)"


@pytest.mark.asyncio
async def test_failed_verify_restarts_cleanly_so_trust_known_applies(monkeypatch, tmp_path):
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    pipeline.fail_verify = True
    with pytest.raises(ValueError, match="verify failed"):
        await pipeline.run(_nonfiction())

    pipeline.fail_verify = False
    await pipeline.run(_nonfiction(notes="edited too"), trust_known=True)
    assert pipeline.calls[0] == "verify(trust_known=True)"
    assert pipeline.calls[-1] == "assemble (2 chapters)"


@pytest.mark.asyncio
async def test_run_started_with_trust_known_wont_resume_without_it(monkeypatch, tmp_path):
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    pipeline.fail_chapter = 2
    with pytest.raises(RuntimeError):
        await pipeline.run(_nonfiction(), trust_known=True)
    pipeline.fail_chapter = None

    with pytest.raises(graph_module.CheckpointMismatch, match="--trust-known"):
        await pipeline.run(_nonfiction())

    await pipeline.run(_nonfiction(), trust_known=True)
    assert pipeline.calls == ["draft 2", "synthesize", "assemble (2 chapters)"]


@pytest.mark.asyncio
async def test_checkpoint_for_a_different_book_with_the_same_slug_is_refused(monkeypatch, tmp_path):
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    pipeline.fail_chapter = 2
    with pytest.raises(RuntimeError):
        await pipeline.run(_nonfiction())

    with pytest.raises(graph_module.CheckpointMismatch, match="different book"):
        await pipeline.run(_nonfiction(isbn="2"))



@pytest.mark.asyncio
async def test_editing_only_notes_resumes_and_uses_the_new_notes(monkeypatch, tmp_path):
    pipeline = _StubbedPipeline(monkeypatch, tmp_path)
    pipeline.fail_chapter = 2
    with pytest.raises(RuntimeError):
        await pipeline.run(_nonfiction(notes="old"))
    pipeline.fail_chapter = None

    book = await pipeline.run(_nonfiction(notes="new"))
    assert pipeline.calls == ["draft 2", "synthesize", "assemble (2 chapters)"]
    assert book.reader_notes == "new"
