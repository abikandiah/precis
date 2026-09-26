"""Builds the whole-book LangGraph pipeline and drives one run.

Orchestration is LangGraph specifically for checkpoint/resume — see
docs/blueprint.md's Orchestration section. The checkpoint DB must live on a
volume mounted into the generation container, not its ephemeral filesystem,
or resume across container restarts doesn't work (config.settings.checkpoint_db_path).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from precis import llm
from precis.config import settings
from precis.known_file import ensure_ready
from precis.pipeline.nodes import assemble, draft, synthesize, verify
from precis.pipeline.state import COMPLETED_STATE_KEY, GraphState
from precis.schema import Book, KnownFile
from precis.search import build_search_client

ProgressCallback = Callable[[str], None]

# Single source of truth for node names — used by build_graph's add_node/
# add_conditional_edges/add_edge calls, _route_after_verify's routing
# decision, _progress_messages' dispatch, and _stream_with_budget's
# terminal-output check below. A node rename that misses one of these
# would previously fail silently (progress narration for that node just
# stops) or, for NODE_ASSEMBLE specifically, break every run outright
# (see _stream_with_budget) — collecting them here means a rename is one
# edit instead of an easy-to-miss find-and-replace across string literals.
NODE_VERIFY = "verify"
NODE_DRAFT_CHAPTER = "draft_one_chapter"
NODE_SYNTHESIZE = "synthesize"
NODE_ASSEMBLE = "assemble"


class RunBudgetExceeded(TimeoutError):
    """Raised when a whole-book run exceeds settings.run_budget_seconds —
    a distinct type from bare TimeoutError so a caller can tell this
    deliberate circuit breaker apart from an incidental timeout elsewhere
    in the stack (e.g. an LLM client's own per-call timeout), rather than
    only by parsing the message string. Still a TimeoutError, so existing
    `except TimeoutError` handling elsewhere doesn't need to change.
    """


class CheckpointMismatch(ValueError):
    """The book's checkpoint got past verify but can't be resumed as-is —
    the known-file was edited (resuming would mix chapters drafted from two
    different files), it names a different book, or the saved run skipped
    verify and this one doesn't. Raised instead of guessing; `--fresh` is
    the explicit way out.
    """


# Known-file fields no pipeline stage reads — `notes` is only copied
# verbatim into the finished book's `reader_notes` (see run_whole_book), so
# editing it mid-run doesn't invalidate any drafted chapter.
_PASS_THROUGH_FIELDS = {"notes"}


def _changed_fields(saved: dict, current: dict) -> list[str]:
    keys = (saved.keys() | current.keys()) - _PASS_THROUGH_FIELDS
    return sorted(k for k in keys if saved.get(k) != current.get(k))


def _finish(book: dict, known_file: KnownFile) -> Book:
    # A resumed or reused run carries the notes from when it started;
    # the reader's current notes win.
    return Book.model_validate({**book, "reader_notes": known_file.notes})


def _resume_blocker(slug: str, saved_values: dict, current: dict, *, trust_known: bool) -> str | None:
    """Why a checkpoint that got past verify can't be resumed as-is, or None."""
    saved_known = saved_values.get("known_file") or {}
    if saved_known.get("isbn") != current.get("isbn"):
        return (
            f"the checkpoint named {slug!r} belongs to a different book (isbn {saved_known.get('isbn')!r}, "
            f"not {current.get('isbn')!r}). Rename this known-file, or rerun with --fresh to discard that checkpoint."
        )
    if changed := _changed_fields(saved_known, current):
        return (
            f"known-file for {slug!r} changed since its checkpoint was saved ({', '.join(changed)}). "
            "Rerun with --fresh to discard the checkpoint and start over, or undo the change to resume."
        )
    # Verify was skipped on the saved run; resuming without --trust-known
    # would finish (or hand back) a book that was never checked.
    if saved_values.get("trust_known") and not trust_known:
        return (
            f"{slug!r} was started with --trust-known, so verify never ran. Rerun with --trust-known to resume, "
            "or with --fresh to start over and verify it."
        )
    return None


def _route_after_verify(state: GraphState) -> str | list[Send]:
    known_file = KnownFile.model_validate(state["known_file"])
    if known_file.is_full_nonfiction_path:
        return [
            Send(
                NODE_DRAFT_CHAPTER,
                {
                    "known_file": state["known_file"],
                    "chapter_number": i + 1,
                    "chapter_title": title,
                },
            )
            for i, title in enumerate(known_file.chapters)
        ]
    return NODE_SYNTHESIZE


def format_chapter_progress(
    number: int, title: str, quality_flag: str | None, *, total_chapters: int | None = None
) -> str:
    """The one place "chapter drafted" progress wording is built — shared
    by _progress_messages (whole-book `generate`, one call per completed
    chapter) and cli.py's `generate-chapter` (single chapter, no sibling
    total known unless the caller passes one), so the two CLI paths can't
    drift into inconsistently-worded output for the same underlying event.
    `total_chapters=None` renders just the bare number, for a caller that
    doesn't have (or want to show) a "N/total" position.
    """
    position = f"{number}/{total_chapters}" if total_chapters is not None else str(number)
    flag = f" (flagged: {_preview(quality_flag)})" if quality_flag else ""
    return f"chapter {position} drafted: {title!r}{flag}"


# A flag carries the critique's whole feedback — the full text belongs in
# the book's warnings[], not in a progress line.
_FLAG_PREVIEW_CHARS = 100


def _preview(text: str) -> str:
    # One line: feedback with its own line breaks would otherwise spill
    # fragments that read as separate progress lines.
    text = " ".join(text.split())
    return text if len(text) <= _FLAG_PREVIEW_CHARS else text[:_FLAG_PREVIEW_CHARS].rstrip() + "…"


def _drafting_message(total_chapters: int, *, resuming: bool = False) -> str:
    chapters = f"{total_chapters} chapter{'s' if total_chapters != 1 else ''}"
    what = f"any not yet drafted of {chapters}" if resuming else chapters
    return f"drafting {what}, up to {settings.concurrency} at a time..."


def _verify_messages(reason: str, total_chapters: int) -> list[str]:
    # The report's first line is the verdict; any differences follow,
    # indented under it so they read as part of this stage.
    first, *rest = reason.splitlines() or [""]
    messages = [f"verify: {first}", *(f"  {line}" for line in rest)]
    if total_chapters:
        messages.append(_drafting_message(total_chapters))
    return messages


def _assemble_message(book: dict) -> str:
    # Flagged chapters aren't counted separately: each already has its own
    # entry in warnings[] (see draft._finalize).
    chapters = book.get("chapters") or []
    details = [f"{len(chapters)} chapters"] if chapters else []
    if warnings := book.get("warnings"):
        details.append(f"{len(warnings)} warning(s) — see the book's warnings")
    return "assemble: book finalized" + (f" ({', '.join(details)})" if details else "")


def _progress_messages(node_name: str, node_update: dict, total_chapters: int) -> list[str]:
    """Translates one node's return dict into zero or more human-readable
    progress lines. A node update is exactly what that node returned (see
    pipeline/nodes/*.py's `run`/`run_one` docstrings for each shape) —
    no new instrumentation needed, this just narrates data the pipeline
    already produces.
    """
    if node_name == NODE_VERIFY:
        return _verify_messages(node_update.get("verify_reason", "known-file confirmed"), total_chapters)
    if node_name == NODE_DRAFT_CHAPTER:
        return [
            format_chapter_progress(c["number"], c["title"], c.get("quality_flag"), total_chapters=total_chapters)
            for c in node_update.get("chapters", [])
        ]
    if node_name == NODE_SYNTHESIZE:
        warnings = node_update.get("warnings") or []
        note = f" — {len(warnings)} warning(s) noted" if warnings else ""
        return [f"synthesize: synopsis/tags/parts complete (parts: {node_update.get('parts_source')}){note}"]
    if node_name == NODE_ASSEMBLE:
        return [_assemble_message(node_update.get("book") or {})]
    return []


async def _stream_with_budget(
    graph: CompiledStateGraph,
    input_state: dict | None,
    config: RunnableConfig,
    *,
    total_chapters: int,
    on_progress: ProgressCallback | None,
) -> dict:
    """Streams node-by-node instead of a single `ainvoke`, purely to drive
    `on_progress` — the returned dict is unchanged from before (assemble's
    own return value, the only node that produces `book`), not a
    hand-rolled re-merge of the graph's per-node updates. `astream`'s
    "updates" mode still checkpoints exactly as `ainvoke` did (checkpointing
    is a StateGraph-engine concern, not tied to which call streams it), so
    the run-budget circuit breaker below behaves identically either way.
    """

    async def _consume() -> dict:
        book_result: dict | None = None
        async for update in graph.astream(input_state, config, stream_mode="updates"):
            for node_name, node_update in update.items():
                if on_progress:
                    for message in _progress_messages(node_name, node_update, total_chapters):
                        on_progress(message)
                if node_name == NODE_ASSEMBLE:
                    book_result = node_update
        if book_result is None:
            raise RuntimeError("graph run finished without reaching assemble — this is a bug in graph.py's routing")
        return book_result

    try:
        return await asyncio.wait_for(_consume(), timeout=settings.run_budget_seconds)
    except TimeoutError as exc:
        # A circuit breaker for a genuinely hung run, not a constraint
        # meant to bind on a normal one — see docs/blueprint.md's Run
        # budget section. Enforced in-process (asyncio.wait_for), not an
        # external process kill — checkpoints persist independently of
        # this cancellation, so nothing completed so far is lost:
        # rerunning the same command resumes rather than starting over.
        raise RunBudgetExceeded(
            f"generation exceeded the {settings.run_budget_seconds}s run budget. "
            "Already-completed work is checkpointed — rerun the same command to resume."
        ) from exc


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node(NODE_VERIFY, verify.run)
    graph.add_node(NODE_DRAFT_CHAPTER, draft.run_one)
    graph.add_node(NODE_SYNTHESIZE, synthesize.run)
    graph.add_node(NODE_ASSEMBLE, assemble.run)

    graph.add_edge(START, NODE_VERIFY)
    graph.add_conditional_edges(NODE_VERIFY, _route_after_verify, [NODE_DRAFT_CHAPTER, NODE_SYNTHESIZE])
    graph.add_edge(NODE_DRAFT_CHAPTER, NODE_SYNTHESIZE)
    graph.add_edge(NODE_SYNTHESIZE, NODE_ASSEMBLE)
    graph.add_edge(NODE_ASSEMBLE, END)

    return graph.compile(checkpointer=checkpointer)


async def run_whole_book(
    known_file: KnownFile,
    *,
    slug: str,
    trust_known: bool = False,
    fresh: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Book:
    """Runs (or resumes) the whole-book pipeline for `slug` — the book's
    one identity, shared by its known-file, its output and its checkpoint
    thread. See docs/blueprint.md's Orchestration section for the resume
    rules implemented below.
    """
    # Enforced here, not just by the CLI's own preflight_check call, since
    # run_whole_book is itself a public entrypoint per the module boundary
    # (docs/blueprint.md) — a caller other than this repo's CLI could invoke
    # it directly. Without this, a full-non-fiction known-file with empty
    # chapters silently completes with no `book` key in the result (the
    # Stage 2 fan-out dispatches zero Send()s and the graph never reaches
    # assemble) instead of failing with a clear message.
    ensure_ready(known_file)
    if not slug:
        raise ValueError("slug is required — it names the book's checkpoint thread")

    checkpoint_dir = os.path.dirname(settings.checkpoint_db_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    # Only the full non-fiction path drafts chapters one by one.
    total_chapters = len(known_file.chapters) if known_file.is_full_nonfiction_path else 0

    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": slug}}
        current = known_file.model_dump()

        saved = None if fresh else await checkpointer.aget_tuple(config)
        saved_values = saved.checkpoint["channel_values"] if saved else {}

        # None resumes the saved thread where it stopped; passing the input
        # state again would restart it from START instead — re-running
        # verify and every chapter, and appending a second copy of each
        # chapter through the `add` reducer.
        input_state: dict | None = {
            "known_file": current,
            "trust_known": trust_known,
            "chapters": [],
            "warnings": [],
        }
        if saved and saved_values.get("verified") and COMPLETED_STATE_KEY not in saved_values:
            if blocker := _resume_blocker(slug, saved_values, current, trust_known=trust_known):
                raise CheckpointMismatch(blocker)
            progress(f"resuming {slug!r} from its checkpoint")
            # Verify already ran, so its update (which announces drafting)
            # won't stream again.
            if total_chapters:
                progress(_drafting_message(total_chapters, resuming=True))
            input_state = None
        elif saved or fresh:
            # Nothing worth keeping: either verify never passed (only its
            # cheap search + model call has run — this is also what lets a
            # rerun with --trust-known take effect after verify failed), or
            # the run finished and its thread outlived a failed output write.
            # A finished book is never handed back from here; the caller
            # deletes the thread once the book is written.
            await checkpointer.adelete_thread(slug)
        if input_state is not None and not trust_known:
            progress("verify: checking the known-file against web search...")

        # Built once per run and threaded through every node via config — see verify.run/draft.run_one, which read
        # config["configurable"] — rather than each of the (potentially
        # many, for draft's per-chapter fan-out) node invocations
        # constructing its own client independently. Never touches the
        # checkpoint DB: config passed to astream() is per-run, not part of
        # the persisted state the checkpointer serializes.
        search_client = build_search_client()
        llm_client = llm.build_client()

        result = await _stream_with_budget(
            graph,
            input_state,
            {
                "configurable": {
                    "thread_id": slug,
                    "search_client": search_client,
                    "llm_client": llm_client,
                },
                "max_concurrency": settings.concurrency,
            },
            total_chapters=total_chapters,
            on_progress=on_progress,
        )
        return _finish(result["book"], known_file)
