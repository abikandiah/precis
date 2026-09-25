"""Builds the whole-book LangGraph pipeline and drives one run.

Orchestration is LangGraph specifically for checkpoint/resume — see
docs/blueprint.md's Orchestration section. The checkpoint DB must live on a
volume mounted into the generation container, not its ephemeral filesystem,
or resume across container restarts doesn't work (config.settings.checkpoint_db_path).
"""

from __future__ import annotations

import asyncio
import hashlib
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
from precis.pipeline.state import GraphState
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


def thread_id_for(known_file: KnownFile, *, trust_known: bool) -> str:
    """Deterministic run identity: the same known-file content and
    trust_known setting resolve to the same thread, so re-running
    `precis generate` unchanged resumes an interrupted run rather than
    starting over. trust_known is part of the identity, not just the
    known-file, because it changes what Stage 1 does — resuming a thread
    that already ran verify with the old value would otherwise silently
    keep the old behavior even though a new value was just requested.
    """
    canonical = known_file.model_dump_json() + f"|trust_known={trust_known}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
    flag = f" (flagged: {quality_flag})" if quality_flag else ""
    return f"chapter {position} drafted: {title!r}{flag}"


def _progress_messages(node_name: str, node_update: dict, total_chapters: int) -> list[str]:
    """Translates one node's return dict into zero or more human-readable
    progress lines. A node update is exactly what that node returned (see
    pipeline/nodes/*.py's `run`/`run_one` docstrings for each shape) —
    no new instrumentation needed, this just narrates data the pipeline
    already produces.
    """
    if node_name == NODE_VERIFY:
        return [f"verify: {node_update.get('verify_reason', 'known-file confirmed')}"]
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
        return ["assemble: book finalized"]
    return []


async def _stream_with_budget(
    graph: CompiledStateGraph,
    input_state: dict,
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
    trust_known: bool = False,
    fresh: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Book:
    # Enforced here, not just by the CLI's own preflight_check call, since
    # run_whole_book is itself a public entrypoint per the module boundary
    # (docs/blueprint.md) — a caller other than this repo's CLI could invoke
    # it directly. Without this, a full-non-fiction known-file with empty
    # chapters silently completes with no `book` key in the result (the
    # Stage 2 fan-out dispatches zero Send()s and the graph never reaches
    # assemble) instead of failing with a clear message.
    ensure_ready(known_file)

    checkpoint_dir = os.path.dirname(settings.checkpoint_db_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Built once per run and threaded through every node via config — see
    # verify.run/draft.run_one, which read config["configurable"] — rather
    # than each of the (potentially many, for draft's per-chapter fan-out)
    # node invocations constructing its own client independently. Never
    # touches the checkpoint DB: config passed to astream() is per-run,
    # not part of the persisted state the checkpointer serializes.
    search_client = build_search_client()
    llm_client = llm.build_client()

    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = thread_id_for(known_file, trust_known=trust_known)

        if fresh:
            await checkpointer.adelete_thread(thread_id)

        result = await _stream_with_budget(
            graph,
            {
                "known_file": known_file.model_dump(),
                "trust_known": trust_known,
                "chapters": [],
                "warnings": [],
            },
            {
                "configurable": {
                    "thread_id": thread_id,
                    "search_client": search_client,
                    "llm_client": llm_client,
                },
                "max_concurrency": settings.concurrency,
            },
            total_chapters=len(known_file.chapters),
            on_progress=on_progress,
        )
        return Book.model_validate(result["book"])
