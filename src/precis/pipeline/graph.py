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
                "draft_one_chapter",
                {
                    "known_file": state["known_file"],
                    "chapter_number": i + 1,
                    "chapter_title": title,
                },
            )
            for i, title in enumerate(known_file.chapters)
        ]
    return "synthesize"


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("verify", verify.run)
    graph.add_node("draft_one_chapter", draft.run_one)
    graph.add_node("synthesize", synthesize.run)
    graph.add_node("assemble", assemble.run)

    graph.add_edge(START, "verify")
    graph.add_conditional_edges("verify", _route_after_verify, ["draft_one_chapter", "synthesize"])
    graph.add_edge("draft_one_chapter", "synthesize")
    graph.add_edge("synthesize", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile(checkpointer=checkpointer)


async def run_whole_book(known_file: KnownFile, *, trust_known: bool = False, fresh: bool = False) -> Book:
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
    # touches the checkpoint DB: config passed to ainvoke() is per-run,
    # not part of the persisted state the checkpointer serializes.
    search_client = build_search_client()
    llm_client = llm.build_client()

    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = thread_id_for(known_file, trust_known=trust_known)

        if fresh:
            await checkpointer.adelete_thread(thread_id)

        try:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "known_file": known_file.model_dump(),
                        "trust_known": trust_known,
                        "chapters": [],
                        "warnings": [],
                    },
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "search_client": search_client,
                            "llm_client": llm_client,
                        },
                        "max_concurrency": settings.concurrency,
                    },
                ),
                timeout=settings.run_budget_seconds,
            )
        except TimeoutError as exc:
            # A circuit breaker for a genuinely hung run, not a constraint
            # meant to bind on a normal one — see docs/blueprint.md's Run
            # budget section. Checkpoints persist independently of this
            # process being killed, so nothing completed so far is lost:
            # rerunning the same command resumes rather than starting over.
            raise TimeoutError(
                f"generation exceeded the {settings.run_budget_seconds}s run budget. "
                "Already-completed work is checkpointed — rerun the same command to resume."
            ) from exc
        return Book.model_validate(result["book"])
