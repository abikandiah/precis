"""Stage 2 — draft chapters (non-fiction full path only). One invocation of
`run_one` per chapter, fanned out via Send() in pipeline/graph.py with
bounded concurrency (config.settings.concurrency). Per chapter:
search-ground, draft key_points + core_claim, critique against search
results, repair-and-retry up to a small local cap for content-quality
failures, falling back to the last schema-valid candidate (sets
`quality_flag`) if critique still fails once retries are exhausted.
Transient/technical failures (rate limits, 5xx) are handled separately by
the LLM client's own built-in retry (see llm.complete) and never touch this
retry budget or `quality_flag` — see docs/blueprint.md's Stage 2 section.

Not yet implemented: needs the search-ground query, the draft prompt, and
the critique prompt/redundancy check.

Return contract (once implemented): `{"chapters": [chapter_dict]}` — a
single-element list, matching the `add` reducer on GraphState.chapters so
N parallel branches concatenate rather than overwrite. `cli.py`'s
`generate-chapter` command calls this function directly (no graph, no
checkpointing — a one-off chapter regen doesn't need either) and expects
the same single-element-list shape.
"""

from __future__ import annotations

from precis.pipeline.state import GraphState


async def run_one(state: GraphState) -> dict:
    raise NotImplementedError(
        f"Stage 2 draft: chapter {state['chapter_number']} "
        f"({state['chapter_title']!r}) drafting/critique loop not yet implemented"
    )
