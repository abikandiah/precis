"""Stage 1 — verify. One search + one model critique against the
known-file's isbn/chapters, scoped narrowly to edition/chapter-list
correctness. Fails fast before any per-chapter work runs. Skippable via
`trust_known`. See docs/blueprint.md's Pipeline stages, Stage 1.

Not yet implemented: needs the search client and the critique prompt.
"""

from __future__ import annotations

from precis.pipeline.state import GraphState


async def run(state: GraphState) -> dict:
    if state.get("trust_known"):
        return {"verified": True}
    raise NotImplementedError(
        "Stage 1 verify: search + critique against known_file isbn/chapters not yet implemented"
    )
