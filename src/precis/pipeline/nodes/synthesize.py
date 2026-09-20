"""Stage 3 — synthesize. synopsis, one_line_takeaway, tags,
key_claims_for_review (non-fiction full path), parts. Fed by the finished
chapters (where they exist) plus its own dedicated themes-oriented search —
deliberately not Stage 1's edition/chapter-list-oriented search results.
Runs after chapter drafting so claims/parts are grounded in real,
already-critiqued content. See docs/blueprint.md's Pipeline stages, Stage 3.

Not yet implemented: needs the themes-oriented search query and the
synthesis prompt(s), and branches on fiction/narrative vs. full non-fiction
for what parts/claims actually mean.
"""

from __future__ import annotations

from precis.pipeline.state import GraphState


async def run(state: GraphState) -> dict:
    raise NotImplementedError("Stage 3 synthesize: not yet implemented")
