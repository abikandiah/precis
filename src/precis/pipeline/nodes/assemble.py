"""Stage 4 — assemble + validate the whole book. Full schema validation
including cross-field checks (parts referencing real chapter numbers,
enforced by Book's model_validator in schema.py). One repair-and-retry pass
if invalid; if that also fails, generation fails outright with no output
file. See docs/blueprint.md's Pipeline stages, Stage 4.

Not yet implemented: needs the repair-and-retry pass for a Book that fails
validation on the first assembly attempt.
"""

from __future__ import annotations

from precis.pipeline.state import GraphState


async def run(state: GraphState) -> dict:
    raise NotImplementedError("Stage 4 assemble: not yet implemented")
