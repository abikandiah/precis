"""LangGraph state schema for the whole-book pipeline.

Every field is a plain dict/list/primitive (Pydantic models `.model_dump()`d
before entering state, reconstructed by whichever node needs the real
model), never a live Pydantic instance. LangGraph's checkpoint serializer
doesn't natively support arbitrary classes — passing a KnownFile/Chapter/
Book instance through state works today but logs a deprecation warning that
it will be a hard error in a future langgraph-checkpoint release. Plain
JSON-safe values sidestep that entirely and don't depend on checkpointer
serialization internals at all.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict


class GraphState(TypedDict, total=False):
    # Set once at graph invocation. KnownFile.model_dump().
    known_file: dict
    trust_known: bool

    # Stage 1 output.
    verified: bool

    # Stage 2 fan-out: per-branch input, delivered via Send() in graph.py.
    # Only visible inside a single draft_one_chapter invocation — never
    # part of the graph's overall merged state.
    chapter_number: int
    chapter_title: str

    # Stage 2 fan-out: accumulated across branches. The `add` reducer means
    # each branch's {"chapters": [one_chapter_dict]} return concatenates
    # rather than overwrites — this is what makes a resumed run redo only
    # the branches that hadn't finished, not the whole batch. See
    # docs/blueprint.md's Orchestration section. Each entry is a
    # Chapter.model_dump().
    chapters: Annotated[list[dict], add]
    warnings: Annotated[list[str], add]

    # Stage 3 output.
    synopsis: str
    one_line_takeaway: str
    tags: list[str]
    parts: list[dict]  # Part.model_dump() each
    parts_source: Literal["known", "generated"]
    key_claims_for_review: list[dict]  # KeyClaim.model_dump() each

    # Stage 4 output. Book.model_dump().
    book: dict
