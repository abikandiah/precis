"""Pydantic models for precis's JSON-file boundary.

Two independent shapes: the known-file (input, written by phase 1 and the
reader, read by phase 3) and the book/chapter output (written by phase 3).
See docs/blueprint.md for the design behind every field here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1"

PLACEHOLDER = "TODO: fill in by hand"


def invalid_chapter_numbers(chapter_numbers: list[int] | None, valid_numbers: set[int]) -> list[int]:
    """The shared "does this part reference a real chapter" check — used by
    both KnownFile's preflight (known_file.py) and Book's own model_validator
    below, so the two can't drift into different definitions of "valid"
    (e.g. a contiguous-range assumption in one and not the other).
    """
    return [n for n in chapter_numbers or [] if n not in valid_numbers]


class KnownPart(BaseModel):
    """Known-file input mirror of Part, minus `summary` — writing the
    summary is still Stage 3's job. `chapter_numbers` only means something
    against a known chapter list, so it's meaningful for the full
    non-fiction path only; narrative non-fiction has no known chapters to
    bind against and must leave it unset. Never meaningful for fiction —
    see preflight_check in known_file.py.
    """

    title: str
    chapter_numbers: list[int] | None = None


class KnownFile(BaseModel):
    """Phase 1/2 input. Deliberately permissive at the schema level —
    phase 1 writes one with empty `chapters`, and the reader fills it in by
    hand afterward. Readiness for generation is a separate, later gate (the
    phase-2 structural preflight check in known_file.py), not enforced here.
    """

    isbn: str
    title: str | None = None
    author: str | None = None
    year: int | None = None
    page_count: int | None = None
    chapters: list[str] = Field(default_factory=list)
    kind: Literal["fiction", "non-fiction"]
    narrative: bool = False
    notes: str | None = None
    parts: list[KnownPart] = Field(default_factory=list)

    @property
    def is_full_nonfiction_path(self) -> bool:
        """Gates both phase-2's chapters-required check (known_file.py) and
        the pipeline's Stage 2 fan-out routing (pipeline/graph.py) — one
        definition so the two can't silently drift apart.
        """
        return self.kind == "non-fiction" and not self.narrative

    @property
    def has_title(self) -> bool:
        """False for both "never filled in" (None) and "phase 1 couldn't
        find it, still a placeholder" — one definition so consumers (e.g.
        the known-file filename in cli.py) don't each reimplement the
        placeholder check.
        """
        return bool(self.title) and self.title != PLACEHOLDER


class Part(BaseModel):
    """Shared shape for both branches; the difference is in the prompt
    behind it (structural grouping vs. spoiler-safe beats), not the schema.
    """

    title: str
    summary: str
    chapter_numbers: list[int] | None = None


class Chapter(BaseModel):
    number: int
    title: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    core_claim: str
    quality_flag: str | None = None
    """Set only when Stage 2 fell back to the last schema-valid candidate
    after critique kept failing past the retry cap — a short reason, not a
    bare bool. Absent (not a falsy value) on a clean pass.
    """


class KeyClaim(BaseModel):
    prompt: str
    answer: str


class Book(BaseModel):
    schema_version: str = SCHEMA_VERSION

    title: str
    author: str
    year: int | None = None
    isbn: str
    page_count: int | None = None
    one_line_takeaway: str
    synopsis: str
    tags: list[str]
    parts: list[Part] | None = None
    parts_source: Literal["known", "generated"] | None = None
    """Whether `parts` is a pass-through of the reader-supplied known-file
    structure or Stage 3's own invention — lets a consumer avoid presenting
    an AI-invented grouping as the book's real published structure.
    """
    reader_notes: str | None = None
    warnings: list[str] = Field(default_factory=list)

    # Non-fiction, full path only. Both present together or both absent —
    # see the model_validator below. min_length=1 only applies when a list
    # is actually given (None still bypasses it) — without it, `chapters=[]`
    # paired with `key_claims_for_review=[]` would satisfy the "both present
    # together" check below while representing a nonsensical zero-chapter
    # "full path" book.
    chapters: list[Chapter] | None = Field(default=None, min_length=1)
    key_claims_for_review: list[KeyClaim] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_chapter_coupling_and_parts_refs(self) -> Book:
        has_chapters = self.chapters is not None
        has_claims = self.key_claims_for_review is not None
        if has_chapters != has_claims:
            raise ValueError(
                "chapters and key_claims_for_review must be present or "
                "absent together — this is the non-fiction full-path pair, "
                "fiction/narrative non-fiction has neither"
            )

        if self.parts and self.chapters is not None:
            known_numbers = {c.number for c in self.chapters}
            for part in self.parts:
                for n in invalid_chapter_numbers(part.chapter_numbers, known_numbers):
                    raise ValueError(
                        f"part {part.title!r} references chapter "
                        f"number {n}, which doesn't exist in chapters"
                    )
        return self
