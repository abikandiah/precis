"""Pydantic models for precis's JSON-file boundary.

Two independent shapes: the known-file (input, written by phase 1 and the
reader, read by phase 3) and the book/chapter output (written by phase 3).
See docs/blueprint.md for the design behind every field here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

SCHEMA_VERSION = "1"

PLACEHOLDER = "TODO: fill in by hand"

# Closed vocabularies for `tags`, curated from BISAC Subject Headings (the
# taxonomy the publishing industry itself uses) rather than left to whatever
# free-form label the model invents per book. Mirrors book-keeper's own
# schema.ts exactly — that's the consumer whose discriminated-union publish
# check requires this closed vocabulary, so the two lists must not drift.
# Split in two because a novel's genres and a non-fiction book's subjects are
# genuinely different vocabularies, not because they share a spectrum.
NONFICTION_TAGS: tuple[str, ...] = (
    "history",
    "philosophy",
    "psychology",
    "science",
    "mathematics",
    "technology",
    "engineering",
    "business",
    "economics",
    "finance",
    "politics",
    "sociology",
    "anthropology",
    "biography-memoir",
    "religion-spirituality",
    "health-wellness",
    "medicine",
    "nature-environment",
    "true-crime",
    "education",
    "art-design",
    "travel",
    "law",
    "humor",
    "reference",
)

FICTION_TAGS: tuple[str, ...] = (
    "fiction-literary",
    "science-fiction",
    "fantasy",
    "mystery-thriller",
    "horror",
    "romance",
    "drama",
    "adventure",
    "dystopian",
    "coming-of-age",
    "poetry",
    "short-stories",
)


def tags_for_kind(kind: Literal["fiction", "non-fiction"]) -> tuple[str, ...]:
    return NONFICTION_TAGS if kind == "non-fiction" else FICTION_TAGS


def invalid_tags(tags: list[str], kind: Literal["fiction", "non-fiction"]) -> list[str]:
    """Shared "are these tags from the right closed vocabulary" check — used
    by both Synthesis's field_validator (synthesize.py, retryable at Stage 3)
    and Book's own field_validator below (the final safety net at Stage 4),
    same reasoning as invalid_chapter_numbers below.
    """
    allowed = tags_for_kind(kind)
    return [t for t in tags if t not in allowed]


def validate_tags(
    tags: list[str], kind: Literal["fiction", "non-fiction"] | None
) -> None:
    """Raises ValueError if `tags` isn't a valid tags list: no duplicates,
    and — when `kind` is known — every entry drawn from the closed
    vocabulary for that kind. Shared by Book's and Synthesis's field
    validators (see synthesize.py) so the two enforce identical rules
    instead of each hand-rolling its own copy, same reasoning as
    invalid_chapter_numbers below. `kind` is only None on Synthesis's
    context-free validation path (e.g. a caller validating it directly
    without a known_file to hand) — vocabulary membership is skipped then,
    since there's nothing to check it against; the duplicate check still
    always runs.
    """
    if len(set(tags)) != len(tags):
        raise ValueError(f"tags must not repeat: {tags!r}")
    if kind is not None:
        bad = invalid_tags(tags, kind)
        if bad:
            raise ValueError(
                f"tags {bad!r} aren't in the closed {kind} vocabulary: {sorted(tags_for_kind(kind))!r}"
            )


def invalid_chapter_numbers(
    chapter_numbers: list[int] | None, valid_numbers: set[int]
) -> list[int]:
    """The shared "does this part reference a real chapter" check — used by
    both KnownFile's preflight (known_file.py) and Book's own model_validator
    below, so the two can't drift into different definitions of "valid"
    (e.g. a contiguous-range assumption in one and not the other).
    """
    return [n for n in chapter_numbers or [] if n not in valid_numbers]


class KnownPart(BaseModel):
    """Known-file input mirror of Part, minus `summary` — writing the
    summary is still Stage 3's job. `chapters` only means something against
    a known chapter list, so it's meaningful for the full non-fiction path
    only; narrative non-fiction has no known chapters to bind against and
    must leave it unset. Never meaningful for fiction — see preflight_check
    in known_file.py.
    """

    title: str
    chapters: list[int] | None = None


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
    kind: Literal["fiction", "non-fiction"]
    narrative: bool = False
    notes: str | None = None
    parts: list[KnownPart] = Field(default_factory=list)
    # Last, after `parts` — chapters is the field most likely to need
    # hand-editing (or be pages long, for a full non-fiction book), so it's
    # the last thing the reader scrolls past when hand-editing the file, not
    # the first thing rendered between the short bibliographic fields.
    chapters: list[str] = Field(default_factory=list)

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
    chapters: list[int] | None = None


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


class TagVocabulary(BaseModel):
    """Export-only reflection of the closed tag vocabulary above — never
    read back in, unlike KnownFile/Book, which are precis's actual
    JSON-file boundary. `precis tags` (cli.py) is how a consumer repo
    (book-keeper's `schema.ts`, or any future one) pulls this instead of
    hand-copying NONFICTION_TAGS/FICTION_TAGS verbatim and risking drift —
    see the comment on those two tuples above.
    """

    schema_version: str = SCHEMA_VERSION
    non_fiction_tags: tuple[str, ...] = tags_for_kind("non-fiction")
    fiction_tags: tuple[str, ...] = tags_for_kind("fiction")


class Book(BaseModel):
    schema_version: str = SCHEMA_VERSION

    title: str
    author: str
    year: int | None = None
    isbn: str
    page_count: int | None = None
    kind: Literal["fiction", "non-fiction"]
    narrative: bool = False
    one_line_takeaway: str
    synopsis: str
    tags: list[str] = Field(min_length=2, max_length=4)
    parts: list[Part] | None = None
    parts_source: Literal["known", "generated"] | None = None
    """Whether `parts` is a pass-through of the reader-supplied known-file
    structure or Stage 3's own invention — lets a consumer avoid presenting
    an AI-invented grouping as the book's real published structure.
    """
    reader_notes: str | None = None
    warnings: list[str] = Field(default_factory=list)

    # Non-fiction, full path only. Both present together or both absent —
    # see the model_validator below. The min_lengths only apply when a list
    # is actually given (None still bypasses them) — without that, `chapters=[]`
    # paired with `key_claims_for_review=[]` would satisfy the "both present
    # together" check below while representing a nonsensical zero-chapter
    # "full path" book. key_claims_for_review's min of 3 (not 1, like
    # chapters) mirrors book-keeper's own schema.ts, which treats a 1-2 claim
    # deck as too thin to be worth a review pass.
    chapters: list[Chapter] | None = Field(default=None, min_length=1)
    key_claims_for_review: list[KeyClaim] | None = Field(default=None, min_length=3)

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, tags: list[str], info: ValidationInfo) -> list[str]:
        """A field_validator (not folded into the whole-model check below)
        specifically so a tags failure gets its own `("tags",)` error
        location instead of the empty `loc` a model_validator's raise would
        produce — assemble.py's `_is_repairable` distinguishes "repairable"
        fields by `loc[0]`, and an empty loc gets bucketed with the
        parts/chapter-number check below, which would wrongly make a
        known-parts-path book's tags failure look like an unrepairable
        parts problem. `info.data.get("kind")` is only absent if `kind`
        itself already failed validation, in which case there's nothing
        meaningful to check membership against anyway.
        """
        validate_tags(tags, info.data.get("kind"))
        return tags

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

        if not self.parts:
            return self

        if self.chapters is not None:
            known_numbers = {c.number for c in self.chapters}
            for part in self.parts:
                for n in invalid_chapter_numbers(part.chapters, known_numbers):
                    raise ValueError(
                        f"part {part.title!r} references chapter "
                        f"number {n}, which doesn't exist in chapters"
                    )
        else:
            # No `chapters` array on this path (fiction/narrative non-fiction)
            # for a part's `chapters` to mean anything against — enforced
            # here, at the one place every Book gets validated, rather than
            # trusted from whichever producer built it (Stage 3's own
            # generated-parts path, Stage 4's repair pass, or any future
            # caller); a producer-side strip can't cover callers it doesn't
            # know about.
            for part in self.parts:
                part.chapters = None
        return self
