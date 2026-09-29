"""Pydantic models for precis's JSON-file boundary: the known-file (input,
bibliographic facts the reader checks) and the book (output, whole-book
notes). See docs/blueprint.md for the design behind every field here.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "2"

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


# 2-4 tags from the closed vocabulary. One definition for every model that
# carries tags, so the write call's response model can't accept what the
# final book then rejects.
Tags = Annotated[list[str], Field(min_length=2, max_length=4)]


def tags_for_kind(kind: Literal["fiction", "non-fiction"]) -> tuple[str, ...]:
    return NONFICTION_TAGS if kind == "non-fiction" else FICTION_TAGS


def validate_tags(tags: list[str], kind: Literal["fiction", "non-fiction"] | None) -> None:
    """Raises ValueError unless `tags` has no duplicates and — when `kind`
    is known — every entry is from that kind's closed vocabulary. Shared by
    Book and the write call's response model so the two enforce one rule.
    """
    if len(set(tags)) != len(tags):
        raise ValueError(f"tags must not repeat: {tags!r}")
    if kind is not None and (bad := [t for t in tags if t not in tags_for_kind(kind)]):
        raise ValueError(f"tags {bad!r} aren't in the closed {kind} vocabulary: {sorted(tags_for_kind(kind))!r}")


class KnownFile(BaseModel):
    """The input: bibliographic facts, looked up from Open Library by
    `create-known-file` and checked by the reader. Permissive about values
    — a lookup that missed leaves placeholders; readiness is
    known_file.preflight_check's job — but not about fields: a misspelled
    or unknown one (say, `note`) is an error, not silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    isbn: str
    title: str | None = None
    author: str | None = None
    year: int | None = None
    page_count: int | None = None
    kind: Literal["fiction", "non-fiction"]
    # A weighting signal for the notes, never quoted into them; carried
    # verbatim into the book's `reader_notes`.
    notes: str | None = None

    @property
    def has_title(self) -> bool:
        """False for both "never filled in" and a leftover placeholder."""
        return bool(self.title) and self.title != PLACEHOLDER

    @property
    def has_author(self) -> bool:
        return bool(self.author) and self.author != PLACEHOLDER


class KeyClaim(BaseModel):
    prompt: str
    answer: str


class TagVocabulary(BaseModel):
    """Export-only reflection of the closed tag vocabulary — `precis tags`
    prints it so a consumer repo (book-keeper's `schema.ts`) can sync its
    own copy instead of hand-copying the tuples above and drifting.
    """

    schema_version: str = SCHEMA_VERSION
    non_fiction_tags: tuple[str, ...] = tags_for_kind("non-fiction")
    fiction_tags: tuple[str, ...] = tags_for_kind("fiction")


SOURCE_ID = re.compile(r"S[1-9][0-9]*")


class Idea(BaseModel):
    """A key idea (non-fiction) or theme (fiction)."""

    title: str = Field(description="The idea or theme, named as the book names it where it has a name.")
    summary: str = Field(
        description="The idea itself (or the theme as the setup raises it), in as few sentences as it needs: one "
        "for a simple rule, more for an argument with steps."
    )
    evidence: str = Field(
        description="Briefly, the study, story, example or figure the author uses (non-fiction), or the "
        "characters and situations that carry the theme (fiction)."
    )
    sources: list[str] = Field(
        default_factory=list,
        description='IDs of the research sources that support this idea, e.g. ["S2", "S5"]; empty when it rests '
        "on your own knowledge of the book.",
    )

    @field_validator("sources")
    @classmethod
    def _check_source_ids(cls, sources: list[str]) -> list[str]:
        if bad := [s for s in sources if not SOURCE_ID.fullmatch(s)]:
            raise ValueError(f"source IDs look like S1, S2, …; got {bad!r}")
        return sources


def notes_shape_problems(
    kind: Literal["fiction", "non-fiction"], ideas: list[Idea], key_claims: list[KeyClaim] | None
) -> list[str]:
    """What makes a book's notes unusable: no ideas, a non-fiction book
    with no review deck, or fiction with one. Shared by `Book` and the
    write and review calls' response models, so it's a retryable validation
    failure at the call and the final book can't disagree with it. There's
    no count to meet: a book has as many ideas as it makes.
    """
    problems = []
    if not ideas:
        problems.append("the notes need at least one idea")
    if kind == "fiction":
        if key_claims:
            problems.append("fiction has no key_claims_for_review")
    elif not key_claims:
        problems.append("non-fiction needs key_claims_for_review")
    return problems


def without_unknown_sources(idea: Idea, source_ids: set[str]) -> tuple[Idea, list[str]]:
    """The idea without citations of sources the research doesn't have, and
    those it dropped. Shared by the write and review calls' response models,
    which both leave a bad citation out rather than send the call back: a
    retry that repeats it would lose the whole call.
    """
    if not (unknown := [s for s in idea.sources if s not in source_ids]):
        return idea, []
    return idea.model_copy(update={"sources": [s for s in idea.sources if s in source_ids]}), unknown


class Book(BaseModel):
    """precis's output: whole-book notes for recalling a book after
    reading it. Fiction's are spoiler-safe and carry no review deck.
    """

    schema_version: str = SCHEMA_VERSION

    title: str
    author: str
    year: int | None = None
    isbn: str
    page_count: int | None = None
    kind: Literal["fiction", "non-fiction"]
    one_line_takeaway: str
    synopsis: str
    ideas: list[Idea]
    key_claims_for_review: list[KeyClaim] | None = None
    tags: Tags
    reader_notes: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, tags: list[str], info: ValidationInfo) -> list[str]:
        validate_tags(tags, info.data.get("kind"))
        return tags

    @model_validator(mode="after")
    def _check_shape(self) -> Book:
        if problems := notes_shape_problems(self.kind, self.ideas, self.key_claims_for_review):
            raise ValueError("; ".join(problems))
        return self
