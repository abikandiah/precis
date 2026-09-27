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


# How many ideas a book's notes carry: key ideas for non-fiction, themes for
# fiction. The count scales with how much a book argues, within these.
IDEA_LIMITS: dict[str, tuple[int, int]] = {"non-fiction": (5, 12), "fiction": (3, 6)}
# Non-fiction's review deck; fiction has none.
KEY_CLAIM_LIMITS = (5, 15)

SOURCE_ID = re.compile(r"S[1-9][0-9]*")


class Idea(BaseModel):
    """A key idea (non-fiction) or theme (fiction)."""

    title: str = Field(description="The idea or theme, named as the book names it where it has a name.")
    summary: str = Field(description="2-4 sentences stating the idea itself (or how the book develops the theme).")
    evidence: str = Field(
        description="The study, story, example or figure the author uses (non-fiction), or the characters and "
        "situations that carry the theme (fiction)."
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
    """The per-kind counts a book's notes must meet. Shared by `Book` and
    the write call's own response model (write.py), so a miscount is a
    retryable validation failure at the call, and the final book can't
    disagree with it.
    """
    problems = []
    low, high = IDEA_LIMITS[kind]
    if not low <= len(ideas) <= high:
        noun = "themes" if kind == "fiction" else "key ideas"
        problems.append(f"a {kind} book needs {low}-{high} ideas ({noun}), got {len(ideas)}")
    if kind == "fiction":
        if key_claims:
            problems.append("fiction has no key_claims_for_review")
    else:
        low, high = KEY_CLAIM_LIMITS
        if not low <= len(key_claims or []) <= high:
            problems.append(f"non-fiction needs {low}-{high} key_claims_for_review, got {len(key_claims or [])}")
    return problems


def citation_problems(ideas: list[Idea], source_ids: set[str]) -> list[str]:
    """Ideas citing a source the research doesn't have. Shared by the write
    and review calls' response models.
    """
    return [
        f"idea {idea.title!r} cites {unknown!r}, which aren't research sources"
        for idea in ideas
        if (unknown := [s for s in idea.sources if s not in source_ids])
    ]


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
