"""Pydantic models for precis's JSON-file boundary: the known-file (input,
bibliographic facts the reader checks) and the book (output, whole-book
notes). See docs/blueprint.md for the design behind every field here.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "4"

# What the notes were written from: "full" from the reader's own copy of the
# book (its book_file), read whole; "overview" from search alone.
Depth = Literal["full", "overview"]

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
    # The reader's own copy of the book (.epub, .pdf or .txt), relative to
    # the known-file; read whole as part of the research (book_file.py).
    book_file: str | None = None
    # Non-fiction built around its own numbered list (48 laws, 7 habits):
    # full notes give one idea per item, past MAX_IDEAS. Set by the reader,
    # not judged by the model, which would take any book with numbered
    # chapters for one.
    numbered_list: bool = False

    @model_validator(mode="after")
    def _list_is_nonfiction(self) -> KnownFile:
        if self.numbered_list and self.kind == "fiction":
            raise ValueError("numbered_list is for non-fiction built around its own list, not a novel")
        return self

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
        description="The idea itself (or the theme as the setup raises it), in 1-3 sentences."
    )
    evidence: str = Field(
        default="",
        description="The study, story or example the author uses (non-fiction), or the characters and "
        "situations that carry the theme (fiction). Empty when you have no specific one — never a general "
        "statement in its place.",
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


# An overview's ideas have no evidence: it states the ideas, not the book's
# own examples, which research from the web can't be trusted for.
FULL_ONLY_IDEA_FIELDS = ("evidence",)

# The most ideas notes have: a book's major points, not a chapter-by-chapter
# account (Making Embedded Systems, read whole, got 62). A ceiling, never a
# target — the prompts say so, since ranges made the model pad to their top.
# Only full notes on a book built around its own numbered list go past it.
MAX_IDEAS = 12


def stripped_schema(
    schema: dict[str, Any], *, fields: tuple[str, ...] = (), idea_fields: tuple[str, ...] = ()
) -> dict[str, Any]:
    """A tool model's JSON schema without `fields` (top-level) and
    `idea_fields` (in its Idea definition): what the notes of this kind and
    depth can't have, so the model is never offered them — rather than
    offered them and told to leave them empty, which it won't always do,
    and paying for output that's thrown away. Raises if a field or the Idea
    definition isn't where pydantic is expected to put it, so a change there
    can't silently start offering them again.
    """
    targets = [(schema, fields)]
    if idea_fields:
        if (idea := schema.get("$defs", {}).get("Idea")) is None:
            raise RuntimeError("the tool schema has no Idea definition to strip fields from")
        targets.append((idea, idea_fields))
    for definition, names in targets:
        for name in names:
            if definition.get("properties", {}).pop(name, None) is None:
                raise RuntimeError(f"the tool schema has no {name!r} field to strip")
            if name in definition.get("required", []):
                definition["required"].remove(name)
    return schema


def has_deck(kind: Literal["fiction", "non-fiction"], depth: Depth) -> bool:
    """Only full non-fiction notes have a review deck: fiction isn't read to
    retain claims, and an overview's would rest on research that can't back
    them — flashcards on shaky claims teach the wrong thing.
    """
    return kind == "non-fiction" and depth == "full"


def notes_shape_problems(
    kind: Literal["fiction", "non-fiction"], depth: Depth, ideas: list[Idea], key_claims: list[KeyClaim] | None
) -> list[str]:
    """What makes a book's notes unusable: no ideas, full non-fiction notes
    with no review deck, or any others with one. Shared by `Book` and the
    write and review calls' response models, so it's a retryable validation
    failure at the call and the final book can't disagree with it. The
    count's ceiling is idea_count_problems'.
    """
    problems = []
    if not ideas:
        problems.append("the notes need at least one idea")
    if has_deck(kind, depth):
        if not key_claims:
            problems.append("full non-fiction notes need key_claims_for_review")
    elif key_claims:
        problems.append("only full non-fiction notes have key_claims_for_review")
    return problems


def idea_count_problems(ideas: list[Idea], depth: Depth, *, numbered_list: bool) -> list[str]:
    """More than MAX_IDEAS ideas, unless they're full notes following the
    book's own numbered list. Checked at the write and review calls, which
    know the known-file — a retryable failure there — not by `Book`, which
    doesn't.
    """
    if len(ideas) <= MAX_IDEAS or (numbered_list and depth == "full"):
        return []
    problem = (
        f"{len(ideas)} ideas — at most {MAX_IDEAS}: keep the book's major points, merging ones a reader would "
        "recall as one and dropping the minor ones"
    )
    return [problem]


def deck_coverage_warnings(
    kind: Literal["fiction", "non-fiction"], depth: Depth, ideas: list[Idea], key_claims: list[KeyClaim] | None
) -> list[str]:
    """A review deck covering under half the ideas. Key claims are one per
    idea a reader needs to remember, so a deck this thin (2 claims for 25 ideas) means most of the notes can't be
    reviewed. A warning, not a retry.
    """
    if not has_deck(kind, depth) or len(key_claims or []) * 2 >= len(ideas):
        return []
    return [f"{len(key_claims or [])} key claims for {len(ideas)} ideas — the review deck covers under half the notes"]


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
    reading it — full, from the reader's own copy read whole, or an overview
    from search: headline ideas without evidence, and no review deck.
    Fiction's are spoiler-safe and carry no review deck; read whole, a
    novel's ending is in `resolution`, which the page shows only behind a
    spoiler warning.
    """

    schema_version: str = SCHEMA_VERSION

    title: str
    author: str
    year: int | None = None
    isbn: str
    page_count: int | None = None
    kind: Literal["fiction", "non-fiction"]
    depth: Depth
    one_line_takeaway: str
    synopsis: str
    ideas: list[Idea]
    resolution: str | None = None
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
        if problems := notes_shape_problems(self.kind, self.depth, self.ideas, self.key_claims_for_review):
            raise ValueError("; ".join(problems))
        if self.resolution is not None and (self.kind, self.depth) != ("fiction", "full"):
            raise ValueError("only fiction read whole (full depth) has a resolution")
        # An overview's evidence is cleared rather than sent back, as a guess:
        # the book's own examples, which search research can't be trusted for.
        if self.depth == "overview":
            self.ideas = [i.model_copy(update={"evidence": ""}) if i.evidence else i for i in self.ideas]
        return self
