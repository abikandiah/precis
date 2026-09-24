"""Stage 4 — assemble + validate the whole book. Full schema validation
including cross-field checks (parts referencing real chapter numbers,
enforced by Book's model_validator in schema.py). One repair-and-retry pass
if invalid; if that also fails, generation fails outright with no output
file. See docs/blueprint.md's Pipeline stages, Stage 4.

No search here — unlike Stages 1-3, this stage doesn't ground anything
against the web, it only assembles already-produced pieces and validates/
repairs structure. The repair pass reuses Stage 3's own response models
(Synthesis/SynthesisWithClaims) rather than redefining near-duplicates,
since it's repairing exactly the fields Stage 3 produced.

That "exactly the fields Stage 3 produced" limit is load-bearing: the
repair pass can only ever change synopsis/one_line_takeaway/tags/parts/
key_claims_for_review, because that's all Synthesis/SynthesisWithClaims
carry. A validation failure on any other field (title, chapters, or the
chapters/key_claims_for_review coupling check) can never be fixed by
asking the model to revise its synthesis — attempting repair anyway would
just reproduce the identical error after wasting an LLM call, which is
exactly what happened before _is_repairable was added below.

`parts` is repairable only when parts_source is "generated". When it's
"known", `parts` is a pass-through of the reader-supplied known-file
structure, already guaranteed valid by known_file.py's preflight check — a
validation error there means something upstream is actually broken, and
letting generic repair "fix" it would silently replace known structure
with an invented one while parts_source still claimed "known". Two
independent guards enforce this: `_is_repairable` skips repair outright
for a `parts`-shaped error, and `_apply_repair` additionally never
overwrites `parts` when parts_source is "known" regardless of which
field's error triggered the repair — so a repair prompted by some other
field can never smuggle invented parts in under a "known" label either.
"""

from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI
from pydantic import ValidationError

from precis import llm
from precis.pipeline.nodes.common import book_header, resolve_llm_client
from precis.pipeline.nodes.synthesize import Synthesis, SynthesisWithClaims
from precis.pipeline.state import GraphState
from precis.schema import Book, KnownFile

_SYSTEM_PROMPT = (
    "You repair the synthesizing material for a book's study-guide JSON, "
    "which failed schema validation. The most common cause is a `parts` "
    "entry referencing a chapter number that doesn't actually exist — use "
    "the real chapter list provided to fix any such references. Preserve "
    "everything else about the content as closely as possible; only change "
    "what's needed to satisfy the validation error given. Call the tool "
    "with the corrected result."
)

# The only fields a repair attempt can possibly change — see the module
# docstring. Any validation error touching a field outside this set means
# repair is guaranteed to fail identically, so it's skipped entirely rather
# than wasted.
_REPAIRABLE_FIELDS = {"synopsis", "one_line_takeaway", "tags", "parts", "key_claims_for_review"}


def _is_repairable(error: ValidationError, parts_source: str | None) -> bool:
    for err in error.errors():
        loc = err["loc"]
        # A whole-model error (empty loc, e.g. the chapters/key_claims_for_review
        # coupling check) isn't attributable to a specific field either way;
        # treated as repairable since the common real case here is the
        # parts/chapter-number check, also raised with an empty loc — the
        # coupling check should be unreachable in practice anyway, since
        # Stage 3's own branching already keeps the two in sync.
        if loc and loc[0] not in _REPAIRABLE_FIELDS:
            return False
        # parts_source == "known" means `parts` is a pass-through of the
        # reader-supplied known-file structure, already guaranteed valid by
        # known_file.py's preflight check — a validation error touching it
        # here means something upstream is actually broken, not a
        # content-quality slip the model can be asked to revise. Letting
        # generic repair "fix" it would silently swap known structure for
        # an invented one while parts_source still claimed "known". Covers
        # the empty-loc whole-model case too (the parts/chapter-number
        # check above), same as the "common real case" it's already keyed
        # off of for the general repairable check.
        if parts_source == "known" and (not loc or loc[0] == "parts"):
            return False
    return True


def _valid_chapter_numbers_block(chapters: list[dict] | None) -> str:
    if not chapters:
        return "(no chapters — this book has no chapters/parts-reference constraint)\n\n"
    lines = [f"{c['number']}. {c['title']}" for c in chapters]
    return "Real chapters (only these numbers may be referenced by parts):\n" + "\n".join(lines) + "\n\n"


def _repair_user_prompt(
    book_kwargs: dict,
    known_file: KnownFile,
    chapters: list[dict] | None,
    error: ValidationError,
) -> str:
    header = book_header(known_file) + "\n\n"
    current = (
        "Current (invalid) synthesized fields:\n"
        f"one_line_takeaway: {book_kwargs['one_line_takeaway']}\n"
        f"synopsis: {book_kwargs['synopsis']}\n"
        f"tags: {book_kwargs['tags']}\n"
        f"parts: {book_kwargs['parts']}\n"
    )
    if book_kwargs.get("key_claims_for_review") is not None:
        current += f"key_claims_for_review: {book_kwargs['key_claims_for_review']}\n"
    current += "\n"
    footer = f"Validation error to fix:\n{error}\n\nCall the tool with the corrected result."
    return header + _valid_chapter_numbers_block(chapters) + current + footer


async def _repair(
    client: AsyncOpenAI,
    known_file: KnownFile,
    book_kwargs: dict,
    chapters: list[dict] | None,
    error: ValidationError,
    is_full_nonfiction_path: bool,
) -> Synthesis | SynthesisWithClaims:
    response_model = SynthesisWithClaims if is_full_nonfiction_path else Synthesis
    return await llm.complete_structured(
        client,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _repair_user_prompt(book_kwargs, known_file, chapters, error)},
        ],
        response_model=response_model,
    )


def _apply_repair(book_kwargs: dict, repaired: Synthesis | SynthesisWithClaims, is_full_nonfiction_path: bool) -> dict:
    book_kwargs = dict(book_kwargs)
    book_kwargs["synopsis"] = repaired.synopsis
    book_kwargs["one_line_takeaway"] = repaired.one_line_takeaway
    book_kwargs["tags"] = repaired.tags
    # Never let a repair triggered by some *other* field's error overwrite
    # known-file-sourced parts with the repair model's invented ones —
    # _is_repairable already skips repair entirely for a `parts`-shaped
    # error here, but that's a separate guard against a separate failure
    # mode; this is what actually keeps parts_source: "known" honest for
    # any repair that does proceed.
    if book_kwargs.get("parts_source") != "known":
        book_kwargs["parts"] = [p.model_dump() for p in repaired.parts]
    if is_full_nonfiction_path:
        assert isinstance(repaired, SynthesisWithClaims)
        book_kwargs["key_claims_for_review"] = [c.model_dump() for c in repaired.key_claims_for_review]
    return book_kwargs


async def run(
    state: GraphState,
    config: RunnableConfig | None = None,
    *,
    llm_client: AsyncOpenAI | None = None,
) -> dict:
    llm_client = resolve_llm_client(config, llm_client)

    known_file = KnownFile.model_validate(state["known_file"])

    # run_whole_book always seeds state["chapters"] as [] (graph.py), even on
    # the fiction/narrative path where Stage 2 never runs and nothing ever
    # appends to it — so an empty list here means "no chapters", not "zero
    # chapters given". Book.chapters has min_length=1 when a list is given at
    # all, so passing [] straight through would fail every fiction book.
    chapters = state.get("chapters") or []

    book_kwargs: dict = {
        "title": known_file.title,
        "author": known_file.author,
        "year": known_file.year,
        "isbn": known_file.isbn,
        "page_count": known_file.page_count,
        "one_line_takeaway": state["one_line_takeaway"],
        "synopsis": state["synopsis"],
        "tags": state["tags"],
        "parts": state.get("parts"),
        "parts_source": state.get("parts_source"),
        "key_claims_for_review": state.get("key_claims_for_review"),
        "reader_notes": known_file.notes,
        "warnings": state.get("warnings", []),
        "chapters": chapters if chapters else None,
    }

    try:
        book = Book.model_validate(book_kwargs)
    except ValidationError as exc:
        if not _is_repairable(exc, book_kwargs["parts_source"]):
            raise ValueError(
                f"Stage 4 assemble: book invalid in a way the repair pass can't fix "
                f"(not a Stage-3-synthesized field): {exc}"
            ) from exc

        client = llm_client or llm.build_client()
        repaired = await _repair(
            client, known_file, book_kwargs, book_kwargs["chapters"], exc, known_file.is_full_nonfiction_path
        )
        repaired_kwargs = _apply_repair(book_kwargs, repaired, known_file.is_full_nonfiction_path)
        try:
            book = Book.model_validate(repaired_kwargs)
        except ValidationError as exc2:
            raise ValueError(f"Stage 4 assemble: book still invalid after one repair attempt: {exc2}") from exc2

    return {"book": book.model_dump()}
