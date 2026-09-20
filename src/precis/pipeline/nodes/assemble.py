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
"""

from langchain_core.runnables import RunnableConfig
from openai import AsyncOpenAI
from pydantic import ValidationError

from precis import llm
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


def _chapters_block(chapters: list[dict] | None) -> str:
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
    header = f'Book: "{known_file.title}" by {known_file.author}\n\n'
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
    return header + _chapters_block(chapters) + current + footer


async def _repair(
    client: AsyncOpenAI,
    known_file: KnownFile,
    book_kwargs: dict,
    chapters: list[dict] | None,
    error: ValidationError,
) -> Synthesis | SynthesisWithClaims:
    response_model = SynthesisWithClaims if known_file.is_full_nonfiction_path else Synthesis
    return await llm.complete_structured(
        client,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _repair_user_prompt(book_kwargs, known_file, chapters, error)},
        ],
        response_model=response_model,
    )


def _apply_repair(book_kwargs: dict, repaired: Synthesis | SynthesisWithClaims) -> dict:
    book_kwargs = dict(book_kwargs)
    book_kwargs["synopsis"] = repaired.synopsis
    book_kwargs["one_line_takeaway"] = repaired.one_line_takeaway
    book_kwargs["tags"] = repaired.tags
    book_kwargs["parts"] = [p.model_dump() for p in repaired.parts]
    if isinstance(repaired, SynthesisWithClaims):
        book_kwargs["key_claims_for_review"] = [c.model_dump() for c in repaired.key_claims_for_review]
    return book_kwargs


async def run(
    state: GraphState,
    config: RunnableConfig | None = None,
    *,
    llm_client: AsyncOpenAI | None = None,
) -> dict:
    configurable = (config or {}).get("configurable", {})
    llm_client = llm_client or configurable.get("llm_client")

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
        "key_claims_for_review": state.get("key_claims_for_review"),
        "reader_notes": known_file.notes,
        "warnings": state.get("warnings", []),
        "chapters": chapters if chapters else None,
    }

    try:
        book = Book.model_validate(book_kwargs)
    except ValidationError as exc:
        client = llm_client or llm.build_client()
        repaired = await _repair(client, known_file, book_kwargs, book_kwargs["chapters"], exc)
        repaired_kwargs = _apply_repair(book_kwargs, repaired)
        try:
            book = Book.model_validate(repaired_kwargs)
        except ValidationError as exc2:
            raise ValueError(f"Stage 4 assemble: book still invalid after one repair attempt: {exc2}") from exc2

    return {"book": book.model_dump()}
