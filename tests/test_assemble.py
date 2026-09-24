from unittest.mock import AsyncMock

import pytest

from precis.pipeline.nodes import assemble
from precis.pipeline.nodes.synthesize import SynthesisWithClaims
from precis.schema import KeyClaim, KnownFile, Part


def _nonfiction_known_file() -> KnownFile:
    return KnownFile(
        isbn="123", title="A Book", author="An Author", kind="non-fiction", narrative=False, chapters=["Ch 1", "Ch 2"]
    )


def _fiction_known_file() -> KnownFile:
    return KnownFile(isbn="456", title="A Novel", author="A Novelist", kind="fiction")


def _chapters() -> list[dict]:
    return [
        {"number": 1, "title": "Ch 1", "key_points": ["a"], "core_claim": "claim 1", "quality_flag": None},
        {"number": 2, "title": "Ch 2", "key_points": ["b"], "core_claim": "claim 2", "quality_flag": None},
    ]


def _nonfiction_state(**overrides) -> dict:
    state = {
        "known_file": _nonfiction_known_file().model_dump(),
        "chapters": _chapters(),
        "one_line_takeaway": "the takeaway",
        "synopsis": "a synopsis",
        "tags": ["tag1", "tag2"],
        "parts": [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}],
        "parts_source": "generated",
        "key_claims_for_review": [{"prompt": "q1", "answer": "a1"}],
        "warnings": [],
    }
    state.update(overrides)
    return state


def _fiction_state(**overrides) -> dict:
    state = {
        "known_file": _fiction_known_file().model_dump(),
        "chapters": [],
        "one_line_takeaway": "the takeaway",
        "synopsis": "a synopsis",
        "tags": ["tag1"],
        "parts": [{"title": "Beginning", "summary": "stakes are introduced", "chapter_numbers": None}],
        "parts_source": "generated",
        "warnings": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_nonfiction_path_valid_on_first_attempt_no_repair_called(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    result = await assemble.run(_nonfiction_state(), llm_client=AsyncMock())

    fake.assert_not_called()
    book = result["book"]
    assert book["title"] == "A Book"
    assert book["author"] == "An Author"
    assert book["isbn"] == "123"
    assert book["one_line_takeaway"] == "the takeaway"
    assert book["synopsis"] == "a synopsis"
    assert book["tags"] == ["tag1", "tag2"]
    assert book["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}]
    assert book["parts_source"] == "generated"
    assert book["key_claims_for_review"] == [{"prompt": "q1", "answer": "a1"}]
    assert len(book["chapters"]) == 2
    assert book["reader_notes"] is None
    assert book["warnings"] == []


@pytest.mark.asyncio
async def test_fiction_path_with_empty_chapters_list_becomes_none(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    result = await assemble.run(_fiction_state(), llm_client=AsyncMock())

    fake.assert_not_called()
    book = result["book"]
    assert book["chapters"] is None
    assert book["key_claims_for_review"] is None


@pytest.mark.asyncio
async def test_reader_notes_passed_through_verbatim(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    known_file = _fiction_known_file()
    known_file = KnownFile.model_validate({**known_file.model_dump(), "notes": "read this slowly"})
    state = _fiction_state(known_file=known_file.model_dump())

    result = await assemble.run(state, llm_client=AsyncMock())

    assert result["book"]["reader_notes"] == "read this slowly"


@pytest.mark.asyncio
async def test_invalid_parts_chapter_reference_triggers_one_repair_call(monkeypatch):
    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
    )

    calls = []

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        calls.append(response_model)
        assert response_model is SynthesisWithClaims
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaim(prompt="q1", answer="a1")],
            parts=[Part(title="Part One", summary="covers ch 1-2", chapter_numbers=[1, 2])],
        )

    monkeypatch.setattr(assemble.llm, "complete_structured", fake_complete_structured)

    result = await assemble.run(state, llm_client=AsyncMock())

    assert len(calls) == 1
    book = result["book"]
    assert book["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}]


@pytest.mark.asyncio
async def test_repair_still_invalid_raises_informative_error(monkeypatch):
    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        # Repaired result still references a nonexistent chapter.
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaim(prompt="q1", answer="a1")],
            parts=[Part(title="Part One", summary="still bad", chapter_numbers=[1, 99])],
        )

    monkeypatch.setattr(assemble.llm, "complete_structured", fake_complete_structured)

    with pytest.raises(ValueError, match="Stage 4 assemble: book still invalid after one repair attempt"):
        await assemble.run(state, llm_client=AsyncMock())


@pytest.mark.asyncio
async def test_unrepairable_field_error_skips_repair_and_raises_immediately(monkeypatch):
    """Regression test: a validation error on a field the repair pass can
    never touch (e.g. title, which Synthesis/SynthesisWithClaims don't
    carry) used to still trigger one wasted repair attempt that was
    guaranteed to reproduce the identical error. It must now raise
    immediately instead, and never call complete_structured at all.
    """
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    known_file = KnownFile.model_validate({**_nonfiction_known_file().model_dump(), "title": None})
    state = _nonfiction_state(known_file=known_file.model_dump())

    with pytest.raises(ValueError, match="not a Stage-3-synthesized field"):
        await assemble.run(state, llm_client=AsyncMock())

    fake.assert_not_called()


@pytest.mark.asyncio
async def test_known_parts_source_skips_repair_and_raises_immediately(monkeypatch):
    """A validation error touching `parts` when parts_source is "known"
    must never be handed to the generic repair pass — that would silently
    swap the reader-supplied known structure for an invented one while the
    book still claimed parts_source: "known". known_file.py's preflight
    check is what's supposed to keep this from ever legitimately firing, so
    reaching it here means something upstream is broken and should surface
    loudly, not get "fixed" by an LLM call.
    """
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
        parts_source="known",
    )

    with pytest.raises(ValueError, match="not a Stage-3-synthesized field"):
        await assemble.run(state, llm_client=AsyncMock())

    fake.assert_not_called()


def test_apply_repair_never_overwrites_known_parts_even_if_invoked():
    """Belt-and-suspenders: even if repair were ever triggered while
    parts_source == "known" (e.g. by a validation error on some other
    field), _apply_repair must not let the repair model's invented parts
    overwrite the known-file-sourced ones — _is_repairable is the first
    guard against this, this is the second, independent of whether the
    first one's error-location logic is airtight.
    """
    book_kwargs = {
        "parts": [{"title": "Part One", "summary": "the real summary", "chapter_numbers": [1, 2]}],
        "parts_source": "known",
    }
    repaired = SynthesisWithClaims(
        synopsis="s",
        one_line_takeaway="t",
        tags=["tag"],
        key_claims_for_review=[KeyClaim(prompt="q", answer="a")],
        parts=[Part(title="invented part", summary="invented summary", chapter_numbers=[1])],
    )

    result = assemble._apply_repair(book_kwargs, repaired, is_full_nonfiction_path=True)

    assert result["parts"] == [{"title": "Part One", "summary": "the real summary", "chapter_numbers": [1, 2]}]


def test_apply_repair_overwrites_generated_parts_as_before():
    book_kwargs = {
        "parts": [{"title": "old", "summary": "old summary", "chapter_numbers": [1]}],
        "parts_source": "generated",
    }
    repaired = SynthesisWithClaims(
        synopsis="s",
        one_line_takeaway="t",
        tags=["tag"],
        key_claims_for_review=[KeyClaim(prompt="q", answer="a")],
        parts=[Part(title="new", summary="new summary", chapter_numbers=[1, 2])],
    )

    result = assemble._apply_repair(book_kwargs, repaired, is_full_nonfiction_path=True)

    assert result["parts"] == [{"title": "new", "summary": "new summary", "chapter_numbers": [1, 2]}]


@pytest.mark.asyncio
async def test_known_parts_source_passes_through_when_already_valid(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(assemble.llm, "complete_structured", fake)

    state = _nonfiction_state(parts_source="known")

    result = await assemble.run(state, llm_client=AsyncMock())

    fake.assert_not_called()
    assert result["book"]["parts_source"] == "known"
    assert result["book"]["parts"] == [{"title": "Part One", "summary": "covers ch 1-2", "chapter_numbers": [1, 2]}]


@pytest.mark.asyncio
async def test_clients_from_config_are_used_when_not_passed_explicitly(monkeypatch):
    llm_client = AsyncMock()
    state = _nonfiction_state(
        parts=[{"title": "Part One", "summary": "covers ch 1-99", "chapter_numbers": [1, 99]}],
    )

    async def fake_complete_structured(client, *, messages, response_model, model=None):
        assert client is llm_client
        return SynthesisWithClaims(
            synopsis="a synopsis",
            one_line_takeaway="the takeaway",
            tags=["tag1", "tag2"],
            key_claims_for_review=[KeyClaim(prompt="q1", answer="a1")],
            parts=[Part(title="Part One", summary="covers ch 1-2", chapter_numbers=[1, 2])],
        )

    monkeypatch.setattr(assemble.llm, "complete_structured", fake_complete_structured)

    result = await assemble.run(state, config={"configurable": {"llm_client": llm_client}})
    assert result["book"]["parts"][0]["chapter_numbers"] == [1, 2]
