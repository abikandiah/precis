from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from precis import write
from precis.research import Research, Source
from precis.schema import KeyClaim, KnownFile

NONFICTION = KnownFile(
    isbn="9780374533557",
    title="Thinking, Fast and Slow",
    author="Daniel Kahneman",
    year=2011,
    kind="non-fiction",
    notes="I care most about the decision-making parts.",
)
FICTION = KnownFile(isbn="9780451524935", title="Nineteen Eighty-Four", author="George Orwell", kind="fiction")
RESEARCH = Research(
    sources=[
        Source(id="S1", title="Review", url="https://a.org", text="Kahneman's two systems."),
        Source(id="S2", title="Interview", url="https://b.org", text="On loss aversion."),
    ],
    warnings=["research warning"],
)


def _idea(n: int, sources: list[str] | None = None) -> dict:
    return {"title": f"Idea {n}", "summary": "s", "evidence": "e", "sources": sources or []}


def _claim(n: int) -> dict:
    return {"prompt": f"Q{n}?", "answer": "A."}


def _draft(kind: str = "non-fiction", ideas: int = 6, claims: int = 6, **extra) -> dict:
    draft = {
        "one_line_takeaway": "take",
        "synopsis": "para one\n\npara two",
        "ideas": [_idea(n, ["S1"] if n == 0 else None) for n in range(ideas)],
        "tags": ["psychology", "science"] if kind == "non-fiction" else ["dystopian", "fiction-literary"],
        **extra,
    }
    if kind == "non-fiction":
        draft["key_claims_for_review"] = [_claim(n) for n in range(claims)]
    return draft


def _validate(model, data: dict, kind: str = "non-fiction", source_ids=("S1", "S2")):
    return model.model_validate(data, context={write.KIND_KEY: kind, write.SOURCE_IDS_KEY: set(source_ids)})


# --- the write call's response model ------------------------------------------------


def test_a_valid_nonfiction_draft_passes():
    draft = _validate(write.DraftWithClaims, _draft())
    assert draft.claims is not None and len(draft.claims) == 6


@pytest.mark.parametrize(
    ("data", "kind", "message"),
    [
        (_draft(ideas=0), "non-fiction", "at least one idea"),
        (_draft(claims=0), "non-fiction", "non-fiction needs key_claims_for_review"),
        (_draft(tags=["psychology", "made-up"]), "non-fiction", "closed non-fiction vocabulary"),
    ],
)
def test_draft_counts_and_tags_are_validated_per_kind(data, kind, message):
    model = write.Draft if kind == "fiction" else write.DraftWithClaims
    with pytest.raises(ValidationError, match=message):
        _validate(model, data, kind)


def test_an_idea_citing_a_source_that_isnt_in_the_research_is_rejected():
    data = _draft()
    data["ideas"][1]["sources"] = ["S7"]
    with pytest.raises(ValidationError, match=r"cites \['S7'\], which aren't research sources"):
        _validate(write.DraftWithClaims, data)


def test_a_malformed_source_id_is_rejected():
    data = _draft()
    data["ideas"][1]["sources"] = ["source 1"]
    with pytest.raises(ValidationError, match="S1, S2"):
        _validate(write.DraftWithClaims, data)


# --- prompts -------------------------------------------------------------------------


def test_context_messages_cache_the_book_and_research_in_the_system_message():
    [system] = write.context_messages(NONFICTION, RESEARCH)
    assert system["role"] == "system"
    [part] = system["content"]  # type: ignore[misc]
    assert part["cache_control"] == {"type": "ephemeral"}
    assert 'Book: "Thinking, Fast and Slow" by Daniel Kahneman (2011)' in part["text"]
    assert '<source id="S2"' in part["text"]
    assert "untrusted reference data" in part["text"]
    # Nothing task-specific, so the review call shares the cached prefix.
    assert "key_claims" not in part["text"] and "spoiler" not in part["text"]


def test_fiction_instructions_are_spoiler_safe_and_ask_for_themes_not_claims():
    text = write._fiction_instructions(FICTION)
    assert "No spoilers" in text and "themes" in text
    assert "key_claims_for_review" not in text
    assert "dystopian" in text and "psychology" not in text


def test_nonfiction_instructions_ask_for_claims_and_pass_reader_notes_on():
    text = write._nonfiction_instructions(NONFICTION)
    assert "key_claims_for_review: 5-15" in text and "ideas: 5-12" in text
    assert "<reader_notes>\nI care most about the decision-making parts.\n</reader_notes>" in text
    assert "<reader_notes>" not in write._fiction_instructions(FICTION)


# --- write_notes ----------------------------------------------------------------------


async def test_write_notes_makes_one_structured_call_and_assembles_the_notes():
    draft = _validate(write.DraftWithClaims, _draft())
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        notes = await write.write_notes(NONFICTION, RESEARCH, client=MagicMock())
    kwargs = call.await_args.kwargs
    assert kwargs["response_model"] is write.DraftWithClaims
    # The same tools the review call sends, so its cache prefix matches.
    assert kwargs["tool_models"] == write.shared_tools("non-fiction")
    assert kwargs["validation_context"] == {write.KIND_KEY: "non-fiction", write.SOURCE_IDS_KEY: {"S1", "S2"}}
    assert kwargs["timeout_seconds"] == write.WRITE_TIMEOUT_SECONDS
    system, user = kwargs["messages"]
    assert system == write.context_messages(NONFICTION, RESEARCH)[0]
    assert user["role"] == "user" and "key_claims_for_review" in user["content"]

    assert (notes.title, notes.author, notes.year, notes.isbn) == (
        "Thinking, Fast and Slow", "Daniel Kahneman", 2011, "9780374533557"
    )
    assert notes.reader_notes == NONFICTION.notes
    assert notes.key_claims_for_review == [KeyClaim(**_claim(n)) for n in range(6)]
    assert notes.ideas[0].sources == ["S1"]
    assert notes.warnings == ["research warning"]


async def test_write_notes_uses_the_fiction_model_for_fiction():
    draft = _validate(write.Draft, _draft("fiction", ideas=4), "fiction")
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        notes = await write.write_notes(FICTION, RESEARCH, client=MagicMock())
    assert call.await_args.kwargs["response_model"] is write.Draft
    assert notes.key_claims_for_review is None


@pytest.mark.parametrize(
    ("mismatch", "fails"),
    [
        (None, False),
        ("Daniel Kahneman and Amos Tversky", False),  # an added co-author
        ("D. Kahneman", False),  # another form of the name
        ("Kahneman, Daniel", False),
        ("N/A", False),  # a placeholder, not an author
        ("None.", False),
        ("unknown", False),
        ("Dr. Daniel Kahneman", False),  # a title isn't a given name
        ("Richard Thaler", True),
        ("Amos Kahneman", True),  # same surname, a different person
    ],
)
async def test_a_reported_different_author_fails_the_run(mismatch, fails):
    draft = _validate(write.DraftWithClaims, _draft(author_mismatch=mismatch))
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)):
        if fails:
            with pytest.raises(write.IdentityError, match=f"credits this book to '{mismatch}'"):
                await write.write_notes(NONFICTION, RESEARCH, client=MagicMock())
        else:
            await write.write_notes(NONFICTION, RESEARCH, client=MagicMock())


async def test_trust_known_turns_a_different_author_into_a_warning():
    draft = _validate(write.DraftWithClaims, _draft(author_mismatch="Richard Thaler"))
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)):
        notes = await write.write_notes(NONFICTION, RESEARCH, trust_known=True, client=MagicMock())
    assert notes.author == "Daniel Kahneman"
    assert any("Richard Thaler" in w and "--trust-known" in w for w in notes.warnings)


async def test_write_notes_caps_retries_tokens_and_time_for_the_long_call():
    draft = _validate(write.DraftWithClaims, _draft())
    client = MagicMock()
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        await write.write_notes(NONFICTION, RESEARCH, client=client)
    client.with_options.assert_called_once_with(max_retries=write.WRITE_MAX_RETRIES)
    assert call.await_args.args[0] is client.with_options.return_value
    assert call.await_args.kwargs["max_tokens"] == write.WRITE_MAX_TOKENS


def test_a_placeholder_author_mismatch_reads_as_none():
    assert _validate(write.DraftWithClaims, _draft(author_mismatch=" N/A ")).author_mismatch is None
    assert _validate(write.DraftWithClaims, _draft(author_mismatch=" Eric Blair ")).author_mismatch == "Eric Blair"


def test_the_author_mismatch_description_covers_pen_names():
    description = write.Draft.model_fields["author_mismatch"].description or ""
    assert "pen name" in description


def test_key_claims_mustnt_just_restate_an_idea_title():
    assert "don't just restate an idea's title as a question" in write._nonfiction_instructions(NONFICTION)
