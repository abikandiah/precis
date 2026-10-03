import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from precis import write
from precis.research import BOOK_FILE_URL, Research, Source
from precis.schema import MAX_IDEAS, KeyClaim, KnownFile
from precis.search import author_names

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
    depth="full",  # non-fiction's deck comes with full notes
)
OVERVIEW = dataclasses.replace(RESEARCH, depth="overview")


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


def _validate(
    model, data: dict, kind: str = "non-fiction", source_ids=("S1", "S2"), depth: str | None = None, numbered_list=False
):
    depth = depth or ("full" if kind == "non-fiction" else "overview")
    context = {
        write.KIND_KEY: kind,
        write.DEPTH_KEY: depth,
        write.SOURCE_IDS_KEY: set(source_ids),
        write.NUMBERED_LIST_KEY: numbered_list,
    }
    return model.model_validate(data, context=context)


def test_more_ideas_than_the_ceiling_are_sent_back_unless_the_book_follows_its_own_list():
    with pytest.raises(ValidationError, match="13 ideas — at most 12"):
        _validate(write.DraftWithClaims, _draft(ideas=13))
    with pytest.raises(ValidationError, match="13 ideas — at most 12"):
        _validate(write.OverviewDraft, _draft(ideas=13), depth="overview", numbered_list=True)
    assert len(_validate(write.DraftWithClaims, _draft(ideas=48), numbered_list=True).ideas) == 48


# --- the write call's response model ------------------------------------------------


def test_a_valid_nonfiction_draft_passes():
    draft = _validate(write.DraftWithClaims, _draft())
    assert draft.claims is not None and len(draft.claims) == 6


@pytest.mark.parametrize(
    ("data", "kind", "message"),
    [
        (_draft(ideas=0), "non-fiction", "at least one idea"),
        (_draft(claims=0), "non-fiction", "full non-fiction notes need key_claims_for_review"),
        (_draft(tags=["psychology", "made-up"]), "non-fiction", "give 2-4 tags from the closed non-fiction vocabulary"),
    ],
)
def test_draft_counts_and_tags_are_validated_per_kind(data, kind, message):
    model = write.Draft if kind == "fiction" else write.DraftWithClaims
    with pytest.raises(ValidationError, match=message):
        _validate(model, data, kind)


def test_a_citation_of_a_source_that_isnt_in_the_research_is_dropped():
    data = _draft()
    data["ideas"][1]["sources"] = ["S1", "S7"]
    assert _validate(write.DraftWithClaims, data).ideas[1].sources == ["S1"]


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
    text = write._fiction_instructions(FICTION, "overview")
    assert "No spoilers" in text and write.FICTION_COUNT_RULE in text
    assert "key_claims_for_review" not in text
    assert "dystopian" in text and "psychology" not in text


def test_nonfiction_instructions_ask_for_claims_and_pass_reader_notes_on():
    text = write._nonfiction_instructions(NONFICTION, "full")
    assert "key_claims_for_review: recall questions" in text and write.IDEA_COUNT_RULE in text
    assert "<reader_notes>\nI care most about the decision-making parts.\n</reader_notes>" in text
    assert "<reader_notes>" not in write._fiction_instructions(FICTION, "overview")


# --- write_notes ----------------------------------------------------------------------


async def test_write_notes_makes_one_structured_call_and_assembles_the_notes():
    draft = _validate(write.DraftWithClaims, _draft())
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        notes = await write.write_notes(NONFICTION, RESEARCH, client=MagicMock())
    kwargs = call.await_args.kwargs
    assert kwargs["response_model"] is write.DraftWithClaims
    # The same tools the review call sends, so its cache prefix matches.
    assert kwargs["tool_models"] == write.shared_tools("non-fiction", "full")
    assert kwargs["validation_context"] == {
        write.KIND_KEY: "non-fiction",
        write.DEPTH_KEY: "full",
        write.SOURCE_IDS_KEY: {"S1", "S2"},
        write.NUMBERED_LIST_KEY: False,
    }
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
    draft = _validate(write.OverviewDraft, _draft("fiction", ideas=4), "fiction")
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        notes = await write.write_notes(FICTION, OVERVIEW, client=MagicMock())
    assert call.await_args.kwargs["response_model"] is write.OverviewDraft
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


@pytest.mark.parametrize(
    ("claimed", "reported", "same"),
    [
        ("Ursula K. Le Guin", "Le Guin", True),
        ("Gabriel García Márquez", "García Márquez", True),
        ("Kingsley Amis", "Martin Amis", False),
    ],
)
def test_same_person_accepts_a_multi_word_surname_alone(claimed, reported, same):
    assert write._same_person(author_names(reported)[0], author_names(claimed)[0]) is same


def test_author_differs_false_overrides_any_name():
    for phrasing in ("No mismatch", "Same author", "not applicable"):
        data = _draft(author_differs=False, author_mismatch=phrasing)
        assert _validate(write.DraftWithClaims, data).author_mismatch is None
    data = _draft(author_differs=True, author_mismatch="Richard Thaler")
    assert _validate(write.DraftWithClaims, data).author_mismatch == "Richard Thaler"


async def test_author_differs_without_a_name_still_fails():
    draft = _validate(write.DraftWithClaims, _draft(author_differs=True, author_mismatch="N/A"))
    with (
        patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)),
        pytest.raises(write.IdentityError, match="an unnamed author"),
    ):
        await write.write_notes(NONFICTION, RESEARCH, client=MagicMock())


def test_a_placeholder_author_mismatch_reads_as_none():
    assert _validate(write.DraftWithClaims, _draft(author_mismatch=" N/A ")).author_mismatch is None
    assert _validate(write.DraftWithClaims, _draft(author_mismatch=" Eric Blair ")).author_mismatch == "Eric Blair"


def test_the_author_differs_description_covers_pen_names():
    description = write.Draft.model_fields["author_differs"].description or ""
    assert "pen name" in description


def test_key_claims_mustnt_just_restate_an_idea_title():
    assert "don't just restate an idea's title as a question" in write._nonfiction_instructions(NONFICTION, "full")


def test_unknown_repeated_or_extra_tags_are_dropped_not_retried():
    tags = [{"name": "x"}, "self-help", "psychology", "psychology", "science", "business", "economics", "history"]
    draft = _validate(write.DraftWithClaims, _draft(tags=tags))
    assert draft.tags == ["psychology", "science", "business", "economics"]


# --- full notes -----------------------------------------------------------------------


def test_the_draft_has_a_deck_for_nonfiction_and_an_ending_for_fiction_read_whole():
    assert write.draft_model("non-fiction", "full") is write.DraftWithClaims
    assert write.draft_model("fiction", "full") is write.DraftWithResolution
    assert write.draft_model("fiction", "overview") is write.OverviewDraft


def test_full_instructions_write_from_the_book():
    nonfiction = write._nonfiction_instructions(NONFICTION, "full")
    assert write.FULL_RULE in nonfiction and write.IDEA_COUNT_RULE in nonfiction
    fiction = write._fiction_instructions(FICTION, "full")
    assert write.FULL_RULE in fiction and write.RESOLUTION_RULE in fiction
    assert "No spoilers anywhere but resolution" in fiction
    overview = write._fiction_instructions(FICTION, "overview")
    assert write.RESOLUTION_RULE not in overview and "evidence" not in overview
    assert write.FULL_RULE not in write._nonfiction_instructions(NONFICTION, "overview")


def test_a_book_built_around_its_own_list_gets_one_idea_per_item_in_full_notes_only():
    listed = NONFICTION.model_copy(update={"numbered_list": True})
    full = write._nonfiction_instructions(listed, "full")
    assert write.LIST_COUNT_RULE in full and write.IDEA_COUNT_RULE not in full
    assert write.OVERVIEW_COUNT_RULE in write._nonfiction_instructions(listed, "overview")


def test_non_fiction_writes_concepts_not_the_books_particulars():
    full = write._nonfiction_instructions(NONFICTION, "full")
    assert write.ALTITUDE_RULE in full and write.ONE_EXAMPLE_RULE in full
    assert "Never a list of specifics or trivia" in full
    # An overview has no evidence to hold its one example.
    overview = write._nonfiction_instructions(NONFICTION, "overview")
    assert write.ALTITUDE_RULE in overview and write.ONE_EXAMPLE_RULE not in overview
    # A novel's themes aren't taught.
    for depth in ("full", "overview"):
        assert write.ALTITUDE_RULE not in write._fiction_instructions(FICTION, depth)


def test_major_points_come_from_the_whole_book_by_rank():
    assert "its later chapters as much as its opening" in write.IDEA_COUNT_RULE
    assert "keep the most important and leave out the weaker" in write.IDEA_COUNT_RULE


def test_the_ceiling_is_stated_as_one_never_as_a_range_to_fill():
    for rule in (write.IDEA_COUNT_RULE, write.FICTION_COUNT_RULE):
        assert "a ceiling, not a target" in rule
    # The number is named once, as the ceiling, so the model can't anchor on it as a goal.
    assert "Never pad toward the ceiling" in write.IDEA_COUNT_RULE and write.IDEA_COUNT_RULE.count(str(MAX_IDEAS)) == 1


def test_full_notes_come_from_the_book_itself():
    book = Research(sources=[Source(id="S1", title="t", url=BOOK_FILE_URL, text="x")], warnings=[], depth="full")
    full = write.context_messages(FICTION, book)[0]["content"][0]["text"]
    assert "notes on its full text" in full and "from the reader's own copy of the book" in full
    assert "from the web" not in full
    assert "from the web" in write.context_messages(FICTION, OVERVIEW)[0]["content"][0]["text"]


async def test_fiction_read_whole_keeps_its_ending_apart():
    book = Research(sources=[Source(id="S1", title="t", url=BOOK_FILE_URL, text="x")], warnings=[], depth="full")
    draft = _validate(write.DraftWithResolution, _draft("fiction", ideas=4, resolution="Winston loves Big Brother."), "fiction")
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        notes = await write.write_notes(FICTION, book, client=MagicMock())
    assert call.await_args.kwargs["response_model"] is write.DraftWithResolution
    assert call.await_args.kwargs["tool_models"] == write.shared_tools("fiction", "full")
    assert (notes.depth, notes.resolution) == ("full", "Winston loves Big Brother.")


def test_an_empty_or_placeholder_ending_is_sent_back():
    for empty in ("", "  ", "N/A."):
        with pytest.raises(ValidationError, match="must say how the story ends"):
            _validate(write.DraftWithResolution, _draft("fiction", ideas=4, resolution=empty), "fiction")


# --- overviews ------------------------------------------------------------------------


def test_an_overview_has_headline_ideas_and_no_deck():
    assert write.draft_model("non-fiction", "overview") is write.OverviewDraft
    text = write._nonfiction_instructions(NONFICTION, "overview")
    assert write.OVERVIEW_COUNT_RULE in text and "an overview" in text
    assert "key_claims_for_review" not in text and write.IDEA_COUNT_RULE not in text
    fiction = write._fiction_instructions(FICTION, "overview")
    assert "evidence" not in fiction and "examples" not in fiction


async def test_a_nonfiction_overview_is_written_without_claims_or_evidence():
    draft = _validate(write.OverviewDraft, {**_draft(), "key_claims_for_review": None}, depth="overview")
    with patch.object(write.llm, "complete_structured", new=AsyncMock(return_value=draft)) as call:
        notes = await write.write_notes(NONFICTION, OVERVIEW, client=MagicMock())
    assert call.await_args.kwargs["response_model"] is write.OverviewDraft
    assert call.await_args.kwargs["validation_context"][write.DEPTH_KEY] == "overview"
    assert notes.depth == "overview" and notes.key_claims_for_review is None
    assert all(i.evidence == "" for i in notes.ideas)


def test_an_overview_draft_with_a_deck_is_sent_back():
    with pytest.raises(ValidationError, match="only full non-fiction notes have key_claims_for_review"):
        _validate(write.DraftWithClaims, _draft(), depth="overview")


def _fields(model) -> tuple[set[str], set[str]]:
    """A tool's top-level fields, and its ideas'."""
    schema = model.model_json_schema()
    return set(schema["properties"]), set(schema["$defs"]["Idea"]["properties"])


def test_each_modes_tools_offer_only_what_its_notes_can_have():
    from precis import review

    every_idea_field = {"title", "summary", "evidence", "sources"}
    expected = {
        ("non-fiction", "full"): (True, False, every_idea_field),
        ("fiction", "full"): (False, True, every_idea_field),
        ("non-fiction", "overview"): (False, False, {"title", "summary", "sources"}),
        ("fiction", "overview"): (False, False, {"title", "summary", "sources"}),
    }
    for (kind, depth), (deck, ending, idea_fields) in expected.items():
        tools = write.shared_tools(kind, depth)
        # The write and review calls send the same tools, so the review reuses the cached research.
        assert tools == [write.draft_model(kind, depth), review.review_model(kind, depth)]
        for tool in tools:
            fields, ideas = _fields(tool)
            assert ("key_claims_for_review" in fields, "resolution" in fields, ideas) == (deck, ending, idea_fields)


def test_stripping_a_field_the_schema_doesnt_have_fails_loudly():
    from precis.schema import stripped_schema

    with pytest.raises(RuntimeError, match="no Idea definition"):
        stripped_schema({"properties": {}}, idea_fields=("evidence",))
    with pytest.raises(RuntimeError, match="no 'resolution' field"):
        stripped_schema({"properties": {}}, fields=("resolution",))


def test_validating_a_draft_without_its_depth_or_numbered_list_fails_loudly():
    with pytest.raises(RuntimeError, match="needs its depth"):
        write.DraftWithClaims.model_validate(_draft(), context={write.KIND_KEY: "non-fiction"})
    with pytest.raises(RuntimeError, match="needs NUMBERED_LIST_KEY"):
        write.DraftWithClaims.model_validate(_draft(), context={write.KIND_KEY: "non-fiction", write.DEPTH_KEY: "full"})
