from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from precis import generate as generate_module
from precis.research import Research
from precis.schema import KnownFile

BOOK = KnownFile(isbn="1", title="T", author="A", kind="fiction")


async def test_generate_researches_then_writes_from_that_research():
    found = Research(sources=[], warnings=["thin research"])
    written = SimpleNamespace(ideas=[], key_claims_for_review=None, warnings=["thin research", "author mismatch"])
    messages: list[str] = []
    with (
        patch.object(generate_module, "research", new=AsyncMock(return_value=found)) as research,
        patch.object(generate_module, "write_notes", new=AsyncMock(return_value=written)) as write,
    ):
        result = await generate_module.generate(
            BOOK, slug="t", trust_known=True, fresh=True, on_progress=messages.append
        )
    assert (result.book, result.research) == (written, found)
    kwargs = research.await_args.kwargs
    assert (kwargs["slug"], kwargs["trust_known"], kwargs["fresh"]) == ("t", True, True)
    assert write.await_args.args == (BOOK, found)
    assert write.await_args.kwargs["trust_known"] is True
    # The write step's own warnings are shown; research printed its own.
    assert "write warning: author mismatch" in messages
    assert "write warning: thin research" not in messages
