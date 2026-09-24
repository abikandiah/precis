from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from pydantic import BaseModel, ValidationInfo, model_validator

from precis.llm import StructuredOutputError, complete_structured


class _Verdict(BaseModel):
    verified: bool
    reason: str


class _CountConstrained(BaseModel):
    items: list[str]

    @model_validator(mode="after")
    def _check_count(self, info: ValidationInfo) -> "_CountConstrained":
        expected = (info.context or {}).get("expected_count")
        if expected is not None and len(self.items) != expected:
            raise ValueError(f"expected {expected} items, got {len(self.items)}")
        return self


def _response_with_tool_calls(tool_calls: list | None, content: str | None = None) -> MagicMock:
    message = MagicMock()
    message.tool_calls = tool_calls
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def _mock_client_returning(tool_calls: list | None, content: str | None = None) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_response_with_tool_calls(tool_calls, content))
    return client


def _mock_client_with_sequence(*responses: MagicMock) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=list(responses))
    return client


def _tool_call(arguments: str) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id="call_1",
        type="function",
        function=Function(name="emit__verdict", arguments=arguments),
    )


@pytest.mark.asyncio
async def test_complete_structured_parses_valid_tool_call():
    client = _mock_client_returning([_tool_call('{"verified": true, "reason": "matches"}')])
    verdict = await complete_structured(client, messages=[], response_model=_Verdict)
    assert verdict == _Verdict(verified=True, reason="matches")


@pytest.mark.asyncio
async def test_complete_structured_raises_when_no_tool_call_made():
    client = _mock_client_returning(None, content="I didn't use the tool")
    with pytest.raises(StructuredOutputError):
        await complete_structured(client, messages=[], response_model=_Verdict)


@pytest.mark.asyncio
async def test_complete_structured_raises_on_malformed_arguments():
    client = _mock_client_returning([_tool_call("not json")])
    with pytest.raises(StructuredOutputError):
        await complete_structured(client, messages=[], response_model=_Verdict)


@pytest.mark.asyncio
async def test_complete_structured_raises_on_schema_mismatch():
    client = _mock_client_returning([_tool_call('{"unrelated_field": 1}')])
    with pytest.raises(StructuredOutputError):
        await complete_structured(client, messages=[], response_model=_Verdict)


@pytest.mark.asyncio
async def test_complete_structured_retries_and_succeeds_on_second_attempt():
    client = _mock_client_with_sequence(
        _response_with_tool_calls(None, content="oops, no tool call"),
        _response_with_tool_calls([_tool_call('{"verified": true, "reason": "matches"}')]),
    )
    verdict = await complete_structured(client, messages=[], response_model=_Verdict, max_attempts=2)
    assert verdict == _Verdict(verified=True, reason="matches")
    assert client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_complete_structured_raises_only_after_exhausting_max_attempts():
    client = _mock_client_returning(None, content="never calls the tool")
    with pytest.raises(StructuredOutputError, match="after 3 attempts"):
        await complete_structured(client, messages=[], response_model=_Verdict, max_attempts=3)
    assert client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_complete_structured_max_attempts_one_means_no_retry():
    client = _mock_client_returning(None, content="never calls the tool")
    with pytest.raises(StructuredOutputError):
        await complete_structured(client, messages=[], response_model=_Verdict, max_attempts=1)
    assert client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_no_validation_context_means_unconstrained():
    """Without validation_context, a model_validator reading `info.context`
    sees None and skips its check — the default, unconstrained behavior
    every existing caller of complete_structured already relies on.
    """
    client = _mock_client_returning([_tool_call('{"items": ["a", "b", "c"]}')])
    result = await complete_structured(client, messages=[], response_model=_CountConstrained)
    assert result.items == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_validation_context_mismatch_triggers_retry_then_succeeds():
    client = _mock_client_with_sequence(
        _response_with_tool_calls([_tool_call('{"items": ["a", "b", "c"]}')]),  # wrong count first
        _response_with_tool_calls([_tool_call('{"items": ["a", "b"]}')]),  # correct on retry
    )
    result = await complete_structured(
        client,
        messages=[],
        response_model=_CountConstrained,
        max_attempts=2,
        validation_context={"expected_count": 2},
    )
    assert result.items == ["a", "b"]
    assert client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_validation_context_mismatch_raises_after_exhausting_attempts():
    client = _mock_client_returning([_tool_call('{"items": ["a", "b", "c"]}')])
    with pytest.raises(StructuredOutputError, match="expected 2 items, got 3"):
        await complete_structured(
            client,
            messages=[],
            response_model=_CountConstrained,
            max_attempts=2,
            validation_context={"expected_count": 2},
        )
    assert client.chat.completions.create.call_count == 2
