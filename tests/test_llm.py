from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from pydantic import BaseModel

from precis.llm import StructuredOutputError, complete_structured


class _Verdict(BaseModel):
    verified: bool
    reason: str


def _mock_client_returning(tool_calls: list | None, content: str | None = None) -> MagicMock:
    message = MagicMock()
    message.tool_calls = tool_calls
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
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
