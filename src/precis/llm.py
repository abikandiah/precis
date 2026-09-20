"""Provider-agnostic LLM gateway client.

Uses the `openai` package purely as an HTTP client for the OpenAI-compatible
chat-completions wire format that OpenRouter (and most gateways) implement —
not a vendor-specific SDK. Base URL, key, and model all come from env vars
via config.settings.

Retries for transient/technical failures (rate limits, 5xx, dropped
connections) are handled by AsyncOpenAI's own built-in retry-with-backoff
(configured via `max_retries` below) rather than hand-rolled here — the SDK
already does this correctly, including respecting a `Retry-After` header and
jittering, which a hand-rolled loop would have to reimplement to match.
Content-quality retries (critique rejects a draft) are a completely separate
loop that lives in the pipeline's Stage 2, not here — see docs/blueprint.md's
Stage 2 section for why the two must stay apart.
"""

from __future__ import annotations

import json

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, ValidationError

from precis.config import settings

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)


class TransientLLMError(Exception):
    """Raised once the client's own built-in retries are exhausted."""


class StructuredOutputError(Exception):
    """Raised when the model doesn't return the requested structured
    output at all (no tool call, or a tool call that fails schema
    validation). Distinct from TransientLLMError: this is a content-quality
    problem, not a technical one, so pipeline retry loops (e.g. Stage 2's
    repair-and-retry) should catch this separately and decide whether to
    retry the content, not the request.
    """


def build_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        max_retries=settings.llm_max_retries,
    )


async def complete(
    client: AsyncOpenAI,
    *,
    messages: list[ChatCompletionMessageParam],
    model: str | None = None,
) -> str:
    try:
        response = await client.chat.completions.create(
            model=model or settings.llm_model,
            messages=messages,
            timeout=settings.llm_call_timeout_seconds,
        )
    except _TRANSIENT_ERRORS as exc:
        raise TransientLLMError(
            f"LLM call failed after exhausting the client's {settings.llm_max_retries} built-in retries"
        ) from exc
    return response.choices[0].message.content or ""


async def complete_structured[T: BaseModel](
    client: AsyncOpenAI,
    *,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    model: str | None = None,
) -> T:
    """Forces schema-shaped output via tool-calling rather than free-form
    JSON-in-prose: `response_model`'s own JSON schema becomes the tool's
    parameters, and the model is required to call it. Tool-calling is
    broadly supported across modern models via OpenRouter — much stronger
    structure adherence than asking nicely in the prompt, while staying
    provider-agnostic (unlike OpenAI-specific structured-output modes).
    """
    tool_name = f"emit_{response_model.__name__.lower()}"
    tool: ChatCompletionFunctionToolParam = {
        "type": "function",
        "function": FunctionDefinition(
            name=tool_name,
            description=f"Emit the {response_model.__name__} result.",
            parameters=response_model.model_json_schema(),
        ),
    }
    tool_choice: ChatCompletionNamedToolChoiceParam = {"type": "function", "function": {"name": tool_name}}

    try:
        response = await client.chat.completions.create(
            model=model or settings.llm_model,
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice,
            timeout=settings.llm_call_timeout_seconds,
        )
    except _TRANSIENT_ERRORS as exc:
        raise TransientLLMError(
            f"LLM call failed after exhausting the client's {settings.llm_max_retries} built-in retries"
        ) from exc

    tool_calls = response.choices[0].message.tool_calls or []
    call = tool_calls[0] if tool_calls else None
    if call is None or not isinstance(call, ChatCompletionMessageFunctionToolCall):
        raise StructuredOutputError(f"model did not call the expected tool {tool_name!r} — no structured output returned")

    try:
        arguments = json.loads(call.function.arguments)
        return response_model.model_validate(arguments)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(f"model's tool call for {tool_name!r} didn't match {response_model.__name__}: {exc}") from exc
