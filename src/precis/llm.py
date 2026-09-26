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
The one exception is a provider error OpenRouter returns inside a 200
response (finish_reason "error", or an error body with no choices) —
invisible to the SDK's retry, so `_create` retries those itself, on a small
budget of its own, and only when the error is one a retry could fix.
`complete_structured` also retries a small, fixed number of times when the
model fails to call the requested tool at all, or calls it with arguments
that don't validate — a horizontal concern every caller of structured
output wants, so it lives here rather than being re-implemented per stage.
This is distinct from a stage's own content-feedback repair loop (e.g.
Stage 2's critique-and-redraft): that stays specific to whatever prompt
shape each stage redrafts, since "revise using this feedback" isn't a
generic operation the way "the tool call didn't parse, try again" is. See
docs/blueprint.md's Stage 2 section for the fuller reasoning.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, ValidationError

from precis.config import settings

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)

# OpenRouter's provider-routing preference: only route to providers that
# support every parameter sent (here, `tools` + a forced `tool_choice`).
# Without it, a model id served by several providers — or a router like
# `openrouter/free` — can land on one that ignores tool_choice and replies
# in prose. Gateways that don't know the field ignore it.
_REQUIRE_TOOL_SUPPORT = {"provider": {"require_parameters": True}}

# A provider error returned inside a 200 response (see _create) is invisible
# to the SDK's retry, so it gets its own small, fixed budget — deliberately
# not PRECIS_LLM_MAX_RETRIES, since each of these attempts already carries
# the SDK's full retry budget for HTTP-level failures, and the two would
# multiply. Backoff mirrors the SDK's: exponential, capped, jittered down by
# up to 25% so parallel chapter drafts don't retry in lockstep.
_PROVIDER_ERROR_RETRIES = 3
_PROVIDER_ERROR_INITIAL_DELAY_SECONDS = 0.5
_PROVIDER_ERROR_MAX_DELAY_SECONDS = 8.0

# How much of a non-tool-call reply to quote in StructuredOutputError.
_REPLY_EXCERPT_CHARS = 200


class TransientLLMError(Exception):
    """Raised when a temporary failure outlasts its retries: either the
    client's built-in retries for an HTTP-level failure (rate limit, 5xx,
    dropped connection), or _create's own for a retryable provider error
    returned inside a 200 response.
    """


class ProviderError(Exception):
    """Raised when the provider reports, inside a 200 response, an error
    that retrying the same request won't fix (e.g. context length exceeded,
    a moderation refusal). Distinct from TransientLLMError so it isn't
    mistaken for something a rerun would get past.
    """


def _gateway_error_message(error: dict[str, Any]) -> str | None:
    """OpenRouter puts the upstream provider's message (e.g. a free model's
    shared pool being rate-limited) in the error's `metadata.raw`, which is
    far more actionable than its generic "Provider returned error".
    """
    metadata = error.get("metadata")
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    message = raw or error.get("message")
    return str(message) if message else None


def _transient_error(exc: Exception) -> TransientLLMError:
    """Wraps the last transient failure with what actually went wrong —
    the status and the gateway's own explanation.
    """
    detail = str(exc)
    if isinstance(exc, APIStatusError):
        body = exc.body if isinstance(exc.body, dict) else {}
        detail = f"HTTP {exc.status_code}: {_gateway_error_message(body) or exc.message}"
    return TransientLLMError(
        f"LLM call failed after exhausting the client's {settings.llm_max_retries} built-in retries ({detail})"
    )


class StructuredOutputError(Exception):
    """Raised when the model doesn't return the requested structured
    output at all (no tool call, or a tool call that fails schema
    validation). Distinct from TransientLLMError: this is a content-quality
    problem, not a technical one, so pipeline retry loops (e.g. Stage 2's
    repair-and-retry) should catch this separately and decide whether to
    retry the content, not the request.
    """


def _provider_error(response: ChatCompletion) -> dict[str, Any] | None:
    """The error a 200 response carries in place of a completion, or None
    if it's a real completion. OpenRouter reports a provider failing
    mid-generation as finish_reason "error" with a non-standard `error`
    object ({code, message, metadata}) on the choice, and one failing
    before generating anything as a body with a top-level `error` and no
    choices. An empty dict when the error has no details.
    """
    if not response.choices:
        error = (response.model_extra or {}).get("error")
    elif response.choices[0].finish_reason == "error":
        error = (response.choices[0].model_extra or {}).get("error")
    else:
        return None
    return error if isinstance(error, dict) else {}


def _is_retryable(error: dict[str, Any]) -> bool:
    """Rate limits, upstream 5xx, and errors with no usable code are worth
    retrying; any other code (400 context length, 403 moderation, ...)
    would fail the same way again.
    """
    try:
        code = int(error["code"])
    except (KeyError, TypeError, ValueError):
        return True
    return code == 429 or code >= 500


def _describe_provider_error(response: ChatCompletion, error: dict[str, Any]) -> str:
    message = _gateway_error_message(error) or "no error details"
    code = error.get("code")
    detail = f"{code}: {message}" if code is not None else message
    return f"model {getattr(response, 'model', None)!r}, {detail}"


def _backoff_seconds(retry: int) -> float:
    delay = min(_PROVIDER_ERROR_INITIAL_DELAY_SECONDS * 2**retry, _PROVIDER_ERROR_MAX_DELAY_SECONDS)
    return delay * (1 - 0.25 * random.random())


# Indirection so tests can skip backoff without patching asyncio.sleep
# for the whole event loop.
_sleep = asyncio.sleep


async def _create(client: AsyncOpenAI, **kwargs: Any) -> ChatCompletion:
    """One chat-completions request, with a provider error returned inside
    a 200 response treated as the failure it is (see _provider_error). The
    SDK's retry never sees these, and to complete_structured an empty
    errored choice would look like the model declining to call the tool.
    Retryable ones get _PROVIDER_ERROR_RETRIES retries with backoff, then
    TransientLLMError; the rest raise ProviderError at once.
    """
    for retry in range(_PROVIDER_ERROR_RETRIES + 1):
        try:
            response = await client.chat.completions.create(**kwargs)
        except _TRANSIENT_ERRORS as exc:
            raise _transient_error(exc) from exc
        error = _provider_error(response)
        if error is None:
            return response
        detail = _describe_provider_error(response, error)
        if not _is_retryable(error):
            raise ProviderError(f"LLM provider returned an error that retrying won't fix ({detail})")
        if retry < _PROVIDER_ERROR_RETRIES:
            await _sleep(_backoff_seconds(retry))
    raise TransientLLMError(
        f"LLM call failed after {_PROVIDER_ERROR_RETRIES} retries of a provider error returned in a 200 response "
        f"({detail})"
    )


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
    response = await _create(
        client,
        model=model or settings.llm_model,
        messages=messages,
        timeout=settings.llm_call_timeout_seconds,
    )
    return response.choices[0].message.content or ""


async def complete_structured[T: BaseModel](
    client: AsyncOpenAI,
    *,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    model: str | None = None,
    max_attempts: int = 2,
    validation_context: dict[str, object] | None = None,
) -> T:
    """Forces schema-shaped output via tool-calling rather than free-form
    JSON-in-prose: `response_model`'s own JSON schema becomes the tool's
    parameters, and the model is required to call it. Tool-calling is
    broadly supported across modern models via OpenRouter — much stronger
    structure adherence than asking nicely in the prompt, while staying
    provider-agnostic (unlike OpenAI-specific structured-output modes).

    Retries the same request up to `max_attempts` times if the model
    doesn't call the tool, or calls it with arguments that don't validate —
    usually one-off noise, not a reason to fail the whole call. Raises
    StructuredOutputError only once that budget is exhausted.

    `validation_context` is passed straight through to `model_validate` —
    it's how a caller wires a runtime-only constraint (something that
    varies per call and can't be a static Field on `response_model`, e.g.
    "must be exactly N items") into a `model_validator` so a mismatch is a
    real schema failure that this retry loop already handles, rather than
    a separate check the caller does after the fact with no chance to
    retry. See synthesize.py's known-parts count check for the motivating
    case.
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

    for attempt in range(max_attempts):
        response = await _create(
            client,
            model=model or settings.llm_model,
            messages=messages,
            tools=[tool],
            tool_choice=tool_choice,
            timeout=settings.llm_call_timeout_seconds,
            extra_body=_REQUIRE_TOOL_SUPPORT,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        call = tool_calls[0] if tool_calls else None
        if call is None or not isinstance(call, ChatCompletionMessageFunctionToolCall):
            if attempt == max_attempts - 1:
                reply = (choice.message.content or "").strip()
                if len(reply) > _REPLY_EXCERPT_CHARS:
                    reply = reply[:_REPLY_EXCERPT_CHARS] + "..."
                raise StructuredOutputError(
                    f"model did not call the expected tool {tool_name!r} after {max_attempts} attempts "
                    f"(model {response.model!r}, finish_reason {choice.finish_reason!r}, replied {reply!r})"
                )
            continue

        try:
            arguments = json.loads(call.function.arguments)
            return response_model.model_validate(arguments, context=validation_context)
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == max_attempts - 1:
                raise StructuredOutputError(
                    f"model's tool call for {tool_name!r} didn't match {response_model.__name__} "
                    f"after {max_attempts} attempts: {exc}"
                ) from exc
            continue

    raise StructuredOutputError(f"complete_structured called with max_attempts={max_attempts} < 1")
