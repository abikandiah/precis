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
that don't validate — something every caller of structured output wants,
so it lives here rather than per call site.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Callable, Sequence
from contextvars import ContextVar
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

from precis import usage
from precis.config import settings

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)

# OpenRouter's provider-routing preference: only route to providers that
# support every parameter sent (here, `tools` + a forced `tool_choice`).
# Without it, a model id served by several providers can land on one
# that ignores tool_choice and replies in prose. Gateways that don't know the field ignore it.
_REQUIRE_TOOL_SUPPORT = {"provider": {"require_parameters": True}}

# Asks OpenRouter to report each call's cost in `usage.cost` (see usage.py).
# Gateways that don't know the field ignore it.
_USAGE_ACCOUNTING = {"usage": {"include": True}}

# A provider error returned inside a 200 response (see _create) is invisible
# to the SDK's retry, so it gets its own small, fixed budget — deliberately
# not PRECIS_LLM_MAX_RETRIES, since each of these attempts already carries
# the SDK's full retry budget for HTTP-level failures, and the two would
# multiply. Backoff mirrors the SDK's: exponential, capped, jittered down by
# up to 25% so parallel calls don't retry in lockstep.
_PROVIDER_ERROR_RETRIES = 3
_PROVIDER_ERROR_INITIAL_DELAY_SECONDS = 0.5
_PROVIDER_ERROR_MAX_DELAY_SECONDS = 8.0

# How much of a non-tool-call reply to quote in StructuredOutputError.
_REPLY_EXCERPT_CHARS = 200

# How much of a validation error to hand back to the model on a retry.
_VALIDATION_FEEDBACK_CHARS = 2_000

# How much of a retry's reason to report through `on_retry`: enough to see
# what was wrong, short enough for one progress line.
_RETRY_REASON_CHARS = 300

# Told why a call is being retried, for progress output.
RetryCallback = Callable[[str], None]


def _no_retry_report(_: str) -> None:
    pass


# The SDK's own HTTP retries (rate limits, 5xx, timeouts, dropped
# connections) happen inside one `create` call, out of this module's sight
# except through the SDK's log. _create puts its call's `on_retry` here, and
# _SDKRetryReporter turns the SDK's log lines into reports to it.
_http_retry_report: ContextVar[RetryCallback | None] = ContextVar("precis_http_retry_report", default=None)
_http_retry_reason: ContextVar[str] = ContextVar("precis_http_retry_reason", default="")


class _SDKRetryReporter(logging.Handler):
    """Reports the openai SDK's HTTP retries to the current call's
    `on_retry`, with what went wrong. Reads the SDK's log messages, so an
    SDK that rewords them only stops the reports; nothing breaks.
    """

    def emit(self, record: logging.LogRecord) -> None:
        report = _http_retry_report.get()
        args = record.args if isinstance(record.args, tuple) else ()
        if report is None or not isinstance(record.msg, str):
            return
        if record.msg.startswith("Encountered a timeout exception"):
            _http_retry_reason.set("timed out")
        elif record.msg.startswith("Encountered an HTTP status error") and args:
            _http_retry_reason.set(f"HTTP {args[0]}")
        elif record.msg.startswith("Encountered exception") and args:
            _http_retry_reason.set(f"connection error: {args[0]}")
        elif record.msg.startswith("Retrying request in") and len(args) == 3:
            delay, attempt, total = args
            reason = _http_retry_reason.get()
            _http_retry_reason.set("")
            report(f"HTTP retry {attempt} of {total} in {delay:.1f}s" + (f" ({reason})" if reason else ""))


def report_sdk_retries() -> None:
    """Routes the SDK's HTTP retries to each call's `on_retry`. For the CLI,
    which owns the process's logging: it sets the SDK's logger to DEBUG (to
    see why a retry happens) and stops it propagating, so nothing else
    prints those records.
    """
    sdk_log = logging.getLogger("openai._base_client")
    if not any(isinstance(h, _SDKRetryReporter) for h in sdk_log.handlers):
        sdk_log.addHandler(_SDKRetryReporter())
        sdk_log.setLevel(logging.DEBUG)
        sdk_log.propagate = False


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
    """OpenRouter puts the upstream provider's message (e.g. a model's
    shared pool being rate-limited) in the error's `metadata.raw`, which is
    far more actionable than its generic "Provider returned error".
    """
    metadata = error.get("metadata")
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    message = raw or error.get("message")
    return str(message) if message else None


def _status_detail(exc: APIStatusError) -> str:
    body = exc.body if isinstance(exc.body, dict) else {}
    return f"HTTP {exc.status_code}: {_gateway_error_message(body) or exc.message}"


def _transient_error(client: AsyncOpenAI, exc: Exception) -> TransientLLMError:
    """Wraps the last transient failure with what actually went wrong —
    the status and the gateway's own explanation.
    """
    detail = _status_detail(exc) if isinstance(exc, APIStatusError) else str(exc)
    return TransientLLMError(
        f"LLM call failed after exhausting the client's {client.max_retries} built-in retries ({detail})"
    )


class StructuredOutputError(Exception):
    """Raised when the model doesn't return the requested structured
    output at all (no tool call, or a tool call that fails schema
    validation). Distinct from TransientLLMError: this is a content-quality
    problem, not a technical one, so a rerun won't necessarily get past it.
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


async def _create(client: AsyncOpenAI, on_retry: RetryCallback = _no_retry_report, **kwargs: Any) -> ChatCompletion:
    """One chat-completions request, with a provider error returned inside
    a 200 response treated as the failure it is (see _provider_error). The
    SDK's retry never sees these, and to complete_structured an empty
    errored choice would look like the model declining to call the tool.
    Retryable ones get _PROVIDER_ERROR_RETRIES retries with backoff, then
    TransientLLMError; the rest raise ProviderError at once.

    Every response is recorded to the current usage scope, errored ones
    included — a provider can bill for a generation that failed partway.
    `on_retry` is told about each retry, the SDK's own HTTP retries included
    once report_sdk_retries() is on. Any other HTTP error (a 400 for a
    context that's too long, a 402 for credits) is a ProviderError, so
    callers handle one error type for "the provider refused".
    """
    kwargs["extra_body"] = {**_USAGE_ACCOUNTING, **kwargs.get("extra_body", {})}
    for retry in range(_PROVIDER_ERROR_RETRIES + 1):
        token = _http_retry_report.set(on_retry)
        try:
            response = await client.chat.completions.create(**kwargs)
        except _TRANSIENT_ERRORS as exc:
            raise _transient_error(client, exc) from exc
        except APIStatusError as exc:
            raise ProviderError(f"LLM request refused ({_status_detail(exc)})") from exc
        finally:
            _http_retry_report.reset(token)
        usage.record_llm_response(response)
        error = _provider_error(response)
        if error is None:
            return response
        detail = _describe_provider_error(response, error)
        if not _is_retryable(error):
            raise ProviderError(f"LLM provider returned an error that retrying won't fix ({detail})")
        if retry < _PROVIDER_ERROR_RETRIES:
            on_retry(f"provider error, retrying ({_excerpt(detail, _RETRY_REASON_CHARS)})")
            await _sleep(_backoff_seconds(retry))
    raise TransientLLMError(
        f"LLM call failed after {_PROVIDER_ERROR_RETRIES} retries of a provider error returned in a 200 response "
        f"({detail})"
    )


def _tool_name(model: type[BaseModel]) -> str:
    return f"emit_{model.__name__.lower()}"


def _tool(model: type[BaseModel]) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": FunctionDefinition(
            name=_tool_name(model),
            description=f"Emit the {model.__name__} result.",
            parameters=model.model_json_schema(),
        ),
    }


def _excerpt(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def _rejection(call: ChatCompletionMessageFunctionToolCall, reason: str) -> list[ChatCompletionMessageParam]:
    """The rejected tool call and why it was rejected, as the turn a retry
    continues from.
    """
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": reason,
        },
    ]


def _describe_error(error: Any) -> str:
    """One pydantic error as "where: what", so a retry report names the
    field ("ideas.3.sources: …"), not just "Field required".
    """
    where = ".".join(str(part) for part in error["loc"])
    return f"{where}: {error['msg']}" if where else error["msg"]


def _parses(arguments: str) -> bool:
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return True


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
    timeout_seconds: float | None = None,
) -> str:
    """A plain-text reply. `timeout_seconds` overrides
    PRECIS_LLM_CALL_TIMEOUT_SECONDS, for a call known to take long.
    """
    response = await _create(
        client,
        model=model or settings.llm_model,
        messages=messages,
        timeout=settings.llm_call_timeout_seconds if timeout_seconds is None else timeout_seconds,
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
    timeout_seconds: float | None = None,
    max_tokens: int | None = None,
    tool_models: Sequence[type[BaseModel]] | None = None,
    on_retry: RetryCallback = _no_retry_report,
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
    retry (write.py's per-kind counts and citation IDs). A retry after a validation failure shows the model its rejected
    call and the error, so it can fix the specific problem rather than
    rolling the dice again.

    Only a call that parsed but didn't validate is shown back: unparseable
    arguments (a reply cut off mid-JSON) can't be echoed as a tool call —
    providers that parse tool input reject the request — so that retry
    resends the original request.

    `timeout_seconds` overrides PRECIS_LLM_CALL_TIMEOUT_SECONDS, and
    `max_tokens` caps the reply, for a call known to generate long output.

    `tool_models` is the full list of tools to send, `response_model`'s
    among them; the call is still forced to `response_model`'s. Calls that
    share a cached prompt prefix must send identical tools, since a
    provider's cache prefix starts with the tool definitions.

    `on_retry` is told why each retry happens — a rejected attempt or a
    provider error — so a run's progress shows what a retry cost, and why.
    """
    models = list(tool_models or [response_model])
    if response_model not in models:
        raise ValueError(f"tool_models must include the response model {response_model.__name__}")
    tool_name = _tool_name(response_model)
    tools = [_tool(model) for model in models]
    tool_choice: ChatCompletionNamedToolChoiceParam = {"type": "function", "function": {"name": tool_name}}

    attempt_messages = list(messages)
    for attempt in range(max_attempts):
        response = await _create(
            client,
            model=model or settings.llm_model,
            messages=attempt_messages,
            on_retry=on_retry,
            tools=tools,
            tool_choice=tool_choice,
            timeout=settings.llm_call_timeout_seconds if timeout_seconds is None else timeout_seconds,
            extra_body=_REQUIRE_TOOL_SUPPORT,
            **({} if max_tokens is None else {"max_tokens": max_tokens}),
        )

        choice = response.choices[0]
        calls = [c for c in choice.message.tool_calls or [] if isinstance(c, ChatCompletionMessageFunctionToolCall)]
        call = next((c for c in calls if c.function.name == tool_name), None)
        if call is None:
            # Another of the tools sent, despite the forced choice: validating
            # its arguments as the response model would be meaningless.
            wrong = calls[0] if calls else None
            if wrong is not None:
                did = f"called {wrong.function.name!r} instead"
            else:
                did = f"replied {_excerpt((choice.message.content or '').strip(), _REPLY_EXCERPT_CHARS)!r}"
            if attempt == max_attempts - 1:
                raise StructuredOutputError(
                    f"model did not call the expected tool {tool_name!r} after {max_attempts} attempts "
                    f"(model {response.model!r}, finish_reason {choice.finish_reason!r}, {did})"
                )
            on_retry(f"attempt {attempt + 1} rejected, retrying: didn't call {tool_name!r}, {did}")
            if wrong is not None and _parses(wrong.function.arguments):
                reason = f"Rejected — that's the wrong tool. Call {tool_name} with the whole result."
                attempt_messages = [*messages, *_rejection(wrong, reason)]
            else:
                attempt_messages = list(messages)
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
            if isinstance(exc, ValidationError):
                problem = "; ".join(_describe_error(e) for e in exc.errors())
                error = _excerpt(str(exc), _VALIDATION_FEEDBACK_CHARS)
                reason = f"Rejected — the result didn't validate:\n{error}\n\nCall {tool_name} again with the whole result, corrected."
                attempt_messages = [*messages, *_rejection(call, reason)]
            else:
                problem = f"its arguments weren't valid JSON (finish_reason {choice.finish_reason!r})"
                attempt_messages = list(messages)
            on_retry(f"attempt {attempt + 1} rejected, retrying: {_excerpt(problem, _RETRY_REASON_CHARS)}")
            continue

    raise StructuredOutputError(f"complete_structured called with max_attempts={max_attempts} < 1")
