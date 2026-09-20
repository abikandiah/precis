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

from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam

from precis.config import settings

_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)


class TransientLLMError(Exception):
    """Raised once the client's own built-in retries are exhausted."""


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
