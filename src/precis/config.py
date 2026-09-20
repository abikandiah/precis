"""Env-driven settings. All of it — LLM gateway, search, budgets — comes
from the environment, never hard-coded, so a consumer (or a future desktop
app settings UI) can supply its own without touching this module's code.

Every field uses `default_factory`, not a bare default expression: a bare
`field: str = os.environ.get(...)` is evaluated exactly once, at class
*definition* time (import time), and that single frozen value would then be
reused for every `Settings()` instance for the rest of the process — env
vars set after the first import of this module (e.g. by an embedding
application configuring its own environment before calling into precis)
would be silently ignored. `default_factory` re-reads the environment on
every instantiation instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # LLM gateway: OpenRouter-style, base URL + key + model via env vars,
    # not a vendor-specific SDK — see docs/blueprint.md's Infrastructure
    # decisions. The `openai` package is used purely as an HTTP client for
    # this wire format, which OpenRouter and most gateways implement.
    llm_base_url: str = field(default_factory=lambda: _env_str("PRECIS_LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    llm_api_key: str = field(default_factory=lambda: _env_str("PRECIS_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env_str("PRECIS_LLM_MODEL", ""))
    llm_max_retries: int = field(default_factory=lambda: _env_int("PRECIS_LLM_MAX_RETRIES", 5))

    # Search: provider-agnostic key name on purpose — swapping the concrete
    # SearchClient implementation (see search.py) shouldn't require
    # renaming this.
    search_api_key: str = field(default_factory=lambda: _env_str("PRECIS_SEARCH_API_KEY", ""))

    # Stage 2 fan-out.
    concurrency: int = field(default_factory=lambda: _env_int("PRECIS_CONCURRENCY", 3))

    # Circuit breakers, not constraints meant to bind on a normal run — see
    # docs/blueprint.md's Run budget section.
    run_budget_seconds: int = field(default_factory=lambda: _env_int("PRECIS_RUN_BUDGET_SECONDS", 60 * 60))
    llm_call_timeout_seconds: int = field(default_factory=lambda: _env_int("PRECIS_LLM_CALL_TIMEOUT_SECONDS", 120))

    # Checkpoint DB path — in the real generation container this must be
    # overridden to a path on a volume mounted into the container, not its
    # own ephemeral filesystem, or resume across container restarts doesn't
    # work. Concrete volume mount is finalized with the generation image's
    # Dockerfile/compose (still open per docs/blueprint.md). The default
    # below is relative-to-cwd so local/dev runs work without that mount
    # already existing.
    checkpoint_db_path: str = field(
        default_factory=lambda: _env_str("PRECIS_CHECKPOINT_DB_PATH", ".precis/checkpoints.sqlite")
    )


settings = Settings()
