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


def _env_positive_int(name: str, default: int) -> int:
    """For a timeout or a concurrency limit, where 0 or a negative number
    would time every call out at once or hang: fails loudly at
    settings-load time instead.
    """
    value = _env_int(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number (got {value})")
    return value


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

    # `precis eval` only: the model that judges two runs against each other
    # (a stronger one than the models under test), and where the eval set
    # and its runs live — see evals/README.md.
    judge_model: str = field(default_factory=lambda: _env_str("PRECIS_JUDGE_MODEL", ""))
    evals_dir: str = field(default_factory=lambda: _env_str("PRECIS_EVALS_DIR", "evals"))

    # Search: provider-agnostic key name on purpose — swapping the concrete
    # SearchClient implementation (see search.py) shouldn't require
    # renaming this.
    search_api_key: str = field(default_factory=lambda: _env_str("PRECIS_SEARCH_API_KEY", ""))

    # Research cache: each book's raw search results, kept so a rerun
    # doesn't search again (see research.py). In Docker this is on the
    # /data volume; the relative default is for local/dev runs.
    cache_dir: str = field(default_factory=lambda: _env_str("PRECIS_CACHE_DIR", ".precis/cache"))

    # `precis eval judge` only: how many books are judged at once.
    concurrency: int = field(default_factory=lambda: _env_positive_int("PRECIS_CONCURRENCY", 3))

    # A circuit breaker for a hung call, not a limit meant to bind — the
    # write call sets its own, longer one (write.py).
    llm_call_timeout_seconds: int = field(
        default_factory=lambda: _env_positive_int("PRECIS_LLM_CALL_TIMEOUT_SECONDS", 120)
    )


settings = Settings()
