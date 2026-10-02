"""Env-driven settings. All of it — LLM gateway, search, budgets — comes
from the environment, never hard-coded, so a consumer (or a future desktop
app settings UI) can supply its own without touching this module's code.

Every field uses `default_factory`, not a bare default expression: a bare
`field: str = os.environ.get(...)` is evaluated exactly once, at class
*definition* time (import time), and that single frozen value would then be
reused for every `Settings()` instance for the rest of the process.
`default_factory` reads the environment on every instantiation instead, so
a fresh `Settings()` (as the tests build) sees the environment as it is
then. The module-level `settings` every module uses is still built once, at
first import: env vars set after that don't reach it.

Numbers are kept as the environment's text and parsed when read, not at
import: `settings` is built when this module is first imported, so a
malformed PRECIS_LLM_MAX_RETRIES parsed there would crash every command with a
traceback — `precis tags` included — before the CLI could report it.
Parsed on read, it fails only the command that uses it, as a ConfigError
the CLI prints cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


class ConfigError(ValueError):
    """A setting's environment variable holds a value precis can't use."""


def _int(name: str, raw: str, default: int, *, minimum: int) -> int:
    """`raw` as a whole number of at least `minimum`, or `default` when
    unset. A timeout of 0 would time every call out at once, so it needs at
    least 1.
    """
    if not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number (got {raw!r})") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum} (got {value})")
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
    llm_max_retries_env: str = field(default_factory=lambda: _env_str("PRECIS_LLM_MAX_RETRIES", ""))

    # Search: provider-agnostic key name on purpose — swapping the concrete
    # SearchClient implementation (see search.py) shouldn't require
    # renaming this.
    search_api_key: str = field(default_factory=lambda: _env_str("PRECIS_SEARCH_API_KEY", ""))

    # Research cache: each book's raw search results, kept so a rerun
    # doesn't search again (see research.py). In Docker this is on the
    # /data volume; the relative default is for local/dev runs.
    cache_dir: str = field(default_factory=lambda: _env_str("PRECIS_CACHE_DIR", ".precis/cache"))

    # A circuit breaker for a hung call, not a limit meant to bind — the
    # write call sets its own, longer one (write.py).
    llm_call_timeout_seconds_env: str = field(
        default_factory=lambda: _env_str("PRECIS_LLM_CALL_TIMEOUT_SECONDS", "")
    )

    @property
    def llm_max_retries(self) -> int:
        return _int("PRECIS_LLM_MAX_RETRIES", self.llm_max_retries_env, 5, minimum=0)

    @property
    def llm_call_timeout_seconds(self) -> int:
        return _int("PRECIS_LLM_CALL_TIMEOUT_SECONDS", self.llm_call_timeout_seconds_env, 120, minimum=1)

    def check_llm(self) -> None:
        """Raises ConfigError for a malformed model-call setting — for a
        command to call before any paid work, since the client reads them
        only once it's built.
        """
        _ = self.llm_max_retries, self.llm_call_timeout_seconds


settings = Settings()
