import os
import subprocess
import sys
from pathlib import Path

import pytest

from precis.config import ConfigError, Settings


@pytest.mark.parametrize(
    ("name", "value", "message", "read"),
    [
        ("PRECIS_LLM_CALL_TIMEOUT_SECONDS", "0", "at least 1", lambda s: s.llm_call_timeout_seconds),
        ("PRECIS_CONCURRENCY", "-1", "at least 1", lambda s: s.concurrency),
        ("PRECIS_LLM_MAX_RETRIES", "abc", "a whole number", lambda s: s.llm_max_retries),
        ("PRECIS_LLM_MAX_RETRIES", "-1", "at least 0", lambda s: s.llm_max_retries),
    ],
)
def test_a_bad_number_fails_when_read_not_when_loaded(monkeypatch, name, value, message, read):
    monkeypatch.setenv(name, value)
    settings = Settings()
    with pytest.raises(ConfigError, match=f"{name} must be {message}"):
        read(settings)


def test_settings_are_read_from_the_environment_at_instantiation(monkeypatch):
    monkeypatch.setenv("PRECIS_LLM_CALL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("PRECIS_CACHE_DIR", "/data/cache")
    settings = Settings()
    assert (settings.llm_call_timeout_seconds, settings.cache_dir) == (60, "/data/cache")
    assert (settings.concurrency, settings.llm_max_retries) == (3, 5)


def _precis(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from precis.cli import main; main()", *args],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_bad_setting_fails_only_the_commands_that_use_it_and_never_with_a_traceback():
    evals = str(Path(__file__).parents[1] / "evals")
    bad = {"PRECIS_CONCURRENCY": "0", "PRECIS_JUDGE_MODEL": "judge/model", "PRECIS_EVALS_DIR": evals}
    assert _precis("tags", **bad).returncode == 0
    judged = _precis("eval", "judge", "a", "b", **bad)
    assert judged.returncode != 0
    assert "PRECIS_CONCURRENCY must be at least 1 (got 0)" in judged.stderr
    assert "Traceback" not in judged.stderr
