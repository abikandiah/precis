import os
import subprocess
import sys

import pytest

from precis.config import ConfigError, Settings


@pytest.mark.parametrize(
    ("name", "value", "message", "read"),
    [
        ("PRECIS_LLM_CALL_TIMEOUT_SECONDS", "0", "at least 1", lambda s: s.llm_call_timeout_seconds),
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
    assert settings.llm_max_retries == 5


def _precis(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from precis.cli import main; main()", *args],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_bad_setting_fails_only_the_commands_that_use_it_before_any_paid_work(tmp_path):
    bad = {"PRECIS_LLM_MAX_RETRIES": "abc", "PRECIS_CACHE_DIR": str(tmp_path / "cache")}
    assert _precis("tags", **bad).returncode == 0
    known = tmp_path / "book.json"
    known.write_text('{"isbn": "1", "title": "A Book", "author": "An Author", "kind": "non-fiction"}')
    generated = _precis("generate", str(known), **bad)
    assert generated.returncode == 2
    assert "precis: PRECIS_LLM_MAX_RETRIES must be a whole number" in generated.stderr
    assert "Traceback" not in generated.stderr
    assert not (tmp_path / "cache").exists()  # nothing searched
