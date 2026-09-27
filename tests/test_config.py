import pytest

from precis.config import Settings


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_non_positive_timeout_fails_at_load(monkeypatch, value):
    monkeypatch.setenv("PRECIS_LLM_CALL_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="PRECIS_LLM_CALL_TIMEOUT_SECONDS"):
        Settings()


def test_settings_are_read_from_the_environment_at_instantiation(monkeypatch):
    monkeypatch.setenv("PRECIS_LLM_CALL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("PRECIS_CACHE_DIR", "/data/cache")
    settings = Settings()
    assert (settings.llm_call_timeout_seconds, settings.cache_dir) == (60, "/data/cache")
