import pytest

from precis.config import Settings


@pytest.mark.parametrize("value", ["0", "-1", "-100"])
@pytest.mark.parametrize("env_var", ["PRECIS_RUN_BUDGET_SECONDS", "PRECIS_LLM_CALL_TIMEOUT_SECONDS"])
def test_non_positive_timeout_env_vars_raise_clearly(monkeypatch, env_var, value):
    """PRECIS_RUN_BUDGET_SECONDS=0 used to silently make every run fail
    instantly (asyncio.wait_for(coro, timeout=0)) instead of failing loudly
    at config-load time with a message pointing at the actual problem.
    """
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ValueError, match=env_var):
        Settings()


def test_positive_timeout_env_vars_are_accepted(monkeypatch):
    monkeypatch.setenv("PRECIS_RUN_BUDGET_SECONDS", "1800")
    monkeypatch.setenv("PRECIS_LLM_CALL_TIMEOUT_SECONDS", "60")
    settings = Settings()
    assert settings.run_budget_seconds == 1800
    assert settings.llm_call_timeout_seconds == 60


def test_defaults_are_positive_when_unset(monkeypatch):
    monkeypatch.delenv("PRECIS_RUN_BUDGET_SECONDS", raising=False)
    monkeypatch.delenv("PRECIS_LLM_CALL_TIMEOUT_SECONDS", raising=False)
    settings = Settings()
    assert settings.run_budget_seconds == 60 * 60
    assert settings.llm_call_timeout_seconds == 120
