from __future__ import annotations

import pytest

from agent.config import (
    get_config,
    get_llm_profile,
    reset_config,
    resolved_config_snapshot,
)


@pytest.fixture(autouse=True)
def reset_config_cache() -> None:
    reset_config()
    yield
    reset_config()


def test_proactive_profile_defaults_to_main_llm_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "main-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("LLM_PROACTIVE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROACTIVE_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROACTIVE_MODEL", raising=False)
    reset_config()

    profile = get_llm_profile("proactive")

    assert profile.api_key == "main-key"
    assert profile.base_url == "https://api.openai.com/v1"
    assert profile.model == "gpt-4o-mini"


def test_proactive_profile_can_override_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "main-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_PROACTIVE_API_KEY", "proactive-key")
    monkeypatch.setenv("LLM_PROACTIVE_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_PROACTIVE_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_PROACTIVE_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_PROACTIVE_MAX_TOKENS", "128")
    reset_config()

    profile = get_llm_profile("proactive")

    assert profile.api_key == "proactive-key"
    assert profile.base_url == "https://api.deepseek.com/v1"
    assert profile.model == "deepseek-chat"
    assert profile.temperature == 0.0
    assert profile.max_tokens == 128


def test_agent_role_reads_role_specific_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "main-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_AGENT_MODEL", "gpt-4.1-mini")
    reset_config()

    profile = get_llm_profile("agent")

    assert profile.model == "gpt-4.1-mini"
    assert profile.api_key == "main-key"


def test_disable_reasoning_defaults_to_agent_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_AGENT_DISABLE_REASONING", raising=False)
    monkeypatch.delenv("LLM_PLANNER_DISABLE_REASONING", raising=False)
    reset_config()

    assert get_llm_profile("agent").disable_reasoning is True
    assert get_llm_profile("planner").disable_reasoning is False


def test_disable_reasoning_role_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_AGENT_DISABLE_REASONING", "false")
    monkeypatch.setenv("LLM_PLANNER_DISABLE_REASONING", "yes")
    reset_config()

    assert get_llm_profile("agent").disable_reasoning is False
    assert get_llm_profile("planner").disable_reasoning is True


def test_pipeline_and_transport_settings_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIPELINE_MAX_RETRIES", "3")
    monkeypatch.setenv("PIPELINE_VERBOSE", "false")
    monkeypatch.setenv("LLM_CONNECT_TIMEOUT_S", "7.5")
    monkeypatch.setenv("LLM_READ_TIMEOUT_S", "120")
    monkeypatch.setenv("LLM_SDK_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_ATTEMPTS", "4")
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_BACKOFF_S", "0.5,1.5,3")
    reset_config()

    cfg = get_config()

    assert cfg.pipeline.max_retries == 3
    assert cfg.pipeline.verbose is False
    assert cfg.llm.connect_timeout_s == 7.5
    assert cfg.llm.read_timeout_s == 120.0
    assert cfg.llm.sdk_max_retries == 1
    assert cfg.llm.transient_retry_attempts == 4
    assert cfg.llm.transient_retry_backoff_s == (0.5, 1.5, 3.0)


def test_canonical_dataset_paths_override_legacy_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    monkeypatch.setenv("DATASET_ICENTIA11K_ROOT", str(canonical))
    monkeypatch.setenv("ICENTIA11K_DATASET_ROOT", str(legacy))
    reset_config()

    assert get_config().datasets.icentia11k_root == canonical


def test_resolved_config_snapshot_excludes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-appear")
    monkeypatch.setenv("LLM_AGENT_MODEL", "agent-model")
    monkeypatch.setenv("LLM_PROACTIVE_TEMPERATURE", "0.2")
    reset_config()

    snapshot = resolved_config_snapshot()

    assert snapshot["llm_profiles"]["agent"]["model"] == "agent-model"
    assert snapshot["llm_profiles"]["proactive"]["temperature"] == 0.2
    assert "must-not-appear" not in repr(snapshot)
    assert "api_key" not in repr(snapshot)
