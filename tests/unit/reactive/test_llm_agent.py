from __future__ import annotations

import pytest

from agent.config import reset_config
from agent.reactive import llm_agent


@pytest.fixture(autouse=True)
def reset_config_cache() -> None:
    reset_config()
    yield
    reset_config()


def test_build_answer_system_prompt_includes_ecg_rules_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Does this ECG show atrial fibrillation?",
    )

    assert "weighted evidence" in prompt
    assert "match_scope == specific" in prompt
    assert "do not treat that absence as proof" in prompt
    assert "covers relatively well" in prompt


def test_build_answer_system_prompt_omits_ecg_rules_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Does this ECG show atrial fibrillation?",
    )

    assert "weighted evidence" not in prompt
    assert "match_scope == specific" not in prompt


def test_build_answer_system_prompt_marks_low_trust_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Does this ECG show symptoms of inverted t-waves in lead aVL?",
    )

    assert "lead-specific, form-related, interval-focused, or region-qualified" in prompt
    assert "Prioritize `get_ecg_description`" in prompt


def test_build_answer_system_prompt_uses_single_lead_note_for_lead_i_high_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Does this ECG show symptoms of digitalis effect in lead I?",
    )

    assert "single-lead mode on Lead I" in prompt
    assert "strong evidence for Lead I" in prompt


def test_build_answer_system_prompt_respects_uncertainty_qualifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "What diagnostic symptoms does this ECG show, including uncertain symptoms?",
    )

    assert "including uncertain symptoms" in prompt
    assert "weak candidates" in prompt


def test_build_answer_system_prompt_includes_ppg_limitations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Is this pulse signal irregular?",
    )

    assert "PPG / pulse-signal evidence" in prompt
    assert "does NOT by itself confirm an ECG diagnosis" in prompt


def test_build_answer_system_prompt_enforces_answer_labels_and_rhythm_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Has the rhythm changed compared with the previous window?",
    )

    assert "first line MUST be exactly one of those labels" in prompt
    assert 'Do not answer "none" unless "none" is one of the allowed labels' in prompt
    assert "indeterminate or insufficient evidence should be answered as \"No\"" in prompt
    assert "NOT on heart-rate mean or interval-duration changes alone" in prompt


def test_build_answer_system_prompt_requires_confirmatory_rhythm_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    prompt = llm_agent.build_answer_system_prompt(
        "Does this window show AF, and has the rhythm changed?",
    )

    assert "RR/PP-interval irregularity metrics" in prompt
    assert "signal_quality_score > 0.6" in prompt
    assert "quality_label" in prompt
    assert "`rhythm_screen` differ" in prompt
    assert "RR-CoV deltas under 0.10" in prompt
    assert "if categorical fields are unchanged, answer \"No\"" in prompt
