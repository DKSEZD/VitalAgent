from __future__ import annotations

from typing import Any

import pytest

from agent.proactive.intervention_decision import (
    CORE_RULES,
    ECG_EXTRA_RULES,
    PHASE2_RULES,
    PPG_EXTRA_RULES,
    InterventionDecision,
    InterventionResult,
    RuleBasedJudge,
    Modality,
    Urgency,
    rules_for_modality,
)
from agent.proactive.llm_judge import CitedSection, JudgeDecision


def _make_state(
    *,
    window_duration_s: float = 10.0,
    **short_overrides: Any,
) -> dict[str, Any]:
    default_short = {
        "hr_bpm": 72.0,
        "rhythm_class": "N",
        "rhythm_regular": True,
        "af_episode_duration_s": None,
        "tachycardia_ratio_5min": 0.0,
        "tachycardia_sample_count": 30,
    }
    default_short.update(short_overrides)
    return {
        "short": default_short,
        "medium": {},
        "long": {},
        "embedding": {"latest_zscore": 0.0},
        "window_meta": {"window_duration_s": window_duration_s},
    }


class _FakeLLMJudge:
    enabled = False

    def __init__(self, decision: JudgeDecision) -> None:
        self._decision = decision

    def evaluate(self, health_state: dict[str, Any]) -> JudgeDecision:
        del health_state
        return self._decision


def test_default_constructor_uses_ecg_modality() -> None:
    decision = RuleBasedJudge()

    assert decision.modality is Modality.ECG
    assert [rule.rule_id for rule in decision.rules] == [
        rule.rule_id for rule in rules_for_modality(Modality.ECG)
    ]


def test_ppg_modality_constructor_returns_core_rules() -> None:
    decision = RuleBasedJudge(modality=Modality.PPG)

    assert decision.modality is Modality.PPG
    assert [rule.rule_id for rule in decision.rules] == [
        rule.rule_id for rule in CORE_RULES
    ]


def test_rules_for_modality_rejects_unknown_input() -> None:
    with pytest.raises(ValueError):
        rules_for_modality("unknown")  # type: ignore[arg-type]


def test_describe_rules_includes_citation_fields_for_each_rule() -> None:
    decision = RuleBasedJudge()

    descriptions = decision.describe_rules()

    assert len(descriptions) == len(decision.rules)
    for item in descriptions:
        assert {
            "rule_id",
            "description",
            "urgency",
            "modality",
            "guideline_id",
            "section_id",
            "cor",
            "paraphrase",
            "url",
        } <= set(item)


def test_phase_1_core_has_exactly_three_rules() -> None:
    assert [rule.rule_id for rule in CORE_RULES] == [
        "extreme_bradycardia",
        "extreme_tachycardia",
        "sustained_tachycardia",
    ]


def test_phase_1_ecg_extra_is_empty() -> None:
    assert ECG_EXTRA_RULES == ()


def test_phase_1_ppg_extra_is_empty() -> None:
    assert PPG_EXTRA_RULES == ()


def test_phase_2_contains_long_af_episode_rule() -> None:
    assert [rule.rule_id for rule in PHASE2_RULES] == ["long_af_episode"]


def test_extreme_bradycardia_fires_at_25_bpm_for_10_seconds() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=25.0))

    assert result.intervene is True
    assert result.urgency is Urgency.CRITICAL
    assert result.triggered_rules == ["extreme_bradycardia"]


def test_extreme_bradycardia_does_not_fire_at_35_bpm() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=35.0))

    assert result.intervene is False


def test_extreme_bradycardia_does_not_fire_when_hr_missing() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=None))

    assert result.intervene is False


def test_extreme_bradycardia_does_not_fire_below_10_second_window() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=25.0, window_duration_s=5.0))

    assert result.intervene is False


def test_extreme_tachycardia_fires_at_210_bpm_for_10_seconds() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=210.0))

    assert result.intervene is True
    assert result.urgency is Urgency.CRITICAL
    assert result.triggered_rules == ["extreme_tachycardia"]


def test_extreme_tachycardia_does_not_fire_at_195_bpm() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=195.0))

    assert result.intervene is False


def test_extreme_tachycardia_does_not_fire_when_hr_missing() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state(hr_bpm=None))

    assert result.intervene is False


def test_long_af_episode_fires_at_24_hours() -> None:
    decision = RuleBasedJudge(include_phase2=True)

    result = decision.evaluate(
        _make_state(rhythm_class="AF", af_episode_duration_s=24 * 3600)
    )

    assert result.intervene is True
    assert result.urgency is Urgency.HIGH
    assert result.triggered_rules == ["long_af_episode"]


def test_long_af_episode_fires_above_24_hours() -> None:
    decision = RuleBasedJudge(include_phase2=True)

    result = decision.evaluate(
        _make_state(rhythm_class="AF", af_episode_duration_s=25 * 3600)
    )

    assert result.intervene is True
    assert result.triggered_rules == ["long_af_episode"]


def test_long_af_episode_does_not_fire_below_24_hours() -> None:
    decision = RuleBasedJudge(include_phase2=True)

    result = decision.evaluate(
        _make_state(rhythm_class="AF", af_episode_duration_s=23 * 3600)
    )

    assert result.intervene is False


def test_long_af_episode_does_not_fire_for_atrial_flutter() -> None:
    decision = RuleBasedJudge(include_phase2=True)

    result = decision.evaluate(
        _make_state(rhythm_class="AFL", af_episode_duration_s=48 * 3600)
    )

    assert result.intervene is False


def test_long_af_episode_does_not_fire_for_normal_rhythm() -> None:
    decision = RuleBasedJudge(include_phase2=True)

    result = decision.evaluate(
        _make_state(rhythm_class="N", af_episode_duration_s=48 * 3600)
    )

    assert result.intervene is False


def test_long_af_episode_does_not_fire_when_duration_missing() -> None:
    decision = RuleBasedJudge(include_phase2=True)

    result = decision.evaluate(_make_state(rhythm_class="AF", af_episode_duration_s=None))

    assert result.intervene is False


def test_sustained_tachycardia_fires_at_full_ratio() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(
        _make_state(tachycardia_ratio_5min=1.0, tachycardia_sample_count=30)
    )

    assert result.intervene is True
    assert result.urgency is Urgency.MEDIUM
    assert result.triggered_rules == ["sustained_tachycardia"]


def test_sustained_tachycardia_fires_at_exact_boundary() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(
        _make_state(tachycardia_ratio_5min=0.80, tachycardia_sample_count=30)
    )

    assert result.intervene is True
    assert result.triggered_rules == ["sustained_tachycardia"]


def test_sustained_tachycardia_does_not_fire_below_ratio_boundary() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(
        _make_state(tachycardia_ratio_5min=0.79, tachycardia_sample_count=30)
    )

    assert result.intervene is False


def test_sustained_tachycardia_does_not_fire_below_sample_threshold() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(
        _make_state(tachycardia_ratio_5min=1.0, tachycardia_sample_count=29)
    )

    assert result.intervene is False


def test_sustained_tachycardia_does_not_fire_when_ratio_missing() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(
        _make_state(tachycardia_ratio_5min=None, tachycardia_sample_count=30)
    )

    assert result.intervene is False


def test_multi_rule_result_uses_highest_urgency_and_keeps_all_rule_ids() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(
        _make_state(
            hr_bpm=210.0,
            tachycardia_ratio_5min=1.0,
            tachycardia_sample_count=30,
        )
    )

    assert result.intervene is True
    assert result.urgency is Urgency.CRITICAL
    assert result.triggered_rules == [
        "extreme_tachycardia",
        "sustained_tachycardia",
    ]


def test_no_rule_fire_returns_none_urgency() -> None:
    decision = RuleBasedJudge()

    result = decision.evaluate(_make_state())

    assert result.intervene is False
    assert result.urgency is Urgency.NONE
    assert result.triggered_rules is None or result.triggered_rules == []


def test_missing_tachycardia_keys_does_not_raise() -> None:
    decision = RuleBasedJudge()
    state = _make_state()
    del state["short"]["tachycardia_ratio_5min"]
    del state["short"]["tachycardia_sample_count"]

    result = decision.evaluate(state)

    assert result.intervene is False


def test_missing_rhythm_class_does_not_raise() -> None:
    decision = RuleBasedJudge()
    state = _make_state(af_episode_duration_s=24 * 3600)
    del state["short"]["rhythm_class"]

    result = decision.evaluate(state)

    assert result.intervene is False


def test_high_level_intervention_decision_merges_rule_and_llm_layers() -> None:
    rule_judge = RuleBasedJudge(
        rules=(),
    )
    rule_judge.evaluate = lambda state, embedding_zscore=0.0: InterventionResult(  # type: ignore[method-assign]
        intervene=True,
        urgency=Urgency.MEDIUM,
        reason="rule reason",
        triggered_rules=["rule-a"],
    )
    llm_judge = _FakeLLMJudge(
        JudgeDecision(
            intervene=True,
            urgency=Urgency.HIGH,
            reason="llm reason",
            advice="",
            cited_sections=[CitedSection("llm", "b")],
        )
    )

    outcome = InterventionDecision(
        rule_judge=rule_judge,
        llm_judge=llm_judge,
    ).evaluate(_make_state())

    assert outcome.rule_result.triggered_rules == ["rule-a"]
    assert [c.fully_qualified_id() for c in outcome.judge_decision.cited_sections] == ["llm.b"]
    assert outcome.final_result.intervene is True
    assert outcome.final_result.urgency is Urgency.HIGH
    assert outcome.final_result.triggered_rules == ["rule-a", "llm:llm.b"]

