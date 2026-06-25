from __future__ import annotations

import pytest

from agent.config import reset_config
from agent.reactive.ecg_diagnosis_policy import classify_ecg_diagnosis_question
from agent.reactive import planner
from agent.schemas import IntentType, Plan, PlanStep


@pytest.fixture(autouse=True)
def reset_config_cache() -> None:
    reset_config()
    yield
    reset_config()


def test_build_planner_system_prompt_includes_ecg_diagnosis_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    prompt = planner.build_planner_system_prompt(
        tools_description="TOOLS",
        tool_names="ecg_diagnosis, analyze_heart_rate",
    )

    assert "ecg_diagnosis" in prompt
    assert "ECGFounder" in prompt
    assert "STRONGEST" in prompt
    assert "WEAKER" in prompt
    assert "lead-specific, form-related, or region-qualified" in prompt


def test_build_planner_system_prompt_omits_ecg_diagnosis_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    prompt = planner.build_planner_system_prompt(
        tools_description="TOOLS",
        tool_names="analyze_heart_rate, get_ecg_description",
    )

    assert "ecg_diagnosis" not in prompt
    assert "ECGFounder" not in prompt
    assert "STRONGEST" not in prompt


def test_classify_ecg_diagnosis_question_marks_high_trust_label_query() -> None:
    profile = classify_ecg_diagnosis_question(
        "Does this ECG show atrial fibrillation?",
    )

    assert profile.is_relevant is True
    assert profile.trust_level == "high"
    assert profile.needs_report_context is True
    assert profile.needs_lead_morphology is False
    assert profile.needs_interval_morphology is False


def test_classify_ecg_diagnosis_question_marks_low_trust_lead_specific_query() -> None:
    profile = classify_ecg_diagnosis_question(
        "Does this ECG show symptoms of inverted t-waves in lead aVL?",
    )

    assert profile.is_relevant is True
    assert profile.trust_level == "low"
    assert profile.needs_lead_morphology is True
    assert profile.extracted_leads == ("aVL",)


def test_classify_ecg_diagnosis_question_marks_low_trust_interval_query() -> None:
    profile = classify_ecg_diagnosis_question(
        "Which diagnostic symptom does this ECG show, long qt-interval or incomplete right bundle branch block, excluding uncertain symptoms?",
    )

    assert profile.is_relevant is True
    assert profile.trust_level == "low"
    assert profile.needs_interval_morphology is True
    assert profile.ignores_uncertain_positive is True


def test_apply_ecg_diagnosis_policy_prioritizes_report_and_morphology_for_low_trust_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    plan_obj = Plan(
        intent=IntentType.ANOMALY_EXPLAIN,
        reasoning="Initial plan",
        original_query="Does this ECG show symptoms of inverted t-waves in lead aVL?",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="ecg_diagnosis",
                tool_args={"record_id": "6432"},
                description="Run ECGFounder",
            ),
        ],
    )

    updated = planner._apply_ecg_diagnosis_policy(
        plan_obj,
        plan_obj.original_query,
        active_record_id="6432",
    )

    assert [step.tool_name for step in updated.steps[:3]] == [
        "get_ecg_description",
        "analyze_lead_morphology",
        "ecg_diagnosis",
    ]
    assert updated.steps[1].tool_args["leads"] == "aVL"
    assert "weakly covered by ECGFounder" in updated.reasoning


def test_apply_ecg_diagnosis_policy_prioritizes_ecgfounder_for_high_trust_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    plan_obj = Plan(
        intent=IntentType.ANOMALY_EXPLAIN,
        reasoning="Initial plan",
        original_query="Does this ECG show atrial fibrillation?",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_heart_rate",
                tool_args={"record_id": "101"},
                description="Check rhythm rate",
            ),
        ],
    )

    updated = planner._apply_ecg_diagnosis_policy(
        plan_obj,
        plan_obj.original_query,
        active_record_id="101",
    )

    assert [step.tool_name for step in updated.steps[:3]] == [
        "ecg_diagnosis",
        "get_ecg_description",
        "analyze_heart_rate",
    ]
    assert updated.steps[0].tool_args["record_id"] == "101"
    assert "well-covered diagnosis labels" in updated.reasoning


def test_classify_ecg_diagnosis_question_marks_high_trust_single_lead_for_lead_i() -> None:
    profile = classify_ecg_diagnosis_question(
        "Does this ECG show symptoms of digitalis effect in lead I?",
    )

    assert profile.is_relevant is True
    assert profile.trust_level == "high"
    assert profile.ecg_diagnosis_mode == "single_lead"
    assert profile.extracted_leads == ("I",)
    assert profile.needs_lead_morphology is False


def test_classify_ecg_diagnosis_question_keeps_low_trust_for_non_i_lead_specific() -> None:
    profile = classify_ecg_diagnosis_question(
        "Does this ECG show symptoms of digitalis effect in lead aVL?",
    )

    assert profile.trust_level == "low"
    assert profile.ecg_diagnosis_mode == "12_lead"
    assert profile.needs_lead_morphology is True


def test_apply_ecg_diagnosis_policy_uses_single_lead_mode_for_lead_i_high_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    plan_obj = Plan(
        intent=IntentType.ANOMALY_EXPLAIN,
        reasoning="Initial plan",
        original_query="Does this ECG show symptoms of digitalis effect in lead I?",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_heart_rate",
                tool_args={"record_id": "101"},
                description="Check rhythm rate",
            ),
        ],
    )

    updated = planner._apply_ecg_diagnosis_policy(
        plan_obj,
        plan_obj.original_query,
        active_record_id="101",
    )

    assert updated.steps[0].tool_name == "ecg_diagnosis"
    assert updated.steps[0].tool_args["mode"] == "single_lead"
    assert "single-lead mode" in updated.reasoning


def test_apply_ecg_diagnosis_policy_skips_ppg_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "true")
    reset_config()

    plan_obj = Plan(
        intent=IntentType.ANOMALY_EXPLAIN,
        reasoning="Initial plan",
        original_query="Is this pulse signal irregular?",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_ppg_rhythm_irregularity",
                tool_args={"record_id": "ppg:001"},
                description="Check pulse irregularity",
            ),
        ],
    )

    updated = planner._apply_ecg_diagnosis_policy(
        plan_obj,
        plan_obj.original_query,
        active_record_id="ppg:001",
    )

    assert updated == plan_obj


def test_create_plan_returns_usage_from_shared_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeService:
        def complete(self, *, profile, messages):
            assert profile.role == "planner"
            assert messages[0]["role"] == "system"
            return type(
                "Result",
                (),
                {
                    "text": '{"intent": "knowledge_qa", "reasoning": "ok", "steps": []}',
                    "usage": {
                        "input_tokens": 7,
                        "cached_input_tokens": 1,
                        "output_tokens": 2,
                        "total_tokens": 9,
                    },
                },
            )()

    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    planner_obj = planner.Planner(llm_service=FakeService())
    plan_obj, usage = planner_obj.create_plan("What does this ECG show?")

    assert plan_obj.intent == IntentType.KNOWLEDGE_QA
    assert usage["total_tokens"] == 9


def test_create_plan_retries_invalid_tool_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[dict[str, str]]] = []

    class FakeService:
        def complete(self, *, profile, messages):
            calls.append(messages)
            tool_name = "invented_tool" if len(calls) == 1 else "medical_knowledge"
            return type(
                "Result",
                (),
                {
                    "text": (
                        '{"intent":"knowledge_qa","reasoning":"ok","steps":['
                        f'{{"step_id":1,"tool_name":"{tool_name}",'
                        '"tool_args":{},"description":"answer"}]}'
                    ),
                    "usage": {
                        "input_tokens": 3,
                        "cached_input_tokens": 0,
                        "output_tokens": 1,
                        "total_tokens": 4,
                    },
                },
            )()

    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    planner_obj = planner.Planner(llm_service=FakeService())
    plan_obj, usage = planner_obj.create_plan("What is atrial fibrillation?")

    assert len(calls) == 2
    assert "invented_tool" in calls[1][1]["content"]
    assert [step.tool_name for step in plan_obj.steps] == ["medical_knowledge"]
    assert usage["total_tokens"] == 8


def test_create_plan_labels_active_ppg_record_in_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_messages: list[dict[str, str]] = []

    class FakeService:
        def complete(self, *, profile, messages):
            seen_messages.extend(messages)
            return type(
                "Result",
                (),
                {
                    "text": '{"intent": "knowledge_qa", "reasoning": "ok", "steps": []}',
                    "usage": {
                        "input_tokens": 3,
                        "cached_input_tokens": 0,
                        "output_tokens": 1,
                        "total_tokens": 4,
                    },
                },
            )()

    monkeypatch.setenv("ENABLE_ECG_DIAGNOSIS", "false")
    reset_config()

    planner_obj = planner.Planner(llm_service=FakeService())
    planner_obj.create_plan("How is this pulse signal?", active_record_id="ppg:001")

    assert any(
        message["role"] == "user" and "Active PPG record ID: ppg:001" in message["content"]
        for message in seen_messages
    )
