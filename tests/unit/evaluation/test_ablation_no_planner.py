from __future__ import annotations

import json

from agent.evaluation.reactive.ablation_no_planner import NoPlannerPlanner
from agent.schemas import IntentType


def _usage() -> dict[str, int]:
    return {
        "input_tokens": 3,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 5,
    }


class FakeLLMService:
    def __init__(
        self,
        text: str = (
            '{"tool_calls": ['
            '{"tool_name": "analyze_ppg_dalia_window_signal", '
            '"tool_args": {"patient_id": "p1"}},'
            '{"tool_name": "state_get_current_monitoring_state", '
            '"tool_args": {"patient_id": "p1"}}]}'
        ),
    ):
        self.text = text
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, *, profile, messages):
        self.messages.append(messages)
        return type(
            "Result",
            (),
            {
                "text": self.text,
                "usage": _usage(),
            },
        )()


def test_no_planner_selects_reproducible_tool_and_llm_args(monkeypatch) -> None:
    tools = [
        {
            "name": "state_get_current_monitoring_state",
            "description": "Get current state",
            "parameters": {"patient_id": {"type": "string"}},
        },
        {
            "name": "analyze_ppg_dalia_window_signal",
            "description": "Analyze PPG-DaLiA signal",
            "parameters": {"patient_id": {"type": "string"}},
        },
    ]
    monkeypatch.setattr("agent.reactive.planner.list_tools", lambda: tools)
    service_a = FakeLLMService()
    service_b = FakeLLMService()

    planner_a = NoPlannerPlanner(
        llm_service=service_a,
        seed=11,
        sample_key="q1",
    )
    planner_b = NoPlannerPlanner(
        llm_service=service_b,
        seed=11,
        sample_key="q1",
    )

    plan_a, usage_a = planner_a.create_plan(
        "What is my heart rate?",
        extra_context='{"patient_id": "p1"}',
    )
    plan_b, usage_b = planner_b.create_plan(
        "What is my heart rate?",
        extra_context='{"patient_id": "p1"}',
    )

    assert plan_a.intent == IntentType.STATUS_QUERY
    assert [step.tool_name for step in plan_a.steps] == [
        step.tool_name for step in plan_b.steps
    ]
    assert plan_a.steps[0].tool_args == {"patient_id": "p1"}
    assert usage_a == usage_b == _usage()
    assert "Pre-selected tools" in service_a.messages[0][1]["content"]


def test_no_planner_falls_back_to_empty_args_on_bad_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.reactive.planner.list_tools",
        lambda: [
            {
                "name": "state_get_current_monitoring_state",
                "description": "Get current state",
                "parameters": {"patient_id": {"type": "string"}},
            }
        ],
    )
    service = FakeLLMService(text="not json")
    planner = NoPlannerPlanner(llm_service=service, seed=1, sample_key="q1")

    plan, usage = planner.create_plan("What is my heart rate?")

    assert plan.steps[0].tool_args == {}
    assert usage == _usage()


def test_no_planner_can_sample_tool_count_range(monkeypatch) -> None:
    tools = [
        {
            "name": f"tool_{index}",
            "description": "Synthetic tool",
            "parameters": {},
        }
        for index in range(1, 8)
    ]
    monkeypatch.setattr("agent.reactive.planner.list_tools", lambda: tools)
    batch = {
        "tool_calls": [
            {"tool_name": f"tool_{index}", "tool_args": {}}
            for index in range(1, 8)
        ]
    }
    service_a = FakeLLMService(text=json.dumps(batch))
    service_b = FakeLLMService(text=json.dumps(batch))

    planner_a = NoPlannerPlanner(
        llm_service=service_a,
        seed=23,
        sample_key="q-range",
        tool_count=1,
        tool_count_max=5,
    )
    planner_b = NoPlannerPlanner(
        llm_service=service_b,
        seed=23,
        sample_key="q-range",
        tool_count=1,
        tool_count_max=5,
    )

    plan_a, usage_a = planner_a.create_plan("Use a random number of tools.")
    plan_b, usage_b = planner_b.create_plan("Use a random number of tools.")

    assert 1 <= len(plan_a.steps) <= 5
    assert [step.tool_name for step in plan_a.steps] == [
        step.tool_name for step in plan_b.steps
    ]
    assert usage_a["total_tokens"] == 5
    assert usage_b == usage_a
    assert len(service_a.messages) == 1


def test_no_planner_defaults_to_restricted_planner_tool_pool(
    monkeypatch,
) -> None:
    narrow_tools = [
        {
            "name": "analyze_ppg_dalia_window_signal",
            "description": "Analyze PPG-DaLiA signal",
            "parameters": {},
        },
        {
            "name": "analyze_wesad_window_signal",
            "description": "Analyze WESAD signal",
            "parameters": {},
        },
    ]
    full_tools = narrow_tools + [
        {
            "name": "analyze_icentia11k_ecg_window_signal",
            "description": "Analyze Icentia11k ECG signal",
            "parameters": {},
        }
    ]
    monkeypatch.setattr("agent.reactive.planner.list_tools", lambda: narrow_tools)
    monkeypatch.setattr("agent.tools.registry.list_tools", lambda: full_tools)
    service = FakeLLMService(
        text=(
            '{"tool_calls": ['
            '{"tool_name": "analyze_ppg_dalia_window_signal", "tool_args": {}},'
            '{"tool_name": "analyze_wesad_window_signal", "tool_args": {}}]}'
        )
    )
    planner = NoPlannerPlanner(
        llm_service=service,
        seed=31,
        sample_key="restricted-pool",
        tool_count=2,
    )

    plan, usage = planner.create_plan("Use the default no-planner pool.")

    assert {step.tool_name for step in plan.steps} == {
        "analyze_ppg_dalia_window_signal",
        "analyze_wesad_window_signal",
    }
    assert usage["total_tokens"] == 5


def test_no_planner_explicit_pool_filters_exposed_planner_pool(
    monkeypatch,
) -> None:
    wesad_tool = {
        "name": "analyze_wesad_window_signal",
        "description": "Analyze WESAD signal",
        "parameters": {},
    }
    icentia_tool = {
        "name": "analyze_icentia11k_ecg_window_signal",
        "description": "Analyze Icentia11k ECG signal",
        "parameters": {},
    }
    monkeypatch.setattr(
        "agent.reactive.planner.list_tools",
        lambda: [wesad_tool, icentia_tool],
    )
    service = FakeLLMService(
        text=(
            '{"tool_calls": ['
            '{"tool_name": "analyze_icentia11k_ecg_window_signal", "tool_args": {}},'
            '{"tool_name": "analyze_wesad_window_signal", "tool_args": {}}]}'
        )
    )
    planner = NoPlannerPlanner(
        llm_service=service,
        seed=37,
        sample_key="full-raw-signal-pool",
        tool_count=2,
        tool_pool_names={
            "analyze_icentia11k_ecg_window_signal",
            "analyze_wesad_window_signal",
        },
    )

    plan, usage = planner.create_plan("Use the explicit no-planner pool.")

    assert {step.tool_name for step in plan.steps} == {
        "analyze_icentia11k_ecg_window_signal",
        "analyze_wesad_window_signal",
    }
    assert usage["total_tokens"] == 5


def test_no_tools_skips_argument_model_call(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.reactive.planner.list_tools",
        lambda: [
            {
                "name": "analyze_wesad_window_signal",
                "description": "Analyze WESAD",
                "parameters": {},
            }
        ],
    )
    service = FakeLLMService()
    planner = NoPlannerPlanner(
        llm_service=service,
        selection_mode="none",
        sample_key="no-tools",
    )

    plan, usage = planner.create_plan("Answer without tools.")

    assert plan.steps == []
    assert usage["total_tokens"] == 0
    assert service.messages == []


def test_all_tools_keeps_exact_set_and_overrides_canonical_locator(monkeypatch) -> None:
    tools = [
        {
            "name": "analyze_pulse_rate",
            "description": "Analyze pulse",
            "parameters": {
                "record_id": {"type": "string"},
                "patient_id": {"type": "string"},
                "window_start_s": {"type": "number"},
                "window_end_s": {"type": "number"},
            },
        },
        {
            "name": "evaluate_proactive_rules",
            "description": "Evaluate rules",
            "parameters": {
                "dataset": {"type": "string"},
                "patient_id": {"type": "string"},
                "window_start_s": {"type": "number"},
                "window_end_s": {"type": "number"},
            },
        },
    ]
    monkeypatch.setattr("agent.reactive.planner.list_tools", lambda: tools)
    service = FakeLLMService(
        text=(
            '{"tool_calls": ['
            '{"tool_name": "analyze_pulse_rate", '
            '"tool_args": {"record_id": "bad", "patient_id": ".", "extra": 1}},'
            '{"tool_name": "unexpected_tool", "tool_args": {}},'
            '{"tool_name": "analyze_pulse_rate", "tool_args": {"patient_id": "dup"}}]}'
        )
    )
    planner = NoPlannerPlanner(
        llm_service=service,
        selection_mode="all",
        canonical_context={
            "dataset": "afppgecg",
            "patient_id": "007",
            "window_start_s": 10.0,
            "window_end_s": 40.0,
        },
    )

    plan, usage = planner.create_plan(
        "Run every applicable tool.",
        active_record_id="ppg:007",
    )

    assert [step.tool_name for step in plan.steps] == [
        "analyze_pulse_rate",
        "evaluate_proactive_rules",
    ]
    assert plan.steps[0].tool_args == {
        "record_id": "ppg:007",
        "patient_id": "007",
        "window_start_s": 10.0,
        "window_end_s": 40.0,
    }
    assert plan.steps[1].tool_args == {
        "dataset": "afppgecg",
        "patient_id": "007",
        "window_start_s": 10.0,
        "window_end_s": 40.0,
    }
    assert usage["total_tokens"] == 5
    assert len(service.messages) == 1
