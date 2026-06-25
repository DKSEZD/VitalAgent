from __future__ import annotations

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
    def __init__(self, text: str = '{"tool_args": {"patient_id": "p1"}}'):
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
    assert "Pre-selected tool" in service_a.messages[0][1]["content"]


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
    service_a = FakeLLMService(text='{"tool_args": {}}')
    service_b = FakeLLMService(text='{"tool_args": {}}')

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
    assert usage_a["total_tokens"] == 5 * len(plan_a.steps)
    assert usage_b == usage_a


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
    service = FakeLLMService(text='{"tool_args": {}}')
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
    assert usage["total_tokens"] == 10


def test_no_planner_explicit_pool_bypasses_restricted_planner_pool(
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
    monkeypatch.setattr("agent.reactive.planner.list_tools", lambda: [wesad_tool])
    monkeypatch.setattr(
        "agent.tools.registry.list_tools",
        lambda: [wesad_tool, icentia_tool],
    )
    service = FakeLLMService(text='{"tool_args": {}}')
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
    assert usage["total_tokens"] == 10
