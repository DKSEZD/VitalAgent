from __future__ import annotations

from agent.evaluation.reactive.protocols import (
    PaperV1NoPlannerPlanner,
    PaperV1ValidationGate,
)
from agent.reactive.validation import ValidationGate


def _usage() -> dict[str, int]:
    return {
        "input_tokens": 3,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 5,
    }


class FakeLLMService:
    def __init__(self):
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, *, profile, messages):
        self.messages.append(messages)
        return type(
            "Result",
            (),
            {
                "text": '{"tool_args": {"patient_id": "llm-selected"}}',
                "usage": _usage(),
            },
        )()


def test_paper_v1_no_planner_preserves_global_pool_and_per_tool_calls(
    monkeypatch,
) -> None:
    tools = [
        {
            "name": "analyze_ppg_dalia_window_signal",
            "description": "Analyze PPG-DaLiA",
            "parameters": {"patient_id": {"type": "string"}},
        },
        {
            "name": "analyze_wesad_window_signal",
            "description": "Analyze WESAD",
            "parameters": {"patient_id": {"type": "string"}},
        },
    ]
    monkeypatch.setattr("agent.reactive.planner.list_tools", lambda: tools[:1])
    monkeypatch.setattr("agent.tools.registry.list_tools", lambda: tools)
    service = FakeLLMService()
    planner = PaperV1NoPlannerPlanner(
        llm_service=service,
        seed=42,
        sample_key="paper-v1",
        tool_count=2,
        tool_pool_names={tool["name"] for tool in tools},
    )

    plan, usage = planner.create_plan(
        "Inspect this window.",
        extra_context='{"patient_id": "canonical"}',
    )

    assert {step.tool_name for step in plan.steps} == {tool["name"] for tool in tools}
    assert all(step.tool_args == {"patient_id": "llm-selected"} for step in plan.steps)
    assert len(service.messages) == 2
    assert all("Pre-selected tool:" in messages[1]["content"] for messages in service.messages)
    assert usage["total_tokens"] == 10


def test_paper_v1_validation_constants_are_frozen_before_raw_tool_additions() -> None:
    extended_protocol_tools = {
        "analyze_icentia11k_ecg_window_signal",
        "analyze_ppg_dalia_window_signal",
        "analyze_wesad_window_signal",
    }

    assert extended_protocol_tools.isdisjoint(PaperV1ValidationGate.REQUIRED_FIELDS)
    assert extended_protocol_tools.isdisjoint(PaperV1ValidationGate.CRITICAL_TOOLS)
    assert extended_protocol_tools <= ValidationGate.REQUIRED_FIELDS.keys()
    assert extended_protocol_tools <= ValidationGate.CRITICAL_TOOLS
