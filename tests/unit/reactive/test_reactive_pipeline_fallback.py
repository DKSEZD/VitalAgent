from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.llm import LLMStreamEvent
from agent.reactive.pipeline import ReactivePipeline
from agent.schemas import IntentType, Plan, PlanStep, RiskLevel, ToolResult, ValidationResult
import agent.tools.proactive_context_tools as proactive_tools


class DummyPlanner:
    def __init__(self, plan: Plan, usage: dict[str, int] | None = None):
        self.plan = plan
        self.usage = usage or {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def create_plan(self, user_query: str, active_record_id: str | None = None):
        return self.plan, self.usage

    def replan(self, previous_plan: Plan, issues: list[str], user_query: str):
        return previous_plan, self.usage


class ContextAwarePlanner(DummyPlanner):
    def create_plan(
        self,
        user_query: str,
        active_record_id: str | None = None,
        extra_context: str = "",
    ):
        return self.plan, self.usage


class DummyLLMAgent:
    def __init__(self, parts: list[str] | None = None) -> None:
        self.generate_answer_called = False
        self.failure_calls: list[list[str]] = []
        self.parts = parts or ["fallback ", "answer"]

    def generate_answer(self, user_query: str, plan: Plan, tool_results: list[ToolResult], extra_context: str = ""):
        self.generate_answer_called = True
        for part in self.parts:
            yield LLMStreamEvent(delta=part)
        yield LLMStreamEvent(usage=_usage(output=5, total=9))

    def generate_failure_response(
        self,
        user_query: str,
        plans: list[Plan],
        issues: list[str],
        answer_set: list[str] | None = None,
    ):
        self.failure_calls.append(list(issues))
        return (
            SimpleNamespace(
                answer="failure answer",
                risk_level=None,
                plan_history=plans,
                retry_count=len(plans) - 1,
                context_used={},
            ),
            _usage(output=3, total=7),
        )

    def _extract_risk(self, answer: str):
        return None


class ProactiveAlertLLMAgent:
    def __init__(self) -> None:
        self.tool_result_snapshot: list[ToolResult] = []

    def generate_answer(
        self,
        user_query: str,
        plan: Plan,
        tool_results: list[ToolResult],
        extra_context: str = "",
    ):
        self.tool_result_snapshot = list(tool_results)
        explanation = tool_results[-1].data["explanation"]
        yield LLMStreamEvent(delta=explanation["plain_language_summary"])
        yield LLMStreamEvent(usage=_usage(output=7, total=11))

    def generate_failure_response(
        self,
        user_query: str,
        plans: list[Plan],
        issues: list[str],
        answer_set: list[str] | None = None,
    ):
        raise AssertionError(f"Unexpected failure response: {issues}")

    def _extract_risk(self, answer: str):
        return None


def _usage(input_tokens: int = 1, cached: int = 0, output: int = 2, total: int = 3) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "total_tokens": total,
    }


def _plan() -> Plan:
    return Plan.model_validate(
        {
            "intent": "record_list",
            "reasoning": "test",
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "list_ecg_records",
                    "tool_args": {},
                    "description": "list",
                },
            ],
        }
    )


def _proactive_alert_plan() -> Plan:
    return Plan(
        intent=IntentType.ANOMALY_EXPLAIN,
        reasoning="Explain the latest proactive alert from PatientContext.",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="proactive_explain_last_alert",
                tool_args={},
                description="Explain the most recent proactive alert.",
            )
        ],
    )


def _general_chat_plan() -> Plan:
    return Plan(
        intent=IntentType.GENERAL_CHAT,
        reasoning="Reply without tools.",
        steps=[],
        original_query="hello",
    )


def _run_non_status(pipeline: ReactivePipeline):
    return [
        response
        for response in pipeline.run("list records", active_record_id="123")
        if response.context_used.get("event") != "status"
    ]


def test_general_chat_emits_final_response_with_complete_usage() -> None:
    pipeline = ReactivePipeline(client=object())
    pipeline.planner = DummyPlanner(
        _general_chat_plan(),
        usage=_usage(input_tokens=4, total=4),
    )
    pipeline.llm_agent = DummyLLMAgent(parts=["hello ", "there"])
    pipeline.llm_agent._extract_risk = lambda answer: RiskLevel.LOW

    responses = [
        response
        for response in pipeline.run("hello")
        if response.context_used.get("event") != "status"
    ]

    assert [response.answer for response in responses] == [
        "hello ",
        "hello there",
        "hello there",
    ]
    final = responses[-1]
    assert final.risk_level == RiskLevel.LOW
    assert final.context_used["tool_results"] == []
    assert final.context_used["token_usage"]["total_tokens"] == 13
    assert final.context_used["validation_passed"] is None


def test_normal_validation_path_streams_partial_answers_and_accumulates_usage(monkeypatch: pytest.MonkeyPatch):
    pipeline = ReactivePipeline(client=object())
    pipeline.planner = DummyPlanner(_plan(), usage=_usage(input_tokens=4, total=4))
    pipeline.llm_agent = DummyLLMAgent(parts=["normal ", "answer"])
    pipeline.validator = SimpleNamespace(
        validate=lambda plan, results: ValidationResult(passed=True, issues=[])
    )

    def fake_call_tool(name: str, **kwargs):
        if name == "list_ecg_records":
            return {
                "success": True,
                "record_ids": ["123", "122"],
                "records": [
                    {"record_id": "123", "patient_id": "p1"},
                    {"record_id": "122", "patient_id": "p1"},
                ],
            }
        raise AssertionError(f"Unexpected tool call: {name}")

    monkeypatch.setattr("agent.reactive.pipeline.call_tool", fake_call_tool)

    responses = _run_non_status(pipeline)

    assert [response.answer for response in responses] == [
        "normal ",
        "normal answer",
        "normal answer",
    ]
    assert responses[-1].context_used["validation_passed"] is True
    assert responses[-1].context_used["token_usage"]["total_tokens"] == 13


def test_reactive_pipeline_explains_loaded_proactive_alert_context() -> None:
    proactive_tools._CONTEXTS.clear()
    proactive_tools._CONTEXTS["01182"] = {
        "patient_id": "01182",
        "alert_summary": {
            "total_alerts": 1,
            "false_alert_count": 0,
            "matched_alert_count": 1,
            "by_rule": {"sustained_tachycardia": 1},
        },
        "recent_alerts": [
            {
                "alert_id": 0,
                "patient_id": "01182",
                "segment_id": "00",
                "window_index": 138,
                "offset_s": 1380.0,
                "urgency": "medium",
                "triggered_rules": ["sustained_tachycardia"],
                "reason": "Heart rate has exceeded 100 bpm in 80% of recent readings.",
                "matched_episode_label": "AF",
                "latency_seconds": 80.0,
                "is_false_alert": False,
            }
        ],
    }

    pipeline = ReactivePipeline(client=object())
    pipeline.planner = ContextAwarePlanner(
        _proactive_alert_plan(),
        usage=_usage(input_tokens=4, total=4),
    )
    llm_agent = ProactiveAlertLLMAgent()
    pipeline.llm_agent = llm_agent

    responses = [
        response
        for response in pipeline.run(
            "刚刚为什么提醒我？",
            active_context={"patient_id": "01182"},
        )
        if response.context_used.get("event") != "status"
    ]

    assert responses[-1].answer
    assert "sustained_tachycardia" in responses[-1].answer
    assert "AF episode" in responses[-1].answer
    assert responses[-1].context_used["validation_passed"] is True
    tool_result = responses[-1].context_used["tool_results"][0]
    assert tool_result["tool_name"] == "proactive_explain_last_alert"
    assert tool_result["data"]["patient_id"] == "01182"
    assert tool_result["data"]["explanation"]["latency_seconds"] == 80.0
    assert llm_agent.tool_result_snapshot[0].data["alert"]["window_index"] == 138

    proactive_tools._CONTEXTS.clear()
