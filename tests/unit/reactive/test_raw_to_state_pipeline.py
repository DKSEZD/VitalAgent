from __future__ import annotations

import json

from agent.benchmarks.vitalbench.eval_loader import AgentInput, WindowLocator
from agent.llm import LLMResult, LLMStreamEvent, zero_usage
from agent.mhealth.schemas import MonitoringState
from agent.reactive.pipeline import ReactivePipeline
from agent.state.mhealth_state_store import get_global_state_store
import agent.tools.proactive_context_tools as proactive_tools
from agent.tools.registry import list_tools


def test_reactive_pipeline_imports_ecg_signal_state_builder_tool() -> None:
    names = {tool["name"] for tool in list_tools()}

    assert "state_build_from_ecg_record" in names


class DeterministicAgentInputLLM:
    def complete(self, *, profile, messages):
        assert profile.role == "planner"
        return LLMResult(
            text=(
                "{"
                '"intent": "status_query",'
                '"reasoning": "Use the active mHealth-QA locator.",'
                '"steps": ['
                "{"
                '"step_id": 1,'
                '"tool_name": "state_get_current_monitoring_state",'
                '"tool_args": {},'
                '"description": "Read the current state using active locator context."'
                "}"
                "]"
                "}"
            ),
            usage=zero_usage(),
        )

    def stream(self, *, profile, messages):
        assert profile.role == "agent"
        yield LLMStreamEvent(delta="Your heart rate is 88 bpm.")
        yield LLMStreamEvent(usage=zero_usage())


def test_reactive_pipeline_agent_input_injects_window_locator_args() -> None:
    store = get_global_state_store()
    store.clear()
    try:
        store.add_states(
            [
                MonitoringState(
                    state_id="ppg-1",
                    patient_id="001",
                    dataset="afppgecg",
                    modality="ppg",
                    subject_id="001",
                    recording_id="001",
                    window_start_s=30.0,
                    window_end_s=60.0,
                    window_duration_s=30.0,
                    hr_bpm=88.0,
                )
            ]
        )
        pipeline = ReactivePipeline(llm_service=DeterministicAgentInputLLM())
        agent_input = AgentInput(
            question_id="q1",
            question="What is my heart rate?",
            window_locator=WindowLocator(
                dataset="afppgecg",
                patient_id="001",
                window_start_s=30.0,
                window_end_s=60.0,
            ),
            answer_set=None,
        )

        responses = [
            response
            for response in pipeline.run_agent_input(agent_input)
            if response.context_used.get("event") != "status"
        ]

        final = responses[-1]
        tool_result = final.context_used["tool_results"][0]
        assert tool_result["tool_name"] == "state_get_current_monitoring_state"
        assert tool_result["data"]["state"]["state_id"] == "ppg-1"
        assert tool_result["data"]["state"]["hr_bpm"] == 88.0
    finally:
        store.clear()


class PreviousWindowAgentInputLLM(DeterministicAgentInputLLM):
    def complete(self, *, profile, messages):
        assert profile.role == "planner"
        return LLMResult(
            text=(
                "{"
                '"intent": "status_query",'
                '"reasoning": "Compare adjacent windows internally.",'
                '"steps": ['
                "{"
                '"step_id": 1,'
                '"tool_name": "state_get_current_monitoring_state",'
                '"tool_args": {},'
                '"description": "Read the current state."'
                "}"
                "]"
                "}"
            ),
            usage=zero_usage(),
        )


class PreviousWindowProactivePlanLLM:
    def __init__(self) -> None:
        self.answer_messages = None

    def complete(self, *, profile, messages):
        assert profile.role == "planner"
        return LLMResult(
            text=(
                "{"
                '"intent": "history_compare",'
                '"reasoning": "Inspect the active monitoring window before answering.",'
                '"steps": ['
                "{"
                '"step_id": 1,'
                '"tool_name": "evaluate_proactive_rules",'
                '"tool_args": {},'
                '"description": "Inspect rhythm changes in the active window."'
                "}"
                "]"
                "}"
            ),
            usage=zero_usage(),
        )

    def stream(self, *, profile, messages):
        assert profile.role == "agent"
        self.answer_messages = messages
        yield LLMStreamEvent(delta="no")
        yield LLMStreamEvent(usage=zero_usage())


def _run_previous_window_proactive_plan(monkeypatch, *, comparison_result: dict):
    executed_tools: list[str] = []

    def fake_call_tool(tool_name: str, **kwargs):
        executed_tools.append(tool_name)
        assert tool_name == "evaluate_proactive_rules"
        return {
            "success": True,
            "data": {
                "rhythm_summary": {
                    "stability_label": "frequent_changes",
                }
            },
            "metadata": {"dataset": kwargs["dataset"]},
        }

    monkeypatch.setattr("agent.reactive.pipeline.call_tool", fake_call_tool)
    monkeypatch.setattr(
        "agent.reactive.pipeline.compare_with_previous_window",
        lambda **kwargs: comparison_result,
    )

    llm = PreviousWindowProactivePlanLLM()
    pipeline = ReactivePipeline(llm_service=llm)
    pipeline.disable_validation = True
    comparison_inputs: list[list[str]] = []
    original_comparison = pipeline._previous_window_comparison_result

    def capture_comparison_input(*, results, active_context):
        comparison_inputs.append([result.tool_name for result in results])
        return original_comparison(results=results, active_context=active_context)

    monkeypatch.setattr(
        pipeline,
        "_previous_window_comparison_result",
        capture_comparison_input,
    )
    agent_input = AgentInput(
        question_id="q_previous_rhythm",
        question="Is my rhythm different from a few moments ago?",
        window_locator=WindowLocator(
            dataset="afppgecg",
            patient_id="001",
            window_start_s=30.0,
            window_end_s=60.0,
            recording_id="001",
        ),
        answer_set=["yes", "no"],
    )
    responses = list(pipeline.run_agent_input(agent_input))
    final = [
        response
        for response in responses
        if response.context_used.get("event") != "status"
    ][-1]
    return final, responses, llm, executed_tools, comparison_inputs


def test_previous_window_success_isolates_answer_context_from_current_window_tool(
    monkeypatch,
) -> None:
    final, responses, llm, executed_tools, comparison_inputs = (
        _run_previous_window_proactive_plan(
            monkeypatch,
            comparison_result={
                "success": True,
                "data": {
                    "comparison": {
                        "assessment": {
                            "current": "regular",
                            "previous": "regular",
                            "changed": False,
                        }
                    }
                },
                "metadata": {"dataset": "afppgecg"},
                "error": "",
            },
        )
    )

    assert executed_tools == ["evaluate_proactive_rules"]
    assert comparison_inputs == [["evaluate_proactive_rules"]]
    assert [
        result["tool_name"] for result in final.context_used["tool_results"]
    ] == ["evaluate_proactive_rules", "previous_window_comparison"]
    assert llm.answer_messages is not None
    answer_user_message = llm.answer_messages[-1]["content"]
    assert "[Step 2] previous_window_comparison" in answer_user_message
    assert "[Step 1] evaluate_proactive_rules" not in answer_user_message
    assert any(
        response.context_used.get("event") == "status"
        and "Temporal orchestration" in response.answer
        for response in responses
    )


def test_previous_window_failure_keeps_current_window_tool_in_answer_context(
    monkeypatch,
) -> None:
    final, _, llm, executed_tools, comparison_inputs = (
        _run_previous_window_proactive_plan(
            monkeypatch,
            comparison_result={
                "success": False,
                "data": {},
                "metadata": {"dataset": "afppgecg"},
                "error": "previous window analyzer failed",
            },
        )
    )

    assert executed_tools == ["evaluate_proactive_rules"]
    assert comparison_inputs == [["evaluate_proactive_rules"]]
    assert [
        result["tool_name"] for result in final.context_used["tool_results"]
    ] == ["evaluate_proactive_rules", "previous_window_comparison"]
    assert llm.answer_messages is not None
    answer_user_message = llm.answer_messages[-1]["content"]
    assert "[Step 1] evaluate_proactive_rules" in answer_user_message
    assert "[Step 2] previous_window_comparison" in answer_user_message


def test_previous_window_scope_appends_internal_comparison_to_answer_context(
    monkeypatch,
) -> None:
    store = get_global_state_store()
    store.clear()
    store.add_states(
        [
            MonitoringState(
                state_id="ppg-current",
                patient_id="001",
                dataset="afppgecg",
                modality="ppg",
                subject_id="001",
                recording_id="001",
                window_start_s=30.0,
                window_end_s=60.0,
                window_duration_s=30.0,
                hr_bpm=90.0,
            )
        ]
    )
    monkeypatch.setattr(
        "agent.reactive.pipeline.compare_with_previous_window",
        lambda **kwargs: {
            "success": True,
            "data": {
                "comparison": {
                    "heart_rate.mean_bpm": {
                        "current": 90.0,
                        "previous": 80.0,
                        "delta": 10.0,
                        "meaningfully_higher": True,
                    }
                },
                "heart_rate_change_interpretation": {
                    "meaningfully_higher": True,
                    "delta_bpm": 10.0,
                },
            },
            "metadata": {"dataset": kwargs["dataset"]},
            "error": "",
        },
    )

    try:
        pipeline = ReactivePipeline(llm_service=PreviousWindowAgentInputLLM())
        agent_input = AgentInput(
            question_id="q_previous",
            question="Is my heart rate higher than a few moments ago?",
            window_locator=WindowLocator(
                dataset="afppgecg",
                patient_id="001",
                window_start_s=30.0,
                window_end_s=60.0,
                recording_id="001",
            ),
            answer_set=["yes", "no"],
        )

        responses = [
            response
            for response in pipeline.run_agent_input(agent_input)
            if response.context_used.get("event") != "status"
        ]
        final = responses[-1]
        comparison = final.context_used["tool_results"][-1]
        assert comparison["tool_name"] == "previous_window_comparison"
        assert comparison["data"]["data"]["comparison"]["heart_rate.mean_bpm"][
            "delta"
        ] == 10.0
        registry_names = {tool["name"] for tool in list_tools()}
        assert "compare_with_previous_window" not in registry_names
        assert "previous_window_comparison" not in registry_names
    finally:
        store.clear()


def test_agent_input_query_includes_recording_id_when_locator_has_one() -> None:
    agent_input = AgentInput(
        question_id="q_ic",
        question="Based on the current window, am I showing signs of AF?",
        window_locator=WindowLocator(
            dataset="icentia11k",
            patient_id="03500",
            window_start_s=900.0,
            window_end_s=1200.0,
            recording_id="17",
        ),
        answer_set=["yes", "no"],
    )

    query = ReactivePipeline._query_from_agent_input(agent_input)

    assert '"recording_id": "17"' in query


def test_af_ppg_max_hr_autofill_enables_recording_context() -> None:
    args = ReactivePipeline._apply_active_locator_args(
        tool_name="analyze_pulse_rate",
        resolved_args={},
        active_record_id=None,
        active_context={
            "dataset": "afppgecg",
            "patient_id": "001",
            "window_start_s": 100.0,
            "window_end_s": 400.0,
            "question": "What was the maximum heart rate captured in this monitoring window?",
        },
    )

    assert args["include_recording_context"] is True


def test_icentia_signal_tool_does_not_receive_subject_id() -> None:
    args = ReactivePipeline._apply_active_locator_args(
        tool_name="analyze_icentia11k_ecg_window_signal",
        resolved_args={},
        active_record_id=None,
        active_context={
            "dataset": "icentia11k",
            "patient_id": "01182",
            "recording_id": "00",
            "window_start_s": 10.0,
            "window_end_s": 20.0,
        },
    )

    assert "subject_id" not in args
    assert args["recording_id"] == "00"
    assert args["window_start_s"] == 10.0
    assert args["window_end_s"] == 20.0


class DeterministicProactiveContextLLM:
    def complete(self, *, profile, messages):
        assert profile.role == "planner"
        return LLMResult(
            text=(
                "{"
                '"intent": "anomaly_explain",'
                '"reasoning": "Load the proactive replay context, then explain the last alert.",'
                '"steps": ['
                "{"
                '"step_id": 1,'
                '"tool_name": "proactive_load_patient_context",'
                '"tool_args": {"clear_existing": true},'
                '"description": "Load proactive PatientContext from active context."'
                "},"
                "{"
                '"step_id": 2,'
                '"tool_name": "proactive_explain_last_alert",'
                '"tool_args": {},'
                '"description": "Explain the latest proactive alert."'
                "}"
                "]"
                "}"
            ),
            usage=zero_usage(),
        )

    def stream(self, *, profile, messages):
        assert profile.role == "agent"
        yield LLMStreamEvent(delta="The latest alert was sustained tachycardia.")
        yield LLMStreamEvent(usage=zero_usage())


def test_reactive_pipeline_loads_proactive_context_path_without_extra_patient_arg(tmp_path) -> None:
    proactive_tools._CONTEXTS.clear()
    context_path = tmp_path / "replay.jsonl"
    context_path.write_text(
        json.dumps(
            {
                "record_type": "patient_context",
                "patient_id": "01182",
                "alert_summary": {"total_alerts": 1},
                "recent_alerts": [
                    {
                        "alert_id": 0,
                        "patient_id": "01182",
                        "window_index": 42,
                        "offset_s": 420.0,
                        "urgency": "medium",
                        "triggered_rules": ["sustained_tachycardia"],
                        "reason": "Heart rate has exceeded 100 bpm in recent readings.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        pipeline = ReactivePipeline(llm_service=DeterministicProactiveContextLLM())
        responses = [
            response
            for response in pipeline.run(
                "刚刚为什么提醒我？",
                active_context={
                    "patient_id": "01182",
                    "proactive_context_path": str(context_path),
                },
            )
            if response.context_used.get("event") != "status"
        ]

        final = responses[-1]
        tool_results = final.context_used["tool_results"]
        assert final.context_used["validation_passed"] is True
        assert tool_results[0]["tool_name"] == "proactive_load_patient_context"
        assert tool_results[0]["success"] is True
        assert tool_results[0]["data"]["context_count"] == 1
        assert tool_results[1]["tool_name"] == "proactive_explain_last_alert"
        assert tool_results[1]["success"] is True
        assert tool_results[1]["data"]["patient_id"] == "01182"
    finally:
        proactive_tools._CONTEXTS.clear()
