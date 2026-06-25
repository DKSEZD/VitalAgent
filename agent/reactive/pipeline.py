"""
Reactive Pipeline — orchestrates the full reactive path:

    User Query
        → Planner (create plan via LLM)
        → Execute tools (load ECG data as text)
        → Validation Gate (check data completeness)
        ─── pass ──→ LLM Agent → Response
        ─── fail ──→ Re-plan → re-execute → Validation
                      ─── pass ──→ LLM Agent → Response
                      ─── fail ──→ Failure Response
"""

from __future__ import annotations

import logging
import json
from typing import Any, Iterator

from openai import OpenAI

from agent.config import get_config
from agent.llm import LLMService, merge_usage, zero_usage
from agent.reactive.llm_agent import LLMAgent
from agent.reactive.planner import Planner
from agent.reactive.previous_window_compare import (
    PREVIOUS_WINDOW_COMPARISON_RESULT,
    compare_with_previous_window,
    needs_previous_window_comparison,
)
from agent.reactive.validation import ValidationGate, ValidationResult
from agent.schemas import AgentResponse, IntentType, Plan, ToolResult
from agent.tools.registry import call_tool

import agent.tools.datasets.ppg_dalia
import agent.tools.datasets.wesad
import agent.tools.datasets.wesad_stress_classifier
import agent.tools.ecg.tools
import agent.tools.mhealth_state_tools
import agent.tools.mhealth_state_build_tools
import agent.tools.mhealth_signal_state_build_tools
import agent.tools.ppg.tools
import agent.tools.proactive_context_tools
import agent.tools.proactive_evaluate_tools
import agent.tools.search_tools  # noqa: F401

logger = logging.getLogger(__name__)


class ReactivePipeline:
    """End-to-end reactive pipeline: query → plan → tools → validate → answer."""

    def __init__(self, client: OpenAI | None = None, llm_service: LLMService | None = None):
        cfg = get_config()
        self.max_retries = cfg.pipeline.max_retries
        self.disable_validation = cfg.pipeline.disable_validation
        self.verbose = cfg.pipeline.verbose

        self.llm_service = llm_service or LLMService(client=client)

        self.planner = Planner(llm_service=self.llm_service)
        self.validator = ValidationGate()
        self.llm_agent = LLMAgent(llm_service=self.llm_service)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_agent_input(self, agent_input: Any) -> Iterator[AgentResponse]:
        """Execute the reactive pipeline from a VitalBench AgentInput object."""

        context = self._active_context_from_agent_input(agent_input)
        query = self._query_from_agent_input(agent_input)
        active_record_id = self._active_record_id_from_context(context)
        yield from self.run(
            user_query=query,
            active_record_id=active_record_id,
            active_context=context,
        )

    def run(
        self,
        user_query: str,
        active_record_id: str | None = None,
        active_context: dict[str, Any] | None = None,
    ) -> Iterator[AgentResponse]:
        """Execute pipeline and stream answer updates from the LLM stage."""
        token_usage = zero_usage()

        self._log(f"[Pipeline] Query: {user_query}", color_stage="query")
        self._log(f"[Pipeline] Active Record: {active_record_id}")
        if active_context:
            self._log(f"[Pipeline] Active Context: {active_context}")

        yield self._status_response("🧭 Planning: Understanding the query and generating an execution plan...")

        planner_extra_context = ""
        if active_context:
            planner_extra_context = (
                "Active multimodal mHealth context:\n"
                + json.dumps(active_context, ensure_ascii=False, indent=2)
            )

        if planner_extra_context:
            plan, planning_usage = self.planner.create_plan(
                user_query=user_query,
                active_record_id=active_record_id,
                extra_context=planner_extra_context,
            )
        else:
            plan, planning_usage = self.planner.create_plan(
                user_query=user_query,
                active_record_id=active_record_id,
            )
        token_usage = merge_usage(token_usage, planning_usage)
        all_plans: list[Plan] = [plan]
        yield self._status_response(
            f"✅ Plan ready: {len(plan.steps)} steps. Starting tool execution..."
        )

        # Non-ECG conversation: skip tools entirely, let LLM reply directly
        if plan.intent == IntentType.GENERAL_CHAT and not plan.steps:
            final_answer, answer_usage = yield from self._stream_answer(
                user_query=user_query,
                plan=plan,
                tool_results=[],
                token_usage=token_usage,
                plan_history=all_plans,
                retry_count=0,
                validation_passed=None,
                validation_issues=[],
            )
            token_usage = merge_usage(token_usage, answer_usage)
            yield AgentResponse(
                answer=final_answer,
                risk_level=self.llm_agent._extract_risk(final_answer),
                context_used={
                    "tool_results": [],
                    "token_usage": token_usage,
                    "validation_passed": None,
                    "validation_issues": [],
                },
                plan_history=all_plans,
                retry_count=0,
            )
            return

        validation = None
        for attempt in range(1 + self.max_retries):
            results: list[ToolResult] = []
            for step in plan.steps:
                yield self._status_response(
                    f"🔧 [{step.step_id}/{len(plan.steps)}] Calling tool {step.tool_name}..."
                )

                resolved_args = self._resolve_args(
                    step.tool_name,
                    step.tool_args,
                    results,
                    active_record_id,
                    active_context,
                )
                raw = call_tool(step.tool_name, **resolved_args)

                success = raw.get("success", False)
                error = raw.get("error") or ""
                data = {k: v for k, v in raw.items() if k not in ("success", "error")}

                result = ToolResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=success,
                    data=data,
                    error=error,
                )
                results.append(result)
                self._log_tool_execution(
                    step.tool_name,
                    resolved_args,
                    success=success,
                    error=error,
                )
                if success:
                    yield self._status_response(f"✅ Tool {step.tool_name} completed")
                else:
                    yield self._status_response(f"⚠️ Tool {step.tool_name} failed: {error}")

            # --- Normal validation path ---
            yield self._status_response("🧪 Validation: Checking completeness and validity of tool outputs...")
            if self.disable_validation:
                self._log("Validation is disabled. Skipping checks and proceeding to answer generation.")
                validation = ValidationResult(passed=True, issues=[])
            else:
                validation = self.validator.validate(plan, results)

            if validation.passed:
                answer_results = list(results)
                recorded_results = list(results)
                question_text = str((active_context or {}).get("question") or user_query)
                if needs_previous_window_comparison(question_text):
                    yield self._status_response(
                        "🕒 Temporal orchestration: Comparing with the previous window..."
                    )
                    comparison = self._previous_window_comparison_result(
                        results=results,
                        active_context=active_context,
                    )
                    answer_results = (
                        [comparison]
                        if comparison.success
                        else [*results, comparison]
                    )
                    recorded_results = [*results, comparison]
                yield self._status_response("✅ Validation passed. Generating answer...")
                final_answer, answer_usage = yield from self._stream_answer(
                    user_query=user_query,
                    plan=plan,
                    tool_results=answer_results,
                    token_usage=token_usage,
                    plan_history=all_plans,
                    retry_count=attempt,
                    validation_passed=validation.passed,
                    validation_issues=validation.issue_messages,
                )

                token_usage = merge_usage(token_usage, answer_usage)
                self._log(
                    "[Token] input=%d cached_input=%d output=%d total=%d"
                    % (
                        token_usage["input_tokens"],
                        token_usage["cached_input_tokens"],
                        token_usage["output_tokens"],
                        token_usage["total_tokens"],
                    )
                )

                yield AgentResponse(
                    answer=final_answer,
                    risk_level=self.llm_agent._extract_risk(final_answer),
                    context_used={
                        "tool_results": [r.model_dump() for r in recorded_results],
                        "token_usage": token_usage,
                        "validation_passed": validation.passed,
                        "validation_issues": validation.issue_messages,
                    },
                    plan_history=all_plans,
                    retry_count=attempt,
                )
                return

            if attempt < self.max_retries:
                yield self._status_response(
                    "🔁 Validation did not pass. Re-planning based on identified issues..."
                )
                plan, replan_usage = self.planner.replan(
                    previous_plan=plan,
                    issues=validation.issue_messages,
                    user_query=user_query,
                )
                token_usage = merge_usage(token_usage, replan_usage)
                all_plans.append(plan)
                yield self._status_response(
                    f"🧭 Re-plan ready: {len(plan.steps)} steps. Re-running tool execution..."
                )

        response, failure_usage = self.llm_agent.generate_failure_response(
            user_query=user_query,
            plans=all_plans,
            issues=validation.issue_messages if validation else ["Unknown error"],
            answer_set=self._answer_set_from_context(active_context),
        )
        token_usage = merge_usage(token_usage, failure_usage)
        response.context_used["token_usage"] = token_usage
        response.context_used["validation_passed"] = validation.passed if validation else None
        response.context_used["validation_issues"] = validation.issue_messages if validation else ["Unknown error"]
        self._log(
            "[Token] input=%d cached_input=%d output=%d total=%d"
            % (
                token_usage["input_tokens"],
                token_usage["cached_input_tokens"],
                token_usage["output_tokens"],
                token_usage["total_tokens"],
            )
        )
        yield response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stream_answer(
        self,
        *,
        user_query: str,
        plan: Plan,
        tool_results: list[ToolResult],
        token_usage: dict[str, int],
        plan_history: list[Plan],
        retry_count: int,
        validation_passed: bool | None,
        validation_issues: list[str],
        extra_context: str = "",
    ) -> Iterator[AgentResponse]:
        """Stream partial answers and return the final text plus usage."""
        final_answer = ""
        answer_usage = zero_usage()
        tool_result_payload = [result.model_dump() for result in tool_results]

        for event in self.llm_agent.generate_answer(
            user_query=user_query,
            plan=plan,
            tool_results=tool_results,
            extra_context=extra_context,
        ):
            if event.usage is not None:
                answer_usage = event.usage
            if not event.delta:
                continue
            final_answer += event.delta
            yield AgentResponse(
                answer=final_answer,
                risk_level=None,
                context_used={
                    "tool_results": tool_result_payload,
                    "token_usage": token_usage,
                    "validation_passed": validation_passed,
                    "validation_issues": validation_issues,
                },
                plan_history=plan_history,
                retry_count=retry_count,
            )

        return final_answer, answer_usage

    def _previous_window_comparison_result(
        self,
        *,
        results: list[ToolResult],
        active_context: dict[str, Any] | None,
    ) -> ToolResult:
        """Run internal temporal orchestration without registering a tool call."""
        step_id = max((result.step_id for result in results), default=0) + 1
        context = active_context or {}
        required = ("dataset", "patient_id", "window_start_s", "window_end_s")
        missing = [key for key in required if context.get(key) is None]
        if missing:
            return ToolResult(
                step_id=step_id,
                tool_name=PREVIOUS_WINDOW_COMPARISON_RESULT,
                success=False,
                error=(
                    "Previous-window orchestration requires active locator fields: "
                    + ", ".join(missing)
                ),
            )

        self._log("[Orchestration] Running previous-window comparison")
        current_data = self._current_window_data_for_comparison(
            dataset=str(context["dataset"]),
            results=results,
        )
        raw = compare_with_previous_window(
            dataset=str(context["dataset"]),
            patient_id=str(context["patient_id"]),
            window_start_s=float(context["window_start_s"]),
            window_end_s=float(context["window_end_s"]),
            recording_id=(
                str(context["recording_id"])
                if context.get("recording_id") is not None
                else None
            ),
            current_data=current_data,
        )
        return ToolResult(
            step_id=step_id,
            tool_name=PREVIOUS_WINDOW_COMPARISON_RESULT,
            success=bool(raw.get("success", False)),
            data={key: value for key, value in raw.items() if key not in ("success", "error")},
            error=str(raw.get("error") or ""),
        )

    @staticmethod
    def _current_window_data_for_comparison(
        *,
        dataset: str,
        results: list[ToolResult],
    ) -> dict[str, Any] | None:
        """Reuse complete current-window analyzer output when it is available."""
        successful = {
            result.tool_name: result
            for result in results
            if result.success and isinstance(result.data.get("data"), dict)
        }
        single_analyzers = {
            "ppg_dalia": "analyze_ppg_dalia_window_signal",
            "icentia11k": "analyze_icentia11k_ecg_window_signal",
            "wesad": "analyze_wesad_window_signal",
        }
        if dataset in single_analyzers:
            result = successful.get(single_analyzers[dataset])
            return None if result is None else dict(result.data["data"])

        if dataset == "afppgecg":
            pulse = successful.get("analyze_pulse_rate")
            rhythm = successful.get("analyze_ppg_rhythm_irregularity")
            if pulse is None or rhythm is None:
                return None
            return {
                **dict(pulse.data["data"]),
                **dict(rhythm.data["data"]),
            }
        return None

    def _resolve_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        prior_results: list[ToolResult],
        active_record_id: str | None,
        active_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve the active-record placeholder and inject locator arguments."""
        del prior_results  # Reserved for result-dependent argument resolution.
        resolved = dict(tool_args)
        if active_record_id:
            for key, value in resolved.items():
                if not isinstance(value, str):
                    continue
                token = value.strip().strip("\"'").lower()
                if token in {"<active_record_id>", "<current_record_id>"}:
                    resolved[key] = str(active_record_id)

        return self._apply_active_locator_args(
            tool_name=tool_name,
            resolved_args=resolved,
            active_record_id=active_record_id,
            active_context=active_context,
        )

    @staticmethod
    def _active_context_from_agent_input(agent_input: Any) -> dict[str, Any]:
        locator = agent_input.window_locator
        context = {
            "dataset": locator.dataset,
            "patient_id": locator.patient_id,
            "subject_id": locator.patient_id,
            "window_start_s": locator.window_start_s,
            "window_end_s": locator.window_end_s,
            "question": str(getattr(agent_input, "question", "") or ""),
        }
        recording_id = getattr(locator, "recording_id", None)
        if recording_id:
            context["recording_id"] = recording_id
            context["segment_id"] = recording_id
        answer_set = getattr(agent_input, "answer_set", None)
        if answer_set:
            context["answer_set"] = list(answer_set)
        return context

    @staticmethod
    def _query_from_agent_input(agent_input: Any) -> str:
        locator = agent_input.window_locator
        locator_payload = {
            "dataset": locator.dataset,
            "patient_id": locator.patient_id,
            "window_start_s": locator.window_start_s,
            "window_end_s": locator.window_end_s,
        }
        recording_id = getattr(locator, "recording_id", None)
        if recording_id:
            locator_payload["recording_id"] = recording_id
        answer_set = getattr(agent_input, "answer_set", None)
        answer_set_note = ""
        if answer_set:
            answer_set_note = (
                "\n\nAnswer using one of these allowed answer labels when possible: "
                + json.dumps(list(answer_set), ensure_ascii=False)
            )
        allows_adaptive_scope = (
            "Observation scope policy: this is an aggregate rhythm/AF question"
            in str(agent_input.question)
        )
        locator_instruction = (
            "Use this locator as the anchor. When tools support explicit scope "
            "parameters, you may use the observation scope policy from the question; "
            "when tools ask for subject_id, use the locator patient_id. Otherwise "
            "use tool arguments dataset, patient_id, window_start_s, window_end_s, "
            "and recording_id whenever supported."
            if allows_adaptive_scope
            else (
                "Use this locator exactly. When tools ask for subject_id, use the "
                "locator patient_id. Do not infer timestamps from IDs or window indices. "
                "Use tool arguments dataset, patient_id, window_start_s, window_end_s, "
                "and recording_id whenever supported."
            )
        )
        return (
            f"{agent_input.question}{answer_set_note}\n\n"
            "Benchmark locator context: "
            f"{json.dumps(locator_payload, ensure_ascii=False)}\n\n"
            f"{locator_instruction}"
        )

    @staticmethod
    def _active_record_id_from_context(context: dict[str, Any]) -> str | None:
        if str(context.get("dataset")) == "afppgecg" and context.get("patient_id"):
            return f"ppg:{context['patient_id']}"
        return None

    @staticmethod
    def _answer_set_from_context(active_context: dict[str, Any] | None) -> list[str] | None:
        if not active_context:
            return None
        answer_set = active_context.get("answer_set")
        if not answer_set:
            return None
        return [str(label) for label in answer_set]

    @staticmethod
    def _apply_active_locator_args(
        *,
        tool_name: str,
        resolved_args: dict[str, Any],
        active_record_id: str | None,
        active_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not active_context:
            return resolved_args

        args = dict(resolved_args)
        dataset = active_context.get("dataset")
        patient_id = active_context.get("patient_id") or active_context.get("subject_id")
        window_start_s = active_context.get("window_start_s")
        window_end_s = active_context.get("window_end_s")
        recording_id = active_context.get("recording_id")

        if tool_name.startswith("state_get_"):
            args.setdefault("dataset", dataset)
            args.setdefault("patient_id", patient_id)
            args.setdefault("subject_id", patient_id)
            if tool_name == "state_get_current_monitoring_state":
                args.setdefault("window_start_s", window_start_s)
                args.setdefault("timestamp_s", window_start_s)
            elif tool_name in {
                "state_get_monitoring_window",
                "state_get_longitudinal_trend",
            }:
                args.setdefault("window_start_s", window_start_s)
                args.setdefault("window_end_s", window_end_s)
                args.setdefault("start_s", window_start_s)
                args.setdefault("end_s", window_end_s)

        if tool_name in {
            "proactive_get_recent_alerts",
            "proactive_explain_last_alert",
        }:
            args.setdefault("patient_id", patient_id)
        if tool_name == "proactive_load_patient_context":
            context_path = (
                active_context.get("proactive_context_path")
                or active_context.get("patient_context_path")
                or active_context.get("context_path")
            )
            if context_path is not None:
                args.setdefault("context_path", context_path)

        if tool_name in {
            "get_ppg_description",
            "get_ppg_metadata",
            "analyze_pulse_rate",
            "analyze_prv",
            "assess_ppg_signal_quality",
            "analyze_ppg_rhythm_irregularity",
            "analyze_ppg_dalia_window_signal",
            "analyze_wesad_window_signal",
            "classify_wesad_stress_state",
            "analyze_icentia11k_ecg_window_signal",
            "evaluate_proactive_rules",
            "analyze_afppgecg_rhythm_context",
        }:
            args.setdefault("dataset", dataset)
            args.setdefault("patient_id", patient_id)
            args.setdefault("window_start_s", window_start_s)
            args.setdefault("window_end_s", window_end_s)
            if tool_name in {
                "analyze_ppg_dalia_window_signal",
                "analyze_wesad_window_signal",
                "classify_wesad_stress_state",
            }:
                args.setdefault("subject_id", patient_id)
            if tool_name == "analyze_icentia11k_ecg_window_signal":
                args.setdefault("recording_id", recording_id)
                args.setdefault("segment_id", recording_id)
            if tool_name == "evaluate_proactive_rules":
                args.setdefault("recording_id", recording_id)
                args.setdefault("segment_id", recording_id)
            if tool_name == "analyze_afppgecg_rhythm_context":
                args.setdefault("anchor_window_start_s", window_start_s)
                args.setdefault("anchor_window_end_s", window_end_s)
            if tool_name == "analyze_pulse_rate" and dataset == "afppgecg":
                question = str(active_context.get("question") or "").lower()
                if any(
                    phrase in question
                    for phrase in (
                        "highest heart rate",
                        "maximum heart rate",
                        "how high did my heart rate",
                    )
                ):
                    args.setdefault("include_recording_context", True)
            if active_record_id:
                args.setdefault("record_id", active_record_id)

        return {key: value for key, value in args.items() if value is not None}

    @staticmethod
    def _status_response(message: str) -> AgentResponse:
        return AgentResponse(
            answer=message,
            risk_level=None,
            context_used={"event": "status"},
            plan_history=[],
            retry_count=0,
        )

    def _log_tool_execution(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        success: bool,
        error: str = "",
    ) -> None:
        color_stage = "tool_result_success" if success else "tool_result_failure"
        self._log(f"  [Exec] {tool_name}({tool_args})", color_stage=color_stage)
        self._log(
            f"  [Exec] → success={success}" + (f", error={error}" if error else ""),
            color_stage=color_stage,
        )

    def _log(self, msg: str, *, color_stage: str | None = None) -> None:
        if self.verbose:
            logger.info(msg, extra={"color_stage": color_stage})

