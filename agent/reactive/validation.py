"""
Validation Gate — checks whether collected tool results are sufficient
and consistent before handing them to the LLM for answer generation.

Four-dimensional validation:
1. Completeness — did every planned step produce a result?
2. Success      — did each tool return success=True?
3. Data presence — are critical fields non-empty?
4. Consistency  — do tool outputs contradict each other?

Issues are graded by severity:
- CRITICAL: core analysis tool failure → triggers replan
- WARNING: supplementary tool failure or consistency conflict → logged, does not block
"""

from __future__ import annotations

import logging

from agent.schemas import (
    IssueSeverity,
    IssueType,
    Plan,
    ToolResult,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ValidationGate:
    """Validates tool execution results against the plan's expectations."""

    # ---- Tool output schema expectations ----
    REQUIRED_FIELDS: dict[str, list[str]] = {
        "get_ecg_description": ["description"],
        "get_ecg_metadata": ["metadata"],
        "get_ppg_description": ["description"],
        "get_ppg_metadata": ["metadata"],
        "list_ecg_records": ["record_ids"],
        "list_ppg_records": ["record_ids"],
        "analyze_heart_rate": ["data", "metadata"],
        "analyze_hrv": ["data", "metadata"],
        "analyze_pulse_rate": ["data", "metadata"],
        "analyze_prv": ["data", "metadata"],
        "analyze_morphology": ["data", "metadata"],
        "ecg_diagnosis": ["data", "metadata"],
        "assess_signal_quality": ["data", "metadata"],
        "assess_ppg_signal_quality": ["data", "metadata"],
        "medical_info_search": ["data"],
        "medical_knowledge": ["data"],
        "analyze_lead_morphology": ["data", "metadata"],
        "assess_all_leads_quality": ["data", "metadata"],
        "analyze_ppg_rhythm_irregularity": ["data", "metadata"],
        "state_load_monitoring_states": ["state_count", "contexts"],
        "state_build_from_wesad_pickle": ["dataset", "state_count", "contexts"],
        "state_build_from_ecg_record": ["dataset", "state_count", "contexts"],
        "state_build_from_ppg_patient": ["dataset", "state_count", "contexts"],
        "state_list_contexts": ["contexts"],
        "state_get_current_monitoring_state": ["state", "evidence"],
        "state_get_monitoring_window": ["state_count", "states", "window"],
        "state_get_previous_monitoring_state": ["state"],
        "state_get_longitudinal_trend": ["target", "summary", "evidence"],
        "state_get_evidence": ["evidence"],
        "state_get_dataset_capabilities": ["capabilities"],
        "proactive_load_patient_context": ["context_count", "contexts"],
        "proactive_list_patient_contexts": ["contexts"],
        "proactive_get_recent_alerts": ["alerts", "alert_summary"],
        "proactive_explain_last_alert": ["alert", "explanation"],
        "evaluate_proactive_rules": ["data", "metadata"],
        "analyze_afppgecg_rhythm_context": ["data", "metadata"],
    }

    # ---- Severity classification ----
    # Tools whose failure is CRITICAL (core ECG analysis, directly needed to answer)
    CRITICAL_TOOLS: set[str] = {
        "analyze_heart_rate",
        "analyze_hrv",
        "analyze_pulse_rate",
        "analyze_prv",
        "analyze_morphology",
        "analyze_lead_morphology",
        "analyze_ppg_rhythm_irregularity",
        "ecg_diagnosis",
        "get_ecg_metadata",
        "get_ppg_metadata",
        "state_load_monitoring_states",
        "state_build_from_wesad_pickle",
        "state_build_from_ecg_record",
        "state_build_from_ppg_patient",
        "state_get_current_monitoring_state",
        "state_get_longitudinal_trend",
        "state_get_evidence",
        "evaluate_proactive_rules",
        "analyze_afppgecg_rhythm_context",
    }
    # All other tools default to WARNING severity when they fail.

    def _severity_for(self, tool_name: str) -> IssueSeverity:
        """Determine issue severity based on tool criticality."""
        if tool_name in self.CRITICAL_TOOLS:
            return IssueSeverity.CRITICAL
        return IssueSeverity.WARNING

    def _is_missing_required_field(self, field_name: str, data: dict[str, object]) -> bool:
        """Return True when a required field should be treated as missing."""
        if field_name not in data:
            return True

        value = data[field_name]
        if value is None:
            return True

        # Many tools legitimately return empty metadata objects.
        if field_name == "metadata":
            return False

        return not value

    # ================================================================
    # Main entry point
    # ================================================================

    def validate(
        self,
        plan: Plan,
        results: list[ToolResult],
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        total_steps = len(plan.steps)
        successful_steps = 0

        result_map: dict[int, ToolResult] = {result.step_id: result for result in results}

        # ---- Check 1-3: Completeness, Success, Data presence ----
        for step in plan.steps:
            result = result_map.get(step.step_id)

            if result is None:
                issues.append(
                    ValidationIssue(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        issue_type=IssueType.MISSING_RESULT,
                        severity=self._severity_for(step.tool_name),
                        message=f"Step {step.step_id} ({step.tool_name}): no result returned.",
                    )
                )
                continue

            if not result.success:
                issues.append(
                    ValidationIssue(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        issue_type=IssueType.TOOL_FAILURE,
                        severity=self._severity_for(step.tool_name),
                        message=f"Step {step.step_id} ({step.tool_name}) failed: {result.error}",
                    )
                )
                continue

            # Data presence
            required = self.REQUIRED_FIELDS.get(step.tool_name, [])
            has_missing_required_field = False
            for field_name in required:
                if self._is_missing_required_field(field_name, result.data):
                    has_missing_required_field = True
                    issues.append(
                        ValidationIssue(
                            step_id=step.step_id,
                            tool_name=step.tool_name,
                            issue_type=IssueType.MISSING_FIELD,
                            severity=self._severity_for(step.tool_name),
                            message=(
                                f"Step {step.step_id} ({step.tool_name}): "
                                f"missing required field '{field_name}'."
                            ),
                        )
                    )

            if not has_missing_required_field:
                successful_steps += 1

        # ---- Check 4: Cross-tool consistency ----
        issues.extend(self._check_consistency(result_map))

        # ---- Compute result ----
        coverage = successful_steps / total_steps if total_steps > 0 else 1.0
        # CRITICAL issues block; WARNING-only issues pass
        passed = not any(issue.severity == IssueSeverity.CRITICAL for issue in issues)

        result = ValidationResult(passed=passed, issues=issues, coverage=coverage)

        n_critical = sum(1 for issue in issues if issue.severity == IssueSeverity.CRITICAL)
        n_warning = sum(1 for issue in issues if issue.severity == IssueSeverity.WARNING)
        logger.info(
            "Validation %s (coverage=%.0f%%, critical=%d, warnings=%d)",
            "PASSED" if passed else "FAILED",
            coverage * 100,
            n_critical,
            n_warning,
        )
        return result

    # ================================================================
    # Cross-tool consistency checks
    # ================================================================

    def _check_consistency(
        self,
        result_map: dict[int, ToolResult],
    ) -> list[ValidationIssue]:
        """
        Check for contradictions between successful tool outputs.
        All consistency issues are WARNING severity — contradictions indicate
        uncertainty, not necessarily errors.
        """
        issues: list[ValidationIssue] = []

        # Collect successful results by tool name for cross-referencing
        by_tool: dict[str, list[ToolResult]] = {}
        for result in result_map.values():
            if result.success and result.data:
                by_tool.setdefault(result.tool_name, []).append(result)

        # Rule 1: HR assessment vs ECG diagnosis contradiction
        issues.extend(self._check_hr_vs_diagnosis(by_tool))

        # Rule 2: HR vs morphology RR interval contradiction
        issues.extend(self._check_hr_vs_morphology(by_tool))

        # Rule 3: Poor signal quality degrades all signal-dependent results
        issues.extend(self._check_signal_quality_impact(by_tool))

        return issues

    def _first_result(
        self,
        by_tool: dict[str, list[ToolResult]],
        tool_name: str,
    ) -> ToolResult | None:
        """Pick the earliest successful result for a tool."""
        matches = by_tool.get(tool_name, [])
        if not matches:
            return None
        return min(matches, key=lambda result: result.step_id)

    def _assessment_contains_any(
        self,
        assessment: str,
        keywords: set[str],
    ) -> bool:
        """Return True when the assessment text includes any keyword."""
        return any(keyword in assessment for keyword in keywords)

    def _check_hr_vs_diagnosis(
        self,
        by_tool: dict[str, list[ToolResult]],
    ) -> list[ValidationIssue]:
        """
        Rule 1: If analyze_heart_rate says tachycardia but ecg_diagnosis
        lists bradycardia-related conditions (or vice versa), flag it.
        """
        hr_result = self._first_result(by_tool, "analyze_heart_rate")
        diag_result = self._first_result(by_tool, "ecg_diagnosis")
        if not hr_result or not diag_result:
            return []

        # Extract HR assessment
        hr_data = hr_result.data.get("data", hr_result.data)
        hr_assessment = ""
        if isinstance(hr_data, dict):
            hr_assessment = str(hr_data.get("assessment", "")).lower()

        # Extract diagnosis present labels
        diag_data = diag_result.data.get("data", diag_result.data)
        present_labels: list[str] = []
        if isinstance(diag_data, dict):
            present_raw = diag_data.get("present", [])
            if isinstance(present_raw, list):
                present_labels = [
                    (item.get("label", "") if isinstance(item, dict) else str(item)).lower()
                    for item in present_raw
                ]

        brady_keywords = {"bradycardia", "sinus bradycardia", "1st degree av block"}
        tachy_keywords = {
            "tachycardia",
            "sinus tachycardia",
            "atrial fibrillation",
            "atrial flutter",
            "supraventricular tachycardia",
        }

        issues: list[ValidationIssue] = []

        if self._assessment_contains_any(hr_assessment, tachy_keywords):
            brady_matches = [
                label
                for label in present_labels
                if any(keyword in label for keyword in brady_keywords)
            ]
            if brady_matches:
                issues.append(
                    ValidationIssue(
                        step_id=None,
                        tool_name="consistency:hr_vs_diagnosis",
                        issue_type=IssueType.CONSISTENCY_CONFLICT,
                        severity=IssueSeverity.WARNING,
                        message=(
                            "Heart rate analysis indicates tachycardia, but ecg_diagnosis "
                            f"lists bradycardia-related findings: {', '.join(brady_matches)}. "
                            "Results may be contradictory."
                        ),
                    )
                )

        elif self._assessment_contains_any(hr_assessment, brady_keywords):
            tachy_matches = [
                label
                for label in present_labels
                if any(keyword in label for keyword in tachy_keywords)
            ]
            if tachy_matches:
                issues.append(
                    ValidationIssue(
                        step_id=None,
                        tool_name="consistency:hr_vs_diagnosis",
                        issue_type=IssueType.CONSISTENCY_CONFLICT,
                        severity=IssueSeverity.WARNING,
                        message=(
                            "Heart rate analysis indicates bradycardia, but ecg_diagnosis "
                            f"lists tachycardia-related findings: {', '.join(tachy_matches)}. "
                            "Results may be contradictory."
                        ),
                    )
                )

        return issues

    def _check_hr_vs_morphology(
        self,
        by_tool: dict[str, list[ToolResult]],
    ) -> list[ValidationIssue]:
        """
        Rule 2: If HR is normal range but morphology reports extreme RR intervals,
        flag the inconsistency.
        """
        hr_result = self._first_result(by_tool, "analyze_heart_rate")
        morph_result = self._first_result(by_tool, "analyze_morphology")
        if not hr_result or not morph_result:
            return []

        hr_data = hr_result.data.get("data", hr_result.data)
        hr_assessment = str(hr_data.get("assessment", "")).lower() if isinstance(hr_data, dict) else ""

        morph_data = morph_result.data.get("data", morph_result.data)
        rr_mean_ms = None
        if isinstance(morph_data, dict):
            rr_info = morph_data.get("rr_interval", {})
            if isinstance(rr_info, dict):
                rr_mean_ms = rr_info.get("mean_ms")

        if (
            hr_assessment == "normal"
            and isinstance(rr_mean_ms, int | float)
            and (rr_mean_ms < 500 or rr_mean_ms > 1200)
        ):
            return [
                ValidationIssue(
                    step_id=None,
                    tool_name="consistency:hr_vs_morphology",
                    issue_type=IssueType.CONSISTENCY_CONFLICT,
                    severity=IssueSeverity.WARNING,
                    message=(
                        "Heart rate is assessed as normal, but morphology reports "
                        f"mean RR interval of {rr_mean_ms:.0f} ms (expected 600-1000 ms "
                        "for normal HR). Check for measurement artifacts."
                    ),
                )
            ]

        return []

    def _check_signal_quality_impact(
        self,
        by_tool: dict[str, list[ToolResult]],
    ) -> list[ValidationIssue]:
        """
        Rule 3: If signal quality is poor/unacceptable, warn that all
        signal-dependent tool results may be unreliable.
        """
        # Check both single-lead and all-leads quality tools
        quality_tools = [
            "assess_signal_quality",
            "assess_all_leads_quality",
            "assess_ppg_signal_quality",
        ]
        poor_quality = False
        quality_detail = ""

        for tool_name in quality_tools:
            for quality_result in by_tool.get(tool_name, []):
                quality_data = quality_result.data.get("data", quality_result.data)
                if not isinstance(quality_data, dict):
                    continue

                overall = str(
                    quality_data.get(
                        "overall_quality",
                        quality_data.get("quality_label", ""),
                    )
                ).lower()
                if overall in ("poor", "unacceptable"):
                    poor_quality = True
                    quality_detail = f"{tool_name} reported quality='{overall}'"
                    break

            if poor_quality:
                break

        if not poor_quality:
            return []

        # List which signal-dependent tools were also in the results
        signal_dependent = {
            "analyze_heart_rate",
            "analyze_hrv",
            "analyze_pulse_rate",
            "analyze_prv",
            "analyze_morphology",
            "analyze_lead_morphology",
            "analyze_ppg_rhythm_irregularity",
            "ecg_diagnosis",
        }
        affected = sorted(tool_name for tool_name in signal_dependent if by_tool.get(tool_name))

        if not affected:
            return []

        return [
            ValidationIssue(
                step_id=None,
                tool_name="consistency:signal_quality",
                issue_type=IssueType.CONSISTENCY_CONFLICT,
                severity=IssueSeverity.WARNING,
                message=(
                    f"Signal quality is poor ({quality_detail}). Results from "
                    f"signal-dependent tools ({', '.join(affected)}) may be unreliable. "
                    "Interpret findings with caution."
                ),
            )
        ]
