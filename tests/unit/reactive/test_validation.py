from __future__ import annotations

from agent.reactive.validation import ValidationGate
from agent.schemas import (
    IntentType,
    IssueSeverity,
    IssueType,
    Plan,
    PlanStep,
    ToolResult,
)


def _plan(*steps: PlanStep) -> Plan:
    return Plan(intent=IntentType.STATUS_QUERY, reasoning="test", steps=list(steps))


def test_validate_passes_empty_plan() -> None:
    result = ValidationGate().validate(_plan(), [])

    assert result.passed is True
    assert result.issues == []
    assert result.issue_messages == []
    assert result.coverage == 1.0


def test_validate_reports_missing_result() -> None:
    plan = _plan(
        PlanStep(step_id=1, tool_name="get_ecg_description", tool_args={}, description="describe")
    )

    result = ValidationGate().validate(plan, [])

    assert result.passed is True
    assert result.coverage == 0.0
    assert result.issue_messages == ["Step 1 (get_ecg_description): no result returned."]
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == IssueType.MISSING_RESULT
    assert result.issues[0].severity == IssueSeverity.WARNING


def test_validate_reports_failed_step() -> None:
    plan = _plan(
        PlanStep(step_id=1, tool_name="get_ecg_description", tool_args={}, description="describe")
    )
    results = [
        ToolResult(step_id=1, tool_name="get_ecg_description", success=False, error="boom"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert result.coverage == 0.0
    assert result.issue_messages == ["Step 1 (get_ecg_description) failed: boom"]
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == IssueType.TOOL_FAILURE
    assert result.issues[0].severity == IssueSeverity.WARNING


def test_validate_reports_missing_required_field_and_excludes_step_from_coverage() -> None:
    plan = _plan(
        PlanStep(step_id=1, tool_name="get_ecg_description", tool_args={}, description="describe")
    )
    results = [
        ToolResult(step_id=1, tool_name="get_ecg_description", success=True, data={}),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert result.coverage == 0.0
    assert result.issue_messages == [
        "Step 1 (get_ecg_description): missing required field 'description'."
    ]
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == IssueType.MISSING_FIELD
    assert result.issues[0].severity == IssueSeverity.WARNING


def test_validate_combines_coverage_and_issues_for_mixed_results() -> None:
    plan = _plan(
        PlanStep(step_id=1, tool_name="get_ecg_description", tool_args={}, description="describe"),
        PlanStep(step_id=2, tool_name="medical_info_search", tool_args={}, description="search"),
    )
    results = [
        ToolResult(step_id=1, tool_name="get_ecg_description", success=True, data={"description": "ok"}),
        ToolResult(step_id=2, tool_name="medical_info_search", success=False, error="timeout"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert result.coverage == 0.5
    assert result.issue_messages == ["Step 2 (medical_info_search) failed: timeout"]
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == IssueType.TOOL_FAILURE
    assert result.issues[0].severity == IssueSeverity.WARNING


def test_warning_tool_failure_still_passes() -> None:
    """medical_info_search failure is WARNING -> validation passes."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="medical_info_search", tool_args={}, description="search"),
    )
    results = [
        ToolResult(step_id=1, tool_name="medical_info_search", success=False, error="timeout"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert len(result.issues) == 1
    assert result.issues[0].severity == IssueSeverity.WARNING
    assert result.issues[0].issue_type == IssueType.TOOL_FAILURE


def test_critical_tool_failure_blocks() -> None:
    """analyze_heart_rate failure is CRITICAL -> validation fails."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
    )
    results = [
        ToolResult(step_id=1, tool_name="analyze_heart_rate", success=False, error="crash"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is False
    assert len(result.issues) == 1
    assert result.issues[0].severity == IssueSeverity.CRITICAL


def test_state_tool_required_fields_are_validated() -> None:
    plan = _plan(
        PlanStep(
            step_id=1,
            tool_name="state_get_current_monitoring_state",
            tool_args={},
            description="state",
        ),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="state_get_current_monitoring_state",
            success=True,
            data={"state": {"state_id": "s1"}},
        ),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is False
    assert result.issues[0].issue_type == IssueType.MISSING_FIELD
    assert result.issues[0].severity == IssueSeverity.CRITICAL
    assert "evidence" in result.issue_messages[0]


def test_state_list_contexts_failure_is_warning() -> None:
    plan = _plan(
        PlanStep(
            step_id=1,
            tool_name="state_list_contexts",
            tool_args={},
            description="list",
        ),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="state_list_contexts",
            success=False,
            error="empty",
        ),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert result.issues[0].severity == IssueSeverity.WARNING


def test_state_dataset_capabilities_failure_is_warning() -> None:
    plan = _plan(
        PlanStep(
            step_id=1,
            tool_name="state_get_dataset_capabilities",
            tool_args={"dataset": "missing"},
            description="caps",
        ),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="state_get_dataset_capabilities",
            success=False,
            error="unknown dataset",
        ),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert result.issues[0].severity == IssueSeverity.WARNING


def test_mixed_critical_and_warning() -> None:
    """One CRITICAL failure + one WARNING failure -> overall fails."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
        PlanStep(step_id=2, tool_name="medical_info_search", tool_args={}, description="search"),
    )
    results = [
        ToolResult(step_id=1, tool_name="analyze_heart_rate", success=False, error="fail"),
        ToolResult(step_id=2, tool_name="medical_info_search", success=False, error="timeout"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is False
    assert result.has_critical is True
    assert len(result.critical_issues) == 1
    assert len(result.warnings) == 1


def test_all_warnings_passes() -> None:
    """Multiple WARNING issues but zero CRITICAL -> validation passes."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="medical_info_search", tool_args={}, description="s1"),
        PlanStep(step_id=2, tool_name="medical_knowledge", tool_args={}, description="s2"),
    )
    results = [
        ToolResult(step_id=1, tool_name="medical_info_search", success=False, error="t1"),
        ToolResult(step_id=2, tool_name="medical_knowledge", success=False, error="t2"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert len(result.issues) == 2
    assert all(issue.severity == IssueSeverity.WARNING for issue in result.issues)


def test_ppg_critical_tool_failure_blocks() -> None:
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_pulse_rate", tool_args={}, description="pulse"),
    )
    results = [
        ToolResult(step_id=1, tool_name="analyze_pulse_rate", success=False, error="noisy"),
    ]

    result = ValidationGate().validate(plan, results)

    assert result.passed is False
    assert result.issues[0].severity == IssueSeverity.CRITICAL


def test_ppg_poor_quality_warns_signal_dependent_tools() -> None:
    plan = _plan(
        PlanStep(step_id=1, tool_name="assess_ppg_signal_quality", tool_args={}, description="quality"),
        PlanStep(step_id=2, tool_name="analyze_pulse_rate", tool_args={}, description="pulse"),
        PlanStep(step_id=3, tool_name="analyze_prv", tool_args={}, description="prv"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="assess_ppg_signal_quality",
            success=True,
            data={"data": {"quality_label": "poor"}, "metadata": {}},
        ),
        ToolResult(
            step_id=2,
            tool_name="analyze_pulse_rate",
            success=True,
            data={"data": {"assessment": "normal"}, "metadata": {}},
        ),
        ToolResult(
            step_id=3,
            tool_name="analyze_prv",
            success=True,
            data={"data": {"time_domain": {"sdnn_ms": 20.0}}, "metadata": {}},
        ),
    ]

    result = ValidationGate().validate(plan, results)

    consistency_issues = [
        issue for issue in result.issues if issue.issue_type == IssueType.CONSISTENCY_CONFLICT
    ]
    assert len(consistency_issues) == 1
    assert "analyze_pulse_rate" in consistency_issues[0].message
    assert "analyze_prv" in consistency_issues[0].message


def test_consistency_hr_tachycardia_vs_brady_diagnosis() -> None:
    """HR says tachycardia, diagnosis says bradycardia -> WARNING."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
        PlanStep(step_id=2, tool_name="ecg_diagnosis", tool_args={}, description="diag"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "tachycardia", "heart_rate": {"mean_bpm": 120}},
                "metadata": {},
            },
        ),
        ToolResult(
            step_id=2,
            tool_name="ecg_diagnosis",
            success=True,
            data={
                "data": {"present": [{"label": "sinus bradycardia", "confidence": 0.8}]},
                "metadata": {},
            },
        ),
    ]
    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    consistency_issues = [
        issue for issue in result.issues if issue.issue_type == IssueType.CONSISTENCY_CONFLICT
    ]
    assert len(consistency_issues) == 1
    assert "tachycardia" in consistency_issues[0].message
    assert "bradycardia" in consistency_issues[0].message


def test_consistency_hr_substring_assessment_matches() -> None:
    """Assessment variants like 'sinus tachycardia' should still match."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
        PlanStep(step_id=2, tool_name="ecg_diagnosis", tool_args={}, description="diag"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "Sinus Tachycardia", "heart_rate": {"mean_bpm": 118}},
                "metadata": {},
            },
        ),
        ToolResult(
            step_id=2,
            tool_name="ecg_diagnosis",
            success=True,
            data={
                "data": {"present": [{"label": "sinus bradycardia", "confidence": 0.8}]},
                "metadata": {},
            },
        ),
    ]

    result = ValidationGate().validate(plan, results)

    consistency_issues = [
        issue for issue in result.issues if issue.issue_type == IssueType.CONSISTENCY_CONFLICT
    ]
    assert len(consistency_issues) == 1


def test_consistency_uses_earliest_result_when_tool_appears_multiple_times() -> None:
    """Repeated tool names should not be overwritten by later results."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr1"),
        PlanStep(step_id=2, tool_name="analyze_heart_rate", tool_args={}, description="hr2"),
        PlanStep(step_id=3, tool_name="ecg_diagnosis", tool_args={}, description="diag"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "tachycardia", "heart_rate": {"mean_bpm": 120}},
                "metadata": {},
            },
        ),
        ToolResult(
            step_id=2,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "normal", "heart_rate": {"mean_bpm": 75}},
                "metadata": {},
            },
        ),
        ToolResult(
            step_id=3,
            tool_name="ecg_diagnosis",
            success=True,
            data={
                "data": {"present": [{"label": "sinus bradycardia", "confidence": 0.8}]},
                "metadata": {},
            },
        ),
    ]

    result = ValidationGate().validate(plan, results)

    consistency_issues = [
        issue for issue in result.issues if issue.tool_name == "consistency:hr_vs_diagnosis"
    ]
    assert len(consistency_issues) == 1


def test_consistency_hr_vs_morphology_uses_tighter_rr_thresholds() -> None:
    """Normal HR with borderline-extreme RR intervals should warn."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
        PlanStep(step_id=2, tool_name="analyze_morphology", tool_args={}, description="morph"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "normal", "heart_rate": {"mean_bpm": 72}},
                "metadata": {},
            },
        ),
        ToolResult(
            step_id=2,
            tool_name="analyze_morphology",
            success=True,
            data={
                "data": {"rr_interval": {"mean_ms": 450}},
                "metadata": {},
            },
        ),
    ]

    result = ValidationGate().validate(plan, results)

    rr_issues = [issue for issue in result.issues if issue.tool_name == "consistency:hr_vs_morphology"]
    assert len(rr_issues) == 1
    assert "450 ms" in rr_issues[0].message


def test_consistency_poor_signal_quality_warns() -> None:
    """Poor signal quality + signal-dependent tools -> WARNING about reliability."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="assess_signal_quality", tool_args={}, description="sq"),
        PlanStep(step_id=2, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="assess_signal_quality",
            success=True,
            data={
                "data": {"overall_quality": "poor"},
                "metadata": {},
            },
        ),
        ToolResult(
            step_id=2,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "normal", "heart_rate": {"mean_bpm": 75}},
                "metadata": {},
            },
        ),
    ]
    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    quality_issues = [issue for issue in result.issues if issue.tool_name == "consistency:signal_quality"]
    assert len(quality_issues) == 1
    assert "unreliable" in quality_issues[0].message


def test_no_consistency_issues_when_tools_agree() -> None:
    """Normal HR + no contradicting diagnosis -> no consistency issues."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="analyze_heart_rate",
            success=True,
            data={
                "data": {"assessment": "normal", "heart_rate": {"mean_bpm": 75}},
                "metadata": {},
            },
        ),
    ]
    result = ValidationGate().validate(plan, results)

    assert result.passed is True
    assert len(result.issues) == 0


def test_issue_messages_returns_plain_strings() -> None:
    """issue_messages property returns list[str] for backward compat."""
    plan = _plan(
        PlanStep(step_id=1, tool_name="analyze_heart_rate", tool_args={}, description="hr"),
    )
    results = [
        ToolResult(step_id=1, tool_name="analyze_heart_rate", success=False, error="boom"),
    ]
    result = ValidationGate().validate(plan, results)

    messages = result.issue_messages
    assert isinstance(messages, list)
    assert all(isinstance(message, str) for message in messages)
    assert "boom" in messages[0]
