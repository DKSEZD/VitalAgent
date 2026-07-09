from __future__ import annotations

import json
from pathlib import Path

from scripts.rebuttal import analyze_mechanism as mod


def _prediction(
    qid: str,
    condition: str,
    *,
    score: bool,
    actual_path: str,
    total_tokens: int = 100,
    llm_call_count: int = 2,
    elapsed_sec: float = 1.0,
    validation_first_passed: bool | None = None,
    validation_first_coverage: float | None = None,
    validation_last_passed: bool | None = None,
    validation_last_coverage: float | None = None,
    coverage_improved: bool | None = None,
    replan_triggered: bool = False,
    plan_changed_after_replan: bool = False,
) -> dict[str, object]:
    return {
        "question_id": qid,
        "condition": condition,
        "score": score,
        "actual_path": actual_path,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed_sec,
        "trace_validation_first_passed": validation_first_passed,
        "trace_validation_first_coverage": validation_first_coverage,
        "trace_validation_last_passed": validation_last_passed,
        "trace_validation_last_coverage": validation_last_coverage,
        "trace_coverage_improved": coverage_improved,
        "trace_replan_triggered": replan_triggered,
        "trace_plan_changed_after_replan": plan_changed_after_replan,
        "trace_llm_call_count": llm_call_count,
    }


def _validation_event(
    qid: str,
    *,
    attempt: int,
    passed: bool,
    coverage: float,
    issue_types: list[str],
) -> dict[str, object]:
    return {
        "question_id": qid,
        "event": "validation_done",
        "attempt": attempt,
        "passed": passed,
        "coverage": coverage,
        "issue_types": issue_types,
    }


def _trace(
    qid: str,
    condition: str,
    events: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "question_id": qid,
        "condition": condition,
        "summary": {},
        "events": events,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_analyze_mechanism_reports_layered_metrics_and_case_lists(tmp_path: Path) -> None:
    predictions = [
        _prediction(
            "q_rescue",
            "agent",
            score=True,
            actual_path="signal_query",
            total_tokens=130,
            llm_call_count=3,
            elapsed_sec=3.0,
            validation_first_passed=False,
            validation_first_coverage=0.5,
            validation_last_passed=True,
            validation_last_coverage=1.0,
            coverage_improved=True,
            replan_triggered=True,
            plan_changed_after_replan=True,
        ),
        _prediction(
            "q_rescue",
            "agent_no_validation",
            score=False,
            actual_path="signal_query",
            total_tokens=100,
            llm_call_count=2,
            elapsed_sec=2.0,
        ),
        _prediction(
            "q_rescue",
            "agent_no_replan",
            score=False,
            actual_path="signal_query",
            total_tokens=110,
            llm_call_count=2,
            elapsed_sec=2.5,
            validation_first_passed=False,
        ),
        _prediction(
            "q_fallback",
            "agent",
            score=True,
            actual_path="failure",
            total_tokens=120,
            llm_call_count=2,
            elapsed_sec=1.5,
            validation_first_passed=True,
            validation_first_coverage=1.0,
            validation_last_passed=True,
            validation_last_coverage=1.0,
            coverage_improved=False,
        ),
        _prediction(
            "q_fallback",
            "agent_no_validation",
            score=True,
            actual_path="signal_query",
            total_tokens=100,
            llm_call_count=2,
            elapsed_sec=1.0,
        ),
        _prediction(
            "q_fallback",
            "agent_no_replan",
            score=True,
            actual_path="signal_query",
            total_tokens=100,
            llm_call_count=2,
            elapsed_sec=1.0,
            validation_first_passed=True,
        ),
        _prediction(
            "q_false_replan",
            "agent",
            score=False,
            actual_path="failure",
            total_tokens=150,
            llm_call_count=3,
            elapsed_sec=4.0,
            validation_first_passed=False,
            validation_first_coverage=0.0,
            validation_last_passed=False,
            validation_last_coverage=0.0,
            coverage_improved=False,
            replan_triggered=True,
            plan_changed_after_replan=False,
        ),
        _prediction(
            "q_false_replan",
            "agent_no_validation",
            score=True,
            actual_path="signal_query",
            total_tokens=100,
            llm_call_count=2,
            elapsed_sec=1.0,
        ),
        _prediction(
            "q_false_replan",
            "agent_no_replan",
            score=False,
            actual_path="failure",
            total_tokens=100,
            llm_call_count=2,
            elapsed_sec=1.0,
            validation_first_passed=False,
        ),
    ]
    traces = [
        _trace(
            "q_rescue",
            "agent",
            [
                _validation_event(
                    "q_rescue",
                    attempt=0,
                    passed=False,
                    coverage=0.5,
                    issue_types=["missing_result"],
                ),
                _validation_event(
                    "q_rescue",
                    attempt=1,
                    passed=True,
                    coverage=1.0,
                    issue_types=[],
                ),
            ],
        ),
        _trace(
            "q_false_replan",
            "agent",
            [
                _validation_event(
                    "q_false_replan",
                    attempt=0,
                    passed=False,
                    coverage=0.0,
                    issue_types=["tool_failure"],
                ),
                _validation_event(
                    "q_false_replan",
                    attempt=1,
                    passed=False,
                    coverage=0.0,
                    issue_types=["tool_failure"],
                ),
            ],
        ),
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    traces_path = tmp_path / "agent_traces.jsonl"
    slices_dir = tmp_path / "slices"
    output_dir = tmp_path / "analysis"
    slices_dir.mkdir()
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(traces_path, traces)
    (slices_dir / "ALL.txt").write_text(
        "q_rescue\nq_fallback\nq_false_replan\n",
        encoding="utf-8",
    )
    (slices_dir / "tool_slice.txt").write_text("q_false_replan\n", encoding="utf-8")

    rc = mod.main(
        [
            "--predictions",
            str(predictions_path),
            "--traces",
            str(traces_path),
            "--slices-dir",
            str(slices_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    report = json.loads((output_dir / "mechanism_report.json").read_text(encoding="utf-8"))
    all_slice = report["slices"]["ALL"]
    assert all_slice["accuracy"]["full"] == {
        "value": 2 / 3,
        "numerator": 2,
        "denominator": 3,
    }
    assert all_slice["layer1_detection"]["validation_fail_rate"]["value"] == 2 / 3
    assert all_slice["layer1_detection"]["detection_precision"] == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert all_slice["layer1_detection"]["detection_recall"] == {
        "value": 1.0,
        "numerator": 1,
        "denominator": 1,
    }
    assert all_slice["layer1_issue_segments"]["_tool_failure"]["detection_precision"][
        "value"
    ] == 0.0
    assert all_slice["layer1_issue_segments"]["_non_tool_failure"]["detection_precision"][
        "value"
    ] == 1.0
    assert all_slice["issue_histograms"]["attempt0_full"] == {
        "missing_result": 1,
        "tool_failure": 1,
    }
    assert all_slice["layer2_repair"]["replan_trigger_rate"]["value"] == 2 / 3
    assert all_slice["layer2_repair"]["plan_change_rate"]["value"] == 0.5
    assert all_slice["layer2_repair"]["evidence_repair_rate"]["value"] == 0.5
    assert all_slice["layer2_repair"]["coverage_improvement_rate"]["value"] == 0.5
    assert all_slice["layer2_repair"]["replan_rescue_rate"]["value"] == 0.5
    assert all_slice["layer2_repair"]["replan_rescue_rate_strict"]["value"] == 0.5
    assert all_slice["layer2_repair"]["false_replan_rate"]["value"] == 0.5
    assert all_slice["layer3_scoring_artifact"]["fallback_correct_rate"]["value"] == 1 / 3
    assert all_slice["layer3_scoring_artifact"]["tool_failure_rate"]["value"] == 1 / 3
    assert all_slice["case_lists"]["evidence_backed_replan_rescues"] == ["q_rescue"]
    assert all_slice["case_lists"]["strict_replan_rescues"] == ["q_rescue"]
    assert all_slice["case_lists"]["false_replans"] == ["q_false_replan"]
    assert all_slice["case_lists"]["fallback_correct_full"] == ["q_fallback"]
    assert all_slice["case_lists"]["validation_detected_unrepaired"] == [
        "q_false_replan"
    ]
    assert all_slice["case_lists"]["tool_failure_cases"] == ["q_false_replan"]
    assert (output_dir / "mechanism_report.md").exists()
    assert (output_dir / "strict_replan_rescues.txt").read_text(
        encoding="utf-8"
    ) == "q_rescue\n"
    assert (output_dir / "case_ids" / "tool_slice" / "tool_failure_cases.txt").read_text(
        encoding="utf-8"
    ) == "q_false_replan\n"

