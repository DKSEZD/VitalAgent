#!/usr/bin/env python3
"""Analyze VitalAgent validation/replan mechanism traces for rebuttal tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FULL_CONDITION = "agent"
NO_VALIDATION_CONDITION = "agent_no_validation"
NO_REPLAN_CONDITION = "agent_no_replan"
REQUIRED_CONDITIONS = (
    FULL_CONDITION,
    NO_VALIDATION_CONDITION,
    NO_REPLAN_CONDITION,
)
CONDITION_LABELS = {
    FULL_CONDITION: "full",
    NO_VALIDATION_CONDITION: "no_validation",
    NO_REPLAN_CONDITION: "no_replan",
}
CASE_LIST_NAMES = (
    "evidence_backed_replan_rescues",
    "strict_replan_rescues",
    "false_replans",
    "fallback_correct_full",
    "validation_detected_unrepaired",
    "tool_failure_cases",
)
ISSUE_CLASS_MAP = {
    "tool_failure": "tool_failure",
    "missing_field": "evidence_missing",
    "missing_result": "evidence_missing",
    "consistency_conflict": "consistency",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(payload)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def load_slices(slices_dir: Path | None, all_question_ids: set[str]) -> dict[str, set[str]]:
    if slices_dir is None:
        return {"ALL": set(all_question_ids)}
    if not slices_dir.exists():
        raise FileNotFoundError(f"slices dir not found: {slices_dir}")
    slices: dict[str, set[str]] = {}
    for path in sorted(slices_dir.glob("*.txt")):
        ids: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    ids.add(value)
        slices[path.stem] = ids
    if "ALL" not in slices:
        slices = {"ALL": set(all_question_ids), **slices}
    return slices


def trace_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("question_id") or ""), str(record.get("condition") or "")


def build_trace_index(
    trace_records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in trace_records:
        key = trace_key(record)
        if key[0] and key[1]:
            index[key] = record
    return index


def _validation_events(trace_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trace_record, dict):
        return []
    events = trace_record.get("events") or []
    if not isinstance(events, list):
        return []
    return [
        event
        for event in events
        if isinstance(event, dict) and event.get("event") == "validation_done"
    ]


def enrich_predictions_with_trace_records(
    records: list[dict[str, Any]],
    trace_index: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Backfill flat trace fields from agent_traces.jsonl when needed."""

    summary_keys = (
        "validation_first_passed",
        "validation_first_coverage",
        "validation_last_passed",
        "validation_last_coverage",
        "coverage_improved",
        "validation_issue_types",
        "replan_count",
        "replan_triggered",
        "initial_tool_names",
        "initial_plan_signature",
        "final_tool_names",
        "final_plan_signature",
        "plan_changed_after_replan",
        "tool_call_count",
        "llm_call_count",
    )

    def fill_if_missing(record: dict[str, Any], key: str, value: Any) -> None:
        if record.get(key) is None and value is not None:
            record[key] = value

    for record in records:
        trace_record = trace_index.get(trace_key(record))
        trace_summary = record.get("trace_summary")
        if not isinstance(trace_summary, dict) and isinstance(trace_record, dict):
            trace_summary = trace_record.get("summary")
            if isinstance(trace_summary, dict):
                record["trace_summary"] = trace_summary
        if isinstance(trace_summary, dict):
            for key in summary_keys:
                fill_if_missing(record, f"trace_{key}", trace_summary.get(key))

        events = _validation_events(trace_record)
        if not events:
            continue
        initial = next(
            (event for event in events if int(event.get("attempt", -1)) == 0),
            None,
        )
        final = events[-1]
        if initial is not None:
            fill_if_missing(
                record,
                "trace_validation_first_passed",
                initial.get("passed"),
            )
            fill_if_missing(
                record,
                "trace_validation_first_coverage",
                initial.get("coverage"),
            )
        fill_if_missing(record, "trace_validation_last_passed", final.get("passed"))
        fill_if_missing(record, "trace_validation_last_coverage", final.get("coverage"))
        first_coverage = record.get("trace_validation_first_coverage")
        last_coverage = record.get("trace_validation_last_coverage")
        if record.get("trace_coverage_improved") is None:
            record["trace_coverage_improved"] = _coverage_improved(
                first_coverage,
                last_coverage,
            )


def join_predictions(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates: list[dict[str, str]] = []
    for record in records:
        qid = str(record.get("question_id") or "")
        condition = str(record.get("condition") or "")
        if condition not in REQUIRED_CONDITIONS:
            continue
        if not qid:
            continue
        if condition in by_qid[qid]:
            duplicates.append({"question_id": qid, "condition": condition})
        by_qid[qid][condition] = record
    complete = {
        qid: conditions
        for qid, conditions in by_qid.items()
        if all(condition in conditions for condition in REQUIRED_CONDITIONS)
    }
    incomplete = {
        qid: sorted(set(REQUIRED_CONDITIONS) - set(conditions))
        for qid, conditions in by_qid.items()
        if qid not in complete
    }
    diagnostics = {
        "question_count_seen": len(by_qid),
        "complete_question_count": len(complete),
        "incomplete_question_count": len(incomplete),
        "incomplete_question_ids": incomplete,
        "duplicate_records": duplicates,
    }
    return complete, diagnostics


def score_true(record: dict[str, Any]) -> bool:
    return record.get("score") is True


def evidence_backed_correct(record: dict[str, Any]) -> bool:
    return score_true(record) and str(record.get("actual_path") or "") != "failure"


def fallback_correct(record: dict[str, Any]) -> bool:
    return score_true(record) and str(record.get("actual_path") or "") == "failure"


def trace_value(record: dict[str, Any], key: str) -> Any:
    flat_key = f"trace_{key}"
    if flat_key in record:
        return record.get(flat_key)
    summary = record.get("trace_summary")
    if isinstance(summary, dict):
        return summary.get(key)
    return None


def as_bool(value: Any) -> bool:
    return value is True


def validation_fired(full_record: dict[str, Any]) -> bool:
    return trace_value(full_record, "validation_first_passed") is False


def replan_triggered(full_record: dict[str, Any]) -> bool:
    return as_bool(trace_value(full_record, "replan_triggered"))


def plan_changed(full_record: dict[str, Any]) -> bool:
    return as_bool(trace_value(full_record, "plan_changed_after_replan"))


def _coverage_improved(first: Any, last: Any) -> bool | None:
    if first is None or last is None:
        return None
    try:
        return float(last) > float(first)
    except (TypeError, ValueError):
        return None


def coverage_improved(full_record: dict[str, Any]) -> bool:
    value = trace_value(full_record, "coverage_improved")
    if value is not None:
        return as_bool(value)
    return as_bool(
        _coverage_improved(
            trace_value(full_record, "validation_first_coverage"),
            trace_value(full_record, "validation_last_coverage"),
        )
    )


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def numeric_value(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        value = trace_value(record, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def issue_types_from_record(
    record: dict[str, Any],
    trace_record: dict[str, Any] | None,
    *,
    attempt0_only: bool,
) -> list[str]:
    events = _validation_events(trace_record)
    selected_events = events
    if attempt0_only:
        selected_events = [
            event for event in events if int(event.get("attempt", -1)) == 0
        ]
    if selected_events:
        issue_types: list[str] = []
        for event in selected_events:
            issue_types.extend(str(value) for value in event.get("issue_types") or [])
        return issue_types
    if attempt0_only:
        value = trace_value(record, "validation_issue_types")
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def issue_classes(issue_types: list[str]) -> set[str]:
    return {ISSUE_CLASS_MAP.get(issue_type, "other") for issue_type in issue_types}


def has_issue_class(
    record: dict[str, Any],
    trace_record: dict[str, Any] | None,
    issue_class: str,
) -> bool:
    return issue_class in issue_classes(
        issue_types_from_record(record, trace_record, attempt0_only=True)
    )


def has_non_tool_failure_issue(
    record: dict[str, Any],
    trace_record: dict[str, Any] | None,
) -> bool:
    types = issue_types_from_record(record, trace_record, attempt0_only=True)
    return bool(types) and not has_issue_class(record, trace_record, "tool_failure")


def sorted_qids(qids: set[str] | list[str]) -> list[str]:
    return sorted(qids)


def analyze_slice(
    slice_name: str,
    requested_qids: set[str],
    joined: dict[str, dict[str, dict[str, Any]]],
    trace_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    qids = sorted_qids(set(joined) & requested_qids)
    missing_qids = sorted_qids(requested_qids - set(joined))
    rows = [joined[qid] for qid in qids]

    def condition_records(condition: str) -> list[dict[str, Any]]:
        return [row[condition] for row in rows]

    def accuracy(condition: str) -> dict[str, Any]:
        records = condition_records(condition)
        return ratio(sum(1 for record in records if score_true(record)), len(records))

    full_by_qid = {qid: joined[qid][FULL_CONDITION] for qid in qids}
    no_validation_by_qid = {
        qid: joined[qid][NO_VALIDATION_CONDITION] for qid in qids
    }
    no_replan_by_qid = {qid: joined[qid][NO_REPLAN_CONDITION] for qid in qids}

    fired_qids = [qid for qid in qids if validation_fired(full_by_qid[qid])]
    no_validation_failed_qids = [
        qid
        for qid in qids
        if not evidence_backed_correct(no_validation_by_qid[qid])
    ]
    triggered_qids = [qid for qid in qids if replan_triggered(full_by_qid[qid])]

    attempt0_issue_histogram: Counter[str] = Counter()
    all_issue_histogram: Counter[str] = Counter()
    issue_class_histogram: Counter[str] = Counter()
    issue_source = "prediction_trace_fields"
    for qid in qids:
        full = full_by_qid[qid]
        trace_record = trace_index.get((qid, FULL_CONDITION))
        if trace_record is not None:
            issue_source = "agent_traces"
        attempt0_types = issue_types_from_record(
            full,
            trace_record,
            attempt0_only=True,
        )
        all_types = issue_types_from_record(
            full,
            trace_record,
            attempt0_only=False,
        )
        attempt0_issue_histogram.update(attempt0_types)
        all_issue_histogram.update(all_types)
        issue_class_histogram.update(issue_classes(attempt0_types))

    tool_failure_qids = [
        qid
        for qid in qids
        if has_issue_class(
            full_by_qid[qid],
            trace_index.get((qid, FULL_CONDITION)),
            "tool_failure",
        )
    ]
    non_tool_failure_qids = [
        qid
        for qid in qids
        if has_non_tool_failure_issue(
            full_by_qid[qid],
            trace_index.get((qid, FULL_CONDITION)),
        )
    ]

    def detection_precision_for(qid_subset: list[str]) -> dict[str, Any]:
        denominator_qids = [qid for qid in qid_subset if validation_fired(full_by_qid[qid])]
        numerator = sum(
            1
            for qid in denominator_qids
            if not evidence_backed_correct(no_validation_by_qid[qid])
        )
        return ratio(numerator, len(denominator_qids))

    evidence_backed_replan_rescues = [
        qid
        for qid in triggered_qids
        if evidence_backed_correct(full_by_qid[qid])
        and not evidence_backed_correct(no_replan_by_qid[qid])
    ]
    strict_replan_rescues = [
        qid for qid in evidence_backed_replan_rescues if plan_changed(full_by_qid[qid])
    ]
    false_replans = [
        qid
        for qid in triggered_qids
        if not evidence_backed_correct(full_by_qid[qid])
        and evidence_backed_correct(no_validation_by_qid[qid])
    ]
    fallback_correct_full = [
        qid for qid in qids if fallback_correct(full_by_qid[qid])
    ]
    validation_detected_unrepaired = [
        qid
        for qid in fired_qids
        if trace_value(full_by_qid[qid], "validation_last_passed") is not True
    ]

    full_tokens = [
        value
        for record in condition_records(FULL_CONDITION)
        if (value := numeric_value(record, "total_tokens")) is not None
    ]
    no_validation_tokens = [
        value
        for record in condition_records(NO_VALIDATION_CONDITION)
        if (value := numeric_value(record, "total_tokens")) is not None
    ]
    full_llm_calls = [
        value
        for record in condition_records(FULL_CONDITION)
        if (value := numeric_value(record, "llm_call_count")) is not None
    ]
    no_validation_llm_calls = [
        value
        for record in condition_records(NO_VALIDATION_CONDITION)
        if (value := numeric_value(record, "llm_call_count")) is not None
    ]
    full_elapsed = [
        value
        for record in condition_records(FULL_CONDITION)
        if (value := numeric_value(record, "elapsed_sec")) is not None
    ]
    no_validation_elapsed = [
        value
        for record in condition_records(NO_VALIDATION_CONDITION)
        if (value := numeric_value(record, "elapsed_sec")) is not None
    ]

    mean_full_tokens = mean(full_tokens)
    mean_no_validation_tokens = mean(no_validation_tokens)
    mean_full_llm_calls = mean(full_llm_calls)
    mean_no_validation_llm_calls = mean(no_validation_llm_calls)
    mean_full_elapsed = mean(full_elapsed)
    mean_no_validation_elapsed = mean(no_validation_elapsed)

    case_lists = {
        "evidence_backed_replan_rescues": sorted_qids(evidence_backed_replan_rescues),
        "strict_replan_rescues": sorted_qids(strict_replan_rescues),
        "false_replans": sorted_qids(false_replans),
        "fallback_correct_full": sorted_qids(fallback_correct_full),
        "validation_detected_unrepaired": sorted_qids(validation_detected_unrepaired),
        "tool_failure_cases": sorted_qids(tool_failure_qids),
    }

    return {
        "slice": slice_name,
        "requested_question_count": len(requested_qids),
        "joined_question_count": len(qids),
        "missing_question_count": len(missing_qids),
        "missing_question_ids": missing_qids,
        "accuracy": {
            CONDITION_LABELS[condition]: accuracy(condition)
            for condition in REQUIRED_CONDITIONS
        },
        "layer1_detection": {
            "validation_fail_rate": ratio(len(fired_qids), len(qids)),
            "detection_precision": ratio(
                sum(
                    1
                    for qid in fired_qids
                    if not evidence_backed_correct(no_validation_by_qid[qid])
                ),
                len(fired_qids),
            ),
            "detection_recall": ratio(
                sum(1 for qid in no_validation_failed_qids if qid in fired_qids),
                len(no_validation_failed_qids),
            ),
        },
        "layer1_issue_segments": {
            "_tool_failure": {
                "question_count": len(tool_failure_qids),
                "validation_fail_rate_within_slice": ratio(
                    len(tool_failure_qids),
                    len(qids),
                ),
                "detection_precision": detection_precision_for(tool_failure_qids),
            },
            "_non_tool_failure": {
                "question_count": len(non_tool_failure_qids),
                "validation_fail_rate_within_slice": ratio(
                    len(non_tool_failure_qids),
                    len(qids),
                ),
                "detection_precision": detection_precision_for(non_tool_failure_qids),
            },
        },
        "issue_histograms": {
            "source": issue_source if qids else "unavailable",
            "attempt0_full": dict(sorted(attempt0_issue_histogram.items())),
            "all_full_validation": dict(sorted(all_issue_histogram.items())),
            "attempt0_issue_classes": dict(sorted(issue_class_histogram.items())),
        },
        "layer2_repair": {
            "replan_trigger_rate": ratio(len(triggered_qids), len(qids)),
            "plan_change_rate": ratio(
                sum(1 for qid in triggered_qids if plan_changed(full_by_qid[qid])),
                len(triggered_qids),
            ),
            "evidence_repair_rate": ratio(
                sum(
                    1
                    for qid in triggered_qids
                    if trace_value(full_by_qid[qid], "validation_last_passed") is True
                ),
                len(triggered_qids),
            ),
            "coverage_improvement_rate": ratio(
                sum(1 for qid in triggered_qids if coverage_improved(full_by_qid[qid])),
                len(triggered_qids),
            ),
            "replan_rescue_rate": ratio(
                len(evidence_backed_replan_rescues),
                len(triggered_qids),
            ),
            "replan_rescue_rate_strict": ratio(
                len(strict_replan_rescues),
                len(triggered_qids),
            ),
            "false_replan_rate": ratio(len(false_replans), len(triggered_qids)),
            "overhead": {
                "delta_avg_total_tokens_full_minus_no_validation": (
                    None
                    if mean_full_tokens is None or mean_no_validation_tokens is None
                    else mean_full_tokens - mean_no_validation_tokens
                ),
                "delta_mean_llm_call_count_full_minus_no_validation": (
                    None
                    if mean_full_llm_calls is None or mean_no_validation_llm_calls is None
                    else mean_full_llm_calls - mean_no_validation_llm_calls
                ),
                "delta_mean_elapsed_sec_full_minus_no_validation": (
                    None
                    if mean_full_elapsed is None or mean_no_validation_elapsed is None
                    else mean_full_elapsed - mean_no_validation_elapsed
                ),
            },
        },
        "layer3_scoring_artifact": {
            "fallback_correct_rate": ratio(len(fallback_correct_full), len(qids)),
            "tool_failure_rate": ratio(len(tool_failure_qids), len(qids)),
        },
        "case_counts": {name: len(values) for name, values in case_lists.items()},
        "case_lists": case_lists,
    }


def analyze_mechanism(
    predictions_jsonl: Path,
    output_dir: Path,
    *,
    agent_traces_jsonl: Path | None = None,
    slices_dir: Path | None = None,
) -> dict[str, Any]:
    predictions = read_jsonl(predictions_jsonl)
    traces = read_jsonl(agent_traces_jsonl) if agent_traces_jsonl else []
    trace_index = build_trace_index(traces)
    enrich_predictions_with_trace_records(predictions, trace_index)
    joined, join_diagnostics = join_predictions(predictions)
    slices = load_slices(slices_dir, set(joined))

    output_dir.mkdir(parents=True, exist_ok=True)
    slice_reports = {
        name: analyze_slice(name, ids, joined, trace_index)
        for name, ids in slices.items()
    }
    report = {
        "inputs": {
            "predictions_jsonl": str(predictions_jsonl),
            "agent_traces_jsonl": str(agent_traces_jsonl)
            if agent_traces_jsonl
            else None,
            "slices_dir": str(slices_dir) if slices_dir else None,
        },
        "condition_mapping": CONDITION_LABELS,
        "join": join_diagnostics,
        "slices": slice_reports,
    }
    write_json(output_dir / "mechanism_report.json", report)
    (output_dir / "mechanism_report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    write_case_lists(output_dir, slice_reports)
    return report


def write_case_lists(output_dir: Path, slice_reports: dict[str, dict[str, Any]]) -> None:
    primary_slice = "ALL" if "ALL" in slice_reports else next(iter(slice_reports), None)
    if primary_slice is not None:
        for name in CASE_LIST_NAMES:
            write_lines(
                output_dir / f"{name}.txt",
                slice_reports[primary_slice]["case_lists"][name],
            )
    case_root = output_dir / "case_ids"
    for slice_name, report in slice_reports.items():
        slice_dir = case_root / slice_name
        slice_dir.mkdir(parents=True, exist_ok=True)
        for name in CASE_LIST_NAMES:
            write_lines(slice_dir / f"{name}.txt", report["case_lists"][name])


def fmt_rate(item: dict[str, Any] | None) -> str:
    if not item:
        return "n/a"
    value = item.get("value")
    numerator = item.get("numerator")
    denominator = item.get("denominator")
    if value is None:
        return f"n/a ({numerator}/{denominator})"
    return f"{value:.3f} ({numerator}/{denominator})"


def fmt_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "n/a"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    slices = report["slices"]
    lines = [
        "# VitalAgent Mechanism Analysis",
        "",
        "All cross-condition metrics are proxies from independent runs. Rescue and "
        "precision/recall metrics use evidence-backed correctness, not raw score.",
        "",
        "## Accuracy",
        "",
    ]
    lines.append(
        md_table(
            ["slice", "n", "full", "no_validation", "no_replan"],
            [
                [
                    name,
                    str(item["joined_question_count"]),
                    fmt_rate(item["accuracy"]["full"]),
                    fmt_rate(item["accuracy"]["no_validation"]),
                    fmt_rate(item["accuracy"]["no_replan"]),
                ]
                for name, item in slices.items()
            ],
        )
    )
    lines.extend(["", "## Layer 1 Detection", ""])
    lines.append(
        md_table(
            [
                "slice",
                "validation_fail_rate",
                "detection_precision",
                "detection_recall",
                "tool_failure precision",
                "non_tool precision",
            ],
            [
                [
                    name,
                    fmt_rate(item["layer1_detection"]["validation_fail_rate"]),
                    fmt_rate(item["layer1_detection"]["detection_precision"]),
                    fmt_rate(item["layer1_detection"]["detection_recall"]),
                    fmt_rate(
                        item["layer1_issue_segments"]["_tool_failure"][
                            "detection_precision"
                        ]
                    ),
                    fmt_rate(
                        item["layer1_issue_segments"]["_non_tool_failure"][
                            "detection_precision"
                        ]
                    ),
                ]
                for name, item in slices.items()
            ],
        )
    )
    lines.extend(["", "## Layer 2 Repair", ""])
    lines.append(
        md_table(
            [
                "slice",
                "replan_trigger",
                "plan_change",
                "evidence_repair",
                "coverage_improve",
                "rescue",
                "strict_rescue",
                "false_replan",
            ],
            [
                [
                    name,
                    fmt_rate(item["layer2_repair"]["replan_trigger_rate"]),
                    fmt_rate(item["layer2_repair"]["plan_change_rate"]),
                    fmt_rate(item["layer2_repair"]["evidence_repair_rate"]),
                    fmt_rate(item["layer2_repair"]["coverage_improvement_rate"]),
                    fmt_rate(item["layer2_repair"]["replan_rescue_rate"]),
                    fmt_rate(item["layer2_repair"]["replan_rescue_rate_strict"]),
                    fmt_rate(item["layer2_repair"]["false_replan_rate"]),
                ]
                for name, item in slices.items()
            ],
        )
    )
    lines.extend(["", "## Layer 2 Overhead", ""])
    lines.append(
        md_table(
            ["slice", "delta tokens", "delta LLM calls", "delta elapsed sec"],
            [
                [
                    name,
                    fmt_delta(
                        item["layer2_repair"]["overhead"][
                            "delta_avg_total_tokens_full_minus_no_validation"
                        ]
                    ),
                    fmt_delta(
                        item["layer2_repair"]["overhead"][
                            "delta_mean_llm_call_count_full_minus_no_validation"
                        ]
                    ),
                    fmt_delta(
                        item["layer2_repair"]["overhead"][
                            "delta_mean_elapsed_sec_full_minus_no_validation"
                        ]
                    ),
                ]
                for name, item in slices.items()
            ],
        )
    )
    lines.extend(["", "## Layer 3 Scoring Artifact", ""])
    lines.append(
        md_table(
            ["slice", "fallback_correct_rate", "tool_failure_rate"],
            [
                [
                    name,
                    fmt_rate(item["layer3_scoring_artifact"]["fallback_correct_rate"]),
                    fmt_rate(item["layer3_scoring_artifact"]["tool_failure_rate"]),
                ]
                for name, item in slices.items()
            ],
        )
    )
    lines.extend(["", "## Issue Histograms", ""])
    for name, item in slices.items():
        hist = item["issue_histograms"]
        lines.append(
            f"- `{name}` source={hist['source']} attempt0="
            f"`{json.dumps(hist['attempt0_full'], sort_keys=True)}` all="
            f"`{json.dumps(hist['all_full_validation'], sort_keys=True)}`"
        )
    lines.extend(["", "## Case Lists", ""])
    lines.append(
        "Top-level case lists are written for `ALL` when present; per-slice lists are "
        "under `case_ids/<slice>/`."
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze validation/replan mechanism metrics from VitalBench eval outputs.",
    )
    parser.add_argument(
        "--predictions",
        "--predictions-jsonl",
        dest="predictions_jsonl",
        type=Path,
        required=True,
        help="Path to predictions.jsonl from vitalbench_eval.",
    )
    parser.add_argument(
        "--traces",
        "--agent-traces-jsonl",
        dest="agent_traces_jsonl",
        type=Path,
        default=None,
        help="Optional path to agent_traces.jsonl for exact issue histograms.",
    )
    parser.add_argument(
        "--slices-dir",
        type=Path,
        default=None,
        help="Directory of *.txt question-id slice files. Defaults to ALL joined qids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for mechanism_report.json/md and case-id files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    analyze_mechanism(
        args.predictions_jsonl,
        args.output_dir,
        agent_traces_jsonl=args.agent_traces_jsonl,
        slices_dir=args.slices_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
