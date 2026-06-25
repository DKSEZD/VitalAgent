"""State-level tools for multimodal mHealth MonitoringState timelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.mhealth.schemas import MonitoringState
from agent.state.mhealth_state_store import get_global_state_store
from agent.tools.registry import register_tool

IDENTITY_FIELDS = {
    "state_id",
    "patient_id",
    "dataset",
    "modality",
    "subject_id",
    "recording_id",
    "window_index",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
}

DEFAULT_STATE_FIELDS = {
    "hr_bpm",
    "mean_hr_bpm",
    "rhythm_class",
    "af_burden_ratio",
    "stress_label",
    "protocol_label_id",
    "stress_burden_ratio",
    "activity_label",
    "previous_hr_bpm",
    "previous_rhythm_class",
    "previous_stress_label",
    "previous_protocol_label_id",
    "previous_activity_label",
}

TREND_TARGETS = {
    "heart_rate_summary",
    "af_burden",
    "rhythm_transition_count",
    "stress_burden",
    "stress_duration",
    "stress_transition_count",
    "activity_distribution",
}

TREND_TARGET_TO_REQUIRED_CAPABILITY = {
    "heart_rate_summary": "supports_hr",
    "af_burden": "supports_af",
    "rhythm_transition_count": "supports_rhythm",
    "stress_burden": "supports_stress",
    "stress_duration": "supports_stress",
    "stress_transition_count": "supports_stress",
    "activity_distribution": "supports_activity",
}


def _compact_state(
    state: MonitoringState,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    payload = state.to_dict()
    wanted = set(fields or DEFAULT_STATE_FIELDS) | IDENTITY_FIELDS
    compact = {
        key: payload[key]
        for key in payload
        if key in wanted and payload[key] is not None
    }
    compact.setdefault("subject_id", _effective_subject_id(state))
    compact.setdefault("recording_id", _effective_recording_id(state))
    compact.setdefault("window_end_s", _effective_window_end_s(state))
    metadata_summary = _metadata_summary(state)
    if metadata_summary:
        compact["metadata_summary"] = metadata_summary
    if fields and "metadata" in fields:
        compact["metadata"] = payload.get("metadata", {})
    if fields and "dataset_specific" in fields:
        compact["dataset_specific"] = payload.get("dataset_specific", {})
    return compact


def _metadata_summary(state: MonitoringState) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if state.metadata:
        summary["metadata_keys"] = sorted(state.metadata)
        summary["metadata_field_count"] = len(state.metadata)
    if state.dataset_specific:
        summary["dataset_specific_keys"] = sorted(state.dataset_specific)
        summary["dataset_specific_field_count"] = len(state.dataset_specific)
    return summary


def _metadata_value(state: MonitoringState, key: str) -> Any:
    value = state.metadata.get(key)
    if value is None:
        value = state.dataset_specific.get(key)
    return value


def _effective_subject_id(state: MonitoringState) -> str | None:
    return state.subject_id or _metadata_value(state, "subject_id") or state.patient_id


def _effective_recording_id(state: MonitoringState) -> str | None:
    return (
        state.recording_id
        or _metadata_value(state, "recording_id")
        or _metadata_value(state, "record_id")
        or _metadata_value(state, "segment_id")
        or _effective_subject_id(state)
    )


def _effective_window_end_s(state: MonitoringState) -> float | None:
    if state.window_end_s is not None:
        return float(state.window_end_s)
    value = _metadata_value(state, "window_end_s")
    if value is not None:
        return float(value)
    if state.window_start_s is not None and state.window_duration_s is not None:
        return float(state.window_start_s) + float(state.window_duration_s)
    return None


def _base_evidence(state: MonitoringState) -> dict[str, Any]:
    return {
        "dataset": state.dataset,
        "subject_id": _effective_subject_id(state),
        "recording_id": _effective_recording_id(state),
        # `record_id` is kept as a compatibility alias for ECG-oriented callers.
        "record_id": _effective_recording_id(state),
        "state_id": state.state_id,
        "window_start_s": state.window_start_s,
        "window_end_s": _effective_window_end_s(state),
    }


def _store_empty_error() -> dict[str, Any]:
    return {"success": False, "error": "No monitoring states are loaded."}


def _has_loaded_states() -> bool:
    return bool(get_global_state_store().list_contexts())


def _dataset_contexts(dataset: str) -> list[dict[str, Any]]:
    return [
        context
        for context in get_global_state_store().list_contexts()
        if context.get("dataset") == dataset
    ]


def _loaded_states_for_dataset(dataset: str) -> list[MonitoringState]:
    store = get_global_state_store()
    states: list[MonitoringState] = []
    for context in _dataset_contexts(dataset):
        start_s = context.get("start_s")
        end_s = context.get("end_s")
        if start_s is None or end_s is None:
            current = store.get_current_state(
                dataset=dataset,
                subject_id=context.get("subject_id"),
                recording_id=context.get("recording_id"),
            )
            if current is not None:
                states.append(current)
            continue
        states.extend(
            store.get_window(
                dataset=dataset,
                subject_id=context.get("subject_id"),
                recording_id=context.get("recording_id"),
                start_s=float(start_s),
                end_s=float(end_s),
            )
        )
    return states


def _capabilities_for_dataset(dataset: str) -> dict[str, Any] | None:
    capabilities = _dataset_capabilities(dataset)
    if capabilities is None:
        return None
    return _apply_loaded_capability_overrides(dataset, dict(capabilities))


def _is_unqualified_context(
    dataset: str | None,
    subject_id: str | None,
    recording_id: str | None,
) -> bool:
    return dataset is None and subject_id is None and recording_id is None


def _is_ambiguous_unqualified_context(
    dataset: str | None,
    subject_id: str | None,
    recording_id: str | None,
) -> bool:
    return (
        _is_unqualified_context(dataset, subject_id, recording_id)
        and len(get_global_state_store().list_contexts()) > 1
    )


def _resolve_subject_id(
    subject_id: str | None,
    patient_id: str | None,
) -> str | None:
    return subject_id or patient_id


def _resolve_timestamp_s(
    timestamp_s: float | None,
    window_start_s: float | None,
) -> float | None:
    if timestamp_s is not None:
        return float(timestamp_s)
    if window_start_s is not None:
        return float(window_start_s)
    return None


def _resolve_start_end_s(
    start_s: float | None,
    end_s: float | None,
    window_start_s: float | None,
    window_end_s: float | None,
) -> tuple[float | None, float | None]:
    resolved_start = start_s if start_s is not None else window_start_s
    resolved_end = end_s if end_s is not None else window_end_s
    return (
        None if resolved_start is None else float(resolved_start),
        None if resolved_end is None else float(resolved_end),
    )


def _summary_supports_answer(target: str, summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    if target == "heart_rate_summary":
        return (
            int(summary.get("count") or 0) > 0
            or int(summary.get("peak_count") or 0) > 0
        )
    if target == "af_burden":
        return (
            int(summary.get("count") or 0) > 0
            or int(summary.get("derived_state_count") or 0) > 0
        )
    if target == "stress_burden":
        ratio_summary = summary.get("stress_burden_ratio")
        ratio_count = 0
        if isinstance(ratio_summary, dict):
            ratio_count = int(ratio_summary.get("count") or 0)
        return int(summary.get("derived_state_count") or 0) > 0 or ratio_count > 0
    if target == "activity_distribution":
        counts = summary.get("counts")
        return isinstance(counts, dict) and bool(counts)
    return any(value is not None for value in summary.values())


@register_tool(
    name="state_load_monitoring_states",
    description="Load MonitoringState JSONL records into the global mHealth StateStore.",
    parameters={
        "state_path": {
            "type": "string",
            "description": "Path to a JSONL file containing MonitoringState dictionaries.",
            "required": True,
        },
        "clear_existing": {
            "type": "boolean",
            "description": "Clear already-loaded states before loading this file.",
            "default": False,
        },
    },
)
def state_load_monitoring_states(
    state_path: str,
    clear_existing: bool = False,
) -> dict[str, Any]:
    store = get_global_state_store()
    try:
        if clear_existing:
            store.clear()
        count = store.load_jsonl(Path(state_path))
        return {
            "success": True,
            "state_count": count,
            "contexts": store.list_contexts(),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@register_tool(
    name="state_list_contexts",
    description="List available dataset/subject/recording timelines in the StateStore.",
)
def state_list_contexts() -> dict[str, Any]:
    return {"success": True, "contexts": get_global_state_store().list_contexts()}


@register_tool(
    name="state_get_current_monitoring_state",
    description="Return the current or latest MonitoringState for a dataset/subject/recording.",
    parameters={
        "dataset": {"type": "string", "description": "Dataset name.", "required": False},
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator. Alias for subject_id.",
            "required": False,
        },
        "subject_id": {"type": "string", "description": "Subject ID.", "required": False},
        "recording_id": {"type": "string", "description": "Recording ID.", "required": False},
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds. Alias for timestamp_s.",
            "required": False,
        },
        "timestamp_s": {
            "type": "number",
            "description": "Timestamp in seconds. Returns containing state or nearest previous state.",
            "required": False,
        },
        "fields": {
            "type": "array",
            "description": "Optional MonitoringState fields to include in addition to identity/window fields.",
            "required": False,
        },
    },
)
def state_get_current_monitoring_state(
    dataset: str | None = None,
    patient_id: str | None = None,
    subject_id: str | None = None,
    recording_id: str | None = None,
    window_start_s: float | None = None,
    timestamp_s: float | None = None,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    subject_id = _resolve_subject_id(subject_id, patient_id)
    timestamp_s = _resolve_timestamp_s(timestamp_s, window_start_s)
    if not _has_loaded_states():
        return _store_empty_error()
    if _is_ambiguous_unqualified_context(dataset, subject_id, recording_id):
        return {
            "success": False,
            "error": (
                "Ambiguous monitoring context. Provide dataset, subject_id, "
                "or recording_id."
            ),
        }
    state = get_global_state_store().get_current_state(
        dataset=dataset,
        subject_id=subject_id,
        recording_id=recording_id,
        timestamp_s=timestamp_s,
    )
    if state is None:
        return {"success": False, "error": "No matching monitoring state found."}
    return {
        "success": True,
        "state": _compact_state(state, fields),
        "evidence": _base_evidence(state),
    }


@register_tool(
    name="state_get_monitoring_window",
    description="Return MonitoringState records overlapping a time range.",
    parameters={
        "dataset": {"type": "string", "description": "Dataset name.", "required": False},
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator. Alias for subject_id.",
            "required": False,
        },
        "subject_id": {"type": "string", "description": "Subject ID.", "required": False},
        "recording_id": {"type": "string", "description": "Recording ID.", "required": False},
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds. Alias for start_s.",
            "required": False,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds. Alias for end_s.",
            "required": False,
        },
        "start_s": {"type": "number", "description": "Start time in seconds.", "required": False},
        "end_s": {"type": "number", "description": "End time in seconds.", "required": False},
        "fields": {"type": "array", "description": "Optional fields to include.", "required": False},
        "limit": {"type": "integer", "description": "Maximum states to return.", "default": 200},
    },
)
def state_get_monitoring_window(
    start_s: float | None = None,
    end_s: float | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
    subject_id: str | None = None,
    recording_id: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    fields: list[str] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    subject_id = _resolve_subject_id(subject_id, patient_id)
    start_s, end_s = _resolve_start_end_s(start_s, end_s, window_start_s, window_end_s)
    if not _has_loaded_states():
        return _store_empty_error()
    if start_s is None or end_s is None:
        return {"success": False, "error": "window_start_s/window_end_s or start_s/end_s are required."}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"success": False, "error": "limit must be an integer."}
    if limit < 1:
        return {"success": False, "error": "limit must be at least 1."}
    states = get_global_state_store().get_window(
        dataset=dataset,
        subject_id=subject_id,
        recording_id=recording_id,
        start_s=float(start_s),
        end_s=float(end_s),
    )
    if not states:
        return {"success": False, "error": "No matching monitoring states found."}
    returned = states[:limit]
    return {
        "success": True,
        "state_count": len(states),
        "returned_count": len(returned),
        "states": [_compact_state(state, fields) for state in returned],
        "truncated": len(states) > limit,
        "window": {"start_s": float(start_s), "end_s": float(end_s)},
    }


@register_tool(
    name="state_get_previous_monitoring_state",
    description="Return a previous state from the same dataset/subject/recording timeline.",
    parameters={
        "state_id": {"type": "string", "description": "Current state ID.", "required": True},
        "offset": {"type": "integer", "description": "How many states back to return.", "default": 1},
        "fields": {"type": "array", "description": "Optional fields to include.", "required": False},
    },
)
def state_get_previous_monitoring_state(
    state_id: str,
    offset: int = 1,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    if not _has_loaded_states():
        return _store_empty_error()
    state = get_global_state_store().get_previous_state(state_id, offset=offset)
    if state is None:
        return {"success": False, "error": "No previous state found."}
    return {"success": True, "state": _compact_state(state, fields)}


@register_tool(
    name="state_get_longitudinal_trend",
    description=(
        "Compute an aggregate summary from MonitoringState fields over a time window. "
        "Use target='heart_rate_summary' for mean HR, highest HR, and HR excursion "
        "threshold tasks. For highest/peak HR questions use "
        "summary.max_observed_hr_bpm, not summary.max; for threshold questions "
        "such as whether HR exceeded X bpm, compare X against "
        "summary.max_observed_hr_bpm."
    ),
    parameters={
        "dataset": {"type": "string", "description": "Dataset name.", "required": False},
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator. Alias for subject_id.",
            "required": False,
        },
        "subject_id": {"type": "string", "description": "Subject ID.", "required": False},
        "recording_id": {"type": "string", "description": "Recording ID.", "required": False},
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds. Alias for start_s.",
            "required": False,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds. Alias for end_s.",
            "required": False,
        },
        "start_s": {"type": "number", "description": "Start time in seconds.", "required": False},
        "end_s": {"type": "number", "description": "End time in seconds.", "required": False},
        "target": {
            "type": "string",
            "description": (
                "Aggregate target such as heart_rate_summary, stress_burden, or "
                "activity_distribution. Use heart_rate_summary for mean HR, highest/peak HR, "
                "and HR excursion threshold tasks; summary.max keeps the legacy "
                "mean/current-HR maximum, while summary.max_observed_hr_bpm is the "
                "monitoring-window peak to use for highest HR and threshold "
                "exceedance questions."
            ),
            "required": True,
        },
    },
)
def state_get_longitudinal_trend(
    start_s: float | None = None,
    end_s: float | None = None,
    target: str = "",
    dataset: str | None = None,
    patient_id: str | None = None,
    subject_id: str | None = None,
    recording_id: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
) -> dict[str, Any]:
    subject_id = _resolve_subject_id(subject_id, patient_id)
    start_s, end_s = _resolve_start_end_s(start_s, end_s, window_start_s, window_end_s)
    if not _has_loaded_states():
        return _store_empty_error()
    if start_s is None or end_s is None:
        return {"success": False, "error": "window_start_s/window_end_s or start_s/end_s are required."}
    if target not in TREND_TARGETS:
        return {"success": False, "error": f"Unsupported trend target: {target}"}

    states = get_global_state_store().get_window(
        dataset=dataset,
        subject_id=subject_id,
        recording_id=recording_id,
        start_s=float(start_s),
        end_s=float(end_s),
    )
    if not states:
        return {"success": False, "error": "No matching monitoring states found."}
    evidence_dataset = dataset or states[0].dataset
    evidence_subject_id = subject_id or _effective_subject_id(states[0])
    evidence_recording_id = recording_id or _effective_recording_id(states[0])
    capabilities = _capabilities_for_dataset(evidence_dataset)
    required_capability = TREND_TARGET_TO_REQUIRED_CAPABILITY.get(target)
    if (
        capabilities is not None
        and required_capability is not None
        and not capabilities.get(required_capability, False)
    ):
        return {
            "success": False,
            "error": (
                f"Dataset '{evidence_dataset}' does not support target "
                f"'{target}'."
            ),
            "unsupported_target": target,
            "hint": "Call state_get_dataset_capabilities to see supported targets.",
        }
    summary = get_global_state_store().summarize_trend(
        dataset=dataset,
        subject_id=subject_id,
        recording_id=recording_id,
        start_s=float(start_s),
        end_s=float(end_s),
        target=target,
    )
    if not _summary_supports_answer(target, summary):
        return {
            "success": False,
            "error": (
                f"No usable state fields were available for target '{target}' "
                "in the requested window."
            ),
            "target": target,
            "state_count": len(states),
            "hint": "Call state_get_dataset_capabilities to confirm target support.",
        }
    return {
        "success": True,
        "target": target,
        "summary": summary,
        "state_count": len(states),
        "window": {"start_s": float(start_s), "end_s": float(end_s)},
        "evidence": {
            "dataset": evidence_dataset,
            "subject_id": evidence_subject_id,
            "recording_id": evidence_recording_id,
            "used_state_count": len(states),
        },
    }


@register_tool(
    name="state_get_evidence",
    description="Return minimal state-grounded evidence fields for a MonitoringState target.",
    parameters={
        "state_id": {"type": "string", "description": "MonitoringState ID.", "required": True},
        "target": {
            "type": "string",
            "description": "Evidence target: heart_rate, rhythm, af, stress, activity.",
            "required": True,
        },
    },
)
def state_get_evidence(state_id: str, target: str) -> dict[str, Any]:
    if not _has_loaded_states():
        return _store_empty_error()
    state = get_global_state_store().get_state_by_id(state_id)
    if state is None:
        return {"success": False, "error": "No matching monitoring state found."}
    try:
        evidence = _target_evidence(state, target)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "evidence": evidence}


@register_tool(
    name="state_get_dataset_capabilities",
    description="Return dataset-level target support for mHealth state tools.",
    parameters={
        "dataset": {"type": "string", "description": "Dataset name.", "required": True},
    },
)
def state_get_dataset_capabilities(dataset: str) -> dict[str, Any]:
    capabilities = _capabilities_for_dataset(dataset)
    if capabilities is None:
        return {
            "success": True,
            "dataset": dataset,
            "capabilities": {},
            "recommended_tools": ["state_list_contexts"],
            "unsupported_targets": [
                "af",
                "rhythm",
                "stress",
                "activity",
            ],
            "hint": "Unknown dataset. Capabilities default to empty.",
        }
    unsupported_targets = [
        target
        for target, key in {
            "af": "supports_af",
            "rhythm": "supports_rhythm",
            "stress": "supports_stress",
            "activity": "supports_activity",
        }.items()
        if not capabilities.get(key, False)
    ]
    return {
        "success": True,
        "dataset": dataset,
        "capabilities": capabilities,
        "recommended_tools": _recommended_tools_for_capabilities(capabilities),
        "unsupported_targets": unsupported_targets,
    }


def _target_evidence(state: MonitoringState, target: str) -> dict[str, Any]:
    evidence = _base_evidence(state)

    if target == "heart_rate":
        evidence.update(
            {
                "hr_bpm": state.hr_bpm,
                "mean_hr_bpm": state.mean_hr_bpm,
                "max_hr_bpm": state.max_hr_bpm,
                "sdnn_ms": state.sdnn_ms,
                "rmssd_ms": state.rmssd_ms,
                "signal_quality_score": state.signal_quality_score,
            }
        )
        return evidence

    if target == "rhythm":
        evidence.update({"rhythm_class": state.rhythm_class})
        if state.dataset == "afppgecg" or state.modality == "ppg":
            evidence["methodological_caveat"] = (
                "PPG rhythm/AF evidence is indirect and cannot observe P-waves."
            )
        return evidence

    if target == "af":
        evidence.update(
            {
                "rhythm_class": state.rhythm_class,
                "af_burden_ratio": state.af_burden_ratio,
            }
        )
        if state.dataset == "afppgecg" or state.modality == "ppg":
            evidence["methodological_caveat"] = (
                "PPG rhythm/AF evidence is indirect and cannot observe P-waves."
            )
        return evidence

    if target == "stress":
        evidence.update(
            {
                "stress_label": state.stress_label,
                "protocol_label_id": state.protocol_label_id,
                "ground_truth_semantics": "WESAD protocol condition",
            }
        )
        return evidence

    if target == "activity":
        evidence.update({"activity_label": state.activity_label})
        return evidence

    raise ValueError(f"Unsupported evidence target: {target}")


def _dataset_capabilities(dataset: str) -> dict[str, Any] | None:
    table: dict[str, dict[str, Any]] = {
        "icentia11k": {
            "supports_ecg": True,
            "supports_ppg": False,
            "supports_hr": True,
            "supports_rhythm": True,
            "supports_af": True,
            "supports_stress": False,
            "supports_activity": False,
        },
        "afppgecg": {
            "supports_ecg": False,
            "supports_ppg": True,
            "supports_hr": True,
            "supports_rhythm": True,
            "supports_af": True,
            "supports_stress": False,
            "supports_activity": False,
            "caveats": {
                "af": "PPG rhythm/AF evidence is indirect and cannot observe P-waves.",
                "rhythm": "PPG rhythm/AF evidence is indirect and cannot observe P-waves.",
            },
        },
        "wesad": {
            "supports_ecg": False,
            "supports_ppg": False,
            "supports_hr": True,
            "supports_rhythm": False,
            "supports_af": False,
            "supports_stress": True,
            "supports_activity": False,
        },
        "ppg_dalia": {
            "supports_ecg": False,
            "supports_ppg": True,
            "supports_hr": True,
            "supports_rhythm": False,
            "supports_af": False,
            "supports_stress": False,
            "supports_activity": True,
        },
    }
    return table.get(dataset)


def _recommended_tools_for_capabilities(capabilities: dict[str, Any]) -> list[str]:
    tools = ["state_get_current_monitoring_state"]
    if capabilities.get("supports_hr"):
        tools.append("state_get_longitudinal_trend(target='heart_rate_summary')")
        tools.append("state_get_evidence(target='heart_rate')")
    if capabilities.get("supports_af"):
        tools.append("state_get_longitudinal_trend(target='af_burden')")
        tools.append("state_get_evidence(target='af')")
    if capabilities.get("supports_rhythm"):
        tools.append("state_get_longitudinal_trend(target='rhythm_transition_count')")
        tools.append("state_get_evidence(target='rhythm')")
    if capabilities.get("supports_stress"):
        tools.append("state_get_longitudinal_trend(target='stress_burden')")
        tools.append("state_get_longitudinal_trend(target='stress_duration')")
        tools.append("state_get_evidence(target='stress')")
    if capabilities.get("supports_activity"):
        tools.append("state_get_longitudinal_trend(target='activity_distribution')")
        tools.append("state_get_evidence(target='activity')")
    return tools


def _apply_loaded_capability_overrides(
    dataset: str,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    states = _loaded_states_for_dataset(dataset)
    if not states:
        return capabilities

    capabilities["supports_ecg"] = any(state.modality == "ecg" for state in states)
    capabilities["supports_ppg"] = any(state.modality == "ppg" for state in states)
    capabilities["supports_hr"] = any(
        state.hr_bpm is not None
        or state.mean_hr_bpm is not None
        or state.max_hr_bpm is not None
        for state in states
    )
    capabilities["supports_rhythm"] = any(
        state.rhythm_class is not None
        or state.previous_rhythm_class is not None
        or state.rhythm_transition_count is not None
        or state.rhythm_transition_count_per_hour is not None
        for state in states
    )
    capabilities["supports_af"] = any(
        state.rhythm_class is not None
        or state.af_burden_ratio is not None
        or state.af_episode_duration_s is not None
        for state in states
    )
    capabilities["supports_stress"] = any(
        state.stress_label is not None
        or state.protocol_label_id is not None
        or state.stress_burden_ratio is not None
        or state.stress_duration_s is not None
        or state.dominant_stress_label is not None
        for state in states
    )
    capabilities["supports_activity"] = any(
        state.activity_label is not None for state in states
    )
    return capabilities

