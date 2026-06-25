"""Signal-derived MonitoringState builder tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.mhealth.schemas import MonitoringState
from agent.mhealth.state_builders.ppg_state_builder import (
    PPGWindowConfig,
    build_monitoring_states_from_ppg_patient,
)
from agent.state.mhealth_state_store import get_global_state_store
from agent.tools.ecg import tools as ecg_tools
from agent.tools.registry import register_tool

_DATASET_NAMES = {
    "icentia11k",
    "afppgecg",
    "wesad",
    "ppg_dalia",
}

_ECG_SOURCE_SEMANTICS = {
    "signal_derived": (
        "HR/HRV fields are signal-derived from ECG R-peak detection."
    ),
    "not_inferred": (
        "rhythm_class and AF fields are left unset by this adapter; use "
        "ecg_diagnosis or get_ecg_description for rhythm-related PTB-XL queries."
    ),
    "ground_truth_boundary": (
        "Signal-derived HR/HRV state; rhythm and AF labels are not inferred by "
        "this adapter."
    ),
}

_PPG_SOURCE_SEMANTICS = {
    "signal_derived": (
        "HR/PRV/SQI fields are signal-derived from PPG peak detection."
    ),
    "reference_annotation": (
        "Rhythm/AF fields are derived from aligned ECG reference annotation, "
        "not from an independent PPG AF classifier."
    ),
    "coverage_rule": (
        "PPG windows require sufficient overlap with ECG reference annotation "
        "before rhythm/AF ground truth is emitted."
    ),
}


def _store_states(
    *,
    states: list[MonitoringState],
    clear_existing: bool,
) -> dict[str, Any]:
    store = get_global_state_store()
    if clear_existing:
        store.clear()
    store.add_states(states)
    return {
        "state_count": len(states),
        "contexts": store.list_contexts(),
    }


def _metadata_from_record(record: Any | None) -> dict[str, Any]:
    metadata = getattr(record, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _resolve_dataset(metadata: dict[str, Any]) -> str:
    for key in ("dataset", "dataset_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value in _DATASET_NAMES:
            return value

    source = metadata.get("source")
    if isinstance(source, str) and source in _DATASET_NAMES:
        return source

    return "icentia11k"


def _resolve_source_path(record: Any | None, metadata: dict[str, Any]) -> str | None:
    for key in ("source_path", "file_path", "path"):
        value = metadata.get(key)
        if value is not None:
            return str(value)

    file_name = metadata.get("file_name")
    if file_name is not None:
        return str(file_name)

    record_path = getattr(record, "path", None)
    return str(record_path) if record_path is not None else None


def _hrv_error(result: dict[str, Any]) -> str | None:
    if result.get("success") is True:
        return None
    error = result.get("error")
    return str(error) if error else "HRV analysis failed."


def _build_local_icentia11k_record(
    record_id: str,
    *,
    clear_existing: bool,
) -> dict[str, Any] | None:
    record_stem = Path(record_id)
    if record_stem.suffix:
        record_stem = record_stem.with_suffix("")
    name = record_stem.name
    if not (name.startswith("p") and "_s" in name):
        return None
    patient_text, segment_text = name.split("_s", 1)
    patient = patient_text.removeprefix("p")
    segment = segment_text
    if not (patient.isdigit() and len(patient) == 5 and segment.isdigit()):
        return None
    if not record_stem.with_suffix(".hea").exists():
        return None

    from agent.data.icentia11k_loader import Icentia11kLoader
    from agent.mhealth.state_builders.icentia11k_state_builder import (
        build_monitoring_states_from_icentia_raw,
    )

    local_root = record_stem.parents[2]
    raw = Icentia11kLoader(local_root=local_root).read_segment(patient, segment)
    states = build_monitoring_states_from_icentia_raw(raw)
    result = _store_states(states=states, clear_existing=clear_existing)
    return {
        "success": True,
        "dataset": "icentia11k",
        "record_id": str(record_stem),
        "patient_id": patient,
        "segment_id": segment,
        "source_path": str(record_stem),
        **result,
    }


@register_tool(
    name="state_build_from_ecg_record",
    description=(
        "Build one signal-derived ECG MonitoringState from an ECG raw signal record. "
        "Uses existing NeuroKit2 heart-rate and HRV analysis tools. rhythm_class "
        "and AF fields are left unset here; use ecg_diagnosis or "
        "get_ecg_description for rhythm-related PTB-XL queries."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to analyze.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Sampling rate for ECG loading and analysis.",
            "default": 100,
        },
        "lead": {
            "type": "string",
            "description": "Lead to analyze. Default 'II'.",
            "default": "II",
        },
        "clear_existing": {
            "type": "boolean",
            "description": "Clear already-loaded states before adding the generated state.",
            "default": False,
        },
    },
)
def state_build_from_ecg_record(
    record_id: str,
    sampling_rate: int = 100,
    lead: str = "II",
    clear_existing: bool = False,
) -> dict[str, Any]:
    try:
        icentia_result = _build_local_icentia11k_record(
            str(record_id),
            clear_existing=clear_existing,
        )
        if icentia_result is not None:
            return icentia_result

        hr_result = ecg_tools.analyze_heart_rate(
            record_id=record_id,
            sampling_rate=sampling_rate,
            lead=lead,
        )
        if hr_result.get("success") is not True:
            return {
                "success": False,
                "record_id": str(record_id),
                "error": str(hr_result.get("error") or "Heart-rate analysis failed."),
                "source_semantics": _ECG_SOURCE_SEMANTICS,
            }

        hr_metadata = dict(hr_result.get("metadata") or {})
        heart_rate = dict((hr_result.get("data") or {}).get("heart_rate") or {})
        metadata_notes: dict[str, Any] = {}

        try:
            record = ecg_tools._load_record(record_id, sampling_rate)
        except Exception as exc:
            record = None
            metadata_notes["record_metadata_error"] = str(exc)
        record_metadata = _metadata_from_record(record)

        hrv_result: dict[str, Any]
        try:
            hrv_result = ecg_tools.analyze_hrv(
                record_id=record_id,
                sampling_rate=sampling_rate,
                lead=lead,
            )
        except Exception as exc:
            hrv_result = {"success": False, "error": str(exc)}

        hrv_metadata_error = _hrv_error(hrv_result)
        time_domain = {}
        if hrv_result.get("success") is True:
            time_domain = dict((hrv_result.get("data") or {}).get("time_domain") or {})

        patient_id = str(record_metadata.get("patient_id") or record_id)
        duration_s = hr_metadata.get("duration_seconds")
        dataset = _resolve_dataset(record_metadata)
        source_path = _resolve_source_path(record, record_metadata)
        resolved_lead = str(hr_metadata.get("lead_analyzed") or lead)
        resolved_sampling_rate = int(hr_metadata.get("sampling_rate") or sampling_rate)

        metadata: dict[str, Any] = {
            "hr_source": "ecg_r_peak_detection",
            "signal_processor": "neurokit2",
            "lead_analyzed": resolved_lead,
            "sampling_rate": resolved_sampling_rate,
            "ground_truth_boundary": _ECG_SOURCE_SEMANTICS["ground_truth_boundary"],
            "source_semantics": _ECG_SOURCE_SEMANTICS,
            "record_id": str(record_id),
        }
        if record_metadata:
            metadata["record_metadata"] = record_metadata
        metadata.update(metadata_notes)
        if source_path is not None:
            metadata["source_path"] = source_path
        if record_metadata.get("source") is not None:
            metadata["record_source"] = record_metadata.get("source")
        if hrv_metadata_error is not None:
            metadata["hrv_error"] = hrv_metadata_error

        state = MonitoringState(
            state_id=f"ecg_signal_{record_id}",
            patient_id=patient_id,
            subject_id=patient_id,
            recording_id=str(record_id),
            dataset=dataset,
            modality="ecg",
            window_index=0,
            window_start_s=0.0,
            window_duration_s=float(duration_s) if duration_s is not None else None,
            hr_bpm=heart_rate.get("mean_bpm"),
            mean_hr_bpm=heart_rate.get("mean_bpm"),
            max_hr_bpm=heart_rate.get("max_bpm"),
            sdnn_ms=time_domain.get("sdnn_ms"),
            rmssd_ms=time_domain.get("rmssd_ms"),
            rhythm_class=None,
            af_burden_ratio=None,
            metadata=metadata,
        )
        result = _store_states(states=[state], clear_existing=clear_existing)
        return {
            "success": True,
            "dataset": dataset,
            "source_path": source_path,
            "record_id": str(record_id),
            "source_semantics": _ECG_SOURCE_SEMANTICS,
            **result,
        }
    except Exception as exc:
        return {
            "success": False,
            "record_id": str(record_id),
            "error": str(exc),
            "source_semantics": _ECG_SOURCE_SEMANTICS,
        }


@register_tool(
    name="state_build_from_ppg_patient",
    description=(
        "Build signal-derived PPG MonitoringState records from one AFPPGECG wrist PPG "
        "patient. PPG vitals come from peak detection; rhythm/AF fields come from "
        "aligned ECG reference annotation."
    ),
    parameters={
        "patient_id": {
            "type": "string",
            "description": "AFPPGECG patient ID, e.g. '001'.",
            "required": True,
        },
        "dataset_root": {
            "type": "string",
            "description": "Optional dataset root containing *_ECG.mat and *_PPG.mat files.",
            "required": False,
        },
        "clear_existing": {
            "type": "boolean",
            "description": "Clear already-loaded states before adding generated states.",
            "default": False,
        },
        "window_sec": {
            "type": "integer",
            "description": "PPG window duration in seconds.",
            "default": 300,
        },
        "max_windows": {
            "type": "integer",
            "description": "Optional maximum number of PPG windows to build.",
            "required": False,
        },
        "rhythm_threshold": {
            "type": "number",
            "description": (
                "AF burden threshold used to map ECG reference annotation to rhythm class."
            ),
            "default": 0.5,
        },
        "ambiguous_rhythm_margin": {
            "type": "number",
            "description": "Optional margin around the rhythm threshold where rhythm_class is left unset.",
            "required": False,
        },
        "min_annotation_coverage_ratio": {
            "type": "number",
            "description": (
                "Minimum fraction of a PPG window covered by ECG reference annotation."
            ),
            "default": 0.8,
        },
    },
)
def state_build_from_ppg_patient(
    patient_id: str,
    dataset_root: str | None = None,
    clear_existing: bool = False,
    window_sec: int = 300,
    max_windows: int | None = None,
    rhythm_threshold: float = 0.5,
    ambiguous_rhythm_margin: float | None = None,
    min_annotation_coverage_ratio: float = 0.8,
) -> dict[str, Any]:
    try:
        window_sec = int(window_sec)
        if window_sec <= 0:
            return {
                "success": False,
                "patient_id": str(patient_id),
                "error": "window_sec must be positive.",
            }

        rhythm_threshold = float(rhythm_threshold)
        if rhythm_threshold < 0 or rhythm_threshold > 1:
            return {
                "success": False,
                "patient_id": str(patient_id),
                "error": "rhythm_threshold must be between 0 and 1.",
            }

        min_annotation_coverage_ratio = float(min_annotation_coverage_ratio)
        if min_annotation_coverage_ratio < 0 or min_annotation_coverage_ratio > 1:
            return {
                "success": False,
                "patient_id": str(patient_id),
                "error": "min_annotation_coverage_ratio must be between 0 and 1.",
                "source_semantics": _PPG_SOURCE_SEMANTICS,
            }

        resolved_max_windows = None if max_windows is None else int(max_windows)
        resolved_margin = (
            None
            if ambiguous_rhythm_margin is None
            else float(ambiguous_rhythm_margin)
        )

        config = PPGWindowConfig(
            window_sec=window_sec,
            max_windows=resolved_max_windows,
            rhythm_threshold=rhythm_threshold,
            ambiguous_rhythm_margin=resolved_margin,
            min_annotation_coverage_ratio=min_annotation_coverage_ratio,
        )
        resolved_root = Path(dataset_root) if dataset_root is not None else None
        states = build_monitoring_states_from_ppg_patient(
            dataset_root=resolved_root,
            patient_id=patient_id,
            config=config,
        )
        if not states:
            return {
                "success": False,
                "dataset": "afppgecg",
                "patient_id": str(patient_id),
                "error": "No PPG MonitoringState records were generated for the requested patient.",
                "source_semantics": _PPG_SOURCE_SEMANTICS,
            }

        for state in states:
            state.metadata.setdefault("source_semantics", _PPG_SOURCE_SEMANTICS)

        result = _store_states(states=states, clear_existing=clear_existing)
        return {
            "success": True,
            "dataset": "afppgecg",
            "patient_id": str(patient_id),
            "source_semantics": _PPG_SOURCE_SEMANTICS,
            **result,
        }
    except Exception as exc:
        return {
            "success": False,
            "dataset": "afppgecg",
            "patient_id": str(patient_id),
            "error": str(exc),
            "source_semantics": _PPG_SOURCE_SEMANTICS,
        }


