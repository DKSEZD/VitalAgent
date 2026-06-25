"""Raw Icentia11k ECG tools for mHealth-QA eval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import get_config
from agent.data.icentia11k_loader import Icentia11kLoader
from agent.mhealth.state_builders.icentia11k_state_builder import compute_max_hr_bpm
from agent.proactive.rhythm_classifier import classify_rhythm
from agent.tools.ecg.window_analysis import analyze_single_lead_ecg_window
from agent.tools.ecg.utils import process_ecg
from agent.tools.registry import register_tool


def _loader(local_root: str | None = None) -> Icentia11kLoader:
    raw_root = local_root or os.getenv("ICENTIA11K_DATASET_ROOT")
    root = Path(raw_root).expanduser() if raw_root else get_config().datasets.icentia11k_root
    if root.exists():
        return Icentia11kLoader(local_root=root)
    return Icentia11kLoader()


def _normalize_segment_id(segment_id: str | int | None, recording_id: str | int | None) -> str:
    candidate = segment_id if segment_id is not None else recording_id
    if candidate is None:
        candidate = "00"
    return Icentia11kLoader.normalize_segment_id(candidate)


@register_tool(
    name="analyze_icentia11k_ecg_window_signal",
    description=(
        "Analyze a raw single-lead Icentia11k ECG window by patient/segment/time. "
        "Returns heart-rate and RR-irregularity metrics. Mean heart rate uses "
        "Icentia11k beat annotations when available for benchmark-compatible HR; "
        "raw R-peak signal estimates are retained separately."
    ),
    parameters={
        "dataset": {
            "type": "string",
            "description": "Dataset name from the mHealth-QA locator. Expected 'icentia11k'.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Icentia11k patient ID from the mHealth-QA locator.",
            "required": True,
        },
        "recording_id": {
            "type": "string",
            "description": "Icentia11k segment ID from the raw-data locator, such as '17' or 's17'.",
            "required": False,
        },
        "segment_id": {
            "type": "string",
            "description": "Alias for recording_id.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start in seconds within the segment.",
            "required": True,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end in seconds within the segment.",
            "required": True,
        },
        "local_root": {
            "type": "string",
            "description": "Optional local Icentia11k dataset root.",
            "required": False,
        },
    },
)
def analyze_icentia11k_ecg_window_signal(
    patient_id: str,
    window_start_s: float,
    window_end_s: float,
    dataset: str | None = None,
    recording_id: str | None = None,
    segment_id: str | None = None,
    local_root: str | None = None,
) -> dict[str, Any]:
    if dataset is not None and str(dataset) != "icentia11k":
        return {"success": False, "error": f"Unsupported dataset for Icentia11k ECG tool: {dataset!r}"}

    try:
        segment = _normalize_segment_id(segment_id, recording_id)
        loader = _loader(local_root)
        header = loader.read_segment_header(patient_id, segment)
        start_sample = int(round(float(window_start_s) * header.fs))
        end_sample = int(round(float(window_end_s) * header.fs))
        if end_sample <= start_sample:
            return {
                "success": False,
                "error": f"Invalid ECG window: start={window_start_s}, end={window_end_s}.",
            }
        raw = loader.read_segment(patient_id, segment, sampfrom=start_sample, sampto=end_sample)
        context_raw = _read_monitoring_context_segment(
            loader=loader,
            patient_id=patient_id,
            segment=segment,
            header=header,
            end_sample=end_sample,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    analysis = analyze_single_lead_ecg_window(
        np.asarray(raw.signal, dtype=float),
        raw.fs,
        process_func=process_ecg,
        rhythm_classifier=classify_rhythm,
    )
    if not analysis.get("success"):
        metadata: dict[str, Any] = {
            "dataset": "icentia11k",
            "patient_id": raw.patient_id,
            "segment_id": raw.segment_id,
            "sampling_rate": raw.fs,
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
        }
        if analysis.get("signal_quality") is not None:
            metadata["signal_quality"] = analysis["signal_quality"]
        return {
            "success": False,
            "error": str(analysis.get("error")),
            "metadata": metadata,
        }

    data = dict(analysis["data"])
    annotation_hr_bpm = _annotation_hr_bpm(raw)
    annotation_hr_30s = _annotation_hr_30s_bpm(raw)
    context_hr_30s = _annotation_monitoring_context_hr_30s_bpm(context_raw)
    if annotation_hr_bpm is not None:
        signal_estimated_heart_rate = dict(data.get("heart_rate") or {})
        heart_rate = dict(signal_estimated_heart_rate)
        heart_rate["mean_bpm"] = annotation_hr_bpm
        heart_rate["mean_bpm_source"] = "icentia11k_beat_annotations"
        if annotation_hr_30s is not None:
            heart_rate.update(annotation_hr_30s)
            heart_rate["max_30s_bpm_source"] = "icentia11k_beat_annotations"
        if context_hr_30s is not None:
            heart_rate.update(context_hr_30s)
            heart_rate["monitoring_context_max_30s_bpm_source"] = (
                "icentia11k_beat_annotations_monitoring_context"
            )
        data["heart_rate"] = heart_rate
        data["signal_estimated_heart_rate"] = signal_estimated_heart_rate
    data["caveats"] = [
        (
            "Mean heart rate uses Icentia11k beat annotations when available; "
            "signal_estimated_heart_rate preserves the raw R-peak detector estimate."
        )
        if annotation_hr_bpm is not None
        else "This is raw single-lead ECG signal analysis, not a direct reading of Icentia11k annotations.",
        *list(data.get("caveats") or []),
    ]

    return {
        "success": True,
        "data": data,
        "metadata": {
            "dataset": "icentia11k",
            "patient_id": raw.patient_id,
            "segment_id": raw.segment_id,
            "sampling_rate": raw.fs,
            "sample_start": raw.sample_start,
            "sample_end": raw.sample_end,
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
            "duration_seconds": round(raw.window_duration_seconds, 1),
            "source": raw.metadata.get("source"),
            "lead_names": raw.metadata.get("lead_names"),
            "beat_annotation_hr_available": annotation_hr_bpm is not None,
        },
    }


def _annotation_hr_bpm(raw: Any) -> float | None:
    beat_annotations = getattr(raw, "beat_annotations", None)
    samples = getattr(beat_annotations, "samples", None)
    duration_seconds = float(getattr(raw, "window_duration_seconds", 0.0) or 0.0)
    if samples is None or duration_seconds <= 0:
        return None
    beat_count = int(np.asarray(samples).size)
    if beat_count <= 0:
        return None
    return round(beat_count / (duration_seconds / 60.0), 1)


def _read_monitoring_context_segment(
    *,
    loader: Icentia11kLoader,
    patient_id: str,
    segment: str,
    header: Any,
    end_sample: int,
) -> Any | None:
    fs = float(getattr(header, "fs", 0.0) or 0.0)
    total_samples = getattr(header, "n_samples", None)
    if fs <= 0 or total_samples is None:
        return None
    context_samples = int(round(1800.0 * fs))
    context_end = min(int(total_samples), max(int(end_sample), context_samples))
    if context_end <= 0:
        return None
    try:
        return loader.read_segment(patient_id, segment, sampfrom=0, sampto=context_end)
    except Exception:
        return None


def _annotation_hr_30s_bpm(raw: Any) -> dict[str, float] | None:
    beat_annotations = getattr(raw, "beat_annotations", None)
    samples = getattr(beat_annotations, "samples", None)
    fs = float(getattr(raw, "fs", 0.0) or 0.0)
    n_samples = int(getattr(raw, "n_samples", 0) or 0)
    if samples is None or fs <= 0 or n_samples <= 0:
        return None

    beat_samples = np.asarray(samples, dtype=int).reshape(-1)
    if beat_samples.size < 2:
        return None

    max_hr = compute_max_hr_bpm(
        beat_samples,
        0,
        n_samples,
        sampling_rate_hz=int(round(fs)),
        subwindow_sec=30,
    )
    min_hr = _annotation_min_hr_30s_bpm(beat_samples, fs_hz=fs, n_samples=n_samples)
    if max_hr is None or min_hr is None:
        return None
    return {
        "max_30s_bpm": round(float(max_hr), 1),
        "min_30s_bpm": round(float(min_hr), 1),
    }


def _annotation_monitoring_context_hr_30s_bpm(raw: Any | None) -> dict[str, float] | None:
    if raw is None:
        return None
    beat_annotations = getattr(raw, "beat_annotations", None)
    samples = getattr(beat_annotations, "samples", None)
    fs = float(getattr(raw, "fs", 0.0) or 0.0)
    n_samples = int(getattr(raw, "n_samples", 0) or 0)
    if samples is None or fs <= 0 or n_samples <= 0:
        return None
    beat_samples = np.asarray(samples, dtype=int).reshape(-1)
    max_hr = compute_max_hr_bpm(
        beat_samples,
        0,
        n_samples,
        sampling_rate_hz=int(round(fs)),
        subwindow_sec=30,
    )
    if max_hr is None:
        return None
    return {"monitoring_context_max_30s_bpm": round(float(max_hr), 1)}


def _annotation_min_hr_30s_bpm(
    beat_samples: np.ndarray,
    *,
    fs_hz: float,
    n_samples: int,
) -> float | None:
    subwindow_samples = int(round(30.0 * fs_hz))
    if subwindow_samples <= 0:
        return None

    values: list[float] = []
    cursor = 0
    while cursor + subwindow_samples <= n_samples:
        sub_end = cursor + subwindow_samples
        beat_count = int(np.sum((beat_samples >= cursor) & (beat_samples < sub_end)))
        if beat_count > 0:
            values.append(float(beat_count / (subwindow_samples / fs_hz) * 60.0))
        cursor = sub_end

    if not values:
        return None
    return float(min(values))


__all__ = ["analyze_icentia11k_ecg_window_signal"]
