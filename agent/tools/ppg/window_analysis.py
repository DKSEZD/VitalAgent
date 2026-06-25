"""Shared helpers for raw PPG/BVP window analysis."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from agent.encoder.ecg_signal_processor import ECGSignalProcessor
from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.tools.ecg.window_analysis import rhythm_screen_from_rr_intervals


def get_nested(payload: dict[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def as_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def slice_by_time(
    values: np.ndarray | None,
    *,
    fs_hz: float,
    start_s: float,
    end_s: float,
) -> np.ndarray | None:
    if values is None or values.size == 0:
        return None
    start = max(0, int(round(start_s * fs_hz)))
    end = min(int(round(end_s * fs_hz)), int(values.shape[0]))
    if end <= start:
        return values[:0]
    return values[start:end]


def finite_1d(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def summary_stats(values: np.ndarray | None) -> dict[str, Any] | None:
    finite = finite_1d(values)
    if finite.size == 0:
        return None
    return {
        "mean": round(float(np.mean(finite)), 4),
        "std": round(float(np.std(finite)), 4),
        "min": round(float(np.min(finite)), 4),
        "max": round(float(np.max(finite)), 4),
        "sample_count": int(finite.size),
    }


def trend_summary(
    values: np.ndarray | None,
    *,
    duration_s: float,
) -> dict[str, Any] | None:
    summary = summary_stats(values)
    if summary is None:
        return None
    finite = finite_1d(values)
    if finite.size >= 2 and duration_s > 0:
        summary["slope_per_min"] = round(
            float((finite[-1] - finite[0]) / duration_s * 60.0),
            4,
        )
    return summary


def motion_summary(acc_window: np.ndarray | None) -> dict[str, Any] | None:
    if acc_window is None or acc_window.size == 0:
        return None
    arr = np.asarray(acc_window, dtype=float)
    if arr.ndim == 1:
        magnitude = arr
    else:
        magnitude = np.linalg.norm(arr, axis=1)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return None

    std = float(np.std(finite))
    if std < 0.02:
        level = "low"
    elif std < 0.08:
        level = "medium"
    else:
        level = "high"

    return {
        "magnitude_mean": round(float(np.mean(finite)), 4),
        "magnitude_std": round(std, 4),
        "motion_level": level,
        "sample_count": int(finite.size),
    }


def quality_label(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.8:
        return "good"
    if score >= 0.5:
        return "acceptable"
    return "poor"


def bvp_pulse_summary(
    *,
    bvp_window: np.ndarray | None,
    acc_window: np.ndarray | None,
    bvp_fs_hz: float,
    processor_cls: type[PPGSignalProcessor] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    caveats: list[str] = []
    if bvp_window is None or bvp_window.size == 0:
        return None, ["No wrist BVP samples were available for this window."]

    processor_type = processor_cls or PPGSignalProcessor
    processor = processor_type(sampling_rate=bvp_fs_hz)
    vitals = processor.extract_vitals(
        np.asarray(bvp_window, dtype=float).reshape(-1),
        fs=bvp_fs_hz,
        accel=acc_window,
    )
    intervals = np.asarray(vitals.rr_intervals_ms or [], dtype=float)
    quality = vitals.signal_quality_score
    summary: dict[str, Any] = {
        "source": "wrist_bvp_peak_detection",
        "mean_pulse_rate_bpm": vitals.hr_bpm,
        "pp_interval_count": int(intervals.size),
        "sdnn_ms": vitals.sdnn_ms,
        "rmssd_ms": vitals.rmssd_ms,
        "signal_quality_score": quality,
        "quality_label": quality_label(float(quality) if quality is not None else None),
    }
    if intervals.size:
        instant = 60_000.0 / intervals
        summary["min_pulse_rate_bpm"] = round(float(np.min(instant)), 1)
        summary["max_pulse_rate_bpm"] = round(float(np.max(instant)), 1)
        summary["mean_pp_interval_ms"] = round(float(np.mean(intervals)), 1)
    else:
        caveats.append("Too few valid BVP peaks were detected for pulse-rate statistics.")
    if quality is not None and quality < 0.5:
        caveats.append("BVP signal quality is low; pulse metrics may be unreliable.")
    return summary, caveats


def chest_ecg_summary(
    ecg_window: np.ndarray | None,
    *,
    ecg_fs_hz: float,
    processor_cls: type[ECGSignalProcessor] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if ecg_window is None or ecg_window.size == 0:
        return None, None
    try:
        processor_type = processor_cls or ECGSignalProcessor
        processor = processor_type(sampling_rate=ecg_fs_hz)
        vitals = processor.extract_vitals(
            np.asarray(ecg_window, dtype=float).reshape(-1),
            fs=ecg_fs_hz,
        )
        intervals = np.asarray(vitals.rr_intervals_ms or [], dtype=float)
        hr_30s = _hr_30s_from_rr_intervals(
            intervals,
            duration_s=float(np.asarray(ecg_window).shape[0]) / float(ecg_fs_hz),
        )
        summary = {
            "source": "chest_ecg_r_peak_detection",
            "mean_bpm": vitals.hr_bpm,
            "rr_interval_count": int(intervals.size),
            "sdnn_ms": vitals.sdnn_ms,
            "rmssd_ms": vitals.rmssd_ms,
            "signal_quality_score": vitals.signal_quality_score,
        }
        if hr_30s:
            summary["max_30s_bpm"] = round(float(max(hr_30s)), 1)
            summary["min_30s_bpm"] = round(float(min(hr_30s)), 1)
        rhythm_block = rhythm_screen_from_rr_intervals(intervals)
        summary.update(
            {
                "irregularity_metrics": rhythm_block["irregularity_metrics"],
                "rhythm_screen": rhythm_block["rhythm_screen"],
                "af_screen_positive": rhythm_block["af_screen_positive"],
                "caveats": rhythm_block["caveats"],
            }
        )
        return summary, None
    except Exception as exc:
        return None, f"Chest ECG analysis failed: {exc}"


def _hr_30s_from_rr_intervals(
    rr_intervals_ms: np.ndarray,
    *,
    duration_s: float,
) -> list[float]:
    intervals = np.asarray(rr_intervals_ms, dtype=float).reshape(-1)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0 or duration_s < 30.0:
        return []

    beat_times = np.concatenate(([0.0], np.cumsum(intervals / 1000.0)))
    beat_times = beat_times[beat_times < duration_s]

    values: list[float] = []
    cursor = 0.0
    while cursor + 30.0 <= duration_s + 1e-9:
        end = cursor + 30.0
        beat_count = int(np.sum((beat_times >= cursor) & (beat_times < end)))
        if beat_count >= 2:
            values.append(float(beat_count / 30.0 * 60.0))
        cursor = end
    return values
