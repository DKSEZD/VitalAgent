"""Shared helpers for raw ECG window analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from agent.proactive.rhythm_classifier import classify_rhythm
from agent.tools.ecg.utils import process_ecg

ECGProcessFunc = Callable[[np.ndarray, float], tuple[dict[str, Any], dict[str, Any]]]
RhythmClassifier = Callable[[list[float]], str]


def round_or_none(value: float | None, digits: int = 1) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def signal_quality_summary(signal: np.ndarray, r_peaks: np.ndarray) -> dict[str, Any]:
    finite_fraction = float(np.mean(np.isfinite(signal))) if signal.size else 0.0
    finite_signal = signal[np.isfinite(signal)]
    amplitude_range = float(np.ptp(finite_signal)) if finite_signal.size else 0.0
    if finite_signal.size == 0:
        label = "unusable"
    elif finite_fraction < 0.98 or amplitude_range <= 1e-6:
        label = "poor"
    elif len(r_peaks) < 3:
        label = "limited"
    else:
        label = "usable"
    return {
        "quality_label": label,
        "finite_fraction": round(finite_fraction, 3),
        "amplitude_range": round(amplitude_range, 6),
        "r_peak_count": int(len(r_peaks)),
    }


def rhythm_screen_from_rr_intervals(
    rr_intervals_ms: list[float] | np.ndarray | None,
    *,
    rhythm_classifier: RhythmClassifier | None = None,
) -> dict[str, Any]:
    """Return the shared ECG-style RR irregularity and AF-screening block."""

    classify = rhythm_classifier or classify_rhythm
    rr = np.asarray([] if rr_intervals_ms is None else rr_intervals_ms, dtype=float)
    rr_clean = rr[(rr >= 200.0) & (rr <= 2500.0)]
    diffs = np.diff(rr_clean)
    rr_cv = (
        float(np.std(rr_clean, ddof=1) / np.mean(rr_clean))
        if len(rr_clean) > 1
        else 0.0
    )
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else 0.0
    pnn50 = (
        float(np.sum(np.abs(diffs) > 50.0) / len(diffs) * 100.0)
        if len(diffs) > 0
        else 0.0
    )
    rhythm_screen = classify(rr_clean.tolist())

    caveats = [
        "AF screening from RR irregularity should be treated as supportive evidence; definitive AF diagnosis requires morphology/P-wave review or a validated classifier.",
    ]
    if rhythm_screen == "Other":
        caveats.append(
            "RR metrics are irregular or ambiguous but do not clearly satisfy the AF screening heuristic."
        )
    elif rhythm_screen == "Unknown":
        caveats.append("Too few clean RR intervals were available for confident rhythm screening.")

    return {
        "irregularity_metrics": {
            "coefficient_of_variation": round(rr_cv, 3),
            "rmssd_ms": round(rmssd, 1),
            "pnn50_percent": round(pnn50, 1),
        },
        "rhythm_screen": rhythm_screen,
        "af_screen_positive": rhythm_screen == "AF",
        "caveats": caveats,
    }


def analyze_single_lead_ecg_window(
    signal: np.ndarray,
    fs_hz: float,
    *,
    process_func: ECGProcessFunc | None = None,
    rhythm_classifier: RhythmClassifier | None = None,
) -> dict[str, Any]:
    """Analyze a raw single-lead ECG window without dataset-specific metadata.

    On success, signal quality is nested under ``data["signal_quality"]``. When
    processing succeeds but too few R-peaks are detected, signal quality is
    returned at the top level so callers can include it in failure metadata.
    """
    process = process_func or process_ecg
    classify = rhythm_classifier or classify_rhythm
    signal = np.asarray(signal, dtype=float)
    try:
        signals, info = process(signal, fs_hz)
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception as exc:
        return {"success": False, "error": f"ECG processing failed: {exc}"}

    if len(r_peaks) < 2:
        return {
            "success": False,
            "error": "Too few R-peaks detected for ECG window analysis.",
            "signal_quality": signal_quality_summary(signal, r_peaks),
        }

    rate_series = signals["ECG_Rate"].dropna()
    rate = np.asarray(rate_series, dtype=float)
    rr_intervals_ms = np.diff(r_peaks) / float(fs_hz) * 1000.0
    rr_clean = rr_intervals_ms[(rr_intervals_ms >= 200.0) & (rr_intervals_ms <= 2500.0)]
    rhythm_block = rhythm_screen_from_rr_intervals(
        rr_clean,
        rhythm_classifier=classify,
    )

    return {
        "success": True,
        "data": {
            "heart_rate": {
                "mean_bpm": round_or_none(float(np.mean(rate)) if rate.size else None),
                "min_bpm": round_or_none(float(np.min(rate)) if rate.size else None),
                "max_bpm": round_or_none(float(np.max(rate)) if rate.size else None),
                "std_bpm": round_or_none(
                    float(np.std(rate, ddof=1)) if rate.size > 1 else 0.0
                ),
            },
            "rr_intervals": {
                "count": int(len(rr_clean)),
                "mean_ms": round_or_none(float(np.mean(rr_clean)) if len(rr_clean) else None),
                "std_ms": round_or_none(
                    float(np.std(rr_clean, ddof=1)) if len(rr_clean) > 1 else 0.0
                ),
            },
            "irregularity_metrics": rhythm_block["irregularity_metrics"],
            "rhythm_screen": rhythm_block["rhythm_screen"],
            "af_screen_positive": rhythm_block["af_screen_positive"],
            "signal_quality": signal_quality_summary(signal, r_peaks),
            "caveats": rhythm_block["caveats"],
        },
    }
