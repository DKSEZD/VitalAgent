"""
ECG utility functions shared across tools.

Organized by function group:
  - General ECG processing
  - Heart rate / HRV helpers
  - Per-lead morphology analysis
  - Signal quality assessment
  - Constants
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import neurokit2 as nk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# Noise types tracked in PTB-XL metadata
NOISE_TYPES = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]

# ST-segment offset thresholds (mV)
ST_ELEVATION_THRESH = 0.1
# Use a slightly more sensitive cutoff for nonspecific ST depression screening.
ST_DEPRESSION_THRESH = -0.05

# T-wave amplitude thresholds (mV)
T_FLAT_THRESH = 0.05

# Morphology timing windows
J_POINT_AFTER_S_MS = 20.0
J_POINT_FALLBACK_AFTER_R_MS = 40.0
ST_MEASURE_CENTER_MS = 60.0
ST_MEASURE_HALF_WINDOW_MS = 10.0
BASELINE_FALLBACK_WINDOW_MS = 20.0
TP_SEGMENT_MIN_MS = 20.0
TP_SEGMENT_EDGE_MARGIN_MS = 10.0

# Q-wave pathological criteria
Q_DURATION_THRESH_MS = 40.0
Q_DEPTH_R_RATIO = 0.33
Q_MIN_DEPTH_MV = 0.1

# ---------------------------------------------------------------------------
# General ECG processing
# ---------------------------------------------------------------------------

def process_ecg(signal, sampling_rate):
    """Shared ECG processing. Returns (signals_df, info_dict)."""
    signals, info = nk.ecg_process(signal, sampling_rate=sampling_rate)
    return signals, info


# ---------------------------------------------------------------------------
# Per-lead morphology analysis
# ---------------------------------------------------------------------------

def _is_missing_sample(sample: Any) -> bool:
    """Return True when a delineation sample is missing."""
    return sample is None or (isinstance(sample, (float, np.floating)) and np.isnan(sample))


def _coerce_sample_index(sample: Any, signal_len: int) -> int | None:
    """Convert a delineation sample to an in-range integer index."""
    if _is_missing_sample(sample):
        return None
    idx = int(sample)
    if 0 <= idx < signal_len:
        return idx
    return None


def _estimate_j_point(
    r_peak: Any,
    s_peak: Any,
    r_onset: Any,
    r_offset: Any,
    sr: int,
    signal_len: int,
) -> int | None:
    """Estimate J-point using the best available delineation landmarks."""
    s_idx = _coerce_sample_index(s_peak, signal_len)
    if s_idx is not None:
        return min(
            signal_len - 1,
            s_idx + int(round(J_POINT_AFTER_S_MS / 1000.0 * sr)),
        )

    r_offset_idx = _coerce_sample_index(r_offset, signal_len)
    if r_offset_idx is not None:
        return r_offset_idx

    r_peak_idx = _coerce_sample_index(r_peak, signal_len)
    if r_peak_idx is None:
        return None

    r_onset_idx = _coerce_sample_index(r_onset, signal_len)
    if r_onset_idx is not None:
        tail_width = int(
            np.clip(
                r_peak_idx - r_onset_idx,
                int(round(0.01 * sr)),
                int(round(0.06 * sr)),
            )
        )
        return min(signal_len - 1, r_peak_idx + tail_width)

    return min(
        signal_len - 1,
        r_peak_idx + int(round(J_POINT_FALLBACK_AFTER_R_MS / 1000.0 * sr)),
    )


def compute_baseline(
    signal: np.ndarray,
    sr: int,
    p_onsets: list,
    t_offsets: list | None = None,
) -> float:
    """Estimate isoelectric baseline from TP segments when available.

    Falls back to the pre-P segment median, then to the median of the
    first 10% of the signal when delineation data is unavailable.
    """
    signal_len = len(signal)
    tp_segments: list[float] = []

    valid_p_onsets = sorted(
        idx for onset in p_onsets if (idx := _coerce_sample_index(onset, signal_len)) is not None
    )
    if t_offsets and valid_p_onsets:
        valid_t_offsets = sorted(
            idx for t_off in t_offsets if (idx := _coerce_sample_index(t_off, signal_len)) is not None
        )
        margin = int(round(TP_SEGMENT_EDGE_MARGIN_MS / 1000.0 * sr))
        min_tp_len = max(1, int(round(TP_SEGMENT_MIN_MS / 1000.0 * sr)))
        p_ptr = 0
        for t_idx in valid_t_offsets:
            while p_ptr < len(valid_p_onsets) and valid_p_onsets[p_ptr] <= t_idx:
                p_ptr += 1
            if p_ptr >= len(valid_p_onsets):
                break

            next_p_idx = valid_p_onsets[p_ptr]
            if next_p_idx - t_idx < min_tp_len:
                continue

            start = t_idx + margin
            end = next_p_idx - margin
            if end <= start:
                start, end = t_idx, next_p_idx
            if end > start:
                tp_segments.append(float(np.median(signal[start:end])))

    if tp_segments:
        return float(np.median(tp_segments))

    segments: list[float] = []
    for onset in p_onsets:
        onset_idx = _coerce_sample_index(onset, signal_len)
        if onset_idx is None:
            continue
        win = max(1, int(round(BASELINE_FALLBACK_WINDOW_MS / 1000.0 * sr)))
        start = max(0, onset_idx - win)
        if start < onset_idx:
            segments.append(float(np.median(signal[start:onset_idx])))
    if segments:
        return float(np.median(segments))
    n = max(1, len(signal) // 10)
    return float(np.median(signal[:n]))


def analyze_single_lead(
    signal: np.ndarray,
    sr: int,
    lead_name: str,
) -> dict[str, Any]:
    """Return per-lead morphology features (ST, T-wave, Q-wave, R-wave) for one lead."""

    result: dict[str, Any] = {"lead": lead_name}

    # --- ECG Processing & Delineation ---
    try:
        _, info = nk.ecg_process(signal, sampling_rate=sr)
    except Exception as e:
        result["error"] = f"ECG processing failed: {e}"
        return result

    r_peaks = info.get("ECG_R_Peaks", [])
    if len(r_peaks) < 2:
        result["error"] = "Too few R-peaks detected."
        return result

    try:
        _, waves = nk.ecg_delineate(signal, r_peaks, sampling_rate=sr, method="dwt")
    except Exception as e:
        result["error"] = f"Delineation failed: {e}"
        return result

    p_onsets = waves.get("ECG_P_Onsets", [])
    q_peaks = waves.get("ECG_Q_Peaks", [])
    r_onsets = waves.get("ECG_R_Onsets", [])
    r_offsets = waves.get("ECG_R_Offsets", [])
    s_peaks = waves.get("ECG_S_Peaks", [])
    t_peaks = waves.get("ECG_T_Peaks", [])
    t_offsets = waves.get("ECG_T_Offsets", [])

    baseline = compute_baseline(signal, sr, p_onsets, t_offsets)

    # --- ST Segment Analysis ---
    st_offsets_mv: list[float] = []
    for i, r_idx in enumerate(r_peaks):
        s_peak = s_peaks[i] if i < len(s_peaks) else None
        r_onset = r_onsets[i] if i < len(r_onsets) else None
        r_offset = r_offsets[i] if i < len(r_offsets) else None
        j_point = _estimate_j_point(r_idx, s_peak, r_onset, r_offset, sr, len(signal))
        if j_point is None:
            continue

        center = j_point + int(round(ST_MEASURE_CENTER_MS / 1000.0 * sr))
        half_window = max(1, int(round(ST_MEASURE_HALF_WINDOW_MS / 1000.0 * sr)))
        seg_start = max(0, center - half_window)
        seg_end = min(len(signal), center + half_window + 1)
        if seg_start >= len(signal):
            continue
        seg_end = max(seg_start + 1, seg_end)

        st_amp = float(np.mean(signal[seg_start:seg_end])) - baseline
        st_offsets_mv.append(st_amp)

    if st_offsets_mv:
        mean_st = float(np.mean(st_offsets_mv))
        if mean_st > ST_ELEVATION_THRESH:
            st_status = "elevation"
        elif mean_st < ST_DEPRESSION_THRESH:
            st_status = "depression"
        else:
            st_status = "normal"
        result["st_segment"] = {
            "status": st_status,
            "mean_offset_mv": round(mean_st, 3),
            "n_beats": len(st_offsets_mv),
        }
    else:
        result["st_segment"] = {
            "status": "unable_to_measure",
            "reason": "No valid J-point found.",
        }

    # --- T-Wave Analysis ---
    t_amps: list[float] = []
    for tp in t_peaks:
        tp_idx = _coerce_sample_index(tp, len(signal))
        if tp_idx is None:
            continue
        t_amps.append(float(signal[tp_idx]) - baseline)

    if t_amps:
        mean_t = float(np.mean(t_amps))
        if abs(mean_t) < T_FLAT_THRESH:
            t_status = "flat"
        elif mean_t > 0:
            t_status = "upright"
        else:
            t_status = "inverted"
        result["t_wave"] = {
            "status": t_status,
            "mean_amplitude_mv": round(mean_t, 3),
            "n_beats": len(t_amps),
        }
    else:
        result["t_wave"] = {
            "status": "unable_to_measure",
            "reason": "T peaks not detected.",
        }

    # --- Q-Wave Analysis ---
    q_durations_ms: list[float] = []
    q_depths_mv: list[float] = []
    r_amplitudes_mv: list[float] = []

    for i in range(min(len(q_peaks), len(r_peaks))):
        q = q_peaks[i] if i < len(q_peaks) else None
        r_on = r_onsets[i] if i < len(r_onsets) else None
        r_idx = r_peaks[i]

        r_amp = (
            float(signal[int(r_idx)]) - baseline if int(r_idx) < len(signal) else 0.0
        )
        r_amplitudes_mv.append(r_amp)

        if q is None or (isinstance(q, float) and np.isnan(q)):
            continue
        q = int(q)
        if q >= len(signal):
            continue

        q_amp = float(signal[q]) - baseline
        if q_amp >= 0:
            continue

        q_depth = abs(q_amp)
        q_depths_mv.append(q_depth)

        if r_on is not None and not (isinstance(r_on, float) and np.isnan(r_on)):
            q_dur_ms = abs(q - int(r_on)) / sr * 1000.0
        else:
            q_dur_ms = abs(int(r_idx) - q) / sr * 1000.0
        q_durations_ms.append(q_dur_ms)

    pathological = False
    if q_depths_mv:
        mean_q_dur = float(np.mean(q_durations_ms)) if q_durations_ms else 0.0
        mean_q_depth = float(np.mean(q_depths_mv))
        mean_r_amp = float(np.mean(r_amplitudes_mv)) if r_amplitudes_mv else 0.0

        depth_significant = mean_q_depth > Q_MIN_DEPTH_MV
        duration_long = mean_q_dur > Q_DURATION_THRESH_MS
        depth_ratio_high = (
            mean_r_amp > 0 and (mean_q_depth / mean_r_amp) > Q_DEPTH_R_RATIO
        )
        if depth_significant and duration_long and depth_ratio_high:
            pathological = True

        result["q_wave"] = {
            "present": True,
            "pathological": pathological,
            "mean_duration_ms": round(mean_q_dur, 1),
            "mean_depth_mv": round(mean_q_depth, 3),
            "n_beats": len(q_depths_mv),
        }
    else:
        result["q_wave"] = {"present": False, "pathological": False}

    # --- R-Wave Amplitude ---
    if r_amplitudes_mv:
        result["r_wave"] = {
            "mean_amplitude_mv": round(float(np.mean(r_amplitudes_mv)), 3),
            "max_amplitude_mv": round(float(np.max(r_amplitudes_mv)), 3),
        }

    # --- Per-Lead Interpretation Summary ---
    findings: list[str] = []
    st_info = result.get("st_segment", {})
    if st_info.get("status") == "elevation":
        findings.append(f"ST elevation ({st_info['mean_offset_mv']:.2f} mV)")
    elif st_info.get("status") == "depression":
        findings.append(f"ST depression ({st_info['mean_offset_mv']:.2f} mV)")

    t_info = result.get("t_wave", {})
    if t_info.get("status") == "inverted":
        findings.append("T-wave inversion")
    elif t_info.get("status") == "flat":
        findings.append("T-wave flattening")

    q_info = result.get("q_wave", {})
    if q_info.get("pathological"):
        findings.append(
            f"Pathological Q wave (dur {q_info.get('mean_duration_ms', 0):.0f} ms, "
            f"depth {q_info.get('mean_depth_mv', 0):.2f} mV)"
        )
    elif q_info.get("present"):
        findings.append("Q wave present (non-pathological)")

    result["findings"] = (
        findings if findings else ["No significant morphological abnormalities"]
    )

    return result


# ---------------------------------------------------------------------------
# Signal quality assessment
# ---------------------------------------------------------------------------

def parse_noise_metadata(record_metadata: dict[str, Any]) -> dict[str, Any]:
    """Parse PTB-XL noise annotations into a structured dict.

    Returns dict like:
        {"baseline_drift": {"present": True, "leads": ["II", "III"]}, ...}
    """
    annotations: dict[str, Any] = {}
    for noise_type in NOISE_TYPES:
        raw = record_metadata.get(noise_type, "")
        if not raw or not str(raw).strip():
            annotations[noise_type] = {"present": False, "leads": []}
            continue
        parts = [p.strip().upper() for p in str(raw).split(",") if p.strip()]
        leads = [p for p in parts if p]
        annotations[noise_type] = {
            "present": len(leads) > 0,
            "leads": leads,
        }
    return annotations


def assess_single_lead_quality(
    signal: np.ndarray,
    sr: int,
    lead_name: str,
    noise_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess quality and noise for a single lead.

    Parameters
    ----------
    signal : 1-D array
    sr : sampling rate
    lead_name : e.g. "II", "V4"
    noise_metadata : parsed PTB-XL noise annotations for this record
        (output of parse_noise_metadata)

    Returns
    -------
    dict with quality_score, quality_label, noise_types, signal_stats
    """
    result: dict[str, Any] = {"lead": lead_name}

    # --- NeuroKit2 signal quality ---
    nk_quality_score = None
    try:
        quality_series = nk.ecg_quality(signal, sampling_rate=sr, method="zhao2018")
        quality_counts: dict[str, int] = {}
        for q in quality_series:
            q_str = str(q)
            quality_counts[q_str] = quality_counts.get(q_str, 0) + 1
        total = len(quality_series)
        excellent_frac = quality_counts.get("Excellent", 0) / total if total > 0 else 0
        acceptable_frac = quality_counts.get("Barely Acceptable", 0) / total if total > 0 else 0
        nk_quality_score = excellent_frac + 0.5 * acceptable_frac
    except Exception as e:
        logger.debug("NK quality failed for lead %s: %s", lead_name, e)

    # --- Basic signal statistics ---
    sig_std = float(np.std(signal))
    sig_range = float(np.ptp(signal))

    quarter = len(signal) // 4
    drift = 0.0
    if quarter > 0:
        drift = abs(float(np.mean(signal[:quarter]) - np.mean(signal[-quarter:])))

    noise_level = float(np.std(np.diff(signal)))

    # Composite quality score
    scores: list[float] = []
    if nk_quality_score is not None:
        scores.append(nk_quality_score)
    if sig_std > 0:
        snr_proxy = sig_std / (noise_level + 1e-8)
        scores.append(min(1.0, snr_proxy / 5.0))
    if sig_range > 0:
        scores.append(max(0.0, 1.0 - drift / sig_range))

    quality_score = round(float(np.mean(scores)), 2) if scores else 0.5

    if quality_score >= 0.8:
        quality_label = "GOOD"
    elif quality_score >= 0.5:
        quality_label = "ACCEPTABLE"
    else:
        quality_label = "POOR"

    result["quality_score"] = quality_score
    result["quality_label"] = quality_label
    result["signal_stats"] = {
        "noise_level": round(noise_level, 4),
        "baseline_drift": round(drift, 4),
    }

    # --- Noise from PTB-XL metadata ---
    lead_upper = lead_name.upper()
    detected_noise: list[str] = []

    if noise_metadata:
        for ntype in NOISE_TYPES:
            info = noise_metadata.get(ntype, {})
            if info.get("present") and lead_upper in info.get("leads", []):
                detected_noise.append(ntype)

    result["noise_types"] = detected_noise
    result["has_noise"] = len(detected_noise) > 0

    return result
