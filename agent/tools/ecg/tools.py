"""
ECG data tools — concrete tool implementations for the reactive pipeline.

All tools return text-friendly representations so the LLM can directly
understand and reason about the ECG data (no embedding needed for now).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import neurokit2 as nk

from agent.encoder.ecgfounder_runtime import (
    load_ecgfounder_runtime,
    resample_signal_to_length,
    z_score_normalize,
)
from agent.tools.registry import is_tool_enabled, register_tool
from agent.data.ecg_loader import get_loader
from agent.data.record_repository import PTBXL_SOURCE, get_record_repository
from agent.tools.ecg.ecgfounder_mapping import (
    UNCOVERED_DIAGNOSTIC_CONCEPTS,
    build_scp_relevant_findings,
)
from agent.tools.ecg.utils import (
    ALL_LEADS,
    NOISE_TYPES,
    analyze_single_lead,
    assess_single_lead_quality,
    parse_noise_metadata,
    process_ecg,
)

logger = logging.getLogger(__name__)

# Backward-compatible aliases kept for existing tests and call sites.
_resample_signal_to_length = resample_signal_to_length
_z_score_normalize = z_score_normalize
ECGDiagnosisMode = Literal["12_lead", "single_lead"]


def _add_interval_range_flags(
    payload: dict[str, Any],
    *,
    value_key: str,
    lower_normal_ms: float | None,
    upper_normal_ms: float | None,
) -> dict[str, Any]:
    """Add explicit range flags so the LLM does not have to infer boundaries from prose."""
    value = payload.get(value_key)
    if value is None:
        return payload

    payload["normal_range_ms"] = {
        "lower": lower_normal_ms,
        "upper": upper_normal_ms,
    }
    if lower_normal_ms is not None:
        payload["is_below_normal"] = bool(value < lower_normal_ms)
    if upper_normal_ms is not None:
        payload["is_above_normal"] = bool(value > upper_normal_ms)

    below = bool(payload.get("is_below_normal", False))
    above = bool(payload.get("is_above_normal", False))
    payload["is_within_normal"] = not below and not above
    return payload


def _format_ecg_diagnosis_finding(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "concept": str(item.get("diagnostic_concept", "")),
        "probability": round(float(item.get("aggregate_probability", 0.0)), 4),
        "match_type": str(item.get("match_type", "")),
        "match_scope": str(item.get("match_scope", "")),
        "note": str(item.get("note", "")),
        "matched_labels": list(item.get("matched_labels", [])),
    }


def _build_ecg_diagnosis_coverage_note(
    *,
    present_findings: list[dict[str, Any]],
    uncertain_findings: list[dict[str, Any]],
) -> str:
    if present_findings:
        status_note = "Mapped findings were detected, but they should be interpreted with coverage limits in mind."
    elif uncertain_findings:
        status_note = "Only weak mapped findings were detected, so the output should be treated as supporting context rather than a diagnosis."
    else:
        status_note = "No mapped concept crossed the confidence threshold, but that does not prove the asked concept is absent."

    uncovered_examples = ", ".join(sorted(UNCOVERED_DIAGNOSTIC_CONCEPTS))
    limitation_note = (
        "Absence from `present` is not proof of absence. "
        "Only `present` findings with `match_scope == specific` should be treated as strong positive evidence. "
        "Findings with `match_scope == partial` or `match_scope == aggregate` are approximate matches and cannot confirm a diagnosis by themselves. "
        "Lead-specific ST/T/Q findings, Q-wave presence, form-related questions, voltage-criteria questions, long-QT questions, and region-qualified ischemia / injury / infarction are weak coverage areas and should be checked against the report and morphology tools. "
        f"Known uncovered or weakly covered diagnostic concepts include: {uncovered_examples}."
    )
    return f"{status_note} {limitation_note}"


def _normalize_ecg_diagnosis_mode(mode: ECGDiagnosisMode | str) -> ECGDiagnosisMode:
    if mode not in {"12_lead", "single_lead"}:
        raise ValueError(
            f"Unsupported ecg_diagnosis mode {mode!r}. Expected '12_lead' or 'single_lead'."
        )
    return mode



def _resolve_record_reference(record_id: str | int) -> tuple[str, str]:
    ref = get_record_repository().parse_record_reference(record_id)
    return ref.source, ref.raw_record_id


@lru_cache(maxsize=512)
def _load_record(record_id: str | int, sampling_rate: int = 100):
    source, raw_record_id = _resolve_record_reference(record_id)
    if source == PTBXL_SOURCE:
        return get_loader(sampling_rate).get_record(raw_record_id)
    return get_record_repository().get_record(record_id, sampling_rate=sampling_rate)


def _resolve_analysis_lead(record: Any, requested_lead: str | None, *, default_lead: str = "II") -> str:
    lead_names = [str(name).upper() for name in getattr(record, "lead_names", [])]
    requested = (requested_lead or default_lead).strip().upper()
    if requested in lead_names:
        return requested

    if int(getattr(record, "num_leads", len(lead_names) or 0)) == 1 and lead_names:
        only_lead = lead_names[0]
        if requested in {"", default_lead.upper(), "I"}:
            return only_lead

    raise ValueError(f"Lead '{requested}' not available. Available: {lead_names}")
def _prepare_ecgfounder_signal(
    record: Any,
    *,
    mode: ECGDiagnosisMode,
) -> tuple[np.ndarray, str | None]:
    signal_array = np.asarray(record.signal, dtype=np.float32)
    lead_names = [str(name).upper() for name in getattr(record, "lead_names", [])]
    record_label = getattr(record, "record_id", "<unknown>")

    if mode == "12_lead":
        if signal_array.ndim != 2:
            raise ValueError(
                f"ECGFounder 12-lead mode expects a 2D signal array, got shape {signal_array.shape!r}."
            )
        if int(record.num_leads) != 12:
            raise ValueError(
                f"ECGFounder 12-lead mode expects 12 leads, but record '{record_label}' has {record.num_leads}."
            )
        return signal_array.T, None

    if int(record.num_leads) == 1:
        if lead_names and lead_names[0] != "I":
            raise ValueError(
                f"Single-lead ECGFounder support is limited to Lead I, but record '{record_label}' exposes {lead_names[0]}."
            )
        if signal_array.ndim == 2:
            if signal_array.shape[1] == 1:
                signal_array = signal_array[:, 0]
            elif signal_array.shape[0] == 1:
                signal_array = signal_array[0]
        if signal_array.ndim != 1:
            raise ValueError(
                f"ECGFounder single-lead mode expects a 1D signal, got shape {signal_array.shape!r}."
            )
        return signal_array.reshape(1, -1), "I"

    if lead_names and "I" not in lead_names:
        raise ValueError(
            f"Single-lead ECGFounder support requires Lead I, but record '{record_label}' does not include it."
        )

    try:
        lead_signal = np.asarray(record.get_lead("I"), dtype=np.float32)
    except Exception as exc:
        raise ValueError(
            f"Single-lead ECGFounder support requires Lead I, but record '{record_label}' does not include it."
        ) from exc

    if lead_signal.ndim != 1:
        raise ValueError(
            f"ECGFounder single-lead mode expects Lead I to be 1D, got shape {lead_signal.shape!r}."
        )
    return lead_signal.reshape(1, -1), "I"




# -----------------------------------------------------------------------
# Tool: list available ECG records
# -----------------------------------------------------------------------
@register_tool(
    name="list_ecg_records",
    description=(
        "List available ECG record IDs. Optionally filter by age range, sex, "
        "or diagnostic class. Returns a list of record IDs with brief metadata."
    ),
    parameters={
        "limit": {
            "type": "integer",
            "description": "Max number of records to return.",
            "default": 20,
        },
        "age_min": {
            "type": "integer",
            "description": "Minimum patient age filter.",
        },
        "age_max": {
            "type": "integer",
            "description": "Maximum patient age filter.",
        },
        "diagnostic_class": {
            "type": "string",
            "description": "Filter by diagnostic class (e.g. NORM, MI, STTC, CD, HYP).",
        },
    },
)
def list_ecg_records(
    limit: int = 20,
    age_min: int | None = None,
    age_max: int | None = None,
    diagnostic_class: str | None = None,
) -> dict[str, Any]:
    records = get_record_repository().list_records(
        limit=limit,
        age_min=age_min,
        age_max=age_max,
        diagnostic_class=diagnostic_class,
    )
    ids = [str(item.get("record_id")) for item in records if item.get("record_id") is not None]

    return {
        "success": True,
        "record_ids": ids,
        "records": records,
        "count": len(ids),
    }


# -----------------------------------------------------------------------
# Tool: get ECG text description (primary tool for LLM understanding)
# -----------------------------------------------------------------------
@register_tool(
    name="get_ecg_description",
    description=(
        "Get a detailed text description of a specific ECG record. "
        "Includes patient info, clinical report, diagnostic codes, "
        "signal statistics, and estimated heart rate. "
        "This is the primary tool for understanding an ECG record."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "The ECG record ID to describe.",
        },
    },
)
def get_ecg_description(record_id: str) -> dict[str, Any]:
    record = _load_record(record_id)
    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    text = record.to_text()
    return {
        "success": True,
        "description": text,
        "record_id": record_id,
    }


# -----------------------------------------------------------------------
# Tool: get ECG metadata
# -----------------------------------------------------------------------
@register_tool(
    name="get_ecg_metadata",
    description=(
        "Retrieve patient and recording metadata for an ECG record. "
        "Includes age, sex, clinical report, recording date, diagnostic codes."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "The ECG record ID.",
        },
    },
)
def get_ecg_metadata(record_id: str) -> dict[str, Any]:
    record = _load_record(record_id)
    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    meta = dict(getattr(record, "metadata", {}) or {})
    meta["record_id"] = record_id
    meta["source"] = meta.get("source", _resolve_record_reference(record_id)[0])
    meta["lead_names"] = list(getattr(record, "lead_names", []))
    meta["num_leads"] = record.num_leads
    meta["duration_seconds"] = record.duration_seconds
    meta["sampling_rate"] = record.sampling_rate

    return {"success": True, "metadata": meta}

# -----------------------------------------------------------------------
# Tool: analyze heart rate
# -----------------------------------------------------------------------
@register_tool(
    name="analyze_heart_rate",
    description=(
        "Analyze heart rate from an ECG record using R-peak detection. "
        "Returns mean/min/max heart rate, RR interval statistics, "
        "and a clinical assessment (bradycardia/normal/tachycardia). "
        "Use this tool when the user asks about heart rate, pulse, "
        "or general cardiac rhythm speed."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to analyze.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Sampling rate to load the signal. 100 or 500. Default 100.",
            "required": False,
            "default": 100,
        },
        "lead": {
            "type": "string",
            "description": "Which lead to analyze. Default 'II' (standard for rhythm analysis).",
            "required": False,
            "default": "II",
        },
    },
)
def analyze_heart_rate(
    record_id: str,
    sampling_rate: int = 100,
    lead: str = "II",
) -> dict[str, Any]:
    """Analyze heart rate using NeuroKit2 R-peak detection."""

    # --- Load record ---
    record = _load_record(record_id, sampling_rate)

    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    try:
        lead = _resolve_analysis_lead(record, lead, default_lead="II")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    signal = record.get_lead(lead)
    sr = record.sampling_rate

    signal

    # R-peak detection via NeuroKit2
    try:
        signals, info = process_ecg(signal, sr)
    except Exception as e:
        return {"success": False, "error": f"ECG processing failed: {e}"}

    r_peak_indices = info["ECG_R_Peaks"]

    if len(r_peak_indices) < 2:
        return {
            "success": False,
            "error": "Too few R-peaks detected. Signal may be too short or noisy.",
        }

    # Compute heart rate
    ecg_rate = signals["ECG_Rate"].dropna()
    hr_mean = round(float(ecg_rate.mean()), 1)
    hr_min = round(float(ecg_rate.min()), 1)
    hr_max = round(float(ecg_rate.max()), 1)
    hr_std = round(float(ecg_rate.std()), 1)

    # RR intervals
    rr_intervals_samples = np.diff(r_peak_indices)
    rr_intervals_ms = (rr_intervals_samples / sr) * 1000
    rr_mean = round(float(np.mean(rr_intervals_ms)), 1)
    rr_std = round(float(np.std(rr_intervals_ms)), 1)

    # Clinical assessment
    if hr_mean < 60:
        assessment = "bradycardia"
        assessment_detail = (
            f"Mean heart rate ({hr_mean} bpm) is below 60 bpm, indicating bradycardia."
        )
    elif hr_mean > 100:
        assessment = "tachycardia"
        assessment_detail = (
            f"Mean heart rate ({hr_mean} bpm) is above 100 bpm, indicating tachycardia."
        )
    else:
        assessment = "normal"
        assessment_detail = (
            f"Mean heart rate ({hr_mean} bpm) is within normal range (60-100 bpm)."
        )

    rr_assessment = "normal"
    rr_assessment_detail = (
        f"Mean RR interval ({rr_mean} ms) is within normal range (600-1000 ms)."
    )
    if rr_mean < 600:
        rr_assessment = "below_normal"
        rr_assessment_detail = f"Mean RR interval ({rr_mean} ms) is below normal range (< 600 ms), indicating fast heart rate."
    elif rr_mean > 1000:
        rr_assessment = "above_normal"
        rr_assessment_detail = f"Mean RR interval ({rr_mean} ms) is above normal range (> 1000 ms), indicating slow heart rate."

    # Check for significant variability
    hr_range = hr_max - hr_min
    variability_note = None
    if hr_range > 30:
        variability_note = (
            f"Heart rate varies significantly ({hr_min}-{hr_max} bpm, range {round(hr_range, 1)} bpm). "
            "Consider rhythm analysis for possible irregularities."
        )

    return {
        "success": True,
        "data": {
            "heart_rate": {
                "mean_bpm": hr_mean,
                "min_bpm": hr_min,
                "max_bpm": hr_max,
                "std_bpm": hr_std,
            },
            "rr_intervals": {
                "mean_ms": rr_mean,
                "std_ms": rr_std,
                "count": len(rr_intervals_ms),
                "assessment": rr_assessment,
                "assessment_detail": rr_assessment_detail,
                "normal_range_ms": {"lower": 600.0, "upper": 1000.0},
                "is_below_normal": rr_mean < 600,
                "is_above_normal": rr_mean > 1000,
                "is_within_normal": 600 <= rr_mean <= 1000,
            },
            "assessment": assessment,
            "assessment_detail": assessment_detail,
            "variability_note": variability_note,
        },
        "metadata": {
            "lead_analyzed": lead,
            "sampling_rate": sr,
            "num_r_peaks": len(r_peak_indices),
            "duration_seconds": round(record.duration_seconds, 1),
        },
    }


# -----------------------------------------------------------------------
# Tool: analyze heart rate variability (HRV)
# -----------------------------------------------------------------------
@register_tool(
    name="analyze_hrv",
    description=(
        "Analyze heart rate variability (HRV) from an ECG record. "
        "Returns time-domain metrics (SDNN, RMSSD, pNN50) and frequency-domain "
        "power (LF, HF, LF/HF ratio) when signal is long enough. "
        "Use when the user asks about HRV, stress, autonomic function, "
        "or cardiac variability."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to analyze.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Sampling rate to load the signal. 100 or 500. Default 100.",
            "required": False,
            "default": 100,
        },
        "lead": {
            "type": "string",
            "description": "Which lead to analyze. Default 'II'.",
            "required": False,
            "default": "II",
        },
    },
)
def analyze_hrv(
    record_id: str,
    sampling_rate: int = 100,
    lead: str = "II",
) -> dict[str, Any]:
    """Analyze heart rate variability using R-peak derived NN intervals."""

    # --- Load record ---
    record = _load_record(record_id, sampling_rate)

    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    try:
        lead = _resolve_analysis_lead(record, lead, default_lead="II")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    signal = record.get_lead(lead)
    sr = record.sampling_rate

    # --- R-peak detection ---
    try:
        signals, info = process_ecg(signal, sr)
    except Exception as e:
        return {"success": False, "error": f"ECG processing failed: {e}"}

    r_peak_indices = info["ECG_R_Peaks"]

    if len(r_peak_indices) < 3:
        return {
            "success": False,
            "error": "Too few R-peaks detected (need ≥3). Signal may be too short or noisy.",
        }

    # --- NN intervals with artifact filtering ---
    rr_samples = np.diff(r_peak_indices)
    nn_intervals_ms = (rr_samples / sr) * 1000.0

    mask = (nn_intervals_ms >= 300) & (nn_intervals_ms <= 2000)
    nn_clean = nn_intervals_ms[mask]

    if len(nn_clean) < 3:
        return {
            "success": False,
            "error": "Too few valid NN intervals after artifact removal.",
        }

    # --- Time-domain HRV ---
    nn_diffs = np.diff(nn_clean)
    time_domain = {
        "mean_nn_ms": round(float(np.mean(nn_clean)), 1),
        "sdnn_ms": round(float(np.std(nn_clean, ddof=1)), 1),
        "rmssd_ms": round(float(np.sqrt(np.mean(nn_diffs**2))), 1),
        "pnn50_percent": round(
            float(np.sum(np.abs(nn_diffs) > 50) / len(nn_diffs) * 100), 1
        )
        if len(nn_diffs) > 0
        else 0.0,
        "nn_count": int(len(nn_clean)),
    }

    # --- Frequency-domain HRV (needs sufficient data) ---
    duration_s = record.duration_seconds
    freq_domain = None

    if len(nn_clean) >= 10 and duration_s >= 10:
        try:
            hrv_freq = nk.hrv_frequency(
                {"ECG_R_Peaks": r_peak_indices},
                sampling_rate=sr,
                show=False,
            )
            fd = {}
            for col, key in [
                ("HRV_LF", "lf_power_ms2"),
                ("HRV_HF", "hf_power_ms2"),
                ("HRV_LFHF", "lf_hf_ratio"),
            ]:
                if col in hrv_freq.columns:
                    v = hrv_freq[col].values[0]
                    if np.isfinite(v):
                        fd[key] = round(float(v), 2)
            freq_domain = fd or None
        except Exception as e:
            logger.warning("Frequency-domain HRV failed: %s", e)

    # --- Caveats ---
    caveats = []
    if duration_s < 30:
        caveats.append(
            f"Recording is very short ({duration_s:.0f}s); HRV metrics have limited reliability."
        )
    elif duration_s < 300:
        caveats.append(
            f"Recording ({duration_s:.0f}s) is below the 5-min clinical standard; interpret with caution."
        )
    if freq_domain is None and duration_s < 120:
        caveats.append(
            "Frequency-domain metrics unavailable due to short recording duration."
        )

    # --- Response ---
    result_data: dict[str, Any] = {"time_domain": time_domain}
    if freq_domain:
        result_data["frequency_domain"] = freq_domain
    if caveats:
        result_data["caveats"] = caveats

    return {
        "success": True,
        "data": result_data,
        "metadata": {
            "lead_analyzed": lead,
            "sampling_rate": sr,
            "num_r_peaks": len(r_peak_indices),
            "duration_seconds": round(duration_s, 1),
            "nn_artifacts_removed": int(len(nn_intervals_ms) - len(nn_clean)),
        },
    }


# -----------------------------------------------------------------------
# Tool: analyze ECG morphology (QRS, PR, QT intervals)
# -----------------------------------------------------------------------
@register_tool(
    name="analyze_morphology",
    description=(
        "Analyze ECG waveform morphology: QRS duration, PR interval, QT interval, "
        "and corrected QT (QTc via Bazett's formula). "
        "Use this tool when the user asks about conduction delays, "
        "bundle branch blocks, interval prolongation, or waveform shape."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to analyze.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Sampling rate. 500 recommended for morphology. Default 500.",
            "required": False,
            "default": 500,
        },
        "lead": {
            "type": "string",
            "description": "Lead to analyze. Default 'II'.",
            "required": False,
            "default": "II",
        },
    },
)
def analyze_morphology(
    record_id: str,
    sampling_rate: int = 500,
    lead: str = "II",
) -> dict[str, Any]:
    """Analyze ECG waveform morphology using NeuroKit2 delineation."""

    record = _load_record(record_id, sampling_rate)

    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    try:
        lead = _resolve_analysis_lead(record, lead, default_lead="II")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    signal = record.get_lead(lead)
    sr = record.sampling_rate

    # --- Process ECG ---
    try:
        signals, info = process_ecg(signal, sr)
    except Exception as e:
        return {"success": False, "error": f"ECG processing failed: {e}"}

    r_peaks = info["ECG_R_Peaks"]
    if len(r_peaks) < 2:
        return {"success": False, "error": "Too few R-peaks for morphology analysis."}

    # --- Delineation ---
    try:
        _, waves = nk.ecg_delineate(signal, r_peaks, sampling_rate=sr, method="dwt")
    except Exception as e:
        return {"success": False, "error": f"ECG delineation failed: {e}"}

    # --- Helper: extract valid intervals with outlier filtering ---
    def _extract_intervals(onsets, offsets, sr_hz, plausible_range=(10, 800)):
        """Compute intervals in ms, filtering NaN and physiologically implausible values."""
        durations = []
        lo, hi = plausible_range
        for on, off in zip(onsets, offsets):
            if on is None or off is None:
                continue
            try:
                on_f, off_f = float(on), float(off)
            except (TypeError, ValueError):
                continue
            if np.isnan(on_f) or np.isnan(off_f):
                continue
            dur_ms = (off_f - on_f) / sr_hz * 1000.0
            if lo <= dur_ms <= hi:
                durations.append(dur_ms)
        return durations

    # QRS duration: R_Onsets → R_Offsets (NeuroKit2 dwt method labels these as QRS boundaries)
    qrs_durations = _extract_intervals(
        waves.get("ECG_R_Onsets", []),
        waves.get("ECG_R_Offsets", []),
        sr,
        plausible_range=(20, 250),  # QRS: 20-250 ms
    )

    # PR interval: P_Onsets → R_Onsets
    pr_intervals = _extract_intervals(
        waves.get("ECG_P_Onsets", []),
        waves.get("ECG_R_Onsets", []),
        sr,
        plausible_range=(60, 400),  # PR: 60-400 ms
    )

    # QT interval: R_Onsets → T_Offsets
    qt_intervals = _extract_intervals(
        waves.get("ECG_R_Onsets", []),
        waves.get("ECG_T_Offsets", []),
        sr,
        plausible_range=(200, 700),  # QT: 200-700 ms
    )

    # Mean RR for QTc
    rr_ms = np.diff(r_peaks) / sr * 1000.0
    mean_rr_s = float(np.mean(rr_ms)) / 1000.0 if len(rr_ms) > 0 else None

    # --- Aggregate ---
    def _stats(values):
        if not values:
            return None
        arr = np.array(values)
        return {
            "mean_ms": round(float(np.mean(arr)), 1),
            "std_ms": round(float(np.std(arr)), 1),
            "min_ms": round(float(np.min(arr)), 1),
            "max_ms": round(float(np.max(arr)), 1),
            "n_beats": len(values),
        }

    qrs_stats = _stats(qrs_durations)
    pr_stats = _stats(pr_intervals)
    qt_stats = _stats(qt_intervals)

    # QTc (Bazett)
    qtc_ms = None
    if qt_stats and mean_rr_s and mean_rr_s > 0:
        qtc_ms = round(float(qt_stats["mean_ms"] / np.sqrt(mean_rr_s)), 1)

    # --- Build result with reference ranges (let LLM interpret) ---
    data: dict[str, Any] = {}

    if qrs_stats:
        qrs_stats["reference_range_ms"] = (
            "60-100 normal, 100-120 borderline, >120 prolonged"
        )
        _add_interval_range_flags(
            qrs_stats,
            value_key="mean_ms",
            lower_normal_ms=60.0,
            upper_normal_ms=100.0,
        )
        data["qrs_duration"] = qrs_stats
    if pr_stats:
        pr_stats["reference_range_ms"] = (
            "<120 short (pre-excitation), 120-200 normal, >200 prolonged (1st degree AV block)"
        )
        _add_interval_range_flags(
            pr_stats,
            value_key="mean_ms",
            lower_normal_ms=120.0,
            upper_normal_ms=200.0,
        )
        data["pr_interval"] = pr_stats
    if qt_stats:
        _add_interval_range_flags(
            qt_stats,
            value_key="mean_ms",
            lower_normal_ms=350.0,
            upper_normal_ms=450.0,
        )
        qt_stats["reference_range_ms"] = "350-450 normal"
        data["qt_interval"] = qt_stats
    if qtc_ms is not None:
        data["qtc_bazett"] = {
            "value_ms": qtc_ms,
            "reference_range_ms": "<350 short, 350-450 normal, 450-500 borderline, >500 prolonged",
        }
        _add_interval_range_flags(
            data["qtc_bazett"],
            value_key="value_ms",
            lower_normal_ms=350.0,
            upper_normal_ms=450.0,
        )

    # Caveats
    caveats = []
    measured = sum(1 for s in [qrs_stats, pr_stats, qt_stats] if s is not None)
    if measured == 0:
        caveats.append(
            "Wave delineation failed for all intervals. Signal may be too noisy or short."
        )
    elif measured < 3:
        missing = []
        if not qrs_stats:
            missing.append("QRS")
        if not pr_stats:
            missing.append("PR")
        if not qt_stats:
            missing.append("QT")
        caveats.append(
            f"Could not measure: {', '.join(missing)}. Partial delineation only."
        )

    if record.duration_seconds < 5:
        caveats.append(
            f"Very short recording ({record.duration_seconds:.0f}s); measurements may be unreliable."
        )

    if caveats:
        data["caveats"] = caveats

    return {
        "success": True,
        "data": data,
        "metadata": {
            "lead_analyzed": lead,
            "sampling_rate": sr,
            "num_r_peaks": len(r_peaks),
            "duration_seconds": round(record.duration_seconds, 1),
            "intervals_measured": measured,
        },
    }


# -----------------------------------------------------------------------
# Tool: assess ECG signal quality
# -----------------------------------------------------------------------
@register_tool(
    name="assess_signal_quality",
    description=(
        "Assess the quality of an ECG signal. Returns an overall quality score "
        "(0-1), per-component scores (NeuroKit2 analysis, SNR, baseline drift), "
        "noise annotations from PTB-XL metadata, and whether the signal is "
        "suitable for analysis. Use when the user asks about recording quality, "
        "noise, or to verify signal reliability before trusting other analysis results."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to assess.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Sampling rate. Default 100.",
            "required": False,
            "default": 100,
        },
        "lead": {
            "type": "string",
            "description": "Lead to assess. Default 'II'.",
            "required": False,
            "default": "II",
        },
    },
)
def assess_signal_quality(
    record_id: str,
    sampling_rate: int = 100,
    lead: str = "II",
) -> dict[str, Any]:
    """Assess ECG signal quality combining algorithmic analysis and PTB-XL metadata."""

    record = _load_record(record_id, sampling_rate)

    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    try:
        lead = _resolve_analysis_lead(record, lead, default_lead="II")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    signal = record.get_lead(lead)
    sr = record.sampling_rate

    # ===================== Component 1: NeuroKit2 quality =====================
    nk_score = None
    nk_breakdown = {}
    try:
        quality_series = nk.ecg_quality(signal, sampling_rate=sr, method="zhao2018")
        counts = {}
        for q in quality_series:
            q_str = str(q)
            counts[q_str] = counts.get(q_str, 0) + 1
        total = len(quality_series)
        if total > 0:
            excellent_frac = counts.get("Excellent", 0) / total
            acceptable_frac = counts.get("Barely Acceptable", 0) / total
            nk_score = round(excellent_frac + 0.5 * acceptable_frac, 3)
            nk_breakdown = {k: round(v / total, 3) for k, v in counts.items()}
    except Exception as e:
        logger.warning("NeuroKit2 quality assessment failed: %s", e)

    # ===================== Component 2: Signal-level metrics =====================
    sig_std = float(np.std(signal))
    sig_range = float(np.ptp(signal))

    # SNR proxy: signal energy vs high-frequency noise energy
    noise_std = float(np.std(np.diff(signal)))
    snr_score = None
    if noise_std > 0:
        snr_ratio = sig_std / noise_std
        # Map to 0-1: ratio < 1 is terrible, > 3 is good
        snr_score = round(float(np.clip((snr_ratio - 1.0) / 2.0, 0.0, 1.0)), 3)

    # Baseline drift: mean difference between first and last quarter
    quarter = len(signal) // 4
    drift_abs = 0.0
    drift_score = None
    if quarter > 0 and sig_range > 0:
        drift_abs = abs(float(np.mean(signal[:quarter]) - np.mean(signal[-quarter:])))
        drift_ratio = drift_abs / sig_range
        # Map to 0-1: ratio > 0.3 is bad, < 0.05 is good
        drift_score = round(float(np.clip(1.0 - drift_ratio / 0.3, 0.0, 1.0)), 3)

    # ===================== Component 3: PTB-XL noise metadata =====================
    meta = record.metadata
    noise_types = [
        "baseline_drift",
        "static_noise",
        "burst_noise",
        "electrodes_problems",
    ]
    noise_annotations = {}

    for noise_type in noise_types:
        raw = meta.get(noise_type, "")
        if not raw or (isinstance(raw, str) and not raw.strip()):
            noise_annotations[noise_type] = {"present": False, "affected_leads": []}
            continue
        parts = [p.strip().upper() for p in str(raw).split(",") if p.strip()]
        leads = [p for p in parts if p]
        noise_annotations[noise_type] = {
            "present": len(leads) > 0,
            "affected_leads": leads,
        }

    # Check noise on queried lead
    noise_on_lead = []
    for ntype, info in noise_annotations.items():
        if info["present"] and lead in info["affected_leads"]:
            noise_on_lead.append(ntype)

    # Metadata-based penalty: each noise type on this lead deducts from score
    metadata_score = round(max(0.0, 1.0 - len(noise_on_lead) * 0.25), 3)

    # ===================== Composite score (weighted) =====================
    # Weights: NeuroKit2 (0.4) + SNR (0.2) + Drift (0.15) + Metadata (0.25)
    weights = {}
    component_scores = {}

    if nk_score is not None:
        weights["neurokit2"] = 0.4
        component_scores["neurokit2"] = nk_score
    if snr_score is not None:
        weights["snr"] = 0.2
        component_scores["snr"] = snr_score
    if drift_score is not None:
        weights["baseline_stability"] = 0.15
        component_scores["baseline_stability"] = drift_score

    weights["metadata"] = 0.25
    component_scores["metadata"] = metadata_score

    # Normalize weights if some components are missing
    total_weight = sum(weights.values())
    if total_weight > 0:
        overall = sum(component_scores[k] * weights[k] for k in weights) / total_weight
        overall = round(overall, 2)
    else:
        overall = 0.5

    # Quality label
    if overall >= 0.8:
        quality_label = "good"
    elif overall >= 0.5:
        quality_label = "acceptable"
    else:
        quality_label = "poor"

    # ===================== Caveats =====================
    caveats = []
    if quality_label == "poor":
        caveats.append(
            "Signal quality is poor. Analysis results from other tools may be unreliable."
        )
    if noise_on_lead:
        caveats.append(
            f"PTB-XL metadata reports noise on lead {lead}: {', '.join(noise_on_lead)}."
        )
    if snr_score is not None and snr_score < 0.3:
        caveats.append("High-frequency noise level is elevated.")
    if drift_score is not None and drift_score < 0.3:
        caveats.append("Significant baseline drift detected.")

    # Consistency check: metadata says clean but signal analysis says noisy, or vice versa
    if metadata_score > 0.9 and overall < 0.5:
        caveats.append(
            "Note: PTB-XL metadata reports no noise but signal analysis indicates poor quality."
        )
    elif metadata_score < 0.5 and overall > 0.8:
        caveats.append(
            "Note: PTB-XL metadata reports noise but signal analysis indicates good quality."
        )

    return {
        "success": True,
        "data": {
            "overall_quality_score": overall,
            "quality_label": quality_label,
            "component_scores": component_scores,
            "signal_metrics": {
                "std": round(sig_std, 4),
                "range": round(sig_range, 4),
                "noise_std": round(noise_std, 4),
                "baseline_drift": round(drift_abs, 4),
            },
            "noise_annotations": noise_annotations,
            "noise_on_queried_lead": noise_on_lead,
            "caveats": caveats if caveats else None,
        },
        "metadata": {
            "lead_analyzed": lead,
            "sampling_rate": sr,
            "duration_seconds": round(record.duration_seconds, 1),
        },
    }


# ---------------------------------------------------------------------------
# Tool: analyze ECG morphology features per lead (ST segment, T wave, Q wave, R wave)
# ---------------------------------------------------------------------------
@register_tool(
    name="analyze_lead_morphology",
    description=(
        "Analyze ECG waveform morphology for each lead independently. "
        "Returns per-lead ST segment status (elevation / depression / normal), "
        "T wave polarity (upright / inverted / flat), Q wave presence and "
        "pathological assessment, and R wave amplitude. "
        "Use this tool when the user asks about form-related symptoms in specific "
        "leads, ST changes, T wave abnormalities, Q waves, or when a comprehensive "
        "per-lead morphological assessment is needed. "
        "Complements analyze_morphology (which reports interval durations) by "
        "providing amplitude and polarity features per lead."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to analyze.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": (
                "Sampling rate to load the signal. 500 recommended for morphology "
                "analysis. Default 500."
            ),
            "required": False,
            "default": 500,
        },
        "leads": {
            "type": "string",
            "description": (
                "Comma-separated list of leads to analyze (e.g. 'II,V1,V6'). "
                "Default: all 12 leads."
            ),
            "required": False,
            "default": "all",
        },
    },
)
def analyze_lead_morphology(
    record_id: str,
    sampling_rate: int = 500,
    leads: str = "all",
) -> dict[str, Any]:
    """Analyze per-lead ECG morphology: ST segment, T wave, Q wave, R wave."""

    record = _load_record(record_id, sampling_rate)

    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    sr = record.sampling_rate

    # --- Resolve requested leads ---
    if leads.strip().lower() == "all":
        target_leads = [l for l in ALL_LEADS if l in record.lead_names]
    else:
        target_leads = [l.strip().upper() for l in leads.split(",") if l.strip()]
        invalid = [l for l in target_leads if l not in record.lead_names]
        if invalid:
            return {
                "success": False,
                "error": (
                    f"Leads not available: {invalid}. Available: {record.lead_names}"
                ),
            }

    if not target_leads:
        return {"success": False, "error": "No valid leads to analyze."}

    # --- Analyze each lead ---
    per_lead_results: list[dict[str, Any]] = []
    leads_analyzed: list[str] = []
    leads_failed: list[str] = []

    for lead_name in target_leads:
        signal = record.get_lead(lead_name)
        lead_result = analyze_single_lead(signal, sr, lead_name)

        if "error" in lead_result:
            leads_failed.append(lead_name)
            logger.warning(
                "Lead %s analysis failed for record %s: %s",
                lead_name,
                record_id,
                lead_result["error"],
            )
        else:
            leads_analyzed.append(lead_name)

        per_lead_results.append(lead_result)

    # --- Build cross-lead summary ---
    st_elevation_leads = [
        r["lead"]
        for r in per_lead_results
        if r.get("st_segment", {}).get("status") == "elevation"
    ]
    st_depression_leads = [
        r["lead"]
        for r in per_lead_results
        if r.get("st_segment", {}).get("status") == "depression"
    ]
    t_inverted_leads = [
        r["lead"]
        for r in per_lead_results
        if r.get("t_wave", {}).get("status") == "inverted"
    ]
    t_flat_leads = [
        r["lead"]
        for r in per_lead_results
        if r.get("t_wave", {}).get("status") == "flat"
    ]
    q_pathological_leads = [
        r["lead"] for r in per_lead_results if r.get("q_wave", {}).get("pathological")
    ]
    q_present_leads = [
        r["lead"] for r in per_lead_results if r.get("q_wave", {}).get("present")
    ]

    # R-wave progression check (V1 → V6)
    r_progression = None
    precordial = ["V1", "V2", "V3", "V4", "V5", "V6"]
    r_amps_precordial: list[tuple[str, float]] = []
    for r in per_lead_results:
        if r.get("lead") in precordial and "r_wave" in r:
            r_amps_precordial.append((r["lead"], r["r_wave"]["mean_amplitude_mv"]))
    if len(r_amps_precordial) >= 4:
        # Sort by precordial order
        r_amps_precordial.sort(key=lambda x: precordial.index(x[0]))
        amps = [a for _, a in r_amps_precordial]
        # Check if R wave generally increases V1 → V4/V5
        if amps[-1] > amps[0] and all(
            amps[i + 1] >= amps[i] * 0.7 for i in range(len(amps) - 1)
        ):
            r_progression = "normal"
        else:
            r_progression = "poor"

    # --- Cross-lead interpretations ---
    interpretations: list[str] = []

    if st_elevation_leads:
        interpretations.append(
            f"ST elevation in {', '.join(st_elevation_leads)} — "
            "may indicate acute injury or pericarditis."
        )
    if st_depression_leads:
        interpretations.append(
            f"ST depression in {', '.join(st_depression_leads)} — "
            "may indicate ischemia or reciprocal changes."
        )
    if t_inverted_leads:
        interpretations.append(
            f"T-wave inversion in {', '.join(t_inverted_leads)} — "
            "may indicate ischemia, strain, or normal variant (depending on leads)."
        )
    if t_flat_leads:
        interpretations.append(
            f"T-wave flattening in {', '.join(t_flat_leads)} — "
            "non-specific finding, may indicate electrolyte imbalance or early ischemia."
        )
    if q_pathological_leads:
        interpretations.append(
            f"Pathological Q waves in {', '.join(q_pathological_leads)} — "
            "may indicate prior myocardial infarction."
        )
    if r_progression == "poor":
        interpretations.append(
            "Poor R-wave progression across precordial leads — "
            "may indicate anterior MI, LVH, or lead placement issue."
        )

    if not interpretations:
        interpretations.append(
            "No significant morphological abnormalities detected across analyzed leads."
        )

    summary: dict[str, Any] = {
        "interpretations": interpretations,
    }
    if st_elevation_leads:
        summary["st_elevation_leads"] = st_elevation_leads
    if st_depression_leads:
        summary["st_depression_leads"] = st_depression_leads
    if t_inverted_leads:
        summary["t_inverted_leads"] = t_inverted_leads
    if t_flat_leads:
        summary["t_flat_leads"] = t_flat_leads
    if q_pathological_leads:
        summary["q_pathological_leads"] = q_pathological_leads
    if q_present_leads:
        summary["q_present_leads"] = q_present_leads
    if r_progression:
        summary["r_wave_progression"] = r_progression

    return {
        "success": True,
        "data": {
            "per_lead": per_lead_results,
            "summary": summary,
        },
        "metadata": {
            "leads_analyzed": leads_analyzed,
            "leads_failed": leads_failed,
            "sampling_rate": sr,
            "duration_seconds": round(record.duration_seconds, 1),
        },
    }

# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------
@register_tool(
    name="assess_all_leads_quality",
    description=(
        "Scan all 12 leads for signal quality and noise detection in a single call. "
        "Returns per-lead quality scores (0-1), quality labels (GOOD/ACCEPTABLE/POOR), "
        "and detected noise types (baseline_drift, static_noise, burst_noise, "
        "electrodes_problems) for each lead. Also provides a cross-lead summary "
        "listing which leads have each noise type. "
        "Use this tool when the user asks about noise in the ECG, signal quality "
        "across leads, which leads have baseline drift or static noise, or any "
        "question about recording quality that may involve multiple leads. "
        "Preferred over assess_signal_quality when the question is not limited "
        "to a single specific lead."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to assess.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Sampling rate. Default 100.",
            "required": False,
            "default": 100,
        },
    },
)
def assess_all_leads_quality(
    record_id: str,
    sampling_rate: int = 100,
) -> dict[str, Any]:
    """Assess signal quality and noise across all 12 leads."""
 
    record = _load_record(record_id, sampling_rate)
 
    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}
 
    sr = record.sampling_rate
 
    # Parse noise metadata once
    noise_meta = parse_noise_metadata(record.metadata)
 
    # --- Scan all leads ---
    per_lead_results: list[dict[str, Any]] = []
    target_leads = [l for l in ALL_LEADS if l in record.lead_names]
 
    for lead_name in target_leads:
        signal = record.get_lead(lead_name)
        lead_result = assess_single_lead_quality(signal, sr, lead_name, noise_meta)
        per_lead_results.append(lead_result)
 
    # --- Cross-lead summary ---
    # Noise type → list of affected leads
    noise_by_type: dict[str, list[str]] = {nt: [] for nt in NOISE_TYPES}
    for r in per_lead_results:
        for nt in r.get("noise_types", []):
            noise_by_type[nt].append(r["lead"])
 
    # Filter to only types that have affected leads
    noise_by_type_filtered = {
        nt: leads for nt, leads in noise_by_type.items() if leads
    }
 
    # Leads with any noise
    noisy_leads = sorted(set(
        lead for leads in noise_by_type_filtered.values() for lead in leads
    ))
 
    # Clean leads (no noise detected)
    clean_leads = [r["lead"] for r in per_lead_results if not r.get("has_noise")]
 
    # Quality distribution
    quality_dist: dict[str, list[str]] = {"GOOD": [], "ACCEPTABLE": [], "POOR": []}
    for r in per_lead_results:
        label = r.get("quality_label", "POOR")
        quality_dist[label].append(r["lead"])
 
    # Average quality
    avg_quality = round(
        float(np.mean([r["quality_score"] for r in per_lead_results])), 2
    ) if per_lead_results else 0.0
 
    # --- Interpretations ---
    interpretations: list[str] = []
 
    if noise_by_type_filtered:
        for nt, leads in sorted(noise_by_type_filtered.items()):
            readable = nt.replace("_", " ")
            interpretations.append(
                f"{readable.capitalize()} detected in: {', '.join(leads)}"
            )
    else:
        interpretations.append("No noise detected in any lead from metadata annotations.")
 
    poor_leads = quality_dist.get("POOR", [])
    if poor_leads:
        interpretations.append(
            f"Poor signal quality in: {', '.join(poor_leads)}. "
            "Analysis results from these leads may be unreliable."
        )
 
    if avg_quality >= 0.8:
        interpretations.append("Overall recording quality is GOOD.")
    elif avg_quality >= 0.5:
        interpretations.append("Overall recording quality is ACCEPTABLE.")
    else:
        interpretations.append("Overall recording quality is POOR.")
 
    return {
        "success": True,
        "data": {
            "per_lead": per_lead_results,
            "summary": {
                "avg_quality_score": avg_quality,
                "noise_by_type": noise_by_type_filtered,
                "noisy_leads": noisy_leads,
                "clean_leads": clean_leads,
                "quality_distribution": {
                    k: v for k, v in quality_dist.items() if v
                },
                "interpretations": interpretations,
            },
        },
        "metadata": {
            "leads_scanned": target_leads,
            "sampling_rate": sr,
            "duration_seconds": round(record.duration_seconds, 1),
        },
    }


# ---------------------------------------------------------------------------
# Tool: ECGFounder-based diagnosis support
# ---------------------------------------------------------------------------
@register_tool(
    name="ecg_diagnosis",
    description=(
        "Run ECGFounder, a pretrained ECG diagnosis model, on a PTB-XL record. "
        "Returns SCP-style mapped diagnostic findings grouped into present (>0.5) and "
        "uncertain (0.35-0.5) evidence, plus a coverage note. Use this for diagnosis-oriented questions "
        "such as arrhythmias, infarction, hypertrophy, bundle branch blocks, pacemaker "
        "rhythms, and other SCP-code-like diagnostic findings."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "ECG record ID to analyze.",
            "required": True,
        },
        "sampling_rate": {
            "type": "integer",
            "description": "Signal loading rate. 500 preferred; 100 will be resampled. Default 500.",
            "required": False,
            "default": 500,
        },
        "mode": {
            "type": "string",
            "description": "Inference mode. Use `12_lead` for the default full-recording model, or `single_lead` to run ECGFounder 1-lead mode on Lead I only.",
            "required": False,
            "default": "12_lead",
        },
        "top_k_predictions": {
            "type": "integer",
            "description": "Legacy parameter retained for compatibility. Raw ECGFounder label lists are not returned.",
            "required": False,
            "default": 10,
        },
        "top_k_findings": {
            "type": "integer",
            "description": "Maximum number of mapped diagnostic concepts to return across present and uncertain groups.",
            "required": False,
            "default": 15,
        },
        "min_probability": {
            "type": "number",
            "description": "Legacy lower bound for displaying mapped findings. Effective minimum is 0.35.",
            "required": False,
            "default": 0.1,
        },
    },
)
def ecg_diagnosis(
    record_id: str,
    sampling_rate: int = 500,
    mode: ECGDiagnosisMode = "12_lead",
    top_k_predictions: int = 10,
    top_k_findings: int = 15,
    min_probability: float = 0.1,
) -> dict[str, Any]:
    """Run ECGFounder and map its outputs to SCP-style diagnostic concepts."""
    if not is_tool_enabled("ecg_diagnosis"):
        return {"success": False, "error": "Tool 'ecg_diagnosis' is disabled."}

    record = _load_record(record_id, sampling_rate)
    if record is None:
        return {"success": False, "error": f"Record '{record_id}' not found."}

    try:
        normalized_mode = _normalize_ecg_diagnosis_mode(mode)
        if normalized_mode == "12_lead" and int(record.num_leads) == 1:
            normalized_mode = "single_lead"
        signal_array, lead_used = _prepare_ecgfounder_signal(record, mode=normalized_mode)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    runtime = load_ecgfounder_runtime(normalized_mode)

    try:
        signal_array = resample_signal_to_length(signal_array, target_length=5000)
    except Exception as exc:
        return {"success": False, "error": f"Failed to resample ECG signal: {exc}"}
    signal_array = z_score_normalize(signal_array)

    torch = runtime["torch"]
    device = runtime["device"]
    model = runtime["model"]
    tasks = runtime["tasks"]

    with torch.no_grad():
        input_tensor = torch.from_numpy(signal_array).unsqueeze(0).to(device)
        logits_tensor = model(input_tensor)
        probabilities_tensor = torch.sigmoid(logits_tensor)

    logits = logits_tensor.detach().cpu().numpy()[0].astype(float).tolist()
    probabilities = probabilities_tensor.detach().cpu().numpy()[0].astype(float).tolist()

    label_scores = {
        label: {"logit": logits[idx], "probability": probabilities[idx]}
        for idx, label in enumerate(tasks)
    }
    effective_min_probability = max(float(min_probability), 0.35)
    scp_relevant_findings = build_scp_relevant_findings(
        label_scores,
        min_probability=effective_min_probability,
        top_k=int(top_k_findings),
    )

    present_findings = []
    uncertain_findings = []
    for item in scp_relevant_findings:
        finding = _format_ecg_diagnosis_finding(item)
        probability = finding["probability"]
        if probability > 0.5:
            present_findings.append(finding)
        elif probability >= effective_min_probability:
            uncertain_findings.append(finding)

    coverage_note = _build_ecg_diagnosis_coverage_note(
        present_findings=present_findings,
        uncertain_findings=uncertain_findings,
    )

    return {
        "success": True,
        "data": {
            "model_name": "ECGFounder-1lead" if normalized_mode == "single_lead" else "ECGFounder-12lead",
            "scp_relevant_findings": {
                "present": present_findings,
                "uncertain": uncertain_findings,
            },
            "coverage_note": coverage_note,
        },
        "metadata": {
            "record_id": str(record_id),
            "inference_mode": normalized_mode,
            "lead_used": lead_used,
            "source": getattr(record, "metadata", {}).get("source", _resolve_record_reference(record_id)[0]),
            "lead_names": list(getattr(record, "lead_names", [])),
            "num_leads": int(record.num_leads),
            "sampling_rate_loaded": int(record.sampling_rate),
            "duration_seconds": round(float(record.duration_seconds), 1),
            "input_signal_shape": [int(signal_array.shape[0]), int(signal_array.shape[1])],
            "model_input_shape": [1, int(signal_array.shape[0]), int(signal_array.shape[1])],
            "num_task_labels": int(len(tasks)),
            "present_threshold": 0.5,
            "uncertain_threshold_min": round(float(effective_min_probability), 4),
            "checkpoint_path": runtime["checkpoint_path"],
            "missing_keys": runtime["missing_keys"],
            "unexpected_keys": runtime["unexpected_keys"],
        },
    }













