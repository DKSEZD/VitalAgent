"""Reactive tool for replaying proactive guideline-grounded rules."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agent.encoder.ecg_signal_processor import ECGSignalProcessor, VitalParams
from agent.proactive.intervention_decision import (
    InterventionResult,
    Modality,
    RuleBasedJudge,
    Urgency,
)
from agent.proactive.rhythm_classifier import classify_rhythm
from agent.proactive.tracker import HealthStateTracker
from agent.tools.datasets import icentia11k as icentia11k_tools
from agent.tools.datasets import ppg_dalia as ppg_dalia_tools
from agent.tools.datasets import wesad as wesad_tools
from agent.tools.ppg.window_analysis import as_float_array, get_nested, slice_by_time
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

SUB_WINDOW_SECONDS = 10.0
MIN_SUSTAINED_RUNS = 3  # 30 s at SUB_WINDOW_SECONDS=10; AF episode minimum per AF-2023.
HIGH_CONFIDENCE_QUALITY_THRESHOLD = 0.6
HIGH_CONFIDENCE_AF_QUALITY_RATIO = 0.5  # require >=50% of AF-labeled windows to be high-quality
SUB_WINDOWS_PER_30S = 3
MAX_REPLAY_SECONDS = 600.0
SUPPORTED_DATASETS = {"icentia11k", "wesad", "ppg_dalia", "afppgecg"}
PHASE2_CAVEAT = (
    "long_af_episode requires >=24h sustained AF and cannot fire from a fresh "
    "tracker within the 600s replay cap. Pass via include_phase2_rules=True is "
    "honored but the rule will be inert under current scope."
)
_URGENCY_RANK = {
    Urgency.NONE: 0,
    Urgency.LOW: 1,
    Urgency.MEDIUM: 2,
    Urgency.HIGH: 3,
    Urgency.CRITICAL: 4,
}


@register_tool(
    name="evaluate_proactive_rules",
    description=(
        "Replay the proactive engine's guideline-grounded rule layer over a "
        "raw signal window from a supported dataset. Slices the requested "
        "time range into 10-second sub-windows, accumulates the HealthStateTracker, "
        "and evaluates the RuleBasedJudge rule set at each sub-window. "
        "Returns which clinical rules fired, the highest urgency tier, per-rule "
        "guideline citations, and a HR + rhythm time-series trace. Useful for "
        "questions about whether a clinical alert should fire, sustained patterns "
        "(e.g. sustained tachycardia), or guideline-grounded judgement. "
        "Supports icentia11k, wesad (chest ECG), ppg_dalia (chest ECG), "
        "and afppgecg (aligned ECG)."
    ),
    parameters={
        "dataset": {
            "type": "string",
            "description": "One of icentia11k|wesad|ppg_dalia|afppgecg.",
            "required": True,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient/subject identifier from the dataset locator.",
            "required": True,
        },
        "window_start_s": {
            "type": "number",
            "description": "Range start in seconds.",
            "required": True,
        },
        "window_end_s": {
            "type": "number",
            "description": (
                "Range end in seconds. Must be >= start + 10s. Capped internally "
                "at 600s (10 min) to bound cost."
            ),
            "required": True,
        },
        "segment_id": {
            "type": "string",
            "description": "Icentia segment ID; ignored for other datasets.",
            "required": False,
        },
        "recording_id": {
            "type": "string",
            "description": "Alias for segment_id on Icentia; ignored for other datasets.",
            "required": False,
        },
        "modality": {
            "type": "string",
            "description": "'ecg' or 'ppg'. Defaults to 'ecg' for these datasets.",
            "required": False,
        },
        "include_phase2_rules": {
            "type": "boolean",
            "description": "Include PHASE2 staged rules (e.g. long_af_episode).",
            "default": False,
        },
    },
)
def evaluate_proactive_rules(
    dataset: str,
    patient_id: str,
    window_start_s: float,
    window_end_s: float,
    segment_id: str | None = None,
    recording_id: str | None = None,
    modality: str | None = None,
    include_phase2_rules: bool = False,
) -> dict[str, Any]:
    try:
        dataset_name = str(dataset)
        if dataset_name not in SUPPORTED_DATASETS:
            return _error_response(
                dataset_name,
                patient_id,
                f"Unsupported dataset for proactive rule replay: {dataset_name!r}",
            )

        start_s = float(window_start_s)
        requested_end_s = float(window_end_s)
        if requested_end_s - start_s < SUB_WINDOW_SECONDS:
            return _error_response(
                dataset_name,
                patient_id,
                "window_end_s - window_start_s must be at least 10 seconds.",
            )

        range_clamped = requested_end_s - start_s > MAX_REPLAY_SECONDS
        end_s = min(requested_end_s, start_s + MAX_REPLAY_SECONDS)
        if range_clamped:
            logger.warning(
                "Clamped evaluate_proactive_rules replay window from %.1fs to %.1fs.",
                requested_end_s - start_s,
                MAX_REPLAY_SECONDS,
            )

        signal, fs_hz, source_metadata = _load_ecg_signal_window(
            dataset=dataset_name,
            patient_id=str(patient_id),
            start_s=start_s,
            end_s=end_s,
            segment_id=segment_id,
            recording_id=recording_id,
        )
        qrs_index_in_window = source_metadata.pop("_qrs_index_in_window", None)
        if signal is None or signal.size == 0:
            return _error_response(dataset_name, patient_id, "No ECG samples were available.")
        if fs_hz <= 0:
            return _error_response(dataset_name, patient_id, f"Invalid sampling rate: {fs_hz!r}.")

        replay = _replay_rules(
            signal=signal,
            fs_hz=float(fs_hz),
            modality=_resolve_modality(modality),
            include_phase2_rules=bool(include_phase2_rules),
            qrs_index_in_window=qrs_index_in_window,
        )

        duration_s = end_s - start_s
        metadata = {
            "dataset": dataset_name,
            "patient_id": str(patient_id),
            "segment_id": source_metadata.get("segment_id") or segment_id or recording_id,
            "recording_id": source_metadata.get("recording_id") or recording_id or segment_id,
            "window_start_s": start_s,
            "window_end_s": end_s,
            "requested_window_end_s": requested_end_s,
            "duration_s": duration_s,
            "sampling_rate": float(fs_hz),
            "sub_window_seconds": SUB_WINDOW_SECONDS,
            "range_clamped": range_clamped,
            **source_metadata,
        }
        if include_phase2_rules:
            metadata["phase2_caveat"] = PHASE2_CAVEAT
        if replay["valid_vital_window_count"] == 0:
            metadata["quality_warning"] = (
                "No sub-window produced usable heart-rate vitals; rule replay returned no alert."
            )
        elif replay["invalid_vital_window_count"] > 0:
            metadata["quality_warning"] = (
                f"{replay['invalid_vital_window_count']} sub-window(s) did not produce usable "
                "heart-rate vitals and were skipped."
            )

        return {
            "success": True,
            "data": replay["data"],
            "metadata": metadata,
        }
    except Exception as exc:
        logger.exception("evaluate_proactive_rules failed")
        return _error_response(str(dataset), str(patient_id), str(exc))


def _load_ecg_signal_window(
    *,
    dataset: str,
    patient_id: str,
    start_s: float,
    end_s: float,
    segment_id: str | None,
    recording_id: str | None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if dataset == "icentia11k":
        return _load_icentia11k_signal(patient_id, start_s, end_s, segment_id, recording_id)
    if dataset == "wesad":
        return _load_wesad_chest_ecg(patient_id, start_s, end_s)
    if dataset == "ppg_dalia":
        return _load_ppg_dalia_chest_ecg(patient_id, start_s, end_s)
    if dataset == "afppgecg":
        return _load_afppgecg_signal(patient_id, start_s, end_s)
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _load_icentia11k_signal(
    patient_id: str,
    start_s: float,
    end_s: float,
    segment_id: str | None,
    recording_id: str | None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    segment = icentia11k_tools._normalize_segment_id(segment_id, recording_id)  # noqa: SLF001
    loader = icentia11k_tools._loader()  # noqa: SLF001
    header = loader.read_segment_header(patient_id, segment)
    start_sample = int(round(start_s * header.fs))
    end_sample = int(round(end_s * header.fs))
    if end_sample <= start_sample:
        raise ValueError("No Icentia11k samples available for requested window.")
    raw = loader.read_segment(patient_id, segment, sampfrom=start_sample, sampto=end_sample)
    return (
        _flatten_ecg(raw.signal),
        float(raw.fs),
        {
            "segment_id": raw.segment_id,
            "recording_id": raw.segment_id,
            "sample_start": raw.sample_start,
            "sample_end": raw.sample_end,
            "dataset_source": raw.metadata.get("source"),
        },
    )


def _load_wesad_chest_ecg(
    patient_id: str,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    path = wesad_tools._resolve_pickle_path(  # noqa: SLF001
        pickle_path=None,
        subject_id=None,
        patient_id=patient_id,
        dataset_root=None,
    )
    payload = wesad_tools._load_wesad_pickle(path)  # noqa: SLF001
    ecg = as_float_array(get_nested(payload, ("signal", "chest", "ECG")))
    if ecg is None or ecg.size == 0:
        raise ValueError("WESAD chest ECG channel is not available.")
    window = slice_by_time(
        ecg,
        fs_hz=wesad_tools.WESAD_CHEST_FS_HZ,
        start_s=start_s,
        end_s=end_s,
    )
    return (
        _flatten_ecg(window),
        float(wesad_tools.WESAD_CHEST_FS_HZ),
        {"source_path": str(path), "subject_id": str(payload.get("subject") or path.stem)},
    )


def _load_ppg_dalia_chest_ecg(
    patient_id: str,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    path = ppg_dalia_tools._resolve_pickle_path(  # noqa: SLF001
        pickle_path=None,
        subject_id=None,
        patient_id=patient_id,
        dataset_root=None,
    )
    payload = ppg_dalia_tools._load_ppg_dalia_pickle(path)  # noqa: SLF001
    ecg = as_float_array(get_nested(payload, ("signal", "chest", "ECG")))
    if ecg is None or ecg.size == 0:
        raise ValueError("PPG-DaLiA chest ECG channel is not available.")
    window = slice_by_time(
        ecg,
        fs_hz=ppg_dalia_tools.PPG_DALIA_CHEST_FS_HZ,
        start_s=start_s,
        end_s=end_s,
    )
    return (
        _flatten_ecg(window),
        float(ppg_dalia_tools.PPG_DALIA_CHEST_FS_HZ),
        {"source_path": str(path), "subject_id": str(payload.get("subject") or path.stem)},
    )


def _load_afppgecg_signal(
    patient_id: str,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    from agent.data.ppg_loader import PPGLoader

    loader = PPGLoader()
    header = loader.read_ecg_record(patient_id, include_signal=False)
    fs_hz = float(header.fs)
    if fs_hz <= 0:
        raise ValueError(f"Invalid AF-PPG-ECG sampling rate: {fs_hz!r}.")

    start_sample = max(0, int(round(start_s * fs_hz)))
    end_sample = min(int(round(end_s * fs_hz)), int(header.n_samples))
    if end_sample <= start_sample:
        raise ValueError("No AF-PPG-ECG aligned ECG samples available for requested window.")

    record = loader.read_ecg_record(
        patient_id,
        include_signal=True,
        sampfrom=start_sample,
        sampto=end_sample,
    )
    ecg = as_float_array(record.signal)
    if ecg is None or ecg.size == 0:
        raise ValueError("AF-PPG-ECG aligned ECG signal is not available.")
    window = slice_by_time(
        ecg,
        fs_hz=fs_hz,
        start_s=0.0,
        end_s=(end_sample - start_sample) / fs_hz,
    )
    source_path = (
        getattr(record, "path", None)
        or record.metadata.get("record_path")
        or header.metadata.get("record_path")
    )
    qrs = np.asarray(record.qrs_index, dtype=int).reshape(-1)
    qrs_window = qrs[(qrs >= start_sample) & (qrs < end_sample)] - start_sample
    return (
        _flatten_ecg(window),
        fs_hz,
        {
            "source_path": str(source_path) if source_path is not None else None,
            "patient_id": str(record.patient_id),
            "ecg_channel": "aligned_ecg",
            "sample_start": start_sample,
            "sample_end": end_sample,
            "_qrs_index_in_window": qrs_window,
        },
    )


def _replay_rules(
    *,
    signal: np.ndarray,
    fs_hz: float,
    modality: Modality,
    include_phase2_rules: bool,
    qrs_index_in_window: np.ndarray | None = None,
) -> dict[str, Any]:
    tracker = HealthStateTracker()
    decision = RuleBasedJudge(
        modality=modality,
        include_phase2=include_phase2_rules,
    )
    processor = ECGSignalProcessor(sampling_rate=int(round(fs_hz)))
    samples_per_window = int(round(SUB_WINDOW_SECONDS * fs_hz))
    n_sub_windows = int(signal.size // samples_per_window) if samples_per_window > 0 else 0

    hr_series: list[float | None] = []
    rhythm_series: list[str] = []
    signal_quality_series: list[float | None] = []
    per_window_triggered_rules: list[list[str]] = []
    per_window_urgency: list[str] = []
    first_trigger_offset_s: dict[str, float] = {}
    first_trigger_reasons: dict[str, str] = {}
    triggered_union: list[str] = []
    highest_urgency = Urgency.NONE
    final_state = tracker.get_current_state()
    valid_count = 0
    invalid_count = 0

    for index in range(n_sub_windows):
        start = index * samples_per_window
        end = start + samples_per_window
        window = signal[start:end]
        vitals = _extract_vitals(processor, window, fs_hz)
        qrs_hr = _qrs_hr_bpm(
            qrs_index_in_window,
            start_sample=start,
            end_sample=end,
            fs_hz=fs_hz,
            min_beats=2,
        )
        if qrs_hr is not None and vitals is not None:
            vitals.hr_bpm = qrs_hr
        quality = _vital_quality(vitals)
        hr = vitals.hr_bpm if vitals is not None else None
        if hr is None:
            hr_series.append(None)
            rhythm_series.append("Unknown")
            signal_quality_series.append(quality)
            per_window_triggered_rules.append([])
            per_window_urgency.append(Urgency.NONE.value)
            invalid_count += 1
            continue

        valid_count += 1
        rhythm = classify_rhythm(vitals.rr_intervals_ms)
        tracker.update_vitals(
            vitals.to_dict(),
            rhythm_class=rhythm,
            window_duration_s=SUB_WINDOW_SECONDS,
        )
        state = tracker.get_current_state()
        state["window_meta"] = {"window_duration_s": SUB_WINDOW_SECONDS}
        final_state = state
        result = decision.evaluate(state)

        hr_series.append(float(hr))
        rhythm_series.append(rhythm)
        signal_quality_series.append(quality)
        rules = list(result.triggered_rules or [])
        per_window_triggered_rules.append(rules)
        per_window_urgency.append(result.urgency.value)
        if _URGENCY_RANK[result.urgency] > _URGENCY_RANK[highest_urgency]:
            highest_urgency = result.urgency

        for rule_id in rules:
            if rule_id not in triggered_union:
                triggered_union.append(rule_id)
                first_trigger_offset_s[rule_id] = index * SUB_WINDOW_SECONDS
                first_trigger_reasons[rule_id] = _rule_reason(decision, rule_id, state, result)

    reason = " ".join(first_trigger_reasons[rule_id] for rule_id in triggered_union)
    if not reason:
        reason = "No proactive guideline rules fired during the replay window."
    rule_descriptions = [
        item
        for item in decision.describe_rules()
        if item.get("rule_id") in set(triggered_union)
    ]
    rhythm_summary = _rhythm_summary(rhythm_series, signal_quality_series)
    hr_30s_series = (
        _hr_30s_series_from_qrs(
            qrs_index_in_window,
            n_sub_windows=n_sub_windows,
            samples_per_window=samples_per_window,
            fs_hz=fs_hz,
        )
        if qrs_index_in_window is not None
        else _hr_30s_series_from_10s(hr_series, n_sub_windows=n_sub_windows)
    )
    hr_30s_values = [value for value in hr_30s_series if value is not None]

    return {
        "valid_vital_window_count": valid_count,
        "invalid_vital_window_count": invalid_count,
        "data": {
            "rule_outcome": {
                "intervene": bool(triggered_union),
                "urgency": highest_urgency.value,
                "reason": reason,
                "triggered_rules": triggered_union,
                "first_trigger_offset_s": first_trigger_offset_s,
            },
            "rule_descriptions": rule_descriptions,
            "rhythm_summary": rhythm_summary,
            "trace": {
                "n_sub_windows": n_sub_windows,
                "valid_vital_window_count": valid_count,
                "invalid_vital_window_count": invalid_count,
                "hr_bpm_series": hr_series,
                "hr_bpm_30s_series": hr_30s_series,
                "max_hr_30s_bpm": max(hr_30s_values) if hr_30s_values else None,
                "min_hr_30s_bpm": min(hr_30s_values) if hr_30s_values else None,
                "sub_window_30s_seconds": SUB_WINDOW_SECONDS * SUB_WINDOWS_PER_30S,
                "rhythm_series": rhythm_series,
                "signal_quality_series": signal_quality_series,
                "per_window_triggered_rules": per_window_triggered_rules,
                "per_window_urgency": per_window_urgency,
                "final_short_window": dict(final_state.get("short") or {}),
            },
            "modality": modality.value,
            "phase2_rules_included": include_phase2_rules,
        },
    }


def _rhythm_summary(
    rhythm_series: list[str],
    signal_quality_series: list[float | None] | None = None,
) -> dict[str, Any]:
    labels = ("N", "AF", "Other", "Unknown")
    counts = {label: 0 for label in labels}
    for raw_label in rhythm_series:
        label = raw_label if raw_label in counts else "Unknown"
        counts[label] += 1

    total = len(rhythm_series)
    fractions = {
        label: (counts[label] / total if total else 0.0)
        for label in labels
    }
    max_runs = {label: 0 for label in labels}
    raw_transition_count = 0
    previous: str | None = None
    current_run = 0
    for raw_label in rhythm_series:
        label = raw_label if raw_label in counts else "Unknown"
        if previous is None:
            current_run = 1
        elif label == previous:
            current_run += 1
        else:
            raw_transition_count += 1
            current_run = 1
        max_runs[label] = max(max_runs[label], current_run)
        previous = label

    high_confidence_af_count = 0
    high_confidence_max_consecutive_af = 0
    quality_gated_window_count = 0
    high_confidence_current_run = 0
    quality_values = signal_quality_series or []
    for raw_label, raw_quality in zip(rhythm_series, quality_values, strict=False):
        quality = _coerce_quality(raw_quality)
        is_high_confidence = (
            quality is not None and quality >= HIGH_CONFIDENCE_QUALITY_THRESHOLD
        )
        if is_high_confidence:
            quality_gated_window_count += 1
        if raw_label == "AF" and is_high_confidence:
            high_confidence_af_count += 1
            high_confidence_current_run += 1
            high_confidence_max_consecutive_af = max(
                high_confidence_max_consecutive_af,
                high_confidence_current_run,
            )
        else:
            high_confidence_current_run = 0

    non_unknown_counts = {label: counts[label] for label in ("AF", "N", "Other")}
    valid_label_count = sum(non_unknown_counts.values())
    if any(non_unknown_counts.values()):
        dominant_class = max(
            ("AF", "N", "Other"),
            key=lambda label: (non_unknown_counts[label], {"AF": 2, "N": 1, "Other": 0}[label]),
        )
    else:
        dominant_class = None

    # Quality-aware AF confidence: a sustained AF episode is "high-confidence"
    # when the raw sustained run is present AND at least half of the AF-labeled
    # sub-windows fall in high-quality windows. This correctly accepts true
    # continuous AF episodes (where the classifier flickers AF/Other within
    # mostly-good signal) while rejecting classifier false-positives that
    # concentrate on artefact-noisy low-quality windows.
    quality_gate_coverage = (
        quality_gated_window_count / valid_label_count
        if valid_label_count
        else 0.0
    )
    af_label_count = counts["AF"]
    high_confidence_af_quality_ratio = (
        high_confidence_af_count / af_label_count
        if af_label_count > 0
        else 0.0
    )

    max_seconds = {
        label: max_runs[label] * SUB_WINDOW_SECONDS
        for label in labels
    }
    sustained_classes = [
        label for label in ("N", "AF", "Other")
        if max_runs[label] >= MIN_SUSTAINED_RUNS
    ]
    dominant_fraction_of_valid = (
        counts[dominant_class] / valid_label_count
        if dominant_class is not None and valid_label_count
        else 0.0
    )
    stability_label, stability_reason = _rhythm_stability_interpretation(
        sustained_classes=sustained_classes,
        dominant_class=dominant_class,
        dominant_fraction_of_valid=dominant_fraction_of_valid,
        valid_label_count=valid_label_count,
    )
    return {
        "dominant_class": dominant_class,
        "valid_label_count": valid_label_count,
        "dominant_fraction_of_valid": dominant_fraction_of_valid,
        "label_counts": counts,
        "label_fractions": fractions,
        "max_consecutive_runs": max_runs,
        "max_consecutive_seconds": max_seconds,
        "raw_transition_count": raw_transition_count,
        "quality_gated_window_count": quality_gated_window_count,
        "high_confidence_af_sub_window_count": high_confidence_af_count,
        "high_confidence_max_consecutive_af": high_confidence_max_consecutive_af,
        "high_confidence_sustained_af_episode_detected": (
            max_runs["AF"] >= MIN_SUSTAINED_RUNS
            and high_confidence_af_quality_ratio >= HIGH_CONFIDENCE_AF_QUALITY_RATIO
        ),
        "high_confidence_af_quality_ratio": high_confidence_af_quality_ratio,
        "quality_gate_coverage": quality_gate_coverage,
        "sustained_episode_classes": sustained_classes,
        "stability_label": stability_label,
        "stability_reason": stability_reason,
        "sustained_af_episode_detected": max_runs["AF"] >= MIN_SUSTAINED_RUNS,
        "sustained_n_episode_detected": max_runs["N"] >= MIN_SUSTAINED_RUNS,
        "sustained_other_episode_detected": max_runs["Other"] >= MIN_SUSTAINED_RUNS,
    }


def _rhythm_stability_interpretation(
    *,
    sustained_classes: list[str],
    dominant_class: str | None,
    dominant_fraction_of_valid: float,
    valid_label_count: int,
) -> tuple[str, str]:
    if valid_label_count == 0 or dominant_class is None:
        return "unknown", "No usable rhythm classifications were available."

    sustained_set = set(sustained_classes)
    if "N" in sustained_set and ({"AF", "Other"} & sustained_set):
        return (
            "frequent_changes",
            "Both normal and irregular rhythm classes have sustained runs.",
        )

    if sustained_set <= {"AF", "Other"} and sustained_set:
        return (
            "stable",
            "Sustained irregular labels are treated as one AF-like rhythm family.",
        )

    if sustained_set == {"N"}:
        return "stable", "Only normal rhythm has a sustained run."

    if dominant_fraction_of_valid >= 0.5:
        return (
            "stable",
            "One dominant rhythm class accounts for at least half of valid windows.",
        )

    return (
        "occasional_changes",
        "No rhythm family is dominant enough to call stable, but sustained cross-family switching was not detected.",
    )


@register_tool(
    name="analyze_afppgecg_rhythm_context",
    description=(
        "Adaptive-scope rhythm analysis for afppgecg. Returns an anchor "
        "summary for the locator window AND an extended-context summary "
        "sampled across the full aligned-ECG recording. Extended fields are "
        "signal-classifier-derived (from RR irregularity over sampled 30s "
        "chunks) — NOT dataset clinical AF annotations. Use for patient-level "
        "rhythm/AF questions on afppgecg: 'lately', 'how often', 'rhythm "
        "variability over the recording', or AF burden patterns. For locator-"
        "bound 'in this 5-min window' questions, prefer evaluate_proactive_rules."
    ),
    parameters={
        "patient_id": {"type": "string", "required": True},
        "anchor_window_start_s": {"type": "number", "required": True},
        "anchor_window_end_s": {"type": "number", "required": True},
        "scope": {
            "type": "string",
            "description": (
                "'locator_window' = anchor only; 'sampled_recording' = anchor + "
                "sampled chunks across full recording."
            ),
            "default": "sampled_recording",
        },
        "n_sampled_chunks": {
            "type": "integer",
            "description": (
                "Number of equally-spaced 30s chunks sampled across the recording "
                "when scope=sampled_recording. Default 60. Min 10, max 200."
            ),
            "default": 60,
        },
    },
)
def analyze_afppgecg_rhythm_context(
    patient_id: str,
    anchor_window_start_s: float | None = None,
    anchor_window_end_s: float | None = None,
    scope: str = "sampled_recording",
    n_sampled_chunks: int = 60,
    dataset: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
) -> dict[str, Any]:
    if dataset is not None and str(dataset) != "afppgecg":
        return {
            "success": False,
            "error": f"Unsupported dataset for AF-PPG-ECG rhythm context: {dataset!r}",
        }
    try:
        from agent.data.ppg_loader import PPGLoader

        start_s = _resolve_anchor_time(anchor_window_start_s, window_start_s, "start")
        end_s = _resolve_anchor_time(anchor_window_end_s, window_end_s, "end")
        if end_s <= start_s:
            return {"success": False, "error": "anchor_window_end_s must be greater than start."}

        loader = PPGLoader()
        record = loader.read_ecg_record(patient_id, include_signal=False)
        fs_hz = float(record.fs)
        qrs_index = np.asarray(record.qrs_index, dtype=float).reshape(-1)
        if fs_hz <= 0 or record.n_samples <= 0:
            return {"success": False, "error": "Invalid AF-PPG-ECG ECG header."}
        if qrs_index.size == 0:
            return {"success": False, "error": "No QRS positions are available."}

        total_recording_seconds = float(record.n_samples) / fs_hz
        anchor_summary = _afppgecg_anchor_summary(
            qrs_index=qrs_index,
            fs_hz=fs_hz,
            start_s=start_s,
            end_s=end_s,
        )

        normalized_scope = str(scope or "sampled_recording")
        extended_summary = None
        if normalized_scope == "sampled_recording":
            extended_summary = _afppgecg_extended_summary(
                qrs_index=qrs_index,
                fs_hz=fs_hz,
                total_recording_seconds=total_recording_seconds,
                n_sampled_chunks=int(n_sampled_chunks),
            )
        elif normalized_scope != "locator_window":
            return {
                "success": False,
                "error": "scope must be 'locator_window' or 'sampled_recording'.",
            }

        return {
            "success": True,
            "data": {
                "anchor_window_summary": anchor_summary,
                "extended_context_summary": extended_summary,
            },
            "metadata": {
                "dataset": "afppgecg",
                "patient_id": str(record.patient_id),
                "qrs_source": "afppgecg_HDF5_QRSindex",
                "sampling_rate": fs_hz,
                "total_recording_seconds": total_recording_seconds,
                "uses_af_annotation": False,
            },
        }
    except Exception as exc:
        logger.exception("analyze_afppgecg_rhythm_context failed")
        return {"success": False, "error": str(exc)}


def _resolve_anchor_time(primary: float | None, alias: float | None, name: str) -> float:
    value = primary if primary is not None else alias
    if value is None:
        raise ValueError(f"anchor_window_{name}_s is required.")
    return float(value)


def _afppgecg_anchor_summary(
    *,
    qrs_index: np.ndarray,
    fs_hz: float,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    start_sample = start_s * fs_hz
    end_sample = end_s * fs_hz
    qrs = qrs_index[(qrs_index >= start_sample) & (qrs_index < end_sample)]
    rr_ms = _rr_intervals_ms_from_qrs(qrs, fs_hz=fs_hz)
    rhythm_label = classify_rhythm(rr_ms.tolist()) if rr_ms.size >= 4 else "Unknown"
    return {
        "scope": "locator_window",
        "window_start_s": start_s,
        "window_end_s": end_s,
        "n_qrs_beats": int(qrs.size),
        "rhythm_label": rhythm_label,
        "irregularity_metrics": _rr_irregularity_metrics(rr_ms),
    }


def _afppgecg_extended_summary(
    *,
    qrs_index: np.ndarray,
    fs_hz: float,
    total_recording_seconds: float,
    n_sampled_chunks: int,
) -> dict[str, Any]:
    chunk_count = max(10, min(int(n_sampled_chunks), 200))
    if total_recording_seconds <= 30.0:
        centers = np.asarray([total_recording_seconds / 2.0], dtype=float)
        chunk_count = 1
    else:
        centers = np.linspace(15.0, total_recording_seconds - 15.0, chunk_count)

    chunk_labels: list[str] = []
    for center in centers:
        start_sample = (float(center) - 15.0) * fs_hz
        end_sample = (float(center) + 15.0) * fs_hz
        qrs = qrs_index[(qrs_index >= start_sample) & (qrs_index < end_sample)]
        if qrs.size < 5:
            chunk_labels.append("Unknown")
            continue
        rr_ms = _rr_intervals_ms_from_qrs(qrs, fs_hz=fs_hz)
        chunk_labels.append(classify_rhythm(rr_ms.tolist()) if rr_ms.size >= 4 else "Unknown")

    valid_labels = [label for label in chunk_labels if label != "Unknown"]
    valid_count = len(valid_labels)
    af_count = sum(1 for label in valid_labels if label == "AF")
    transition_count = _transition_count_ignoring_unknown(chunk_labels)
    transition_rate_per_hour = (
        float(transition_count / (total_recording_seconds / 3600.0))
        if total_recording_seconds > 0
        else 0.0
    )
    variability_label = _variability_label_from_transition_rate(
        transition_count=transition_count,
        transition_rate_per_hour=transition_rate_per_hour,
    )
    rr_full = _rr_intervals_ms_from_qrs(qrs_index, fs_hz=fs_hz)
    return {
        "scope": "sampled_recording",
        "total_recording_seconds": total_recording_seconds,
        "n_sampled_chunks": int(chunk_count),
        "chunk_duration_seconds": 30.0,
        "valid_chunk_count": int(valid_count),
        "chunk_labels": chunk_labels,
        "af_chunk_count": int(af_count),
        "n_chunk_chunk_count": int(valid_count),
        "af_chunk_fraction": float(af_count / valid_count) if valid_count else 0.0,
        "max_consecutive_af_chunks": _max_consecutive_label(chunk_labels, "AF"),
        "sustained_af_evidence": _max_consecutive_label(chunk_labels, "AF") >= MIN_SUSTAINED_RUNS,
        "signal_derived_transition_count": int(transition_count),
        "signal_derived_transition_per_hour_sampled": transition_rate_per_hour,
        "signal_derived_variability_label_sampled": variability_label,
        "signal_derived_variability_reason": (
            "Bucketed from transition_count / total_recording_hours, not from "
            "the raw transition count alone: stable = 0 transitions, "
            "occasional_changes = >0 and <=3 transitions/hour, "
            "frequent_changes = >3 transitions/hour."
        ),
        "rr_sample_entropy": _sample_entropy(rr_full),
        "rr_poincare_sd1_sd2_ratio": _poincare_sd1_sd2_ratio(rr_full),
        "caveat": (
            "Extended-context metrics are signal-classifier-derived from sampled "
            "chunks across the recording. They reflect the agent's own observation "
            "and may differ from clinical AF burden labels."
        ),
    }


def _rr_intervals_ms_from_qrs(qrs_index: np.ndarray, *, fs_hz: float) -> np.ndarray:
    qrs = np.asarray(qrs_index, dtype=float).reshape(-1)
    if qrs.size < 2 or fs_hz <= 0:
        return np.asarray([], dtype=float)
    rr_ms = np.diff(np.sort(qrs)) / float(fs_hz) * 1000.0
    return rr_ms[np.isfinite(rr_ms) & (rr_ms > 0)]


def _rr_irregularity_metrics(rr_ms: np.ndarray) -> dict[str, Any]:
    rr = np.asarray(rr_ms, dtype=float).reshape(-1)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size == 0:
        return {
            "rr_interval_count": 0,
            "mean_rr_ms": None,
            "coefficient_of_variation": None,
            "rmssd_ms": None,
            "pnn50_percent": None,
        }
    diffs = np.diff(rr)
    mean_rr = float(np.mean(rr))
    return {
        "rr_interval_count": int(rr.size),
        "mean_rr_ms": round(mean_rr, 3),
        "coefficient_of_variation": (
            round(float(np.std(rr) / mean_rr), 6) if mean_rr > 0 else None
        ),
        "rmssd_ms": round(float(np.sqrt(np.mean(diffs**2))), 3) if diffs.size else 0.0,
        "pnn50_percent": (
            round(float(np.mean(np.abs(diffs) > 50.0) * 100.0), 3)
            if diffs.size
            else 0.0
        ),
    }


def _transition_count_ignoring_unknown(labels: list[str]) -> int:
    previous: str | None = None
    transitions = 0
    for label in labels:
        if label == "Unknown":
            continue
        if previous is not None and label != previous:
            transitions += 1
        previous = label
    return transitions


def _variability_label_from_transition_rate(
    *,
    transition_count: int,
    transition_rate_per_hour: float,
) -> str:
    if transition_count <= 0:
        return "stable"
    if transition_rate_per_hour <= 3.0:
        return "occasional_changes"
    return "frequent_changes"


def _max_consecutive_label(labels: list[str], target: str) -> int:
    best = 0
    current = 0
    for label in labels:
        if label == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _sample_entropy(rr_ms: np.ndarray, *, m: int = 2) -> float | None:
    rr = np.asarray(rr_ms, dtype=float).reshape(-1)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < m + 2:
        return None
    if rr.size > 1000:
        indices = np.linspace(0, rr.size - 1, 1000).astype(int)
        rr = rr[indices]
    std = float(np.std(rr))
    if std <= 0:
        return 0.0
    radius = 0.2 * std
    count_m = _template_match_count(rr, m=m, radius=radius)
    count_m1 = _template_match_count(rr, m=m + 1, radius=radius)
    if count_m <= 0 or count_m1 <= 0:
        return None
    return round(float(-np.log(count_m1 / count_m)), 6)


def _template_match_count(values: np.ndarray, *, m: int, radius: float) -> int:
    n_templates = values.size - m + 1
    if n_templates <= 1:
        return 0
    templates = np.asarray([values[index : index + m] for index in range(n_templates)])
    distances = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
    matches = (distances <= radius)
    return int(np.sum(matches) - n_templates)


def _poincare_sd1_sd2_ratio(rr_ms: np.ndarray) -> float | None:
    rr = np.asarray(rr_ms, dtype=float).reshape(-1)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 3:
        return None
    diff = np.diff(rr)
    sd_rr = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    sd_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    sd1 = np.sqrt(0.5) * sd_diff
    sd2_sq = max(0.0, 2.0 * sd_rr**2 - 0.5 * sd_diff**2)
    sd2 = np.sqrt(sd2_sq)
    if sd2 <= 0:
        return None
    return round(float(sd1 / sd2), 6)


def _hr_30s_series_from_10s(
    hr_series: list[float | None],
    *,
    n_sub_windows: int,
) -> list[float | None]:
    hr_30s_series: list[float | None] = []
    complete_windows = (n_sub_windows // SUB_WINDOWS_PER_30S) * SUB_WINDOWS_PER_30S
    for index in range(0, complete_windows, SUB_WINDOWS_PER_30S):
        bin_hr = [
            float(value)
            for value in hr_series[index : index + SUB_WINDOWS_PER_30S]
            if value is not None
        ]
        hr_30s_series.append(float(np.mean(bin_hr)) if bin_hr else None)
    return hr_30s_series


def _hr_30s_series_from_qrs(
    qrs_index: np.ndarray | None,
    *,
    n_sub_windows: int,
    samples_per_window: int,
    fs_hz: float,
) -> list[float | None]:
    if qrs_index is None:
        return []
    hr_30s_series: list[float | None] = []
    samples_per_30s = samples_per_window * SUB_WINDOWS_PER_30S
    complete_windows = (n_sub_windows // SUB_WINDOWS_PER_30S) * SUB_WINDOWS_PER_30S
    for index in range(0, complete_windows, SUB_WINDOWS_PER_30S):
        start_sample = index * samples_per_window
        end_sample = start_sample + samples_per_30s
        hr_30s_series.append(
            _qrs_hr_bpm(
                qrs_index,
                start_sample=start_sample,
                end_sample=end_sample,
                fs_hz=fs_hz,
                min_beats=2,
            )
        )
    return hr_30s_series


def _qrs_hr_bpm(
    qrs_index: np.ndarray | None,
    *,
    start_sample: int,
    end_sample: int,
    fs_hz: float,
    min_beats: int = 1,
) -> float | None:
    if qrs_index is None or fs_hz <= 0 or end_sample <= start_sample:
        return None
    qrs = np.asarray(qrs_index, dtype=float).reshape(-1)
    beat_count = int(np.sum((qrs >= start_sample) & (qrs < end_sample)))
    if beat_count < min_beats:
        return None
    duration_s = (end_sample - start_sample) / float(fs_hz)
    if duration_s <= 0:
        return None
    return round(float(beat_count / duration_s * 60.0), 1)


def _extract_vitals(
    processor: ECGSignalProcessor,
    window: np.ndarray,
    fs_hz: float,
) -> VitalParams | None:
    try:
        return processor.extract_vitals(window, fs=int(round(fs_hz)))
    except Exception:
        logger.exception("ECG vital extraction failed during proactive rule replay")
        return None


def _vital_quality(vitals: VitalParams | None) -> float | None:
    if vitals is None:
        return None
    return _coerce_quality(getattr(vitals, "signal_quality_score", None))


def _coerce_quality(value: Any) -> float | None:
    if value is None:
        return None
    try:
        quality = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(quality):
        return None
    return quality


def _rule_reason(
    decision: RuleBasedJudge,
    rule_id: str,
    state: dict[str, Any],
    result: InterventionResult,
) -> str:
    rule = decision.get_rule(rule_id)
    if rule is None:
        return result.reason
    try:
        return rule.trigger(state) or result.reason
    except Exception:
        logger.exception("Rule reason extraction failed: %s", rule_id)
        return result.reason


def _resolve_modality(value: str | None) -> Modality:
    if value is None:
        return Modality.ECG
    lowered = str(value).strip().lower()
    if lowered == "ecg":
        return Modality.ECG
    if lowered == "ppg":
        return Modality.PPG
    raise ValueError("modality must be 'ecg' or 'ppg'.")


def _flatten_ecg(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(arr.shape[0], -1)[:, 0]


def _error_response(dataset: str, patient_id: str, error: str) -> dict[str, Any]:
    return {
        "success": False,
        "dataset": dataset,
        "patient_id": patient_id,
        "error": error,
    }


__all__ = ["evaluate_proactive_rules", "analyze_afppgecg_rhythm_context"]
