"""Build MonitoringState objects from original UCI PPG-DaLiA subject pickles.

PPG-DaLiA is used here for wearable PPG/BVP heart-rate and protocol activity
context QA. ECG-derived HR labels are used only as dataset-construction ground
truth. Protocol activity labels are context labels; this is not a stress,
sleep, or clinical diagnosis dataset, and no agent-side classifier tools are
implemented here.

First-pass PPG-DaLiA QA generation uses activity labels as MonitoringState
metadata/context while generated questions mainly come from existing HR
templates. Activity-specific templates are intentionally left for a later pass.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.mhealth.schemas import MonitoringState


PPG_DALIA_HR_LABEL_FS_HZ = 0.5
PPG_DALIA_BVP_FS_HZ = 64
PPG_DALIA_WRIST_ACC_FS_HZ = 32
PPG_DALIA_DEFAULT_ACTIVITY_FS_HZ = 4
PPG_DALIA_WRIST_LOW_RATE_FS_HZ = 4
PPG_DALIA_CHEST_FS_HZ = 700


PPG_DALIA_ACTIVITY_LABELS: dict[int, str] = {
    0: "transition",
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
    9: "clean_baseline",
}

TRANSITION_ACTIVITY_NAMES = {"transition", "unknown", "no_activity"}


@dataclass(frozen=True)
class PPGDaLiAWindowConfig:
    """Configuration for PPG-DaLiA clean-window state generation."""

    window_secs: tuple[int, ...] = (30, 60)
    window_stride_sec: int | None = None
    hr_label_fs_hz: float = PPG_DALIA_HR_LABEL_FS_HZ
    bvp_fs_hz: int = PPG_DALIA_BVP_FS_HZ
    wrist_acc_fs_hz: int = PPG_DALIA_WRIST_ACC_FS_HZ
    max_hr_subwindow_sec: int = 30
    activity_label_fs_hz: float | None = None
    source_variant: str = "uci_original_pickle"
    hr_label_source: str = "ecg_derived"
    reference_signal: str = "chest_ecg"
    min_activity_fraction: float = 0.8
    max_windows_per_duration: int | None = None
    include_transition_windows: bool = False


@dataclass(frozen=True)
class PPGDaLiAActivityWindow:
    """Dominant PPG-DaLiA activity label for a candidate window."""

    activity_label_id: int | None
    activity_label: str | None
    activity_label_fraction: float | None
    is_transition_or_unknown: bool


@dataclass(frozen=True)
class PPGDaLiAPulseWindow:
    """Signal-derived pulse metrics from the wrist BVP window."""

    hr_bpm: float | None
    max_hr_bpm: float | None
    sdnn_ms: float | None
    rmssd_ms: float | None
    signal_quality_score: float | None


def load_ppg_dalia_subject_pickle(path: Path | str) -> dict[str, Any]:
    """Load one original UCI PPG-DaLiA `S*.pkl` subject file."""

    with Path(path).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")

    if not isinstance(payload, dict):
        raise ValueError(f"PPG-DaLiA pickle did not contain a dict: {path}")

    if "label" not in payload:
        raise ValueError(f"PPG-DaLiA pickle has no HR label vector: {path}")

    return payload


def build_monitoring_states_from_ppg_dalia_subject(
    pickle_path: Path | str,
    *,
    config: PPGDaLiAWindowConfig = PPGDaLiAWindowConfig(),
) -> list[MonitoringState]:
    """Build clean-window HR/activity MonitoringState objects from one subject."""

    _validate_config(config)

    payload = load_ppg_dalia_subject_pickle(pickle_path)
    subject_id = _normalize_subject_id(payload.get("subject"), Path(pickle_path).stem)
    hr_labels = _as_float_1d(payload.get("label"))
    activity_labels = _extract_activity_labels(payload)
    wrist_acc = _extract_wrist_acc(payload)
    bvp = _extract_wrist_bvp(payload)

    if hr_labels.size == 0:
        return []

    activity_fs_hz = _resolve_activity_label_fs_hz(
        activity_labels=activity_labels,
        hr_labels=hr_labels,
        bvp=bvp,
        config=config,
    )
    recording_duration_s = _infer_recording_duration_s(
        hr_labels=hr_labels,
        activity_labels=activity_labels,
        activity_fs_hz=activity_fs_hz,
        bvp=bvp,
        config=config,
    )

    states: list[MonitoringState] = []
    for window_sec in config.window_secs:
        states.extend(
            _build_states_for_duration(
                subject_id=subject_id,
                hr_labels=hr_labels,
                activity_labels=activity_labels,
                activity_fs_hz=activity_fs_hz,
                wrist_acc=wrist_acc,
                bvp=bvp,
                recording_duration_s=recording_duration_s,
                window_sec=window_sec,
                config=config,
            )
        )

    return states


def _validate_config(config: PPGDaLiAWindowConfig) -> None:
    if not config.window_secs:
        raise ValueError("window_secs must contain at least one duration.")

    if any(window_sec <= 0 for window_sec in config.window_secs):
        raise ValueError("All window durations must be positive.")

    if config.window_stride_sec is not None and config.window_stride_sec <= 0:
        raise ValueError("window_stride_sec must be positive when provided.")

    if config.hr_label_fs_hz <= 0:
        raise ValueError("hr_label_fs_hz must be positive.")

    if config.bvp_fs_hz <= 0:
        raise ValueError("bvp_fs_hz must be positive.")

    if config.wrist_acc_fs_hz <= 0:
        raise ValueError("wrist_acc_fs_hz must be positive.")

    if config.max_hr_subwindow_sec <= 0:
        raise ValueError("max_hr_subwindow_sec must be positive.")

    if config.activity_label_fs_hz is not None and config.activity_label_fs_hz <= 0:
        raise ValueError("activity_label_fs_hz must be positive when provided.")

    if not 0 < config.min_activity_fraction <= 1:
        raise ValueError("min_activity_fraction must be in (0, 1].")

    if config.max_windows_per_duration is not None and config.max_windows_per_duration < 0:
        raise ValueError("max_windows_per_duration must be non-negative when provided.")


def _build_states_for_duration(
    *,
    subject_id: str,
    hr_labels: np.ndarray,
    activity_labels: np.ndarray | None,
    activity_fs_hz: float,
    wrist_acc: np.ndarray | None,
    bvp: np.ndarray | None,
    recording_duration_s: float,
    window_sec: int,
    config: PPGDaLiAWindowConfig,
) -> list[MonitoringState]:
    stride_sec = config.window_stride_sec or window_sec
    states: list[MonitoringState] = []
    previous_state: MonitoringState | None = None
    processor = PPGSignalProcessor(sampling_rate=config.bvp_fs_hz)
    raw_window_index = 0
    cursor_s = 0.0

    while cursor_s + window_sec <= recording_duration_s + 1e-9:
        if (
            config.max_windows_per_duration is not None
            and len(states) >= config.max_windows_per_duration
        ):
            break

        start_s = cursor_s
        end_s = cursor_s + window_sec
        hr_window = _slice_by_time(hr_labels, config.hr_label_fs_hz, start_s, end_s)
        finite_hr = hr_window[np.isfinite(hr_window)]

        activity = _classify_activity_window(
            activity_labels,
            start_s=start_s,
            end_s=end_s,
            activity_fs_hz=activity_fs_hz,
        )
        if _should_skip_activity_window(activity, config=config):
            cursor_s += stride_sec
            raw_window_index += 1
            continue

        bvp_window = _slice_signal_by_time(
            bvp,
            fs_hz=config.bvp_fs_hz,
            start_s=start_s,
            end_s=end_s,
        )
        acc_window = _slice_signal_by_time(
            wrist_acc,
            fs_hz=config.wrist_acc_fs_hz,
            start_s=start_s,
            end_s=end_s,
        )
        pulse = _derive_bvp_pulse_window(
            bvp_window=bvp_window,
            acc_window=acc_window,
            processor=processor,
            bvp_fs_hz=config.bvp_fs_hz,
            max_hr_subwindow_sec=config.max_hr_subwindow_sec,
        )
        if pulse.hr_bpm is None:
            cursor_s += stride_sec
            raw_window_index += 1
            continue

        reference_hr = float(np.mean(finite_hr)) if finite_hr.size else None
        motion_level = _derive_motion_level(
            wrist_acc,
            start_s=start_s,
            end_s=end_s,
            acc_fs_hz=config.wrist_acc_fs_hz,
        )
        clean_index = len(states)
        state_id = (
            f"ppg_dalia_subject_{subject_id}_"
            f"{window_sec}s_clean_window_{clean_index:04d}"
        )
        metadata = _build_window_metadata(
            subject_id=subject_id,
            raw_window_index=raw_window_index,
            clean_index=clean_index,
            start_s=start_s,
            end_s=end_s,
            window_sec=window_sec,
            stride_sec=stride_sec,
            hr_labels=hr_labels,
            finite_hr=finite_hr,
            activity=activity,
            activity_fs_hz=activity_fs_hz,
            previous_state=previous_state,
            bvp=bvp,
            wrist_acc=wrist_acc,
            pulse=pulse,
            config=config,
        )

        state = MonitoringState(
            state_id=state_id,
            patient_id=subject_id,
            dataset="ppg_dalia",
            modality="ppg",
            window_index=clean_index,
            window_start_s=start_s,
            window_duration_s=window_sec,
            hr_bpm=pulse.hr_bpm,
            previous_hr_bpm=previous_state.hr_bpm if previous_state is not None else None,
            activity_label=activity.activity_label,
            previous_activity_label=(
                previous_state.activity_label if previous_state is not None else None
            ),
            max_hr_bpm=pulse.max_hr_bpm,
            mean_hr_bpm=pulse.hr_bpm,
            ecg_reference_hr_bpm=(
                reference_hr if config.hr_label_source == "ecg_derived" else None
            ),
            sdnn_ms=pulse.sdnn_ms,
            rmssd_ms=pulse.rmssd_ms,
            signal_quality_score=pulse.signal_quality_score,
            motion_level=motion_level,
            recording_duration_s=recording_duration_s,
            metadata=metadata,
        )

        states.append(state)
        previous_state = state
        cursor_s += stride_sec
        raw_window_index += 1

    return states


def _classify_activity_window(
    activity_labels: np.ndarray | None,
    *,
    start_s: float,
    end_s: float,
    activity_fs_hz: float,
) -> PPGDaLiAActivityWindow:
    if activity_labels is None or activity_labels.size == 0:
        return PPGDaLiAActivityWindow(
            activity_label_id=None,
            activity_label=None,
            activity_label_fraction=None,
            is_transition_or_unknown=False,
        )

    window = _slice_by_time(
        activity_labels,
        activity_fs_hz,
        start_s,
        end_s,
    )
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return PPGDaLiAActivityWindow(
            activity_label_id=None,
            activity_label=None,
            activity_label_fraction=None,
            is_transition_or_unknown=True,
        )

    ids = np.rint(finite).astype(int)
    values, counts = np.unique(ids, return_counts=True)
    best_index = int(np.argmax(counts))
    label_id = int(values[best_index])
    fraction = float(counts[best_index]) / float(ids.size)
    label = map_ppg_dalia_activity_label(label_id)

    return PPGDaLiAActivityWindow(
        activity_label_id=label_id,
        activity_label=label,
        activity_label_fraction=fraction,
        is_transition_or_unknown=label in TRANSITION_ACTIVITY_NAMES,
    )


def _should_skip_activity_window(
    activity: PPGDaLiAActivityWindow,
    *,
    config: PPGDaLiAWindowConfig,
) -> bool:
    if activity.activity_label is None:
        return False

    if config.include_transition_windows:
        return False

    if activity.is_transition_or_unknown:
        return True

    fraction = activity.activity_label_fraction
    return fraction is not None and fraction < config.min_activity_fraction


def map_ppg_dalia_activity_label(label_id: int) -> str:
    """Map numeric PPG-DaLiA protocol activity labels to stable names."""

    return PPG_DALIA_ACTIVITY_LABELS.get(int(label_id), "unknown")


def _derive_motion_level(
    wrist_acc: np.ndarray | None,
    *,
    start_s: float,
    end_s: float,
    acc_fs_hz: int,
) -> str | None:
    """Return a coarse heuristic motion bucket from wrist ACC variability.

    Empatica E4 wrist ACC is commonly represented in g-like units in
    PPG-DaLiA-derived files, but exact scaling should be verified on original
    pickles before motion_level is used as a main benchmark target. The current
    thresholds are placeholders for exploratory metadata only.
    """

    if wrist_acc is None or wrist_acc.size == 0:
        return None

    acc_window = _slice_by_time(wrist_acc, acc_fs_hz, start_s, end_s)
    if acc_window.size == 0 or not np.isfinite(acc_window).any():
        return None

    arr = np.asarray(acc_window, dtype=float)
    if arr.ndim == 1:
        magnitude = arr
    else:
        magnitude = np.linalg.norm(arr, axis=1)

    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return None

    score = float(np.std(finite))
    if score < 0.02:
        return "low"
    if score < 0.08:
        return "medium"
    return "high"


def _build_window_metadata(
    *,
    subject_id: str,
    raw_window_index: int,
    clean_index: int,
    start_s: float,
    end_s: float,
    window_sec: int,
    stride_sec: int,
    hr_labels: np.ndarray,
    finite_hr: np.ndarray,
    activity: PPGDaLiAActivityWindow,
    activity_fs_hz: float,
    previous_state: MonitoringState | None,
    bvp: np.ndarray | None,
    wrist_acc: np.ndarray | None,
    pulse: PPGDaLiAPulseWindow,
    config: PPGDaLiAWindowConfig,
) -> dict[str, Any]:
    hr_start = int(np.ceil(start_s * config.hr_label_fs_hz))
    hr_end = int(np.ceil(end_s * config.hr_label_fs_hz))

    metadata: dict[str, Any] = {
        "dataset_source": "ppg_dalia",
        "source_variant": config.source_variant,
        "subject_id": subject_id,
        "device_setting": "wearable",
        "primary_signal": "wrist_ppg_bvp",
        "reference_signal": config.reference_signal,
        "hr_label_source": config.hr_label_source,
        "hr_source": "wrist_bvp_peak_detection",
        "mean_hr_bpm_source": "wrist_bvp_peak_detection",
        "max_hr_bpm_source": "max_wrist_bvp_peak_hr_over_subwindows",
        "activity_label_source": "protocol_activity",
        "raw_window_index": raw_window_index,
        "clean_window_index": clean_index,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "window_sec": window_sec,
        "window_stride_sec": stride_sec,
        "window_overlap": stride_sec < window_sec,
        "hr_label_sampling_rate_hz": config.hr_label_fs_hz,
        "hr_label_time_semantics": (
            "Treated as a 0.5 Hz time series for benchmark construction. "
            "PPG-DaLiA HR labels are ECG-derived values commonly produced "
            "from 8s windows with 2s shift; exact center/end alignment should "
            "be verified against original dataset documentation before paper "
            "claims depend on sub-window timing."
        ),
        "bvp_sampling_rate_hz": config.bvp_fs_hz,
        "wrist_acc_sampling_rate_hz": config.wrist_acc_fs_hz,
        "activity_label_sampling_rate_hz": activity_fs_hz,
        "activity_label_sampling_rate_source": (
            "configured"
            if config.activity_label_fs_hz is not None
            else "inferred_from_activity_length_and_hr_bvp_duration"
        ),
        "chest_sampling_rate_hz": PPG_DALIA_CHEST_FS_HZ,
        "hr_label_window_start_index": max(0, hr_start),
        "hr_label_window_end_index": min(hr_end, hr_labels.size),
        "hr_label_window_count": int(finite_hr.size),
        "ecg_reference_hr_bpm": float(np.mean(finite_hr)) if finite_hr.size else None,
        "activity_label_id": activity.activity_label_id,
        "activity_label": activity.activity_label,
        "activity_label_fraction": activity.activity_label_fraction,
        "previous_activity_label": (
            previous_state.activity_label if previous_state is not None else None
        ),
        "min_activity_fraction": config.min_activity_fraction,
        "include_transition_windows": config.include_transition_windows,
        "activity_context_boundary": (
            "Current PPG-DaLiA first pass uses activity labels as state "
            "metadata/context. Generated QA mainly covers HR-related templates; "
            "activity-specific QA can be added in a later template pass."
        ),
        "motion_level_semantics": (
            "Heuristic derived from wrist ACC magnitude variability. Validate "
            "thresholds before using motion_level as a main benchmark target."
        ),
        "previous_clean_window_state_id": (
            previous_state.state_id if previous_state is not None else None
        ),
        "previous_clean_window_start_s": (
            previous_state.window_start_s if previous_state is not None else None
        ),
        "previous_clean_window_time_gap_s": (
            start_s - (previous_state.window_start_s + previous_state.window_duration_s)
            if previous_state is not None
            and previous_state.window_start_s is not None
            and previous_state.window_duration_s is not None
            else None
        ),
        "previous_window_comparison_semantics": (
            "previous_hr_bpm refers to the previous retained clean PPG-DaLiA "
            "window of the same duration, not necessarily the immediately "
            "adjacent raw time window. If transition or low-purity windows were "
            "skipped, this comparison may cross time gaps and activity changes. "
            "Treat ta4 on PPG-DaLiA as an HR-direction sanity check under "
            "protocol-clean activity context, not cardiac-event detection."
        ),
        "ground_truth_boundary": (
            "PPG-DaLiA provides ECG-derived HR reference labels and protocol "
            "activity context. State hr_bpm fields are derived from wrist BVP "
            "peak detection; ECG-derived labels are retained only as reference "
            "metadata/ecg_reference_hr_bpm."
        ),
    }

    metadata.update(
        {
            "bvp_pulse_mean_hr_bpm": pulse.hr_bpm,
            "bvp_pulse_max_hr_bpm": pulse.max_hr_bpm,
            "bvp_pulse_sdnn_ms": pulse.sdnn_ms,
            "bvp_pulse_rmssd_ms": pulse.rmssd_ms,
            "bvp_pulse_signal_quality_score": pulse.signal_quality_score,
            "max_hr_subwindow_sec": config.max_hr_subwindow_sec,
        }
    )

    if bvp is not None:
        metadata.update(
            _signal_window_metadata(
                bvp,
                fs_hz=config.bvp_fs_hz,
                start_s=start_s,
                end_s=end_s,
                prefix="wrist_bvp",
            )
        )

    if wrist_acc is not None:
        metadata.update(
            _signal_window_metadata(
                wrist_acc,
                fs_hz=config.wrist_acc_fs_hz,
                start_s=start_s,
                end_s=end_s,
                prefix="wrist_acc",
            )
        )

    return metadata


def _signal_window_metadata(
    signal: np.ndarray,
    *,
    fs_hz: float,
    start_s: float,
    end_s: float,
    prefix: str,
) -> dict[str, Any]:
    start = int(round(start_s * fs_hz))
    end = min(int(round(end_s * fs_hz)), int(signal.shape[0]))
    return {
        f"{prefix}_total_samples": int(signal.shape[0]),
        f"{prefix}_window_start_sample": max(0, start),
        f"{prefix}_window_end_sample": max(0, end),
        f"{prefix}_window_samples": max(0, end - max(0, start)),
    }


def _derive_bvp_pulse_window(
    *,
    bvp_window: np.ndarray | None,
    acc_window: np.ndarray | None,
    processor: PPGSignalProcessor,
    bvp_fs_hz: int,
    max_hr_subwindow_sec: int,
) -> PPGDaLiAPulseWindow:
    if bvp_window is None or bvp_window.size == 0:
        return PPGDaLiAPulseWindow(None, None, None, None, None)

    signal = np.asarray(bvp_window, dtype=float).reshape(-1)
    try:
        vitals = processor.extract_vitals(signal, fs=bvp_fs_hz, accel=acc_window)
    except Exception:
        return PPGDaLiAPulseWindow(None, None, None, None, None)

    hr_bpm = _finite_float_or_none(vitals.hr_bpm)
    interval_max_hr = _max_hr_from_intervals(vitals.rr_intervals_ms)
    subwindow_max_hr = _compute_bvp_subwindow_max_hr(
        signal=signal,
        acc_window=acc_window,
        processor=processor,
        bvp_fs_hz=bvp_fs_hz,
        subwindow_sec=max_hr_subwindow_sec,
    )
    max_candidates = [
        value for value in (interval_max_hr, subwindow_max_hr, hr_bpm) if value is not None
    ]

    return PPGDaLiAPulseWindow(
        hr_bpm=hr_bpm,
        max_hr_bpm=max(max_candidates) if max_candidates else None,
        sdnn_ms=_finite_float_or_none(vitals.sdnn_ms),
        rmssd_ms=_finite_float_or_none(vitals.rmssd_ms),
        signal_quality_score=_finite_float_or_none(vitals.signal_quality_score),
    )


def _compute_bvp_subwindow_max_hr(
    *,
    signal: np.ndarray,
    acc_window: np.ndarray | None,
    processor: PPGSignalProcessor,
    bvp_fs_hz: int,
    subwindow_sec: int,
) -> float | None:
    subwindow_samples = int(round(subwindow_sec * bvp_fs_hz))
    if subwindow_samples <= 0 or signal.size == 0:
        return None

    values: list[float] = []
    cursor = 0
    while cursor < signal.size:
        end = min(cursor + subwindow_samples, signal.size)
        sub_acc = _slice_acc_for_bvp_subwindow(
            acc_window=acc_window,
            start_sample=cursor,
            end_sample=end,
            total_bvp_samples=signal.size,
        )
        try:
            vitals = processor.extract_vitals(signal[cursor:end], fs=bvp_fs_hz, accel=sub_acc)
        except Exception:
            vitals = None
        if vitals is not None:
            hr = _finite_float_or_none(vitals.hr_bpm)
            if hr is not None:
                values.append(hr)
        cursor = end

    return max(values) if values else None


def _slice_acc_for_bvp_subwindow(
    *,
    acc_window: np.ndarray | None,
    start_sample: int,
    end_sample: int,
    total_bvp_samples: int,
) -> np.ndarray | None:
    if acc_window is None or acc_window.size == 0 or total_bvp_samples <= 0:
        return None
    arr = np.asarray(acc_window)
    ratio = arr.shape[0] / float(total_bvp_samples)
    acc_start = int(round(start_sample * ratio))
    acc_end = int(round(end_sample * ratio))
    return arr[acc_start:acc_end]


def _max_hr_from_intervals(intervals_ms: Sequence[float] | None) -> float | None:
    if intervals_ms is None:
        return None
    intervals = np.asarray(intervals_ms, dtype=float)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0:
        return None
    return round(float(np.max(60_000.0 / intervals)), 1)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def _infer_recording_duration_s(
    *,
    hr_labels: np.ndarray,
    activity_labels: np.ndarray | None,
    activity_fs_hz: float,
    bvp: np.ndarray | None,
    config: PPGDaLiAWindowConfig,
) -> float:
    durations = [hr_labels.size / float(config.hr_label_fs_hz)]

    if activity_labels is not None and activity_labels.size:
        durations.append(activity_labels.size / float(activity_fs_hz))

    if bvp is not None and bvp.size:
        durations.append(bvp.shape[0] / float(config.bvp_fs_hz))

    return min(durations)


def _resolve_activity_label_fs_hz(
    *,
    activity_labels: np.ndarray | None,
    hr_labels: np.ndarray,
    bvp: np.ndarray | None,
    config: PPGDaLiAWindowConfig,
) -> float:
    if config.activity_label_fs_hz is not None:
        return float(config.activity_label_fs_hz)

    if activity_labels is None or activity_labels.size == 0:
        return float(PPG_DALIA_DEFAULT_ACTIVITY_FS_HZ)

    reference_durations: list[float] = []
    if bvp is not None and bvp.size:
        reference_durations.append(bvp.shape[0] / float(config.bvp_fs_hz))

    if hr_labels.size:
        reference_durations.append(hr_labels.size / float(config.hr_label_fs_hz))

    reference_durations = [duration for duration in reference_durations if duration > 0]
    if not reference_durations:
        return float(PPG_DALIA_DEFAULT_ACTIVITY_FS_HZ)

    reference_duration_s = float(np.median(reference_durations))
    inferred_fs = float(activity_labels.size) / reference_duration_s
    if inferred_fs <= 0 or not np.isfinite(inferred_fs):
        return float(PPG_DALIA_DEFAULT_ACTIVITY_FS_HZ)

    return inferred_fs


def _extract_wrist_bvp(payload: dict[str, Any]) -> np.ndarray | None:
    return _extract_nested_array(payload, ("signal", "wrist", "BVP"))


def _extract_wrist_acc(payload: dict[str, Any]) -> np.ndarray | None:
    return _extract_nested_array(payload, ("signal", "wrist", "ACC"))


def _extract_activity_labels(payload: dict[str, Any]) -> np.ndarray | None:
    # Activity labels are protocol context labels. Their sampling rate is
    # resolved separately from signal duration rather than assumed here.
    activity = payload.get("activity")
    if activity is None:
        return None
    return _as_float_1d(activity)


def _extract_nested_array(payload: dict[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]

    return np.asarray(value)


def _slice_by_time(
    values: np.ndarray,
    fs_hz: float,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    start = max(0, int(np.ceil(start_s * fs_hz)))
    end = min(int(np.ceil(end_s * fs_hz)), int(values.shape[0]))
    if end <= start:
        return values[:0]
    return values[start:end]


def _slice_signal_by_time(
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


def _as_float_1d(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _normalize_subject_id(subject_raw: Any, fallback: str) -> str:
    if isinstance(subject_raw, bytes):
        subject_raw = subject_raw.decode("utf-8", errors="replace")

    subject_id = str(subject_raw or fallback).strip()
    return subject_id or fallback


__all__ = [
    "PPG_DALIA_HR_LABEL_FS_HZ",
    "PPG_DALIA_BVP_FS_HZ",
    "PPG_DALIA_WRIST_ACC_FS_HZ",
    "PPG_DALIA_DEFAULT_ACTIVITY_FS_HZ",
    "PPG_DALIA_WRIST_LOW_RATE_FS_HZ",
    "PPG_DALIA_CHEST_FS_HZ",
    "PPG_DALIA_ACTIVITY_LABELS",
    "PPGDaLiAWindowConfig",
    "PPGDaLiAActivityWindow",
    "load_ppg_dalia_subject_pickle",
    "map_ppg_dalia_activity_label",
    "build_monitoring_states_from_ppg_dalia_subject",
]

