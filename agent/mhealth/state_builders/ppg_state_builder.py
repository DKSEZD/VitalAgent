"""Build VitalBench MonitoringState objects from AFPPGECG wrist PPG records.

PPG-derived vitals come from pulse peak detection. PPG window rhythm labels
are assigned from aligned ECG reference AF annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from agent.data.ppg_loader import ECGReferenceRecord, PPGLoader
from agent.data.streaming import PPGStreamer, PPGWindow
from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.mhealth.schemas import MonitoringState, RhythmClass


@dataclass(frozen=True)
class PPGWindowConfig:
    window_sec: int = 300
    max_hr_subwindow_sec: int = 30
    rhythm_threshold: float = 0.5
    ambiguous_rhythm_margin: float | None = None
    max_windows: int | None = None
    min_annotation_coverage_ratio: float = 0.8


@dataclass(frozen=True)
class AFInterval:
    start_s: float
    end_s: float
    is_af: bool


@dataclass(frozen=True)
class PPGRecordingAggregates:
    af_burden_ratio: float | None
    max_hr_bpm: float | None
    rhythm_transition_count_per_hour: float | None


@dataclass(frozen=True)
class _PendingPPGState:
    window: PPGWindow
    window_start_s_ecg: float
    window_end_s_ecg: float
    current_af_burden: float | None
    rhythm_class: RhythmClass | None
    hr_bpm: float | None
    sdnn_ms: float | None
    rmssd_ms: float | None
    signal_quality_score: float | None
    max_hr_bpm: float | None
    annotation_coverage_ratio: float


def build_af_intervals_from_ecg_reference(ecg: ECGReferenceRecord) -> list[AFInterval]:
    """Convert ECG beat-level AF annotations to half-open second intervals."""

    labels = np.asarray(ecg.af_annotation, dtype=int).reshape(-1)
    qrs = np.asarray(ecg.qrs_index, dtype=int).reshape(-1)
    fs = float(ecg.fs)

    if fs <= 0 or labels.size == 0:
        return []

    intervals: list[AFInterval] = []

    if qrs.size >= 2:
        for i in range(min(labels.size, qrs.size - 1)):
            start_s = float(qrs[i]) / fs
            end_s = float(qrs[i + 1]) / fs
            if end_s <= start_s:
                continue
            intervals.append(
                AFInterval(
                    start_s=start_s,
                    end_s=end_s,
                    is_af=bool(labels[i] == 1),
                )
            )
        return intervals

    cursor = 0.0
    rr_values = np.asarray(ecg.rr_seconds, dtype=float).reshape(-1)
    for rr, label in zip(rr_values, labels, strict=False):
        rr_s = float(rr)
        if rr_s <= 0:
            continue
        start_s = cursor
        end_s = cursor + rr_s
        intervals.append(
            AFInterval(
                start_s=start_s,
                end_s=end_s,
                is_af=bool(label == 1),
            )
        )
        cursor = end_s

    return intervals


def compute_af_burden_for_window(
    intervals: Sequence[AFInterval],
    start_s: float,
    end_s: float,
) -> float | None:
    """Return AF overlap duration divided by valid ECG-annotated overlap."""

    if end_s <= start_s:
        return None

    af_duration = 0.0
    annotated_duration = 0.0

    for interval in intervals:
        overlap = _overlap_seconds(interval.start_s, interval.end_s, start_s, end_s)
        if overlap <= 0:
            continue

        annotated_duration += overlap
        if interval.is_af:
            af_duration += overlap

    if annotated_duration <= 0:
        return None

    return af_duration / annotated_duration


def build_monitoring_states_from_ppg_patient(
    *,
    dataset_root: Path | str | None = None,
    patient_id: str | int,
    config: PPGWindowConfig = PPGWindowConfig(),
) -> list[MonitoringState]:
    """Build state-grounded VitalBench inputs from one AFPPGECG patient."""

    if config.window_sec <= 0:
        raise ValueError("window_sec must be positive.")

    if config.rhythm_threshold < 0 or config.rhythm_threshold > 1:
        raise ValueError("rhythm_threshold must be between 0 and 1.")

    if config.max_windows is not None and config.max_windows < 0:
        raise ValueError("max_windows must be non-negative when provided.")

    if config.min_annotation_coverage_ratio < 0 or config.min_annotation_coverage_ratio > 1:
        raise ValueError("min_annotation_coverage_ratio must be between 0 and 1.")

    loader = PPGLoader(dataset_root)
    ecg = loader.read_ecg_record(patient_id, include_signal=False)
    ppg = loader.read_ppg_record(patient_id, include_signals=True)

    patient = str(ecg.patient_id)
    intervals = build_af_intervals_from_ecg_reference(ecg)
    annotation_bounds = _annotation_bounds(intervals)
    if annotation_bounds is None:
        return []

    processor = PPGSignalProcessor()
    streamer = PPGStreamer(window_seconds=config.window_sec, drop_last=True)

    pending_states: list[_PendingPPGState] = []

    for window in streamer.stream_recording(ppg):
        if config.max_windows is not None and len(pending_states) >= config.max_windows:
            break

        window_start_s_ecg, window_end_s_ecg = _window_ecg_relative_bounds(
            window=window,
            ecg=ecg,
        )

        annotation_coverage_ratio = compute_annotation_coverage_ratio(
            intervals,
            window_start_s_ecg,
            window_end_s_ecg,
        )
        if annotation_coverage_ratio < config.min_annotation_coverage_ratio:
            continue

        vitals = processor.extract_vitals(
            window.signal,
            fs=window.fs,
            accel=window.accel,
        )
        max_hr_bpm = compute_ppg_max_hr_bpm(
            window=window,
            processor=processor,
            subwindow_sec=config.max_hr_subwindow_sec,
        )
        current_af_burden = compute_af_burden_for_window(
            intervals,
            window_start_s_ecg,
            window_end_s_ecg,
        )

        pending_states.append(
            _PendingPPGState(
                window=window,
                window_start_s_ecg=window_start_s_ecg,
                window_end_s_ecg=window_end_s_ecg,
                current_af_burden=current_af_burden,
                rhythm_class=_map_af_burden_to_rhythm(
                    current_af_burden,
                    threshold=config.rhythm_threshold,
                    ambiguous_margin=config.ambiguous_rhythm_margin,
                ),
                hr_bpm=vitals.hr_bpm,
                sdnn_ms=vitals.sdnn_ms,
                rmssd_ms=vitals.rmssd_ms,
                signal_quality_score=vitals.signal_quality_score,
                max_hr_bpm=max_hr_bpm,
                annotation_coverage_ratio=annotation_coverage_ratio,
            )
        )

    if not pending_states:
        return []

    recording_start_s = pending_states[0].window_start_s_ecg
    recording_end_s = pending_states[-1].window_end_s_ecg
    ppg_covered_duration_s = max(0.0, recording_end_s - recording_start_s)

    recording_aggregates = _compute_recording_aggregates(
        intervals=intervals,
        pending_states=pending_states,
        recording_start_s=recording_start_s,
        recording_end_s=recording_end_s,
    )

    states: list[MonitoringState] = []
    previous_state: MonitoringState | None = None
    previous_end_s: float | None = None

    for pending in pending_states:
        window = pending.window
        previous_gap_s = (
            None
            if previous_end_s is None
            else max(0.0, pending.window_start_s_ecg - previous_end_s)
        )
        same_contiguous_timeline = previous_gap_s is not None and previous_gap_s <= 1e-6
        effective_previous_state = previous_state if same_contiguous_timeline else None
        metadata = {
            "chunk_index": window.chunk_index,
            "window_start_time": (
                window.window_start_time.isoformat()
                if window.window_start_time is not None
                else None
            ),
            "ppg_stream_offset_s": window.stream_offset_s,
            "ecg_relative_start_s": pending.window_start_s_ecg,
            "ecg_relative_end_s": pending.window_end_s_ecg,
            "current_window_af_burden_ratio": pending.current_af_burden,
            "annotation_coverage_ratio": pending.annotation_coverage_ratio,
            "min_annotation_coverage_ratio": config.min_annotation_coverage_ratio,
            "rhythm_gt_source": "aligned_ecg_af_annotation",
            "rhythm_threshold": config.rhythm_threshold,
            "hr_source": "ppg_peak_detection",
            "max_hr_bpm_source": "max_ppg_peak_hr_over_subwindows",
            "dataset_source": "afppgecg",
            "recording_aggregate_scope": (
                "aligned_ecg_time_span_between_retained_ppg_windows"
            ),
            "rhythm_transition_source": "aligned_ecg_reference_annotation",
            "previous_gap_s": previous_gap_s,
            "window_sec": config.window_sec,
            "max_hr_subwindow_sec": config.max_hr_subwindow_sec,
        }

        state = MonitoringState(
            state_id=f"ppg_patient_{patient}_window_{window.window_index}",
            patient_id=patient,
            dataset="afppgecg",
            modality="ppg",
            window_index=window.window_index,
            window_start_s=pending.window_start_s_ecg,
            window_duration_s=pending.window_end_s_ecg - pending.window_start_s_ecg,
            hr_bpm=pending.hr_bpm,
            previous_hr_bpm=(
                effective_previous_state.hr_bpm
                if effective_previous_state is not None
                else None
            ),
            rhythm_class=pending.rhythm_class,
            previous_rhythm_class=(
                effective_previous_state.rhythm_class
                if effective_previous_state is not None
                else None
            ),
            af_burden_ratio=recording_aggregates.af_burden_ratio,
            max_hr_bpm=recording_aggregates.max_hr_bpm,
            rhythm_transition_count_per_hour=(
                recording_aggregates.rhythm_transition_count_per_hour
            ),
            sdnn_ms=pending.sdnn_ms,
            rmssd_ms=pending.rmssd_ms,
            signal_quality_score=pending.signal_quality_score,
            rhythm_transition_count=None,
            recording_duration_s=ppg_covered_duration_s,
            metadata=metadata,
        )

        states.append(state)
        previous_state = state
        previous_end_s = pending.window_end_s_ecg

    return states


def _compute_recording_aggregates(
    *,
    intervals: Sequence[AFInterval],
    pending_states: Sequence[_PendingPPGState],
    recording_start_s: float,
    recording_end_s: float,
) -> PPGRecordingAggregates:
    af_burden_ratio = compute_af_burden_for_window(
        intervals,
        recording_start_s,
        recording_end_s,
    )

    max_hr_values = [
        float(pending.max_hr_bpm)
        for pending in pending_states
        if pending.max_hr_bpm is not None
    ]
    max_hr_bpm = max(max_hr_values) if max_hr_values else None
    rhythm_transition_count_per_hour = compute_reference_af_transition_count_per_hour(
        intervals,
        recording_start_s,
        recording_end_s,
    )

    return PPGRecordingAggregates(
        af_burden_ratio=af_burden_ratio,
        max_hr_bpm=max_hr_bpm,
        rhythm_transition_count_per_hour=rhythm_transition_count_per_hour,
    )


def _map_af_burden_to_rhythm(
    burden: float | None,
    *,
    threshold: float,
    ambiguous_margin: float | None,
) -> RhythmClass | None:
    if burden is None:
        return None

    if ambiguous_margin is not None and abs(burden - threshold) <= ambiguous_margin:
        return None

    return "AF" if burden >= threshold else "N"


def compute_annotation_coverage_ratio(
    intervals: Sequence[AFInterval],
    start_s: float,
    end_s: float,
) -> float:
    duration_s = end_s - start_s
    if duration_s <= 0:
        return 0.0
    return _annotated_overlap_seconds(intervals, start_s, end_s) / duration_s


def compute_reference_af_transition_count_per_hour(
    intervals: Sequence[AFInterval],
    start_s: float,
    end_s: float,
) -> float | None:
    duration_hours = (end_s - start_s) / 3600.0
    if duration_hours <= 0:
        return None

    labels: list[bool] = []
    for interval in intervals:
        if _overlap_seconds(interval.start_s, interval.end_s, start_s, end_s) <= 0:
            continue
        if not labels or labels[-1] != interval.is_af:
            labels.append(interval.is_af)

    if not labels:
        return None
    return max(0, len(labels) - 1) / duration_hours


def compute_ppg_max_hr_bpm(
    *,
    window: PPGWindow,
    processor: PPGSignalProcessor,
    subwindow_sec: int,
) -> float | None:
    subwindow_samples = int(subwindow_sec * window.fs)
    if subwindow_samples <= 0:
        return None

    values: list[float] = []
    cursor = 0
    while cursor < window.n_samples:
        sub_end = min(cursor + subwindow_samples, window.n_samples)
        signal = window.signal[cursor:sub_end]
        accel = _slice_accel_for_ppg_window(
            accel=window.accel,
            start_sample=cursor,
            end_sample=sub_end,
            total_ppg_samples=window.n_samples,
        )
        vitals = processor.extract_vitals(signal, fs=window.fs, accel=accel)
        if vitals.hr_bpm is not None:
            values.append(float(vitals.hr_bpm))
        cursor = sub_end

    return max(values) if values else None


def _window_ecg_relative_bounds(
    *,
    window: PPGWindow,
    ecg: ECGReferenceRecord,
) -> tuple[float, float]:
    if window.window_start_time is not None:
        start_s = (window.window_start_time - ecg.start_time).total_seconds()
    else:
        start_s = float(window.stream_offset_s)

    duration_s = window.n_samples / window.fs
    return start_s, start_s + duration_s


def _annotation_bounds(intervals: Sequence[AFInterval]) -> tuple[float, float] | None:
    if not intervals:
        return None
    return min(interval.start_s for interval in intervals), max(
        interval.end_s for interval in intervals
    )


def _annotated_overlap_seconds(
    intervals: Sequence[AFInterval],
    start_s: float,
    end_s: float,
) -> float:
    return sum(
        _overlap_seconds(interval.start_s, interval.end_s, start_s, end_s)
        for interval in intervals
    )


def _slice_accel_for_ppg_window(
    *,
    accel: np.ndarray | None,
    start_sample: int,
    end_sample: int,
    total_ppg_samples: int,
) -> np.ndarray | None:
    if accel is None or total_ppg_samples <= 0:
        return None
    accel_array = np.asarray(accel)
    if accel_array.ndim != 2 or len(accel_array) == 0:
        return None
    ratio = len(accel_array) / total_ppg_samples
    accel_start = int(round(start_sample * ratio))
    accel_end = int(round(end_sample * ratio))
    return accel_array[accel_start:accel_end]


def _overlap_seconds(
    left_start_s: float,
    left_end_s: float,
    right_start_s: float,
    right_end_s: float,
) -> float:
    return max(0.0, min(left_end_s, right_end_s) - max(left_start_s, right_start_s))


def _ranges_overlap(
    left_start_s: float,
    left_end_s: float,
    right_start_s: float,
    right_end_s: float,
) -> bool:
    return _overlap_seconds(left_start_s, left_end_s, right_start_s, right_end_s) > 0


__all__ = [
    "PPGWindowConfig",
    "AFInterval",
    "PPGRecordingAggregates",
    "build_af_intervals_from_ecg_reference",
    "compute_annotation_coverage_ratio",
    "compute_af_burden_for_window",
    "compute_reference_af_transition_count_per_hour",
    "compute_ppg_max_hr_bpm",
    "build_monitoring_states_from_ppg_patient",
]

