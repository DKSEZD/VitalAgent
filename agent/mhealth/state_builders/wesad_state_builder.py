"""Build stress-aware MonitoringState objects from WESAD subject pickles.

WESAD stores synchronised signals and a 700 Hz protocol label vector in each
`S*.pkl` file. This builder turns clean 30s/60s protocol-label windows into
state-grounded inputs for stress/affect QA templates.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agent.mhealth.schemas import MonitoringState, StressClass


WESAD_LABEL_FS_HZ = 700
WESAD_WRIST_BVP_FS_HZ = 64

WESAD_PROTOCOL_LABELS: dict[int, StressClass] = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

IGNORED_WESAD_LABELS = {
    0: "not_defined_or_transient",
    5: "ignore",
    6: "ignore",
    7: "ignore",
}


@dataclass(frozen=True)
class WESADWindowConfig:
    """Configuration for WESAD clean-window state generation."""

    window_secs: tuple[int, ...] = (30, 60)
    window_stride_sec: int | None = None
    label_sampling_rate_hz: int = WESAD_LABEL_FS_HZ
    allowed_protocol_label_ids: tuple[int, ...] = (1, 2, 3)
    min_label_fraction: float = 1.0
    max_windows_per_duration: int | None = None


@dataclass(frozen=True)
class WESADWindowLabel:
    """Dominant protocol label for a candidate WESAD window."""

    protocol_label_id: int
    stress_label: StressClass
    label_fraction: float
    is_clean: bool


@dataclass(frozen=True)
class WESADSubjectAggregates:
    """Subject-level aggregate fields over allowed WESAD protocol labels."""

    stress_burden_ratio: float | None
    stress_duration_s: float | None
    dominant_stress_label: StressClass | None
    stress_transition_count: int | None
    stress_transition_count_per_hour: float | None
    valid_protocol_duration_s: float


def load_wesad_subject_pickle(path: Path | str) -> dict[str, Any]:
    """Load one WESAD `S*.pkl` file using the encoding expected by WESAD."""

    with Path(path).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")

    if "label" not in payload:
        raise ValueError(f"WESAD pickle has no label vector: {path}")

    return payload


def classify_wesad_window_labels(
    labels: Sequence[int] | np.ndarray,
    *,
    allowed_protocol_label_ids: Sequence[int] = (1, 2, 3),
    min_label_fraction: float = 1.0,
) -> WESADWindowLabel | None:
    """Return a clean protocol label for a label segment, or None if unusable."""

    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if label_array.size == 0:
        return None

    if not 0 < min_label_fraction <= 1:
        raise ValueError("min_label_fraction must be in (0, 1].")

    values, counts = np.unique(label_array, return_counts=True)
    best_index = int(np.argmax(counts))
    protocol_label_id = int(values[best_index])
    label_fraction = float(counts[best_index]) / float(label_array.size)

    allowed = set(int(label_id) for label_id in allowed_protocol_label_ids)
    is_clean = protocol_label_id in allowed and label_fraction >= min_label_fraction
    if not is_clean:
        return None

    stress_label = WESAD_PROTOCOL_LABELS.get(protocol_label_id)
    if stress_label is None:
        return None

    return WESADWindowLabel(
        protocol_label_id=protocol_label_id,
        stress_label=stress_label,
        label_fraction=label_fraction,
        is_clean=True,
    )


def build_monitoring_states_from_wesad_subject(
    pickle_path: Path | str,
    *,
    config: WESADWindowConfig = WESADWindowConfig(),
) -> list[MonitoringState]:
    """Build clean-window stress MonitoringState objects from one WESAD subject."""

    if not config.window_secs:
        raise ValueError("window_secs must contain at least one duration.")

    if config.label_sampling_rate_hz <= 0:
        raise ValueError("label_sampling_rate_hz must be positive.")

    if config.window_stride_sec is not None and config.window_stride_sec <= 0:
        raise ValueError("window_stride_sec must be positive when provided.")

    if config.max_windows_per_duration is not None and config.max_windows_per_duration < 0:
        raise ValueError("max_windows_per_duration must be non-negative when provided.")

    payload = load_wesad_subject_pickle(pickle_path)
    subject_id = _normalize_subject_id(payload.get("subject"), Path(pickle_path).stem)
    labels = np.asarray(payload["label"], dtype=int).reshape(-1)
    recording_duration_s = labels.size / float(config.label_sampling_rate_hz)
    bvp_signal = _extract_wrist_bvp(payload)
    subject_aggregates = compute_wesad_subject_aggregates(
        labels,
        allowed_protocol_label_ids=config.allowed_protocol_label_ids,
        sampling_rate_hz=config.label_sampling_rate_hz,
    )

    states: list[MonitoringState] = []

    for window_sec in config.window_secs:
        if window_sec <= 0:
            raise ValueError("All window durations must be positive.")

        states.extend(
            _build_states_for_duration(
                labels=labels,
                bvp_signal=bvp_signal,
                subject_id=subject_id,
                window_sec=window_sec,
                recording_duration_s=recording_duration_s,
                subject_aggregates=subject_aggregates,
                config=config,
            )
        )

    return states


def _build_states_for_duration(
    *,
    labels: np.ndarray,
    bvp_signal: np.ndarray | None,
    subject_id: str,
    window_sec: int,
    recording_duration_s: float,
    subject_aggregates: WESADSubjectAggregates,
    config: WESADWindowConfig,
) -> list[MonitoringState]:
    label_fs = int(config.label_sampling_rate_hz)
    window_samples = int(round(window_sec * label_fs))
    stride_sec = config.window_stride_sec or window_sec
    stride_samples = int(round(stride_sec * label_fs))

    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must resolve to positive sample counts.")

    states: list[MonitoringState] = []
    previous_state: MonitoringState | None = None
    raw_window_index = 0
    cursor = 0

    while cursor + window_samples <= labels.size:
        if (
            config.max_windows_per_duration is not None
            and len(states) >= config.max_windows_per_duration
        ):
            break

        window_labels = labels[cursor : cursor + window_samples]
        classified = classify_wesad_window_labels(
            window_labels,
            allowed_protocol_label_ids=config.allowed_protocol_label_ids,
            min_label_fraction=config.min_label_fraction,
        )

        if classified is not None:
            clean_index = len(states)
            start_s = cursor / float(label_fs)
            end_s = (cursor + window_samples) / float(label_fs)
            metadata = _build_window_metadata(
                subject_id=subject_id,
                raw_window_index=raw_window_index,
                window_start_label_sample=cursor,
                window_end_label_sample=cursor + window_samples,
                start_s=start_s,
                end_s=end_s,
                window_sec=window_sec,
                stride_sec=stride_sec,
                label_fs=label_fs,
                bvp_signal=bvp_signal,
                classified=classified,
                previous_state=previous_state,
                min_label_fraction=config.min_label_fraction,
            )
            state = MonitoringState(
                state_id=(
                    f"wesad_subject_{subject_id}_"
                    f"{window_sec}s_clean_window_{clean_index:04d}"
                ),
                patient_id=subject_id,
                dataset="wesad",
                modality="wearable",
                window_index=clean_index,
                window_start_s=start_s,
                window_duration_s=window_sec,
                stress_label=classified.stress_label,
                protocol_label_id=classified.protocol_label_id,
                previous_stress_label=(
                    previous_state.stress_label if previous_state is not None else None
                ),
                previous_protocol_label_id=(
                    previous_state.protocol_label_id
                    if previous_state is not None
                    else None
                ),
                stress_burden_ratio=subject_aggregates.stress_burden_ratio,
                stress_duration_s=subject_aggregates.stress_duration_s,
                dominant_stress_label=subject_aggregates.dominant_stress_label,
                stress_transition_count=subject_aggregates.stress_transition_count,
                stress_transition_count_per_hour=(
                    subject_aggregates.stress_transition_count_per_hour
                ),
                recording_duration_s=recording_duration_s,
                metadata=metadata,
            )

            states.append(state)
            previous_state = state

        cursor += stride_samples
        raw_window_index += 1

    return states


def compute_wesad_subject_aggregates(
    labels: Sequence[int] | np.ndarray,
    *,
    allowed_protocol_label_ids: Sequence[int] = (1, 2, 3),
    sampling_rate_hz: int = WESAD_LABEL_FS_HZ,
) -> WESADSubjectAggregates:
    """Compute subject-level stress aggregates over allowed protocol labels."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    label_array = np.asarray(labels, dtype=int).reshape(-1)
    allowed = set(int(label_id) for label_id in allowed_protocol_label_ids)
    valid_mask = np.isin(label_array, list(allowed))
    valid_labels = label_array[valid_mask]
    valid_duration_s = valid_labels.size / float(sampling_rate_hz)

    if valid_labels.size == 0:
        return WESADSubjectAggregates(
            stress_burden_ratio=None,
            stress_duration_s=None,
            dominant_stress_label=None,
            stress_transition_count=None,
            stress_transition_count_per_hour=None,
            valid_protocol_duration_s=0.0,
        )

    stress_samples = int(np.sum(valid_labels == 2))
    stress_duration_s = stress_samples / float(sampling_rate_hz)
    stress_burden_ratio = stress_samples / float(valid_labels.size)
    dominant_label_id = _dominant_protocol_label_id(valid_labels)
    dominant_stress_label = WESAD_PROTOCOL_LABELS.get(dominant_label_id)
    transition_count = _count_protocol_transitions(valid_labels)
    transition_count_per_hour = None
    if valid_duration_s > 0:
        transition_count_per_hour = transition_count / (valid_duration_s / 3600.0)

    return WESADSubjectAggregates(
        stress_burden_ratio=stress_burden_ratio,
        stress_duration_s=stress_duration_s,
        dominant_stress_label=dominant_stress_label,
        stress_transition_count=transition_count,
        stress_transition_count_per_hour=transition_count_per_hour,
        valid_protocol_duration_s=valid_duration_s,
    )


def _dominant_protocol_label_id(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    priority = {
        2: 3,  # stress
        3: 2,  # amusement
        4: 1,  # meditation
        1: 0,  # baseline
    }
    return int(
        max(
            zip(values, counts, strict=True),
            key=lambda item: (int(item[1]), priority.get(int(item[0]), -1)),
        )[0]
    )


def _count_protocol_transitions(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0

    collapsed = [int(labels[0])]
    for label in labels[1:]:
        label_id = int(label)
        if label_id != collapsed[-1]:
            collapsed.append(label_id)

    return max(0, len(collapsed) - 1)


def _normalize_subject_id(subject_raw: Any, fallback: str) -> str:
    if isinstance(subject_raw, bytes):
        subject_raw = subject_raw.decode("utf-8", errors="replace")

    subject_id = str(subject_raw or fallback).strip()
    return subject_id or fallback


def _extract_wrist_bvp(payload: dict[str, Any]) -> np.ndarray | None:
    signal = payload.get("signal")
    if not isinstance(signal, dict):
        return None

    wrist = signal.get("wrist")
    if not isinstance(wrist, dict):
        return None

    bvp = wrist.get("BVP")
    if bvp is None:
        return None

    return np.asarray(bvp)


def _build_window_metadata(
    *,
    subject_id: str,
    raw_window_index: int,
    window_start_label_sample: int,
    window_end_label_sample: int,
    start_s: float,
    end_s: float,
    window_sec: int,
    stride_sec: int,
    label_fs: int,
    bvp_signal: np.ndarray | None,
    classified: WESADWindowLabel,
    previous_state: MonitoringState | None,
    min_label_fraction: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_source": "wesad",
        "subject_id": subject_id,
        "raw_window_index": raw_window_index,
        "window_start_label_sample": window_start_label_sample,
        "window_end_label_sample": window_end_label_sample,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "window_sec": window_sec,
        "window_stride_sec": stride_sec,
        "window_overlap": stride_sec < window_sec,
        "label_sampling_rate_hz": label_fs,
        "protocol_label_id": classified.protocol_label_id,
        "protocol_label_name": classified.stress_label,
        "protocol_label_fraction": classified.label_fraction,
        "clean_window_definition": (
            "dominant allowed protocol label fraction >= "
            f"{min_label_fraction}"
        ),
        "ignored_protocol_labels": dict(IGNORED_WESAD_LABELS),
        "previous_clean_window_state_id": (
            previous_state.state_id if previous_state is not None else None
        ),
        "aggregate_scope": "subject_allowed_protocol_labels",
    }

    if bvp_signal is not None:
        bvp_start = int(round(start_s * WESAD_WRIST_BVP_FS_HZ))
        bvp_end = int(round(end_s * WESAD_WRIST_BVP_FS_HZ))
        bvp_end = min(bvp_end, int(bvp_signal.shape[0]))
        metadata.update(
            {
                "wrist_bvp_sampling_rate_hz": WESAD_WRIST_BVP_FS_HZ,
                "wrist_bvp_total_samples": int(bvp_signal.shape[0]),
                "wrist_bvp_window_start_sample": bvp_start,
                "wrist_bvp_window_end_sample": bvp_end,
                "wrist_bvp_window_samples": max(0, bvp_end - bvp_start),
            }
        )

    return metadata


__all__ = [
    "WESAD_LABEL_FS_HZ",
    "WESAD_WRIST_BVP_FS_HZ",
    "WESAD_PROTOCOL_LABELS",
    "IGNORED_WESAD_LABELS",
    "WESADWindowConfig",
    "WESADWindowLabel",
    "WESADSubjectAggregates",
    "load_wesad_subject_pickle",
    "classify_wesad_window_labels",
    "compute_wesad_subject_aggregates",
    "build_monitoring_states_from_wesad_subject",
]

