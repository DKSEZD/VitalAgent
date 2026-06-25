from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence

from agent.data.icentia11k_loader import RawECGData
from agent.mhealth.schemas import MonitoringState


MonitoringRhythm = Literal["N", "AF", "AFL"]


ICENTIA_TO_MONITORING_RHYTHM: dict[str, MonitoringRhythm] = {
    "N": "N",
    "AFIB": "AF",
    "AF": "AF",
    # Temporary v0.1 evaluation convention: collapse atrial flutter into the
    # AF class so Tier B can target atrial-arrhythmia switching.
    "AFL": "AF",
}

RHYTHM_LABEL_MAPPING_VERSION = "icentia11k_v1_afl_collapsed_to_af"


# Deterministic tie-breaker for dominant rhythm.
# If two rhythms occupy exactly the same duration, prefer clinically more
# salient labels.
#
# IMPORTANT: This is a screening-oriented bias by design. At exact 50/50
# overlap, AF wins over N. This means generated GT slightly favours
# positive AF labels at ambiguous boundaries. Document this in the paper
# method section. v0.2 should consider dropping windows whose top-2
# rhythms are within e.g. 0.45-0.55 of total duration as
# "ambiguous_rhythm_window" rather than relying on this priority.
RHYTHM_PRIORITY: dict[MonitoringRhythm, int] = {
    "AF": 2,
    "AFL": 1,
    "N": 0,
}


@dataclass(frozen=True)
class RhythmInterval:
    """A rhythm-labelled interval in sample indices.

    Icentia/WFDB annotations are sample-based. Therefore start_sample and
    end_sample are sample indices, not seconds.

    Intervals are half-open: [start_sample, end_sample).
    """

    start_sample: int
    end_sample: int
    label: str


@dataclass(frozen=True)
class IcentiaWindowConfig:
    """Configuration for converting Icentia11k segment annotations to states.

    Window durations are specified in seconds, while ECG samples and rhythm
    intervals are represented by sample indices.
    """

    sampling_rate_hz: int = 250
    current_window_sec: int = 300
    previous_window_sec: int = 300
    window_stride_sec: int = 300
    max_hr_subwindow_sec: int = 30
    rhythm_min_overlap_sec: float = 2.0


@dataclass(frozen=True)
class IcentiaSegmentAggregates:
    """Monitoring-window aggregate fields copied to each window state."""

    af_burden_ratio: Optional[float]
    max_hr_bpm: Optional[float]
    rhythm_transition_count_per_hour: Optional[float]


def map_icentia_rhythm(label: Optional[str]) -> Optional[MonitoringRhythm]:
    """Map Icentia11k rhythm labels to MonitoringState rhythm classes.

    Icentia11k rhythm aux_note labels are typically:
    - N
    - AFIB
    - AFL

    In the current QA-generation convention, AFL is intentionally collapsed to
    AF so the benchmark can ask about atrial-arrhythmia switching. This should
    be revisited if AF and AFL are evaluated as separate clinical classes.

    Unknown, missing, or empty labels are returned as None so that
    rhythm-dependent QA templates can be skipped.
    """

    if label is None:
        return None

    normalized = label.strip()

    if not normalized:
        return None

    # WFDB aux_note rhythm starts may appear as "(N", "(AFIB", "(AFL".
    if normalized.startswith("("):
        normalized = normalized[1:].strip()

    if not normalized or normalized == ")":
        return None

    return ICENTIA_TO_MONITORING_RHYTHM.get(normalized)


def parse_rhythm_intervals_from_aux_notes(
    annotation_samples: Sequence[int],
    aux_notes: Sequence[str],
    *,
    segment_end_sample: int,
) -> list[RhythmInterval]:
    """Parse WFDB aux_note rhythm markers into rhythm intervals.

    Icentia11k rhythm annotations use start markers such as "(N", "(AFIB",
    "(AFL", and optional end markers such as ")" or "AFIB)".

    This function does not require wfdb directly. You can pass:
    - ann.sample
    - ann.aux_note
    after reading the annotation file with wfdb.rdann(...).

    Robustness:
    - Empty "(" markers are skipped.
    - If a new "(X" appears before the previous rhythm closes, the previous
      one is implicitly closed at the new start sample.
    - If the last rhythm never closes, it is closed at segment_end_sample.
    - Annotations sharing the same sample index are kept in their original
      input order using stable sorting by (sample, input_index).
    """

    if len(annotation_samples) != len(aux_notes):
        raise ValueError("annotation_samples and aux_notes must have the same length.")

    intervals: list[RhythmInterval] = []

    active_label: Optional[str] = None
    active_start: Optional[int] = None

    indexed = list(enumerate(zip(annotation_samples, aux_notes)))
    indexed.sort(key=lambda item: (item[1][0], item[0]))

    for _, (sample, aux_note) in indexed:
        sample = int(sample)
        note = (aux_note or "").strip()

        if not note:
            continue

        if note.startswith("("):
            new_label = note[1:].strip()

            # Skip lone "(" with no rhythm label attached.
            if not new_label:
                continue

            # If a new rhythm begins before the previous one explicitly
            # closes, close the previous interval at this sample.
            if (
                active_label is not None
                and active_start is not None
                and sample > active_start
            ):
                intervals.append(
                    RhythmInterval(
                        start_sample=active_start,
                        end_sample=sample,
                        label=active_label,
                    )
                )

            active_label = new_label
            active_start = sample

        elif note.endswith(")"):
            if (
                active_label is not None
                and active_start is not None
                and sample > active_start
            ):
                intervals.append(
                    RhythmInterval(
                        start_sample=active_start,
                        end_sample=sample,
                        label=active_label,
                    )
                )

            active_label = None
            active_start = None

    # If the last rhythm interval remains open, close it at the segment end.
    if (
        active_label is not None
        and active_start is not None
        and segment_end_sample > active_start
    ):
        intervals.append(
            RhythmInterval(
                start_sample=active_start,
                end_sample=int(segment_end_sample),
                label=active_label,
            )
        )

    return intervals


def compute_hr_bpm(
    beat_samples: Sequence[int],
    start_sample: int,
    end_sample: int,
    sampling_rate_hz: int = 250,
) -> Optional[float]:
    """Compute average HR over a sample interval.

    Returns:
    - None if the interval duration is invalid.
    - None if the interval contains zero detected beats. This is treated
      as "unmeasurable" rather than "0 bpm" because Icentia11k beat
      detectors typically emit no beats during severe noise or signal
      loss, and a 0 bpm answer would be clinically wrong.
    """

    duration_sec = (end_sample - start_sample) / sampling_rate_hz

    if duration_sec <= 0:
        return None

    beat_count = sum(start_sample <= beat < end_sample for beat in beat_samples)

    if beat_count == 0:
        return None

    return beat_count / (duration_sec / 60.0)


def _overlap_samples(
    interval: RhythmInterval,
    start_sample: int,
    end_sample: int,
) -> int:
    overlap_start = max(start_sample, interval.start_sample)
    overlap_end = min(end_sample, interval.end_sample)
    return max(0, overlap_end - overlap_start)


def dominant_rhythm(
    intervals: Sequence[RhythmInterval],
    start_sample: int,
    end_sample: int,
    *,
    sampling_rate_hz: int = 250,
    min_overlap_sec: float = 2.0,
) -> Optional[MonitoringRhythm]:
    """Return the rhythm with the largest valid overlap duration.

    Notes
    -----
    - Intervals whose overlap with the query window is below min_overlap_sec
      are filtered out as noise. This means very short paroxysmal AF episodes
      within a window will not flip the dominant rhythm.
    - Ties are broken by RHYTHM_PRIORITY: AF > AFL > N.
    """

    min_overlap_samples = int(min_overlap_sec * sampling_rate_hz)
    duration_by_label: dict[MonitoringRhythm, int] = {}

    for interval in intervals:
        overlap = _overlap_samples(interval, start_sample, end_sample)

        if overlap < min_overlap_samples:
            continue

        mapped = map_icentia_rhythm(interval.label)

        if mapped is None:
            continue

        duration_by_label[mapped] = duration_by_label.get(mapped, 0) + overlap

    if not duration_by_label:
        return None

    return max(
        duration_by_label.items(),
        key=lambda item: (item[1], RHYTHM_PRIORITY[item[0]]),
    )[0]


def compute_af_burden_ratio(
    intervals: Sequence[RhythmInterval],
    start_sample: int,
    end_sample: int,
    *,
    sampling_rate_hz: int = 250,
    min_overlap_sec: float = 2.0,
) -> Optional[float]:
    """Compute AF burden as AF duration / valid rhythm-labelled duration."""

    min_overlap_samples = int(min_overlap_sec * sampling_rate_hz)

    total_duration = 0
    af_duration = 0

    for interval in intervals:
        overlap = _overlap_samples(interval, start_sample, end_sample)

        if overlap < min_overlap_samples:
            continue

        mapped = map_icentia_rhythm(interval.label)

        if mapped is None:
            continue

        total_duration += overlap

        if mapped == "AF":
            af_duration += overlap

    if total_duration == 0:
        return None

    return af_duration / total_duration


def compute_max_hr_bpm(
    beat_samples: Sequence[int],
    start_sample: int,
    end_sample: int,
    *,
    sampling_rate_hz: int = 250,
    subwindow_sec: int = 30,
) -> Optional[float]:
    """Compute max HR using short subwindows.

    Subwindows with zero beats are skipped, so silent / noisy stretches do not
    artificially pull the max HR estimate.
    """

    subwindow_samples = int(subwindow_sec * sampling_rate_hz)

    if subwindow_samples <= 0:
        return None

    if end_sample <= start_sample:
        return None

    values: list[float] = []

    cursor = start_sample

    while cursor < end_sample:
        sub_end = min(cursor + subwindow_samples, end_sample)

        hr = compute_hr_bpm(
            beat_samples=beat_samples,
            start_sample=cursor,
            end_sample=sub_end,
            sampling_rate_hz=sampling_rate_hz,
        )

        if hr is not None:
            values.append(hr)

        cursor = sub_end

    if not values:
        return None

    return max(values)


def compute_rhythm_transition_count_per_hour(
    intervals: Sequence[RhythmInterval],
    start_sample: int,
    end_sample: int,
    *,
    sampling_rate_hz: int = 250,
    min_overlap_sec: float = 2.0,
) -> Optional[float]:
    """Compute rhythm transition count normalized per hour.

    Repeated consecutive labels are collapsed, so:
    N -> N -> AF -> AF -> N
    counts as 2 transitions.

    Intended scope
    --------------
    This metric is intended for segment-level aggregation. Results on short
    windows are unstable: a single transition inside a 5-minute window
    extrapolates to 12 transitions/hour, which is not meaningful.
    """

    duration_hours = (end_sample - start_sample) / sampling_rate_hz / 3600.0

    if duration_hours <= 0:
        return None

    min_overlap_samples = int(min_overlap_sec * sampling_rate_hz)

    collapsed_labels: list[MonitoringRhythm] = []

    for interval in sorted(intervals, key=lambda x: x.start_sample):
        overlap = _overlap_samples(interval, start_sample, end_sample)

        if overlap < min_overlap_samples:
            continue

        mapped = map_icentia_rhythm(interval.label)

        if mapped is None:
            continue

        if not collapsed_labels or collapsed_labels[-1] != mapped:
            collapsed_labels.append(mapped)

    transition_count = max(0, len(collapsed_labels) - 1)

    return transition_count / duration_hours


def compute_segment_aggregates(
    *,
    beat_samples: Sequence[int],
    rhythm_intervals: Sequence[RhythmInterval],
    segment_start_sample: int,
    segment_end_sample: int,
    config: IcentiaWindowConfig,
) -> IcentiaSegmentAggregates:
    """Compute aggregate fields for Tier B monitoring-window questions."""

    return IcentiaSegmentAggregates(
        af_burden_ratio=compute_af_burden_ratio(
            rhythm_intervals,
            segment_start_sample,
            segment_end_sample,
            sampling_rate_hz=config.sampling_rate_hz,
            min_overlap_sec=config.rhythm_min_overlap_sec,
        ),
        max_hr_bpm=compute_max_hr_bpm(
            beat_samples,
            segment_start_sample,
            segment_end_sample,
            sampling_rate_hz=config.sampling_rate_hz,
            subwindow_sec=config.max_hr_subwindow_sec,
        ),
        rhythm_transition_count_per_hour=compute_rhythm_transition_count_per_hour(
            rhythm_intervals,
            segment_start_sample,
            segment_end_sample,
            sampling_rate_hz=config.sampling_rate_hz,
            min_overlap_sec=config.rhythm_min_overlap_sec,
        ),
    )


def build_monitoring_state_from_icentia_window(
    *,
    patient_id: str,
    segment_id: str,
    window_index: int,
    beat_samples: Sequence[int],
    rhythm_intervals: Sequence[RhythmInterval],
    current_start_sample: int,
    segment_start_sample: int,
    segment_end_sample: int,
    config: IcentiaWindowConfig = IcentiaWindowConfig(),
    segment_aggregates: Optional[IcentiaSegmentAggregates] = None,
) -> MonitoringState:
    """Build one MonitoringState from one current window.

    The current window must fit entirely inside the segment. Truncated windows
    are rejected to keep window-level fields comparable across states.
    """

    sr = config.sampling_rate_hz

    current_window_samples = int(config.current_window_sec * sr)
    previous_window_samples = int(config.previous_window_sec * sr)

    if current_window_samples <= 0:
        raise ValueError("current_window_sec must be positive.")

    if current_start_sample < segment_start_sample:
        raise ValueError("current_start_sample is before segment_start_sample.")

    current_end_sample = current_start_sample + current_window_samples

    if current_end_sample > segment_end_sample:
        raise ValueError(
            "Current window does not fit inside the segment. "
            "Refusing to build a truncated window state."
        )

    previous_start_sample = current_start_sample - previous_window_samples

    hr_bpm = compute_hr_bpm(
        beat_samples=beat_samples,
        start_sample=current_start_sample,
        end_sample=current_end_sample,
        sampling_rate_hz=sr,
    )

    rhythm_class = dominant_rhythm(
        rhythm_intervals,
        current_start_sample,
        current_end_sample,
        sampling_rate_hz=sr,
        min_overlap_sec=config.rhythm_min_overlap_sec,
    )

    if previous_start_sample < segment_start_sample:
        previous_hr_bpm = None
        previous_rhythm_class = None
    else:
        previous_hr_bpm = compute_hr_bpm(
            beat_samples=beat_samples,
            start_sample=previous_start_sample,
            end_sample=current_start_sample,
            sampling_rate_hz=sr,
        )

        previous_rhythm_class = dominant_rhythm(
            rhythm_intervals,
            previous_start_sample,
            current_start_sample,
            sampling_rate_hz=sr,
            min_overlap_sec=config.rhythm_min_overlap_sec,
        )

    if segment_aggregates is None:
        segment_aggregates = compute_segment_aggregates(
            beat_samples=beat_samples,
            rhythm_intervals=rhythm_intervals,
            segment_start_sample=segment_start_sample,
            segment_end_sample=segment_end_sample,
            config=config,
        )

    window_start_s = (current_start_sample - segment_start_sample) / sr
    window_duration_s = (current_end_sample - current_start_sample) / sr

    return MonitoringState(
        state_id=(
            f"icentia11k_patient_{patient_id}_"
            f"segment_{segment_id}_window_{window_index}"
        ),
        patient_id=str(patient_id),
        dataset="icentia11k",
        modality="ecg",
        window_index=window_index,
        window_start_s=window_start_s,
        window_duration_s=window_duration_s,
        hr_bpm=hr_bpm,
        previous_hr_bpm=previous_hr_bpm,
        rhythm_class=rhythm_class,
        previous_rhythm_class=previous_rhythm_class,
        af_burden_ratio=segment_aggregates.af_burden_ratio,
        max_hr_bpm=segment_aggregates.max_hr_bpm,
        rhythm_transition_count_per_hour=(
            segment_aggregates.rhythm_transition_count_per_hour
        ),
        metadata={
            "segment_id": str(segment_id),
            "segment_start_sample": segment_start_sample,
            "segment_end_sample": segment_end_sample,
            "current_start_sample": current_start_sample,
            "current_end_sample": current_end_sample,
            "previous_start_sample": (
                previous_start_sample
                if previous_start_sample >= segment_start_sample
                else None
            ),
            "sampling_rate_hz": sr,
            "current_window_sec": config.current_window_sec,
            "previous_window_sec": config.previous_window_sec,
            "window_stride_sec": config.window_stride_sec,
            "max_hr_subwindow_sec": config.max_hr_subwindow_sec,
            "rhythm_min_overlap_sec": config.rhythm_min_overlap_sec,
            "rhythm_class_source": "wfdb_aux_note_dominant_overlap",
            "rhythm_label_mapping_version": RHYTHM_LABEL_MAPPING_VERSION,
            "aggregate_scope": "monitoring_window_level",
            "aggregate_scope_samples": segment_end_sample - segment_start_sample,
            "aggregate_scope_seconds": (
                segment_end_sample - segment_start_sample
            )
            / sr,
        },
    )


def build_monitoring_states_from_icentia_segment(
    *,
    patient_id: str,
    segment_id: str,
    beat_samples: Sequence[int],
    rhythm_intervals: Sequence[RhythmInterval],
    segment_start_sample: int,
    segment_end_sample: int,
    config: IcentiaWindowConfig = IcentiaWindowConfig(),
) -> list[MonitoringState]:
    """Build multiple MonitoringState objects from one Icentia11k segment."""

    sr = config.sampling_rate_hz
    stride_samples = int(config.window_stride_sec * sr)
    current_window_samples = int(config.current_window_sec * sr)

    if stride_samples <= 0:
        raise ValueError("window_stride_sec must be positive.")

    if current_window_samples <= 0:
        raise ValueError("current_window_sec must be positive.")

    if segment_end_sample <= segment_start_sample:
        raise ValueError("segment_end_sample must be greater than segment_start_sample.")

    segment_aggregates = compute_segment_aggregates(
        beat_samples=beat_samples,
        rhythm_intervals=rhythm_intervals,
        segment_start_sample=segment_start_sample,
        segment_end_sample=segment_end_sample,
        config=config,
    )

    states: list[MonitoringState] = []

    cursor = segment_start_sample
    window_index = 0

    while cursor + current_window_samples <= segment_end_sample:
        states.append(
            build_monitoring_state_from_icentia_window(
                patient_id=patient_id,
                segment_id=segment_id,
                window_index=window_index,
                beat_samples=beat_samples,
                rhythm_intervals=rhythm_intervals,
                current_start_sample=cursor,
                segment_start_sample=segment_start_sample,
                segment_end_sample=segment_end_sample,
                config=config,
                segment_aggregates=segment_aggregates,
            )
        )

        cursor += stride_samples
        window_index += 1

    return states


def rhythm_intervals_from_loader_output(raw: RawECGData) -> list[RhythmInterval]:
    """Convert Icentia11kLoader RhythmAnnotation to state-builder intervals.

    The current Icentia11kLoader returns rhythm annotation positions relative
    to the returned RawECGData signal window. Therefore this adapter also
    produces local sample indices in [0, raw.n_samples].
    """

    if raw.rhythm_annotations is None:
        return []

    intervals: list[RhythmInterval] = []

    for start, end, rhythm_type in zip(
        raw.rhythm_annotations.start_samples,
        raw.rhythm_annotations.end_samples,
        raw.rhythm_annotations.rhythm_types,
        strict=False,
    ):
        start_i = int(start)
        end_i = int(end)

        # Skip malformed or zero-length spans rather than letting them affect
        # aggregation metrics.
        if end_i <= start_i:
            continue

        intervals.append(
            RhythmInterval(
                start_sample=start_i,
                end_sample=end_i,
                label=str(rhythm_type),
            )
        )

    return intervals


def beat_samples_from_loader_output(raw: RawECGData) -> list[int]:
    """Convert Icentia11kLoader BeatAnnotation to state-builder beat samples.

    The current Icentia11kLoader returns beat samples relative to the returned
    RawECGData signal window.
    """

    if raw.beat_annotations is None:
        return []

    return [int(sample) for sample in raw.beat_annotations.samples]


def build_monitoring_states_from_icentia_raw(
    raw: RawECGData,
    *,
    config: IcentiaWindowConfig = IcentiaWindowConfig(),
) -> list[MonitoringState]:
    """Build MonitoringState objects from existing Icentia11kLoader output.

    RawECGData returned by Icentia11kLoader uses local sample coordinates for
    beat and rhythm annotations. Tier A current/previous-window fields and
    Tier B aggregate fields are therefore computed over the returned monitoring
    window, not necessarily the full source segment. The source segment length
    is preserved in metadata when available.
    """

    if int(raw.fs) != int(config.sampling_rate_hz):
        config = replace(config, sampling_rate_hz=int(raw.fs))

    beat_samples = beat_samples_from_loader_output(raw)
    rhythm_intervals = rhythm_intervals_from_loader_output(raw)

    states = build_monitoring_states_from_icentia_segment(
        patient_id=raw.patient_id,
        segment_id=raw.segment_id,
        beat_samples=beat_samples,
        rhythm_intervals=rhythm_intervals,
        segment_start_sample=0,
        segment_end_sample=int(raw.n_samples),
        config=config,
    )

    for state in states:
        state.state_id = (
            f"icentia11k_patient_{raw.patient_id}_"
            f"segment_{raw.segment_id}_samples_{raw.sample_start}_{raw.sample_end}_"
            f"window_{state.window_index}"
        )
        state.metadata.update(
            {
                "source_sample_start": int(raw.sample_start),
                "source_sample_end": int(raw.sample_end),
                "source_segment_total_samples": int(raw.segment_total_samples),
                "aggregate_scope": "monitoring_window_level",
                "aggregate_scope_samples": int(raw.n_samples),
                "aggregate_scope_seconds": int(raw.n_samples) / int(raw.fs),
            }
        )

    return states


__all__ = [
    "MonitoringRhythm",
    "ICENTIA_TO_MONITORING_RHYTHM",
    "RHYTHM_PRIORITY",
    "RhythmInterval",
    "IcentiaWindowConfig",
    "IcentiaSegmentAggregates",
    "map_icentia_rhythm",
    "parse_rhythm_intervals_from_aux_notes",
    "compute_hr_bpm",
    "dominant_rhythm",
    "compute_af_burden_ratio",
    "compute_max_hr_bpm",
    "compute_rhythm_transition_count_per_hour",
    "compute_segment_aggregates",
    "build_monitoring_state_from_icentia_window",
    "build_monitoring_states_from_icentia_segment",
    "rhythm_intervals_from_loader_output",
    "beat_samples_from_loader_output",
    "build_monitoring_states_from_icentia_raw",
]

