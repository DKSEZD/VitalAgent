"""Helpers for deriving window-level labels on AFPPGECG PPG windows."""

from __future__ import annotations

from dataclasses import dataclass
import re

from agent.data.ppg_loader import ECGReferenceRecord, PPGRecording

EPS = 1e-9


@dataclass(frozen=True)
class SimultaneousInterval:
    chunk_index: int
    chunk_offset_s: float
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class RhythmSegment:
    start_s: float
    end_s: float
    label: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class WindowLabelResult:
    window_id: str
    af_burden: float
    rhythm_label: str
    qa_eligible: bool
    fully_covered: bool


@dataclass(frozen=True)
class ParsedPPGWindowID:
    patient_id: str
    chunk_index: int
    window_index: int
    start_sample: int
    end_sample: int


_WINDOW_ID_RE = re.compile(
    r"^ppg:(?P<patient_id>[^:]+):chunk(?P<chunk_index>\d{3}):"
    r"w(?P<window_index>\d{6}):s(?P<start_sample>\d{9})-e(?P<end_sample>\d{9})$"
)


def format_ppg_window_id(
    *,
    patient_id: str,
    chunk_index: int,
    window_index: int,
    start_sample: int,
    end_sample: int,
) -> str:
    return (
        f"ppg:{patient_id}:chunk{chunk_index:03d}:"
        f"w{window_index:06d}:s{start_sample:09d}-e{end_sample:09d}"
    )


def parse_ppg_window_id(window_id: str) -> ParsedPPGWindowID:
    match = _WINDOW_ID_RE.match(str(window_id).strip())
    if match is None:
        raise ValueError(f"Invalid PPG window_id: {window_id!r}")

    parsed = ParsedPPGWindowID(
        patient_id=match.group("patient_id"),
        chunk_index=int(match.group("chunk_index")),
        window_index=int(match.group("window_index")),
        start_sample=int(match.group("start_sample")),
        end_sample=int(match.group("end_sample")),
    )
    if parsed.end_sample <= parsed.start_sample:
        raise ValueError(
            f"Invalid PPG window_id {window_id!r}: end_sample must be greater than start_sample."
        )
    return parsed


def build_simultaneous_intervals(
    ecg: ECGReferenceRecord,
    ppg_record: PPGRecording,
) -> list[SimultaneousInterval]:
    intervals: list[SimultaneousInterval] = []
    for chunk in ppg_record.chunks:
        offset_s = (chunk.start_time - ecg.start_time).total_seconds()
        start_s = max(0.0, offset_s)
        end_s = min(ecg.duration_seconds, offset_s + chunk.duration_seconds)
        if end_s > start_s:
            intervals.append(
                SimultaneousInterval(
                    chunk_index=chunk.chunk_index,
                    chunk_offset_s=offset_s,
                    start_s=start_s,
                    end_s=end_s,
                )
            )
    return intervals


def build_rhythm_segments(ecg: ECGReferenceRecord) -> list[RhythmSegment]:
    qrs_times_s = ecg.qrs_index.astype(float) / float(ecg.fs)
    labels = ecg.af_annotation.astype(int)

    n_intervals = min(len(labels), len(qrs_times_s) - 1)
    if n_intervals <= 0:
        return []

    starts_s = qrs_times_s[:n_intervals]
    ends_s = qrs_times_s[1 : n_intervals + 1]
    labels = labels[:n_intervals]

    boundaries = [0]
    for idx in range(1, n_intervals):
        if labels[idx] != labels[idx - 1]:
            boundaries.append(idx)
    boundaries.append(n_intervals)

    segments: list[RhythmSegment] = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end_idx <= start_idx:
            continue
        segments.append(
            RhythmSegment(
                start_s=float(starts_s[start_idx]),
                end_s=float(ends_s[end_idx - 1]),
                label=int(labels[start_idx]),
            )
        )
    return segments


def transition_counts(segments: list[RhythmSegment]) -> dict[str, int]:
    total = 0
    non_af_to_af = 0
    af_to_non_af = 0
    for previous, current in zip(segments[:-1], segments[1:], strict=True):
        if previous.label == current.label:
            continue
        total += 1
        if previous.label == 0 and current.label == 1:
            non_af_to_af += 1
        elif previous.label == 1 and current.label == 0:
            af_to_non_af += 1
    return {
        "total": total,
        "non_af_to_af": non_af_to_af,
        "af_to_non_af": af_to_non_af,
    }


def iter_window_starts(
    interval: SimultaneousInterval,
    *,
    window_seconds: float,
    stride_seconds: float,
) -> list[float]:
    if stride_seconds <= 0:
        raise ValueError(f"stride_seconds must be positive, got {stride_seconds!r}")
    starts: list[float] = []
    start_s = interval.start_s
    latest_start = interval.end_s - window_seconds
    while start_s <= latest_start + EPS:
        starts.append(round(start_s, 9))
        start_s += stride_seconds
    return starts


def compute_af_burden(
    window_start_s: float,
    window_end_s: float,
    segments: list[RhythmSegment],
) -> tuple[float, bool]:
    covered_s = 0.0
    af_s = 0.0

    for segment in segments:
        overlap_start = max(window_start_s, segment.start_s)
        overlap_end = min(window_end_s, segment.end_s)
        if overlap_end <= overlap_start:
            continue
        overlap_s = overlap_end - overlap_start
        covered_s += overlap_s
        if segment.label == 1:
            af_s += overlap_s

    window_duration_s = window_end_s - window_start_s
    fully_covered = covered_s >= (window_duration_s - EPS)
    if window_duration_s <= 0:
        return 0.0, False
    return af_s / window_duration_s, fully_covered


def label_time_window(
    *,
    window_id: str,
    window_start_s: float,
    window_end_s: float,
    segments: list[RhythmSegment],
    af_min: float = 0.80,
    nsr_max: float = 0.05,
) -> WindowLabelResult:
    af_burden, fully_covered = compute_af_burden(window_start_s, window_end_s, segments)

    if not fully_covered:
        return WindowLabelResult(
            window_id=window_id,
            af_burden=af_burden,
            rhythm_label="unlabeled",
            qa_eligible=False,
            fully_covered=False,
        )

    if af_burden <= nsr_max + EPS:
        return WindowLabelResult(
            window_id=window_id,
            af_burden=af_burden,
            rhythm_label="NSR",
            qa_eligible=True,
            fully_covered=True,
        )
    if af_burden >= af_min - EPS:
        return WindowLabelResult(
            window_id=window_id,
            af_burden=af_burden,
            rhythm_label="AF",
            qa_eligible=True,
            fully_covered=True,
        )
    return WindowLabelResult(
        window_id=window_id,
        af_burden=af_burden,
        rhythm_label="transition",
        qa_eligible=False,
        fully_covered=True,
    )
