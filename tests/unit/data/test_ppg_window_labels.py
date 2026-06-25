from __future__ import annotations

from datetime import datetime

import numpy as np

from agent.data.ppg_loader import (
    ECGReferenceRecord,
    PPGChunk,
    PPGRecording,
    RelativeTimestamp,
)
from agent.data.ppg_window_labels import (
    build_rhythm_segments,
    build_simultaneous_intervals,
    format_ppg_window_id,
    label_time_window,
    parse_ppg_window_id,
    transition_counts,
)


def _ecg_record() -> ECGReferenceRecord:
    return ECGReferenceRecord(
        patient_id="001",
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00", anchor=datetime(2000, 1, 1)),
        fs=1,
        n_samples=5,
        qrs_index=np.array([0, 1, 2, 3, 4], dtype=int),
        rr_seconds=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        af_annotation=np.array([0, 0, 1, 1], dtype=int),
    )


def _ppg_recording() -> PPGRecording:
    chunk = PPGChunk(
        patient_id="001",
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00", anchor=datetime(2000, 1, 1)),
        ppg_fs=10,
        accel_fs=10,
        n_ppg_samples=40,
    )
    return PPGRecording(patient_id="001", chunks=[chunk])


def test_format_ppg_window_id_matches_streaming_convention() -> None:
    result = format_ppg_window_id(
        patient_id="001",
        chunk_index=0,
        window_index=2,
        start_sample=10,
        end_sample=30,
    )

    assert result == "ppg:001:chunk000:w000002:s000000010-e000000030"


def test_parse_ppg_window_id_round_trips_formatted_value() -> None:
    parsed = parse_ppg_window_id("ppg:001:chunk000:w000002:s000000010-e000000030")

    assert parsed.patient_id == "001"
    assert parsed.chunk_index == 0
    assert parsed.window_index == 2
    assert parsed.start_sample == 10
    assert parsed.end_sample == 30


def test_build_rhythm_segments_and_transition_counts() -> None:
    segments = build_rhythm_segments(_ecg_record())

    assert [(segment.start_s, segment.end_s, segment.label) for segment in segments] == [
        (0.0, 2.0, 0),
        (2.0, 4.0, 1),
    ]
    assert transition_counts(segments) == {
        "total": 1,
        "non_af_to_af": 1,
        "af_to_non_af": 0,
    }


def test_build_simultaneous_intervals_captures_overlap() -> None:
    intervals = build_simultaneous_intervals(_ecg_record(), _ppg_recording())

    assert len(intervals) == 1
    assert intervals[0].start_s == 0.0
    assert intervals[0].end_s == 4.0


def test_label_time_window_marks_transition_and_eligibility() -> None:
    segments = build_rhythm_segments(_ecg_record())

    transition = label_time_window(
        window_id="w0",
        window_start_s=1.0,
        window_end_s=3.0,
        segments=segments,
        af_min=0.8,
        nsr_max=0.05,
    )
    pure_af = label_time_window(
        window_id="w1",
        window_start_s=2.0,
        window_end_s=4.0,
        segments=segments,
        af_min=0.8,
        nsr_max=0.05,
    )

    assert transition.fully_covered is True
    assert transition.rhythm_label == "transition"
    assert transition.qa_eligible is False
    assert transition.af_burden == 0.5

    assert pure_af.rhythm_label == "AF"
    assert pure_af.qa_eligible is True
    assert pure_af.af_burden == 1.0
