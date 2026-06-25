from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent.data.ppg_loader import (
    ECGReferenceRecord,
    PPGChunk,
    PPGRecording,
    RelativeTimestamp,
)
from agent.mhealth.state_builders import ppg_state_builder as builder
from agent.mhealth.state_builders.ppg_state_builder import (
    AFInterval,
    PPGWindowConfig,
    build_af_intervals_from_ecg_reference,
    build_monitoring_states_from_ppg_patient,
    compute_af_burden_for_window,
)
from agent.benchmarks.vitalbench.state_to_qa_generator import generate_qa_from_states


def _synthetic_ppg(duration_s: float, hr_bpm: float = 60.0, fs: int = 100) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    f0 = hr_bpm / 60.0
    return (
        np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(4 * np.pi * f0 * t)
        + 0.2 * np.sin(6 * np.pi * f0 * t)
    )


def _synthetic_ecg_reference() -> ECGReferenceRecord:
    return ECGReferenceRecord(
        patient_id="001",
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
        fs=1,
        n_samples=36,
        qrs_index=np.array([0, 12, 24, 36], dtype=int),
        rr_seconds=np.array([12.0, 12.0, 12.0], dtype=float),
        af_annotation=np.array([1, 0, 1], dtype=int),
    )


def _synthetic_ppg_recording() -> PPGRecording:
    fs = 100
    duration_s = 36.0
    signal = _synthetic_ppg(duration_s=duration_s, hr_bpm=60.0, fs=fs)
    n_accel = int(duration_s * 50)
    accel = np.zeros(n_accel, dtype=float)
    chunk = PPGChunk(
        patient_id="001",
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
        ppg_fs=fs,
        accel_fs=50,
        n_ppg_samples=len(signal),
        green_signal=signal,
        ambient_signal=np.zeros_like(signal),
        accel_x=accel,
        accel_y=accel,
        accel_z=accel,
    )
    return PPGRecording(patient_id="001", chunks=[chunk])


def _single_chunk_ppg_recording(
    *,
    signal: np.ndarray,
    patient_id: str = "001",
    start_time: str = "00:00:00",
    fs: int = 100,
) -> PPGRecording:
    accel = np.zeros(int(len(signal) * 0.5), dtype=float)
    chunk = PPGChunk(
        patient_id=patient_id,
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text=start_time),
        ppg_fs=fs,
        accel_fs=50,
        n_ppg_samples=len(signal),
        green_signal=signal,
        ambient_signal=np.zeros_like(signal),
        accel_x=accel,
        accel_y=accel,
        accel_z=accel,
    )
    return PPGRecording(patient_id=patient_id, chunks=[chunk])


def _ecg_reference_covering(end_s: int, patient_id: str = "001") -> ECGReferenceRecord:
    return ECGReferenceRecord(
        patient_id=patient_id,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
        fs=1,
        n_samples=end_s,
        qrs_index=np.array([0, end_s], dtype=int),
        rr_seconds=np.array([float(end_s)], dtype=float),
        af_annotation=np.array([0], dtype=int),
    )


class _MeanBasedPPGProcessor:
    def extract_vitals(self, signal, fs=None, accel=None):
        hr = 120.0 if float(np.mean(signal)) > 0.5 else 60.0
        return SimpleNamespace(
            hr_bpm=hr,
            sdnn_ms=10.0,
            rmssd_ms=5.0,
            signal_quality_score=0.9,
        )


def test_build_af_intervals_from_ecg_reference_uses_qrs_index() -> None:
    ecg = _synthetic_ecg_reference()

    intervals = build_af_intervals_from_ecg_reference(ecg)

    assert intervals == [
        AFInterval(start_s=0.0, end_s=12.0, is_af=True),
        AFInterval(start_s=12.0, end_s=24.0, is_af=False),
        AFInterval(start_s=24.0, end_s=36.0, is_af=True),
    ]


def test_compute_af_burden_for_window_handles_zero_full_and_partial() -> None:
    intervals = [
        AFInterval(start_s=0.0, end_s=10.0, is_af=False),
        AFInterval(start_s=10.0, end_s=20.0, is_af=True),
    ]

    assert compute_af_burden_for_window(intervals, 0.0, 10.0) == pytest.approx(0.0)
    assert compute_af_burden_for_window(intervals, 10.0, 20.0) == pytest.approx(1.0)
    assert compute_af_burden_for_window(intervals, 5.0, 15.0) == pytest.approx(0.5)


def test_ppg_state_builder_creates_grounded_states(monkeypatch) -> None:
    ecg = _synthetic_ecg_reference()
    ppg = _synthetic_ppg_recording()

    class FakePPGLoader:
        def __init__(self, dataset_root: Path | str | None = None):
            self.dataset_root = dataset_root

        def read_ecg_record(self, patient_id, include_signal: bool = False):
            return ecg

        def read_ppg_record(self, patient_id, include_signals: bool = False):
            return ppg

    monkeypatch.setattr(builder, "PPGLoader", FakePPGLoader)

    states = build_monitoring_states_from_ppg_patient(
        dataset_root=Path("unused"),
        patient_id="001",
        config=PPGWindowConfig(window_sec=12, max_windows=3),
    )

    assert len(states) == 3
    assert {state.dataset for state in states} == {"afppgecg"}
    assert {state.modality for state in states} == {"ppg"}
    assert [state.rhythm_class for state in states] == ["AF", "N", "AF"]
    assert states[0].previous_hr_bpm is None
    assert states[0].previous_rhythm_class is None
    assert states[1].previous_hr_bpm == states[0].hr_bpm
    assert states[1].previous_rhythm_class == "AF"
    assert states[2].previous_rhythm_class == "N"
    assert all(state.hr_bpm is not None for state in states)
    assert all(state.sdnn_ms is not None for state in states)
    assert all(state.rmssd_ms is not None for state in states)
    assert all(state.signal_quality_score is not None for state in states)
    assert all(
        state.metadata["rhythm_gt_source"] == "aligned_ecg_af_annotation"
        for state in states
    )
    assert all(state.metadata["hr_source"] == "ppg_peak_detection" for state in states)
    assert all(
        state.metadata["max_hr_bpm_source"] == "max_ppg_peak_hr_over_subwindows"
        for state in states
    )
    assert all(
        state.metadata["rhythm_transition_source"] == "aligned_ecg_reference_annotation"
        for state in states
    )
    assert all(
        state.metadata["recording_aggregate_scope"]
        == "aligned_ecg_time_span_between_retained_ppg_windows"
        for state in states
    )

    aggregate_burdens = {state.af_burden_ratio for state in states}
    transition_rates = {state.rhythm_transition_count_per_hour for state in states}
    max_hrs = {state.max_hr_bpm for state in states}

    assert len(aggregate_burdens) == 1
    assert next(iter(aggregate_burdens)) == pytest.approx(2 / 3)
    assert len(transition_rates) == 1
    assert next(iter(transition_rates)) == pytest.approx(200.0)
    assert max_hrs == {max(state.hr_bpm for state in states)}
    assert all(state.recording_duration_s == pytest.approx(36.0) for state in states)

    samples = generate_qa_from_states(states)

    assert samples
    assert all(sample.dataset == "afppgecg" for sample in samples)
    assert all(sample.modality == "ppg" for sample in samples)
    assert any("On PPG datasets" in sample.notes for sample in samples)

    for sample in samples:
        json.dumps(sample.to_jsonl_record(), ensure_ascii=False)


def test_ppg_max_hr_uses_subwindow_peak_not_window_mean(monkeypatch) -> None:
    fs = 100
    signal = np.zeros(300 * fs, dtype=float)
    signal[-30 * fs :] = 1.0
    ecg = _ecg_reference_covering(300)
    ppg = _single_chunk_ppg_recording(signal=signal, fs=fs)

    class FakePPGLoader:
        def __init__(self, dataset_root: Path | str | None = None):
            self.dataset_root = dataset_root

        def read_ecg_record(self, patient_id, include_signal: bool = False):
            return ecg

        def read_ppg_record(self, patient_id, include_signals: bool = False):
            return ppg

    monkeypatch.setattr(builder, "PPGLoader", FakePPGLoader)
    monkeypatch.setattr(builder, "PPGSignalProcessor", _MeanBasedPPGProcessor)

    states = build_monitoring_states_from_ppg_patient(
        dataset_root=Path("unused"),
        patient_id="001",
        config=PPGWindowConfig(window_sec=300, max_hr_subwindow_sec=30),
    )

    assert len(states) == 1
    assert states[0].hr_bpm == pytest.approx(60.0)
    assert states[0].max_hr_bpm == pytest.approx(120.0)


def test_ppg_low_annotation_coverage_window_is_skipped(monkeypatch) -> None:
    fs = 100
    signal = np.zeros(300 * fs, dtype=float)
    ecg = _ecg_reference_covering(10)
    ppg = _single_chunk_ppg_recording(signal=signal, fs=fs)

    class FakePPGLoader:
        def __init__(self, dataset_root: Path | str | None = None):
            self.dataset_root = dataset_root

        def read_ecg_record(self, patient_id, include_signal: bool = False):
            return ecg

        def read_ppg_record(self, patient_id, include_signals: bool = False):
            return ppg

    monkeypatch.setattr(builder, "PPGLoader", FakePPGLoader)
    monkeypatch.setattr(builder, "PPGSignalProcessor", _MeanBasedPPGProcessor)

    states = build_monitoring_states_from_ppg_patient(
        dataset_root=Path("unused"),
        patient_id="001",
        config=PPGWindowConfig(window_sec=300),
    )

    assert states == []


def test_ppg_chunk_gap_uses_ecg_time_axis_and_resets_previous_state(monkeypatch) -> None:
    fs = 100
    first = np.zeros(30 * fs, dtype=float)
    second = np.ones(30 * fs, dtype=float)
    accel = np.zeros(15 * fs, dtype=float)
    chunks = [
        PPGChunk(
            patient_id="001",
            chunk_index=0,
            start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
            ppg_fs=fs,
            accel_fs=50,
            n_ppg_samples=len(first),
            green_signal=first,
            ambient_signal=np.zeros_like(first),
            accel_x=accel,
            accel_y=accel,
            accel_z=accel,
        ),
        PPGChunk(
            patient_id="001",
            chunk_index=1,
            start_timestamp=RelativeTimestamp(day_index=1, time_text="01:00:00"),
            ppg_fs=fs,
            accel_fs=50,
            n_ppg_samples=len(second),
            green_signal=second,
            ambient_signal=np.zeros_like(second),
            accel_x=accel,
            accel_y=accel,
            accel_z=accel,
        ),
    ]
    ecg = _ecg_reference_covering(3630)
    ppg = PPGRecording(patient_id="001", chunks=chunks)

    class FakePPGLoader:
        def __init__(self, dataset_root: Path | str | None = None):
            self.dataset_root = dataset_root

        def read_ecg_record(self, patient_id, include_signal: bool = False):
            return ecg

        def read_ppg_record(self, patient_id, include_signals: bool = False):
            return ppg

    monkeypatch.setattr(builder, "PPGLoader", FakePPGLoader)
    monkeypatch.setattr(builder, "PPGSignalProcessor", _MeanBasedPPGProcessor)

    states = build_monitoring_states_from_ppg_patient(
        dataset_root=Path("unused"),
        patient_id="001",
        config=PPGWindowConfig(window_sec=30),
    )

    assert len(states) == 2
    assert states[0].window_start_s == pytest.approx(0.0)
    assert states[1].window_start_s == pytest.approx(3600.0)
    assert states[1].metadata["ppg_stream_offset_s"] == pytest.approx(30.0)
    assert states[1].metadata["previous_gap_s"] == pytest.approx(3570.0)
    assert states[1].previous_hr_bpm is None
    assert states[1].previous_rhythm_class is None

