from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from agent.encoder.ecg_signal_processor import VitalParams
from agent.mhealth.schemas import MonitoringState
from agent.mhealth.state_builders import ppg_dalia_state_builder as builder
from agent.mhealth.state_builders.ppg_dalia_state_builder import (
    PPGDaLiAWindowConfig,
    build_monitoring_states_from_ppg_dalia_subject,
)
from agent.benchmarks.vitalbench.state_to_qa_generator import (
    generate_qa_from_states,
    write_jsonl,
)


def _write_synthetic_ppg_dalia_pickle(
    path: Path,
    *,
    hr_labels: np.ndarray,
    activity_labels: np.ndarray | None,
    bvp_hr_labels: np.ndarray | None = None,
    subject: str | bytes = "S_TEST",
) -> None:
    duration_s = hr_labels.size / 0.5
    bvp_samples = int(round(duration_s * 64))
    acc_samples = int(round(duration_s * 32))
    chest_samples = int(round(duration_s * 700))
    bvp_reference = np.asarray(
        bvp_hr_labels if bvp_hr_labels is not None else np.nan_to_num(hr_labels, nan=72.0),
        dtype=float,
    ).reshape(-1)
    bvp_times = np.arange(bvp_samples, dtype=float) / 64.0
    bvp_label_indices = np.minimum((bvp_times * 0.5).astype(int), bvp_reference.size - 1)
    bvp_signal = bvp_reference[bvp_label_indices].reshape(-1, 1)

    payload = {
        "subject": subject,
        "label": np.asarray(hr_labels, dtype=float),
        "signal": {
            "wrist": {
                "BVP": bvp_signal,
                "ACC": np.zeros((acc_samples, 3), dtype=float),
                "EDA": np.zeros((int(round(duration_s * 4)), 1), dtype=float),
                "TEMP": np.zeros((int(round(duration_s * 4)), 1), dtype=float),
            },
            "chest": {
                "ECG": np.zeros((chest_samples, 1), dtype=float),
                "Resp": np.zeros((chest_samples, 1), dtype=float),
                "ACC": np.zeros((chest_samples, 3), dtype=float),
            },
        },
    }
    if activity_labels is not None:
        payload["activity"] = np.asarray(activity_labels, dtype=float).reshape(-1, 1)

    with path.open("wb") as f:
        pickle.dump(payload, f)


class _MeanEncodedBVPProcessor:
    def __init__(self, sampling_rate: int = 64) -> None:
        self.sampling_rate = sampling_rate

    def extract_vitals(self, signal, fs=None, accel=None):
        del fs, accel
        arr = np.asarray(signal, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return VitalParams(signal_quality_score=0.0)
        hr = round(float(np.mean(finite)), 1)
        interval = 60_000.0 / hr if hr > 0 else 1000.0
        return VitalParams(
            hr_bpm=hr,
            rr_intervals_ms=[interval, interval, interval, interval],
            sdnn_ms=12.3,
            rmssd_ms=18.4,
            signal_quality_score=0.87,
        )


def _patch_bvp_processor(monkeypatch) -> None:
    monkeypatch.setattr(builder, "PPGSignalProcessor", _MeanEncodedBVPProcessor)


def test_schema_accepts_ppg_dalia_dataset() -> None:
    state = MonitoringState(
        state_id="ppg_dalia_subject_S_TEST_30s_clean_window_0000",
        patient_id="S_TEST",
        dataset="ppg_dalia",
        modality="ppg",
        hr_bpm=72.0,
        activity_label="sitting",
        mean_hr_bpm=72.0,
        ecg_reference_hr_bpm=72.0,
    )

    assert state.dataset == "ppg_dalia"
    assert state.modality == "ppg"
    assert state.activity_label == "sitting"
    assert state.mean_hr_bpm == 72.0
    assert state.ecg_reference_hr_bpm == 72.0


def test_ppg_dalia_builder_creates_bvp_hr_activity_states(tmp_path: Path, monkeypatch) -> None:
    _patch_bvp_processor(monkeypatch)
    # Four 30s windows. HR labels are 0.5 Hz, so 15 labels per 30s window.
    hr_labels = np.concatenate(
        [
            np.full(15, 70.0),
            np.full(15, 80.0),
            np.full(15, 95.0),
            np.full(15, 110.0),
        ]
    )
    activity_labels = np.concatenate(
        [
            np.full(120, 1),  # sitting
            np.full(120, 1),
            np.full(120, 7),  # walking
            np.full(120, 7),
        ]
    )
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_ppg_dalia_pickle(
        pkl_path,
        hr_labels=hr_labels,
        activity_labels=activity_labels,
    )

    states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(window_secs=(30,)),
    )

    assert len(states) == 4
    assert {state.dataset for state in states} == {"ppg_dalia"}
    assert {state.modality for state in states} == {"ppg"}
    assert [state.hr_bpm for state in states] == [70.0, 80.0, 95.0, 110.0]
    assert [state.mean_hr_bpm for state in states] == [70.0, 80.0, 95.0, 110.0]
    assert [state.ecg_reference_hr_bpm for state in states] == [
        70.0,
        80.0,
        95.0,
        110.0,
    ]
    assert [state.max_hr_bpm for state in states] == [70.0, 80.0, 95.0, 110.0]
    assert [state.previous_hr_bpm for state in states] == [None, 70.0, 80.0, 95.0]
    assert [state.activity_label for state in states] == [
        "sitting",
        "sitting",
        "walking",
        "walking",
    ]
    assert [state.previous_activity_label for state in states] == [
        None,
        "sitting",
        "sitting",
        "walking",
    ]
    assert states[0].metadata["dataset_source"] == "ppg_dalia"
    assert states[0].metadata["source_variant"] == "uci_original_pickle"
    assert states[0].metadata["reference_signal"] == "chest_ecg"
    assert states[0].metadata["hr_label_source"] == "ecg_derived"
    assert states[0].metadata["hr_source"] == "wrist_bvp_peak_detection"
    assert states[0].metadata["activity_label_source"] == "protocol_activity"
    assert states[0].metadata["bvp_sampling_rate_hz"] == 64
    assert states[0].metadata["wrist_bvp_window_samples"] == 30 * 64
    assert states[0].sdnn_ms == 12.3
    assert states[0].rmssd_ms == 18.4
    assert states[0].signal_quality_score == 0.87


def test_ppg_dalia_builder_records_custom_hr_provenance(tmp_path: Path, monkeypatch) -> None:
    _patch_bvp_processor(monkeypatch)
    hr_labels = np.full(15, 75.0)
    activity_labels = np.full(120, 1)
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_ppg_dalia_pickle(
        pkl_path,
        hr_labels=hr_labels,
        activity_labels=activity_labels,
    )

    states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(
            window_secs=(30,),
            source_variant="edge_impulse_e4_csv_converted",
            hr_label_source="e4_hr_csv",
            reference_signal="empatica_e4_hr",
        ),
    )

    assert len(states) == 1
    state = states[0]
    assert state.hr_bpm == 75.0
    assert state.mean_hr_bpm == 75.0
    assert state.ecg_reference_hr_bpm is None
    assert state.metadata["source_variant"] == "edge_impulse_e4_csv_converted"
    assert state.metadata["hr_label_source"] == "e4_hr_csv"
    assert state.metadata["reference_signal"] == "empatica_e4_hr"


def test_ppg_dalia_builder_min_activity_fraction_filters_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_bvp_processor(monkeypatch)
    hr_labels = np.concatenate([np.full(15, 70.0), np.full(15, 80.0)])
    activity_labels = np.concatenate(
        [
            np.concatenate([np.full(84, 1), np.full(36, 7)]),  # 70% sitting
            np.full(120, 7),  # 100% walking
        ]
    )
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_ppg_dalia_pickle(
        pkl_path,
        hr_labels=hr_labels,
        activity_labels=activity_labels,
    )

    states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(
            window_secs=(30,),
            min_activity_fraction=0.8,
        ),
    )

    assert len(states) == 1
    assert states[0].activity_label == "walking"
    assert states[0].previous_hr_bpm is None
    assert states[0].metadata["raw_window_index"] == 1


def test_ppg_dalia_builder_can_include_transition_windows(tmp_path: Path, monkeypatch) -> None:
    _patch_bvp_processor(monkeypatch)
    hr_labels = np.full(15, 72.0)
    activity_labels = np.full(120, 0)
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_ppg_dalia_pickle(
        pkl_path,
        hr_labels=hr_labels,
        activity_labels=activity_labels,
    )

    default_states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(window_secs=(30,)),
    )
    included_states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(
            window_secs=(30,),
            include_transition_windows=True,
        ),
    )

    assert default_states == []
    assert len(included_states) == 1
    assert included_states[0].activity_label == "transition"


def test_ppg_dalia_builder_keeps_bvp_windows_without_finite_ecg_hr_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_bvp_processor(monkeypatch)
    hr_labels = np.concatenate([np.full(15, np.nan), np.full(15, 82.0)])
    bvp_hr_labels = np.concatenate([np.full(15, 72.0), np.full(15, 82.0)])
    activity_labels = np.full(240, 1)
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_ppg_dalia_pickle(
        pkl_path,
        hr_labels=hr_labels,
        bvp_hr_labels=bvp_hr_labels,
        activity_labels=activity_labels,
    )

    states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(window_secs=(30,)),
    )

    assert len(states) == 2
    assert states[0].hr_bpm == 72.0
    assert states[0].ecg_reference_hr_bpm is None
    assert states[0].previous_hr_bpm is None
    assert states[1].hr_bpm == 82.0
    assert states[1].ecg_reference_hr_bpm == 82.0
    assert states[1].previous_hr_bpm == 72.0
    assert states[1].metadata["raw_window_index"] == 1


def test_ppg_dalia_states_generate_hr_qa_and_serialize_jsonl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_bvp_processor(monkeypatch)
    hr_labels = np.concatenate([np.full(15, 70.0), np.full(15, 108.0)])
    activity_labels = np.concatenate([np.full(120, 1), np.full(120, 7)])
    pkl_path = tmp_path / "S_TEST.pkl"
    output_path = tmp_path / "ppg_dalia_samples.jsonl"
    _write_synthetic_ppg_dalia_pickle(
        pkl_path,
        hr_labels=hr_labels,
        activity_labels=activity_labels,
    )

    states = build_monitoring_states_from_ppg_dalia_subject(
        pkl_path,
        config=PPGDaLiAWindowConfig(window_secs=(30,)),
    )
    samples = generate_qa_from_states(states)
    write_jsonl(samples, output_path)

    assert samples
    assert {sample.dataset for sample in samples} == {"ppg_dalia"}
    assert {sample.modality for sample in samples} == {"ppg"}

    template_ids = {sample.template_id for sample in samples}
    assert "ta2_heart_rate_category_current" in template_ids
    assert "ta3_heart_rate_query_current" in template_ids
    assert "ta4_heart_rate_higher_than_previous" in template_ids
    assert "tb3_heart_rate_excursion_threshold_monitoring_window" in template_ids
    assert "tb4_highest_heart_rate_query_monitoring_window" in template_ids
    assert "ta1_af_presence_current" not in template_ids
    assert "ta6_stress_presence_current" not in template_ids

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == len(samples)
    assert records[0]["dataset"] == "ppg_dalia"
    for sample in samples:
        record = sample.to_jsonl_record()
        assert record["window_start_s"] is not None
        assert record["window_duration_s"] == 30.0
    for record in records:
        assert record["window_start_s"] is not None
        assert record["window_duration_s"] == 30.0


