from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import agent.tools.mhealth_state_build_tools as build_tools
import agent.tools.mhealth_signal_state_build_tools as signal_build_tools
import agent.tools.mhealth_state_tools as state_tools
from agent.state.mhealth_state_store import get_global_state_store
from agent.tools.registry import list_tools


@pytest.fixture(autouse=True)
def clear_global_store():
    get_global_state_store().clear()
    yield
    get_global_state_store().clear()


def test_state_build_tools_are_registered() -> None:
    names = {tool["name"] for tool in list_tools()}

    assert {
        "state_build_from_ecg_record",
        "state_build_from_ppg_patient",
        "state_build_from_ppg_dalia_pickle",
        "state_build_from_wesad_pickle",
    }.issubset(names)


def test_wesad_raw_to_state_direct_smoke_when_sample_exists() -> None:
    sample_path = Path("dataset/wesad_sample/WESAD/S11/S11.pkl")
    if not sample_path.exists():
        pytest.skip("WESAD local sample is not available.")

    result = build_tools.state_build_from_wesad_pickle(
        str(sample_path),
        max_windows=2,
    )

    assert result["success"] is True
    assert result["dataset"] == "wesad"
    assert result["state_count"] > 0
    assert result["contexts"][0]["subject_id"] == "S11"

    current = state_tools.state_get_current_monitoring_state(
        dataset="wesad",
        subject_id="S11",
        recording_id="S11",
        timestamp_s=120,
    )
    assert current["success"] is True
    assert current["state"]["stress_label"] in {"baseline", "stress", "amusement"}


class _FakeRateSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def dropna(self) -> np.ndarray:
        return self._values


@pytest.fixture
def ecg_record_fixture() -> SimpleNamespace:
    sampling_rate = 100
    signal = np.linspace(0.0, 1.0, sampling_rate * 10, dtype=np.float32)
    return SimpleNamespace(
        record_id="fixture-ecg-001",
        lead_names=["I"],
        num_leads=1,
        sampling_rate=sampling_rate,
        duration_seconds=10.0,
        metadata={"patient_id": "patient-ecg-001", "source": "unit_fixture"},
        get_lead=lambda lead: signal if lead == "I" else None,
        signal=signal.reshape(-1, 1),
    )


def test_ecg_signal_raw_to_state_direct_smoke(
    monkeypatch: pytest.MonkeyPatch,
    ecg_record_fixture: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        signal_build_tools.ecg_tools,
        "_load_record",
        lambda *_args, **_kwargs: ecg_record_fixture,
    )
    monkeypatch.setattr(
        signal_build_tools.ecg_tools,
        "process_ecg",
        lambda signal, sr: (
            {"ECG_Rate": _FakeRateSeries([60.0, 62.0, 64.0])},
            {"ECG_R_Peaks": np.array([0, 100, 200, 300, 400, 500])},
        ),
    )
    monkeypatch.setattr(
        signal_build_tools.ecg_tools.nk,
        "hrv_frequency",
        lambda *_args, **_kwargs: SimpleNamespace(columns=[]),
    )

    result = signal_build_tools.state_build_from_ecg_record("fixture-ecg-001")

    assert result["success"] is True
    assert result["dataset"] == "icentia11k"
    assert result["record_id"] == "fixture-ecg-001"
    assert result["state_count"] == 1
    assert "source_semantics" in result
    assert (
        "rhythm_class and AF fields are left unset"
        in result["source_semantics"]["not_inferred"]
    )

    current = state_tools.state_get_current_monitoring_state(
        dataset="icentia11k",
        subject_id="patient-ecg-001",
        recording_id="fixture-ecg-001",
        timestamp_s=5,
    )
    assert current["success"] is True
    assert current["state"]["hr_bpm"] == 62.0
    assert "rhythm_class" not in current["state"]
    assert "af_burden_ratio" not in current["state"]


def test_signal_build_defaults_accumulate_existing_contexts(
    monkeypatch: pytest.MonkeyPatch,
    ecg_record_fixture: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        signal_build_tools.ecg_tools,
        "_load_record",
        lambda *_args, **_kwargs: ecg_record_fixture,
    )
    monkeypatch.setattr(
        signal_build_tools.ecg_tools,
        "process_ecg",
        lambda signal, sr: (
            {"ECG_Rate": _FakeRateSeries([60.0, 62.0, 64.0])},
            {"ECG_R_Peaks": np.array([0, 100, 200, 300, 400, 500])},
        ),
    )
    monkeypatch.setattr(
        signal_build_tools.ecg_tools.nk,
        "hrv_frequency",
        lambda *_args, **_kwargs: SimpleNamespace(columns=[]),
    )

    first = signal_build_tools.state_build_from_ecg_record("fixture-ecg-001")
    second = signal_build_tools.state_build_from_ecg_record("fixture-ecg-002")

    assert first["success"] is True
    assert second["success"] is True
    assert {
        context["recording_id"]
        for context in get_global_state_store().list_contexts()
    } == {"fixture-ecg-001", "fixture-ecg-002"}


def test_signal_builds_share_store_by_default(
    monkeypatch: pytest.MonkeyPatch,
    ecg_record_fixture: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        signal_build_tools.ecg_tools,
        "_load_record",
        lambda *_args, **_kwargs: ecg_record_fixture,
    )
    monkeypatch.setattr(
        signal_build_tools.ecg_tools,
        "process_ecg",
        lambda signal, sr: (
            {"ECG_Rate": _FakeRateSeries([60.0, 62.0, 64.0])},
            {"ECG_R_Peaks": np.array([0, 100, 200, 300, 400, 500])},
        ),
    )
    monkeypatch.setattr(
        signal_build_tools.ecg_tools.nk,
        "hrv_frequency",
        lambda *_args, **_kwargs: SimpleNamespace(columns=[]),
    )

    first = signal_build_tools.state_build_from_ecg_record("fixture-ecg-001")
    second = signal_build_tools.state_build_from_ecg_record("fixture-ecg-002")

    assert first["success"] is True
    assert second["success"] is True
    assert {
        context["recording_id"]
        for context in get_global_state_store().list_contexts()
    } == {"fixture-ecg-001", "fixture-ecg-002"}


def test_ppg_signal_raw_to_state_direct_smoke_when_sample_exists() -> None:
    sample_root = Path("dataset/zenodo_af_monitoring_v11242869/raw")
    patient_id = "001"
    if not (
        (sample_root / f"{patient_id}_ECG.mat").exists()
        and (sample_root / f"{patient_id}_PPG.mat").exists()
    ):
        pytest.skip("AFPPGECG local sample is not available.")

    result = signal_build_tools.state_build_from_ppg_patient(
        patient_id=patient_id,
        dataset_root=str(sample_root),
        max_windows=2,
    )

    assert result["success"] is True
    assert result["dataset"] == "afppgecg"
    assert result["patient_id"] == patient_id
    assert result["state_count"] > 0
    assert "PPG peak detection" in result["source_semantics"]["signal_derived"]
    assert (
        "aligned ECG reference annotation"
        in result["source_semantics"]["reference_annotation"]
    )

    context = result["contexts"][0]
    timestamp_s = float(context["start_s"] or 0.0)
    current = state_tools.state_get_current_monitoring_state(
        dataset="afppgecg",
        subject_id=str(context["subject_id"]),
        timestamp_s=timestamp_s,
        fields=["hr_bpm", "signal_quality_score"],
    )

    assert current["success"] is True
    assert (
        current["state"].get("hr_bpm") is not None
        or current["state"].get("signal_quality_score") is not None
    )


def test_ppg_signal_state_builder_validates_parameters() -> None:
    bad_window = signal_build_tools.state_build_from_ppg_patient(
        patient_id="001",
        window_sec=0,
    )
    bad_threshold = signal_build_tools.state_build_from_ppg_patient(
        patient_id="001",
        rhythm_threshold=1.1,
    )
    bad_coverage = signal_build_tools.state_build_from_ppg_patient(
        patient_id="001",
        min_annotation_coverage_ratio=1.1,
    )

    assert bad_window["success"] is False
    assert "window_sec must be positive" in bad_window["error"]
    assert bad_threshold["success"] is False
    assert "rhythm_threshold must be between 0 and 1" in bad_threshold["error"]
    assert bad_coverage["success"] is False
    assert (
        "min_annotation_coverage_ratio must be between 0 and 1"
        in bad_coverage["error"]
    )
