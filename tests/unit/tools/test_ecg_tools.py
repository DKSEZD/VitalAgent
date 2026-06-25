from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent.config import reset_config
from agent.tools.ecg import tools as ecg_tools


@pytest.fixture(autouse=True)
def reset_config_cache() -> None:
    reset_config()
    yield
    reset_config()


def test_add_interval_range_flags_marks_above_normal() -> None:
    payload = {"rr_mean_ms": 1100.0}

    result = ecg_tools._add_interval_range_flags(
        payload,
        value_key="rr_mean_ms",
        lower_normal_ms=600.0,
        upper_normal_ms=1000.0,
    )

    assert result["normal_range_ms"] == {"lower": 600.0, "upper": 1000.0}
    assert result["is_below_normal"] is False
    assert result["is_above_normal"] is True
    assert result["is_within_normal"] is False


def test_add_interval_range_flags_leaves_missing_values_unchanged() -> None:
    payload = {}

    result = ecg_tools._add_interval_range_flags(
        payload,
        value_key="rr_mean_ms",
        lower_normal_ms=600.0,
        upper_normal_ms=1000.0,
    )

    assert result == {}


def test_resample_signal_to_length_interpolates_and_casts_float32() -> None:
    signal = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=float)

    result = ecg_tools._resample_signal_to_length(signal, target_length=5)

    assert result.shape == (2, 5)
    assert result.dtype == np.float32
    assert np.allclose(result[:, 0], np.array([0.0, 2.0], dtype=np.float32))
    assert np.allclose(result[:, -1], np.array([2.0, 4.0], dtype=np.float32))


def test_resample_signal_to_length_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="Expected 2D signal array"):
        ecg_tools._resample_signal_to_length(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="too short"):
        ecg_tools._resample_signal_to_length(np.array([[1.0], [2.0]]))


def test_z_score_normalize_returns_centered_float32_array() -> None:
    result = ecg_tools._z_score_normalize(np.array([1.0, 2.0, 3.0], dtype=float))

    assert result.dtype == np.float32
    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(result)) == pytest.approx(1.0, rel=1e-6)


def test_list_ecg_records_returns_repository_records(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = SimpleNamespace(
        list_records=lambda **_kwargs: [
            {"record_id": "101", "patient_id": None, "recording_date": None, "source": "ptbxl"},
            {"record_id": "apple_watch:aw-1", "patient_id": "watch-user", "recording_date": None, "source": "apple_watch"},
        ]
    )
    monkeypatch.setattr(ecg_tools, "get_record_repository", lambda: fake_repo)

    result = ecg_tools.list_ecg_records(limit=2)

    assert result == {
        "success": True,
        "record_ids": ["101", "apple_watch:aw-1"],
        "records": [
            {"record_id": "101", "patient_id": None, "recording_date": None, "source": "ptbxl"},
            {"record_id": "apple_watch:aw-1", "patient_id": "watch-user", "recording_date": None, "source": "apple_watch"},
        ],
        "count": 2,
    }
class _FakeRateSeries:
    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=float)

    def dropna(self) -> np.ndarray:
        return self._values


def _single_lead_record(fs: int = 100) -> SimpleNamespace:
    signal = np.linspace(0.0, 1.0, fs * 10, dtype=np.float32)
    return SimpleNamespace(
        record_id="apple_watch:aw-test",
        lead_names=["I"],
        num_leads=1,
        sampling_rate=fs,
        duration_seconds=10.0,
        metadata={"source": "apple_watch"},
        get_lead=lambda lead: signal if lead == "I" else None,
        signal=signal.reshape(-1, 1),
    )


def test_analyze_heart_rate_defaults_to_lead_i_for_single_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _single_lead_record()
    monkeypatch.setattr(ecg_tools, "_load_record", lambda *_args, **_kwargs: record)
    monkeypatch.setattr(
        ecg_tools,
        "process_ecg",
        lambda signal, sr: (
            {"ECG_Rate": _FakeRateSeries([60.0, 61.0, 62.0])},
            {"ECG_R_Peaks": np.array([0, 100, 200, 300])},
        ),
    )

    result = ecg_tools.analyze_heart_rate("apple_watch:aw-test")

    assert result["success"] is True
    assert result["metadata"]["lead_analyzed"] == "I"


def test_analyze_hrv_defaults_to_lead_i_for_single_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _single_lead_record()
    monkeypatch.setattr(ecg_tools, "_load_record", lambda *_args, **_kwargs: record)
    monkeypatch.setattr(
        ecg_tools,
        "process_ecg",
        lambda signal, sr: (
            {},
            {"ECG_R_Peaks": np.array([0, 100, 200, 300, 400, 500])},
        ),
    )
    monkeypatch.setattr(ecg_tools.nk, "hrv_frequency", lambda *_args, **_kwargs: SimpleNamespace(columns=[]))

    result = ecg_tools.analyze_hrv("apple_watch:aw-test")

    assert result["success"] is True
    assert result["metadata"]["lead_analyzed"] == "I"


def test_analyze_morphology_defaults_to_lead_i_for_single_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _single_lead_record(fs=500)
    monkeypatch.setattr(ecg_tools, "_load_record", lambda *_args, **_kwargs: record)
    monkeypatch.setattr(
        ecg_tools,
        "process_ecg",
        lambda signal, sr: ({}, {"ECG_R_Peaks": np.array([100, 300, 500])}),
    )
    monkeypatch.setattr(
        ecg_tools.nk,
        "ecg_delineate",
        lambda *_args, **_kwargs: (
            None,
            {
                "ECG_R_Onsets": [90, 290, 490],
                "ECG_R_Offsets": [120, 320, 520],
                "ECG_P_Onsets": [60, 260, 460],
                "ECG_T_Offsets": [180, 380, 580],
            },
        ),
    )

    result = ecg_tools.analyze_morphology("apple_watch:aw-test")

    assert result["success"] is True
    assert result["metadata"]["lead_analyzed"] == "I"


def test_assess_signal_quality_defaults_to_lead_i_for_single_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _single_lead_record()
    monkeypatch.setattr(ecg_tools, "_load_record", lambda *_args, **_kwargs: record)
    monkeypatch.setattr(ecg_tools.nk, "ecg_quality", lambda *_args, **_kwargs: ["Excellent", "Excellent"])

    result = ecg_tools.assess_signal_quality("apple_watch:aw-test")

    assert result["success"] is True
    assert result["metadata"]["lead_analyzed"] == "I"


def test_analyze_lead_morphology_handles_single_lead_record(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _single_lead_record()
    monkeypatch.setattr(ecg_tools, "_load_record", lambda *_args, **_kwargs: record)
    monkeypatch.setattr(
        ecg_tools,
        "analyze_single_lead",
        lambda signal, sr, lead_name: {
            "lead": lead_name,
            "st_segment": {"status": "normal"},
            "t_wave": {"status": "upright"},
            "q_wave": {"present": False, "pathological": False},
            "r_wave": {"mean_amplitude_mv": 0.8},
        },
    )

    result = ecg_tools.analyze_lead_morphology("apple_watch:aw-test")

    assert result["success"] is True
    assert result["metadata"]["leads_analyzed"] == ["I"]
    assert len(result["data"]["per_lead"]) == 1


def test_ecg_diagnosis_auto_switches_single_lead_mode_for_one_lead_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def __init__(self, array: np.ndarray):
            self.array = np.asarray(array, dtype=np.float32)

        def unsqueeze(self, axis: int) -> "FakeTensor":
            return FakeTensor(np.expand_dims(self.array, axis))

        def to(self, _device: object) -> "FakeTensor":
            return self

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.array

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeTorch:
        @staticmethod
        def from_numpy(array: np.ndarray) -> FakeTensor:
            return FakeTensor(array)

        @staticmethod
        def sigmoid(tensor: FakeTensor) -> FakeTensor:
            return FakeTensor(1.0 / (1.0 + np.exp(-tensor.array)))

        @staticmethod
        def no_grad() -> FakeNoGrad:
            return FakeNoGrad()

    class FakeModel:
        def __call__(self, _input_tensor: FakeTensor) -> FakeTensor:
            return FakeTensor(np.zeros((1, 1), dtype=np.float32))

    record = _single_lead_record(fs=500)
    runtime_calls: list[str] = []
    monkeypatch.setattr(ecg_tools, "_load_record", lambda *_args, **_kwargs: record)
    monkeypatch.setattr(
        ecg_tools,
        "load_ecgfounder_runtime",
        lambda mode="12_lead": runtime_calls.append(mode) or {
            "mode": mode,
            "torch": FakeTorch(),
            "device": "cpu",
            "model": FakeModel(),
            "tasks": ["TASK_A"],
            "checkpoint_path": "checkpoint-1lead.pth",
            "missing_keys": [],
            "unexpected_keys": [],
        },
    )
    monkeypatch.setattr(ecg_tools, "build_scp_relevant_findings", lambda *_args, **_kwargs: [])

    result = ecg_tools.ecg_diagnosis("apple_watch:aw-test", mode="12_lead")

    assert result["success"] is True
    assert runtime_calls == ["single_lead"]
    assert result["metadata"]["inference_mode"] == "single_lead"
    assert result["metadata"]["source"] == "apple_watch"


