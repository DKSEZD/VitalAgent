from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agent.data.icentia11k_loader import BeatAnnotation, RawECGData
from agent.tools.datasets import icentia11k as icentia11k_tools


class _FakeRate:
    def __init__(self, values):
        self.values = values

    def dropna(self):
        return self.values


class _FakeLoader:
    def read_segment_header(self, patient_id: str, segment_id: str):
        assert patient_id == "03500"
        assert segment_id == "17"
        return SimpleNamespace(fs=250)

    def read_segment(self, patient_id: str, segment_id: str, sampfrom: int, sampto: int):
        assert patient_id == "03500"
        assert segment_id == "17"
        assert sampfrom == 250
        assert sampto == 750
        return RawECGData(
            signal=np.sin(np.linspace(0, 8 * np.pi, 500)),
            fs=250,
            patient_id="03500",
            segment_id="17",
            n_samples=500,
            sample_start=sampfrom,
            sample_end=sampto,
            segment_total_samples=1000,
            window_duration_seconds=2.0,
            metadata={"source": "local", "lead_names": ["ECG"]},
        )


class _FakeAnnotationLoader(_FakeLoader):
    def read_segment(self, patient_id: str, segment_id: str, sampfrom: int, sampto: int):
        raw = super().read_segment(patient_id, segment_id, sampfrom, sampto)
        raw.beat_annotations = BeatAnnotation(
            samples=np.asarray([10, 110, 210], dtype=int),
            symbols=["N", "N", "N"],
        )
        return raw


class _FakeMonitoringContextLoader:
    def read_segment_header(self, patient_id: str, segment_id: str):
        assert patient_id == "03500"
        assert segment_id == "17"
        return SimpleNamespace(fs=10, n_samples=18000)

    def read_segment(self, patient_id: str, segment_id: str, sampfrom: int, sampto: int):
        assert patient_id == "03500"
        assert segment_id == "17"
        raw = RawECGData(
            signal=np.zeros(max(0, sampto - sampfrom), dtype=float),
            fs=10,
            patient_id="03500",
            segment_id="17",
            n_samples=max(0, sampto - sampfrom),
            sample_start=sampfrom,
            sample_end=sampto,
            segment_total_samples=18000,
            window_duration_seconds=max(0, sampto - sampfrom) / 10,
            metadata={"source": "local", "lead_names": ["ECG"]},
        )
        if sampfrom == 10 and sampto == 30:
            raw.beat_annotations = BeatAnnotation(
                samples=np.asarray([0, 10], dtype=int),
                symbols=["N", "N"],
            )
        elif sampfrom == 0 and sampto == 18000:
            raw.beat_annotations = BeatAnnotation(
                samples=np.arange(0, 70, dtype=int),
                symbols=["N"] * 70,
            )
        return raw


def test_analyze_icentia11k_ecg_window_signal_uses_raw_signal_only(monkeypatch) -> None:
    monkeypatch.setattr(icentia11k_tools, "_loader", lambda local_root=None: _FakeLoader())
    monkeypatch.setattr(
        icentia11k_tools,
        "process_ecg",
        lambda signal, fs: (
            {"ECG_Rate": _FakeRate([60.0, 61.0, 62.0])},
            {"ECG_R_Peaks": np.array([0, 250, 500])},
        ),
    )

    result = icentia11k_tools.analyze_icentia11k_ecg_window_signal(
        dataset="icentia11k",
        patient_id="03500",
        recording_id="segment_17",
        window_start_s=1.0,
        window_end_s=3.0,
    )

    assert result["success"] is True
    assert result["data"]["heart_rate"]["mean_bpm"] == 61.0
    assert result["data"]["rr_intervals"]["count"] == 2
    assert result["metadata"]["segment_id"] == "17"
    assert "rhythm_annotations" not in str(result)
    assert "beat_annotations" not in str(result)


def test_analyze_icentia11k_ecg_window_signal_prefers_annotation_hr_when_available(monkeypatch) -> None:
    monkeypatch.setattr(icentia11k_tools, "_loader", lambda local_root=None: _FakeAnnotationLoader())
    monkeypatch.setattr(
        icentia11k_tools,
        "analyze_single_lead_ecg_window",
        lambda signal, fs, process_func, rhythm_classifier: {
            "success": True,
            "data": {
                "heart_rate": {
                    "mean_bpm": 120.0,
                    "min_bpm": 70.0,
                    "max_bpm": 120.0,
                    "std_bpm": 10.0,
                },
                "rr_intervals": {"count": 3},
            },
        },
    )

    result = icentia11k_tools.analyze_icentia11k_ecg_window_signal(
        dataset="icentia11k",
        patient_id="03500",
        recording_id="segment_17",
        window_start_s=1.0,
        window_end_s=3.0,
    )

    assert result["success"] is True
    assert result["data"]["heart_rate"]["mean_bpm"] == 90.0
    assert result["data"]["heart_rate"]["mean_bpm_source"] == "icentia11k_beat_annotations"
    assert result["data"]["signal_estimated_heart_rate"]["mean_bpm"] == 120.0
    assert result["metadata"]["beat_annotation_hr_available"] is True


def test_analyze_icentia11k_ecg_window_signal_reports_monitoring_context_max(monkeypatch) -> None:
    monkeypatch.setattr(
        icentia11k_tools,
        "_loader",
        lambda local_root=None: _FakeMonitoringContextLoader(),
    )
    monkeypatch.setattr(
        icentia11k_tools,
        "analyze_single_lead_ecg_window",
        lambda signal, fs, process_func, rhythm_classifier: {
            "success": True,
            "data": {
                "heart_rate": {
                    "mean_bpm": 60.0,
                    "min_bpm": 60.0,
                    "max_bpm": 60.0,
                    "std_bpm": 0.0,
                },
                "rr_intervals": {"count": 2},
            },
        },
    )

    result = icentia11k_tools.analyze_icentia11k_ecg_window_signal(
        dataset="icentia11k",
        patient_id="03500",
        recording_id="segment_17",
        window_start_s=1.0,
        window_end_s=3.0,
    )

    assert result["success"] is True
    heart_rate = result["data"]["heart_rate"]
    assert heart_rate["monitoring_context_max_30s_bpm"] == 140.0
    assert (
        heart_rate["monitoring_context_max_30s_bpm_source"]
        == "icentia11k_beat_annotations_monitoring_context"
    )
