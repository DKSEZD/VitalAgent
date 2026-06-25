from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from agent.data.ppg_loader import PPGChunk, PPGRecording, RelativeTimestamp
from agent.data.ppg_window_labels import format_ppg_window_id
from agent.encoder.ecg_signal_processor import VitalParams
from agent.tools.ppg import tools as ppg_tools


@dataclass
class _FakeLoader:
    recording: PPGRecording
    ecg_start_time: datetime | None = None

    def available_patient_ids(self) -> list[str]:
        return [self.recording.patient_id]

    def patient_files_complete(self, patient_id: str) -> bool:
        return patient_id == self.recording.patient_id

    def read_ppg_record(self, patient_id: str, *, include_signals: bool) -> PPGRecording:
        assert patient_id == self.recording.patient_id
        assert include_signals in {True, False}
        return self.recording

    def read_ecg_record(self, patient_id: str, *, include_signal: bool = False):
        if self.ecg_start_time is None:
            raise FileNotFoundError("No fake ECG record configured.")
        assert patient_id == self.recording.patient_id
        assert include_signal is False
        return SimpleNamespace(start_time=self.ecg_start_time)

    def summarize_ppg(self, record: PPGRecording) -> dict[str, object]:
        assert record is self.recording
        return {
            "patient_id": record.patient_id,
            "chunk_count": 1,
            "total_hours": record.total_duration_seconds / 3600.0,
            "longest_chunk_hours": record.longest_chunk_seconds / 3600.0,
            "chunk_hours": [record.chunks[0].duration_seconds / 3600.0],
            "chunk_start_times": [record.chunks[0].start_time.isoformat(sep=" ")],
        }


def _recording() -> PPGRecording:
    chunk = PPGChunk(
        patient_id="001",
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
        ppg_fs=100,
        accel_fs=50,
        n_ppg_samples=1000,
        green_signal=np.linspace(0.0, 1.0, 1000, dtype=float),
        ambient_signal=np.zeros(1000, dtype=float),
        accel_x=np.zeros(500, dtype=float),
        accel_y=np.zeros(500, dtype=float),
        accel_z=np.zeros(500, dtype=float),
        metadata={},
    )
    return PPGRecording(patient_id="001", chunks=[chunk], metadata={})


def _recording_duration(duration_s: float) -> PPGRecording:
    fs = 100
    n_ppg_samples = int(duration_s * fs)
    chunk = PPGChunk(
        patient_id="001",
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
        ppg_fs=fs,
        accel_fs=50,
        n_ppg_samples=n_ppg_samples,
        green_signal=np.linspace(0.0, 1.0, n_ppg_samples, dtype=float),
        ambient_signal=np.zeros(n_ppg_samples, dtype=float),
        accel_x=np.zeros(int(duration_s * 50), dtype=float),
        accel_y=np.zeros(int(duration_s * 50), dtype=float),
        accel_z=np.zeros(int(duration_s * 50), dtype=float),
        metadata={},
    )
    return PPGRecording(patient_id="001", chunks=[chunk], metadata={})


def _recording_with_late_ppg_start() -> PPGRecording:
    chunk = PPGChunk(
        patient_id="001",
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:10:00"),
        ppg_fs=100,
        accel_fs=50,
        n_ppg_samples=1000,
        green_signal=np.linspace(0.0, 1.0, 1000, dtype=float),
        ambient_signal=np.zeros(1000, dtype=float),
        accel_x=np.zeros(500, dtype=float),
        accel_y=np.zeros(500, dtype=float),
        accel_z=np.zeros(500, dtype=float),
        metadata={},
    )
    return PPGRecording(patient_id="001", chunks=[chunk], metadata={})


def test_list_ppg_records_returns_prefixed_ids(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)

    result = ppg_tools.list_ppg_records(limit=5)

    assert result["success"] is True
    assert result["record_ids"] == ["ppg:001"]
    assert result["records"][0]["source"] == "ppg"
    assert result["records"][0]["chunk_count"] == 1


def test_analyze_pulse_rate_returns_expected_shape(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)
    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=72.5,
            rr_intervals_ms=[800.0, 820.0, 780.0, 810.0],
            sdnn_ms=17.1,
            rmssd_ms=24.3,
            signal_quality_score=0.88,
        ),
    )

    result = ppg_tools.analyze_pulse_rate("ppg:001")

    assert result["success"] is True
    assert result["data"]["heart_rate"]["mean_bpm"] == 72.5
    assert result["data"]["pp_intervals"]["count"] == 4
    assert result["metadata"]["record_id"] == "ppg:001"
    assert result["metadata"]["signal_quality_score"] == 0.88


def test_analyze_pulse_rate_reports_30s_subwindow_max(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording_duration(90.0))
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)
    hr_values = iter([80.0, 70.0, 112.4, 90.0])

    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=next(hr_values),
            rr_intervals_ms=[800.0, 820.0, 780.0, 810.0],
            signal_quality_score=0.88,
        ),
    )

    result = ppg_tools.analyze_pulse_rate(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=0.0,
        window_end_s=90.0,
    )

    assert result["success"] is True
    assert result["data"]["heart_rate"]["mean_bpm"] == 80.0
    assert result["data"]["heart_rate"]["max_30s_bpm"] == 112.4
    assert result["data"]["heart_rate"]["min_30s_bpm"] == 70.0
    assert (
        result["data"]["heart_rate"]["max_30s_bpm_source"]
        == "ppg_peak_detection_30s_subwindows"
    )


def test_analyze_pulse_rate_reports_recording_context_max_when_requested(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording_duration(600.0))
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)
    ppg_tools._RECORDING_CONTEXT_MAX_30S_CACHE.clear()
    hr_values = iter([80.0, 70.0, 112.4, 90.0])

    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=next(hr_values),
            rr_intervals_ms=[800.0, 820.0, 780.0, 810.0],
            signal_quality_score=0.88,
        ),
    )
    monkeypatch.setattr(
        ppg_tools,
        "compute_ppg_max_hr_bpm",
        lambda *, window, processor, subwindow_sec: 191.5
        if window.window_index == 1
        else 120.0,
    )

    result = ppg_tools.analyze_pulse_rate(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=0.0,
        window_end_s=90.0,
        include_recording_context=True,
    )

    assert result["success"] is True
    assert result["data"]["heart_rate"]["recording_context_max_30s_bpm"] == 191.5
    assert (
        result["data"]["heart_rate"]["recording_context_max_30s_bpm_source"]
        == "ppg_peak_detection_recording_context_30s_subwindows"
    )


def test_assess_ppg_signal_quality_reports_quality_label(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)
    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(signal_quality_score=0.31),
    )
    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "_motion_intensity",
        staticmethod(lambda accel: 0.75),
    )

    result = ppg_tools.assess_ppg_signal_quality("ppg:001")

    assert result["success"] is True
    assert result["data"]["quality_label"] == "poor"
    assert result["data"]["component_scores"]["motion_intensity"] == 0.75


def test_analyze_ppg_rhythm_irregularity_marks_irregular(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)
    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=95.0,
            rr_intervals_ms=[600.0, 900.0, 650.0, 980.0, 700.0],
            signal_quality_score=0.9,
        ),
    )

    result = ppg_tools.analyze_ppg_rhythm_irregularity("ppg:001")

    assert result["success"] is True
    assert result["data"]["assessment"] == "irregular"
    assert result["data"]["is_irregular"] is True


def test_analyze_ppg_rhythm_irregularity_low_quality_blocks_positive(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)
    monkeypatch.setattr(
        ppg_tools.PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=95.0,
            rr_intervals_ms=[600.0, 1000.0, 620.0, 1100.0, 650.0],
            signal_quality_score=0.503,
        ),
    )

    result = ppg_tools.analyze_ppg_rhythm_irregularity("ppg:001")

    assert result["success"] is True
    assert result["data"]["assessment"] == "indeterminate"
    assert result["data"]["is_irregular"] is False


def test_analyze_pulse_rate_supports_window_id(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)

    captured: dict[str, object] = {}

    def fake_extract_vitals(self, signal, fs, accel=None):
        captured["signal_len"] = len(signal)
        captured["signal_first"] = float(signal[0])
        captured["signal_last"] = float(signal[-1])
        return VitalParams(
            hr_bpm=72.5,
            rr_intervals_ms=[800.0, 820.0, 780.0, 810.0],
            sdnn_ms=17.1,
            rmssd_ms=24.3,
            signal_quality_score=0.88,
        )

    monkeypatch.setattr(ppg_tools.PPGSignalProcessor, "extract_vitals", fake_extract_vitals)

    window_id = format_ppg_window_id(
        patient_id="001",
        chunk_index=0,
        window_index=3,
        start_sample=100,
        end_sample=300,
    )
    result = ppg_tools.analyze_pulse_rate("ppg:001", window_id=window_id)

    assert result["success"] is True
    assert captured == {
        "signal_len": 200,
        "signal_first": float(fake_loader.recording.chunks[0].green_signal[100]),
        "signal_last": float(fake_loader.recording.chunks[0].green_signal[299]),
    }
    assert result["metadata"]["window_id"] == window_id
    assert result["metadata"]["window_index"] == 3
    assert result["metadata"]["window_duration_seconds"] == 2.0


def test_analyze_pulse_rate_accepts_window_locator_aliases(monkeypatch) -> None:
    fake_loader = _FakeLoader(_recording())
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)

    captured: dict[str, object] = {}

    def fake_extract_vitals(self, signal, fs, accel=None):
        captured["signal_len"] = len(signal)
        captured["signal_first"] = float(signal[0])
        captured["signal_last"] = float(signal[-1])
        return VitalParams(
            hr_bpm=72.5,
            rr_intervals_ms=[800.0, 820.0, 780.0, 810.0],
            sdnn_ms=17.1,
            rmssd_ms=24.3,
            signal_quality_score=0.88,
        )

    monkeypatch.setattr(ppg_tools.PPGSignalProcessor, "extract_vitals", fake_extract_vitals)

    result = ppg_tools.analyze_pulse_rate(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=1.0,
        window_end_s=3.0,
    )

    assert result["success"] is True
    assert captured == {
        "signal_len": 200,
        "signal_first": float(fake_loader.recording.chunks[0].green_signal[100]),
        "signal_last": float(fake_loader.recording.chunks[0].green_signal[299]),
    }
    assert result["metadata"]["record_id"] == "ppg:001"
    assert result["metadata"]["window_start_s"] == 1.0
    assert result["metadata"]["window_end_s"] == 3.0
    assert result["metadata"]["window_duration_seconds"] == 2.0


def test_analyze_pulse_rate_uses_ecg_relative_locator_for_af_ppg(monkeypatch) -> None:
    fake_loader = _FakeLoader(
        _recording_with_late_ppg_start(),
        ecg_start_time=datetime(2000, 1, 1, 0, 0, 0),
    )
    monkeypatch.setattr(ppg_tools, "_loader", lambda: fake_loader)

    captured: dict[str, object] = {}

    def fake_extract_vitals(self, signal, fs, accel=None):
        captured["signal_len"] = len(signal)
        captured["signal_first"] = float(signal[0])
        captured["signal_last"] = float(signal[-1])
        return VitalParams(
            hr_bpm=72.5,
            rr_intervals_ms=[800.0, 820.0, 780.0, 810.0],
            sdnn_ms=17.1,
            rmssd_ms=24.3,
            signal_quality_score=0.88,
        )

    monkeypatch.setattr(ppg_tools.PPGSignalProcessor, "extract_vitals", fake_extract_vitals)

    result = ppg_tools.analyze_pulse_rate(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=601.0,
        window_end_s=603.0,
    )

    assert result["success"] is True
    assert captured == {
        "signal_len": 200,
        "signal_first": float(fake_loader.recording.chunks[0].green_signal[100]),
        "signal_last": float(fake_loader.recording.chunks[0].green_signal[299]),
    }
    assert result["metadata"]["time_reference"] == "ecg_recording_start"


def test_ppg_tools_reject_non_ppg_dataset_even_with_record_id() -> None:
    result = ppg_tools.analyze_pulse_rate(
        record_id="ppg:S12",
        dataset="ppg_dalia",
        patient_id="S12",
        window_start_s=0.0,
        window_end_s=60.0,
    )

    assert result["success"] is False
    assert "Unsupported raw PPG dataset" in result["error"]
