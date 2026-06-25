from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from agent.encoder.ecg_signal_processor import ECGSignalProcessor, VitalParams
from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.tools.datasets import ppg_dalia as ppg_dalia_tools
from agent.tools.registry import list_tools


def _write_ppg_dalia_pickle(path: Path) -> None:
    duration_s = 60.0
    payload = {
        "subject": "S_TEST",
        "label": np.concatenate([np.full(15, 70.0), np.full(15, 90.0)]),
        "activity": np.concatenate([np.full(120, 1), np.full(120, 7)]).reshape(-1, 1),
        "signal": {
            "wrist": {
                "BVP": np.sin(np.linspace(0.0, 24.0 * np.pi, int(duration_s * 64))).reshape(-1, 1),
                "ACC": np.column_stack(
                    [
                        np.zeros(int(duration_s * 32)),
                        np.zeros(int(duration_s * 32)),
                        np.full(int(duration_s * 32), 0.01),
                    ]
                ),
                "EDA": np.linspace(1.0, 2.0, int(duration_s * 4)).reshape(-1, 1),
                "TEMP": np.linspace(32.0, 33.0, int(duration_s * 4)).reshape(-1, 1),
            },
            "chest": {
                "ECG": np.zeros((int(duration_s * 700), 1), dtype=float),
            },
        },
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_analyze_ppg_dalia_window_signal_returns_raw_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_ppg_dalia_pickle(pkl_path)

    monkeypatch.setattr(
        PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=82.0,
            rr_intervals_ms=[720.0, 740.0, 730.0, 750.0],
            sdnn_ms=12.3,
            rmssd_ms=18.4,
            signal_quality_score=0.87,
        ),
    )

    result = ppg_dalia_tools.analyze_ppg_dalia_window_signal(
        pickle_path=str(pkl_path),
        dataset="ppg_dalia",
        window_start_s=30.0,
        window_end_s=60.0,
    )

    assert result["success"] is True
    assert result["dataset"] == "ppg_dalia"
    assert result["subject_id"] == "S_TEST"
    assert result["data"]["window"]["duration_s"] == 30.0
    assert result["data"]["bvp_pulse"]["mean_pulse_rate_bpm"] == 82.0
    assert result["data"]["bvp_pulse"]["quality_label"] == "good"
    assert result["data"]["ecg_reference_hr_label"]["mean_bpm"] == 90.0
    assert result["data"]["activity"]["dominant_activity_label"] == "walking"
    assert result["data"]["motion"]["motion_level"] == "low"
    assert result["data"]["eda"]["sample_count"] == 120
    assert result["metadata"]["window_sample_counts"]["wrist_bvp"] == 30 * 64
    assert result["metadata"]["window_sample_counts"]["hr_label"] == 15


def test_analyze_ppg_dalia_window_signal_surfaces_chest_ecg_rhythm_screen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_ppg_dalia_pickle(pkl_path)

    monkeypatch.setattr(
        PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=82.0,
            rr_intervals_ms=[720.0, 740.0, 730.0, 750.0, 735.0, 745.0],
            signal_quality_score=0.87,
        ),
    )
    monkeypatch.setattr(
        ECGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs: VitalParams(
            hr_bpm=75.0,
            rr_intervals_ms=[800.0, 805.0, 795.0, 802.0, 798.0, 801.0],
            sdnn_ms=3.2,
            rmssd_ms=4.1,
            signal_quality_score=0.95,
        ),
    )

    result = ppg_dalia_tools.analyze_ppg_dalia_window_signal(
        pickle_path=str(pkl_path),
        dataset="ppg_dalia",
        window_start_s=0.0,
        window_end_s=30.0,
        include_chest_ecg_analysis=True,
    )

    chest_ecg = result["data"]["chest_ecg"]
    assert chest_ecg["rhythm_screen"] in {"N", "AF", "Other", "Unknown"}
    assert isinstance(chest_ecg["af_screen_positive"], bool)
    assert chest_ecg["af_screen_positive"] is (chest_ecg["rhythm_screen"] == "AF")
    assert set(chest_ecg["irregularity_metrics"]) == {
        "coefficient_of_variation",
        "rmssd_ms",
        "pnn50_percent",
    }


def test_analyze_ppg_dalia_window_signal_resolves_subject_from_dataset_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    subject_dir = tmp_path / "S1"
    subject_dir.mkdir()
    pkl_path = subject_dir / "S1.pkl"
    _write_ppg_dalia_pickle(pkl_path)

    monkeypatch.setattr(
        PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=75.0,
            rr_intervals_ms=[800.0, 810.0, 790.0],
            signal_quality_score=0.7,
        ),
    )

    result = ppg_dalia_tools.analyze_ppg_dalia_window_signal(
        subject_id="1",
        dataset_root=str(tmp_path),
        window_start_s=0.0,
        window_end_s=30.0,
    )

    assert result["success"] is True
    assert result["source_path"].endswith("S1.pkl")
    assert result["data"]["activity"]["dominant_activity_label"] == "sitting"


def test_analyze_ppg_dalia_window_signal_rejects_wrong_dataset(tmp_path: Path) -> None:
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_ppg_dalia_pickle(pkl_path)

    result = ppg_dalia_tools.analyze_ppg_dalia_window_signal(
        pickle_path=str(pkl_path),
        dataset="wesad",
        window_start_s=0.0,
        window_end_s=30.0,
    )

    assert result["success"] is False
    assert "Unsupported dataset" in result["error"]


def test_ppg_dalia_tool_is_registered() -> None:
    names = {tool["name"] for tool in list_tools()}

    assert "analyze_ppg_dalia_window_signal" in names
