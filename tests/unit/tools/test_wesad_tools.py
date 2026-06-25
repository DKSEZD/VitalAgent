from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from agent.encoder.ecg_signal_processor import ECGSignalProcessor, VitalParams
from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.tools.datasets import wesad as wesad_tools
from agent.tools.datasets import wesad_stress_classifier as classifier_tools
from agent.tools.registry import list_tools


def _write_wesad_pickle(path: Path, *, subject: str = "S_TEST") -> None:
    duration_s = 60.0
    labels = np.concatenate(
        [
            np.full(int(30 * 700), 1),
            np.full(int(30 * 700), 2),
        ]
    )
    payload = {
        "subject": subject,
        "label": labels,
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
                "EDA": np.linspace(1.0, 3.0, int(duration_s * 4)).reshape(-1, 1),
                "TEMP": np.linspace(32.0, 33.0, int(duration_s * 4)).reshape(-1, 1),
            },
            "chest": {
                "ECG": np.zeros((int(duration_s * 700), 1), dtype=float),
                "Resp": np.linspace(-1.0, 1.0, int(duration_s * 700)).reshape(-1, 1),
            },
        },
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _write_classifier_wesad_subject(root: Path, subject: str, *, phase: float = 0.0) -> Path:
    duration_s = 240.0
    subject_dir = root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    path = subject_dir / f"{subject}.pkl"
    labels = np.concatenate(
        [
            np.full(int(60 * 700), 1),
            np.full(int(60 * 700), 2),
            np.full(int(60 * 700), 3),
            np.full(int(60 * 700), 4),
        ]
    )
    bvp_t = np.arange(int(duration_s * 64)) / 64.0
    bvp = np.concatenate(
        [
            np.sin(2 * np.pi * 1.0 * bvp_t[: int(60 * 64)] + phase),
            np.sin(2 * np.pi * 1.45 * bvp_t[: int(60 * 64)] + phase),
            np.sin(2 * np.pi * 1.2 * bvp_t[: int(60 * 64)] + phase),
            np.sin(2 * np.pi * 0.85 * bvp_t[: int(60 * 64)] + phase),
        ]
    )
    acc = np.concatenate(
        [
            np.full((int(60 * 32), 3), 0.01),
            np.full((int(60 * 32), 3), 0.22),
            np.full((int(60 * 32), 3), 0.08),
            np.full((int(60 * 32), 3), 0.02),
        ]
    )
    eda = np.concatenate(
        [
            np.full(int(60 * 4), 1.0),
            np.full(int(60 * 4), 4.0),
            np.full(int(60 * 4), 2.0),
            np.full(int(60 * 4), 0.6),
        ]
    )
    temp = np.full(int(duration_s * 4), 32.5)
    resp = np.sin(np.linspace(0.0, 80.0 * np.pi, int(duration_s * 700)))
    payload = {
        "subject": subject,
        "label": labels,
        "signal": {
            "wrist": {
                "BVP": bvp.reshape(-1, 1),
                "ACC": acc,
                "EDA": eda.reshape(-1, 1),
                "TEMP": temp.reshape(-1, 1),
            },
            "chest": {
                "ECG": np.zeros((int(duration_s * 700), 1), dtype=float),
                "Resp": resp.reshape(-1, 1),
            },
        },
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def test_analyze_wesad_window_signal_returns_raw_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_wesad_pickle(pkl_path)

    monkeypatch.setattr(
        PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=76.0,
            rr_intervals_ms=[790.0, 800.0, 810.0, 805.0],
            sdnn_ms=9.1,
            rmssd_ms=12.2,
            signal_quality_score=0.91,
        ),
    )

    result = wesad_tools.analyze_wesad_window_signal(
        pickle_path=str(pkl_path),
        dataset="wesad",
        window_start_s=30.0,
        window_end_s=60.0,
    )

    assert result["success"] is True
    assert result["dataset"] == "wesad"
    assert result["subject_id"] == "S_TEST"
    assert result["data"]["window"]["duration_s"] == 30.0
    assert result["data"]["protocol_label"]["dominant_protocol_label"] == "stress"
    assert result["data"]["protocol_label"]["dominant_protocol_label_id"] == 2
    assert result["data"]["protocol_label"]["is_clean_protocol_window"] is True
    assert result["data"]["bvp_pulse"]["mean_pulse_rate_bpm"] == 76.0
    assert result["data"]["bvp_pulse"]["quality_label"] == "good"
    assert result["data"]["motion"]["motion_level"] == "low"
    assert result["data"]["eda"]["sample_count"] == 120
    assert result["data"]["temperature"]["sample_count"] == 120
    assert result["data"]["respiration"]["sample_count"] == 30 * 700
    assert result["metadata"]["window_sample_counts"]["protocol_label"] == 30 * 700
    assert result["metadata"]["window_sample_counts"]["wrist_bvp"] == 30 * 64
    assert any("not clinical stress diagnoses" in caveat for caveat in result["data"]["caveats"])


def test_analyze_wesad_window_signal_surfaces_chest_ecg_rhythm_screen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_wesad_pickle(pkl_path)

    monkeypatch.setattr(
        PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=76.0,
            rr_intervals_ms=[790.0, 800.0, 810.0, 805.0, 795.0, 800.0],
            signal_quality_score=0.91,
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

    result = wesad_tools.analyze_wesad_window_signal(
        pickle_path=str(pkl_path),
        dataset="wesad",
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


def test_analyze_wesad_window_signal_resolves_subject_from_dataset_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    subject_dir = tmp_path / "S2"
    subject_dir.mkdir()
    pkl_path = subject_dir / "S2.pkl"
    _write_wesad_pickle(pkl_path, subject="S2")

    monkeypatch.setattr(
        PPGSignalProcessor,
        "extract_vitals",
        lambda self, signal, fs, accel=None: VitalParams(
            hr_bpm=72.0,
            rr_intervals_ms=[820.0, 830.0, 810.0],
            signal_quality_score=0.72,
        ),
    )

    result = wesad_tools.analyze_wesad_window_signal(
        subject_id="2",
        dataset_root=str(tmp_path),
        window_start_s=0.0,
        window_end_s=30.0,
    )

    assert result["success"] is True
    assert result["source_path"].endswith("S2.pkl")
    assert result["data"]["protocol_label"]["dominant_protocol_label"] == "baseline"


def test_analyze_wesad_window_signal_rejects_wrong_dataset(tmp_path: Path) -> None:
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_wesad_pickle(pkl_path)

    result = wesad_tools.analyze_wesad_window_signal(
        pickle_path=str(pkl_path),
        dataset="ppg_dalia",
        window_start_s=0.0,
        window_end_s=30.0,
    )

    assert result["success"] is False
    assert "Unsupported dataset" in result["error"]


def test_wesad_tool_is_registered() -> None:
    names = {tool["name"] for tool in list_tools()}

    assert "analyze_wesad_window_signal" in names
    assert "classify_wesad_stress_state" in names


def test_wesad_stress_classifier_trains_with_subject_exclusion(tmp_path: Path) -> None:
    root = tmp_path / "WESAD"
    _write_classifier_wesad_subject(root, "S_TRAIN1", phase=0.0)
    _write_classifier_wesad_subject(root, "S_TRAIN2", phase=0.2)
    active = _write_classifier_wesad_subject(root, "S_TEST", phase=0.1)
    model_path = tmp_path / "model.pkl"

    result = classifier_tools.classify_wesad_stress_state(
        pickle_path=str(active),
        dataset_root=str(root),
        dataset="wesad",
        window_start_s=60.0,
        window_end_s=120.0,
        model_path=str(model_path),
        evaluate_subject_wise=True,
    )

    assert result["success"] is True
    assert result["data"]["predicted_stress_label"] in {
        "baseline",
        "stress",
        "amusement",
        "meditation",
    }
    guard = result["metadata"]["leakage_guard"]
    assert guard["active_subject_id"] == "S_TEST"
    assert guard["active_subject_excluded_from_training"] is True
    train_subjects = result["metadata"]["training"]["train_subjects"]
    assert train_subjects == ["S_TRAIN1", "S_TRAIN2"]
    assert "S_TEST" not in train_subjects
    assert result["metadata"]["subject_wise_eval"]["method"] == "leave_one_subject_out"


def test_wesad_stress_classifier_blocks_release_subject_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "WESAD"
    active = _write_classifier_wesad_subject(root, "S2", phase=0.1)
    model_path = tmp_path / "model.pkl"
    monkeypatch.setattr(classifier_tools, "release_wesad_subjects", lambda: {"S2"})

    result = classifier_tools.classify_wesad_stress_state(
        pickle_path=str(active),
        dataset_root=str(root),
        dataset="wesad",
        window_start_s=60.0,
        window_end_s=120.0,
        model_path=str(model_path),
    )

    assert result["success"] is False
    assert "Insufficient non-leaking WESAD training data" in result["error"]
    assert "S2" in result["leakage_guard"]["release_eval_subjects"]
