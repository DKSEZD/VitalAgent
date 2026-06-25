from __future__ import annotations

import numpy as np

from agent.tools.ecg import window_analysis


class _FakeRate:
    def __init__(self, values: list[float]):
        self.values = values

    def dropna(self) -> list[float]:
        return self.values


def test_analyze_single_lead_ecg_window_returns_rr_metrics() -> None:
    result = window_analysis.analyze_single_lead_ecg_window(
        np.sin(np.linspace(0.0, 4.0 * np.pi, 500)),
        250,
        process_func=lambda signal, fs: (
            {"ECG_Rate": _FakeRate([60.0, 61.0, 62.0])},
            {"ECG_R_Peaks": np.array([0, 250, 500])},
        ),
        rhythm_classifier=lambda rr_ms: "Normal",
    )

    assert result["success"] is True
    assert result["data"]["heart_rate"]["mean_bpm"] == 61.0
    assert result["data"]["rr_intervals"]["count"] == 2
    assert result["data"]["rhythm_screen"] == "Normal"
    assert result["data"]["af_screen_positive"] is False


def test_analyze_single_lead_ecg_window_reports_quality_when_too_few_peaks() -> None:
    result = window_analysis.analyze_single_lead_ecg_window(
        np.ones(100),
        250,
        process_func=lambda signal, fs: (
            {"ECG_Rate": _FakeRate([])},
            {"ECG_R_Peaks": np.array([10])},
        ),
    )

    assert result["success"] is False
    assert result["error"] == "Too few R-peaks detected for ECG window analysis."
    assert result["signal_quality"]["quality_label"] == "poor"
    assert result["signal_quality"]["r_peak_count"] == 1
