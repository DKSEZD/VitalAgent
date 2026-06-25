from __future__ import annotations

import numpy as np
import pytest

nk = pytest.importorskip("neurokit2")

from agent.data.streaming import ECGWindow  # noqa: E402
from agent.encoder import ECGSignalProcessor  # noqa: E402


def test_detect_r_peaks_finds_multiple_beats_on_simulated_ecg() -> None:
    signal = nk.ecg_simulate(duration=10, sampling_rate=250)
    processor = ECGSignalProcessor()

    r_peaks = processor.detect_r_peaks(signal, fs=250)

    assert isinstance(r_peaks, np.ndarray)
    assert r_peaks.dtype.kind in {"i", "u"}
    assert len(r_peaks) >= 2


def test_compute_hrv_returns_metrics_for_clean_rr_series() -> None:
    processor = ECGSignalProcessor()
    rr_intervals_ms = np.array([820.0, 840.0, 860.0, 830.0, 850.0], dtype=float)

    hrv = processor.compute_hrv(rr_intervals_ms)

    assert hrv["sdnn_ms"] is not None
    assert hrv["rmssd_ms"] is not None
    assert hrv["sdnn_ms"] > 0
    assert hrv["rmssd_ms"] > 0


def test_assess_signal_quality_returns_unit_interval_score() -> None:
    signal = nk.ecg_simulate(duration=10, sampling_rate=250)
    processor = ECGSignalProcessor()

    score = processor.assess_signal_quality(signal, fs=250)

    assert 0.0 <= score <= 1.0


def test_extract_vitals_returns_expected_fields_on_simulated_ecg() -> None:
    signal = nk.ecg_simulate(duration=10, sampling_rate=250)
    processor = ECGSignalProcessor()

    vitals = processor.extract_vitals(signal, fs=250)

    assert vitals.hr_bpm is not None
    assert vitals.rr_intervals_ms
    assert vitals.signal_quality_score is not None


def test_extract_vitals_uses_call_level_sampling_rate() -> None:
    processor = ECGSignalProcessor()
    signal = np.zeros(600, dtype=float)

    processor.detect_r_peaks = lambda *_args, **_kwargs: np.array([0, 250, 500], dtype=int)  # type: ignore[method-assign]
    processor.assess_signal_quality = lambda *_args, **_kwargs: 0.5  # type: ignore[method-assign]

    vitals_250 = processor.extract_vitals(signal, fs=250)
    vitals_500 = processor.extract_vitals(signal, fs=500)

    assert vitals_250.hr_bpm == 60.0
    assert vitals_500.hr_bpm == 120.0


def test_extract_vitals_falls_back_to_instance_sampling_rate_when_fs_is_missing() -> None:
    processor = ECGSignalProcessor(sampling_rate=500)
    signal = np.zeros(600, dtype=float)

    processor.detect_r_peaks = lambda *_args, **_kwargs: np.array([0, 250, 500], dtype=int)  # type: ignore[method-assign]
    processor.assess_signal_quality = lambda *_args, **_kwargs: 0.5  # type: ignore[method-assign]

    vitals = processor.extract_vitals(signal)

    assert vitals.hr_bpm == 120.0


def test_extract_vitals_returns_partial_vitals_when_too_few_peaks_detected() -> None:
    processor = ECGSignalProcessor()
    signal = np.zeros(20, dtype=float)

    processor.detect_r_peaks = lambda *_args, **_kwargs: np.array([5], dtype=int)  # type: ignore[method-assign]
    processor.assess_signal_quality = lambda *_args, **_kwargs: 0.25  # type: ignore[method-assign]

    vitals = processor.extract_vitals(signal, fs=250)

    assert vitals.hr_bpm is None
    assert vitals.rr_intervals_ms is None
    assert vitals.signal_quality_score == 0.25


def test_extract_vitals_runs_when_shared_processing_fails_and_fallback_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = nk.ecg_simulate(duration=10, sampling_rate=250)
    processor = ECGSignalProcessor()

    monkeypatch.setattr(processor, "_process_ecg_shared", lambda *_args, **_kwargs: (None, None))

    vitals = processor.extract_vitals(signal, fs=250)

    assert vitals.hr_bpm is not None
    assert vitals.rr_intervals_ms


def test_resolve_fs_rejects_non_integer_float_sampling_rate() -> None:
    processor = ECGSignalProcessor()

    with pytest.raises(ValueError, match="integer value"):
        processor.extract_vitals(np.zeros(600, dtype=float), fs=250.5)


def test_extract_vitals_accepts_window_sampling_rate_contract() -> None:
    signal = nk.ecg_simulate(duration=10, sampling_rate=250)
    window = ECGWindow(
        signal=signal,
        fs=250,
        patient_id="00012",
        segment_id="03",
        window_index=0,
        start_sample=0,
        end_sample=len(signal),
        n_samples=len(signal),
        stream_offset_s=0.0,
    )
    processor = ECGSignalProcessor()

    vitals = processor.extract_vitals(window.signal, fs=window.fs)

    assert vitals.hr_bpm is not None
    assert vitals.rr_intervals_ms
