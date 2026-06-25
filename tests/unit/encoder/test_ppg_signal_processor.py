from __future__ import annotations

import numpy as np
import pytest

from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.encoder.ecg_signal_processor import VitalParams

FS = 100  # AFPPGECG PPG sampling rate


def _synthetic_ppg(duration_s: float = 12.0, hr_bpm: float = 60.0, fs: int = FS) -> np.ndarray:
    """Synthetic PPG: sum of sine harmonics mimicking systolic waveform."""
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    f0 = hr_bpm / 60.0
    signal = (
        np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(4 * np.pi * f0 * t)
        + 0.2 * np.sin(6 * np.pi * f0 * t)
    )
    return signal


def _flat_signal(n: int = 1000) -> np.ndarray:
    return np.zeros(n, dtype=float)


# ---------------------------------------------------------------------------
# detect_peaks
# ---------------------------------------------------------------------------

def test_detect_peaks_finds_peaks_in_synthetic_ppg() -> None:
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    peaks = proc.detect_peaks(signal, fs=FS)

    assert isinstance(peaks, np.ndarray)
    assert peaks.dtype.kind in {"i", "u"}
    # At 60 bpm over 12 s expect roughly 12 peaks; allow generous bounds
    assert 6 <= len(peaks) <= 20


def test_detect_peaks_returns_empty_for_flat_signal() -> None:
    proc = PPGSignalProcessor(sampling_rate=FS)
    peaks = proc.detect_peaks(_flat_signal(), fs=FS)
    assert len(peaks) == 0


def test_detect_peaks_uses_instance_fs_when_not_given() -> None:
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)
    peaks_explicit = proc.detect_peaks(signal, fs=FS)
    peaks_default = proc.detect_peaks(signal)
    # Both paths must find the same peaks (deterministic)
    np.testing.assert_array_equal(peaks_explicit, peaks_default)


# ---------------------------------------------------------------------------
# extract_vitals — happy path
# ---------------------------------------------------------------------------

def test_extract_vitals_returns_vital_params_instance() -> None:
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=70.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    vitals = proc.extract_vitals(signal, fs=FS)

    assert isinstance(vitals, VitalParams)


def test_extract_vitals_plausible_pulse_rate() -> None:
    signal = _synthetic_ppg(duration_s=15.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    vitals = proc.extract_vitals(signal, fs=FS)

    assert vitals.hr_bpm is not None
    assert 40.0 <= vitals.hr_bpm <= 120.0


def test_extract_vitals_populates_pp_intervals() -> None:
    signal = _synthetic_ppg(duration_s=15.0, hr_bpm=75.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    vitals = proc.extract_vitals(signal, fs=FS)

    assert vitals.rr_intervals_ms is not None
    assert len(vitals.rr_intervals_ms) >= 2
    # Each PP interval should be physiologically plausible
    for pp in vitals.rr_intervals_ms:
        assert 250.0 <= pp <= 2000.0


def test_extract_vitals_computes_prv() -> None:
    signal = _synthetic_ppg(duration_s=20.0, hr_bpm=65.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    vitals = proc.extract_vitals(signal, fs=FS)

    assert vitals.sdnn_ms is not None
    assert vitals.rmssd_ms is not None


def test_extract_vitals_quality_score_in_unit_interval() -> None:
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    vitals = proc.extract_vitals(signal, fs=FS)

    assert vitals.signal_quality_score is not None
    assert 0.0 <= vitals.signal_quality_score <= 1.0


# ---------------------------------------------------------------------------
# extract_vitals — partial / degenerate inputs
# ---------------------------------------------------------------------------

def test_extract_vitals_returns_partial_vitals_for_flat_signal() -> None:
    proc = PPGSignalProcessor(sampling_rate=FS)
    vitals = proc.extract_vitals(_flat_signal(1000), fs=FS)

    assert vitals.hr_bpm is None
    assert vitals.rr_intervals_ms is None
    assert vitals.signal_quality_score is not None
    assert vitals.signal_quality_score == pytest.approx(0.0, abs=1e-6)


def test_extract_vitals_ignores_lead_index() -> None:
    """lead_index is a no-op; different values must give identical output."""
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    v0 = proc.extract_vitals(signal, lead_index=0, fs=FS)
    v1 = proc.extract_vitals(signal, lead_index=2, fs=FS)

    assert v0.hr_bpm == v1.hr_bpm
    assert v0.rr_intervals_ms == v1.rr_intervals_ms


def test_extract_vitals_uses_instance_sampling_rate_when_fs_missing() -> None:
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    vitals_explicit = proc.extract_vitals(signal, fs=FS)
    vitals_default = proc.extract_vitals(signal)

    assert vitals_explicit.hr_bpm == vitals_default.hr_bpm


# ---------------------------------------------------------------------------
# Accel motion quality penalty
# ---------------------------------------------------------------------------

def test_quality_lower_with_high_motion_accel() -> None:
    signal = _synthetic_ppg(duration_s=12.0, hr_bpm=60.0)
    proc = PPGSignalProcessor(sampling_rate=FS)

    n_accel = int(12.0 * 50)  # 50 Hz accel
    accel_rest = np.zeros((n_accel, 3), dtype=float)
    accel_motion = np.ones((n_accel, 3), dtype=float) * 2.0  # 2 g — high motion

    v_rest = proc.extract_vitals(signal, fs=FS, accel=accel_rest)
    v_motion = proc.extract_vitals(signal, fs=FS, accel=accel_motion)

    assert v_rest.signal_quality_score is not None
    assert v_motion.signal_quality_score is not None
    assert v_motion.signal_quality_score < v_rest.signal_quality_score


# ---------------------------------------------------------------------------
# _resolve_fs validation
# ---------------------------------------------------------------------------

def test_resolve_fs_rejects_non_integer_float() -> None:
    proc = PPGSignalProcessor(sampling_rate=FS)
    with pytest.raises(ValueError, match="integer value"):
        proc.extract_vitals(np.zeros(100), fs=100.5)


def test_resolve_fs_rejects_zero() -> None:
    proc = PPGSignalProcessor(sampling_rate=FS)
    with pytest.raises(ValueError, match="positive"):
        proc.extract_vitals(np.zeros(100), fs=0)


# ---------------------------------------------------------------------------
# Duck-type compatibility with ProactivePipeline
# ---------------------------------------------------------------------------

def test_ppg_processor_is_pipeline_compatible() -> None:
    """PPGSignalProcessor must work as drop-in for ECGSignalProcessor in pipeline."""
    from agent.proactive.pipeline import ProactivePipeline
    from agent.proactive.intervention_decision import Modality
    from agent.data.streaming import PPGWindow

    proc = PPGSignalProcessor(sampling_rate=FS)
    pipeline = ProactivePipeline(modality=Modality.PPG, signal_processor=proc)

    signal = _synthetic_ppg(duration_s=10.0, hr_bpm=70.0)
    window = PPGWindow(
        signal=signal,
        fs=FS,
        patient_id="001",
        chunk_index=0,
        window_index=0,
        start_sample=0,
        end_sample=len(signal),
        n_samples=len(signal),
        stream_offset_s=0.0,
    )

    result = pipeline.process_window(window)

    assert "intervene" in result
    assert "rhythm_class" in result
    assert "window_index" in result
