"""
Traditional Signal Processing — extracts interpretable vital parameters
from raw ECG signals, running in parallel with the ECG Encoder.

Reuses ``agent.tools.ecg.utils.process_ecg`` (shared with Reactive tools)
for R-peak detection and HR computation.  Falls back to lightweight local
methods when the shared function fails (e.g. very short or noisy windows).

Responsibilities:
    - Heart Rate (HR): R-peak detection → RR interval → BPM
    - HRV metrics: SDNN, RMSSD
    - Signal quality assessment
    - Morphological params (QRS, PR, QT/QTc) — Phase 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from numbers import Real

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VitalParams:
    """Structured output of traditional signal processing."""

    hr_bpm: float | None = None
    rr_intervals_ms: list[float] | None = None
    # HRV
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    # Morphology  [Phase 2]
    qrs_duration_ms: float | None = None
    pr_interval_ms: float | None = None
    qt_interval_ms: float | None = None
    qtc_interval_ms: float | None = None
    # Signal quality
    signal_quality_score: float | None = None
    baseline_drift_level: float | None = None
    noise_level: float | None = None

    def to_dict(self) -> dict:
        """Convenience serialiser for HealthStateTracker consumption."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ECGSignalProcessor:
    """
    Extracts clinically interpretable parameters from raw ECG.

    Delegates to ``agent.tools.ecg.utils.process_ecg`` (NeuroKit2) where
    possible, keeping the same processing path as the Reactive tools.
    """

    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_r_peaks(
        self,
        signal: np.ndarray,
        fs: int | None = None,
        lead_index: int = 0,
    ) -> np.ndarray:
        """
        Detect R-peak locations in an ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Shape ``(n_samples,)`` for single-lead, or ``(num_leads, n_samples)``.
        fs : int | None
            Sampling rate for this call. Falls back to ``self.sampling_rate``.
        lead_index : int
            Which lead to analyse when multi-lead input is given.

        Returns
        -------
        np.ndarray
            Indices of detected R-peaks.
        """
        single = self._select_lead(signal, lead_index)
        effective_fs = self._resolve_fs(fs)
        r_peaks, _ = self._detect_r_peaks_with_rate(single, effective_fs)
        if r_peaks is None:
            return np.array([], dtype=int)
        return np.asarray(r_peaks, dtype=int)

    def compute_hrv(self, rr_intervals_ms: np.ndarray) -> dict[str, float | None]:
        """
        Compute HRV metrics from RR intervals.

        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            RR intervals in milliseconds.

        Returns
        -------
        dict
            ``{"sdnn_ms": ..., "rmssd_ms": ...}``
        """
        rr_intervals_ms = np.asarray(rr_intervals_ms, dtype=float)
        if rr_intervals_ms.size == 0:
            return {"sdnn_ms": None, "rmssd_ms": None}

        mask = (rr_intervals_ms >= 300.0) & (rr_intervals_ms <= 2000.0)
        nn_clean = rr_intervals_ms[mask]
        if len(nn_clean) < 3:
            return {"sdnn_ms": None, "rmssd_ms": None}

        nn_diffs = np.diff(nn_clean)
        return {
            "sdnn_ms": round(float(np.std(nn_clean, ddof=1)), 2),
            "rmssd_ms": round(float(np.sqrt(np.mean(nn_diffs ** 2))), 2),
        }

    def assess_signal_quality(
        self,
        signal: np.ndarray,
        fs: int | None = None,
        lead_index: int = 0,
    ) -> float:
        """
        Assess overall signal quality (0 = poor, 1 = excellent).

        Parameters
        ----------
        signal : np.ndarray
            Shape ``(n_samples,)`` for single-lead, or ``(num_leads, n_samples)``.
        fs : int | None
            Sampling rate for this call. Falls back to ``self.sampling_rate``.
        lead_index : int
            Which lead to analyse when multi-lead input is given.

        Returns
        -------
        float
            Quality score in ``[0, 1]``.
        """
        single = self._select_lead(signal, lead_index)
        effective_fs = self._resolve_fs(fs)
        return self._estimate_quality(single, effective_fs)

    def extract_vitals(
        self,
        signal: np.ndarray,
        lead_index: int = 0,
        fs: int | None = None,
    ) -> VitalParams:
        """
        Extract vital parameters from an ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Shape ``(n_samples,)`` for single-lead, or ``(num_leads, n_samples)``.
        lead_index : int
            Which lead to analyse when multi-lead input is given.
        fs : int | None
            Sampling rate for this call. Falls back to ``self.sampling_rate``.

        Returns
        -------
        VitalParams
        """
        single = self._select_lead(signal, lead_index)
        effective_fs = self._resolve_fs(fs)
        r_peaks = self.detect_r_peaks(single, fs=effective_fs)

        if r_peaks is None or len(r_peaks) < 2:
            logger.warning("Fewer than 2 R-peaks detected; returning partial vitals.")
            return VitalParams(signal_quality_score=self.assess_signal_quality(single, fs=effective_fs))

        # ----------------------------------------------------------
        # HR
        # ----------------------------------------------------------
        rr_mean_s = np.mean(np.diff(r_peaks)) / effective_fs
        hr_bpm = round(60.0 / rr_mean_s, 1) if rr_mean_s > 0 else None

        # ----------------------------------------------------------
        # RR intervals → HRV  (same filtering as agent.tools.ecg.tools.analyze_hrv)
        # ----------------------------------------------------------
        rr_samples = np.diff(r_peaks)
        nn_ms = (rr_samples / effective_fs) * 1000.0
        rr_list = nn_ms.tolist()
        hrv = self.compute_hrv(nn_ms)

        # ----------------------------------------------------------
        # Signal quality
        # ----------------------------------------------------------
        quality = self.assess_signal_quality(single, fs=effective_fs)

        return VitalParams(
            hr_bpm=hr_bpm,
            rr_intervals_ms=rr_list,
            sdnn_ms=hrv["sdnn_ms"],
            rmssd_ms=hrv["rmssd_ms"],
            signal_quality_score=quality,
        )

    # ------------------------------------------------------------------
    # Shared Reactive-tool path
    # ------------------------------------------------------------------

    def _detect_r_peaks_with_rate(
        self,
        single_lead: np.ndarray,
        sampling_rate: int,
    ) -> tuple[np.ndarray | None, object | None]:
        """Run the full detection chain once and return peaks plus optional rate output."""
        r_peaks, ecg_rate = self._process_ecg_shared(single_lead, sampling_rate)
        if r_peaks is None or len(r_peaks) < 2:
            r_peaks = self._fallback_r_peaks(single_lead, sampling_rate)
            ecg_rate = None
        return r_peaks, ecg_rate

    def _process_ecg_shared(self, single_lead: np.ndarray, sampling_rate: int):
        """
        Call ``agent.tools.ecg.utils.process_ecg`` — the same function used
        by ``analyze_heart_rate`` and ``analyze_hrv`` in the Reactive tools.

        Returns
        -------
        (r_peaks, ecg_rate) or (None, None) on failure.
            r_peaks : np.ndarray of R-peak indices
            ecg_rate : pd.Series of instantaneous HR (may be None)
        """
        try:
            from agent.tools.ecg.utils import process_ecg

            signals, info = process_ecg(single_lead, sampling_rate)
            r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            ecg_rate = signals.get("ECG_Rate")
            if ecg_rate is not None:
                ecg_rate = ecg_rate.dropna()
            return r_peaks, ecg_rate
        except ImportError:
            logger.debug("agent.tools.ecg.utils.process_ecg not available; using fallback.")
            return None, None
        except Exception as exc:
            logger.debug("process_ecg failed (%s); using fallback.", exc)
            return None, None

    # ------------------------------------------------------------------
    # Fallback methods (when shared path is unavailable or fails)
    # ------------------------------------------------------------------

    def _fallback_r_peaks(self, single_lead: np.ndarray, sampling_rate: int) -> np.ndarray | None:
        """NeuroKit2 ecg_peaks as standalone fallback, then scipy if that also fails."""
        try:
            import neurokit2 as nk

            cleaned = nk.ecg_clean(single_lead, sampling_rate=sampling_rate)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
            return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        except Exception:
            pass

        try:
            from scipy.signal import find_peaks

            threshold = np.mean(single_lead) + 0.6 * np.std(single_lead)
            min_distance = int(0.3 * sampling_rate)
            peaks, _ = find_peaks(single_lead, height=threshold, distance=min_distance)
            return peaks
        except Exception:
            return None

    def _estimate_quality(self, single_lead: np.ndarray, sampling_rate: int) -> float:
        """Quick signal quality score in [0, 1]."""
        try:
            import neurokit2 as nk

            cleaned = nk.ecg_clean(single_lead, sampling_rate=sampling_rate)
            q = nk.ecg_quality(cleaned, sampling_rate=sampling_rate, approach="zhao2018")
            score = float(np.nanmean(q)) if hasattr(q, "__len__") else float(q)
            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return self._fallback_quality(single_lead)

    @staticmethod
    def _fallback_quality(single_lead: np.ndarray) -> float:
        """SNR heuristic: signal power / high-freq noise power."""
        if len(single_lead) < 10:
            return 0.0
        sig_power = np.var(single_lead)
        noise_power = np.var(np.diff(single_lead))
        if noise_power == 0:
            return 1.0
        return round(min(1.0, sig_power / noise_power / 10.0), 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_fs(self, fs: int | None) -> int:
        if fs is None:
            effective_fs = self.sampling_rate
        else:
            if isinstance(fs, bool) or not isinstance(fs, Real):
                raise ValueError(f"Sampling rate must be numeric, got {fs!r}")
            if isinstance(fs, float) and not float(fs).is_integer():
                raise ValueError(f"Sampling rate must be an integer value, got {fs!r}")
            effective_fs = int(fs)
        if effective_fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {effective_fs!r}")
        return effective_fs

    @staticmethod
    def _select_lead(signal: np.ndarray, lead_index: int) -> np.ndarray:
        if signal.ndim == 1:
            return signal
        if signal.ndim == 2:
            return signal[lead_index]
        raise ValueError(f"Expected 1-D or 2-D signal, got shape {signal.shape}")
