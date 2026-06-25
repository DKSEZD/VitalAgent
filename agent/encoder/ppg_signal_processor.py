"""PPG signal processing — systolic peak detection, pulse rate, PRV, quality."""

from __future__ import annotations

import logging
from numbers import Real

import numpy as np

from agent.encoder.ecg_signal_processor import VitalParams

logger = logging.getLogger(__name__)


class PPGSignalProcessor:
    """
    Extracts vital parameters from raw PPG (photoplethysmography) signals.

    Output is ``VitalParams`` — the same structure as ``ECGSignalProcessor`` —
    so the downstream pipeline (tracker, rhythm classifier, intervention
    decision) is completely signal-source agnostic.

    Field mapping:
        hr_bpm           → pulse rate (beats per minute)
        rr_intervals_ms  → pulse-to-pulse (PP) intervals in milliseconds
        sdnn_ms          → pulse rate variability SDNN
        rmssd_ms         → pulse rate variability RMSSD
        signal_quality_score → composite quality score in [0, 1]
    """

    # Physiological plausibility bounds for PP intervals
    _PP_MIN_MS: float = 250.0   # 240 bpm upper limit
    _PP_MAX_MS: float = 2000.0  # 30 bpm lower limit

    # Minimum peaks required for meaningful statistics
    _MIN_PEAKS: int = 4

    def __init__(self, sampling_rate: int = 100) -> None:
        self.sampling_rate = sampling_rate

    # ------------------------------------------------------------------
    # Public API (mirrors ECGSignalProcessor)
    # ------------------------------------------------------------------

    def detect_peaks(
        self,
        signal: np.ndarray,
        fs: int | None = None,
    ) -> np.ndarray:
        """
        Detect systolic peaks in a PPG green channel signal.

        Returns sample indices of detected peaks (may be empty).
        """
        fs = self._resolve_fs(fs)
        cleaned = self._clean_signal(signal, fs)
        peaks = self._detect_peaks_primary(cleaned, fs)
        if peaks is None or len(peaks) < self._MIN_PEAKS:
            peaks = self._detect_peaks_fallback(cleaned, fs)
        if peaks is None or len(peaks) == 0:
            return np.array([], dtype=int)
        peaks = self._suppress_dicrotic_notch(np.asarray(peaks, dtype=int))
        return peaks

    def extract_vitals(
        self,
        signal: np.ndarray,
        lead_index: int = 0,
        fs: int | None = None,
        accel: np.ndarray | None = None,
    ) -> VitalParams:
        """
        Extract VitalParams from one PPG green-channel window.

        Parameters
        ----------
        signal : np.ndarray, shape (n_samples,)
            Raw or lightly pre-processed green channel in ADC units.
        lead_index : int
            Ignored — PPG is single-channel. Kept for duck-type
            compatibility with ``ECGSignalProcessor.extract_vitals``.
        fs : int | None
            Sampling rate. Falls back to ``self.sampling_rate``.
        accel : np.ndarray | None
            Accelerometer array, shape ``(n_accel_samples, 3)`` (XYZ),
            same window duration as ``signal``. If supplied, high-motion
            windows receive a lower quality score.
        """
        fs = self._resolve_fs(fs)
        signal = np.asarray(signal, dtype=float).ravel()

        quality = self._assess_quality(signal, fs, accel=accel)
        peaks = self.detect_peaks(signal, fs=fs)

        if len(peaks) < self._MIN_PEAKS:
            logger.debug(
                "Too few PPG peaks detected (%d); returning partial vitals.", len(peaks)
            )
            return VitalParams(signal_quality_score=quality)

        pp_ms = (np.diff(peaks).astype(float) / fs) * 1000.0

        # Remove physiologically implausible intervals (artefact / missed peaks)
        mask = (pp_ms >= self._PP_MIN_MS) & (pp_ms <= self._PP_MAX_MS)
        pp_clean = pp_ms[mask]

        if len(pp_clean) < 2:
            return VitalParams(signal_quality_score=quality)

        pulse_rate = round(60_000.0 / float(np.mean(pp_clean)), 1)
        prv = self._compute_prv(pp_clean)

        return VitalParams(
            hr_bpm=pulse_rate,
            rr_intervals_ms=pp_clean.tolist(),
            sdnn_ms=prv["sdnn_ms"],
            rmssd_ms=prv["rmssd_ms"],
            signal_quality_score=quality,
        )

    # ------------------------------------------------------------------
    # Signal cleaning
    # ------------------------------------------------------------------

    def _clean_signal(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Bandpass filter via NeuroKit2 if available, else scipy fallback."""
        try:
            import neurokit2 as nk
            return np.asarray(nk.ppg_clean(signal, sampling_rate=fs), dtype=float)
        except Exception:
            return self._bandpass_fallback(signal, fs)

    @staticmethod
    def _bandpass_fallback(signal: np.ndarray, fs: int) -> np.ndarray:
        """0.5–4 Hz third-order Butterworth bandpass (covers 30–240 bpm)."""
        from scipy.signal import butter, filtfilt
        nyq = fs / 2.0
        low = 0.5 / nyq
        high = min(4.0 / nyq, 0.99)
        b, a = butter(3, [low, high], btype="band")
        return np.asarray(filtfilt(b, a, signal), dtype=float)

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def _detect_peaks_primary(self, signal: np.ndarray, fs: int) -> np.ndarray | None:
        """NeuroKit2 PPG peak detector using Elgendi method (robust to dicrotic notch)."""
        try:
            import neurokit2 as nk
            _, info = nk.ppg_peaks(signal, sampling_rate=fs, method="elgendi")
            peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)
            return peaks if len(peaks) >= self._MIN_PEAKS else None
        except Exception:
            return None

    def _detect_peaks_fallback(self, signal: np.ndarray, fs: int) -> np.ndarray | None:
        """scipy find_peaks fallback with prominence-based threshold."""
        try:
            from scipy.signal import find_peaks
            # Minimum distance: 200 bpm cap → 60/200 * fs samples
            min_distance = max(1, int(0.3 * fs))
            prominence = max(1e-3, float(np.std(signal)) * 0.5)
            peaks, _ = find_peaks(
                signal,
                height=float(np.percentile(signal, 50)),
                distance=min_distance,
                prominence=prominence,
            )
            return peaks if len(peaks) >= self._MIN_PEAKS else None
        except Exception:
            return None

    @staticmethod
    def _suppress_dicrotic_notch(peaks: np.ndarray) -> np.ndarray:
        """Drop peaks whose interval to the prior peak is < 0.5 * P75 of all intervals.

        Uses the 75th percentile (not median) as the reference heartbeat period so
        the threshold remains anchored to the true beat period even when dicrotic wave
        peaks outnumber systolic peaks and would pull the median down to the short
        systolic-to-dicrotic gap. Iterates twice to catch residual artifacts.
        """
        for _ in range(2):
            if len(peaks) < 2:
                break
            intervals = np.diff(peaks.astype(float))
            reference_pp = float(np.percentile(intervals, 75))
            threshold = 0.5 * reference_pp
            keep = np.ones(len(peaks), dtype=bool)
            for i in range(1, len(peaks)):
                if keep[i - 1] and (peaks[i] - peaks[i - 1]) < threshold:
                    keep[i] = False
            peaks = peaks[keep]
        return peaks

    # ------------------------------------------------------------------
    # Pulse rate variability
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_prv(pp_ms: np.ndarray) -> dict[str, float | None]:
        if len(pp_ms) < 3:
            return {"sdnn_ms": None, "rmssd_ms": None}
        diffs = np.diff(pp_ms)
        return {
            "sdnn_ms": round(float(np.std(pp_ms, ddof=1)), 2),
            "rmssd_ms": round(float(np.sqrt(np.mean(diffs**2))), 2),
        }

    # ------------------------------------------------------------------
    # Signal quality
    # ------------------------------------------------------------------

    def _assess_quality(
        self,
        signal: np.ndarray,
        fs: int,
        accel: np.ndarray | None = None,
    ) -> float:
        """
        Composite quality score in [0, 1].

        Based on signal SNR (power vs high-frequency noise) with an
        optional motion penalty from accelerometer SMA when accel is given.
        """
        score = self._snr_score(signal)
        if accel is not None:
            motion = self._motion_intensity(accel)
            # Scale score down by up to 0.5 at peak motion
            score = score * (1.0 - 0.5 * motion)
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _snr_score(signal: np.ndarray) -> float:
        """Signal power vs second-derivative noise, clamped to [0, 1]."""
        if len(signal) < 10:
            return 0.0
        if float(np.ptp(signal)) < 1e-9:
            return 0.0
        sig_std = float(np.std(signal))
        noise_std = float(np.std(np.diff(signal, n=2))) + 1e-9
        return float(np.tanh((sig_std / noise_std) / 5.0))

    @staticmethod
    def _motion_intensity(accel: np.ndarray) -> float:
        """
        Motion intensity in [0, 1] from accelerometer SMA.

        SMA = mean(|ax| + |ay| + |az|). Uses soft saturation so that
        typical wrist movement at ~0.5 g maps to ~0.5 and rest maps to ~0.
        """
        accel = np.asarray(accel, dtype=float)
        if accel.ndim != 2 or accel.shape[1] < 3:
            return 0.0
        sma = float(np.mean(np.abs(accel[:, 0]) + np.abs(accel[:, 1]) + np.abs(accel[:, 2])))
        return float(np.tanh(sma * 2.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_fs(self, fs: int | None) -> int:
        if fs is None:
            return self.sampling_rate
        if isinstance(fs, bool) or not isinstance(fs, Real):
            raise ValueError(f"Sampling rate must be numeric, got {fs!r}")
        if isinstance(fs, float) and not float(fs).is_integer():
            raise ValueError(f"Sampling rate must be an integer value, got {fs!r}")
        result = int(fs)
        if result <= 0:
            raise ValueError(f"Sampling rate must be positive, got {result!r}")
        return result
