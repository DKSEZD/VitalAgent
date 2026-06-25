"""
Health State Tracker — maintains a temporal health profile via
multi-scale sliding windows.

Responsibilities (§2.1 Proactive Engine):
    - Short window  (real-time ~ 5 min):  current HR, rhythm
    - Medium window (1 hr ~ 24 hr):       HRV, anomaly frequency, HR stats
    - Long window   (7 days ~ 30 days):   baseline establishment, long-term trends

Dual-channel input:
    - Numeric channel:   vital params from traditional signal processing
    - Embedding channel: ECG encoder embeddings for distribution drift detection
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window data structures
# ---------------------------------------------------------------------------
@dataclass
class ShortWindow:
    """Real-time ~ 5 min: current beat-level metrics."""

    hr_bpm: float | None = None
    rhythm_regular: bool | None = None
    latest_rr_intervals_ms: list[float] = field(default_factory=list)
    # Fraction of the recent rolling window with HR > 100 bpm.
    # None until the rolling window has at least one sample.
    tachycardia_ratio_5min: float | None = None
    tachycardia_sample_count: int = 0
    # Rhythm classification from RR-interval-based classifier.
    # "N" = normal sinus, "AF" = atrial fibrillation,
    # "Other" = ambiguous, None / "Unknown" = not enough data.
    rhythm_class: str | None = None
    # Seconds the current AF episode has lasted (None if not in AF).
    af_episode_duration_s: float | None = None
    timestamp: str = ""


@dataclass
class MediumWindow:
    """1 hr ~ 24 hr: aggregated statistics."""

    mean_hr: float | None = None
    min_hr: float | None = None
    max_hr: float | None = None
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    anomaly_count: int = 0
    timestamp_start: str = ""
    timestamp_end: str = ""


@dataclass
class LongWindow:
    """7 days ~ 30 days: baseline and trends."""

    resting_hr_baseline: float | None = None
    hr_trend: list[float] = field(default_factory=list)  # daily avg HR
    hrv_trend: list[float] = field(default_factory=list)  # daily SDNN
    anomaly_frequency: list[int] = field(default_factory=list)  # daily anomaly count


@dataclass
class EmbeddingBaseline:
    """Personal embedding distribution baseline (§3.4)."""

    mean_vector: np.ndarray | None = None  # μ_base
    distance_mean: float = 0.0  # μ_d
    distance_std: float = 1.0  # σ_d
    sample_count: int = 0
    cold_start_complete: bool = False  # True after enough samples

    # How many embeddings before cold-start is considered done.
    # At 10-second windows, 7 days ≈ 60480 windows — but for practical
    # MVP we use a much smaller threshold (e.g. 100 windows).
    COLD_START_THRESHOLD: int = 100


# ---------------------------------------------------------------------------
# Internal ring buffers for numeric aggregation
# ---------------------------------------------------------------------------
# We keep raw HR/HRV readings in deques so we can recompute aggregates.
# Max lengths are tuned for Icentia11k (250 Hz, 10-sec windows → 1 window/10s):
#   short:  ~30 windows  = 5 min
#   medium: ~360 windows = 1 hr  (MVP; expand to 8640 for 24 hr if needed)
# RR intervals are more numerous than windows, so we keep a larger RR buffer
# for computing medium-window HRV directly from the concatenated NN series.

_SHORT_MAXLEN = 30
_MEDIUM_MAXLEN = 360
_RR_PER_WINDOW_MAX = 30
# Sustained-tachycardia tracking.
# Tracks a rolling window of HR readings used to compute the fraction
# above a threshold. At 10 s windows, 30 entries = 5 minutes.
_TACHY_WINDOW_LEN = 30           # number of recent HR samples retained
_TACHY_HR_THRESHOLD_BPM = 100.0  # SVT-2015 sinus-tachycardia definition

# ---------------------------------------------------------------------------
# Health State Tracker
# ---------------------------------------------------------------------------
class HealthStateTracker:
    """
    Maintains a temporal health profile with three sliding windows
    and dual-channel anomaly detection.
    """

    def __init__(self) -> None:
        self.short_window = ShortWindow()
        self.medium_window = MediumWindow()
        self.long_window = LongWindow()
        self.embedding_baseline = EmbeddingBaseline()

        # Internal ring buffers
        self._hr_short: deque[float] = deque(maxlen=_SHORT_MAXLEN)
        self._rr_short: deque[float] = deque(maxlen=_SHORT_MAXLEN * 30)  # many RR per window
        self._hr_medium: deque[float] = deque(maxlen=_MEDIUM_MAXLEN)
        self._rr_medium: deque[float] = deque(maxlen=_MEDIUM_MAXLEN * _RR_PER_WINDOW_MAX)
        self._hr_tachy_window: deque[float] = deque(maxlen=_TACHY_WINDOW_LEN)

        # AF episode tracking. We accumulate only on confirmed AF windows;
        # "Unknown" preserves the current episode but does not add time.
        self._af_episode_duration_s: float | None = None
        self._window_counter: int = 0

        # Embedding accumulation during cold-start
        self._embedding_distances: list[float] = []

        # Latest embedding drift z-score (persisted so get_current_state can expose it)
        self._latest_embedding_zscore: float = 0.0

        logger.info("HealthStateTracker initialised.")

    # ------------------------------------------------------------------
    # Numeric channel
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Numeric channel
    # ------------------------------------------------------------------

    def update_vitals(
        self,
        vital_params: dict[str, Any],
        update_baseline: bool = True,
        rhythm_class: str | None = None,
        window_duration_s: float = 10.0,
    ) -> None:
        """
        Update windows with new vital parameters from ECGSignalProcessor.

        Parameters
        ----------
        vital_params : dict
            Output of ``VitalParams.to_dict()``, keys such as
            ``hr_bpm``, ``rr_intervals_ms``, ``sdnn_ms``, ``rmssd_ms``,
            ``signal_quality_score``.
        update_baseline : bool
            Whether the long-window resting HR baseline may be updated.
        rhythm_class : str | None
            Output of ``rhythm_classifier.classify_rhythm()``. One of
            "N", "AF", "Other", "Unknown", or None.
        window_duration_s : float
            Duration of one window in seconds. Used for AF episode
            timing. Default 10.0 (Icentia11k standard).
        """
        hr = vital_params.get("hr_bpm")
        rr_list = vital_params.get("rr_intervals_ms", [])

        # --- Short window: HR + tachycardia tracking ---
        self.short_window.hr_bpm = hr

        if hr is not None:
            self._hr_short.append(hr)
        # Always append (use NaN for missing) so the deque truly spans the
        # last N windows of wall time rather than the last N valid readings.
        self._hr_tachy_window.append(hr if hr is not None else float("nan"))
        valid = [h for h in self._hr_tachy_window if not math.isnan(h)]
        if valid:
            total = len(self._hr_tachy_window)
            above = sum(1 for h in valid if h > _TACHY_HR_THRESHOLD_BPM)
            self.short_window.tachycardia_ratio_5min = above / total
            self.short_window.tachycardia_sample_count = total
        else:
            self.short_window.tachycardia_ratio_5min = None
            self.short_window.tachycardia_sample_count = 0

        # --- Short window: RR intervals ---
        if rr_list:
            for rr in rr_list:
                self._rr_short.append(rr)
            self.short_window.latest_rr_intervals_ms = list(rr_list[-10:])
        else:
            self.short_window.latest_rr_intervals_ms = []

        # --- Short window: rhythm class + AF episode tracking ---
        self._window_counter += 1
        self.short_window.rhythm_class = rhythm_class

        if rhythm_class == "AF":
            if self._af_episode_duration_s is None:
                self._af_episode_duration_s = 0.0
            self._af_episode_duration_s += window_duration_s
        elif rhythm_class in ("N", "Other"):
            # Only confirmed non-AF rhythms reset the episode.
            # "Unknown" (insufficient data) preserves the running episode
            # so transient signal-quality drops don't break AHRE timing.
            self._af_episode_duration_s = None

        self.short_window.af_episode_duration_s = self._af_episode_duration_s

        # --- Short window: rhythm_regular (derived from rhythm_class) ---
        # When rhythm_class is available, derive rhythm_regular from it.
        # When not provided (None / "Unknown"), fall back to the old
        # CV heuristic so existing behaviour is preserved.
        if rhythm_class == "N":
            self.short_window.rhythm_regular = True
        elif rhythm_class in ("AF", "Other"):
            self.short_window.rhythm_regular = False
        elif rr_list and len(rr_list) >= 5:
            # Fallback: CV-based regularity check (pre-Day-5 behaviour)
            rr_arr = np.array(rr_list[-20:])
            cv = np.std(rr_arr) / np.mean(rr_arr) if np.mean(rr_arr) > 0 else 1.0
            self.short_window.rhythm_regular = bool(cv < 0.15)
        else:
            self.short_window.rhythm_regular = None

        # --- Medium window update ---
        if hr is not None:
            self._hr_medium.append(hr)
        if rr_list:
            for rr in rr_list:
                self._rr_medium.append(rr)

        if self._hr_medium:
            hr_arr = np.array(self._hr_medium)
            self.medium_window.mean_hr = round(float(np.mean(hr_arr)), 1)
            self.medium_window.min_hr = round(float(np.min(hr_arr)), 1)
            self.medium_window.max_hr = round(float(np.max(hr_arr)), 1)
        medium_hrv = self._compute_medium_hrv()
        self.medium_window.sdnn_ms = medium_hrv["sdnn_ms"]
        self.medium_window.rmssd_ms = medium_hrv["rmssd_ms"]

        # --- Long window: running baseline ---
        # In MVP, we don't have real day boundaries.
        # We just track the running baseline from medium-window data.
        if update_baseline and self.medium_window.mean_hr is not None:
            if self.long_window.resting_hr_baseline is None:
                self.long_window.resting_hr_baseline = self.medium_window.mean_hr
            else:
                current_baseline = self.long_window.resting_hr_baseline
                deviation = abs(self.medium_window.mean_hr - current_baseline)
                deviation_ratio = deviation / current_baseline if current_baseline > 0 else 0.0

                # Only adapt baseline when the current medium HR remains close
                # to the established resting profile.
                if deviation_ratio < 0.20:
                    alpha = 0.05
                    self.long_window.resting_hr_baseline = round(
                        alpha * self.medium_window.mean_hr
                        + (1 - alpha) * current_baseline,
                        1,
                    )

    def _compute_medium_hrv(self) -> dict[str, float | None]:
        """Compute medium-window HRV from the concatenated RR interval buffer."""
        if len(self._rr_medium) < 3:
            return {"sdnn_ms": None, "rmssd_ms": None}

        rr_arr = np.asarray(self._rr_medium, dtype=float)
        mask = (rr_arr >= 300.0) & (rr_arr <= 2000.0)
        nn_clean = rr_arr[mask]
        if len(nn_clean) < 3:
            return {"sdnn_ms": None, "rmssd_ms": None}

        diffs = np.diff(nn_clean)
        sdnn_ms = round(float(np.std(nn_clean, ddof=1)), 2)
        rmssd_ms = round(float(np.sqrt(np.mean(diffs ** 2))), 2) if len(diffs) else None
        return {"sdnn_ms": sdnn_ms, "rmssd_ms": rmssd_ms}

    # ------------------------------------------------------------------
    # Embedding channel
    # ------------------------------------------------------------------

    def update_embedding(self, embedding: np.ndarray) -> float:
        """
        Process a new ECG embedding: compute drift z-score.

        During cold-start, accumulates distances to build baseline stats.
        After cold-start, returns z-score of current distance.

        Returns
        -------
        float
            Z-score (0.0 during cold-start).
        """
        bl = self.embedding_baseline

        if bl.mean_vector is None:
            # Very first embedding — initialise baseline mean
            bl.mean_vector = embedding.copy().astype(float)
            bl.sample_count = 1
            self._latest_embedding_zscore = 0.0
            return 0.0

        # Compute L2 distance to current baseline mean
        dist = float(np.linalg.norm(embedding - bl.mean_vector))

        if not bl.cold_start_complete:
            # Accumulate during cold-start
            self._embedding_distances.append(dist)
            self._update_baseline(embedding)

            if bl.sample_count >= bl.COLD_START_THRESHOLD:
                # Finalise cold-start stats
                dists = np.array(self._embedding_distances)
                bl.distance_mean = float(np.mean(dists))
                bl.distance_std = float(np.std(dists)) if len(dists) > 1 else 1.0
                if bl.distance_std < 1e-8:
                    bl.distance_std = 1.0
                bl.cold_start_complete = True
                logger.info(
                    "Embedding cold-start complete after %d samples "
                    "(μ_d=%.4f, σ_d=%.4f).",
                    bl.sample_count,
                    bl.distance_mean,
                    bl.distance_std,
                )
            self._latest_embedding_zscore = 0.0
            return 0.0

        # Post cold-start: compute z-score
        z = (dist - bl.distance_mean) / bl.distance_std

        # Slowly update baseline with non-anomalous embeddings (|z| < 3)
        if abs(z) < 3.0:
            self._update_baseline(embedding)
            # Also update distance stats with EMA
            alpha = 0.01
            bl.distance_mean = alpha * dist + (1 - alpha) * bl.distance_mean
            # Running std via Welford-like EMA (approximate)
            bl.distance_std = alpha * abs(dist - bl.distance_mean) + (1 - alpha) * bl.distance_std
            if bl.distance_std < 1e-8:
                bl.distance_std = 1.0

        self._latest_embedding_zscore = round(z, 4)
        return self._latest_embedding_zscore

    def _update_baseline(self, embedding: np.ndarray) -> None:
        """Incrementally update the personal embedding baseline mean (EMA)."""
        bl = self.embedding_baseline
        bl.sample_count += 1
        # Running mean via incremental formula
        bl.mean_vector = bl.mean_vector + (embedding - bl.mean_vector) / bl.sample_count

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_current_state(self) -> dict[str, Any]:
        """
        Return a snapshot of the current health state across all windows.
        Used by RuleBasedJudge.evaluate().
        """
        return {
            "short": {
                "hr_bpm": self.short_window.hr_bpm,
                "rhythm_class": self.short_window.rhythm_class,
                "rhythm_regular": self.short_window.rhythm_regular,
                "af_episode_duration_s": self.short_window.af_episode_duration_s,
                "tachycardia_ratio_5min": self.short_window.tachycardia_ratio_5min,
                "tachycardia_sample_count": self.short_window.tachycardia_sample_count,
            },
            "medium": {
                "mean_hr": self.medium_window.mean_hr,
                "min_hr": self.medium_window.min_hr,
                "max_hr": self.medium_window.max_hr,
                "sdnn_ms": self.medium_window.sdnn_ms,
                "rmssd_ms": self.medium_window.rmssd_ms,
                "anomaly_count": self.medium_window.anomaly_count,
                "window_count": len(self._hr_medium),
            },
            "long": {
                "resting_hr_baseline": self.long_window.resting_hr_baseline,
                "hr_trend": self.long_window.hr_trend,
                "hrv_trend": self.long_window.hrv_trend,
                "anomaly_frequency": self.long_window.anomaly_frequency,
            },
            "embedding": {
                "cold_start_complete": self.embedding_baseline.cold_start_complete,
                "sample_count": self.embedding_baseline.sample_count,
                "distance_mean": self.embedding_baseline.distance_mean,
                "distance_std": self.embedding_baseline.distance_std,
                "latest_zscore": self._latest_embedding_zscore,
            },
        }

    def get_trend_summary(self) -> dict[str, Any]:
        """Return long-window trend data for historical comparison."""
        return {
            "resting_hr_baseline": self.long_window.resting_hr_baseline,
            "hr_trend": self.long_window.hr_trend,
            "hrv_trend": self.long_window.hrv_trend,
            "anomaly_frequency": self.long_window.anomaly_frequency,
        }

    # ------------------------------------------------------------------
    # Anomaly & trend bookkeeping
    # ------------------------------------------------------------------

    def increment_anomaly_count(self) -> None:
        """Called by the pipeline when a distinct alert event is emitted."""
        self.medium_window.anomaly_count += 1

    def append_daily_snapshot(self) -> None:
        """
        Snapshot current medium-window aggregates into long-window trends.

        The name is retained for compatibility, but the current MVP uses
        this as a generic periodic snapshot hook. ``ProactivePipeline``
        calls it at fixed window-count intervals and once at run end so
        trend rules can observe medium-window history.
        """
        if self.medium_window.mean_hr is not None:
            self.long_window.hr_trend.append(self.medium_window.mean_hr)
        if self.medium_window.sdnn_ms is not None:
            self.long_window.hrv_trend.append(self.medium_window.sdnn_ms)
        self.long_window.anomaly_frequency.append(self.medium_window.anomaly_count)

        # Reset medium-window anomaly counter for next period
        self.medium_window.anomaly_count = 0
