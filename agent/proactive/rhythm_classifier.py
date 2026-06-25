"""
Rhythm classifier — lightweight AF detection from RR intervals.

This module provides a single-function classifier that takes a window's
RR interval series and returns a coarse rhythm label. It is designed
for continuous monitoring on resource-constrained wearable devices:
the entire classification runs in under 100 μs per 10-second window.

Algorithm (three-feature heuristic):
    1. **Coefficient of Variation (CV)** of RR intervals.
       CV = std(RR) / mean(RR). Normal sinus rhythm has low CV
       (regular R-R); AF has high CV (irregularly irregular).

    2. **Shannon entropy** of the ΔRR (successive differences)
       distribution, binned into fixed-width bins. AF produces a
       near-uniform ΔRR distribution (high entropy); normal sinus
       rhythm produces a concentrated distribution (low entropy).

    3. **Turning Point Ratio (TPR)** of the RR series. A turning point
       is an RR value that is either a local minimum or local maximum
       (strictly less than or strictly greater than both neighbours).
       For a purely random sequence, TPR converges to 2/3 ≈ 0.667.
       AF's irregularly-irregular pattern produces TPR close to this
       theoretical value. Normal sinus rhythm and rhythms with frequent
       ectopic beats (PVC/PAC bigeminy, trigeminy) produce regular
       short-long-short alternation patterns whose TPR deviates from
       the random baseline — typically higher (> 0.75) because every
       ectopic creates a local extremum in a predictable pattern.

    Decision logic:
       - CV > threshold AND entropy > threshold AND TPR in AF range
         → "AF"
       - CV < normal threshold → "N" (normal sinus rhythm)
       - Otherwise → "Other" (ambiguous — let the LLM judge decide)

    If the RR series is too short to compute meaningful statistics,
    return "Unknown".

Threshold calibration:
    Current values are drawn from Tateno & Glass (2001), Dash et al.
    (2009), and follow-up literature on RR-irregularity-based AF
    screening. They should be calibrated on annotated Icentia11k
    patients in Day 6 evaluation.

References:
    - Tateno K, Glass L. Automatic detection of atrial fibrillation
      using the coefficient of sample entropy. Med Biol Eng Comput.
      2001;39(6):664-671.
    - Dash S, et al. Automatic real time detection of atrial
      fibrillation. Ann Biomed Eng. 2009;37(9):1701-1709.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — calibrate on annotated patients in Day 6 evaluation
# ---------------------------------------------------------------------------

# Calibrated on Icentia11k patient 01182 (AF/AFL ground truth) via
# grid search in scripts/proactive/calibrate_rhythm.py (Day 6).  Sweep covered:
#   CV ∈ {0.06,0.08,0.10,0.12,0.14}, ENT ∈ {0.40,0.50,0.60,0.70,0.80},
#   TPR_LOW ∈ {0.35,0.40,0.45,0.50,0.55,0.60}, TPR_HIGH ∈ {0.70,…,0.90}
# Best combination by transition-F1 (debounce=2): CV=0.06, ENT=0.50,
# TPR_LOW=0.45, TPR_HIGH=0.80 → AF sensitivity=87%, transition F1=0.74.

# Coefficient of variation thresholds
_AF_CV_THRESHOLD: float = 0.06       # above → candidate AF (calibrated Day 6)
_NORMAL_CV_THRESHOLD: float = 0.05   # below → confident normal sinus

# Shannon entropy of ΔRR thresholds
_AF_ENTROPY_THRESHOLD: float = 0.50  # above → candidate AF (calibrated Day 6)

# Turning Point Ratio range for AF.
# Pure random series → TPR ≈ 0.667. AF is near-random, so we expect
# TPR in a band around this theoretical value. Frequent ectopics
# (PVC/PAC) produce short-long-short alternation → TPR tends to be
# higher (> 0.80) because each ectopic is a predictable turning point.
# Lower bound relaxed from 0.55 → 0.45 to capture AF windows with
# slightly more regular intervals (seen in AFL and fast AF).
_AF_TPR_LOW: float = 0.45   # below → too regular for AF (calibrated Day 6)
_AF_TPR_HIGH: float = 0.80  # above → likely ectopic pattern, not AF (calibrated Day 6)

# Minimum number of RR/PP intervals required for classification.
# With fewer than this, statistics are too noisy to be meaningful.
# At typical HR of 60-180 bpm, a 10-second window has 8-30 intervals.
# Lowered from 8 to 5 to cover PPG bradycardia / partial-window cases
# where peak detection yields fewer pulses than the ECG equivalent.
_MIN_RR_COUNT: int = 5

# Number of bins for the ΔRR histogram used in entropy calculation.
_ENTROPY_BINS: int = 16


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_rhythm(rr_intervals_ms: list[float] | None) -> str:
    """
    Classify the cardiac rhythm from a window's RR interval series.

    Parameters
    ----------
    rr_intervals_ms : list[float] | None
        RR intervals in milliseconds for one window. None or empty
        returns "Unknown".

    Returns
    -------
    str
        One of:
        - ``"N"``       — normal sinus rhythm (regular RR)
        - ``"AF"``      — atrial fibrillation (irregularly irregular)
        - ``"Other"``   — ambiguous, does not clearly fit N or AF
        - ``"Unknown"`` — insufficient data for classification
    """
    if not rr_intervals_ms or len(rr_intervals_ms) < _MIN_RR_COUNT:
        return "Unknown"

    rr = np.asarray(rr_intervals_ms, dtype=float)

    # Filter out physiologically implausible intervals (artefacts)
    mask = (rr >= 200.0) & (rr <= 2500.0)
    rr_clean = rr[mask]
    if len(rr_clean) < _MIN_RR_COUNT:
        return "Unknown"

    cv = _coefficient_of_variation(rr_clean)
    entropy = _delta_rr_entropy(rr_clean)
    tpr = _turning_point_ratio(rr_clean)

    # Decision logic
    #
    # AF requires ALL three conditions:
    #   1. High CV (irregularly irregular)
    #   2. High entropy (near-uniform ΔRR distribution)
    #   3. TPR in the "random" band (not the ectopic pattern)
    #
    # The TPR gate is what distinguishes AF from frequent ectopics:
    # PVC/PAC bigeminy/trigeminy creates short-long alternation where
    # nearly every beat is a turning point (TPR > 0.75), while AF's
    # randomness keeps TPR closer to 0.667.
    if (
        cv > _AF_CV_THRESHOLD
        and entropy > _AF_ENTROPY_THRESHOLD
        and _AF_TPR_LOW <= tpr <= _AF_TPR_HIGH
    ):
        logger.debug(
            "Rhythm classified as AF (CV=%.3f, entropy=%.3f, TPR=%.3f)",
            cv, entropy, tpr,
        )
        return "AF"

    if cv < _NORMAL_CV_THRESHOLD:
        return "N"

    # Intermediate CV, or AF-like CV+entropy but ectopic TPR pattern.
    logger.debug(
        "Rhythm classified as Other (CV=%.3f, entropy=%.3f, TPR=%.3f)",
        cv, entropy, tpr,
    )
    return "Other"


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _coefficient_of_variation(rr: np.ndarray) -> float:
    """CV = std(RR) / mean(RR). Dimensionless measure of RR irregularity."""
    mean = float(np.mean(rr))
    if mean <= 0:
        return 0.0
    return float(np.std(rr, ddof=0)) / mean


def _delta_rr_entropy(rr: np.ndarray, n_bins: int = _ENTROPY_BINS) -> float:
    """
    Shannon entropy of the ΔRR distribution, normalised to [0, 1].

    ΔRR = successive differences of the RR series. We bin these into
    ``n_bins`` equal-width bins spanning the observed range and compute
    the Shannon entropy of the resulting histogram, normalised by
    log2(n_bins) so the output is in [0, 1]:
        0.0 = all ΔRR values in one bin (perfectly regular)
        1.0 = uniform distribution across all bins (maximally irregular)

    Returns 0.0 if there are fewer than 2 intervals (no differences).
    """
    if len(rr) < 2:
        return 0.0

    delta_rr = np.diff(rr)
    if len(delta_rr) == 0:
        return 0.0

    # If all ΔRR values are identical (or nearly so), entropy is 0.
    rng = float(np.max(delta_rr)) - float(np.min(delta_rr))
    if rng < 1e-6:
        return 0.0

    # Build histogram
    counts, _ = np.histogram(delta_rr, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return 0.0

    # Shannon entropy (normalised)
    probs = counts / total
    # Filter out zero-probability bins to avoid log2(0)
    probs = probs[probs > 0]
    raw_entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(n_bins)
    if max_entropy <= 0:
        return 0.0

    return raw_entropy / max_entropy


def _turning_point_ratio(rr: np.ndarray) -> float:
    """
    Fraction of interior RR values that are local extrema.

    A turning point at index i means RR[i] is strictly greater than
    both neighbours (local max) or strictly less than both neighbours
    (local min). The ratio is computed over all interior points
    (indices 1 through N-2).

    For a purely random i.i.d. sequence, TPR → 2/3 ≈ 0.667.
    For AF (irregularly irregular), TPR is close to this value.
    For regular rhythms with ectopic beats (bigeminy, trigeminy),
    the alternating short-long pattern pushes TPR above 0.75.

    Returns 0.0 if the series has fewer than 3 values (no interior
    points).

    Reference: Dash S, et al. Ann Biomed Eng. 2009;37(9):1701-1709.
    """
    if len(rr) < 3:
        return 0.0

    n_interior = len(rr) - 2
    turning_points = 0

    for i in range(1, len(rr) - 1):
        if (rr[i] > rr[i - 1] and rr[i] > rr[i + 1]) or \
           (rr[i] < rr[i - 1] and rr[i] < rr[i + 1]):
            turning_points += 1

    return turning_points / n_interior
