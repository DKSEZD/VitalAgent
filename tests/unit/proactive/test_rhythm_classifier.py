"""Tests for agent.proactive.rhythm_classifier (CV + entropy + TPR)."""

from __future__ import annotations

import numpy as np
import pytest

from agent.proactive.rhythm_classifier import (
    _AF_CV_THRESHOLD,
    _AF_ENTROPY_THRESHOLD,
    _AF_TPR_HIGH,
    _AF_TPR_LOW,
    _MIN_RR_COUNT,
    _coefficient_of_variation,
    _delta_rr_entropy,
    _turning_point_ratio,
    classify_rhythm,
)


# ---------------------------------------------------------------------------
# classify_rhythm — edge cases
# ---------------------------------------------------------------------------


def test_none_input_returns_unknown() -> None:
    assert classify_rhythm(None) == "Unknown"


def test_empty_list_returns_unknown() -> None:
    assert classify_rhythm([]) == "Unknown"


def test_too_few_intervals_returns_unknown() -> None:
    assert classify_rhythm([800.0] * (_MIN_RR_COUNT - 1)) == "Unknown"


def test_implausible_intervals_filtered_out() -> None:
    assert classify_rhythm([100.0] * 10) == "Unknown"


# ---------------------------------------------------------------------------
# classify_rhythm — normal sinus rhythm
# ---------------------------------------------------------------------------


def test_regular_rr_classified_as_normal() -> None:
    """Perfectly regular RR (CV ≈ 0) should be classified as N."""
    rr = [800.0] * 20
    assert classify_rhythm(rr) == "N"


def test_slightly_variable_rr_still_normal() -> None:
    """Small physiological variation should still be N."""
    rng = np.random.RandomState(42)
    rr = (800.0 + rng.normal(0, 10, 20)).tolist()
    result = classify_rhythm(rr)
    assert result == "N"


# ---------------------------------------------------------------------------
# classify_rhythm — AF
# ---------------------------------------------------------------------------


def test_true_af_pattern_random_rr() -> None:
    """Simulate AF with uniformly distributed RR (random, TPR ≈ 0.667).
    Should be classified as AF."""
    rng = np.random.RandomState(123)
    # Uniform RR in AF-like range — genuinely random sequence
    rr = rng.uniform(400, 1000, 40).tolist()

    # Verify features are in AF zone
    rr_arr = np.asarray(rr)
    cv = _coefficient_of_variation(rr_arr)
    entropy = _delta_rr_entropy(rr_arr)
    tpr = _turning_point_ratio(rr_arr)

    assert cv > _AF_CV_THRESHOLD, f"CV={cv:.3f}"
    assert entropy > _AF_ENTROPY_THRESHOLD, f"entropy={entropy:.3f}"
    assert _AF_TPR_LOW <= tpr <= _AF_TPR_HIGH, f"TPR={tpr:.3f}"

    assert classify_rhythm(rr) == "AF"


# ---------------------------------------------------------------------------
# classify_rhythm — ectopic beats → Other (not AF)
# ---------------------------------------------------------------------------


def test_bigeminy_pattern_classified_as_other_not_af() -> None:
    """PVC bigeminy: alternating normal-short-normal-short.
    High CV + high entropy, but TPR > 0.75 (every beat is a turning
    point in the short-long alternation). Should be Other, not AF."""
    # Simulate bigeminy: 800ms, 450ms, 800ms, 450ms, ...
    rr = []
    for _ in range(15):
        rr.extend([800.0, 450.0])

    rr_arr = np.asarray(rr)
    cv = _coefficient_of_variation(rr_arr)
    tpr = _turning_point_ratio(rr_arr)

    # Verify this IS the ectopic pattern we're targeting
    assert cv > _AF_CV_THRESHOLD, f"CV={cv:.3f} should be high"
    assert tpr > _AF_TPR_HIGH, f"TPR={tpr:.3f} should be above AF range"

    result = classify_rhythm(rr)
    assert result == "Other", (
        f"Bigeminy should be Other, got {result} (CV={cv:.3f}, TPR={tpr:.3f})"
    )


def test_trigeminy_pattern_classified_as_other_not_af() -> None:
    """PVC trigeminy: normal-normal-short repeating pattern.
    Should be Other because the pattern is structured rather than
    irregularly irregular AF."""
    rr = []
    for _ in range(10):
        rr.extend([800.0, 810.0, 450.0])

    rr_arr = np.asarray(rr)
    cv = _coefficient_of_variation(rr_arr)
    entropy = _delta_rr_entropy(rr_arr)
    tpr = _turning_point_ratio(rr_arr)

    assert cv > _AF_CV_THRESHOLD, f"CV={cv:.3f}"
    assert entropy < _AF_ENTROPY_THRESHOLD, (
        f"entropy={entropy:.3f} should stay below the AF threshold"
    )

    result = classify_rhythm(rr)
    assert result == "Other", (
        f"Trigeminy should be Other, got {result} "
        f"(CV={cv:.3f}, entropy={entropy:.3f}, TPR={tpr:.3f})"
    )


# ---------------------------------------------------------------------------
# classify_rhythm — Other (ambiguous)
# ---------------------------------------------------------------------------


def test_structured_oscillation_classified_as_other() -> None:
    """Structured sawtooth-like RR oscillation (respiratory sinus arrhythmia
    analogue): CV is well above the AF threshold but ΔRR concentrates in
    only two bins (up/down steps), so entropy is well below the AF entropy
    threshold.  Should be Other because AF requires BOTH high CV AND high
    entropy."""
    # Triangle wave: ramp up then ramp down, step=40 ms
    half = 10
    rr = [800.0 + i * 40.0 for i in range(half)] + [
        800.0 + (half - i) * 40.0 for i in range(half)
    ]
    rr_arr = np.asarray(rr)
    cv = _coefficient_of_variation(rr_arr)
    entropy = _delta_rr_entropy(rr_arr)

    # CV is clearly above the AF threshold (high variability)
    assert cv > _AF_CV_THRESHOLD, f"CV={cv:.4f} should be above AF threshold"
    # Entropy is well below (structured deltas: only +40 and -40)
    assert entropy < _AF_ENTROPY_THRESHOLD, (
        f"entropy={entropy:.4f} should be below AF entropy threshold"
    )

    result = classify_rhythm(rr)
    assert result == "Other", (
        f"Structured oscillation should be Other, got {result} "
        f"(CV={cv:.3f}, entropy={entropy:.3f})"
    )


# ---------------------------------------------------------------------------
# Turning Point Ratio
# ---------------------------------------------------------------------------


def test_tpr_constant_series_is_zero() -> None:
    """No turning points in a constant series."""
    rr = np.array([800.0] * 10)
    assert _turning_point_ratio(rr) == 0.0


def test_tpr_monotonic_series_is_zero() -> None:
    """Monotonically increasing → no turning points."""
    rr = np.arange(800.0, 820.0, 1.0)
    assert _turning_point_ratio(rr) == 0.0


def test_tpr_alternating_series_is_one() -> None:
    """Perfectly alternating (up-down-up-down) → every interior point
    is a turning point → TPR = 1.0."""
    rr = np.array([800.0, 400.0, 800.0, 400.0, 800.0, 400.0, 800.0])
    assert _turning_point_ratio(rr) == pytest.approx(1.0)


def test_tpr_random_series_near_two_thirds() -> None:
    """Large random series should have TPR ≈ 2/3."""
    rng = np.random.RandomState(42)
    rr = rng.uniform(400, 1200, 1000)
    tpr = _turning_point_ratio(rr)
    assert 0.60 < tpr < 0.73, f"TPR={tpr:.3f}, expected near 0.667"


def test_tpr_too_short_returns_zero() -> None:
    assert _turning_point_ratio(np.array([800.0, 900.0])) == 0.0


# ---------------------------------------------------------------------------
# Other feature functions (unchanged from previous version)
# ---------------------------------------------------------------------------


def test_cv_of_constant_series_is_zero() -> None:
    rr = np.array([800.0, 800.0, 800.0, 800.0])
    assert _coefficient_of_variation(rr) == pytest.approx(0.0)


def test_cv_positive_for_variable_series() -> None:
    rr = np.array([600.0, 1200.0, 600.0, 1200.0])
    assert _coefficient_of_variation(rr) > 0.2


def test_entropy_of_constant_deltas_is_zero() -> None:
    rr = np.array([800.0, 810.0, 820.0, 830.0, 840.0])
    assert _delta_rr_entropy(rr) == pytest.approx(0.0)


def test_entropy_of_uniform_deltas_is_high() -> None:
    rng = np.random.RandomState(42)
    rr = np.cumsum(rng.uniform(300, 1200, 100)) + 500.0
    entropy = _delta_rr_entropy(rr)
    assert entropy > 0.7


def test_entropy_single_interval_is_zero() -> None:
    assert _delta_rr_entropy(np.array([800.0])) == 0.0
