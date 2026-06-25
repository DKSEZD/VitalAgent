from __future__ import annotations

import numpy as np
import pytest

from agent.proactive.tracker import HealthStateTracker


def test_update_vitals_clears_stale_rhythm_when_rr_count_is_insufficient() -> None:
    tracker = HealthStateTracker()

    tracker.update_vitals(
        {
            "hr_bpm": 72.0,
            "rr_intervals_ms": [600.0, 1200.0, 620.0, 1180.0, 640.0, 1160.0],
            "sdnn_ms": 55.0,
            "rmssd_ms": 20.0,
        }
    )
    assert tracker.short_window.rhythm_regular is False

    tracker.update_vitals(
        {
            "hr_bpm": 71.0,
            "rr_intervals_ms": [810.0, 805.0, 815.0, 800.0],
            "sdnn_ms": 56.0,
            "rmssd_ms": 21.0,
        }
    )

    assert tracker.short_window.rhythm_regular is None
    assert tracker.short_window.latest_rr_intervals_ms == [810.0, 805.0, 815.0, 800.0]


def test_update_vitals_computes_medium_hrv_from_concatenated_rr_buffer() -> None:
    tracker = HealthStateTracker()

    first_rr = [700.0, 780.0, 820.0, 900.0]
    second_rr = [710.0, 790.0, 810.0, 890.0]

    tracker.update_vitals(
        {
            "hr_bpm": 70.0,
            "rr_intervals_ms": first_rr,
            "sdnn_ms": 1.0,
            "rmssd_ms": 1.0,
        }
    )
    tracker.update_vitals(
        {
            "hr_bpm": 71.0,
            "rr_intervals_ms": second_rr,
            "sdnn_ms": 2.0,
            "rmssd_ms": 2.0,
        }
    )

    expected_rr = np.asarray(first_rr + second_rr, dtype=float)
    expected_sdnn = round(float(np.std(expected_rr, ddof=1)), 2)
    expected_rmssd = round(float(np.sqrt(np.mean(np.diff(expected_rr) ** 2))), 2)

    assert tracker.medium_window.sdnn_ms == expected_sdnn
    assert tracker.medium_window.rmssd_ms == expected_rmssd


def test_update_vitals_skips_baseline_update_when_medium_hr_deviates_too_far() -> None:
    tracker = HealthStateTracker()

    tracker.update_vitals(
        {
            "hr_bpm": 70.0,
            "rr_intervals_ms": [850.0, 860.0, 845.0, 855.0],
        }
    )
    assert tracker.long_window.resting_hr_baseline == 70.0

    tracker.update_vitals(
        {
            "hr_bpm": 150.0,
            "rr_intervals_ms": [400.0, 405.0, 395.0, 410.0],
        }
    )

    assert tracker.medium_window.mean_hr == 110.0
    assert tracker.long_window.resting_hr_baseline == 70.0


def test_update_vitals_tracks_tachycardia_ratio() -> None:
    tracker = HealthStateTracker()

    state = tracker.get_current_state()
    assert state["short"]["tachycardia_ratio_5min"] is None
    assert state["short"]["tachycardia_sample_count"] == 0

    tracker.update_vitals({"hr_bpm": 70.0, "rr_intervals_ms": [860.0] * 10})
    state = tracker.get_current_state()
    assert state["short"]["tachycardia_ratio_5min"] == 0.0
    assert state["short"]["tachycardia_sample_count"] == 1

    for _ in range(30):
        tracker.update_vitals({"hr_bpm": 135.0, "rr_intervals_ms": [444.0] * 10})

    state = tracker.get_current_state()
    assert state["short"]["tachycardia_ratio_5min"] == 1.0
    assert state["short"]["tachycardia_sample_count"] == 30


def test_tachycardia_ratio_reflects_mixed_history() -> None:
    tracker = HealthStateTracker()

    for _ in range(10):
        tracker.update_vitals({"hr_bpm": 80.0, "rr_intervals_ms": [750.0] * 8})
    for _ in range(20):
        tracker.update_vitals({"hr_bpm": 110.0, "rr_intervals_ms": [545.0] * 12})

    state = tracker.get_current_state()
    assert state["short"]["tachycardia_sample_count"] == 30
    assert state["short"]["tachycardia_ratio_5min"] == pytest.approx(20 / 30)


def test_tachycardia_window_is_rolling() -> None:
    tracker = HealthStateTracker()

    for _ in range(30):
        tracker.update_vitals({"hr_bpm": 135.0, "rr_intervals_ms": [444.0] * 10})
    assert tracker.get_current_state()["short"]["tachycardia_ratio_5min"] == 1.0

    for _ in range(30):
        tracker.update_vitals({"hr_bpm": 70.0, "rr_intervals_ms": [860.0] * 10})

    state = tracker.get_current_state()
    assert state["short"]["tachycardia_ratio_5min"] == 0.0
    assert state["short"]["tachycardia_sample_count"] == 30
