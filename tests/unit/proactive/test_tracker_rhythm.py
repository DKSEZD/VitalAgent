"""Tests for rhythm_class and AF episode tracking in HealthStateTracker."""

from __future__ import annotations

from agent.proactive.tracker import HealthStateTracker


def _vitals(hr: float = 72.0, rr: list[float] | None = None):
    return {
        "hr_bpm": hr,
        "rr_intervals_ms": rr or [830.0] * 10,
        "sdnn_ms": 50.0,
        "rmssd_ms": 15.0,
    }


# ---------------------------------------------------------------------------
# rhythm_class pass-through
# ---------------------------------------------------------------------------


def test_rhythm_class_appears_in_state() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="AF")

    state = tracker.get_current_state()
    assert state["short"]["rhythm_class"] == "AF"


def test_rhythm_class_defaults_to_none() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals())

    state = tracker.get_current_state()
    assert state["short"]["rhythm_class"] is None


def test_rhythm_class_updates_each_window() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="N")
    assert tracker.short_window.rhythm_class == "N"

    tracker.update_vitals(_vitals(), rhythm_class="AF")
    assert tracker.short_window.rhythm_class == "AF"

    tracker.update_vitals(_vitals(), rhythm_class="Other")
    assert tracker.short_window.rhythm_class == "Other"


# ---------------------------------------------------------------------------
# rhythm_regular derived from rhythm_class
# ---------------------------------------------------------------------------


def test_rhythm_regular_true_when_class_is_n() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="N")
    assert tracker.short_window.rhythm_regular is True


def test_rhythm_regular_false_when_class_is_af() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="AF")
    assert tracker.short_window.rhythm_regular is False


def test_rhythm_regular_false_when_class_is_other() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="Other")
    assert tracker.short_window.rhythm_regular is False


def test_rhythm_regular_falls_back_to_cv_when_class_is_none() -> None:
    """When rhythm_class is not provided, the old CV heuristic should
    still set rhythm_regular."""
    tracker = HealthStateTracker()
    # Regular RR → CV < 0.15 → rhythm_regular=True
    tracker.update_vitals(
        _vitals(rr=[800.0, 802.0, 798.0, 801.0, 799.0, 800.0, 803.0, 797.0]),
        rhythm_class=None,
    )
    assert tracker.short_window.rhythm_regular is True


def test_rhythm_regular_none_when_class_unknown_and_few_rr() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(
        {"hr_bpm": 70.0, "rr_intervals_ms": [800.0, 810.0]},
        rhythm_class="Unknown",
    )
    # "Unknown" doesn't map to N/AF/Other, and only 2 RR → None
    assert tracker.short_window.rhythm_regular is None


# ---------------------------------------------------------------------------
# AF episode duration — strict mode
# ---------------------------------------------------------------------------


def test_single_af_window_duration_equals_one_window() -> None:
    """First AF window contributes exactly one window_duration_s."""
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] == 10.0


def test_consecutive_af_windows_accumulate_duration() -> None:
    tracker = HealthStateTracker()
    for _ in range(5):
        tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] == 50.0  # 5 * 10s


def test_non_af_window_resets_episode() -> None:
    """Confirmed non-AF (N / Other) resets the counter."""
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] == 20.0

    # Interrupt with a normal window
    tracker.update_vitals(_vitals(), rhythm_class="N", window_duration_s=10.0)
    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] is None

    # AF resumes — new episode, single-window duration
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)
    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] == 10.0


def test_unknown_window_preserves_af_episode() -> None:
    """A transient 'Unknown' window (e.g. low-quality PPG) must not reset
    the running AF episode — only confirmed N/Other resets it."""
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)
    tracker.update_vitals(_vitals(), rhythm_class="Unknown", window_duration_s=10.0)
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)

    state = tracker.get_current_state()
    # Unknown preserves the episode but does not add time.
    assert state["short"]["af_episode_duration_s"] == 30.0


def test_consecutive_unknown_windows_freeze_af_duration() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=10.0)

    for _ in range(3):
        tracker.update_vitals(
            {"hr_bpm": None, "rr_intervals_ms": []},
            rhythm_class="Unknown",
            window_duration_s=10.0,
        )

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] == 10.0


def test_af_episode_duration_none_when_not_in_af() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals(), rhythm_class="N")

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] is None


def test_af_episode_duration_none_when_rhythm_class_not_provided() -> None:
    tracker = HealthStateTracker()
    tracker.update_vitals(_vitals())

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] is None


def test_custom_window_duration_scales_episode() -> None:
    """Non-standard window length (e.g. 5s) should be respected."""
    tracker = HealthStateTracker()
    for _ in range(4):
        tracker.update_vitals(_vitals(), rhythm_class="AF", window_duration_s=5.0)

    state = tracker.get_current_state()
    assert state["short"]["af_episode_duration_s"] == 20.0  # 4 * 5s
