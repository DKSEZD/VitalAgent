from __future__ import annotations

import pytest

from agent.proactive.aggregator import aggregate_monitoring_window


def _result(
    rhythm_class: str,
    *,
    hr_bpm: float | None = 72.0,
    duration_s: float = 10.0,
    offset_s: float = 0.0,
) -> dict:
    return {
        "rhythm_class": rhythm_class,
        "vitals": {"hr_bpm": hr_bpm},
        "window_duration_s": duration_s,
        "stream_offset_s": offset_s,
    }


def test_aggregate_all_normal_has_zero_af_burden_and_transitions() -> None:
    aggregates = aggregate_monitoring_window(
        [
            _result("N", offset_s=0.0),
            _result("N", offset_s=10.0),
            _result("N", offset_s=20.0),
        ]
    )

    assert aggregates.af_burden_ratio == 0.0
    assert aggregates.rhythm_transition_count_per_hour == 0.0
    assert aggregates.confirmed_af_episode_count == 0
    assert aggregates.confirmed_af_duration_s == 0.0
    assert aggregates.confirmed_af_burden_ratio == 0.0
    assert aggregates.debounced_rhythm_transition_count_per_hour == 0.0


def test_aggregate_partial_af_burden_uses_window_durations() -> None:
    aggregates = aggregate_monitoring_window(
        [
            _result("N", duration_s=10.0),
            _result("AF", duration_s=20.0),
            _result("AF", duration_s=10.0),
        ]
    )

    assert aggregates.af_burden_ratio == pytest.approx(0.75)
    assert aggregates.confirmed_af_episode_count == 0
    assert aggregates.confirmed_af_duration_s == 0.0


def test_aggregate_max_hr_ignores_missing_values() -> None:
    aggregates = aggregate_monitoring_window(
        [
            _result("N", hr_bpm=None),
            _result("N", hr_bpm=88.0),
            _result("N", hr_bpm=81.0),
        ]
    )

    assert aggregates.max_hr_bpm == 88.0


def test_aggregate_empty_stream_returns_none_fields() -> None:
    aggregates = aggregate_monitoring_window([])

    assert aggregates.af_burden_ratio is None
    assert aggregates.max_hr_bpm is None
    assert aggregates.rhythm_transition_count_per_hour is None
    assert aggregates.confirmed_af_duration_s is None
    assert aggregates.confirmed_af_burden_ratio is None


def test_aggregate_confirms_af_after_three_consecutive_windows() -> None:
    aggregates = aggregate_monitoring_window(
        [
            _result("AF", duration_s=10.0),
            _result("AF", duration_s=10.0),
            _result("AF", duration_s=10.0),
            _result("N", duration_s=10.0),
        ]
    )

    assert aggregates.af_burden_ratio == pytest.approx(0.75)
    assert aggregates.confirmed_af_episode_count == 1
    assert aggregates.confirmed_af_duration_s == pytest.approx(30.0)
    assert aggregates.confirmed_af_burden_ratio == pytest.approx(0.75)


def test_aggregate_short_af_spike_is_not_confirmed() -> None:
    aggregates = aggregate_monitoring_window(
        [
            _result("N", offset_s=0.0),
            _result("AF", offset_s=10.0),
            _result("N", offset_s=20.0),
            _result("N", offset_s=30.0),
        ]
    )

    assert aggregates.af_burden_ratio == pytest.approx(0.25)
    assert aggregates.confirmed_af_episode_count == 0
    assert aggregates.confirmed_af_duration_s == 0.0
    assert aggregates.debounced_rhythm_transition_count_per_hour == 0.0


def test_aggregate_debounced_transitions_require_stable_new_label() -> None:
    aggregates = aggregate_monitoring_window(
        [
            _result("N", offset_s=0.0),
            _result("AF", offset_s=10.0),
            _result("AF", offset_s=20.0),
            _result("AF", offset_s=30.0),
            _result("N", offset_s=40.0),
            _result("N", offset_s=50.0),
            _result("N", offset_s=60.0),
        ]
    )

    assert aggregates.rhythm_transition_count_per_hour == pytest.approx(
        2 / (70.0 / 3600.0)
    )
    assert aggregates.debounced_rhythm_transition_count_per_hour == pytest.approx(
        2 / (70.0 / 3600.0)
    )
