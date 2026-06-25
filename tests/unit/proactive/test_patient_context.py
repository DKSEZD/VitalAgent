from __future__ import annotations

from agent.proactive.patient_context import PatientContext


def _tracker_state(
    *,
    window_count: int = 30,
    resting_hr_baseline: float | None = 70.0,
    sdnn_ms: float | None = 55.0,
    rmssd_ms: float | None = 32.0,
    cold_start_complete: bool = False,
    distance_mean: float = 0.25,
    distance_std: float = 0.05,
) -> dict[str, object]:
    return {
        "medium": {
            "window_count": window_count,
            "sdnn_ms": sdnn_ms,
            "rmssd_ms": rmssd_ms,
        },
        "long": {
            "resting_hr_baseline": resting_hr_baseline,
        },
        "embedding": {
            "cold_start_complete": cold_start_complete,
            "distance_mean": distance_mean,
            "distance_std": distance_std,
        },
    }


def test_snapshot_baseline_numeric_not_ready_below_min_windows() -> None:
    ctx = PatientContext(patient_id="00012")

    ctx.snapshot_baseline_from_tracker(
        _tracker_state(window_count=29),
        embedding_enabled=False,
    )

    assert ctx.baseline.numeric_ready is False
    assert ctx.baseline.embedding_ready is True
    assert ctx.baseline.is_complete is False


def test_snapshot_baseline_complete_when_numeric_ready_and_embedding_disabled() -> None:
    ctx = PatientContext(patient_id="00012")

    ctx.snapshot_baseline_from_tracker(
        _tracker_state(window_count=30),
        embedding_enabled=False,
    )

    assert ctx.baseline.numeric_ready is True
    assert ctx.baseline.embedding_ready is True
    assert ctx.baseline.is_complete is True


def test_snapshot_baseline_incomplete_when_embedding_enabled_but_cold_start_pending() -> None:
    ctx = PatientContext(patient_id="00012")

    ctx.snapshot_baseline_from_tracker(
        _tracker_state(window_count=30, cold_start_complete=False),
        embedding_enabled=True,
    )

    assert ctx.baseline.numeric_ready is True
    assert ctx.baseline.embedding_ready is False
    assert ctx.baseline.is_complete is False


def test_snapshot_baseline_complete_when_embedding_enabled_and_cold_start_done() -> None:
    ctx = PatientContext(patient_id="00012")

    ctx.snapshot_baseline_from_tracker(
        _tracker_state(
            window_count=30,
            cold_start_complete=True,
            distance_mean=0.4,
            distance_std=0.08,
        ),
        embedding_enabled=True,
    )

    assert ctx.baseline.numeric_ready is True
    assert ctx.baseline.embedding_ready is True
    assert ctx.baseline.is_complete is True
    assert ctx.baseline.embedding_distance_mean == 0.4
    assert ctx.baseline.embedding_distance_std == 0.08


def test_snapshot_baseline_numeric_not_ready_when_required_metrics_missing() -> None:
    missing_cases = [
        _tracker_state(resting_hr_baseline=None),
        _tracker_state(sdnn_ms=None),
        _tracker_state(rmssd_ms=None),
    ]

    for state in missing_cases:
        ctx = PatientContext(patient_id="00012")
        ctx.snapshot_baseline_from_tracker(state, embedding_enabled=False)

        assert ctx.baseline.numeric_ready is False
        assert ctx.baseline.is_complete is False


def test_baseline_to_dict_exports_readiness_and_is_complete() -> None:
    ctx = PatientContext(patient_id="00012")
    ctx.snapshot_baseline_from_tracker(
        _tracker_state(window_count=30, cold_start_complete=True),
        embedding_enabled=True,
    )

    payload = ctx.baseline.to_dict()

    assert payload["numeric_ready"] is True
    assert payload["embedding_ready"] is True
    assert payload["is_complete"] is True


def test_record_alert_exports_recent_alert_summary() -> None:
    ctx = PatientContext(patient_id="00012", max_recent_alerts=2)

    first = ctx.record_alert(
        segment_id="03",
        window_index=10,
        offset_s=100.0,
        urgency="medium",
        triggered_rules=["sustained_tachycardia"],
        reason="first alert",
        matched_episode_id=0,
        matched_episode_label="AF",
        latency_seconds=40.0,
    )
    second = ctx.record_alert(
        segment_id="03",
        window_index=12,
        offset_s=120.0,
        urgency="medium",
        triggered_rules=["sustained_tachycardia"],
        reason="second alert",
        is_false_alert=True,
    )
    ctx.record_alert(
        segment_id="03",
        window_index=14,
        offset_s=140.0,
        urgency="high",
        triggered_rules=["experimental_short_af_episode"],
        reason="third alert",
    )

    payload = ctx.to_dict()

    assert first.alert_id == 0
    assert second.alert_id == 1
    assert payload["alert_summary"]["total_alerts"] == 2
    assert payload["alert_summary"]["by_rule"] == {
        "sustained_tachycardia": 1,
        "experimental_short_af_episode": 1,
    }
    assert payload["alert_summary"]["false_alert_count"] == 1
    assert payload["alert_summary"]["matched_alert_count"] == 0
    assert [alert["window_index"] for alert in payload["recent_alerts"]] == [12, 14]
