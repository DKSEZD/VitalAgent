from __future__ import annotations

from typing import Iterable

import numpy as np

from agent.data.streaming import ECGWindow
from agent.proactive.intervention_decision import InterventionResult, Urgency
from agent.proactive.pipeline import ProactivePipeline


def _make_window(window_index: int = 0, fs: int = 250) -> ECGWindow:
    signal = np.zeros(fs * 10, dtype=float)
    return ECGWindow(
        signal=signal,
        fs=fs,
        patient_id="00012",
        segment_id="03",
        window_index=window_index,
        start_sample=window_index * len(signal),
        end_sample=(window_index + 1) * len(signal),
        n_samples=len(signal),
        stream_offset_s=float(window_index * 10),
    )


class _DummyVitals:
    def __init__(
        self,
        hr_bpm: float = 72.0,
        rr_intervals_ms: list[float] | None = None,
        sdnn_ms: float = 60.0,
        rmssd_ms: float = 12.0,
    ) -> None:
        self._payload = {
            "hr_bpm": hr_bpm,
            "rr_intervals_ms": rr_intervals_ms or [820.0, 830.0, 840.0, 835.0, 825.0],
            "sdnn_ms": sdnn_ms,
            "rmssd_ms": rmssd_ms,
        }

    def to_dict(self) -> dict[str, float | list[float]]:
        return dict(self._payload)


def _install_vitals_stub(pipeline: ProactivePipeline, vitals: _DummyVitals | None = None) -> None:
    payload = vitals or _DummyVitals()
    pipeline.signal_processor.extract_vitals = lambda *args, **kwargs: payload  # type: ignore[method-assign]


def _install_decision_sequence(
    pipeline: ProactivePipeline,
    results: Iterable[InterventionResult],
) -> None:
    sequence = iter(results)
    pipeline.decision.rule_judge.evaluate = lambda state, embedding_zscore=0.0: next(sequence)  # type: ignore[method-assign]


def test_pipeline_process_window_passes_window_sampling_rate() -> None:
    pipeline = ProactivePipeline()
    captured: dict[str, int] = {}

    def fake_extract_vitals(signal, lead_index=0, fs=None):
        captured["fs"] = fs
        return _DummyVitals(sdnn_ms=10.0)

    pipeline.signal_processor.extract_vitals = fake_extract_vitals  # type: ignore[method-assign]
    _install_decision_sequence(
        pipeline,
        [
            InterventionResult(
                intervene=False,
                urgency=Urgency.NONE,
                reason="No intervention rules triggered.",
                triggered_rules=[],
            )
        ],
    )

    result = pipeline.process_window(_make_window(fs=500))

    assert captured["fs"] == 500
    assert result["intervene"] is False
    assert result["condition_active"] is False


def test_pipeline_import_and_process_window_shape() -> None:
    pipeline = ProactivePipeline()
    _install_vitals_stub(pipeline, _DummyVitals(hr_bpm=75.0, sdnn_ms=55.0, rmssd_ms=14.0))
    _install_decision_sequence(
        pipeline,
        [
            InterventionResult(
                intervene=True,
                urgency=Urgency.CRITICAL,
                reason="Critical bradycardia threshold crossed",
                triggered_rules=["extreme_bradycardia"],
            )
        ],
    )

    result = pipeline.process_window(_make_window())

    assert result["intervene"] is True
    assert result["condition_active"] is True
    assert result["urgency"] == "critical"
    assert result["segment_id"] == "03"
    assert result["modality"] == "ecg"
    assert result["rule_layer"]["triggered_rules"] == ["extreme_bradycardia"]
    assert result["judge_layer"]["intervene"] is False


def test_process_window_deduplicates_identical_active_alerts() -> None:
    pipeline = ProactivePipeline()
    _install_vitals_stub(pipeline)
    _install_decision_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.CRITICAL, "first", ["extreme_bradycardia"]),
            InterventionResult(True, Urgency.CRITICAL, "repeat", ["extreme_bradycardia"]),
        ],
    )

    first = pipeline.process_window(_make_window(0))
    second = pipeline.process_window(_make_window(1))

    assert first["intervene"] is True
    assert first["condition_active"] is True
    assert second["intervene"] is False
    assert second["condition_active"] is True
    assert pipeline.tracker.medium_window.anomaly_count == 1


def test_process_window_reemits_after_recovery_and_recurrence() -> None:
    pipeline = ProactivePipeline()
    _install_vitals_stub(pipeline)
    _install_decision_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.CRITICAL, "active", ["extreme_bradycardia"]),
            InterventionResult(False, Urgency.NONE, "", []),
            InterventionResult(True, Urgency.CRITICAL, "active-again", ["extreme_bradycardia"]),
        ],
    )

    first = pipeline.process_window(_make_window(0))
    recovered = pipeline.process_window(_make_window(1))
    recurrent = pipeline.process_window(_make_window(2))

    assert first["intervene"] is True
    assert recovered["intervene"] is False
    assert recovered["condition_active"] is False
    assert recurrent["intervene"] is True
    assert pipeline.tracker.medium_window.anomaly_count == 2


def test_process_window_reemits_when_triggered_rules_change() -> None:
    pipeline = ProactivePipeline()
    _install_vitals_stub(pipeline)
    _install_decision_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.CRITICAL, "rule-a", ["extreme_bradycardia"]),
            InterventionResult(True, Urgency.MEDIUM, "rule-b", ["sustained_tachycardia"]),
        ],
    )

    first = pipeline.process_window(_make_window(0))
    second = pipeline.process_window(_make_window(1))

    assert first["intervene"] is True
    assert second["intervene"] is True
    assert pipeline.tracker.medium_window.anomaly_count == 2


def test_process_window_reemits_on_urgency_escalation() -> None:
    pipeline = ProactivePipeline()
    _install_vitals_stub(pipeline)
    _install_decision_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.MEDIUM, "medium", ["sustained_tachycardia"]),
            InterventionResult(True, Urgency.CRITICAL, "critical", ["sustained_tachycardia"]),
        ],
    )

    first = pipeline.process_window(_make_window(0))
    second = pipeline.process_window(_make_window(1))

    assert first["intervene"] is True
    assert second["intervene"] is True
    assert second["urgency"] == "critical"


def test_run_on_windows_appends_hourly_and_final_snapshots() -> None:
    pipeline = ProactivePipeline()
    snapshot_calls: list[int] = []

    pipeline.process_window = lambda window: {  # type: ignore[method-assign]
        "intervene": False,
        "condition_active": False,
        "urgency": "none",
        "reason": "",
        "triggered_rules": [],
        "vitals": {},
        "embedding_zscore": 0.0,
        "window_index": window.window_index,
        "patient_id": window.patient_id,
        "segment_id": window.segment_id,
        "stream_offset_s": window.stream_offset_s,
    }
    pipeline.tracker.append_daily_snapshot = lambda: snapshot_calls.append(len(snapshot_calls) + 1)  # type: ignore[method-assign]

    results = pipeline.run_on_windows(_make_window(i) for i in range(361))

    assert len(results) == 361
    assert snapshot_calls == [1, 2]
