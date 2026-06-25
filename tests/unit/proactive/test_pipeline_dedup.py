"""
Tests for judge-invocation dedup in ProactivePipeline.

These tests verify that when the LLM judge is in LIVE mode, the
pipeline skips redundant judge calls across windows whose rule-layer
outcome has not changed. The tests also verify that:

  - STUB mode behaviour is unchanged (no caching, judge called every
    time it would have been called before).
  - The dedup cache is force-refreshed after a configured number of
    windows even if the rule outcome has not changed.
  - Changes to the rule outcome (triggered rules or urgency)
    invalidate the cache.
  - Raw state changes alone do not invalidate the cache when the
    rule outcome is unchanged.
  - ``judge_layer_cached`` is surfaced in the per-window result dict.

These tests DO NOT touch a real LLM. They install a fake judge whose
``enabled`` property and ``evaluate`` method are both mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from agent.data.streaming import ECGWindow
from agent.proactive.intervention_decision import (
    InterventionResult,
    Urgency,
    _JUDGE_FORCE_REFRESH_WINDOWS,
    _rule_outcome_signature,
)
from agent.proactive.llm_judge import JudgeDecision
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


class _FakeVitals:
    """Minimal VitalParams replacement that vitals.to_dict() can serialise."""

    def __init__(self, hr_bpm: float | None = 75.0) -> None:
        self._hr = hr_bpm

    def to_dict(self) -> dict[str, Any]:
        return {
            "hr_bpm": self._hr,
            "rr_intervals_ms": [800.0] * 6 if self._hr is not None else None,
            "sdnn_ms": 50.0,
            "rmssd_ms": 15.0,
        }


def _install_vitals(pipeline: ProactivePipeline, hr_bpm: float | None = 75.0) -> None:
    vitals = _FakeVitals(hr_bpm=hr_bpm)
    pipeline.signal_processor.extract_vitals = lambda *a, **kw: vitals  # type: ignore[method-assign]


def _install_rule_layer_noop(pipeline: ProactivePipeline) -> None:
    pipeline.decision.rule_judge.evaluate = lambda state, embedding_zscore=0.0: InterventionResult(  # type: ignore[method-assign]
        intervene=False,
        urgency=Urgency.NONE,
        reason="",
        triggered_rules=[],
    )


def _install_rule_layer_sequence(
    pipeline: ProactivePipeline,
    results: list[InterventionResult],
) -> None:
    sequence = iter(results)
    pipeline.decision.rule_judge.evaluate = lambda state, embedding_zscore=0.0: next(sequence)  # type: ignore[method-assign]


def _install_live_fake_judge(
    pipeline: ProactivePipeline,
    responses: list[JudgeDecision] | None = None,
) -> MagicMock:
    fake_judge = MagicMock()
    fake_judge.enabled = True

    if responses is None:
        responses = [JudgeDecision.no_alert()]
    fake_judge.evaluate.side_effect = lambda state, patient_context=None: (
        responses[min(fake_judge.evaluate.call_count - 1, len(responses) - 1)]
    )
    pipeline.decision.llm_judge = fake_judge
    return fake_judge


def _stub_state_on_tracker(
    pipeline: ProactivePipeline,
    hr_bpm: float | None,
    tachy_ratio: float | None = None,
    rhythm_class: str | None = None,
    rhythm_regular: bool | None = None,
) -> None:
    def _state() -> dict[str, Any]:
        return {
            "short": {
                "hr_bpm": hr_bpm,
                "rhythm_class": rhythm_class,
                "rhythm_regular": rhythm_regular,
                "tachycardia_ratio_5min": tachy_ratio,
                "tachycardia_sample_count": 30 if tachy_ratio is not None else 0,
            },
            "medium": {},
            "long": {},
            "embedding": {"latest_zscore": 0.0},
        }

    pipeline.tracker.get_current_state = _state  # type: ignore[method-assign]


def test_rule_outcome_signature_for_no_alert_is_empty_rule_set_and_none_urgency() -> None:
    result = InterventionResult(
        intervene=False,
        urgency=Urgency.NONE,
        reason="",
        triggered_rules=[],
    )

    assert _rule_outcome_signature(result) == (frozenset(), Urgency.NONE)


def test_rule_outcome_signature_ignores_trigger_order_but_keeps_urgency() -> None:
    result_a = InterventionResult(
        intervene=True,
        urgency=Urgency.CRITICAL,
        reason="",
        triggered_rules=["extreme_bradycardia", "sustained_tachycardia"],
    )
    result_b = InterventionResult(
        intervene=True,
        urgency=Urgency.CRITICAL,
        reason="",
        triggered_rules=["sustained_tachycardia", "extreme_bradycardia"],
    )
    result_c = InterventionResult(
        intervene=True,
        urgency=Urgency.HIGH,
        reason="",
        triggered_rules=["extreme_bradycardia", "sustained_tachycardia"],
    )

    assert _rule_outcome_signature(result_a) == _rule_outcome_signature(result_b)
    assert _rule_outcome_signature(result_a) != _rule_outcome_signature(result_c)


def test_stub_mode_always_calls_judge_and_never_sets_cached() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=135.0)
    _install_rule_layer_noop(pipeline)

    fake_judge = MagicMock()
    fake_judge.enabled = False
    fake_judge.evaluate.return_value = JudgeDecision.no_alert()
    pipeline.decision.llm_judge = fake_judge

    for i in range(5):
        result = pipeline.process_window(_make_window(i))
        assert result["judge_layer_cached"] is False

    assert fake_judge.evaluate.call_count == 5
    assert pipeline.decision._judge_cache is None
    assert pipeline.decision._last_judge_signature is None


def test_live_mode_caches_when_rule_outcome_is_unchanged() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=135.0)
    _install_rule_layer_noop(pipeline)
    _stub_state_on_tracker(
        pipeline,
        hr_bpm=135.0,
        tachy_ratio=0.6,
        rhythm_class="N",
        rhythm_regular=True,
    )
    fake_judge = _install_live_fake_judge(pipeline)

    r0 = pipeline.process_window(_make_window(0))
    r1 = pipeline.process_window(_make_window(1))
    r2 = pipeline.process_window(_make_window(2))

    assert fake_judge.evaluate.call_count == 1
    assert r0["judge_layer_cached"] is False
    assert r1["judge_layer_cached"] is True
    assert r2["judge_layer_cached"] is True


def test_live_mode_keeps_cache_when_raw_state_changes_but_rule_outcome_is_same() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=135.0)
    _install_rule_layer_noop(pipeline)
    fake_judge = _install_live_fake_judge(pipeline)

    _stub_state_on_tracker(
        pipeline,
        hr_bpm=135.0,
        tachy_ratio=0.6,
        rhythm_class="N",
        rhythm_regular=True,
    )
    pipeline.process_window(_make_window(0))

    _stub_state_on_tracker(
        pipeline,
        hr_bpm=152.0,
        tachy_ratio=0.2,
        rhythm_class="AF",
        rhythm_regular=False,
    )
    r1 = pipeline.process_window(_make_window(1))

    assert fake_judge.evaluate.call_count == 1
    assert r1["judge_layer_cached"] is True


def test_live_mode_reinvokes_when_triggered_rules_change() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=95.0)
    _install_rule_layer_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.CRITICAL, "first", ["extreme_bradycardia"]),
            InterventionResult(True, Urgency.MEDIUM, "second", ["sustained_tachycardia"]),
        ],
    )
    fake_judge = _install_live_fake_judge(pipeline)

    pipeline.process_window(_make_window(0))
    r1 = pipeline.process_window(_make_window(1))

    assert fake_judge.evaluate.call_count == 2
    assert r1["judge_layer_cached"] is False


def test_live_mode_reinvokes_when_urgency_changes_for_same_rule_set() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=105.0)
    _install_rule_layer_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.MEDIUM, "medium", ["sustained_tachycardia"]),
            InterventionResult(True, Urgency.CRITICAL, "critical", ["sustained_tachycardia"]),
        ],
    )
    fake_judge = _install_live_fake_judge(pipeline)

    pipeline.process_window(_make_window(0))
    r1 = pipeline.process_window(_make_window(1))

    assert fake_judge.evaluate.call_count == 2
    assert r1["judge_layer_cached"] is False


def test_live_mode_reinvokes_when_rule_layer_clears_alert() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=72.0)
    _install_rule_layer_sequence(
        pipeline,
        [
            InterventionResult(True, Urgency.CRITICAL, "active", ["extreme_bradycardia"]),
            InterventionResult(False, Urgency.NONE, "", []),
        ],
    )
    fake_judge = _install_live_fake_judge(pipeline)

    pipeline.process_window(_make_window(0))
    r1 = pipeline.process_window(_make_window(1))

    assert fake_judge.evaluate.call_count == 2
    assert r1["judge_layer_cached"] is False


def test_live_mode_force_refresh_after_configured_windows() -> None:
    pipeline = ProactivePipeline()
    _install_vitals(pipeline, hr_bpm=135.0)
    _install_rule_layer_noop(pipeline)
    _stub_state_on_tracker(
        pipeline,
        hr_bpm=135.0,
        tachy_ratio=0.6,
        rhythm_class="N",
        rhythm_regular=True,
    )
    fake_judge = _install_live_fake_judge(pipeline)

    total = _JUDGE_FORCE_REFRESH_WINDOWS + 2
    cached_flags = []
    for i in range(total):
        r = pipeline.process_window(_make_window(i))
        cached_flags.append(r["judge_layer_cached"])

    assert cached_flags[0] is False
    assert all(cached_flags[1:1 + _JUDGE_FORCE_REFRESH_WINDOWS])
    assert cached_flags[1 + _JUDGE_FORCE_REFRESH_WINDOWS] is False
    assert fake_judge.evaluate.call_count == 2
