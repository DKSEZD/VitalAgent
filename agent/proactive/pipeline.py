"""
Proactive Pipeline — orchestrates the proactive monitoring path.

Flow (Phase 1):

    ECG window
        |
        v
    Signal Processor -> vital params
        |
        v
    Health State Tracker (numeric + optional embedding channel)
        |
        v
    InterventionDecision
        +--> RuleBasedJudge               (guideline-grounded rules)
        +--> ProactiveLLMJudge            (contextual LLM reasoning, dedup'd)
        v
    Final intervention result

The pipeline owns signal processing and health-state tracking. The
high-level ``InterventionDecision`` owns the judge layers and final
decision merge.

Phase 2: wire in real ECG encoder embeddings; enable the LLM judge in
live mode (currently a stub that returns no-alert).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from agent.encoder import ECGSignalProcessor
from agent.proactive.intervention_decision import (
    InterventionDecision,
    InterventionResult,
    Modality,
    Urgency,
)
from agent.proactive.llm_judge import JudgeDecision, ProactiveLLMJudge
from agent.proactive.tracker import HealthStateTracker
from agent.proactive.rhythm_classifier import classify_rhythm

if TYPE_CHECKING:
    from agent.data.streaming import ECGWindow

logger = logging.getLogger(__name__)

_TREND_SNAPSHOT_INTERVAL_WINDOWS = 360


@dataclass(frozen=True)
class _ActiveAlertState:
    triggered_rules: frozenset[str]
    urgency: Urgency


class ProactivePipeline:
    """End-to-end proactive monitoring pipeline."""

    def __init__(
        self,
        sampling_rate: int = 250,
        enable_embedding: bool = False,
        embedding_dim: int = 768,
        modality: Modality = Modality.ECG,
        decision: InterventionDecision | None = None,
        llm_judge: ProactiveLLMJudge | None = None,
        signal_processor: object | None = None,
    ) -> None:
        self.signal_processor = (
            signal_processor
            if signal_processor is not None
            else ECGSignalProcessor(sampling_rate=sampling_rate)
        )
        self.tracker = HealthStateTracker()
        self.modality = modality
        self.decision = decision if decision is not None else InterventionDecision(
            modality=modality,
            llm_judge=llm_judge,
        )
        self.enable_embedding = enable_embedding
        self.embedding_dim = embedding_dim

        # Alert-emit dedup (pre-existing).
        self._last_active_alert: _ActiveAlertState | None = None
        self._last_emitted_alert: _ActiveAlertState | None = None

        logger.info(
            "ProactivePipeline initialised (modality=%s, sampling_rate=%d, "
            "embedding=%s, rules=%d, llm_judge=%s).",
            self.modality.value,
            sampling_rate,
            enable_embedding,
            len(self.decision.rules),
            "active" if self.decision.llm_judge.enabled else "stub",
        )

    # ------------------------------------------------------------------
    # Per-window processing
    # ------------------------------------------------------------------

    def process_window(self, window: Any) -> dict[str, Any]:
        """Process one window through both decision layers."""
        # 1. Signal processing -> vital params
        #    PPGWindow carries accel for motion-quality gating; ECGWindow does not.
        extra = {}
        accel = getattr(window, "accel", None)
        if accel is not None:
            extra["accel"] = accel
        vitals = self.signal_processor.extract_vitals(
            window.signal, lead_index=0, fs=window.fs, **extra
        )
        vitals_dict = vitals.to_dict()

        # 1b. Rhythm classification from RR intervals
        rr_intervals = vitals_dict.get("rr_intervals_ms")
        rhythm_class = classify_rhythm(rr_intervals)

        # 2. Tracker update (numeric channel + rhythm)
        window_duration_s = float(window.n_samples) / float(window.fs or 1)
        self.tracker.update_vitals(
            vitals_dict,
            rhythm_class=rhythm_class,
            window_duration_s=window_duration_s,
        )

        # 3. Embedding channel (stub in Phase 1)
        if self.enable_embedding:
            embedding = self._encode_embedding(window.signal)
            self.tracker.update_embedding(embedding)

        state = self.tracker.get_current_state()

        # Expose window-level metadata that triggers may need
        # (e.g. how long a sustain-window rule should consider the reading).
        state["window_meta"] = {
            "window_duration_s": float(window.n_samples) / float(window.fs or 1),
        }

        # 4. High-level intervention decision: run rule-based and LLM
        #    judges, then merge their outputs.
        decision_outcome = self.decision.evaluate(state)
        merged = decision_outcome.final_result

        condition_active = merged.intervene
        emit_intervention = self._should_emit_intervention(merged)

        # 6. Bookkeeping: count distinct emitted events for any future
        #    long-window rules.
        if emit_intervention:
            self.tracker.increment_anomaly_count()

        return {
            "intervene": emit_intervention,
            "condition_active": condition_active,
            "urgency": merged.urgency.value,
            "reason": merged.reason,
            "triggered_rules": merged.triggered_rules or [],
            "vitals": vitals_dict,
            "embedding_zscore": state["embedding"]["latest_zscore"],
            "window_index": window.window_index,
            "patient_id": window.patient_id,
            "segment_id": window.segment_id,
            "stream_offset_s": window.stream_offset_s,
            "modality": self.modality.value,
            "rhythm_class": rhythm_class,
            # New in Phase 1: separate rule vs judge breakdown for
            # downstream evaluation and paper ablation.
            "rule_layer": _result_to_dict(decision_outcome.rule_result),
            "judge_layer": _judge_to_dict(decision_outcome.judge_decision),
            "judge_layer_cached": decision_outcome.judge_cached,
        }

    # Keep backward compat with skeleton's method name
    def process_segment(self, window: ECGWindow) -> dict[str, Any] | None:
        result = self.process_window(window)
        if result["intervene"]:
            return result
        return None

    # ------------------------------------------------------------------
    # Stream-level processing
    # ------------------------------------------------------------------

    def run_on_windows(
        self,
        windows: Iterator[ECGWindow],
        max_windows: int | None = None,
        log_every: int = 100,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        interventions = 0
        windows_since_snapshot = 0

        for i, window in enumerate(windows):
            if max_windows is not None and i >= max_windows:
                break

            result = self.process_window(window)
            results.append(result)
            windows_since_snapshot += 1

            if windows_since_snapshot >= _TREND_SNAPSHOT_INTERVAL_WINDOWS:
                self.tracker.append_daily_snapshot()
                windows_since_snapshot = 0

            if result["intervene"]:
                interventions += 1
                logger.info(
                    "[Window %d] INTERVENTION: %s (urgency=%s)",
                    window.window_index,
                    result["reason"],
                    result["urgency"],
                )

            if (i + 1) % log_every == 0:
                logger.info(
                    "Processed %d windows (%d interventions so far).",
                    i + 1,
                    interventions,
                )

        if windows_since_snapshot > 0:
            self.tracker.append_daily_snapshot()

        logger.info(
            "Pipeline complete: %d windows, %d interventions (%.1f%%).",
            len(results),
            interventions,
            100 * interventions / max(len(results), 1),
        )
        return results

    # ------------------------------------------------------------------
    # Alert-emit deduplication (unchanged behaviour from previous version)
    # ------------------------------------------------------------------

    def _should_emit_intervention(self, result: InterventionResult) -> bool:
        """
        Emit an intervention only when the alert state changes.

        An alert state is a (triggered_rules, urgency) tuple. When the
        condition deactivates we reset; when it re-activates at equal or
        higher urgency we emit again. Repeating the identical alert is
        suppressed to avoid alert fatigue.
        """
        if not result.intervene:
            if self._last_active_alert is not None:
                logger.debug("Condition resolved; resetting active-alert state.")
            self._last_active_alert = None
            self._last_emitted_alert = None
            return False

        current = _ActiveAlertState(
            triggered_rules=frozenset(result.triggered_rules or []),
            urgency=result.urgency,
        )

        # First time active in this episode, always emit.
        if self._last_active_alert is None:
            self._last_active_alert = current
            self._last_emitted_alert = current
            return True

        # Unchanged alert, suppress.
        if current == self._last_emitted_alert:
            self._last_active_alert = current
            return False

        # Changed rule set or escalated urgency, re-emit.
        self._last_active_alert = current
        self._last_emitted_alert = current
        return True

    # ------------------------------------------------------------------
    # Embedding stub
    # ------------------------------------------------------------------

    def _encode_embedding(self, signal: np.ndarray) -> np.ndarray:
        del signal
        return np.random.randn(self.embedding_dim).astype(np.float32)


def _result_to_dict(result: InterventionResult) -> dict[str, Any]:
    return {
        "intervene": result.intervene,
        "urgency": result.urgency.value,
        "reason": result.reason,
        "triggered_rules": result.triggered_rules or [],
    }


def _judge_to_dict(decision: JudgeDecision) -> dict[str, Any]:
    return {
        "intervene": decision.intervene,
        "urgency": decision.urgency.value,
        "reason": decision.reason,
        "advice": decision.advice,
        "cited_sections": [
            {"guideline_id": c.guideline_id, "section_id": c.section_id}
            for c in decision.cited_sections
        ],
    }


