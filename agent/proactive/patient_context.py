"""
Patient Context — lightweight in-memory event layer for the Proactive Engine.

Positioned as an **eval-first record keeper**: its primary job is to
produce stable episode / trend / baseline records for offline evaluation.
InterventionDecision may optionally query it, but does not depend on it
for core rule evaluation.

Four components:

1. **Episode Log**: records detected anomaly episodes with temporal span.
2. **Baseline Profile**: serialisable personal baseline with save/load.
3. **Trend Buffer**: timestamped periodic snapshots for temporal comparison.
4. **Alert Memory**: recent emitted interventions for follow-up QA.

Design principles:
    - All temporal info (offsets, durations) comes from explicit caller
      input. PatientContext never infers timing from constants.
    - Episode state is a clean two-value enum (ACTIVE / RESOLVED)
      with a separate ``was_escalated`` flag for history tracking.
    - Pure in-memory for evaluation; JSON serialisation via to_dict/from_dict.
    - Pipeline owns one PatientContext per patient run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_NUMERIC_BASELINE_MIN_WINDOWS = 30


# =====================================================================
# 1. Episode Log
# =====================================================================

class EpisodeStatus(str, Enum):
    """Two-state lifecycle of a detected anomaly episode."""
    ACTIVE = "active"
    RESOLVED = "resolved"


@dataclass
class AnomalyEpisode:
    """
    A contiguous anomaly episode detected by the Proactive Engine.

    An episode starts when intervention is first triggered and ends when
    the condition resolves (intervene=False). If the same condition
    re-triggers after recovery, it becomes a new episode.

    State modelling:
        - ``status``: ACTIVE while the condition persists, RESOLVED once closed.
        - ``was_escalated``: True if urgency increased at any point during
          the episode. Orthogonal to status — a resolved episode can have
          was_escalated=True.
    """
    episode_id: int = 0
    episode_type: str = ""
    urgency_initial: str = "none"
    urgency_max: str = "none"
    was_escalated: bool = False
    start_window: int = 0
    end_window: int | None = None
    start_offset_s: float = 0.0
    end_offset_s: float | None = None
    triggered_rules: list[str] = field(default_factory=list)
    vitals_at_start: dict[str, Any] = field(default_factory=dict)
    vitals_at_peak: dict[str, Any] = field(default_factory=dict)
    status: EpisodeStatus = EpisodeStatus.ACTIVE

    @property
    def duration_windows(self) -> int | None:
        if self.end_window is None:
            return None
        return self.end_window - self.start_window + 1

    @property
    def duration_s(self) -> float | None:
        if self.end_offset_s is None:
            return None
        return self.end_offset_s - self.start_offset_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_type": self.episode_type,
            "urgency_initial": self.urgency_initial,
            "urgency_max": self.urgency_max,
            "was_escalated": self.was_escalated,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "start_offset_s": self.start_offset_s,
            "end_offset_s": self.end_offset_s,
            "triggered_rules": self.triggered_rules,
            "vitals_at_start": self.vitals_at_start,
            "vitals_at_peak": self.vitals_at_peak,
            "status": self.status.value,
        }


_URGENCY_ORDER: dict[str, int] = {
    "none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}


class EpisodeLog:
    """
    Chronological log of anomaly episodes.

    All timing is supplied by the caller — the log never infers offsets.
    """

    def __init__(self) -> None:
        self._episodes: list[AnomalyEpisode] = []
        self._next_id: int = 0
        self._active_episode: AnomalyEpisode | None = None

    @property
    def episodes(self) -> list[AnomalyEpisode]:
        return list(self._episodes)

    @property
    def active_episode(self) -> AnomalyEpisode | None:
        return self._active_episode

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    def open_episode(
        self,
        window_index: int,
        stream_offset_s: float,
        primary_rule: str,
        urgency: str,
        triggered_rules: list[str],
        vitals: dict[str, Any],
    ) -> AnomalyEpisode:
        """
        Start a new anomaly episode.

        If there is already an active episode, the caller must close it
        first. This method raises ValueError to prevent silent data loss.
        """
        if self._active_episode is not None:
            raise ValueError(
                f"Cannot open episode: episode #{self._active_episode.episode_id} "
                f"is still active. Close it first."
            )

        episode = AnomalyEpisode(
            episode_id=self._next_id,
            episode_type=primary_rule,
            urgency_initial=urgency,
            urgency_max=urgency,
            was_escalated=False,
            start_window=window_index,
            start_offset_s=stream_offset_s,
            triggered_rules=list(triggered_rules),
            vitals_at_start=dict(vitals),
            vitals_at_peak=dict(vitals),
            status=EpisodeStatus.ACTIVE,
        )
        self._next_id += 1
        self._episodes.append(episode)
        self._active_episode = episode

        logger.debug(
            "Episode #%d opened: type=%s urgency=%s @window=%d",
            episode.episode_id, primary_rule, urgency, window_index,
        )
        return episode

    def update_active_episode(
        self,
        urgency: str,
        triggered_rules: list[str],
        vitals: dict[str, Any],
    ) -> None:
        """
        Update the active episode with new window data.

        Tracks urgency escalation and accumulates triggered rules.
        """
        ep = self._active_episode
        if ep is None:
            return

        for rule in triggered_rules:
            if rule not in ep.triggered_rules:
                ep.triggered_rules.append(rule)

        if _URGENCY_ORDER.get(urgency, 0) > _URGENCY_ORDER.get(ep.urgency_max, 0):
            ep.urgency_max = urgency
            ep.vitals_at_peak = dict(vitals)
            ep.was_escalated = True

    def close_episode(
        self,
        window_index: int,
        stream_offset_s: float,
    ) -> AnomalyEpisode | None:
        """
        Close the active episode (condition resolved).

        Status is unconditionally set to RESOLVED. Escalation history
        is preserved in ``was_escalated``.
        """
        ep = self._active_episode
        if ep is None:
            return None

        ep.end_window = window_index
        ep.end_offset_s = stream_offset_s
        ep.status = EpisodeStatus.RESOLVED
        self._active_episode = None

        logger.debug(
            "Episode #%d resolved: duration=%s windows, escalated=%s",
            ep.episode_id, ep.duration_windows, ep.was_escalated,
        )
        return ep

    # ------------------------------------------------------------------
    # Queries (for eval and optional decision support)
    # ------------------------------------------------------------------

    def count_recent_episodes(
        self,
        episode_type: str | None = None,
        within_last_n_windows: int | None = None,
        current_window: int = 0,
    ) -> int:
        """Count episodes matching criteria."""
        count = 0
        for ep in self._episodes:
            if episode_type is not None and ep.episode_type != episode_type:
                continue
            if within_last_n_windows is not None:
                if (current_window - ep.start_window) > within_last_n_windows:
                    continue
            count += 1
        return count

    def get_last_episode(
        self,
        episode_type: str | None = None,
    ) -> AnomalyEpisode | None:
        for ep in reversed(self._episodes):
            if episode_type is None or ep.episode_type == episode_type:
                return ep
        return None

    def get_episode_summary(self) -> dict[str, Any]:
        """Summary stats for eval output."""
        from collections import Counter
        type_counts = Counter(ep.episode_type for ep in self._episodes)
        durations = [ep.duration_s for ep in self._episodes if ep.duration_s is not None]
        escalated_count = sum(1 for ep in self._episodes if ep.was_escalated)
        return {
            "total_episodes": len(self._episodes),
            "by_type": dict(type_counts),
            "escalated_count": escalated_count,
            "active": self._active_episode is not None,
            "mean_duration_s": sum(durations) / len(durations) if durations else None,
            "max_urgency_seen": max(
                (ep.urgency_max for ep in self._episodes),
                key=lambda u: _URGENCY_ORDER.get(u, 0),
                default="none",
            ),
        }


# =====================================================================
# 2. Baseline Profile
# =====================================================================

@dataclass
class BaselineProfile:
    """
    Serialisable personal baseline built during the cold-start period.

    Can be saved/loaded for longitudinal monitoring across sessions.
    """
    patient_id: str = ""
    resting_hr_bpm: float | None = None
    sdnn_baseline_ms: float | None = None
    rmssd_baseline_ms: float | None = None
    rhythm_distribution: dict[str, float] = field(default_factory=dict)
    embedding_mean: list[float] | None = None
    embedding_distance_mean: float = 0.0
    embedding_distance_std: float = 1.0
    baseline_windows: int = 0
    numeric_ready: bool = False
    embedding_ready: bool = False

    @property
    def is_complete(self) -> bool:
        """Baseline is complete only when all enabled channels are ready."""
        return self.numeric_ready and self.embedding_ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "resting_hr_bpm": self.resting_hr_bpm,
            "sdnn_baseline_ms": self.sdnn_baseline_ms,
            "rmssd_baseline_ms": self.rmssd_baseline_ms,
            "rhythm_distribution": self.rhythm_distribution,
            "embedding_mean": self.embedding_mean,
            "embedding_distance_mean": self.embedding_distance_mean,
            "embedding_distance_std": self.embedding_distance_std,
            "baseline_windows": self.baseline_windows,
            "numeric_ready": self.numeric_ready,
            "embedding_ready": self.embedding_ready,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Baseline profile saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> BaselineProfile:
        with open(path) as f:
            data = json.load(f)
        logger.info("Baseline profile loaded from %s", path)
        return cls.from_dict(data)


# =====================================================================
# 3. Trend Buffer
# =====================================================================

@dataclass
class TrendSnapshot:
    """
    A single timestamped snapshot of aggregated vitals.

    All timing comes from the caller; no internal inference.
    """
    window_range_start: int = 0
    window_range_end: int = 0
    offset_s_start: float = 0.0
    offset_s_end: float = 0.0
    mean_hr_bpm: float | None = None
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    anomaly_count: int = 0
    rhythm_regular_pct: float | None = None
    signal_quality_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TrendBuffer:
    """
    Timestamped trend data supporting temporal comparisons.

    Stores pre-aggregated snapshots with explicit time ranges,
    enabling queries like "last 4 hours vs previous 4 hours".
    """

    def __init__(self, max_snapshots: int = 200) -> None:
        self._snapshots: list[TrendSnapshot] = []
        self._max_snapshots = max_snapshots

    @property
    def snapshots(self) -> list[TrendSnapshot]:
        return list(self._snapshots)

    def record_snapshot(self, snapshot: TrendSnapshot) -> None:
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

    def get_recent(self, n: int = 4) -> list[TrendSnapshot]:
        return self._snapshots[-n:]

    def compare_periods(
        self,
        recent_n: int = 4,
        previous_n: int = 4,
    ) -> dict[str, Any] | None:
        """
        Compare the most recent N snapshots against the N before them.

        Returns None if insufficient data.
        """
        total_needed = recent_n + previous_n
        if len(self._snapshots) < total_needed:
            return None

        recent = self._snapshots[-recent_n:]
        previous = self._snapshots[-total_needed:-recent_n]

        def _safe_mean(lst: list[TrendSnapshot], attr: str) -> float | None:
            vals = [getattr(s, attr) for s in lst if getattr(s, attr) is not None]
            return sum(vals) / len(vals) if vals else None

        recent_hr = _safe_mean(recent, "mean_hr_bpm")
        previous_hr = _safe_mean(previous, "mean_hr_bpm")
        recent_sdnn = _safe_mean(recent, "sdnn_ms")
        previous_sdnn = _safe_mean(previous, "sdnn_ms")
        recent_anomaly = sum(s.anomaly_count for s in recent)
        previous_anomaly = sum(s.anomaly_count for s in previous)

        return {
            "hr_delta": (recent_hr - previous_hr) if recent_hr is not None and previous_hr is not None else None,
            "sdnn_delta": (recent_sdnn - previous_sdnn) if recent_sdnn is not None and previous_sdnn is not None else None,
            "anomaly_delta": recent_anomaly - previous_anomaly,
            "recent_hr_mean": recent_hr,
            "previous_hr_mean": previous_hr,
            "recent_sdnn_mean": recent_sdnn,
            "previous_sdnn_mean": previous_sdnn,
        }

    def get_trend_summary(self) -> dict[str, Any]:
        if not self._snapshots:
            return {"snapshot_count": 0}

        hrs = [s.mean_hr_bpm for s in self._snapshots if s.mean_hr_bpm is not None]
        sdnns = [s.sdnn_ms for s in self._snapshots if s.sdnn_ms is not None]
        return {
            "snapshot_count": len(self._snapshots),
            "hr_range": [min(hrs), max(hrs)] if hrs else None,
            "sdnn_range": [min(sdnns), max(sdnns)] if sdnns else None,
            "total_anomalies": sum(s.anomaly_count for s in self._snapshots),
            "comparison_4h": self.compare_periods(4, 4),
        }


# =====================================================================
# 4. Alert Memory
# =====================================================================

@dataclass
class AlertRecord:
    """
    Structured summary of one emitted proactive intervention.

    Stores reason, rules, timing, and optional replay-alignment metadata,
    but never raw signal.
    """
    alert_id: int = 0
    patient_id: str = ""
    segment_id: str = ""
    window_index: int = 0
    offset_s: float = 0.0
    urgency: str = "none"
    triggered_rules: list[str] = field(default_factory=list)
    reason: str = ""
    condition_active: bool = False
    rule_layer: dict[str, Any] = field(default_factory=dict)
    judge_layer: dict[str, Any] = field(default_factory=dict)
    matched_episode_id: int | None = None
    matched_episode_label: str | None = None
    matched_episode_start_offset_s: float | None = None
    matched_episode_duration_s: float | None = None
    latency_seconds: float | None = None
    is_false_alert: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "segment_id": self.segment_id,
            "window_index": self.window_index,
            "offset_s": self.offset_s,
            "urgency": self.urgency,
            "triggered_rules": list(self.triggered_rules),
            "reason": self.reason,
            "condition_active": self.condition_active,
            "rule_layer": self.rule_layer,
            "judge_layer": self.judge_layer,
            "matched_episode_id": self.matched_episode_id,
            "matched_episode_label": self.matched_episode_label,
            "matched_episode_start_offset_s": self.matched_episode_start_offset_s,
            "matched_episode_duration_s": self.matched_episode_duration_s,
            "latency_seconds": self.latency_seconds,
            "is_false_alert": self.is_false_alert,
        }


# =====================================================================
# 5. PatientContext — unified interface
# =====================================================================

class PatientContext:
    """
    Eval-first event layer for a single patient monitoring session.

    Primary job: produce stable episode / trend / baseline records
    for offline evaluation. InterventionDecision may optionally query
    it but does not depend on it for core rules.

    All temporal values are supplied by the pipeline — PatientContext
    never infers timing from hardcoded constants.
    """

    def __init__(self, patient_id: str = "", max_recent_alerts: int = 50) -> None:
        self.patient_id = patient_id
        self.episode_log = EpisodeLog()
        self.baseline = BaselineProfile(patient_id=patient_id)
        self.trend_buffer = TrendBuffer()
        self._alerts: list[AlertRecord] = []
        self._next_alert_id = 0
        self._max_recent_alerts = max(1, int(max_recent_alerts))

        logger.info("PatientContext created for patient=%s", patient_id)

    # ------------------------------------------------------------------
    # Alert memory
    # ------------------------------------------------------------------

    def record_alert(
        self,
        *,
        window_index: int,
        offset_s: float,
        urgency: str,
        patient_id: str | None = None,
        segment_id: str = "",
        triggered_rules: list[str] | tuple[str, ...] | None = None,
        reason: str = "",
        condition_active: bool = False,
        rule_layer: dict[str, Any] | None = None,
        judge_layer: dict[str, Any] | None = None,
        matched_episode_id: int | None = None,
        matched_episode_label: str | None = None,
        matched_episode_start_offset_s: float | None = None,
        matched_episode_duration_s: float | None = None,
        latency_seconds: float | None = None,
        is_false_alert: bool = False,
    ) -> AlertRecord:
        """Store one emitted intervention as follow-up QA context."""
        alert = AlertRecord(
            alert_id=self._next_alert_id,
            patient_id=patient_id or self.patient_id,
            segment_id=segment_id,
            window_index=window_index,
            offset_s=offset_s,
            urgency=urgency,
            triggered_rules=list(triggered_rules or []),
            reason=reason,
            condition_active=condition_active,
            rule_layer=dict(rule_layer or {}),
            judge_layer=dict(judge_layer or {}),
            matched_episode_id=matched_episode_id,
            matched_episode_label=matched_episode_label,
            matched_episode_start_offset_s=matched_episode_start_offset_s,
            matched_episode_duration_s=matched_episode_duration_s,
            latency_seconds=latency_seconds,
            is_false_alert=is_false_alert,
        )
        self._next_alert_id += 1
        self._alerts.append(alert)
        if len(self._alerts) > self._max_recent_alerts:
            self._alerts = self._alerts[-self._max_recent_alerts:]
        return alert

    def get_recent_alerts(self, n: int = 5) -> list[AlertRecord]:
        """Return recent alerts, newest last."""
        return self._alerts[-max(0, int(n)):]

    def get_last_alert(self) -> AlertRecord | None:
        return self._alerts[-1] if self._alerts else None

    def get_alert_summary(self) -> dict[str, Any]:
        from collections import Counter

        rule_counts: Counter[str] = Counter()
        for alert in self._alerts:
            for rule in alert.triggered_rules or ["(none)"]:
                rule_counts[rule] += 1

        return {
            "total_alerts": len(self._alerts),
            "false_alert_count": sum(1 for alert in self._alerts if alert.is_false_alert),
            "matched_alert_count": sum(
                1 for alert in self._alerts
                if alert.matched_episode_id is not None
            ),
            "by_rule": dict(rule_counts),
            "last_alert": (
                self._alerts[-1].to_dict()
                if self._alerts else None
            ),
        }

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def snapshot_baseline_from_tracker(
        self,
        tracker_state: dict[str, Any],
        *,
        embedding_enabled: bool,
        numeric_min_windows: int = _DEFAULT_NUMERIC_BASELINE_MIN_WINDOWS,
    ) -> None:
        """
        Capture baseline profile from HealthStateTracker's current state.

        Typically called once when cold-start period completes.
        """
        medium = tracker_state.get("medium", {})
        long_ = tracker_state.get("long", {})
        emb = tracker_state.get("embedding", {})

        self.baseline.resting_hr_bpm = long_.get("resting_hr_baseline")
        self.baseline.sdnn_baseline_ms = medium.get("sdnn_ms")
        self.baseline.rmssd_baseline_ms = medium.get("rmssd_ms")
        self.baseline.baseline_windows = medium.get("window_count", 0)
        self.baseline.numeric_ready = (
            self.baseline.resting_hr_bpm is not None
            and self.baseline.sdnn_baseline_ms is not None
            and self.baseline.rmssd_baseline_ms is not None
            and self.baseline.baseline_windows >= numeric_min_windows
        )

        if emb.get("cold_start_complete"):
            self.baseline.embedding_distance_mean = emb.get("distance_mean", 0.0)
            self.baseline.embedding_distance_std = emb.get("distance_std", 1.0)
        self.baseline.embedding_ready = (
            True if not embedding_enabled else bool(emb.get("cold_start_complete"))
        )

        logger.info(
            "Baseline snapshot: resting_hr=%.1f, sdnn=%.1f, windows=%d, "
            "numeric_ready=%s, embedding_ready=%s, complete=%s",
            self.baseline.resting_hr_bpm or 0.0,
            self.baseline.sdnn_baseline_ms or 0.0,
            self.baseline.baseline_windows,
            self.baseline.numeric_ready,
            self.baseline.embedding_ready,
            self.baseline.is_complete,
        )

    # ------------------------------------------------------------------
    # Trend recording
    # ------------------------------------------------------------------

    def record_trend_snapshot(
        self,
        tracker_state: dict[str, Any],
        window_range: tuple[int, int],
        offset_range: tuple[float, float],
        anomaly_count_in_period: int = 0,
    ) -> None:
        """
        Record a periodic trend snapshot. All timing from caller.
        """
        medium = tracker_state.get("medium", {})

        snapshot = TrendSnapshot(
            window_range_start=window_range[0],
            window_range_end=window_range[1],
            offset_s_start=offset_range[0],
            offset_s_end=offset_range[1],
            mean_hr_bpm=medium.get("mean_hr"),
            sdnn_ms=medium.get("sdnn_ms"),
            rmssd_ms=medium.get("rmssd_ms"),
            anomaly_count=anomaly_count_in_period,
        )
        self.trend_buffer.record_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Full export (for eval framework)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "baseline": self.baseline.to_dict(),
            "episode_summary": self.episode_log.get_episode_summary(),
            "episodes": [ep.to_dict() for ep in self.episode_log.episodes],
            "alert_summary": self.get_alert_summary(),
            "recent_alerts": [alert.to_dict() for alert in self.get_recent_alerts(10)],
            "trend_summary": self.trend_buffer.get_trend_summary(),
            "trend_snapshots": [s.to_dict() for s in self.trend_buffer.snapshots],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("PatientContext saved to %s", path)
