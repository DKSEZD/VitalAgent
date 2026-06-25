"""Offline replay runtime for proactive monitoring.

This module turns a finite dataset window iterator into a live-like stream:
each window is processed in order, per-window records can be emitted
immediately, and intervention events are surfaced as soon as the proactive
pipeline fires.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from agent.proactive.pipeline import _TREND_SNAPSHOT_INTERVAL_WINDOWS

logger = logging.getLogger(__name__)

ReplayCallback = Callable[[dict[str, Any]], None]


class ProactivePipelineLike(Protocol):
    """Minimal protocol required by the replay runner."""

    tracker: Any

    def process_window(self, window: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReplayConfig:
    """Runtime controls for replaying a window stream."""

    max_windows: int | None = None
    emit_all_windows: bool = True
    emit_patient_context: bool = True
    output_jsonl: Path | None = None
    append_snapshots: bool = True
    snapshot_interval_windows: int = _TREND_SNAPSHOT_INTERVAL_WINDOWS
    sleep_per_window_s: float = 0.0
    realtime_speed: float | None = None


@dataclass(frozen=True)
class ReplaySummary:
    """Aggregate replay outcome."""

    windows_processed: int = 0
    intervention_count: int = 0
    active_condition_windows: int = 0
    first_offset_s: float | None = None
    last_offset_s: float | None = None
    elapsed_wall_s: float = 0.0
    output_jsonl: str | None = None
    interventions: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["interventions"] = list(self.interventions)
        return payload


def replay_windows(
    windows: Iterable[Any],
    pipeline: ProactivePipelineLike,
    config: ReplayConfig | None = None,
    *,
    on_record: ReplayCallback | None = None,
    on_intervention: ReplayCallback | None = None,
) -> ReplaySummary:
    """Replay windows through ``pipeline`` and emit live-like records.

    Parameters
    ----------
    windows:
        Any iterable of ECG/PPG window-like objects accepted by
        ``ProactivePipeline.process_window``.
    pipeline:
        Proactive pipeline instance or a test double exposing
        ``process_window`` and ``tracker``.
    config:
        Replay controls. By default every processed window is emitted and
        trend snapshots are appended using the pipeline's standard cadence.
    on_record:
        Optional callback invoked for emitted JSON-serialisable records.
    on_intervention:
        Optional callback invoked only when an intervention event is emitted.
    """
    cfg = config or ReplayConfig()
    _validate_config(cfg)

    start_wall = time.monotonic()
    previous_offset_s: float | None = None
    first_offset_s: float | None = None
    last_offset_s: float | None = None
    windows_processed = 0
    active_condition_windows = 0
    windows_since_snapshot = 0
    interventions: list[dict[str, Any]] = []
    patient_context = None

    writer = _JsonlWriter(cfg.output_jsonl)
    try:
        for source_index, window in enumerate(windows):
            if cfg.max_windows is not None and source_index >= cfg.max_windows:
                break

            _sleep_for_replay_clock(cfg, previous_offset_s, window)
            result = pipeline.process_window(window)

            offset_s = _window_offset_s(window, result)
            if first_offset_s is None:
                first_offset_s = offset_s
            last_offset_s = offset_s
            previous_offset_s = offset_s
            windows_processed += 1
            windows_since_snapshot += 1

            if bool(result.get("condition_active", False)):
                active_condition_windows += 1

            record = _window_record(window, result, source_index=source_index)
            if cfg.emit_all_windows:
                _emit_record(record, writer, on_record)

            if bool(result.get("intervene", False)):
                event = _intervention_record(window, result, source_index=source_index)
                interventions.append(event)
                patient_context = _record_patient_context_alert(
                    patient_context,
                    event,
                )
                _emit_record(event, writer, on_record)
                if on_intervention is not None:
                    on_intervention(event)
                logger.info(
                    "[Replay window %s] INTERVENTION: %s (urgency=%s)",
                    event.get("window_index"),
                    event.get("reason") or "-",
                    event.get("urgency"),
                )

            if (
                cfg.append_snapshots
                and cfg.snapshot_interval_windows > 0
                and windows_since_snapshot >= cfg.snapshot_interval_windows
            ):
                _append_snapshot_if_available(pipeline)
                windows_since_snapshot = 0

        if cfg.append_snapshots and windows_since_snapshot > 0:
            _append_snapshot_if_available(pipeline)
        if cfg.emit_patient_context and patient_context is not None:
            _emit_record(
                {"record_type": "patient_context"} | patient_context.to_dict(),
                writer,
                on_record,
            )
    finally:
        writer.close()

    return ReplaySummary(
        windows_processed=windows_processed,
        intervention_count=len(interventions),
        active_condition_windows=active_condition_windows,
        first_offset_s=first_offset_s,
        last_offset_s=last_offset_s,
        elapsed_wall_s=time.monotonic() - start_wall,
        output_jsonl=str(cfg.output_jsonl) if cfg.output_jsonl is not None else None,
        interventions=tuple(interventions),
    )


def _validate_config(config: ReplayConfig) -> None:
    if config.max_windows is not None and config.max_windows < 0:
        raise ValueError("max_windows must be non-negative or None.")
    if config.snapshot_interval_windows < 0:
        raise ValueError("snapshot_interval_windows must be non-negative.")
    if config.sleep_per_window_s < 0:
        raise ValueError("sleep_per_window_s must be non-negative.")
    if config.realtime_speed is not None and config.realtime_speed <= 0:
        raise ValueError("realtime_speed must be positive when provided.")


def _sleep_for_replay_clock(
    config: ReplayConfig,
    previous_offset_s: float | None,
    window: Any,
) -> None:
    if config.sleep_per_window_s > 0:
        time.sleep(config.sleep_per_window_s)
        return
    if config.realtime_speed is None or previous_offset_s is None:
        return
    current_offset_s = _window_offset_s(window, {})
    delta_s = max(0.0, current_offset_s - previous_offset_s)
    if delta_s > 0:
        time.sleep(delta_s / config.realtime_speed)


def _window_record(
    window: Any,
    result: dict[str, Any],
    *,
    source_index: int,
) -> dict[str, Any]:
    return {
        "record_type": "window",
        "source_index": source_index,
        "patient_id": str(result.get("patient_id", getattr(window, "patient_id", ""))),
        "segment_id": str(result.get("segment_id", getattr(window, "segment_id", ""))),
        "window_index": int(result.get("window_index", getattr(window, "window_index", 0))),
        "offset_s": _window_offset_s(window, result),
        "intervene": bool(result.get("intervene", False)),
        "condition_active": bool(result.get("condition_active", False)),
        "urgency": str(result.get("urgency", "none")),
        "triggered_rules": list(result.get("triggered_rules") or []),
        "rhythm_class": result.get("rhythm_class"),
        "result": _json_safe(result),
    }


def _intervention_record(
    window: Any,
    result: dict[str, Any],
    *,
    source_index: int,
) -> dict[str, Any]:
    return {
        "record_type": "intervention_event",
        "source_index": source_index,
        "patient_id": str(result.get("patient_id", getattr(window, "patient_id", ""))),
        "segment_id": str(result.get("segment_id", getattr(window, "segment_id", ""))),
        "window_index": int(result.get("window_index", getattr(window, "window_index", 0))),
        "offset_s": _window_offset_s(window, result),
        "urgency": str(result.get("urgency", "none")),
        "triggered_rules": list(result.get("triggered_rules") or []),
        "reason": str(result.get("reason", "")),
        "condition_active": bool(result.get("condition_active", False)),
        "rule_layer": _json_safe(result.get("rule_layer") or {}),
        "judge_layer": _json_safe(result.get("judge_layer") or {}),
        "vitals": _json_safe(result.get("vitals") or {}),
    }


def _window_offset_s(window: Any, result: dict[str, Any]) -> float:
    value = result.get("stream_offset_s", getattr(window, "stream_offset_s", 0.0))
    return float(value or 0.0)


def _emit_record(
    record: dict[str, Any],
    writer: "_JsonlWriter",
    callback: ReplayCallback | None,
) -> None:
    writer.write(record)
    if callback is not None:
        callback(record)


def _append_snapshot_if_available(pipeline: ProactivePipelineLike) -> None:
    append = getattr(getattr(pipeline, "tracker", None), "append_daily_snapshot", None)
    if callable(append):
        append()


def _record_patient_context_alert(
    patient_context: Any,
    event: dict[str, Any],
) -> Any:
    from agent.proactive.patient_context import PatientContext

    if patient_context is None:
        patient_context = PatientContext(patient_id=str(event.get("patient_id") or ""))
    patient_context.record_alert(
        patient_id=str(event.get("patient_id") or ""),
        segment_id=str(event.get("segment_id") or ""),
        window_index=int(event.get("window_index") or 0),
        offset_s=float(event.get("offset_s") or 0.0),
        urgency=str(event.get("urgency") or "none"),
        triggered_rules=list(event.get("triggered_rules") or []),
        reason=str(event.get("reason") or ""),
        condition_active=bool(event.get("condition_active", False)),
        rule_layer=dict(event.get("rule_layer") or {}),
        judge_layer=dict(event.get("judge_layer") or {}),
    )
    return patient_context


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class _JsonlWriter:
    def __init__(self, path: Path | None) -> None:
        self._handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
