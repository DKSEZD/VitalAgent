"""Reusable live-monitoring session for CLI and future web demos."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from agent.demo.data_sources import DataSource
from agent.proactive.patient_context import PatientContext
from agent.proactive.pipeline import ProactivePipeline
from agent.reactive.pipeline import ReactivePipeline
from agent.schemas import AgentResponse
from agent.tools.proactive_context_tools import register_patient_context

logger = logging.getLogger(__name__)

WindowCallback = Callable[[dict[str, Any]], None]
AlertCallback = Callable[[dict[str, Any]], None]
StatusCallback = Callable[[str], None]


def _decimate(
    signal: Any,
    fs: float,
    target: int = 250,
) -> tuple[list[float | None], float]:
    """Return a JSON-safe preview spanning the complete signal window."""
    if target <= 0:
        raise ValueError("target must be positive.")
    values = np.asarray(signal).reshape(-1)
    if values.size == 0:
        return [], float(fs)
    step = max(1, math.ceil(values.size / target))
    preview: list[float | None] = []
    for value in values[::step][:target]:
        number = float(value)
        preview.append(round(number, 6) if math.isfinite(number) else None)
    return preview, float(fs) / step


@dataclass(frozen=True)
class SessionSnapshot:
    offset_s_global: float
    duration_s: float | None
    windows_processed: int
    alert_count: int
    speed: float
    paused: bool
    running: bool
    completed: bool
    error: str | None
    fast_forwarding: bool
    current_segment_id: str | None


@dataclass
class _Cursor:
    offset_s_global: float = 0.0
    segment_id: str | None = None
    segment_local_end_s: float | None = None


class LiveMonitorSession:
    """Run proactive monitoring in a paced thread and answer live questions."""

    def __init__(
        self,
        source: DataSource,
        sampling_rate: int = 250,
        speed: float = 30.0,
        on_window: WindowCallback | None = None,
        on_alert: AlertCallback | None = None,
        emit_signal_preview: bool = False,
        *,
        proactive_pipeline: Any | None = None,
        reactive_pipeline: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive.")

        self.source = source
        self.window_seconds = float(getattr(source, "window_seconds", 10.0))
        self.proactive_pipeline = proactive_pipeline or ProactivePipeline(
            sampling_rate=sampling_rate
        )
        self.reactive_pipeline = reactive_pipeline or ReactivePipeline()
        self._ctx = PatientContext(patient_id=source.patient_id)
        self._clock = clock
        self._on_window = on_window
        self._on_alert = on_alert
        self._emit_signal_preview = bool(emit_signal_preview)

        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._cursor = _Cursor()
        self._speed = float(speed)
        self._stop_requested = False
        self._paused = False
        self._running = False
        self._completed = False
        self._error: str | None = None
        self._fast_forwarding = False
        self._windows_processed = 0
        self._alert_count = 0
        self._clock_origin_wall: float | None = None
        self._clock_origin_offset = 0.0

    def start(self) -> None:
        with self._cond:
            if self._running:
                return
            if self._thread is not None or self._completed:
                raise RuntimeError("LiveMonitorSession instances cannot be restarted.")
            if self._stop_requested:
                raise RuntimeError("Cannot start a stopped LiveMonitorSession.")

            empty_context = self._ctx.to_dict()
            self._running = True
            self._clock_origin_wall = self._clock()
            self._clock_origin_offset = self._cursor.offset_s_global
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"live-monitor-{self.source.patient_id}",
                daemon=True,
            )

        register_patient_context(empty_context, clear_existing=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stop_requested = True
            self._cond.notify_all()

    def wait(self, timeout: float | None = None) -> None:
        with self._cond:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def pause(self) -> None:
        with self._cond:
            self._paused = True
            self._cond.notify_all()

    def resume(self) -> None:
        with self._cond:
            if not self._paused:
                return
            self._paused = False
            self._rebase_clock_locked()
            self._cond.notify_all()

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive.")
        with self._cond:
            self._speed = float(speed)
            self._rebase_clock_locked()
            self._cond.notify_all()

    def fast_forward_to_next_alert(self) -> None:
        with self._cond:
            if self._completed or self._stop_requested:
                return
            self._fast_forwarding = True
            self._paused = False
            self._cond.notify_all()

    def ask(
        self,
        question: str,
        on_status: StatusCallback | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        final_response: AgentResponse | None = None
        for response in self.ask_stream(question, history=history):
            if response.context_used.get("event") == "status":
                if on_status is not None:
                    on_status(response.answer)
                continue
            final_response = response

        if final_response is None:
            raise RuntimeError("Reactive pipeline produced no final response.")
        return final_response

    def ask_stream(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> Iterator[AgentResponse]:
        """Yield reactive status and answer frames anchored to the live cursor."""
        with self._cond:
            segment_id = self._cursor.segment_id
            end_s = self._cursor.segment_local_end_s

        if segment_id is None or end_s is None:
            raise RuntimeError("No monitoring window has been processed yet.")

        active_context = {
            "dataset": self.source.dataset,
            "patient_id": self.source.patient_id,
            "subject_id": self.source.patient_id,
            "recording_id": segment_id,
            "segment_id": segment_id,
            "timestamp_s": end_s,
            "window_start_s": max(0.0, end_s - self.window_seconds),
            "window_end_s": end_s,
        }
        history_block = ""
        if history:
            turns = []
            for turn in history:
                prior_question = str(turn.get("question") or "").strip()
                prior_answer = str(turn.get("answer") or "").strip()[:600]
                if prior_question or prior_answer:
                    turns.append(f"Q: {prior_question}\nA: {prior_answer}")
            if turns:
                history_block = (
                    "\n\nConversation so far (most recent last):\n"
                    + "\n".join(turns)
                )

        constrained_query = (
            question
            + history_block
            + f"\n\n[Monitoring constraint] Only analyze data up to t={end_s:.0f}s "
            f"within the current recording segment ({segment_id}). "
            "Do not request later timestamps."
        )

        yield from self.reactive_pipeline.run(
            user_query=constrained_query,
            active_context=active_context,
        )

    def snapshot(self) -> SessionSnapshot:
        with self._cond:
            return SessionSnapshot(
                offset_s_global=self._cursor.offset_s_global,
                duration_s=self.source.duration_s,
                windows_processed=self._windows_processed,
                alert_count=self._alert_count,
                speed=self._speed,
                paused=self._paused,
                running=self._running,
                completed=self._completed,
                error=self._error,
                fast_forwarding=self._fast_forwarding,
                current_segment_id=self._cursor.segment_id,
            )

    def _run_loop(self) -> None:
        try:
            for window in self.source.iter_windows():
                if not self._wait_until_window(window):
                    break

                result = self.proactive_pipeline.process_window(window)
                window_duration_s = float(window.n_samples) / float(window.fs)
                local_end_s = float(window.start_sample) / float(window.fs) + window_duration_s
                intervene = bool(result.get("intervene", False))

                with self._cond:
                    if self._stop_requested:
                        break
                    self._cursor = _Cursor(
                        offset_s_global=float(window.stream_offset_s),
                        segment_id=str(window.segment_id),
                        segment_local_end_s=local_end_s,
                    )
                    self._windows_processed += 1
                    if self._fast_forwarding and intervene:
                        self._fast_forwarding = False
                        self._rebase_clock_locked()

                window_record = self._window_record(window, result, local_end_s)
                if self._on_window is not None:
                    self._on_window(window_record)

                if intervene:
                    self._handle_alert(window, result)
        except Exception as exc:
            logger.exception("Live monitoring session failed")
            with self._cond:
                self._error = str(exc)
        finally:
            with self._cond:
                self._running = False
                self._completed = True
                self._fast_forwarding = False
                self._cond.notify_all()

    def _wait_until_window(self, window: Any) -> bool:
        offset_s = float(window.stream_offset_s)
        with self._cond:
            while True:
                if self._stop_requested:
                    return False
                if self._paused:
                    self._cond.wait()
                    continue
                if self._fast_forwarding:
                    return True

                if self._clock_origin_wall is None:
                    self._clock_origin_wall = self._clock()
                    self._clock_origin_offset = offset_s
                deadline = self._clock_origin_wall + (
                    offset_s - self._clock_origin_offset
                ) / self._speed
                remaining = deadline - self._clock()
                if remaining <= 0:
                    if remaining < -self.window_seconds / self._speed:
                        logger.warning("Live monitor is %.3fs behind schedule", -remaining)
                    return True
                self._cond.wait(timeout=remaining)

    def _handle_alert(self, window: Any, result: dict[str, Any]) -> None:
        with self._cond:
            alert = self._ctx.record_alert(
                patient_id=str(result.get("patient_id") or window.patient_id),
                segment_id=str(result.get("segment_id") or window.segment_id),
                window_index=int(result.get("window_index", window.window_index)),
                offset_s=float(result.get("stream_offset_s", window.stream_offset_s)),
                urgency=str(result.get("urgency") or "none"),
                triggered_rules=list(result.get("triggered_rules") or []),
                reason=str(result.get("reason") or ""),
                condition_active=bool(result.get("condition_active", False)),
                rule_layer=dict(result.get("rule_layer") or {}),
                judge_layer=dict(result.get("judge_layer") or {}),
            )
            self._alert_count += 1
            context_snapshot = self._ctx.to_dict()
            event = alert.to_dict()
            if self._emit_signal_preview:
                preview, fs_preview = _decimate(window.signal, window.fs)
                event["window"] = {
                    "signal_preview": preview,
                    "fs_preview": fs_preview,
                    "offset": float(window.stream_offset_s),
                }

        register_patient_context(context_snapshot)
        if self._on_alert is not None:
            self._on_alert(event)

    def _rebase_clock_locked(self) -> None:
        self._clock_origin_wall = self._clock()
        self._clock_origin_offset = self._cursor.offset_s_global

    def _window_record(
        self,
        window: Any,
        result: dict[str, Any],
        local_end_s: float,
    ) -> dict[str, Any]:
        record = {
            "patient_id": str(result.get("patient_id") or window.patient_id),
            "segment_id": str(result.get("segment_id") or window.segment_id),
            "window_index": int(result.get("window_index", window.window_index)),
            "offset_s": float(result.get("stream_offset_s", window.stream_offset_s)),
            "segment_local_end_s": local_end_s,
            "intervene": bool(result.get("intervene", False)),
            "condition_active": bool(result.get("condition_active", False)),
            "urgency": str(result.get("urgency") or "none"),
            "triggered_rules": list(result.get("triggered_rules") or []),
            "result": result,
        }
        if self._emit_signal_preview:
            preview, fs_preview = _decimate(window.signal, window.fs)
            record["signal_preview"] = preview
            record["fs_preview"] = fs_preview
        return record
