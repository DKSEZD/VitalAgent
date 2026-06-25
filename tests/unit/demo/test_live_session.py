from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from agent.data.streaming import ECGWindow
from agent.demo.live_session import LiveMonitorSession, _decimate
from agent.schemas import AgentResponse
from agent.tools.proactive_context_tools import proactive_get_recent_alerts


def _window(index: int) -> ECGWindow:
    fs = 250
    samples = fs * 10
    return ECGWindow(
        signal=np.zeros(samples),
        fs=fs,
        patient_id="01182",
        segment_id="00",
        window_index=index,
        start_sample=index * samples,
        end_sample=(index + 1) * samples,
        n_samples=samples,
        stream_offset_s=float(index * 10),
    )


@dataclass
class _FakeSource:
    windows: list[ECGWindow]
    error: Exception | None = None
    dataset: str = "icentia11k"
    patient_id: str = "01182"
    duration_s: float | None = 20.0
    window_seconds: float = 10.0

    def iter_windows(self) -> Iterator[ECGWindow]:
        if self.error is not None:
            raise self.error
        yield from self.windows


class _StubProactivePipeline:
    def __init__(self, block_first: bool = False) -> None:
        self.block_first = block_first
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def process_window(self, window: ECGWindow) -> dict[str, Any]:
        if self.block_first and window.window_index == 0:
            self.first_started.set()
            assert self.release_first.wait(timeout=2.0)
        intervene = window.window_index == 1
        return {
            "intervene": intervene,
            "condition_active": intervene,
            "urgency": "medium" if intervene else "none",
            "reason": "test alert" if intervene else "",
            "triggered_rules": ["test_rule"] if intervene else [],
            "window_index": window.window_index,
            "patient_id": window.patient_id,
            "segment_id": window.segment_id,
            "stream_offset_s": window.stream_offset_s,
            "rule_layer": {"intervene": intervene},
            "judge_layer": {"intervene": False},
        }


class _StubReactivePipeline:
    def __init__(self) -> None:
        self.query: str | None = None
        self.active_context: dict[str, Any] | None = None

    def run(self, user_query: str, active_context: dict[str, Any]):
        self.query = user_query
        self.active_context = active_context
        yield AgentResponse(answer="planning", context_used={"event": "status"})
        yield AgentResponse(answer="final answer", context_used={})


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1000.0
        return self.value


def test_live_alert_registration_fast_forward_and_reactive_context() -> None:
    proactive = _StubProactivePipeline(block_first=True)
    reactive = _StubReactivePipeline()
    alerts: list[dict[str, Any]] = []
    session = LiveMonitorSession(
        _FakeSource([_window(0), _window(1)]),
        speed=1_000_000.0,
        on_alert=alerts.append,
        proactive_pipeline=proactive,
        reactive_pipeline=reactive,
        clock=_AdvancingClock(),
    )

    session.start()
    assert proactive.first_started.wait(timeout=2.0)

    empty = proactive_get_recent_alerts(patient_id="01182")
    assert empty["success"] is True
    assert empty["alert_count"] == 0

    session.fast_forward_to_next_alert()
    proactive.release_first.set()
    session.wait(timeout=2.0)

    snapshot = session.snapshot()
    assert snapshot.completed is True
    assert snapshot.error is None
    assert snapshot.fast_forwarding is False
    assert snapshot.windows_processed == 2
    assert snapshot.alert_count == 1
    assert len(alerts) == 1

    recent = proactive_get_recent_alerts(patient_id="01182")
    assert recent["success"] is True
    assert recent["alert_count"] == 1
    assert recent["alerts"][0]["reason"] == "test alert"

    statuses: list[str] = []
    response = session.ask("刚刚为什么提醒我？", on_status=statuses.append)
    assert response.answer == "final answer"
    assert statuses == ["planning"]
    assert reactive.active_context is not None
    assert reactive.active_context["window_end_s"] == 20.0
    assert reactive.active_context["recording_id"] == "00"
    assert reactive.query is not None
    assert "Only analyze data up to t=20s" in reactive.query
    assert "current recording segment (00)" in reactive.query


def test_data_source_error_is_recorded_in_snapshot() -> None:
    session = LiveMonitorSession(
        _FakeSource([], error=RuntimeError("broken source")),
        proactive_pipeline=_StubProactivePipeline(),
        reactive_pipeline=_StubReactivePipeline(),
        clock=_AdvancingClock(),
    )

    session.start()
    session.wait(timeout=2.0)

    snapshot = session.snapshot()
    assert snapshot.running is False
    assert snapshot.completed is True
    assert snapshot.error == "broken source"


def test_decimate_spans_signal_and_sanitizes_non_finite_values() -> None:
    signal = np.arange(499, dtype=float)
    signal[20] = np.nan
    signal[40] = np.inf

    preview, fs_preview = _decimate(signal, fs=250, target=250)

    assert len(preview) == 250
    assert preview[10] is None
    assert preview[20] is None
    assert preview[-1] == 498.0
    assert fs_preview == 125.0


def test_signal_preview_and_alert_carry_triggering_window() -> None:
    proactive = _StubProactivePipeline()
    window_records: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    session = LiveMonitorSession(
        _FakeSource([_window(1)]),
        speed=1_000_000.0,
        on_window=window_records.append,
        on_alert=alerts.append,
        emit_signal_preview=True,
        proactive_pipeline=proactive,
        reactive_pipeline=_StubReactivePipeline(),
        clock=_AdvancingClock(),
    )

    session.start()
    session.wait(timeout=2.0)

    assert len(window_records[0]["signal_preview"]) == 250
    assert alerts[0]["window"]["signal_preview"] == window_records[0]["signal_preview"]
    assert alerts[0]["window"]["offset"] == 10.0


def test_ask_stream_yields_status_and_final_with_anchored_context() -> None:
    reactive = _StubReactivePipeline()
    session = LiveMonitorSession(
        _FakeSource([_window(0)]),
        proactive_pipeline=_StubProactivePipeline(),
        reactive_pipeline=reactive,
        clock=_AdvancingClock(),
    )
    session.start()
    session.wait(timeout=2.0)

    frames = list(session.ask_stream("现在怎么样？"))

    assert [frame.context_used.get("event") for frame in frames] == ["status", None]
    assert reactive.active_context is not None
    assert reactive.active_context["window_end_s"] == 10.0
    assert reactive.query is not None
    assert "Only analyze data up to t=10s" in reactive.query


def test_ask_stream_adds_optional_conversation_history() -> None:
    reactive = _StubReactivePipeline()
    session = LiveMonitorSession(
        _FakeSource([_window(0)]),
        proactive_pipeline=_StubProactivePipeline(),
        reactive_pipeline=reactive,
        clock=_AdvancingClock(),
    )
    session.start()
    session.wait(timeout=2.0)

    list(
        session.ask_stream(
            "那现在呢？",
            history=[{"question": "之前怎么样？", "answer": "之前心率正常。"}],
        )
    )
    assert reactive.query is not None
    assert "Conversation so far (most recent last):" in reactive.query
    assert "Q: 之前怎么样？" in reactive.query
    assert "A: 之前心率正常。" in reactive.query

    list(session.ask_stream("独立问题", history=None))
    assert reactive.query is not None
    assert "Conversation so far" not in reactive.query
