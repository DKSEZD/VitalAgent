from __future__ import annotations

import math
import threading
import time

import pytest

from agent import web_demo
from agent.data.apple_watch_loader import AppleWatchECGLoader, AppleWatchUploadError
from agent.data.record_repository import RecordRepository
from agent.demo.live_session import SessionSnapshot
import agent.data.record_repository as record_repository_module


@pytest.fixture
def repo_with_tmp_loader(tmp_path, monkeypatch):
    repo = RecordRepository(apple_watch_loader=AppleWatchECGLoader(storage_root=tmp_path))
    monkeypatch.setattr(record_repository_module, "_record_repository", repo)
    return repo


def test_upload_ecg_json_and_list_records(repo_with_tmp_loader) -> None:
    response = web_demo.upload_ecg_json(
        {
            "samples": [0.0, 0.1, 0.2, 0.3],
            "sampling_rate": 512,
            "patient_id": "watch-user",
        }
    )
    records = web_demo.list_records(limit=10)

    assert response["success"] is True
    assert response["record_id"].startswith("apple_watch:aw-")
    assert records[0]["record_id"] == response["record_id"]
    assert records[0]["source"] == "apple_watch"
    assert records[0]["lead_names"] == ["I"]
    assert records[0]["num_leads"] == 1


def test_upload_ecg_file_rejects_invalid_sampling_rate(repo_with_tmp_loader) -> None:
    with pytest.raises(AppleWatchUploadError, match="sampling_rate"):
        web_demo.upload_ecg_file(
            filename="watch.csv",
            content=b"sample\n0.0\n0.1\n",
            fields={"sampling_rate": "abc"},
        )


class _FakeSource:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.patient_id = kwargs["patient_id"]
        self.duration_s = 120.0
        self.window_seconds = kwargs["window_seconds"]
        self.dataset = "icentia11k"
        self.__class__.instances.append(self)


class _FakeSession:
    instances = []
    active_starts = 0
    max_active_starts = 0
    counter_lock = threading.Lock()

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.completed = False
        self.windows_processed = 1
        self.__class__.instances.append(self)

    def start(self):
        with self.counter_lock:
            self.__class__.active_starts += 1
            self.__class__.max_active_starts = max(
                self.__class__.max_active_starts,
                self.__class__.active_starts,
            )
        time.sleep(0.02)
        self.started = True
        with self.counter_lock:
            self.__class__.active_starts -= 1

    def stop(self):
        self.stopped = True

    def wait(self, timeout=None):
        return None

    def snapshot(self):
        return SessionSnapshot(
            offset_s_global=10.0,
            duration_s=120.0,
            windows_processed=self.windows_processed,
            alert_count=0,
            speed=30.0,
            paused=False,
            running=not self.completed,
            completed=self.completed,
            error=None,
            fast_forwarding=False,
            current_segment_id="00",
        )


@pytest.fixture
def fake_live_runtime(monkeypatch):
    _FakeSource.instances.clear()
    _FakeSession.instances.clear()
    _FakeSession.active_starts = 0
    _FakeSession.max_active_starts = 0
    monkeypatch.setattr(web_demo, "Icentia11kDataSource", _FakeSource)
    monkeypatch.setattr(web_demo, "LiveMonitorSession", _FakeSession)
    return web_demo.LiveWebRuntime()


def _window_record(offset: float) -> dict:
    return {
        "offset_s": offset,
        "segment_id": "00",
        "window_index": int(offset / 10),
        "intervene": False,
        "signal_preview": [offset, offset + 1],
        "fs_preview": 25.0,
        "result": {"vitals": {"hr_bpm": 72.0}, "rhythm_class": "N"},
    }


def test_sanitize_json_recursively_cleans_non_finite_and_numpy() -> None:
    import numpy as np

    cleaned = web_demo._sanitize_json(
        {"a": [math.nan, {"b": math.inf}], "c": np.float32(-math.inf)}
    )

    assert cleaned == {"a": [None, {"b": None}], "c": None}


def test_start_parses_segments_and_returns_snapshot(fake_live_runtime) -> None:
    result = fake_live_runtime.start_session(
        {"patient": "01182", "segments": ["00", "01"], "speed": 30}
    )

    assert result["running"] is True
    assert _FakeSource.instances[-1].kwargs["segments"] == [
        ("00", 0.0, None),
        ("01", 0.0, None),
    ]


def test_stale_generation_callback_is_dropped_and_new_subscriber_gets_latest(
    fake_live_runtime,
) -> None:
    fake_live_runtime.start_session({"segments": ["00"]})
    old = _FakeSession.instances[-1]
    fake_live_runtime.start_session({"segments": ["01"]})
    new = _FakeSession.instances[-1]

    old.kwargs["on_window"](_window_record(10.0))
    new.kwargs["on_window"](_window_record(20.0))
    fake_live_runtime._latest_dirty = False

    first = next(fake_live_runtime.iter_events())
    assert first["offset_s"] == 20.0


def test_completed_stream_flushes_alert_before_done(fake_live_runtime) -> None:
    fake_live_runtime.start_session({"segments": ["00"]})
    session = _FakeSession.instances[-1]
    session.kwargs["on_window"](_window_record(10.0))
    session.kwargs["on_alert"]({"reason": "alert", "window": {"offset": 10.0}})
    session.completed = True

    events = list(fake_live_runtime.iter_events(sleeper=lambda _: None))
    types = [event["type"] for event in events]

    assert types.index("alert") < types.index("done")
    assert types[0] == "window"


def test_status_is_not_starved_by_continuous_windows(fake_live_runtime) -> None:
    fake_live_runtime.start_session({"segments": ["00"]})
    session = _FakeSession.instances[-1]
    session.kwargs["on_window"](_window_record(10.0))
    clock_value = [0.0]

    def clock():
        clock_value[0] += 0.6
        return clock_value[0]

    events = fake_live_runtime.iter_events(clock=clock, sleeper=lambda _: None)
    assert next(events)["type"] == "window"
    session.kwargs["on_window"](_window_record(20.0))
    observed = [next(events)["type"] for _ in range(2)]
    assert "status" in observed


def test_chat_in_progress_blocks_start(fake_live_runtime) -> None:
    fake_live_runtime._chat_lock.acquire()
    try:
        with pytest.raises(web_demo.LiveSessionConflictError):
            fake_live_runtime.start_session({"segments": ["00"]})
    finally:
        fake_live_runtime._chat_lock.release()


def test_concurrent_starts_are_serialized(fake_live_runtime) -> None:
    errors = []

    def start(segment):
        try:
            fake_live_runtime.start_session({"segments": [segment]})
        except Exception as exc:  # pragma: no cover - assertion captures failures
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(segment,)) for segment in ("00", "01")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert errors == []
    assert _FakeSession.max_active_starts == 1


def test_control_without_session_returns_conflict() -> None:
    runtime = web_demo.LiveWebRuntime()
    with pytest.raises(web_demo.NoLiveSessionError):
        runtime.control("pause")


def test_chat_history_is_session_scoped_bounded_and_resettable(fake_live_runtime) -> None:
    fake_live_runtime.start_session({"segments": ["00"]})
    session = fake_live_runtime.acquire_chat_session()
    assert session is _FakeSession.instances[-1]
    for index in range(14):
        fake_live_runtime.record_chat_turn(f"q{index}", f"a{index}")
    assert fake_live_runtime.chat_history_for_prompt() == [
        {"question": f"q{index}", "answer": f"a{index}"}
        for index in range(10, 14)
    ]
    fake_live_runtime.release_chat_session()

    fake_live_runtime.reset_chat_history()
    assert fake_live_runtime._chat_history == []
