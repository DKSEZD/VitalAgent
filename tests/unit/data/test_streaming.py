from __future__ import annotations

import numpy as np

from agent.data.ppg_loader import PPGChunk, PPGRecording, RelativeTimestamp
from agent.data.streaming import PPGStreamer


def _ppg_recording() -> PPGRecording:
    chunk = PPGChunk(
        patient_id="001",
        chunk_index=0,
        start_timestamp=RelativeTimestamp(day_index=1, time_text="00:00:00"),
        ppg_fs=10,
        accel_fs=10,
        n_ppg_samples=40,
        green_signal=np.arange(40, dtype=float),
        ambient_signal=np.arange(40, dtype=float),
        accel_x=np.zeros(40, dtype=float),
        accel_y=np.zeros(40, dtype=float),
        accel_z=np.zeros(40, dtype=float),
    )
    return PPGRecording(patient_id="001", chunks=[chunk])


def test_ppg_streamer_defaults_to_non_overlapping_windows() -> None:
    recording = _ppg_recording()

    windows = list(PPGStreamer(window_seconds=1.0).stream_recording(recording))

    assert [window.start_sample for window in windows] == [0, 10, 20, 30]
    assert [window.n_samples for window in windows] == [10, 10, 10, 10]


def test_ppg_streamer_supports_stride_seconds_for_overlap() -> None:
    recording = _ppg_recording()

    windows = list(
        PPGStreamer(window_seconds=2.0, stride_seconds=1.0).stream_recording(recording)
    )

    assert [window.start_sample for window in windows] == [0, 10, 20]
    assert [window.end_sample for window in windows] == [20, 30, 40]
    assert [window.window_index for window in windows] == [0, 1, 2]


def test_ppg_window_exposes_stable_window_id() -> None:
    recording = _ppg_recording()

    window = next(iter(PPGStreamer(window_seconds=1.0).stream_recording(recording)))

    assert window.window_id == "ppg:001:chunk000:w000000:s000000000-e000000010"
