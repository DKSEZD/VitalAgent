from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from agent.data.icentia11k_loader import Icentia11kLoader, RawECGData
from agent.demo.data_sources import Icentia11kDataSource


def _raw(segment_id: str, *, seconds: int, sample_start: int = 0) -> RawECGData:
    fs = 250
    n_samples = fs * seconds
    return RawECGData(
        signal=np.zeros(n_samples),
        fs=fs,
        patient_id="01182",
        segment_id=segment_id,
        n_samples=n_samples,
        sample_start=sample_start,
        sample_end=sample_start + n_samples,
        segment_total_samples=sample_start + n_samples,
        segment_start_time=datetime(2020, 1, 1),
        window_start_time=datetime(2020, 1, 1),
        window_duration_seconds=float(seconds),
    )


def test_icentia_source_precomputes_stable_duration_and_chains_actual_offsets(
    monkeypatch,
) -> None:
    headers = {
        "00": SimpleNamespace(n_samples=17 * 250, fs=250),
        "01": SimpleNamespace(n_samples=24 * 250, fs=250),
    }
    monkeypatch.setattr(
        Icentia11kLoader,
        "read_segment_header",
        lambda _self, _patient_id, segment_id: headers[str(segment_id)],
    )
    source = Icentia11kDataSource(
        patient_id="01182",
        segments=[("00", 2.0, None), ("01", 4.0, None)],
        window_seconds=10.0,
    )
    assert source.duration_s == 35.0

    raws = iter(
        [
            _raw("00", seconds=15, sample_start=500),
            _raw("01", seconds=20, sample_start=1000),
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_read_segment(**kwargs):
        calls.append(kwargs)
        return next(raws)

    source._loader.read_segment = fake_read_segment  # type: ignore[method-assign]

    iterator = source.iter_windows()
    first_window = next(iterator)
    assert source.duration_s == 35.0
    windows = [first_window, *iterator]

    assert calls == [
        {"patient_id": "01182", "segment_id": "00", "sampfrom": 500, "sampto": None},
        {"patient_id": "01182", "segment_id": "01", "sampfrom": 1000, "sampto": None},
    ]
    assert [window.window_index for window in windows] == [0, 1, 2]
    assert [window.stream_offset_s for window in windows] == [0.0, 15.0, 25.0]
    assert source.duration_s == 35.0
