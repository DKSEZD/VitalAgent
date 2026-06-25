from __future__ import annotations

import numpy as np
import pytest

from agent.data.single_lead import (
    SingleLeadNormalizationError,
    build_single_lead_ecg_record,
    coerce_single_lead_signal,
    derive_sampling_rate_from_time_ms,
)


def test_coerce_single_lead_signal_accepts_vector_and_column_shapes() -> None:
    vector = coerce_single_lead_signal([0.1, 0.2, 0.3])
    column = coerce_single_lead_signal(np.array([[0.1], [0.2], [0.3]], dtype=float))
    row = coerce_single_lead_signal(np.array([[0.1, 0.2, 0.3]], dtype=float))

    assert vector.shape == (3,)
    assert column.shape == (3,)
    assert row.shape == (3,)
    assert vector.dtype == np.float32


def test_coerce_single_lead_signal_rejects_empty_and_multilead_inputs() -> None:
    with pytest.raises(SingleLeadNormalizationError, match="cannot be empty"):
        coerce_single_lead_signal([])

    with pytest.raises(SingleLeadNormalizationError, match="single-lead signal"):
        coerce_single_lead_signal(np.zeros((10, 2), dtype=float))


def test_derive_sampling_rate_from_time_ms_uses_even_spacing() -> None:
    sampling_rate = derive_sampling_rate_from_time_ms([0.0, 4.0, 8.0, 12.0])

    assert sampling_rate == 250


def test_build_single_lead_ecg_record_normalizes_lead_and_metadata() -> None:
    record = build_single_lead_ecg_record(
        record_id="aw-123",
        signal=[0.0, 0.1, 0.2, 0.3],
        sampling_rate=512,
        metadata={"source": "apple_watch", "patient_id": "p-1"},
    )

    assert record.record_id == "aw-123"
    assert record.signal.shape == (4, 1)
    assert record.lead_names == ["I"]
    assert record.metadata["source"] == "apple_watch"
    assert record.metadata["num_leads"] == 1
    assert record.metadata["lead_names"] == ["I"]
