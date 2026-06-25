from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from agent.data.streaming import ECGStreamer
from agent.data.afppgecg_adapter import (
    AFPPGECGSegmentLoader,
    build_afppgecg_rhythm_change_annotation,
)
from agent.evaluation.proactive.metrics import derive_window_truth_label


def _write_char_dataset(handle: h5py.File, name: str, text: str) -> h5py.Dataset:
    values = np.array([[ord(ch)] for ch in text], dtype=np.uint16)
    ds = handle.create_dataset(name, data=values)
    ds.attrs["MATLAB_class"] = np.bytes_(b"char")
    ds.attrs["MATLAB_int_decode"] = np.int32(2)
    return ds


def _write_scalar_dataset(handle: h5py.File, name: str, value: float) -> h5py.Dataset:
    ds = handle.create_dataset(name, data=np.array([[value]], dtype=float))
    ds.attrs["MATLAB_class"] = np.bytes_(b"double")
    return ds


def _write_signal_header(handle: h5py.File, fs: int) -> None:
    group = handle.create_group("signalHeader")
    group.attrs["MATLAB_class"] = np.bytes_(b"struct")
    values = {
        "signal_labels": [_write_char_dataset(handle, "sl_ecg", "ECG")],
        "physical_dimension": [_write_char_dataset(handle, "pd_ecg", "uV")],
        "physical_min": [_write_scalar_dataset(handle, "pmin_ecg", -1000)],
        "physical_max": [_write_scalar_dataset(handle, "pmax_ecg", 1000)],
        "samples_in_record": [_write_scalar_dataset(handle, "sir_ecg", fs)],
    }
    for field_name, datasets in values.items():
        group.create_dataset(
            field_name,
            data=np.array([[dataset.ref] for dataset in datasets], dtype=h5py.ref_dtype),
        )


def _create_ecg_fixture(path: Path) -> None:
    fs = 10
    with h5py.File(path, "w") as handle:
        ecg = handle.create_dataset(
            "ECG",
            data=np.arange(30, dtype=float).reshape(-1, 1),
        )
        ecg.attrs["MATLAB_class"] = np.bytes_(b"double")

        qrs = handle.create_dataset(
            "QRSindex",
            data=np.array([[0], [10], [20]], dtype=float),
        )
        qrs.attrs["MATLAB_class"] = np.bytes_(b"double")
        qrs.attrs["MATLAB_global"] = np.uint8(1)

        rr = handle.create_dataset("rr", data=np.array([[1], [1]], dtype=float))
        rr.attrs["MATLAB_class"] = np.bytes_(b"double")

        af = handle.create_dataset(
            "AF_annotation",
            data=np.array([[0], [1], [0]], dtype=float),
        )
        af.attrs["MATLAB_class"] = np.bytes_(b"double")

        _write_char_dataset(handle, "recording_startday", "01")
        _write_char_dataset(handle, "recording_starttime", "00:00:00")
        _write_signal_header(handle, fs)


def test_afppgecg_rhythm_change_annotation_uses_wfdb_style_markers() -> None:
    annotation = build_afppgecg_rhythm_change_annotation(
        qrs_index=np.array([5, 10, 20, 25]),
        af_annotation=np.array([0, 0, 1, 0]),
    )

    assert annotation is not None
    assert annotation.sample.tolist() == [0, 20, 25]
    assert annotation.symbol == ["+", "+", "+"]
    assert annotation.aux_note == ["(N", "(AFIB", "(N"]


def test_afppgecg_adapter_streams_windows_with_downstream_truth_labels(tmp_path: Path) -> None:
    _create_ecg_fixture(tmp_path / "001_ECG.mat")
    loader = AFPPGECGSegmentLoader(dataset_root=tmp_path)

    assert loader.available_patient_ids() == ["001"]
    assert loader.get_patient_segments("1")[0].segment_id == "ecg"

    windows = list(
        ECGStreamer(window_seconds=1.0).stream_patient(
            loader,
            patient_id="1",
            max_windows=3,
        )
    )

    assert len(windows) == 3
    assert [window.segment_id for window in windows] == ["ecg", "ecg", "ecg"]
    assert all(window.annotations is not None for window in windows)

    labels = [derive_window_truth_label(window) for window in windows]
    assert labels == ["N", "AF", "N"]
    assert windows[1].annotations is not None
    assert windows[1].annotations.rhythm_spans[0].rhythm_type == "AFIB"
