from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd

from agent.data.ppg_loader import PPGLoader


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


def _write_ref_vector(handle: h5py.File, name: str, datasets: list[h5py.Dataset]) -> h5py.Dataset:
    refs = np.array([[dataset.ref] for dataset in datasets], dtype=h5py.ref_dtype)
    ds = handle.create_dataset(name, data=refs)
    ds.attrs["MATLAB_class"] = np.bytes_(b"cell")
    return ds


def _write_signal_header(handle: h5py.File, values: dict[str, list[h5py.Dataset]]) -> None:
    group = handle.create_group("signalHeader")
    group.attrs["MATLAB_class"] = np.bytes_(b"struct")
    for field_name, datasets in values.items():
        group.create_dataset(
            field_name,
            data=np.array([[dataset.ref] for dataset in datasets], dtype=h5py.ref_dtype),
        )


def _create_ecg_file(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        ecg = handle.create_dataset("ECG", data=np.arange(20, dtype=float).reshape(-1, 1))
        ecg.attrs["MATLAB_class"] = np.bytes_(b"double")

        qrs = handle.create_dataset("QRSindex", data=np.array([[0], [5], [10], [15]], dtype=float))
        qrs.attrs["MATLAB_class"] = np.bytes_(b"double")
        qrs.attrs["MATLAB_global"] = np.uint8(1)

        rr = handle.create_dataset("rr", data=np.array([[0.5], [0.5], [0.5]], dtype=float))
        rr.attrs["MATLAB_class"] = np.bytes_(b"double")

        af = handle.create_dataset("AF_annotation", data=np.array([[0], [1], [1]], dtype=float))
        af.attrs["MATLAB_class"] = np.bytes_(b"double")

        _write_char_dataset(handle, "recording_startday", "01")
        _write_char_dataset(handle, "recording_starttime", "00:00:10")

        _write_signal_header(
            handle,
            {
                "signal_labels": [_write_char_dataset(handle, "sl_ecg", "ECG")],
                "physical_dimension": [_write_char_dataset(handle, "pd_ecg", "uV")],
                "physical_min": [_write_scalar_dataset(handle, "pmin_ecg", -1000)],
                "physical_max": [_write_scalar_dataset(handle, "pmax_ecg", 1000)],
                "samples_in_record": [_write_scalar_dataset(handle, "sir_ecg", 500)],
            },
        )


def _create_ppg_file(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        green_1 = handle.create_dataset("green_1", data=np.arange(300, dtype=float).reshape(1, -1))
        green_1.attrs["MATLAB_class"] = np.bytes_(b"double")
        green_2 = handle.create_dataset("green_2", data=np.arange(500, dtype=float).reshape(1, -1))
        green_2.attrs["MATLAB_class"] = np.bytes_(b"double")
        ambient_1 = handle.create_dataset("ambient_1", data=np.arange(300, dtype=float).reshape(1, -1))
        ambient_1.attrs["MATLAB_class"] = np.bytes_(b"double")
        ambient_2 = handle.create_dataset("ambient_2", data=np.arange(500, dtype=float).reshape(1, -1))
        ambient_2.attrs["MATLAB_class"] = np.bytes_(b"double")
        ax_1 = handle.create_dataset("ax_1", data=np.arange(150, dtype=float).reshape(1, -1))
        ax_1.attrs["MATLAB_class"] = np.bytes_(b"double")
        ax_2 = handle.create_dataset("ax_2", data=np.arange(250, dtype=float).reshape(1, -1))
        ax_2.attrs["MATLAB_class"] = np.bytes_(b"double")
        ay_1 = handle.create_dataset("ay_1", data=np.arange(150, dtype=float).reshape(1, -1))
        ay_1.attrs["MATLAB_class"] = np.bytes_(b"double")
        ay_2 = handle.create_dataset("ay_2", data=np.arange(250, dtype=float).reshape(1, -1))
        ay_2.attrs["MATLAB_class"] = np.bytes_(b"double")
        az_1 = handle.create_dataset("az_1", data=np.arange(150, dtype=float).reshape(1, -1))
        az_1.attrs["MATLAB_class"] = np.bytes_(b"double")
        az_2 = handle.create_dataset("az_2", data=np.arange(250, dtype=float).reshape(1, -1))
        az_2.attrs["MATLAB_class"] = np.bytes_(b"double")

        _write_ref_vector(handle, "PPG_GREEN", [green_1, green_2])
        _write_ref_vector(handle, "PPG_AMBIENT", [ambient_1, ambient_2])
        _write_ref_vector(handle, "Accelerometer_X", [ax_1, ax_2])
        _write_ref_vector(handle, "Accelerometer_Y", [ay_1, ay_2])
        _write_ref_vector(handle, "Accelerometer_Z", [az_1, az_2])

        _write_ref_vector(
            handle,
            "recording_startday",
            [
                _write_char_dataset(handle, "day_1", "01"),
                _write_char_dataset(handle, "day_2", "02"),
            ],
        )
        _write_ref_vector(
            handle,
            "recording_starttime",
            [
                _write_char_dataset(handle, "time_1", "01:00:00"),
                _write_char_dataset(handle, "time_2", "02:00:00"),
            ],
        )

        _write_signal_header(
            handle,
            {
                "signal_labels": [
                    _write_char_dataset(handle, "sl_green", "PPG_GREEN"),
                    _write_char_dataset(handle, "sl_ambient", "PPG_AMBIENT"),
                    _write_char_dataset(handle, "sl_ax", "Accelerometer_X"),
                    _write_char_dataset(handle, "sl_ay", "Accelerometer_Y"),
                    _write_char_dataset(handle, "sl_az", "Accelerometer_Z"),
                ],
                "physical_dimension": [
                    _write_char_dataset(handle, "pd_green", "ADC unit"),
                    _write_char_dataset(handle, "pd_ambient", "ADC unit"),
                    _write_char_dataset(handle, "pd_ax", "g"),
                    _write_char_dataset(handle, "pd_ay", "g"),
                    _write_char_dataset(handle, "pd_az", "g"),
                ],
                "physical_min": [
                    _write_scalar_dataset(handle, "pmin_green", -1),
                    _write_scalar_dataset(handle, "pmin_ambient", -1),
                    _write_scalar_dataset(handle, "pmin_ax", -8),
                    _write_scalar_dataset(handle, "pmin_ay", -8),
                    _write_scalar_dataset(handle, "pmin_az", -8),
                ],
                "physical_max": [
                    _write_scalar_dataset(handle, "pmax_green", 1),
                    _write_scalar_dataset(handle, "pmax_ambient", 1),
                    _write_scalar_dataset(handle, "pmax_ax", 8),
                    _write_scalar_dataset(handle, "pmax_ay", 8),
                    _write_scalar_dataset(handle, "pmax_az", 8),
                ],
                "samples_in_record": [
                    _write_scalar_dataset(handle, "sir_green", 100),
                    _write_scalar_dataset(handle, "sir_ambient", 100),
                    _write_scalar_dataset(handle, "sir_ax", 50),
                    _write_scalar_dataset(handle, "sir_ay", 50),
                    _write_scalar_dataset(handle, "sir_az", 50),
                ],
            },
        )


def _create_subject_info_xlsx(path: Path) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    shared_strings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="6" uniqueCount="6">
  <si><t>Patient ID</t></si>
  <si><t>Age, years</t></si>
  <si><t>Unnamed: 2</t></si>
  <si><t>001</t></si>
  <si><t>002</t></si>
  <si><t>note</t></si>
</sst>"""
    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="B1" t="s"><v>1</v></c>
      <c r="C1" t="s"><v>2</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>3</v></c>
      <c r="B2"><v>67</v></c>
      <c r="C2" t="s"><v>5</v></c>
    </row>
    <row r="3">
      <c r="A3" t="s"><v>4</v></c>
      <c r="B3"><v>62</v></c>
    </row>
  </sheetData>
</worksheet>"""
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def test_ppg_loader_reads_synthetic_patient(tmp_path: Path) -> None:
    _create_ecg_file(tmp_path / "001_ECG.mat")
    _create_ppg_file(tmp_path / "001_PPG.mat")
    loader = PPGLoader(tmp_path)

    ecg = loader.read_ecg_record("1")
    ppg = loader.read_ppg_record("1")
    alignment = loader.compute_alignment("1")

    assert ecg.patient_id == "001"
    assert ecg.fs == 500
    assert ecg.n_samples == 20
    assert ecg.af_annotation.tolist() == [0, 1, 1]
    assert ppg.chunk_count == 2
    assert [chunk.n_ppg_samples for chunk in ppg.chunks] == [300, 500]
    assert loader.patient_files_complete("001") is True
    assert alignment["all_ppg_chunks_inside_ecg_window"] is False


def test_available_patient_ids_ignores_hidden_and_noncanonical_files(
    tmp_path: Path,
) -> None:
    for name in (
        "001_ECG.mat",
        "001_PPG.mat",
        "2_ECG.mat",
        "._001_ECG.mat",
        "._001_PPG.mat",
        ".hidden_ECG.mat",
        "patient003_ECG.mat",
        "notes_PPG.mat",
    ):
        (tmp_path / name).touch()

    assert PPGLoader(tmp_path).available_patient_ids() == ["001", "002"]


def test_ppg_loader_summaries_from_synthetic_patient(tmp_path: Path) -> None:
    _create_ecg_file(tmp_path / "001_ECG.mat")
    _create_ppg_file(tmp_path / "001_PPG.mat")
    loader = PPGLoader(tmp_path)

    ecg_summary = loader.summarize_ecg_af(loader.read_ecg_record("001"))
    ppg_summary = loader.summarize_ppg(loader.read_ppg_record("001"))

    assert ecg_summary["af_episode_count"] == 1
    assert ecg_summary["transition_count"] == 1
    assert ecg_summary["non_af_to_af_transitions"] == 1
    assert ecg_summary["af_to_non_af_transitions"] == 0
    assert ppg_summary["chunk_count"] == 2
    assert ppg_summary["longest_chunk_hours"] == 500 / 100 / 3600


def test_ppg_loader_resolves_blank_leading_day_cells() -> None:
    days = PPGLoader._resolve_ppg_day_indices(
        ["", "", "", "01", "02"],
        ["12:17:35", "22:08:40", "23:46:49", "22:27:47", "00:13:45"],
    )

    assert days == [0, 0, 0, 1, 2]


def test_load_subject_info_falls_back_without_openpyxl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _create_subject_info_xlsx(tmp_path / "subject_info.xlsx")
    loader = PPGLoader(tmp_path)

    def _raise_import_error(*args, **kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'")

    monkeypatch.setattr(pd, "read_excel", _raise_import_error)

    df = loader.load_subject_info()

    assert df["Patient ID"].tolist() == ["001", "002"]
    assert df["Age, years"].tolist() == [67, 62]
