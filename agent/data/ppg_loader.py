"""PPG-first loader for the AFPPGECG dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd

from agent.config import get_config


DEFAULT_PPG_DATASET_ROOT = Path("dataset/zenodo_af_monitoring_v11242869/raw")
DEFAULT_ALIGNMENT_ANCHOR = datetime(2000, 1, 1, 0, 0, 0)


class PPGDatasetNotFoundError(FileNotFoundError):
    """Raised when the requested patient file is unavailable."""


@dataclass
class RelativeTimestamp:
    """Day/time pair used by the dataset for date-shifted alignment."""

    day_index: int
    time_text: str
    anchor: datetime = DEFAULT_ALIGNMENT_ANCHOR

    @property
    def as_datetime(self) -> datetime:
        parsed = datetime.strptime(self.time_text, "%H:%M:%S").time()
        return datetime.combine(
            self.anchor.date() + timedelta(days=self.day_index - 1),
            parsed,
        )


@dataclass
class ECGReferenceRecord:
    """ECG reference recording with beat-level AF annotations."""

    patient_id: str
    start_timestamp: RelativeTimestamp
    fs: int
    n_samples: int
    qrs_index: np.ndarray
    rr_seconds: np.ndarray
    af_annotation: np.ndarray
    signal: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def start_time(self) -> datetime:
        return self.start_timestamp.as_datetime

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.fs

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.duration_seconds)


@dataclass
class PPGChunk:
    """One contiguous wrist-PPG chunk."""

    patient_id: str
    chunk_index: int
    start_timestamp: RelativeTimestamp
    ppg_fs: int
    accel_fs: int
    n_ppg_samples: int
    green_signal: np.ndarray | None = None
    ambient_signal: np.ndarray | None = None
    accel_x: np.ndarray | None = None
    accel_y: np.ndarray | None = None
    accel_z: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def start_time(self) -> datetime:
        return self.start_timestamp.as_datetime

    @property
    def duration_seconds(self) -> float:
        return self.n_ppg_samples / self.ppg_fs

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.duration_seconds)


@dataclass
class PPGRecording:
    """Patient-level wrist-PPG recording represented as multiple chunks."""

    patient_id: str
    chunks: list[PPGChunk]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def total_duration_seconds(self) -> float:
        return sum(chunk.duration_seconds for chunk in self.chunks)

    @property
    def longest_chunk_seconds(self) -> float:
        return max((chunk.duration_seconds for chunk in self.chunks), default=0.0)


class PPGLoader:
    """Read patient ECG+PPG files from the AFPPGECG dataset."""

    def __init__(self, dataset_root: Path | str | None = None):
        self.dataset_root = (
            Path(dataset_root)
            if dataset_root
            else get_config().datasets.afppgecg_root
        )

    @staticmethod
    def normalize_patient_id(patient_id: str | int) -> str:
        digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        return digits.zfill(3)

    def build_ecg_path(self, patient_id: str | int) -> Path:
        patient = self.normalize_patient_id(patient_id)
        return self.dataset_root / f"{patient}_ECG.mat"

    def build_ppg_path(self, patient_id: str | int) -> Path:
        patient = self.normalize_patient_id(patient_id)
        return self.dataset_root / f"{patient}_PPG.mat"

    def available_patient_ids(self) -> list[str]:
        ids: set[str] = set()
        for path in self.dataset_root.glob("*_ECG.mat"):
            ids.add(path.stem.split("_", 1)[0])
        for path in self.dataset_root.glob("*_PPG.mat"):
            ids.add(path.stem.split("_", 1)[0])
        return sorted(ids)

    def patient_files_complete(self, patient_id: str | int) -> bool:
        patient = self.normalize_patient_id(patient_id)
        try:
            with h5py.File(self.build_ecg_path(patient), "r"):
                pass
            with h5py.File(self.build_ppg_path(patient), "r"):
                pass
        except (OSError, FileNotFoundError):
            return False
        return True

    def load_subject_info(self) -> pd.DataFrame:
        path = self.dataset_root / "subject_info.xlsx"
        if not path.exists():
            raise PPGDatasetNotFoundError(f"Missing subject info file: {path}")
        try:
            df = pd.read_excel(path)
        except ImportError:
            df = self._read_subject_info_xlsx_fallback(path)
        df = df[df["Patient ID"].notna()].copy()
        df["Patient ID"] = df["Patient ID"].astype(int).astype(str).str.zfill(3)
        return df

    def read_ecg_record(
        self,
        patient_id: str | int,
        *,
        include_signal: bool = False,
        sampfrom: int = 0,
        sampto: int | None = None,
    ) -> ECGReferenceRecord:
        path = self.build_ecg_path(patient_id)
        if not path.exists():
            raise PPGDatasetNotFoundError(f"Missing ECG file: {path}")

        patient = self.normalize_patient_id(patient_id)
        with h5py.File(path, "r") as handle:
            header = self._read_signal_header(handle)
            start_timestamp = self._read_scalar_timestamp(
                handle=handle,
                day_dataset="recording_startday",
                time_dataset="recording_starttime",
            )
            n_samples = int(handle["ECG"].shape[0])
            signal = None
            if include_signal:
                stop = n_samples if sampto is None else min(int(sampto), n_samples)
                if sampfrom < 0 or sampfrom >= stop:
                    raise ValueError(
                        f"Invalid ECG sample range sampfrom={sampfrom}, sampto={sampto}"
                    )
                signal = np.asarray(handle["ECG"][sampfrom:stop, 0], dtype=float)

            qrs_index = np.asarray(handle["QRSindex"][()], dtype=float).reshape(-1).astype(int)
            rr_seconds = np.asarray(handle["rr"][()], dtype=float).reshape(-1)
            af_annotation = np.asarray(handle["AF_annotation"][()], dtype=float).reshape(-1).astype(int)

        return ECGReferenceRecord(
            patient_id=patient,
            start_timestamp=start_timestamp,
            fs=int(header["samples_in_record"][0]),
            n_samples=n_samples,
            qrs_index=qrs_index,
            rr_seconds=rr_seconds,
            af_annotation=af_annotation,
            signal=signal,
            metadata={
                "record_path": str(path),
                "signal_header": header,
            },
        )

    def read_ppg_record(
        self,
        patient_id: str | int,
        *,
        include_signals: bool = False,
    ) -> PPGRecording:
        path = self.build_ppg_path(patient_id)
        if not path.exists():
            raise PPGDatasetNotFoundError(f"Missing PPG file: {path}")

        patient = self.normalize_patient_id(patient_id)
        chunks: list[PPGChunk] = []
        with h5py.File(path, "r") as handle:
            header = self._read_signal_header(handle)

            green_refs = self._read_object_refs(handle["PPG_GREEN"])
            ambient_refs = self._read_object_refs(handle["PPG_AMBIENT"])
            accel_x_refs = self._read_object_refs(handle["Accelerometer_X"])
            accel_y_refs = self._read_object_refs(handle["Accelerometer_Y"])
            accel_z_refs = self._read_object_refs(handle["Accelerometer_Z"])
            start_days = self._read_char_cell_array(handle["recording_startday"])
            start_times = self._read_char_cell_array(handle["recording_starttime"])
            day_indices = self._resolve_ppg_day_indices(start_days, start_times)

            for index, (
                green_ref,
                ambient_ref,
                x_ref,
                y_ref,
                z_ref,
                day_index,
                time_text,
            ) in enumerate(
                zip(
                    green_refs,
                    ambient_refs,
                    accel_x_refs,
                    accel_y_refs,
                    accel_z_refs,
                    day_indices,
                    start_times,
                    strict=True,
                )
            ):
                green_ds = handle[green_ref]
                n_samples = int(green_ds.shape[-1])

                green_signal = None
                ambient_signal = None
                accel_x = None
                accel_y = None
                accel_z = None
                if include_signals:
                    green_signal = np.asarray(green_ds[()], dtype=float).reshape(-1)
                    ambient_signal = np.asarray(handle[ambient_ref][()], dtype=float).reshape(-1)
                    accel_x = np.asarray(handle[x_ref][()], dtype=float).reshape(-1)
                    accel_y = np.asarray(handle[y_ref][()], dtype=float).reshape(-1)
                    accel_z = np.asarray(handle[z_ref][()], dtype=float).reshape(-1)

                chunks.append(
                    PPGChunk(
                        patient_id=patient,
                        chunk_index=index,
                        start_timestamp=RelativeTimestamp(
                            day_index=day_index,
                            time_text=time_text,
                        ),
                        ppg_fs=int(header["samples_in_record"][0]),
                        accel_fs=int(header["samples_in_record"][2]),
                        n_ppg_samples=n_samples,
                        green_signal=green_signal,
                        ambient_signal=ambient_signal,
                        accel_x=accel_x,
                        accel_y=accel_y,
                        accel_z=accel_z,
                        metadata={
                            "green_shape": list(green_ds.shape),
                            "ambient_shape": list(handle[ambient_ref].shape),
                            "raw_recording_startday": start_days[index],
                        },
                    )
                )

        return PPGRecording(
            patient_id=patient,
            chunks=chunks,
            metadata={
                "record_path": str(path),
                "signal_header": header,
            },
        )

    def compute_alignment(self, patient_id: str | int) -> dict[str, Any]:
        ecg = self.read_ecg_record(patient_id, include_signal=False)
        ppg = self.read_ppg_record(patient_id, include_signals=False)

        chunk_offsets_seconds = [
            (chunk.start_time - ecg.start_time).total_seconds()
            for chunk in ppg.chunks
        ]
        chunk_overlaps_ecg = [
            chunk.start_time < ecg.end_time and chunk.end_time > ecg.start_time
            for chunk in ppg.chunks
        ]

        return {
            "patient_id": ecg.patient_id,
            "ecg_start": ecg.start_time.isoformat(sep=" "),
            "ecg_end": ecg.end_time.isoformat(sep=" "),
            "ecg_duration_seconds": ecg.duration_seconds,
            "ppg_chunk_count": ppg.chunk_count,
            "ppg_chunk_starts": [chunk.start_time.isoformat(sep=" ") for chunk in ppg.chunks],
            "ppg_chunk_ends": [chunk.end_time.isoformat(sep=" ") for chunk in ppg.chunks],
            "ppg_chunk_offset_seconds_from_ecg_start": chunk_offsets_seconds,
            "ppg_chunk_overlaps_ecg": chunk_overlaps_ecg,
            "all_ppg_chunks_overlap_ecg": all(chunk_overlaps_ecg),
            "all_ppg_chunks_inside_ecg_window": all(
                chunk.start_time >= ecg.start_time and chunk.end_time <= ecg.end_time
                for chunk in ppg.chunks
            ),
        }

    @staticmethod
    def summarize_ecg_af(record: ECGReferenceRecord) -> dict[str, Any]:
        labels = record.af_annotation.astype(int)
        rr = record.rr_seconds
        af_mask = labels == 1
        transitions = labels[1:] != labels[:-1] if len(labels) > 1 else np.array([], dtype=bool)
        non_af_to_af = (labels[:-1] == 0) & (labels[1:] == 1) if len(labels) > 1 else np.array([], dtype=bool)
        af_to_non_af = (labels[:-1] == 1) & (labels[1:] == 0) if len(labels) > 1 else np.array([], dtype=bool)
        af_episode_starts = af_mask & np.concatenate(([True], ~af_mask[:-1])) if len(labels) else np.array([], dtype=bool)

        total_rr = float(rr.sum()) if len(rr) else 0.0
        af_rr = float(rr[af_mask].sum()) if len(rr) else 0.0
        return {
            "patient_id": record.patient_id,
            "ecg_hours": record.duration_seconds / 3600.0,
            "qrs_count": int(record.qrs_index.size),
            "annotated_intervals": int(labels.size),
            "af_interval_count": int(af_mask.sum()),
            "af_hours": af_rr / 3600.0,
            "af_fraction": (af_rr / total_rr) if total_rr > 0 else 0.0,
            "af_episode_count": int(af_episode_starts.sum()),
            "transition_count": int(transitions.sum()),
            "non_af_to_af_transitions": int(non_af_to_af.sum()),
            "af_to_non_af_transitions": int(af_to_non_af.sum()),
        }

    @staticmethod
    def summarize_ppg(record: PPGRecording) -> dict[str, Any]:
        chunk_seconds = [chunk.duration_seconds for chunk in record.chunks]
        chunk_starts = [chunk.start_time.isoformat(sep=" ") for chunk in record.chunks]
        return {
            "patient_id": record.patient_id,
            "chunk_count": record.chunk_count,
            "total_hours": record.total_duration_seconds / 3600.0,
            "longest_chunk_hours": record.longest_chunk_seconds / 3600.0,
            "chunk_hours": [seconds / 3600.0 for seconds in chunk_seconds],
            "chunk_start_times": chunk_starts,
        }

    @staticmethod
    def _read_object_refs(dataset: h5py.Dataset) -> list[h5py.Reference]:
        return list(np.asarray(dataset[()]).reshape(-1))

    @classmethod
    def _read_char_cell_array(cls, dataset: h5py.Dataset) -> list[str]:
        handle = dataset.file
        values: list[str] = []
        for ref in cls._read_object_refs(dataset):
            values.append(cls._decode_char_array(handle[ref]))
        return values

    @classmethod
    def _resolve_ppg_day_indices(
        cls,
        start_days: list[str],
        start_times: list[str],
    ) -> list[int]:
        """Resolve blank PPG day cells using adjacent chunk timestamps.

        A few raw PPG files contain empty ``recording_startday`` cells for
        leading chunks. The chunks are still ordered chronologically and later
        chunks carry explicit day numbers, so infer missing days by walking
        backward from the next explicit day or forward from the previous
        resolved day.
        """

        if len(start_days) != len(start_times):
            raise ValueError("start_days and start_times must have equal length.")

        resolved: list[int | None] = []
        for day_text in start_days:
            stripped = str(day_text).strip()
            resolved.append(int(stripped) if stripped else None)

        if all(day is None for day in resolved):
            raise ValueError("No recording_startday values are available.")

        time_seconds = [cls._time_text_to_seconds(time_text) for time_text in start_times]

        # Fill leading blanks from the first explicit day, walking backward and
        # decrementing the day when clock time crosses midnight in reverse.
        first_explicit = next(index for index, day in enumerate(resolved) if day is not None)
        for index in range(first_explicit - 1, -1, -1):
            next_day = resolved[index + 1]
            if next_day is None:
                raise ValueError("Internal error while resolving PPG start days.")
            day = next_day
            if time_seconds[index] > time_seconds[index + 1]:
                day -= 1
            resolved[index] = day

        for index in range(first_explicit + 1, len(resolved)):
            if resolved[index] is not None:
                continue
            previous_day = resolved[index - 1]
            if previous_day is None:
                raise ValueError("Internal error while resolving PPG start days.")
            day = previous_day
            if time_seconds[index] < time_seconds[index - 1]:
                day += 1
            resolved[index] = day

        return [int(day) for day in resolved if day is not None]

    @staticmethod
    def _time_text_to_seconds(time_text: str) -> int:
        parsed = datetime.strptime(time_text, "%H:%M:%S").time()
        return parsed.hour * 3600 + parsed.minute * 60 + parsed.second

    @classmethod
    def _read_signal_header(cls, handle: h5py.File) -> dict[str, list[Any]]:
        group = handle["signalHeader"]
        out: dict[str, list[Any]] = {}
        for field_name in group.keys():
            refs = cls._read_object_refs(group[field_name])
            values: list[Any] = []
            for ref in refs:
                ds = handle[ref]
                if cls._is_char_dataset(ds):
                    values.append(cls._decode_char_array(ds))
                else:
                    values.append(np.asarray(ds[()]).reshape(-1).tolist()[0])
            out[field_name] = values
        return out

    @classmethod
    def _read_scalar_timestamp(
        cls,
        *,
        handle: h5py.File,
        day_dataset: str,
        time_dataset: str,
    ) -> RelativeTimestamp:
        return RelativeTimestamp(
            day_index=int(cls._decode_char_array(handle[day_dataset])),
            time_text=cls._decode_char_array(handle[time_dataset]),
        )

    @staticmethod
    def _is_char_dataset(dataset: h5py.Dataset) -> bool:
        matlab_class = dataset.attrs.get("MATLAB_class")
        return isinstance(matlab_class, (bytes, np.bytes_)) and matlab_class == b"char"

    @staticmethod
    def _decode_char_array(dataset: h5py.Dataset) -> str:
        arr = np.asarray(dataset[()]).reshape(-1)
        return "".join(chr(int(value)) for value in arr if int(value) != 0)

    @classmethod
    def _read_subject_info_xlsx_fallback(cls, path: Path) -> pd.DataFrame:
        with ZipFile(path) as archive:
            shared_strings = cls._xlsx_shared_strings(archive)
            sheet_xml = cls._xlsx_first_sheet_xml(archive)
            rows = cls._xlsx_sheet_rows(sheet_xml, shared_strings)

        if not rows:
            return pd.DataFrame()

        headers = cls._xlsx_headers(rows[0])
        records: list[list[Any]] = []
        width = len(headers)
        for row in rows[1:]:
            values = list(row)
            if len(values) < width:
                values.extend([None] * (width - len(values)))
            elif len(values) > width:
                values = values[:width]
            records.append(values)
        return pd.DataFrame(records, columns=headers)

    @staticmethod
    def _xlsx_shared_strings(archive: ZipFile) -> list[str]:
        shared_strings_path = "xl/sharedStrings.xml"
        if shared_strings_path not in archive.namelist():
            return []

        root = ET.fromstring(archive.read(shared_strings_path))
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        strings: list[str] = []
        for item in root.findall("main:si", namespace):
            text_parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
            strings.append("".join(text_parts))
        return strings

    @staticmethod
    def _xlsx_first_sheet_xml(archive: ZipFile) -> bytes:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        workbook_ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}

        first_sheet = workbook.find("main:sheets/main:sheet", workbook_ns)
        if first_sheet is None:
            raise ValueError("Workbook contains no sheets.")
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if rel_id is None:
            raise ValueError("Workbook sheet is missing relationship id.")

        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("rel:Relationship", rel_ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if target is None:
            raise ValueError(f"Unable to resolve worksheet target for relationship {rel_id}.")
        return archive.read(f"xl/{target}")

    @classmethod
    def _xlsx_sheet_rows(cls, sheet_xml: bytes, shared_strings: list[str]) -> list[list[Any]]:
        root = ET.fromstring(sheet_xml)
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[list[Any]] = []

        for row in root.findall("main:sheetData/main:row", namespace):
            cells: dict[int, Any] = {}
            max_col = -1
            for cell in row.findall("main:c", namespace):
                ref = cell.attrib.get("r", "")
                col_idx = cls._xlsx_column_index(ref)
                max_col = max(max_col, col_idx)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", namespace)
                inline_node = cell.find("main:is/main:t", namespace)

                if cell_type == "s" and value_node is not None and value_node.text is not None:
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None and value_node.text is not None:
                    value = cls._xlsx_numeric_or_text(value_node.text)
                else:
                    value = None
                cells[col_idx] = value

            if max_col < 0:
                rows.append([])
                continue
            row_values = [None] * (max_col + 1)
            for idx, value in cells.items():
                row_values[idx] = value
            rows.append(row_values)
        return rows

    @staticmethod
    def _xlsx_headers(raw_headers: list[Any]) -> list[str]:
        headers: list[str] = []
        for idx, value in enumerate(raw_headers):
            text = "" if value is None else str(value).strip()
            headers.append(text if text else f"Unnamed: {idx}")
        return headers

    @staticmethod
    def _xlsx_column_index(cell_ref: str) -> int:
        match = re.match(r"([A-Z]+)", cell_ref)
        if not match:
            return 0
        letters = match.group(1)
        index = 0
        for letter in letters:
            index = index * 26 + (ord(letter) - ord("A") + 1)
        return index - 1

    @staticmethod
    def _xlsx_numeric_or_text(value: str) -> Any:
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number


# Backward-compatible aliases while older scripts migrate to the PPG-first API.
DEFAULT_AFPPGECG_ROOT = DEFAULT_PPG_DATASET_ROOT
AFPPGECGNotFoundError = PPGDatasetNotFoundError
ECGAFRecord = ECGReferenceRecord
AFPPGECGLoader = PPGLoader
