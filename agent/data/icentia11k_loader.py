"""
Icentia11k ECG data loader.

Reads single-lead continuous ECG from Icentia11k, either from a local mirror
or directly from PhysioNet via WFDB's ``pn_dir`` support.

Dataset layout:
    icentia11k-continuous-ecg/1.0/p{id[:2]}/p{full_id}/p{full_id}_s{seg}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from agent.data.single_lead import coerce_single_lead_signal

logger = logging.getLogger(__name__)

PHYSIONET_DB = "icentia11k-continuous-ecg/1.0"
FS = 250


@dataclass
class BeatAnnotation:
    """Beat-level annotations relative to the returned signal window."""

    samples: np.ndarray
    symbols: list[str]


@dataclass
class RhythmAnnotation:
    """Rhythm spans relative to the returned signal window.

    Spans use half-open sample intervals: [start_sample, end_sample).
    """

    start_samples: list[int]
    end_samples: list[int]
    rhythm_types: list[str]


@dataclass
class RawECGData:
    """Loader output for a single signal window."""

    signal: np.ndarray
    fs: int
    patient_id: str
    segment_id: str
    n_samples: int
    sample_start: int
    sample_end: int
    segment_total_samples: int
    segment_start_time: datetime | None = None
    window_start_time: datetime | None = None
    window_duration_seconds: float = 0.0
    beat_annotations: BeatAnnotation | None = None
    rhythm_annotations: RhythmAnnotation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentInfo:
    """Header-only metadata for a segment."""

    patient_id: str
    segment_id: str
    n_samples: int
    duration_seconds: float
    fs: int
    start_time: datetime | None = None


class Icentia11kLoader:
    """
    Remote (PhysioNet) or local reader for Icentia11k.

    WFDB calls used:
        wfdb.rdheader(record_name, pn_dir=...) -> header only
        wfdb.rdsamp(record_name, pn_dir=..., sampfrom=, sampto=...) -> signal
        wfdb.rdann(record_name, "atr", pn_dir=..., sampfrom=, sampto=...) -> annotations
    """

    def __init__(self, local_root: Path | str | None = None):
        self.local_root = Path(local_root).expanduser() if local_root else None
        self.remote = self.local_root is None

    @staticmethod
    def _relative_record_path(patient_id: str | int, segment_id: str | int) -> str:
        patient = Icentia11kLoader.normalize_patient_id(patient_id)
        segment = Icentia11kLoader.normalize_segment_id(segment_id)
        return f"p{patient[:2]}/p{patient}/p{patient}_s{segment}"

    @staticmethod
    def normalize_patient_id(patient_id: str | int) -> str:
        digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        return digits.zfill(5)

    @staticmethod
    def normalize_segment_id(segment_id: str | int) -> str:
        digits = "".join(ch for ch in str(segment_id) if ch.isdigit())
        if not digits:
            raise ValueError(f"Invalid segment_id: {segment_id!r}")
        return digits.zfill(2)

    def build_record_name(self, patient_id: str | int, segment_id: str | int) -> str:
        patient = self.normalize_patient_id(patient_id)
        segment = self.normalize_segment_id(segment_id)
        relpath = self._relative_record_path(patient, segment)
        if self.remote:
            return relpath
        if self.local_root is None:
            raise RuntimeError("local_root is required for local record access.")
        return str(self.local_root / relpath)

    def _wfdb_record_name(
        self,
        patient_id: str | int,
        segment_id: str | int,
        source: str,
    ) -> str:
        patient = self.normalize_patient_id(patient_id)
        segment = self.normalize_segment_id(segment_id)
        if source == "remote":
            return f"p{patient}_s{segment}"
        return self.build_record_name(patient, segment)

    def _wfdb_kwargs(
        self,
        patient_id: str | int,
        segment_id: str | int,
        source: str,
    ) -> dict[str, Any]:
        if source != "remote":
            return {}
        patient = self.normalize_patient_id(patient_id)
        return {"pn_dir": f"{PHYSIONET_DB}/p{patient[:2]}/p{patient}"}

    def _access_modes(self, preferred: str | None = None) -> tuple[str, ...]:
        if self.local_root is None:
            return ("remote",)
        modes = ["local", "remote"]
        if preferred == "remote":
            modes.reverse()
        return tuple(modes)

    def _record_name_for_source(
        self,
        patient_id: str | int,
        segment_id: str | int,
        source: str,
    ) -> str:
        patient = self.normalize_patient_id(patient_id)
        segment = self.normalize_segment_id(segment_id)
        if source == "remote":
            return self._relative_record_path(patient, segment)
        return self.build_record_name(patient, segment)

    def _read_header(self, patient_id: str | int, segment_id: str | int) -> tuple[Any, str]:
        last_exc: Exception | None = None
        for source in self._access_modes():
            record_name = self._wfdb_record_name(patient_id, segment_id, source)
            try:
                return (
                    wfdb.rdheader(record_name, **self._wfdb_kwargs(patient_id, segment_id, source)),
                    source,
                )
            except Exception as exc:
                if source == "local" and self._is_missing_record_error(exc):
                    logger.info(
                        "Local Icentia11k record missing for patient %s segment %s; "
                        "falling back to PhysioNet.",
                        patient_id,
                        segment_id,
                    )
                    last_exc = exc
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise FileNotFoundError(f"Unable to read header for {patient_id=} {segment_id=}")

    def _read_signal(
        self,
        patient_id: str,
        segment_id: str,
        sampfrom: int,
        sampto: int,
        preferred_source: str,
    ) -> tuple[tuple[np.ndarray, dict[str, Any]], str]:
        last_exc: Exception | None = None
        for source in self._access_modes(preferred_source):
            record_name = self._wfdb_record_name(patient_id, segment_id, source)
            try:
                return (
                    wfdb.rdsamp(
                        record_name,
                        sampfrom=sampfrom,
                        sampto=sampto,
                        **self._wfdb_kwargs(patient_id, segment_id, source),
                    ),
                    source,
                )
            except Exception as exc:
                if source == "local" and self._is_missing_record_error(exc):
                    logger.info(
                        "Local Icentia11k signal missing for patient %s segment %s; "
                        "falling back to PhysioNet.",
                        patient_id,
                        segment_id,
                    )
                    last_exc = exc
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise FileNotFoundError(f"Unable to read signal for {patient_id=} {segment_id=}")

    def _read_annotation(
        self,
        patient_id: str,
        segment_id: str,
        sampfrom: int,
        sampto: int,
        preferred_source: str,
    ) -> tuple[Any | None, str | None]:
        last_exc: Exception | None = None
        for source in self._access_modes(preferred_source):
            record_name = self._wfdb_record_name(patient_id, segment_id, source)
            try:
                return (
                    wfdb.rdann(
                        record_name,
                        "atr",
                        sampfrom=sampfrom,
                        sampto=sampto,
                        **self._wfdb_kwargs(patient_id, segment_id, source),
                    ),
                    source,
                )
            except Exception as exc:
                if self._is_missing_record_error(exc):
                    if source == "local":
                        logger.info(
                            "Local Icentia11k annotations missing for patient %s segment %s; "
                            "falling back to PhysioNet.",
                            patient_id,
                            segment_id,
                        )
                        last_exc = exc
                        continue
                    last_exc = exc
                    break
                raise
        if last_exc is not None:
            logger.warning(
                "Annotations missing for patient %s segment %s: %s",
                patient_id,
                segment_id,
                last_exc,
            )
        return None, None

    def read_segment(
        self,
        patient_id: str,
        segment_id: str,
        sampfrom: int = 0,
        sampto: int | None = None,
    ) -> RawECGData:
        """Read signal and annotations for one segment or signal slice."""
        if sampfrom < 0:
            raise ValueError(f"sampfrom must be >= 0, got {sampfrom}")

        patient = self.normalize_patient_id(patient_id)
        segment = self.normalize_segment_id(segment_id)
        header, header_source = self._read_header(patient, segment)
        segment_total_samples = int(header.sig_len)
        fs = int(header.fs)
        segment_start_time = self._extract_start_datetime(header)

        actual_sampto = segment_total_samples if sampto is None else min(int(sampto), segment_total_samples)
        if actual_sampto <= sampfrom:
            raise ValueError(
                f"Invalid sample range for patient {patient} segment {segment}: "
                f"sampfrom={sampfrom}, sampto={actual_sampto}, total={segment_total_samples}"
            )

        (signal, fields), signal_source = self._read_signal(
            patient,
            segment,
            sampfrom=sampfrom,
            sampto=actual_sampto,
            preferred_source=header_source,
        )
        signal_1d = self._coerce_single_lead(signal)

        beat_annotations: BeatAnnotation | None = None
        rhythm_annotations: RhythmAnnotation | None = None
        annotation, _annotation_source = self._read_annotation(
            patient,
            segment,
            sampfrom=sampfrom,
            sampto=actual_sampto,
            preferred_source=signal_source,
        )
        if annotation is not None:
            beat_annotations = self._parse_beat_annotations(annotation, sampfrom)
            rhythm_annotations = self._parse_rhythm_annotations(
                annotation,
                sampfrom=sampfrom,
                window_n_samples=signal_1d.shape[0],
            )

        metadata = {
            "record_name": self._record_name_for_source(patient, segment, signal_source),
            "source": "physionet" if signal_source == "remote" else "local",
            "lead_names": list(fields.get("sig_name", [])),
            "units": list(fields.get("units", [])),
        }
        window_start_time = None
        if segment_start_time is not None:
            window_start_time = segment_start_time + timedelta(seconds=sampfrom / fs)

        return RawECGData(
            signal=signal_1d,
            fs=fs,
            patient_id=patient,
            segment_id=segment,
            n_samples=signal_1d.shape[0],
            sample_start=sampfrom,
            sample_end=actual_sampto,
            segment_total_samples=segment_total_samples,
            segment_start_time=segment_start_time,
            window_start_time=window_start_time,
            window_duration_seconds=signal_1d.shape[0] / fs,
            beat_annotations=beat_annotations,
            rhythm_annotations=rhythm_annotations,
            metadata=metadata,
        )

    def read_segment_header(self, patient_id: str, segment_id: str) -> SegmentInfo:
        """Read header only without loading signal samples."""
        patient = self.normalize_patient_id(patient_id)
        segment = self.normalize_segment_id(segment_id)
        header, _source = self._read_header(patient, segment)
        n_samples = int(header.sig_len)
        fs = int(header.fs)
        return SegmentInfo(
            patient_id=patient,
            segment_id=segment,
            n_samples=n_samples,
            duration_seconds=n_samples / fs,
            fs=fs,
            start_time=self._extract_start_datetime(header),
        )

    def get_patient_segments(self, patient_id: str) -> list[SegmentInfo]:
        """Enumerate segment ids s00, s01, s02... until a segment is missing."""
        patient = self.normalize_patient_id(patient_id)
        segments: list[SegmentInfo] = []
        segment_idx = 0

        while True:
            segment = f"{segment_idx:02d}"
            try:
                segments.append(self.read_segment_header(patient, segment))
            except Exception as exc:
                if self._is_missing_record_error(exc):
                    break
                raise
            segment_idx += 1

        return segments

    @staticmethod
    def _coerce_single_lead(signal: np.ndarray) -> np.ndarray:
        try:
            return coerce_single_lead_signal(signal)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _clean_aux_note(note: str | None) -> str:
        if note is None:
            return ""
        return note.replace("\x00", "").strip()

    @staticmethod
    def _extract_start_datetime(header: Any) -> datetime | None:
        base_datetime = getattr(header, "base_datetime", None)
        if isinstance(base_datetime, datetime):
            return base_datetime

        base_date = getattr(header, "base_date", None)
        base_time = getattr(header, "base_time", None)

        if isinstance(base_date, datetime):
            return base_date
        if isinstance(base_date, date) and isinstance(base_time, time):
            return datetime.combine(base_date, base_time)
        return None

    @classmethod
    def _parse_beat_annotations(cls, annotation: Any, sampfrom: int) -> BeatAnnotation | None:
        beat_samples: list[int] = []
        beat_symbols: list[str] = []

        for sample, symbol, _aux_note in cls._iter_annotations(annotation):
            if symbol == "+" or not symbol:
                continue
            beat_samples.append(sample - sampfrom)
            beat_symbols.append(symbol)

        if not beat_samples:
            return None

        return BeatAnnotation(samples=np.asarray(beat_samples, dtype=int), symbols=beat_symbols)

    @classmethod
    def _parse_rhythm_annotations(
        cls,
        annotation: Any,
        sampfrom: int,
        window_n_samples: int,
    ) -> RhythmAnnotation | None:
        """Parse rhythm annotations into half-open local spans.

        Output spans are relative to the returned RawECGData signal window and
        use half-open intervals: [start_sample, end_sample).

        Icentia/WFDB rhythm markers may include:
        - start markers such as "(N", "(AFIB", "(AFL"
        - end markers such as ")" or "AFIB)"

        If a new rhythm start appears before the previous rhythm is explicitly
        closed, the previous rhythm is closed at the new start sample.
        If a rhythm remains open at the end of the returned signal window, it is
        clipped to window_n_samples.
        """

        intervals: list[tuple[int, int, str]] = []

        active_label: str | None = None
        active_start: int | None = None

        # _iter_annotations() preserves the original zip order. Sort stably by
        # sample so co-located markers keep deterministic input order.
        indexed_annotations = list(enumerate(cls._iter_annotations(annotation)))
        indexed_annotations.sort(key=lambda item: (item[1][0], item[0]))

        for _, (sample, _symbol, aux_note) in indexed_annotations:
            note = cls._clean_aux_note(aux_note)
            if not note:
                continue

            local_sample = int(sample) - int(sampfrom)

            if note.startswith("("):
                new_label = note[1:].strip()
                if not new_label:
                    continue

                # Clip starts to the returned signal window. In normal rdann
                # sliced reads, local_sample should already be within range,
                # but clipping keeps the function robust for mocked tests.
                if local_sample < 0:
                    local_start = 0
                elif local_sample >= window_n_samples:
                    # A start after the returned window cannot affect this raw
                    # slice. If an active label exists, it remains open until
                    # the window end.
                    continue
                else:
                    local_start = local_sample

                if active_label is not None and active_start is not None:
                    if local_start > active_start:
                        intervals.append((active_start, local_start, active_label))

                active_label = new_label
                active_start = local_start

            elif note.endswith(")"):
                if active_label is None or active_start is None:
                    continue

                # End markers outside the left edge close at 0; end markers
                # outside the right edge close at window_n_samples.
                local_end = min(max(local_sample, 0), window_n_samples)

                if local_end > active_start:
                    intervals.append((active_start, local_end, active_label))

                active_label = None
                active_start = None

        if active_label is not None and active_start is not None:
            if window_n_samples > active_start:
                intervals.append((active_start, window_n_samples, active_label))

        if not intervals:
            return None

        return RhythmAnnotation(
            start_samples=[start for start, _end, _label in intervals],
            end_samples=[end for _start, end, _label in intervals],
            rhythm_types=[label for _start, _end, label in intervals],
        )

    @staticmethod
    def _iter_annotations(annotation: Any) -> list[tuple[int, str, str]]:
        samples = np.asarray(getattr(annotation, "sample", []), dtype=int)
        symbols = list(getattr(annotation, "symbol", []))
        aux_notes = list(getattr(annotation, "aux_note", []))
        return [
            (int(sample), symbol or "", aux_note or "")
            for sample, symbol, aux_note in zip_longest(samples, symbols, aux_notes, fillvalue="")
        ]

    @staticmethod
    def _is_missing_record_error(exc: Exception) -> bool:
        message = str(exc).lower()
        missing_markers = (
            "no such file",
            "not found",
            "404",
            "could not find",
            "pn_dir",
        )
        return isinstance(exc, FileNotFoundError) or any(marker in message for marker in missing_markers)


