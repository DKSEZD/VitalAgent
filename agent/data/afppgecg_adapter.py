"""ECG segment adapter for the AFPPGECG dataset hosted on Zenodo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from agent.data.ppg_loader import PPGLoader

logger = logging.getLogger(__name__)

AFPPGECG_ECG_SEGMENT_ID = "ecg"


@dataclass(frozen=True)
class AFPPGECGRhythmChangeAnnotation:
    """WFDB-like rhythm change markers derived from beat-level AF labels."""

    sample: np.ndarray
    symbol: list[str]
    aux_note: list[str]


@dataclass
class AFPPGECGRhythmAnnotation:
    """Rhythm spans relative to the returned signal window."""

    start_samples: list[int]
    end_samples: list[int]
    rhythm_types: list[str]


@dataclass
class AFPPGECGRawECGData:
    """Loader output for one AFPPGECG ECG slice."""

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
    beat_annotations: object | None = None
    rhythm_annotations: AFPPGECGRhythmAnnotation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AFPPGECGSegmentInfo:
    """Header-only metadata for the single virtual AFPPGECG ECG segment."""

    patient_id: str
    segment_id: str
    n_samples: int
    duration_seconds: float
    fs: int
    start_time: datetime | None = None


class AFPPGECGSegmentLoader:
    """
    Wrap ``PPGLoader`` as an ``ECGSegmentReader`` for proactive ECG streaming.

    Each AFPPGECG subject has one continuous ECG recording, exposed here as a
    single virtual segment named ``"ecg"``.
    """

    def __init__(
        self,
        dataset_root: Path | str | None = None,
        ppg_loader: PPGLoader | None = None,
    ) -> None:
        if ppg_loader is not None:
            self.ppg_loader = ppg_loader
        else:
            self.ppg_loader = PPGLoader(dataset_root)
        self.dataset_root = self.ppg_loader.dataset_root

    @staticmethod
    def normalize_patient_id(value: str | int) -> str:
        """Normalize AFPPGECG subject ids to three digits."""
        return PPGLoader.normalize_patient_id(value)

    @staticmethod
    def normalize_segment_id(value: str | int) -> str:
        """Normalize accepted AFPPGECG virtual segment aliases."""
        text = str(value).strip().lower()
        digits = "".join(ch for ch in text if ch.isdigit())
        if text == AFPPGECG_ECG_SEGMENT_ID or digits in {"0", "00", "000"}:
            return AFPPGECG_ECG_SEGMENT_ID
        raise ValueError(
            f"Invalid AFPPGECG ECG segment_id: {value!r}; expected "
            f"{AFPPGECG_ECG_SEGMENT_ID!r}."
        )

    def available_patient_ids(self) -> list[str]:
        """Return patient ids with an ECG MAT file under the dataset root."""
        ids = {
            path.stem.split("_", 1)[0]
            for path in self.dataset_root.glob("*_ECG.mat")
            if path.is_file()
        }
        return sorted(ids)

    def get_patient_segments(self, patient_id: str) -> list[AFPPGECGSegmentInfo]:
        """Return the single virtual ECG segment for a patient."""
        patient = self.normalize_patient_id(patient_id)
        record = self.ppg_loader.read_ecg_record(patient, include_signal=False)
        return [
            AFPPGECGSegmentInfo(
                patient_id=patient,
                segment_id=AFPPGECG_ECG_SEGMENT_ID,
                n_samples=int(record.n_samples),
                duration_seconds=float(record.duration_seconds),
                fs=int(record.fs),
                start_time=record.start_time,
            )
        ]

    def read_segment(
        self,
        patient_id: str,
        segment_id: str,
        sampfrom: int = 0,
        sampto: int | None = None,
    ) -> AFPPGECGRawECGData:
        """Read a signal slice and matching rhythm spans for the ECG segment."""
        if sampfrom < 0:
            raise ValueError(f"sampfrom must be >= 0, got {sampfrom}")

        patient = self.normalize_patient_id(patient_id)
        segment = self.normalize_segment_id(segment_id)
        record = self.ppg_loader.read_ecg_record(
            patient,
            include_signal=True,
            sampfrom=sampfrom,
            sampto=sampto,
        )
        if record.signal is None:
            raise RuntimeError("PPGLoader returned no ECG signal despite include_signal=True.")

        signal = np.asarray(record.signal, dtype=float).reshape(-1)
        actual_sampto = int(sampfrom) + int(signal.shape[0])
        rhythm_annotations = build_afppgecg_rhythm_annotation_spans(
            qrs_index=record.qrs_index,
            af_annotation=record.af_annotation,
            segment_total_samples=record.n_samples,
            sampfrom=sampfrom,
            sampto=actual_sampto,
        )

        metadata = dict(record.metadata)
        metadata.update(
            {
                "dataset": "afppgecg",
                "source": "local",
                "qrs_count": int(np.asarray(record.qrs_index).size),
                "af_annotation_count": int(np.asarray(record.af_annotation).size),
            }
        )

        window_start_time = record.start_time + timedelta(seconds=sampfrom / record.fs)
        return AFPPGECGRawECGData(
            signal=signal,
            fs=int(record.fs),
            patient_id=patient,
            segment_id=segment,
            n_samples=int(signal.shape[0]),
            sample_start=int(sampfrom),
            sample_end=actual_sampto,
            segment_total_samples=int(record.n_samples),
            segment_start_time=record.start_time,
            window_start_time=window_start_time,
            window_duration_seconds=float(signal.shape[0]) / float(record.fs),
            rhythm_annotations=rhythm_annotations,
            metadata=metadata,
        )


def build_afppgecg_rhythm_change_annotation(
    qrs_index: np.ndarray,
    af_annotation: np.ndarray,
) -> AFPPGECGRhythmChangeAnnotation | None:
    """
    Convert AFPPGECG beat-level AF labels to WFDB-like rhythm change markers.

    The first marker is emitted at sample 0. Subsequent markers are emitted at
    ``qrs_index[i]`` whenever ``af_annotation[i]`` changes from the previous
    label. AF is encoded as ``"(AFIB"`` and non-AF as ``"(N"``.
    """
    qrs, labels = _coerce_aligned_qrs_and_labels(qrs_index, af_annotation)
    if labels.size == 0:
        return None

    samples = [0]
    aux_notes = [_label_to_aux_note(bool(labels[0]))]

    for index in range(1, labels.size):
        if bool(labels[index]) == bool(labels[index - 1]):
            continue
        samples.append(int(qrs[index]))
        aux_notes.append(_label_to_aux_note(bool(labels[index])))

    return AFPPGECGRhythmChangeAnnotation(
        sample=np.asarray(samples, dtype=int),
        symbol=["+"] * len(samples),
        aux_note=aux_notes,
    )


def build_afppgecg_rhythm_annotation_spans(
    *,
    qrs_index: np.ndarray,
    af_annotation: np.ndarray,
    segment_total_samples: int,
    sampfrom: int = 0,
    sampto: int | None = None,
) -> AFPPGECGRhythmAnnotation | None:
    """Build clipped rhythm spans compatible with ``ECGStreamer``."""
    total_samples = int(segment_total_samples)
    if total_samples <= 0:
        return None

    window_start = int(sampfrom)
    window_end = total_samples if sampto is None else min(int(sampto), total_samples)
    if window_end <= window_start:
        raise ValueError(
            f"Invalid sample range sampfrom={window_start}, sampto={window_end}"
        )

    changes = build_afppgecg_rhythm_change_annotation(qrs_index, af_annotation)
    if changes is None:
        return None

    change_samples = [int(sample) for sample in changes.sample]
    change_labels = [
        _aux_note_to_rhythm_type(note)
        for note in changes.aux_note
    ]

    starts: list[int] = []
    ends: list[int] = []
    rhythm_types: list[str] = []
    for index, (span_start, rhythm_type) in enumerate(
        zip(change_samples, change_labels, strict=True)
    ):
        span_end = (
            change_samples[index + 1]
            if index + 1 < len(change_samples)
            else total_samples
        )
        clipped_start = max(span_start, window_start)
        clipped_end = min(span_end, window_end)
        if clipped_end <= clipped_start:
            continue
        starts.append(clipped_start - window_start)
        ends.append(clipped_end - window_start)
        rhythm_types.append(rhythm_type)

    if not starts:
        return None

    return AFPPGECGRhythmAnnotation(
        start_samples=starts,
        end_samples=ends,
        rhythm_types=rhythm_types,
    )


def _coerce_aligned_qrs_and_labels(
    qrs_index: np.ndarray,
    af_annotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qrs = np.asarray(qrs_index, dtype=int).reshape(-1)
    labels = np.asarray(af_annotation, dtype=bool).reshape(-1)
    usable = min(qrs.size, labels.size)
    if usable == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=bool)

    qrs = qrs[:usable]
    labels = labels[:usable]
    order = np.argsort(qrs, kind="stable")
    return qrs[order], labels[order]


def _label_to_aux_note(is_af: bool) -> str:
    return "(AFIB" if is_af else "(N"


def _aux_note_to_rhythm_type(aux_note: str) -> str:
    token = str(aux_note).strip()
    if token.startswith("("):
        token = token[1:]
    return "AFIB" if token.upper() == "AFIB" else "N"
