"""
Streaming utilities — ECG and PPG.

Dataset-agnostic: converts loader output into fixed-size windows for the
proactive pipeline. Both ``ECGStreamer`` and ``PPGStreamer`` yield window
objects whose ``.signal`` / ``.fs`` / ``.n_samples`` fields are compatible
with ``ProactivePipeline.process_window`` via duck-typing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Iterator, Protocol, Sequence

import numpy as np

from agent.data.ppg_window_labels import format_ppg_window_id

if TYPE_CHECKING:
    from agent.data.ppg_loader import PPGChunk, PPGRecording

logger = logging.getLogger(__name__)


@dataclass
class RhythmSpan:
    """Rhythm annotation span relative to a single window."""

    start_sample: int
    end_sample: int
    rhythm_type: str


@dataclass
class WindowAnnotations:
    """Annotations scoped to a single window."""

    beat_samples: np.ndarray | None = None
    beat_symbols: list[str] | None = None
    rhythm_spans: list[RhythmSpan] = field(default_factory=list)


@dataclass
class ECGWindow:
    """
    Unit consumed by ``ProactivePipeline.process_segment()``.

    The proactive path is single-lead in v1, so ``signal`` is always 1-D.
    """

    signal: np.ndarray
    fs: int
    patient_id: str
    segment_id: str
    window_index: int
    start_sample: int
    end_sample: int
    n_samples: int
    stream_offset_s: float
    window_start_time: datetime | None = None
    annotations: WindowAnnotations | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class PPGWindow:
    """
    PPG window — duck-typed to ``ECGWindow`` for ``ProactivePipeline`` compat.

    ``signal`` is always the green channel (1-D, ADC units).  Ambient and
    accelerometer arrays are carried as optional extras for downstream
    quality assessment inside ``PPGSignalProcessor``.
    """

    signal: np.ndarray          # green PPG channel, shape (n_samples,)
    fs: int                     # PPG sampling rate (typically 100 Hz)
    patient_id: str
    chunk_index: int            # source chunk within the recording
    window_index: int
    start_sample: int           # sample offset within the source chunk
    end_sample: int
    n_samples: int
    stream_offset_s: float      # seconds from recording start
    window_start_time: datetime | None = None
    ambient_signal: np.ndarray | None = None
    accel: np.ndarray | None = None  # shape (n_accel_samples, 3) — XYZ
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def segment_id(self) -> str:
        """Synthetic segment_id so duck-typing with ECGWindow works."""
        return f"chunk{self.chunk_index:03d}"

    @property
    def window_id(self) -> str:
        """Stable identifier for this window within a patient recording."""
        return format_ppg_window_id(
            patient_id=self.patient_id,
            chunk_index=self.chunk_index,
            window_index=self.window_index,
            start_sample=self.start_sample,
            end_sample=self.end_sample,
        )


class PPGStreamer:
    """Cuts PPG chunks from a recording into fixed-size windows."""

    def __init__(
        self,
        window_seconds: float = 10.0,
        stride_seconds: float | None = None,
        drop_last: bool = True,
    ) -> None:
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds if stride_seconds is not None else window_seconds
        self.drop_last = drop_last

    def stream_recording(self, recording: PPGRecording) -> Iterator[PPGWindow]:
        """Stream every chunk of a ``PPGRecording`` as sequential windows."""
        global_index = 0
        offset_s = 0.0
        for chunk in recording.chunks:
            for window in self._stream_chunk(
                chunk,
                window_index_offset=global_index,
                stream_offset_base_s=offset_s,
            ):
                global_index += 1
                yield window
            offset_s += chunk.duration_seconds

    def _stream_chunk(
        self,
        chunk: PPGChunk,
        window_index_offset: int = 0,
        stream_offset_base_s: float = 0.0,
    ) -> Iterator[PPGWindow]:
        """Slide a fixed window over one PPG chunk."""
        if chunk.green_signal is None:
            raise ValueError(
                f"PPGChunk {chunk.chunk_index} has no green signal — "
                "reload with include_signals=True."
            )

        fs = chunk.ppg_fs
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {chunk.ppg_fs!r}")
        window_samples = self._seconds_to_samples(self.window_seconds, fs, "window_seconds")
        stride_samples = self._seconds_to_samples(self.stride_seconds, fs, "stride_seconds")
        total_samples = chunk.n_ppg_samples

        # Ratio for downsampled accelerometer (typically ppg_fs/accel_fs = 2)
        accel_ratio = chunk.accel_fs / chunk.ppg_fs if chunk.ppg_fs > 0 else 0.0
        has_accel = (
            accel_ratio > 0
            and chunk.accel_x is not None
            and chunk.accel_y is not None
            and chunk.accel_z is not None
        )

        window_count = 0
        for local_start in range(0, total_samples, stride_samples):
            local_end = min(local_start + window_samples, total_samples)
            window_len = local_end - local_start

            if window_len <= 0:
                break
            if window_len < window_samples and self.drop_last:
                break

            green = chunk.green_signal[local_start:local_end]
            ambient = (
                chunk.ambient_signal[local_start:local_end]
                if chunk.ambient_signal is not None
                else None
            )

            accel = None
            if has_accel:
                a_start = int(local_start * accel_ratio)
                a_end = int(local_end * accel_ratio)
                ax = chunk.accel_x[a_start:a_end]   # type: ignore[index]
                ay = chunk.accel_y[a_start:a_end]   # type: ignore[index]
                az = chunk.accel_z[a_start:a_end]   # type: ignore[index]
                n = min(len(ax), len(ay), len(az))
                accel = np.stack([ax[:n], ay[:n], az[:n]], axis=1)

            window_start_time = None
            if chunk.start_time is not None:
                window_start_time = chunk.start_time + timedelta(
                    seconds=local_start / fs
                )

            yield PPGWindow(
                signal=green,
                fs=fs,
                patient_id=chunk.patient_id,
                chunk_index=chunk.chunk_index,
                window_index=window_index_offset + window_count,
                start_sample=local_start,
                end_sample=local_end,
                n_samples=window_len,
                stream_offset_s=stream_offset_base_s + (local_start / fs),
                window_start_time=window_start_time,
                ambient_signal=ambient,
                accel=accel,
                metadata={"chunk_index": chunk.chunk_index, "ppg_fs": fs},
            )
            window_count += 1

    @staticmethod
    def _seconds_to_samples(value_seconds: float, fs: int, field_name: str) -> int:
        samples = int(round(float(value_seconds) * fs))
        if samples <= 0:
            raise ValueError(
                f"{field_name} must yield at least 1 sample, got {value_seconds!r} at fs={fs}."
            )
        return samples


class SegmentInfoLike(Protocol):
    """Minimal segment metadata required by ``ECGStreamer``."""

    patient_id: str
    segment_id: str
    n_samples: int
    duration_seconds: float
    fs: int
    start_time: datetime | None


class RawECGDataLike(Protocol):
    """Minimal loader output required by ``stream_segment``."""

    signal: np.ndarray
    fs: int
    patient_id: str
    segment_id: str
    n_samples: int
    sample_start: int
    sample_end: int
    segment_total_samples: int
    segment_start_time: datetime | None
    window_start_time: datetime | None
    metadata: dict[str, object]
    beat_annotations: object | None
    rhythm_annotations: object | None


class ECGSegmentReader(Protocol):
    """Minimal loader protocol for patient-level streaming."""

    def read_segment(
        self,
        patient_id: str,
        segment_id: str,
        sampfrom: int = 0,
        sampto: int | None = None,
    ) -> RawECGDataLike: ...

    def get_patient_segments(self, patient_id: str) -> list[SegmentInfoLike]: ...


class ECGStreamer:
    """Cuts a single-lead raw ECG segment into sequential windows."""

    def __init__(
        self,
        window_seconds: float = 10.0,
        stride_seconds: float | None = None,
        drop_last: bool = True,
    ):
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds if stride_seconds is not None else window_seconds
        self.drop_last = drop_last

    def stream_segment(
        self,
        raw: RawECGDataLike,
        segment_time_offset_s: float = 0.0,
        window_index_offset: int = 0,
    ) -> Iterator[ECGWindow]:
        """Slide a fixed window over a single loader output object."""
        signal = np.asarray(raw.signal)
        if signal.ndim != 1:
            raise ValueError(
                f"ECGStreamer expects a 1-D single-lead signal, got shape {signal.shape!r}"
            )

        fs = int(raw.fs)
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {raw.fs!r}")

        window_samples = self._seconds_to_samples(self.window_seconds, fs, "window_seconds")
        stride_samples = self._seconds_to_samples(self.stride_seconds, fs, "stride_seconds")
        total_samples = int(raw.n_samples)

        window_count = 0
        for local_start in range(0, total_samples, stride_samples):
            local_end = min(local_start + window_samples, total_samples)
            window_len = local_end - local_start

            if window_len <= 0:
                break
            if window_len < window_samples and self.drop_last:
                break

            annotations = self._slice_annotations(raw, local_start, local_end)
            start_sample = int(raw.sample_start) + local_start
            end_sample = start_sample + window_len
            window_start_time = None
            if raw.window_start_time is not None:
                window_start_time = raw.window_start_time + timedelta(seconds=local_start / fs)

            metadata = dict(raw.metadata)
            metadata["segment_total_samples"] = int(raw.segment_total_samples)
            metadata["segment_start_time"] = raw.segment_start_time

            yield ECGWindow(
                signal=signal[local_start:local_end],
                fs=fs,
                patient_id=raw.patient_id,
                segment_id=raw.segment_id,
                window_index=window_index_offset + window_count,
                start_sample=start_sample,
                end_sample=end_sample,
                n_samples=window_len,
                stream_offset_s=segment_time_offset_s + (local_start / fs),
                window_start_time=window_start_time,
                annotations=annotations,
                metadata=metadata,
            )
            window_count += 1

    def stream_patient(
        self,
        loader: ECGSegmentReader,
        patient_id: str,
        max_segments: int | None = None,
        max_windows: int | None = None,
    ) -> Iterator[ECGWindow]:
        """Chain all segments for a patient into a continuous window stream."""
        segments = self._ordered_segments(loader.get_patient_segments(patient_id))
        if max_segments is not None:
            segments = segments[:max_segments]

        segment_offsets = self._compute_segment_offsets(segments)
        yielded = 0
        window_index_offset = 0

        for segment, segment_time_offset_s in zip(segments, segment_offsets):
            raw = loader.read_segment(patient_id=patient_id, segment_id=segment.segment_id)
            segment_windows = 0

            for window in self.stream_segment(
                raw,
                segment_time_offset_s=segment_time_offset_s,
                window_index_offset=window_index_offset,
            ):
                if max_windows is not None and yielded >= max_windows:
                    return
                yield window
                yielded += 1
                segment_windows += 1

            window_index_offset += segment_windows

    @staticmethod
    def _seconds_to_samples(seconds: float, fs: int, field_name: str) -> int:
        samples = int(round(seconds * fs))
        if samples <= 0:
            raise ValueError(
                f"{field_name} must produce a positive sample count, got "
                f"{seconds!r}s at fs={fs}"
            )
        return samples

    @staticmethod
    def _ordered_segments(segments: Sequence[SegmentInfoLike]) -> list[SegmentInfoLike]:
        segments_list = list(segments)
        if segments_list and all(segment.start_time is not None for segment in segments_list):
            return sorted(segments_list, key=lambda segment: segment.start_time)
        return segments_list

    @classmethod
    def _compute_segment_offsets(cls, segments: Sequence[SegmentInfoLike]) -> list[float]:
        if not segments:
            return []

        offsets = [0.0]
        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            prev_offset = offsets[-1]

            if prev.start_time is not None and curr.start_time is not None:
                delta_s = (curr.start_time - prev.start_time).total_seconds()
                offsets.append(max(prev_offset, prev_offset + delta_s))
            else:
                offsets.append(prev_offset + float(prev.duration_seconds))

        return offsets

    @classmethod
    def _slice_annotations(
        cls,
        raw: RawECGDataLike,
        local_start: int,
        local_end: int,
    ) -> WindowAnnotations | None:
        beat_samples = None
        beat_symbols = None

        raw_beat_annotations = raw.beat_annotations
        if raw_beat_annotations is not None:
            raw_samples = np.asarray(getattr(raw_beat_annotations, "samples", []), dtype=int)
            raw_symbols = list(getattr(raw_beat_annotations, "symbols", []))
            if raw_samples.size and raw_symbols:
                mask = (raw_samples >= local_start) & (raw_samples < local_end)
                if np.any(mask):
                    beat_samples = raw_samples[mask] - local_start
                    beat_symbols = [
                        symbol
                        for symbol, keep in zip(raw_symbols, mask, strict=False)
                        if keep
                    ]

        rhythm_spans: list[RhythmSpan] = []
        raw_rhythm_annotations = raw.rhythm_annotations
        if raw_rhythm_annotations is not None:
            starts = list(getattr(raw_rhythm_annotations, "start_samples", []))
            ends = list(getattr(raw_rhythm_annotations, "end_samples", []))
            types = list(getattr(raw_rhythm_annotations, "rhythm_types", []))
            for span_start, span_end, rhythm_type in zip(starts, ends, types, strict=False):
                clipped_start = max(int(span_start), local_start)
                clipped_end = min(int(span_end), local_end)
                if clipped_end <= clipped_start:
                    continue
                rhythm_spans.append(
                    RhythmSpan(
                        start_sample=clipped_start - local_start,
                        end_sample=clipped_end - local_start,
                        rhythm_type=str(rhythm_type),
                    )
                )

        if beat_samples is None and beat_symbols is None and not rhythm_spans:
            return None

        return WindowAnnotations(
            beat_samples=beat_samples,
            beat_symbols=beat_symbols,
            rhythm_spans=rhythm_spans,
        )
