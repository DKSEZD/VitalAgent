"""Data sources for the live proactive/reactive monitoring demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol

from agent.data.icentia11k_loader import FS, Icentia11kLoader
from agent.data.streaming import ECGStreamer


class DataSource(Protocol):
    """Minimal window source consumed by :class:`LiveMonitorSession`."""

    dataset: str
    patient_id: str
    duration_s: float | None
    window_seconds: float

    def iter_windows(self) -> Iterator[Any]: ...


class Icentia11kDataSource:
    """Chain selected Icentia11k segment slices into one window timeline."""

    dataset = "icentia11k"

    def __init__(
        self,
        patient_id: str,
        segments: list[tuple[str, float, float | None]],
        window_seconds: float = 10.0,
        local_root: Path | str | None = None,
    ) -> None:
        if not segments:
            raise ValueError("At least one Icentia11k segment is required.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        self._loader = Icentia11kLoader(local_root=local_root)
        self.patient_id = self._loader.normalize_patient_id(patient_id)
        self.segments = [
            (self._loader.normalize_segment_id(segment_id), float(start_s), end_s)
            for segment_id, start_s, end_s in segments
        ]
        self.window_seconds = float(window_seconds)
        self.duration_s: float | None = None
        self._validate_segments()
        self._precompute_duration()

    def _validate_segments(self) -> None:
        for segment_id, start_s, end_s in self.segments:
            if start_s < 0:
                raise ValueError(f"Segment {segment_id}: start_s must be non-negative.")
            if end_s is not None and float(end_s) <= start_s:
                raise ValueError(f"Segment {segment_id}: end_s must be greater than start_s.")

    def _precompute_duration(self) -> None:
        """Best-effort total duration from lightweight segment headers."""
        try:
            total_s = 0.0
            for segment_id, start_s, end_s in self.segments:
                header = self._loader.read_segment_header(self.patient_id, segment_id)
                segment_total_s = float(header.n_samples) / float(header.fs)
                slice_end_s = min(
                    float(end_s) if end_s is not None else segment_total_s,
                    segment_total_s,
                )
                total_s += max(0.0, slice_end_s - start_s)
            self.duration_s = total_s
        except Exception:
            # Remote and partial mirrors may not expose every header. Playback
            # can still proceed, so leave the total unknown until iteration ends.
            self.duration_s = None

    def iter_windows(self) -> Iterator[Any]:
        streamer = ECGStreamer(window_seconds=self.window_seconds)
        cumulative_offset_s = 0.0
        cumulative_window_count = 0

        for segment_id, start_s, end_s in self.segments:
            sampfrom = int(round(start_s * FS))
            sampto = None if end_s is None else int(round(float(end_s) * FS))
            raw = self._loader.read_segment(
                patient_id=self.patient_id,
                segment_id=segment_id,
                sampfrom=sampfrom,
                sampto=sampto,
            )
            actual_duration_s = float(raw.n_samples) / float(raw.fs)

            segment_window_count = 0
            for window in streamer.stream_segment(
                raw,
                segment_time_offset_s=cumulative_offset_s,
                window_index_offset=cumulative_window_count,
            ):
                segment_window_count += 1
                yield window

            cumulative_offset_s += actual_duration_s
            cumulative_window_count += segment_window_count

        if self.duration_s is None:
            self.duration_s = cumulative_offset_s


class AfPpgEcgDataSource:
    """Reserved source shape for a future AF PPG/ECG live demo."""

    dataset = "afppgecg"

    def __init__(
        self,
        patient_id: str,
        segments: list[tuple[str, float, float | None]] | None = None,
        window_seconds: float = 10.0,
        **_: Any,
    ) -> None:
        self.patient_id = str(patient_id)
        self.segments = list(segments or [])
        self.window_seconds = float(window_seconds)
        self.duration_s: float | None = None

    def iter_windows(self) -> Iterator[Any]:
        raise NotImplementedError("afppgecg demo source not wired yet")
        yield  # pragma: no cover
