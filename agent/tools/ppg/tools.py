"""PPG data tools for the reactive pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.data.ppg_loader import PPGChunk, PPGLoader, PPGRecording
from agent.data.streaming import PPGStreamer
from agent.data.ppg_window_labels import parse_ppg_window_id
from agent.encoder.ppg_signal_processor import PPGSignalProcessor
from agent.mhealth.state_builders.ppg_state_builder import compute_ppg_max_hr_bpm
from agent.tools.ppg.window_analysis import quality_label
from agent.tools.registry import register_tool

PPG_SOURCE = "ppg"
_RECORDING_CONTEXT_MAX_30S_CACHE: dict[str, float | None] = {}


def _loader() -> PPGLoader:
    return PPGLoader()


def _normalize_patient_id(record_id: str | int) -> str:
    text = str(record_id).strip()
    if ":" in text:
        source, raw = text.split(":", 1)
        if source.strip().lower() != PPG_SOURCE:
            raise ValueError(
                f"Unsupported PPG record source {source!r}. Expected '{PPG_SOURCE}'."
            )
        return PPGLoader.normalize_patient_id(raw)
    return PPGLoader.normalize_patient_id(text)


def _record_id_for_patient(patient_id: str | int) -> str:
    return f"{PPG_SOURCE}:{PPGLoader.normalize_patient_id(patient_id)}"


def _resolve_record_id(
    record_id: str | int | None = None,
    *,
    dataset: str | None = None,
    patient_id: str | int | None = None,
) -> str:
    if dataset is not None and dataset not in {"afppgecg", "ppg"}:
        raise ValueError(f"Unsupported raw PPG dataset: {dataset!r}")
    if record_id is not None:
        return str(record_id)
    if patient_id is None:
        raise ValueError("record_id or patient_id is required.")
    return _record_id_for_patient(patient_id)


def _load_ppg_record(record_id: str | int, *, include_signals: bool) -> PPGRecording:
    patient_id = _normalize_patient_id(record_id)
    return _loader().read_ppg_record(patient_id, include_signals=include_signals)


def _select_chunk(recording: PPGRecording, chunk_index: int | None = None) -> PPGChunk:
    if not recording.chunks:
        raise ValueError(f"PPG record '{recording.patient_id}' contains no chunks.")
    if chunk_index is None:
        return max(recording.chunks, key=lambda chunk: chunk.duration_seconds)
    if chunk_index < 0 or chunk_index >= len(recording.chunks):
        raise ValueError(
            f"Invalid chunk_index={chunk_index}; available range is 0-{len(recording.chunks) - 1}."
        )
    return recording.chunks[chunk_index]


def _slice_accel_axis(
    values: np.ndarray | None,
    *,
    start_sample: int,
    end_sample: int,
    accel_ratio: float,
) -> np.ndarray | None:
    if values is None:
        return None
    accel_start = int(start_sample * accel_ratio)
    accel_end = int(end_sample * accel_ratio)
    return np.asarray(values[accel_start:accel_end], dtype=float)


def _resolve_signal_view(
    recording: PPGRecording,
    *,
    dataset: str | None,
    chunk_index: int | None,
    window_id: str | None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
) -> tuple[PPGChunk, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    chunk = _select_chunk(recording, chunk_index)
    if chunk.green_signal is None:
        raise ValueError("PPG green-channel signal is unavailable.")

    if window_id is None and window_start_s is None and window_end_s is None:
        return (
            chunk,
            np.asarray(chunk.green_signal, dtype=float),
            (
                np.asarray(chunk.ambient_signal, dtype=float)
                if chunk.ambient_signal is not None
                else None
            ),
            _chunk_accel_matrix(chunk),
            {},
        )

    if window_id is None:
        base_time, time_reference = _locator_base_time(recording, dataset)
        return _resolve_signal_view_by_time(
            recording,
            chunk_index=chunk_index,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            base_time=base_time,
            time_reference=time_reference,
        )

    parsed = parse_ppg_window_id(window_id)
    if parsed.patient_id != recording.patient_id:
        raise ValueError(
            f"window_id patient {parsed.patient_id!r} does not match record patient {recording.patient_id!r}."
        )
    if chunk_index is not None and parsed.chunk_index != chunk_index:
        raise ValueError(
            f"window_id chunk_index={parsed.chunk_index} does not match requested chunk_index={chunk_index}."
        )

    chunk = _select_chunk(recording, parsed.chunk_index)
    if chunk.green_signal is None:
        raise ValueError("PPG green-channel signal is unavailable.")
    if parsed.end_sample > chunk.n_ppg_samples:
        raise ValueError(
            f"window_id end_sample={parsed.end_sample} exceeds chunk length {chunk.n_ppg_samples}."
        )

    green = np.asarray(chunk.green_signal[parsed.start_sample:parsed.end_sample], dtype=float)
    ambient = (
        np.asarray(chunk.ambient_signal[parsed.start_sample:parsed.end_sample], dtype=float)
        if chunk.ambient_signal is not None
        else None
    )

    accel = None
    if chunk.ppg_fs > 0 and chunk.accel_fs > 0:
        accel_ratio = chunk.accel_fs / chunk.ppg_fs
        ax = _slice_accel_axis(
            chunk.accel_x,
            start_sample=parsed.start_sample,
            end_sample=parsed.end_sample,
            accel_ratio=accel_ratio,
        )
        ay = _slice_accel_axis(
            chunk.accel_y,
            start_sample=parsed.start_sample,
            end_sample=parsed.end_sample,
            accel_ratio=accel_ratio,
        )
        az = _slice_accel_axis(
            chunk.accel_z,
            start_sample=parsed.start_sample,
            end_sample=parsed.end_sample,
            accel_ratio=accel_ratio,
        )
        if ax is not None and ay is not None and az is not None:
            n = min(len(ax), len(ay), len(az))
            accel = np.stack([ax[:n], ay[:n], az[:n]], axis=1)

    metadata = {
        "window_id": window_id,
        "window_index": parsed.window_index,
        "window_start_sample": parsed.start_sample,
        "window_end_sample": parsed.end_sample,
        "window_duration_seconds": round(
            (parsed.end_sample - parsed.start_sample) / float(chunk.ppg_fs),
            3,
        ),
    }
    return chunk, green, ambient, accel, metadata


def _resolve_signal_view_by_time(
    recording: PPGRecording,
    *,
    chunk_index: int | None,
    window_start_s: float | None,
    window_end_s: float | None,
    base_time: Any,
    time_reference: str,
) -> tuple[PPGChunk, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if window_start_s is None or window_end_s is None:
        raise ValueError("window_start_s and window_end_s must be provided together.")

    start_s = float(window_start_s)
    end_s = float(window_end_s)
    if end_s <= start_s:
        raise ValueError("window_end_s must be greater than window_start_s.")

    candidates = [_select_chunk(recording, chunk_index)] if chunk_index is not None else recording.chunks

    for chunk in candidates:
        if chunk.green_signal is None:
            continue
        chunk_start_s = (chunk.start_time - base_time).total_seconds()
        chunk_end_s = chunk_start_s + chunk.duration_seconds
        if start_s < chunk_start_s or end_s > chunk_end_s:
            continue

        start_sample = int(round((start_s - chunk_start_s) * chunk.ppg_fs))
        end_sample = int(round((end_s - chunk_start_s) * chunk.ppg_fs))
        start_sample = max(0, min(start_sample, chunk.n_ppg_samples))
        end_sample = max(start_sample, min(end_sample, chunk.n_ppg_samples))
        if end_sample <= start_sample:
            raise ValueError("Resolved PPG window contains no samples.")

        green = np.asarray(chunk.green_signal[start_sample:end_sample], dtype=float)
        ambient = (
            np.asarray(chunk.ambient_signal[start_sample:end_sample], dtype=float)
            if chunk.ambient_signal is not None
            else None
        )
        accel = None
        if chunk.ppg_fs > 0 and chunk.accel_fs > 0:
            accel_ratio = chunk.accel_fs / chunk.ppg_fs
            ax = _slice_accel_axis(
                chunk.accel_x,
                start_sample=start_sample,
                end_sample=end_sample,
                accel_ratio=accel_ratio,
            )
            ay = _slice_accel_axis(
                chunk.accel_y,
                start_sample=start_sample,
                end_sample=end_sample,
                accel_ratio=accel_ratio,
            )
            az = _slice_accel_axis(
                chunk.accel_z,
                start_sample=start_sample,
                end_sample=end_sample,
                accel_ratio=accel_ratio,
            )
            if ax is not None and ay is not None and az is not None:
                n = min(len(ax), len(ay), len(az))
                accel = np.stack([ax[:n], ay[:n], az[:n]], axis=1)

        metadata = {
            "window_start_s": start_s,
            "window_end_s": end_s,
            "window_start_sample": start_sample,
            "window_end_sample": end_sample,
            "window_duration_seconds": round((end_sample - start_sample) / float(chunk.ppg_fs), 3),
            "time_reference": time_reference,
        }
        return chunk, green, ambient, accel, metadata

    raise ValueError(
        "No PPG chunk fully contains the requested window_start_s/window_end_s."
    )


def _locator_base_time(recording: PPGRecording, dataset: str | None) -> tuple[Any, str]:
    if dataset == "afppgecg":
        try:
            return (
                _loader().read_ecg_record(recording.patient_id, include_signal=False).start_time,
                "ecg_recording_start",
            )
        except Exception:
            pass
    return min(chunk.start_time for chunk in recording.chunks), "first_ppg_chunk_start"


def _chunk_accel_matrix(chunk: PPGChunk) -> np.ndarray | None:
    if chunk.accel_x is None or chunk.accel_y is None or chunk.accel_z is None:
        return None
    return np.column_stack((chunk.accel_x, chunk.accel_y, chunk.accel_z))


def _build_record_summary(recording: PPGRecording) -> dict[str, Any]:
    loader = _loader()
    summary = loader.summarize_ppg(recording)
    first_chunk = recording.chunks[0] if recording.chunks else None
    return {
        "record_id": _record_id_for_patient(recording.patient_id),
        "patient_id": recording.patient_id,
        "source": PPG_SOURCE,
        "chunk_count": summary["chunk_count"],
        "total_hours": round(float(summary["total_hours"]), 3),
        "longest_chunk_hours": round(float(summary["longest_chunk_hours"]), 3),
        "chunk_hours": [round(float(hours), 3) for hours in summary["chunk_hours"]],
        "chunk_start_times": summary["chunk_start_times"],
        "ppg_sampling_rate": first_chunk.ppg_fs if first_chunk is not None else None,
        "accel_sampling_rate": first_chunk.accel_fs if first_chunk is not None else None,
    }


@register_tool(
    name="list_ppg_records",
    description=(
        "List available PPG patient records from the wrist-PPG dataset. "
        "Returns patient-level PPG record IDs in the form 'ppg:<patient_id>' "
        "plus chunk-count and duration summaries."
    ),
    parameters={
        "limit": {
            "type": "integer",
            "description": "Max number of PPG records to return.",
            "default": 20,
        },
    },
)
def list_ppg_records(limit: int = 20) -> dict[str, Any]:
    loader = _loader()
    records: list[dict[str, Any]] = []

    for patient_id in loader.available_patient_ids():
        if len(records) >= limit:
            break
        if not loader.patient_files_complete(patient_id):
            continue
        try:
            recording = loader.read_ppg_record(patient_id, include_signals=False)
        except Exception:
            continue
        records.append(_build_record_summary(recording))

    return {
        "success": True,
        "record_ids": [item["record_id"] for item in records],
        "records": records,
        "count": len(records),
    }


@register_tool(
    name="get_ppg_description",
    description=(
        "Get a text description of a patient-level PPG recording. Includes chunk "
        "count, total duration, longest chunk, and sampling rates."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "PPG record ID in the form 'ppg:<patient_id>'.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset name from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator.",
            "required": False,
        },
    },
)
def get_ppg_description(
    record_id: str | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
) -> dict[str, Any]:
    try:
        record_id = _resolve_record_id(record_id, dataset=dataset, patient_id=patient_id)
        recording = _load_ppg_record(record_id, include_signals=False)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    summary = _build_record_summary(recording)
    description = "\n".join(
        [
            f"PPG record ID: {summary['record_id']}",
            f"Patient ID: {summary['patient_id']}",
            f"Source: {summary['source']}",
            f"Chunk count: {summary['chunk_count']}",
            f"Total duration: {summary['total_hours']:.3f} hours",
            f"Longest chunk: {summary['longest_chunk_hours']:.3f} hours",
            f"PPG sampling rate: {summary['ppg_sampling_rate']} Hz",
            f"Accelerometer sampling rate: {summary['accel_sampling_rate']} Hz",
            "Chunk start times: " + ", ".join(summary["chunk_start_times"][:5]),
        ]
    )
    return {
        "success": True,
        "description": description,
        "record_id": summary["record_id"],
    }


@register_tool(
    name="get_ppg_metadata",
    description=(
        "Retrieve structured metadata for a patient-level PPG recording, including "
        "chunk counts, durations, and sampling rates."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "PPG record ID in the form 'ppg:<patient_id>'.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset name from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator.",
            "required": False,
        },
    },
)
def get_ppg_metadata(
    record_id: str | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
) -> dict[str, Any]:
    try:
        record_id = _resolve_record_id(record_id, dataset=dataset, patient_id=patient_id)
        recording = _load_ppg_record(record_id, include_signals=False)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    return {"success": True, "metadata": _build_record_summary(recording)}


@register_tool(
    name="analyze_pulse_rate",
    description=(
        "Analyze pulse rate from a PPG recording. Returns mean/min/max pulse rate, "
        "30-second subwindow max/min pulse rate when the requested window is long "
        "enough, PP-interval statistics, and a bradycardia/normal/tachycardia assessment."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "PPG record ID in the form 'ppg:<patient_id>'.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset name from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "chunk_index": {
            "type": "integer",
            "description": (
                "Which chunk to analyze. Defaults to the longest available chunk."
            ),
            "required": False,
        },
        "window_id": {
            "type": "string",
            "description": (
                "Optional exact PPG window ID. If provided, analyze that window "
                "instead of the full chunk."
            ),
            "required": False,
        },
        "include_recording_context": {
            "type": "boolean",
            "description": (
                "When true, also scan the patient's full PPG recording and return "
                "recording_context_max_30s_bpm. Use this only for monitoring-window "
                "highest/max HR questions where the benchmark asks for aggregate context."
            ),
            "required": False,
        },
    },
)
def analyze_pulse_rate(
    record_id: str | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    chunk_index: int | None = None,
    window_id: str | None = None,
    include_recording_context: bool = False,
) -> dict[str, Any]:
    try:
        record_id = _resolve_record_id(record_id, dataset=dataset, patient_id=patient_id)
        recording = _load_ppg_record(record_id, include_signals=True)
        chunk, signal, _ambient, accel, window_meta = _resolve_signal_view(
            recording,
            dataset=dataset,
            chunk_index=chunk_index,
            window_id=window_id,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    processor = PPGSignalProcessor(sampling_rate=chunk.ppg_fs)
    vitals = processor.extract_vitals(signal, fs=chunk.ppg_fs, accel=accel)
    intervals = list(vitals.rr_intervals_ms or [])

    if vitals.hr_bpm is None or len(intervals) < 2:
        return {
            "success": False,
            "error": "Too few valid PPG peaks detected. Signal may be too short or noisy.",
        }

    instantaneous_bpm = 60_000.0 / np.asarray(intervals, dtype=float)
    hr_mean = round(float(vitals.hr_bpm), 1)
    hr_min = round(float(np.min(instantaneous_bpm)), 1)
    hr_max = round(float(np.max(instantaneous_bpm)), 1)
    hr_std = round(float(np.std(instantaneous_bpm)), 1)
    hr_30s = _ppg_hr_subwindow_summary(
        signal=signal,
        fs=chunk.ppg_fs,
        accel=accel,
        processor=processor,
        subwindow_seconds=30.0,
    )
    pp_mean = round(float(np.mean(intervals)), 1)
    pp_std = round(float(np.std(intervals)), 1)

    if hr_mean < 60:
        assessment = "bradycardia"
        assessment_detail = (
            f"Mean pulse rate ({hr_mean} bpm) is below 60 bpm, indicating bradycardia."
        )
    elif hr_mean > 100:
        assessment = "tachycardia"
        assessment_detail = (
            f"Mean pulse rate ({hr_mean} bpm) is above 100 bpm, indicating tachycardia."
        )
    else:
        assessment = "normal"
        assessment_detail = (
            f"Mean pulse rate ({hr_mean} bpm) is within normal range (60-100 bpm)."
        )

    pp_assessment = "normal"
    pp_assessment_detail = (
        f"Mean PP interval ({pp_mean} ms) is within normal range (600-1000 ms)."
    )
    if pp_mean < 600:
        pp_assessment = "below_normal"
        pp_assessment_detail = (
            f"Mean PP interval ({pp_mean} ms) is below normal range (< 600 ms), "
            "indicating fast pulse rate."
        )
    elif pp_mean > 1000:
        pp_assessment = "above_normal"
        pp_assessment_detail = (
            f"Mean PP interval ({pp_mean} ms) is above normal range (> 1000 ms), "
            "indicating slow pulse rate."
        )

    variability_note = None
    if (hr_max - hr_min) > 30:
        variability_note = (
            f"Pulse rate varies significantly ({hr_min}-{hr_max} bpm). "
            "Consider reviewing rhythm irregularity and signal quality."
        )

    heart_rate = {
        "mean_bpm": hr_mean,
        "min_bpm": hr_min,
        "max_bpm": hr_max,
        "std_bpm": hr_std,
    }
    pulse_rate = dict(heart_rate)
    if hr_30s is not None:
        heart_rate.update(hr_30s)
        pulse_rate.update(hr_30s)
    if include_recording_context:
        recording_context_max = _recording_context_max_30s_bpm(recording)
        if recording_context_max is not None:
            heart_rate["recording_context_max_30s_bpm"] = round(
                float(recording_context_max),
                1,
            )
            heart_rate["recording_context_max_30s_bpm_source"] = (
                "ppg_peak_detection_recording_context_30s_subwindows"
            )
            pulse_rate["recording_context_max_30s_bpm"] = heart_rate[
                "recording_context_max_30s_bpm"
            ]
            pulse_rate["recording_context_max_30s_bpm_source"] = heart_rate[
                "recording_context_max_30s_bpm_source"
            ]

    return {
        "success": True,
        "record_id": _record_id_for_patient(recording.patient_id),
        "data": {
            "heart_rate": heart_rate,
            "pulse_rate": pulse_rate,
            "pp_intervals": {
                "mean_ms": pp_mean,
                "std_ms": pp_std,
                "count": len(intervals),
                "assessment": pp_assessment,
                "assessment_detail": pp_assessment_detail,
                "normal_range_ms": {"lower": 600.0, "upper": 1000.0},
                "is_below_normal": pp_mean < 600,
                "is_above_normal": pp_mean > 1000,
                "is_within_normal": 600 <= pp_mean <= 1000,
            },
            "assessment": assessment,
            "assessment_detail": assessment_detail,
            "variability_note": variability_note,
        },
        "metadata": {
            "record_id": _record_id_for_patient(recording.patient_id),
            "patient_id": recording.patient_id,
            "chunk_index": chunk.chunk_index,
            "chunk_start_time": chunk.start_time.isoformat(sep=" "),
            "sampling_rate": chunk.ppg_fs,
            "duration_seconds": round(
                float(window_meta.get("window_duration_seconds", chunk.duration_seconds)),
                1,
            ),
            "signal_quality_score": vitals.signal_quality_score,
            "num_ppg_intervals": len(intervals),
            **window_meta,
        },
    }


def _ppg_hr_subwindow_summary(
    *,
    signal: np.ndarray,
    fs: int,
    accel: np.ndarray | None,
    processor: PPGSignalProcessor,
    subwindow_seconds: float,
) -> dict[str, Any] | None:
    subwindow_samples = int(round(float(subwindow_seconds) * fs))
    if subwindow_samples <= 0 or signal.size < subwindow_samples:
        return None

    values: list[float] = []
    cursor = 0
    while cursor < signal.size:
        sub_end = min(cursor + subwindow_samples, signal.size)
        sub_signal = np.asarray(signal[cursor:sub_end], dtype=float)
        if sub_signal.size <= 0:
            break
        sub_accel = _slice_accel_for_window(
            accel=accel,
            start_sample=cursor,
            end_sample=sub_end,
            total_ppg_samples=signal.size,
        )
        vitals = processor.extract_vitals(sub_signal, fs=fs, accel=sub_accel)
        if vitals.hr_bpm is not None:
            values.append(float(vitals.hr_bpm))
        cursor = sub_end

    if not values:
        return None
    return {
        "max_30s_bpm": round(float(max(values)), 1),
        "min_30s_bpm": round(float(min(values)), 1),
        "max_30s_bpm_source": "ppg_peak_detection_30s_subwindows",
    }


def _recording_context_max_30s_bpm(recording: PPGRecording) -> float | None:
    patient = str(recording.patient_id)
    if patient in _RECORDING_CONTEXT_MAX_30S_CACHE:
        return _RECORDING_CONTEXT_MAX_30S_CACHE[patient]

    processor = PPGSignalProcessor()
    best: float | None = None
    streamer = PPGStreamer(window_seconds=300.0, drop_last=True)
    for window in streamer.stream_recording(recording):
        value = compute_ppg_max_hr_bpm(
            window=window,
            processor=processor,
            subwindow_sec=30,
        )
        if value is not None:
            best = float(value) if best is None else max(best, float(value))

    _RECORDING_CONTEXT_MAX_30S_CACHE[patient] = best
    return best


def _slice_accel_for_window(
    *,
    accel: np.ndarray | None,
    start_sample: int,
    end_sample: int,
    total_ppg_samples: int,
) -> np.ndarray | None:
    if accel is None or total_ppg_samples <= 0:
        return None
    accel_values = np.asarray(accel)
    if accel_values.size == 0:
        return None
    ratio = accel_values.shape[0] / float(total_ppg_samples)
    start = int(round(start_sample * ratio))
    end = int(round(end_sample * ratio))
    return accel_values[start:end]


@register_tool(
    name="analyze_prv",
    description=(
        "Analyze pulse rate variability (PRV) from a PPG recording. Returns "
        "time-domain variability metrics derived from PP intervals."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "PPG record ID in the form 'ppg:<patient_id>'.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset name from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "chunk_index": {
            "type": "integer",
            "description": (
                "Which chunk to analyze. Defaults to the longest available chunk."
            ),
            "required": False,
        },
        "window_id": {
            "type": "string",
            "description": (
                "Optional exact PPG window ID. If provided, analyze that window "
                "instead of the full chunk."
            ),
            "required": False,
        },
    },
)
def analyze_prv(
    record_id: str | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    chunk_index: int | None = None,
    window_id: str | None = None,
) -> dict[str, Any]:
    try:
        record_id = _resolve_record_id(record_id, dataset=dataset, patient_id=patient_id)
        recording = _load_ppg_record(record_id, include_signals=True)
        chunk, signal, _ambient, accel, window_meta = _resolve_signal_view(
            recording,
            dataset=dataset,
            chunk_index=chunk_index,
            window_id=window_id,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    processor = PPGSignalProcessor(sampling_rate=chunk.ppg_fs)
    vitals = processor.extract_vitals(signal, fs=chunk.ppg_fs, accel=accel)
    intervals = np.asarray(vitals.rr_intervals_ms or [], dtype=float)

    if intervals.size < 3 or vitals.sdnn_ms is None or vitals.rmssd_ms is None:
        return {
            "success": False,
            "error": "Too few valid PP intervals for PRV analysis.",
        }

    diffs = np.diff(intervals)
    time_domain = {
        "mean_nn_ms": round(float(np.mean(intervals)), 1),
        "sdnn_ms": float(vitals.sdnn_ms),
        "rmssd_ms": float(vitals.rmssd_ms),
        "pnn50_percent": round(
            float(np.sum(np.abs(diffs) > 50.0) / len(diffs) * 100.0),
            1,
        )
        if len(diffs) > 0
        else 0.0,
        "nn_count": int(len(intervals)),
    }

    caveats = []
    duration_seconds = float(window_meta.get("window_duration_seconds", chunk.duration_seconds))
    if duration_seconds < 30:
        caveats.append(
            f"Analyzed chunk is very short ({duration_seconds:.0f}s); PRV metrics have limited reliability."
        )
    elif duration_seconds < 300:
        caveats.append(
            f"Analyzed chunk ({duration_seconds:.0f}s) is below the 5-min clinical standard; interpret PRV with caution."
        )
    if vitals.signal_quality_score is not None and vitals.signal_quality_score < 0.5:
        caveats.append("Signal quality is low; PRV metrics may be unreliable.")

    return {
        "success": True,
        "record_id": _record_id_for_patient(recording.patient_id),
        "data": {
            "time_domain": time_domain,
            "caveats": caveats if caveats else None,
        },
        "metadata": {
            "record_id": _record_id_for_patient(recording.patient_id),
            "patient_id": recording.patient_id,
            "chunk_index": chunk.chunk_index,
            "chunk_start_time": chunk.start_time.isoformat(sep=" "),
            "sampling_rate": chunk.ppg_fs,
            "duration_seconds": round(duration_seconds, 1),
            "signal_quality_score": vitals.signal_quality_score,
            **window_meta,
        },
    }


@register_tool(
    name="assess_ppg_signal_quality",
    description=(
        "Assess the quality of a PPG chunk. Returns a 0-1 quality score, a quality "
        "label, and motion/coverage caveats."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "PPG record ID in the form 'ppg:<patient_id>'.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset name from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "chunk_index": {
            "type": "integer",
            "description": (
                "Which chunk to analyze. Defaults to the longest available chunk."
            ),
            "required": False,
        },
        "window_id": {
            "type": "string",
            "description": (
                "Optional exact PPG window ID. If provided, analyze that window "
                "instead of the full chunk."
            ),
            "required": False,
        },
    },
)
def assess_ppg_signal_quality(
    record_id: str | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    chunk_index: int | None = None,
    window_id: str | None = None,
) -> dict[str, Any]:
    try:
        record_id = _resolve_record_id(record_id, dataset=dataset, patient_id=patient_id)
        recording = _load_ppg_record(record_id, include_signals=True)
        chunk, signal, _ambient, accel, window_meta = _resolve_signal_view(
            recording,
            dataset=dataset,
            chunk_index=chunk_index,
            window_id=window_id,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    processor = PPGSignalProcessor(sampling_rate=chunk.ppg_fs)
    vitals = processor.extract_vitals(signal, fs=chunk.ppg_fs, accel=accel)
    quality_score = float(vitals.signal_quality_score or 0.0)
    motion_intensity = (
        round(float(processor._motion_intensity(accel)), 3)  # noqa: SLF001
        if accel is not None
        else None
    )
    quality_label_value = quality_label(quality_score)

    caveats = []
    if quality_label_value == "poor":
        caveats.append("Signal quality is poor. Derived pulse metrics may be unreliable.")
    if motion_intensity is not None and motion_intensity > 0.5:
        caveats.append("High accelerometer motion suggests motion artefact.")

    return {
        "success": True,
        "record_id": _record_id_for_patient(recording.patient_id),
        "data": {
            "overall_quality_score": round(quality_score, 3),
            "quality_label": quality_label_value,
            "component_scores": {
                "signal_quality": round(quality_score, 3),
                "motion_intensity": motion_intensity,
            },
            "caveats": caveats if caveats else None,
        },
        "metadata": {
            "record_id": _record_id_for_patient(recording.patient_id),
            "patient_id": recording.patient_id,
            "chunk_index": chunk.chunk_index,
            "chunk_start_time": chunk.start_time.isoformat(sep=" "),
            "sampling_rate": chunk.ppg_fs,
            "duration_seconds": round(
                float(window_meta.get("window_duration_seconds", chunk.duration_seconds)),
                1,
            ),
            **window_meta,
        },
    }


@register_tool(
    name="analyze_ppg_rhythm_irregularity",
    description=(
        "Analyze PP-interval irregularity from a PPG chunk. Useful for irregular "
        "pulse screening, but not for definitive ECG diagnosis."
    ),
    parameters={
        "record_id": {
            "type": "string",
            "description": "PPG record ID in the form 'ppg:<patient_id>'.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Dataset name from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Patient ID from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "chunk_index": {
            "type": "integer",
            "description": (
                "Which chunk to analyze. Defaults to the longest available chunk."
            ),
            "required": False,
        },
        "window_id": {
            "type": "string",
            "description": (
                "Optional exact PPG window ID. If provided, analyze that window "
                "instead of the full chunk."
            ),
            "required": False,
        },
    },
)
def analyze_ppg_rhythm_irregularity(
    record_id: str | None = None,
    dataset: str | None = None,
    patient_id: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    chunk_index: int | None = None,
    window_id: str | None = None,
) -> dict[str, Any]:
    try:
        record_id = _resolve_record_id(record_id, dataset=dataset, patient_id=patient_id)
        recording = _load_ppg_record(record_id, include_signals=True)
        chunk, signal, _ambient, accel, window_meta = _resolve_signal_view(
            recording,
            dataset=dataset,
            chunk_index=chunk_index,
            window_id=window_id,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    processor = PPGSignalProcessor(sampling_rate=chunk.ppg_fs)
    vitals = processor.extract_vitals(signal, fs=chunk.ppg_fs, accel=accel)
    intervals = np.asarray(vitals.rr_intervals_ms or [], dtype=float)
    quality_score = float(vitals.signal_quality_score or 0.0)

    if intervals.size < 3:
        return {
            "success": False,
            "error": "Too few valid PP intervals for rhythm irregularity analysis.",
        }

    diffs = np.diff(intervals)
    cv = float(np.std(intervals, ddof=1) / np.mean(intervals)) if intervals.size > 1 else 0.0
    pnn50 = (
        float(np.sum(np.abs(diffs) > 50.0) / len(diffs) * 100.0)
        if len(diffs) > 0
        else 0.0
    )
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) > 0 else 0.0

    if quality_score < 0.6:
        assessment = "indeterminate"
        assessment_detail = (
            "Signal quality is too low for reliable PPG rhythm irregularity assessment."
        )
    else:
        irregular_evidence_count = sum(
            [
                cv >= 0.18,
                pnn50 >= 35.0,
                rmssd >= 120.0,
            ]
        )
        if irregular_evidence_count >= 2:
            assessment = "irregular"
            assessment_detail = (
                "PP intervals show substantial beat-to-beat variability consistent with an irregular pulse pattern."
            )
        else:
            assessment = "regular"
            assessment_detail = (
                "PP intervals are relatively stable and do not suggest a strongly irregular pulse pattern."
            )

    caveats = [
        "PPG irregularity can support screening for rhythm disturbance, but it does not confirm an ECG diagnosis."
    ]
    if quality_score < 0.6:
        caveats.append(
            "Signal quality below 0.60 blocks positive irregularity calls because motion artefact can inflate PP-interval variability."
        )

    return {
        "success": True,
        "record_id": _record_id_for_patient(recording.patient_id),
        "data": {
            "pp_intervals": {
                "count": int(len(intervals)),
                "mean_ms": round(float(np.mean(intervals)), 1),
                "std_ms": round(float(np.std(intervals, ddof=1)), 1),
            },
            "irregularity_metrics": {
                "coefficient_of_variation": round(cv, 3),
                "rmssd_ms": round(rmssd, 1),
                "pnn50_percent": round(pnn50, 1),
            },
            "assessment": assessment,
            "assessment_detail": assessment_detail,
            "is_irregular": assessment == "irregular",
            "caveats": caveats,
        },
        "metadata": {
            "record_id": _record_id_for_patient(recording.patient_id),
            "patient_id": recording.patient_id,
            "chunk_index": chunk.chunk_index,
            "chunk_start_time": chunk.start_time.isoformat(sep=" "),
            "sampling_rate": chunk.ppg_fs,
            "duration_seconds": round(
                float(window_meta.get("window_duration_seconds", chunk.duration_seconds)),
                1,
            ),
            "signal_quality_score": round(quality_score, 3),
            **window_meta,
        },
    }
