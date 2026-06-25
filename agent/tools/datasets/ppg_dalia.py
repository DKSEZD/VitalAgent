"""PPG-DaLiA raw-signal tools for the reactive pipeline."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import get_config
from agent.mhealth.state_builders.ppg_dalia_state_builder import (
    PPG_DALIA_BVP_FS_HZ,
    PPG_DALIA_CHEST_FS_HZ,
    PPG_DALIA_HR_LABEL_FS_HZ,
    PPG_DALIA_WRIST_ACC_FS_HZ,
    PPG_DALIA_WRIST_LOW_RATE_FS_HZ,
    map_ppg_dalia_activity_label,
)
from agent.tools.ppg.window_analysis import (
    as_float_array,
    bvp_pulse_summary,
    chest_ecg_summary,
    finite_1d,
    get_nested,
    motion_summary,
    slice_by_time,
    trend_summary,
)
from agent.tools.registry import register_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PPG_DALIA_ROOT = (
    PROJECT_ROOT / "dataset" / "ppg_dalia_sample" / "converted_original_like"
)


def _normalize_subject_id(value: str | int | None) -> str:
    if value is None:
        raise ValueError("subject_id or patient_id is required when pickle_path is not provided.")
    text = str(value).strip()
    if not text:
        raise ValueError("subject_id must not be empty.")
    return text if text.upper().startswith("S") else f"S{text}"


def _resolve_pickle_path(
    *,
    pickle_path: str | None,
    subject_id: str | int | None,
    patient_id: str | int | None,
    dataset_root: str | None,
) -> Path:
    if pickle_path:
        return Path(pickle_path)

    subject = _normalize_subject_id(subject_id if subject_id is not None else patient_id)
    roots = (
        [Path(dataset_root)]
        if dataset_root
        else [get_config().datasets.ppg_dalia_root, SAMPLE_PPG_DALIA_ROOT]
    )
    candidates = [root / subject / f"{subject}.pkl" for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find PPG-DaLiA pickle for {subject}. Checked: {checked}")


@lru_cache(maxsize=2)
def _load_ppg_dalia_pickle_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    with Path(path_text).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"PPG-DaLiA pickle did not contain a dict: {path_text}")
    if "signal" not in payload:
        raise ValueError(f"PPG-DaLiA pickle has no signal tree: {path_text}")
    return payload


def _load_ppg_dalia_pickle(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"PPG-DaLiA pickle not found: {resolved}")
    return _load_ppg_dalia_pickle_cached(str(resolved), resolved.stat().st_mtime_ns)


def _infer_activity_fs_hz(
    *,
    activity: np.ndarray | None,
    bvp: np.ndarray | None,
    hr_labels: np.ndarray | None,
) -> float:
    if activity is None or activity.size == 0:
        return float(PPG_DALIA_WRIST_LOW_RATE_FS_HZ)

    reference_durations: list[float] = []
    if bvp is not None and bvp.size:
        reference_durations.append(bvp.shape[0] / float(PPG_DALIA_BVP_FS_HZ))
    if hr_labels is not None and hr_labels.size:
        reference_durations.append(hr_labels.size / float(PPG_DALIA_HR_LABEL_FS_HZ))
    reference_durations = [duration for duration in reference_durations if duration > 0]
    if not reference_durations:
        return float(PPG_DALIA_WRIST_LOW_RATE_FS_HZ)

    inferred = float(activity.reshape(-1).size) / float(np.median(reference_durations))
    if inferred <= 0 or not np.isfinite(inferred):
        return float(PPG_DALIA_WRIST_LOW_RATE_FS_HZ)
    return inferred


def _activity_summary(
    activity_window: np.ndarray | None,
) -> dict[str, Any] | None:
    finite = finite_1d(activity_window)
    if finite.size == 0:
        return None

    ids = np.rint(finite).astype(int)
    values, counts = np.unique(ids, return_counts=True)
    best = int(np.argmax(counts))
    label_id = int(values[best])
    fraction = float(counts[best]) / float(ids.size)
    return {
        "dominant_activity_label_id": label_id,
        "dominant_activity_label": map_ppg_dalia_activity_label(label_id),
        "dominant_activity_fraction": round(fraction, 4),
        "sample_count": int(ids.size),
        "label_counts": {
            map_ppg_dalia_activity_label(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        },
    }


def _hr_label_summary(hr_window: np.ndarray | None) -> dict[str, Any] | None:
    finite = finite_1d(hr_window)
    if finite.size == 0:
        return None
    return {
        "source": "ecg_derived_hr_label",
        "mean_bpm": round(float(np.mean(finite)), 1),
        "min_bpm": round(float(np.min(finite)), 1),
        "max_bpm": round(float(np.max(finite)), 1),
        "label_count": int(finite.size),
        "sampling_rate_hz": PPG_DALIA_HR_LABEL_FS_HZ,
        "time_semantics": "0.5 Hz HR label series; commonly derived from ECG windows shifted every 2 seconds.",
    }

@register_tool(
    name="analyze_ppg_dalia_window_signal",
    description=(
        "Analyze raw synchronized PPG-DaLiA signals for one subject window. "
        "Returns wrist BVP pulse metrics, BVP quality, ECG-derived HR labels, "
        "activity context, wrist motion, EDA, temperature, and caveats."
    ),
    parameters={
        "subject_id": {
            "type": "string",
            "description": "PPG-DaLiA subject ID, e.g. 'S1'.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Alias for subject_id from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "pickle_path": {
            "type": "string",
            "description": "Optional direct path to one original-like PPG-DaLiA S{n}.pkl file.",
            "required": False,
        },
        "dataset_root": {
            "type": "string",
            "description": "Optional root containing S{n}/S{n}.pkl subject folders.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Optional dataset name from an mHealth-QA WindowLocator; expected 'ppg_dalia'.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds from the synchronized subject timeline.",
            "required": True,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds from the synchronized subject timeline.",
            "required": True,
        },
        "include_chest_ecg_analysis": {
            "type": "boolean",
            "description": "Also run ECG R-peak analysis on the chest ECG window.",
            "default": False,
        },
    },
)
def analyze_ppg_dalia_window_signal(
    subject_id: str | None = None,
    patient_id: str | None = None,
    pickle_path: str | None = None,
    dataset_root: str | None = None,
    dataset: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    include_chest_ecg_analysis: bool = False,
) -> dict[str, Any]:
    try:
        if dataset is not None and dataset != "ppg_dalia":
            raise ValueError(f"Unsupported dataset for PPG-DaLiA raw tool: {dataset!r}")
        if window_start_s is None or window_end_s is None:
            raise ValueError("window_start_s and window_end_s are required.")
        start_s = float(window_start_s)
        end_s = float(window_end_s)
        if end_s <= start_s:
            raise ValueError("window_end_s must be greater than window_start_s.")

        path = _resolve_pickle_path(
            pickle_path=pickle_path,
            subject_id=subject_id,
            patient_id=patient_id,
            dataset_root=dataset_root,
        )
        payload = _load_ppg_dalia_pickle(path)
        resolved_subject = str(payload.get("subject") or path.stem)
        duration_s = end_s - start_s

        bvp = as_float_array(get_nested(payload, ("signal", "wrist", "BVP")))
        wrist_acc = as_float_array(get_nested(payload, ("signal", "wrist", "ACC")))
        eda = as_float_array(get_nested(payload, ("signal", "wrist", "EDA")))
        temp = as_float_array(get_nested(payload, ("signal", "wrist", "TEMP")))
        chest_ecg = as_float_array(get_nested(payload, ("signal", "chest", "ECG")))
        hr_labels = as_float_array(payload.get("label"))
        activity = as_float_array(payload.get("activity"))
        activity_fs_hz = _infer_activity_fs_hz(
            activity=activity,
            bvp=bvp,
            hr_labels=hr_labels,
        )

        bvp_window = slice_by_time(
            bvp,
            fs_hz=PPG_DALIA_BVP_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        acc_window = slice_by_time(
            wrist_acc,
            fs_hz=PPG_DALIA_WRIST_ACC_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        eda_window = slice_by_time(
            eda,
            fs_hz=PPG_DALIA_WRIST_LOW_RATE_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        temp_window = slice_by_time(
            temp,
            fs_hz=PPG_DALIA_WRIST_LOW_RATE_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        hr_window = slice_by_time(
            hr_labels,
            fs_hz=PPG_DALIA_HR_LABEL_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        activity_window = slice_by_time(
            activity,
            fs_hz=activity_fs_hz,
            start_s=start_s,
            end_s=end_s,
        )

        pulse, caveats = bvp_pulse_summary(
            bvp_window=bvp_window,
            acc_window=acc_window,
            bvp_fs_hz=PPG_DALIA_BVP_FS_HZ,
        )
        motion = motion_summary(acc_window)
        if motion is not None and motion["motion_level"] == "high":
            caveats.append("High wrist acceleration suggests motion artefact risk.")

        chest_ecg_result = None
        if include_chest_ecg_analysis:
            ecg_window = slice_by_time(
                chest_ecg,
                fs_hz=PPG_DALIA_CHEST_FS_HZ,
                start_s=start_s,
                end_s=end_s,
            )
            chest_ecg_result, ecg_caveat = chest_ecg_summary(
                ecg_window,
                ecg_fs_hz=PPG_DALIA_CHEST_FS_HZ,
            )
            if ecg_caveat:
                caveats.append(ecg_caveat)

        return {
            "success": True,
            "dataset": "ppg_dalia",
            "subject_id": resolved_subject,
            "source_path": str(path),
            "data": {
                "window": {
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": duration_s,
                },
                "bvp_pulse": pulse,
                "ecg_reference_hr_label": _hr_label_summary(hr_window),
                "chest_ecg": chest_ecg_result,
                "activity": _activity_summary(activity_window),
                "motion": motion,
                "eda": trend_summary(eda_window, duration_s=duration_s),
                "temperature": trend_summary(temp_window, duration_s=duration_s),
                "caveats": caveats,
            },
            "metadata": {
                "dataset_source": "ppg_dalia",
                "source_variant": "uci_original_pickle_or_original_like",
                "sampling_rates_hz": {
                    "wrist_bvp": PPG_DALIA_BVP_FS_HZ,
                    "wrist_acc": PPG_DALIA_WRIST_ACC_FS_HZ,
                    "wrist_eda": PPG_DALIA_WRIST_LOW_RATE_FS_HZ,
                    "wrist_temp": PPG_DALIA_WRIST_LOW_RATE_FS_HZ,
                    "hr_label": PPG_DALIA_HR_LABEL_FS_HZ,
                    "activity_label": round(float(activity_fs_hz), 6),
                    "chest_ecg": PPG_DALIA_CHEST_FS_HZ,
                },
                "window_sample_counts": {
                    "wrist_bvp": int(bvp_window.shape[0]) if bvp_window is not None else 0,
                    "wrist_acc": int(acc_window.shape[0]) if acc_window is not None else 0,
                    "hr_label": int(hr_window.shape[0]) if hr_window is not None else 0,
                    "activity": int(activity_window.shape[0]) if activity_window is not None else 0,
                },
                "ground_truth_boundary": (
                    "PPG-DaLiA HR labels are ECG-derived reference values. "
                    "Raw BVP pulse metrics are signal-derived estimates and should be "
                    "interpreted with signal quality and motion caveats."
                ),
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "dataset": "ppg_dalia",
            "error": str(exc),
        }


__all__ = ["analyze_ppg_dalia_window_signal"]
