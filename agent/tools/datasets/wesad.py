"""WESAD raw-signal tools for the reactive pipeline."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import get_config
from agent.mhealth.state_builders.wesad_state_builder import (
    IGNORED_WESAD_LABELS,
    WESAD_LABEL_FS_HZ,
    WESAD_PROTOCOL_LABELS,
    WESAD_WRIST_BVP_FS_HZ,
)
from agent.tools.ppg.window_analysis import (
    as_float_array,
    bvp_pulse_summary,
    chest_ecg_summary,
    get_nested,
    motion_summary,
    slice_by_time,
    summary_stats,
    trend_summary,
)
from agent.tools.registry import register_tool


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_WESAD_ROOT = PROJECT_ROOT / "dataset" / "wesad_sample" / "WESAD"

WESAD_WRIST_ACC_FS_HZ = 32
WESAD_WRIST_LOW_RATE_FS_HZ = 4
WESAD_CHEST_FS_HZ = 700


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
        else [get_config().datasets.wesad_root, SAMPLE_WESAD_ROOT]
    )
    candidates = [root / subject / f"{subject}.pkl" for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find WESAD pickle for {subject}. Checked: {checked}")


@lru_cache(maxsize=2)
def _load_wesad_pickle_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    with Path(path_text).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"WESAD pickle did not contain a dict: {path_text}")
    if "signal" not in payload or "label" not in payload:
        raise ValueError(f"WESAD pickle must contain signal and label: {path_text}")
    return payload


def _load_wesad_pickle(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"WESAD pickle not found: {resolved}")
    return _load_wesad_pickle_cached(str(resolved), resolved.stat().st_mtime_ns)


def _label_name(label_id: int) -> str:
    mapped = WESAD_PROTOCOL_LABELS.get(int(label_id))
    if mapped is not None:
        return str(mapped)
    ignored = IGNORED_WESAD_LABELS.get(int(label_id))
    if ignored is not None:
        return str(ignored)
    return "unknown"


def _protocol_label_summary(
    label_window: np.ndarray | None,
    *,
    min_label_fraction: float,
) -> dict[str, Any] | None:
    if label_window is None or label_window.size == 0:
        return None

    labels = np.asarray(label_window, dtype=int).reshape(-1)
    values, counts = np.unique(labels, return_counts=True)
    best = int(np.argmax(counts))
    label_id = int(values[best])
    fraction = float(counts[best]) / float(labels.size)
    allowed = label_id in WESAD_PROTOCOL_LABELS
    return {
        "dominant_protocol_label_id": label_id,
        "dominant_protocol_label": _label_name(label_id),
        "dominant_label_fraction": round(fraction, 4),
        "is_clean_protocol_window": bool(allowed and fraction >= min_label_fraction),
        "min_label_fraction": min_label_fraction,
        "sample_count": int(labels.size),
        "label_counts": {
            _label_name(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        },
        "label_sampling_rate_hz": WESAD_LABEL_FS_HZ,
    }


@register_tool(
    name="analyze_wesad_window_signal",
    description=(
        "Analyze raw synchronized WESAD signals for one subject window. Returns "
        "protocol-label purity, wrist BVP pulse metrics, wrist motion, EDA, "
        "temperature, respiration summary, optional chest ECG summary, and caveats."
    ),
    parameters={
        "subject_id": {
            "type": "string",
            "description": "WESAD subject ID, e.g. 'S2'.",
            "required": False,
        },
        "patient_id": {
            "type": "string",
            "description": "Alias for subject_id from an mHealth-QA WindowLocator.",
            "required": False,
        },
        "pickle_path": {
            "type": "string",
            "description": "Optional direct path to one WESAD S{n}.pkl file.",
            "required": False,
        },
        "dataset_root": {
            "type": "string",
            "description": "Optional root containing S{n}/S{n}.pkl subject folders.",
            "required": False,
        },
        "dataset": {
            "type": "string",
            "description": "Optional dataset name from an mHealth-QA WindowLocator; expected 'wesad'.",
            "required": False,
        },
        "window_start_s": {
            "type": "number",
            "description": "Window start time in seconds from the synchronized WESAD timeline.",
            "required": True,
        },
        "window_end_s": {
            "type": "number",
            "description": "Window end time in seconds from the synchronized WESAD timeline.",
            "required": True,
        },
        "include_chest_ecg_analysis": {
            "type": "boolean",
            "description": "Also run ECG R-peak analysis on the chest ECG window.",
            "default": False,
        },
        "min_label_fraction": {
            "type": "number",
            "description": "Dominant protocol-label fraction required to mark a clean window.",
            "default": 0.8,
        },
    },
)
def analyze_wesad_window_signal(
    subject_id: str | None = None,
    patient_id: str | None = None,
    pickle_path: str | None = None,
    dataset_root: str | None = None,
    dataset: str | None = None,
    window_start_s: float | None = None,
    window_end_s: float | None = None,
    include_chest_ecg_analysis: bool = False,
    min_label_fraction: float = 0.8,
) -> dict[str, Any]:
    try:
        if dataset is not None and dataset != "wesad":
            raise ValueError(f"Unsupported dataset for WESAD raw tool: {dataset!r}")
        if window_start_s is None or window_end_s is None:
            raise ValueError("window_start_s and window_end_s are required.")
        start_s = float(window_start_s)
        end_s = float(window_end_s)
        if end_s <= start_s:
            raise ValueError("window_end_s must be greater than window_start_s.")
        min_label_fraction = float(min_label_fraction)
        if not 0 < min_label_fraction <= 1:
            raise ValueError("min_label_fraction must be in (0, 1].")

        path = _resolve_pickle_path(
            pickle_path=pickle_path,
            subject_id=subject_id,
            patient_id=patient_id,
            dataset_root=dataset_root,
        )
        payload = _load_wesad_pickle(path)
        resolved_subject = str(payload.get("subject") or path.stem)
        duration_s = end_s - start_s

        labels = as_float_array(payload.get("label"))
        wrist_bvp = as_float_array(get_nested(payload, ("signal", "wrist", "BVP")))
        wrist_acc = as_float_array(get_nested(payload, ("signal", "wrist", "ACC")))
        wrist_eda = as_float_array(get_nested(payload, ("signal", "wrist", "EDA")))
        wrist_temp = as_float_array(get_nested(payload, ("signal", "wrist", "TEMP")))
        chest_ecg = as_float_array(get_nested(payload, ("signal", "chest", "ECG")))
        chest_resp = as_float_array(get_nested(payload, ("signal", "chest", "Resp")))

        label_window = slice_by_time(
            labels,
            fs_hz=WESAD_LABEL_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        bvp_window = slice_by_time(
            wrist_bvp,
            fs_hz=WESAD_WRIST_BVP_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        acc_window = slice_by_time(
            wrist_acc,
            fs_hz=WESAD_WRIST_ACC_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        eda_window = slice_by_time(
            wrist_eda,
            fs_hz=WESAD_WRIST_LOW_RATE_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        temp_window = slice_by_time(
            wrist_temp,
            fs_hz=WESAD_WRIST_LOW_RATE_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )
        resp_window = slice_by_time(
            chest_resp,
            fs_hz=WESAD_CHEST_FS_HZ,
            start_s=start_s,
            end_s=end_s,
        )

        bvp_pulse, caveats = bvp_pulse_summary(
            bvp_window=bvp_window,
            acc_window=acc_window,
            bvp_fs_hz=WESAD_WRIST_BVP_FS_HZ,
        )
        motion = motion_summary(acc_window)
        if motion is not None and motion["motion_level"] == "high":
            caveats.append("High wrist acceleration suggests motion artefact risk.")

        chest_ecg_result = None
        if include_chest_ecg_analysis:
            ecg_window = slice_by_time(
                chest_ecg,
                fs_hz=WESAD_CHEST_FS_HZ,
                start_s=start_s,
                end_s=end_s,
            )
            chest_ecg_result, ecg_caveat = chest_ecg_summary(
                ecg_window,
                ecg_fs_hz=WESAD_CHEST_FS_HZ,
            )
            if ecg_caveat:
                caveats.append(ecg_caveat)

        caveats.append(
            "WESAD protocol labels are lab condition labels, not clinical stress diagnoses."
        )

        return {
            "success": True,
            "dataset": "wesad",
            "subject_id": resolved_subject,
            "source_path": str(path),
            "data": {
                "window": {
                    "start_s": start_s,
                    "end_s": end_s,
                    "duration_s": duration_s,
                },
                "protocol_label": _protocol_label_summary(
                    label_window,
                    min_label_fraction=min_label_fraction,
                ),
                "bvp_pulse": bvp_pulse,
                "motion": motion,
                "eda": trend_summary(eda_window, duration_s=duration_s),
                "temperature": trend_summary(temp_window, duration_s=duration_s),
                "respiration": summary_stats(resp_window),
                "chest_ecg": chest_ecg_result,
                "caveats": caveats,
            },
            "metadata": {
                "dataset_source": "wesad",
                "source_variant": "wesad_subject_pickle",
                "sampling_rates_hz": {
                    "protocol_label": WESAD_LABEL_FS_HZ,
                    "wrist_bvp": WESAD_WRIST_BVP_FS_HZ,
                    "wrist_acc": WESAD_WRIST_ACC_FS_HZ,
                    "wrist_eda": WESAD_WRIST_LOW_RATE_FS_HZ,
                    "wrist_temp": WESAD_WRIST_LOW_RATE_FS_HZ,
                    "chest_ecg": WESAD_CHEST_FS_HZ,
                    "chest_resp": WESAD_CHEST_FS_HZ,
                },
                "window_sample_counts": {
                    "protocol_label": int(label_window.shape[0]) if label_window is not None else 0,
                    "wrist_bvp": int(bvp_window.shape[0]) if bvp_window is not None else 0,
                    "wrist_acc": int(acc_window.shape[0]) if acc_window is not None else 0,
                    "wrist_eda": int(eda_window.shape[0]) if eda_window is not None else 0,
                    "wrist_temp": int(temp_window.shape[0]) if temp_window is not None else 0,
                    "chest_resp": int(resp_window.shape[0]) if resp_window is not None else 0,
                },
                "ground_truth_boundary": (
                    "The protocol label is dataset ground truth for lab condition. "
                    "BVP, EDA, motion, temperature, respiration, and ECG summaries "
                    "are signal-derived context, not independent stress diagnoses."
                ),
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "dataset": "wesad",
            "error": str(exc),
        }


__all__ = ["analyze_wesad_window_signal"]
