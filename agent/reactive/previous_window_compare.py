"""Internal temporal orchestration for adjacent raw-signal windows.

This module is deliberately not part of the tool registry. The reactive
pipeline invokes it after planning when the controlled question text requests
a previous-window comparison.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.tools.datasets.icentia11k import analyze_icentia11k_ecg_window_signal
from agent.tools.datasets.ppg_dalia import analyze_ppg_dalia_window_signal
from agent.tools.datasets.wesad import analyze_wesad_window_signal
from agent.mhealth.schemas import canonicalize_dataset_name
from agent.tools.ppg.tools import analyze_ppg_rhythm_irregularity, analyze_pulse_rate

SUPPORTED_DATASETS = ("afppgecg", "ppg_dalia", "icentia11k", "wesad")
HEART_RATE_MEANINGFUL_DELTA_BPM = 5.0
PREVIOUS_WINDOW_COMPARISON_RESULT = "previous_window_comparison"
PREVIOUS_WINDOW_PHRASES = frozenset(
    {
        "previous window",
        "previous clean window",
        "last window",
        "few moments ago",
        "moments ago",
        "recently",
    }
)

METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "afppgecg": (
        "heart_rate.mean_bpm",
        "pp_intervals.mean_ms",
        "irregularity_metrics.coefficient_of_variation",
        "irregularity_metrics.rmssd_ms",
        "assessment",
    ),
    "ppg_dalia": (
        "bvp_pulse.mean_pulse_rate_bpm",
        "eda.mean",
        "motion.motion_level",
    ),
    "icentia11k": (
        "heart_rate.mean_bpm",
        "rr_intervals.cv",
        "rhythm_assessment",
    ),
    "wesad": (
        "bvp_pulse.mean_pulse_rate_bpm",
        "eda.mean",
        "respiration.mean",
        "motion.motion_level",
    ),
}

HEART_RATE_METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "afppgecg": ("heart_rate.mean_bpm",),
    "ppg_dalia": ("bvp_pulse.mean_pulse_rate_bpm",),
    "icentia11k": ("heart_rate.mean_bpm",),
    "wesad": ("bvp_pulse.mean_pulse_rate_bpm",),
}


def _get_path(data: dict[str, Any] | None, path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _with_meaningful_hr_change(entry: dict[str, Any]) -> dict[str, Any]:
    delta = float(entry["delta"])
    enriched = dict(entry)
    enriched["min_meaningful_delta_bpm"] = HEART_RATE_MEANINGFUL_DELTA_BPM
    enriched["meaningfully_higher"] = delta >= HEART_RATE_MEANINGFUL_DELTA_BPM
    enriched["meaningfully_lower"] = delta <= -HEART_RATE_MEANINGFUL_DELTA_BPM
    enriched["changed_meaningfully"] = (
        enriched["meaningfully_higher"] or enriched["meaningfully_lower"]
    )
    if enriched["meaningfully_higher"]:
        direction = "higher"
    elif enriched["meaningfully_lower"]:
        direction = "lower"
    else:
        direction = "not_meaningfully_changed"
    enriched["meaningful_direction"] = direction
    return enriched


def _comparison(
    current: dict[str, Any],
    previous: dict[str, Any],
    dataset: str,
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    hr_paths = set(HEART_RATE_METRIC_PATHS.get(dataset, ()))
    for path in METRIC_PATHS[dataset]:
        current_value = _get_path(current, path)
        previous_value = _get_path(previous, path)
        if current_value is None or previous_value is None:
            continue
        if _is_number(current_value) and _is_number(previous_value):
            delta = float(current_value) - float(previous_value)
            entry = {
                "current": current_value,
                "previous": previous_value,
                "delta": delta,
                "abs_delta": abs(delta),
                "direction": (
                    "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"
                ),
                "changed": delta != 0,
            }
            entries[path] = _with_meaningful_hr_change(entry) if path in hr_paths else entry
        elif isinstance(current_value, (str, bool)) and isinstance(
            previous_value, (str, bool)
        ):
            entries[path] = {
                "current": current_value,
                "previous": previous_value,
                "delta": None,
                "abs_delta": None,
                "direction": None,
                "changed": current_value != previous_value,
            }
    return entries


def _heart_rate_change_interpretation(
    comparison: dict[str, Any],
    dataset: str,
) -> dict[str, Any] | None:
    for path in HEART_RATE_METRIC_PATHS.get(dataset, ()):
        entry = comparison.get(path)
        if not isinstance(entry, dict) or "min_meaningful_delta_bpm" not in entry:
            continue
        delta = float(entry["delta"])
        threshold = float(entry["min_meaningful_delta_bpm"])
        if entry["meaningfully_higher"]:
            summary = (
                f"Heart rate is meaningfully higher by {delta:.1f} bpm "
                f"(threshold: at least {threshold:.1f} bpm)."
            )
        elif entry["meaningfully_lower"]:
            summary = (
                f"Heart rate is meaningfully lower by {abs(delta):.1f} bpm "
                f"(threshold: at least {threshold:.1f} bpm)."
            )
        else:
            summary = (
                f"Heart rate changed by {delta:.1f} bpm, which is below the "
                f"{threshold:.1f} bpm threshold for a meaningful change."
            )
        return {
            "metric_path": path,
            "current_bpm": entry["current"],
            "previous_bpm": entry["previous"],
            "delta_bpm": delta,
            "raw_direction": entry["direction"],
            "min_meaningful_delta_bpm": threshold,
            "meaningfully_higher": entry["meaningfully_higher"],
            "meaningfully_lower": entry["meaningfully_lower"],
            "changed_meaningfully": entry["changed_meaningfully"],
            "meaningful_direction": entry["meaningful_direction"],
            "summary": summary,
        }
    return None


def _afppgecg(dataset: str, patient_id: str, start_s: float, end_s: float) -> dict[str, Any]:
    args = {
        "dataset": dataset,
        "patient_id": patient_id,
        "window_start_s": start_s,
        "window_end_s": end_s,
    }
    pulse = analyze_pulse_rate(**args)
    if not pulse.get("success"):
        return pulse
    rhythm = analyze_ppg_rhythm_irregularity(**args)
    if not rhythm.get("success"):
        return rhythm
    return {
        "success": True,
        "data": {
            **dict(pulse.get("data") or {}),
            **dict(rhythm.get("data") or {}),
        },
    }


def _ppg_dalia(dataset: str, patient_id: str, start_s: float, end_s: float) -> dict[str, Any]:
    return analyze_ppg_dalia_window_signal(
        dataset=dataset,
        patient_id=patient_id,
        window_start_s=start_s,
        window_end_s=end_s,
    )


def _call_analyzer(
    func: Callable[..., dict[str, Any]],
    dataset: str,
    patient_id: str,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    return func(
        dataset=dataset,
        patient_id=patient_id,
        window_start_s=start_s,
        window_end_s=end_s,
    )


def _icentia11k(
    dataset: str,
    patient_id: str,
    start_s: float,
    end_s: float,
    *,
    recording_id: str | None = None,
) -> dict[str, Any]:
    return analyze_icentia11k_ecg_window_signal(
        dataset=dataset,
        patient_id=patient_id,
        window_start_s=start_s,
        window_end_s=end_s,
        recording_id=recording_id,
        segment_id=recording_id,
    )


DISPATCHERS: dict[str, Callable[..., dict[str, Any]]] = {
    "afppgecg": _afppgecg,
    "ppg_dalia": _ppg_dalia,
    "icentia11k": _icentia11k,
    "wesad": lambda dataset, patient_id, start_s, end_s: _call_analyzer(
        analyze_wesad_window_signal,
        dataset,
        patient_id,
        start_s,
        end_s,
    ),
}


def _empty_response(
    dataset: str,
    patient_id: str,
    start_s: float,
    end_s: float,
    offset: int,
    recording_id: str | None,
) -> dict[str, Any]:
    return {
        "success": True,
        "data": {
            "current": None,
            "previous": None,
            "comparison": {},
            "heart_rate_change_interpretation": None,
            "previous_window_locator": None,
            "caveats": [],
        },
        "metadata": {
            "dataset": dataset,
            "patient_id": patient_id,
            "recording_id": recording_id,
            "current_window": {"window_start_s": start_s, "window_end_s": end_s},
            "previous_window_offset": offset,
        },
        "error": "",
    }


def _run_dispatcher(
    dataset: str,
    patient_id: str,
    start_s: float,
    end_s: float,
    *,
    recording_id: str | None,
) -> dict[str, Any]:
    dispatcher = DISPATCHERS[dataset]
    if recording_id is None:
        return dispatcher(dataset, patient_id, start_s, end_s)
    try:
        return dispatcher(
            dataset,
            patient_id,
            start_s,
            end_s,
            recording_id=recording_id,
        )
    except TypeError:
        return dispatcher(dataset, patient_id, start_s, end_s)


def needs_previous_window_comparison(question_text: str) -> bool:
    """Return whether a controlled VitalBench question asks for the previous window."""
    normalized = str(question_text or "").lower()
    return any(phrase in normalized for phrase in PREVIOUS_WINDOW_PHRASES)


def compare_with_previous_window(
    dataset: str,
    patient_id: str,
    window_start_s: float,
    window_end_s: float,
    previous_window_offset: int = 1,
    recording_id: str | None = None,
    current_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare one locator window with the preceding equal-duration window."""
    dataset = canonicalize_dataset_name(dataset)
    start_s = float(window_start_s)
    end_s = float(window_end_s)
    offset = int(previous_window_offset)
    response = _empty_response(dataset, patient_id, start_s, end_s, offset, recording_id)
    if dataset not in DISPATCHERS:
        response["success"] = False
        response["error"] = (
            f"Previous-window comparison does not support dataset '{dataset}' "
            f"(only {', '.join(SUPPORTED_DATASETS)})."
        )
        return response
    if end_s <= start_s or offset < 1:
        response["success"] = False
        response["error"] = (
            "window_end_s must be greater than window_start_s and "
            "previous_window_offset must be >= 1."
        )
        return response

    if current_data is None:
        current_result = _run_dispatcher(
            dataset,
            patient_id,
            start_s,
            end_s,
            recording_id=recording_id,
        )
        if not current_result.get("success"):
            response["success"] = False
            response["error"] = str(
                current_result.get("error") or "current window analyzer failed"
            )
            return response
        current_data = dict(current_result.get("data") or {})

    response["data"]["current"] = dict(current_data)
    width = end_s - start_s
    previous_start = start_s - width * offset
    previous_end = start_s
    if previous_start < 0:
        response["data"]["caveats"].append(
            "Previous window range starts before recording origin "
            f"(start_s={previous_start:.1f}); only the current window was analyzed."
        )
        return response

    previous_result = _run_dispatcher(
        dataset,
        patient_id,
        previous_start,
        previous_end,
        recording_id=recording_id,
    )
    response["data"]["previous_window_locator"] = {
        "window_start_s": previous_start,
        "window_end_s": previous_end,
    }
    if not previous_result.get("success"):
        error = str(previous_result.get("error") or "unknown error")
        response["data"]["caveats"].append(f"previous window analyzer failed: {error}")
        return response

    previous_data = dict(previous_result.get("data") or {})
    response["data"]["previous"] = previous_data
    comparison = _comparison(current_data, previous_data, dataset)
    response["data"]["comparison"] = comparison
    response["data"]["heart_rate_change_interpretation"] = (
        _heart_rate_change_interpretation(comparison, dataset)
    )
    return response


__all__ = [
    "compare_with_previous_window",
    "DISPATCHERS",
    "HEART_RATE_MEANINGFUL_DELTA_BPM",
    "METRIC_PATHS",
    "PREVIOUS_WINDOW_COMPARISON_RESULT",
    "PREVIOUS_WINDOW_PHRASES",
    "SUPPORTED_DATASETS",
    "needs_previous_window_comparison",
]
