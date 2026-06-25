"""Aggregate proactive per-window outputs for VitalBench Tier B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_CONFIRMATION_WINDOWS = 3


@dataclass(frozen=True)
class MonitoringWindowAggregates:
    """
    Monitoring-window aggregate fields predicted from proactive outputs.

    Definitions
    -----------
    af_burden_ratio:
        ``sum(window_duration_s where rhythm_class == "AF") / sum(window_duration_s)``.
        This is intentionally prediction-derived: proactive only sees its own
        per-window ``rhythm_class`` output, not ground-truth rhythm intervals.
        It is therefore not exactly equivalent to the state builder's
        annotation-overlap ``af_burden_ratio`` and should be disclosed as a
        methodological difference when reported.
    max_hr_bpm:
        ``max(vitals.hr_bpm for window in stream if vitals.hr_bpm is not None)``.
    rhythm_transition_count_per_hour:
        Collapse the per-window ``rhythm_class`` sequence by merging repeated
        adjacent labels, count ``len(collapsed) - 1``, then divide by total
        hours in the monitoring window.
    confirmed_af_*:
        Episode-confirmed AF aggregates. A run of AF is counted only when it
        lasts at least ``confirmation_windows`` consecutive windows. The
        default is 3, i.e. 30 seconds for the 10-second proactive windows used
        by Icentia11k.
    debounced_rhythm_transition_count_per_hour:
        Transition rate after the same confirmation rule is applied to rhythm
        labels. This is the guideline-adapter signal used by proactive QA.

    Stress and sleep aggregate fields are present for shape compatibility with
    Tier B concepts, but proactive ECG currently does not infer them and leaves
    them as ``None``.
    """

    af_burden_ratio: float | None = None
    max_hr_bpm: float | None = None
    rhythm_transition_count_per_hour: float | None = None
    confirmed_af_episode_count: int = 0
    confirmed_af_duration_s: float | None = None
    confirmed_af_burden_ratio: float | None = None
    debounced_rhythm_transition_count_per_hour: float | None = None
    confirmation_windows: int = DEFAULT_CONFIRMATION_WINDOWS
    stress_burden_ratio: float | None = None
    stress_duration_s: float | None = None
    dominant_stress_label: str | None = None


def aggregate_monitoring_window(
    window_results: Iterable[dict[str, Any]],
    *,
    confirmation_windows: int = DEFAULT_CONFIRMATION_WINDOWS,
) -> MonitoringWindowAggregates:
    """Aggregate proactive ``process_window`` result dictionaries."""
    results = list(window_results)
    if not results:
        return MonitoringWindowAggregates(confirmation_windows=confirmation_windows)

    durations = _window_durations_s(results)
    total_duration_s = sum(durations)
    if total_duration_s <= 0:
        return MonitoringWindowAggregates(
            max_hr_bpm=_max_hr_bpm(results),
            confirmation_windows=confirmation_windows,
        )

    af_duration_s = sum(
        duration
        for result, duration in zip(results, durations, strict=False)
        if str(result.get("rhythm_class") or "") == "AF"
    )
    transitions = _transition_count(results)
    confirmed_af_episode_count, confirmed_af_duration_s = _confirmed_af_episodes(
        results,
        durations,
        min_consecutive=confirmation_windows,
    )
    debounced_transitions = _debounced_transition_count(
        results,
        min_consecutive=confirmation_windows,
    )
    total_hours = total_duration_s / 3600.0

    return MonitoringWindowAggregates(
        af_burden_ratio=af_duration_s / total_duration_s,
        max_hr_bpm=_max_hr_bpm(results),
        rhythm_transition_count_per_hour=(
            transitions / total_hours
            if total_hours > 0 and transitions is not None
            else None
        ),
        confirmed_af_episode_count=confirmed_af_episode_count,
        confirmed_af_duration_s=confirmed_af_duration_s,
        confirmed_af_burden_ratio=confirmed_af_duration_s / total_duration_s,
        debounced_rhythm_transition_count_per_hour=(
            debounced_transitions / total_hours
            if total_hours > 0 and debounced_transitions is not None
            else None
        ),
        confirmation_windows=confirmation_windows,
    )


def _window_durations_s(results: list[dict[str, Any]]) -> list[float]:
    explicit = [_duration_from_result(result) for result in results]
    if all(duration is not None and duration > 0 for duration in explicit):
        return [float(duration) for duration in explicit if duration is not None]

    offsets = [_offset_from_result(result) for result in results]
    positive_deltas = [
        later - earlier
        for earlier, later in zip(offsets, offsets[1:], strict=False)
        if later is not None and earlier is not None and later > earlier
    ]
    inferred = positive_deltas[-1] if positive_deltas else 0.0
    return [
        float(duration)
        if duration is not None and duration > 0
        else float(inferred)
        for duration in explicit
    ]


def _duration_from_result(result: dict[str, Any]) -> float | None:
    for key in ("window_duration_s", "duration_s"):
        value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    meta = result.get("window_meta")
    if isinstance(meta, dict) and meta.get("window_duration_s") is not None:
        try:
            return float(meta["window_duration_s"])
        except (TypeError, ValueError):
            return None
    return None


def _offset_from_result(result: dict[str, Any]) -> float | None:
    value = result.get("stream_offset_s")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_hr_bpm(results: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for result in results:
        vitals = result.get("vitals")
        if not isinstance(vitals, dict):
            continue
        hr = vitals.get("hr_bpm")
        if hr is None:
            continue
        try:
            values.append(float(hr))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _transition_count(results: list[dict[str, Any]]) -> int | None:
    labels = [
        str(result.get("rhythm_class"))
        for result in results
        if result.get("rhythm_class") not in (None, "", "Unknown")
    ]
    if not labels:
        return None

    collapsed: list[str] = []
    for label in labels:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return max(0, len(collapsed) - 1)


def _known_rhythm_label(result: dict[str, Any]) -> str | None:
    label = result.get("rhythm_class")
    if label in (None, "", "Unknown"):
        return None
    return str(label)


def _confirmed_af_episodes(
    results: list[dict[str, Any]],
    durations: list[float],
    *,
    min_consecutive: int,
) -> tuple[int, float]:
    min_consecutive = max(1, int(min_consecutive))
    episode_count = 0
    duration_s = 0.0
    run_duration_s = 0.0
    run_windows = 0

    for result, duration in zip(results, durations, strict=False):
        if _known_rhythm_label(result) == "AF":
            run_windows += 1
            run_duration_s += duration
            continue

        if run_windows >= min_consecutive:
            episode_count += 1
            duration_s += run_duration_s
        run_windows = 0
        run_duration_s = 0.0

    if run_windows >= min_consecutive:
        episode_count += 1
        duration_s += run_duration_s

    return episode_count, duration_s


def _debounced_transition_count(
    results: list[dict[str, Any]],
    *,
    min_consecutive: int,
) -> int | None:
    labels = [_known_rhythm_label(result) for result in results]
    known_labels = [label for label in labels if label is not None]
    if not known_labels:
        return None

    min_consecutive = max(1, int(min_consecutive))
    confirmed_label: str | None = None
    pending_label: str | None = None
    pending_count = 0
    transitions = 0

    for label in known_labels:
        if confirmed_label is None:
            confirmed_label = label
            continue

        if label == confirmed_label:
            pending_label = None
            pending_count = 0
            continue

        if label == pending_label:
            pending_count += 1
        else:
            pending_label = label
            pending_count = 1

        if pending_count >= min_consecutive:
            transitions += 1
            confirmed_label = label
            pending_label = None
            pending_count = 0

    return transitions
