"""In-memory store for mHealth MonitoringState timelines."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from agent.mhealth.schemas import MonitoringState

TimelineKey = tuple[str | None, str | None, str | None]


def _metadata_value(state: MonitoringState, key: str) -> Any:
    value = state.metadata.get(key)
    if value is None:
        value = state.dataset_specific.get(key)
    return value


def _effective_subject_id(state: MonitoringState) -> str | None:
    return (
        state.subject_id
        or _metadata_value(state, "subject_id")
        or state.patient_id
    )


def _effective_recording_id(state: MonitoringState) -> str | None:
    return (
        state.recording_id
        or _metadata_value(state, "recording_id")
        or _metadata_value(state, "record_id")
        or _metadata_value(state, "segment_id")
        or _effective_subject_id(state)
    )


def _effective_window_start_s(state: MonitoringState) -> float | None:
    if state.window_start_s is not None:
        return float(state.window_start_s)
    value = _metadata_value(state, "window_start_s")
    return float(value) if value is not None else None


def _effective_window_end_s(state: MonitoringState) -> float | None:
    if state.window_end_s is not None:
        return float(state.window_end_s)
    value = _metadata_value(state, "window_end_s")
    if value is not None:
        return float(value)
    start_s = _effective_window_start_s(state)
    if start_s is not None and state.window_duration_s is not None:
        return start_s + float(state.window_duration_s)
    return None


def _sort_key(state: MonitoringState) -> tuple[str, str, str, float, int]:
    return (
        state.dataset or "",
        _effective_subject_id(state) or "",
        _effective_recording_id(state) or "",
        float(_effective_window_start_s(state) or 0.0),
        int(state.window_index or 0),
    )


def _timeline_key(state: MonitoringState) -> TimelineKey:
    return (state.dataset, _effective_subject_id(state), _effective_recording_id(state))


def _matches(
    state: MonitoringState,
    dataset: str | None,
    subject_id: str | None,
    recording_id: str | None,
) -> bool:
    if dataset is not None and state.dataset != dataset:
        return False
    if subject_id is not None and _effective_subject_id(state) != subject_id:
        return False
    if recording_id is not None and _effective_recording_id(state) != recording_id:
        return False
    return True


def _overlaps(state: MonitoringState, start_s: float, end_s: float) -> bool:
    state_start = _effective_window_start_s(state)
    state_end = _effective_window_end_s(state)
    if state_start is None or state_end is None:
        return False
    return _intervals_overlap(start_s, end_s, state_start, state_end)


def _intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return start_a < end_b and end_a > start_b


def _duration_s(state: MonitoringState) -> float:
    if state.window_duration_s is not None:
        return float(state.window_duration_s)
    start_s = _effective_window_start_s(state)
    end_s = _effective_window_end_s(state)
    if start_s is not None and end_s is not None:
        return float(end_s - start_s)
    return 0.0


class MHealthStateStore:
    """Store and query MonitoringState timelines across mHealth datasets."""

    def __init__(self) -> None:
        self._states_by_id: dict[str, MonitoringState] = {}
        self._timelines: dict[TimelineKey, list[MonitoringState]] = {}

    def clear(self) -> None:
        self._states_by_id.clear()
        self._timelines.clear()

    def add_states(self, states: Iterable[MonitoringState]) -> None:
        for state in sorted(states, key=_sort_key):
            self._states_by_id[state.state_id] = state

        self._rebuild_timelines()

    def load_jsonl(self, path: str | Path) -> int:
        loaded: list[MonitoringState] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                loaded.append(MonitoringState.from_dict(json.loads(line)))
        self.add_states(loaded)
        return len(loaded)

    def list_contexts(self) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        for (dataset, subject_id, recording_id), states in sorted(
            self._timelines.items(),
            key=lambda item: (
                item[0][0] or "",
                item[0][1] or "",
                item[0][2] or "",
            ),
        ):
            if not states:
                continue
            starts = [
                _effective_window_start_s(s)
                for s in states
                if _effective_window_start_s(s) is not None
            ]
            ends = [
                _effective_window_end_s(s)
                for s in states
                if _effective_window_end_s(s) is not None
            ]
            modalities = [s.modality for s in states if s.modality is not None]
            contexts.append(
                {
                    "dataset": dataset,
                    "subject_id": subject_id,
                    "recording_id": recording_id,
                    "modality": modalities[0] if modalities else None,
                    "state_count": len(states),
                    "start_s": min(starts) if starts else None,
                    "end_s": max(ends) if ends else None,
                }
            )
        return contexts

    def get_state_by_id(self, state_id: str) -> MonitoringState | None:
        return self._states_by_id.get(state_id)

    def get_current_state(
        self,
        dataset: str | None = None,
        subject_id: str | None = None,
        recording_id: str | None = None,
        timestamp_s: float | None = None,
    ) -> MonitoringState | None:
        states = self._matching_states(dataset, subject_id, recording_id)
        if not states:
            return None

        if timestamp_s is None:
            return states[-1]

        previous: MonitoringState | None = None
        for state in states:
            state_start = _effective_window_start_s(state)
            state_end = _effective_window_end_s(state)
            if state_start is None or state_end is None:
                continue
            if state_start <= timestamp_s < state_end:
                return state
            if state_start <= timestamp_s:
                previous = state
            elif state_start > timestamp_s:
                break
        return previous

    def get_window(
        self,
        dataset: str | None,
        subject_id: str | None,
        recording_id: str | None,
        start_s: float,
        end_s: float,
    ) -> list[MonitoringState]:
        if end_s <= start_s:
            return []
        return [
            state
            for state in self._matching_states(dataset, subject_id, recording_id)
            if _overlaps(state, start_s, end_s)
        ]

    def get_previous_state(
        self,
        state_id: str,
        offset: int = 1,
    ) -> MonitoringState | None:
        if offset < 1:
            offset = 1
        state = self.get_state_by_id(state_id)
        if state is None:
            return None
        timeline = self._timelines.get(_timeline_key(state), [])
        for index, candidate in enumerate(timeline):
            if candidate.state_id == state_id:
                previous_index = index - offset
                if previous_index >= 0:
                    return timeline[previous_index]
                return None
        return None

    def summarize_trend(
        self,
        dataset: str | None,
        subject_id: str | None,
        recording_id: str | None,
        start_s: float,
        end_s: float,
        target: str,
    ) -> dict[str, Any]:
        states = self.get_window(dataset, subject_id, recording_id, start_s, end_s)

        if target == "heart_rate_summary":
            mean_or_current_values: list[float] = []
            peak_values: list[float] = []
            for state in states:
                preferred_hr = (
                    state.hr_bpm if state.hr_bpm is not None else state.mean_hr_bpm
                )
                if preferred_hr is not None:
                    mean_or_current_values.append(float(preferred_hr))
                if state.max_hr_bpm is not None:
                    peak_values.append(float(state.max_hr_bpm))
                elif preferred_hr is not None:
                    peak_values.append(float(preferred_hr))

            max_mean_or_current_hr_bpm = (
                max(mean_or_current_values) if mean_or_current_values else None
            )
            return {
                "mean": (
                    sum(mean_or_current_values) / len(mean_or_current_values)
                    if mean_or_current_values
                    else None
                ),
                "min": min(mean_or_current_values) if mean_or_current_values else None,
                "max": max_mean_or_current_hr_bpm,
                "count": len(mean_or_current_values),
                "max_mean_or_current_hr_bpm": max_mean_or_current_hr_bpm,
                "max_observed_hr_bpm": max(peak_values) if peak_values else None,
                "peak_count": len(peak_values),
            }

        if target == "af_burden":
            ratios = [
                float(state.af_burden_ratio)
                for state in states
                if state.af_burden_ratio is not None
            ]
            summary: dict[str, Any] = {}
            if ratios:
                summary.update(
                    {
                        "mean": sum(ratios) / len(ratios),
                        "max": max(ratios),
                        "latest": ratios[-1],
                        "count": len(ratios),
                    }
                )
            rhythm_states = [state for state in states if state.rhythm_class is not None]
            if rhythm_states:
                summary["derived_fraction_af_states"] = sum(
                    1 for state in rhythm_states if state.rhythm_class == "AF"
                ) / len(rhythm_states)
                summary["derived_state_count"] = len(rhythm_states)
            return summary

        if target == "rhythm_transition_count":
            return {"count": self._transition_count(states, "rhythm_class")}

        if target == "stress_burden":
            labelled = [state for state in states if state.stress_label is not None]
            ratios = [
                float(state.stress_burden_ratio)
                for state in states
                if state.stress_burden_ratio is not None
            ]
            summary = {
                "derived_fraction_stress_states": (
                    sum(1 for state in labelled if state.stress_label == "stress")
                    / len(labelled)
                    if labelled
                    else None
                ),
                "derived_state_count": len(labelled),
            }
            if ratios:
                summary["stress_burden_ratio"] = {
                    "mean": sum(ratios) / len(ratios),
                    "max": max(ratios),
                    "latest": ratios[-1],
                    "count": len(ratios),
                }
            return summary

        if target == "stress_duration":
            return {
                "duration_s": sum(
                    _duration_s(state)
                    for state in states
                    if state.stress_label == "stress"
                )
            }

        if target == "stress_transition_count":
            return {"count": self._transition_count(states, "stress_label")}

        if target == "activity_distribution":
            return {"counts": self._value_counts(states, "activity_label")}

        raise ValueError(f"Unsupported trend target: {target}")

    def _rebuild_timelines(self) -> None:
        timelines: dict[TimelineKey, list[MonitoringState]] = {}
        for state in sorted(self._states_by_id.values(), key=_sort_key):
            timelines.setdefault(_timeline_key(state), []).append(state)
        self._timelines = timelines

    def _matching_states(
        self,
        dataset: str | None,
        subject_id: str | None,
        recording_id: str | None,
    ) -> list[MonitoringState]:
        states = [
            state
            for state in self._states_by_id.values()
            if _matches(state, dataset, subject_id, recording_id)
        ]
        return sorted(states, key=_sort_key)

    def _transition_count(self, states: list[MonitoringState], field_name: str) -> int:
        values = [
            getattr(state, field_name)
            for state in states
            if getattr(state, field_name) is not None
        ]
        return sum(1 for previous, current in zip(values, values[1:]) if previous != current)

    def _value_counts(
        self,
        states: list[MonitoringState],
        field_name: str,
        *,
        exclude_none: bool = False,
    ) -> dict[str, int]:
        values = []
        for state in states:
            value = getattr(state, field_name)
            if value is None and exclude_none:
                continue
            if value is None:
                continue
            values.append(str(value))
        return dict(Counter(values))


_GLOBAL_STATE_STORE = MHealthStateStore()


def get_global_state_store() -> MHealthStateStore:
    return _GLOBAL_STATE_STORE

