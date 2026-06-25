"""Builder-wrapper tools for loading local mHealth datasets into StateStore."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.mhealth.state_builders.wesad_state_builder import (
    WESADWindowConfig,
    build_monitoring_states_from_wesad_subject,
)
from agent.mhealth.state_builders.ppg_dalia_state_builder import (
    PPGDaLiAWindowConfig,
    build_monitoring_states_from_ppg_dalia_subject,
)
from agent.state.mhealth_state_store import get_global_state_store
from agent.tools.registry import register_tool


def _store_states(
    *,
    states: list[Any],
    clear_existing: bool,
) -> dict[str, Any]:
    store = get_global_state_store()
    if clear_existing:
        store.clear()
    store.add_states(states)
    return {
        "state_count": len(states),
        "contexts": store.list_contexts(),
    }


@register_tool(
    name="state_build_from_wesad_pickle",
    description="Build WESAD MonitoringState records from a local subject pickle.",
    parameters={
        "pickle_path": {
            "type": "string",
            "description": "Path to a WESAD subject .pkl file.",
            "required": True,
        },
        "clear_existing": {
            "type": "boolean",
            "description": "Clear already-loaded states before adding generated states.",
            "default": False,
        },
        "window_secs": {
            "type": "array",
            "description": "Clean-window durations to build in seconds.",
            "default": [30, 60],
        },
        "max_windows": {
            "type": "integer",
            "description": "Optional maximum windows per duration.",
            "required": False,
        },
    },
)
def state_build_from_wesad_pickle(
    pickle_path: str,
    clear_existing: bool = False,
    window_secs: list[int] | None = None,
    max_windows: int | None = None,
) -> dict[str, Any]:
    try:
        path = Path(pickle_path)
        durations = tuple(int(duration) for duration in (window_secs or [30, 60]))
        config = WESADWindowConfig(
            window_secs=durations,
            max_windows_per_duration=max_windows,
        )
        states = build_monitoring_states_from_wesad_subject(path, config=config)
        result = _store_states(states=states, clear_existing=clear_existing)
        return {
            "success": True,
            "dataset": "wesad",
            "source_path": str(path),
            **result,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@register_tool(
    name="state_build_from_ppg_dalia_pickle",
    description="Build PPG-DaLiA MonitoringState records from a local subject pickle.",
    parameters={
        "pickle_path": {
            "type": "string",
            "description": "Path to one original UCI PPG-DaLiA subject .pkl file.",
            "required": True,
        },
        "clear_existing": {
            "type": "boolean",
            "description": "Clear already-loaded states before adding generated states.",
            "default": False,
        },
        "window_secs": {
            "type": "array",
            "description": "Clean-window durations to build in seconds.",
            "default": [30, 60],
        },
        "max_windows": {
            "type": "integer",
            "description": "Optional maximum windows per duration.",
            "required": False,
        },
        "window_stride_sec": {
            "type": "integer",
            "description": "Optional stride for clean-window generation.",
            "required": False,
        },
        "activity_label_fs_hz": {
            "type": "number",
            "description": "Optional sampling rate for PPG-DaLiA activity labels.",
            "required": False,
        },
        "min_activity_fraction": {
            "type": "number",
            "description": "Minimum dominant protocol-activity fraction for clean windows.",
            "default": 0.8,
        },
        "include_transition_windows": {
            "type": "boolean",
            "description": "Include transition/unknown activity windows.",
            "default": False,
        },
    },
)
def state_build_from_ppg_dalia_pickle(
    pickle_path: str,
    clear_existing: bool = False,
    window_secs: list[int] | None = None,
    max_windows: int | None = None,
    window_stride_sec: int | None = None,
    activity_label_fs_hz: float | None = None,
    min_activity_fraction: float = 0.8,
    include_transition_windows: bool = False,
) -> dict[str, Any]:
    try:
        path = Path(pickle_path)
        durations = tuple(int(duration) for duration in (window_secs or [30, 60]))
        config = PPGDaLiAWindowConfig(
            window_secs=durations,
            window_stride_sec=window_stride_sec,
            activity_label_fs_hz=activity_label_fs_hz,
            min_activity_fraction=float(min_activity_fraction),
            max_windows_per_duration=max_windows,
            include_transition_windows=bool(include_transition_windows),
        )
        states = build_monitoring_states_from_ppg_dalia_subject(path, config=config)
        result = _store_states(states=states, clear_existing=clear_existing)
        return {
            "success": True,
            "dataset": "ppg_dalia",
            "source_path": str(path),
            **result,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}

