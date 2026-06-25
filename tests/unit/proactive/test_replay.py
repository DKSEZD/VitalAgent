from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agent.data.streaming import ECGWindow
from agent.proactive.replay import ReplayConfig, replay_windows


def _make_window(window_index: int = 0, fs: int = 250) -> ECGWindow:
    signal = np.zeros(fs * 10, dtype=float)
    return ECGWindow(
        signal=signal,
        fs=fs,
        patient_id="00012",
        segment_id="03",
        window_index=window_index,
        start_sample=window_index * len(signal),
        end_sample=(window_index + 1) * len(signal),
        n_samples=len(signal),
        stream_offset_s=float(window_index * 10),
    )


class _ReplayPipeline:
    def __init__(self, intervention_windows: set[int]) -> None:
        self.intervention_windows = intervention_windows
        self.snapshot_calls = 0
        self.tracker = SimpleNamespace(
            append_daily_snapshot=lambda: self._append_snapshot(),
        )

    def process_window(self, window: ECGWindow) -> dict[str, object]:
        intervene = window.window_index in self.intervention_windows
        return {
            "intervene": intervene,
            "condition_active": intervene,
            "urgency": "medium" if intervene else "none",
            "reason": "test alert" if intervene else "",
            "triggered_rules": ["test_rule"] if intervene else [],
            "vitals": {"hr_bpm": np.float64(130.0) if intervene else np.float64(72.0)},
            "window_index": window.window_index,
            "patient_id": window.patient_id,
            "segment_id": window.segment_id,
            "stream_offset_s": window.stream_offset_s,
            "rhythm_class": "AF" if intervene else "N",
            "rule_layer": {"intervene": intervene},
            "judge_layer": {"intervene": False},
        }

    def _append_snapshot(self) -> None:
        self.snapshot_calls += 1


def test_replay_windows_emits_window_and_intervention_records() -> None:
    pipeline = _ReplayPipeline(intervention_windows={1})
    records: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []

    summary = replay_windows(
        (_make_window(i) for i in range(3)),
        pipeline,
        ReplayConfig(snapshot_interval_windows=2),
        on_record=records.append,
        on_intervention=interventions.append,
    )

    assert summary.windows_processed == 3
    assert summary.intervention_count == 1
    assert summary.active_condition_windows == 1
    assert summary.first_offset_s == 0.0
    assert summary.last_offset_s == 20.0
    assert pipeline.snapshot_calls == 2
    assert [record["record_type"] for record in records] == [
        "window",
        "window",
        "intervention_event",
        "window",
        "patient_context",
    ]
    assert interventions == [records[2]]
    assert interventions[0]["window_index"] == 1
    assert interventions[0]["triggered_rules"] == ["test_rule"]
    assert summary.interventions[0]["reason"] == "test alert"
    assert records[-1]["patient_id"] == "00012"
    assert records[-1]["alert_summary"]["total_alerts"] == 1
    assert records[-1]["recent_alerts"][0]["reason"] == "test alert"


def test_replay_windows_can_emit_interventions_only_to_jsonl(tmp_path) -> None:
    output_jsonl = tmp_path / "replay.jsonl"
    pipeline = _ReplayPipeline(intervention_windows={0, 2})
    records: list[dict[str, object]] = []

    summary = replay_windows(
        (_make_window(i) for i in range(4)),
        pipeline,
        ReplayConfig(
            emit_all_windows=False,
            max_windows=3,
            output_jsonl=output_jsonl,
            append_snapshots=False,
        ),
        on_record=records.append,
    )

    lines = output_jsonl.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]

    assert summary.windows_processed == 3
    assert summary.intervention_count == 2
    assert [record["record_type"] for record in records] == [
        "intervention_event",
        "intervention_event",
        "patient_context",
    ]
    assert [payload["window_index"] for payload in payloads[:2]] == [0, 2]
    assert payloads[0]["vitals"]["hr_bpm"] == 130.0
    assert payloads[-1]["record_type"] == "patient_context"
    assert payloads[-1]["alert_summary"]["total_alerts"] == 2
    assert pipeline.snapshot_calls == 0


def test_replay_windows_can_skip_patient_context_record() -> None:
    records: list[dict[str, object]] = []

    replay_windows(
        (_make_window(i) for i in range(1)),
        _ReplayPipeline(intervention_windows={0}),
        ReplayConfig(emit_patient_context=False),
        on_record=records.append,
    )

    assert [record["record_type"] for record in records] == [
        "window",
        "intervention_event",
    ]


@pytest.mark.parametrize(
    "config",
    [
        ReplayConfig(max_windows=-1),
        ReplayConfig(snapshot_interval_windows=-1),
        ReplayConfig(sleep_per_window_s=-0.1),
        ReplayConfig(realtime_speed=0),
    ],
)
def test_replay_config_rejects_invalid_values(config: ReplayConfig) -> None:
    with pytest.raises(ValueError):
        replay_windows([], _ReplayPipeline(intervention_windows=set()), config)
