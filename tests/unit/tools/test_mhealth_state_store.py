from __future__ import annotations

import json

import pytest

from agent.mhealth.schemas import MonitoringState
from agent.state.mhealth_state_store import MHealthStateStore


def _state(
    state_id: str,
    dataset: str,
    subject_id: str,
    recording_id: str,
    start_s: float,
    end_s: float,
    *,
    modality: str = "ecg",
    window_index: int = 0,
    **kwargs,
) -> MonitoringState:
    return MonitoringState(
        state_id=state_id,
        patient_id=subject_id,
        dataset=dataset,
        modality=modality,
        subject_id=subject_id,
        recording_id=recording_id,
        window_index=window_index,
        window_start_s=start_s,
        window_end_s=end_s,
        window_duration_s=end_s - start_s,
        **kwargs,
    )


@pytest.fixture
def synthetic_states() -> list[MonitoringState]:
    return [
        _state(
            "ic-0",
            "icentia11k",
            "p001",
            "seg00",
            0,
            30,
            window_index=0,
            hr_bpm=70,
            rhythm_class="N",
        ),
        _state(
            "ic-1",
            "icentia11k",
            "p001",
            "seg00",
            30,
            60,
            window_index=1,
            hr_bpm=80,
            mean_hr_bpm=95,
            rhythm_class="AF",
            af_burden_ratio=0.5,
        ),
        _state(
            "ic-2",
            "icentia11k",
            "p001",
            "seg00",
            60,
            90,
            window_index=2,
            hr_bpm=90,
            rhythm_class="AF",
            af_burden_ratio=1.0,
        ),
        _state(
            "wesad-0",
            "wesad",
            "S11",
            "S11",
            0,
            30,
            modality="wearable",
            stress_label="baseline",
            protocol_label_id=1,
        ),
        _state(
            "wesad-1",
            "wesad",
            "S11",
            "S11",
            30,
            60,
            modality="wearable",
            window_index=1,
            stress_label="stress",
            protocol_label_id=2,
            stress_burden_ratio=1.0,
        ),
        _state(
            "dalia-0",
            "ppg_dalia",
            "S1",
            "S1",
            0,
            30,
            modality="ppg",
            hr_bpm=72,
            activity_label="sitting",
        ),
        _state(
            "dalia-1",
            "ppg_dalia",
            "S1",
            "S1",
            30,
            60,
            modality="ppg",
            window_index=1,
            hr_bpm=98,
            activity_label="walking",
        ),
    ]


@pytest.fixture
def store(synthetic_states: list[MonitoringState]) -> MHealthStateStore:
    store = MHealthStateStore()
    store.add_states(reversed(synthetic_states))
    return store


def test_add_states_stores_states(store: MHealthStateStore) -> None:
    assert len(store.list_contexts()) == 3
    assert store.get_state_by_id("ic-0") is not None


def test_load_jsonl_roundtrip(
    tmp_path,
    synthetic_states: list[MonitoringState],
) -> None:
    path = tmp_path / "states.jsonl"
    path.write_text(
        "\n".join(json.dumps(state.to_dict()) for state in synthetic_states) + "\n",
        encoding="utf-8",
    )
    store = MHealthStateStore()

    count = store.load_jsonl(path)

    assert count == len(synthetic_states)
    assert store.get_state_by_id("dalia-1").activity_label == "walking"


def test_load_jsonl_warns_on_unknown_monitoring_state_fields(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    synthetic_states: list[MonitoringState],
) -> None:
    path = tmp_path / "states.jsonl"
    payload = synthetic_states[0].to_dict()
    payload["new_builder_field"] = "not-yet-in-schema"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    store = MHealthStateStore()

    count = store.load_jsonl(path)

    assert count == 1
    assert "Dropping unknown MonitoringState fields" in caplog.text
    assert "new_builder_field" in caplog.text


def test_get_current_state_returns_containing_state(store: MHealthStateStore) -> None:
    state = store.get_current_state("icentia11k", "p001", "seg00", 35)

    assert state is not None
    assert state.state_id == "ic-1"


def test_get_current_state_returns_nearest_previous_state(store: MHealthStateStore) -> None:
    state = store.get_current_state("icentia11k", "p001", "seg00", 95)

    assert state is not None
    assert state.state_id == "ic-2"


def test_get_current_state_returns_latest_without_timestamp(store: MHealthStateStore) -> None:
    state = store.get_current_state("icentia11k", "p001", "seg00")

    assert state is not None
    assert state.state_id == "ic-2"


def test_queries_use_metadata_context_and_duration_when_top_level_fields_missing() -> None:
    state = _state(
        "meta-0",
        "wesad",
        "S11",
        "S11",
        90,
        120,
        modality="wearable",
        stress_label="baseline",
    )
    state.subject_id = None
    state.recording_id = None
    state.window_end_s = None
    state.metadata = {"subject_id": "S11", "window_end_s": 120.0}
    store = MHealthStateStore()
    store.add_states([state])

    current = store.get_current_state("wesad", "S11", "S11", 95)
    contexts = store.list_contexts()
    window = store.get_window("wesad", "S11", "S11", 90, 120)

    assert current is state
    assert contexts[0]["subject_id"] == "S11"
    assert contexts[0]["recording_id"] == "S11"
    assert contexts[0]["end_s"] == 120.0
    assert window == [state]


def test_get_window_returns_overlapping_states(store: MHealthStateStore) -> None:
    states = store.get_window("icentia11k", "p001", "seg00", 25, 65)

    assert [state.state_id for state in states] == ["ic-0", "ic-1", "ic-2"]


def test_get_previous_state_returns_same_timeline_previous(store: MHealthStateStore) -> None:
    state = store.get_previous_state("ic-2")

    assert state is not None
    assert state.state_id == "ic-1"


def test_list_contexts_returns_dataset_subject_recording_summaries(
    store: MHealthStateStore,
) -> None:
    contexts = store.list_contexts()
    dalia = next(context for context in contexts if context["dataset"] == "ppg_dalia")

    assert dalia == {
        "dataset": "ppg_dalia",
        "subject_id": "S1",
        "recording_id": "S1",
        "modality": "ppg",
        "state_count": 2,
        "start_s": 0,
        "end_s": 60,
    }


def test_summarize_trend_heart_rate_summary(store: MHealthStateStore) -> None:
    summary = store.summarize_trend("icentia11k", "p001", "seg00", 0, 90, "heart_rate_summary")

    assert summary == {
        "mean": 80.0,
        "min": 70.0,
        "max": 90.0,
        "count": 3,
        "max_mean_or_current_hr_bpm": 90.0,
        "max_observed_hr_bpm": 90.0,
        "peak_count": 3,
    }


def test_summarize_trend_heart_rate_summary_uses_max_hr_for_observed_peak() -> None:
    store = MHealthStateStore()
    store.add_states(
        [
            _state(
                "hr-peak-0",
                "ppg_dalia",
                "S1",
                "S1",
                0,
                30,
                modality="ppg",
                mean_hr_bpm=70,
                max_hr_bpm=120,
            ),
            _state(
                "hr-peak-1",
                "ppg_dalia",
                "S1",
                "S1",
                30,
                60,
                modality="ppg",
                window_index=1,
                mean_hr_bpm=80,
                max_hr_bpm=110,
            ),
        ]
    )

    summary = store.summarize_trend("ppg_dalia", "S1", "S1", 0, 60, "heart_rate_summary")

    assert summary["mean"] == 75.0
    assert summary["max"] == 80.0
    assert summary["max_mean_or_current_hr_bpm"] == 80.0
    assert summary["max_observed_hr_bpm"] == 120.0
    assert summary["peak_count"] == 2


def test_summarize_trend_heart_rate_summary_peak_falls_back_to_mean_current() -> None:
    store = MHealthStateStore()
    store.add_states(
        [
            _state(
                "hr-fallback-0",
                "ppg_dalia",
                "S1",
                "S1",
                0,
                30,
                modality="ppg",
                mean_hr_bpm=70,
            ),
            _state(
                "hr-fallback-1",
                "ppg_dalia",
                "S1",
                "S1",
                30,
                60,
                modality="ppg",
                window_index=1,
                hr_bpm=84,
                mean_hr_bpm=80,
            ),
        ]
    )

    summary = store.summarize_trend("ppg_dalia", "S1", "S1", 0, 60, "heart_rate_summary")

    assert summary["max"] == 84.0
    assert summary["max_mean_or_current_hr_bpm"] == 84.0
    assert summary["max_observed_hr_bpm"] == 84.0
    assert summary["peak_count"] == 2


def test_summarize_trend_af_burden(store: MHealthStateStore) -> None:
    summary = store.summarize_trend("icentia11k", "p001", "seg00", 0, 90, "af_burden")

    assert summary["mean"] == 0.75
    assert summary["max"] == 1.0
    assert summary["latest"] == 1.0
    assert summary["derived_fraction_af_states"] == pytest.approx(2 / 3)


def test_summarize_trend_stress_burden(store: MHealthStateStore) -> None:
    summary = store.summarize_trend("wesad", "S11", "S11", 0, 60, "stress_burden")

    assert summary["derived_fraction_stress_states"] == 0.5
    assert summary["stress_burden_ratio"]["latest"] == 1.0


def test_summarize_trend_activity_distribution(store: MHealthStateStore) -> None:
    summary = store.summarize_trend("ppg_dalia", "S1", "S1", 0, 60, "activity_distribution")

    assert summary == {"counts": {"sitting": 1, "walking": 1}}

