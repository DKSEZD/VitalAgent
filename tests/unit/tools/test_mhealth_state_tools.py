from __future__ import annotations

import json

import pytest

import agent.tools.mhealth_state_tools as state_tools
from agent.mhealth.schemas import MonitoringState
from agent.state.mhealth_state_store import MHealthStateStore, get_global_state_store
from agent.tools.registry import list_tools
from scripts.vitalbench.generate_vitalbench import write_states_jsonl


def _state(
    state_id: str,
    dataset: str,
    subject_id: str,
    recording_id: str,
    start_s: float,
    end_s: float,
    *,
    modality: str,
    window_index: int,
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
def states() -> list[MonitoringState]:
    return [
        _state(
            "ppg-0",
            "afppgecg",
            "001",
            "001",
            0,
            30,
            modality="ppg",
            window_index=0,
            hr_bpm=70,
            max_hr_bpm=104,
            rhythm_class="N",
        ),
        _state(
            "ppg-1",
            "afppgecg",
            "001",
            "001",
            30,
            60,
            modality="ppg",
            window_index=1,
            hr_bpm=88,
            max_hr_bpm=118,
            rhythm_class="AF",
            af_burden_ratio=1.0,
            dataset_specific={"quality": "synthetic"},
            metadata={"annotation_source": "synthetic", "builder": "unit-test"},
        ),
        _state(
            "wesad-0",
            "wesad",
            "S11",
            "S11",
            0,
            30,
            modality="wearable",
            window_index=0,
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
        ),
        _state(
            "dalia-0",
            "ppg_dalia",
            "S1",
            "S1",
            0,
            30,
            modality="ppg",
            window_index=0,
            hr_bpm=75,
            activity_label="sitting",
        ),
    ]


@pytest.fixture(autouse=True)
def clear_global_store():
    get_global_state_store().clear()
    yield
    get_global_state_store().clear()


@pytest.fixture
def loaded_store(states: list[MonitoringState]) -> None:
    get_global_state_store().add_states(states)


def test_state_tools_are_registered() -> None:
    names = {tool["name"] for tool in list_tools()}

    assert {
        "state_load_monitoring_states",
        "state_list_contexts",
        "state_get_current_monitoring_state",
        "state_get_monitoring_window",
        "state_get_previous_monitoring_state",
        "state_get_longitudinal_trend",
        "state_get_evidence",
        "state_get_dataset_capabilities",
    }.issubset(names)


def test_state_load_monitoring_states_loads_jsonl(
    tmp_path,
    states: list[MonitoringState],
) -> None:
    path = tmp_path / "states.jsonl"
    path.write_text(
        "\n".join(json.dumps(state.to_dict()) for state in states) + "\n",
        encoding="utf-8",
    )

    result = state_tools.state_load_monitoring_states(str(path), clear_existing=True)

    assert result["success"] is True
    assert result["state_count"] == len(states)
    assert len(result["contexts"]) == 3


def test_state_list_contexts_returns_contexts(loaded_store: None) -> None:
    result = state_tools.state_list_contexts()

    assert result["success"] is True
    assert any(context["dataset"] == "ppg_dalia" for context in result["contexts"])


def test_state_get_current_monitoring_state_returns_compact_state_and_evidence(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_current_monitoring_state(
        dataset="afppgecg",
        subject_id="001",
        recording_id="001",
        timestamp_s=35,
        fields=["hr_bpm"],
    )

    assert result["success"] is True
    assert result["state"]["state_id"] == "ppg-1"
    assert result["state"]["hr_bpm"] == 88
    assert "metadata" not in result["state"]
    assert result["state"]["metadata_summary"] == {
        "metadata_keys": ["annotation_source", "builder"],
        "metadata_field_count": 2,
        "dataset_specific_keys": ["quality"],
        "dataset_specific_field_count": 1,
    }
    assert result["evidence"]["window_start_s"] == 30


def test_state_tools_accept_window_locator_aliases(loaded_store: None) -> None:
    current = state_tools.state_get_current_monitoring_state(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=30.0,
    )
    window = state_tools.state_get_monitoring_window(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=30.0,
        window_end_s=60.0,
    )
    trend = state_tools.state_get_longitudinal_trend(
        dataset="afppgecg",
        patient_id="001",
        window_start_s=30.0,
        window_end_s=60.0,
        target="heart_rate_summary",
    )

    assert current["success"] is True
    assert current["state"]["state_id"] == "ppg-1"
    assert window["success"] is True
    assert window["states"][0]["state_id"] == "ppg-1"
    assert trend["success"] is True
    assert trend["summary"]["count"] == 1


def test_state_get_current_monitoring_state_can_return_full_metadata_when_requested(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_current_monitoring_state(
        dataset="afppgecg",
        subject_id="001",
        recording_id="001",
        timestamp_s=35,
        fields=["metadata", "dataset_specific"],
    )

    assert result["success"] is True
    assert result["state"]["metadata"] == {
        "annotation_source": "synthetic",
        "builder": "unit-test",
    }
    assert result["state"]["dataset_specific"] == {"quality": "synthetic"}


def test_state_get_heart_rate_evidence_excludes_ecg_reference_label() -> None:
    state = _state(
        "dalia-reference",
        "ppg_dalia",
        "S1",
        "S1",
        0,
        30,
        modality="ppg",
        window_index=0,
        hr_bpm=74.0,
        mean_hr_bpm=74.0,
        max_hr_bpm=92.0,
        ecg_reference_hr_bpm=80.0,
        signal_quality_score=0.86,
    )
    get_global_state_store().add_states([state])

    result = state_tools.state_get_evidence("dalia-reference", "heart_rate")

    assert result["success"] is True
    assert result["evidence"]["hr_bpm"] == 74.0
    assert result["evidence"]["max_hr_bpm"] == 92.0
    assert "ecg_reference_hr_bpm" not in result["evidence"]


def test_state_get_monitoring_window_respects_limit_and_truncated(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_monitoring_window(
        dataset="afppgecg",
        subject_id="001",
        recording_id="001",
        start_s=0,
        end_s=60,
        limit=1,
        fields=["hr_bpm"],
    )

    assert result["success"] is True
    assert result["state_count"] == 2
    assert result["returned_count"] == 1
    assert result["truncated"] is True
    assert [state["state_id"] for state in result["states"]] == ["ppg-0"]


def test_state_get_monitoring_window_rejects_zero_limit(loaded_store: None) -> None:
    result = state_tools.state_get_monitoring_window(
        dataset="afppgecg",
        subject_id="001",
        recording_id="001",
        start_s=0,
        end_s=60,
        limit=0,
    )

    assert result == {"success": False, "error": "limit must be at least 1."}


def test_state_get_previous_monitoring_state_works(loaded_store: None) -> None:
    result = state_tools.state_get_previous_monitoring_state("ppg-1", fields=["hr_bpm"])

    assert result["success"] is True
    assert result["state"]["state_id"] == "ppg-0"
    assert result["state"]["hr_bpm"] == 70


def test_state_get_longitudinal_trend_returns_expected_summary(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_longitudinal_trend(
        dataset="wesad",
        subject_id="S11",
        recording_id="S11",
        start_s=0,
        end_s=60,
        target="stress_duration",
    )

    assert result["success"] is True
    assert result["summary"] == {"duration_s": 30.0}
    assert result["evidence"]["used_state_count"] == 2


def test_state_get_longitudinal_trend_heart_rate_summary_returns_observed_peak(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_longitudinal_trend(
        dataset="afppgecg",
        subject_id="001",
        recording_id="001",
        window_start_s=0,
        window_end_s=60,
        target="heart_rate_summary",
    )

    assert result["success"] is True
    assert result["summary"]["max"] == 88.0
    assert result["summary"]["max_mean_or_current_hr_bpm"] == 88.0
    assert result["summary"]["max_observed_hr_bpm"] == 118.0
    assert result["summary"]["max_observed_hr_bpm"] > result["summary"]["max"]
    assert result["summary"]["peak_count"] == 2


def test_state_get_longitudinal_trend_fills_evidence_from_matching_state(
    states: list[MonitoringState],
) -> None:
    get_global_state_store().clear()
    get_global_state_store().add_states(
        [state for state in states if state.dataset == "ppg_dalia"]
    )

    result = state_tools.state_get_longitudinal_trend(
        start_s=0,
        end_s=30,
        target="activity_distribution",
    )

    assert result["success"] is True
    assert result["evidence"]["dataset"] == "ppg_dalia"
    assert result["evidence"]["subject_id"] == "S1"
    assert result["evidence"]["recording_id"] == "S1"


def test_state_get_longitudinal_trend_rejects_unsupported_dataset_target(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_longitudinal_trend(
        dataset="ppg_dalia",
        subject_id="S1",
        recording_id="S1",
        start_s=0,
        end_s=30,
        target="af_burden",
    )

    assert result["success"] is False
    assert result["unsupported_target"] == "af_burden"
    assert "does not support target" in result["error"]


def test_state_get_current_monitoring_state_rejects_ambiguous_context(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_current_monitoring_state()

    assert result == {
        "success": False,
        "error": "Ambiguous monitoring context. Provide dataset, subject_id, or recording_id.",
    }


def test_state_get_evidence_returns_target_specific_evidence(
    loaded_store: None,
) -> None:
    result = state_tools.state_get_evidence("dalia-0", "activity")

    assert result["success"] is True
    assert result["evidence"]["activity_label"] == "sitting"


def test_state_get_evidence_includes_ppg_af_caveat(loaded_store: None) -> None:
    result = state_tools.state_get_evidence("ppg-1", "af")

    assert result["success"] is True
    assert "cannot observe P-waves" in result["evidence"]["methodological_caveat"]


def test_state_get_dataset_capabilities_returns_capability_table() -> None:
    result = state_tools.state_get_dataset_capabilities("icentia11k")

    assert result["success"] is True
    assert result["capabilities"]["supports_ecg"] is True
    assert result["capabilities"]["supports_af"] is True
    assert "state_get_evidence(target='af')" in result["recommended_tools"]


def test_state_get_dataset_capabilities_returns_structured_ppg_caveats() -> None:
    result = state_tools.state_get_dataset_capabilities("afppgecg")

    assert result["success"] is True
    assert result["capabilities"]["caveats"]["af"].endswith("cannot observe P-waves.")
    assert result["capabilities"]["caveats"]["rhythm"].endswith("cannot observe P-waves.")


def test_state_get_dataset_capabilities_unknown_dataset_degrades_to_empty() -> None:
    result = state_tools.state_get_dataset_capabilities("unknown_dataset")

    assert result["success"] is True
    assert result["capabilities"] == {}
    assert result["recommended_tools"] == ["state_list_contexts"]
    assert "Unknown dataset" in result["hint"]


def test_ppg_dalia_capabilities_do_not_support_af_or_stress() -> None:
    result = state_tools.state_get_dataset_capabilities("ppg_dalia")

    assert result["capabilities"]["supports_af"] is False
    assert result["capabilities"]["supports_stress"] is False
    assert {"af", "stress"}.issubset(set(result["unsupported_targets"]))


def test_tools_return_clear_error_when_no_state_loaded() -> None:
    result = state_tools.state_get_current_monitoring_state(dataset="ppg_dalia")

    assert result == {"success": False, "error": "No monitoring states are loaded."}


def test_tools_return_clear_error_when_no_match_found(loaded_store: None) -> None:
    result = state_tools.state_get_current_monitoring_state(dataset="ppg_dalia", subject_id="missing")

    assert result == {"success": False, "error": "No matching monitoring state found."}


def test_states_jsonl_export_can_be_loaded_by_state_store(
    tmp_path,
    states: list[MonitoringState],
) -> None:
    path = tmp_path / "exported_states.jsonl"

    write_states_jsonl(states, path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    store = MHealthStateStore()
    loaded = store.load_jsonl(path)

    assert records
    assert all("state_id" in record and "window_start_s" in record for record in records)
    assert loaded == len(states)
    assert store.get_state_by_id("dalia-0").activity_label == "sitting"

