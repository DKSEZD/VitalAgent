from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from agent.mhealth.state_builders.wesad_state_builder import (
    WESADWindowConfig,
    build_monitoring_states_from_wesad_subject,
    classify_wesad_window_labels,
    compute_wesad_subject_aggregates,
)
from agent.benchmarks.vitalbench.state_to_qa_generator import generate_qa_from_states


def _write_synthetic_wesad_pickle(
    path: Path,
    labels: np.ndarray,
    *,
    subject: str | bytes = "S_TEST",
) -> None:
    payload = {
        "subject": subject,
        "label": labels.astype(np.int32),
        "signal": {
            "wrist": {
                "BVP": np.zeros((int(labels.size / 700 * 64), 1), dtype=float),
            }
        },
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_classify_wesad_window_labels_requires_clean_allowed_label() -> None:
    assert classify_wesad_window_labels([1, 1, 1]).stress_label == "baseline"
    assert classify_wesad_window_labels([2, 2, 2]).stress_label == "stress"
    assert classify_wesad_window_labels([4, 4, 4]) is None
    assert (
        classify_wesad_window_labels(
            [4, 4, 4],
            allowed_protocol_label_ids=(1, 2, 3, 4),
        ).stress_label
        == "meditation"
    )
    assert classify_wesad_window_labels([0, 0, 0]) is None
    assert classify_wesad_window_labels([5, 5, 5]) is None
    assert classify_wesad_window_labels([6, 6, 6]) is None
    assert classify_wesad_window_labels([7, 7, 7]) is None
    assert classify_wesad_window_labels([1, 2, 2], min_label_fraction=1.0) is None
    partial = classify_wesad_window_labels([1] * 9 + [0], min_label_fraction=0.9)
    assert partial is not None
    assert partial.stress_label == "baseline"
    assert partial.label_fraction == 0.9


def test_wesad_builder_creates_30s_and_60s_clean_stress_states(tmp_path: Path) -> None:
    fs = 700
    labels = np.concatenate(
        [
            np.full(60 * fs, 1, dtype=np.int32),
            np.full(60 * fs, 2, dtype=np.int32),
            np.full(60 * fs, 3, dtype=np.int32),
        ]
    )
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_wesad_pickle(pkl_path, labels)

    states = build_monitoring_states_from_wesad_subject(
        pkl_path,
        config=WESADWindowConfig(window_secs=(30, 60)),
    )

    assert len(states) == 9
    assert {state.dataset for state in states} == {"wesad"}
    assert {state.modality for state in states} == {"wearable"}

    states_30s = [state for state in states if state.window_duration_s == 30]
    states_60s = [state for state in states if state.window_duration_s == 60]

    assert [state.stress_label for state in states_30s] == [
        "baseline",
        "baseline",
        "stress",
        "stress",
        "amusement",
        "amusement",
    ]
    assert [state.stress_label for state in states_60s] == [
        "baseline",
        "stress",
        "amusement",
    ]

    assert states_30s[0].previous_stress_label is None
    assert states_30s[1].previous_stress_label == "baseline"
    assert states_30s[2].previous_stress_label == "baseline"
    assert states_60s[0].previous_stress_label is None
    assert states_60s[1].previous_stress_label == "baseline"

    assert states_30s[2].metadata["protocol_label_fraction"] == 1.0
    assert states_30s[2].metadata["wrist_bvp_sampling_rate_hz"] == 64
    assert states_30s[2].metadata["wrist_bvp_window_samples"] == 30 * 64
    assert states_30s[2].metadata["window_start_label_sample"] == 60 * fs
    assert states_30s[2].stress_burden_ratio == 1 / 3
    assert states_30s[2].stress_duration_s == 60.0
    assert states_30s[2].dominant_stress_label == "stress"
    assert states_30s[2].stress_transition_count == 2
    assert states_30s[2].stress_transition_count_per_hour == 40.0
    assert states_30s[2].metadata["aggregate_scope"] == "subject_allowed_protocol_labels"


def test_compute_wesad_subject_aggregates_uses_allowed_labels_only() -> None:
    fs = 700
    labels = np.concatenate(
        [
            np.full(30 * fs, 1, dtype=np.int32),
            np.full(30 * fs, 0, dtype=np.int32),
            np.full(60 * fs, 2, dtype=np.int32),
            np.full(30 * fs, 4, dtype=np.int32),
            np.full(30 * fs, 3, dtype=np.int32),
        ]
    )

    aggregates = compute_wesad_subject_aggregates(labels)

    assert aggregates.valid_protocol_duration_s == 120.0
    assert aggregates.stress_duration_s == 60.0
    assert aggregates.stress_burden_ratio == 0.5
    assert aggregates.dominant_stress_label == "stress"
    assert aggregates.stress_transition_count == 2
    assert aggregates.stress_transition_count_per_hour == 60.0


def test_wesad_builder_decodes_bytes_subject_and_limits_windows(tmp_path: Path) -> None:
    fs = 700
    labels = np.full(120 * fs, 2, dtype=np.int32)
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_wesad_pickle(pkl_path, labels, subject=b"S_BYTES")

    states = build_monitoring_states_from_wesad_subject(
        pkl_path,
        config=WESADWindowConfig(
            window_secs=(30, 60),
            max_windows_per_duration=1,
        ),
    )

    assert len(states) == 2
    assert {state.patient_id for state in states} == {"S_BYTES"}
    assert all("b'" not in state.state_id for state in states)
    assert [state.window_duration_s for state in states] == [30, 60]


def test_wesad_builder_supports_partial_clean_and_drops_ignore_windows(
    tmp_path: Path,
) -> None:
    fs = 700
    labels = np.concatenate(
        [
            np.concatenate(
                [
                    np.full(27 * fs, 1, dtype=np.int32),
                    np.full(3 * fs, 0, dtype=np.int32),
                ]
            ),
            np.full(30 * fs, 5, dtype=np.int32),
            np.full(30 * fs, 2, dtype=np.int32),
        ]
    )
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_wesad_pickle(pkl_path, labels)

    states = build_monitoring_states_from_wesad_subject(
        pkl_path,
        config=WESADWindowConfig(
            window_secs=(30,),
            min_label_fraction=0.9,
        ),
    )

    assert [state.stress_label for state in states] == ["baseline", "stress"]
    assert [state.metadata["raw_window_index"] for state in states] == [0, 2]
    assert states[0].metadata["protocol_label_fraction"] == 0.9
    assert states[1].previous_stress_label == "baseline"


def test_wesad_builder_excludes_meditation_by_default_and_can_include_it(
    tmp_path: Path,
) -> None:
    fs = 700
    labels = np.concatenate(
        [
            np.full(30 * fs, 1, dtype=np.int32),
            np.full(30 * fs, 4, dtype=np.int32),
            np.full(30 * fs, 2, dtype=np.int32),
        ]
    )
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_wesad_pickle(pkl_path, labels)

    default_states = build_monitoring_states_from_wesad_subject(
        pkl_path,
        config=WESADWindowConfig(window_secs=(30,)),
    )
    assert [state.stress_label for state in default_states] == ["baseline", "stress"]

    with_meditation = build_monitoring_states_from_wesad_subject(
        pkl_path,
        config=WESADWindowConfig(
            window_secs=(30,),
            allowed_protocol_label_ids=(1, 2, 3, 4),
        ),
    )
    assert [state.stress_label for state in with_meditation] == [
        "baseline",
        "meditation",
        "stress",
    ]


def test_wesad_states_generate_stress_qa_only_when_stress_fields_present(
    tmp_path: Path,
) -> None:
    fs = 700
    labels = np.concatenate(
        [
            np.full(60 * fs, 1, dtype=np.int32),
            np.full(60 * fs, 2, dtype=np.int32),
            np.full(60 * fs, 4, dtype=np.int32),
        ]
    )
    pkl_path = tmp_path / "S_TEST.pkl"
    _write_synthetic_wesad_pickle(pkl_path, labels)

    states = build_monitoring_states_from_wesad_subject(
        pkl_path,
        config=WESADWindowConfig(window_secs=(60,)),
    )
    samples = generate_qa_from_states(states)

    assert samples
    assert {sample.dataset for sample in samples} == {"wesad"}
    assert {sample.modality for sample in samples} == {"wearable"}

    template_ids = {sample.template_id for sample in samples}
    assert {
        "ta6_stress_presence_current",
        "ta7_stress_state_current",
        "ta9_stress_changed_from_previous",
        "tb6_stress_presence_monitoring_window",
        "tb7_stress_ratio_query_monitoring_window",
        "tb8_stress_duration_query_monitoring_window",
        "tb9_dominant_stress_state_monitoring_window",
    }.issubset(template_ids)
    assert "ta8_protocol_label_id_query_current" not in template_ids
    assert "ta2_heart_rate_category_current" not in template_ids

    choose_samples = [
        sample for sample in samples if sample.template_id == "ta7_stress_state_current"
    ]
    assert choose_samples
    assert all(
        sample.answer[0] in {"baseline", "stress", "amusement", "meditation"}
        for sample in choose_samples
    )

    stress_verify_samples = [
        sample
        for sample in samples
        if sample.template_id == "ta6_stress_presence_current"
        and sample.source_state_id.endswith("0001")
    ]
    assert stress_verify_samples
    assert {sample.answer[0] for sample in stress_verify_samples} == {"yes"}

    tier_b_samples = [sample for sample in samples if sample.tier == "B"]
    assert tier_b_samples
    assert {sample.time_scope for sample in tier_b_samples} == {
        "monitoring_window_aggregate"
    }

    stress_ratio_samples = [
        sample
        for sample in samples
        if sample.template_id == "tb7_stress_ratio_query_monitoring_window"
    ]
    assert stress_ratio_samples
    assert {sample.answer[0] for sample in stress_ratio_samples} == {"0.5"}

    stress_duration_samples = [
        sample
        for sample in samples
        if sample.template_id == "tb8_stress_duration_query_monitoring_window"
    ]
    assert stress_duration_samples
    assert {sample.answer[0] for sample in stress_duration_samples} == {"1"}

    dominant_samples = [
        sample
        for sample in samples
        if sample.template_id == "tb9_dominant_stress_state_monitoring_window"
    ]
    assert dominant_samples
    assert {sample.answer[0] for sample in dominant_samples} == {"stress"}

