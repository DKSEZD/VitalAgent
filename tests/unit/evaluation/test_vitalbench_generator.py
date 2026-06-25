"""
Unit tests for the VitalBench state-grounded dataset generator.

These tests validate template generation with hand-crafted state fixtures:

    sample_states_fixture.py
        -> state_to_qa_generator.py
        -> VitalBenchSample

They are lightweight smoke tests for the VitalBench dataset generator.
"""

from __future__ import annotations

import pytest

from tests.unit.evaluation.sample_states_fixture import get_demo_states
from agent.benchmarks.vitalbench.state_to_qa_generator import (
    generate_qa_from_state,
    generate_qa_from_states,
    summarize_samples,
)
from agent.mhealth.schemas import MonitoringState


def _generate_demo_samples():
    return generate_qa_from_states(get_demo_states())


def _build_test_state(
    *,
    window_start_s: float | None,
    window_duration_s: float | None,
) -> MonitoringState:
    return MonitoringState(
        state_id="demo_window_time_0001",
        patient_id="demo_patient_001",
        dataset="afppgecg",
        modality="ecg",
        window_start_s=window_start_s,
        window_duration_s=window_duration_s,
        hr_bpm=72.0,
        previous_hr_bpm=70.0,
        rhythm_class="N",
        previous_rhythm_class="N",
        af_burden_ratio=0.0,
        max_hr_bpm=85.0,
        rhythm_transition_count_per_hour=0.0,
    )


def test_qa_jsonl_record_includes_window_time_fields() -> None:
    """JSONL output must expose window_start_s and window_duration_s at top level."""
    state = _build_test_state(window_start_s=480.0, window_duration_s=30.0)
    samples = generate_qa_from_state(state)
    assert samples, "expected at least one sample for the test state"

    record = samples[0].to_jsonl_record()
    assert record["window_start_s"] == 480.0
    assert record["window_duration_s"] == 30.0


def test_qa_jsonl_record_propagates_none_window_time() -> None:
    """Missing window time should serialize as null, not be dropped."""
    state = _build_test_state(window_start_s=None, window_duration_s=None)
    samples = generate_qa_from_state(state)
    if not samples:
        return

    record = samples[0].to_jsonl_record()
    assert "window_start_s" in record
    assert "window_duration_s" in record
    assert record["window_start_s"] is None
    assert record["window_duration_s"] is None


def test_demo_generation_sample_count() -> None:
    samples = _generate_demo_samples()

    # 8 demo states.
    # A full state generates 39 samples:
    #   Tier A: 5 templates x 3 phrasings = 15
    #   Tier B: tb1/tb2/tb4/tb5 = 4 x 3 = 12
    #           tb3 = 4 thresholds x 3 phrasings = 12
    #           total Tier B = 24
    #
    # One demo state has Unknown rhythm, which should skip:
    #   ta1_af_presence_current: 3 samples
    #   ta5_rhythm_changed_from_previous: 3 samples
    #
    # Expected total = 8 * 39 - 6 = 306
    assert len(samples) == 306


def test_question_ids_are_unique() -> None:
    samples = _generate_demo_samples()

    question_ids = [sample.question_id for sample in samples]

    assert len(question_ids) == len(set(question_ids))


def test_generator_rejects_duplicate_question_ids() -> None:
    state = get_demo_states()[0]

    with pytest.raises(ValueError, match="Duplicated question_id values found"):
        generate_qa_from_states([state, state])


def test_tier_b_samples_use_monitoring_window_scope() -> None:
    samples = _generate_demo_samples()

    tier_b_samples = [sample for sample in samples if sample.tier == "B"]

    assert tier_b_samples
    assert {
        sample.time_scope for sample in tier_b_samples
    } == {"monitoring_window_aggregate"}
    assert all("recording" not in sample.question.lower() for sample in tier_b_samples)


def test_summary_distribution_matches_expected_demo_output() -> None:
    samples = _generate_demo_samples()
    summary = summarize_samples(samples)

    assert summary["sample_count"] == 306

    assert summary["by_dataset"] == {
        "afppgecg": 306,
    }

    assert summary["by_modality"] == {
        "ecg": 156,
        "ppg": 150,
    }

    assert summary["by_tier"] == {
        "A": 114,
        "B": 192,
    }

    assert summary["by_question_type"] == {
        "single_choose": 72,
        "single_query": 48,
        "single_verify": 186,
    }

    assert summary["by_evaluation_method"] == {
        "boolean_match": 186,
        "normalized_match": 72,
        "numeric_tolerance": 48,
    }


def test_single_choose_samples_have_answer_set() -> None:
    samples = _generate_demo_samples()

    choose_samples = [
        sample for sample in samples if sample.question_type == "single_choose"
    ]

    assert choose_samples
    assert all(sample.answer_set for sample in choose_samples)

    for sample in choose_samples:
        assert sample.answer[0] in sample.answer_set
        assert sample.evaluation_method == "normalized_match"


def test_single_verify_samples_have_yes_no_answer_set() -> None:
    samples = _generate_demo_samples()

    verify_samples = [
        sample for sample in samples if sample.question_type == "single_verify"
    ]

    assert verify_samples
    assert all(sample.answer_set == ["yes", "no"] for sample in verify_samples)

    for sample in verify_samples:
        assert sample.answer[0] in {"yes", "no"}
        assert sample.evaluation_method == "boolean_match"


def test_single_query_samples_have_numeric_tolerance() -> None:
    samples = _generate_demo_samples()

    query_samples = [
        sample for sample in samples if sample.question_type == "single_query"
    ]

    assert query_samples
    assert all(sample.answer_type == "numeric" for sample in query_samples)
    assert all(sample.numeric_tolerance is not None for sample in query_samples)
    assert all(sample.evaluation_method == "numeric_tolerance" for sample in query_samples)

    for sample in query_samples:
        # Current implementation stores numeric answers as compact strings.
        float(sample.answer[0])


def test_tb3_threshold_parameter_expansion() -> None:
    samples = _generate_demo_samples()

    tb3_samples = [
        sample
        for sample in samples
        if sample.template_id == "tb3_heart_rate_excursion_threshold_monitoring_window"
    ]

    assert tb3_samples

    thresholds = sorted(
        {
            int(sample.generation_params["threshold_bpm"])
            for sample in tb3_samples
        }
    )

    assert thresholds == [100, 110, 120, 130]

    # 8 demo states x 4 thresholds x 3 phrasings = 96
    assert len(tb3_samples) == 96

    for sample in tb3_samples:
        assert "threshold_bpm" in sample.generation_params
        assert sample.evidence["field"] == "max_hr_bpm"
        assert sample.evidence["operator"] == ">"
        assert sample.evidence["threshold"] in {100.0, 110.0, 120.0, 130.0}


def test_ta4_min_delta_parameter_is_recorded() -> None:
    samples = _generate_demo_samples()

    ta4_samples = [
        sample
        for sample in samples
        if sample.template_id == "ta4_heart_rate_higher_than_previous"
    ]

    assert ta4_samples

    for sample in ta4_samples:
        assert sample.generation_params == {"min_delta_bpm": 5.0}
        assert sample.evidence["threshold"] == 5.0
        assert sample.evidence["operator"] == ">="
        assert sample.evidence["previous_window_offset"] == 1


def test_unknown_rhythm_state_skips_rhythm_dependent_templates() -> None:
    samples = _generate_demo_samples()

    unknown_state_samples = [
        sample
        for sample in samples
        if sample.source_state_id == "demo_ppg_unknown_rhythm_0008"
    ]

    assert unknown_state_samples

    template_ids = {sample.template_id for sample in unknown_state_samples}

    assert "ta1_af_presence_current" not in template_ids
    assert "ta5_rhythm_changed_from_previous" not in template_ids

    # Non-rhythm templates should still be generated.
    assert "ta2_heart_rate_category_current" in template_ids
    assert "ta3_heart_rate_query_current" in template_ids
    assert "ta4_heart_rate_higher_than_previous" in template_ids


def test_ppg_af_samples_include_methodological_caveat() -> None:
    samples = _generate_demo_samples()

    ppg_af_samples = [
        sample
        for sample in samples
        if sample.modality == "ppg"
        and sample.template_id
        in {
            "ta1_af_presence_current",
            "tb1_af_presence_monitoring_window",
        }
    ]

    assert ppg_af_samples

    for sample in ppg_af_samples:
        assert "methodological_caveat" in sample.evidence
        assert "indirect rhythm-irregularity inference" in sample.evidence[
            "methodological_caveat"
        ]


def test_ecg_af_samples_do_not_include_ppg_caveat() -> None:
    samples = _generate_demo_samples()

    ecg_af_samples = [
        sample
        for sample in samples
        if sample.modality == "ecg"
        and sample.template_id
        in {
            "ta1_af_presence_current",
            "tb1_af_presence_monitoring_window",
        }
    ]

    assert ecg_af_samples

    for sample in ecg_af_samples:
        assert "methodological_caveat" not in sample.evidence


def test_tb5_evidence_does_not_contain_af_episode_threshold() -> None:
    samples = _generate_demo_samples()

    tb5_samples = [
        sample
        for sample in samples
        if sample.template_id == "tb5_rhythm_variability_monitoring_window"
    ]

    assert tb5_samples

    for sample in tb5_samples:
        assert sample.evidence["field"] == "rhythm_transition_count_per_hour"
        assert "af_episode_min_duration_s" not in sample.evidence


def test_every_sample_has_required_public_fields() -> None:
    samples = _generate_demo_samples()

    for sample in samples:
        record = sample.to_jsonl_record()

        assert record["question_id"]
        assert record["template_id"]
        assert record["source_state_id"]
        assert record["patient_id"]
        assert record["dataset"]
        assert record["modality"]
        assert "window_start_s" in record
        assert "window_duration_s" in record
        assert record["question"]
        assert record["answer"]
        assert record["answer_type"]
        assert record["ground_truth_source"]
        assert record["evaluation_method"]
        assert record["evidence"]
        assert record["expected_tools"]


def test_to_jsonl_record_is_serializable() -> None:
    import json

    samples = _generate_demo_samples()

    for sample in samples:
        json.dumps(sample.to_jsonl_record(), ensure_ascii=False)

def test_ecg_af_sample_notes_do_not_include_ppg_caveat() -> None:
    samples = generate_qa_from_states(get_demo_states())

    ecg_af_samples = [
        sample
        for sample in samples
        if sample.modality == "ecg"
        and sample.target in {
            "af_presence_current",
            "af_presence_monitoring_window",
            "af_frequency_monitoring_window",
        }
    ]

    assert ecg_af_samples
    for sample in ecg_af_samples:
        assert "On PPG datasets" not in sample.notes
        assert "P-wave absence" not in sample.notes


def test_ppg_af_sample_notes_include_methodological_caveat() -> None:
    samples = generate_qa_from_states(get_demo_states())

    ppg_af_samples = [
        sample
        for sample in samples
        if sample.modality == "ppg"
        and sample.target in {
            "af_presence_current",
            "af_presence_monitoring_window",
            "af_frequency_monitoring_window",
        }
    ]

    assert ppg_af_samples
    for sample in ppg_af_samples:
        assert "On PPG datasets" in sample.notes
        assert "P-wave absence" in sample.notes


def test_wearable_modality_does_not_receive_ppg_af_caveat():
    from agent.mhealth.schemas import MonitoringState

    state = MonitoringState(
        state_id="demo_wearable_stress_0001",
        patient_id="demo_wesad_001",
        dataset="wesad",
        modality="wearable",
        stress_label="stress",
        protocol_label_id=2,
    )

    samples = generate_qa_from_states([state])

    assert samples
    assert {sample.modality for sample in samples} == {"wearable"}
    for sample in samples:
        assert "On PPG datasets" not in sample.notes
        assert "P-wave absence" not in sample.notes


