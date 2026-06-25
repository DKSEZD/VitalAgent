from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, asdict, fields
from pathlib import Path

import pytest

from agent.benchmarks.vitalbench.eval_loader import (
    AgentInput,
    EvalSpec,
    QASample,
    WindowLocator,
    load_qa_jsonl,
)


SAMPLE_JSONL = "\n".join(
    [
        json.dumps(
            {
                "question_id": "ppg_dalia_S7_w0016__tb3__p0",
                "template_id": (
                    "tb3_heart_rate_excursion_threshold_monitoring_window"
                ),
                "source_state_id": "ppg_dalia_subject_S7_30s_clean_window_0016",
                "patient_id": "S7",
                "dataset": "ppg_dalia",
                "modality": "ppg",
                "window_start_s": 480.0,
                "window_duration_s": 30,
                "tier": "B",
                "question_type": "single_verify",
                "target": "heart_rate_excursion_threshold_monitoring_window",
                "time_scope": "monitoring_window_aggregate",
                "difficulty_tier": "easy",
                "question": (
                    "Did my heart rate ever go above 120 bpm during this "
                    "monitoring window?"
                ),
                "answer": ["no"],
                "answer_type": "yes_no",
                "answer_set": ["yes", "no"],
                "ground_truth_source": "derived_from_temporal_aggregation",
                "evaluation_method": "boolean_match",
                "numeric_tolerance": None,
                "evidence": {
                    "field": "max_hr_bpm",
                    "value": 55.16,
                    "threshold": 120.0,
                },
                "expected_tools": ["get_longitudinal_trend"],
                "phrasing_index": 0,
                "generation_params": {"threshold_bpm": 120},
                "notes": "Threshold sampled to avoid overfitting.",
            }
        ),
        json.dumps(
            {
                "question_id": "wesad_S14_w0069__tb7__p0",
                "template_id": "tb7_stress_ratio_query_monitoring_window",
                "source_state_id": "wesad_subject_S14_30s_clean_window_0069",
                "patient_id": "S14",
                "dataset": "wesad",
                "modality": "wearable",
                "window_start_s": 4590.0,
                "window_duration_s": 30,
                "tier": "B",
                "question_type": "single_query",
                "target": "stress_ratio_monitoring_window",
                "time_scope": "monitoring_window_aggregate",
                "difficulty_tier": "medium",
                "question": (
                    "What fraction of this monitoring window was labelled as "
                    "stress?"
                ),
                "answer": ["0.3"],
                "answer_type": "numeric",
                "answer_set": [],
                "ground_truth_source": "derived_from_temporal_aggregation",
                "evaluation_method": "numeric_tolerance",
                "numeric_tolerance": 0.02,
                "evidence": {"field": "stress_burden_ratio", "value": 0.303},
                "expected_tools": ["get_longitudinal_trend"],
                "phrasing_index": 0,
                "generation_params": {},
                "notes": "Numeric ratio in [0, 1].",
            }
        ),
    ]
)


def test_load_qa_jsonl_yields_sample_views(tmp_path: Path) -> None:
    """Each row produces a QASample with agent_input, eval_spec, gt_metadata."""
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))

    assert len(samples) == 2
    for sample in samples:
        assert isinstance(sample, QASample)
        assert isinstance(sample.agent_input, AgentInput)
        assert isinstance(sample.eval_spec, EvalSpec)
        assert isinstance(sample.gt_metadata, dict)


def test_loader_dataclasses_are_frozen(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")
    sample = next(load_qa_jsonl(p))

    assert all(
        cls.__dataclass_params__.frozen
        for cls in (WindowLocator, AgentInput, EvalSpec, QASample)
    )
    with pytest.raises(FrozenInstanceError):
        sample.agent_input.question = "leaked mutation"


def test_window_locator_constructed_from_top_level_fields(tmp_path: Path) -> None:
    """WindowLocator pulls dataset / patient / start / duration from JSONL."""
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))
    loc = samples[0].agent_input.window_locator

    assert loc.dataset == "ppg_dalia"
    assert loc.patient_id == "S7"
    assert loc.window_start_s == 480.0
    assert loc.window_end_s == 510.0
    assert loc.recording_id is None


@pytest.mark.parametrize("legacy_name", ["af_ppg_ecg", "zenodo", "zenodo_af_monitoring"])
def test_loader_canonicalizes_legacy_afppgecg_names(
    tmp_path: Path,
    legacy_name: str,
) -> None:
    row = json.loads(SAMPLE_JSONL.splitlines()[0])
    row["dataset"] = legacy_name
    p = tmp_path / "legacy.jsonl"
    p.write_text(json.dumps(row), encoding="utf-8")

    sample = next(load_qa_jsonl(p))

    assert sample.agent_input.window_locator.dataset == "afppgecg"


def test_icentia_locator_extracts_segment_as_raw_recording_id(tmp_path: Path) -> None:
    row = {
        "question_id": "q_ic",
        "source_state_id": "icentia11k_patient_03500_segment_17_samples_0_450000_window_3",
        "patient_id": "03500",
        "dataset": "icentia11k",
        "window_start_s": 900.0,
        "window_duration_s": 300.0,
        "question": "Based on the current window, am I showing signs of AF?",
        "answer": ["no"],
        "answer_type": "yes_no",
        "answer_set": ["yes", "no"],
        "evaluation_method": "boolean_match",
        "numeric_tolerance": None,
    }
    p = tmp_path / "qa.jsonl"
    p.write_text(json.dumps(row), encoding="utf-8")

    sample = next(load_qa_jsonl(p))

    assert sample.agent_input.window_locator.recording_id == "17"


def test_agent_input_has_no_answer_or_evidence(tmp_path: Path) -> None:
    """Boundary: answer, evidence, expected_tools must NOT leak to AgentInput."""
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))
    forbidden_keys = {
        "answer",
        "answer_type",
        "evidence",
        "expected_tools",
        "evaluation_method",
        "target",
        "ground_truth_source",
        "tier",
        "template_id",
        "source_state_id",
    }

    for sample in samples:
        agent_dict = asdict(sample.agent_input)
        actual_keys = set(_flatten_dict_keys(agent_dict))
        leaked = forbidden_keys & actual_keys
        assert not leaked, f"AgentInput leaked GT/eval fields: {leaked}"


def test_agent_input_schema_has_no_answer_fields() -> None:
    field_names = {field.name for field in fields(AgentInput)}

    assert "answer" not in field_names
    assert "answer_type" not in field_names


def test_eval_spec_carries_grading_metadata(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))
    spec = samples[1].eval_spec

    assert spec.answer == ["0.3"]
    assert spec.answer_type == "numeric"
    assert spec.evaluation_method == "numeric_tolerance"
    assert spec.numeric_tolerance == 0.02


def test_gt_metadata_contains_unconsumed_fields(tmp_path: Path) -> None:
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))
    md = samples[0].gt_metadata

    assert "evidence" in md
    assert "expected_tools" in md
    assert "template_id" in md
    assert "tier" in md
    assert "target" in md
    assert "question" not in md
    assert "answer" not in md
    assert "window_start_s" not in md
    assert "patient_id" not in md
    assert "answer_type" not in md


def test_empty_answer_set_normalized_to_none(tmp_path: Path) -> None:
    """Templates with answer_set=[] should expose None to the agent."""
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))
    wesad_sample = samples[1]

    assert wesad_sample.agent_input.answer_set is None
    assert wesad_sample.eval_spec.answer_set is None


def test_verify_question_preserves_answer_set(tmp_path: Path) -> None:
    """Yes/no verify questions must keep their answer_set."""
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    samples = list(load_qa_jsonl(p))
    ppg_sample = samples[0]

    assert ppg_sample.agent_input.answer_set == ["yes", "no"]


def test_missing_window_time_raises(tmp_path: Path) -> None:
    """Samples without window_start_s should fail loudly."""
    bad_row = json.loads(SAMPLE_JSONL.splitlines()[0])
    bad_row["window_start_s"] = None
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(bad_row), encoding="utf-8")

    with pytest.raises(ValueError, match="window_start_s"):
        list(load_qa_jsonl(p))


def test_invalid_json_raises_with_line_number(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text("not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSON at .*:1"):
        list(load_qa_jsonl(p))


def test_non_object_json_row_raises_with_line_number(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid QA row at .*:1"):
        list(load_qa_jsonl(p))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(load_qa_jsonl(tmp_path / "does_not_exist.jsonl"))


def test_streaming_does_not_load_all_at_once(tmp_path: Path) -> None:
    """load_qa_jsonl must be a generator, not load entire file in memory."""
    p = tmp_path / "qa.jsonl"
    p.write_text(SAMPLE_JSONL, encoding="utf-8")

    result = load_qa_jsonl(p)

    assert inspect.isgenerator(result) or hasattr(result, "__next__")


def test_load_real_ppg_dalia_jsonl() -> None:
    """Smoke test against actual generated PPG-DaLiA output."""
    fixture = Path("dataset/mhealth_qa/full/ppg_dalia/qa/S1_qa.jsonl")
    if not fixture.exists():
        pytest.skip("PPG-DaLiA smoke fixture not committed")

    sample = next(load_qa_jsonl(fixture))
    loc = sample.agent_input.window_locator

    assert loc.dataset == "ppg_dalia"
    assert loc.window_start_s >= 0
    assert loc.window_end_s > loc.window_start_s


def _flatten_dict_keys(d: dict) -> list[str]:
    """Recursively collect all keys from a nested dict."""
    keys = []
    for key, value in d.items():
        keys.append(key)
        if isinstance(value, dict):
            keys.extend(_flatten_dict_keys(value))
    return keys
