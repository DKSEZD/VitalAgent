from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path

from agent.benchmarks.vitalbench.eval_loader import (
    AgentInput,
    EvalSpec,
    QASample,
    WindowLocator,
)
from agent.benchmarks.vitalbench.schemas import VitalBenchSample
from agent.mhealth.schemas import MonitoringState
from agent.evaluation.reactive import vitalbench_eval as eval_mod
from agent.llm import LLMStreamEvent, zero_usage
from agent.reactive.validation import ValidationGate
from agent.schemas import (
    IntentType,
    IssueSeverity,
    IssueType,
    Plan,
    PlanStep,
    ToolResult,
    ValidationIssue,
    ValidationResult,
)
from agent.state.mhealth_state_store import get_global_state_store


def make_state(
    state_id: str = "s1",
    *,
    dataset: str = "demo",
    subject_id: str = "p1",
    recording_id: str = "r1",
    window_index: int = 0,
) -> MonitoringState:
    return MonitoringState(
        state_id=state_id,
        patient_id=subject_id,
        dataset=dataset,
        modality="ppg",
        subject_id=subject_id,
        recording_id=recording_id,
        window_index=window_index,
        window_start_s=float(window_index * 30),
        window_end_s=float((window_index + 1) * 30),
        hr_bpm=72.0,
    )


def make_sample(question_id: str = "q1") -> VitalBenchSample:
    return VitalBenchSample(
        question_id=question_id,
        template_id="ta1",
        source_state_id="s1",
        patient_id="p1",
        dataset="demo",
        modality="ppg",
        window_start_s=0.0,
        window_duration_s=30.0,
        tier="A",
        question_type="single_verify",
        target="heart_rate",
        time_scope="current",
        difficulty_tier="easy",
        question="Is my heart rate normal?",
        answer=["yes"],
        answer_type="yes_no",
        answer_set=["yes", "no"],
        evaluation_method="boolean_match",
    )


def make_eval_sample(question_id: str = "q1") -> QASample:
    return QASample(
        agent_input=AgentInput(
            question_id=question_id,
            question="What is my current heart rate in beats per minute?",
            window_locator=WindowLocator(
                dataset="demo",
                patient_id="p1",
                window_start_s=0.0,
                window_end_s=30.0,
                recording_id="r1",
            ),
            answer_set=None,
        ),
        eval_spec=EvalSpec(
            question_id=question_id,
            answer=["72"],
            answer_type="numeric",
            evaluation_method="numeric_tolerance",
            numeric_tolerance=5.0,
            answer_set=None,
        ),
        gt_metadata={
            "source_state_id": "s1",
            "template_id": "ta3",
            "modality": "ppg",
            "question_type": "single_query",
            "target": "heart_rate_numeric_current",
        },
    )








def _sample_with_state(index: int) -> tuple[QASample, MonitoringState]:
    sample = make_eval_sample(f"q{index}")
    state = make_state(f"s{index}", window_index=index)
    return (
        replace(
            sample,
            agent_input=replace(sample.agent_input, question_id=f"q{index}"),
            eval_spec=replace(sample.eval_spec, question_id=f"q{index}", answer=[str(70 + index)]),
            gt_metadata={**sample.gt_metadata, "source_state_id": f"s{index}"},
        ),
        state,
    )






















def test_scorer_handles_cot_sc_yes_no_majority() -> None:
    spec = replace(
        make_eval_sample().eval_spec,
        answer=["yes"],
        answer_type="yes_no",
        evaluation_method="boolean_match",
    )
    score, error_class, method = eval_mod.score_prediction_details(
        spec,
        ["Yes\n...", "Yes\n...", "No\n...", "Yes\n...", "No\n..."],
    )

    assert score is True
    assert error_class is None
    assert method == "cot_sc_majority"


def test_scorer_accepts_official_answer_prefix_yes_no() -> None:
    spec = replace(
        make_eval_sample().eval_spec,
        answer=["no"],
        answer_type="yes_no",
        evaluation_method="boolean_match",
    )
    score, error_class, method = eval_mod.score_prediction_details(spec, "Answer: No")

    assert score is True
    assert error_class is None
    assert method is None


def test_scorer_accepts_official_answer_prefix_category_without_answer_set() -> None:
    spec = replace(
        make_eval_sample().eval_spec,
        answer=["Supine"],
        answer_type="category",
        evaluation_method="normalized_match",
        answer_set=None,
    )
    score, error_class, method = eval_mod.score_prediction_details(spec, "Answer: Supine")

    assert score is True
    assert error_class is None
    assert method is None


def test_scorer_handles_cot_sc_numeric_median() -> None:
    spec = replace(make_eval_sample().eval_spec, answer=["5"], answer_type="numeric")
    score, error_class, method = eval_mod.score_prediction_details(
        spec,
        ["1", "9", "5", "3", "7"],
    )

    assert score is True
    assert error_class is None
    assert method == "cot_sc_median"


def test_manifest_lookup_by_dataset_and_subject() -> None:
    manifest = eval_mod.SourceManifest(
        {
            "entries": [
                {
                    "dataset": "afppgecg",
                    "subject_id": "001",
                    "recording_id": "001",
                    "builder_tool": "state_build_from_ppg_patient",
                    "builder_args": {"patient_id": "001"},
                }
            ]
        }
    )

    hit = manifest.lookup("afppgecg", "001", "001")
    missing = manifest.lookup("afppgecg", "999", "999")

    assert hit.build_supported is True
    assert hit.builder_tool == "state_build_from_ppg_patient"
    assert missing.build_supported is False
    assert "No source manifest entry" in (missing.error or "")




def test_eval_state_build_missing_context_emits_string_error(monkeypatch) -> None:
    manifest = eval_mod.SourceManifest(
        {
            "entries": [
                {
                    "dataset": "demo",
                    "subject_id": "p1",
                    "recording_id": "r1",
                    "builder_tool": "fake_state_builder",
                    "builder_args": {},
                }
            ]
        }
    )
    monkeypatch.setattr(eval_mod, "call_tool", lambda *_args, **_kwargs: {"success": True})
    eval_mod.set_active_source_manifest(manifest)
    store = get_global_state_store()
    store.clear()
    try:
        result = eval_mod.eval_state_build_missing_context("demo", "p1", "r1")

        assert result["success"] is True
        assert result["error"] == ""
        ToolResult(
            step_id=1,
            tool_name="eval_state_build_missing_context",
            success=True,
            data={key: value for key, value in result.items() if key not in ("success", "error")},
            error=result["error"],
        )
    finally:
        eval_mod.set_active_source_manifest(None)
        store.clear()


def test_eval_state_build_missing_context_uses_loaded_context_cache(monkeypatch) -> None:
    manifest = eval_mod.SourceManifest(
        {
            "entries": [
                {
                    "dataset": "wesad",
                    "subject_id": "S2",
                    "recording_id": "S2",
                    "builder_tool": "state_build_from_wesad_pickle",
                    "builder_args": {"pickle_path": "unused.pkl"},
                }
            ]
        }
    )
    builder_calls: list[str] = []

    def _fake_call_tool(name: str, **_kwargs):
        builder_calls.append(name)
        return {"success": True}

    monkeypatch.setattr(eval_mod, "call_tool", _fake_call_tool)
    eval_mod.set_active_source_manifest(manifest)
    store = get_global_state_store()
    store.clear()
    store.add_states([make_state("wesad-loaded", dataset="wesad", subject_id="S2", recording_id="S2")])
    try:
        result = eval_mod.eval_state_build_missing_context("wesad", "S2", "S2")

        assert result["success"] is True
        assert result["cached"] is True
        assert result["generated_state_count"] == 0
        assert result["builder_result"] == {"success": True, "cached": True}
        assert result["error"] == ""
        assert builder_calls == []
    finally:
        eval_mod.set_active_source_manifest(None)
        store.clear()


def test_partial_preload_split_is_deterministic_and_context_level() -> None:
    states = [
        make_state("a1", subject_id="a", recording_id="ra", window_index=0),
        make_state("a2", subject_id="a", recording_id="ra", window_index=1),
        make_state("b1", subject_id="b", recording_id="rb", window_index=0),
        make_state("c1", subject_id="c", recording_id="rc", window_index=0),
    ]

    first = eval_mod.split_preload_contexts(states, 0.34, 42)
    second = eval_mod.split_preload_contexts(states, 0.34, 42)

    assert first == second
    grouped = eval_mod.group_states_by_context(states)
    for key in first.missing_contexts:
        assert all(eval_mod.context_key_for_state(state) == key for state in grouped[key])
    assert first.routing_analysis_limited is False


def test_partial_preload_split_handles_single_context() -> None:
    states = [make_state("a1"), make_state("a2", window_index=1)]

    split = eval_mod.split_preload_contexts(states, 0.25, 42)

    assert split.routing_analysis_limited is True
    assert len(split.missing_contexts) == 0


def test_deterministic_scorers() -> None:
    assert eval_mod.boolean_match("Detected in this window.", ["yes"]) is True
    assert eval_mod.boolean_match("Normal.", ["no"]) is True
    assert eval_mod.normalized_match("A. low heart rate", ["low"], ["low", "normal", "high"]) is True
    assert eval_mod.numeric_tolerance_match("about 74 bpm", ["72"], 3) is True
    assert eval_mod.numeric_tolerance_match("about 80 bpm", ["72"], 3) is False


def _numeric_eval_spec(gold: str, tolerance: float | None = None) -> EvalSpec:
    return EvalSpec(
        question_id="q_numeric",
        answer=[gold],
        answer_type="numeric",
        evaluation_method="numeric_tolerance",
        numeric_tolerance=tolerance,
        answer_set=None,
    )


def test_numeric_relative_tolerance_passes_for_close_heart_rate(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_NUMERIC_RELATIVE_TOLERANCE", raising=False)

    score, error_class, numeric_match_method = eval_mod.score_prediction_details(
        _numeric_eval_spec("94"),
        "95.6",
    )

    assert score is True
    assert error_class is None
    assert numeric_match_method == "relative_5pct"


def test_numeric_relative_tolerance_fails_outside_five_percent(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_NUMERIC_RELATIVE_TOLERANCE", raising=False)

    score, error_class, numeric_match_method = eval_mod.score_prediction_details(
        _numeric_eval_spec("94"),
        "100",
    )

    assert score is False
    assert error_class == "unknown"
    assert numeric_match_method == "relative_5pct"


def test_numeric_relative_tolerance_zero_gold_uses_half_point_fallback(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_NUMERIC_RELATIVE_TOLERANCE", raising=False)

    score, error_class, numeric_match_method = eval_mod.score_prediction_details(
        _numeric_eval_spec("0"),
        "0.3",
    )

    assert score is True
    assert error_class is None
    assert numeric_match_method == "absolute"


def test_build_wrapper_forces_clear_existing_and_preserves_store(monkeypatch) -> None:
    store = get_global_state_store()
    store.clear()
    store.add_states([make_state("existing", subject_id="keep", recording_id="keep")])
    manifest = eval_mod.SourceManifest(
        {
            "entries": [
                {
                    "dataset": "demo",
                    "subject_id": "p1",
                    "recording_id": "r1",
                    "builder_tool": "fake_builder",
                    "builder_args": {"clear_existing": True, "patient_id": "p1"},
                }
            ]
        }
    )
    eval_mod.set_active_source_manifest(manifest)
    captured = {}

    def fake_call_tool(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        store.add_states([make_state("new", subject_id="p1", recording_id="r1")])
        return {"success": True}

    monkeypatch.setattr(eval_mod, "call_tool", fake_call_tool)

    result = eval_mod.eval_state_build_missing_context("demo", "p1", "r1")

    assert result["success"] is True
    assert captured["kwargs"]["clear_existing"] is False
    contexts = store.list_contexts()
    assert any(context["subject_id"] == "keep" for context in contexts)
    assert any(context["subject_id"] == "p1" for context in contexts)


def test_build_wrapper_manifest_missing_returns_clean_error() -> None:
    eval_mod.set_active_source_manifest(eval_mod.SourceManifest({"entries": []}))

    result = eval_mod.eval_state_build_missing_context("demo", "missing", "missing")

    assert result["success"] is False
    assert result["error_class"] == "build_unavailable"


def test_predictions_schema_contains_routing_fields(tmp_path: Path) -> None:
    state = make_state()
    sample = make_sample()
    states_path = tmp_path / "states.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "out"
    states_path.write_text(json.dumps(state.to_dict()) + "\n", encoding="utf-8")
    qa_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "version": "vitalbench_eval_manifest_v0.2",
                "entries": [
                    {
                        "dataset": "demo",
                        "subject_id": "p1",
                        "recording_id": "r1",
                        "builder_tool": None,
                        "build_supported": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = eval_mod.main(
        [
            "--qa-jsonl",
            str(qa_path),
            "--states-jsonl",
            str(states_path),
            "--source-manifest",
            str(manifest_path),
            "--max-samples",
            "1",
            "--conditions",
            "agent",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert rc == 0
    record = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "expected_path" in record
    assert "actual_path" in record
    assert "actual_tools_called" in record
    assert "build_success" in record
    assert "query_success" in record
    assert record["window_locator"] == {
        "dataset": "demo",
        "patient_id": "p1",
        "window_start_s": 0.0,
        "window_end_s": 30.0,
    }
    assert "gt_metadata" in record
    assert record["gt_metadata"]["source_state_id"] == "s1"
    assert "answer" not in record["agent_input"]
    assert "evidence" not in record["agent_input"]


def test_trace_agent_writes_agent_traces_jsonl(tmp_path: Path) -> None:
    sample = make_sample()
    qa_path = tmp_path / "qa.jsonl"
    states_path = tmp_path / "states.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "out"
    qa_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    states_path.write_text("", encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"version": "vitalbench_eval_manifest_v0.2", "entries": []}),
        encoding="utf-8",
    )

    rc = eval_mod.main(
        [
            "--qa-jsonl",
            str(qa_path),
            "--states-jsonl",
            str(states_path),
            "--source-manifest",
            str(manifest_path),
            "--max-samples",
            "1",
            "--conditions",
            "agent",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--trace-agent",
        ]
    )

    assert rc == 0
    trace_record = json.loads((output_dir / "agent_traces.jsonl").read_text(encoding="utf-8"))
    assert trace_record["question_id"] == "q1"
    assert trace_record["condition"] == "agent"
    assert trace_record["tier"] == "A"
    assert trace_record["template_id"] == "ta1"
    assert trace_record["target"] == "heart_rate"
    assert trace_record["dataset"] == "demo"
    assert trace_record["summary"]["event_count"] >= 1
    assert trace_record["events"][0]["event"] == "dry_run"
    assert trace_record["events"][0]["attempt"] == 0

    prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["trace_summary"]["event_count"] >= 1
    assert prediction["trace_validation_first_passed"] is None
    assert prediction["trace_validation_first_coverage"] is None
    assert prediction["trace_validation_last_passed"] is None
    assert prediction["trace_validation_last_coverage"] is None
    assert prediction["trace_coverage_improved"] is None
    assert prediction["trace_replan_count"] == 0
    assert prediction["trace_replan_triggered"] is False
    assert prediction["trace_tool_call_count"] == 0
    assert prediction["trace_llm_call_count"] == 0
    assert (output_dir / "error_analysis_summary.json").exists()
    assert (output_dir / "error_analysis_failures.csv").exists()
    assert (output_dir / "error_analysis_failures.jsonl").exists()


def test_remaining_agent_ablation_conditions_dry_run_write_records(tmp_path: Path) -> None:
    sample = make_sample()
    qa_path = tmp_path / "qa.jsonl"
    states_path = tmp_path / "states.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "out"
    qa_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    states_path.write_text("", encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"version": "vitalbench_eval_manifest_v0.2", "entries": []}),
        encoding="utf-8",
    )

    rc = eval_mod.main(
        [
            "--qa-jsonl",
            str(qa_path),
            "--states-jsonl",
            str(states_path),
            "--source-manifest",
            str(manifest_path),
            "--max-samples",
            "1",
            "--conditions",
            "agent_no_validation",
            "agent_no_replan",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert rc == 0
    records = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["condition"] for record in records] == [
        "agent_no_validation",
        "agent_no_replan",
    ]
    eval_payload = json.loads((output_dir / "eval.json").read_text(encoding="utf-8"))
    assert set(eval_payload["aggregate"]) == {"agent_no_validation", "agent_no_replan"}
    assert set(eval_payload["resolved_config"]["llm_profiles"]) == {
        "agent",
        "planner",
        "proactive",
    }
    assert "api_key" not in json.dumps(eval_payload["resolved_config"])


def test_filter_samples_allows_supported_dataset_filter(tmp_path: Path) -> None:
    ppg_sample = make_sample("q_ppg")
    wesad_sample = make_sample("q_wesad")
    wesad_sample.dataset = "wesad"
    wesad_sample.modality = "wearable"
    wesad_sample.patient_id = "S11"

    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(
        json.dumps(ppg_sample.to_jsonl_record()) + "\n"
        + json.dumps(wesad_sample.to_jsonl_record()) + "\n",
        encoding="utf-8",
    )

    samples = eval_mod.load_qa_jsonl(qa_path)
    filtered = eval_mod.filter_samples(
        samples,
        tier="A",
        dataset=None,
        question_type=None,
        max_samples=None,
        limit_per_target=None,
        seed=42,
    )

    assert {sample.agent_input.question_id for sample in filtered} == {"q_ppg", "q_wesad"}

    wesad_only = eval_mod.filter_samples(
        samples,
        tier="A",
        dataset="wesad",
        question_type=None,
        max_samples=None,
        limit_per_target=None,
        seed=42,
    )

    assert [sample.agent_input.question_id for sample in wesad_only] == ["q_wesad"]


def test_agent_sample_uses_only_agent_input_boundary(monkeypatch) -> None:
    import contextlib
    import agent.reactive.pipeline as pipeline_mod

    captured = {}

    class FakeFinal:
        answer = "yes"
        context_used = {
            "tool_results": [],
            "token_usage": {
                "input_tokens": 4,
                "cached_input_tokens": 1,
                "output_tokens": 3,
                "total_tokens": 7,
            },
        }

        def model_dump(self):
            return {"answer": self.answer}

    class FakePipeline:
        def run_agent_input(self, agent_input):
            captured["agent_input"] = agent_input
            yield FakeFinal()

    monkeypatch.setattr(eval_mod, "restricted_mhealth_tool_pool", contextlib.nullcontext)
    monkeypatch.setattr(pipeline_mod, "ReactivePipeline", FakePipeline)

    agent_input = AgentInput(
        question_id="q1",
        question="Is my heart rate normal?",
        window_locator=WindowLocator(
            dataset="demo",
            patient_id="p1",
            window_start_s=0.0,
            window_end_s=30.0,
        ),
        answer_set=["yes", "no"],
    )

    result = eval_mod.run_agent_sample(agent_input, dry_run=False)

    assert result["prediction"] == "yes"
    assert result["token_usage"] == {
        "input_tokens": 4,
        "cached_input_tokens": 1,
        "output_tokens": 3,
        "total_tokens": 7,
    }
    boundary_payload = json.dumps(captured["agent_input"].__dict__, default=str)
    for forbidden in [
        "source_state_id",
        "template_id",
        "evidence",
        "expected_tools",
        "ground_truth_source",
        "answer_type",
        "evaluation_method",
        "target",
    ]:
        assert forbidden not in boundary_payload


def test_agent_ablation_flags_configure_pipeline(monkeypatch) -> None:
    import contextlib
    import agent.reactive.pipeline as pipeline_mod

    captured = {}

    class FakeFinal:
        answer = "yes"
        context_used = {
            "tool_results": [],
            "token_usage": {
                "input_tokens": 1,
                "cached_input_tokens": 0,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        }

        def model_dump(self):
            return {"answer": self.answer}

    class FakePipeline:
        def __init__(self):
            self.disable_validation = False
            self.max_retries = 1

        def run_agent_input(self, _agent_input):
            captured["disable_validation"] = self.disable_validation
            captured["max_retries"] = self.max_retries
            yield FakeFinal()

    monkeypatch.setattr(eval_mod, "restricted_mhealth_tool_pool", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(eval_mod, "restricted_signal_tool_pool", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(pipeline_mod, "ReactivePipeline", FakePipeline)

    agent_input = make_eval_sample().agent_input

    eval_mod.run_agent_sample(
        agent_input,
        dry_run=False,
        no_validation_baseline=True,
    )
    assert captured["disable_validation"] is True

    eval_mod.run_agent_sample(
        agent_input,
        dry_run=False,
        no_replan_baseline=True,
    )
    assert captured["disable_validation"] is False
    assert captured["max_retries"] == 0


def test_internal_previous_window_result_is_excluded_from_tool_counts() -> None:
    actual_path, names, _, query_success = eval_mod.infer_actual_path(
        [
            {
                "tool_name": "analyze_wesad_window_signal",
                "success": True,
                "data": {},
            },
            {
                "tool_name": "previous_window_comparison",
                "success": True,
                "data": {"comparison": {}},
            },
        ]
    )

    assert actual_path == "signal_query"
    assert names == ["analyze_wesad_window_signal"]
    assert query_success is True


def test_raw_mode_does_not_preload_state(tmp_path: Path) -> None:
    state = make_state()
    sample = make_sample()
    states_path = tmp_path / "states.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "out"
    states_path.write_text(json.dumps(state.to_dict()) + "\n", encoding="utf-8")
    qa_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"version": "vitalbench_eval_manifest_v0.2", "entries": []}),
        encoding="utf-8",
    )

    rc = eval_mod.main(
        [
            "--qa-jsonl",
            str(qa_path),
            "--states-jsonl",
            str(states_path),
            "--source-manifest",
            str(manifest_path),
            "--max-samples",
            "1",
            "--conditions",
            "agent",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--agent-data-mode",
            "raw",
        ]
    )

    assert rc == 0
    payload = json.loads((output_dir / "eval.json").read_text(encoding="utf-8"))
    assert payload["agent_data_mode"] == "raw"
    assert payload["preloaded_contexts"] == []
    assert payload["missing_contexts"] == []


def test_raw_result_sanitizer_removes_reference_labels() -> None:
    payload = {
        "success": True,
        "data": {
            "protocol_label": {"dominant_protocol_label": "stress"},
            "ecg_reference_hr_label": {"mean_bpm": 72.0},
            "activity": {"dominant_activity_label": "walking"},
            "bvp_pulse": {"mean_pulse_rate_bpm": 71.2},
        },
    }

    sanitized = eval_mod.sanitize_raw_tool_result(payload)

    assert "protocol_label" not in sanitized["data"]
    assert "ecg_reference_hr_label" not in sanitized["data"]
    assert "activity" not in sanitized["data"]
    assert sanitized["data"]["bvp_pulse"]["mean_pulse_rate_bpm"] == 71.2


def test_project_label_generated_states_keeps_signal_fields() -> None:
    signal_state = make_state("signal", dataset="demo")
    label_state = make_state("label", dataset="icentia11k")
    label_state.rhythm_class = "AF"
    label_state.max_hr_bpm = 128.0
    label_state.metadata = {
        "rhythm_class_source": "wfdb_aux_note_dominant_overlap",
        "sampling_rate_hz": 250,
    }

    visible = eval_mod.project_label_generated_states([signal_state, label_state])

    assert eval_mod.is_label_generated_state(signal_state) is False
    assert eval_mod.is_label_generated_state(label_state) is True
    assert len(visible) == 2
    assert visible[0] == signal_state
    assert visible[1].state_id == "label"
    assert visible[1].rhythm_class is None
    assert visible[1].max_hr_bpm == 128.0
    assert visible[1].metadata == {"sampling_rate_hz": 250}
    assert eval_mod.is_label_generated_state(visible[1]) is False


def test_eval_state_mode_projects_label_generated_state_facts(tmp_path: Path) -> None:
    state = make_state(dataset="icentia11k")
    state.modality = "ecg"
    state.af_burden_ratio = 0.5
    state.max_hr_bpm = 128.0
    state.rhythm_class = "AF"
    state.rhythm_transition_count_per_hour = 36.0
    state.metadata = {"rhythm_class_source": "wfdb_aux_note_dominant_overlap"}
    sample = make_sample()
    sample.tier = "B"
    sample.template_id = "tb1"
    sample.target = "af_presence_monitoring_window"
    sample.question_type = "single_verify"
    sample.answer = ["yes"]
    sample.answer_type = "yes_no"
    sample.answer_set = ["yes", "no"]
    sample.evaluation_method = "boolean_match"
    sample.question = "Did AF occur during the monitoring window?"
    states_path = tmp_path / "states.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "out"
    states_path.write_text(json.dumps(state.to_dict()) + "\n", encoding="utf-8")
    qa_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"version": "vitalbench_eval_manifest_v0.2", "entries": []}),
        encoding="utf-8",
    )

    rc = eval_mod.main(
        [
            "--qa-jsonl",
            str(qa_path),
            "--states-jsonl",
            str(states_path),
            "--source-manifest",
            str(manifest_path),
            "--tier",
            "B",
            "--max-samples",
            "1",
            "--conditions",
            "agent",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--agent-data-mode",
            "state",
        ]
    )

    assert rc == 0
    eval_payload = json.loads((output_dir / "eval.json").read_text(encoding="utf-8"))
    assert eval_payload["project_label_generated_state_facts"] is True
    assert eval_payload["label_generated_states_projected"] == 1


def test_restricted_mhealth_tool_pool_blocks_ecg_tools_and_tolerates_context_args() -> None:
    import agent.reactive.pipeline as pipeline_mod
    import agent.reactive.planner as planner_mod

    plan = Plan(intent=IntentType.STATUS_QUERY, reasoning="test", steps=[])

    with eval_mod.restricted_mhealth_tool_pool():
        assert planner_mod._apply_ecg_diagnosis_policy(plan, "Does this ECG show AF?") is plan

        blocked = pipeline_mod.call_tool("ecg_diagnosis")
        contexts = pipeline_mod.call_tool("state_list_contexts", dataset="demo")

    assert blocked["success"] is False
    assert "disabled during VitalBench eval" in blocked["error"]
    assert contexts["success"] is True
    assert "contexts" in contexts


def test_restricted_signal_tool_pool_blocks_state_tools() -> None:
    import agent.reactive.pipeline as pipeline_mod
    import agent.reactive.planner as planner_mod

    with eval_mod.restricted_signal_tool_pool():
        tool_names = {tool["name"] for tool in planner_mod.list_tools()}
        blocked = pipeline_mod.call_tool("state_get_current_monitoring_state")

    assert "state_get_current_monitoring_state" not in tool_names
    assert "analyze_heart_rate" in tool_names
    assert "get_ecg_description" in tool_names
    assert "analyze_icentia11k_ecg_window_signal" in tool_names
    assert "analyze_ppg_dalia_window_signal" in tool_names
    assert "analyze_wesad_window_signal" in tool_names
    assert blocked["success"] is False
    assert "raw-data eval" in blocked["error"]


def test_restricted_signal_tool_pool_is_dataset_specific_for_ppg_dalia() -> None:
    import agent.reactive.pipeline as pipeline_mod
    import agent.reactive.planner as planner_mod

    with eval_mod.restricted_signal_tool_pool(dataset="ppg_dalia"):
        tool_names = {tool["name"] for tool in planner_mod.list_tools()}
        blocked = pipeline_mod.call_tool(
            "analyze_pulse_rate",
            dataset="ppg_dalia",
            patient_id="S10",
            window_start_s=0,
            window_end_s=60,
        )

    assert tool_names == {
        "analyze_ppg_dalia_window_signal",
        "evaluate_proactive_rules",
    }
    assert blocked["success"] is False
    assert "for dataset 'ppg_dalia'" in blocked["error"]


def test_restricted_signal_tool_pool_exposes_proactive_rules_for_supported_ecg_datasets() -> None:
    import agent.reactive.planner as planner_mod

    for dataset in ("icentia11k", "wesad", "ppg_dalia", "afppgecg"):
        with eval_mod.restricted_signal_tool_pool(dataset=dataset):
            tool_names = {tool["name"] for tool in planner_mod.list_tools()}

        assert "evaluate_proactive_rules" in tool_names


def test_rhythm_context_in_raw_tool_pool_for_afppgecg() -> None:
    import agent.reactive.planner as planner_mod

    with eval_mod.restricted_signal_tool_pool(
        dataset="afppgecg",
        adaptive_scope_enabled=True,
    ):
        tool_names = {tool["name"] for tool in planner_mod.list_tools()}

    assert "analyze_afppgecg_rhythm_context" in tool_names


def test_rhythm_context_hidden_without_adaptive_scope_gate() -> None:
    import agent.reactive.planner as planner_mod

    with eval_mod.restricted_signal_tool_pool(dataset="afppgecg"):
        tool_names = {tool["name"] for tool in planner_mod.list_tools()}

    assert "analyze_afppgecg_rhythm_context" not in tool_names


def test_adaptive_scope_policy_only_enables_tier_b_af_aggregate_targets() -> None:
    assert eval_mod._adaptive_scope_enabled(
        dataset="afppgecg",
        tier="B",
        target="af_frequency_monitoring_window",
    )
    assert not eval_mod._adaptive_scope_enabled(
        dataset="afppgecg",
        tier="A",
        target="af_frequency_monitoring_window",
    )
    assert not eval_mod._adaptive_scope_enabled(
        dataset="afppgecg",
        tier="B",
        target="highest_heart_rate_numeric_monitoring_window",
    )


def test_ppg_dalia_raw_eval_keeps_chest_ecg_analysis_disabled(monkeypatch) -> None:
    import agent.reactive.pipeline as pipeline_mod
    import agent.tools.registry as registry_mod

    captured = {}

    def fake_call_tool(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return {"success": True, "data": {"ok": True}}

    monkeypatch.setattr(registry_mod, "call_tool", fake_call_tool)

    with eval_mod.restricted_signal_tool_pool(dataset="ppg_dalia"):
        result = pipeline_mod.call_tool(
            "analyze_ppg_dalia_window_signal",
            dataset="ppg_dalia",
            patient_id="S14",
            window_start_s=4110.0,
            window_end_s=4140.0,
        )

    assert result["success"] is True
    assert captured["name"] == "analyze_ppg_dalia_window_signal"
    assert "include_chest_ecg_analysis" not in captured["kwargs"]


def test_raw_signal_observability_marks_annotation_targets() -> None:
    assert eval_mod.raw_signal_observability("heart_rate_numeric_current") == "raw_signal_observable"
    assert eval_mod.raw_signal_observability("stress_state_current") == "annotation_grounded"
    assert eval_mod.raw_signal_observability("unknown_target") == "unknown"


def test_restricted_signal_tool_pool_records_tool_trace() -> None:
    import agent.reactive.pipeline as pipeline_mod

    trace = eval_mod.EvalTraceRecorder("q_trace", quiet=True, result_max_chars=200)
    with eval_mod.restricted_signal_tool_pool(trace):
        blocked = pipeline_mod.call_tool("state_get_current_monitoring_state", dataset="demo")

    assert blocked["success"] is False
    events = trace.events
    assert [event["event"] for event in events] == ["tool_start", "tool_done"]
    assert events[0]["tool_name"] == "state_get_current_monitoring_state"
    assert events[1]["success"] is False
    assert "result" in events[1]


def test_traced_validation_gate_records_structured_issue() -> None:
    trace = eval_mod.EvalTraceRecorder(
        "q_validation",
        condition="agent",
        quiet=True,
        result_max_chars=200,
    )
    plan = Plan(
        intent=IntentType.STATUS_QUERY,
        reasoning="check heart rate",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_heart_rate",
                tool_args={},
                description="Analyze HR",
            )
        ],
    )
    results = [
        ToolResult(
            step_id=1,
            tool_name="analyze_heart_rate",
            success=False,
            error="tool failed",
        )
    ]

    with eval_mod.traced_validation_gate(trace):
        validation = ValidationGate().validate(plan, results)

    assert validation.passed is False
    event = trace.events[0]
    assert event["event"] == "validation_done"
    assert event["attempt"] == 0
    assert event["passed"] is False
    assert event["has_critical"] is True
    assert event["coverage"] == 0.0
    assert event["issue_types"] == ["tool_failure"]
    assert event["issue_severities"] == ["critical"]
    assert event["tool_names"] == ["analyze_heart_rate"]
    assert event["issue_tool_names"] == ["analyze_heart_rate"]

    summary = trace.summarize()
    assert summary["validation_first_passed"] is False
    assert summary["validation_first_coverage"] == 0.0
    assert summary["validation_last_passed"] is False
    assert summary["validation_last_coverage"] == 0.0
    assert summary["coverage_improved"] is False
    assert summary["validation_critical_first"] is True
    assert summary["validation_issue_types"] == ["tool_failure"]


def test_run_agent_sample_trace_records_deterministic_replan(monkeypatch) -> None:
    import agent.reactive.llm_agent as llm_agent_mod
    import agent.reactive.pipeline as pipeline_mod
    import agent.reactive.planner as planner_mod

    initial_plan = Plan(
        intent=IntentType.STATUS_QUERY,
        reasoning="initial",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_heart_rate",
                tool_args={"record_id": "r1"},
                description="Analyze HR",
            )
        ],
    )
    revised_plan = Plan(
        intent=IntentType.STATUS_QUERY,
        reasoning="revised",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_hrv",
                tool_args={"record_id": "r1"},
                description="Analyze HRV",
            )
        ],
    )
    validation_calls: list[list[str]] = []

    def fake_create_plan(self, *args, **kwargs):
        return initial_plan, zero_usage()

    def fake_replan(self, *, previous_plan, issues, user_query):
        assert previous_plan == initial_plan
        assert issues == ["first attempt failed"]
        assert user_query
        return revised_plan, zero_usage()

    def fake_validate(self, plan, results):
        validation_calls.append([step.tool_name for step in plan.steps])
        if len(validation_calls) == 1:
            return ValidationResult(
                passed=False,
                coverage=0.5,
                issues=[
                    ValidationIssue(
                        step_id=1,
                        tool_name="analyze_heart_rate",
                        issue_type=IssueType.TOOL_FAILURE,
                        severity=IssueSeverity.CRITICAL,
                        message="first attempt failed",
                    )
                ],
            )
        return ValidationResult(passed=True, coverage=1.0, issues=[])

    def fake_call_tool(name: str, **kwargs):
        return {
            "success": True,
            "data": {"tool_name": name, "args": kwargs},
            "metadata": {"source": "test"},
        }

    def fake_generate_answer(self, user_query, plan, tool_results, extra_context=""):
        assert plan == revised_plan
        assert [result.tool_name for result in tool_results] == ["analyze_hrv"]
        yield LLMStreamEvent(delta="yes")
        yield LLMStreamEvent(usage=zero_usage())

    monkeypatch.setattr(planner_mod.Planner, "create_plan", fake_create_plan)
    monkeypatch.setattr(planner_mod.Planner, "replan", fake_replan)
    monkeypatch.setattr(ValidationGate, "validate", fake_validate)
    monkeypatch.setattr(pipeline_mod, "call_tool", fake_call_tool)
    monkeypatch.setattr(llm_agent_mod.LLMAgent, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(
        eval_mod,
        "restricted_signal_tool_pool",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    sample = make_eval_sample("q_replan_path")
    trace = eval_mod.EvalTraceRecorder(
        sample.agent_input.question_id,
        condition="agent",
        tier="A",
        template_id="ta3",
        target="heart_rate_numeric_current",
        dataset="demo",
        quiet=True,
        result_max_chars=200,
    )

    result = eval_mod.run_agent_sample(
        sample.agent_input,
        dry_run=False,
        data_mode="raw",
        trace=trace,
        benchmark_tier="A",
        benchmark_target="heart_rate_numeric_current",
    )

    assert result["prediction"] == "yes"
    validation_events = [
        event for event in trace.events if event["event"] == "validation_done"
    ]
    assert [event["attempt"] for event in validation_events] == [0, 1]
    assert validation_events[0]["issue_types"] == ["tool_failure"]
    assert validation_events[0]["coverage"] < 1.0
    assert validation_events[1]["passed"] is True

    replan_events = [event for event in trace.events if event["event"] == "replan_done"]
    assert len(replan_events) == 1
    replan = replan_events[0]
    assert replan["attempt"] == 1
    assert replan["from_attempt"] == 0
    assert replan["to_attempt"] == 1
    assert replan["previous_tool_names"] == ["analyze_heart_rate"]
    assert replan["new_tool_names"] == ["analyze_hrv"]
    assert replan["added_tools"] == ["analyze_hrv"]
    assert replan["removed_tools"] == ["analyze_heart_rate"]
    assert replan["issues_used_for_replan"] == ["first attempt failed"]

    summary = trace.summarize()
    assert summary["replan_count"] == 1
    assert summary["plan_changed_after_replan"] is True
    assert summary["validation_first_coverage"] == 0.5
    assert summary["validation_last_passed"] is True
    assert summary["validation_last_coverage"] == 1.0
    assert summary["coverage_improved"] is True
    assert summary["initial_plan_signature"] != summary["final_plan_signature"]
    assert summary["initial_tool_names"] == ["analyze_heart_rate"]
    assert summary["final_tool_names"] == ["analyze_hrv"]


def test_trace_summary_reports_orchestration_duration() -> None:
    trace = eval_mod.EvalTraceRecorder(
        "q_orchestration",
        quiet=True,
        result_max_chars=200,
    )
    trace.add(
        "orchestration_done",
        {"orchestration": "previous_window_comparison", "success": True},
        duration_sec=1.25,
    )

    summary = trace.summarize()

    assert summary["orchestration_duration_sec"] == 1.25
    assert summary["tool_duration_sec"] == 0.0


def test_trace_summary_reports_plan_signatures_and_replan_change() -> None:
    initial_plan = Plan(
        intent=IntentType.STATUS_QUERY,
        reasoning="initial",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_heart_rate",
                tool_args={"record_id": "r1"},
                description="Analyze HR",
            )
        ],
    )
    revised_plan = Plan(
        intent=IntentType.STATUS_QUERY,
        reasoning="revised",
        steps=[
            PlanStep(
                step_id=1,
                tool_name="analyze_hrv",
                tool_args={"record_id": "r1"},
                description="Analyze HRV",
            )
        ],
    )
    trace = eval_mod.EvalTraceRecorder(
        "q_replan",
        condition="agent",
        quiet=True,
        result_max_chars=200,
    )

    trace.add("planner_done", eval_mod._plan_trace_payload(initial_plan))
    trace.current_attempt = 1
    trace.add(
        "replan_done",
        {
            **eval_mod._plan_trace_payload(revised_plan),
            "attempt": 1,
            "from_attempt": 0,
            "to_attempt": 1,
        },
    )
    trace.add("tool_done", {"tool_name": "analyze_hrv", "success": True})
    trace.add("answer_done", {"chars": 3})

    summary = trace.summarize()

    assert summary["replan_count"] == 1
    assert summary["replan_triggered"] is True
    assert summary["initial_tool_names"] == ["analyze_heart_rate"]
    assert summary["final_tool_names"] == ["analyze_hrv"]
    assert summary["initial_plan_signature"] != summary["final_plan_signature"]
    assert summary["plan_changed_after_replan"] is True
    assert summary["tool_call_count"] == 1
    assert summary["llm_call_count"] == 3
