from __future__ import annotations

from agent.benchmarks.vitalbench.templates import QA_TEMPLATES
from agent.reactive import previous_window_compare as temporal
from agent.reactive.pipeline import ReactivePipeline
from agent.schemas import ToolResult
from agent.tools.registry import list_tools


def test_previous_window_comparison_computes_meaningful_hr_delta(monkeypatch) -> None:
    calls: list[tuple[float, float]] = []

    def fake_dispatcher(
        dataset: str,
        patient_id: str,
        start_s: float,
        end_s: float,
    ) -> dict:
        calls.append((start_s, end_s))
        mean_bpm = 90.0 if start_s == 100.0 else 80.0
        return {
            "success": True,
            "data": {
                "heart_rate": {"mean_bpm": mean_bpm},
                "pp_intervals": {"mean_ms": 700.0},
                "irregularity_metrics": {
                    "coefficient_of_variation": 0.1,
                    "rmssd_ms": 50.0,
                },
                "assessment": "regular",
            },
        }

    monkeypatch.setitem(temporal.DISPATCHERS, "afppgecg", fake_dispatcher)

    result = temporal.compare_with_previous_window(
        "afppgecg",
        "001",
        100.0,
        130.0,
    )

    assert result["success"] is True
    assert calls == [(100.0, 130.0), (70.0, 100.0)]
    hr_change = result["data"]["comparison"]["heart_rate.mean_bpm"]
    assert hr_change["delta"] == 10.0
    assert hr_change["meaningfully_higher"] is True
    interpretation = result["data"]["heart_rate_change_interpretation"]
    assert interpretation["changed_meaningfully"] is True
    assert interpretation["meaningful_direction"] == "higher"
    assert "compare_with_previous_window" not in {
        tool["name"] for tool in list_tools()
    }


def test_previous_window_comparison_accepts_legacy_dataset_name(monkeypatch) -> None:
    seen_datasets: list[str] = []

    def fake_dispatcher(
        dataset: str,
        patient_id: str,
        start_s: float,
        end_s: float,
    ) -> dict:
        seen_datasets.append(dataset)
        return {"success": True, "data": {}}

    monkeypatch.setitem(temporal.DISPATCHERS, "afppgecg", fake_dispatcher)

    result = temporal.compare_with_previous_window("af_ppg_ecg", "001", 100.0, 130.0)

    assert result["success"] is True
    assert result["metadata"]["dataset"] == "afppgecg"
    assert seen_datasets == ["afppgecg", "afppgecg"]


def test_previous_window_comparison_reuses_current_data(monkeypatch) -> None:
    calls: list[tuple[float, float]] = []

    def fake_dispatcher(
        dataset: str,
        patient_id: str,
        start_s: float,
        end_s: float,
    ) -> dict:
        calls.append((start_s, end_s))
        mean_bpm = 90.0 if start_s == 100.0 else 80.0
        return {
            "success": True,
            "data": {
                "heart_rate": {"mean_bpm": mean_bpm},
                "pp_intervals": {"mean_ms": 750.0},
                "irregularity_metrics": {
                    "coefficient_of_variation": 0.1,
                    "rmssd_ms": 50.0,
                },
                "assessment": "regular",
            },
        }

    monkeypatch.setitem(temporal.DISPATCHERS, "afppgecg", fake_dispatcher)
    current_data = {
        "heart_rate": {"mean_bpm": 90.0},
        "pp_intervals": {"mean_ms": 750.0},
        "irregularity_metrics": {
            "coefficient_of_variation": 0.1,
            "rmssd_ms": 50.0,
        },
        "assessment": "regular",
    }

    baseline = temporal.compare_with_previous_window(
        "afppgecg",
        "001",
        100.0,
        130.0,
    )
    calls.clear()
    reused = temporal.compare_with_previous_window(
        "afppgecg",
        "001",
        100.0,
        130.0,
        current_data=current_data,
    )

    assert calls == [(70.0, 100.0)]
    assert reused["data"] == baseline["data"]
    assert reused["data"]["current"] == current_data
    assert reused["data"]["comparison"]["heart_rate.mean_bpm"]["delta"] == 10.0


def test_afppgecg_current_data_requires_and_merges_both_components() -> None:
    pulse = ToolResult(
        step_id=1,
        tool_name="analyze_pulse_rate",
        success=True,
        data={"data": {"heart_rate": {"mean_bpm": 90.0}}},
    )
    rhythm = ToolResult(
        step_id=2,
        tool_name="analyze_ppg_rhythm_irregularity",
        success=True,
        data={"data": {"assessment": "regular", "is_irregular": False}},
    )

    assert ReactivePipeline._current_window_data_for_comparison(
        dataset="afppgecg",
        results=[pulse],
    ) is None
    assert ReactivePipeline._current_window_data_for_comparison(
        dataset="afppgecg",
        results=[pulse, rhythm],
    ) == {
        "heart_rate": {"mean_bpm": 90.0},
        "assessment": "regular",
        "is_irregular": False,
    }


def test_previous_window_phrase_matcher() -> None:
    for phrase in temporal.PREVIOUS_WINDOW_PHRASES:
        assert temporal.needs_previous_window_comparison(
            f"Did this change {phrase}?"
        )
    assert temporal.needs_previous_window_comparison("What is my heart rate now?") is False


def test_previous_window_phrase_matcher_matches_only_previous_window_templates() -> None:
    for template in QA_TEMPLATES:
        expected = template.time_scope == "previous_window_comparison"
        for question in (template.canonical, *template.paraphrases):
            assert temporal.needs_previous_window_comparison(question) is expected, (
                template.template_id,
                question,
            )
