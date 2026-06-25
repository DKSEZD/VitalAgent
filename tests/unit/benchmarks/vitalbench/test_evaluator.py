from __future__ import annotations

import pytest

from agent.benchmarks.vitalbench.eval_loader import EvalSpec
from agent.benchmarks.vitalbench.evaluator import evaluate_prediction


def _spec(
    *,
    evaluation_method: str,
    answer: list[str],
    answer_type: str = "category",
    numeric_tolerance: float | None = None,
    answer_set: list[str] | None = None,
) -> EvalSpec:
    return EvalSpec(
        question_id="q1",
        answer=answer,
        answer_type=answer_type,
        evaluation_method=evaluation_method,
        numeric_tolerance=numeric_tolerance,
        answer_set=answer_set,
    )


def test_evaluate_prediction_dispatches_boolean_match() -> None:
    spec = _spec(
        evaluation_method="boolean_match",
        answer=["yes"],
        answer_type="yes_no",
        answer_set=["yes", "no"],
    )

    score, error_class = evaluate_prediction(spec, "Yes, this window is positive.")

    assert score is True
    assert error_class is None


def test_evaluate_prediction_dispatches_normalized_match() -> None:
    spec = _spec(
        evaluation_method="normalized_match",
        answer=["low"],
        answer_type="category",
        answer_set=["low", "normal", "high"],
    )

    score, error_class = evaluate_prediction(spec, "A. low heart rate")

    assert score is True
    assert error_class is None


def test_normalized_match_prefers_answer_prefix_over_later_option_mentions() -> None:
    spec = _spec(
        evaluation_method="normalized_match",
        answer=["normal"],
        answer_type="category",
        answer_set=["low", "normal", "high"],
    )

    score, error_class = evaluate_prediction(
        spec,
        "normal\n\nThe heart rate is not elevated or low.",
    )

    assert score is True
    assert error_class is None


def test_evaluate_prediction_dispatches_numeric_tolerance() -> None:
    spec = _spec(
        evaluation_method="numeric_tolerance",
        answer=["72"],
        answer_type="numeric",
        numeric_tolerance=3.0,
    )

    score, error_class = evaluate_prediction(spec, "about 74 bpm")

    assert score is True
    assert error_class is None


def test_numeric_tolerance_uses_direct_answer_not_earlier_stats() -> None:
    spec = _spec(
        evaluation_method="numeric_tolerance",
        answer=["62.3"],
        answer_type="numeric",
        numeric_tolerance=5.0,
    )

    score, error_class = evaluate_prediction(
        spec,
        "The tool reported min 31 bpm and max 122 bpm; the answer is approximately 60 bpm.",
    )

    assert score is True
    assert error_class is None


def test_numeric_tolerance_ignores_normal_range_distractors() -> None:
    spec = _spec(
        evaluation_method="numeric_tolerance",
        answer=["62.3"],
        answer_type="numeric",
        numeric_tolerance=5.0,
    )

    score, error_class = evaluate_prediction(
        spec,
        "The normal range is 60-100 bpm, but my estimate is about 80 bpm.",
    )

    assert score is False
    assert error_class == "unknown"


def test_evaluate_prediction_uses_method_not_answer_type_fallback() -> None:
    spec = _spec(
        evaluation_method="normalized_match",
        answer=["72"],
        answer_type="numeric",
        numeric_tolerance=3.0,
    )

    score, error_class = evaluate_prediction(spec, "about 74 bpm")

    assert score is False
    assert error_class == "unknown"


def test_evaluate_prediction_rejects_unknown_method() -> None:
    spec = _spec(
        evaluation_method="exact_match",
        answer=["yes"],
        answer_type="yes_no",
    )

    with pytest.raises(ValueError, match="Unsupported VitalBench evaluation_method"):
        evaluate_prediction(spec, "yes")
