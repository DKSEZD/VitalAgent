"""Deterministic scoring for VitalBench EvalSpec objects."""

from __future__ import annotations

import os
import re
import string
from statistics import median

from agent.benchmarks.vitalbench.eval_loader import EvalSpec


YES_VARIANTS = {"yes", "true", "present", "detected", "abnormal", "positive"}
NO_VARIANTS = {"no", "false", "absent", "not detected", "normal", "negative"}
SUPPORTED_EVALUATION_METHODS = frozenset(
    {
        "boolean_match",
        "normalized_match",
        "numeric_tolerance",
    }
)


def _strip_answer_prefix(text: str) -> str:
    return re.sub(r"^\s*answer\s*:\s*", "", text, count=1, flags=re.IGNORECASE)


def _strip_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    text = _strip_answer_prefix(text)
    return " ".join(text.lower().translate(table).split())


def _first_boolean(text: str) -> bool | None:
    normalized = _strip_punctuation(text)
    tokens = normalized.split()
    if not tokens:
        return None
    first = tokens[0]
    first_two = " ".join(tokens[:2])
    if first_two in NO_VARIANTS:
        return False
    if first in YES_VARIANTS:
        return True
    if first in NO_VARIANTS:
        return False
    return None


def boolean_match(prediction: str, gold: str | list[str]) -> bool:
    """Score yes/no answers by the first boolean-like token."""

    gold_text = gold[0] if isinstance(gold, list) and gold else str(gold)
    pred_value = _first_boolean(prediction)
    gold_value = _first_boolean(gold_text)
    return pred_value is not None and gold_value is not None and pred_value == gold_value


def normalized_match(
    prediction: str,
    gold: str | list[str],
    answer_set: list[str] | None = None,
) -> bool:
    """Score categorical answers after simple punctuation/case normalization."""

    gold_values = gold if isinstance(gold, list) else [str(gold)]
    pred_norm = _strip_punctuation(prediction)
    if answer_set:
        members = {_strip_punctuation(member): member for member in answer_set}
        for member_norm in members:
            if pred_norm == member_norm or pred_norm.startswith(member_norm + " "):
                pred_norm = member_norm
                break
        else:
            pred_tokens = pred_norm.split()
            for member_norm in members:
                if member_norm in pred_tokens:
                    pred_norm = member_norm
                    break
    return any(pred_norm == _strip_punctuation(value) for value in gold_values)


def _category_label(prediction: str, answer_set: list[str] | None) -> str | None:
    pred_norm = _strip_punctuation(prediction)
    if not pred_norm:
        return None
    if answer_set:
        members = {_strip_punctuation(member): member for member in answer_set}
        for member_norm, member in members.items():
            if pred_norm == member_norm or pred_norm.startswith(member_norm + " "):
                return member
        pred_tokens = pred_norm.split()
        for member_norm, member in members.items():
            if member_norm in pred_tokens:
                return member
        return None
    return pred_norm


_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
_NUMERIC_CUE_RE = re.compile(
    r"\b(?:answer|approximately|approx|about|around|estimate|estimated|mean|average|avg|"
    r"heart\s*rate|pulse\s*rate|bpm|beats\s+per\s+minute|percent|ratio|fraction|seconds?|"
    r"minutes?|ms|milliseconds?)\b|%",
    re.IGNORECASE,
)
_NUMERIC_DISTRACTOR_RE = re.compile(
    r"\b(?:min|max|minimum|maximum|std|standard\s+deviation|range|normal\s+range|"
    r"threshold|quality|score|confidence|count|window|start|end|sample|sampling|cv|rmssd|pnn50)\b",
    re.IGNORECASE,
)


def _first_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def _number_candidates(text: str) -> list[tuple[float, int, int]]:
    return [(float(match.group(0)), match.start(), match.end()) for match in _NUMBER_RE.finditer(text)]


def _preferred_prediction_numbers(text: str) -> list[float]:
    """Return numbers that look like direct numeric answers, not supporting stats."""

    candidates = _number_candidates(text)
    preferred: list[float] = []
    for value, start, end in candidates:
        context = text[max(0, start - 36): min(len(text), end + 36)]
        if _NUMERIC_DISTRACTOR_RE.search(context):
            continue
        if _NUMERIC_CUE_RE.search(context):
            preferred.append(value)
    return preferred


def numeric_tolerance_match(
    prediction: str,
    gold: str | list[str],
    tolerance: float | None,
) -> bool:
    """Score numeric answers by absolute tolerance around the first number."""

    score, _method = numeric_tolerance_match_details(prediction, gold, tolerance)
    return score


def _configured_relative_tolerance() -> float:
    raw_value = os.getenv("EVAL_NUMERIC_RELATIVE_TOLERANCE")
    if raw_value is None or raw_value.strip() == "":
        return 0.05
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 0.05


def numeric_tolerance_match_details(
    prediction: str,
    gold: str | list[str],
    tolerance: float | None,
) -> tuple[bool, str]:
    """Score numeric answers and report the numeric match strategy used."""

    gold_text = gold[0] if isinstance(gold, list) and gold else str(gold)
    gold_number = _first_number(gold_text)
    use_absolute = tolerance is not None and float(tolerance) > 0
    default_method = "absolute" if use_absolute else "exact"
    if gold_number is None:
        return False, default_method
    preferred_numbers = _preferred_prediction_numbers(prediction)
    pred_numbers = preferred_numbers or [
        number for number, _start, _end in _number_candidates(prediction)[:1]
    ]
    if not pred_numbers:
        return False, default_method
    if use_absolute:
        return (
            any(abs(pred_number - gold_number) <= float(tolerance) for pred_number in pred_numbers),
            "absolute",
        )
    if any(pred_number == gold_number for pred_number in pred_numbers):
        return True, "exact"
    relative_tolerance = _configured_relative_tolerance()
    if relative_tolerance <= 0:
        return False, "exact"
    if gold_number == 0:
        return any(abs(pred_number - gold_number) <= 0.5 for pred_number in pred_numbers), "absolute"
    allowed_delta = relative_tolerance * abs(gold_number)
    return (
        any(abs(pred_number - gold_number) <= allowed_delta for pred_number in pred_numbers),
        "relative_5pct",
    )


def _majority_vote(values: list[object | None]) -> object | None:
    counts: dict[object, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    best_count = max(counts.values())
    tied = {value for value, count in counts.items() if count == best_count}
    for value in values:
        if value in tied:
            return value
    return None


def _evaluate_prediction_list(
    eval_spec: EvalSpec,
    prediction: list[str],
) -> tuple[bool, str | None, str | None]:
    if not prediction:
        return evaluate_prediction_details(eval_spec, "")
    method = str(eval_spec.evaluation_method)
    if method == "boolean_match":
        voted = _majority_vote([_first_boolean(item) for item in prediction])
        if voted is None:
            return evaluate_prediction_details(eval_spec, prediction[0])
        gold_text = eval_spec.answer[0] if eval_spec.answer else ""
        gold_value = _first_boolean(gold_text)
        score = gold_value is not None and voted == gold_value
        return score, None if score else "unknown", "cot_sc_majority"
    if method == "normalized_match":
        voted = _majority_vote([
            _category_label(item, eval_spec.answer_set) for item in prediction
        ])
        if voted is None:
            return evaluate_prediction_details(eval_spec, prediction[0])
        score = any(_strip_punctuation(str(voted)) == _strip_punctuation(value) for value in eval_spec.answer)
        return score, None if score else "unknown", "cot_sc_majority"
    if method == "numeric_tolerance":
        numbers: list[float] = []
        for item in prediction:
            preferred = _preferred_prediction_numbers(item)
            if preferred:
                numbers.append(preferred[0])
                continue
            fallback = _first_number(item)
            if fallback is not None:
                numbers.append(fallback)
        if not numbers:
            return evaluate_prediction_details(eval_spec, prediction[0])
        score, _method = numeric_tolerance_match_details(
            str(float(median(numbers))),
            eval_spec.answer,
            eval_spec.numeric_tolerance,
        )
        return score, None if score else "unknown", "cot_sc_median"
    return evaluate_prediction_details(eval_spec, prediction[0])


def evaluate_prediction(eval_spec: EvalSpec, prediction: str | list[str]) -> tuple[bool, str | None]:
    """Score a prediction using only EvalSpec.evaluation_method."""

    score, error_class, _numeric_match_method = evaluate_prediction_details(eval_spec, prediction)
    return score, error_class


def evaluate_prediction_details(
    eval_spec: EvalSpec,
    prediction: str | list[str],
) -> tuple[bool, str | None, str | None]:
    """Score a prediction and include numeric match metadata when applicable."""

    if isinstance(prediction, list):
        return _evaluate_prediction_list(eval_spec, [str(item) for item in prediction])

    method = str(eval_spec.evaluation_method)
    numeric_match_method = None
    if method == "boolean_match":
        score = boolean_match(prediction, eval_spec.answer)
    elif method == "normalized_match":
        score = normalized_match(prediction, eval_spec.answer, eval_spec.answer_set)
    elif method == "numeric_tolerance":
        score, numeric_match_method = numeric_tolerance_match_details(
            prediction,
            eval_spec.answer,
            eval_spec.numeric_tolerance,
        )
    else:
        raise ValueError(
            "Unsupported VitalBench evaluation_method "
            f"{method!r}; expected one of {sorted(SUPPORTED_EVALUATION_METHODS)}"
        )
    return score, None if score else "unknown", numeric_match_method


__all__ = [
    "SUPPORTED_EVALUATION_METHODS",
    "boolean_match",
    "evaluate_prediction_details",
    "normalized_match",
    "numeric_tolerance_match",
    "numeric_tolerance_match_details",
    "evaluate_prediction",
]
