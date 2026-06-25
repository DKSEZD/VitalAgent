"""
State-to-QA generator for VitalBench.

This module converts structured MonitoringState objects into QA samples
using the templates defined in templates.py.

Core idea:
    MonitoringState + QATemplate + optional generation parameters
        -> VitalBenchSample with explicit evidence

The generator is deterministic by default. It does not call an LLM.

Important boundary:
- This generator only consumes already-computed MonitoringState fields.
- It does not perform signal processing, rhythm classification, AF episode
  filtering, or baseline construction.
- Therefore, evidence fields only record values actually used by this
  generator to derive the QA answer.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.benchmarks.vitalbench.schemas import VitalBenchSample
from agent.benchmarks.vitalbench.templates import QATemplate, get_templates
from agent.mhealth.schemas import MonitoringState


# ---------------------------------------------------------------------------
# Generator-level config
# ---------------------------------------------------------------------------

# The previous window is defined as the immediately preceding window.
# This constant is kept here, not in templates.py, because it is part of
# dataset generation configuration rather than question semantics.
PREVIOUS_WINDOW_OFFSET = 1


@dataclass(frozen=True)
class AnswerPayload:
    """
    Intermediate answer payload returned by answer functions.
    """

    answer: str
    answer_type: str
    evidence: dict[str, Any]


# ---------------------------------------------------------------------------
# Public generator API
# ---------------------------------------------------------------------------


def generate_qa_from_state(
    state: MonitoringState,
) -> list[VitalBenchSample]:
    """
    Generate all applicable QA samples from one MonitoringState.

    Parameters
    ----------
    state:
        Structured monitoring state.
    Returns
    -------
    list[VitalBenchSample]
        Generated QA samples.
    """
    templates = get_templates(
        dataset=state.dataset,
        modality=state.modality,
    )

    samples: list[VitalBenchSample] = []

    for template in templates:
        if not state.has_required_fields(template.required_fields):
            continue

        for generation_params in _expand_template_parameters(template):
            answer_payload = _answer_template(template, state, generation_params)
            if answer_payload is None:
                continue

            for phrasing_index, raw_question in enumerate(template.all_phrasings):
                question = _format_question(raw_question, generation_params)
                question_id = build_question_id(
                    state=state,
                    template=template,
                    phrasing_index=phrasing_index,
                    generation_params=generation_params,
                )

                samples.append(
                    VitalBenchSample(
                        question_id=question_id,
                        template_id=template.template_id,
                        source_state_id=state.state_id,
                        patient_id=state.patient_id,
                        dataset=state.dataset,
                        modality=state.modality,
                        window_start_s=state.window_start_s,
                        window_end_s=state.window_end_s,
                        window_duration_s=state.window_duration_s,
                        tier=template.tier,
                        question_type=template.question_type,
                        target=template.target,
                        time_scope=template.time_scope,
                        difficulty_tier=template.difficulty_tier,
                        question=question,
                        answer=[answer_payload.answer],
                        answer_type=answer_payload.answer_type,
                        answer_set=list(template.answer_set),
                        ground_truth_source=template.ground_truth_source,
                        evaluation_method=template.evaluation_method,
                        numeric_tolerance=template.numeric_tolerance,
                        evidence=answer_payload.evidence,
                        expected_tools=list(template.expected_tools),
                        phrasing_index=phrasing_index,
                        generation_params=dict(generation_params),
                        notes=_build_sample_notes(template=template, state=state),
                    )
                )

    return samples


def generate_qa_from_states(
    states: Iterable[MonitoringState],
) -> list[VitalBenchSample]:
    """
    Generate QA samples from multiple states.
    """
    samples: list[VitalBenchSample] = []
    for state in states:
        samples.extend(generate_qa_from_state(state))
    validate_unique_question_ids(samples)
    return samples


def validate_unique_question_ids(samples: Iterable[VitalBenchSample]) -> None:
    """Raise ValueError if duplicated question IDs are found."""

    seen: set[str] = set()
    duplicates: list[str] = []

    for sample in samples:
        if sample.question_id in seen:
            duplicates.append(sample.question_id)
        seen.add(sample.question_id)

    if duplicates:
        raise ValueError(
            "Duplicated question_id values found: "
            + ", ".join(sorted(set(duplicates))[:10])
        )


def write_jsonl(
    samples: Iterable[VitalBenchSample],
    output_path: str | Path,
) -> None:
    """
    Write generated QA samples to a JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_jsonl_record(), ensure_ascii=False) + "\n")


def summarize_samples(samples: Iterable[VitalBenchSample]) -> dict[str, Any]:
    """
    Return simple dataset statistics for sanity checking.
    """
    sample_list = list(samples)

    def count_by(field_name: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for sample in sample_list:
            value = str(getattr(sample, field_name))
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    return {
        "sample_count": len(sample_list),
        "by_dataset": count_by("dataset"),
        "by_modality": count_by("modality"),
        "by_tier": count_by("tier"),
        "by_question_type": count_by("question_type"),
        "by_target": count_by("target"),
        "by_difficulty": count_by("difficulty_tier"),
        "by_ground_truth_source": count_by("ground_truth_source"),
        "by_evaluation_method": count_by("evaluation_method"),
    }


# ---------------------------------------------------------------------------
# Question ID / parameter handling
# ---------------------------------------------------------------------------


def build_question_id(
    *,
    state: MonitoringState,
    template: QATemplate,
    phrasing_index: int,
    generation_params: dict[str, Any],
) -> str:
    """
    Build a stable question ID.

    Examples
    --------
    patient_001__ta1_af_presence_current__p0
    patient_001__tb3_heart_rate_excursion_threshold_monitoring_window__threshold_bpm120__p1
    """
    param_part = _generation_param_id_part(generation_params)
    if param_part:
        return f"{state.state_id}__{template.template_id}__{param_part}__p{phrasing_index}"
    return f"{state.state_id}__{template.template_id}__p{phrasing_index}"


def _generation_param_id_part(generation_params: dict[str, Any]) -> str:
    """
    Build a deterministic string segment for concrete generation parameters.
    """
    if not generation_params:
        return ""

    parts = []
    for key in sorted(generation_params):
        value = generation_params[key]
        safe_value = str(value).replace(".", "p")
        parts.append(f"{key}{safe_value}")

    return "__".join(parts)


def _expand_template_parameters(template: QATemplate) -> list[dict[str, Any]]:
    """
    Expand template parameters into concrete generation parameter dicts.

    Example
    -------
    parameters={"threshold_bpm": (100, 110, 120, 130)}

    becomes:
    [
        {"threshold_bpm": 100},
        {"threshold_bpm": 110},
        {"threshold_bpm": 120},
        {"threshold_bpm": 130},
    ]
    """
    if not template.parameters:
        return [{}]

    keys = list(template.parameters.keys())
    values = [template.parameters[key] for key in keys]

    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        expanded.append(dict(zip(keys, combo, strict=True)))

    return expanded


def _format_question(question: str, generation_params: dict[str, Any]) -> str:
    """
    Fill parameterized question text.

    Example
    -------
    "Did my HR go above {threshold_bpm} bpm?"
    """
    if not generation_params:
        return question
    return question.format(**generation_params)

PPG_AF_CAVEAT_TARGETS = {
    "af_presence_current",
    "af_presence_monitoring_window",
    "af_frequency_monitoring_window",
}


PPG_AF_METHOD_CAVEAT = (
    "On PPG datasets, 'AF presence' is inferred from rhythm irregularity "
    "rather than direct observation of P-wave absence. The GT comes from "
    "dataset annotation, but the agent's tool response on PPG is fundamentally "
    "an indirect inference. This is a methodological caveat for the paper, "
    "not a model bug."
)


def _build_sample_notes(
    *,
    template: QATemplate,
    state: MonitoringState,
) -> str:
    """Build sample-level notes with modality-specific caveats."""

    notes = template.notes.strip()

    if state.modality == "ppg" and template.target in PPG_AF_CAVEAT_TARGETS:
        if notes:
            notes += " "
        notes += PPG_AF_METHOD_CAVEAT

    return notes


# ---------------------------------------------------------------------------
# Answer dispatch
# ---------------------------------------------------------------------------


AnswerFn = Callable[[MonitoringState, dict[str, Any]], AnswerPayload | None]


def _answer_template(
    template: QATemplate,
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Dispatch one template to its registered answer function.
    """
    try:
        fn = ANSWER_FUNCTIONS[template.answer_fn]
    except KeyError as exc:
        raise KeyError(
            f"No answer function registered for template answer_fn={template.answer_fn!r}"
        ) from exc

    return fn(state, generation_params)


def _hr_provenance_evidence(state: MonitoringState) -> dict[str, Any]:
    """Return optional HR provenance details recorded by dataset builders."""

    metadata = state.metadata
    evidence: dict[str, Any] = {}
    for key in (
        "source_variant",
        "hr_source",
        "mean_hr_bpm_source",
        "max_hr_bpm_source",
    ):
        value = metadata.get(key)
        if value is not None:
            evidence[key] = value

    label_source = metadata.get("hr_label_source")
    if label_source is not None:
        if evidence.get("hr_source") is None:
            evidence["hr_source"] = label_source
        elif label_source != evidence.get("hr_source"):
            evidence["reference_hr_source"] = label_source

    reference_signal = metadata.get("reference_signal")
    if reference_signal is not None:
        evidence["reference_signal"] = reference_signal

    return evidence


# ---------------------------------------------------------------------------
# Answer functions: Tier A
# ---------------------------------------------------------------------------


def answer_af_presence_current(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Answer whether the current window indicates AF.
    """
    del generation_params

    rhythm = state.rhythm_class
    if rhythm is None:
        return None

    answer = "yes" if rhythm == "AF" else "no"

    evidence: dict[str, Any] = {
        "field": "rhythm_class",
        "value": rhythm,
        "operator": "==",
        "threshold": "AF",
    }

    if state.modality == "ppg":
        evidence["methodological_caveat"] = (
            "For PPG, AF presence is an indirect rhythm-irregularity inference, "
            "not direct observation of P-wave absence."
        )

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence=evidence,
    )


def answer_heart_rate_category_current(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Categorize current heart rate into low / normal / high.
    """
    del generation_params

    hr = state.hr_bpm
    if hr is None:
        return None

    if hr < 60:
        category = "low"
    elif hr > 100:
        category = "high"
    else:
        category = "normal"

    evidence = {
            "field": "hr_bpm",
            "value": hr,
            "unit": "bpm",
            "bins": {
                "low": "<60",
                "normal": "60-100",
                "high": ">100",
            },
        }
    evidence.update(_hr_provenance_evidence(state))

    return AnswerPayload(
        answer=category,
        answer_type="category",
        evidence=evidence,
    )


def answer_heart_rate_query_current(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Return the current heart rate as a numeric answer.
    """
    del generation_params

    hr = state.hr_bpm
    if hr is None:
        return None

    evidence = {
            "field": "hr_bpm",
            "value": hr,
            "unit": "bpm",
            "numeric_tolerance_note": "Evaluate with template.numeric_tolerance.",
        }
    evidence.update(_hr_provenance_evidence(state))

    return AnswerPayload(
        answer=_format_number(hr),
        answer_type="numeric",
        evidence=evidence,
    )


def answer_heart_rate_higher_than_previous(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Answer whether current HR is higher than the previous window by at least
    min_delta_bpm.
    """
    current = state.hr_bpm
    previous = state.previous_hr_bpm

    if current is None or previous is None:
        return None

    min_delta_bpm = float(generation_params.get("min_delta_bpm", 5.0))
    delta = current - previous

    answer = "yes" if delta >= min_delta_bpm else "no"

    evidence = {
            "fields": ["hr_bpm", "previous_hr_bpm"],
            "current_hr_bpm": current,
            "previous_hr_bpm": previous,
            "delta_bpm": delta,
            "operator": ">=",
            "threshold": min_delta_bpm,
            "previous_window_offset": PREVIOUS_WINDOW_OFFSET,
        }
    evidence.update(_hr_provenance_evidence(state))

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence=evidence,
    )


def answer_rhythm_changed_from_previous(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Answer whether the current rhythm class differs from the previous window.
    """
    del generation_params

    current = state.rhythm_class
    previous = state.previous_rhythm_class

    if current is None or previous is None:
        return None

    answer = "yes" if current != previous else "no"

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence={
            "fields": ["rhythm_class", "previous_rhythm_class"],
            "current_rhythm_class": current,
            "previous_rhythm_class": previous,
            "operator": "!=",
            "previous_window_offset": PREVIOUS_WINDOW_OFFSET,
        },
    )


def answer_stress_presence_current(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Answer whether the current clean WESAD window is labelled stress."""
    del generation_params

    label = state.stress_label
    if label is None:
        return None

    answer = "yes" if label == "stress" else "no"

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence={
            "field": "stress_label",
            "value": label,
            "operator": "==",
            "threshold": "stress",
            "protocol_label_id": state.protocol_label_id,
            "ground_truth_semantics": "WESAD protocol condition",
        },
    )


def answer_stress_state_current(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Return the current WESAD protocol stress/affect class."""
    del generation_params

    label = state.stress_label
    if label is None:
        return None

    return AnswerPayload(
        answer=label,
        answer_type="category",
        evidence={
            "field": "stress_label",
            "value": label,
            "protocol_label_id": state.protocol_label_id,
            "answer_set": ["baseline", "stress", "amusement", "meditation"],
            "ground_truth_semantics": "WESAD protocol condition",
        },
    )


def answer_stress_changed_from_previous(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Answer whether the current WESAD protocol class changed."""
    del generation_params

    current = state.stress_label
    previous = state.previous_stress_label
    if current is None or previous is None:
        return None

    answer = "yes" if current != previous else "no"

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence={
            "fields": ["stress_label", "previous_stress_label"],
            "current_stress_label": current,
            "previous_stress_label": previous,
            "current_protocol_label_id": state.protocol_label_id,
            "previous_protocol_label_id": state.previous_protocol_label_id,
            "operator": "!=",
            "previous_window_offset": PREVIOUS_WINDOW_OFFSET,
            "ground_truth_semantics": "WESAD protocol condition",
        },
    )


# ---------------------------------------------------------------------------
# Answer functions: Tier B
# ---------------------------------------------------------------------------


def answer_af_presence_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Answer whether monitoring-window AF burden is non-zero.

    The generator assumes af_burden_ratio has already been computed upstream.
    It does not apply AF episode duration filtering here.
    """
    del generation_params

    burden = state.af_burden_ratio
    if burden is None:
        return None

    answer = "yes" if burden > 0 else "no"

    evidence: dict[str, Any] = {
        "field": "af_burden_ratio",
        "value": burden,
        "operator": ">",
        "threshold": 0,
    }

    if state.modality == "ppg":
        evidence["methodological_caveat"] = (
            "For PPG, AF presence is an indirect rhythm-irregularity inference, "
            "not direct observation of P-wave absence."
        )

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence=evidence,
    )


def answer_af_frequency_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Convert monitoring-window AF burden into a user-facing frequency category.

    The generator assumes af_burden_ratio has already been computed upstream.
    It does not apply AF episode duration filtering here.
    """
    del generation_params

    burden = state.af_burden_ratio
    if burden is None:
        return None

    if burden == 0:
        category = "not_at_all"
    elif burden < 0.05:
        category = "occasionally"
    elif burden < 0.30:
        category = "often"
    else:
        category = "most_of_the_time"

    return AnswerPayload(
        answer=category,
        answer_type="category",
        evidence={
            "field": "af_burden_ratio",
            "value": burden,
            "bins": {
                "not_at_all": "==0",
                "occasionally": "(0, 0.05)",
                "often": "[0.05, 0.30)",
                "most_of_the_time": ">=0.30",
            },
        },
    )


def answer_heart_rate_excursion_threshold_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Answer whether max HR exceeded the sampled threshold.
    """
    max_hr = state.max_hr_bpm
    if max_hr is None:
        return None

    threshold = float(generation_params["threshold_bpm"])
    answer = "yes" if max_hr > threshold else "no"

    evidence = {
            "field": "max_hr_bpm",
            "value": max_hr,
            "unit": "bpm",
            "operator": ">",
            "threshold": threshold,
        }
    evidence.update(_hr_provenance_evidence(state))

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence=evidence,
    )


def answer_highest_heart_rate_query_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Return the maximum HR during the monitoring window as a numeric answer.
    """
    del generation_params

    max_hr = state.max_hr_bpm
    if max_hr is None:
        return None

    evidence = {
            "field": "max_hr_bpm",
            "value": max_hr,
            "unit": "bpm",
            "numeric_tolerance_note": "Evaluate with template.numeric_tolerance.",
        }
    evidence.update(_hr_provenance_evidence(state))

    return AnswerPayload(
        answer=_format_number(max_hr),
        answer_type="numeric",
        evidence=evidence,
    )


def answer_rhythm_variability_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """
    Convert transition rate into rhythm variability category.
    """
    del generation_params

    transitions_per_hour = state.rhythm_transition_count_per_hour
    if transitions_per_hour is None:
        return None

    if transitions_per_hour == 0:
        category = "stable"
    elif transitions_per_hour <= 3:
        category = "occasional_changes"
    else:
        category = "frequent_changes"

    return AnswerPayload(
        answer=category,
        answer_type="category",
        evidence={
            "field": "rhythm_transition_count_per_hour",
            "value": transitions_per_hour,
            "bins": {
                "stable": "0",
                "occasional_changes": "(0, 3]",
                "frequent_changes": ">3",
            },
        },
    )


def answer_stress_presence_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Answer whether any allowed WESAD protocol time was labelled stress."""
    del generation_params

    burden = state.stress_burden_ratio
    if burden is None:
        return None

    answer = "yes" if burden > 0 else "no"

    return AnswerPayload(
        answer=answer,
        answer_type="yes_no",
        evidence={
            "field": "stress_burden_ratio",
            "value": burden,
            "operator": ">",
            "threshold": 0,
            "aggregate_scope": state.metadata.get("aggregate_scope"),
            "ground_truth_semantics": "WESAD protocol condition",
        },
    )


def answer_stress_ratio_query_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Return the fraction of allowed protocol time labelled stress."""
    del generation_params

    burden = state.stress_burden_ratio
    if burden is None:
        return None

    return AnswerPayload(
        answer=_format_number(burden),
        answer_type="numeric",
        evidence={
            "field": "stress_burden_ratio",
            "value": burden,
            "unit": "ratio",
            "aggregate_scope": state.metadata.get("aggregate_scope"),
            "numeric_tolerance_note": "Evaluate with template.numeric_tolerance.",
        },
    )


def answer_stress_duration_query_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Return stress-labelled duration in minutes."""
    del generation_params

    duration_s = state.stress_duration_s
    if duration_s is None:
        return None

    duration_min = duration_s / 60.0

    return AnswerPayload(
        answer=_format_number(duration_min),
        answer_type="numeric",
        evidence={
            "field": "stress_duration_s",
            "value": duration_s,
            "answer_unit": "minutes",
            "answer_value_min": duration_min,
            "aggregate_scope": state.metadata.get("aggregate_scope"),
            "numeric_tolerance_note": "Evaluate with template.numeric_tolerance.",
        },
    )


def answer_dominant_stress_state_monitoring_window(
    state: MonitoringState,
    generation_params: dict[str, Any],
) -> AnswerPayload | None:
    """Return the dominant WESAD protocol state in the aggregate window."""
    del generation_params

    label = state.dominant_stress_label
    if label is None:
        return None

    return AnswerPayload(
        answer=label,
        answer_type="category",
        evidence={
            "field": "dominant_stress_label",
            "value": label,
            "answer_set": ["baseline", "stress", "amusement", "meditation"],
            "aggregate_scope": state.metadata.get("aggregate_scope"),
            "ground_truth_semantics": "WESAD protocol condition",
        },
    )


ANSWER_FUNCTIONS: dict[str, AnswerFn] = {
    # Tier A
    "answer_af_presence_current": answer_af_presence_current,
    "answer_heart_rate_category_current": answer_heart_rate_category_current,
    "answer_heart_rate_query_current": answer_heart_rate_query_current,
    "answer_heart_rate_higher_than_previous": answer_heart_rate_higher_than_previous,
    "answer_rhythm_changed_from_previous": answer_rhythm_changed_from_previous,
    "answer_stress_presence_current": answer_stress_presence_current,
    "answer_stress_state_current": answer_stress_state_current,
    "answer_stress_changed_from_previous": answer_stress_changed_from_previous,
    # Tier B
    "answer_af_presence_monitoring_window": answer_af_presence_monitoring_window,
    "answer_af_frequency_monitoring_window": answer_af_frequency_monitoring_window,
    "answer_heart_rate_excursion_threshold_monitoring_window": (
        answer_heart_rate_excursion_threshold_monitoring_window
    ),
    "answer_highest_heart_rate_query_monitoring_window": (
        answer_highest_heart_rate_query_monitoring_window
    ),
    "answer_rhythm_variability_monitoring_window": (
        answer_rhythm_variability_monitoring_window
    ),
    "answer_stress_presence_monitoring_window": (
        answer_stress_presence_monitoring_window
    ),
    "answer_stress_ratio_query_monitoring_window": (
        answer_stress_ratio_query_monitoring_window
    ),
    "answer_stress_duration_query_monitoring_window": (
        answer_stress_duration_query_monitoring_window
    ),
    "answer_dominant_stress_state_monitoring_window": (
        answer_dominant_stress_state_monitoring_window
    ),
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_number(value: float) -> str:
    """
    Format numeric answers compactly.

    Examples
    --------
    82.0 -> "82"
    82.4 -> "82.4"
    """
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


__all__ = [
    "PREVIOUS_WINDOW_OFFSET",
    "AnswerPayload",
    "generate_qa_from_state",
    "generate_qa_from_states",
    "validate_unique_question_ids",
    "write_jsonl",
    "summarize_samples",
    "build_question_id",
]


