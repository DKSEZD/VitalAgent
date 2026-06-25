"""Shared QA template types.

Keeping QATemplate outside dataset-specific template modules avoids circular
imports between the central registry and per-dataset template files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# DatasetName / ModalityName are owned by the runtime schema; re-export them
# here so benchmark templates share a single source of truth and cannot drift.
from agent.mhealth.schemas import DatasetName, ModalityName


QuestionType = Literal[
    "single_verify",
    "single_choose",
    "single_query",
]

Tier = Literal[
    "A",
    "B",
]

TimeScope = Literal[
    "current_window",
    "previous_window_comparison",
    "monitoring_window_aggregate",
]

DifficultyTier = Literal[
    "easy",
    "medium",
    "hard",
]

GroundTruthSource = Literal[
    "dataset_annotation",
    "derived_from_signal",
    "derived_from_temporal_aggregation",
    "derived_from_previous_window",
    "derived_from_baseline",
]

EvaluationMethod = Literal[
    "exact_match",
    "normalized_match",
    "boolean_match",
    "numeric_tolerance",
    "llm_judge_optional",
]


@dataclass(frozen=True)
class QATemplate:
    """
    A dataset-aware VitalBench template.

    Attributes
    ----------
    template_id:
        Stable ID used in generated question IDs.
    tier:
        Tier A = current-window question.
        Tier B = monitoring-window aggregate question.
    question_type:
        single_verify = yes/no.
        single_choose = closed-set category.
        single_query = short numeric/text answer.
    target:
        Semantic target of the question.
    time_scope:
        Temporal scope needed to answer the question.
    canonical:
        Main question wording.
    paraphrases:
        Alternative phrasings. The generator may emit one QA sample per
        phrasing.
    answer_set:
        Closed answer set for verify/choose questions.
        Empty for numeric single-query questions.
    answer_fn:
        Name of the answer function used by the generator.
    required_fields:
        State fields required before this template can be applied.
    expected_tools:
        Tools that a reactive agent should ideally call.
    dataset_compatibility:
        Which datasets can support this template.
    modality:
        Which signal modality can support this template.
    difficulty_tier:
        Evaluation difficulty bucket.
    ground_truth_source:
        Where the GT answer comes from.
    evaluation_method:
        How predictions should be scored.
    numeric_tolerance:
        Absolute tolerance for numeric single-query answers.
    parameters:
        Optional generation parameters, e.g. threshold candidates.
    notes:
        Implementation or evaluation notes.
    """

    template_id: str
    tier: Tier
    question_type: QuestionType
    target: str
    time_scope: TimeScope
    canonical: str
    paraphrases: tuple[str, ...]
    answer_set: tuple[str, ...]
    answer_fn: str
    required_fields: tuple[str, ...]
    expected_tools: tuple[str, ...]
    dataset_compatibility: tuple[DatasetName, ...]
    modality: tuple[ModalityName, ...]
    difficulty_tier: DifficultyTier
    ground_truth_source: GroundTruthSource
    evaluation_method: EvaluationMethod
    numeric_tolerance: float | None = None
    parameters: dict[str, tuple[float | int | str, ...]] = field(default_factory=dict)
    notes: str = ""

    @property
    def all_phrasings(self) -> tuple[str, ...]:
        """Return canonical + paraphrases."""
        return (self.canonical, *self.paraphrases)


__all__ = [
    "QuestionType",
    "Tier",
    "TimeScope",
    "DatasetName",
    "ModalityName",
    "DifficultyTier",
    "GroundTruthSource",
    "EvaluationMethod",
    "QATemplate",
]
