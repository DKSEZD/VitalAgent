"""Schemas for VitalBench benchmark samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.benchmarks.vitalbench.template_types import (
    DifficultyTier,
    EvaluationMethod,
    GroundTruthSource,
    QuestionType,
    Tier,
)
from agent.mhealth.schemas import DatasetName, ModalityName, canonicalize_dataset_name


@dataclass(kw_only=True)
class VitalBenchSample:
    """
    One generated VitalBench sample.

    The answer is stored as list[str] to support benchmark-style multiple references,
    where each sample may have one or more acceptable answers.

    answer_set is also stored in each sample so the evaluator does not
    need to query the template registry when doing normalized matching.
    """

    # ------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------

    question_id: str
    template_id: str
    source_state_id: str

    patient_id: str
    dataset: DatasetName
    modality: ModalityName
    window_start_s: float | None = None
    window_end_s: float | None = None
    window_duration_s: float | None = None

    # ------------------------------------------------------------------
    # Template metadata
    # ------------------------------------------------------------------

    tier: Tier
    question_type: QuestionType
    target: str
    time_scope: str
    difficulty_tier: DifficultyTier

    # ------------------------------------------------------------------
    # QA content
    # ------------------------------------------------------------------

    question: str
    answer: list[str]
    answer_type: str
    # Expected examples:
    # - "yes_no"
    # - "category"
    # - "numeric"

    answer_set: list[str] = field(default_factory=list)
    # Empty for numeric single-query.
    # For verify: ["yes", "no"].
    # For choose: e.g. ["low", "normal", "high"].

    # ------------------------------------------------------------------
    # Ground truth / evaluation metadata
    # ------------------------------------------------------------------

    ground_truth_source: GroundTruthSource = "derived_from_signal"
    evaluation_method: EvaluationMethod = "exact_match"
    numeric_tolerance: float | None = None

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    evidence: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Tool expectation
    # ------------------------------------------------------------------

    expected_tools: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Optional generation metadata
    # ------------------------------------------------------------------

    phrasing_index: int = 0
    generation_params: dict[str, Any] = field(default_factory=dict)
    # Example:
    # - {"threshold_bpm": 120}
    # - {"min_delta_bpm": 5.0}

    notes: str = ""

    def __post_init__(self) -> None:
        self.dataset = canonicalize_dataset_name(self.dataset)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """Convert QA sample to a serializable dictionary."""
        return asdict(self)

    @property
    def primary_answer(self) -> str:
        """Return the first answer for primary-answer evaluation."""
        return self.answer[0] if self.answer else ""

    def to_jsonl_record(self) -> dict[str, Any]:
        """
        Return the JSONL representation.

        Currently identical to to_dict(), but kept as a separate method so
        future export-specific normalization can be added without changing
        the dataclass itself.
        """
        return self.to_dict()
