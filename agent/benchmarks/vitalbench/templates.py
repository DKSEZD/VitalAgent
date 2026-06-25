"""
VitalBench template definitions.

This file defines modality-aware, dataset-aware QA templates for the
mHealth ECG/PPG agent evaluation.

Design principles:
- First-person, user-facing questions.
- Avoid logically duplicated verify templates.
- Avoid dead classes that cannot be produced from automatic GT.
- Closed-set questions first, but include numeric single-query templates
  with explicit tolerance-based evaluation.
- Track whether ground truth comes from dataset annotation, signal-derived
  features, or temporal aggregation.
- Keep system-internal checks such as signal quality outside this benchmark
  template registry.
"""

from __future__ import annotations

from agent.benchmarks.vitalbench.template_types import (
    DatasetName,
    ModalityName,
    QATemplate,
    QuestionType,
    Tier,
)


# ---------------------------------------------------------------------------
# Tier A: current-window / local temporal questions
# ---------------------------------------------------------------------------

TIER_A_TEMPLATES: tuple[QATemplate, ...] = (
    QATemplate(
        template_id="ta1_af_presence_current",
        tier="A",
        question_type="single_verify",
        target="af_presence_current",
        time_scope="current_window",
        canonical="Does my recording indicate atrial fibrillation right now?",
        paraphrases=(
            "At this moment, is there a sign of AF in my data?",
            "Based on the current window, am I showing signs of AF?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_af_presence_current",
        required_fields=("rhythm_class",),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("icentia11k", "afppgecg"),
        modality=("ecg", "ppg"),
        difficulty_tier="easy",
        ground_truth_source="dataset_annotation",
        evaluation_method="boolean_match",
        notes=(
            "This replaces the previous duplicated pair of positive/negative "
            "AF verify templates. Negative phrasing is intentionally excluded "
            "from the main benchmark to avoid inflating question count and "
            "introducing negation bias."
        ),
    ),
    QATemplate(
        template_id="ta2_heart_rate_category_current",
        tier="A",
        question_type="single_choose",
        target="heart_rate_category_current",
        time_scope="current_window",
        canonical=(
            "Is my heart rate at this moment "
            "(A) low, (B) normal, or (C) high?"
        ),
        paraphrases=(
            "Based on my recording right now, where does my heart rate fall: low, normal, or high?",
            "At this moment, is my heart rate low, normal, or high?",
        ),
        answer_set=("low", "normal", "high"),
        answer_fn="answer_heart_rate_category_current",
        required_fields=("hr_bpm",),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("icentia11k", "afppgecg", "ppg_dalia"),
        modality=("ecg", "ppg"),
        difficulty_tier="easy",
        ground_truth_source="derived_from_signal",
        evaluation_method="normalized_match",
        notes=(
            "Default bins: low < 60 bpm, normal 60-100 bpm, high > 100 bpm. "
            "These are screening categories, not diagnoses. Single-choose answers "
            "should be evaluated with normalized matching so variants such as "
            "'A. low', 'low heart rate', or localized wording can map to the "
            "canonical answer label."
        ),
    ),
    QATemplate(
        template_id="ta3_heart_rate_query_current",
        tier="A",
        question_type="single_query",
        target="heart_rate_numeric_current",
        time_scope="current_window",
        canonical="What is my heart rate right now?",
        paraphrases=(
            "How fast is my heart beating at this moment?",
            "What is my current heart rate in beats per minute?",
        ),
        answer_set=(),
        answer_fn="answer_heart_rate_query_current",
        required_fields=("hr_bpm",),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("icentia11k", "afppgecg", "ppg_dalia"),
        modality=("ecg", "ppg"),
        difficulty_tier="easy",
        ground_truth_source="derived_from_signal",
        evaluation_method="numeric_tolerance",
        numeric_tolerance=5.0,
        notes=(
            "Single-query numeric answer. Evaluate with ±5 bpm tolerance, "
            "not exact string match."
        ),
    ),
    QATemplate(
        template_id="ta4_heart_rate_higher_than_previous",
        tier="A",
        question_type="single_verify",
        target="heart_rate_higher_than_previous_window",
        time_scope="previous_window_comparison",
        canonical="Is my heart rate higher now than in the previous window?",
        paraphrases=(
            "Compared with a few moments ago, is my heart rate higher now?",
            "Did my heart rate increase compared with the last window?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_heart_rate_higher_than_previous",
        required_fields=("hr_bpm", "previous_hr_bpm"),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("icentia11k", "afppgecg", "ppg_dalia"),
        modality=("ecg", "ppg"),
        difficulty_tier="medium",
        ground_truth_source="derived_from_previous_window",
        evaluation_method="boolean_match",
        parameters={
            "min_delta_bpm": (5.0,),
        },
        notes=(
            "This replaces baseline-comparison in the first version. It avoids "
            "requiring a debated personal baseline definition. The answer should "
            "be positive only when current HR exceeds the previous-window HR by "
            "at least min_delta_bpm. The previous window is defined by the "
            "generator configuration, not by this template."
        ),
    ),
    QATemplate(
        template_id="ta5_rhythm_changed_from_previous",
        tier="A",
        question_type="single_verify",
        target="rhythm_changed_from_previous_window",
        time_scope="previous_window_comparison",
        canonical="Has my rhythm changed compared with the previous window?",
        paraphrases=(
            "Is my rhythm now different from what it was a few moments ago?",
            "Based on my recording, did my heartbeat pattern shift recently?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_rhythm_changed_from_previous",
        required_fields=("rhythm_class", "previous_rhythm_class"),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("icentia11k", "afppgecg"),
        modality=("ecg", "ppg"),
        difficulty_tier="medium",
        ground_truth_source="derived_from_previous_window",
        evaluation_method="boolean_match",
        notes=(
            "The previous window duration should follow the dataset generation "
            "config, e.g. 10s or 30s."
        ),
    ),
    QATemplate(
        template_id="ta6_stress_presence_current",
        tier="A",
        question_type="single_verify",
        target="stress_presence_current",
        time_scope="current_window",
        canonical="Does this wearable recording indicate a stress condition right now?",
        paraphrases=(
            "Based on this current window, is the protocol label stress?",
            "At this moment, is my wearable data labelled as stress?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_stress_presence_current",
        required_fields=("stress_label",),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="easy",
        ground_truth_source="dataset_annotation",
        evaluation_method="boolean_match",
        notes=(
            "WESAD stress GT is the lab protocol condition. It should be "
            "treated as stress-condition annotation, not a clinical diagnosis."
        ),
    ),
    QATemplate(
        template_id="ta7_stress_state_current",
        tier="A",
        question_type="single_choose",
        target="stress_state_current",
        time_scope="current_window",
        canonical=(
            "Which protocol state best matches this wearable window: "
            "(A) baseline, (B) stress, (C) amusement, or (D) meditation?"
        ),
        paraphrases=(
            "For this window, is the labelled state baseline, stress, amusement, or meditation?",
            "What is the current WESAD protocol condition: baseline, stress, amusement, or meditation?",
        ),
        answer_set=("baseline", "stress", "amusement", "meditation"),
        answer_fn="answer_stress_state_current",
        required_fields=("stress_label",),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="easy",
        ground_truth_source="dataset_annotation",
        evaluation_method="normalized_match",
        notes=(
            "Use only clean windows whose WESAD protocol label is stable "
            "throughout the generated window."
        ),
    ),
    QATemplate(
        template_id="ta9_stress_changed_from_previous",
        tier="A",
        question_type="single_verify",
        target="stress_state_changed_from_previous_window",
        time_scope="previous_window_comparison",
        canonical="Has the labelled stress-related state changed compared with the previous clean window?",
        paraphrases=(
            "Is this WESAD protocol state different from the previous clean window?",
            "Compared with the previous clean window, did the labelled stress or affective state change?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_stress_changed_from_previous",
        required_fields=("stress_label", "previous_stress_label"),
        expected_tools=("get_current_monitoring_state",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="medium",
        ground_truth_source="derived_from_previous_window",
        evaluation_method="boolean_match",
        notes=(
            "Previous window means the previous generated clean WESAD window "
            "for the same subject and window duration."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Tier B: monitoring-window aggregate questions
# ---------------------------------------------------------------------------

TIER_B_TEMPLATES: tuple[QATemplate, ...] = (
    QATemplate(
        template_id="tb1_af_presence_monitoring_window",
        tier="B",
        question_type="single_verify",
        target="af_presence_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical="Have I had any atrial fibrillation episodes during this monitoring window?",
        paraphrases=(
            "Lately, has my data shown any AF?",
            "During this monitoring window, did my data capture any episodes of atrial fibrillation?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_af_presence_monitoring_window",
        required_fields=("af_burden_ratio",),
        expected_tools=("get_longitudinal_trend", "get_recent_alerts"),
        dataset_compatibility=("icentia11k", "afppgecg"),
        modality=("ecg", "ppg"),
        difficulty_tier="easy",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="boolean_match",
        notes=(
            "Positive when monitoring-window AF burden is greater than zero."
        ),
    ),
    QATemplate(
        template_id="tb2_af_frequency_monitoring_window",
        tier="B",
        question_type="single_choose",
        target="af_frequency_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical=(
            "Lately, have I been having AF "
            "(A) not at all, (B) occasionally, (C) often, or (D) most of the time?"
        ),
        paraphrases=(
            "During this monitoring window, was AF absent, occasional, frequent, or present most of the time?",
            "How often did my recent data show AF: not at all, occasionally, often, or most of the time?",
        ),
        answer_set=("not_at_all", "occasionally", "often", "most_of_the_time"),
        answer_fn="answer_af_frequency_monitoring_window",
        required_fields=("af_burden_ratio",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("icentia11k", "afppgecg"),
        modality=("ecg", "ppg"),
        difficulty_tier="medium",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="normalized_match",
        notes=(
            "User-facing version of AF burden. Suggested bins: "
            "not_at_all = 0; occasionally = (0, 0.05); often = [0.05, 0.30); "
            "most_of_the_time >= 0.30. These bins can be adjusted later. "
            "Single-choose answers should be evaluated with normalized matching, "
            "not raw exact string matching."
        ),
    ),
    QATemplate(
        template_id="tb3_heart_rate_excursion_threshold_monitoring_window",
        tier="B",
        question_type="single_verify",
        target="heart_rate_excursion_threshold_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical="Did my heart rate ever go above {threshold_bpm} bpm during this monitoring window?",
        paraphrases=(
            "Lately, has my data captured any moments where my heart rate exceeded {threshold_bpm} bpm?",
            "During this monitoring window, did my heart rate spike above {threshold_bpm} bpm?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_heart_rate_excursion_threshold_monitoring_window",
        required_fields=("max_hr_bpm",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("icentia11k", "afppgecg", "ppg_dalia"),
        modality=("ecg", "ppg"),
        difficulty_tier="easy",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="boolean_match",
        parameters={
            "threshold_bpm": (100, 110, 120, 130),
        },
        notes=(
            "The threshold should be sampled during generation to avoid "
            "overfitting to a single hard-coded value."
        ),
    ),
    QATemplate(
        template_id="tb4_highest_heart_rate_query_monitoring_window",
        tier="B",
        question_type="single_query",
        target="highest_heart_rate_numeric_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical="What was my highest heart rate during this monitoring window?",
        paraphrases=(
            "What was the maximum heart rate captured in this monitoring window?",
            "How high did my heart rate get during this monitoring window?",
        ),
        answer_set=(),
        answer_fn="answer_highest_heart_rate_query_monitoring_window",
        required_fields=("max_hr_bpm",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("icentia11k", "afppgecg", "ppg_dalia"),
        modality=("ecg", "ppg"),
        difficulty_tier="easy",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="numeric_tolerance",
        numeric_tolerance=5.0,
        notes=(
            "Single-query numeric answer. Evaluate with ±5 bpm tolerance. "
            "A secondary LLM judge can be used only for formatting variants, "
            "not as the primary metric."
        ),
    ),
    QATemplate(
        template_id="tb5_rhythm_variability_monitoring_window",
        tier="B",
        question_type="single_choose",
        target="rhythm_variability_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical=(
            "How variable was my rhythm during this monitoring window: "
            "(A) stable throughout, (B) occasional changes, or (C) frequent changes?"
        ),
        paraphrases=(
            "Lately, has my rhythm been stable, occasionally changing, or frequently changing?",
            "During this monitoring window, did my heartbeat pattern stay stable or change often?",
        ),
        answer_set=("stable", "occasional_changes", "frequent_changes"),
        answer_fn="answer_rhythm_variability_monitoring_window",
        required_fields=("rhythm_transition_count_per_hour",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("icentia11k", "afppgecg"),
        modality=("ecg", "ppg"),
        difficulty_tier="hard",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="normalized_match",
        notes=(
            "Suggested bins: stable = 0 transitions/hour; "
            "occasional_changes = 1-3 transitions/hour; "
            "frequent_changes > 3 transitions/hour. These are evaluation bins, "
            "not clinical thresholds. Single-choose answers should be evaluated "
            "with normalized matching rather than exact string matching."
        ),
    ),
    QATemplate(
        template_id="tb6_stress_presence_monitoring_window",
        tier="B",
        question_type="single_verify",
        target="stress_presence_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical="During this monitoring window, was there any period labelled as stress?",
        paraphrases=(
            "Across this monitoring window, did the WESAD protocol include stress at any point?",
            "Was stress present anywhere in the labelled wearable monitoring period?",
        ),
        answer_set=("yes", "no"),
        answer_fn="answer_stress_presence_monitoring_window",
        required_fields=("stress_burden_ratio",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="easy",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="boolean_match",
        notes=(
            "Positive when the subject-level allowed WESAD protocol labels "
            "contain any stress-labelled duration. This is benchmark GT, not "
            "an inference-time wearable stress classifier."
        ),
    ),
    QATemplate(
        template_id="tb7_stress_ratio_query_monitoring_window",
        tier="B",
        question_type="single_query",
        target="stress_ratio_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical="What fraction of this monitoring window was labelled as stress?",
        paraphrases=(
            "What proportion of the allowed WESAD protocol time was stress?",
            "How much of this labelled monitoring period was stress, as a ratio?",
        ),
        answer_set=(),
        answer_fn="answer_stress_ratio_query_monitoring_window",
        required_fields=("stress_burden_ratio",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="medium",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="numeric_tolerance",
        numeric_tolerance=0.02,
        notes=(
            "Numeric ratio in [0, 1]. Default WESAD generation aggregates over "
            "allowed labels 1/2/3 unless meditation is explicitly included."
        ),
    ),
    QATemplate(
        template_id="tb8_stress_duration_query_monitoring_window",
        tier="B",
        question_type="single_query",
        target="stress_duration_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical="How many minutes in this monitoring window were labelled as stress?",
        paraphrases=(
            "What was the total stress-labelled duration in minutes?",
            "Across this labelled monitoring period, how many minutes were stress?",
        ),
        answer_set=(),
        answer_fn="answer_stress_duration_query_monitoring_window",
        required_fields=("stress_duration_s",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="medium",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="numeric_tolerance",
        numeric_tolerance=0.1,
        notes=(
            "Numeric answer is minutes. Evidence stores both seconds and "
            "minutes so downstream reporting can choose either unit."
        ),
    ),
    QATemplate(
        template_id="tb9_dominant_stress_state_monitoring_window",
        tier="B",
        question_type="single_choose",
        target="dominant_stress_state_monitoring_window",
        time_scope="monitoring_window_aggregate",
        canonical=(
            "Which WESAD protocol state dominated this monitoring window: "
            "(A) baseline, (B) stress, (C) amusement, or (D) meditation?"
        ),
        paraphrases=(
            "Across this labelled monitoring period, which protocol state lasted the longest?",
            "Was the dominant labelled state baseline, stress, amusement, or meditation?",
        ),
        answer_set=("baseline", "stress", "amusement", "meditation"),
        answer_fn="answer_dominant_stress_state_monitoring_window",
        required_fields=("dominant_stress_label",),
        expected_tools=("get_longitudinal_trend",),
        dataset_compatibility=("wesad",),
        modality=("wearable",),
        difficulty_tier="medium",
        ground_truth_source="derived_from_temporal_aggregation",
        evaluation_method="normalized_match",
        notes=(
            "Dominant state is computed over allowed WESAD protocol labels. "
            "Ties are broken deterministically with stress before amusement, "
            "meditation, then baseline."
        ),
    ),
)


QA_TEMPLATES: tuple[QATemplate, ...] = (
    *TIER_A_TEMPLATES,
    *TIER_B_TEMPLATES,
)


def get_templates(
    *,
    tier: Tier | None = None,
    question_type: QuestionType | None = None,
    target: str | None = None,
    dataset: DatasetName | None = None,
    modality: ModalityName | None = None,
) -> tuple[QATemplate, ...]:
    """
    Return templates filtered by tier, question_type, target, dataset, or modality.

    Examples
    --------
    get_templates(tier="A")
    get_templates(question_type="single_verify")
    get_templates(dataset="afppgecg", modality="ppg")
    """
    templates = QA_TEMPLATES

    if tier is not None:
        templates = tuple(t for t in templates if t.tier == tier)

    if question_type is not None:
        templates = tuple(t for t in templates if t.question_type == question_type)

    if target is not None:
        templates = tuple(t for t in templates if t.target == target)

    if dataset is not None:
        templates = tuple(t for t in templates if dataset in t.dataset_compatibility)

    if modality is not None:
        templates = tuple(t for t in templates if modality in t.modality)

    return templates


def count_question_phrasings(templates: tuple[QATemplate, ...] = QA_TEMPLATES) -> int:
    """Count canonical + paraphrase question phrasings."""
    return sum(len(t.all_phrasings) for t in templates)


def summarize_templates() -> dict[str, object]:
    """Return a small summary useful for sanity checks."""
    templates = QA_TEMPLATES

    by_tier: dict[str, int] = {}
    by_question_type: dict[str, int] = {}
    by_gt_source: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}

    for template in templates:
        by_tier[template.tier] = by_tier.get(template.tier, 0) + 1
        by_question_type[template.question_type] = (
            by_question_type.get(template.question_type, 0) + 1
        )
        by_gt_source[template.ground_truth_source] = (
            by_gt_source.get(template.ground_truth_source, 0) + 1
        )
        by_difficulty[template.difficulty_tier] = (
            by_difficulty.get(template.difficulty_tier, 0) + 1
        )

    return {
        "semantic_template_count": len(templates),
        "question_phrasing_count": count_question_phrasings(templates),
        "by_tier": by_tier,
        "by_question_type": by_question_type,
        "by_ground_truth_source": by_gt_source,
        "by_difficulty": by_difficulty,
        "template_ids": [template.template_id for template in templates],
    }


if __name__ == "__main__":
    from pprint import pprint

    pprint(summarize_templates())
    print()
    print("AF-PPG-ECG PPG-compatible templates:")
    pprint(
        [
            template.template_id
            for template in get_templates(dataset="afppgecg", modality="ppg")
        ]
    )

