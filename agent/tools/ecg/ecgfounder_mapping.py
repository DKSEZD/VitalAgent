"""ECGFounder label mapping for SCP-derived diagnostic concepts."""

from __future__ import annotations

from typing import Any


DIAGNOSTIC_CONCEPT_MAPPINGS: dict[str, dict[str, Any]] = {
    "atrial fibrillation": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["ATRIAL FIBRILLATION"],
        "note": "Exact label match.",
    },
    "atrial flutter": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["ATRIAL FLUTTER"],
        "note": "Exact label match.",
    },
    "incomplete left bundle branch block": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["INCOMPLETE LEFT BUNDLE BRANCH BLOCK"],
        "note": "Exact label match.",
    },
    "incomplete right bundle branch block": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["INCOMPLETE RIGHT BUNDLE BRANCH BLOCK"],
        "note": "Exact label match.",
    },
    "left anterior fascicular block": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["LEFT ANTERIOR FASCICULAR BLOCK"],
        "note": "Exact label match.",
    },
    "left posterior fascicular block": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["LEFT POSTERIOR FASCICULAR BLOCK"],
        "note": "Exact label match.",
    },
    "left ventricular hypertrophy": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["LEFT VENTRICULAR HYPERTROPHY"],
        "note": "Exact label match.",
    },
    "normal ecg": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["NORMAL ECG"],
        "note": "Exact label match.",
    },
    "sinus bradycardia": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["SINUS BRADYCARDIA"],
        "note": "Exact label match.",
    },
    "sinus rhythm": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["SINUS RHYTHM"],
        "note": "Exact label match.",
    },
    "sinus tachycardia": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["SINUS TACHYCARDIA"],
        "note": "Exact label match.",
    },
    "supraventricular tachycardia": {
        "match_type": "direct",
        "match_scope": "specific",
        "labels": ["SUPRAVENTRICULAR TACHYCARDIA"],
        "note": "Exact label match.",
    },
    "any diagnostic symptoms": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": ["ABNORMAL ECG"],
        "note": "Broad abnormal-vs-normal category.",
    },
    "any form-related symptoms": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "ABNORMAL ECG",
            "LEFT VENTRICULAR HYPERTROPHY",
            "RIGHT VENTRICULAR HYPERTROPHY",
            "BIVENTRICULAR HYPERTROPHY",
            "RIGHT BUNDLE BRANCH BLOCK",
            "LEFT BUNDLE BRANCH BLOCK",
            "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
            "NONSPECIFIC INTRAVENTRICULAR BLOCK",
            "ANTERIOR INFARCT",
            "ANTEROSEPTAL INFARCT",
            "ANTEROLATERAL INFARCT",
            "INFERIOR INFARCT",
            "LATERAL INFARCT",
            "POSTERIOR INFARCT",
        ],
        "note": "Broad morphology/form category.",
    },
    "any kind of abnormal symptoms": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": ["ABNORMAL ECG"],
        "note": "Broad abnormal-vs-normal category.",
    },
    "any rhythm-related symptoms": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "SINUS BRADYCARDIA",
            "SINUS TACHYCARDIA",
            "ATRIAL FIBRILLATION",
            "ATRIAL FLUTTER",
            "SUPRAVENTRICULAR TACHYCARDIA",
            "UNDETERMINED RHYTHM",
            "JUNCTIONAL RHYTHM",
            "VENTRICULAR TACHYCARDIA",
            "WIDE QRS TACHYCARDIA",
        ],
        "note": "Broad rhythm category.",
    },
    "atrial premature complex": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["PREMATURE ATRIAL COMPLEXES"],
        "note": "Pluralized label.",
    },
    "bigeminal pattern (unknown origin, supraventricular, or ventricular)": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["IN A PATTERN OF BIGEMINY"],
        "note": "Same phenomenon, phrased differently.",
    },
    "complete left bundle branch block": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["LEFT BUNDLE BRANCH BLOCK"],
        "note": "ECGFounder omits 'complete'.",
    },
    "complete right bundle branch block": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["RIGHT BUNDLE BRANCH BLOCK"],
        "note": "ECGFounder omits 'complete'.",
    },
    "conduction disturbance": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "RIGHT BUNDLE BRANCH BLOCK",
            "LEFT BUNDLE BRANCH BLOCK",
            "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
            "NONSPECIFIC INTRAVENTRICULAR BLOCK",
            "LEFT ANTERIOR FASCICULAR BLOCK",
            "LEFT POSTERIOR FASCICULAR BLOCK",
            "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",
            "INCOMPLETE LEFT BUNDLE BRANCH BLOCK",
            "WITH 1ST DEGREE AV BLOCK",
        ],
        "note": "Broad conduction umbrella category.",
    },
    "digitalis effect": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["OR DIGITALIS EFFECT"],
        "note": "Same concept embedded in a longer label.",
    },
    "first degree av block": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["WITH 1ST DEGREE AV BLOCK", "WITH PROLONGED AV CONDUCTION"],
        "note": "Equivalent clinical interpretation.",
    },
    "high qrs voltage": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY"],
        "note": "Closest voltage-based hypertrophy label.",
    },
    "hypertrophy": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "LEFT VENTRICULAR HYPERTROPHY",
            "RIGHT VENTRICULAR HYPERTROPHY",
            "BIVENTRICULAR HYPERTROPHY",
        ],
        "note": "Broad hypertrophy category.",
    },
    "inverted t-waves": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "T WAVE INVERSION NOW EVIDENT IN",
            "T WAVE INVERSION NO LONGER EVIDENT IN",
            "T WAVE INVERSION MORE EVIDENT IN",
            "T WAVE INVERSION LESS EVIDENT IN",
        ],
        "note": "Same abnormality family.",
    },
    "ischemic in anterior leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["ANTERIOR INJURY PATTERN"],
        "note": "Partial match: injury pattern, not explicit ischemia.",
    },
    "ischemic in anterolateral leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["ANTEROLATERAL INJURY PATTERN"],
        "note": "Partial regional match.",
    },
    "ischemic in anteroseptal leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["ANTERIOR INJURY PATTERN", "ANTEROSEPTAL INFARCT"],
        "note": "Partial regional match only.",
    },
    "ischemic in inferior leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["INFERIOR INJURY PATTERN"],
        "note": "Partial regional match.",
    },
    "ischemic in inferolateral leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["INFEROLATERAL INJURY PATTERN"],
        "note": "Partial regional match.",
    },
    "ischemic in lateral leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["LATERAL INJURY PATTERN"],
        "note": "Partial regional match.",
    },
    "left atrial overload/enlargement": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["LEFT ATRIAL ENLARGEMENT"],
        "note": "Same concept, different wording.",
    },
    "long qt-interval": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["PROLONGED QT"],
        "note": "Same concept, different wording.",
    },
    "low amplitude t-wave": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["T WAVE AMPLITUDE HAS DECREASED IN", "NONSPECIFIC T WAVE ABNORMALITY"],
        "note": "Closest T-wave attenuation / abnormality labels.",
    },
    "low qrs voltages in the frontal and horizontal leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["LOW VOLTAGE QRS"],
        "note": "Same concept, less lead-specific wording.",
    },
    "myocardial infarction": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "ACUTE MI",
            "ACUTE MI / STEMI",
            "ANTERIOR INFARCT",
            "ANTEROSEPTAL INFARCT",
            "ANTEROLATERAL INFARCT",
            "INFERIOR INFARCT",
            "INFERIOR-POSTERIOR INFARCT",
            "LATERAL INFARCT",
            "POSTERIOR INFARCT",
        ],
        "note": "Broad infarct family; would require aggregation.",
    },
    "myocardial infarction in anterior leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["ANTERIOR INFARCT"],
        "note": "Regional infarct match.",
    },
    "myocardial infarction in anterolateral leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["ANTEROLATERAL INFARCT"],
        "note": "Regional infarct match.",
    },
    "myocardial infarction in anteroseptal leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["ANTEROSEPTAL INFARCT"],
        "note": "Regional infarct match.",
    },
    "myocardial infarction in inferior leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["INFERIOR INFARCT"],
        "note": "Regional infarct match.",
    },
    "myocardial infarction in inferoposterior leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["INFERIOR-POSTERIOR INFARCT"],
        "note": "Regional infarct match.",
    },
    "myocardial infarction in lateral leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["LATERAL INFARCT"],
        "note": "Regional infarct match.",
    },
    "myocardial infarction in posterior leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["POSTERIOR INFARCT"],
        "note": "Regional infarct match.",
    },
    "non-diagnostic t abnormalities": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["NONSPECIFIC T WAVE ABNORMALITY"],
        "note": "Closest nonspecific T-wave label.",
    },
    "non-specific intraventricular conduction disturbance (block)": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": [
            "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
            "NONSPECIFIC INTRAVENTRICULAR BLOCK",
        ],
        "note": "Same family, wording differs.",
    },
    "non-specific st changes": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": [
            "NONSPECIFIC ST ABNORMALITY",
            "NON-SPECIFIC CHANGE IN ST SEGMENT IN",
            "NONSPECIFIC ST AND T WAVE ABNORMALITY",
        ],
        "note": "Same family, wording differs.",
    },
    "non-specific st depression": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": [
            "ST NOW DEPRESSED IN",
            "ST MORE DEPRESSED IN",
            "ST LESS DEPRESSED IN",
            "NONSPECIFIC ST ABNORMALITY",
        ],
        "note": "Depression-specific but lead/text phrasing differs.",
    },
    "non-specific st elevation": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": [
            "ST ELEVATION NOW PRESENT IN",
            "ST MORE ELEVATED IN",
            "ST LESS ELEVATED IN",
        ],
        "note": "Elevation-specific but lead/text phrasing differs.",
    },
    "non-specific t-wave changes": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["NONSPECIFIC T WAVE ABNORMALITY"],
        "note": "Same family.",
    },
    "normal functioning artificial pacemaker": {
        "match_type": "semantic",
        "match_scope": "aggregate",
        "labels": [
            "ELECTRONIC ATRIAL PACEMAKER",
            "ELECTRONIC VENTRICULAR PACEMAKER",
            "AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER",
            "VENTRICULAR-PACED RHYTHM",
            "ATRIAL-PACED RHYTHM",
        ],
        "note": "Pacemaker presence is covered; 'normal functioning' is not explicit.",
    },
    "prolonged pr interval": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["WITH PROLONGED AV CONDUCTION", "WITH 1ST DEGREE AV BLOCK"],
        "note": "Equivalent interpretation.",
    },
    "right atrial overload/enlargement": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["RIGHT ATRIAL ENLARGEMENT"],
        "note": "Same concept, different wording.",
    },
    "sinus arrhythmia": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["WITH SINUS ARRHYTHMIA", "WITH MARKED SINUS ARRHYTHMIA"],
        "note": "Same concept.",
    },
    "st/t change": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["NONSPECIFIC ST AND T WAVE ABNORMALITY"],
        "note": "Same family.",
    },
    "subendocardial injury in anterolateral leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["ANTEROLATERAL INJURY PATTERN"],
        "note": "Regional injury match.",
    },
    "subendocardial injury in anteroseptal leads": {
        "match_type": "semantic",
        "match_scope": "partial",
        "labels": ["ANTERIOR INJURY PATTERN"],
        "note": "Partial regional match only.",
    },
    "subendocardial injury in inferior leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["INFERIOR INJURY PATTERN"],
        "note": "Regional injury match.",
    },
    "subendocardial injury in inferolateral leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["INFEROLATERAL INJURY PATTERN"],
        "note": "Regional injury match.",
    },
    "subendocardial injury in lateral leads": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["LATERAL INJURY PATTERN"],
        "note": "Regional injury match.",
    },
    "t-wave abnormality": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["NONSPECIFIC T WAVE ABNORMALITY"],
        "note": "Same family.",
    },
    "ventricular premature complex": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["PREMATURE VENTRICULAR COMPLEXES"],
        "note": "Pluralized label.",
    },
    "voltage criteria (qrs) for left ventricular hypertrophy": {
        "match_type": "semantic",
        "match_scope": "specific",
        "labels": ["VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY"],
        "note": "Same concept with parenthetical '(qrs)' normalized.",
    },
}


UNCOVERED_DIAGNOSTIC_CONCEPTS: dict[str, str] = {
    "abnormal qrs": "No explicit ABNORMAL QRS or generic QRS-morphology abnormality label.",
    "myocardial infarction in inferolateral leads": (
        "No explicit inferolateral infarct label; only INFERIOR-POSTERIOR INFARCT and "
        "INFEROLATERAL INJURY PATTERN."
    ),
    "myocardial infarction in inferoposterolateral leads": "No single label for this combined region.",
    "non-specific ischemic": (
        "No explicit nonspecific ischemia label; only injury / infarct / ST-T abnormality families."
    ),
    "q waves present": "No explicit Q-wave finding label.",
}


def build_scp_relevant_findings(
    label_scores: dict[str, dict[str, float]],
    *,
    min_probability: float = 0.1,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Aggregate ECGFounder label scores into SCP-style diagnostic concepts."""
    findings: list[dict[str, Any]] = []
    for concept, spec in DIAGNOSTIC_CONCEPT_MAPPINGS.items():
        matched = []
        for label in spec["labels"]:
            score = label_scores.get(label)
            if score is None:
                continue
            matched.append(
                {
                    "label": label,
                    "probability": round(float(score["probability"]), 4),
                    "logit": round(float(score["logit"]), 4),
                }
            )
        if not matched:
            continue
        matched.sort(key=lambda item: item["probability"], reverse=True)
        aggregate_probability = max(item["probability"] for item in matched)
        aggregate_logit = max(item["logit"] for item in matched)
        findings.append(
            {
                "diagnostic_concept": concept,
                "match_type": spec["match_type"],
                "match_scope": spec["match_scope"],
                "aggregate_probability": round(float(aggregate_probability), 4),
                "aggregate_logit": round(float(aggregate_logit), 4),
                "matched_labels": matched[:5],
                "note": spec["note"],
            }
        )

    scope_priority = {"specific": 2, "partial": 1, "aggregate": 0}
    type_priority = {"direct": 1, "semantic": 0}
    findings.sort(
        key=lambda item: (
            scope_priority.get(str(item["match_scope"]), 0),
            item["aggregate_probability"],
            type_priority.get(str(item["match_type"]), 0),
            item["aggregate_logit"],
        ),
        reverse=True,
    )
    filtered = [item for item in findings if item["aggregate_probability"] >= min_probability]
    specific_filtered = [item for item in filtered if item["match_scope"] != "aggregate"]
    if specific_filtered:
        filtered = specific_filtered
    if filtered:
        return filtered[:top_k]
    return findings[:top_k]
