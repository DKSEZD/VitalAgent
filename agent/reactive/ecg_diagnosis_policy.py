from __future__ import annotations

from dataclasses import dataclass
import re


_HIGH_TRUST_PHRASES: tuple[str, ...] = (
    "atrial fibrillation",
    "atrial flutter",
    "sinus bradycardia",
    "sinus tachycardia",
    "sinus rhythm",
    "complete left bundle branch block",
    "incomplete left bundle branch block",
    "complete right bundle branch block",
    "incomplete right bundle branch block",
    "left anterior fascicular block",
    "left posterior fascicular block",
    "first degree av block",
    "1st degree av block",
    "left ventricular hypertrophy",
    "left atrial overload/enlargement",
    "left atrial enlargement",
    "right atrial overload/enlargement",
    "right atrial enlargement",
    "normal ecg",
    "digitalis effect",
)

_LOW_TRUST_PHRASES: tuple[str, ...] = (
    "including uncertain symptoms",
    "excluding uncertain symptoms",
    "form-related",
    "what leads",
    "q wave",
    "q waves",
    "st depression",
    "st changes",
    "st segment",
    "t-wave",
    "t wave",
    "voltage criteria",
    "high qrs voltage",
    "long qt",
    "qt-interval",
    "qt interval",
    "prolonged qt",
    "qtc",
)

_REGION_TERMS: tuple[str, ...] = (
    "anterior",
    "inferior",
    "lateral",
    "posterior",
    "anteroseptal",
    "anterolateral",
    "inferolateral",
    "inferoposterior",
    "inferoposterolateral",
)

_REGION_DIAGNOSIS_TERMS: tuple[str, ...] = (
    "myocardial infarction",
    "infarction",
    "infarct",
    "ischemic",
    "ischemia",
    "injury",
)

_GENERAL_DIAGNOSIS_HINTS: tuple[str, ...] = (
    "diagnostic symptom",
    "diagnostic symptoms",
    "form-related symptom",
    "form-related symptoms",
    "rhythm-related symptom",
    "rhythm-related symptoms",
    "show symptoms of",
)

_LEAD_PATTERN = re.compile(
    r"\b(?:aVR|aVL|aVF|V[1-6]|III|II|I)\b",
    flags=re.IGNORECASE,
)

_CANONICAL_LEADS = {
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
    "I": "I",
    "II": "II",
    "III": "III",
}


@dataclass(frozen=True)
class EcgDiagnosisQuestionProfile:
    is_relevant: bool
    trust_level: str
    needs_report_context: bool
    needs_lead_morphology: bool
    needs_interval_morphology: bool
    extracted_leads: tuple[str, ...]
    allows_uncertain_positive: bool
    ignores_uncertain_positive: bool
    ecg_diagnosis_mode: str = "12_lead"


def extract_leads_from_query(user_query: str) -> tuple[str, ...]:
    leads: list[str] = []
    seen: set[str] = set()
    for match in _LEAD_PATTERN.finditer(user_query):
        lead = _CANONICAL_LEADS[match.group(0).upper()]
        if lead in seen:
            continue
        seen.add(lead)
        leads.append(lead)
    return tuple(leads)


def classify_ecg_diagnosis_question(user_query: str) -> EcgDiagnosisQuestionProfile:
    query = " ".join(user_query.lower().split())
    extracted_leads = extract_leads_from_query(user_query)

    has_high_trust_phrase = any(phrase in query for phrase in _HIGH_TRUST_PHRASES)
    has_low_trust_phrase = any(phrase in query for phrase in _LOW_TRUST_PHRASES)
    has_region_diagnosis = any(region in query for region in _REGION_TERMS) and any(
        term in query for term in _REGION_DIAGNOSIS_TERMS
    )
    has_lead_specific_diagnosis = (
        (
            extracted_leads
            or " in lead " in query
            or " in leads " in query
            or " lead " in query
            or " leads " in query
        )
        and any(
            term in query
            for term in (
                "symptom",
                "diagnostic",
                "form-related",
                "rhythm-related",
                "st ",
                "t-wave",
                "t wave",
                "q wave",
                "q waves",
                "inverted t",
                "ischemic",
                "injury",
                "infarct",
                "myocardial infarction",
                "voltage criteria",
            )
        )
    )
    has_interval_focus = any(
        term in query
        for term in ("long qt", "qt-interval", "qt interval", "prolonged qt", "qtc")
    )
    has_form_or_wave_focus = any(
        term in query
        for term in (
            "form-related",
            "q wave",
            "q waves",
            "st depression",
            "st changes",
            "st segment",
            "t-wave",
            "t wave",
            "voltage criteria",
            "high qrs voltage",
        )
    )
    has_general_hint = any(hint in query for hint in _GENERAL_DIAGNOSIS_HINTS)

    # Lead I + high-trust concept: eligible for single-lead ECGFounder with
    # high trust, rather than demoting to low trust as for other leads.
    is_lead_i_only = extracted_leads == ("I",)
    is_single_lead_high_trust = (
        is_lead_i_only
        and has_high_trust_phrase
        and has_lead_specific_diagnosis
        and not has_region_diagnosis
        and not has_interval_focus
        and not has_form_or_wave_focus
    )

    is_low_trust = (
        has_low_trust_phrase
        or has_region_diagnosis
        or (has_lead_specific_diagnosis and not is_single_lead_high_trust)
        or has_interval_focus
        or has_form_or_wave_focus
    )
    is_relevant = bool(
        is_low_trust
        or has_high_trust_phrase
        or has_general_hint
        or is_single_lead_high_trust
    )

    if is_single_lead_high_trust:
        trust_level = "high"
        ecg_diagnosis_mode = "single_lead"
    elif is_low_trust:
        trust_level = "low"
        ecg_diagnosis_mode = "12_lead"
    elif has_high_trust_phrase:
        trust_level = "high"
        ecg_diagnosis_mode = "12_lead"
    elif is_relevant:
        trust_level = "neutral"
        ecg_diagnosis_mode = "12_lead"
    else:
        trust_level = "none"
        ecg_diagnosis_mode = "12_lead"

    return EcgDiagnosisQuestionProfile(
        is_relevant=is_relevant,
        trust_level=trust_level,
        needs_report_context=is_relevant,
        needs_lead_morphology=is_low_trust and (
            has_form_or_wave_focus
            or has_lead_specific_diagnosis
            or has_region_diagnosis
        ),
        needs_interval_morphology=is_low_trust and has_interval_focus,
        extracted_leads=extracted_leads,
        allows_uncertain_positive="including uncertain symptoms" in query,
        ignores_uncertain_positive="excluding uncertain symptoms" in query,
        ecg_diagnosis_mode=ecg_diagnosis_mode,
    )


__all__ = [
    "EcgDiagnosisQuestionProfile",
    "classify_ecg_diagnosis_question",
    "extract_leads_from_query",
]
