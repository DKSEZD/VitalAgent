"""Frozen validation configuration used by the submitted paper protocol."""

from __future__ import annotations

from agent.reactive.validation import ValidationGate


class PaperV1ValidationGate(ValidationGate):
    """ValidationGate with the exact tool configuration from ``main@2ed4008``."""

    REQUIRED_FIELDS: dict[str, list[str]] = {
        "get_ecg_description": ["description"],
        "get_ecg_metadata": ["metadata"],
        "get_ppg_description": ["description"],
        "get_ppg_metadata": ["metadata"],
        "list_ecg_records": ["record_ids"],
        "list_ppg_records": ["record_ids"],
        "analyze_heart_rate": ["data", "metadata"],
        "analyze_hrv": ["data", "metadata"],
        "analyze_pulse_rate": ["data", "metadata"],
        "analyze_prv": ["data", "metadata"],
        "analyze_morphology": ["data", "metadata"],
        "ecg_diagnosis": ["data", "metadata"],
        "assess_signal_quality": ["data", "metadata"],
        "assess_ppg_signal_quality": ["data", "metadata"],
        "medical_info_search": ["data"],
        "medical_knowledge": ["data"],
        "analyze_lead_morphology": ["data", "metadata"],
        "assess_all_leads_quality": ["data", "metadata"],
        "analyze_ppg_rhythm_irregularity": ["data", "metadata"],
        "state_load_monitoring_states": ["state_count", "contexts"],
        "state_build_from_wesad_pickle": ["dataset", "state_count", "contexts"],
        "state_build_from_ecg_record": ["dataset", "state_count", "contexts"],
        "state_build_from_ppg_patient": ["dataset", "state_count", "contexts"],
        "state_list_contexts": ["contexts"],
        "state_get_current_monitoring_state": ["state", "evidence"],
        "state_get_monitoring_window": ["state_count", "states", "window"],
        "state_get_previous_monitoring_state": ["state"],
        "state_get_longitudinal_trend": ["target", "summary", "evidence"],
        "state_get_evidence": ["evidence"],
        "state_get_dataset_capabilities": ["capabilities"],
        "proactive_load_patient_context": ["context_count", "contexts"],
        "proactive_list_patient_contexts": ["contexts"],
        "proactive_get_recent_alerts": ["alerts", "alert_summary"],
        "proactive_explain_last_alert": ["alert", "explanation"],
        "evaluate_proactive_rules": ["data", "metadata"],
        "analyze_afppgecg_rhythm_context": ["data", "metadata"],
    }

    CRITICAL_TOOLS: set[str] = {
        "analyze_heart_rate",
        "analyze_hrv",
        "analyze_pulse_rate",
        "analyze_prv",
        "analyze_morphology",
        "analyze_lead_morphology",
        "analyze_ppg_rhythm_irregularity",
        "ecg_diagnosis",
        "get_ecg_metadata",
        "get_ppg_metadata",
        "state_load_monitoring_states",
        "state_build_from_wesad_pickle",
        "state_build_from_ecg_record",
        "state_build_from_ppg_patient",
        "state_get_current_monitoring_state",
        "state_get_longitudinal_trend",
        "state_get_evidence",
        "evaluate_proactive_rules",
        "analyze_afppgecg_rhythm_context",
    }
